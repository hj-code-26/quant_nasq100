"""진입일 복원을 **실계좌에서** 확인하는 읽기 전용 프로브.

    python research/sell_execution_audit/entry_probe.py --selftest   # 가짜 브로커로 프로브 자체 점검
    python research/sell_execution_audit/entry_probe.py              # 실계좌 (읽기 전용)
    python research/sell_execution_audit/entry_probe.py --replay research/out/entry_probe_raw.json

답하는 질문은 하나다: **ENTRY_UNVERIFIED 가 비는가.**
비면 만기 청산이 정상 동작한다. 안 비면 어떤 원인인지까지 분류한다.

하지 않는 것
  · 주문·취소·정정을 내지 않는다. HTTP 는 아래 GET 5종 + 토큰 발급만 허용하고,
    그 외 요청은 가드가 예외로 막는다 (기동 시 가드 자체를 자가 점검한다).
  · 운영 DB(trading_decisions.db)를 열지 않는다 — autotrade.DB_PATH 를 임시 파일로 돌린다.
  · 운영 소스·.env·토큰 파일에 쓰지 않는다. **비밀값을 출력하지 않는다** (계좌번호·토큰·
    자격증명은 어디에도 찍지 않고, 주문 식별자는 앞 4자만 남긴다).

주의
  · 허용 IP 에서 실행해야 한다 (2026-09-10 run 27 이 403 ip-not-allowed 로 죽었다).
  · 유효한 공유 토큰이 있으면 그것을 쓴다. 없으면 새로 발급하는데, 토스는 새 토큰을 내면
    이전 토큰을 무효화한다 — 봇이 도는 중이면 봇은 401 후 공유 파일에서 회복한다.
    걱정되면 사이클 사이(정각 근처가 아닌 때)에 돌릴 것.
"""
import argparse
import builtins
import datetime
import json
import pathlib
import sqlite3
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "research" / "out"
RAW = OUT / "entry_probe_raw.json"
REPORT = OUT / "entry_probe_report.txt"

# 이 프로브가 낼 수 있는 요청의 전부.
ALLOWED_GET = ("/api/v1/holdings", "/api/v1/buying-power", "/api/v1/orders",
               "/api/v1/candles", "/api/v1/market-calendar/US")
ALLOWED_POST = ("/oauth2/token",)          # 토큰 발급. 주문 경로는 전부 막힌다.
BLOCK_WRITE = ("trading_decisions.db", ".env", "autotrade.log", "server.log")


class ProbeViolation(RuntimeError):
    pass


def readonly_guard():
    """GET 화이트리스트 + 운영 파일 쓰기 차단. 설치 후 스스로 점검한다."""
    import requests

    real_req = requests.sessions.Session.request

    def _request(self, method, url, *a, **kw):
        m = str(method).upper()
        path = "/" + str(url).split("//", 1)[-1].split("/", 1)[-1].split("?")[0]
        if m == "GET" and any(path.startswith(p) for p in ALLOWED_GET):
            return real_req(self, method, url, *a, **kw)
        if m == "POST" and any(path.startswith(p) for p in ALLOWED_POST):
            return real_req(self, method, url, *a, **kw)
        raise ProbeViolation(f"허용되지 않은 요청 차단: {m} {path}")

    requests.sessions.Session.request = _request

    real_open = builtins.open

    def _open(file, mode="r", *a, **kw):
        if any(c in str(mode) for c in "wxa+"):
            name = pathlib.Path(str(file)).name
            if name in BLOCK_WRITE or (name.endswith(".py")
                                       and pathlib.Path(str(file)).parent == ROOT):
                raise ProbeViolation(f"운영 파일 쓰기 차단: {name}")
        return real_open(file, mode, *a, **kw)

    builtins.open = _open

    real_connect = sqlite3.connect

    def _connect(database, *a, **kw):
        s = str(database)
        if "trading_decisions.db" in s and "mode=ro" not in s:
            raise ProbeViolation("운영 DB 열기 차단")
        return real_connect(database, *a, **kw)

    sqlite3.connect = _connect

    # --- 가드 자가 점검: 막혀야 할 것이 정말 막히는가 ---
    s = requests.Session()
    for m, u in (("POST", "https://openapi.tossinvest.com/api/v1/orders"),
                 ("POST", "https://openapi.tossinvest.com/api/v1/orders/X/cancel"),
                 ("POST", "https://openapi.tossinvest.com/api/v1/orders/X/modify"),
                 ("GET", "https://example.com/anything")):
        try:
            s.request(m, u)
            raise AssertionError(f"가드가 {m} {u} 를 막지 못했다")
        except ProbeViolation:
            pass
    try:
        sqlite3.connect(str(ROOT / "trading_decisions.db"))
        raise AssertionError("가드가 운영 DB 를 막지 못했다")
    except ProbeViolation:
        pass
    print("가드 자가 점검 통과: 주문 생성·취소·정정 POST, 외부 호스트, 운영 DB 모두 차단됨\n")


def mask(v, keep=4):
    v = "" if v is None else str(v)
    return v[:keep] + "…" if len(v) > keep else v


def slim(rows):
    """스냅샷에 남길 최소 필드. 계좌 식별정보는 애초에 이 응답에 없다."""
    return [{"orderId": mask(r.get("orderId")), "clientOrderId": mask(r.get("clientOrderId")),
             "symbol": r.get("symbol"), "side": r.get("side"), "status": r.get("status"),
             "quantity": r.get("quantity"), "orderedAt": r.get("orderedAt"),
             "execution": {"filledQuantity": (r.get("execution") or {}).get("filledQuantity")}}
            for r in rows]


def collect_live():
    from toss import TossClient
    import autotrade as at

    t = TossClient()
    holdings = at.account_state(t)["holdings"]
    closed, closed_ok = t.orders_all("CLOSED")
    opens, opens_ok = t.orders_all("OPEN")
    try:
        idx = at.index_daily(t)
        cal = [d.astimezone(at.NY).date().isoformat() for d in idx.index] if idx is not None else []
    except Exception as e:                                   # noqa: BLE001
        print(f"  (달력용 지수 일봉 조회 실패: {e} — 주말만 빼는 근사로 계산한다)")
        cal = []
    return {"holdings": {s: {k: h[k] for k in ("quantity", "avg_price", "last_price",
                                               "market_value", "pnl_pct")}
                         for s, h in holdings.items()},
            "closed": slim(closed), "closed_complete": closed_ok,
            "open": slim(opens), "open_complete": opens_ok,
            "calendar": cal, "captured_at": datetime.datetime.now().isoformat(timespec="seconds")}


def collect_fake():
    """--selftest: 두 가지 실패 원인을 일부러 심어 프로브가 분류하는지 본다."""
    def row(sym, side, qty, ts, i):
        return {"orderId": f"O{i}", "clientOrderId": f"c{i}", "symbol": sym, "side": side,
                "status": "FILLED", "quantity": str(qty), "orderedAt": ts,
                "execution": {"filledQuantity": str(qty)}}
    closed = [row("GOOD", "BUY", 4, "2026-08-14T22:40:00+09:00", 1),
              row("PART", "BUY", 1, "2026-09-01T22:40:00+09:00", 2)]
    closed += [row("NOISE", "BUY", 1, f"2026-07-{i % 28 + 1:02d}T22:40:00+09:00", 100 + i)
               for i in range(40)]                            # 20건 창을 넘기는 잡음
    cal = []
    d = datetime.date(2026, 6, 1)
    while d <= datetime.datetime.now().date():
        if d.weekday() < 5:
            cal.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return {"holdings": {
                "GOOD": {"quantity": 4.0, "avg_price": 38.0, "last_price": 36.4,
                         "market_value": 145.6, "pnl_pct": -4.2},
                "PART": {"quantity": 3.0, "avg_price": 100.0, "last_price": 99.0,
                         "market_value": 297.0, "pnl_pct": -1.0},   # 이력엔 1주뿐
                "OFFHR": {"quantity": 1.0, "avg_price": 32.7, "last_price": 31.2,
                          "market_value": 31.2, "pnl_pct": -4.6}},  # 이력에 아예 없음
            "closed": closed, "closed_complete": True,
            "open": [], "open_complete": True, "calendar": cal,
            "captured_at": "selftest"}


class Replay:
    """수집된 스냅샷을 브로커처럼 보이게 하는 어댑터 (네트워크 없음)."""

    def __init__(self, snap):
        self.snap = snap

    def orders_all(self, status="OPEN", **kw):
        if status == "CLOSED":
            return self.snap["closed"], self.snap["closed_complete"]
        return self.snap["open"], self.snap["open_complete"]

    def orders(self, status="OPEN", **kw):
        rows, _ = self.orders_all(status)
        return {"orders": rows, "nextCursor": None, "hasNext": False}


def analyse(snap):
    import autotrade as at

    at.DB_PATH = pathlib.Path(tempfile.mkdtemp(prefix="probe-")) / "scratch.db"
    at.initialize_db()
    at.TRADING_DAYS = [datetime.date.fromisoformat(d) for d in snap["calendar"]]
    holdings = {s: dict(h) for s, h in snap["holdings"].items()}
    entries = at.position_entry_map(Replay(snap), holdings)
    unver = set(at.ENTRY_UNVERIFIED)

    hist_syms, hist_qty = set(), {}
    for r in snap["closed"] + snap["open"]:
        f = at.filled_qty(r)
        if f <= 0:
            continue
        hist_syms.add(r["symbol"])
        sign = 1 if str(r.get("side", "")).upper() == "BUY" else -1
        hist_qty[r["symbol"]] = hist_qty.get(r["symbol"], 0.0) + sign * f

    lines = []
    w = lines.append
    w(f"수집 시각: {snap['captured_at']}")
    w(f"체결 이력: CLOSED {len(snap['closed'])}건(완전={snap['closed_complete']}) · "
      f"OPEN {len(snap['open'])}건(완전={snap['open_complete']}) · "
      f"거래일 달력 {len(snap['calendar'])}일")
    w(f"MAX_HOLD_DAYS={at.MAX_HOLD_DAYS} · HOLD_EXTEND_TOP={at.HOLD_EXTEND_TOP}")
    w("")
    w(f"{'종목':8}{'보유수량':>12}{'복원수량':>12}  {'진입일':12}{'보유일':>7}{'만기까지':>9}  판정")
    w("-" * 92)
    causes = {}
    for sym in sorted(holdings):
        held = float(holdings[sym]["quantity"])
        got = hist_qty.get(sym, 0.0)
        e = entries.get(sym)
        days = at.trading_days_since(e) if e else None
        if sym not in unver:
            verdict, cause = "OK", None
        elif not snap["closed_complete"] or not snap["open_complete"]:
            verdict, cause = "UNVERIFIED", "목록 불완전(페이지를 다 못 읽음)"
        elif sym not in hist_syms:
            verdict, cause = "UNVERIFIED", "이력에 없음 — Open API 미지원 호가(시간외 종가 등) 매수 의심"
        else:
            verdict, cause = "UNVERIFIED", f"수량 불일치 (이력 {got:g} ≠ 보유 {held:g})"
        if cause:
            causes.setdefault(cause, []).append(sym)
        left = "" if days is None else max(0, at.MAX_HOLD_DAYS - days)
        w(f"{sym:8}{held:>12.6f}{got:>12.6f}  {str(e)[:10]:12}"
          f"{'' if days is None else days:>7}{left:>9}  {verdict}"
          + (f" — {cause}" if cause else ""))
    w("")
    if not unver:
        w("판정: PASS — 전 종목의 진입일이 체결 이력으로 검증됐다. 만기 청산이 정상 동작한다.")
    else:
        w(f"판정: FAIL — {len(unver)}/{len(holdings)} 종목 미검증. 이 종목들은 만기 청산에서 "
          "제외된다(손절·노출 축소는 그대로 적용).")
        for cause, syms in causes.items():
            w(f"  · {cause}: {', '.join(syms)}")
        w("")
        w("다음 행동:")
        if any("목록 불완전" in c for c in causes):
            w("  1) orders_all 의 max_pages 를 올린다 (현재 20페이지×100건=2000건).")
        if any("이력에 없음" in c for c in causes):
            w("  2) 토스 앱 거래내역에서 해당 종목의 최초 매수일을 확인한다. Open API 로는")
            w("     복원할 수 없는 주문이므로(스펙상 조회 불가), 운영자 입력이 유일한 경로다.")
            w("     → 감사 가능한 진입일 오버라이드가 필요하면 말해 달라 (아직 만들지 않았다).")
        if any("수량 불일치" in c for c in causes):
            w("  3) 이력 일부만 보이는 경우다. 같은 종목을 앱에서 추가 매수했는지 확인한다.")
    return "\n".join(lines), (not unver)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="가짜 데이터로 프로브 자체 점검")
    ap.add_argument("--replay", help="저장된 스냅샷으로 재분석 (네트워크 없음)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    # autotrade 를 import 하면 logging.FileHandler 가 autotrade.log 를 연다. 이 핸들러는
    # builtins.open 가드를 거치지 않으므로, 기록이 생기기 전에 떼어 낸다 (운영 로그 무오염).
    import logging
    import autotrade  # noqa: F401
    for h in list(logging.getLogger().handlers):
        if isinstance(h, logging.FileHandler):
            logging.getLogger().removeHandler(h)
            h.close()

    if a.selftest:
        snap = collect_fake()
    elif a.replay:
        snap = json.loads(pathlib.Path(a.replay).read_text(encoding="utf-8"))
    else:
        readonly_guard()
        print("실계좌 읽기 전용 조회 중 (주문 없음)…")
        snap = collect_live()

    report, ok = analyse(snap)
    print(report)
    if not (a.no_save or a.replay or a.selftest):   # 실계좌 수집일 때만 저장
        OUT.mkdir(parents=True, exist_ok=True)
        RAW.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
        REPORT.write_text(report, encoding="utf-8")
        print(f"\n저장: {RAW.relative_to(ROOT)} (식별자 마스킹됨) · {REPORT.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
