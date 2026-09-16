"""매도 실행 엔진 시나리오 — 가짜 브로커·가짜 시계·임시 SQLite 만 쓴다.

    python research/sell_execution_audit/scenarios.py          # 사람이 읽는 표
    python research/sell_execution_audit/scenarios.py --json   # A/B 드라이버용

**이 파일은 전략을 평가하지 않는다.** 검증 대상은 실행 엔진뿐이다:
주문 의도 생성(INTENT) → 제출(SUBMIT) → 접수(ACK) → 체결(FILL) 의 각 경계에서
상태를 혼동하지 않는가, 모르는 것을 아는 척하지 않는가.

격리: research.isolation.guard() — 자격증명 제거, 운영 DB/로그/토스 호스트 차단.
각 시나리오는 예외를 잡아 FAIL 로 돌려준다. 그래야 **수정 전 버전**(속성 자체가
없는 상태)에서도 크래시 없이 A/B 비교가 된다.
"""
import argparse
import datetime
import json
import pathlib
import sqlite3
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.isolation import guard                      # noqa: E402

guard()

import autotrade as at                                    # noqa: E402
from toss import TossError                                # noqa: E402

# 운영 DB 를 절대 건드리지 않는다 — import 직후 임시 경로로 돌린다.
_TMP = pathlib.Path(tempfile.mkdtemp(prefix="sellaudit-"))
assert at.DB_PATH == ROOT / "trading_decisions.db", at.DB_PATH
at.DB_PATH = _TMP / "scratch.db"
at.initialize_db()

# 전략값은 시나리오 안에서만 고정한다 (.env·운영 기본값을 바꾸는 게 아니다).
at.MAX_POSITIONS, at.MAX_POSITION_PCT = 10, 15
at.CASH_RESERVE_PCT, at.MIN_ORDER_USD = 10, 5
at.VOL_TARGET_PCT, at.HOLD_EXTEND_TOP, at.HOLD_EXTEND_SHADOW = 0, 0, 0
at.MAX_HOLD_DAYS, at.STOP_LOSS_PCT, at.MOMENTUM_EXIT = 20, 15, False

NOW = datetime.datetime.now(at.KST)
REGULAR = (NOW - datetime.timedelta(hours=1), NOW - datetime.timedelta(hours=1),
           NOW + datetime.timedelta(hours=3), NOW + datetime.timedelta(hours=4))
OFF = (NOW - datetime.timedelta(hours=1), NOW + datetime.timedelta(hours=6),
       NOW + datetime.timedelta(hours=9), NOW + datetime.timedelta(hours=10))


# ────────────────────────────── 가짜 브로커 ──────────────────────────────
class FakeBroker:
    """주문 생명주기를 가진 가짜 브로커.

    OPEN 목록과 CLOSED 목록을 따로 들고, create/cancel/fill 로 상태를 옮긴다.
    `fail` 에 넣은 이름의 호출만 실패한다. `post_reaches_broker=True` 면
    create_order 가 **브로커에는 기록된 뒤** 예외를 던진다 (POST 타임아웃 재현).
    """

    def __init__(self, holdings=None, cash=0.0, opens=(), closed=(), sellable=None,
                 fail=(), post_reaches_broker=False, session_open=True):
        self.h = dict(holdings or {})
        self.cash = cash
        self.open_rows = [dict(o) for o in opens]
        self.closed_rows = [dict(o) for o in closed]
        self._sellable = sellable
        self.fail = set(fail)
        self.post_reaches_broker = post_reaches_broker
        self.session_open = session_open
        self.calls = []                      # 상태 변경 호출만 기록

    # ---- 조회 ----
    def holdings(self, symbol=None):
        return {"items": [
            {"symbol": s, "name": s, "currency": "USD", "quantity": str(v["quantity"]),
             "averagePurchasePrice": str(v["avg_price"]), "lastPrice": str(v["last_price"]),
             "marketValue": {"amount": str(v["quantity"] * v["last_price"])},
             "profitLoss": {"rate": str(v["last_price"] / v["avg_price"] - 1)}}
            for s, v in self.h.items()]}

    def buying_power(self, currency="USD"):
        return {"cashBuyingPower": str(self.cash)}

    def orders(self, status="OPEN", symbol=None):
        if "orders" in self.fail:
            raise TossError(503, "retry-exhausted", "주문 조회 실패")
        rows = self.open_rows if status == "OPEN" else self.closed_rows
        if symbol:
            rows = [r for r in rows if r.get("symbol") == symbol]
        return {"orders": [dict(r) for r in rows]}

    def sellable_quantity(self, symbol):
        if "sellable" in self.fail:
            raise TossError(503, "retry-exhausted", "매도가능 조회 실패")
        if self._sellable is None:
            return {}
        return {"sellableQuantity": self._sellable}

    def us_market_calendar(self, date=None):
        if not self.session_open:
            return {"today": {}}
        n = datetime.datetime.now(at.NY)
        return {"today": {"regularMarket": {
            "startTime": (n - datetime.timedelta(hours=1)).isoformat(),
            "endTime": (n + datetime.timedelta(hours=3)).isoformat()}}}

    # ---- 상태 변경 ----
    def create_order(self, symbol, side, order_type, quantity=None, price=None,
                     order_amount=None, time_in_force=None, client_order_id=None):
        self.calls.append(("create", symbol, side, quantity, order_amount, client_order_id))
        row = {"orderId": f"B{len(self.calls)}", "clientOrderId": client_order_id,
               "symbol": symbol, "side": side, "status": "OPEN",
               "quantity": quantity, "execution": {"filledQuantity": "0"}}
        if "create" in self.fail:
            if self.post_reaches_broker:     # 브로커엔 닿았고 응답만 못 받았다
                self.open_rows.append(row)
            raise TossError(504, "timeout", "게이트웨이 타임아웃")
        self.open_rows.append(row)
        return {"orderId": row["orderId"], "clientOrderId": client_order_id}

    def cancel_order(self, order_id):
        self.calls.append(("cancel", order_id))
        if "cancel" in self.fail:
            raise TossError(400, "already-filled", "이미 체결되어 취소 불가")
        for r in list(self.open_rows):
            if r.get("orderId") == order_id:
                self.open_rows.remove(r)
                self.closed_rows.append({**r, "status": "CANCELED"})
        return {"orderId": order_id}

    def modify_order(self, *a, **k):
        self.calls.append(("modify", a, k))
        return {}

    # ---- 테스트 편의 ----
    def n(self, kind):
        return len([c for c in self.calls if c[0] == kind])

    def last_qty(self):
        c = [x for x in self.calls if x[0] == "create"]
        return c[-1][3] if c else None


def hold(qty, avg, last):
    return {"name": "x", "quantity": qty, "avg_price": avg, "last_price": last,
            "market_value": qty * last, "pnl_pct": round((last / avg - 1) * 100, 2)}


def opn(side, qty, filled=0.0, oid="o1"):
    return {"orderId": oid, "side": side.upper(), "quantity": qty, "filled": filled,
            "remaining": qty - filled, "price": None, "state": "ACKNOWLEDGED"}


def acct(cash, holdings=None, open_by=None, unknown=False):
    """account_state 가 만드는 모양. 수정 전 버전은 open_by_symbol 을 모르지만,
    그 버전의 validate_orders 는 open_orders 만 보므로 그대로 동작한다."""
    h = holdings or {}
    by = dict(open_by or {})
    return {"cash": cash, "holdings": h, "open_orders": sorted(by),
            "open_by_symbol": by, "open_unknown": unknown,
            "total_value": cash + sum(x["market_value"] for x in h.values())}


def dec(sym, decision, price, ret20, account):
    f, tier = at.momentum_tier(ret20)
    h = account["holdings"].get(sym, {})
    return {"symbol": sym, "decision": decision, "percentage": 80, "reason": "t",
            "status": {"current_price": price, "stock_balance": h.get("quantity", 0.0),
                       "avg_buy_price": h.get("avg_price", 0.0), "pnl_pct": h.get("pnl_pct"),
                       "ret_20d_pct": ret20, "momentum_tier": tier, "size_factor": f}}


def vo(account, decisions=None, plan=None, entries=None, session=REGULAR, **kw):
    return at.validate_orders({"orders": plan or [], "summary": ""}, decisions or {},
                              account, session, entries or {}, **kw)


def tmpdb():
    p = pathlib.Path(tempfile.mkdtemp(dir=_TMP)) / "t.db"
    at.DB_PATH = p
    at.initialize_db()
    return p


def statuses(db):
    with sqlite3.connect(db) as c:
        return [r[0] for r in c.execute("SELECT status FROM orders ORDER BY id")]


REAL_TRADING_DAYS_SINCE = at.trading_days_since     # 스텁을 씌우기 전의 진짜 함수


def days_stub():
    """entries 에 '보유 거래일수'를 정수로 직접 넣게 한다 (달력·현재시각 의존 제거)."""
    at.trading_days_since = lambda ts, calendar=None: ts if isinstance(ts, int) else None


days_stub()

SCEN = []


def scenario(sid, title, group):
    def deco(fn):
        SCEN.append((sid, title, group, fn))
        return fn
    return deco


# ═══════════════ A. 미체결이 있는 종목의 강제 매도 의도 (INTENT 단계) ═══════════════
@scenario("A1", "만기 초과 + 기존 미체결 매수 → 매도 의도가 남고 cancel_first", "A")
def _a1():
    a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("BUY", 3)]})
    out, skip = vo(a, entries={"AAPL": 20})
    assert len(out) == 1 and out[0]["side"] == "sell", f"out={out} skip={_s(skip)}"
    assert out[0].get("cancel_first") is True, out[0]
    return f"매도 의도 1건, cancel_first=True, qty={out[0]['quantity']}"


@scenario("A2", "만기 초과 + 목표를 덮는 미체결 매도 → 취소·재발행 없음", "A")
def _a2():
    a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("SELL", 10)]})
    out, skip = vo(a, entries={"AAPL": 20})
    assert not out, f"불필요한 재발행: {out}"
    assert any("PENDING_SELL_SUFFICIENT" in s["skipped"] for s in skip), _s(skip)
    return "재발행 0건 + 사유 PENDING_SELL_SUFFICIENT"


@scenario("A3", "만기 초과 + 목표보다 적은 미체결 매도 → 의도 유지", "A")
def _a3():
    a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("SELL", 4)]})
    out, skip = vo(a, entries={"AAPL": 20})
    assert len(out) == 1 and out[0]["cancel_first"] is True, f"out={out} skip={_s(skip)}"
    return "취소 후 재발행 (실제 수량은 제출 단계에서 sellable 로 clamp)"


@scenario("A4", "노출 축소 + 기존 미체결 주문 → 축소 매도가 남는다", "A")
def _a4():
    real = at.exposure_cap
    at.exposure_cap = lambda verbose=False: 0.5
    try:
        a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("BUY", 1)]})
        out, skip = vo(a, entries={"AAPL": 1})
        assert len(out) == 1 and "상한" in out[0]["reason"], f"out={out} skip={_s(skip)}"
        return f"축소 매도 1건 ${out[0]['amount_usd']}"
    finally:
        at.exposure_cap = real


@scenario("A5", "노출 축소가 이미 예약된 매도를 중복해서 또 깎지 않는다", "A")
def _a5():
    real = at.exposure_cap
    at.exposure_cap = lambda verbose=False: 0.5
    try:
        a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("SELL", 10)]})
        out, _ = vo(a, entries={"AAPL": 1})
        assert not out, f"예약분을 또 팔았다: {out}"
        return "전량이 이미 매도 주문으로 떠 있어 추가 축소 0건"
    finally:
        at.exposure_cap = real


@scenario("A6", "재량(LLM) 매도는 미체결이 있으면 예전처럼 미룬다", "A")
def _a6():
    old = at.STOP_LOSS_PCT
    at.STOP_LOSS_PCT = 0
    try:
        a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("BUY", 1)]})
        out, skip = vo(a, {"AAPL": dec("AAPL", "sell", 105, 25, a)},
                       plan=[{"symbol": "AAPL", "side": "sell", "sell_pct": 50, "reason": "t"}])
        assert not out and any("미체결" in s["skipped"] for s in skip), f"{out} {_s(skip)}"
        return "재량 매도는 보류 (경합을 만들 이유가 없다)"
    finally:
        at.STOP_LOSS_PCT = old


# ═══════════════ B. 수량 UNKNOWN (SUBMIT 단계) ═══════════════
SELL = {"symbol": "AAPL", "side": "sell", "quantity": 10.0, "cancel_first": True,
        "amount_usd": 1000.0, "price": 100.0, "reason": "손절", "whole": False}


def _submit(b, o=None):
    """place_order 결과를 (결과종류, 제출수량) 으로."""
    try:
        at.place_order(b, dict(o or SELL))
        return "SUBMITTED", b.last_qty()
    except getattr(at, "Deferred", ()) as e:
        return "DEFERRED", str(e)
    except TossError as e:
        return "TOSSERROR", str(e)


@scenario("B1", "매도가능 수량 조회 실패 → DEFERRED, 제출 0건", "B")
def _b1():
    b = FakeBroker(opens=[{"orderId": "o1", "symbol": "AAPL", "side": "BUY", "quantity": "3"}],
                   fail={"sellable"})
    kind, detail = _submit(b)
    assert kind == "DEFERRED", f"{kind}: {detail} (제출수량={b.last_qty()})"
    assert b.n("create") == 0, b.calls
    return f"DEFERRED, create 0회 — {detail}"


@scenario("B2", "미체결 조회 실패 → DEFERRED (취소 여부를 모르는 채 제출 금지)", "B")
def _b2():
    b = FakeBroker(fail={"orders"})
    kind, detail = _submit(b)
    assert kind == "DEFERRED", f"{kind}: {detail} (제출수량={b.last_qty()})"
    assert b.n("create") == 0, b.calls
    return f"DEFERRED, create 0회 — {detail}"


@scenario("B3", "매도 가능 수량 0 → 제출 없음", "B")
def _b3():
    b = FakeBroker(sellable=0)
    kind, detail = _submit(b)
    assert b.n("create") == 0, b.calls
    assert kind == "DEFERRED", f"{kind}: {detail}"
    return f"제출 0건 — {detail}"


@scenario("B4", "취소 중 부분 체결 → 남은 목표 수량만 제출", "B")
def _b4():
    b = FakeBroker(opens=[{"orderId": "o1", "symbol": "AAPL", "side": "SELL",
                           "quantity": "4"}], sellable=6.0)
    kind, qty = _submit(b)
    assert kind == "SUBMITTED" and qty == "6", f"{kind} qty={qty} calls={b.calls}"
    return "계획 10 → 재조회 6 → 6 제출 (취소 1회 선행)"


@scenario("B5", "확인된 수량이 계획보다 많아도 계획을 넘지 않는다", "B")
def _b5():
    b = FakeBroker(sellable=99.0)
    kind, qty = _submit(b)
    assert kind == "SUBMITTED" and qty == "10", f"{kind} qty={qty}"
    return "sellable 99 이지만 계획 10 만 제출"


@scenario("B6", "미체결 상태 자체가 UNKNOWN 이면 그 사이클 제출 보류", "B")
def _b6():
    a = acct(100, {"AAPL": hold(10, 100, 80)}, unknown=True)
    out, skip = vo(a, {"AAPL": dec("AAPL", "hold", 80, 25, a)})
    assert not out, f"UNKNOWN 인데 제출했다: {out}"
    assert skip and any("DEFERRED_OPEN_UNKNOWN" in s["skipped"] for s in skip), _s(skip)
    return "주문 0건 + 의도는 DEFERRED_OPEN_UNKNOWN 으로 기록"


# ═══════════════ C. 대사 (ACK vs NOT_FOUND vs UNKNOWN) ═══════════════
@scenario("C1", "대사 조회 실패는 NOT_FOUND 가 아니라 UNKNOWN", "C")
def _c1():
    b = FakeBroker(fail={"orders"})
    r = at.find_order(b, "coid-1")
    assert isinstance(r, tuple), f"find_order 가 단일값을 돌려준다({r!r}) — 실패와 부재를 구분 못 한다"
    found, known = r
    assert found is None and known is False, r
    return "(None, known=False)"


@scenario("C2", "CLOSED 에서 찾으면 known=True", "C")
def _c2():
    b = FakeBroker(closed=[{"orderId": "B1", "clientOrderId": "c", "status": "FILLED",
                            "quantity": "3", "execution": {"filledQuantity": "3"}}])
    r = at.find_order(b, "c")
    assert isinstance(r, tuple), f"find_order 가 단일값: {r!r}"
    found, known = r
    assert known is True and found is not None, r
    return f"orderId={found.get('orderId')}"


@scenario("C3", "CLOSED 의 FILLED / CANCELED / REJECTED / 부분체결 구분", "C")
def _c3():
    f = at.order_state
    cases = {
        "FILLED": {"status": "FILLED", "quantity": "3", "execution": {"filledQuantity": "3"}},
        "CANCELED": {"status": "CANCELED", "quantity": "3",
                     "execution": {"filledQuantity": "0"}},
        "REJECTED": {"status": "REJECTED", "quantity": "3"},
        "PARTIALLY_FILLED": {"status": "OPEN", "quantity": "5",
                             "execution": {"filledQuantity": "2"}},
        "UNKNOWN": {"status": "CLOSED", "quantity": "3",
                    "execution": {"filledQuantity": "0"}},
    }
    got = {k: f(v) for k, v in cases.items()}
    assert got == {k: k for k in cases}, got
    assert f(None) == "UNKNOWN", f(None)
    return "5개 상태 + None→UNKNOWN 모두 일치"


@scenario("C4", "부분 체결 뒤 취소는 CANCELED (남은 목표는 다음 사이클)", "C")
def _c4():
    st = at.order_state({"status": "CANCELED", "quantity": "10",
                         "execution": {"filledQuantity": "4"}})
    assert st == "CANCELED", st
    return "CANCELED (체결 4/10 은 계좌 수량으로 반영된다)"


# ═══════════════ D. 제출 결과의 구조화 (SUBMIT → ACK) ═══════════════
PLAN = [{"symbol": "AAPL", "side": "sell", "quantity": 10.0, "amount_usd": 1000.0,
         "price": 100.0, "reason": "만기", "cancel_first": False},
        {"symbol": "MSFT", "side": "sell", "quantity": 1.0, "amount_usd": 100.0,
         "price": 100.0, "reason": "만기", "cancel_first": False}]


@scenario("D1", "place_all 이 종목별 결과를 돌려준다", "D")
def _d1():
    tmpdb()
    b = FakeBroker()
    res = at.place_all(b, 1, PLAN, dry=False)
    assert isinstance(res, dict), f"place_all 이 {type(res).__name__} 을 돌려준다 — 종목별 결과 없음"
    assert set(res) == {"AAPL", "MSFT"}, res
    return repr(res)


@scenario("D2", "거절은 성공(ACKNOWLEDGED)이 아니다", "D")
def _d2():
    tmpdb()
    b = FakeBroker(fail={"create"})
    res = at.place_all(b, 1, PLAN, dry=False)
    assert isinstance(res, dict), "place_all 이 결과를 안 돌려준다"
    assert all(v != "ACKNOWLEDGED" for v in res.values()), res
    return repr(res)


@scenario("D3", "POST 타임아웃이지만 브로커는 접수 → 중복 제출 금지", "D")
def _d3():
    db = tmpdb()
    b = FakeBroker(fail={"create"}, post_reaches_broker=True)
    res = at.place_all(b, 1, [PLAN[0]], dry=False)
    assert b.n("create") == 1, f"재제출했다: {b.calls}"
    st = statuses(db)
    assert any("ACK" in s or "대사 확인" in s for s in st), st
    return f"create 1회, 상태={st}"


@scenario("D4", "대사 API 도 실패 → UNKNOWN (NOT_FOUND 로 간주 금지)", "D")
def _d4():
    db = tmpdb()
    b = FakeBroker(fail={"create", "orders"})
    res = at.place_all(b, 1, [PLAN[0]], dry=False)
    st = statuses(db)
    assert isinstance(res, dict) and res.get("AAPL") == "UNKNOWN", f"res={res} st={st}"
    assert any("UNKNOWN" in s for s in st), st
    assert b.n("create") == 1, b.calls
    return f"res={res}, DB={st}"


@scenario("D5", "위험관리 제출 실패가 '이미 주문함'으로 제외되지 않는다", "D")
def _d5():
    tmpdb()
    b = FakeBroker(fail={"create"})
    res = at.place_all(b, 1, PLAN, dry=False)
    assert isinstance(res, dict), "place_all 결과가 없어 run_cycle 이 계획만으로 제외한다"
    ok = {s for s, v in res.items()
          if v in ("ACKNOWLEDGED", "FILLED", "PARTIALLY_FILLED", "DRY_RUN", "UNKNOWN")}
    assert ok == set(), f"거절인데 성공으로 분류: {res}"
    return f"제외 대상 {sorted(ok)} (전부 실패라 본 패스가 다시 본다)"


@scenario("D6", "DRY_RUN 에서 create/cancel/modify 호출 0회", "D")
def _d6():
    db = tmpdb()
    b = FakeBroker(opens=[{"orderId": "o1", "symbol": "AAPL", "side": "BUY", "quantity": "3"}],
                   sellable=5.0)
    plan = [dict(PLAN[0], cancel_first=True), dict(PLAN[1])]
    at.place_all(b, 1, plan, dry=True)
    assert b.calls == [], b.calls
    assert len(statuses(db)) == 2, statuses(db)
    return "상태변경 호출 0회, 의도 2건은 DB 에 기록"


@scenario("D7", "제출 직전 세션 재검증 — 휴장이면 제출 0건", "D")
def _d7():
    db = tmpdb()
    b = FakeBroker(session_open=False)
    at.place_all(b, 1, PLAN, dry=False)
    assert b.n("create") == 0, b.calls
    st = statuses(db)
    assert all("DEFER" in s.upper() for s in st), st
    return f"create 0회, 상태={st}"


@scenario("D8", "toss._call — 상태 변경(POST)은 연결 오류에 재전송하지 않는다", "D")
def _d8():
    import requests
    import toss as T

    class Sess:
        def __init__(self):
            self.n = 0

        def request(self, method, url, **kw):
            self.n += 1
            raise requests.ConnectionError("연결 끊김")

    c = T.TossClient.__new__(T.TossClient)
    c._s = Sess()
    c.account_seq = "X"
    c._token, c._token_exp = "tok", 9e18
    try:
        c._call("POST", "/api/v1/orders", json={"a": 1}, account=True)
        raise AssertionError("예외 없이 끝났다")
    except TossError as e:
        pass
    posts = c._s.n
    c._s.n = 0
    try:
        c._call("GET", "/api/v1/orders", account=True)
    except TossError:
        pass
    gets = c._s.n
    assert posts == 1, f"POST 를 {posts}회 보냈다 — 중복 주문 위험"
    assert gets > 1, f"GET 재시도가 사라졌다({gets}회) — 읽기 견고성 회귀"
    return f"POST {posts}회(재전송 없음) / GET {gets}회(재시도 유지)"


# ═══════════════ J. 현금 교착 산수 (run 63 스냅샷) ═══════════════
# 출처: 운영 DB 사본의 runs(id=63) 과 trading_decisions(run_id=63, stock_balance>0).
# 읽기 전용으로 읽어 여기 상수로 박았다 — 이 파일은 운영 DB 를 열지 않는다.
RUN63_CASH = 0.01
RUN63_TOTAL_REPORTED = 568.83          # runs.total_value (account_state 가 그 시점에 계산한 값)
RUN63_HOLDINGS = {                     # (수량, 평단, 현재가) — trading_decisions 기록
    "MRNA": (0.408386, 146.8216, 142.42), "MU": (0.058791, 1035.866901, 931.355),
    "SMCI": (4.0, 38.09, 36.43), "MSTR": (1.0, 142.97, 129.51),
    "TEAM": (0.138645, 178.873497, 189.53), "KDP": (1.0, 32.7435, 31.25),
    "TSLA": (0.344463, 368.5729, 357.79)}


def _run63(extra_cash=0.0, drop=()):
    h = {s: hold(q, a, p) for s, (q, a, p) in RUN63_HOLDINGS.items() if s not in drop}
    cash = RUN63_CASH + extra_cash
    return acct(cash, h)


@scenario("J1", "run 63 가용예산 = cash − total_value × CASH_RESERVE_PCT/100", "J")
def _j1():
    a = _run63()
    reserve = a["total_value"] * at.CASH_RESERVE_PCT / 100
    budget = a["cash"] - reserve
    # 재구성 총자산과 기록된 총자산의 차이 (시세 스냅샷 시점 차)
    gap = abs(a["total_value"] - RUN63_TOTAL_REPORTED)
    assert gap < 0.5, f"재구성 총자산 {a['total_value']} vs 기록 {RUN63_TOTAL_REPORTED}"
    assert budget < at.MIN_ORDER_USD, budget
    return (f"총자산 ${a['total_value']:.2f}(기록 ${RUN63_TOTAL_REPORTED}, 차 ${gap:.2f}) · "
            f"유지선 ${reserve:.2f} · 가용 ${budget:.2f} < 최소 ${at.MIN_ORDER_USD}")


@scenario("J2", "매도 '체결' 후 예산 — 어떤 매도가 교착을 실제로 푸는가", "J")
def _j2():
    rows = []
    for sym in RUN63_HOLDINGS:
        q, _, px = RUN63_HOLDINGS[sym]
        a = _run63(extra_cash=q * px, drop=(sym,))
        b = a["cash"] - a["total_value"] * at.CASH_RESERVE_PCT / 100
        rows.append((sym, round(q * px, 2), round(b, 2), b >= at.MIN_ORDER_USD))
    rows.sort(key=lambda r: -r[1])
    solved = [r[0] for r in rows if r[3]]
    assert solved, rows
    return " · ".join(f"{s}(${v})→${b}{'✔' if ok else '✘'}" for s, v, b, ok in rows)


@scenario("J3", "total_value 는 USD 보유만 — 다른 통화 현금·주식은 빠진다", "J")
def _j3():
    b = FakeBroker(cash=100.0)
    b.h = {}
    raw = b.holdings()
    raw["items"].append({"symbol": "005930", "name": "x", "currency": "KRW",
                         "quantity": "10", "averagePurchasePrice": "70000",
                         "lastPrice": "80000", "marketValue": {"amount": "800000"},
                         "profitLoss": {"rate": "0.14"}})
    b.holdings = lambda symbol=None: raw
    a = at.account_state(b)
    assert a["total_value"] == 100.0 and not a["holdings"], a
    return ("KRW 보유는 holdings·total_value 에서 제외된다 — 원화 자산이 있으면 "
            "CASH_RESERVE_PCT·MAX_POSITION_PCT 의 분모가 실제보다 작다")


@scenario("J4", "cash 는 cashBuyingPower 그대로 — 미체결 매수 금액을 코드가 따로 빼지 않는다", "J")
def _j4():
    b = FakeBroker(cash=500.0, opens=[{"orderId": "o1", "symbol": "AAPL", "side": "BUY",
                                       "quantity": "1", "price": "200"}])
    a = at.account_state(b)
    assert a["cash"] == 500.0, a
    return ("cash=cashBuyingPower(500). 이 값이 미체결 매수를 이미 차감한 값인지는 "
            "**API 계약 미확인** — 아니라면 예산이 과대계상된다")


# ═══════════════ E. 진입일 복원 ═══════════════
def closed_row(sym, side, qty, ts, oid="x"):
    return {"orderId": oid, "symbol": sym, "side": side, "orderedAt": ts,
            "status": "FILLED", "quantity": str(qty),
            "execution": {"filledQuantity": str(qty)}}


def entry_map(b, holdings):
    try:
        return at.position_entry_map(b, holdings)
    except TypeError:                      # 수정 전: holdings 인자를 받지 않는다
        return at.position_entry_map(b)


@scenario("E1", "부분 매도 후 최초 진입일 유지", "E")
def _e1():
    b = FakeBroker(closed=[closed_row("AAPL", "BUY", 10, "2026-01-02T10:00:00+00:00", "1"),
                           closed_row("AAPL", "SELL", 4, "2026-01-08T10:00:00+00:00", "2")])
    e = entry_map(b, {"AAPL": hold(6, 100, 100)})
    assert str(e.get("AAPL")).startswith("2026-01-02"), e
    return f"AAPL={e['AAPL']}"


@scenario("E2", "완전 청산 후 재매수 → 진입일 갱신", "E")
def _e2():
    b = FakeBroker(closed=[closed_row("AAPL", "BUY", 10, "2026-01-02T10:00:00+00:00", "1"),
                           closed_row("AAPL", "SELL", 10, "2026-01-08T10:00:00+00:00", "2"),
                           closed_row("AAPL", "BUY", 5, "2026-02-02T10:00:00+00:00", "3")])
    e = entry_map(b, {"AAPL": hold(5, 100, 100)})
    assert str(e.get("AAPL")).startswith("2026-02-02"), e
    return f"AAPL={e['AAPL']}"


@scenario("E3", "추가 매수는 시계를 리셋하지 않는다", "E")
def _e3():
    b = FakeBroker(closed=[closed_row("AAPL", "BUY", 5, "2026-01-02T10:00:00+00:00", "1"),
                           closed_row("AAPL", "BUY", 5, "2026-01-20T10:00:00+00:00", "2")])
    e = entry_map(b, {"AAPL": hold(10, 100, 100)})
    assert str(e.get("AAPL")).startswith("2026-01-02"), e
    return f"AAPL={e['AAPL']}"


@scenario("E4", "로컬 대체 경로가 submitted 를 진입으로 쓰지 않는다", "E")
def _e4():
    db = tmpdb()
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO orders (run_id,timestamp,symbol,side,status,quantity) "
                  "VALUES (1,?,?,?,?,?)",
                  ("2026-01-02T00:00:00+09:00", "AAPL", "buy", "submitted", 10))
    e = entry_map(None, {"AAPL": hold(10, 100, 100)})
    assert "AAPL" not in e, f"접수만 된 주문을 진입으로 썼다: {e}"
    return "진입일 없음 (체결 증거가 아니므로)"


@scenario("E5", "로컬 대체 경로가 submitted 매도로 진입일을 지우지 않는다", "E")
def _e5():
    db = tmpdb()
    with sqlite3.connect(db) as c:
        for ts, side, st, q in [("2026-01-02T00:00:00+09:00", "buy", "FILLED", 10),
                                ("2026-01-05T00:00:00+09:00", "sell", "submitted", 10)]:
            c.execute("INSERT INTO orders (run_id,timestamp,symbol,side,status,quantity) "
                      "VALUES (1,?,?,?,?,?)", (ts, "AAPL", side, st, q))
    e = entry_map(None, {"AAPL": hold(10, 100, 100)})
    assert str(e.get("AAPL")).startswith("2026-01-02"), \
        f"접수만 된 매도를 전량 청산으로 취급했다: {e}"
    return f"AAPL={e['AAPL']} (submitted 매도는 무시)"


@scenario("E6", "이력이 holdings 와 어긋나면 탐지하고 만기 판단에서 뺀다", "E")
def _e6():
    b = FakeBroker(closed=[closed_row("AAPL", "BUY", 3, "2026-01-02T10:00:00+00:00", "1")])
    e = entry_map(b, {"AAPL": hold(10, 100, 100), "MSFT": hold(1, 10, 10)})
    unv = getattr(at, "ENTRY_UNVERIFIED", None)
    assert unv is not None, "ENTRY_UNVERIFIED 가 없다 — 불완전 이력을 탐지하지 않는다"
    assert "AAPL" in unv and "MSFT" in unv, f"unverified={unv} entries={e}"
    assert "MSFT" not in e, e
    return f"unverified={sorted(unv)} (AAPL 수량불일치, MSFT 이력없음)"


@scenario("E7", "검증되지 않은 진입일로는 만기 청산하지 않는다", "E")
def _e7():
    unv = getattr(at, "ENTRY_UNVERIFIED", None)
    assert unv is not None, "ENTRY_UNVERIFIED 가 없다"
    a = acct(100, {"AAPL": hold(10, 100, 105)})
    unv.clear()
    out1, _ = vo(a, entries={"AAPL": 20})
    unv.add("AAPL")
    out2, _ = vo(a, entries={"AAPL": 20})
    unv.clear()
    assert out1 and not out2, f"검증={out1} 미검증={out2}"
    return "검증됨 → 매도 1건 / 미검증 → 0건"


# ═══════════════ F. 재시작 · 동시 실행 · 시간 버킷 ═══════════════
@scenario("F1", "재시작 후 같은 슬롯의 같은 의도 → 중복 제출 금지", "F")
def _f1():
    db = tmpdb()
    o = dict(PLAN[0])
    coid = at.intent_key(o)
    b1 = FakeBroker()
    at.place_all(b1, 1, [o], dry=False)
    # 프로세스가 죽었다 → 새 프로세스가 같은 의도를 다시 낸다
    b2 = FakeBroker(fail={"create"},
                    opens=[{"orderId": "REAL1", "clientOrderId": coid, "symbol": "AAPL",
                            "status": "OPEN", "quantity": "10",
                            "execution": {"filledQuantity": "0"}}])
    at.place_all(b2, 2, [o], dry=False)
    assert b2.n("create") == 1, f"재제출: {b2.calls}"
    st = statuses(db)
    assert any("대사 확인" in s for s in st), st
    return f"2회차는 대사로 접수 확인 후 종료 — 상태={st[-1]}"


@scenario("F2", "다른 프로세스가 같은 슬롯에 같은 의도 → 같은 clientOrderId", "F")
def _f2():
    t = datetime.datetime(2026, 3, 4, 10, 5, tzinfo=at.NY)
    o = {"symbol": "AAPL", "side": "sell"}
    assert at.intent_key(o, t) == at.intent_key(o, t), "키가 프로세스마다 다르다"
    t2 = datetime.datetime(2026, 3, 4, 10, 29, tzinfo=at.NY)
    assert at.intent_key(o, t) == at.intent_key(o, t2), "같은 슬롯인데 키가 다르다"
    return at.intent_key(o, t)


@scenario("F3", "30분 버킷 경계 — 다음 슬롯은 새 키 (중복 억제가 풀린다)", "F")
def _f3():
    o = {"symbol": "AAPL", "side": "sell"}
    a = at.intent_key(o, datetime.datetime(2026, 3, 4, 10, 29, tzinfo=at.NY))
    bkey = at.intent_key(o, datetime.datetime(2026, 3, 4, 10, 31, tzinfo=at.NY))
    assert a != bkey, (a, bkey)
    return f"{a} → {bkey} (브로커 중복거절에 기대지 못하는 구간)"


@scenario("F4", "다음 슬롯에서도 유효한 미체결 매도가 있으면 재제출하지 않는다", "F")
def _f4():
    """F3 의 구멍을 실제로 막는 것은 키가 아니라 **계좌 대사**다 (TEAM 09-09 재현)."""
    a = acct(100, {"TEAM": hold(0.09306, 180, 180)}, {"TEAM": [opn("SELL", 0.09306)]})
    out, skip = vo(a, entries={"TEAM": 20})
    assert not out, f"다음 슬롯에서 중복 제출: {out}"
    assert any("PENDING_SELL_SUFFICIENT" in s["skipped"] for s in skip), _s(skip)
    return "미체결 매도가 목표를 덮으므로 재제출 0건"


@scenario("F5", "같은 슬롯의 정당한 후속 매도까지 막지는 않는다", "F")
def _f5():
    """부분 체결로 목표가 남으면 같은 슬롯이어도 나가야 한다."""
    a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("SELL", 10, filled=7)]})
    out, skip = vo(a, entries={"AAPL": 20})
    assert len(out) == 1, f"잔량 3 인데 후속 매도가 막혔다: out={out} skip={_s(skip)}"
    return "미체결 잔량 3 < 목표 10 → cancel_first 로 재대사"


# ═══════════════ G. 체결 전 대금 재사용 ═══════════════
@scenario("G1", "매도를 '제출'했다고 슬롯·현금이 생기지 않는다", "G")
def _g1():
    a = acct(10, {s: hold(1, 100, 100) for s in ("A", "B", "C")})
    old = at.MAX_POSITIONS
    at.MAX_POSITIONS = 3
    try:
        ds = {s: dec(s, "hold", 100, 25, a) for s in ("A", "B", "C")}
        ds["NEW"] = dec("NEW", "buy", 50, 25, a)
        out, skip = vo(a, ds, plan=[{"symbol": "A", "side": "sell", "sell_pct": 100,
                                     "reason": "t"},
                                    {"symbol": "NEW", "side": "buy", "reason": "t"}])
        sides = [(o["symbol"], o["side"]) for o in out]
        assert sides == [("A", "sell")], f"{sides} / {_s(skip)}"
        return "매도 1건만 — 매수는 최대 종목 수·현금에서 막힌다"
    finally:
        at.MAX_POSITIONS = old


def _s(skip):
    return [s.get("skipped") for s in skip]


# ═══════════════ H. 만기 도달 (가짜 시계) ═══════════════
@scenario("H1", "가짜 달력으로 만기 도달 재현 (휴장일 반영)", "H")
def _h1():
    fn = REAL_TRADING_DAYS_SINCE
    try:
        # 2026-09-15 진입, 달력에서 09-17 을 휴장으로 뺀다
        cal = [datetime.date(2026, 9, d) for d in range(15, 31) if d not in (17, 19, 20)]
        cal += [datetime.date(2026, 10, d) for d in range(1, 16)
                if datetime.date(2026, 10, d).weekday() < 5]
        entry = "2026-09-15T13:30:00+00:00"
        n = fn(entry, cal)
        assert n is not None and n > 0, n
        return f"달력 {len(cal)}일 기준 보유 {n}거래일 (오늘까지)"
    finally:
        days_stub()


@scenario("H2", "진입일을 모르면 만기로 간주하지 않는다", "H")
def _h2():
    a = acct(100, {"AAPL": hold(10, 100, 105)})
    out, _ = vo(a, entries={})
    assert not out, f"진입일 없이 만기 청산: {out}"
    return "entries 없음 → 만기 판단 없음 (오늘 날짜로 채우지 않는다)"


@scenario("H3", "만기 도달 + 장외 소수점 잔량 → 제출 안 하고 사유를 남긴다", "H")
def _h3():
    a = acct(100, {"AAPL": hold(0.62, 20, 20)})
    out, skip = vo(a, entries={"AAPL": 20}, session=OFF)
    assert not out and any("FRACTIONAL_SESSION" in s["skipped"] or "정규장" in s["skipped"]
                           for s in skip), f"{out} {_s(skip)}"
    return "장외 소수점은 거절될 주문이므로 내지 않는다 (다음 정규장 재평가)"


@scenario("H4", "만기 도달 + 정규장 → 전량 매도 의도", "H")
def _h4():
    a = acct(100, {"AAPL": hold(0.62, 20, 20)})
    out, skip = vo(a, entries={"AAPL": 20}, session=REGULAR)
    assert len(out) == 1 and out[0]["quantity"] == 0.62, f"{out} {_s(skip)}"
    return f"전량 {out[0]['quantity']}주 (소수점 허용 세션)"


@scenario("H5", "가짜 체결 뒤 계좌를 다시 읽어야 현금·슬롯이 생긴다", "H")
def _h5():
    b = FakeBroker(holdings={"AAPL": hold(4, 36, 36.43)}, cash=0.01)
    a0 = at.account_state(b)
    before = a0["cash"] - a0["total_value"] * at.CASH_RESERVE_PCT / 100
    b.h.pop("AAPL")                       # 체결되었다고 가정
    b.cash = 0.01 + 4 * 36.43
    a1 = at.account_state(b)
    after = a1["cash"] - a1["total_value"] * at.CASH_RESERVE_PCT / 100
    assert before < 0 <= after, (before, after)
    return f"가용예산 ${before:.2f} → ${after:.2f} (체결 후 계좌 재조회로만 반영)"


@scenario("H6", "run 63 보유일 스냅샷 → 몇 거래일 뒤 만기 의도가 생기는가", "H")
def _h6():
    """★ 진입일 원본은 **확인하지 못했다.** 로컬 DB·로그에 브로커 체결 이력이 없다.
    로그에 남은 것은 코드가 계산한 '보유 거래일수'뿐이므로 그 값을 입력으로 쓴다.
    따라서 아래는 **날짜가 아니라 남은 거래일 수**다 — 달력은 런타임에 지수 일봉에서
    채워지므로(TRADING_DAYS) 오프라인에서 특정 날짜를 확정할 수 없다."""
    held = {"MU": 5, "MRNA": 1, "SMCI": 7, "MSTR": 7, "TEAM": 4, "KDP": 7, "TSLA": 5}
    a = _run63()
    out = {}
    for sym, h0 in held.items():
        for extra in range(0, 30):
            o, _ = vo(a, entries={sym: h0 + extra})
            if any(x["symbol"] == sym for x in o):
                out[sym] = extra
                break
    assert set(out) == set(held), out
    assert out["SMCI"] == at.MAX_HOLD_DAYS - 7, out
    return ("만기까지 남은 거래일: " +
            ", ".join(f"{k}+{v}" for k, v in sorted(out.items(), key=lambda kv: kv[1])))


@scenario("H7", "거래일 달력이 없으면 만기가 앞당겨진다 (휴장일 수만큼)", "H")
def _h7():
    fn = REAL_TRADING_DAYS_SINCE
    start = datetime.date(2026, 6, 1)
    today = datetime.datetime.now(at.NY).date()
    # 달력 있음: 평일에서 '휴장일' 3일을 뺀 목록
    cal = []
    d = start
    holidays = {start + datetime.timedelta(days=k) for k in (10, 20, 30)}
    while d <= today:
        if d.weekday() < 5 and d not in holidays:
            cal.append(d)
        d += datetime.timedelta(days=1)
    with_cal = fn(start.isoformat() + "T13:30:00+00:00", cal)
    no_cal = fn(start.isoformat() + "T13:30:00+00:00", [])
    n_hol = sum(1 for d in holidays if d.weekday() < 5 and start < d <= today)
    assert no_cal - with_cal == n_hol, (with_cal, no_cal, n_hol)
    return (f"달력 사용 {with_cal}거래일 vs 주말만 제외 {no_cal}거래일 — "
            f"달력이 비면 평일 휴장 {n_hol}일만큼 만기가 일찍 온다")


# ═══════════════ I. 변동성 타겟 활성화 ═══════════════
def _equity(rows):
    """rows = [(date, total_value, stock_value, cashflow)] 를 임시 DB 에 넣는다."""
    db = tmpdb()
    with sqlite3.connect(db) as c:
        c.executemany("INSERT INTO equity (date,total_value,stock_value,cashflow,timestamp) "
                      "VALUES (?,?,?,?,'t')", rows)
    return db


def _series(n, value=1000.0, stock=1.0, cashflow=None, deposit_on=None, deposit=0.0):
    out, v = [], value
    d = datetime.date(2026, 1, 1)
    for i in range(n):
        cf = None
        if deposit_on is not None and i == deposit_on:
            v += deposit
            cf = deposit if cashflow else None
        out.append((d.isoformat(), v, v * stock, cf))
        d += datetime.timedelta(days=1)
    return out


@scenario("I1", "관측치가 VOL_WINDOW+1 미만이면 비활성 (None)", "I")
def _i1():
    old = at.VOL_TARGET_PCT
    at.VOL_TARGET_PCT = 30
    try:
        _equity(_series(at.VOL_WINDOW))
        assert at._vol_target_cap() is None, "모자란 이력으로 상한을 냈다"
        return f"{at.VOL_WINDOW}행 < {at.VOL_WINDOW + 1} → None"
    finally:
        at.VOL_TARGET_PCT = old


@scenario("I2", "시장가치가 일정하면 변동성 0 → 상한 없음(축소 없음)", "I")
def _i2():
    old = at.VOL_TARGET_PCT
    at.VOL_TARGET_PCT = 30
    try:
        _equity(_series(at.VOL_WINDOW + 1))
        cap = at._vol_target_cap()
        assert cap is None, f"변동성 0 인데 상한 {cap}"
        return "변동성 0 → None (강제 축소 없음)"
    finally:
        at.VOL_TARGET_PCT = old


@scenario("I3", "입금만 있고 시장가치 불변 — cashflow 기재 시 허위 변동성 없음", "I")
def _i3():
    old = at.VOL_TARGET_PCT
    at.VOL_TARGET_PCT = 30
    try:
        _equity(_series(at.VOL_WINDOW + 1, deposit_on=30, deposit=200.0, cashflow=True))
        cap = at._vol_target_cap()
        assert cap is None, f"cashflow 를 적었는데도 상한 {cap} 이 생겼다"
        return "cashflow 기재 → 변동성 0 유지, 축소 없음"
    finally:
        at.VOL_TARGET_PCT = old


@scenario("I4", "입금을 cashflow 에 안 적으면 허위 변동성 → 강제 축소가 생긴다", "I")
def _i4():
    old = at.VOL_TARGET_PCT
    at.VOL_TARGET_PCT = 30
    try:
        _equity(_series(at.VOL_WINDOW + 1, deposit_on=30, deposit=200.0, cashflow=False))
        cap = at._vol_target_cap()
        assert cap is not None and cap < 1.0, f"허위 변동성이 안 잡혔다: cap={cap}"
        return (f"NULL 방치 → 상한 {cap * 100:.1f}% 로 축소가 걸린다 "
                "(입금이 '하루 수익률'로 잡힌 결과)")
    finally:
        at.VOL_TARGET_PCT = old


@scenario("I5", "NULL cashflow 는 '0' 으로 취급된다 — 미확인과 구분되지 않는다", "I")
def _i5():
    db = _equity(_series(3))
    with sqlite3.connect(db) as c:
        rows = list(c.execute("SELECT COALESCE(cashflow,0), cashflow FROM equity"))
    assert all(r[0] == 0 for r in rows) and all(r[1] is None for r in rows), rows
    return "COALESCE(cashflow,0) — NULL(미기재)과 0(입출금 없음)이 같은 값이 된다"


# ────────────────────────────── 실행 ──────────────────────────────
def run_all():
    out = []
    for sid, title, group, fn in SCEN:
        at.DB_PATH = _TMP / "scratch.db"
        try:
            detail = fn()
            out.append({"id": sid, "group": group, "title": title, "ok": True,
                        "detail": str(detail)})
        except Exception as e:                       # noqa: BLE001
            out.append({"id": sid, "group": group, "title": title, "ok": False,
                        "detail": f"{type(e).__name__}: {e}"[:300]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = run_all()
    if a.json:
        print(json.dumps(res, ensure_ascii=False))
        return 0
    g = None
    for r in res:
        if r["group"] != g:
            g = r["group"]
            print()
        print(f"  {'ok  ' if r['ok'] else 'FAIL'}  {r['id']}  {r['title']}")
        print(f"          {r['detail']}")
    bad = [r["id"] for r in res if not r["ok"]]
    print(f"\n통과 {len(res) - len(bad)} / {len(res)}" + (f" · 실패 {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
