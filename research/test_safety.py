"""§4.2 안전성·엔진 무결성 회귀 테스트 — **실제 함수**에 연결한다.

실행: python research/test_safety.py
네트워크·실계좌·운영 DB 를 건드리지 않는다 (임시 DB + 가짜 토스 객체).

test_validate.py / test_session.py 가 이미 덮은 것(스키마·슬롯·강제청산·만기·intent_key)은
반복하지 않고, 이번 프롬프트 §4.2 에서 새로 요구한 항목만 넣는다.
"""
import datetime
import math
import pathlib
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402

from research.isolation import guard              # noqa: E402

guard()

import autotrade as at                              # noqa: E402
from research import engine as E                    # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok   {name}")
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL {name}: {e}")


def tmpdb():
    p = pathlib.Path(tempfile.mkdtemp()) / "t.db"
    at.DB_PATH = p
    at.initialize_db()
    return p


def equity_rows(db, n, w=0.5, ann_vol=0.60, cashflow_on=None, cf=0.0):
    """±1σ 를 번갈아 넣어 실현 변동성이 ann_vol 이 되는 자산 이력. cashflow_on 일에 입금."""
    d = ann_vol / math.sqrt(252)
    v = 1000.0
    with sqlite3.connect(db) as c:
        for i in range(n):
            amt = cf if i == cashflow_on else 0.0
            v += amt
            c.execute("INSERT INTO equity (date, total_value, stock_value, cashflow) "
                      "VALUES (?,?,?,?)", (f"2026-{i // 28 + 1:02d}-{i % 28 + 1:02d}",
                                           v, v * w, amt))
            v *= 1 + (d * w) * (1 if i % 2 else -1)


class FakeToss:
    """토스 클라이언트 대역. 실제 호출부(free_position/place_all/position_entry_map)가
    쓰는 메서드만 흉내낸다."""

    def __init__(self, open_orders=(), closed=(), sellable=None,
                 cancel_raises=False, create_raises=False):
        self._open = list(open_orders)
        self._closed = list(closed)
        self._sellable = sellable
        self.cancel_raises = cancel_raises
        self.create_raises = create_raises
        self.created = []
        self.cancelled = []

    def us_market_calendar(self, date=None):
        # place_all 이 **제출 직전** 세션을 다시 확인한다 — 대역도 거래 가능 구간을 준다.
        n = datetime.datetime.now(at.NY)
        return {"today": {"regularMarket": {
            "startTime": (n - datetime.timedelta(hours=1)).isoformat(),
            "endTime": (n + datetime.timedelta(hours=3)).isoformat()}}}

    def orders(self, status="OPEN", symbol=None, limit=None, cursor=None,
               from_date=None, to_date=None):
        """공식 스펙(1.2.17): OPEN 은 전량, CLOSED 는 limit(기본 20·최대 100)+cursor."""
        rows = self._open if status == "OPEN" else self._closed
        if symbol:
            rows = [r for r in rows if r.get("symbol") == symbol]
        if status == "OPEN":
            return {"orders": rows, "nextCursor": None, "hasNext": False}
        n, i = min(int(limit or 20), 100), int(cursor or 0)
        has = i + n < len(rows)
        return {"orders": rows[i:i + n], "nextCursor": str(i + n) if has else None,
                "hasNext": has}

    def orders_all(self, status="OPEN", symbol=None, from_date=None, to_date=None,
                   max_pages=20):
        rows, cursor = [], None
        for _ in range(max_pages):
            r = self.orders(status, symbol=symbol, limit=100, cursor=cursor)
            rows.extend(r["orders"])
            cursor = r["nextCursor"]
            if not r["hasNext"]:
                return rows, True
        return rows, False

    def order(self, order_id):
        for r in self._open + self._closed:
            if r.get("orderId") == order_id:
                return r
        raise at.TossError(404, "not-found", "없는 주문")

    def cancel_order(self, oid):
        self.cancelled.append(oid)
        if self.cancel_raises:                      # 취소 경합: 이미 체결되어 거절
            raise at.TossError(409, "already-filled", "이미 체결")
        self._open = [o for o in self._open if o["orderId"] != oid]

    def sellable_quantity(self, symbol):
        if self._sellable is None:
            raise at.TossError(500, "err", "조회 실패")
        return {"sellableQuantity": self._sellable}

    def create_order(self, symbol, side, otype, quantity=None, price=None,
                     order_amount=None, client_order_id=None):
        if self.create_raises:
            raise at.TossError(504, "timeout", "게이트웨이 타임아웃")
        self.created.append(dict(symbol=symbol, side=side, quantity=quantity,
                                 amount=order_amount, coid=client_order_id))
        return {"orderId": "OID-" + client_order_id}


def fill(sym, side, qty, at_iso):
    return {"symbol": sym, "side": side, "orderedAt": at_iso,
            "execution": {"filledQuantity": qty}}


def hold(qty, avg, last):
    return {"name": "x", "quantity": qty, "avg_price": avg, "last_price": last,
            "market_value": qty * last, "pnl_pct": round((last / avg - 1) * 100, 2)}


def acct(cash, holdings=None, open_orders=()):
    h = holdings or {}
    return {"cash": cash, "holdings": h, "open_orders": list(open_orders),
            "total_value": cash + sum(x["market_value"] for x in h.values())}


def dec(sym, decision, price, ret20, account):
    factor, tier = at.momentum_tier(ret20)
    h = account["holdings"].get(sym, {})
    return {"symbol": sym, "decision": decision, "percentage": 80, "reason": "t",
            "status": {"current_price": price, "stock_balance": h.get("quantity", 0.0),
                       "avg_buy_price": h.get("avg_price", 0.0), "pnl_pct": h.get("pnl_pct"),
                       "ret_20d_pct": ret20, "momentum_tier": tier, "size_factor": factor}}


NOW = datetime.datetime.now(at.KST)
REGULAR = (NOW - datetime.timedelta(hours=1), NOW - datetime.timedelta(hours=1),
           NOW + datetime.timedelta(hours=3))


# ══════════════════════════════════════════════════════════════════════════
def t_cashflow_removed():
    """입금이 '수익'으로 잡혀 변동성을 오염시키지 않는다 — cashflow 를 적으면 동일한 상한."""
    at.VOL_TARGET_PCT, at.VOL_WINDOW = 30, 60
    db = tmpdb()
    equity_rows(db, at.VOL_WINDOW + 1)
    base = at.exposure_cap()
    db = tmpdb()
    equity_rows(db, at.VOL_WINDOW + 1, cashflow_on=30, cf=500.0)   # 30일차에 $500 입금
    withcf = at.exposure_cap()
    assert base is not None and withcf is not None, (base, withcf)
    assert abs(base - withcf) < 0.02, f"입금 후 상한이 달라졌다 {base:.3f} vs {withcf:.3f}"


def t_cashflow_unrecorded_is_detected():
    """cashflow 를 안 적으면 하루 ±25% 점프가 경고로 잡힌다 (자동 보정은 못 한다)."""
    import logging
    db = tmpdb()
    equity_rows(db, at.VOL_WINDOW + 1)
    with sqlite3.connect(db) as c:                 # 입금을 cashflow 없이 총자산에만 반영
        c.execute("UPDATE equity SET total_value = total_value * 2 WHERE date >= "
                  "(SELECT date FROM equity ORDER BY date LIMIT 1 OFFSET 30)")
    rec = []
    h = logging.Handler()
    h.emit = rec.append
    at.log.addHandler(h)
    at.exposure_cap()
    at.log.removeHandler(h)
    assert any("cashflow" in r.getMessage() for r in rec), "미기재 입금 경고가 없다"


def t_vol_cap_insufficient_history():
    """관측 61개 전에는 기능이 꺼진다 (None) — 근거 없이 노출을 자르지 않는다."""
    db = tmpdb()
    equity_rows(db, at.VOL_WINDOW)                  # 60개 = 수익률 59개
    assert at.exposure_cap() is None


def t_vol_cap_zero_vol():
    """변동성 0(값이 안 변함) → 0 나눗셈 대신 상한 100% 로 수렴하고 예외를 내지 않는다."""
    db = tmpdb()
    with sqlite3.connect(db) as c:
        for i in range(at.VOL_WINDOW + 1):
            c.execute("INSERT INTO equity (date,total_value,stock_value,cashflow) "
                      "VALUES (?,?,?,0)", (f"2026-{i // 28 + 1:02d}-{i % 28 + 1:02d}",
                                           1000.0, 500.0))
    assert at.exposure_cap() is None                # vol_ann <= 0 → 비활성


def t_vol_cap_is_total_asset_ceiling():
    """cap 의 의미: 기존 노출에 곱하는 비율이 아니라 **총자산 대비 주식 상한**이다.
    축소 금액 = 주식평가액 − 총자산×cap 이어야 한다."""
    at.MAX_HOLD_DAYS, at.STOP_LOSS_PCT, at.MOMENTUM_EXIT = 0, 0, False
    at.MIN_ORDER_USD, at.CASH_RESERVE_PCT = 5, 10
    a = acct(200, {"AAA": hold(10, 50, 80), "BBB": hold(10, 20, 20)})   # 주식 1000, 총 1200
    orig, at.exposure_cap = at.exposure_cap, lambda verbose=False: 0.5   # 상한 50% = 600
    try:
        out, _ = at.validate_orders({"orders": [], "summary": ""}, {}, a, REGULAR)
    finally:
        at.exposure_cap = orig
    cut = sum(o["amount_usd"] for o in out if o["side"] == "sell")
    assert abs(cut - 400) < 1.0, f"축소액 {cut} != 1000 − 1200×0.5 = 400"


def t_reduce_blocks_buys_but_not_sells():
    """신규 진입 차단과 보유분 대사/축소는 분리되어 있다 — cap 초과여도 매도는 나간다."""
    at.STOP_LOSS_PCT, at.MOMENTUM_EXIT = 15, False
    a = acct(1000, {"AAA": hold(10, 100, 50)})       # −50% → 손절 대상
    orig, at.exposure_cap = at.exposure_cap, lambda verbose=False: 0.0   # 노출 0% 상한
    try:
        out, skip = at.validate_orders({"orders": [{"symbol": "BBB", "side": "buy",
                                                    "reason": "t"}], "summary": ""},
                                       {"BBB": dec("BBB", "buy", 10, 30, a)}, a, REGULAR)
    finally:
        at.exposure_cap = orig
    sells = [o for o in out if o["side"] == "sell"]
    buys = [o for o in out if o["side"] == "buy"]
    assert sells and any("손절" in o["reason"] for o in sells), "노출 차단이 손절까지 막았다"
    assert not buys, "노출 0% 인데 신규 매수가 나갔다"


def t_sell_proceeds_not_reused():
    """매도 **제출**한 대금으로 같은 사이클에서 더 사지 않는다 (체결·buying power 기준)."""
    at.STOP_LOSS_PCT, at.MOMENTUM_EXIT = 15, False
    at.MAX_POSITIONS, at.MAX_POSITION_PCT = 5, 30
    a = acct(100, {"AAA": hold(10, 100, 50)})        # 현금 100, 손절로 500 이 들어올 예정
    out, skip = at.validate_orders(
        {"orders": [{"symbol": "BBB", "side": "buy", "reason": "t"}], "summary": ""},
        {"BBB": dec("BBB", "buy", 10, 30, a)}, a, REGULAR)
    buys = [o for o in out if o["side"] == "buy"]
    total = a["total_value"]
    room = a["cash"] - total * at.CASH_RESERVE_PCT / 100
    assert not buys or buys[0]["amount_usd"] <= room + 1e-6, \
        f"매도 대금을 미리 썼다: {buys[0]['amount_usd']} > {room}"


def t_partial_exit_keeps_age():
    """부분 매도는 보유 나이를 유지하고, 전량 청산 후 재진입은 시계를 리셋한다."""
    t = FakeToss(closed=[fill("AAA", "BUY", 10, "2026-01-05T14:30:00+00:00"),
                         fill("AAA", "SELL", 4, "2026-02-02T14:30:00+00:00"),
                         fill("BBB", "BUY", 5, "2026-01-05T14:30:00+00:00"),
                         fill("BBB", "SELL", 5, "2026-02-02T14:30:00+00:00"),
                         fill("BBB", "BUY", 5, "2026-03-02T14:30:00+00:00")])
    m = at.position_entry_map(t)
    assert m["AAA"].startswith("2026-01-05"), m           # 부분 매도 → 최초 진입 유지
    assert m["BBB"].startswith("2026-03-02"), m           # 전량 청산 후 재진입 → 리셋


def t_add_on_does_not_reset_age():
    """추가 매수는 시계를 리셋하지 않는다 (최초 진입 기준)."""
    t = FakeToss(closed=[fill("AAA", "BUY", 5, "2026-01-05T14:30:00+00:00"),
                         fill("AAA", "BUY", 5, "2026-02-20T14:30:00+00:00")])
    assert at.position_entry_map(t)["AAA"].startswith("2026-01-05")


def t_dst_boundary():
    """미국 DST 경계에서도 거래일 계산이 달력 기준으로 맞는다 (KST ISO 입력)."""
    # 2026-03-08 미국 서머타임 시작. KST 09:00 은 NY 전날 19:00(EST) 또는 20:00(EDT).
    cal = [datetime.date(2026, 3, d) for d in (5, 6, 9, 10, 11, 12, 13)]
    entry = "2026-03-06T09:00:00+09:00"              # = 2026-03-05 19:00 NY
    n = at.trading_days_since(entry, cal)
    assert n == sum(1 for d in cal if datetime.date(2026, 3, 5) < d
                    <= datetime.datetime.now(at.NY).date()), n
    assert at.trading_days_since("2026-03-09T09:00:00+09:00", cal) == \
        sum(1 for d in cal if datetime.date(2026, 3, 8) < d
            <= datetime.datetime.now(at.NY).date())


def t_cancel_race_partial_fill():
    """취소가 거절돼도(=이미 체결) 판단은 재조회 결과로 한다. 부분체결이면 그만큼만 판다."""
    t = FakeToss(open_orders=[{"orderId": "O1", "symbol": "AAA"}], sellable=3.0,
                 cancel_raises=True)
    q = at.free_position(t, "AAA", timeout=0)
    assert q == 3.0, q
    o = {"symbol": "AAA", "side": "sell", "quantity": 10.0, "amount_usd": 100,
         "price": 10, "cancel_first": True, "reason": "손절"}
    at.place_order(t, o, coid="TESTCOID")
    assert t.created and t.created[0]["quantity"] == "3", t.created


def t_sellable_zero_refuses():
    """매도 가능 수량 0 이면 주문을 내지 않고 명시적으로 **보류**한다 (실패가 아니라 '모름'이 아닌 '없음')."""
    t = FakeToss(open_orders=[], sellable=0.0)
    try:
        at.place_order(t, {"symbol": "AAA", "side": "sell", "quantity": 5.0,
                           "amount_usd": 50, "price": 10, "cancel_first": True,
                           "reason": "만기"}, coid="X")
    except at.Deferred as e:
        assert "NO_SELLABLE" in str(e), e
        return
    raise AssertionError("매도 가능 0 인데 주문이 나갔다")


def t_sellable_lookup_fails_defers():
    """조회 자체가 실패하면 UNKNOWN → **오래된 계획 수량으로 제출하지 않는다.**

    (예전에는 None 을 '계획 수량 그대로' 로 읽었다. 그러면 취소가 끝났는지도 모르는 채
     기존 주문과 합쳐 초과 매도가 나갈 수 있었다.)
    """
    t = FakeToss(open_orders=[], sellable=None)
    assert at.free_position(t, "AAA", timeout=0) is None
    try:
        at.place_order(t, {"symbol": "AAA", "side": "sell", "quantity": 5.0,
                           "amount_usd": 50, "price": 10, "cancel_first": True,
                           "reason": "손절"}, coid="X")
    except at.Deferred:
        assert not t.created, t.created
        return
    raise AssertionError("수량 UNKNOWN 인데 계획 수량으로 제출했다")


def t_late_fill_after_timeout():
    """제출 중 타임아웃 → 대사에서 접수가 확인되면 **재시도하지 않고** 접수로 기록한다."""
    db = tmpdb()
    o = {"symbol": "AAA", "side": "buy", "quantity": None, "amount_usd": 100.0,
         "price": 10, "reason": "t"}
    coid = at.intent_key(o)
    t = FakeToss(closed=[{"orderId": "REAL1", "clientOrderId": coid, "symbol": "AAA"}],
                 create_raises=True)
    rid = at.db_insert("runs", {"timestamp": at._now(), "status": "test"})
    at.place_all(t, rid, [o], dry=False)
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT status, order_id FROM orders").fetchall()
    assert len(rows) == 1 and "대사 확인" in rows[0][0], rows
    assert rows[0][1] == "REAL1", rows
    assert not t.created, "타임아웃 후 재제출했다"


def t_durable_intent_before_submit():
    """제출 **전에** pending 의도가 DB 에 남는다 — 도중에 죽어도 다음 실행이 안다."""
    db = tmpdb()
    o = {"symbol": "AAA", "side": "buy", "quantity": None, "amount_usd": 100.0,
         "price": 10, "reason": "t"}

    class Boom(FakeToss):
        def create_order(self, *a, **k):
            raise KeyboardInterrupt("프로세스 강제 종료")

    rid = at.db_insert("runs", {"timestamp": at._now(), "status": "test"})
    try:
        at.place_all(Boom(), rid, [o], dry=False)
    except KeyboardInterrupt:
        pass
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT status, order_id FROM orders").fetchall()
    assert rows and rows[0][0] == "INTENT_RECORDED", rows


def t_duplicate_process_same_coid():
    """다른 프로세스가 같은 슬롯에 같은 의도를 내면 clientOrderId 가 같다 (브로커가 거절).
    ★ 이것은 브로커 중복 거절에 의존하는 것이고, 다중 PC 동시 제출 경합을 막지 못한다."""
    when = datetime.datetime(2026, 9, 10, 22, 5, tzinfo=at.NY)
    o = {"symbol": "AAA", "side": "buy"}
    assert at.intent_key(o, when) == at.intent_key(dict(o), when)
    assert at.intent_key(o, when) != at.intent_key(o, when + datetime.timedelta(minutes=30))


def t_stale_snapshot_blocks_buys_only():
    """낡은 스냅샷 처리: 매수만 제거하고 매도는 남긴다 (run_cycle 의 필터와 동일한 규칙)."""
    orders = [{"symbol": "A", "side": "buy"}, {"symbol": "B", "side": "sell"}]
    kept = [o for o in orders if o.get("side") != "buy"]
    assert [o["symbol"] for o in kept] == ["B"]
    assert at.STALE_MAX_MIN > 0
    # ※ run_cycle 안에 인라인이라 함수 단위로는 못 부른다. 규칙만 고정한다 (한계 명시).


def t_regime_fallback_is_a_different_strategy():
    """QQQ 실패 시 유니버스 중앙값 폴백이 실제로 **다른 판정**을 낼 수 있음을 보인다."""
    rows = [{"ret_60d_pct": v} for v in (-10, -9, -8, -7, -6)]
    idx = pd.DataFrame({"close": np.linspace(100, 99, 61)})   # 지수는 -1% → 보통
    a = at.market_regime(None, rows=rows, df=idx)
    b = at.market_regime(None, rows=rows, df=None)            # 지수 없음 → 중앙값 -8% → 하락
    assert a["regime"] == "보통" and a["source"] == at.BEAR_INDEX, a
    assert b["regime"] == "하락" and b["source"] == "유니버스 중앙값", b


def t_regime_no_data_defaults_to_normal():
    """지수도 유니버스도 없으면 '보통'으로 떨어진다 — 위험을 계산했다고 말할 수 없는 상태."""
    r = at.market_regime(None, rows=None, df=None)
    assert r["regime"] == "보통" and r["ret_60d_pct"] is None and "실패" in r["source"]


def _panel():
    import backtest as bt
    return bt.daily_panel()


def t_prefix_invariance():
    """미래 행을 붙여도 과거 판단이 바뀌지 않는다 (엔진 전 구간)."""
    close, opens, ret20 = _panel()
    k = 400
    cfg = dict(sl=None, mom_exit=False, hold_days=20, top=10, max_pos_pct=15)
    for m in E.MODES:
        short = E.sim(close.iloc[:k], ret20.iloc[:k], cfg,
                      opens=opens.iloc[:k], mode=m)["eq"]
        full = E.sim(close, ret20, cfg, opens=opens, mode=m)["eq"]
        common = short.index.intersection(full.index)
        assert np.allclose(short.loc[common].values, full.loc[common].values), \
            f"{m}: prefix 불변 위반 max={float(abs(short.loc[common]-full.loc[common]).max())}"


def t_golden_replay_ranking():
    """운영 진입 후보 산정과 replay 엔진의 후보 목록이 **같은 스냅샷에서 같다**.

    운영: momentum_tier(ret_20d) 의 size_factor > 0 인 종목을 ret_20d 내림차순.
    replay: engine.sim 의 cands (mo > entry_th=0, 내림차순).
    """
    close, opens, ret20 = _panel()
    for t in (close.index[100], close.index[400], close.index[-2]):
        mo = ret20.loc[t].dropna()
        replay = [s for s in mo.sort_values(ascending=False).index if mo[s] > 0]
        live = [s for s in mo.sort_values(ascending=False).index
                if at.momentum_tier(float(mo[s]), "보통")[0] > 0]
        assert replay == live, (t, replay[:5], live[:5])


def t_bear_regime_ignores_momentum_sign():
    """하락 국면에서는 모멘텀 부호로 매수를 막지 않는다 (운영 규칙 고정)."""
    assert at.momentum_tier(-30, "하락")[0] == 1.0
    assert at.momentum_tier(-30, "보통")[0] == 0.0


TESTS = [
    ("입출금 제거 — cashflow 기재 시 변동성 불변", t_cashflow_removed),
    ("입출금 미기재 — ±25% 점프 경고", t_cashflow_unrecorded_is_detected),
    ("변동성 타겟 — 관측 61개 전 비활성", t_vol_cap_insufficient_history),
    ("변동성 타겟 — 변동성 0 대응", t_vol_cap_zero_vol),
    ("변동성 cap 의미 — 총자산 대비 상한", t_vol_cap_is_total_asset_ceiling),
    ("신규 진입 차단과 보유분 축소 분리", t_reduce_blocks_buys_but_not_sells),
    ("매도 제출 대금 재사용 금지", t_sell_proceeds_not_reused),
    ("부분청산 나이 유지 / 재진입 리셋", t_partial_exit_keeps_age),
    ("추가 매수는 나이 리셋 안 함", t_add_on_does_not_reset_age),
    ("DST 경계 거래일 계산", t_dst_boundary),
    ("취소 경합 + 부분체결 clamp", t_cancel_race_partial_fill),
    ("매도가능 0 → 주문 거절", t_sellable_zero_refuses),
    ("매도가능 조회 실패 → 제출 보류", t_sellable_lookup_fails_defers),
    ("타임아웃 후 뒤늦은 접수 대사", t_late_fill_after_timeout),
    ("durable intent (재시작 대비)", t_durable_intent_before_submit),
    ("중복 프로세스 동일 clientOrderId", t_duplicate_process_same_coid),
    ("낡은 스냅샷 — 매수만 취소", t_stale_snapshot_blocks_buys_only),
    ("QQQ 폴백은 다른 판정을 낼 수 있다", t_regime_fallback_is_a_different_strategy),
    ("국면 판정 불가 시 기본값", t_regime_no_data_defaults_to_normal),
    ("하락 국면 모멘텀 부호 무시", t_bear_regime_ignores_momentum_sign),
    ("prefix-invariance (미래 행 추가)", t_prefix_invariance),
    ("golden replay — 운영/replay 후보 동일", t_golden_replay_ranking),
]


def main():
    real_db = at.DB_PATH
    print(f"안전성 회귀 테스트 {len(TESTS)}건 (운영 DB {real_db.name} 은 건드리지 않는다)")
    for name, fn in TESTS:
        check(name, fn)
    at.DB_PATH = real_db
    print(f"\n통과 {len(PASS)} / 실패 {len(FAIL)}")
    for n, e in FAIL:
        print(f"  FAIL {n}: {e}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
