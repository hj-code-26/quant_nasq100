"""§3 불변식 테스트 + 결함 주입(mutation) 검증 + 손계산 원장 fixture.

실행: python research/test_invariance.py

세 가지를 확인한다.
 1. **당일 종가 불변식** — t 일 종가만 바꿔도 t 일 시가 주문이 바뀌지 않는다.
    (기존 prefix-invariance 와 다른 테스트다: prefix 는 '미래 행 추가', 이건 '당일 종가 변경'.)
 2. **결함 주입** — 알려진 결함을 되살리면 해당 테스트가 **실제로 실패하는지** 본다.
    통과만 보는 테스트는 아무것도 증명하지 않는다.
 3. **손계산 원장 fixture** — 엔진과 독립적으로 손으로 계산한 값과 일치하는지 본다.
"""
import importlib.util
import pathlib
import tempfile
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from research.isolation import guard    # noqa: E402

guard()

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

from research import engine as E        # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok   {name}")
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL {name}: {e}")


def orders_on(led, day):
    return sorted((r["symbol"], r["side"], round(r["qty"], 9), round(r["fill_px"], 6))
                  for r in led if r["order_date"] == day)


def panel():
    import backtest as bt
    return bt.daily_panel()


# ══ 1. 당일 종가 불변식 ═══════════════════════════════════════════════════
CFG_SL = dict(sl=25, mom_exit=True, rotate_guard=True)     # 손절이 있어야 판정가 누수가 드러난다


def _order_days(mode, cfg, close, opens, ret20, n=6):
    """실제로 주문이 난 날짜만 고른다 — 빈 날을 비교하면 아무것도 증명하지 못한다."""
    led = []
    E.sim(close, ret20, cfg, opens=opens, mode=mode, ledger=led)
    days = sorted({r["order_date"] for r in led})
    step = max(1, len(days) // n)
    return led, [close.index.get_loc(d) for d in days[::step][:n]]


def _perturb_days(mode, cfg=None, factor=1.30, ks=None):
    """t 일 종가만 factor 배로 바꾸고 **그 날 주문**이 같은지 본다. 다른 날은 손대지 않는다.
    주문이 실제로 있는 날만 센다 — 빈 날은 자동으로 같아져 아무것도 증명하지 못한다."""
    close, opens, ret20 = panel()
    cfg = cfg or CFG_SL
    base_led, auto_ks = _order_days(mode, cfg, close, opens, ret20)
    ks = ks or auto_ks
    out = []
    for k in ks:
        day = close.index[k]
        c2 = close.copy()
        c2.iloc[k] = c2.iloc[k] * factor        # ★ 당일 종가만 변경
        pert = []
        E.sim(c2, c2.pct_change(20) * 100, cfg, opens=opens, mode=mode, ledger=pert)
        a, b = orders_on(base_led, day), orders_on(pert, day)
        out.append((day, a, b))
    return out


def _assert_same(rows, mode):
    assert [r for r in rows if r[1]], f"모드 {mode}: 검사한 날에 주문이 없다 — 테스트가 무력하다"
    for day, a, b in rows:
        assert a == b, f"모드 {mode} {day.date()} 주문이 당일 종가에 반응했다: {a} != {b}"


def t_close_perturbation_mode_c():
    _assert_same(_perturb_days("C", factor=1.30), "C")


def t_close_perturbation_mode_c_down():
    """반대 방향(-25%) 섭동에서도 성립하는지."""
    _assert_same(_perturb_days("C", factor=0.75), "C")


# ══ 2. 결함 주입 ══════════════════════════════════════════════════════════
def mutate_autotrade(replacements):
    """autotrade.py 를 **디스크는 그대로 두고** 메모리에서만 변형해 모듈로 올린다."""
    src = (ROOT / "autotrade.py").read_text(encoding="utf-8")
    tmp = pathlib.Path(tempfile.mkdtemp()) / "mutant.db"
    src = src.replace('DB_PATH = ROOT / "trading_decisions.db"',
                      f"DB_PATH = pathlib.Path(r'{tmp}')", 1)
    for old, new in replacements:
        assert old in src, f"변형 앵커를 못 찾았다: {old[:60]}"
        src = src.replace(old, new, 1)
    mod = types.ModuleType("autotrade_mutant")
    mod.__file__ = str(ROOT / "autotrade.py")
    spec = importlib.util.spec_from_loader("autotrade_mutant", loader=None)
    mod.__spec__ = spec
    sys.modules["autotrade_mutant"] = mod
    exec(compile(src, str(ROOT / "autotrade.py"), "exec"), mod.__dict__)
    return mod


def expect_fail(fn, why):
    """결함을 넣었을 때 테스트가 **실패해야** 한다. 통과하면 그 테스트는 무력하다."""
    try:
        fn()
    except AssertionError:
        return
    except Exception as e:                       # noqa: BLE001
        raise AssertionError(f"{why}: 예상한 AssertionError 가 아니라 {type(e).__name__}: {e}")
    raise AssertionError(f"{why}: 결함을 넣었는데도 테스트가 통과했다 (테스트가 무력하다)")


def t_mutant_leak_breaks_close_invariance():
    """누수 복원(모드 B = 당일 종가로 판정) → 당일 종가 불변식이 깨져야 한다."""
    expect_fail(lambda: _assert_same(_perturb_days("B", factor=1.30), "B"), "모드 B 누수")


def t_mutant_slot_reuse_breaks_max_positions():
    """P0-4 복원: 매도 **제출**만으로 슬롯을 비우면 최대 종목 수를 넘긴다."""
    mut = mutate_autotrade([(
        "        done.add(sym)\n        # ★ 슬롯은 여기서 비우지 않는다.",
        "        done.add(sym)\n        positions.discard(sym)\n"
        "        # ★ 슬롯은 여기서 비우지 않는다.")])
    from research import test_safety as TS
    mut.MAX_POSITIONS, mut.MAX_POSITION_PCT, mut.CASH_RESERVE_PCT = 1, 30, 10
    mut.MIN_ORDER_USD, mut.STOP_LOSS_PCT, mut.MOMENTUM_EXIT, mut.MAX_HOLD_DAYS = 5, 15, False, 0

    def run():
        a = TS.acct(1000, {"AAA": TS.hold(10, 100, 50)})     # AAA 는 손절 대상
        f, tier = mut.momentum_tier(30)
        d = {"BBB": {"symbol": "BBB", "decision": "buy", "percentage": 80, "reason": "t",
                     "status": {"current_price": 10, "stock_balance": 0.0,
                                "avg_buy_price": 0.0, "pnl_pct": None, "ret_20d_pct": 30,
                                "momentum_tier": tier, "size_factor": f}}}
        out, skip = mut.validate_orders(
            {"orders": [{"symbol": "BBB", "side": "buy", "reason": "t"}], "summary": ""},
            d, a, TS.REGULAR)
        buys = [o for o in out if o["side"] == "buy"]
        assert not buys, "매도 제출만으로 슬롯이 비어 신규 매수가 나갔다"
    expect_fail(run, "P0-4 슬롯 재사용")


def t_mutant_partial_fill_ignored():
    """부분체결 clamp 제거 → 매도가능 수량보다 많이 파는 주문이 나간다."""
    mut = mutate_autotrade([(
        '            o = {**o, "quantity": min(o["quantity"], q)}',
        "            pass")])
    from research.test_safety import FakeToss

    def run():
        t = FakeToss(open_orders=[{"orderId": "O1", "symbol": "AAA"}], sellable=3.0,
                     cancel_raises=True)
        mut.place_order(t, {"symbol": "AAA", "side": "sell", "quantity": 10.0,
                            "amount_usd": 100, "price": 10, "cancel_first": True,
                            "reason": "손절"}, coid="X")
        assert t.created and t.created[0]["quantity"] == "3", t.created
    expect_fail(run, "부분체결 clamp 제거")


def t_mutant_duplicate_order_key():
    """intent_key 에 프로세스 고유값을 되살리면 → 다른 프로세스가 다른 id 를 내 중복 주문."""
    mut = mutate_autotrade([(
        '    return f"at{t:%y%m%d}{t.hour:02d}{t.minute // 30 * 30:02d}'
        '{o[\'symbol\']}{o[\'side\'][0].upper()}"',
        "    import os\n"
        '    return f"at{os.getpid()}{t:%y%m%d}{o[\'symbol\']}{o[\'side\'][0].upper()}"')])

    def run():
        import datetime
        when = datetime.datetime(2026, 9, 10, 22, 5, tzinfo=mut.NY)
        o = {"symbol": "AAA", "side": "buy"}
        k1 = mut.intent_key(o, when)
        k2 = mutate_autotrade([(
            '    return f"at{t:%y%m%d}{t.hour:02d}{t.minute // 30 * 30:02d}'
            '{o[\'symbol\']}{o[\'side\'][0].upper()}"',
            "    import os\n"
            '    return f"at{os.getpid() + 1}{t:%y%m%d}'
            '{o[\'symbol\']}{o[\'side\'][0].upper()}"')]).intent_key(o, when)
        assert k1 == k2, "다른 프로세스가 다른 clientOrderId 를 낸다 → 중복 주문"
    expect_fail(run, "intent_key 프로세스 고유값")


def t_mutant_stale_filter_removed():
    """낡은 스냅샷 필터 제거 → 소스 수준 점검이 실패해야 한다."""
    src = (ROOT / "autotrade.py").read_text(encoding="utf-8")
    anchor = ('            plan["orders"] = [o for o in plan["orders"] '
              'if o.get("side") != "buy"]')
    assert anchor in src, "현재 소스에 낡은 스냅샷 매수 취소가 있다"

    def run():
        mutated = src.replace(anchor, "            pass", 1)
        assert anchor in mutated, "낡은 스냅샷일 때 신규 매수를 취소하지 않는다"
    expect_fail(run, "stale 필터 제거")


def t_mutant_cash_reserve_removed():
    """현금 유지 비중 제거 → '매도 대금 재사용 금지' 테스트가 잡아내야 한다."""
    mut = mutate_autotrade([(
        '    cash_left = account["cash"] - total * CASH_RESERVE_PCT / 100',
        '    cash_left = account["cash"] + sum(h["market_value"] '
        'for h in holdings.values())')])
    from research import test_safety as TS
    mut.STOP_LOSS_PCT, mut.MOMENTUM_EXIT, mut.MAX_HOLD_DAYS = 15, False, 0
    mut.MAX_POSITIONS, mut.MAX_POSITION_PCT, mut.CASH_RESERVE_PCT = 5, 30, 10
    mut.MIN_ORDER_USD = 5

    def run():
        a = TS.acct(100, {"AAA": TS.hold(10, 100, 50)})
        f, tier = mut.momentum_tier(30)
        d = {"BBB": {"symbol": "BBB", "decision": "buy", "percentage": 80, "reason": "t",
                     "status": {"current_price": 10, "stock_balance": 0.0,
                                "avg_buy_price": 0.0, "pnl_pct": None, "ret_20d_pct": 30,
                                "momentum_tier": tier, "size_factor": f}}}
        out, _ = mut.validate_orders(
            {"orders": [{"symbol": "BBB", "side": "buy", "reason": "t"}], "summary": ""},
            d, a, TS.REGULAR)
        buys = [o for o in out if o["side"] == "buy"]
        room = a["cash"] - a["total_value"] * 10 / 100
        assert not buys or buys[0]["amount_usd"] <= room + 1e-6, \
            f"매도 대금·보유 평가액을 매수 여력으로 썼다: {buys[0]['amount_usd']} > {room}"
    expect_fail(run, "현금 유지 비중 제거")


# ══ 3. 손계산 원장 fixture ════════════════════════════════════════════════
def hand_fixture():
    """엔진과 무관하게 손으로 따라갈 수 있는 최소 패널.

    거래일 30개, 종목 X·Y.
      Y: 종가·시가 모두 100 고정 → ret20 = 0 → entry_th=0 초과가 아니라 후보 아님.
      X: 20행까지 100 고정, 이후 매일 종가 +10 (100,110,120,…). 시가 = 전일 종가.
    설정: top=1, 종목당 100%, 현금 유지 0%, 비용 0, 최소금액 0, 손절/모멘텀청산 없음,
          만기 없음, 소수점 주문 허용.
    """
    n = 30
    idx = pd.bdate_range("2026-01-01", periods=n, tz="UTC")
    xc = [100.0] * 21 + [100.0 + 10 * (i - 20) for i in range(21, n)]
    yc = [100.0] * n
    xo = [100.0] + xc[:-1]                       # 시가 = 전일 종가
    yo = [100.0] * n
    close = pd.DataFrame({"X": xc, "Y": yc}, index=idx)
    opens = pd.DataFrame({"X": xo, "Y": yo}, index=idx)
    return close, opens, close.pct_change(20) * 100


HAND_CFG = dict(top=1, max_pos_pct=100, reserve_pct=0, min_usd=0, cost_pct=0,
                sl=None, mom_exit=False, hold_days=None, frac=True)


def t_hand_ledger_mode_c():
    """손계산:
      dates = index[20:] → 행 20..29 (10일). n=0 은 모드 C 가 건너뛴다.
      ret20(행 21) = 110/100 − 1 = +10% > 0  → 신호는 행 21 종가에 확정.
      모드 C: 행 22 개장에 매수. 행 22 시가 = 행 21 종가 = 110.
      수량 = 1000 / 110 = 9.0909090909…  (비용 0, 금액 주문)
      이후 청산 규칙이 없으므로 끝까지 보유. 마지막 행 29 종가 = 100 + 10×9 = 190.
      최종 NAV = 9.0909090909… × 190 = 1727.2727…
    """
    close, opens, ret20 = hand_fixture()
    led = []
    r = E.sim(close, ret20, HAND_CFG, cash0=1000.0, opens=opens, mode="C", ledger=led)
    buys = [x for x in led if x["side"] == "BUY"]
    assert len(buys) == 1 and buys[0]["symbol"] == "X", led
    assert buys[0]["order_date"] == close.index[22], buys[0]["order_date"]
    assert abs(buys[0]["fill_px"] - 110.0) < 1e-9, buys[0]["fill_px"]
    qty_hand = 1000.0 / 110.0
    assert abs(buys[0]["qty"] - qty_hand) < 1e-9, (buys[0]["qty"], qty_hand)
    nav_hand = qty_hand * 190.0
    assert abs(float(r["eq"].iloc[-1]) - nav_hand) < 1e-9, (r["eq"].iloc[-1], nav_hand)
    assert not [x for x in led if x["side"] == "SELL"], "청산 규칙이 없는데 매도가 있다"


def t_hand_ledger_mode_a_is_one_day_ahead():
    """같은 fixture 에서 모드 A 는 행 21 **종가 110** 에 산다 — 하루 빠르고 같은 가격.
    (A 의 이득은 이 fixture 에서 '하루 더 보유'로 나타난다.)"""
    close, opens, ret20 = hand_fixture()
    led = []
    E.sim(close, ret20, HAND_CFG, cash0=1000.0, opens=opens, mode="A", ledger=led)
    buys = [x for x in led if x["side"] == "BUY"]
    assert len(buys) == 1 and buys[0]["order_date"] == close.index[21], buys
    assert abs(buys[0]["fill_px"] - 110.0) < 1e-9


def t_hand_ledger_integer_order_uses_prior_close():
    """정수 주문(frac=False): 수량은 **t-1 종가**로 산정한다 (t 시가로 정하면 미래 정보).
    손계산: 예산 1000, t-1 종가 110 → int(1000/110) = 9주. 체결가는 t 시가 110.
    """
    close, opens, ret20 = hand_fixture()
    led = []
    E.sim(close, ret20, dict(HAND_CFG, frac=False), cash0=1000.0, opens=opens,
          mode="C", ledger=led)
    b = [x for x in led if x["side"] == "BUY"][0]
    assert b["qty"] == 9.0, b["qty"]
    assert abs(b["fill_px"] - 110.0) < 1e-9


def t_hand_accounting_identity():
    """원장만으로 최종 NAV 를 재구성한다 (엔진 값과 독립)."""
    close, opens, ret20 = hand_fixture()
    led = []
    r = E.sim(close, ret20, HAND_CFG, cash0=1000.0, opens=opens, mode="C", ledger=led)
    cash = 1000.0
    pos = {}
    for x in led:
        if x["side"] == "BUY":
            cash -= x["gross"] + x["cost_usd"]
            pos[x["symbol"]] = pos.get(x["symbol"], 0.0) + x["qty"]
        else:
            cash += x["gross"] - x["cost_usd"]
            pos[x["symbol"]] -= x["qty"]
    nav = cash + sum(q * close.iloc[-1][s] for s, q in pos.items())
    assert abs(nav - float(r["eq"].iloc[-1])) < 1e-9, (nav, r["eq"].iloc[-1])


TESTS = [
    ("당일 종가 불변식 (모드 C, 상승 섭동)", t_close_perturbation_mode_c),
    ("당일 종가 불변식 (모드 C, 하락 섭동)", t_close_perturbation_mode_c_down),
    ("주입: 누수 복원 → 불변식 실패", t_mutant_leak_breaks_close_invariance),
    ("주입: 슬롯 재사용 → 최대 종목 수 실패", t_mutant_slot_reuse_breaks_max_positions),
    ("주입: 부분체결 clamp 제거 → 실패", t_mutant_partial_fill_ignored),
    ("주입: intent_key 프로세스 고유값 → 중복 주문", t_mutant_duplicate_order_key),
    ("주입: stale 필터 제거 → 실패", t_mutant_stale_filter_removed),
    ("주입: 현금 유지 제거 → 대금 재사용 실패", t_mutant_cash_reserve_removed),
    ("손계산 원장 (모드 C)", t_hand_ledger_mode_c),
    ("손계산 원장 (모드 A 대조)", t_hand_ledger_mode_a_is_one_day_ahead),
    ("손계산 원장 (정수 주문 = t-1 종가 산정)", t_hand_ledger_integer_order_uses_prior_close),
    ("손계산 회계 항등식", t_hand_accounting_identity),
]


def main():
    print(f"불변식·결함주입·손계산 테스트 {len(TESTS)}건")
    for n, f in TESTS:
        check(n, f)
    print(f"\n통과 {len(PASS)} / 실패 {len(FAIL)}")
    for n, e in FAIL:
        print(f"  FAIL {n}: {e}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
