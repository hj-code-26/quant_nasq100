"""validate_orders / forced_exits 자체 점검 — 돈이 나가는 유일한 경로.

네트워크·Claude 없이 dict 만 넣고 규칙이 지켜지는지 본다.  실행: python test_validate.py
"""
import datetime

import autotrade as at

# .env 값에 흔들리지 않게 규칙을 고정한다.
at.MAX_POSITIONS = 3
at.MAX_POSITION_PCT = 30
at.CASH_RESERVE_PCT = 10
at.MIN_ORDER_USD = 5
at.STOP_LOSS_PCT = 15
at.MOMENTUM_EXIT = True
at.RESERVE_RESTORE = 0          # 유지선 복원은 전용 블록에서만 켠다 (.env 에 켜져 있어도)

NOW = datetime.datetime.now(at.KST)
REGULAR = (NOW - datetime.timedelta(hours=1), NOW - datetime.timedelta(hours=1),
           NOW + datetime.timedelta(hours=3))   # 정규장 한복판 → 금액·소수점 주문 가능
CLOSED = None                                   # 휴장/장외 → 정수 주만
assert at.fractional_allowed(REGULAR) and not at.extended_hours(REGULAR)
assert not at.fractional_allowed(CLOSED)


def hold(qty, avg, last):
    return {"name": "x", "quantity": qty, "avg_price": avg, "last_price": last,
            "market_value": qty * last, "pnl_pct": round((last / avg - 1) * 100, 2)}


def acct(cash, holdings=None, open_orders=()):
    """open_orders 는 종목 목록(=미체결 매수 1건씩) 또는 {종목: [주문상세]}.

    account_state 가 상세(방향·미체결 잔량)를 같이 주므로 테스트도 같은 모양을 쓴다 —
    '미체결이 있다' 만으로는 기존 매도와 목표를 대사할 수 없기 때문이다.
    """
    h = holdings or {}
    by = (dict(open_orders) if isinstance(open_orders, dict) else
          {s: [{"orderId": "o-" + s, "side": "BUY", "quantity": 1.0, "filled": 0.0,
                "remaining": 1.0, "price": None, "state": "ACKNOWLEDGED"}]
           for s in open_orders})
    return {"cash": cash, "holdings": h, "open_orders": sorted(by),
            "open_by_symbol": by, "open_unknown": False,
            "total_value": cash + sum(x["market_value"] for x in h.values())}


def dec(sym, decision, price, ret20, account):
    factor, tier = at.momentum_tier(ret20)
    h = account["holdings"].get(sym, {})
    return {"symbol": sym, "decision": decision, "percentage": 80, "reason": "테스트",
            "status": {"current_price": price, "stock_balance": h.get("quantity", 0.0),
                       "avg_buy_price": h.get("avg_price", 0.0), "pnl_pct": h.get("pnl_pct"),
                       "ret_20d_pct": ret20, "momentum_tier": tier, "size_factor": factor}}


def run(orders, decisions, account, session=REGULAR):
    return at.validate_orders({"orders": orders, "summary": ""}, decisions, account, session)


def buy(sym, reason="테스트"):
    return {"symbol": sym, "side": "buy", "reason": reason}


def sell(sym, pct, reason="테스트"):
    return {"symbol": sym, "side": "sell", "sell_pct": pct, "reason": reason}


# --- 매수: 금액은 코드가 정한다 (종목당 한도 × 모멘텀 배수) ---
a = acct(1000)
out, skip = run([buy("AAPL")], {"AAPL": dec("AAPL", "buy", 200, 25, a)}, a)
assert len(out) == 1 and not skip
assert out[0]["amount_usd"] == 300 and out[0]["quantity"] is None   # 1000×30% × 1.0
# 중간 구간은 배수 0.7
a = acct(1000)
out, _ = run([buy("AAPL")], {"AAPL": dec("AAPL", "buy", 200, 15, a)}, a)
assert out[0]["amount_usd"] == 210

# --- 현금 유지선을 넘는 매수는 잘린다 ---
a = acct(50, {"MSFT": hold(5, 100, 190)})          # 총자산 1000, 현금 50, 유지선 100 → 여유 -50
out, skip = run([buy("AAPL")], {"AAPL": dec("AAPL", "buy", 200, 25, a),
                                "MSFT": dec("MSFT", "hold", 190, 25, a)}, a)
assert not out and "최소" in skip[0]["skipped"]
a = acct(200, {"MSFT": hold(5, 100, 160)})         # 총자산 1000, 여유 = 200 − 100 = 100
out, _ = run([buy("AAPL")], {"AAPL": dec("AAPL", "buy", 200, 25, a),
                             "MSFT": dec("MSFT", "hold", 160, 25, a)}, a)
assert out[0]["amount_usd"] == 100                 # 한도 300 이 아니라 현금 여유 100 까지만

# --- 모멘텀이 음수면 매수하지 않는다 ---
a = acct(1000)
out, skip = run([buy("AAPL")], {"AAPL": dec("AAPL", "buy", 200, -3, a)}, a)
assert not out and "모멘텀 음수" in skip[0]["skipped"]

# --- 미체결 주문이 있는 종목은 건드리지 않는다 (매수·매도 모두) ---
a = acct(1000, {"AAPL": hold(5, 200, 220)}, open_orders=["AAPL"])
out, skip = run([buy("AAPL"), sell("AAPL", 100)], {"AAPL": dec("AAPL", "buy", 220, 25, a)}, a)
assert not out and len(skip) == 2 and all("미체결" in s["skipped"] for s in skip)

# --- 최대 보유 종목 수 ---
a = acct(1000, {s: hold(1, 100, 100) for s in ("A", "B", "C")})
ds = {s: dec(s, "hold", 100, 25, a) for s in ("A", "B", "C")}
ds["AAPL"] = dec("AAPL", "buy", 200, 25, a)
out, skip = run([buy("AAPL")], ds, a)
assert not out and "최대 보유 종목 수" in skip[0]["skipped"]
# 매도를 '제출'했다고 슬롯이 비지는 않는다 — 체결로만 자리가 난다 (P0-4)
out, skip = run([sell("A", 100), buy("AAPL")], ds, a)
assert [o["symbol"] for o in out] == ["A"]
assert len(skip) == 1 and "최대 보유 종목 수" in skip[0]["skipped"]

# --- 정규장 밖: 정수 주만 ---
a = acct(1000)
out, _ = run([buy("AAPL")], {"AAPL": dec("AAPL", "buy", 200, 25, a)}, a, CLOSED)
assert out[0]["quantity"] == 1 and out[0]["amount_usd"] == 200      # 300 → 1주(200)
a = acct(1000)
out, skip = run([buy("AAPL")], {"AAPL": dec("AAPL", "buy", 500, 25, a)}, a, CLOSED)
assert not out and "1주 미만" in skip[0]["skipped"]                  # 한도 300 < 500

# --- 매도: 보유하지 않은 종목, 최소 금액, 전량 예외 ---
a = acct(100)
out, skip = run([sell("AAPL", 100)], {}, a)
assert not out and "보유하지 않은" in skip[0]["skipped"]
a = acct(100, {"AAPL": hold(10, 20, 20)})                           # 평가액 200
out, skip = run([sell("AAPL", 1)], {"AAPL": dec("AAPL", "sell", 20, 25, a)}, a)
assert not out and "최소 주문" in skip[0]["skipped"]                 # 1% = $2 < $5
a = acct(100, {"AAPL": hold(0.1, 20, 20)})                          # 평가액 $2
out, skip = run([sell("AAPL", 100)], {"AAPL": dec("AAPL", "sell", 20, 25, a)}, a)
assert out[0]["quantity"] == 0.1 and not skip                       # 전량 매도는 최소 금액 예외
out, _ = run([sell("AAPL", 50)], {"AAPL": dec("AAPL", "sell", 20, 25, a)}, a, CLOSED)
assert not out                                                      # 장외 부분 매도 → 정수 주 0
# 장외 **전량** 매도도 정수 주만 — 토스는 소수점 수량을 정규장 시장가 매도로만 받는다 (WDAY 400 재현)
a = acct(100, {"AAPL": hold(2.6, 20, 20)})
out, _ = run([sell("AAPL", 100)], {"AAPL": dec("AAPL", "sell", 20, 25, a)}, a, CLOSED)
assert out[0]["quantity"] == 2 and out[0]["whole"]                  # 2주 지금, 0.6주는 정규장에서
a = acct(100, {"AAPL": hold(0.621832, 20, 20)})
out, skip = run([sell("AAPL", 100)], {"AAPL": dec("AAPL", "sell", 20, 25, a)}, a, CLOSED)
assert not out and "정규장" in skip[0]["skipped"]                    # 1주 미만 → 거절될 주문을 내지 않는다

# --- 강제 청산: 손절 ---
a = acct(100, {"AAPL": hold(10, 100, 80)})                          # -20%
d = {"AAPL": dec("AAPL", "hold", 80, 25, a)}                        # Claude 는 hold 를 고집
assert [o["symbol"] for o in at.forced_exits(d, a)] == ["AAPL"]
out, _ = run([], d, a)
assert len(out) == 1 and out[0]["side"] == "sell" and out[0]["quantity"] == 10
assert "손절" in out[0]["reason"]

# --- 강제 청산: 모멘텀 음수 (갈아탈 후보가 있을 때만) ---
a = acct(100, {"AAPL": hold(10, 100, 105)})                         # 손절선은 안 닿음
d = {"AAPL": dec("AAPL", "hold", 105, -2, a), "MSFT": dec("MSFT", "buy", 400, 25, a)}
out, _ = run([], d, a)
assert len(out) == 1 and "모멘텀 청산" in out[0]["reason"]
# 대체 후보가 없으면 회전이 아니라 저점 매도일 뿐이므로 들고 간다
assert at.forced_exits({"AAPL": dec("AAPL", "hold", 105, -2, a)}, a) == []
# 판단이 아예 없어도(Claude 호출 실패) 손절은 나간다
a = acct(100, {"AAPL": hold(10, 100, 80)})
assert len(at.forced_exits({}, a)) == 1
# 모멘텀 정상 + 손실 미달이면 그대로 둔다
a = acct(100, {"AAPL": hold(10, 100, 105)})
assert at.forced_exits({"AAPL": dec("AAPL", "hold", 105, 25, a)}, a) == []

# --- 강제 청산과 Claude 매도가 겹치면 주문은 1건, 같은 사이클에 재매수하지 않는다 ---
a = acct(1000, {"AAPL": hold(10, 100, 80)})
d = {"AAPL": dec("AAPL", "sell", 80, 25, a)}
out, skip = run([sell("AAPL", 100), buy("AAPL")], d, a)
assert len(out) == 1 and "손절" in out[0]["reason"]
assert len(skip) == 1 and "강제 청산" in skip[0]["skipped"]

# --- 스위치를 끄면 강제 청산도 없다 ---
at.STOP_LOSS_PCT, at.MOMENTUM_EXIT = 0, False
a = acct(100, {"AAPL": hold(10, 100, 50)})
assert at.forced_exits({"AAPL": dec("AAPL", "hold", 50, -30, a)}, a) == []
at.STOP_LOSS_PCT, at.MOMENTUM_EXIT = 15, True

# --- 만기 청산: 보유 거래일이 상한을 넘으면 Claude 판단과 무관하게 전량 매도 ---
at.MAX_HOLD_DAYS = 20
old_days = at.trading_days_since
at.trading_days_since = lambda d: None if d is None else d       # 테스트는 일수를 직접 넣는다
a = acct(100, {"AAPL": hold(10, 100, 105)})
d = {"AAPL": dec("AAPL", "hold", 105, 25, a)}
out, _ = run([], d, a, session=REGULAR)                          # entries 없으면 만기 판단 불가
assert out == []
out, _ = at.validate_orders({"orders": [], "summary": ""}, d, a, REGULAR, {"AAPL": 20})
assert len(out) == 1 and "만기" in out[0]["reason"]
out, _ = at.validate_orders({"orders": [], "summary": ""}, d, a, REGULAR, {"AAPL": 19})
assert out == []
at.trading_days_since = old_days

# --- 변동성 타겟: 노출 상한 초과분은 실제 매도 주문으로 나가야 한다 (plan 에만 넣고 끝나면 안 된다) ---
old_cap = at.exposure_cap
at.exposure_cap = lambda verbose=False: 0.5                      # 주식 노출 상한 50%
a = acct(100, {"AAPL": hold(10, 100, 105)})                      # 주식 1050 / 총 1150 → 초과분 있음
out, _ = at.validate_orders({"orders": [], "summary": ""},
                            {"AAPL": dec("AAPL", "hold", 105, 25, a)}, a, REGULAR, {})
assert len(out) == 1 and "변동성 타겟" in out[0]["reason"]
at.exposure_cap = old_cap



# --- P0-2: 위험 축소는 미체결 때문에 미뤄지지 않는다 (취소 후 매도) ---
at.STOP_LOSS_PCT = 15
a = acct(100, {"AAPL": hold(10, 100, 80)}, open_orders=["AAPL"])   # -20%, 미체결 있음
d = {"AAPL": dec("AAPL", "hold", 80, 25, a)}
out, skip = run([], d, a)
assert len(out) == 1 and out[0]["cancel_first"] is True and "손절" in out[0]["reason"]
# 반대로 Claude 의 재량 매도는 예전처럼 미룬다 (경합을 만들 이유가 없다)
at.STOP_LOSS_PCT = 0
a = acct(100, {"AAPL": hold(10, 100, 105)}, open_orders=["AAPL"])
out, skip = run([sell("AAPL", 50)], {"AAPL": dec("AAPL", "sell", 105, 25, a)}, a)
assert not out and "미체결" in skip[0]["skipped"]
at.STOP_LOSS_PCT = 15

# --- P0-3: decisions 없이도(=LLM 전면 실패) 손절·만기가 나온다 ---
a = acct(100, {"AAPL": hold(10, 100, 80), "MSFT": hold(1, 10, 10)})
out, _ = at.validate_orders({"orders": [], "summary": ""}, {}, a, REGULAR, {})
assert [o["symbol"] for o in out] == ["AAPL"] and "손절" in out[0]["reason"]

# --- 같은 종목 중복 주문은 한 건만 나간다 ---
at.STOP_LOSS_PCT = 0
a = acct(1000, {"AAPL": hold(10, 100, 105)})
d = {"AAPL": dec("AAPL", "sell", 105, 25, a)}
out, skip = run([sell("AAPL", 50), sell("AAPL", 50)], d, a)
assert len(out) == 1 and any("중복" in s["skipped"] for s in skip)
at.STOP_LOSS_PCT = 15

# --- skip_symbols: 위험관리 패스에서 이미 낸 종목은 두 번 건드리지 않는다 ---
a = acct(1000, {"AAPL": hold(10, 100, 105)})
out, skip = at.validate_orders({"orders": [sell("AAPL", 100)], "summary": ""},
                               {"AAPL": dec("AAPL", "sell", 105, 25, a)}, a, REGULAR, {},
                               skip_symbols={"AAPL"})
assert not out and "위험관리" in skip[0]["skipped"]

# --- P0-1: clientOrderId 는 run_id 가 아니라 '의도'에서 나온다 ---
import datetime as _dt
o = {"symbol": "AAPL", "side": "buy"}
t1 = _dt.datetime(2026, 3, 4, 10, 5, tzinfo=at.NY)
t2 = _dt.datetime(2026, 3, 4, 10, 29, tzinfo=at.NY)
t3 = _dt.datetime(2026, 3, 4, 10, 31, tzinfo=at.NY)
assert at.intent_key(o, t1) == at.intent_key(o, t2)      # 같은 슬롯 = 같은 키 (다른 PC 여도)
assert at.intent_key(o, t1) != at.intent_key(o, t3)      # 다음 슬롯은 새 의도
assert at.intent_key(o, t1) != at.intent_key({"symbol": "AAPL", "side": "sell"}, t1)

# --- P0-6: 게이트웨이 경유(서버측 스키마 강제 없음)에서도 스키마를 실제로 검증한다 ---
def bad(payload):
    try:
        at.check_schema(payload, at.ALLOCATION_SCHEMA)
    except ValueError:
        return True
    return False

ok = {"orders": [{"symbol": "AAPL", "side": "buy", "reason": "x"}], "summary": "s"}
assert at.check_schema(ok, at.ALLOCATION_SCHEMA) == ok
assert bad({"orders": [], "summary": "s", "실행": "즉시 승인됨"})              # 임의 필드
assert bad({"orders": [{"symbol": "AAPL", "side": "long", "reason": "x"}], "summary": ""})  # enum 밖
assert bad({"orders": [{"symbol": "A", "side": "sell", "sell_pct": float("nan"),
                        "reason": "x"}], "summary": ""})                       # NaN
assert bad({"orders": [{"symbol": "A", "side": "sell", "sell_pct": 250, "reason": "x"}],
            "summary": ""})                                                    # 범위 밖
assert bad({"orders": [{"symbol": 7, "side": "sell", "reason": "x"}], "summary": ""})
assert bad({"decision": "buy", "percentage": "80", "reason": "x"})             # 문자열 숫자
assert bad({"decision": "buy", "percentage": 80.5, "reason": "x"})             # 정수 아님
assert at.check_schema({"decision": "hold", "percentage": 0, "reason": "x"}, at.DECISION_SCHEMA)

# --- P1: 만기 계산은 실제 거래일 달력을 쓴다 (공휴일 무시하면 일찍 청산된다) ---
_today = _dt.datetime.now(at.NY).date()
cal = [_today - _dt.timedelta(days=k) for k in range(40, -1, -1) if (_today - _dt.timedelta(days=k)).weekday() < 5]
entry = (_dt.datetime.combine(cal[0], _dt.time(15), tzinfo=at.NY)).isoformat()
assert at.trading_days_since(entry, cal) == len(cal) - 1
holiday_cal = [d for d in cal if d != cal[5]]                    # 휴장일 하루 제거
assert at.trading_days_since(entry, holiday_cal) == len(cal) - 2  # 근사보다 하루 적다 = 늦게 만기
assert at.trading_days_since(None, cal) is None


# --- P1: 변동성 타겟은 '현금 섞인 계좌 변동성'을 절대 상한으로 쓰면 안 된다 ---
# 합성 반례: 변동성 60% 자산을 절반만 들고 있으면 계좌 변동성은 30%.
#   옛 방식 cap = 30/30 = 100%  → 다 사라 → 다음엔 계좌 60% → cap 50% → 진동
#   새 방식 cap = 30/(30/0.5) = 50% → 목표 비중에서 고정 (진동 없음)
import math, pathlib, sqlite3, tempfile
_tmp = pathlib.Path(tempfile.mkdtemp()) / "eq.db"
_old_db, at.DB_PATH = at.DB_PATH, _tmp
at.VOL_TARGET_PCT, at.VOL_WINDOW = 30, 60
at.initialize_db()
_daily = 0.60 / math.sqrt(252)
with sqlite3.connect(_tmp) as _c:
    v = 1000.0
    for i in range(at.VOL_WINDOW + 1):                  # ±1σ 를 번갈아 → 실현 변동성 ≈ 60%
        w = 0.5                                          # 주식 비중은 계속 50%
        _c.execute("INSERT INTO equity (date, total_value, stock_value, cashflow) VALUES (?,?,?,0)",
                   (f"2026-01-{i + 1:03d}", v, v * w))
        v *= 1 + (_daily * w) * (1 if i % 2 else -1)
_cap = at.exposure_cap(verbose=True)
assert 0.40 < _cap < 0.62, _cap                          # 100% 가 아니라 50% 부근


# --- 오버레이: 하락 국면 노출 축소 (REGIME_DERISK). 기본값 OFF 가 지켜져야 한다 ---
def _idx(closes):
    return at.pd.DataFrame({"close": closes},
                           index=at.pd.date_range("2025-01-01", periods=len(closes), freq="D"))

FLAT, DROP = [100.0] * 260, [80.0] * 40                  # 252일 고점 100 → 현재 80 = -20%
_bear = _idx(FLAT + DROP)

at.BEAR_EXPOSURE_PCT, at.BEAR_DD_PCT, at.BEAR_OFF_DAYS = None, 15, 3
assert at.bear_derisk(_bear) is None                     # ★ 기본값 OFF — 신호가 켜져도 개입 없음
assert at.exposure_cap() == _cap                         # 상한도 변동성 타겟 그대로

at.BEAR_EXPOSURE_PCT = 70
assert at.bear_derisk(_bear) == 0.70                     # -20% ≤ -15% → ON
assert at.bear_derisk(_idx(FLAT + [95.0] * 40)) is None   # -5% 는 임계 미달 → OFF
assert at.bear_derisk(_idx(FLAT[:100])) is None          # 252봉 미만이면 판정 안 한다
assert at.bear_derisk(None) is None                      # 지수를 못 받은 날은 개입하지 않는다
# ★ 판정 불가 ≠ 신호 OFF. 데이터가 없으면 보호가 조용히 사라지는 것이므로 구분해서 로그한다
assert at.bear_data_ok(_bear) and not at.bear_data_ok(None) and not at.bear_data_ok(_idx(FLAT[:252]))

# 복귀 지연: 오늘은 고점 회복이어도 최근 BEAR_OFF_DAYS 일 안에 ON 이 있으면 유지한다
_rebound = _idx(FLAT + [80.0] * 38 + [100.0, 100.0])
assert at.bear_derisk(_rebound) == 0.70                  # 2일 전이 ON → 지연 3일이라 유지
at.BEAR_OFF_DAYS = 1
assert at.bear_derisk(_rebound) is None                  # 지연 1일이면 오늘만 본다 → 해제
at.BEAR_OFF_DAYS = 3

# ★ 미래 참조 없음: 데이터를 t 에서 잘라도 t 의 판정이 바뀌면 안 된다
for _k in (270, 285, 299):
    assert at.bear_derisk(_bear.iloc[:_k]) == at.bear_derisk(_bear.iloc[:_k]), _k
assert at.bear_derisk(_bear.iloc[:261]) == 0.70          # 하락 첫날에 이미 켜진다 (지연 없음)
assert at.bear_derisk(_bear.iloc[:260]) is None          # 그 전날은 아직 아니다

# 상한은 둘 중 낮은 쪽 — 겹쳐도 0~100% 를 벗어나지 않는다
at.BEAR_CAP = at.bear_derisk(_bear)
assert at.exposure_cap() == min(_cap, 0.70) == _cap      # 변동성 타겟 50% 가 더 낮다
at.BEAR_EXPOSURE_PCT = 30
at.BEAR_CAP = at.bear_derisk(_bear)
assert at.exposure_cap() == 0.30                         # 이번엔 오버레이가 더 낮다

# 축소 매도에 사유 코드가 남는가 (나중에 오버레이 기여를 분리하려면 필수)
_a = acct(0, {"AAPL": hold(10, 100, 100)})               # 주식 100% → 상한 30% 초과
_o, _ = run([], {}, _a)
_r = [o for o in _o if "REGIME_DERISK" in o.get("reason", "")]
assert _r and _r[0]["side"] == "sell", _o
at.BEAR_CAP, at.BEAR_EXPOSURE_PCT = None, None           # 킬 스위치 — 기준선 동작으로 복귀
assert at.exposure_cap() == _cap

at.DB_PATH = _old_db


# ---------- A안 D1~D4: 코드·프롬프트 계약과 관측성 ----------
# D1. 하락 국면은 모멘텀으로 매수를 막지 않는다. 프롬프트(instructions.md 6-1)가 이 구간을
#     모르면 LLM 이 "약 구간이면 hold" 로 저변동성 후보를 전부 거부한다 (2026-09-16 run 63).
_f_bear, _t_bear = at.momentum_tier(3.7, "하락")
assert _f_bear == 1.0 and "모멘텀 무관" in _t_bear
assert at.momentum_tier(3.7, "보통")[0] == 0.4           # 평상시엔 약 구간 배수 0.4
assert at.momentum_tier(-1.0, "보통")[0] == 0.0          # 음수는 매수 안 함
assert at.momentum_tier(-1.0, "하락")[0] == 1.0          # 하락 국면은 음수여도 배수 유지
_md = (at.ROOT / "instructions.md").read_text(encoding="utf-8")
assert _t_bear in _md, "instructions.md 에 하락 국면 구간 이름이 없다 — LLM 이 약 구간으로 오인한다"
assert "시장 국면" in _md and "6-1" in _md

# D2. 1단계 페이로드 라벨이 screen() 의 실제 정렬 기준과 일치해야 한다.
#     예전엔 하락 국면에도 "20일 수익률 내림차순" 이라 적어 LLM 에 거짓 입력을 줬다.
_payloads = []
_real_ask = at.ask_claude
at.ask_claude = lambda f, payload, schema, **kw: (_payloads.append(payload)
                                                  or {"candidates": []})
_rows = [{"symbol": "EA", "ret_20d_pct": 2.0, "atr_pct": 0.2},
         {"symbol": "NVDA", "ret_20d_pct": 30.0, "atr_pct": 5.0}]
at.pick_candidates(_rows, acct(100), "하락")
at.pick_candidates(_rows, acct(100), "보통")
at.ask_claude = _real_ask
_bear_p, _norm_p = _payloads
assert _bear_p["시장 국면"] == "하락"
assert any("atr_pct" in k for k in _bear_p), _bear_p.keys()
assert not any("20일 수익률 내림차순" in k for k in _bear_p), "하락인데 모멘텀 정렬이라 라벨링"
assert "atr_pct" in _bear_p["선정 규칙"]["표 정렬 기준"]
assert any("20일 수익률 내림차순" in k for k in _norm_p), _norm_p.keys()
# 프롬프트가 코드 판정을 쓰는가 — 자체 판정(표 중앙값)이 남아 있으면 두 판정이 엇갈린다
_sc = (at.ROOT / "instructions_screen.md").read_text(encoding="utf-8")
assert "중앙값이 −5% 미만" not in _sc and "시장 국면" in _sc

# D3. 주문이 0건이어도 퍼널이 로그에 남아야 한다 (43사이클 무주문을 로그로 못 봤다).
_seen = []
_real_info, _real_warn = at.log.info, at.log.warning
at.log.info = lambda m, *a, **kw: _seen.append(("info", m % a if a else m))
at.log.warning = lambda m, *a, **kw: _seen.append(("warn", m % a if a else m))
try:
    _a = acct(0.01, {"MU": hold(1, 100, 100)})           # 현금 $0.01 → 가용 음수
    _d = {"AAPL": dec("AAPL", "buy", 50, 25.0, _a)}
    _o, _sk = run([buy("AAPL")], _d, _a)
    at.log_funnel([1] * 101, [{"symbol": "AAPL"}], _d,
                  {"orders": [buy("AAPL")], "summary": ""}, [], _o, _sk, _a)
finally:
    at.log.info, at.log.warning = _real_info, _real_warn
assert _o == [], _o                                       # 가용현금 음수 → 주문 0건
_txt = " ".join(m for _, m in _seen)
assert "퍼널:" in _txt and "가용현금" in _txt, _txt
assert any(k == "warn" and "매수 판단" in m for k, m in _seen), "전량 차단인데 경고가 없다"
assert "차단 1건" in _txt, _txt

# D4. 고아 run(status='running') 은 기동 때 interrupted 로 마감된다.
import sqlite3 as _sq
import tempfile as _tmp
_tmpdb = pathlib.Path(_tmp.mkdtemp()) / "t.db"
at.DB_PATH = _tmpdb
at.initialize_db()
at.db_insert("runs", {"timestamp": at._now(), "status": "running"})
at.db_insert("runs", {"timestamp": at._now(), "status": "done"})
at.initialize_db()                                        # 재기동
with _sq.connect(_tmpdb) as _c:
    _st = [r[0] for r in _c.execute("SELECT status FROM runs ORDER BY id")]
assert _st == ["interrupted", "done"], _st
at.initialize_db()
with _sq.connect(_tmpdb) as _c:                           # 멱등 — 두 번 돌려도 done 은 그대로
    assert [r[0] for r in _c.execute("SELECT status FROM runs ORDER BY id")] == _st
at.DB_PATH = _old_db

# ---------- 조건부 만기: HOLD_EXTEND_TOP ----------
# "시간이 됐으니 판다" → "시간이 됐으니 같은 기준으로 다시 묻는다".
# 근거 research/exit_condition.md — 사전등록 DSR 기준은 통과하지 못했다(기본 꺼짐).
_old_extend = at.HOLD_EXTEND_TOP
_ENTRY = (datetime.datetime.now(at.NY) - datetime.timedelta(days=120)).isoformat()
_a = acct(0, {"AAPL": hold(1, 100, 100), "TSLA": hold(1, 100, 100)})
_e = {"AAPL": _ENTRY, "TSLA": _ENTRY}                      # 둘 다 만기 초과
_rows = [{"symbol": "AAPL", "ret_20d_pct": 25.0, "atr_pct": 3.0},
         {"symbol": "ZZZZ", "ret_20d_pct": 20.0, "atr_pct": 0.5},
         {"symbol": "TSLA", "ret_20d_pct": 1.0, "atr_pct": 9.0}]

at.HOLD_EXTEND_TOP = 0                                     # 꺼짐 = 현행 — 시간만으로 전량 매도
assert at.extendable(_rows, _a["holdings"], "보통") == set()
_o, _ = at.validate_orders({"orders": [], "summary": ""}, {}, _a, REGULAR, _e)
assert sorted(o["symbol"] for o in _o) == ["AAPL", "TSLA"], _o
assert all("만기" in o["reason"] for o in _o)

at.HOLD_EXTEND_TOP = 2                                     # 켜짐 — 상위 2위 안이면 유지
assert at.extendable(_rows, _a["holdings"], "보통") == {"AAPL"}   # ZZZZ 는 미보유
_o, _ = at.validate_orders({"orders": [], "summary": ""}, {}, _a, REGULAR, _e,
                           extend_ok=at.extendable(_rows, _a["holdings"], "보통"))
assert [o["symbol"] for o in _o] == ["TSLA"], _o            # 순위 밖만 판다
assert "상위 2위 밖" in _o[0]["reason"], _o[0]["reason"]

# 평상시엔 진입과 같은 조건(20일 수익률 > 0)을 함께 건다 — 음수면 연장 자격 없음
_neg = [{"symbol": "AAPL", "ret_20d_pct": -5.0, "atr_pct": 3.0}]
assert at.extendable(_neg, _a["holdings"], "보통") == set()
# 하락 국면은 저변동성으로 고르므로 모멘텀 부호를 보지 않는다 (momentum_tier 와 같은 규칙)
assert at.extendable(_neg, _a["holdings"], "하락") == {"AAPL"}

# 연장은 손절·노출축소를 막지 못한다 — 위험 축소가 만기 연장보다 우선이다
at.STOP_LOSS_PCT = 15
_a2 = acct(0, {"AAPL": hold(1, 100, 80)})                  # -20% → 손절 대상
_o, _ = at.validate_orders({"orders": [], "summary": ""}, {}, _a2, REGULAR,
                           {"AAPL": _ENTRY}, extend_ok={"AAPL"})
assert [o["symbol"] for o in _o] == ["AAPL"] and "손절" in _o[0]["reason"], _o

# 프롬프트에 나가는 만기 문구가 실제 설정을 따라간다 (옛 "20 거래일" 고정 문구 제거)
assert "상위 2위" in at.hold_rule() and "고정이 아니라" in at.hold_rule()
at.HOLD_EXTEND_TOP = 0
assert "상위" not in at.hold_rule() and str(at.MAX_HOLD_DAYS) in at.hold_rule()
at.MAX_HOLD_DAYS = 0
assert "없음" in at.hold_rule()
at.MAX_HOLD_DAYS, at.HOLD_EXTEND_TOP = 20, _old_extend

# ---------- 만기 연장 섀도 (주문에 영향 없이 판정만 기록) ----------
at.DB_PATH = _tmpdb
at.initialize_db()
at.HOLD_EXTEND_TOP, at.HOLD_EXTEND_SHADOW = 0, 2       # 실주문은 현행, 섀도만 상위 2위
_h = {"AAPL": hold(1, 100, 100), "TSLA": hold(1, 100, 100)}
at.log_hold_shadow(1, _rows, _h, {"AAPL": _ENTRY, "TSLA": _ENTRY}, "보통")
with _sq.connect(_tmpdb) as _c:
    _sh = {r[0]: r[1:] for r in _c.execute(
        "SELECT symbol, rank, would_extend, at_expiry, live_extend_top FROM hold_shadow")}
assert _sh["AAPL"] == (1, 1, 1, 0), _sh                # 1위 → 연장 판정, 실운영은 꺼짐
# 순위는 **유니버스 전체** 기준이다 (미보유 ZZZZ 가 2위) — 시뮬의 '상위 N위' 와 같은 정의.
# 보유 종목끼리만 매기면 보유가 적을 때 전부 연장돼 버린다.
assert _sh["TSLA"] == (3, 0, 1, 0), _sh
at.HOLD_EXTEND_SHADOW = 1
at.log_hold_shadow(2, _rows, _h, {"AAPL": _ENTRY, "TSLA": _ENTRY}, "보통")
with _sq.connect(_tmpdb) as _c:
    _sh2 = {r[0]: r[1] for r in _c.execute(
        "SELECT symbol, would_extend FROM hold_shadow WHERE run_id=2")}
assert _sh2 == {"AAPL": 1, "TSLA": 0}, _sh2            # 상위 1위만 연장

# 섀도는 주문을 만들지 않는다 — 같은 상태에서 실제 주문은 여전히 전량 만기 청산
_o, _ = at.validate_orders({"orders": [], "summary": ""}, {}, acct(0, _h), REGULAR,
                           {"AAPL": _ENTRY, "TSLA": _ENTRY})
assert sorted(o["symbol"] for o in _o) == ["AAPL", "TSLA"], _o

# 순위 정의가 extendable 과 같은가 (두 곳이 갈리면 섀도가 거짓말이 된다)
at.HOLD_EXTEND_TOP = 2
assert at.extendable(_rows, _h, "보통") == {
    s for s, r in at.hold_ranks(_rows, "보통").items() if 0 < r <= 2} & set(_h)
at.HOLD_EXTEND_SHADOW = 0                              # 꺼짐 → 기록 없음
at.log_hold_shadow(3, _rows, _h, {"AAPL": _ENTRY}, "보통")
with _sq.connect(_tmpdb) as _c:
    assert _c.execute("SELECT COUNT(*) FROM hold_shadow WHERE run_id=3").fetchone()[0] == 0
at.HOLD_EXTEND_TOP, at.HOLD_EXTEND_SHADOW = _old_extend, 20
at.DB_PATH = _old_db

# --- 현금 유지선 복원 (RESERVE_RESTORE): 현금이 바닥난 계좌의 매수 영구 교착을 푼다 ---
_old = (at.STOP_LOSS_PCT, at.MOMENTUM_EXIT, at.MAX_HOLD_DAYS, at.RESERVE_RESTORE, at.exposure_cap)
at.STOP_LOSS_PCT, at.MOMENTUM_EXIT, at.MAX_HOLD_DAYS = 0, False, 0
at.exposure_cap = lambda verbose=False: None
_h = {"AAPL": hold(1, 500, 600), "MSFT": hold(1, 300, 400)}       # 주식 1000, 현금 0.01 → 유지선 100
at.RESERVE_RESTORE = 0                                             # 꺼짐(기본) → 현행 동작 그대로
_o, _ = at.validate_orders({"orders": [], "summary": ""}, {}, acct(0.01, _h), REGULAR, {})
assert _o == [], _o
at.RESERVE_RESTORE = 0.5
_o, _ = at.validate_orders({"orders": [], "summary": ""}, {}, acct(0.01, _h), REGULAR, {})
assert len(_o) == 1 and _o[0]["symbol"] == "AAPL" and "유지선" in _o[0]["reason"], _o   # 큰 종목부터
assert abs(_o[0]["amount_usd"] - 99.99) < 0.1, _o    # 유지선까지만 판다 (전량 아님, sell_pct 반올림 오차)
# 현금이 유지선의 절반 이상이면 발동하지 않는다 (가격 상승으로 조금 모자란 건 정상 상태)
_o, _ = at.validate_orders({"orders": [], "summary": ""}, {}, acct(60, _h), REGULAR, {})
assert _o == [], _o
# 이미 떠 있는 매도가 부족분을 덮으면 두 번 팔지 않는다
_pend = {"MSFT": [{"orderId": "p", "side": "SELL", "quantity": 1.0, "filled": 0.0, "remaining": 1.0,
                   "price": None, "state": "ACKNOWLEDGED"}]}
_o, _ = at.validate_orders({"orders": [], "summary": ""}, {}, acct(0.01, _h, _pend), REGULAR, {})
assert _o == [], _o
# 만기 청산으로 들어올 돈도 먼저 친다 — AAPL 만기 전량이면 추가 복원 매도는 없다
at.MAX_HOLD_DAYS = 20
_old_days = at.trading_days_since
at.trading_days_since = lambda d: None if d is None else d
_o, _ = at.validate_orders({"orders": [], "summary": ""}, {}, acct(0.01, _h), REGULAR, {"AAPL": 20})
assert [(o["symbol"], "만기" in o["reason"]) for o in _o] == [("AAPL", True)], _o
at.trading_days_since = _old_days
# 장외(정수 주만)에서 1주 미만 부분 매도는 다음 정규장으로 미룬다 — 소수점 주문 거절(400) 방지
at.MAX_HOLD_DAYS = 0
_o, _sk = at.validate_orders({"orders": [], "summary": ""}, {}, acct(0.01, _h), CLOSED, {})
assert _o == [] and any("FRACTIONAL_SESSION" in s["skipped"] for s in _sk), (_o, _sk)
(at.STOP_LOSS_PCT, at.MOMENTUM_EXIT, at.MAX_HOLD_DAYS, at.RESERVE_RESTORE, at.exposure_cap) = _old

# --- 살 돈이 없으면 매수 분석(Claude)을 건너뛴다 ---
assert not at.can_buy(acct(0.01, _h)) and at.can_buy(acct(200, _h))
assert not at.can_buy(acct(104, _h))                               # 104 − 유지선 110.4 < 5
_hold_only = {"AAPL": {"decision": "hold"}}
assert not at.needs_allocation(False, _hold_only)                  # 보유 판단이 전부 hold → 배분 생략
assert at.needs_allocation(False, {"AAPL": {"decision": "sell"}})   # 매도 판단이 있으면 배분한다
assert at.needs_allocation(True, _hold_only)                       # 살 수 있으면 배분한다

print("validate_orders OK")
