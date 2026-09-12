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
    h = holdings or {}
    return {"cash": cash, "holdings": h, "open_orders": list(open_orders),
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

print("validate_orders OK")
