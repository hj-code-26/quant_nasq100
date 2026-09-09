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
# 전량 매도로 자리가 나면 같은 사이클에서 매수할 수 있다
out, skip = run([sell("A", 100), buy("AAPL")], ds, a)
assert [o["symbol"] for o in out] == ["A", "AAPL"] and not skip

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

print("validate_orders OK")
