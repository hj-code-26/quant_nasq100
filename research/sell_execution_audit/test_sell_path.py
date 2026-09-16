"""매도 실행 경로 재현·회귀 테스트 — 네트워크·LLM·실주문 없이 돈이 나가는 길만 본다.

실행:  python research/sell_execution_audit/test_sell_path.py

여기서 검증하는 불변식 (전략 기준은 건드리지 않는다):
  1. 매도 '의도'가 말없이 사라지지 않는다 (미체결 주문이 있어도 강제 청산은 남는다).
  2. 확인할 수 없는 상태(UNKNOWN)를 안전한 상태로 추정하지 않는다 — 보류하고 기록한다.
  3. 확인된 매도 가능 수량과 남은 목표량을 넘겨 팔지 않는다.
  4. ACKNOWLEDGED(접수) 와 FILLED(체결) 는 다르다. 취소 요청은 취소 완료가 아니다.
  5. 정상적인 거래 제한(소수점·세션·최소금액)은 버그가 아니므로 없애지 않는다.

가짜 브로커만 쓴다. create_order/cancel_order/modify_order 호출은 전부 기록되고,
DRY_RUN 경로에서는 0 회여야 한다.
"""
import datetime
import pathlib
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import autotrade as at                                        # noqa: E402
from toss import TossError                                    # noqa: E402

# 전략값은 테스트 안에서 고정한다 (.env 에 흔들리지 않게). 운영 기본값을 바꾸는 게 아니다.
at.MAX_POSITIONS, at.MAX_POSITION_PCT = 3, 30
at.CASH_RESERVE_PCT, at.MIN_ORDER_USD = 10, 5
at.STOP_LOSS_PCT, at.MOMENTUM_EXIT, at.MAX_HOLD_DAYS = 15, False, 20
at.VOL_TARGET_PCT, at.HOLD_EXTEND_TOP, at.HOLD_EXTEND_SHADOW = 0, 0, 0
# 테스트는 '보유 거래일수'를 entries 에 직접 숫자로 넣는다 (달력·시계 의존 제거).
at.trading_days_since = lambda ts, calendar=None: ts if isinstance(ts, int) else None

NOW = datetime.datetime.now(at.KST)
REGULAR = (NOW - datetime.timedelta(hours=1), NOW - datetime.timedelta(hours=1),
           NOW + datetime.timedelta(hours=3), NOW + datetime.timedelta(hours=4))
OFF = (NOW - datetime.timedelta(hours=1), NOW + datetime.timedelta(hours=6),
       NOW + datetime.timedelta(hours=9), NOW + datetime.timedelta(hours=10))
assert at.fractional_allowed(REGULAR) and not at.extended_hours(REGULAR)
assert not at.fractional_allowed(OFF) and at.extended_hours(OFF)

FAILS = []


def check(name, cond, detail=""):
    print(("  ok  " if cond else "  FAIL") + "  " + name +
          (" — " + str(detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def hold(qty, avg, last):
    return {"name": "x", "quantity": qty, "avg_price": avg, "last_price": last,
            "market_value": qty * last, "pnl_pct": round((last / avg - 1) * 100, 2)}


def acct(cash, holdings=None, open_orders=None, **extra):
    h = holdings or {}
    a = {"cash": cash, "holdings": h, "open_orders": sorted(open_orders or {}),
         "open_by_symbol": dict(open_orders or {}), "open_unknown": False,
         "total_value": cash + sum(x["market_value"] for x in h.values())}
    a.update(extra)
    return a


def opn(side, qty, filled=0.0, price=None, oid="o1"):
    """미체결 주문 한 건 (account_state 가 만드는 모양)."""
    return {"orderId": oid, "side": side.upper(), "quantity": qty,
            "filled": filled, "remaining": qty - filled, "price": price}


def dec(sym, decision, price, ret20, account):
    factor, tier = at.momentum_tier(ret20)
    h = account["holdings"].get(sym, {})
    return {"symbol": sym, "decision": decision, "percentage": 80, "reason": "테스트",
            "status": {"current_price": price, "stock_balance": h.get("quantity", 0.0),
                       "avg_buy_price": h.get("avg_price", 0.0), "pnl_pct": h.get("pnl_pct"),
                       "ret_20d_pct": ret20, "momentum_tier": tier, "size_factor": factor}}


def vo(account, decisions=None, plan=None, entries=None, session=REGULAR, **kw):
    return at.validate_orders({"orders": plan or [], "summary": ""}, decisions or {},
                              account, session, entries or {}, **kw)


class FakeToss:
    """주문 API 를 흉내내는 가짜 브로커. 실주문은 절대 나가지 않는다."""

    def __init__(self, opens=(), sellable=None, fail=(), closed=()):
        self.open_rows = [dict(o) for o in opens]
        self.closed_rows = [dict(o) for o in closed]
        self._sellable = sellable
        self.fail = set(fail)          # {"orders","sellable","cancel","create"}
        self.calls = []                # 상태 변경 호출 기록

    # --- 조회 ---
    def orders(self, status="OPEN", symbol=None):
        if "orders" in self.fail:
            raise TossError(503, "retry-exhausted", "조회 실패")
        rows = self.open_rows if status == "OPEN" else self.closed_rows
        if symbol:
            rows = [r for r in rows if r.get("symbol") == symbol]
        return {"orders": [dict(r) for r in rows]}

    def sellable_quantity(self, symbol):
        if "sellable" in self.fail:
            raise TossError(503, "retry-exhausted", "매도가능 조회 실패")
        return {} if self._sellable is None else {"sellableQuantity": self._sellable}

    def us_market_calendar(self, date=None):
        n = datetime.datetime.now(at.NY)
        return {"today": {"regularMarket": {
            "startTime": (n - datetime.timedelta(hours=1)).isoformat(),
            "endTime": (n + datetime.timedelta(hours=3)).isoformat()}}}

    # --- 상태 변경 ---
    def cancel_order(self, order_id):
        self.calls.append(("cancel", order_id))
        if "cancel" in self.fail:
            raise TossError(400, "already-filled", "이미 체결")
        self.open_rows = [r for r in self.open_rows if r.get("orderId") != order_id]
        return {"orderId": order_id}

    def modify_order(self, *a, **k):
        self.calls.append(("modify", a, k))
        return {}

    def create_order(self, symbol, side, order_type, quantity=None, price=None,
                     order_amount=None, time_in_force=None, client_order_id=None):
        self.calls.append(("create", symbol, side, quantity, order_amount, client_order_id))
        if "create" in self.fail:
            raise TossError(400, "invalid-request", "거절")
        return {"orderId": "B-" + (client_order_id or "x"), "clientOrderId": client_order_id}


def tmpdb():
    p = pathlib.Path(tempfile.mkdtemp()) / "t.db"
    at.DB_PATH = p
    at.initialize_db()
    return p


def orders_rows(db):
    with sqlite3.connect(db) as c:
        return c.execute("SELECT symbol, side, quantity, status FROM orders ORDER BY id").fetchall()


def deferred(fn, *a, **k):
    """place_order 가 '보류'로 끝났는지 — 제출도, 오래된 수량 사용도 아니어야 한다."""
    try:
        fn(*a, **k)
        return False, "제출됨"
    except at.Deferred as e:
        return True, str(e)
    except Exception as e:             # noqa: BLE001
        return False, type(e).__name__ + ": " + str(e)


# ══════════════════════════════════════════════════════════════════════════
print("\n[1] 만기·노출 축소 의도가 미체결 주문 때문에 사라지지 않는다")
# ══════════════════════════════════════════════════════════════════════════
# 1-a 만기 초과 + 기존 미체결 **매수**
a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("BUY", 3)]})
out, skip = vo(a, entries={"AAPL": 20})
check("1-a 만기+미체결 매수 → 매도 주문이 남는다", len(out) == 1 and out[0]["side"] == "sell",
      "out=%s skip=%s" % (out, [s.get("skipped") for s in skip]))
check("1-a 반대 방향 미체결은 먼저 취소한다", bool(out) and out[0].get("cancel_first") is True)

# 1-b 만기 초과 + 기존 미체결 **매도**(목표보다 적은 수량) → 의도가 남는다
a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("SELL", 4)]})
out, skip = vo(a, entries={"AAPL": 20})
check("1-b 만기+부분 매도 미체결 → 의도가 남는다", len(out) == 1,
      "skip=%s" % [s.get("skipped") for s in skip])

# 1-c 이미 목표를 충족하는 유효한 매도 주문 → 취소·재발행하지 않는다
a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("SELL", 10)]})
out, skip = vo(a, entries={"AAPL": 20})
check("1-c 목표를 덮는 매도 미체결 → 불필요한 재발행 없음",
      not out and any("이미" in s["skipped"] for s in skip),
      "out=%s skip=%s" % (out, [s.get("skipped") for s in skip]))

# 1-d 노출 상한 초과 + 기존 미체결 주문
_real_cap = at.exposure_cap
at.exposure_cap = lambda verbose=False: 0.5
a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("BUY", 1)]})
out, skip = vo(a, entries={"AAPL": 1})
check("1-d 노출 축소+미체결 → 축소 매도가 남는다",
      len(out) == 1 and "상한" in out[0]["reason"],
      "out=%s skip=%s" % (out, [s.get("skipped") for s in skip]))

# 1-e 노출 축소는 이미 예약된 매도(미체결 SELL)를 중복해서 또 깎지 않는다
a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("SELL", 10)]})
out, skip = vo(a, entries={"AAPL": 1})
check("1-e 이미 예약된 매도만큼은 다시 줄이지 않는다", not out, "out=%s" % out)
at.exposure_cap = _real_cap

# 1-f 손절도 같은 경로 (기존 동작 회귀)
a = acct(100, {"AAPL": hold(10, 100, 80)}, {"AAPL": [opn("BUY", 1)]})
out, _ = vo(a, {"AAPL": dec("AAPL", "hold", 80, 25, a)})
check("1-f 손절+미체결 → cancel_first 경로 유지",
      len(out) == 1 and out[0]["cancel_first"] is True)

# 1-g Claude 의 재량 매도는 예전처럼 미룬다 (경합을 만들 이유가 없다)
at.STOP_LOSS_PCT = 0
a = acct(100, {"AAPL": hold(10, 100, 105)}, {"AAPL": [opn("BUY", 1)]})
out, skip = vo(a, {"AAPL": dec("AAPL", "sell", 105, 25, a)},
               plan=[{"symbol": "AAPL", "side": "sell", "sell_pct": 50, "reason": "t"}])
check("1-g 재량 매도는 미체결이 있으면 미룬다", not out and "미체결" in skip[0]["skipped"])
at.STOP_LOSS_PCT = 15

# ══════════════════════════════════════════════════════════════════════════
print("\n[2] UNKNOWN 을 0 이나 정상값으로 치환하지 않는다")
# ══════════════════════════════════════════════════════════════════════════
o = {"symbol": "AAPL", "side": "sell", "quantity": 10.0, "cancel_first": True,
     "amount_usd": 1000.0, "price": 100.0, "reason": "손절", "whole": False}

t = FakeToss(opens=[{"orderId": "o1", "symbol": "AAPL", "side": "BUY", "quantity": 3}],
             fail={"sellable"})
ok, why = deferred(at.place_order, t, dict(o))
check("2-a 매도가능 수량 조회 실패 → 보류(Deferred)", ok, why)
check("2-a 보류면 create 호출 0회", not [c for c in t.calls if c[0] == "create"])

t = FakeToss(fail={"orders"})
ok, why = deferred(at.place_order, t, dict(o))
check("2-b 미체결 조회 실패 → 보류 (오래된 계획 수량으로 제출하지 않음)", ok, why)
check("2-b 보류면 create 호출 0회", not [c for c in t.calls if c[0] == "create"])

t = FakeToss(sellable=0)
ok, why = deferred(at.place_order, t, dict(o))
check("2-c 매도 가능 수량 0 → 제출 없음",
      ok and not [c for c in t.calls if c[0] == "create"], why)

# 2-d 취소 중 부분 체결 → 재조회된 잔량만 제출
t = FakeToss(opens=[{"orderId": "o1", "symbol": "AAPL", "side": "SELL", "quantity": 4}],
             sellable=6.0)
at.place_order(t, dict(o))
created = [c for c in t.calls if c[0] == "create"]
check("2-d 취소 후 재조회 수량(6)만 제출", len(created) == 1 and created[0][3] == "6",
      "calls=%s" % (t.calls,))

# 2-e 확인된 수량이 계획보다 많아도 계획을 넘지 않는다
t = FakeToss(sellable=99.0)
at.place_order(t, dict(o))
check("2-e 계획 수량(10) 초과 매도 없음",
      [c for c in t.calls if c[0] == "create"][0][3] == "10")

# ══════════════════════════════════════════════════════════════════════════
print("\n[3] 접수·체결·불명확 상태를 구분한다")
# ══════════════════════════════════════════════════════════════════════════
t = FakeToss(fail={"orders"})
found, known = at.find_order(t, "coid-1")
check("3-a 대사 조회 실패는 NOT_FOUND 가 아니라 UNKNOWN", found is None and known is False)

t = FakeToss(closed=[{"orderId": "B1", "clientOrderId": "coid-1", "status": "FILLED",
                      "execution": {"filledQuantity": "3"}}])
found, known = at.find_order(t, "coid-1")
check("3-b CLOSED 에서 찾으면 known=True", known is True and found is not None)
check("3-c FILLED 구분", at.order_state(found) == "FILLED", at.order_state(found))
check("3-c CANCELED 구분",
      at.order_state({"status": "CANCELED", "execution": {"filledQuantity": "0"}}) == "CANCELED")
check("3-c REJECTED 구분", at.order_state({"status": "REJECTED"}) == "REJECTED")
check("3-c 부분 체결 구분",
      at.order_state({"status": "OPEN", "quantity": "5",
                      "execution": {"filledQuantity": "2"}}) == "PARTIALLY_FILLED")

# ══════════════════════════════════════════════════════════════════════════
print("\n[4] place_all: 제출 실패가 성공처럼 취급되지 않는다 · DRY_RUN 은 호출 0회")
# ══════════════════════════════════════════════════════════════════════════
plan = [{"symbol": "AAPL", "side": "sell", "quantity": 10.0, "amount_usd": 1000.0,
         "price": 100.0, "reason": "만기", "cancel_first": False},
        {"symbol": "MSFT", "side": "sell", "quantity": 1.0, "amount_usd": 100.0,
         "price": 100.0, "reason": "만기", "cancel_first": False}]
db = tmpdb()
t = FakeToss(fail={"create"})
res = at.place_all(t, 1, plan, dry=False)
check("4-a place_all 이 종목별 결과를 돌려준다",
      isinstance(res, dict) and set(res) == {"AAPL", "MSFT"}, repr(res))
check("4-b 거절은 성공이 아니다",
      all(v != "ACKNOWLEDGED" for v in (res or {}).values()), repr(res))

db = tmpdb()
t = FakeToss()
res = at.place_all(t, 1, plan, dry=True)
check("4-c DRY_RUN 에서 create/cancel/modify 호출 0회", t.calls == [], repr(t.calls))
check("4-d DRY_RUN 기록은 남는다", len(orders_rows(db)) == 2)

# 4-e POST 타임아웃 뒤 실제로는 접수된 주문 → 재제출하지 않는다
db = tmpdb()
coid = at.intent_key(plan[0])
t = FakeToss(fail={"create"},
             opens=[{"orderId": "B9", "symbol": "AAPL", "clientOrderId": coid, "status": "OPEN"}])
res = at.place_all(t, 1, [plan[0]], dry=False)
check("4-e 대사로 접수 확인 → 중복 제출 없음",
      len([c for c in t.calls if c[0] == "create"]) == 1 and res.get("AAPL") == "ACKNOWLEDGED",
      "res=%s calls=%s" % (res, t.calls))

# 4-f 대사 자체가 실패하면 UNKNOWN — 재제출도, 성공 처리도 하지 않는다
db = tmpdb()
t = FakeToss(fail={"create", "orders"})
res = at.place_all(t, 1, [plan[0]], dry=False)
check("4-f 대사 실패는 UNKNOWN", res.get("AAPL") == "UNKNOWN", repr(res))
check("4-f UNKNOWN 은 DB 에 그대로 남는다",
      any("UNKNOWN" in (r[3] or "") for r in orders_rows(db)), repr(orders_rows(db)))

# ══════════════════════════════════════════════════════════════════════════
print("\n[5] 세션·미체결 상태 불명")
# ══════════════════════════════════════════════════════════════════════════
db = tmpdb()


class ClosedToss(FakeToss):
    def us_market_calendar(self, date=None):
        return {"today": {}}


t = ClosedToss()
res = at.place_all(t, 1, plan, dry=False)
check("5-a 제출 직전 세션 재검증 — 휴장이면 제출 0건",
      not [c for c in t.calls if c[0] == "create"], repr(t.calls))
check("5-b 보류는 기록으로 남는다",
      all("DEFERRED" in (r[3] or "") for r in orders_rows(db)), repr(orders_rows(db)))

a = acct(100, {"AAPL": hold(10, 100, 80)}, {})
a["open_unknown"] = True
out, skip = vo(a, {"AAPL": dec("AAPL", "hold", 80, 25, a)})
check("5-c 미체결 상태를 모르면 제출을 보류한다 (의도는 사유와 함께 남는다)",
      not out and skip and any("UNKNOWN" in s["skipped"] for s in skip),
      "out=%s skip=%s" % (out, [s.get("skipped") for s in skip]))

# ══════════════════════════════════════════════════════════════════════════
print("\n[6] 진입일 복원")
# ══════════════════════════════════════════════════════════════════════════
def closed_row(sym, side, qty, ts, oid="x"):
    return {"orderId": oid, "symbol": sym, "side": side, "orderedAt": ts,
            "status": "FILLED", "execution": {"filledQuantity": str(qty)}}


def hist(rows):
    return FakeToss(closed=rows)


# 6-a 부분 매도는 최초 진입일을 유지한다
t = hist([closed_row("AAPL", "BUY", 10, "2026-01-02T10:00:00+00:00", "1"),
          closed_row("AAPL", "SELL", 4, "2026-01-08T10:00:00+00:00", "2")])
e = at.position_entry_map(t, {"AAPL": hold(6, 100, 100)})
check("6-a 부분 매도 후 진입일 유지", str(e.get("AAPL")).startswith("2026-01-02"), repr(e))

# 6-b 완전 청산 후 재매수는 새 진입일
t = hist([closed_row("AAPL", "BUY", 10, "2026-01-02T10:00:00+00:00", "1"),
          closed_row("AAPL", "SELL", 10, "2026-01-08T10:00:00+00:00", "2"),
          closed_row("AAPL", "BUY", 5, "2026-02-02T10:00:00+00:00", "3")])
e = at.position_entry_map(t, {"AAPL": hold(5, 100, 100)})
check("6-b 재매수는 새 진입일", str(e.get("AAPL")).startswith("2026-02-02"), repr(e))

# 6-c 추가 매수는 최초 진입 기준 (시계 리셋 없음)
t = hist([closed_row("AAPL", "BUY", 5, "2026-01-02T10:00:00+00:00", "1"),
          closed_row("AAPL", "BUY", 5, "2026-01-20T10:00:00+00:00", "2")])
e = at.position_entry_map(t, {"AAPL": hold(10, 100, 100)})
check("6-c 추가 매수는 최초 진입 유지", str(e.get("AAPL")).startswith("2026-01-02"), repr(e))

# 6-d 이력이 일부만 있으면(복원 수량 ≠ holdings) 탐지한다
t = hist([closed_row("AAPL", "BUY", 3, "2026-01-02T10:00:00+00:00", "1")])
e = at.position_entry_map(t, {"AAPL": hold(10, 100, 100), "MSFT": hold(1, 10, 10)})
check("6-d 불완전 이력 탐지 (수량 불일치)", "AAPL" in at.ENTRY_UNVERIFIED,
      repr(at.ENTRY_UNVERIFIED))
check("6-d 이력에 없는 보유는 ENTRY_UNKNOWN",
      "MSFT" not in e and "MSFT" in at.ENTRY_UNVERIFIED, repr(at.ENTRY_UNVERIFIED))

# 6-e 진입일이 검증되지 않은 종목은 만기 청산하지 않는다
a = acct(100, {"AAPL": hold(10, 100, 105)})
at.ENTRY_UNVERIFIED = set()
out, _ = vo(a, entries={"AAPL": 20})
at.ENTRY_UNVERIFIED = {"AAPL"}
out2, skip2 = vo(a, entries={"AAPL": 20})
at.ENTRY_UNVERIFIED = set()
check("6-e 진입일 미검증 종목은 만기 청산하지 않는다", bool(out) and not out2,
      "out=%s out2=%s" % (out, out2))

# 6-f 로컬 대체 경로가 submitted 매수를 진입으로 쓰지 않는다
db = tmpdb()
with sqlite3.connect(db) as c:
    c.execute("INSERT INTO orders (run_id,timestamp,symbol,side,status) VALUES (1,?,?,?,?)",
              ("2026-01-02T00:00:00+09:00", "AAPL", "buy", "submitted"))
e = at.position_entry_map(None, {"AAPL": hold(10, 100, 100)})
check("6-f submitted 주문은 체결 증거가 아니다", "AAPL" not in e, repr(e))

# ══════════════════════════════════════════════════════════════════════════
print("\n[7] 정상 제한과 매수 경로 회귀")
# ══════════════════════════════════════════════════════════════════════════
at.ENTRY_UNVERIFIED = set()
a = acct(100, {"AAPL": hold(0.62, 20, 20)})
out, skip = vo(a, {"AAPL": dec("AAPL", "sell", 20, 25, a)},
               plan=[{"symbol": "AAPL", "side": "sell", "sell_pct": 100, "reason": "t"}],
               session=OFF)
check("7-a 장외 소수점 잔량은 내지 않고 사유를 남긴다",
      not out and skip and "정규장" in skip[0]["skipped"],
      "out=%s skip=%s" % (out, [s.get("skipped") for s in skip]))
a = acct(1000)
out, _ = vo(a, {"AAPL": dec("AAPL", "buy", 200, 25, a)},
            plan=[{"symbol": "AAPL", "side": "buy", "reason": "t"}])
check("7-b 매수 경로 회귀 (한도 30% × 배수 1.0)", len(out) == 1 and out[0]["amount_usd"] == 300,
      repr(out))

print("\n" + ("실패 %d건: %s" % (len(FAILS), FAILS) if FAILS else "전부 통과"))
sys.exit(1 if FAILS else 0)
