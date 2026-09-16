"""라이브 무거래 퍼널 — 운영 DB 사본 + 실제 validate_orders 로 '어디서 막혔나'를 센다.

읽기 전용. research.isolation.guard() 로 운영 DB·로그·브로커를 전부 막고,
DB 는 research/out/ 로 복사한 사본만 쓴다. 네트워크·LLM 호출 없음.

  python research/live_funnel.py            # 사후 집계 + 차단 단계 재현
"""
import csv
import datetime
import json
import pathlib
import shutil
import sqlite3
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"
sys.path.insert(0, str(ROOT))

# --- 격리 전에 운영 DB 를 사본으로 뜬다 (읽기만) ---
OUT.mkdir(parents=True, exist_ok=True)
SNAP = OUT / "live_snapshot.db"
shutil.copyfile(ROOT / "trading_decisions.db", SNAP)

from research.isolation import guard        # noqa: E402
guard()

import autotrade as A                       # noqa: E402
A.DB_PATH = SNAP                            # 운영 DB 를 절대 안 연다


# ---------- A) 사후 집계: run 별 퍼널 ----------
def history():
    c = sqlite3.connect(f"file:{SNAP}?mode=ro", uri=True)
    runs = {r[0]: r for r in c.execute(
        "SELECT id, timestamp, status, total_value, cash FROM runs")}
    dec, orders = {}, {}
    for rid, d, n in c.execute(
            "SELECT run_id, decision, COUNT(*) FROM trading_decisions GROUP BY run_id, decision"):
        dec.setdefault(rid, {})[d] = n
    for rid, st, n in c.execute(
            "SELECT run_id, status, COUNT(*) FROM orders GROUP BY run_id, status"):
        orders.setdefault(rid, {})[st] = n
    rows = []
    for rid in sorted(runs):
        _, ts, status, tv, cash = runs[rid]
        d, o = dec.get(rid, {}), orders.get(rid, {})
        sent = sum(n for s, n in o.items() if not s.startswith("skipped"))
        skip = sum(n for s, n in o.items() if s.startswith("skipped"))
        rows.append({"run": rid, "timestamp": ts, "status": status,
                     "total_value": tv, "cash": cash,
                     "reserve_usd": None if tv is None else round(tv * A.CASH_RESERVE_PCT / 100, 2),
                     "cash_left": None if tv is None else round(cash - tv * A.CASH_RESERVE_PCT / 100, 2),
                     "buy_decisions": d.get("buy", 0), "sell_decisions": d.get("sell", 0),
                     "hold_decisions": d.get("hold", 0),
                     "orders_sent": sent, "orders_skipped": skip,
                     "skip_reasons": "; ".join(sorted(s[9:] for s in o if s.startswith("skipped")))})
    p = OUT / "live_funnel.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows, p


# ---------- B) 차단 단계 재현: 마지막 완료 run 의 계좌로 validate_orders ----------
def snapshot_account(run_id):
    """trading_decisions 의 보유 수량·현재가로 계좌를 복원한다 (평가액 = 수량 × 현재가)."""
    c = sqlite3.connect(f"file:{SNAP}?mode=ro", uri=True)
    tv, cash = c.execute("SELECT total_value, cash FROM runs WHERE id=?", (run_id,)).fetchone()
    holdings, decisions = {}, {}
    for sym, d, pct, bal, avg, px in c.execute(
            "SELECT symbol, decision, percentage, stock_balance, avg_buy_price, current_price "
            "FROM trading_decisions WHERE run_id=?", (run_id,)):
        if bal and bal > 0:
            holdings[sym] = {"name": sym, "quantity": bal, "avg_price": avg or px,
                             "last_price": px, "market_value": bal * px,
                             "pnl_pct": round((px / avg - 1) * 100, 2) if avg else 0.0}
        decisions[sym] = {"symbol": sym, "decision": d, "percentage": pct,
                          "reason": "", "status": {
                              "current_price": px, "avg_buy_price": avg,
                              "stock_balance": bal or 0.0, "pnl_pct":
                                  round((px / avg - 1) * 100, 2) if avg and bal else None,
                              "ret_20d_pct": None, "momentum_tier": "", "size_factor": 1.0}}
    acct = {"cash": cash, "holdings": holdings, "open_orders": [],
            "total_value": round(cash + sum(h["market_value"] for h in holdings.values()), 2)}
    return acct, decisions


def regular_session():
    now = datetime.datetime.now(A.KST)
    h = datetime.timedelta(hours=1)
    return (now - 2 * h, now - h, now + 2 * h, now + 3 * h)   # 정규장 한복판 = 소수점 허용


def replay(run_id, cash_override=None, label=""):
    acct, decisions = snapshot_account(run_id)
    if cash_override is not None:
        acct = {**acct, "cash": cash_override,
                "total_value": round(cash_override
                                     + sum(h["market_value"] for h in acct["holdings"].values()), 2)}
    # 미보유 후보 전부에 매수 주문을 넣어 본다 — '신호가 있었다면 나갔을까' 를 본다
    plan = {"orders": [{"symbol": s, "side": "buy", "reason": "replay"}
                       for s in decisions if s not in acct["holdings"]], "summary": ""}
    entries = {s: datetime.datetime.now(A.NY).isoformat() for s in acct["holdings"]}
    orders, skipped = A.validate_orders(plan, decisions, acct, regular_session(), entries)
    return {"label": label, "run": run_id, "cash": acct["cash"],
            "total_value": acct["total_value"],
            "reserve_usd": round(acct["total_value"] * A.CASH_RESERVE_PCT / 100, 2),
            "cash_left": round(acct["cash"] - acct["total_value"] * A.CASH_RESERVE_PCT / 100, 2),
            "buy_candidates": len(plan["orders"]), "orders_out": len(orders),
            "skipped": [(o["symbol"], o["skipped"]) for o in skipped]}


def main():
    rows, path = history()
    print(f"[A] run 별 퍼널 → {path}")
    print(f"{'run':>4} {'시각':<20} {'총자산':>8} {'현금':>7} {'여유현금':>9} "
          f"{'buy':>4} {'sell':>4} {'hold':>4} {'주문':>4} {'제외':>4}")
    for r in rows[-16:]:
        print(f"{r['run']:>4} {r['timestamp'][:19]:<20} "
              f"{r['total_value'] or 0:>8.2f} {r['cash'] or 0:>7.2f} "
              f"{r['cash_left'] if r['cash_left'] is not None else 0:>9.2f} "
              f"{r['buy_decisions']:>4} {r['sell_decisions']:>4} {r['hold_decisions']:>4} "
              f"{r['orders_sent']:>4} {r['orders_skipped']:>4}")
    sent = sum(r["orders_sent"] for r in rows)
    print(f"\n전체 {len(rows)} run · 실제 접수 주문 {sent}건 · "
          f"제외 {sum(r['orders_skipped'] for r in rows)}건")
    last_sent = max((r["run"] for r in rows if r["orders_sent"]), default=None)
    print(f"마지막으로 주문이 실제 나간 run: {last_sent} "
          f"({next(r['timestamp'] for r in rows if r['run'] == last_sent)})")

    print("\n[B] 차단 단계 재현 — 실제 validate_orders (LLM·브로커 없음)")
    last_done = max(r["run"] for r in rows if r["status"] == "done")
    cases = [replay(last_done, None, "현행 계좌 그대로"),
             replay(last_done, 60.0, "현금 $60 (현금유지 한도 直前)"),
             replay(last_done, 120.0, "현금 $120 (한도 초과분 확보)")]
    for c in cases:
        print(f"\n  · {c['label']}: 현금 ${c['cash']:.2f} / 총자산 ${c['total_value']:.2f} "
              f"→ 유지선 ${c['reserve_usd']:.2f}, 가용 ${c['cash_left']:.2f}")
        print(f"    매수 후보 {c['buy_candidates']}종목 → 실제 주문 {c['orders_out']}건")
        for sym, why in c["skipped"][:4]:
            print(f"      제외 {sym}: {why}")
    (OUT / "live_funnel_replay.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  → {OUT / 'live_funnel_replay.json'}")


if __name__ == "__main__":
    main()
