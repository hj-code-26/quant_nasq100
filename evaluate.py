"""판단 성적표 — 기록된 판단을 이후 실제 가격과 대조한다.

  python evaluate.py            # 전체
  python evaluate.py --days 5   # 최근 5일 판단만

지표
  · 판단별 이후 수익률: buy 는 +면 적중, sell 은 -면 적중, hold 는 |수익률| < 1% 면 적중(관망이 옳았음)
  · 모델별·판단별 적중률과 평균 수익률
  · 번복률: 같은 종목에서 직전 판단과 다른 판단을 낸 비율 (낮을수록 안정적)
표본이 며칠치 뿐이면 숫자는 잡음이다. 최소 2~4주 쌓인 뒤 봐라.
"""
import collections
import datetime
import sqlite3
import sys

from autotrade import DB_PATH, KST
from toss import shared_client

HOLD_BAND = 1.0   # hold 적중 판정 폭 (%)


def main():
    days = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 3650
    since = (datetime.datetime.now(KST) - datetime.timedelta(days=days)).isoformat()
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute("""SELECT d.run_id, coalesce(r.model, '?'), d.timestamp, d.symbol,
                                   d.decision, d.percentage, d.current_price
                            FROM trading_decisions d LEFT JOIN runs r ON r.id = d.run_id
                            WHERE d.run_id IS NOT NULL AND d.timestamp >= ? AND d.current_price > 0
                            ORDER BY d.symbol, d.id""", (since,)).fetchall()
    if not rows:
        print("판단 기록 없음")
        return
    syms = sorted({r[3] for r in rows})
    px = {p["symbol"]: float(p["lastPrice"]) for p in shared_client().prices(*syms)}

    by_model = collections.defaultdict(lambda: collections.defaultdict(list))
    flips, prev = collections.Counter(), {}
    print(f"{'run':>4} {'모델':22} {'시각':16} {'종목':6} {'판단':5} {'당시가':>9} {'현재가':>9} {'수익률':>7} 적중")
    for run_id, model, ts, sym, dec, pct, p0 in rows:
        p1 = px.get(sym)
        if not p1:
            continue
        ret = (p1 / p0 - 1) * 100
        hit = (ret > 0) if dec == "buy" else (ret < 0) if dec == "sell" else abs(ret) < HOLD_BAND
        m = model.split("/")[-1]
        by_model[m][dec].append((ret, hit))
        if sym in prev and prev[sym] != dec:
            flips[m] += 1
        flips[m + "_n"] += 1
        prev[sym] = dec
        print(f"{run_id:>4} {m:22} {ts[5:16]:16} {sym:6} {dec:5} {p0:9.2f} {p1:9.2f} {ret:+6.2f}% {'○' if hit else '×'}")

    print("\n=== 모델별 성적 (현재가 기준, 보유 기간 무시) ===")
    for m, decs in by_model.items():
        parts = []
        for dec in ("buy", "sell", "hold"):
            xs = decs.get(dec) or []
            if xs:
                acc = sum(h for _, h in xs) / len(xs) * 100
                avg = sum(r for r, _ in xs) / len(xs)
                parts.append(f"{dec} {len(xs)}건 적중 {acc:.0f}% 평균 {avg:+.2f}%")
        n, f = flips[m + "_n"], flips[m]
        print(f"{m:22} " + " | ".join(parts) + f" | 번복 {f}/{n} ({f / max(n, 1) * 100:.0f}%)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
