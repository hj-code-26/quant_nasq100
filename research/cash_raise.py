"""현금을 만들려면 무엇을 얼마나 팔아야 하나 — 산수와 시스템 규칙만. 읽기 전용·오프라인.

    python research/cash_raise.py

투자 판단을 하지 않는다. 답하는 것은 셋뿐이다.
  ① 어떤 보유분을 팔면 매수 교착(가용현금 < 최소주문)이 실제로 풀리는가 — 실제 validate_orders 로 확인
  ② 교착을 풀려면 최소 얼마를 팔아야 하는가
  ③ 시스템 자신의 규칙(선별 정렬)은 지금 어느 종목을 상위로 보는가 — 평상시/하락 국면 각각

③ 은 yfinance 일봉으로 screen() 과 같은 식으로 계산한다 (토스는 격리로 차단).
"""
import pathlib
import shutil
import sqlite3
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

OUT.mkdir(parents=True, exist_ok=True)
SNAP = OUT / "cash_raise_snapshot.db"
shutil.copyfile(ROOT / "trading_decisions.db", SNAP)

from research.isolation import guard        # noqa: E402

guard()

import autotrade as A                       # noqa: E402

A.DB_PATH = SNAP


def account_from_db():
    c = sqlite3.connect(f"file:{SNAP}?mode=ro", uri=True)
    rid = c.execute("SELECT MAX(run_id) FROM trading_decisions "
                    "WHERE stock_balance > 0").fetchone()[0]
    cash = c.execute("SELECT cash FROM runs WHERE id=?", (rid,)).fetchone()[0]
    h = {}
    for sym, bal, avg, px in c.execute(
            "SELECT symbol, stock_balance, avg_buy_price, current_price FROM trading_decisions "
            "WHERE run_id=? AND stock_balance>0", (rid,)):
        h[sym] = {"name": sym, "quantity": bal, "avg_price": avg, "last_price": px,
                  "market_value": bal * px,
                  "pnl_pct": round((px / avg - 1) * 100, 2) if avg else 0.0}
    return rid, {"cash": cash, "holdings": h, "open_orders": [],
                 "total_value": round(cash + sum(x["market_value"] for x in h.values()), 2)}


def can_buy(acct, session):
    """실제 validate_orders 로 '매수가 한 건이라도 나가는가' 를 본다."""
    d = {"XXXX": {"symbol": "XXXX", "decision": "buy", "percentage": 80, "reason": "probe",
                  "status": {"current_price": 10.0, "stock_balance": 0.0, "avg_buy_price": 0.0,
                             "pnl_pct": None, "ret_20d_pct": 25.0,
                             "momentum_tier": "강", "size_factor": 1.0}}}
    o, sk = A.validate_orders({"orders": [{"symbol": "XXXX", "side": "buy", "reason": "probe"}],
                               "summary": ""}, d, acct, session)
    buys = [x for x in o if x["side"] == "buy"]
    return (buys[0]["amount_usd"] if buys else 0.0,
            "" if buys else next((x["skipped"] for x in sk), ""))


def ranks():
    """screen() 과 같은 식으로 오늘의 순위. 평상시=20일 수익률 내림차순, 하락=atr_pct 오름차순."""
    import yfinance as yf
    from nasdaq100 import TICKERS
    px = yf.download(list(TICKERS), period="6mo", progress=False, auto_adjust=False,
                     group_by="ticker")
    rows = []
    for s in TICKERS:
        try:
            d = px[s].dropna()
        except KeyError:
            continue
        if len(d) < 61:
            continue
        c = d["Close"]
        rows.append({"symbol": s, "ret_20d_pct": (c.iloc[-1] / c.iloc[-21] - 1) * 100,
                     "atr_pct": (d["High"] - d["Low"]).tail(14).mean() / c.iloc[-1] * 100})
    df = pd.DataFrame(rows)
    df["모멘텀순위"] = df["ret_20d_pct"].rank(ascending=False).astype(int)
    df["저변동성순위"] = df["atr_pct"].rank(ascending=True).astype(int)
    return df.set_index("symbol"), len(df)


def main():
    rid, base = account_from_db()
    now = A.datetime.datetime.now(A.KST)
    hr = A.datetime.timedelta(hours=1)
    session = (now - 2 * hr, now - hr, now + 2 * hr, now + 3 * hr)   # 정규장 가정
    total = base["total_value"]
    reserve = total * A.CASH_RESERVE_PCT / 100
    need = reserve + A.MIN_ORDER_USD - base["cash"]
    print(f"기준 run {rid} · 총자산 ${total:.2f} · 현금 ${base['cash']:.2f}")
    print(f"현금유지선 ${reserve:.2f} (총자산 {A.CASH_RESERVE_PCT:.0f}%) · 최소주문 ${A.MIN_ORDER_USD}")
    print(f"→ 매수가 한 건이라도 나가려면 현금이 ${reserve + A.MIN_ORDER_USD:.2f} 이상이어야 한다."
          f"  **최소 ${need:.2f} 어치를 팔아야 한다.**\n")

    print("① 한 종목만 전량 매도했을 때 (실제 validate_orders 로 확인)")
    print(f"{'매도':<7}{'평가액':>9}{'손익%':>8}{'매도후현금':>11}{'가용현금':>10}  결과")
    out = []
    for sym, h in sorted(base["holdings"].items(), key=lambda kv: -kv[1]["market_value"]):
        mv = h["market_value"]
        acct = {"cash": round(base["cash"] + mv, 2),
                "holdings": {k: v for k, v in base["holdings"].items() if k != sym},
                "open_orders": [], "total_value": total}
        amt, why = can_buy(acct, session)
        left = acct["cash"] - total * A.CASH_RESERVE_PCT / 100
        ok = "매수 가능 ${:.2f}".format(amt) if amt else "여전히 막힘"
        print(f"{sym:<7}{mv:>9.2f}{h['pnl_pct']:>8.2f}{acct['cash']:>11.2f}{left:>10.2f}  {ok}")
        out.append({"매도": sym, "평가액": round(mv, 2), "손익%": h["pnl_pct"],
                    "가용현금": round(left, 2), "매수가능액": round(amt, 2), "차단사유": why})
    pd.DataFrame(out).to_csv(OUT / "cash_raise.csv", index=False, encoding="utf-8-sig")

    print("\n② 시스템 자신의 선별 규칙은 지금 이 종목들을 어떻게 보는가")
    try:
        r, n = ranks()
    except Exception as e:                   # noqa: BLE001
        print(f"   순위 계산 실패 ({e}) — 네트워크 없이 돌렸다면 정상이다.")
        return
    tb = r.loc[[s for s in base["holdings"] if s in r.index]].copy()
    tb["평가액"] = [round(base["holdings"][s]["market_value"], 2) for s in tb.index]
    tb = tb[["평가액", "ret_20d_pct", "모멘텀순위", "atr_pct", "저변동성순위"]].round(2)
    print(f"   (유니버스 {n}종목 기준. 평상시 선별=모멘텀순위, 하락 국면 선별=저변동성순위)")
    print(tb.sort_values("저변동성순위").to_string())
    print(f"\n→ {OUT / 'cash_raise.csv'}")


if __name__ == "__main__":
    main()
