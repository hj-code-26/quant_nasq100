"""SOXL 규칙(M3 40%/10%밴드 + R1 70% + R2p -30% 관망 + R3 63일 축소)을 SOXL 과 구조가
같은 다른 3배 레버리지 ETF 에 개별 적용 — 사용자가 지목한 후보 (2026-09-17):
    TQQQ(나스닥100 3배) · TECL(XLK 기술주 3배) · LABU(S&P 바이오텍 3배) ·
    FNGU(FANG+ 10종목 3배 ETN) · DPST(지역은행 3배)

    python research/soxl_rules_etfs.py   # → research/out/soxl_rules_etfs.csv (Yahoo 신규 다운로드)

규칙 파라미터는 research/soxl_rules.py 의 M3·risk=price 를 그대로 재사용 — 운영값과 동일.
탐색용 — 사전등록·판정 기준 없음.
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research.isolation import guard        # noqa: E402
guard()

from research import soxl_rules as S         # noqa: E402

TICKERS = ["TQQQ", "TECL", "LABU", "FNGU", "DPST", "SOXL"]   # SOXL = 기준선으로 같이 비교
CFG = {"module": "M3", "risk": "price", "target": 0.40}      # 운영값과 동일
DATA = ROOT / "research" / "data"


def fetch(sym):
    cache = DATA / f"{sym}_daily.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    import yfinance as yf
    df = yf.download(sym, start="2010-01-01", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    pickle.dump(df, open(cache, "wb"))
    return df


def main():
    rf = (pickle.load(open(DATA / "soxl_close.pkl", "rb"))["^IRX"].ffill() / 100 / 252)
    rows = []
    for sym in TICKERS:
        px = fetch(sym)
        px["rf"] = rf.reindex(px.index).ffill().fillna(0)
        eq, w, _ = S.sim(px, CFG)
        bh = px["close"] / px["close"].iloc[0]
        st, bhst = S.stat(eq, px["rf"]), S.stat(bh, px["rf"])
        rows.append({"종목": sym, "상장일": str(px.index[0].date()), "일수": len(px),
                     "규칙 CAGR%": st["CAGR%"], "규칙 Sharpe": st["Sharpe"], "규칙 MDD%": st["MDD%"],
                     "평균비중%": round(w.mean() * 100, 1),
                     "매수보유 CAGR%": bhst["CAGR%"], "매수보유 Sharpe": bhst["Sharpe"], "매수보유 MDD%": bhst["MDD%"],
                     "규칙이 Sharpe 개선": st["Sharpe"] > bhst["Sharpe"]})
    tb = pd.DataFrame(rows).sort_values("규칙 Sharpe", ascending=False)
    tb.to_csv(ROOT / "research" / "out" / "soxl_rules_etfs.csv", index=False, encoding="utf-8-sig")
    pd.options.display.width = 200
    print(tb.to_string(index=False))


if __name__ == "__main__":
    main()
