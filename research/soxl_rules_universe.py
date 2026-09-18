"""SOXL 규칙(M3 리밸런싱 40%/밴드10%p + R1(현금30%)+R2p(-30% 관망)+R3(63일 축소))을
SOXL 대신 나스닥100 PIT 유니버스 종목 각각에 개별 적용 — 종목별 단일자산+현금 백테스트.

지금 못 사는 SOXL 대신, 이미 확인 중인 나스닥100 종목에 같은 기계적 규칙을 하나씩
꽂아보면 어떤지 보는 탐색용 스크립트다 (사전등록 없음 — 채택 판단용 아니라 탐색용).

    python research/soxl_rules_universe.py   # → research/out/soxl_rules_universe.csv

규칙은 research/soxl_rules.py 의 M3·risk=price 시뮬레이터를 그대로 재사용한다
(운영 SOXL_TARGET=0.40·SOXL_BAND=0.10·R1 70%·R2p -30%·R3 63일과 동일 파라미터).
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import soxl_rules as S        # noqa: E402  (guard() 는 여기서 켜진다)

MIN_DAYS = 500                              # 최소 이만큼 종가 있어야 대상 (약 2년)
CFG = {"module": "M3", "risk": "price", "target": 0.40}   # 운영값과 동일


def load():
    panel = pickle.load(open(ROOT / "research" / "data" / "pit" / "ohlcv_panel.pkl", "rb"))
    close = pickle.load(open(ROOT / "research" / "data" / "soxl_close.pkl", "rb"))
    rf = (close["^IRX"].ffill() / 100 / 252)
    return panel["Close"], rf


def run_symbol(c, rf):
    c = c.dropna()
    if len(c) < MIN_DAYS:
        return None
    px = pd.DataFrame({"open": np.nan, "high": np.nan, "low": np.nan, "close": c})
    px["rf"] = rf.reindex(px.index).ffill().fillna(0)
    eq, w, _ = S.sim(px, CFG)
    bh = c / c.iloc[0]
    st, bhst = S.stat(eq, px["rf"]), S.stat(bh, px["rf"])
    return {"종목": c.name, "일수": len(c), "CAGR%": st["CAGR%"], "Sharpe": st["Sharpe"], "MDD%": st["MDD%"],
            "평균비중%": round(w.mean() * 100, 1),
            "매수보유 CAGR%": bhst["CAGR%"], "매수보유 Sharpe": bhst["Sharpe"], "매수보유 MDD%": bhst["MDD%"],
            "규칙이 Sharpe 개선": st["Sharpe"] > bhst["Sharpe"]}


def main():
    close, rf = load()
    rows = [r for s in close.columns if (r := run_symbol(close[s], rf)) is not None]
    tb = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    tb.to_csv(ROOT / "research" / "out" / "soxl_rules_universe.csv", index=False, encoding="utf-8-sig")

    pd.options.display.width = 200
    summary = pd.DataFrame({
        "규칙 적용": tb[["CAGR%", "Sharpe", "MDD%", "평균비중%"]].median(),
        "매수보유": tb[["매수보유 CAGR%", "매수보유 Sharpe", "매수보유 MDD%"]].set_axis(
            ["CAGR%", "Sharpe", "MDD%"], axis=1).median().reindex(["CAGR%", "Sharpe", "MDD%", "평균비중%"]),
    })
    print(f"대상 {len(tb)}종목 (최소 {MIN_DAYS}일)")
    print("\n## 중앙값 — 규칙 적용 vs 매수보유\n")
    print(summary.to_string())
    print(f"\n규칙이 매수보유보다 Sharpe 높은 종목: {tb['규칙이 Sharpe 개선'].sum()}/{len(tb)} "
          f"({tb['규칙이 Sharpe 개선'].mean():.0%})")
    print("\n## 상위 10 (Sharpe)\n")
    print(tb.head(10).to_string(index=False))
    print("\n## 하위 10 (Sharpe)\n")
    print(tb.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
