"""SOXL 규칙(개별종목 적용판, research/soxl_rules_universe.py)이 매수보유보다 Sharpe 를 개선한
35종목 중 상관관계 낮은 5~10종목을 골라 동일가중 멀티슬롯으로 묶었을 때 성과 — 탐색용.

    python research/soxl_rules_multislot.py   # → research/out/soxl_rules_multislot.csv

구성: 슬롯마다 **독립된 자본**으로 같은 규칙(M3 40%/10%밴드 + R1 70% + R2p -30% + R3 63일,
운영 SOXL 값과 동일)을 그 종목에 적용하고, 슬롯 수익률을 동일가중 평균해 포트폴리오를 만든다
(슬롯끼리 현금을 다투지 않는다 — 실제로도 슬롯별 배분 자본이 다를 것이므로 단순화).
종목 선택: 후보군(규칙이 Sharpe 개선한 35종목) 안에서 Sharpe 높은 순으로 그리디하게 추가하되,
이미 고른 종목들과의 평균 상관계수가 임계값(0.55) 넘으면 건너뛴다.
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import soxl_rules as S        # noqa: E402

CFG = {"module": "M3", "risk": "price", "target": 0.40}   # research/soxl_rules_universe.py 와 동일
CORR_MAX = 0.55
N_SLOTS = (5, 10)


def load():
    panel = pickle.load(open(ROOT / "research" / "data" / "pit" / "ohlcv_panel.pkl", "rb"))
    close_all = pickle.load(open(ROOT / "research" / "data" / "soxl_close.pkl", "rb"))
    rf = (close_all["^IRX"].ffill() / 100 / 252)
    return panel["Close"], rf


def sim_symbol(c, rf):
    c = c.dropna()
    px = pd.DataFrame({"open": np.nan, "high": np.nan, "low": np.nan, "close": c})
    px["rf"] = rf.reindex(px.index).ffill().fillna(0)
    eq, w, _ = S.sim(px, CFG)
    return eq.pct_change().fillna(0), c / c.iloc[0]


def greedy_select(cands, ret, k):
    """cands: Sharpe 내림차순 종목 리스트. 상관 임계값 안에서 그리디로 k개 채운다."""
    picked = [cands[0]]
    for s in cands[1:]:
        if len(picked) >= k:
            break
        avg_corr = np.mean([ret[s].corr(ret[p]) for p in picked])
        if avg_corr <= CORR_MAX:
            picked.append(s)
    return picked


def portfolio_stat(rets, rf, ref_index):
    r = pd.concat(rets, axis=1).fillna(0)          # 미상장 구간은 0(현금 취급)
    port_r = r.mean(axis=1)
    eq = (1 + port_r).cumprod()
    return S.stat(eq, rf.reindex(eq.index).ffill().fillna(0)), eq


def main():
    close, rf = load()
    cand_tb = pd.read_csv(ROOT / "research" / "out" / "soxl_rules_universe.csv")
    cands = cand_tb[cand_tb["규칙이 Sharpe 개선"]].sort_values("Sharpe", ascending=False)["종목"].tolist()

    rule_ret, bh_ret = {}, {}
    for s in cands:
        rule_ret[s], bh_ret[s] = sim_symbol(close[s], rf)

    corr = pd.DataFrame({s: rule_ret[s] for s in cands}).corr()
    print("## 규칙적용 수익률 상관행렬 (일부)\n")
    print(corr.round(2).iloc[:10, :10].to_string())

    rows = []
    for k in range(N_SLOTS[0], N_SLOTS[1] + 1):
        picked = greedy_select(cands, rule_ret, k)
        if len(picked) < k:
            continue     # 임계값 안에서 k개를 못 채우면 스킵
        st, eq = portfolio_stat([rule_ret[s] for s in picked], rf, None)
        st_bh, _ = portfolio_stat([bh_ret[s].pct_change().fillna(0) for s in picked], rf, None)
        avg_corr = np.mean([corr.loc[a, b] for i, a in enumerate(picked) for b in picked[i + 1:]])
        rows.append({"슬롯수": len(picked), "종목": ",".join(picked), "평균상관": round(avg_corr, 2),
                     "포트 CAGR%": st["CAGR%"], "포트 Sharpe": st["Sharpe"], "포트 MDD%": st["MDD%"],
                     "매수보유 CAGR%": st_bh["CAGR%"], "매수보유 Sharpe": st_bh["Sharpe"], "매수보유 MDD%": st_bh["MDD%"]})

    tb = pd.DataFrame(rows)
    tb.to_csv(ROOT / "research" / "out" / "soxl_rules_multislot.csv", index=False, encoding="utf-8-sig")
    pd.options.display.width = 200
    print("\n## 멀티슬롯 결과 (슬롯 수별)\n")
    print(tb.drop(columns="종목").to_string(index=False))
    print("\n## 구성 종목\n")
    for _, r in tb.iterrows():
        print(f"{r['슬롯수']}종목: {r['종목']}")


if __name__ == "__main__":
    main()
