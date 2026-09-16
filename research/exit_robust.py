"""G-2 — exit_study 후보의 사후 강건성. **선택이 아니라 확인**이다 (시도 수는 DSR N 에 더한다).

    python research/exit_robust.py

  ① 이웃 파라미터: X14 연장 기준 상위 10/30/40 — 봉우리인지 고원인지
  ② 최소 변경안: 순위 126일(E2)·12-1(E3) + 만기연장(X14) — 운영에 실제로 옮길 수 있는 조합
  ③ 연도 하나 빼기: ΔSharpe(vs L0) 부호가 한 해(특히 2020)에 매달려 있는가
  ④ QQQ 매수보유 대비: 후보가 벤치마크를 이기는가
"""
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from research import exit_study as ES      # noqa: E402  (guard() 는 거기서 켜진다)
from research import corr_limits as C      # noqa: E402

EXTRA = {
    "X14b 만기 때 상위10이면 연장": ES.v(extend=10),
    "X14c 만기 때 상위30이면 연장": ES.v(extend=30),
    "X14d 만기 때 상위40이면 연장": ES.v(extend=40),
    "P1 E2+X14 (126일·연장20)": ES.v(rank="mom126", extend=20),
    "P2 E3+X14 (12-1·연장20)": ES.v(rank="mom12_1", extend=20),
}


def loyo(r, b):
    """연도 하나씩 빼고 Sharpe 차 (점추정)."""
    x, y = b.align(r, join="inner")
    out = {}
    for yr in sorted(set(x.index.year)):
        k = x.index.year != yr
        out[yr] = C.stat(y[k])["sharpe"] - C.stat(x[k])["sharpe"]
    return pd.Series(out)


def main():
    grid = dict(ES.GRID)
    combo_cfg = ES.v(**{k: val for p in ("R4 부가규칙 전부 제거(단순 슬롯)", "X14 만기 때 상위20이면 연장",
                                         "M4 대기현금 QQQ 보유", "E2 순위=126일 모멘텀")
                        for k, val in ES.GRID[p].items() if ES.L0.get(k, "∅") != val})
    grid["C 조합: R4 + X14 + M4 + E2"] = combo_cfg
    grid.update(EXTRA)
    n_trials = len(grid) - 1
    with mp.Pool(min(11, mp.cpu_count() - 1), initializer=ES._init) as pool:
        res = ES.run_grid(grid, pool)
    tb = ES.evaluate(res, list(grid), n_trials)
    keep = [ES.BASE, "X14 만기 때 상위20이면 연장", *EXTRA, "E2 순위=126일 모멘텀", "E3 순위=12-1개월 모멘텀",
            "X12 만기 40일", "X2 추적손절 15%", "C 조합: R4 + X14 + M4 + E2"]
    sub = tb[tb["설정"].isin(keep)]
    sub.to_csv(ES.OUT / "g2_grid.csv", index=False, encoding="utf-8-sig")
    print(f"### ①② 이웃·최소변경안 (DSR N={n_trials} 로 재계산 — 전체 시도 합산)")
    print(sub.to_string(index=False))

    P = ES.panels()["PIT"]
    q = ES.benchmarks(P)["지수 매수보유"]
    base = res[("PIT", ES.BASE, ES.COST)][0]
    names = ["X14 만기 때 상위20이면 연장", "E2 순위=126일 모멘텀", "P1 E2+X14 (126일·연장20)",
             "P2 E3+X14 (12-1·연장20)", "X12 만기 40일", "X2 추적손절 15%", "C 조합: R4 + X14 + M4 + E2"]
    ly = pd.DataFrame({n.split()[0]: loyo(res[("PIT", n, ES.COST)][0], base) for n in names}).round(3)
    ly.to_csv(ES.OUT / "g2_loyo.csv", encoding="utf-8-sig")
    print("\n### ③ 연도 하나 빼고 ΔSharpe vs L0 (행 = 뺀 연도)")
    print(ly.to_string())
    print("  최소값:", ly.min().round(3).to_dict())

    rows = []
    for n in [ES.BASE] + names:
        r = res[("PIT", n, ES.COST)][0]
        x, y = q.align(r, join="inner")
        bs = C.paired_block_boot(x.values, y.values)
        m = C.stat(r)
        rows.append({"설정": n, "CAGR%": round(m["cagr"], 2), "Sharpe": round(m["sharpe"], 3),
                     "MDD%": round(m["mdd"], 1), "QQQ 대비 ΔSh": round(bs["d_sharpe"], 3),
                     "CI": f"{bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f}",
                     "누적수익 차": round(bs["d_cum"], 2)})
    qt = pd.DataFrame(rows)
    qt.to_csv(ES.OUT / "g2_vs_qqq.csv", index=False, encoding="utf-8-sig")
    m = C.stat(q)
    print(f"\n### ④ QQQ 매수보유 대비 (QQQ: CAGR {m['cagr']:.2f}% · Sharpe {m['sharpe']:.3f} · MDD {m['mdd']:.1f}%)")
    print(qt.to_string(index=False))


if __name__ == "__main__":
    main()
