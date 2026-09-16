"""만기(시간)만이 청산 조건인 문제 — 조건부 청산 후보를 운영에 옮길 수 있는 형태로 잰다.

    python research/exit_condition.py

사전등록 그리드(g_grid.csv)에서 이미 잰 것:
  · 가격 기반 청산(손절·추적손절·모멘텀음전·50일선) → 전부 REJECT. 판 종목이 오히려 올랐다.
  · X14 '만기 때 선별 상위 N위면 연장' → ΔSh +0.296 (CI +0.083~+0.506), DSR 0.76. 최고 성적.
  · 만기 40일 → 1999~2014 만 보고 골라도 2015~2026 에서 이긴다 (유일한 표본 외 근거).

여기서 새로 재는 것 — **운영 동등성**:
  X14 는 연장할 때 보유 시계를 리셋해 H일을 더 준다. 운영은 진입 시각을 브로커 체결 이력에서
  읽으므로 리셋할 수단이 없다. 만기 이후 **매 사이클** 재확인하고 순위에서 빠지는 날 판다
  (`extend_daily`). 둘이 같은 성적인지 확인해야 X14 의 근거를 운영에 가져다 쓸 수 있다.

출력: research/out/exit_condition.csv
"""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from research import exit_study as ES       # noqa: E402  (guard() 는 여기서 켜진다)
from research import corr_limits as C       # noqa: E402

CASES = {
    "L0 현행 (만기 20일, 시간만)":              ES.v(),
    "만기 40일 (표본 외 근거 있음)":             ES.v(hold=40),
    "X14 상위20 연장·시계 리셋 (백테스트 형태)":   ES.v(extend=20),
    "X14L 상위20 연장·매일 재확인 (운영 형태)":    ES.v(extend=20, extend_daily=True),
    "X14L 상위10 연장·매일 재확인":              ES.v(extend=10, extend_daily=True),
    "X14L 상위30 연장·매일 재확인":              ES.v(extend=30, extend_daily=True),
    "X14L 상위40 연장·매일 재확인":              ES.v(extend=40, extend_daily=True),
    "X14L+40 만기40·상위20·매일 재확인":         ES.v(hold=40, extend=20, extend_daily=True),
}


def scorecard():
    """사전등록 채택 기준 전체(두 반기·25bp·CI·DSR·편향구간 방향)로 다시 매긴다.

    N 은 누적 시도 수다 — G 연구 37 + live_gap 6 + 여기 7 = 50. 시도가 늘면 DSR 은 내려간다.
    """
    import multiprocessing as mp
    grid = {ES.BASE: ES.L0, **{k: v for k, v in CASES.items() if k != list(CASES)[0]}}
    with mp.Pool(min(8, mp.cpu_count() - 1), initializer=ES._init) as pool:
        res = ES.run_grid(grid, pool)
    tb = ES.evaluate(res, list(grid), n_trials=50)
    tb.to_csv(ES.OUT / "exit_condition_scorecard.csv", index=False, encoding="utf-8-sig")
    return tb


def rows_for(P, label, cost):
    out = {}
    for name, cfg in CASES.items():
        r, turn, exits, w = ES.sim(P, {**cfg, "cost_bp": cost})
        out[name] = (r, turn, exits, w)
    return out


def main():
    panels = ES.panels()
    P, B = panels["PIT"], panels["BIAS"]
    res10 = rows_for(P, "PIT", ES.COST)
    res25 = rows_for(P, "PIT", ES.STRESS)
    resb = rows_for(B, "BIAS", ES.COST)
    base = res10["L0 현행 (만기 20일, 시간만)"][0]

    rows = []
    for name in CASES:
        r, turn, exits, w = res10[name]
        s = C.stat(r)
        why = pd.Series([e[2] for e in exits]).value_counts() if exits else pd.Series(dtype=int)
        rec = {"설정": name, "CAGR%": round(s["cagr"], 2), "Sharpe": round(s["sharpe"], 3),
               "MDD%": round(s["mdd"], 1), "연회전율": round(turn, 2),
               "청산건수": len(exits), "만기청산": int(why.get("만기", 0)),
               "노출축소": int(why.get("노출축소", 0)),
               "Sh 25bp": round(C.stat(res25[name][0])["sharpe"], 3),
               "Sh 편향99~14": round(C.stat(resb[name][0])["sharpe"], 3)}
        if name == "L0 현행 (만기 20일, 시간만)":
            rec["ΔSh"], rec["ΔSh 95%CI"] = 0.0, "(기준)"
        else:
            a, b = base.align(r, join="inner")
            bs = C.paired_block_boot(a, b)
            rec["ΔSh"] = round(bs["d_sharpe"], 3)
            rec["ΔSh 95%CI"] = f"{bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f}"
        rows.append(rec)
    tb = pd.DataFrame(rows)
    tb.to_csv(ES.OUT / "exit_condition.csv", index=False, encoding="utf-8-sig")
    print(f"PIT {P['dates'][0].date()} ~ {P['dates'][-1].date()} · 편도 {ES.COST}bp\n")
    print(tb.to_string(index=False))

    # 운영 형태와 백테스트 형태가 같은가 — 이게 아니면 X14 근거를 운영에 못 가져온다
    a, b = res10["X14 상위20 연장·시계 리셋 (백테스트 형태)"][0].align(
        res10["X14L 상위20 연장·매일 재확인 (운영 형태)"][0], join="inner")
    bs = C.paired_block_boot(a, b)
    print(f"\n운영 형태 − 백테스트 형태: ΔSharpe {bs['d_sharpe']:+.3f} "
          f"(95% CI {bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f})")
    print("\n### 사전등록 채택 기준 전체 (누적 시도 N=50)")
    print(scorecard().to_string(index=False))
    print(f"\n→ {ES.OUT / 'exit_condition.csv'}, exit_condition_scorecard.csv")


if __name__ == "__main__":
    main()
