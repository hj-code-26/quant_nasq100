"""진입 지연 — 모멘텀 신호 직후의 '단기 고점 매수'를 피하면 나아지는가.

    python research/entry_delay.py     # → research/out/entry_delay.csv, entry_delay_mechanism.csv

동기 (timing_study.md)
  평상 국면 진입가가 ±10일 종가평균보다 +2.49% 비싸다(같은 날 전 종목 대비, 월클러스터 t 14.4).
  그런데 20일 성과는 대조군과 같다(−0.02%). 단기 반전(ret5 IC 음수)과 같은 방향.

사전 등록 (결과 보기 전 고정 — 2026-09-17)
  기준선  L0 (운영 근사, exit_study.L0). 코드 기본값에서 L0 수치 재현 확인 (CAGR 7.63 · Sharpe 0.450)
  시도    D1 지연 1일 · D2 지연 3일 · D3 지연 5일
            = 진입 순위·자격(20일수익률>0)·모멘텀등급을 k 거래일 전 정보로 매긴다.
              국면 판정·지수 편입 여부·청산은 오늘 기준 그대로.
          D4 눌림 확인 = 오늘 후보 중 최근 5일 수익률 < 0 인 종목만 진입
          D5 최근 5일 제외 모멘텀 = 순위를 t−25 → t−5 수익률로 (자격 조건은 현행 20일수익률>0)
          → 5 시도. 누적 N = 50(exit_condition 까지) + 5 = 55
  판정    exit_study.evaluate 그대로: ① 두 반기 Sharpe > L0 ② 25bp Sharpe > L0 ③ ΔSh CI 가 0 제외
          ④ DSR ≥ 0.95. 전부 = ADOPT · ①② = CANDIDATE · 그 외 REJECT
  조건부  D1~D3 은 이웃이다 — CANDIDATE 가 하나만 나오고 이웃 부호가 다르면 봉우리로 보고 근거 불가
  조합    하지 않는다 (X14 와의 결합도 이번 시도에 포함하지 않는다)
  기전    (판정에 쓰지 않음) 각 설정의 진입 프리미엄 BPR_대조 와 20일 대조군대비 — 가설대로 움직였는가
  한계    timing_study 를 보고 만든 가설이다 — 같은 2015~2026 표본에서 동기와 검정을 했다.
          편향 구간(1999~2014) 방향이 유일한 반독립 확인이다.
"""
import multiprocessing as mp
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import exit_study as ES      # noqa: E402  (guard() 가 여기서 켜진다)
from research import timing_study as TS    # noqa: E402

N_TRIALS = 55
GRID = {
    ES.BASE: ES.L0,
    "D1 진입 지연 1일": ES.v(delay=1),
    "D2 진입 지연 3일": ES.v(delay=3),
    "D3 진입 지연 5일": ES.v(delay=5),
    "D4 눌림 확인 (5일수익률<0)": ES.v(pullback=True),
    "D5 순위=최근5일 제외 20일 모멘텀": ES.v(rank="mom20s5"),
}


def mechanism(P, res):
    rows = []
    for name in GRID:
        ex = res[("PIT", name, ES.COST)][2]
        ex = [e for e in ex if e[2] == "만기"]
        df = TS.trades(P, ex)
        rec = {"설정": name, "거래": len(df)}
        for lab, g in (("전체", df), ("평상", df[df["국면"] == "평상(모멘텀)"])):
            for col in ("BPR_대조", "대조군대비"):
                _, mt = TS.cmean_t(g[col], g["진입일"].dt.to_period("M"))
                rec[f"{lab} {col}%"] = round(g[col].mean() * 100, 2)
                rec[f"{lab} {col} 월t"] = round(mt, 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    with mp.Pool(min(11, mp.cpu_count() - 1), initializer=ES._init) as pool:
        res = ES.run_grid(GRID, pool)
    tb = ES.evaluate(res, list(GRID), N_TRIALS)
    tb.to_csv(ES.OUT / "entry_delay.csv", index=False, encoding="utf-8-sig")
    print(f"### 사전등록 판정 — PIT 2015~2026, 편도 {ES.COST:.0f}bp, 누적 N={N_TRIALS}\n")
    print(tb.to_string(index=False))

    mc = mechanism(ES.panels()["PIT"], res)
    mc.to_csv(ES.OUT / "entry_delay_mechanism.csv", index=False, encoding="utf-8-sig")
    print("\n### 기전 (판정 미사용) — 진입 프리미엄과 20일 성과, 같은 날 전 종목 대비\n")
    print(mc.to_string(index=False))


if __name__ == "__main__":
    main()
