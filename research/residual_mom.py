"""잔차 모멘텀 · Frog-in-the-pan — 문헌 근거가 있는데 아직 안 잰 선별 규칙.

    python research/residual_mom.py     # → research/out/residual_mom.csv, residual_mom_mechanism.csv

출처
  Blitz·Huij·Martens (2011, JEF) Residual Momentum — 요인 노출을 뺀 잔차로 순위, 위험조정 수익 약 2배
  Da·Gurun·Warachka (2014, RFS) Frog in the Pan — 꾸준히 오른 종목의 모멘텀이 더 강하고 반전하지 않는다

사전 등록 (2026-09-17, 결과 보기 전 커밋)
  기준선  L0 (exit_study.L0). 코드 기본값에서 L0 재현 확인 (CAGR 7.63 · Sharpe 0.450)
  지표    잔차 = 일수익 − β·지수일수익. β = 후행 252일 공분산/분산 (최소 126일). 지수는 ^NDX
          (원논문은 FF 3요인 36개월 — 여기서는 1요인, 일봉. 원형 재현이 아니라 우리 구조에 옮긴 형태다)
          rmom20   = 최근 20일 잔차합 / 20일 잔차 표준편차
          rmom12_1 = t−252 ~ t−21 잔차합 / 같은 창 표준편차 (최소 200일)
          id20     = sign(20일수익) × (20일 중 하락일 비율 − 상승일 비율). 낮을수록 꾸준한 상승
  시도    F1 순위 = rmom20
          F2 순위 = rmom12_1
          F3 20일 모멘텀 상위 20개 중 id20 낮은 순으로 슬롯 채움
          공통: 진입 자격(20일수익률>0)·모멘텀등급 사이징·하락 국면 저변동성 선별·청산은 L0 그대로
          → 3 시도. 누적 N = 55(entry_delay 까지) + 3 = 58
  판정    exit_study.evaluate 그대로: ① 두 반기 Sharpe > L0 ② 25bp Sharpe > L0 ③ ΔSh CI 0 제외
          ④ DSR ≥ 0.95. 전부 = ADOPT · ①② = CANDIDATE · 그 외 REJECT
  하지 않음 조합(D5·X14 포함), 파라미터 이웃 탐색(상위 20 → 10/30, 창 길이) — 하면 새 시도
  불변식  세 설정 모두 prefix(끝 400일 삭제 후 재계산) 괴리 < 1e-12 를 먼저 assert
  기전    (판정 미사용) timing_study 방식의 진입 프리미엄(BPR_대조)과 20일 같은날 대조군대비
  표본 외 이 두 규칙은 우리 표본을 보고 만든 가설이 아니다(외부 문헌). 그래도 N 에 더한다.
"""
import multiprocessing as mp
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import exit_study as ES      # noqa: E402  (guard() 가 여기서 켜진다)
from research import entry_delay as ED     # noqa: E402

N_TRIALS = 58
GRID = {
    ES.BASE: ES.L0,
    "F1 순위=20일 잔차 모멘텀": ES.v(rank="rmom20"),
    "F2 순위=12-1개월 잔차 모멘텀": ES.v(rank="rmom12_1"),
    "F3 모멘텀 상위20 중 꾸준함(id20) 순": ES.v(fip=20),
}


def prefix_checks(k_back=400):
    close, _, ndx, qqq = ES.raw()
    full_p = ES.build(close, ndx, qqq, ES.PIT_START)
    part_p = ES.build(close.iloc[:-k_back], ndx, qqq, ES.PIT_START)
    for name, cfg in GRID.items():
        if name == ES.BASE:
            continue
        full, part = ES.sim(full_p, cfg)[0], ES.sim(part_p, cfg)[0]
        gap = float((full.loc[part.index] - part).abs().max())
        assert gap < 1e-12, f"{name}: prefix 불변식 실패 (괴리 {gap})"
        print(f"prefix 불변식 통과 — {name} (괴리 {gap:.1e})")


def main():
    prefix_checks()
    with mp.Pool(min(11, mp.cpu_count() - 1), initializer=ES._init) as pool:
        res = ES.run_grid(GRID, pool)
    tb = ES.evaluate(res, list(GRID), N_TRIALS)
    tb.to_csv(ES.OUT / "residual_mom.csv", index=False, encoding="utf-8-sig")
    print(f"\n### 사전등록 판정 — PIT 2015~2026, 편도 {ES.COST:.0f}bp, 누적 N={N_TRIALS}\n")
    print(tb.to_string(index=False))

    ED.GRID = GRID                              # 기전 표는 entry_delay 와 같은 계산을 재사용
    mc = ED.mechanism(ES.panels()["PIT"], res)
    mc.to_csv(ES.OUT / "residual_mom_mechanism.csv", index=False, encoding="utf-8-sig")
    print("\n### 기전 (판정 미사용) — 진입 프리미엄과 20일 성과, 같은 날 전 종목 대비\n")
    print(mc.to_string(index=False))


if __name__ == "__main__":
    main()
