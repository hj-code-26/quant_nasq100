"""SOXL 규칙 2차 — 1차(soxl_rules.md, 9설정 전부 REJECT)에서 드러난 약점을 겨냥해 조정.

    python research/soxl_rules_v2.py   # → research/soxl_rules_v2.md, research/out/soxl_rules_v2.csv

1차에서 본 것 (이 조정안은 그 결과를 **보고** 만들었다 — 독립 검증이 아니다)
  · 리스크 규칙 없으면 합성 1994~ 에서 무한매수법 −99.5%, 스윙 −97.4% → 2차는 전부 R1+R2p+R3 위에서 조정
  · 가격 낙폭 −30% 관망이 생존을 만들었지만 평균 SOXL 비중 17% 로 수익을 크게 깎았다
  · 스윙은 RSI≤30 을 폭락장에서 사고 20일선 교차로 휩소 청산 — 전 구간 최하
  · 20/80 리밸런싱이 가장 안정적이었으나 비중이 낮아 실제 구간 Sharpe < SOXX

사전 등록 (2026-09-17, 결과 보기 전 커밋) — 시뮬레이터·자료·비용·체결·판정 기준은 1차와 동일
  공통   리스크 R1(현금 30%) + R2p(SOXL 252일 고점 대비 낙폭 관망) + R3(63거래일 50% 축소)
  A1 무한매수법 + 쿼터손절: 40회분 소진 후 종가 < 평단이면 보유 1/4 종가 매도, 10회분을 다시 쓸 수 있게 한다
  A2 무한매수법, 관망 기준 −30% → −50%
  B1 스윙 + 추세 필터: RSI≤30 진입은 종가 > 200일선일 때만 (200일선 터치 반등 진입은 그대로)
  B2 B1 + 청산 이동평균 20일 → 50일 하향 교차
  C1 리밸런싱 목표 30% (밴드 ±10%p → 20~40%)
  C2 리밸런싱 목표 40% (30~50%)
  C3 리밸런싱 동적 목표: SOXL 종가 > 200일선이면 30%, 아니면 10% (밴드 ±10%p)
  → 7 시도. 누적 N = 92 + 7 = 99. 조합은 하지 않는다
  판정   1차와 같다: ① 합성 1994~ MDD > −60% ② 실제 2010~ Sharpe > SOXX 매수보유 ③ 두 구간 CAGR > 현금
  불변식 1차 기본값 수치 재현(M1·R1+R2p+R3 실제 13.1%·0.76·−29.8%) + prefix(7설정) + 현금 = 단기금리 — assert
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import soxl_rules as S       # noqa: E402  (guard() 는 여기서 켜진다)

R = {"risk": "price"}
GRID = {
    "A1 무한매수법 + 쿼터손절": {"module": "M1", **R, "quarter": True},
    "A2 무한매수법 · 관망 −50%": {"module": "M1", **R, "dd": -0.50},
    "B1 스윙 + 200일선 추세 필터": {"module": "M2", **R, "trend": True},
    "B2 B1 + 50일선 청산": {"module": "M2", **R, "trend": True, "exit_ma": 50},
    "C1 리밸런싱 목표 30%": {"module": "M3", **R, "target": 0.30},
    "C2 리밸런싱 목표 40%": {"module": "M3", **R, "target": 0.40},
    "C3 리밸런싱 동적 30%/10%": {"module": "M3", **R, "dyn": True},
}

if __name__ == "__main__":
    act, _, _ = S.load()
    e, _, _ = S.sim(act, S.GRID["M1 · R1+R2p+R3"])
    st = S.stat(e, act["rf"])
    assert (st["CAGR%"], st["Sharpe"], st["MDD%"]) == (13.1, 0.76, -29.8), st
    print("1차 기본값 재현 확인 (M1·R1+R2p+R3 실제 13.1% · 0.76 · −29.8%)")
    if "--check" in sys.argv:
        S.check(act, GRID)
        print("불변식 prefix(7설정)·현금 통과")
        sys.exit(0)
    S.main(GRID, tag="soxl_rules_v2", title="SOXL 규칙 2차 (사전등록, 누적 N=99)")
