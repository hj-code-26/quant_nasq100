"""현금 유지선 복원 매도 — 운영 교착 해소용 규칙이 전략 성적을 해치지 않는지 (비열등성 검정).

    python research/reserve_restore.py   # → research/out/reserve_restore.csv

배경 (live_diagnosis.md · fix_options.md)
  실계좌 현금 $0.01 · 유지선 $57 → 가용현금 음수 → 매수 0건. 매도 경로는 20거래일 만기뿐 (첫 만기 ~10-05).
  사용자 결정(2026-09-17): 현금이 유지선보다 크게 모자라면 큰 종목부터 팔아 유지선을 채운다.
  이 규칙은 **성적 개선이 목적이 아니다** — 설계된 현금 규칙이 다시 작동하게 하는 것. 그래서 판정은 비열등성이다.

사전 등록 (2026-09-17, 결과 보기 전 커밋)
  규칙    매 거래일, 청산·노출축소 뒤 진입 전: 현금 < 유지선(10%) × 0.5 이면 유지선까지 큰 종목부터 매도.
          최소 주문(min_frac) 미만 조각은 팔지 않는다. 운영 구현도 같은 순서·같은 트리거로 만든다.
  시도    R0 L0 + 복원(트리거 0.5)
          R1 L0 + 복원 + 실계좌 최소주문 $5/$570
          → 2 시도. 누적 N = 70 + 2 = 72
  비교    R0 ↔ L0,  R1 ↔ L0·실계좌 최소주문 (각각 짝지은 블록 부트스트랩)
  판정    PASS(운영에 넣어도 됨) = ΔSharpe 95% CI 하한 > −0.10 **그리고** 25bp Sharpe 하락 < 0.05
          FAIL 이면 운영 규칙을 기본 꺼짐으로만 넣고 사용자에게 알린다
  서술    발동 일수, 회전율 변화, CAGR·MDD
  불변식  restore 를 주지 않으면 L0 수치 그대로 (CAGR 7.63 · Sharpe 0.450) — assert
"""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import exit_study as ES      # noqa: E402  (guard() 가 여기서 켜진다)
from research import corr_limits as C      # noqa: E402

N_TRIALS = 72
LIVE_MIN_FRAC = 5 / 570
PAIRS = {
    "R0 L0 + 유지선 복원": (ES.v(restore=0.5), ES.v()),
    "R1 L0 + 유지선 복원 · 실계좌 최소주문": (ES.v(restore=0.5, min_frac=LIVE_MIN_FRAC), ES.v(min_frac=LIVE_MIN_FRAC)),
}


def main():
    P = ES.panels()["PIT"]
    s0 = C.stat(ES.sim(P, ES.v())[0])
    assert round(s0["cagr"], 2) == 7.63 and round(s0["sharpe"], 3) == 0.450, s0
    rows = []
    for name, (cfg, base) in PAIRS.items():
        probe = {}
        r, tv, ex, w = ES.sim(P, {**cfg, "probe": probe})
        b, btv, _, bw = ES.sim(P, base)
        r25, b25 = ES.sim(P, {**cfg, "cost_bp": ES.STRESS})[0], ES.sim(P, {**base, "cost_bp": ES.STRESS})[0]
        x, y = b.align(r, join="inner")
        bs = C.paired_block_boot(x.values, y.values)
        s, sb = C.stat(r), C.stat(b)
        d25 = C.stat(r25)["sharpe"] - C.stat(b25)["sharpe"]
        ok = bs["d_sharpe_lo"] > -0.10 and d25 > -0.05
        rows.append({"설정": name, "CAGR%": round(s["cagr"], 2), "기준 CAGR%": round(sb["cagr"], 2),
                     "Sharpe": round(s["sharpe"], 3), "기준 Sharpe": round(sb["sharpe"], 3),
                     "MDD%": round(s["mdd"], 1), "기준 MDD%": round(sb["mdd"], 1),
                     "회전율": round(tv, 2), "기준 회전율": round(btv, 2),
                     "발동일": probe.get("복원 발동", 0), "전량복원청산": sum(e[2] == "유지선복원" for e in ex),
                     "ΔSh (CI)": f"{bs['d_sharpe']:+.3f} ({bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f})",
                     "25bp ΔSh": round(d25, 3), "판정": "PASS" if ok else "FAIL"})
    tb = pd.DataFrame(rows)
    tb.to_csv(ES.OUT / "reserve_restore.csv", index=False, encoding="utf-8-sig")
    print(f"### 비열등성 — PIT 2015~2026, 편도 {ES.COST:.0f}bp, 누적 N={N_TRIALS}\n")
    print(tb.to_string(index=False))


if __name__ == "__main__":
    main()
