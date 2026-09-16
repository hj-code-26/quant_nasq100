"""라이브와 백테스트의 간극을 두 가지로 쪼개 잰다. 읽기 전용·오프라인 (guard 는 exit_study 가 켠다).

    python research/live_gap.py

  G1. 최소 주문 금액 — 백테스트는 NAV 의 0.1% 를 최소 주문으로 본다. 실계좌는 $5/$570 = 0.88%.
      계좌가 작을수록 '현금은 있는데 못 산다' 가 잦아진다. 얼마나 잦은지, 성적은 얼마나 깎이는지.
  G2. 하락 국면 진입 — 코드는 허용(momentum_tier 가 하락이면 배수 1.0)하는데
      2단계 프롬프트에는 그 구간이 없어 LLM 이 "약 구간이면 hold" 로 전부 거부한다.
      그 거부가 성적에 얼마인지.

출력: research/out/live_gap.csv
"""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from research import exit_study as ES       # noqa: E402  (guard() 는 여기서 켜진다)
from research import corr_limits as C       # noqa: E402

OUT = ES.OUT

# 실계좌 기준 최소 주문 비율 = MIN_ORDER_USD / 총자산 (2026-09-16 기준 $5 / $568.83)
LIVE_MIN_FRAC = 5.0 / 568.83

CASES = {
    "L0 현행(백테스트 가정 min 0.1%)":        ES.v(),
    f"G1a 최소주문 {LIVE_MIN_FRAC:.2%} (실계좌 $570)": ES.v(min_frac=LIVE_MIN_FRAC),
    "G1b 최소주문 2% (계좌 $250 상당)":        ES.v(min_frac=0.02),
    "G1c 최소주문 5% (계좌 $100 상당)":        ES.v(min_frac=0.05),
    "G2 하락 국면 신규진입 차단(현 프롬프트)":   ES.v(bear_noentry=True),
    "G1+G2 둘 다 (현 운영 실태)":             ES.v(min_frac=LIVE_MIN_FRAC, bear_noentry=True),
}


def main():
    P = ES.panels()["PIT"]
    rows, curves = [], {}
    for name, cfg in CASES.items():
        probe = {}
        r, turn, exits, w = ES.sim(P, {**cfg, "probe": probe})
        s = C.stat(r)
        curves[name] = r
        free = probe.get("빈슬롯일", 0)
        rows.append({
            "설정": name,
            "CAGR%": round(s["cagr"], 2), "MDD%": round(s["mdd"], 1),
            "Sharpe": round(s["sharpe"], 3), "변동성%": round(s["vol"], 1),
            "평균주식비중%": round(w * 100, 1), "연회전율": round(turn, 2),
            "청산건수": len(exits),
            "빈슬롯일": free,
            "현금유지에 막힌 날%": round(probe.get("현금유지에 막힘", 0) / max(free, 1) * 100, 1),
            "최소주문에 막힌 날%": round(probe.get("최소주문에 막힘", 0) / max(free, 1) * 100, 1)})
    tb = pd.DataFrame(rows)
    base = curves["L0 현행(백테스트 가정 min 0.1%)"]
    # ΔSharpe 는 점추정만 보면 과신한다 — 짝 유지 블록 부트스트랩 CI 를 같이 낸다.
    d, ci = [], []
    for n in tb["설정"]:
        a, b = base.align(curves[n], join="inner")
        if n == tb["설정"].iloc[0]:
            d.append(0.0); ci.append("(기준)"); continue
        bs = C.paired_block_boot(a, b)
        d.append(round(bs["d_sharpe"], 3))
        ci.append(f"{bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f}")
    tb["ΔSharpe(vs L0)"], tb["ΔSh 95%CI"] = d, ci
    tb.to_csv(OUT / "live_gap.csv", index=False, encoding="utf-8-sig")
    print(f"PIT 패널 {P['dates'][0].date()} ~ {P['dates'][-1].date()}  비용 {ES.COST}bp\n")
    print(tb.to_string(index=False))
    print(f"\n→ {OUT / 'live_gap.csv'}")
    print("\n주의: 이 표는 '시뮬 안에서의 차이'다. 실전 동등성은 여전히 미입증(conclusion.md ③).")


if __name__ == "__main__":
    main()
