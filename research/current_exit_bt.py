"""현행 매도 조건(운영 기본값) 그대로의 백테스트 — 채택 판정이 아니라 **현황 보고**.

    python research/current_exit_bt.py     # → research/out/current_exit_bt.{csv,md}

현행 운영값 (.env 에 전략 키 없음 → autotrade.py 기본값):
  MAX_HOLD_DAYS=20 · STOP_LOSS_PCT=0 · MOMENTUM_EXIT=0 · HOLD_EXTEND_TOP=0 ·
  BEAR_EXPOSURE_PCT=None · VOL_TARGET_PCT=30 · MAX_POSITIONS=10 · MAX_POSITION_PCT=15 ·
  CASH_RESERVE_PCT=10 · MIN_ORDER_USD=5
이 조합은 research/exit_study.py 의 L0 과 같다. 엔진은 L0 을 그대로 쓰고 새로 만들지 않는다.
격리: exit_study import 시 research.isolation.guard() 가 켜진다 (운영 DB·브로커 차단).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import exit_study as ES      # noqa: E402  (guard() 가 여기서 켜진다)
from research import corr_limits as C      # noqa: E402

OUT = ROOT / "research" / "out"

# 운영 계좌 규모의 최소 주문: $5 / 총자산 ~$570 ≈ 0.88% (L0 은 0.1% 가정)
LIVE_MIN_FRAC = 5 / 570

CASES = {
    "현행 L0 (편도 10bp)": ES.L0,
    "현행 L0 · 비용 25bp": {**ES.L0, "cost_bp": ES.STRESS},
    "현행 L0 · 실계좌 최소주문($5/$570)": {**ES.L0, "min_frac": LIVE_MIN_FRAC},
}


def row(name, r, tv=None, w=None):
    s = C.stat(r)
    return {"구성": name, "기간": f"{r.index[0]:%Y-%m}~{r.index[-1]:%Y-%m}",
            "CAGR%": round(s["cagr"], 2), "연변동성%": round(s["vol"], 1),
            "Sharpe": round(s["sharpe"], 3), "MDD%": round(s["mdd"], 1),
            "Calmar": round(s["calmar"], 2),
            "연turnover": None if tv is None else round(tv, 1),
            "평균주식비중%": None if w is None else round(w * 100)}


def main():
    P = ES.panels()
    gap = ES.check(P["PIT"])                    # 엔진 동치 불변식 (실패하면 assert)
    print(f"엔진 불변식 통과 (괴리 {gap:.1e})\n")

    tables, yearly, reasons = {}, None, {}
    for pname in ("PIT", "BIAS"):
        rows, rets = [], {}
        for name, cfg in CASES.items():
            if pname == "BIAS" and name != "현행 L0 (편도 10bp)":
                continue
            r, tv, ex, w = ES.sim(P[pname], cfg)
            rows.append(row(name, r, tv, w))
            rets[name] = r
            if name == "현행 L0 (편도 10bp)":
                reasons[pname] = pd.Series([x[2] for x in ex]).value_counts()
        bm = ES.benchmarks(P[pname])
        label = "QQQ 매수보유" if pname == "PIT" else "NDX 지수 매수보유(가격)"
        rows.append(row(label, bm["지수 매수보유"]))
        rets[label] = bm["지수 매수보유"]
        tables[pname] = pd.DataFrame(rows)
        if pname == "PIT":
            base = rets["현행 L0 (편도 10bp)"]
            q = rets[label].reindex(base.index).fillna(0)
            yearly = pd.DataFrame({
                "현행 L0 %": (1 + base).groupby(base.index.year).prod().sub(1).mul(100).round(1),
                "QQQ %": (1 + q).groupby(q.index.year).prod().sub(1).mul(100).round(1)})
            yearly["차이 %p"] = (yearly["현행 L0 %"] - yearly["QQQ %"]).round(1)
            x, y = q.align(base, join="inner")
            bs = C.paired_block_boot(x.values, y.values)

    OUT.mkdir(parents=True, exist_ok=True)
    md = []
    for pname, title in (("PIT", "주 표본 — PIT 구성종목 2015-01~2026-09"),
                         ("BIAS", "참고 — 1999~2014 현재 구성종목 (생존편향 있음)")):
        md.append(f"## {title}\n\n" + tables[pname].to_string(index=False))
        md.append("청산 사유 건수: " + ", ".join(f"{k} {v}" for k, v in reasons[pname].items()))
        tables[pname].to_csv(OUT / f"current_exit_bt_{pname}.csv", index=False,
                             encoding="utf-8-sig")
    md.append("## 연도별 (PIT, 편도 10bp)\n\n" + yearly.to_string())
    md.append(f"현행 L0 − QQQ Sharpe 차 {bs['d_sharpe']:+.3f} "
              f"(95% 블록부트스트랩 {bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f})")
    text = "\n\n".join(md)
    (OUT / "current_exit_bt.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
