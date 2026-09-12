"""기존 코드 vs PIT 반영 코드 — **같은 데이터·같은 규칙**으로 수익률을 나란히 낸다.

    python research/pit_compare.py   → research/out/pit_compare.csv

두 번 돌린다: `pit.ENABLED=False`(기존) / `True`(수정). 나머지는 한 글자도 다르지 않다.
불변식: 커버리지 시작(2015-01-01) **이전** 구간은 일별 수익률이 **완전히 같아야** 한다 —
PIT 마스크가 그 구간엔 손대지 않기 때문이다. 매 실행 assert 한다.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from research.isolation import guard        # noqa: E402

guard()

import pit                                  # noqa: E402
from research import corr_limits as C       # noqa: E402

OUT = pathlib.Path(__file__).with_name("out")
SPLIT = "2015-01-01"


def run(enabled):
    pit.ENABLED = enabled
    import backtest_rules
    f, _ = backtest_rules.features(False)
    mom, ret1, close = f["mom"], f["ret1"], f["close"]
    rank = np.array(mom.rank(axis=1, ascending=False, method="first").to_numpy(float), copy=True)
    rank[np.isnan(rank)] = np.inf
    r, _, tv = C.simulate_g(C.SLOTS, C.HOLD, rank, np.array(ret1.to_numpy(float), copy=True),
                            close.index, None, None, C.COST_BPS)
    return r, tv


def main():
    old, tv_o = run(False)
    new, tv_n = run(True)

    pre_o, pre_n = old[:SPLIT], new[:SPLIT]          # 커버리지 이전 = 손대지 않은 구간
    gap = float((pre_o - pre_n).abs().max())
    assert gap == 0.0, f"2015 이전 구간이 달라졌다 (괴리 {gap}) — 마스크가 범위를 넘었다"
    print(f"불변식 통과 — 2015-01-01 이전 {len(pre_o)}일 일별 수익률 완전 일치 (괴리 0.0)")

    rows = []
    for label, lo, hi in (("탐색 1999~2014 (마스크 없음)", "1900-01-01", SPLIT),
                          ("검증 2015~2026 (마스크 적용)", SPLIT, "2100-01-01"),
                          ("전체", "1900-01-01", "2100-01-01")):
        for name, r, tv in (("기존 (오늘의 유니버스)", old, tv_o), ("수정 (당시 구성종목)", new, tv_n)):
            m = C.stat(r[lo:hi])
            rows.append({"구간": label, "코드": name, "총수익%": round(m["ret"], 1),
                         "CAGR%": round(m["cagr"], 2), "변동성%": round(m["vol"], 2),
                         "Sharpe": round(m["sharpe"], 3), "MDD%": round(m["mdd"], 2),
                         "Calmar": round(m["calmar"], 2)})
    tb = pd.DataFrame(rows)
    tb.to_csv(OUT / "pit_compare.csv", index=False, encoding="utf-8-sig")
    print(f"\n### 장기 패널 (1998~2026, 슬롯 {C.SLOTS}·만기 {C.HOLD}일·편도 {C.COST_BPS}bp)")
    print(tb.to_string(index=False))

    a, b = old[SPLIT:], new[SPLIT:]
    bs = C.paired_block_boot(a.values, b.values)
    print(f"\n검증 구간 수정−기존: Sharpe {bs['d_sharpe']:+.3f} "
          f"(95% CI {bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f})")
    print("산출물: pit_compare.csv")


if __name__ == "__main__":
    main()
