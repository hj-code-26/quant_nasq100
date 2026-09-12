"""PIT 구성종목 (2015~) — 검증 구간의 생존 편향을 줄인다. **제거가 아니라 축소다.**

    python research/pit_universe.py --fetch    # 없는 티커 시세 내려받기 (yfinance, 1회)
    python research/pit_universe.py            # 편향 크기 측정

출처: jmccarrell/n100tickers (MIT). 연도별 YAML = 1월 1일 구성종목 + 날짜별 union/difference.
  → research/data/pit/n100-YYYY.yaml (오프라인 사본, LICENSE 동봉)
  커버리지는 **2015-01-01 이후**다. 그 전 구간(1999~2014)은 여전히 오늘의 유니버스다.

무엇을 고치나
  기존 백테스트는 **오늘의** 나스닥100 을 2015년에도 들고 있었다. PLTR·APP·MSTR 처럼
  급등 뒤에 편입된 종목을 편입 전부터 살 수 있었다는 뜻이다 (= 미래를 안 채로 고른 것).
  여기서는 t 일에 **실제로 지수에 있던** 종목만 매수 후보로 둔다.

무엇을 못 고치나 (사용자 결정: 폐지 종목은 범위 밖)
  · 인수·합병으로 사라진 회사의 가격을 yfinance 로 못 받으면 그 종목은 그냥 빠진다.
    그 회사들이 편입돼 있던 기간의 수익(대개 피인수 프리미엄)이 표본에서 사라진다.
  · 따라서 결과를 "생존 편향 제거"라고 부르지 않는다. **편향 축소**이고, 남은 결손은
    pit_missing.csv 에 종목·기간·일수로 전부 적는다.
  · 티커 변경(FB→META 등)은 새 티커 이력이 옛 구간까지 이어져 있으면 자동으로 이어진다
    (rename_map). 이어지지 않으면 결손으로 남긴다.
"""
import argparse
import datetime
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

DATA = pathlib.Path(__file__).with_name("data") / "pit"
PRICES = DATA / "pit_prices.pkl"
OUT = pathlib.Path(__file__).with_name("out")
START = "2015-01-01"

from pit import RENAME, events, mask as pit_mask   # 구성종목 로더는 최상위 pit.py 하나뿐이다


def membership():
    return events()


def daily_mask(ev, dates, symbols):
    return pit_mask(dates, symbols, quiet=True)


# ---------- 패널 ----------
def panel():
    """2015~ 결합 가격 패널 + PIT 마스크 + 결손 집계."""
    base = pickle.load(open("data_cache/yf_ohlcv.pkl", "rb"))["close"].drop(columns=["^NDX"])
    old = pickle.load(PRICES.open("rb"))
    add = [c for c in old.columns if c not in base.columns]
    close = base.join(old[add], how="outer").loc[START:]
    close = close.loc[:, close.notna().sum() > 60]
    syms = list(close.columns)
    ev = membership()
    mask = daily_mask(ev, close.index, syms)

    # 결손 = 그날 지수 구성종목인데 가격이 없는 종목-일수
    have = set(syms)
    miss = {}
    tot_md = gap_md = 0
    keys, cur, k = list(ev), set(), 0
    for t, d in enumerate(close.index):
        while k < len(keys) and keys[k] <= d.date():
            cur = {RENAME.get(s, s) for s in ev[keys[k]]}
            k += 1
        tot_md += len(cur)
        for s in cur:
            if s not in have or not np.isfinite(close.iloc[t][s]):
                gap_md += 1
                m = miss.setdefault(s, [0, d.date(), d.date()])
                m[0] += 1
                m[2] = d.date()
    md = pd.DataFrame([{"symbol": s, "결손_일수": v[0], "first": v[1], "last": v[2],
                        "사유": "가격 없음 (인수·합병·폐지 추정)" if s not in have else "구간 결측"}
                       for s, v in sorted(miss.items(), key=lambda x: -x[1][0])])
    return close, np.array(mask), md, tot_md, gap_md


def main():
    import backtest_slots  # noqa: F401  (동일 저울 확인용 import 경로 유지)
    from nasdaq100 import TICKERS
    from research import corr_limits as C

    close, mask, md, tot_md, gap_md = panel()
    dates = close.index
    ret = close.pct_change().to_numpy(float)
    mom = (close / close.shift(20) - 1)
    print(f"구간 {dates[0].date()}~{dates[-1].date()} ({len(dates)}거래일), 패널 {close.shape[1]}종목")
    print(f"구성종목-일수 {tot_md:,} 중 가격 결손 {gap_md:,} ({gap_md / tot_md * 100:.1f}%) "
          f"— 종목 {len(md)}개")
    print(f"PIT 마스크: 하루 평균 매수 가능 종목 {mask.sum(1).mean():.1f}개")

    def ranks(valid):
        m = mom.where(valid)
        r = np.array(m.rank(axis=1, ascending=False, method="first").to_numpy(float), copy=True)
        r[np.isnan(r)] = np.inf
        return r

    cur_valid = pd.DataFrame(np.tile([c in set(TICKERS) for c in close.columns], (len(dates), 1)),
                             index=dates, columns=close.columns)
    rank_cur = ranks(cur_valid)                                    # 지금까지의 방식
    rank_pit = ranks(pd.DataFrame(mask, index=dates, columns=close.columns))   # PIT

    # 불변식: PIT 런은 **그날 구성종목이 아닌 종목을 절대 후보로 두지 않는다**
    assert np.isinf(rank_pit[~mask]).all(), "PIT 마스크 밖 종목이 순위에 남아 있다"
    assert (rank_pit[mask] != np.inf).sum() > 0
    # 기존 방식은 실제로 미래 편입 종목을 후보로 뒀는가 (편향이 존재한다는 증거)
    early = int(((~mask) & np.isfinite(rank_cur)).sum())
    print(f"기존 방식이 '그날 지수에 없던 종목'을 후보로 둔 종목-일수: {early:,}")

    # 분해용: PIT ∩ 오늘의 유니버스 — 되살린 옛 구성종목 80개를 빼고 '미래 편입'만 막는다
    rank_pitnow = ranks(pd.DataFrame(mask, index=dates, columns=close.columns) & cur_valid)

    rows, curves = [], {}
    for name, rk, cfg in (("오늘의 유니버스(기존)", rank_cur, None),
                          ("PIT ∩ 오늘의 유니버스 (분해용)", rank_pitnow, None),
                          ("PIT 구성종목", rank_pit, None),
                          ("PIT + 상관군 rho0.60/cap3", rank_pit, (0.60, 3))):
        g = C.group_matrix(ret, cfg[0]) if cfg else None
        r, maxw, tv = C.simulate_g(C.SLOTS, C.HOLD, rk, ret, dates, g,
                                   None if cfg is None else cfg[1], C.COST_BPS)
        curves[name] = r
        m = C.stat(r)
        rows.append({"구성": name, "총수익%": round(m["ret"], 1), "CAGR%": round(m["cagr"], 2),
                     "변동성%": round(m["vol"], 2), "Sharpe": round(m["sharpe"], 3),
                     "MDD%": round(m["mdd"], 2), "Calmar": round(m["calmar"], 2),
                     "최대단일비중%": round(maxw * 100, 1), "연turnover": round(tv, 1)})
    qqq = pickle.load((DATA / "QQQ_2015.pkl").open("rb"))
    q = qqq.squeeze().reindex(dates).ffill().pct_change().dropna()
    mq = C.stat(q)
    rows.append({"구성": "QQQ 매수보유 (총수익)", "총수익%": round(mq["ret"], 1),
                 "CAGR%": round(mq["cagr"], 2), "변동성%": round(mq["vol"], 2),
                 "Sharpe": round(mq["sharpe"], 3), "MDD%": round(mq["mdd"], 2),
                 "Calmar": round(mq["calmar"], 2), "최대단일비중%": 100.0, "연turnover": 0.0})
    tb = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    tb.to_csv(OUT / "pit_bias.csv", index=False, encoding="utf-8-sig")
    md.to_csv(OUT / "pit_missing.csv", index=False, encoding="utf-8-sig")
    print(f"\n### 검증 구간 편향 크기 (슬롯 {C.SLOTS} · 만기 {C.HOLD}일 · 편도 {C.COST_BPS}bp)")
    print(tb.to_string(index=False))

    a, b = curves["오늘의 유니버스(기존)"], curves["PIT 구성종목"]
    bs = C.paired_block_boot(a.values, b.values)
    pd.DataFrame([bs]).to_csv(OUT / "pit_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(f"\nPIT − 기존: Sharpe {bs['d_sharpe']:+.3f} "
          f"(95% CI {bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f}), "
          f"누적수익 배수 차 {bs['d_cum']:+.3f} (CI {bs['d_cum_lo']:+.3f}~{bs['d_cum_hi']:+.3f})")

    print("\n### 가격을 못 받아 빠진 옛 구성종목 (상위 12)")
    print(md.head(12).to_string(index=False))
    print("\n산출물: pit_bias.csv, pit_missing.csv, pit_bootstrap.csv")
    print("★ 이것은 편향 **축소**다. 결손 종목의 기간이 표본에서 빠져 있으므로 "
          "'생존 편향 제거'라고 쓰지 않는다.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="옛 구성종목 시세 내려받기 (1회)")
    if ap.parse_args().fetch:
        fetch_missing()
    else:
        main()
