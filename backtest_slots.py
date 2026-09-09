"""보유 종목 수(슬롯) 재검토 — 운영 시스템과 **같은 구조**로 시뮬레이션한다.

왜 새 파일인가.
  backtest_rules.py 의 포트폴리오는 '겹치는 트랜치'다. 매일 자본의 1/보유일을 새로 넣기 때문에
  상위 5종목만 골라도 실제로는 평균 **17종목**을 동시에 들고 있게 된다.
  반면 운영 시스템(autotrade.py)은 MAX_POSITIONS 개 슬롯이 전부다. 항상 정확히 5종목이다.
  → 지금까지의 백테스트는 실전보다 3배 이상 분산된 포트폴리오를 재고 있었다.

여기서는 운영 구조를 그대로 흉내낸다.
  · 슬롯 N 개. 빈 슬롯이 생기면 그때 보유하지 않은 종목 중 모멘텀 1위부터 채운다
  · 진입 금액 = 그 시점 현금 / 빈 슬롯 수 (운영의 '한도 안에서 균등 배분'에 대응)
  · 청산 = 보유 hold 거래일 경과 (만기 청산). 청산 전까지 비중은 그대로 흘러간다 (리밸런스 없음)
  · 리밸런스를 안 하는 것도 의도적이다 — 운영 코드가 그렇게 동작한다

사용:  python backtest_slots.py
       python backtest_slots.py --vt     (살아남은 안에 변동성 타겟팅까지 얹어 비교)

★ 생존 편향: 유니버스가 오늘의 나스닥 100. 절대 CAGR 은 부풀려져 있다. 설정 간 상대 비교로 읽을 것.
"""
import sys

import numpy as np
import pandas as pd

from backtest_rules import SPLIT, features, metrics

SLOTS = (3, 5, 8, 10, 15, 20)
HOLDS = (20, 60)


def simulate(N, hold, rank, ret, dates):
    """슬롯 N 개, 만기 hold 일. 반환: (자본곡선, 평균 보유종목수, 최대 단일종목 비중)."""
    n_days, n_sym = ret.shape
    pos = np.zeros(n_sym)              # 종목별 평가액
    day = np.full(n_sym, -1)           # 진입일 인덱스
    cash = 1.0
    eq = np.empty(n_days)
    maxw = 0.0
    for t in range(n_days):
        held = pos > 0
        if held.any():                                    # 1) 평가액 갱신
            pos[held] *= 1 + np.nan_to_num(ret[t][held])
            gone = held & (t - day >= hold)               # 2) 만기 청산
            if gone.any():
                cash += pos[gone].sum()
                pos[gone] = 0.0
                day[gone] = -1
        held = pos > 0
        free = N - int(held.sum())
        if free > 0 and cash > 1e-12:                     # 3) 빈 슬롯 채우기
            rk = rank[t]
            order = np.argsort(rk, kind="stable")
            picks = [c for c in order[:N + n_sym // 4]
                     if np.isfinite(rk[c]) and not held[c]][:free]
            if picks:
                per = cash / len(picks)
                for c in picks:
                    pos[c] = per
                    day[c] = t
                cash -= per * len(picks)
        v = cash + pos.sum()
        eq[t] = v
        if v > 0 and pos.max() / v > maxw:
            maxw = pos.max() / v
    curve = pd.Series(eq, index=dates)
    return curve.pct_change().dropna(), maxw


def voltarget(c, target=30, win=60):
    rv = c.rolling(win).std() * np.sqrt(252) * 100
    return (c * (target / rv.shift(1)).clip(upper=1.0)).dropna()


def sm(c):
    return metrics(c["1999-01-01":SPLIT]), metrics(c[SPLIT:"2030-01-01"])


def main():
    f, lab = features("--refresh" in sys.argv)
    close, mom, ret1 = f["close"], f["mom"], f["ret1"]
    rank = np.array(mom.rank(axis=1, ascending=False, method="first").to_numpy(float), copy=True)
    rank[np.isnan(rank)] = np.inf
    ret = np.array(ret1.to_numpy(float), copy=True)
    dates = close.index
    print(f"종목 {close.shape[1]}개, 기간 {dates.min().date()} ~ {dates.max().date()}, "
          f"탐색/검증 경계 {SPLIT}")
    print("구조: 슬롯 N개를 항상 채운다 (운영 autotrade.py 와 동일). 리밸런스 없음.")
    print("★ 생존 편향으로 절대 CAGR 은 부풀려져 있다. 설정 간 상대 비교로만 읽을 것.\n")

    results = {}
    for hold in HOLDS:
        print(f"### 만기 청산 {hold}일")
        print(f"{'슬롯':>5}{'탐CAGR':>9}{'탐MDD':>9}{'탐Sh':>7}  |{'검CAGR':>9}{'검MDD':>9}"
              f"{'검Sh':>7}{'최대단일비중':>13}")
        for n in SLOTS:
            r, maxw = simulate(n, hold, rank, ret, dates)
            results[(n, hold)] = r
            a, b = sm(r)
            print(f"{n:>5}{a['cagr']:+8.1f}%{a['mdd']:8.1f}%{a['sharpe']:7.2f}  |"
                  f"{b['cagr']:+8.1f}%{b['mdd']:8.1f}%{b['sharpe']:7.2f}{maxw * 100:12.1f}%")
        print()

    print("### 국면별 연율 수익률 (전 구간)")
    print(f"{'설정':22}{'상승장':>9}{'횡보장':>9}{'하락장':>9}{'MDD':>9}")
    for (n, hold), r in results.items():
        if n in (3, 5, 10, 15):
            m = metrics(r, lab)
            print(f"슬롯 {n:2} · 만기 {hold:3}일     {m['상승']:+8.1f}%{m['횡보']:+8.1f}%"
                  f"{m['하락']:+8.1f}%{m['mdd']:8.1f}%")

    if "--vt" in sys.argv:
        print("\n### 변동성 타겟 30% 를 얹으면")
        print(f"{'설정':22}{'탐CAGR':>9}{'탐MDD':>9}{'탐Sh':>7}  |{'검CAGR':>9}{'검MDD':>9}{'검Sh':>7}")
        for key in ((5, 20), (5, 60), (10, 60), (15, 60)):
            c = voltarget(results[key])
            a, b = sm(c)
            print(f"슬롯 {key[0]:2} · 만기 {key[1]:3}일 +VT{a['cagr']:+8.1f}%{a['mdd']:8.1f}%"
                  f"{a['sharpe']:7.2f}  |{b['cagr']:+8.1f}%{b['mdd']:8.1f}%{b['sharpe']:7.2f}")

    print("\n### 최악 5개 연도 (%)")
    keys = [(5, 20), (5, 60), (10, 60), (15, 60)]
    yr = pd.DataFrame({f"슬롯{n}·{h}일": (1 + results[(n, h)]).resample("YE").prod() - 1
                       for n, h in keys}) * 100
    yr.index = yr.index.year
    print(yr.loc[yr[f"슬롯5·20일"].nsmallest(6).index].round(1).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
