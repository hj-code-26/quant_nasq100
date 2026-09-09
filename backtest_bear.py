"""하락장 전용 전략 탐색 — 국면에 따라 **진입 규칙만** 갈아끼운다.

출발점 (backtest_slots.py 제안안: 슬롯 10 · 만기 20일 · 모멘텀 상위)
    상승장 승률 67.6% / 횡보장 51.7% / **하락장 41.1%, 평균 -3.12%**
하락장이 유일한 구멍이다. 국면 필터(하락장 현금)는 검증에서 실패했으므로(backtest_rules.py J·K),
"쉬는" 대신 **다른 규칙으로 바꾸는" 쪽을 검증한다.

단서는 backtest_volume.py ④ 격자의 하락장 행이다.
    모멘텀 음수 행 = 승률 56.7~59.9%   /   모멘텀 >20% 행 = 승률 48.9~54.4%
    → 하락장에서는 모멘텀이 뒤집힌다 (평균회귀). 이걸 슬롯 구조에서 실제로 잰다.

방법
  · 슬롯 구조는 backtest_slots.py 와 동일 (운영 autotrade.py 와 같은 구조)
  · 국면 라벨은 **그 날까지의 데이터만** 쓴다 (지수 60일 수익률, 거래량 20일/250일 비율)
    → 실시간으로 계산 가능하다. 사후 라벨이 아니다
  · 상승·횡보장에서는 모든 안이 동일(모멘텀 상위). **하락장 진입 규칙만** 바꾼다
  · 거래는 **진입일 국면**으로 분류한다 (진입 규칙을 평가하므로)
  · 탐색 1999~2014 / 검증 2015~2026 으로 나눠 **하락장 승률을 각각** 본다
    탐색의 하락장 = 닷컴·금융위기 / 검증의 하락장 = 2018 4분기·코로나·2022

사용:  python backtest_bear.py

★ 생존 편향: 유니버스가 오늘의 나스닥 100. 하락장 절대 승률은 실제보다 높게 나온다
  (망한 회사가 표본에 없다). 안(案) 간 상대 비교로만 읽을 것.
★ 국면 임계값(±3%, 1.05)은 전체 분포를 보고 정했다. 약한 사후 편향이 있다.
"""
import sys

import numpy as np
import pandas as pd

from backtest_rules import SPLIT, features, metrics

N_SLOT = 10
HOLD = 20
BIG_LOSS = -10.0


def build_ranks(f, lab):
    """국면별로 갈아끼울 '그 날의 종목 순위' 후보들. 값이 작을수록 먼저 산다."""
    mom, slope, vol20 = f["mom"], f["slope"], f["vol20"]
    inf = np.inf
    R = {}
    R["mom_hi"] = mom.rank(axis=1, ascending=False, method="first")       # 모멘텀 상위 (기본)
    R["mom_lo"] = mom.rank(axis=1, ascending=True, method="first")        # 모멘텀 하위 (역발상)
    R["lowvol"] = vol20.rank(axis=1, ascending=True, method="first")      # 저변동성
    # 격자 최고 칸: 모멘텀 음수 중 기울기가 가장 급락한 것
    neg = mom < 0
    R["neg_dip"] = slope.where(neg).rank(axis=1, ascending=True, method="first")
    # 모멘텀 음수 중 낙폭이 가장 큰 것 (순수 평균회귀)
    R["neg_deep"] = mom.where(neg).rank(axis=1, ascending=True, method="first")
    R["cash"] = mom * np.nan                                             # 아무것도 안 삼
    out = {}
    for k, v in R.items():
        a = np.array(v.to_numpy(float), copy=True)
        a[np.isnan(a)] = inf
        out[k] = a
    return out


def simulate(rank_by_day, ret, dates, hold_by_day=None, n=N_SLOT, hold=HOLD):
    """슬롯 n 개. rank_by_day[t] 가 그 날의 순위. 반환: (일별수익률, 거래표)."""
    n_days, n_sym = ret.shape
    pos = np.zeros(n_sym); ent = np.zeros(n_sym)
    day = np.full(n_sym, -1); dur = np.full(n_sym, hold)
    cash = 1.0
    eq = np.empty(n_days)
    trades = []
    for t in range(n_days):
        held = pos > 0
        if held.any():
            pos[held] *= 1 + np.nan_to_num(ret[t][held])
            gone = held & (t - day >= dur)
            if gone.any():
                for c in np.flatnonzero(gone):
                    trades.append(((pos[c] / ent[c] - 1) * 100, t - day[c], day[c]))
                cash += pos[gone].sum()
                pos[gone] = 0.0; ent[gone] = 0.0; day[gone] = -1
        held = pos > 0
        free = n - int(held.sum())
        if free > 0 and cash > 1e-12:
            rk = rank_by_day[t]
            order = np.argsort(rk, kind="stable")
            picks = [c for c in order[:n + n_sym // 4]
                     if np.isfinite(rk[c]) and not held[c]][:free]
            if picks:
                per = cash / len(picks)
                h = hold if hold_by_day is None else int(hold_by_day[t])
                for c in picks:
                    pos[c] = per; ent[c] = per; day[c] = t; dur[c] = h
                cash -= per * len(picks)
        eq[t] = cash + pos.sum()
    curve = pd.Series(eq, index=dates).pct_change().dropna()
    tr = pd.DataFrame(trades, columns=["ret", "days", "entry_i"])
    tr["entry"] = dates[tr["entry_i"].to_numpy()]
    return curve, tr


def stack_rank(ranks, regime, plan):
    """plan = {'상승': 키, '횡보': 키, '하락': 키} → 날짜별로 해당 순위 행렬을 골라 쌓는다."""
    out = np.empty_like(ranks["mom_hi"])
    for kind, key in plan.items():
        m = (regime == kind)
        out[m] = ranks[key][m]
    return out


def bear_stats(tr, lab, lo, hi):
    g = tr[(tr["entry"] >= lo) & (tr["entry"] < hi)].copy()
    g["방향"] = lab["방향"].reindex(g["entry"]).to_numpy()
    b = g[g["방향"] == "하락"]["ret"]
    if len(b) < 20:
        return None
    return {"n": len(b), "win": (b > 0).mean() * 100, "mean": b.mean(),
            "big": (b <= BIG_LOSS).mean() * 100}


def main():
    f, lab = features("--refresh" in sys.argv)
    close, ret1 = f["close"], f["ret1"]
    dates = close.index
    ret = np.array(ret1.to_numpy(float), copy=True)
    ranks = build_ranks(f, lab)
    regime = lab["방향"].reindex(dates).to_numpy()          # 그 날까지의 데이터로 계산된 라벨
    vol_regime = lab["국면"].reindex(dates).to_numpy()

    plans = {
        "0 기준: 하락장에도 모멘텀 상위": {"상승": "mom_hi", "횡보": "mom_hi", "하락": "mom_hi"},
        "A 하락장 → 모멘텀 하위 (역발상)": {"상승": "mom_hi", "횡보": "mom_hi", "하락": "mom_lo"},
        "B 하락장 → 음수모멘텀 중 급락": {"상승": "mom_hi", "횡보": "mom_hi", "하락": "neg_dip"},
        "C 하락장 → 음수모멘텀 중 최대낙폭": {"상승": "mom_hi", "횡보": "mom_hi", "하락": "neg_deep"},
        "D 하락장 → 저변동성": {"상승": "mom_hi", "횡보": "mom_hi", "하락": "lowvol"},
        "E 하락장 → 현금 (비교용)": {"상승": "mom_hi", "횡보": "mom_hi", "하락": "cash"},
    }
    print(f"종목 {close.shape[1]}개, 기간 {dates.min().date()} ~ {dates.max().date()}")
    print(f"구조: 슬롯 {N_SLOT} · 만기 {HOLD}일 · 진입일 국면으로 거래 분류 · 경계 {SPLIT}")
    print("★ 생존 편향으로 하락장 절대 승률은 부풀려져 있다. 안 간 상대 비교로 읽을 것.\n")

    print("### 하락장 거래만 (탐색 / 검증 따로)")
    print(f"{'안':34}{'탐 n':>7}{'탐승률':>8}{'탐평균':>8}{'탐-10%':>8}  |"
          f"{'검 n':>7}{'검승률':>8}{'검평균':>8}{'검-10%':>8}")
    curves, trades = {}, {}
    for name, plan in plans.items():
        rk = stack_rank(ranks, regime, plan)
        c, tr = simulate(rk, ret, dates)
        curves[name], trades[name] = c, tr
        a = bear_stats(tr, lab, "1999-01-01", SPLIT)
        b = bear_stats(tr, lab, SPLIT, "2030-01-01")
        fa = (f"{a['n']:>7,}{a['win']:7.1f}%{a['mean']:+7.2f}%{a['big']:7.1f}%" if a
              else f"{'표본부족':>30}")
        fb = (f"{b['n']:>7,}{b['win']:7.1f}%{b['mean']:+7.2f}%{b['big']:7.1f}%" if b
              else f"{'표본부족':>30}")
        print(f"{name:34}{fa}  |{fb}")

    print("\n### 전체 포트폴리오 (하락장 규칙 교체가 전체에 미치는 영향)")
    print(f"{'안':34}{'탐CAGR':>9}{'탐MDD':>9}{'탐Sh':>7}  |{'검CAGR':>9}{'검MDD':>9}{'검Sh':>7}")
    for name, c in curves.items():
        a = metrics(c["1999-01-01":SPLIT]); b = metrics(c[SPLIT:"2030-01-01"])
        print(f"{name:34}{a['cagr']:+8.1f}%{a['mdd']:8.1f}%{a['sharpe']:7.2f}  |"
              f"{b['cagr']:+8.1f}%{b['mdd']:8.1f}%{b['sharpe']:7.2f}")

    print("\n### 하락장을 거래량으로 쪼개면 (실림·하락 vs 마름·하락, 전 구간)")
    print(f"{'안':34}{'실림 n':>8}{'승률':>8}{'평균':>8}  |{'마름 n':>8}{'승률':>8}{'평균':>8}")
    for name, tr in trades.items():
        g = tr.copy()
        g["국면"] = pd.Series(vol_regime, index=dates).reindex(g["entry"]).to_numpy()
        cells = ""
        for k in ("실림·하락", "마름·하락"):
            x = g[g["국면"] == k]["ret"]
            cells += (f"{len(x):>8,}{(x > 0).mean() * 100:7.1f}%{x.mean():+7.2f}%" if len(x) >= 20
                      else f"{'-':>24}")
            cells += "  |" if k == "실림·하락" else ""
        print(f"{name:34}{cells}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
