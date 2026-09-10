"""수수료·슬리피지를 물렸을 때 결론이 버티는지 확인한다.

지금까지의 모든 백테스트(backtest_rules / _slots / _bear)는 **거래비용 0** 을 가정했다.
그런데 규칙마다 회전율이 3배씩 차이 난다 (슬롯5·만기20일 = 연 11.1배 vs 만기60일 = 연 3.6배).
비용을 넣으면 우열이 뒤집힐 수 있으므로, 운영 구성과 같은 슬롯 시뮬레이션에 비용을 물려 다시 잰다.

비용 모델
  · `cost_bp` = **한 방향** 거래비용 (bp). 매수·매도 양쪽에 각각 물린다
  · 왕복 비용 = 2 × cost_bp. 수수료 + 스프레드 절반(시장가 주문) + 체결 오차를 합친 값으로 본다
  · 토스 실제 수수료율을 못 받아(API 403) 특정 값을 단정하지 않는다. 대신 0~50bp 를 훑어
    **결론이 뒤집히는 지점**을 찾는다. 본인 수수료율 + 스프레드 절반을 넣어 해당 행을 보면 된다
  · 주문 크기가 $60~90 수준이라 시장충격은 무시한다 (나스닥100 대형주 기준)

구성은 운영 코드(autotrade.py)와 같다.
  슬롯 N개 · 보유 hold 거래일 만기 청산 · 하락 국면(QQQ 60일 < -3%)이면 저변동성 선별 전환
  · 계좌 실현 변동성으로 노출 축소(변동성 타겟)

사용:  python backtest_costs.py            (전체)
       python backtest_costs.py --quick    (핵심 표만)

★ 생존 편향은 그대로 남아 있다. 절대 CAGR 이 아니라 **비용에 따른 순위 변화**를 보는 도구다.
"""
import sys

import numpy as np
import pandas as pd

from backtest_bear import build_ranks
from backtest_rules import SPLIT, features, metrics
from market_data import INDEX, load_ohlcv

SLOTS, HOLD = 10, 20
VOL_TARGET, VOL_WIN = 30, 60
COSTS = (0, 5, 10, 15, 25, 50)      # 한 방향 bp


def setup(refresh=False):
    f, lab = features(refresh)
    close, ret1 = f["close"], f["ret1"]
    dates = close.index
    ret = np.array(ret1.to_numpy(float), copy=True)
    ranks = build_ranks(f, lab)
    d = load_ohlcv(refresh)
    idx = d["close"][INDEX].dropna()
    bear = (((idx / idx.shift(60) - 1) * 100).reindex(dates) < -3).fillna(False).to_numpy()
    atr = ((d["high"] - d["low"]).rolling(14).mean() / d["close"] * 100)
    atr = atr.drop(columns=[INDEX]).reindex(index=dates, columns=close.columns)
    ranks["atr_all"] = np.nan_to_num(
        atr.rank(axis=1, ascending=True, method="first").to_numpy(float), nan=np.inf)
    return ret, dates, ranks, bear


def simulate(ret, dates, ranks, bear, slots=SLOTS, hold=HOLD, cost_bp=0.0,
             stop=None, bear_switch=True, vol_target=VOL_TARGET, vol_win=VOL_WIN):
    """운영 구성 슬롯 시뮬레이션 + 거래비용. 반환: (일별수익률, 지표 dict)."""
    rk_all = np.where(bear[:, None], ranks["atr_all"], ranks["mom_hi"]) if bear_switch \
        else ranks["mom_hi"]
    c = cost_bp / 10000.0
    n_days, n_sym = ret.shape
    pos = np.zeros(n_sym); ent = np.zeros(n_sym); day = np.full(n_sym, -1)
    cash = 1.0
    eq = np.empty(n_days)
    rets = []; prev = 1.0
    traded = 0.0                      # 누적 거래대금 / 그 시점 자산 (회전율)
    trades = []; n_stop = 0

    def sell(mask, keep=0.0):
        """mask 종목을 (1-keep) 비율만큼 판다. 비용을 물리고 현금으로."""
        nonlocal cash, traded
        amt = pos[mask].sum() * (1 - keep)
        eq_now = cash + pos.sum()
        if eq_now > 0:
            traded += amt / eq_now        # 자산 대비 비율로 누적해야 28년 복리에 안 휩쓸린다
        cash += amt * (1 - c)
        pos[mask] *= keep

    for t in range(n_days):
        held = pos > 0
        if held.any():
            pos[held] *= 1 + np.nan_to_num(ret[t][held])
            pnl = np.where(held, pos / np.where(ent > 0, ent, 1) - 1, 0.0)
            hit = (held & (pnl <= -(stop or 999) / 100)) if stop else np.zeros(n_sym, bool)
            gone = (held & (t - day >= hold)) | hit
            if gone.any():
                for k in np.flatnonzero(gone):
                    trades.append((pos[k] / ent[k] - 1) * 100 - 2 * cost_bp / 100)
                n_stop += int(hit.sum())
                sell(gone)
                pos[gone] = 0.0; ent[gone] = 0.0; day[gone] = -1

        v = cash + pos.sum()
        rets.append(v / prev - 1 if prev > 0 else 0.0); prev = v

        cap = 1.0
        if vol_target and len(rets) > vol_win:
            rv = float(np.std(rets[-vol_win - 1:-1], ddof=1)) * np.sqrt(252) * 100
            if rv > 0:
                cap = min(1.0, vol_target / rv)
        held = pos > 0
        if cap < 1.0 and held.any() and v > 0 and pos.sum() / v > cap:
            sell(held, keep=cap * v / pos.sum())          # 노출 축소 (비용 물림)

        held = pos > 0
        free = slots - int(held.sum())
        if free > 0 and cash > 1e-12:
            v = cash + pos.sum()
            room = v * cap - pos.sum() if vol_target else cash
            budget = max(0.0, min(cash, room))
            if budget > 1e-12:
                r = rk_all[t]
                order = np.argsort(r, kind="stable")
                picks = [k for k in order[:slots + n_sym // 4]
                         if np.isfinite(r[k]) and not held[k]][:free]
                if picks:
                    per = budget / len(picks)
                    if v > 0:
                        traded += per * len(picks) / v
                    for k in picks:
                        pos[k] = per * (1 - c)            # 비용만큼 덜 사진다
                        ent[k] = pos[k]; day[k] = t
                    cash -= per * len(picks)
        eq[t] = cash + pos.sum()

    curve = pd.Series(eq, index=dates).pct_change().dropna()
    yrs = (dates[-1] - dates[0]).days / 365.25
    x = pd.Series(trades)
    return curve, {"turnover": traded / yrs, "trades_yr": len(x) / yrs,
                   "win": (x > 0).mean() * 100 if len(x) else np.nan,
                   "n_stop": n_stop,
                   "cost_yr": traded / yrs * cost_bp / 10000 * 100}   # 자산 대비 연 %


def sm(c):
    return metrics(c["1999-01-01":SPLIT]), metrics(c[SPLIT:"2030-01-01"])


def row(label, curve, st, base=None):
    a, b = sm(curve)
    d = "" if base is None else f"{b['cagr'] - base:+7.1f}"
    return (f"{label:26}{a['cagr']:+8.1f}%{a['sharpe']:7.2f}  |{b['cagr']:+8.1f}%"
            f"{b['mdd']:8.1f}%{b['sharpe']:7.2f}{d}{st['trades_yr']:8.0f}"
            f"{st['turnover']:6.1f}배{st['cost_yr']:7.2f}%"), b


def main():
    ret, dates, ranks, bear = setup("--refresh" in sys.argv)
    A = (ret, dates, ranks, bear)
    head = (f"{'구성':26}{'탐CAGR':>9}{'탐Sh':>7}  |{'검CAGR':>9}{'검MDD':>9}{'검Sh':>7}"
            f"{'검Δ':>7}{'연거래':>8}{'회전':>7}{'연비용':>8}")
    print(f"기간 {dates.min().date()} ~ {dates.max().date()},  기준 구성 = 슬롯{SLOTS}·만기{HOLD}일"
          f"·하락장 저변동성 전환·변동성타겟{VOL_TARGET}%")
    print("★ 생존 편향은 그대로다. 절대 CAGR 이 아니라 비용에 따른 순위 변화를 볼 것.")
    print("  연비용 = 연 거래대금 × 한방향 bp (자산 대비 %). 왕복이므로 실제 부담은 이 값이다.\n")

    print("### ① 거래비용 민감도 (기준 구성)")
    print(head)
    base_cagr = None
    for bp in COSTS:
        c, st = simulate(*A, cost_bp=bp)
        line, b = row(f"한방향 {bp}bp", c, st, base_cagr)
        if base_cagr is None:
            base_cagr = b["cagr"]
        print(line)

    print("\n### ② 슬롯 수 — 회전율이 다르므로 비용에 다르게 반응한다")
    for bp in (0, 10, 25):
        print(f"\n[한방향 {bp}bp]"); print(head)
        for n in (5, 8, 10, 15):
            c, st = simulate(*A, slots=n, cost_bp=bp)
            print(row(f"슬롯 {n}", c, st)[0])

    print("\n### ③ 만기 — 회전율 차이가 가장 큰 축")
    for bp in (0, 10, 25):
        print(f"\n[한방향 {bp}bp]"); print(head)
        for h in (10, 20, 40, 60):
            c, st = simulate(*A, hold=h, cost_bp=bp)
            print(row(f"만기 {h}일", c, st)[0])

    if "--quick" in sys.argv:
        return

    print("\n### ④ 손절 — 발동할수록 회전이 늘어 비용이 겹친다")
    for bp in (0, 10, 25):
        print(f"\n[한방향 {bp}bp]"); print(head)
        for stp in (None, 15, 25):
            c, st = simulate(*A, stop=stp, cost_bp=bp)
            print(row("손절 없음" if stp is None else f"손절 -{stp}%", c, st)[0])

    print("\n### ⑤ 하락장 전환 · 변동성 타겟 — 비용을 물려도 유지되는가")
    for bp in (0, 10, 25):
        print(f"\n[한방향 {bp}bp]"); print(head)
        for label, kw in (("둘 다 없음", dict(bear_switch=False, vol_target=0)),
                          ("하락장 전환만", dict(bear_switch=True, vol_target=0)),
                          ("변동성 타겟만", dict(bear_switch=False, vol_target=VOL_TARGET)),
                          ("둘 다 (운영 구성)", dict(bear_switch=True, vol_target=VOL_TARGET))):
            c, st = simulate(*A, cost_bp=bp, **kw)
            print(row(label, c, st)[0])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
