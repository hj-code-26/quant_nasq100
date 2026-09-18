"""멀티슬롯 SOXL 규칙 — 여러 종목이 **현금을 공유**하는 한 계좌에서 각자 밴드 리밸런싱 +
R1(슬롯당 상한) + R2p(종목 자체 252일 고점 대비 관망) + R3(63일 축소) 를 적용.

research/soxl_rules_multislot.py(슬롯별 독립자본 가정, 탐색용 1차)의 후속 — 이번엔 실제 운영처럼
현금 하나를 슬롯들이 나눠 쓴다. autotrade.py 의 soxl_orders()/soxl_plan() 을 다종목으로 확장한
것과 같은 결정 로직이다(신주문 코드는 아직 없음 — 이 스크립트는 성과 검증용).

    python research/soxl_multislot_shared.py

규칙 (심볼별로 동일)
    슬롯 목표비중 = TOTAL_TARGET / N          (기본 90% 투자, 10% 현금 유지)
    밴드   ±BAND (기본 5%p)
    R1     슬롯당 상한 = 목표 + R1_HEADROOM   (한 종목이 지나치게 커지는 것 방지)
    R2p    그 종목 자체 252일 고점 대비 DD_LIM 이하면 그 종목 신규 매수만 중단
    R3     보유 63거래일 도달 시 50% 축소 (슬롯당 1회)
    체결   결정 당일 종가 (연구용 근사 — soxl_rules.py 와 동일한 단순화)
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import soxl_rules as S        # noqa: E402  (stat() 재사용)

FEE = S.FEE
MIN_TRADE = 0.001                            # 최소 거래 규모(V 대비) — 이보다 작으면 스킵


def load(symbols):
    panel = pickle.load(open(ROOT / "research" / "data" / "pit" / "ohlcv_panel.pkl", "rb"))
    close_all = pickle.load(open(ROOT / "research" / "data" / "soxl_close.pkl", "rb"))
    rf = (close_all["^IRX"].ffill() / 100 / 252)
    close = panel["Close"][symbols]
    return close, rf


def sim(close, rf, cfg):
    symbols = list(close.columns)
    n = len(symbols)
    total_target = cfg.get("total_target", 0.90)
    each_target = total_target / n
    band = cfg.get("band", 0.05)
    r1_cap = each_target + cfg.get("r1_headroom", 0.15)
    dd_lim = cfg.get("dd", -0.30)
    trim_days = cfg.get("trim_days", 63)
    FEE = cfg.get("fee", S.FEE)          # 스트레스 테스트용 슬리피지 상향 오버라이드

    c = close.to_numpy(float)
    hi252 = close.rolling(252, min_periods=1).max().to_numpy(float)
    rf_v = rf.reindex(close.index).ffill().fillna(0).to_numpy(float)
    dates = close.index
    T = len(dates)

    cash = 1.0
    qty = np.zeros(n)
    age = np.zeros(n, dtype=int)
    trimmed = np.zeros(n, dtype=bool)
    eq = np.ones(T)
    wts = np.zeros((T, n))
    trades = 0
    # pending[i]: 전날 결정 → 오늘 종가 체결. ("sell", frac) | ("buy", 어제 계산한 목표 매수 금액$)
    pending = [[] for _ in range(n)]

    for t in range(T):
        if t > 0:
            cash *= 1 + rf_v[t]
        price = c[t]
        listed = ~np.isnan(price)

        # 1) 어제 결정 → 오늘 종가 체결. 매도 먼저(현금 확보), 그다음 매수.
        for i in range(n):
            for kind, x in pending[i]:
                if kind == "sell" and qty[i] > 1e-12:
                    q = qty[i] * min(1.0, x)
                    cash += q * price[i] * (1 - FEE)
                    qty[i] -= q
                    trades += 1
        V = cash + np.nansum(qty * price)
        need = {}
        for i in range(n):
            for kind, x in pending[i]:
                if kind == "buy":
                    blocked = hi252[t, i] > 0 and price[i] / hi252[t, i] - 1 <= dd_lim
                    if not blocked:
                        need[i] = min(x, r1_cap * V - qty[i] * price[i])   # R2p·R1 은 체결일 기준으로 다시 본다
        for i in sorted(need, key=lambda k: -need[k]):
            amt = min(need[i], cash / (1 + FEE))
            if amt <= MIN_TRADE * V:
                continue
            cash -= amt * (1 + FEE)
            qty[i] += amt / price[i]
            trades += 1
        pending = [[] for _ in range(n)]

        assert cash >= -1e-9 and (qty >= -1e-9).all(), (t, cash, qty)
        V = cash + np.nansum(qty * price)
        eq[t] = V
        wts[t] = np.where(listed, qty * np.where(listed, price, 0) / max(V, 1e-12), 0.0)

        # 2) 오늘 종가 기준 결정 → 내일 체결 (pending 에 쌓는다)
        weight = wts[t]
        for i in range(n):
            if qty[i] > 1e-12:
                age[i] += 1
            else:
                age[i], trimmed[i] = 0, False
            if not listed[i]:
                continue
            acts = []
            if weight[i] > each_target + band:
                acts.append(("sell", 1 - each_target / weight[i]))
            elif weight[i] < each_target - band:
                acts.append(("buy", each_target * V - qty[i] * price[i]))
            if qty[i] > 1e-12 and age[i] >= trim_days and not trimmed[i]:
                acts.append(("sell", 0.5))
                trimmed[i] = True
            pending[i] = acts

    return pd.Series(eq, dates), pd.DataFrame(wts, dates, symbols), trades


def bench_equal_weight(close, rf):
    r = close.pct_change().fillna(0)
    n_listed = close.notna().sum(axis=1).clip(lower=1)
    port_r = (r.where(close.notna(), 0).sum(axis=1) / n_listed)
    return (1 + port_r).cumprod()


def main():
    groups = {
        "5종목": ["NVDA", "PLTR", "CTAS", "SHOP", "RKLB"],
        "6종목": ["NVDA", "PLTR", "CTAS", "SHOP", "RKLB", "MU"],
    }
    rows = []
    for gname, syms in groups.items():
        close, rf = load(syms)
        close = close.dropna(how="all")
        for total_target in (0.70, 0.90, 1.00):
            for band in (0.03, 0.05, 0.08):
                cfg = {"total_target": total_target, "band": band}
                eq, wts, trades = sim(close, rf, cfg)
                st = S.stat(eq, rf.reindex(eq.index).ffill().fillna(0))
                yrs = len(eq) / 252
                rows.append({"그룹": gname, "총목표": total_target, "밴드": band,
                             **st, "평균투자비중%": round(wts.sum(axis=1).mean() * 100, 1),
                             "연매매": round(trades / yrs, 1)})
        # 스트레스: 슬리피지 편도 0.12% → 0.25% (leaders_bt.py STRESS_SLIP 과 동일 가정)
        eq_s, wts_s, trades_s = sim(close, rf, {"total_target": 0.70, "band": 0.05, "fee": 0.0025})
        st_s = S.stat(eq_s, rf.reindex(eq_s.index).ffill().fillna(0))
        rows.append({"그룹": gname, "총목표": "0.7(스트레스 슬리피지0.25%)", "밴드": 0.05, **st_s,
                     "평균투자비중%": round(wts_s.sum(axis=1).mean() * 100, 1),
                     "연매매": round(trades_s / (len(eq_s) / 252), 1)})
        bh = bench_equal_weight(close, rf)
        st_bh = S.stat(bh, rf.reindex(bh.index).ffill().fillna(0))
        rows.append({"그룹": gname, "총목표": "매수보유(동일가중)", "밴드": "-", **st_bh})

    tb = pd.DataFrame(rows)
    tb.to_csv(ROOT / "research" / "out" / "soxl_multislot_shared.csv", index=False, encoding="utf-8-sig")
    pd.options.display.width = 200
    print(tb.to_string(index=False))


if __name__ == "__main__":
    main()
