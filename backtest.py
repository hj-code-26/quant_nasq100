"""보조지표 조건 → 이후 수익률 백테스트 (나스닥 100, 토스 일봉).

질문: "지표가 이런 값일 때 사면 이후 5일·20일에 올랐는가?" 를 전 종목·전 기간에서 센다.
  · 기준선 = 아무 조건 없이 아무 날이나 샀을 때의 평균 수익률·상승 확률
  · 각 조건의 표본 수, 평균 수익률, 상승 확률, 기준선 대비 차이, t-통계량
  · 시간순 앞 2/3(탐색) / 뒤 1/3(검증) 으로 나눠, 탐색 구간에서 좋아 보인 조건이 검증 구간에서도 유지되는지 본다

사용:  python backtest.py            (첫 실행은 종목당 일봉 ~750개 수집, 캐시 후 재사용)
       python backtest.py --refresh  (캐시 무시)

주의: 겹치는 표본(이웃 날짜는 미래 구간을 공유)이라 t-통계량은 유효 표본 n/horizon 으로 깎아 계산한다.
      종목 간 같은 날의 상관(시장 전체가 오르는 날)은 보정하지 않으므로 t 는 여전히 낙관적이다. |t|>3 정도는 돼야 믿을 만하다.
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

from autotrade import add_indicators, candles, momentum_tier, rule_score
from nasdaq100 import TICKERS
from toss import shared_client

CACHE = pathlib.Path(__file__).with_name("data_cache")
DAYS = 750          # 약 3년
HORIZONS = (5, 20)


def load(sym, refresh=False):
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{sym}_1d.pkl"
    if p.exists() and not refresh:
        return pickle.load(p.open("rb"))
    df = candles(shared_client(), sym, "1d", DAYS)
    pickle.dump(df, p.open("wb"))
    return df


def features(df):
    df = add_indicators(df.copy())
    f = pd.DataFrame(index=df.index)
    f["close_gt_sma20"] = df["close"] > df["SMA_20"]
    f["sma20_gt_sma50"] = df["SMA_20"] > df["SMA_50"]
    f["ret_20d"] = df["close"].pct_change(20) * 100
    f["ret_5d"] = df["close"].pct_change(5) * 100
    f["macd_hist"] = df.get("MACDh_12_26_9")
    f["rsi"] = df["RSI_14"]
    f["bbp"] = df.get("BBP_20_2.0_2.0")
    f["stoch_k"] = df.get("STOCHk_14_3_3")
    f["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    # 규칙 점수: 각 날짜에 대해 그 날까지의 데이터로 계산 (미래 참조 없음)
    score = []
    for i in range(len(df)):
        score.append(rule_score(df.iloc[:i + 1]) if i >= 60 else np.nan)
    f["score"] = score
    for h in HORIZONS:
        f[f"fwd{h}"] = (df["close"].shift(-h) / df["close"] - 1) * 100
    return f


def stats(sub, h, base_mean, base_hit):
    x = sub[f"fwd{h}"].dropna()
    n = len(x)
    if n < 30:
        return None
    n_eff = max(1, n // h)
    t = (x.mean() - base_mean) / (x.std(ddof=1) / np.sqrt(n_eff)) if x.std() > 0 else 0.0
    return {"n": n, "mean": x.mean(), "hit": (x > 0).mean() * 100,
            "d_mean": x.mean() - base_mean, "d_hit": (x > 0).mean() * 100 - base_hit, "t": t}


def report(name, all_df, conds, split_date):
    """조건별 성적표. 탐색 구간과 검증 구간을 나란히."""
    print(f"\n### {name}")
    for h in HORIZONS:
        train = all_df[all_df["date"] < split_date]
        test = all_df[all_df["date"] >= split_date]
        bt = train[f"fwd{h}"].dropna(); btest = test[f"fwd{h}"].dropna()
        print(f"\n[{h}일 뒤]  기준선 탐색: 평균 {bt.mean():+.2f}% 상승확률 {(bt > 0).mean() * 100:.1f}% (n={len(bt):,})"
              f" | 검증: 평균 {btest.mean():+.2f}% 상승확률 {(btest > 0).mean() * 100:.1f}% (n={len(btest):,})")
        print(f"{'조건':34} {'탐색 n':>7} {'평균':>7} {'상승%':>6} {'Δ평균':>6} {'t':>5} | {'검증 n':>7} {'평균':>7} {'상승%':>6} {'Δ평균':>6} {'t':>5}")
        for label, mask in conds:
            m = mask.fillna(False).astype(bool)
            a = stats(train[m.loc[train.index]], h, bt.mean(), (bt > 0).mean() * 100)
            b = stats(test[m.loc[test.index]], h, btest.mean(), (btest > 0).mean() * 100)
            fa = f"{a['n']:>7,} {a['mean']:+6.2f}% {a['hit']:5.1f}% {a['d_mean']:+5.2f} {a['t']:5.1f}" if a else f"{'-':>34}"
            fb = f"{b['n']:>7,} {b['mean']:+6.2f}% {b['hit']:5.1f}% {b['d_mean']:+5.2f} {b['t']:5.1f}" if b else f"{'-':>34}"
            print(f"{label:34} {fa} | {fb}")


def main():
    refresh = "--refresh" in sys.argv
    if "--daily" in sys.argv:         # 계좌 단위 일별 시뮬 (지표 계산 불필요)
        daily_report(refresh)
        return
    frames = []
    for i, sym in enumerate(TICKERS):
        try:
            df = load(sym, refresh)
            if len(df) < 120:
                continue
            f = features(df)
            f["symbol"] = sym
            frames.append(f)
        except Exception as e:  # noqa: BLE001
            print(f"{sym} 실패: {e}", file=sys.stderr)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(TICKERS)} 종목 처리", file=sys.stderr)
    all_df = pd.concat(frames).dropna(subset=["score", "rsi", "bbp", "macd_hist"])
    all_df = all_df.rename_axis("date").reset_index().sort_values("date").reset_index(drop=True)
    split = all_df["date"].iloc[int(len(all_df) * 2 / 3)]
    print(f"종목 {len(frames)}개, 종목·일 표본 {len(all_df):,}개, 기간 {all_df['date'].min().date()} ~ {all_df['date'].max().date()}, 검증 구간 시작 {split.date()}")

    d = all_df
    if "--strategy" in sys.argv:      # 조건표는 건너뛰고 선별 규칙 비교만
        strategy_report(all_df, split)
        return
    if "--live-exits" in sys.argv:    # 운영 중인 강제청산 규칙 검증
        live_exits_report(all_df, split)
        return
    if "--exits" in sys.argv:         # 목표가/손절가 청산 규칙 비교만
        exits_report(all_df, split)
        return
    report("① 현재 규칙 점수의 구성 요소 (각각 단독)", d, [
        ("종가 > SMA20", d.close_gt_sma20),
        ("종가 < SMA20", ~d.close_gt_sma20),
        ("SMA20 > SMA50 (정배열)", d.sma20_gt_sma50),
        ("20일 수익률 > 0", d.ret_20d > 0),
        ("MACD 히스토그램 > 0", d.macd_hist > 0),
        ("MACD 히스토그램 < 0", d.macd_hist < 0),
        ("RSI 40~65 (규칙의 '적정')", d.rsi.between(40, 65)),
        ("RSI > 75 (규칙의 '과열')", d.rsi > 75),
        ("BB%B 0.2~0.85 (규칙의 '적정')", d.bbp.between(0.2, 0.85)),
        ("BB%B > 1 (규칙의 '과열')", d.bbp > 1),
        ("거래량 > 20일 평균 1.2배", d.vol_ratio > 1.2),
    ], split)
    report("② 규칙 점수 구간 (스크리닝이 실제로 쓰는 값)", d, [
        (f"score = {s}", d.score == s) for s in sorted(d.score.unique())
    ] + [("score >= 5 (현재 상위권)", d.score >= 5), ("score <= 1", d.score <= 1)], split)
    report("③ RSI 구간", d, [
        (f"RSI {lo}~{hi}", d.rsi.between(lo, hi)) for lo, hi in
        ((0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100))
    ], split)
    report("④ BB%B 구간", d, [
        (f"BB%B {lo}~{hi}", d.bbp.between(lo, hi)) for lo, hi in
        ((-9, 0), (0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 9))
    ], split)
    report("⑤ 20일 수익률 구간 (모멘텀 vs 과열)", d, [
        (f"20일 {lo}~{hi}%", d.ret_20d.between(lo, hi)) for lo, hi in
        ((-99, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 20), (20, 999))
    ], split)
    report("⑥ 스토캐스틱 %K 구간", d, [
        (f"%K {lo}~{hi}", d.stoch_k.between(lo, hi)) for lo, hi in
        ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100))
    ], split)
    report("⑦ 조합 (프롬프트가 '좋다'고 가르치는 상태)", d, [
        ("정배열 + MACD>0 + RSI<70 + BB%B<1", d.close_gt_sma20 & d.sma20_gt_sma50 & (d.macd_hist > 0) & (d.rsi < 70) & (d.bbp < 1)),
        ("정배열 + MACD>0 + RSI 40~60", d.close_gt_sma20 & d.sma20_gt_sma50 & (d.macd_hist > 0) & d.rsi.between(40, 60)),
        ("정배열 + BB%B 0.5~0.8", d.close_gt_sma20 & d.sma20_gt_sma50 & d.bbp.between(0.5, 0.8)),
        ("정배열 + 20일 수익률 0~10%", d.close_gt_sma20 & d.sma20_gt_sma50 & d.ret_20d.between(0, 10)),
        ("역배열 + RSI<35 (과매도 반등)", ~d.close_gt_sma20 & ~d.sma20_gt_sma50 & (d.rsi < 35)),
        ("정배열 + RSI<40 (추세 중 눌림)", d.close_gt_sma20 & d.sma20_gt_sma50 & (d.rsi < 40)),
        ("정배열 + 5일 수익률 < -3% (눌림)", d.sma20_gt_sma50 & (d.ret_5d < -3)),
    ], split)
    strategy_report(all_df, split)


def strategy_report(all_df, split_date, top=5):
    """매일 규칙대로 상위 top 종목을 뽑아 h일 보유했을 때의 평균 수익률 (선별 규칙 비교)."""
    print(f"\n### ⑧ 선별 규칙 비교 — 매일 상위 {top}종목 선택, 동일 비중")
    rules = {
        "기존 규칙 점수 (score, 동점은 20일 수익률)": lambda g: g.sort_values(["score", "ret_20d"], ascending=False),
        "20일 모멘텀": lambda g: g.sort_values("ret_20d", ascending=False),
        "20일 모멘텀 + 5일 눌림(-3%) 우선": lambda g: g.assign(_dip=(g.ret_5d < -3).astype(int))
                                              .sort_values(["_dip", "ret_20d"], ascending=False),
        "20일 모멘텀 > 20% 만 (없으면 미보유)": lambda g: g[g.ret_20d > 20].sort_values("ret_20d", ascending=False),
        "20일 모멘텀 상위 + RSI<70 (과열 제외)": lambda g: g[g.rsi < 70].sort_values("ret_20d", ascending=False),
        "20일 모멘텀 하위 (역발상)": lambda g: g.sort_values("ret_20d", ascending=True),
        "[적용안] 모멘텀 상위 + 구간 가중 사이즈": lambda g: g.sort_values("ret_20d", ascending=False)
                                              .assign(_w=lambda x: x.ret_20d.map(lambda r: momentum_tier(r)[0])),
    }
    for h in HORIZONS:
        col = f"fwd{h}"
        d = all_df.dropna(subset=[col])
        rows = []
        for name, pick in rules.items():
            out = {}
            for label, part in (("탐색", d[d["date"] < split_date]), ("검증", d[d["date"] >= split_date])):
                per_day = []
                base = []
                for _, g in part.groupby("date"):
                    sel = pick(g).head(top)
                    if "_w" in sel:            # 구간 가중: 배수 0 인 종목은 현금(수익 0)으로 둔다
                        w = sel["_w"].values
                        if len(sel):
                            per_day.append(float((sel[col].values * w).sum() / top))
                    elif len(sel):
                        per_day.append(sel[col].mean())
                    base.append(g[col].mean())
                x = pd.Series(per_day); b = pd.Series(base)
                n_eff = max(1, len(x) // h)
                diff = x.mean() - b.mean()
                t = diff / (x.std(ddof=1) / np.sqrt(n_eff)) if len(x) > 1 else 0
                out[label] = (len(x), x.mean(), (x > 0).mean() * 100, b.mean(), diff, t)
            rows.append((name, out))
        print(f"\n[{h}일 보유]  {'규칙':38} {'일수':>5} {'평균':>7} {'양(+)일%':>7} {'유니버스':>8} {'Δ':>6} {'t':>5} | {'일수':>5} {'평균':>7} {'양(+)일%':>7} {'유니버스':>8} {'Δ':>6} {'t':>5}")
        for name, out in rows:
            a, b = out["탐색"], out["검증"]
            print(f"{'':12}{name:38} {a[0]:>5} {a[1]:+6.2f}% {a[2]:6.1f}% {a[3]:+7.2f}% {a[4]:+5.2f} {a[5]:5.1f} | "
                  f"{b[0]:>5} {b[1]:+6.2f}% {b[2]:6.1f}% {b[3]:+7.2f}% {b[4]:+5.2f} {b[5]:5.1f}")


def live_exits_report(all_df, split_date, top=5, hold=60):
    """운영 중인 강제청산 규칙(손절 -15% + 20일 모멘텀 음수)을 실제로 시뮬레이션한다.
    exits_report 와 달리 종가 기준이고, 청산 사유별 건수·기여도를 같이 낸다.
    보유 상한 hold 일 = '규칙이 안 내보내면 언제까지 들고 가나' 의 상한선일 뿐."""
    print(f"\n### ⑩ 운영 청산 규칙 검증 — 모멘텀 상위 {top}종목 진입, 최대 {hold}일 보유 (종가 기준)")
    raw, mom = {}, {}
    for sym, g in all_df.groupby("symbol"):
        raw[sym] = pickle.load((CACHE / f"{sym}_1d.pkl").open("rb"))[["close", "low"]]
        mom[sym] = g.set_index("date")["ret_20d"]
    picks = (all_df.dropna(subset=["ret_20d"]).sort_values("ret_20d", ascending=False)
             .groupby("date").head(top)[["date", "symbol"]])

    rules = {
        "청산 없음 (만기 보유)":            dict(sl=None, mom=False),
        "손절 -15% 만 (현재 STOP_LOSS_PCT)": dict(sl=15, mom=False),
        "손절 -25% 만":                    dict(sl=25, mom=False),
        "모멘텀 음수 청산만 (MOMENTUM_EXIT)": dict(sl=None, mom=True),
        "손절 -15% + 모멘텀 음수 (현재 운영)": dict(sl=15, mom=True),
        "손절 -8% + 모멘텀 음수":            dict(sl=8, mom=True),
        "모멘텀 -3% 이탈 시 청산 (완충)":       dict(sl=None, mom=True, mom_th=-3),
        "모멘텀 -5% 이탈 시 청산 (완충)":       dict(sl=None, mom=True, mom_th=-5),
        "손절 -25% + 모멘텀 -5% 완충":        dict(sl=25, mom=True, mom_th=-5),
    }

    def simulate(sl, use_mom, part, mom_th=0):
        rets, days, why = [], [], {"손절": 0, "모멘텀": 0, "만기": 0}
        for date, sym in part.itertuples(index=False):
            df, ms = raw[sym], mom[sym]
            i = df.index.get_loc(date)
            if i + hold >= len(df):
                continue
            entry = df["close"].iloc[i]
            r = None
            for k in range(1, hold + 1):
                d_k = df.index[i + k]
                c = df["close"].iloc[i + k]
                if sl and c <= entry * (1 - sl / 100):
                    r, held, tag = (c / entry - 1) * 100, k, "손절"; break
                if use_mom and ms.get(d_k, 1.0) < mom_th:
                    r, held, tag = (c / entry - 1) * 100, k, "모멘텀"; break
            if r is None:
                r, held, tag = (df["close"].iloc[i + hold] / entry - 1) * 100, hold, "만기"
            why[tag] += 1
            rets.append(r); days.append(held)
        x = pd.Series(rets)
        n = max(1, len(x))
        # 연율화: 건당 평균 수익을 평균 보유일로 나눠 일수익 → 252일
        dh = pd.Series(days).mean() if days else 1
        # 왕복 비용(수수료+슬리피지)을 건당 차감한 뒤 연율화 — 회전이 빠른 규칙일수록 비용을 더 문다
        ann = [(x.mean() - c) / dh * 252 for c in (0, 0.25, 0.5)]
        return len(x), x.mean(), (x > 0).mean() * 100, dh, ann, {k: v * 100 / n for k, v in why.items()}

    for label, part in (("탐색", picks[picks["date"] < split_date]), ("검증", picks[picks["date"] >= split_date])):
        print(f"\n[{label} 구간]  {'규칙':34} {'건수':>6} {'평균':>7} {'양(+)%':>7} {'보유일':>6}"
              f" {'연율화(비용 0/0.25/0.5%)':>26}   청산사유 %(손절/모멘텀/만기)")
        for name, kw in rules.items():
            n, m, hit, dh, ann, why = simulate(kw["sl"], kw["mom"], part, kw.get("mom_th", 0))
            print(f"{'':11}{name:34} {n:>6,} {m:+6.2f}% {hit:6.1f}% {dh:6.1f}"
                  f" {ann[0]:+7.1f}%{ann[1]:+7.1f}%{ann[2]:+7.1f}%   "
                  f"{why['손절']:.0f} / {why['모멘텀']:.0f} / {why['만기']:.0f}")


def exits_report(all_df, split_date, top=5, hold=20):
    """모멘텀 상위 top 종목을 hold 일 보유하되, 목표가(+tp%)·손절가(-sl%)에 닿으면 즉시 청산.
    일봉 고가/저가로 터치를 판정한다 (같은 날 둘 다 닿으면 손절로 간주 — 보수적)."""
    print(f"\n### ⑨ 청산 규칙 비교 — 모멘텀 상위 {top}종목, 최대 {hold}일 보유, 목표가/손절가 터치 시 즉시 청산")
    raw = {}
    for sym in all_df["symbol"].unique():
        df = pickle.load((CACHE / f"{sym}_1d.pkl").open("rb"))
        raw[sym] = df[["close", "high", "low"]]
    picks = (all_df.dropna(subset=["fwd20"]).sort_values("ret_20d", ascending=False)
             .groupby("date").head(top)[["date", "symbol"]])
    grid_tp = (None, 5, 10, 15, 20, 30)
    grid_sl = (None, 5, 8, 10, 15)

    def simulate(tp, sl, part):
        rets, days_held = [], []
        for date, sym in part.itertuples(index=False):
            df = raw[sym]
            i = df.index.get_loc(date)
            if i + hold >= len(df):
                continue
            entry = df["close"].iloc[i]
            r = None
            for k in range(1, hold + 1):
                hi, lo = df["high"].iloc[i + k], df["low"].iloc[i + k]
                if sl and lo <= entry * (1 - sl / 100):
                    r, held = -sl, k; break
                if tp and hi >= entry * (1 + tp / 100):
                    r, held = tp, k; break
            if r is None:
                r, held = (df["close"].iloc[i + hold] / entry - 1) * 100, hold
            rets.append(r); days_held.append(held)
        x = pd.Series(rets)
        return len(x), x.mean(), (x > 0).mean() * 100, x.std(ddof=1), pd.Series(days_held).mean()

    for label, part in (("탐색", picks[picks["date"] < split_date]), ("검증", picks[picks["date"] >= split_date])):
        print(f"\n[{label}]  값 = 건당 평균 수익률% (양(+)비율%, 평균 보유일)   행=목표가, 열=손절가")
        head = '목표가 \ 손절가'
        print(f"{head:12}" + "".join(f"{('없음' if s is None else f'-{s}%'):>22}" for s in grid_sl))
        for tp in grid_tp:
            row = f"{('없음' if tp is None else f'+{tp}%'):12}"
            for sl in grid_sl:
                n, m, hit, sd, dh = simulate(tp, sl, part)
                row += f"{m:+6.2f} ({hit:4.1f}%, {dh:4.1f}일)".rjust(22)
            print(row)
        n, m, hit, sd, dh = simulate(None, None, part)
        print(f"   (기준: 청산 규칙 없음 {n:,}건, 평균 {m:+.2f}%, 표준편차 {sd:.1f}%)")


def daily_panel(refresh=False):
    """종가·20일 수익률 패널 (날짜 × 종목). 지표는 필요 없어 features() 를 건너뛴다."""
    closes = {}
    for sym in TICKERS:
        try:
            df = load(sym, refresh)
            if len(df) >= 120:
                closes[sym] = df["close"]
        except Exception as e:  # noqa: BLE001
            print(f"{sym} 실패: {e}", file=sys.stderr)
    close = pd.DataFrame(closes).sort_index()
    return close, close.pct_change(20) * 100


def daily_sim(close, ret20, cfg, cash0=10000.0):
    """계좌 단위 일별 시뮬. 매일 종가에 청산 → 진입 순으로 처리한다.
    운영 코드와 같은 제약을 건다: 최대 종목 수, 종목당 비중 한도, 현금 유지 비중, 최소 주문 금액,
    왕복 비용. 소수점 주문 불가(frac=False)면 정수 주만 산다 — 소액 계좌에서 이게 제일 크게 문다."""
    top = cfg.get("top", 5)
    max_pos_pct = cfg.get("max_pos_pct", 30)
    reserve = cfg.get("reserve_pct", 10)
    min_usd = cfg.get("min_usd", 5)
    cost = cfg.get("cost_pct", 0.25) / 100        # 편도 (수수료+슬리피지)
    sl = cfg.get("sl")                            # 평단 대비 -sl% 면 청산
    mom_exit = cfg.get("mom_exit", True)
    mom_th = cfg.get("mom_th", 0)                 # 20일 수익률이 이 값 미만이면 청산
    entry_th = cfg.get("entry_th", 0)             # 20일 수익률이 이 값 초과일 때만 진입
    rotate_guard = cfg.get("rotate_guard", False)  # 갈아탈 후보 없으면 모멘텀 청산 보류
    frac = cfg.get("frac", True)
    rebal = cfg.get("rebal", 1)                   # 며칠에 한 번 신규 진입을 볼지

    cash, pos = cash0, {}                         # pos: sym -> [수량, 평단]
    equity, trades, holds = [], [], []
    dates = close.index[20:]
    for n, t in enumerate(dates):
        px, mo = close.loc[t], ret20.loc[t]
        cands = [s for s in mo.dropna().sort_values(ascending=False).index
                 if mo[s] > entry_th and s not in pos and pd.notna(px.get(s))]
        for sym in list(pos):
            p = px.get(sym)
            if pd.isna(p):
                continue
            qty, avg, since = pos[sym]
            why = None
            if sl and p <= avg * (1 - sl / 100):
                why = "손절"
            elif mom_exit and pd.notna(mo.get(sym)) and mo[sym] < mom_th:
                why = None if (rotate_guard and not cands) else "모멘텀"
            if why:
                cash += qty * p * (1 - cost)
                trades.append((why, (p / avg - 1) * 100)); holds.append(n - since)
                del pos[sym]
        if n % rebal:
            equity.append(cash + sum(q * px.get(s, a) for s, (q, a, _) in pos.items()))
            continue
        for sym in cands:
            if len(pos) >= top:
                break
            total = cash + sum(q * px.get(s, a) for s, (q, a, _) in pos.items())
            budget = min(total * max_pos_pct / 100, cash - total * reserve / 100)
            p = px[sym]
            qty = budget / p / (1 + cost) if frac else float(int(budget / p / (1 + cost)))
            if qty <= 0 or qty * p < min_usd:
                continue
            cash -= qty * p * (1 + cost)
            pos[sym] = [qty, p, n]
        equity.append(cash + sum(q * px.get(s, a) for s, (q, a, _) in pos.items()))
    eq = pd.Series(equity, index=dates)
    r = eq.pct_change().dropna()
    years = len(eq) / 252
    cagr = (eq.iloc[-1] / cash0) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    kinds = pd.Series([w for w, _ in trades])
    return {"eq": eq, "총수익": eq.iloc[-1] / cash0 - 1, "CAGR": cagr, "MDD": mdd,
            "Sharpe": sharpe, "거래": len(trades),
            "평균보유일": float(np.mean(holds)) if holds else 0,
            "손절%": (kinds == "손절").mean() * 100 if len(kinds) else 0,
            "승률": float(np.mean([p > 0 for _, p in trades]) * 100) if trades else 0}


def daily_report(refresh=False):
    """규칙 조합별 계좌 단위 성적. 탐색 / 검증 구간을 나눠 같은 표로 낸다."""
    close, ret20 = daily_panel(refresh)
    split = close.index[int(len(close) * 2 / 3)]
    print(f"종목 {close.shape[1]}개, 기간 {close.index[0].date()} ~ {close.index[-1].date()}, "
          f"검증 구간 시작 {split.date()}")
    print("### ⑪ 계좌 단위 일별 시뮬 — 초기 $10,000, 편도 비용 0.25%, 최대 5종목·종목당 30%·현금 10% 유지")

    rules = {
        "① 현재 운영 (손절15+모멘텀0)":      dict(sl=15, mom_exit=True),
        "② 손절 25 + 모멘텀 0":            dict(sl=25, mom_exit=True),
        "③ 손절 없음 + 모멘텀 0":           dict(sl=None, mom_exit=True),
        "④ 손절 25 + 모멘텀 0 + 회전가드":    dict(sl=25, mom_exit=True, rotate_guard=True),
        "⑤ 손절 25 + 모멘텀 -5 완충":       dict(sl=25, mom_exit=True, mom_th=-5),
        "⑥ 손절 25 + 청산 없음(모멘텀 무시)":  dict(sl=25, mom_exit=False),
        "⑦ ④ + 진입 문턱 20일>10%":        dict(sl=25, mom_exit=True, rotate_guard=True, entry_th=10),
        "⑧ ④ + 주 1회 진입(rebal 5)":      dict(sl=25, mom_exit=True, rotate_guard=True, rebal=5),
        "⑨ ④ 를 소수점 주문 없이 (정수 주)":   dict(sl=25, mom_exit=True, rotate_guard=True, frac=False),
    }
    for label, sub in (("탐색", close.loc[:split]), ("검증", close.loc[split:])):
        r20 = ret20.loc[sub.index]
        print(f"\n[{label} 구간 {sub.index[0].date()}~{sub.index[-1].date()}]"
              f"  {'규칙':32} {'총수익':>8} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'거래':>5} {'승률':>6} {'보유일':>6} {'손절%':>6}")
        bh = (sub.iloc[-1] / sub.iloc[0] - 1).mean() * 100          # 유니버스 동일비중 보유
        for name, cfg in rules.items():
            m = daily_sim(sub, r20, cfg)
            print(f"{'':11}{name:32} {m['총수익'] * 100:+7.1f}% {m['CAGR'] * 100:+7.1f}% "
                  f"{m['MDD'] * 100:+7.1f}% {m['Sharpe']:7.2f} {m['거래']:5d} {m['승률']:5.1f}% "
                  f"{m['평균보유일']:6.1f} {m['손절%']:5.1f}%")
        print(f"{'':11}{'(기준: 나스닥100 동일비중 보유)':32} {bh:+7.1f}%")

    # 한 구간의 성적은 운일 수 있다. 6개월 창을 한 달씩 밀며 규칙끼리 직접 붙인다.
    print("\n[6개월 롤링 창 — 창별 수익률 중앙값과 기준선(동일비중) 대비 승률]")
    windows = [close.index[i:i + 126] for i in range(0, len(close) - 126, 21)]
    for name in ("① 현재 운영 (손절15+모멘텀0)", "② 손절 25 + 모멘텀 0",
                 "④ 손절 25 + 모멘텀 0 + 회전가드", "⑧ ④ + 주 1회 진입(rebal 5)"):
        rets, wins = [], 0
        for w in windows:
            sub = close.loc[w]
            m = daily_sim(sub, ret20.loc[w], rules[name])
            base = (sub.iloc[-1] / sub.iloc[0] - 1).mean()
            rets.append(m["총수익"]); wins += m["총수익"] > base
        print(f"{'':11}{name:32} 창 {len(windows)}개, 중앙값 {np.median(rets) * 100:+6.1f}%, "
              f"기준선 대비 승률 {wins / len(windows) * 100:4.0f}%, 최악 {min(rets) * 100:+6.1f}%")

    # 소액 계좌: 정수 주 제약 + 최소 주문 금액이 실제로 얼마나 무는지
    print("\n[계좌 규모별 — ④ 규칙, 소수점 주문 불가(정규장 외) 가정]")
    for cash0 in (500, 2000, 10000, 50000):
        m = daily_sim(close, ret20, dict(rules["④ 손절 25 + 모멘텀 0 + 회전가드"], frac=False), cash0)
        print(f"{'':11}초기 ${cash0:>6,}  총수익 {m['총수익'] * 100:+7.1f}%  거래 {m['거래']:3d}  "
              f"MDD {m['MDD'] * 100:+6.1f}%")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
