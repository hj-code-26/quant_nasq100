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
        print(f"{'목표가\\손절가':12}" + "".join(f"{('없음' if s is None else f'-{s}%'):>22}" for s in grid_sl))
        for tp in grid_tp:
            row = f"{('없음' if tp is None else f'+{tp}%'):12}"
            for sl in grid_sl:
                n, m, hit, sd, dh = simulate(tp, sl, part)
                row += f"{m:+6.2f} ({hit:4.1f}%, {dh:4.1f}일)".rjust(22)
            print(row)
        n, m, hit, sd, dh = simulate(None, None, part)
        print(f"   (기준: 청산 규칙 없음 {n:,}건, 평균 {m:+.2f}%, 표준편차 {sd:.1f}%)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
