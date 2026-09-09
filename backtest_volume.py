"""거래량 기반 국면 분류 → 국면별 하락방어선 + 모멘텀 미분(기울기) 분석.

backtest_regimes.py 는 지수 수익률만으로 국면을 나눴다. 여기서는 **거래량**을 축으로 넣는다.
"거래량이 실린 하락"과 "거래량이 마른 하락"은 다른 국면이라는 전제를 실제로 검증한다.

  ① 국면 분류    : 시장 달러거래량 팽창(高/低) × 지수 방향(상승/횡보/하락) = 6칸
  ② 하락방어선   : 국면별로 손절선을 -3% ~ -20% 로 훑어 어디가 방어선인지 찾는다
  ③ 모멘텀 미분  : ret_20d 의 1차 미분(기울기)·2차 미분(가속도) 별 이후 20일 수익률
  ④ 격자         : 모멘텀 수준 × 기울기 조합에서 무엇이 오르고 무엇이 떨어지는지

사용:
  python backtest_volume.py                (전부)
  python backtest_volume.py --defense      (② 만)
  python backtest_volume.py --momentum     (③④ 만)

정의
  · 시장 거래량비 vr = 유니버스 합산 달러거래량의 20일 평균 / 250일 평균
    (종목 수 변화·장기 성장이 상쇄되도록 비율로 쓴다. 그 날까지의 데이터만 사용)
  · 방향 dir60 = 지수 60일 수익률.  > +3% 상승 / < -3% 하락 / 그 사이 횡보
  · 모멘텀 mom = ret_20d,  기울기 slope = (mom - mom 5일전) / 5   [%p/일]
    가속도 accel = (slope - slope 5일전) / 5                      [%p/일²]
  · 손절 체결가는 갭을 반영한다: 시가가 이미 손절가 아래면 시가로, 아니면 손절가로 체결.

★ 생존 편향: 유니버스가 오늘의 나스닥 100 이다 (1998년 56종목 → 2026년 102종목).
  과거 구간 절대 수익률은 부풀려져 있다. 규칙·구간 간 상대 비교로만 읽을 것.
"""
import sys

import numpy as np
import pandas as pd

from market_data import INDEX, load_ohlcv

TOP = 5                 # 매일 진입할 종목 수 (운영 MAX_POSITIONS 와 동일)
HOLD = 20               # 기본 보유일 (백테스트가 검증한 고정 20일 청산)
STOPS = (None, 3, 5, 8, 10, 12, 15, 20)
VR_HI = 1.05            # 거래량비 중앙값 — 이 위를 '거래량 실림'으로 본다
DIR_BAND = 3.0          # 지수 60일 수익률 ±3% 를 횡보로 본다
SLOPE_BINS = [-99, -2.0, -0.5, 0.5, 2.0, 99]
SLOPE_NAMES = ["급락(< -2)", "하락(-2~-0.5)", "평탄(±0.5)", "상승(0.5~2)", "급등(> 2)"]
MOM_BINS = [-999, 0, 10, 20, 999]
MOM_NAMES = ["음수", "0~10%", "10~20%", "> 20%"]


def prepare(refresh=False):
    d = load_ohlcv(refresh)
    idx = d["close"][INDEX].dropna()
    px = {k: v.drop(columns=[INDEX]) for k, v in d.items()}
    close = px["close"]

    # --- 시장 거래량비 (그 날까지의 데이터만) ---
    dollar_vol = (close * px["volume"]).sum(axis=1, min_count=1)
    vr = dollar_vol.rolling(20).mean() / dollar_vol.rolling(250).mean()

    # --- 지수 방향 ---
    dir60 = (idx / idx.shift(60) - 1) * 100

    lab = pd.DataFrame({"vr": vr, "dir60": dir60}).dropna()
    lab["방향"] = np.where(lab.dir60 > DIR_BAND, "상승",
                         np.where(lab.dir60 < -DIR_BAND, "하락", "횡보"))
    lab["거래량"] = np.where(lab.vr > VR_HI, "실림", "마름")
    lab["국면"] = lab["거래량"] + "·" + lab["방향"]

    # --- 종목별 모멘텀과 그 미분 ---
    mom = (close / close.shift(20) - 1) * 100
    slope = (mom - mom.shift(5)) / 5
    accel = (slope - slope.shift(5)) / 5
    fwd20 = (close.shift(-20) / close - 1) * 100
    return px, idx, lab, mom, slope, accel, fwd20


def make_picks(mom, lab):
    """매일 그 날 데이터가 있는 종목 중 mom 상위 TOP 개 (미래 참조 없음)."""
    m = mom.loc[mom.index.isin(lab.index)]
    rank = m.rank(axis=1, ascending=False, method="first")
    sel = (rank <= TOP).stack()                 # pandas 3.0: stack() 은 NaN 을 안 버린다
    picks = sel[sel].index.to_frame(index=False)
    picks.columns = ["date", "symbol"]
    days = m.notna().any(axis=1).sum()
    assert len(picks) <= days * TOP, f"진입 {len(picks):,}건 > {days:,}일 × {TOP}"
    return picks


def build_series(px):
    """{symbol: (open, close, low, {날짜: 인덱스})} — 손절 시뮬레이션용 배열."""
    out = {}
    for sym in px["close"].columns:
        s = px["close"][sym].dropna()
        if len(s) < 60:
            continue
        out[sym] = (px["open"][sym].reindex(s.index).to_numpy(float),
                    s.to_numpy(float),
                    px["low"][sym].reindex(s.index).to_numpy(float),
                    {d: i for i, d in enumerate(s.index)})
    return out


# ---------- ② 하락방어선 ----------
def sweep_stops(series, picks, hold=HOLD, stops=STOPS):
    """고정 hold 일 보유 + 손절선. 갭 반영: 시가가 손절가 아래면 시가 체결."""
    acc = {sl: ([], []) for sl in stops}
    for date, sym in picks.itertuples(index=False):
        op, cl, lo, idx = series[sym]
        i = idx.get(date)
        if i is None or i + hold >= len(cl):
            continue
        entry = cl[i]
        p_lo, p_op, p_cl = lo[i + 1:i + hold + 1], op[i + 1:i + hold + 1], cl[i + 1:i + hold + 1]
        base = (p_cl[-1] / entry - 1) * 100
        for sl in stops:
            r, d = acc[sl]
            if sl is None:
                r.append(base); d.append(hold); continue
            thr = entry * (1 - sl / 100)
            hit = np.flatnonzero(p_lo <= thr)
            if len(hit):
                k = hit[0]
                fill = min(thr, p_op[k]) if not np.isnan(p_op[k]) else thr   # 갭이면 시가 체결
                r.append((fill / entry - 1) * 100); d.append(k + 1)
            else:
                r.append(base); d.append(hold)
    rows = []
    for sl in stops:
        x, dd = pd.Series(acc[sl][0]), pd.Series(acc[sl][1])
        if not len(x):
            continue
        rows.append((sl, {"n": len(x), "mean": x.mean(), "hit": (x > 0).mean() * 100,
                          "days": dd.mean(), "bp": x.mean() / dd.mean() * 100,
                          "worst": x.min(), "p5": x.quantile(0.05),
                          "cut": (dd < hold).mean() * 100}))
    return rows


def defense_report(series, picks, lab):
    print(f"\n### ② 국면별 하락방어선 — 모멘텀 상위 {TOP} 진입, 고정 {HOLD}일 보유 + 손절선")
    reg = lab["국면"].reindex(picks["date"]).to_numpy()
    dirn = lab["방향"].reindex(picks["date"]).to_numpy()
    groups = [("전체", slice(None))]
    groups += [(f"{k}장", dirn == k) for k in ("상승", "횡보", "하락")]
    groups += [(k, reg == k) for k in
               ("실림·상승", "마름·상승", "실림·횡보", "마름·횡보", "실림·하락", "마름·하락")]
    for name, m in groups:
        part = picks if isinstance(m, slice) else picks[m]
        rows = sweep_stops(series, part)
        if not rows:
            print(f"\n[{name}] 표본 없음"); continue
        base_bp = rows[0][1]["bp"]
        print(f"\n[{name}]  진입 {rows[0][1]['n']:,}건")
        print(f"{'손절선':>8}{'건당':>8}{'승률':>7}{'보유일':>7}{'일당bp':>8}"
              f"{'Δbp':>7}{'하위5%':>8}{'최악건':>8}{'손절발동':>9}")
        for sl, s in rows:
            tag = "없음" if sl is None else f"-{sl}%"
            print(f"{tag:>8}{s['mean']:+7.2f}%{s['hit']:6.1f}%{s['days']:7.1f}{s['bp']:8.1f}"
                  f"{s['bp'] - base_bp:+7.1f}{s['p5']:+7.1f}%{s['worst']:+7.1f}%{s['cut']:8.1f}%")


# ---------- ③④ 모멘텀 미분 ----------
def stack_frame(lab, mom, slope, accel, fwd20):
    """전 종목·전 거래일을 한 줄씩 편 표 (국면 라벨 붙임)."""
    dates = lab.index
    def s(df, name):
        return df.loc[df.index.isin(dates)].stack().rename(name)
    d = pd.concat([s(mom, "mom"), s(slope, "slope"), s(accel, "accel"),
                   s(fwd20, "fwd20")], axis=1).dropna()
    d = d.reset_index(names=["date", "symbol"])
    d["방향"] = lab["방향"].reindex(d["date"]).to_numpy()
    d["거래량"] = lab["거래량"].reindex(d["date"]).to_numpy()
    d["기울기"] = pd.cut(d.slope, SLOPE_BINS, labels=SLOPE_NAMES)
    d["모멘텀"] = pd.cut(d.mom, MOM_BINS, labels=MOM_NAMES)
    return d


def momentum_report(d):
    print("\n### ③ 모멘텀 기울기(1차 미분)별 이후 20일 수익률 — 전 종목·전 거래일")
    for kind in ("상승", "횡보", "하락"):
        part = d[d["방향"] == kind]
        base = part.fwd20.mean()
        print(f"\n[{kind}장]  표본 {len(part):,}  기준선(이 국면 전체 평균) {base:+.2f}%")
        print(f"{'모멘텀 기울기':18}{'표본':>10}{'평균':>8}{'상승%':>7}{'Δ기준선':>9}"
              f"{'거래량실림':>10}{'거래량마름':>10}")
        for name in SLOPE_NAMES:
            g = part[part["기울기"] == name]
            if len(g) < 200:
                continue
            hi = g[g["거래량"] == "실림"].fwd20.mean()
            lo = g[g["거래량"] == "마름"].fwd20.mean()
            print(f"{name:18}{len(g):>10,}{g.fwd20.mean():+7.2f}%{(g.fwd20 > 0).mean() * 100:6.1f}%"
                  f"{g.fwd20.mean() - base:+8.2f}{hi:+9.2f}%{lo:+9.2f}%")

    print("\n### ④ 모멘텀 수준 × 기울기 격자 — 값 = 이후 20일 평균 수익률 (상승확률%)")
    for kind in ("상승", "횡보", "하락"):
        part = d[d["방향"] == kind]
        print(f"\n[{kind}장]")
        print(f"{'모멘텀/기울기':16}" + "".join(f"{n:>20}" for n in SLOPE_NAMES))
        for mname in MOM_NAMES:
            row = f"{mname:16}"
            for sname in SLOPE_NAMES:
                g = part[(part["모멘텀"] == mname) & (part["기울기"] == sname)]
                row += (f"{g.fwd20.mean():+6.2f}% ({(g.fwd20 > 0).mean() * 100:4.1f}%)".rjust(20)
                        if len(g) >= 200 else f"{'-':>20}")
            print(row)

    print("\n### ④-2 가속도(2차 미분) — 기울기가 음수인 구간에서 가속도 부호가 갈리는가")
    neg = d[d.slope < 0]
    print(f"{'국면':10}{'가속도<0 (더 꺾임)':>24}{'가속도>0 (감속 중)':>24}")
    for kind in ("상승", "횡보", "하락"):
        p = neg[neg["방향"] == kind]
        a, b = p[p.accel < 0], p[p.accel > 0]
        print(f"{kind + '장':10}{a.fwd20.mean():+9.2f}% (n={len(a):>9,}){b.fwd20.mean():+9.2f}% (n={len(b):>9,})")


def main():
    px, idx, lab, mom, slope, accel, fwd20 = prepare("--refresh" in sys.argv)
    print(f"기간 {lab.index.min().date()} ~ {lab.index.max().date()}, {len(lab):,}거래일")
    print("★ 유니버스가 '오늘의' 나스닥 100 (1998년 56종목 → 2026년 102종목). "
          "절대 수익률은 생존 편향으로 부풀려져 있다. 상대 비교로만 읽을 것.")

    print(f"\n### ① 국면 분류 — 거래량비(20일/250일) > {VR_HI} '실림' / 지수 60일 수익률 ±{DIR_BAND:.0f}%")
    ct = pd.crosstab(lab["거래량"], lab["방향"]).reindex(index=["실림", "마름"],
                                                     columns=["상승", "횡보", "하락"])
    print(ct.to_string())
    print(f"\n{'국면':12}{'거래일':>8}{'비중':>7}{'지수 60일 수익률 평균':>22}{'거래량비 평균':>15}")
    for k, g in lab.groupby("국면"):
        print(f"{k:12}{len(g):>8,}{len(g) / len(lab) * 100:6.1f}%"
              f"{g.dir60.mean():>21.1f}%{g.vr.mean():>14.2f}")

    do_def = "--momentum" not in sys.argv
    do_mom = "--defense" not in sys.argv
    if do_def:
        series = build_series(px)
        picks = make_picks(mom, lab)
        print(f"\n진입 표본 {len(picks):,}건 (매일 모멘텀 상위 {TOP})")
        defense_report(series, picks, lab)
    if do_mom:
        momentum_report(stack_frame(lab, mom, slope, accel, fwd20))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
