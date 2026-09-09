"""청산 규칙을 **시장 국면별로** 실측한다 (하락장·횡보장 포함).

backtest.py 는 토스 일봉 750개(2023-11~)만 쓴다. 그 구간은 거의 전부 상승장이라
"운영 규칙 vs 개선안" 비교가 상승장 한 국면에서만 검증된 셈이다. 여기서는 yfinance 로
1998년까지 끌어와 닷컴 붕괴·금융위기·2022 하락장·횡보 구간에서 같은 비교를 다시 한다.

  python backtest_regimes.py              (첫 실행은 다운로드, 이후 캐시)
  python backtest_regimes.py --refresh    (캐시 무시하고 다시 받음)
  python backtest_regimes.py --periods    (기계적 국면 대신 이름 붙은 위기 구간별로)

★ 생존 편향 (반드시 읽을 것)
  종목 유니버스가 **오늘의** 나스닥 100 이다. 2000년에 상장폐지·피인수된 회사는 아예 없고,
  살아남아 오늘까지 큰 회사만 남아 있다. 따라서 과거 구간의 **절대 수익률은 심하게 부풀려져 있다**.
  다만 모든 청산 규칙이 같은 유니버스·같은 진입을 공유하므로, **규칙 간 상대 비교**는
  절대 수준보다 훨씬 믿을 만하다. 이 파일은 그 상대 비교를 위한 것이다.

  국면 분류도 사후적(hindsight)이다. "국면을 미리 맞힐 수 있다"는 뜻이 아니라
  "그 국면에 있었을 때 규칙이 어떻게 굴러갔나"를 귀속시키는 용도다.
"""
import pathlib
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

from backtest import simulate_exits
from nasdaq100 import TICKERS

warnings.filterwarnings("ignore")

CACHE = pathlib.Path(__file__).with_name("data_cache") / "yf_long.pkl"
START = "1998-01-01"
INDEX = "^NDX"          # 나스닥 100 지수 — 국면 분류 기준
TOP = 5                 # 매일 진입할 종목 수 (운영 MAX_POSITIONS 와 동일)

# 비교할 청산 규칙 — backtest.py ⑩ 과 같은 정의
RULES = [
    ("고정 20일 청산 (⑧ 이 검증한 것)", dict(hold=20)),
    ("[운영] ret_20d < 0 이면 매도", dict(mom_below=0)),
    ("[개선안] ret_20d < 5% 이면 매도", dict(mom_below=5)),
    ("[개선안] ret_20d < 10% 이면 매도", dict(mom_below=10)),
    ("[운영] + -15% 손절", dict(mom_below=0, sl=15)),
    ("[개선안] < 5% + -15% 손절", dict(mom_below=5, sl=15)),
]

# 이름 붙은 구간 (--periods). 사후적으로 널리 통용되는 국면 구분.
PERIODS = [
    ("닷컴 붕괴 (NDX -83%)",      "2000-03-27", "2002-10-09", "하락"),
    ("회복·상승",                 "2002-10-10", "2007-10-31", "상승"),
    ("금융위기 (NDX -54%)",       "2007-11-01", "2009-03-09", "하락"),
    ("QE 상승장",                 "2009-03-10", "2015-06-30", "상승"),
    ("2015~16 조정·횡보",         "2015-07-01", "2016-06-30", "횡보"),
    ("2016~18 상승",              "2016-07-01", "2018-09-30", "상승"),
    ("2018 4분기 급락",           "2018-10-01", "2018-12-24", "하락"),
    ("2019~20 상승",              "2018-12-26", "2020-02-19", "상승"),
    ("코로나 크래시",             "2020-02-20", "2020-03-23", "하락"),
    ("코로나 이후 상승",          "2020-03-24", "2021-11-19", "상승"),
    ("2022 하락장 (NDX -35%)",    "2021-11-22", "2022-12-28", "하락"),
    ("2023~ 현재 상승장",         "2023-01-03", "2026-09-08", "상승"),
]


def load(refresh=False):
    """일봉 종가 (배당·분할 조정). 컬럼 = 종목, 인덱스 = 날짜."""
    if CACHE.exists() and not refresh:
        return pickle.load(CACHE.open("rb"))
    import yfinance as yf
    syms = list(dict.fromkeys(list(TICKERS) + [INDEX]))
    print(f"yfinance 에서 {len(syms)}종목 {START}~ 일봉 다운로드…", file=sys.stderr)
    raw = yf.download(syms, start=START, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"].dropna(how="all")
    CACHE.parent.mkdir(exist_ok=True)
    pickle.dump(close, CACHE.open("wb"))
    return close


def build(close):
    """종목별 (종가, ret_20d, 날짜→인덱스) 와, 매일의 모멘텀 상위 TOP 진입 목록."""
    px = close.drop(columns=[INDEX])
    ret20 = (px / px.shift(20) - 1) * 100

    series, picks = {}, []
    for sym in px.columns:
        s = px[sym].dropna()
        if len(s) < 60:
            continue
        c = s.to_numpy(float)
        r = np.full(len(c), np.nan)
        r[20:] = (c[20:] / c[:-20] - 1) * 100
        series[sym] = (c, r, {d: i for i, d in enumerate(s.index)})

    # 매일 그 날 데이터가 있는 종목 중 ret_20d 상위 TOP 개를 진입 (미래 참조 없음)
    r = ret20[list(series)]
    rank = r.rank(axis=1, ascending=False, method="first")
    # pandas 3.0 의 stack() 은 NaN 을 버리지 않는다. 불리언으로 만들어 True 만 남긴다.
    sel = (rank <= TOP).stack()
    picks = sel[sel].index.to_frame(index=False)
    picks.columns = ["date", "symbol"]
    days = r.notna().any(axis=1).sum()
    assert len(picks) <= days * TOP, f"진입 {len(picks):,}건 > {days:,}일 × {TOP}"
    return series, picks


def regimes(close):
    """지수의 직전 1년(252거래일) 수익률로 국면을 나눈다. 그 날까지의 데이터만 쓴다."""
    idx = close[INDEX].dropna()
    r252 = (idx / idx.shift(252) - 1) * 100
    dd = (idx / idx.cummax() - 1) * 100
    lab = pd.Series("횡보", index=idx.index)
    lab[r252 > 10] = "상승"
    lab[r252 < -10] = "하락"
    lab[r252.isna()] = None
    return lab, r252, dd


def table(series, picks, label):
    rows = []
    for name, kw in RULES:
        s = simulate_exits(series, picks, **kw)
        if s:
            rows.append((name, s))
    if not rows:
        print(f"  {label}: 표본 없음")
        return
    print(f"\n[{label}]  진입 {rows[0][1]['n']:,}건")
    print(f"{'청산 규칙':34}{'건당':>8}{'승률':>7}{'보유일':>7}{'일당bp':>8}{'최악건':>8}{'-10%이하':>9}")
    for name, s in rows:
        print(f"{name:34}{s['mean']:+7.2f}%{s['hit']:6.1f}%{s['days']:7.1f}"
              f"{s['bp']:8.1f}{s['worst']:+7.1f}%{s['big_loss']:8.1f}%")


def main():
    close = load("--refresh" in sys.argv)
    series, picks = build(close)
    lab, r252, dd = regimes(close)
    picks = picks[picks["date"].isin(lab.dropna().index)]
    print(f"종목 {len(series)}개, 기간 {close.index.min().date()} ~ {close.index.max().date()}, "
          f"진입 표본 {len(picks):,}건 (매일 모멘텀 상위 {TOP})")
    print("★ 유니버스가 '오늘의' 나스닥 100 이라 과거 구간 절대 수익률은 생존 편향으로 부풀려져 있다. "
          "규칙 간 상대 비교로만 읽을 것.")

    if "--periods" in sys.argv:
        print("\n### 이름 붙은 구간별")
        for name, a, b, kind in PERIODS:
            m = (picks["date"] >= a) & (picks["date"] <= b)
            table(series, picks[m], f"{name} [{kind}] {a[:7]}~{b[:7]}")
        return

    print("\n### 국면별 (지수 직전 252일 수익률: >+10% 상승 / <-10% 하락 / 그 외 횡보)")
    d = lab.reindex(picks["date"]).to_numpy()
    for kind in ("상승", "횡보", "하락"):
        days = lab[lab == kind]
        span = f"{len(days):,}거래일, 지수 평균 낙폭 {dd.reindex(days.index).mean():.1f}%"
        table(series, picks[d == kind], f"{kind}장 — {span}")
    table(series, picks, "전체 (1999~2026)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
