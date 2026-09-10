"""§1 QQQ 국면지표 이력 확보 — 권한·라이선스 범위 확인 후 수집.

확인한 것
  · 유료 구매 없음. 운영 토큰 신규 발급 없음. 토스 API 호출 없음(격리 하네스가 차단).
  · yfinance 는 **이미 이 저장소의 의존성**이다(backtest_regimes.py 가 ^NDX 를 이걸로 받는다).
    같은 경로를 재사용하는 것이므로 새 데이터 계약이 아니다.
  · 출처: Yahoo Finance 무료 일봉. 라이선스는 Yahoo 의 개인·비상업 이용 조건이며,
    재배포용이 아니다. 캐시는 research/data/ 안에만 둔다(운영 data_cache 를 건드리지 않는다).

실행: python research/fetch_index.py
"""
import hashlib
import json
import pathlib
import pickle
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from research.isolation import guard   # noqa: E402

guard()                                # 토스·비밀값 차단, yahoo 만 허용

import pandas as pd                    # noqa: E402

DATA = pathlib.Path(__file__).with_name("data")
DATA.mkdir(exist_ok=True)
PKL = DATA / "QQQ_1d.pkl"
META = DATA / "QQQ_1d.meta.json"
SYMBOL = "QQQ"
START, END = "2023-06-01", "2026-09-10"


def fetch():
    import yfinance as yf
    df = yf.download(SYMBOL, start=START, end=END, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.dropna(subset=["close"])


def load(refresh=False):
    """캐시 우선. 없으면 받는다."""
    if PKL.exists() and not refresh:
        with open(PKL, "rb") as f:
            return pickle.load(f)
    df = fetch()
    with open(PKL, "wb") as f:
        pickle.dump(df, f)
    with open(META, "w", encoding="utf-8") as f:
        json.dump({
            "symbol": SYMBOL, "source": "Yahoo Finance via yfinance (무료·비상업 조건)",
            "license": "재배포 금지. research/data 내부 캐시 전용",
            "paid": False, "broker_api_used": False, "new_token_issued": False,
            "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
            "rows": int(len(df)),
            "sha256": hashlib.sha256(df.to_csv().encode()).hexdigest(),
        }, f, ensure_ascii=False, indent=2)
    return df


def aligned_ret60(panel_index, refresh=False):
    """패널 거래일에 맞춘 QQQ 60거래일 수익률(%). 결측일은 직전 값으로 채우지 않는다."""
    df = load(refresh)
    r = (df["close"] / df["close"].shift(60) - 1) * 100
    key = pd.Index(pd.to_datetime(panel_index).tz_localize(None).normalize())
    return pd.Series(r.reindex(key).values, index=panel_index, name="qqq_ret60")


def main():
    df = load()
    with open(META, encoding="utf-8") as f:
        meta = json.load(f)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    import backtest as bt
    close, _, _ = bt.daily_panel()
    s = aligned_ret60(close.index)
    print(f"\n패널 {len(close)}일 중 QQQ 60일수익률 결측 {int(s.isna().sum())}일 "
          f"(앞 60일 워밍업 포함)")
    print(f"마지막 값 {s.dropna().iloc[-1]:+.2f}% @ {s.dropna().index[-1].date()}")


if __name__ == "__main__":
    main()
