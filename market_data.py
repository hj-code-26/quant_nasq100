"""장기 일봉 데이터 로더 (yfinance) — backtest_regimes.py / backtest_volume.py 공용.

토스 API 는 750봉(약 3년)만 준다. 하락장·횡보장을 보려면 더 긴 이력이 필요해서
yfinance 로 1998년부터 받아 캐시한다. 배당·분할 조정(auto_adjust) 종가 기준.

★ 유니버스가 **오늘의** 나스닥 100 이라 생존 편향이 있다. 과거 구간의 절대 수익률은
  부풀려져 있고, 특히 하락장이 심하다 (망한 회사가 표본에 없다).
  규칙 간 **상대 비교**로만 읽을 것.
"""
import pathlib
import pickle
import sys
import warnings

from nasdaq100 import TICKERS

warnings.filterwarnings("ignore")

CACHE = pathlib.Path(__file__).with_name("data_cache") / "yf_ohlcv.pkl"
START = "1998-01-01"
INDEX = "^NDX"          # 나스닥 100 지수 — 방향 분류 기준 (지수 자체는 거래량이 없다)
FIELDS = ("Open", "Close", "High", "Low", "Volume")


def load_ohlcv(refresh=False):
    """{'close','high','low','volume': DataFrame} — 인덱스=날짜, 컬럼=종목(+지수는 close 에만)."""
    if CACHE.exists() and not refresh:
        return pickle.load(CACHE.open("rb"))
    import yfinance as yf
    syms = list(dict.fromkeys(list(TICKERS) + [INDEX]))
    print(f"yfinance 에서 {len(syms)}종목 {START}~ 일봉(OHLCV) 다운로드…", file=sys.stderr)
    raw = yf.download(syms, start=START, auto_adjust=True, progress=False, threads=True)
    out = {f.lower(): raw[f].dropna(how="all") for f in FIELDS}
    CACHE.parent.mkdir(exist_ok=True)
    pickle.dump(out, CACHE.open("wb"))
    return out
