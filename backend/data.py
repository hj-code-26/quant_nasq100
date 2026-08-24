"""주가 데이터 수집 (yfinance) — 일봉 이력 + 실시간 호가."""
import threading
import time

import pandas as pd
import yfinance as yf

_lock = threading.Lock()
_hist_cache: dict = {}   # ticker -> (timestamp, df)
_HIST_TTL = 300          # 5분


def history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """일봉 OHLCV. 5분 캐시."""
    key = (ticker.upper(), period)
    with _lock:
        hit = _hist_cache.get(key)
        if hit and time.time() - hit[0] < _HIST_TTL:
            return hit[1]
    df = yf.download(ticker, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"데이터를 가져올 수 없습니다: {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    with _lock:
        _hist_cache[key] = (time.time(), df)
    return df


def quote(ticker: str) -> dict:
    """실시간(지연 포함) 현재가."""
    t = yf.Ticker(ticker)
    price = prev = None
    try:
        fi = t.fast_info
        price = fi.get("last_price") or fi.get("lastPrice")
        prev = fi.get("previous_close") or fi.get("previousClose")
    except Exception:
        pass
    if price is None:
        intr = t.history(period="1d", interval="1m")
        if not intr.empty:
            price = float(intr["Close"].iloc[-1])
    out = {"ticker": ticker.upper(), "price": float(price) if price else None,
           "prev_close": float(prev) if prev else None, "ts": time.time()}
    if out["price"] and out["prev_close"]:
        out["change_pct"] = (out["price"] / out["prev_close"] - 1) * 100
    else:
        out["change_pct"] = None
    return out
