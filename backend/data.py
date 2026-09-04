"""주가 데이터 수집 (yfinance) — 일봉 이력 + 실시간 호가.

일봉 스냅샷 고정 (재현성 보정)
------------------------------
yfinance 는 `auto_adjust=True` 의 배당 조정계수를 요청마다 미세하게 다르게
돌려준다 — 실측 결과 '행 단위 공통 배수'로 상대오차 최대 2.2e-7, 최근
구간은 동일하고 과거 구간만 흔들린다. 값 자체는 무시할 크기지만 GAF -> VAE
학습 궤적이 이 섭동에 혼돈적으로 반응해서, 같은 코드·같은 시드로 재실행해도
표본 외 정확도가 ±3%p, 누적 수익률이 종목에 따라 20%p 넘게 달라졌다.

확률 단계에서 눌러보는 보정(가격 양자화, VAE 시드 앙상블, 중립밴드 확대)은
측정해 본 결과 정확도 흩어짐만 줄이고 수익률 흩어짐은 줄이지 못했다.
포지션 플립이 이산적이라 몇 번의 매매가 결과를 지배하기 때문이다.
그래서 원인 쪽을 고정한다 — 같은 거래일에는 같은 일봉을 쓴다.

디스크 스냅샷을 (종목, 기간, 날짜)로 저장하고 그날 안에서는 재다운로드하지
않는다. 이러면 같은 날 실행한 백테스트·A/B 비교가 완전히 재현된다.
끄려면 환경변수 QUANT_SNAPSHOT=0, 강제 갱신은 history(..., refresh=True).
"""
import datetime
import os
import pathlib
import threading
import time

import pandas as pd
import yfinance as yf

_lock = threading.Lock()
_hist_cache: dict = {}   # ticker -> (timestamp, df)
_HIST_TTL = 300          # 5분

SNAP_DIR = (pathlib.Path(__file__).resolve().parent.parent
            / "models_cache" / "snapshots")
SNAPSHOT = os.environ.get("QUANT_SNAPSHOT", "1") != "0"
SNAP_KEEP_DAYS = 7       # 오래된 스냅샷 정리 기준


def _snap_path(ticker: str, period: str, day: str) -> pathlib.Path:
    return SNAP_DIR / f"{ticker}_{period}_{day}.pkl"


def _prune_snapshots(keep_days: int = SNAP_KEEP_DAYS) -> None:
    """keep_days 보다 오래된 스냅샷 삭제 (용량 관리)."""
    cutoff = (datetime.date.today()
              - datetime.timedelta(days=keep_days)).isoformat()
    try:
        for p in SNAP_DIR.glob("*.pkl"):
            day = p.stem.rsplit("_", 1)[-1]
            if len(day) == 10 and day < cutoff:
                p.unlink(missing_ok=True)
    except OSError:
        pass


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def _download(ticker: str, period: str, tries: int = 3):
    """레이트리밋 재시도. 재시도가 없으면 250종목 스캔에서 뒤쪽 수십 종목이
    빈 응답으로 조용히 누락되고, 표에는 '실패'가 아니라 그냥 행이 빠진다."""
    for i in range(tries):
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            return df
        if i < tries - 1:
            time.sleep(1.5 * (i + 1))
    return None


def history(ticker: str, period: str = "5y", refresh: bool = False) -> pd.DataFrame:
    """일봉 OHLCV. 메모리 5분 캐시 + 당일 디스크 스냅샷(재현성)."""
    tk = ticker.upper()
    key = (tk, period)
    if not refresh:
        with _lock:
            hit = _hist_cache.get(key)
            if hit and time.time() - hit[0] < _HIST_TTL:
                return hit[1]

    day = datetime.date.today().isoformat()
    snap = _snap_path(tk, period, day)
    if SNAPSHOT and not refresh and snap.exists():
        try:
            df = pd.read_pickle(snap)
            with _lock:
                _hist_cache[key] = (time.time(), df)
            return df
        except (OSError, ValueError, EOFError):
            snap.unlink(missing_ok=True)   # 손상된 스냅샷은 버리고 재다운로드

    df = _download(ticker, period)
    if df is None or df.empty:
        raise ValueError(f"데이터를 가져올 수 없습니다: {ticker}")
    df = _normalize(df)
    if SNAPSHOT:
        try:
            SNAP_DIR.mkdir(parents=True, exist_ok=True)
            df.to_pickle(snap)
            _prune_snapshots()
        except OSError:
            pass          # 스냅샷 실패는 조회 실패가 아니다
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
