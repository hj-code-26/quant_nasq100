"""
gs-quant (Goldman Sachs Quant) timeseries 분석 루틴.

https://github.com/goldmansachs/gs-quant 의 gs_quant.timeseries 모듈을
사용하여 기술적 지표와 리스크 지표를 계산한다 (GS 세션 불필요, 로컬 계산).
"""
import math

import numpy as np
import pandas as pd
import gs_quant.timeseries as ts


def _last(series) -> float:
    try:
        v = float(series.dropna().iloc[-1])
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def analytics(close: pd.Series, window: int = 20) -> dict:
    """gs-quant 함수 기반 핵심 지표 요약."""
    out = {}
    w = ts.Window(window, 0)
    try:
        out["ma20"] = _last(ts.moving_average(close, w))
        out["ma60"] = _last(ts.moving_average(close, ts.Window(60, 0)))
    except Exception:
        out["ma20"] = out["ma60"] = float("nan")
    try:
        # 연율화 실현 변동성 (%)
        out["volatility_20d"] = _last(ts.volatility(close, w))
    except Exception:
        out["volatility_20d"] = float("nan")
    try:
        out["rsi_14"] = _last(ts.relative_strength_index(close, 14))
    except Exception:
        out["rsi_14"] = float("nan")
    try:
        out["macd"] = _last(ts.macd(close, 12, 26))
    except Exception:
        out["macd"] = float("nan")
    try:
        bb = ts.bollinger_bands(close, w, 2)
        out["bb_low"] = _last(bb.iloc[:, 0])
        out["bb_high"] = _last(bb.iloc[:, 1])
    except Exception:
        out["bb_low"] = out["bb_high"] = float("nan")
    try:
        rets = ts.returns(close).dropna()
        out["ret_1d"] = _last(rets)
        out["max_drawdown_1y"] = _last(ts.max_drawdown(close.tail(252), ts.Window(252, 0)))
    except Exception:
        out["ret_1d"] = out["max_drawdown_1y"] = float("nan")
    return out


def series_for_chart(close: pd.Series, window: int = 20) -> dict:
    """차트 오버레이용 시계열 (MA/볼린저)."""
    res = {}
    try:
        res["ma20"] = ts.moving_average(close, ts.Window(window, 0))
        res["ma60"] = ts.moving_average(close, ts.Window(60, 0))
        bb = ts.bollinger_bands(close, ts.Window(window, 0), 2)
        res["bb_low"], res["bb_high"] = bb.iloc[:, 0], bb.iloc[:, 1]
    except Exception:
        pass
    return res


def expected_price(close: pd.Series, p_up: float) -> dict:
    """
    예상 주가 계산.

    논문 모형은 상승/하락의 '확률'을 산출하므로, 기대 수익률을
      E[r] = p_up * E[r | r>=0] + (1 - p_up) * E[r | r<0]
    로 계산해 가격으로 환산한다. 신뢰 구간은 gs-quant 실현 변동성의
    일간 ±1σ 밴드를 사용한다.
    """
    rets = ts.returns(close).dropna()
    up_mean = float(rets[rets >= 0].mean()) if (rets >= 0).any() else 0.0
    dn_mean = float(rets[rets < 0].mean()) if (rets < 0).any() else 0.0
    exp_ret = p_up * up_mean + (1.0 - p_up) * dn_mean
    last = float(close.iloc[-1])
    try:
        ann_vol = _last(ts.volatility(close, ts.Window(20, 0))) / 100.0
        daily_sigma = ann_vol / math.sqrt(252) if math.isfinite(ann_vol) else float(rets.std())
    except Exception:
        daily_sigma = float(rets.std())
    target = last * (1.0 + exp_ret)
    return {
        "last_close": last,
        "expected_return_pct": exp_ret * 100.0,
        "expected_price": target,
        "band_low": last * (1.0 + exp_ret - daily_sigma),
        "band_high": last * (1.0 + exp_ret + daily_sigma),
        "daily_sigma_pct": daily_sigma * 100.0,
        "up_day_mean_pct": up_mean * 100.0,
        "down_day_mean_pct": dn_mean * 100.0,
    }


# ---------------------------------------------------------------------------
# 기술적 지표 특징 행렬 (논문 설계 반영)
#
# 조병호(2021) 「인공지능 머신러닝 기술을 이용한 주식 종목 매수/매도 추천
# 시스템의 분석 및 설계」의 엔진 설계는 '여러 기술적 분석 기법의 결과값을
# 머신러닝 입력으로 결합'하는 것이 핵심이다(단일 기법 평균 56.7% vs 결합
# 78.3% 주장). 기존 구현은 아래 지표들을 화면 표시에만 썼고 분류기 입력은
# VAE 잠재변수뿐이었다. 이 함수는 지표들을 학습 가능한 특징으로 만들어
# 잠재변수와 결합할 수 있게 한다.
#
# 모든 특징은 t 시점까지의 데이터만 사용하는 인과적(causal) 롤링 계산이며,
# 수준(level)이 아니라 비율/정규화 값이라 시계열 간 비교가 가능하다.
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "ret_1d", "ret_5d", "ret_20d", "ma20_gap", "ma60_gap", "rsi_14",
    "macd_norm", "bb_pctb", "vol_20d", "volume_ratio", "range_14", "mom_60d",
]


def _align(series, index) -> pd.Series:
    """gs-quant 반환 시계열을 원본 인덱스에 맞춰 정렬."""
    s = pd.Series(series)
    try:
        s = s.reindex(index)
    except Exception:
        s = pd.Series(np.asarray(s, dtype=float), index=index[-len(s):]).reindex(index)
    return s.astype(float)


def feature_matrix(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    일봉 DataFrame -> 기술적 지표 특징 행렬 (인덱스는 df 와 동일).

    gs-quant timeseries 함수를 우선 사용하고, 실패 시 동등한 pandas 계산으로
    대체한다. 선행 구간의 NaN 은 0(중립)으로 채운다.
    """
    close = df["Close"].astype(float)
    idx = df.index
    f = pd.DataFrame(index=idx)

    f["ret_1d"] = close.pct_change(1)
    f["ret_5d"] = close.pct_change(5)
    f["ret_20d"] = close.pct_change(20)

    try:
        ma20 = _align(ts.moving_average(close, ts.Window(window, 0)), idx)
        ma60 = _align(ts.moving_average(close, ts.Window(60, 0)), idx)
    except Exception:
        ma20 = close.rolling(window).mean()
        ma60 = close.rolling(60).mean()
    f["ma20_gap"] = close / ma20 - 1.0
    f["ma60_gap"] = close / ma60 - 1.0

    try:
        rsi = _align(ts.relative_strength_index(close, 14), idx)
    except Exception:
        d = close.diff()
        up = d.clip(lower=0).rolling(14).mean()
        dn = (-d.clip(upper=0)).rolling(14).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    f["rsi_14"] = rsi / 100.0 - 0.5

    try:
        macd = _align(ts.macd(close, 12, 26), idx)
    except Exception:
        macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    f["macd_norm"] = macd / close

    try:
        bb = ts.bollinger_bands(close, ts.Window(window, 0), 2)
        lo, hi = _align(bb.iloc[:, 0], idx), _align(bb.iloc[:, 1], idx)
    except Exception:
        sd = close.rolling(window).std()
        lo, hi = ma20 - 2 * sd, ma20 + 2 * sd
    f["bb_pctb"] = (close - lo) / (hi - lo) - 0.5

    try:
        f["vol_20d"] = _align(ts.volatility(close, ts.Window(window, 0)), idx) / 100.0
    except Exception:
        f["vol_20d"] = close.pct_change().rolling(window).std() * math.sqrt(252)

    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        f["volume_ratio"] = vol / vol.rolling(window).mean() - 1.0
    else:
        f["volume_ratio"] = 0.0

    if {"High", "Low"} <= set(df.columns):
        f["range_14"] = ((df["High"] - df["Low"]) / close).rolling(14).mean()
    else:
        f["range_14"] = 0.0

    f["mom_60d"] = close / close.shift(60) - 1.0

    f = f[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return f.astype(float)
