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
