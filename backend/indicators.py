"""
보조지표 구간 분석 — 지표의 '실제 영향'을 과거 데이터로 측정해 예측에 반영.

각 지표(RSI, 스토캐스틱, 볼린저 %B, MACD 히스토그램)의 특정 구간(과매도/과매수 등)에
대해, 해당 구간에 있었던 과거 날들의 '익일 상승 비율'과 평균 익일 수익률을 계산한다.
현재 지표가 신호 구간에 있으면 그 구간의 역사적 상승확률로 상승/하락을 예측하고,
활성 신호들의 가중 평균을 모형 확률과 블렌딩한다.
"""
import numpy as np
import pandas as pd

MIN_SAMPLES = 15     # 구간 통계 신뢰 최소 표본 수
BLEND_W = 0.3        # ponytail: 고정 블렌딩 가중치, 성능 검증 후 백테스트 기반 최적화 가능


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _stoch_k(df: pd.DataFrame, n: int = 14) -> pd.Series:
    lo = df["Low"].rolling(n).min()
    hi = df["High"].rolling(n).max()
    return 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)


def _pct_b(close: pd.Series, n: int = 20) -> pd.Series:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return (close - ma) / (4 * sd).replace(0, np.nan) + 0.5


def _macd_hist(close: pd.Series) -> pd.Series:
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return macd - macd.ewm(span=9, adjust=False).mean()


def compute_series(df: pd.DataFrame) -> dict:
    """차트 표시용 지표 시계열."""
    close = df["Close"]
    return {
        "rsi": _rsi(close),
        "stoch_k": _stoch_k(df),
        "pct_b": _pct_b(close),
        "macd_hist": _macd_hist(close),
    }


# (지표키, 구간라벨, 마스크함수, 신호방향 가설)
_ZONES = [
    ("rsi", "RSI < 30 (과매도)", lambda s: s < 30),
    ("rsi", "RSI > 70 (과매수)", lambda s: s > 70),
    ("stoch_k", "스토캐스틱 %K < 20 (과매도)", lambda s: s < 20),
    ("stoch_k", "스토캐스틱 %K > 80 (과매수)", lambda s: s > 80),
    ("pct_b", "볼린저 %B < 0 (하단 이탈)", lambda s: s < 0),
    ("pct_b", "볼린저 %B > 1 (상단 이탈)", lambda s: s > 1),
    ("macd_hist", "MACD 히스토그램 > 0 (상승 모멘텀)", lambda s: s > 0),
    ("macd_hist", "MACD 히스토그램 < 0 (하락 모멘텀)", lambda s: s < 0),
]


def analyze(df: pd.DataFrame, p_model: float) -> dict:
    """
    구간별 역사적 익일 상승확률 계산 + 현재 활성 신호 판정 + 모형 확률과 블렌딩.

    반환:
      zones: 구간별 통계(표본수, 상승비율, 평균 익일수익률, 현재 활성 여부)
      signals: 현재 활성 구간의 상승/하락 신호
      p_indicator: 활성 구간 가중평균 상승확률 (없으면 None)
      p_combined: 모형 확률과 블렌딩한 최종 상승확률
    """
    series = compute_series(df)
    next_ret = df["Close"].pct_change().shift(-1)

    zones, active = [], []
    for key, label, mask_fn in _ZONES:
        s = series[key]
        mask = mask_fn(s) & s.notna() & next_ret.notna()
        rets = next_ret[mask]
        n = int(mask.sum())
        p_up = float((rets > 0).mean()) if n else None
        avg = float(rets.mean() * 100) if n else None
        last = float(s.iloc[-1])
        cur = last if np.isfinite(last) else None
        is_active = bool(cur is not None and mask_fn(pd.Series([cur])).iloc[0])
        direction = "중립"
        if is_active and n >= MIN_SAMPLES:
            direction = "상승" if p_up > 0.5 else "하락"
        z = {"indicator": key, "label": label, "n": n, "p_up": p_up,
             "avg_next_ret_pct": avg, "active": is_active,
             "current": cur, "direction": direction}
        zones.append(z)
        if is_active and n >= MIN_SAMPLES:
            active.append(z)

    p_ind = None
    if active:
        w = np.array([min(z["n"], 100) for z in active], float)  # 표본수 가중(상한 100)
        p = np.array([z["p_up"] for z in active])
        p_ind = float((w * p).sum() / w.sum())

    p_combined = p_model if p_ind is None else (1 - BLEND_W) * p_model + BLEND_W * p_ind
    return {
        "zones": zones,
        "signals": active,
        "p_indicator": p_ind,
        "p_model": p_model,
        "blend_weight": BLEND_W if p_ind is not None else 0.0,
        "p_combined": float(p_combined),
    }


def p_indicator_asof(df: pd.DataFrame) -> np.ndarray:
    """
    각 시점 t 에 대해 't 이전 데이터만으로' 계산한 활성 구간 가중평균 상승확률.

    analyze() 는 전체 기간 통계를 쓴다 — 내일을 예측할 때는 올바르지만,
    과거 시점의 예측을 채점할 때 그대로 쓰면 그 시점에 알 수 없었던 미래
    수익률이 구간 통계에 섞인다. walk-forward 채점용으로는 이 함수를 쓴다.

    반환: 길이 len(df) 의 배열. 활성 구간이 없거나 표본이 모자라면 NaN.
    """
    series = compute_series(df)
    nr = df["Close"].pct_change().shift(-1).to_numpy(float)
    n = len(df)
    zones = []
    for key, _label, mask_fn in _ZONES:
        s = series[key]
        m = (mask_fn(s) & s.notna()).to_numpy(bool)
        valid = m & np.isfinite(nr)          # 라벨(익일 수익률)을 아는 행만
        zones.append((m, np.cumsum(valid), np.cumsum(valid & (nr > 0))))
    out = np.full(n, np.nan)
    for t in range(1, n):
        lim = t - 1        # t 시점에 익일 수익률을 아는 마지막 행
        ws, ps = [], []
        for m, cum_n, cum_hit in zones:
            if not m[t]:
                continue                     # 현재 활성 구간만 반영
            cnt = int(cum_n[lim])
            if cnt < MIN_SAMPLES:
                continue
            ws.append(min(cnt, 100))
            ps.append(cum_hit[lim] / cnt)
        if ws:
            w = np.asarray(ws, float)
            out[t] = float((w * np.asarray(ps, float)).sum() / w.sum())
    return out


def blend(p_model: float, p_ind: float) -> float:
    """analyze() 와 동일한 블렌딩 규칙 (채점 경로에서 재사용)."""
    if p_ind is None or not np.isfinite(p_ind):
        return float(p_model)
    return float((1 - BLEND_W) * p_model + BLEND_W * p_ind)


def chart_series(df: pd.DataFrame, tail: int) -> dict:
    """프론트 차트용 최근 tail 개 지표값 (NaN -> None)."""
    series = compute_series(df)
    out = {}
    for k, s in series.items():
        v = s.tail(tail).to_numpy(float)
        out[k] = [None if not np.isfinite(x) else float(x) for x in v]
    return out


# ---------- 지표 조합(콤보) 분석 ----------
RSI_EDGES = [0, 30, 50, 70, 101]
RSI_LABELS = ["RSI<30", "RSI 30-50", "RSI 50-70", "RSI>70"]
PCTB_EDGES = [-np.inf, 0, 0.5, 1, np.inf]
PCTB_LABELS = ["%B<0", "%B 0-0.5", "%B 0.5-1", "%B>1"]
MACD_LABELS = ["MACD -", "MACD +"]
MIN_COMBO = 20       # 3중 조합 통계 최소 표본


def combo_analysis(df: pd.DataFrame, tail: int = 180) -> dict:
    """
    RSI × MACD × 볼린저 %B 동시 조건별 역사적 익일 상승률.

    각 지표를 구간화(bin)해 조합 상태를 만들고, 과거 그 조합이었던 날들의
    실제 익일 상승 비율/평균 수익률을 계산한다. 반환:
      grid_rsi_macd / grid_rsi_pctb : 2차원 히트맵용 통계
      combos : 표본 >= MIN_COMBO 인 3중 조합 순위 (상승률 내림차순)
      current : 현재 조합과 그 역사적 통계
      timeline : 최근 tail 일의 일별 조합 상승률 (타이밍 시각화용)
    """
    s = compute_series(df)
    next_ret = df["Close"].pct_change().shift(-1)
    rsi_b = pd.cut(s["rsi"], RSI_EDGES, labels=False)
    pctb_b = pd.cut(s["pct_b"], PCTB_EDGES, labels=False)
    macd_b = (s["macd_hist"] > 0).astype(float).where(s["macd_hist"].notna())
    valid = rsi_b.notna() & pctb_b.notna() & macd_b.notna()
    hasret = next_ret.notna()

    def stats(mask):
        m = mask & hasret
        n = int(m.sum())
        if not n:
            return {"n": 0, "p_up": None, "avg": None}
        r = next_ret[m]
        return {"n": n, "p_up": float((r > 0).mean()),
                "avg": float(r.mean() * 100)}

    grid_rm = [[stats(valid & (rsi_b == i) & (macd_b == j)) for j in (0, 1)]
               for i in range(4)]
    grid_rp = [[stats(valid & (rsi_b == i) & (pctb_b == k)) for k in range(4)]
               for i in range(4)]

    combos = []
    for i in range(4):
        for j in (0, 1):
            for k in range(4):
                st = stats(valid & (rsi_b == i) & (macd_b == j) & (pctb_b == k))
                if st["n"] >= MIN_COMBO:
                    combos.append({
                        "rsi": i, "macd": j, "pctb": k,
                        "label": f"{RSI_LABELS[i]} · {MACD_LABELS[j]} · {PCTB_LABELS[k]}",
                        **st})
    combos.sort(key=lambda c: c["p_up"], reverse=True)

    cur = None
    if bool(valid.iloc[-1]):
        i, j, k = int(rsi_b.iloc[-1]), int(macd_b.iloc[-1]), int(pctb_b.iloc[-1])
        cur = {"rsi": i, "macd": j, "pctb": k,
               "label": f"{RSI_LABELS[i]} · {MACD_LABELS[j]} · {PCTB_LABELS[k]}",
               **stats(valid & (rsi_b == i) & (macd_b == j) & (pctb_b == k))}

    lut = {(c["rsi"], c["macd"], c["pctb"]): c["p_up"] for c in combos}
    timeline = []
    for t in range(max(0, len(df) - tail), len(df)):
        if not bool(valid.iloc[t]):
            timeline.append(None)
            continue
        timeline.append(lut.get((int(rsi_b.iloc[t]), int(macd_b.iloc[t]),
                                 int(pctb_b.iloc[t]))))
    return {"rsi_labels": RSI_LABELS, "pctb_labels": PCTB_LABELS,
            "macd_labels": MACD_LABELS,
            "grid_rsi_macd": grid_rm, "grid_rsi_pctb": grid_rp,
            "combos": combos, "current": cur, "timeline": timeline}


if __name__ == "__main__":  # 자체 점검
    idx = pd.date_range("2022-01-03", periods=300, freq="B")
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.02, 300)), index=idx)
    df = pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99})
    r = analyze(df, p_model=0.6)
    assert 0.0 <= r["p_combined"] <= 1.0
    assert len(r["zones"]) == len(_ZONES)
    macd_zones = [z for z in r["zones"] if z["indicator"] == "macd_hist"]
    assert sum(z["active"] for z in macd_zones) == 1  # 히스토그램은 항상 한쪽
    cs = chart_series(df, 50)
    assert all(len(v) == 50 for v in cs.values())
    print("indicators self-check OK", {k: round(v, 3) if isinstance(v, float) else v
                                       for k, v in r.items() if k.startswith("p_")})
    cb = combo_analysis(df, tail=60)
    assert len(cb["timeline"]) == 60
    assert len(cb["grid_rsi_macd"]) == 4 and len(cb["grid_rsi_macd"][0]) == 2
    assert len(cb["grid_rsi_pctb"]) == 4 and len(cb["grid_rsi_pctb"][0]) == 4
    assert all(c["n"] >= MIN_COMBO for c in cb["combos"])
    assert all(cb["combos"][i]["p_up"] >= cb["combos"][i+1]["p_up"]
               for i in range(len(cb["combos"]) - 1))
    print("combo self-check OK:", len(cb["combos"]), "combos, current =",
          cb["current"] and cb["current"]["label"])
