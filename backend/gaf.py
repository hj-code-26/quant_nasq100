"""
GAF (Gramian Angular Field) 인코딩 — 논문 3.2.2절 구현.

1차원 시계열을 극좌표계 기반 2차원 행렬로 변환하여
시간적 상관관계를 보존한 이미지를 생성한다.

  식 (3.1): x~_i = ((x_i - max(X)) + (x_i - min(X))) / (max(X) - min(X))
  식 (3.2): phi_i = arccos(x~_i)
  GADF    : sin(phi_i - phi_j)   (각도의 차 — 논문이 채택한 방식)
"""
import numpy as np


def _rescale(x: np.ndarray) -> np.ndarray:
    """식 (3.1): [-1, 1] 구간으로 표준화."""
    x = np.asarray(x, dtype=np.float64)
    rng = x.max() - x.min()
    if rng == 0:
        return np.zeros_like(x)
    xs = ((x - x.max()) + (x - x.min())) / rng
    return np.clip(xs, -1.0, 1.0)


def gadf(x: np.ndarray) -> np.ndarray:
    """GADF 행렬 (N x N), 값 범위 [-1, 1]."""
    phi = np.arccos(_rescale(x))
    return np.sin(phi[:, None] - phi[None, :])


def gasf(x: np.ndarray) -> np.ndarray:
    """GASF 행렬 (비교용, 각도의 합)."""
    phi = np.arccos(_rescale(x))
    return np.cos(phi[:, None] + phi[None, :])


def _tile(fn, open_, high, low, close) -> np.ndarray:
    """논문 [그림 3.5]: [[종가, 시가], [고가, 저가]] 2x2 타일."""
    top = np.hstack([fn(close), fn(open_)])
    bot = np.hstack([fn(high), fn(low)])
    img = np.vstack([top, bot])
    return ((img + 1.0) / 2.0).astype(np.float32)


def ohlc_gaf_image(open_, high, low, close, dual: bool = True) -> np.ndarray:
    """
    시가/고가/저가/종가의 GAF 변환 2x2 타일 이미지.

    dual=True 이면 GADF + GASF 2채널 (개선사항: 논문 3.2.2절에서
    GASF가 정확도 측면에서 우세한 경우가 있다고 언급 — 두 인코딩을
    모두 입력해 정보 손실을 줄인다). False 면 GADF 1채널 (논문 기본).
    반환: (C, 2N, 2N) float32, 값 범위 [0, 1].
    """
    ch = [_tile(gadf, open_, high, low, close)]
    if dual:
        ch.append(_tile(gasf, open_, high, low, close))
    return np.stack(ch)


def build_dataset(df, window: int = 20, dual: bool = True, horizon: int = 1):
    """
    슬라이딩 윈도우(=1일 이동, 논문 3.1절)로 GAF 이미지 데이터셋 생성.

    df: Open/High/Low/Close 컬럼을 가진 DataFrame (일봉)
    horizon: 라벨의 전망 기간(거래일). 1=익일, 5=1주, 21=1개월.
    반환: X (M, C, 2W, 2W), y (M,)  — y=1 이면 horizon 일 뒤 종가 상승(>=), 0 하락,
          dates: 각 윈도우의 마지막 거래일 (예측 기준일)
    윈도우 마지막 날 t 에 대해 라벨은 C_{t+horizon} >= C_t.
    라벨을 만들 수 없는 마지막 horizon 개 윈도우는 y=-1 (실시간 예측용).

    주의: horizon>1 이면 이웃 표본이 미래 구간을 공유해 서로 독립이 아니다.
    유효 표본 수는 대략 n/horizon 이므로 유의성 검정 때 반드시 보정해야 한다.
    """
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    n = len(df)
    X, y, dates = [], [], []
    for t in range(window - 1, n):
        s = t - window + 1
        X.append(ohlc_gaf_image(o[s:t + 1], h[s:t + 1], l[s:t + 1],
                                c[s:t + 1], dual=dual))
        if t + horizon < n:
            y.append(1 if c[t + horizon] >= c[t] else 0)
        else:
            y.append(-1)
        dates.append(df.index[t])
    return np.stack(X), np.asarray(y, dtype=np.int64), dates
