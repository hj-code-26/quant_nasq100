"""
분석 파이프라인 — 논문 [그림 1.1] 흐름도 구현 + 개선사항.

  주가 데이터 -> GAF(GADF+GASF 2채널) 인코딩 -> VAE 잠재변수 추출
    -> 로지스틱/SVM/랜덤포레스트 앙상블 -> 익일 상승확률
    -> 임계값 보정 + 표본 외 백테스트 + gs-quant 분석
    -> 예상 주가(1/5/20일) + 시기(Walk-forward) 분석

개선사항: 시드 고정(재현성), 모델 디스크 캐시(같은 기준일 재학습 생략),
GADF+GASF 2채널, 보정 임계값, 다중 시계 예상주가, 백테스트.
"""
import base64
import io
import math
import pathlib
import threading
import time

import joblib
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["AppleGothic", "Apple SD Gothic Neo", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from . import backtest as bt
from . import data as data_mod
from . import gaf, models, quant, vae

WINDOW = 20          # 논문: 20일 촛대 차트 기간
LATENT_DIM = 15      # 논문: 잠재 변수 15
EPOCHS = 25
CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "models_cache"

_lock = threading.Lock()
_store: dict = {}    # ticker -> {"status", "progress", "result", ...}


def get_state(ticker: str) -> dict:
    with _lock:
        return dict(_store.get(ticker.upper(), {"status": "idle"}))


def _set(ticker: str, **kw):
    with _lock:
        st = _store.setdefault(ticker.upper(), {})
        st.update(kw)


def _gaf_png(img: np.ndarray) -> str:
    """GAF 2x2 타일(GADF 채널)을 base64 PNG 로 렌더링."""
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100)
    ax.imshow(img[0], cmap="rainbow", vmin=0, vmax=1)
    n = img.shape[-1] // 2
    ax.axhline(n - 0.5, color="white", lw=1)
    ax.axvline(n - 0.5, color="white", lw=1)
    for (x, y, t) in [(n * 0.5, -1.5, "종가"), (n * 1.5, -1.5, "시가"),
                      (n * 0.5, n * 2 + 2.5, "고가"), (n * 1.5, n * 2 + 2.5, "저가")]:
        ax.text(x, y, t, ha="center", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _cache_paths(ticker: str):
    base = CACHE_DIR / ticker.upper()
    return base.with_suffix(".pt"), base.with_suffix(".joblib")


def _load_cached_model(ticker, as_of, n_channels):
    """같은 기준일의 학습 모델이 있으면 로드 (개선사항: 재학습 생략)."""
    pt, jb = _cache_paths(ticker)
    if not (pt.exists() and jb.exists()):
        return None
    try:
        meta = joblib.load(jb)
        if meta.get("as_of") != as_of:
            return None
        model = vae.ConvVAE(img=meta["img"], latent_dim=LATENT_DIM,
                            channels=n_channels)
        model.load_state_dict(torch.load(pt, map_location="cpu"))
        model.to(vae.device()).eval()
        return model
    except Exception:
        return None


def _save_model(ticker, model, as_of, img):
    pt, jb = _cache_paths(ticker)
    try:
        torch.save(model.state_dict(), pt)
        joblib.dump({"as_of": as_of, "img": img}, jb)
    except OSError:
        pass


def multi_horizon(close, p_up, threshold=0.5):
    """
    다중 시계 예상 주가 (개선사항).
    1일: 모형 확률 기반 기대수익률.  5/20일: 1일 모형 시각 + 이후 과거
    평균 드리프트, 밴드는 ±1σ·√h 로 확장.
    """
    rets = close.pct_change().dropna()
    up_m = float(rets[rets >= 0].mean()) if (rets >= 0).any() else 0.0
    dn_m = float(rets[rets < 0].mean()) if (rets < 0).any() else 0.0
    drift = float(rets.tail(252).mean())
    sigma = float(rets.tail(60).std())
    r1 = p_up * up_m + (1 - p_up) * dn_m
    last = float(close.iloc[-1])
    out = []
    for h in (1, 5, 20):
        r = r1 + (h - 1) * drift
        band = sigma * math.sqrt(h)
        out.append({"horizon": h, "expected_price": last * (1 + r),
                    "expected_return_pct": r * 100,
                    "band_low": last * (1 + r - band),
                    "band_high": last * (1 + r + band)})
    return out


def run_analysis(ticker: str, period: str = "5y"):
    """전체 파이프라인 실행 (백그라운드 스레드에서 호출)."""
    ticker = ticker.upper()
    try:
        _set(ticker, status="loading", progress=0.02, message="주가 데이터 수집 중…")
        df = data_mod.history(ticker, period)
        as_of = str(df.index[-1].date())

        _set(ticker, progress=0.08, message="GAF(GADF+GASF) 인코딩 중…")
        X, y, dates = gaf.build_dataset(df, WINDOW, dual=True)
        labeled = y >= 0
        Xl, yl = X[labeled], y[labeled]
        dl = [d for d, m in zip(dates, labeled) if m]

        model = _load_cached_model(ticker, as_of, X.shape[1])
        if model is None:
            _set(ticker, status="training", progress=0.10,
                 message=f"VAE 학습 중… (이미지 {len(Xl):,}장 × {X.shape[1]}채널)")

            def prog(ep, total, loss):
                _set(ticker, progress=0.10 + 0.5 * ep / total,
                     message=f"VAE 학습 중… epoch {ep}/{total} (loss {loss:.1f})")

            model = vae.train_vae(Xl, epochs=EPOCHS, latent_dim=LATENT_DIM,
                                  progress=prog)
            _save_model(ticker, model, as_of, X.shape[-1])
        else:
            _set(ticker, status="training", progress=0.55,
                 message="캐시된 모델 로드 (동일 기준일 — 재학습 생략)")

        _set(ticker, progress=0.62, message="잠재 변수 추출 중…")
        Z_all = vae.extract_latents(model, X)
        Zl = Z_all[labeled]

        _set(ticker, progress=0.70, message="Walk-forward 검증 중… (6 walks)")
        wf = models.walk_forward(Zl, yl, dl, n_walks=6, test_days=21)

        # 개선사항: 표본 외 예측으로 결정 임계값 보정 + 백테스트
        cal = bt.calibrate_threshold(wf["per_day"])
        closes_by_date = {str(i.date()): float(v)
                          for i, v in df["Close"].items()}
        backtest = bt.run_backtest(wf["per_day"], closes_by_date)

        _set(ticker, progress=0.86, message="분류 모형 학습 및 예측 중…")
        bundle = models.fit_classifiers(Zl, yl)
        probs_last = models.predict_proba(bundle, Z_all[-1:])
        p_up = float(probs_last["ensemble"][0])
        threshold = cal["threshold"]

        k = min(120, len(Z_all))
        probs_hist = models.predict_proba(bundle, Z_all[-k:])["ensemble"]
        signal_series = [
            {"date": str(dates[len(dates) - k + i].date()),
             "prob": float(probs_hist[i])}
            for i in range(k)
        ]

        _set(ticker, progress=0.93, message="gs-quant 지표 계산 중…")
        close = df["Close"]
        ind = quant.analytics(close, WINDOW)
        exp = quant.expected_price(close, p_up)
        horizons = multi_horizon(close, p_up, threshold)
        overlays = quant.series_for_chart(close, WINDOW)

        tail = df.tail(180)
        candles = [
            {"date": str(idx.date()), "o": float(r.Open), "h": float(r.High),
             "l": float(r.Low), "c": float(r.Close), "v": float(r.Volume)}
            for idx, r in tail.iterrows()
        ]
        ov = {}
        for name, s in overlays.items():
            s = s.reindex(tail.index)
            ov[name] = [None if (v != v) else float(v) for v in s.to_numpy(float)]

        # 신뢰 구간: |p - threshold| 가 작으면 '중립' (개선사항)
        conf = abs(p_up - threshold)
        direction = ("중립" if conf < 0.03
                     else "상승" if p_up >= threshold else "하락")

        result = {
            "ticker": ticker,
            "as_of": as_of,
            "n_images": int(len(X)),
            "window": WINDOW,
            "latent_dim": LATENT_DIM,
            "channels": int(X.shape[1]),
            "prediction": {
                "p_up": p_up,
                "direction": direction,
                "threshold": threshold,
                "calibrated_acc": cal["accuracy"],
                "confidence": conf,
                "per_model": {k2: float(v[0]) for k2, v in probs_last.items()},
                "horizons": horizons,
                **exp,
            },
            "indicators": ind,
            "walk_forward": wf,
            "backtest": backtest,
            "signals": signal_series,
            "candles": candles,
            "overlays": ov,
            "gaf_png": _gaf_png(X[-1]),
            "trained_at": time.time(),
        }
        _set(ticker, status="done", progress=1.0, message="완료", result=result)
    except Exception as e:  # noqa: BLE001
        _set(ticker, status="error", message=f"오류: {e}")


def start_analysis(ticker: str, period: str = "5y", force: bool = False) -> dict:
    st = get_state(ticker)
    if st.get("status") in ("loading", "training") and not force:
        return st
    if st.get("status") == "done" and not force:
        age = time.time() - st.get("result", {}).get("trained_at", 0)
        if age < 3600:
            return st
    threading.Thread(target=run_analysis, args=(ticker, period), daemon=True).start()
    time.sleep(0.1)
    return get_state(ticker)
