"""
특징 구성 A/B 비교 실행기.

같은 종목·같은 기간·같은 VAE(캐시 재사용)·같은 라벨 위에서 분류기 입력만
바꿔 walk-forward 를 세 번 돌린다.

  ① VAE 잠재변수 15차원만        (조예서 2022 방식)
  ② 기술적 지표 12종만           (조병호 2021 엔진 설계)
  ③ 둘을 결합한 27차원           (본 구현)

세 변형이 같은 날짜를 예측하므로 우열은 짝지은 McNemar 검정으로 판정한다.
정확도 숫자만 나열하면 표본 126일 기준 ±8%p 의 노이즈를 실력 차이로
오독하게 되기 때문이다.
"""
import threading
import time

from . import data as data_mod
from . import gaf, models, pipeline, vae

_lock = threading.Lock()
_store: dict = {}


def get_state(ticker: str) -> dict:
    with _lock:
        return dict(_store.get(ticker.upper(), {"status": "idle"}))


def _set(ticker: str, **kw):
    with _lock:
        _store.setdefault(ticker.upper(), {}).update(kw)


def run(ticker: str, period: str = "5y"):
    ticker = ticker.upper()
    t0 = time.time()
    try:
        _set(ticker, status="running", progress=0.05,
             message="주가 데이터 · GAF 인코딩…", result=None)
        df = data_mod.history(ticker, period)
        as_of = str(df.index[-1].date())
        X, y, dates = gaf.build_dataset(df, pipeline.WINDOW, dual=True)
        labeled = y >= 0
        Xl, yl = X[labeled], y[labeled]
        dl = [d for d, m in zip(dates, labeled) if m]

        fit_end, leak_free = pipeline.vae_fit_end(len(Xl))
        model = pipeline._load_cached_model(ticker, as_of, X.shape[1], fit_end)
        if model is None:
            _set(ticker, progress=0.15,
                 message="VAE 학습 중… (검증 구간 제외, 캐시 없음)")
            model = vae.train_vae(Xl[:fit_end], epochs=pipeline.EPOCHS,
                                  latent_dim=pipeline.LATENT_DIM)
            pipeline._save_model(ticker, model, as_of, X.shape[-1], fit_end)
        else:
            _set(ticker, progress=0.15, message="캐시된 VAE 재사용")

        _set(ticker, progress=0.25, message="잠재변수 · 지표 결합 중…")
        Z_all, feat_names = pipeline.build_features(model, X, df, dates)
        Zl = Z_all[labeled]

        def prog(i, total, label):
            _set(ticker, progress=0.30 + 0.65 * i / total,
                 message=f"Walk-forward 비교 중… ({i + 1}/{total}) {label}")

        ab = models.ablation(Zl, yl, dl, n_latent=pipeline.LATENT_DIM,
                             n_walks=pipeline.N_WALKS,
                             test_days=pipeline.TEST_DAYS, progress=prog)
        ab.update({
            "ticker": ticker,
            "as_of": as_of,
            "period": period,
            "feature_names": feat_names,
            "n_latent": pipeline.LATENT_DIM,
            "vae_leak_free": bool(leak_free),
            "vae_fit_end": str(dl[fit_end - 1].date()) if fit_end else None,
            "label_eps_bp": gaf.LABEL_EPS * 10000.0,
            "elapsed_sec": round(time.time() - t0, 1),
            "ran_at": time.time(),
        })
        _set(ticker, status="done", progress=1.0,
             message=f"완료 ({ab['elapsed_sec']:.0f}초)", result=ab)
    except Exception as e:  # noqa: BLE001
        _set(ticker, status="error", message=f"오류: {e}")


def start(ticker: str, period: str = "5y", force: bool = False) -> dict:
    st = get_state(ticker)
    if st.get("status") == "running" and not force:
        return {k: v for k, v in st.items() if k != "result"}
    if st.get("status") == "done" and not force:
        if time.time() - st.get("result", {}).get("ran_at", 0) < 3600:
            return {k: v for k, v in st.items() if k != "result"}
    threading.Thread(target=run, args=(ticker, period), daemon=True).start()
    time.sleep(0.1)
    st = get_state(ticker)
    return {k: v for k, v in st.items() if k != "result"}
