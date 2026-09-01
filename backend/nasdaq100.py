"""
나스닥 유니버스 전체 스캔 (250종목) — 구성 종목 전체에 GAF·VAE 파이프라인(고속 모드)을
적용해 Walk-forward 표본 외 정확도 순으로 정렬한다.

고속 모드: 2년 데이터, VAE 8 epochs, 3 walks — 종목당 수 초.
정렬 기준: 앙상블 Walk-forward 평균 정확도 (내림차순).
"""
import json
import pathlib
import threading
import time

import numpy as np

from . import backtest as bt
from . import data as data_mod
from . import gaf, models, pipeline, vae

# 스캔 유니버스 (250종목) — NASDAQ-100 구성 종목 + 나스닥 상장 유동성 상위 종목.
# 확장분은 6개월 일봉 존재 여부로 생존 확인 후 일평균 거래대금순 선정 (2026-08 기준).
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA",
    "COST", "NFLX", "AMD", "PEP", "ADBE", "CSCO", "QCOM", "TMUS", "INTU",
    "AMAT", "TXN", "CMCSA", "ISRG", "AMGN", "HON", "BKNG", "VRTX", "SBUX",
    "ADP", "GILD", "MU", "LRCX", "ADI", "PANW", "MDLZ", "REGN", "INTC",
    "KLAC", "SNPS", "CDNS", "PYPL", "MAR", "CSX", "ASML", "ORLY", "CRWD",
    "ABNB", "MRVL", "NXPI", "CTAS", "ROP", "PCAR", "AEP", "MNST", "FTNT",
    "WDAY", "ADSK", "CPRT", "ROST", "KDP", "PAYX", "ODFL", "DDOG", "FAST",
    "CHTR", "EXC", "GEHC", "IDXX", "VRSK", "CCEP", "EA", "XEL", "CTSH",
    "TTWO", "MCHP", "DXCM", "AZN", "KHC", "ON", "BIIB", "ZS", "MRNA",
    "WBD", "DLTR", "ARM", "CEG", "TEAM", "PDD", "MELI", "CDW", "GFS",
    "BKR", "LULU", "SMCI", "APP", "PLTR", "AXON", "LIN", "DASH", "TRI",
    "MSTR", "SHOP",
]

# --- 확장분 (149종목) ---
TICKERS += [
    "LITE", "WDC", "STX", "HOOD", "COIN", "TER", "SOFI", "MPWR", "CVNA",
    "MDB", "AKAM", "FANG", "EBAY", "ROKU", "EXPE", "TWLO", "TTD", "CASY",
    "OKTA", "ENTG", "NTAP", "IBKR", "ALNY", "HBAN", "ULTA", "ZM", "INSM",
    "TSCO", "FITB", "PINS", "NTRA", "AMKR", "CHRW", "ONTO", "SITM", "SWKS",
    "RMBS", "AFRM", "FOXA", "W", "BIDU", "ILMN", "UTHR", "ENPH", "JD", "VIAV",
    "AEIS", "FIVE", "JBHT", "FFIV", "LSCC", "SNAP", "PODD", "TECH", "TYL",
    "VRSN", "DECK", "TROW", "NVMI", "FORM", "ETSY", "JAZZ", "CHWY", "WING",
    "SANM", "SAIA", "EXPD", "DOCU", "FWONK", "DUOL", "TXRH", "ALGN", "NBIX",
    "JKHY", "NTRS", "EPAM", "NTNX", "GRAB", "INCY", "CHKP", "SFM", "SEDG",
    "AXSM", "MEDP", "IONS", "SSNC", "GTLB", "GEN", "CACI", "WIX", "ARWR",
    "VSAT", "OLLI", "MTCH", "TCOM", "HALO", "UPST", "ICLR", "EXEL", "CROX",
    "SIRI", "MNDY", "TWST", "PNFP", "NWSA", "RUN", "BMRN", "EWBC", "ALGM",
    "NTES", "CAKE", "LOGI", "QRVO", "PTCT", "DBX", "UCTT", "HSIC", "ZION",
    "SYNA", "BILL", "CRUS", "BSY", "CRSP", "TENB", "QLYS", "PCTY", "SLAB",
    "LSTR", "ALKS", "CORT", "RNG", "ACLS", "HQY", "PLXS", "LKQ", "CAMT",
    "WTFC", "NTLA", "BOX", "SEIC", "VIRT", "ICHR", "VECO", "VRNS", "POWI",
    "FIVN", "APPF", "NICE", "CBSH",
]

_STATE = {"status": "idle", "done": 0, "total": 0, "current": "",
          "results": [], "started_at": None, "stop": False}
_lock = threading.Lock()
# v2: VAE 누수 제거 + 지표 결합 + 롤링 임계값으로 산출 방식이 바뀌어
# 이전 결과와 비교 불가하므로 캐시 파일을 분리한다.
CACHE = (pathlib.Path(__file__).resolve().parent.parent / "models_cache"
         / "scan_results_v3.json")   # v3: 표본 외 백테스트 승률 추가


def get_state():
    with _lock:
        st = dict(_STATE)
        st["results"] = sorted(st["results"],
                               key=lambda r: r.get("accuracy", 0), reverse=True)
        return st


def _set(**kw):
    with _lock:
        _STATE.update(kw)


def _append(row):
    with _lock:
        _STATE["results"].append(row)
        try:
            CACHE.write_text(json.dumps(_STATE["results"], ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


def load_cached():
    if CACHE.exists():
        try:
            with _lock:
                if not _STATE["results"]:
                    _STATE["results"] = json.loads(CACHE.read_text(encoding="utf-8"))
                    _STATE["status"] = "done" if _STATE["results"] else "idle"
                    _STATE["done"] = _STATE["total"] = len(_STATE["results"])
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            pass


def scan_one(ticker: str, period="2y", epochs=8, n_walks=3,
             horizon: int = 1) -> dict:
    """
    단일 종목 고속 분석 -> 정확도/예측 요약.

    horizon: 라벨 전망 기간(거래일). 1=익일, 5=1주, 21=1개월.
    horizon>1 이면 학습 구간 끝 horizon-1 개를 잘라 라벨 누수를 막는다.
    또한 이웃 표본이 미래 구간을 공유하므로(독립 아님) 유효 표본은
    대략 n/horizon 이다 - 유의성 판단 때 반드시 감안해야 한다.

    정확도는 표본이 작아(3 walks x 21일 = 63일) 표준오차가 6%p 수준이다.
    250 종목을 이 값으로 줄세우면 상위권은 실력이 아니라 다중비교로 뽑힌
    노이즈일 수 있으므로, 표본 수·95% 신뢰구간·다수클래스 기준선 대비
    p-value 를 함께 반환해 화면에서 같이 보여준다.
    """
    df = data_mod.history(ticker, period)
    if len(df) < 150:
        raise ValueError("데이터 부족")
    X, y, dates = gaf.build_dataset(df, 20, horizon=horizon)
    labeled = y >= 0
    Xl, yl = X[labeled], y[labeled]
    dl = [d for d, m in zip(dates, labeled) if m]
    # 검증 구간을 제외하고 VAE 학습 (수정사항 ②)
    fit_end, leak_free = pipeline.vae_fit_end(len(Xl), n_walks, 21)
    model = vae.train_vae(Xl[:fit_end], epochs=epochs, latent_dim=15)
    # 잠재변수 + 기술적 지표 결합 (수정사항 ④)
    Z_all, feat_names = pipeline.build_features(model, X, df, dates)
    Zl = Z_all[labeled]
    wf = models.walk_forward(Zl, yl, dl, n_walks=n_walks, test_days=21,
                             embargo=horizon - 1)
    bundle = models.fit_classifiers(Zl, yl)
    p_up = float(models.predict_proba(bundle, Z_all[-1:])["ensemble"][0])
    cal = bt.rolling_calibration(wf["per_day"])
    # 롤링 보정 임계값(그날 이전 walk 들로만 결정)으로 표본 외 매매를 흉내낸다.
    # 임계값을 이 표본에서 탐색하지 않으므로 승률에 탐색 편향이 없다 —
    # 다만 3 walks(63일)뿐이라 매매 횟수가 적어 승률 자체의 오차는 크다.
    btr = bt.run_backtest(wf["per_day"],
                          {str(i.date()): float(v) for i, v in df["Close"].items()})
    bs = btr["stats"] if btr else {}
    pooled = wf["pooled"]
    avg = wf["average"]["ensemble"]
    last = float(df["Close"].iloc[-1])
    rets = df["Close"].pct_change().dropna()
    up_m = float(rets[rets >= 0].mean()) if (rets >= 0).any() else 0.0
    dn_m = float(rets[rets < 0].mean()) if (rets < 0).any() else 0.0
    exp_ret = p_up * up_m + (1 - p_up) * dn_m
    conf = abs(p_up - cal["threshold"])
    return {
        "ticker": ticker,
        "horizon": horizon,
        "accuracy": pooled["accuracy"],          # 임계값 0.5 고정, 전 구간 합산
        "baseline": pooled["baseline"],          # 다수 클래스만 찍었을 때
        "n": pooled["n"],
        "n_eff": pooled["n"] // max(1, horizon),  # 겹침 보정 유효 표본
        "ci_low": pooled["ci_low"],
        "ci_high": pooled["ci_high"],
        "p_value": pooled["p_value"],
        "significant": pooled["significant"],
        "rolling_acc": cal["accuracy"],          # 롤링 보정 임계값 기준
        "rolling_n": cal["n_eval"],
        "insample_acc": cal["insample_accuracy"],  # 편향값(참고용)
        "f1": avg.get("f1", float("nan")),
        "auc": avg.get("auc", float("nan")),
        "threshold": cal["threshold"],
        "p_up": p_up,
        "direction": ("중립" if conf < bt.NEUTRAL_BAND
                      else "상승" if p_up >= cal["threshold"] else "하락"),
        "leak_free": bool(leak_free),
        "last_close": last,
        "expected_price": last * (1 + exp_ret),
        "expected_return_pct": exp_ret * 100,
        "as_of": str(df.index[-1].date()),
        # 표본 외 백테스트 (롤링 임계값) — 후보 선별의 1차 기준
        "win_rate_pct": bs.get("win_rate_pct", float("nan")),
        "bt_return_pct": bs.get("strategy_return_pct", float("nan")),
        "bt_buyhold_pct": bs.get("buyhold_return_pct", float("nan")),
        "bt_trades": bs.get("trades", 0),
        "bt_sharpe": bs.get("sharpe", float("nan")),
        "ver": 3,
    }


def _run(tickers, epochs):
    _set(status="running", done=0, total=len(tickers), results=[],
         started_at=time.time(), stop=False)
    for i, t in enumerate(tickers):
        if _STATE["stop"]:
            _set(status="stopped", current="")
            return
        _set(current=t, done=i)
        try:
            _append(scan_one(t, epochs=epochs))
        except Exception as e:  # noqa: BLE001
            _append({"ticker": t, "error": str(e)[:120], "accuracy": -1})
        _set(done=i + 1)
    _set(status="done", current="")


def start_scan(limit: int = 0, epochs: int = 8) -> dict:
    if _STATE["status"] == "running":
        return get_state()
    tickers = TICKERS[:limit] if limit else TICKERS
    threading.Thread(target=_run, args=(tickers, epochs), daemon=True).start()
    time.sleep(0.05)
    return get_state()


def stop_scan():
    _set(stop=True)
    return get_state()


load_cached()


def scan_stats() -> dict:
    """스캔 결과가 '우연'과 구분되는지 이항검정.

    종목당 표본이 63개뿐이라 우연만으로도 정확도 표준편차가 6.3%p 다.
    따라서 '정확도 55% 이상 N개' 같은 집계는 의미가 없고, 우연 기대치와
    비교해야 한다. 다중검정(250종목 중 최고를 고름) 보정도 함께 계산한다.
    """
    import math
    from scipy import stats as sps

    rows = [x for x in _STATE["results"] if x.get("accuracy", -1) >= 0]
    if len(rows) < 2:
        return {"n_tickers": len(rows), "ready": False}
    acc = [x["accuracy"] for x in rows]
    vals = sorted(set(round(a, 6) for a in acc))
    gaps = [b - a for a, b in zip(vals, vals[1:]) if b - a > 1e-9]
    n = round(1 / min(gaps)) if gaps else 63          # 종목당 표본 수 역산
    N = len(acc)

    buckets = []
    for th in (0.50, 0.55, 0.60):
        k = math.ceil(th * n - 1e-9)
        p_one = float(1 - sps.binom.cdf(k - 1, n, 0.5))
        obs = sum(a >= th - 1e-9 for a in acc)
        buckets.append({"threshold": th, "observed": obs,
                        "expected_by_chance": round(p_one * N, 1),
                        "ratio": round(obs / (p_one * N), 2) if p_one * N else None})

    hits = sum(round(a * n) for a in acc)
    pooled_p = float(sps.binomtest(hits, N * n, 0.5, alternative="greater").pvalue)
    best = max(acc)
    p_best = float(1 - sps.binom.cdf(round(best * n) - 1, n, 0.5))
    return {
        "ready": True, "n_tickers": N, "samples_per_ticker": n,
        "mean_accuracy": sum(acc) / N,
        "median_accuracy": sorted(acc)[N // 2],
        "chance_std": math.sqrt(0.25 / n),
        "buckets": buckets,
        "pooled": {"hits": hits, "total": N * n, "accuracy": hits / (N * n),
                   "p_value": pooled_p, "significant": pooled_p < 0.05},
        "best": {"ticker": max(rows, key=lambda v: v["accuracy"])["ticker"],
                 "accuracy": best, "p_value": p_best,
                 "p_value_corrected": float(1 - (1 - p_best) ** N),
                 "significant": (1 - (1 - p_best) ** N) < 0.05},
    }
