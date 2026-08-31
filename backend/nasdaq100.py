"""
NASDAQ-100 전체 스캔 — 구성 종목 전체에 GAF·VAE 파이프라인(고속 모드)을
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
from . import gaf, models, vae

# NASDAQ-100 구성 종목 (2026 기준 근사 — 편입/편출 시 수정)
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

_STATE = {"status": "idle", "done": 0, "total": 0, "current": "",
          "results": [], "started_at": None, "stop": False}
_lock = threading.Lock()
CACHE = pathlib.Path(__file__).resolve().parent.parent / "models_cache" / "scan_results.json"


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


def scan_one(ticker: str, period="2y", epochs=8, n_walks=3) -> dict:
    """단일 종목 고속 분석 → 정확도/예측 요약."""
    df = data_mod.history(ticker, period)
    if len(df) < 150:
        raise ValueError("데이터 부족")
    X, y, dates = gaf.build_dataset(df, 20)
    labeled = y >= 0
    Xl, yl = X[labeled], y[labeled]
    dl = [d for d, m in zip(dates, labeled) if m]
    model = vae.train_vae(Xl, epochs=epochs, latent_dim=15)
    Z_all = vae.extract_latents(model, X)
    Zl = Z_all[labeled]
    wf = models.walk_forward(Zl, yl, dl, n_walks=n_walks, test_days=21)
    bundle = models.fit_classifiers(Zl, yl)
    p_up = float(models.predict_proba(bundle, Z_all[-1:])["ensemble"][0])
    cal = bt.calibrate_threshold(wf["per_day"])
    avg = wf["average"]["ensemble"]
    last = float(df["Close"].iloc[-1])
    rets = df["Close"].pct_change().dropna()
    up_m = float(rets[rets >= 0].mean()) if (rets >= 0).any() else 0.0
    dn_m = float(rets[rets < 0].mean()) if (rets < 0).any() else 0.0
    exp_ret = p_up * up_m + (1 - p_up) * dn_m
    return {
        "ticker": ticker,
        "accuracy": avg.get("accuracy", float("nan")),
        "f1": avg.get("f1", float("nan")),
        "auc": avg.get("auc", float("nan")),
        "calibrated_acc": cal["accuracy"],
        "threshold": cal["threshold"],
        "p_up": p_up,
        "direction": "상승" if p_up >= cal["threshold"] else "하락",
        "last_close": last,
        "expected_price": last * (1 + exp_ret),
        "expected_return_pct": exp_ret * 100,
        "as_of": str(df.index[-1].date()),
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
