"""FastAPI 서버 — 분석/스캔/AI감사 API + 프론트엔드 서빙."""
import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from . import ai_auditor
from . import data as data_mod
from . import nasdaq100, pipeline

ROOT = pathlib.Path(__file__).resolve().parent.parent
app = FastAPI(title="GAF-VAE Quant", version="2.0")


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/quote")
def api_quote(ticker: str = "AAPL"):
    try:
        return data_mod.quote(ticker)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, str(e))


@app.post("/api/analyze")
def api_analyze(ticker: str = "AAPL", period: str = "5y", force: bool = False):
    """분석 시작(비동기). 상태는 /api/status 로 폴링."""
    return JSONResponse(_strip(pipeline.start_analysis(ticker, period, force)))


@app.get("/api/status")
def api_status(ticker: str = "AAPL"):
    return JSONResponse(_strip(pipeline.get_state(ticker)))


@app.get("/api/result")
def api_result(ticker: str = "AAPL"):
    st = pipeline.get_state(ticker)
    if st.get("status") != "done":
        raise HTTPException(404, "분석 결과가 아직 없습니다")
    return JSONResponse(st["result"])


# ---------- NASDAQ-100 스캔 ----------

@app.post("/api/scan/start")
def api_scan_start(limit: int = 0):
    return JSONResponse(nasdaq100.start_scan(limit=limit))


@app.post("/api/scan/stop")
def api_scan_stop():
    return JSONResponse(nasdaq100.stop_scan())


@app.get("/api/scan/status")
def api_scan_status():
    return JSONResponse(nasdaq100.get_state())


# ---------- AI 코드 감사 ----------

@app.post("/api/audit/start")
def api_audit_start(auto_fix: bool = True):
    return JSONResponse(ai_auditor.start_audit(auto_fix))


@app.get("/api/audit/status")
def api_audit_status():
    return JSONResponse(ai_auditor.get_state())


def _strip(st: dict) -> dict:
    """상태 응답에는 대용량 result 제외."""
    return {k: v for k, v in st.items() if k != "result"} | (
        {"has_result": "result" in st}
    )
