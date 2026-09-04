"""FastAPI 서버 — 분석/스캔/매매 API + 프론트엔드 서빙."""
import math
import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from . import ablation
from . import data as data_mod
from . import nasdaq100, pipeline, trader
from .toss import TossClient, TossError

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _json_safe(o):
    """NaN/Infinity 를 null 로 치환 — JSON.parse 가 깨지지 않도록."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o
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
    return JSONResponse(_json_safe(st["result"]))


# ---------- 특징 구성 A/B 비교 ----------

@app.post("/api/ablation/start")
def api_ablation_start(ticker: str = "AAPL", period: str = "5y",
                       force: bool = False):
    """잠재변수만 / 지표만 / 결합 을 같은 walk 로 비교 (비동기)."""
    return JSONResponse(_json_safe(ablation.start(ticker, period, force)))


@app.get("/api/ablation/status")
def api_ablation_status(ticker: str = "AAPL"):
    st = ablation.get_state(ticker)
    return JSONResponse(_json_safe(
        {k: v for k, v in st.items() if k != "result"}
        | {"has_result": bool(st.get("result"))}))


@app.get("/api/ablation/result")
def api_ablation_result(ticker: str = "AAPL"):
    st = ablation.get_state(ticker)
    if not st.get("result"):
        raise HTTPException(404, "비교 결과가 아직 없습니다")
    return JSONResponse(_json_safe(st["result"]))


# ---------- NASDAQ-100 스캔 ----------

@app.post("/api/scan/start")
def api_scan_start(limit: int = 0):
    return JSONResponse(_json_safe(nasdaq100.start_scan(limit=limit)))


@app.post("/api/scan/stop")
def api_scan_stop():
    return JSONResponse(_json_safe(nasdaq100.stop_scan()))


@app.get("/api/scan/status")
def api_scan_status():
    return JSONResponse(_json_safe(nasdaq100.get_state()))


@app.get("/api/scan/stats")
def api_scan_stats():
    """스캔 결과의 통계적 유의성 (우연 대비)."""
    return JSONResponse(nasdaq100.scan_stats())


# ---------- 토스증권 계좌 (읽기 전용) ----------

def _toss():
    try:
        return TossClient()
    except KeyError as e:
        raise HTTPException(503, f"환경변수 미설정: {e}. .env 를 확인하세요")


def _toss_call(fn):
    try:
        return JSONResponse(fn(_toss()))
    except TossError as e:
        raise HTTPException(e.status if e.status < 500 else 502,
                            f"{e.code}: {e}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, str(e))


@app.get("/api/toss/summary")
def api_toss_summary():
    """계좌 + 보유자산 + 매수가능금액을 한 번에."""
    def go(t):
        out = {"accounts": t.accounts(), "account_seq": t.account_seq}
        if t.account_seq:
            out["holdings"] = t.holdings()
            out["buying_power"] = {c: t.buying_power(c) for c in ("USD", "KRW")}
            out["commissions"] = t.commissions()
        return out
    return _toss_call(go)


@app.get("/api/toss/orders")
def api_toss_orders(status: str = "OPEN"):
    return _toss_call(lambda t: t.orders(status=status))


@app.post("/api/toss/orders/{order_id}/cancel")
def api_toss_cancel(order_id: str):
    return _toss_call(lambda t: t.cancel_order(order_id))


# ---------- 매매 계획 ----------

@app.post("/api/trade/plan")
def api_trade_plan_start(ticker: str = "AAPL", force: bool = False):
    """계획 산출 시작(비동기). 상태는 GET /api/trade/plan 으로 폴링."""
    return JSONResponse(trader.start_plan(ticker, force))


@app.get("/api/trade/plan")
def api_trade_plan(ticker: str = "AAPL"):
    """계획 산출 상태 + 완료 시 계획."""
    return JSONResponse(trader.get_plan_state(ticker))


@app.post("/api/trade/execute")
def api_trade_execute(ticker: str = "AAPL", live: bool = False,
                      confirm: str = ""):
    """계획 실행. live=True 는 실계좌 주문 — confirm 에 티커를 정확히 보내야 한다."""
    st = trader.get_plan_state(ticker)
    plan = st.get("plan")
    if st.get("status") != "done" or not plan:
        raise HTTPException(409, "계획을 먼저 산출하세요")
    if live:
        if confirm.strip().upper() != ticker.strip().upper():
            raise HTTPException(400, "실주문 확인 실패: confirm 값이 티커와 다릅니다")
        if not plan["tradable"]:
            raise HTTPException(422,
                "표본 외 예측이 우연과 구분되지 않습니다 (통계적 유의성 없음). "
                "실주문을 거부합니다.")
    try:
        return JSONResponse({"plan": plan, "executed": trader.execute(plan, live=live)})
    except TossError as e:
        raise HTTPException(502, f"{e.code}: {e}")


# ---------- 스캔 → 계획 (포트폴리오) ----------

@app.post("/api/trade/portfolio")
def api_portfolio_start(top: int = 5, force: bool = False,
                        strict: bool = True):
    """스캔 상위 종목으로 계획 산출 시작(비동기)."""
    return JSONResponse(_json_safe(trader.start_portfolio(top, force, strict)))


@app.get("/api/trade/portfolio")
def api_portfolio():
    return JSONResponse(_json_safe(trader.get_portfolio_state()))


@app.post("/api/trade/portfolio/execute")
def api_portfolio_execute(live: bool = False, confirm: str = ""):
    """포트폴리오 체결. live=True 면 confirm 에 정확히 EXECUTE 를 보내야 한다."""
    st = trader.get_portfolio_state()
    if st["status"] != "done":
        raise HTTPException(409, "계획을 먼저 산출하세요")
    tradable = [p for p in st["plans"] if p.get("tradable")]
    if not tradable:
        raise HTTPException(422, "거래 자격 있는 종목이 없습니다 — 주문하지 않습니다.")
    if live and confirm.strip().upper() != "EXECUTE":
        raise HTTPException(400, "실주문 확인 실패: confirm=EXECUTE 필요")
    try:
        return JSONResponse(_json_safe(
            {"tickers": [p["ticker"] for p in tradable],
             "executed": trader.execute_portfolio(live=live)}))
    except TossError as e:
        raise HTTPException(502, f"{e.code}: {e}")


@app.get("/api/trade/plans")
def api_trade_plans(min_acc: float = trader.MIN_SCAN_ACC):
    """산출한 계획을 기준일별로 정리 (스캔 롤링 보정 정확도 min_acc 이상만)."""
    return JSONResponse(_json_safe(trader.plans_by_date(min_acc)))


@app.get("/api/trade/journal")
def api_trade_journal():
    """거래 일지 (dry-run 포함)."""
    if not trader.JOURNAL.exists():
        return JSONResponse([])
    import json as _json
    rows = [_json.loads(l) for l in
            trader.JOURNAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    return JSONResponse(rows[-200:])


def _strip(st: dict) -> dict:
    """상태 응답에는 대용량 result 제외."""
    return _json_safe({k: v for k, v in st.items() if k != "result"} | (
        {"has_result": "result" in st}
    ))
