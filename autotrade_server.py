"""자동매매 웹 대시보드 — streamlit_app.py 와 같은 역할을 FastAPI + 정적 프론트로.

실행: python3 autotrade_server.py   (기본 http://127.0.0.1:8877, PORT 로 변경)
봇은 이 프로세스 안에서 백그라운드 스레드로 돈다. 서버를 끄면 자동 실행도 멈춘다.
대시보드는 .env 의 DRY_RUN 과 무관하게 항상 모의로 시작한다. 실주문은 확인 문구를 보내야 켜진다.
"""
import collections
import datetime
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time

import schedule
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import autotrade as at
from toss import shared_client

ROOT = at.ROOT
FRONT = ROOT / "frontend_autotrade" / "index.html"
LOG_PATH = ROOT / "autotrade.log"
LIVE_CONFIRM = "실주문"


class Controller:
    def __init__(self):
        self.busy = False
        self.auto = False
        self.dry_run = True
        self.last_run = None
        self.history = collections.deque(maxlen=20)
        self._lock = threading.Lock()

    def run_once(self):
        with self._lock:
            if self.busy:
                return False
            self.busy = True
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self):
        try:
            at.run_cycle(dry_run=self.dry_run)
        finally:
            self.busy = False
            self.last_run = datetime.datetime.now(at.KST)
            self.history.appendleft(self.last_run)

    def start_auto(self):
        if self.auto:
            return
        self.auto = True
        schedule.clear()
        for t in at.TRADE_TIMES:
            schedule.every().day.at(t).do(self.run_once)

        def loop():
            while self.auto:
                schedule.run_pending()
                time.sleep(1)
        threading.Thread(target=loop, daemon=True).start()

    def stop_auto(self):
        self.auto = False
        schedule.clear()

    def next_run(self):
        return schedule.next_run() if self.auto and schedule.jobs else None


at.initialize_db()
ctl = Controller()
app = FastAPI(title="Claude 자동매매", version="1.0")


def q(sql, params=()):
    with sqlite3.connect(at.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params)]


def _run_row(r):
    try:
        r["candidates"] = json.loads(r["candidates"]) if r.get("candidates") else []
    except (TypeError, ValueError):
        r["candidates"] = []
    return r


@app.get("/")
def index():
    return FileResponse(FRONT)


@app.get("/api/state")
def state():
    nxt = ctl.next_run()
    runs = q("SELECT id, timestamp, dry_run, status, total_value, cash, claude_calls, "
             "input_tokens, output_tokens, cost_usd FROM runs ORDER BY id DESC LIMIT 1")
    tok = q("SELECT COALESCE(SUM(input_tokens),0) i, COALESCE(SUM(output_tokens),0) o, "
            "COALESCE(SUM(cost_usd),0) c, COUNT(*) n FROM runs")[0]
    return {
        "busy": ctl.busy, "auto": ctl.auto, "dry_run": ctl.dry_run,
        "last_run": ctl.last_run.isoformat() if ctl.last_run else None,
        "next_run": nxt.isoformat() if nxt else None,
        "now": datetime.datetime.now(at.KST).isoformat(),
        "config": {"model": at.MODEL, "base_url": at.BASE_URL, "gateway": at.GATEWAY,
                   "top_n": at.TOP_N, "screen_n": at.SCREEN_N, "max_positions": at.MAX_POSITIONS,
                   "max_position_pct": at.MAX_POSITION_PCT, "cash_reserve_pct": at.CASH_RESERVE_PCT,
                   "min_order_usd": at.MIN_ORDER_USD, "trade_times": at.TRADE_TIMES,
                   "env_dry_run": at.DRY_RUN},
        "last": runs[0] if runs else None,
        "totals": {"runs": tok["n"], "input_tokens": int(tok["i"]),
                   "output_tokens": int(tok["o"]), "cost_usd": round(tok["c"], 4)},
    }


@app.get("/api/account")
def account():
    runs = q("SELECT timestamp, total_value, cash FROM runs WHERE total_value IS NOT NULL "
             "ORDER BY id ASC")
    initial = runs[0] if runs else None
    try:
        t = shared_client()
        acct = at.account_state(t)
        try:
            open_orders = (t.orders("OPEN") or {}).get("orders", [])
        except Exception:  # noqa: BLE001
            open_orders = []
        live = True
        err = None
    except Exception as e:  # noqa: BLE001
        last = runs[-1] if runs else None
        acct = {"cash": last["cash"] if last else None,
                "total_value": last["total_value"] if last else None,
                "holdings": {}, "open_orders": []}
        open_orders, live, err = [], False, str(e)
    total = acct.get("total_value")
    ret = (total / initial["total_value"] - 1) * 100 if (initial and total and initial["total_value"]) else None
    started = initial["timestamp"] if initial else None
    return {"live": live, "error": err, "cash": acct["cash"], "total_value": total,
            "holdings": acct["holdings"], "open_orders": open_orders,
            "initial_value": initial["total_value"] if initial else None,
            "started": started, "return_pct": round(ret, 2) if ret is not None else None,
            "runs": len(runs)}


@app.get("/api/runs")
def runs(limit: int = 50):
    return [_run_row(r) for r in q("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))]


@app.get("/api/runs/{run_id}")
def run_detail(run_id: int):
    rows = q("SELECT * FROM runs WHERE id=?", (run_id,))
    if not rows:
        raise HTTPException(404, "no such run")
    return {"run": _run_row(rows[0]),
            "decisions": q("SELECT * FROM trading_decisions WHERE run_id=? ORDER BY id", (run_id,)),
            "orders": q("SELECT * FROM orders WHERE run_id=? ORDER BY id", (run_id,))}


@app.get("/api/decisions")
def decisions(limit: int = 300):
    return q("SELECT * FROM trading_decisions ORDER BY id DESC LIMIT ?", (limit,))


@app.get("/api/orders")
def orders(limit: int = 300):
    return q("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))


@app.get("/api/log")
def log_tail(lines: int = 40):
    if not LOG_PATH.exists():
        return {"lines": []}
    return {"lines": LOG_PATH.read_text(encoding="utf-8").splitlines()[-lines:]}


@app.post("/api/run")
def run_now():
    if not ctl.run_once():
        raise HTTPException(409, "already running")
    return {"ok": True, "dry_run": ctl.dry_run}


class AutoBody(BaseModel):
    on: bool


@app.post("/api/auto")
def set_auto(body: AutoBody):
    ctl.start_auto() if body.on else ctl.stop_auto()
    return {"auto": ctl.auto}


class ModeBody(BaseModel):
    live: bool
    confirm: str = ""


@app.post("/api/mode")
def set_mode(body: ModeBody):
    if ctl.busy:
        raise HTTPException(409, "실행 중에는 모드를 바꿀 수 없습니다")
    if body.live and body.confirm.strip() != LIVE_CONFIRM:
        raise HTTPException(400, f"확인 문구 「{LIVE_CONFIRM}」을 입력해야 실주문이 켜집니다")
    ctl.dry_run = not body.live
    return {"dry_run": ctl.dry_run}


def _port_holder(port):
    """이미 그 포트를 듣고 있는 프로세스가 있으면 (pid, 명령어) 를 돌려준다."""
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                             capture_output=True, text=True, timeout=5).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    pid = out[0]
    try:
        cmd = subprocess.run(["ps", "-p", pid, "-o", "command="],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        cmd = ""
    return pid, cmd


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8877))
    held = _port_holder(port)
    if held:
        pid, cmd = held
        print(f"포트 {port} 를 이미 PID {pid} 가 쓰고 있습니다.\n  {cmd}\n"
              f"기존 서버를 끄려면:  kill {pid}\n"
              f"다른 포트로 띄우려면: PORT=8878 python3 autotrade_server.py")
        sys.exit(1)
    print(f"자동매매 대시보드 → http://127.0.0.1:{port}   (모의 실행으로 시작)")
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)
