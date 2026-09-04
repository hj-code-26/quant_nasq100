"""자동매매 대시보드 — 실행 제어 + 기록 (gpt-bitcoin streamlit_app.py 대응).

실행: streamlit run streamlit_app.py
봇은 이 프로세스 안에서 백그라운드 스레드로 돈다. 대시보드를 끄면 자동 실행도 멈춘다.
"""
import collections
import datetime
import sqlite3
import threading
import time

import pandas as pd
import schedule
import streamlit as st

import autotrade as at
from toss import shared_client

st.set_page_config(page_title="Claude 자동매매", layout="wide")


# ---------- 봇 제어 (프로세스당 하나, 재실행 사이에도 유지) ----------
class Controller:
    def __init__(self):
        self.busy = False
        self.auto = False
        self.dry_run = True          # 대시보드는 항상 모의로 시작. 실주문은 토글 + 확인 문구.
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
            import autotrade   # 스트림릿이 모듈을 다시 불러와도 현재 모듈을 쓰도록 매번 import
            autotrade.run_cycle(dry_run=self.dry_run)   # 모드는 전역이 아니라 인자로 명시
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


@st.cache_resource
def controller():
    at.initialize_db()
    return Controller()


ctl = controller()


def q(sql, params=()):
    with sqlite3.connect(at.DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


# ---------- 제어 패널 ----------
st.title("Claude 자동매매")
st.caption(f"모델 {at.MODEL} @ {at.BASE_URL} · 후보 {at.TOP_N} · 최대 {at.MAX_POSITIONS}종목 · "
           f"종목당 {at.MAX_POSITION_PCT:.0f}% · 현금 유지 {at.CASH_RESERVE_PCT:.0f}% · "
           f"실행 시각 {', '.join(at.TRADE_TIMES)} KST")

c = st.columns([1, 1, 1, 2])
if c[0].button("지금 1회 실행", type="primary", disabled=ctl.busy, width="stretch"):
    ctl.run_once()
    st.rerun()
if ctl.auto:
    if c[1].button("자동 실행 중지", width="stretch"):
        ctl.stop_auto()
        st.rerun()
else:
    if c[1].button("자동 실행 시작", width="stretch"):
        ctl.start_auto()
        st.rerun()
if c[2].button("새로고침", width="stretch"):
    st.rerun()

with c[3]:
    # 대시보드는 .env 의 DRY_RUN 과 무관하게 항상 모의로 시작한다. 실주문은 토글 + 확인 문구.
    want_live = st.toggle("실주문 (끄면 모의: 판단·배분만 기록)", value=False, disabled=ctl.busy)
    confirmed = False
    if want_live:
        confirmed = st.text_input("확인 문구 「실주문」 입력", value="", disabled=ctl.busy).strip() == "실주문"
        if confirmed:
            st.error("실주문 모드 — 다음 실행부터 검증을 통과한 주문이 토스 계좌로 나갑니다.")
        else:
            st.warning("확인 문구를 입력해야 실주문이 켜집니다. 지금은 모의 모드.")
    ctl.dry_run = not (want_live and confirmed)
    st.caption("현재 모드: " + ("**실주문**" if not ctl.dry_run else "모의 (DRY_RUN)"))


@st.fragment(run_every="3s")
def status_panel():
    s = st.columns(4)
    s[0].metric("상태", "실행 중…" if ctl.busy else "대기")
    s[1].metric("자동 실행", "켜짐" if ctl.auto else "꺼짐")
    nxt = ctl.next_run()
    s[2].metric("다음 실행", nxt.strftime("%m-%d %H:%M") if nxt else "—")
    s[3].metric("마지막 실행", ctl.last_run.strftime("%m-%d %H:%M") if ctl.last_run else "—")
    log_path = at.ROOT / "autotrade.log"
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").splitlines()[-25:]
        st.code("\n".join(lines), language="text")


status_panel()

# ---------- 계좌 (토스 실시간, 실패하면 마지막 기록) ----------
runs = q("SELECT * FROM runs ORDER BY id DESC")
try:
    t = shared_client()
    items = [i for i in (t.holdings() or {}).get("items") or [] if i.get("currency") == "USD"]
    cash = float(t.buying_power("USD")["cashBuyingPower"])
    value = cash + sum(float(i["marketValue"]["amount"]) for i in items)
    holdings = pd.DataFrame([{
        "종목": i["symbol"], "이름": i.get("name"), "수량": float(i["quantity"]),
        "평단": float(i["averagePurchasePrice"]), "현재가": float(i["lastPrice"]),
        "평가액": float(i["marketValue"]["amount"]),
        "손익%": round(float(i["profitLoss"]["rate"]) * 100, 2)} for i in items])
    live_acct = True
except Exception:  # noqa: BLE001
    if runs.empty:
        st.info("토스 연결 실패, 기록도 없음. .env 를 확인하세요.")
        st.stop()
    last = runs.iloc[0]
    cash, value, holdings, live_acct = last["cash"], last["total_value"], pd.DataFrame(), False

first = runs.dropna(subset=["total_value"]).iloc[-1] if not runs.empty and runs["total_value"].notna().any() else None
initial = float(first["total_value"]) if first is not None else value
started = pd.to_datetime(first["timestamp"]) if first is not None else None
elapsed = pd.Timestamp.now(tz="Asia/Seoul") - started if started is not None else None

st.subheader("계좌")
c = st.columns(4)
c[0].metric("수익률", f"{(value / initial - 1) * 100:+.2f}%" if initial else "—")
c[1].metric("총 자산" + ("" if live_acct else " (마지막 기록)"), f"${value:,.2f}")
c[2].metric("초기 자산", f"${initial:,.2f}")
c[3].metric("투자 기간", f"{elapsed.days}일 {elapsed.seconds // 3600}시간" if elapsed is not None else "—")
c = st.columns(4)
c[0].metric("현금 (USD)", f"${cash:,.2f}")
c[1].metric("보유 종목", f"{len(holdings)}개")
c[2].metric("실행 횟수", f"{len(runs)}회")
c[3].metric("마지막 기록", str(runs.iloc[0]["timestamp"])[:16] if not runs.empty else "—")
if not holdings.empty:
    st.dataframe(holdings, width="stretch", hide_index=True)

if runs.empty:
    st.info("아직 실행 기록이 없습니다. 위의 「지금 1회 실행」을 누르세요.")
    st.stop()

st.subheader("토큰 사용량")
tok = runs.fillna({"input_tokens": 0, "output_tokens": 0, "cost_usd": 0, "claude_calls": 0})
last_tok = tok.iloc[0]
c = st.columns(4)
c[0].metric("누적 입력 토큰", f"{int(tok['input_tokens'].sum()):,}")
c[1].metric("누적 출력 토큰", f"{int(tok['output_tokens'].sum()):,}")
c[2].metric("누적 추정 비용", f"${tok['cost_usd'].sum():,.3f}")
c[3].metric("마지막 실행", f"{int(last_tok['claude_calls'])}회 · "
            f"{int(last_tok['input_tokens']):,} / {int(last_tok['output_tokens']):,} · "
            f"${last_tok['cost_usd']:.3f}")
st.caption("추정 비용은 모델별 Anthropic API 단가 환산값. OmniRoute 구독 경유면 실제 청구는 다를 수 있음.")

st.subheader("실행 기록")
st.dataframe(runs.drop(columns=["id"]).rename(columns={
    "timestamp": "시각", "dry_run": "모의", "total_value": "총자산", "cash": "현금",
    "candidates": "후보", "summary": "배분 요약", "status": "상태", "model": "모델",
    "claude_calls": "호출", "input_tokens": "입력 토큰", "output_tokens": "출력 토큰",
    "cost_usd": "추정 $"}),
    width="stretch", hide_index=True)

st.subheader("주문")
st.dataframe(q("SELECT timestamp, run_id, symbol, side, quantity, amount_usd, price, status, reason "
               "FROM orders ORDER BY id DESC LIMIT 300"), width="stretch", hide_index=True)

st.subheader("종목별 판단")
st.dataframe(q("SELECT timestamp, run_id, symbol, decision, percentage, reason, stock_balance, "
               "avg_buy_price, current_price FROM trading_decisions ORDER BY id DESC LIMIT 500"),
             width="stretch", hide_index=True)
