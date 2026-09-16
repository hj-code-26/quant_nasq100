"""Claude 자동매매 — 나스닥 100 → 후보 선별 → 종목별 판단 → 포트폴리오 배분 → 주문.

3단계 깔때기 (gpt-bitcoin 의 단일 종목 흐름을 여러 종목으로 확장):
  1) 스크리닝  : 나스닥 100 전 종목 일봉 지표로 규칙 점수 → 상위 SCREEN_N 개 표를 Claude 가 보고
                 후보 TOP_N 개 선정 (Claude 호출 1회)
  2) 종목별 판단: 후보 + 현재 보유 종목 각각 시간봉·호가·뉴스까지 붙여 Claude 판단 (병렬)
  3) 배분       : 종목별 판단 + 계좌 상태 + 규칙을 Claude 에 주고 최종 주문 목록 (호출 1회)
                 → 코드가 규칙(최대 종목 수·비중·현금 유지·최소 주문)으로 다시 검증 → 주문

실행:  python autotrade.py            (즉시 1회 + TRADE_TIMES 에 반복. 예약 실행은 정규장 시간에만)
       python autotrade.py --once     (1회만, 장 시간 무시)
설정:  .env 참고. DRY_RUN=1 이면 주문 없이 전 과정을 기록만 한다.
"""
import collections
import concurrent.futures
import datetime
import json
import logging
import os
import pathlib
import sqlite3
import sys
import threading
import time
import zoneinfo

import anthropic
import pandas as pd
import requests
import schedule

from nasdaq100 import TICKERS
from toss import TossError, shared_client

ROOT = pathlib.Path(__file__).resolve().parent
DB_PATH = ROOT / "trading_decisions.db"
KST = zoneinfo.ZoneInfo("Asia/Seoul")
NY = zoneinfo.ZoneInfo("America/New_York")

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
# 추정 단가 ($/백만 토큰, 입력·출력). 게이트웨이 구독 경유면 실제 청구는 0 이지만 API 환산 비용을 보여준다.
PRICES = {"fable": (10, 50), "opus": (5, 25), "sonnet": (2, 10), "haiku": (1, 5)}
# 셸 환경변수가 .env 보다 우선한다. 셸에 ANTHROPIC_BASE_URL 이 이미 있으면 .env 값은 무시되니 시작 로그로 확인.
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
GATEWAY = "api.anthropic.com" not in BASE_URL             # OmniRoute 등 게이트웨이 경유 여부
DRY_RUN = os.environ.get("DRY_RUN", "1") == "1"           # 기본은 모의. 실주문은 DRY_RUN=0
# 실행 시각 (KST): 프리장 개장 17:00 → 정규장 개장 22:30 → 1시간 뒤 23:30 → 00:00 부터 2시간 간격
# → 정규장 마감(05:00) 뒤 애프터장 06:00·08:00.
# 겨울(서머타임 해제)엔 한 시간씩 밀려서 앞선 실행은 "장 시작 전", 마지막은 "장 마감 후"로 자동 건너뛴다.
TRADE_TIMES = [t.strip() for t in
               os.environ.get("TRADE_TIMES",
                              "17:00,22:30,23:30,00:00,02:00,04:00,06:00,08:00").split(",")]
# 사전 분석: 정규장 개장 전에 스크리닝·판단까지만 돌려 로그·DB 에 남긴다 (주문은 안 낸다).
# 소수점 매수가 정규장에만 되니 실주문은 개장 뒤지만, 무엇을 살지는 미리 봐 둔다.
# 서머타임에 따라 정규장 개장이 22:30/23:30 KST 로 밀리므로 두 시각을 다 걸어두고,
# 개장까지 ANALYSIS_LEAD_MIN 분 넘게 남은 쪽은 실행 시점에 스스로 건너뛴다.
ANALYSIS_TIMES = [t.strip() for t in
                  os.environ.get("ANALYSIS_TIMES", "22:00,23:00").split(",") if t.strip()]
ANALYSIS_LEAD_MIN = 60          # ANALYSIS_TIMES 를 바꾸면 이 창도 같이 볼 것
PREMARKET = os.environ.get("PREMARKET", "1") == "1"       # 프리장(17:00~22:30 KST)에도 주문할지
AFTERMARKET = os.environ.get("AFTERMARKET", "1") == "1"   # 애프터장(05:00~09:00 KST)에도 주문할지
PREMARKET_SLIP = float(os.environ.get("PREMARKET_SLIP", 0.5))  # 장외 지정가 버퍼 %

SCREEN_N = int(os.environ.get("SCREEN_N", 40))            # 규칙 점수 상위 몇 개를 Claude 에 보여줄지
TOP_N = int(os.environ.get("TOP_N", 10))                  # Claude 가 고르는 후보 수
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", 10))  # 동시 보유 최대 종목 수 (5→10: 낙폭·집중도 개선)
MAX_POSITION_PCT = float(os.environ.get("MAX_POSITION_PCT", 15))   # 종목당 최대 비중 (총자산 대비 %)
CASH_RESERVE_PCT = float(os.environ.get("CASH_RESERVE_PCT", 10))   # 항상 남겨둘 현금 비중 (%)
MIN_ORDER_USD = float(os.environ.get("MIN_ORDER_USD", 5))
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", 0))    # 평단 대비 -N% 면 전량 매도 (0 이면 끔)
# 기본 꺼짐: -10~-30% 전 구간에서 성적이 나빠졌다. -25%는 검증 Sharpe 1.34→0.95, CAGR 33.0→24.2%
MOMENTUM_EXIT = os.environ.get("MOMENTUM_EXIT", "0") == "1"  # 20일 수익률 음전 시 전량 매도.
# 기본 꺼짐: 같은 백테스트에서 MAX_HOLD_DAYS 만기 청산(승률 56.4%)이 모멘텀 청산(42.1%)보다 우위였다
# 스크리닝~배분(LLM 왕복)이 이 시간을 넘기면 시세·계좌가 낡은 것으로 보고 **신규 진입만** 막는다.
# 위험 축소(손절·만기·노출)는 낡아도 계속 나간다 — 막아야 할 것은 낡은 값으로 사는 일이다.
STALE_MAX_MIN = float(os.environ.get("STALE_MAX_MIN", 20))
WORKERS = int(os.environ.get("WORKERS", 3))               # 종목별 판단 병렬 수 (Claude 속도 제한 고려)
# 아래 셋은 1998~2026 나스닥100 백테스트(탐색 1999~2014 / 검증 2015~2026)에서 나온 값이다.
# 근거는 backtest_bear.py, backtest_slots.py, backtest_regimes.py 참고.
MAX_HOLD_DAYS = int(os.environ.get("MAX_HOLD_DAYS", 20))  # 만기 청산 (거래일). 0 이면 끔
BEAR_INDEX = os.environ.get("BEAR_INDEX", "QQQ")          # 국면 판정용 지수 ETF. 빈 값이면 유니버스 중앙값 사용
BEAR_RET60_PCT = float(os.environ.get("BEAR_RET60_PCT", -3))   # 60일 수익률이 이 밑이면 하락 국면
# 변동성 타겟: 계좌 일별 수익률의 실현 변동성이 목표를 넘으면 주식 노출을 줄인다 (0 이면 끔).
# 신규 매수만 조이는 방식은 슬롯이 늘 차 있어 아무 효과가 없었다 — 보유분을 줄여야 작동한다.
VOL_TARGET_PCT = float(os.environ.get("VOL_TARGET_PCT", 30))
VOL_WINDOW = int(os.environ.get("VOL_WINDOW", 60))         # 실현 변동성 계산 창 (거래일)

# ---------- 오버레이: 하락 국면 노출 축소 (REGIME_DERISK) ----------
# ★ 기본값 OFF. 켜려면 BEAR_EXPOSURE_PCT 에 값을 넣는다 (예: 70). 빈 값이면 기준선 동작.
#   이 변수 하나가 킬 스위치다 — 비우면 오버레이가 전부 사라지고 기존 경로만 남는다.
# 검증: backtest_bear_exposure.py. 판정 **ADOPT_LIMITED** (ADOPT 아님).
#   통과: 방어 효과(전 구간 MDD -63.3%→-47.2%), 보험료(CAGR +0.1%p), Calmar 탐·검 동시 우위,
#         회전율 감소(25.0x→24.4x), 비용 2배에서도 결론 유지
#   미달: 다중검정 보정 후 유의성 없음(DSR 0.086, 기준 0.95), 부트스트랩 ΔMDD 95% 구간이
#         0 을 걸침(-3.6 ~ +18.1%p), 인접 파라미터가 고원이 아니라 뾰족한 봉우리
#   → 사람이 페이퍼·섀도로 확인하기 전에는 켜지 말 것. 권장 검증값은 70 / 15 / 3 이다.
_bx = os.environ.get("BEAR_EXPOSURE_PCT", "").strip()
BEAR_EXPOSURE_PCT = float(_bx) if _bx else None   # 하락 국면 주식 노출 상한 (%). None 이면 끔
BEAR_DD_PCT = float(os.environ.get("BEAR_DD_PCT", 15))     # 지수 252일 고점 대비 이만큼 빠지면 ON
BEAR_OFF_DAYS = int(os.environ.get("BEAR_OFF_DAYS", 3))    # OFF 가 N일 연속돼야 해제 (휩소 방지)
BEAR_CAP = None      # 사이클마다 run_cycle 이 채운다. None = 오버레이 미적용

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # 윈도우 콘솔(cp949)에서 한글 로그가 깨지지 않게
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(ROOT / "autotrade.log", encoding="utf-8")])
log = logging.getLogger("autotrade")

# ---------- Claude 출력 스키마 ----------
CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {
        "type": "object",
        "properties": {"symbol": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["symbol", "reason"], "additionalProperties": False}}},
    "required": ["candidates"], "additionalProperties": False}
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["buy", "sell", "hold"]},
        "percentage": {"type": "integer", "minimum": 0, "maximum": 100},
        "reason": {"type": "string"}},
    "required": ["decision", "percentage", "reason"], "additionalProperties": False}
ALLOCATION_SCHEMA = {   # 매수 금액은 Claude 가 아니라 코드가 정한다 (모멘텀 구간 × 한도)
    "type": "object",
    "properties": {
        "orders": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "sell_pct": {"type": "number", "minimum": 0, "maximum": 100},
                "reason": {"type": "string"}},
            "required": ["symbol", "side", "reason"], "additionalProperties": False}},
        "summary": {"type": "string"}},
    "required": ["orders", "summary"], "additionalProperties": False}


# ---------- DB ----------
def initialize_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, dry_run INTEGER,
            total_value REAL, cash REAL, candidates TEXT, summary TEXT, status TEXT);
        CREATE TABLE IF NOT EXISTS trading_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, timestamp TEXT, symbol TEXT,
            decision TEXT, percentage INTEGER, reason TEXT, stock_balance REAL,
            usd_balance REAL, avg_buy_price REAL, current_price REAL);
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, timestamp TEXT, symbol TEXT,
            side TEXT, quantity REAL, amount_usd REAL, price REAL, order_id TEXT,
            status TEXT, reason TEXT);
        CREATE TABLE IF NOT EXISTS equity (
            date TEXT PRIMARY KEY, total_value REAL, timestamp TEXT);""")
        ecols = [r[1] for r in conn.execute("PRAGMA table_info(equity)")]
        for col, typ in (("stock_value", "REAL"), ("cashflow", "REAL")):
            if col not in ecols:
                conn.execute(f"ALTER TABLE equity ADD COLUMN {col} {typ}")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(trading_decisions)")]
        if "run_id" not in cols:   # 단일 종목 버전 DB 호환
            conn.execute("ALTER TABLE trading_decisions ADD COLUMN run_id INTEGER")
        rcols = [r[1] for r in conn.execute("PRAGMA table_info(runs)")]
        for col, typ in (("model", "TEXT"), ("claude_calls", "INTEGER"),
                         ("input_tokens", "INTEGER"), ("output_tokens", "INTEGER"),
                         ("cost_usd", "REAL")):
            if col not in rcols:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        # 고아 run 마감 — status='running' 인 채로 남은 것은 프로세스가 죽은 것이다
        # (2026-09-16 재부팅으로 run 64 가 영구 running, 04·06·08시 사이클이 통째로 소실됐다).
        # 이 프로세스가 유일한 실행자라는 가정 위에서만 맞다 — 여러 인스턴스를 동시에 돌리면
        # 살아 있는 run 을 마감해 버린다. 그래서 상태만 바꾸고 주문은 건드리지 않는다.
        orphans = [r[0] for r in conn.execute("SELECT id FROM runs WHERE status='running'")]
        if orphans:
            conn.execute("UPDATE runs SET status='interrupted' WHERE status='running'")
            log.warning("이전 실행이 비정상 종료됨 — run %s 를 interrupted 로 마감한다. "
                        "미체결 주문은 계좌 조회로 다시 확인된다", orphans)


def log_equity(total_value, stock_value=None):
    """하루에 한 줄, 그 날 마지막 총자산과 주식 평가액. 변동성 타겟이 쓰는 유일한 이력이다.

    cashflow(그 날의 입금+/출금-)는 **코드가 채우지 않는다** — 토스 Open API 에 입출금
    조회가 없어서 자동 판별이 불가능하다. 입출금·환전을 했으면 그 날 행의 cashflow 를
    직접 채워야 수익률이 오염되지 않는다. 안 채우면 exposure_cap 이 큰 점프를 경고한다.
    """
    if not total_value:
        return
    today = datetime.datetime.now(NY).date().isoformat()   # 미국 거래일 기준
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO equity (date, total_value, stock_value, timestamp) "
                     "VALUES (?, ?, ?, ?) "
                     "ON CONFLICT(date) DO UPDATE SET total_value=excluded.total_value, "
                     "stock_value=excluded.stock_value, timestamp=excluded.timestamp",
                     (today, float(total_value),
                      None if stock_value is None else float(stock_value), _now()))


def exposure_cap(verbose=False):
    """주식에 둘 수 있는 최대 비중(0~1). 제약이 하나도 없으면 None.

    변동성 타겟(후행)과 하락 국면 오버레이(선행) 중 **더 낮은 쪽**을 쓴다. 둘은 겹쳐도
    범위를 벗어나지 않는다 — min 이므로 0~100% 안에 갇힌다.
    """
    caps = [c for c in (_vol_target_cap(verbose), BEAR_CAP) if c is not None]
    return min(caps) if caps else None


def bear_data_ok(df):
    """오버레이 신호를 **판정할 수 있는가**. 252일 고점 + 당일 = 253봉이 필요하다.

    신호 OFF 와 판정 불가를 섞지 않으려고 따로 뒀다 — 지수를 못 받은 날 조용히 꺼지면
    보호가 사라진 것을 로그에서 알아볼 수 없다.
    """
    return df is not None and len(df) >= 253


def bear_derisk(df):
    """하락 국면 노출 축소 오버레이 — ON 이면 허용 주식 비중(0~1), 아니면 None.

    신호: 지수가 252거래일 고점 대비 BEAR_DD_PCT% 이상 빠졌으면 ON.
          OFF 가 BEAR_OFF_DAYS 일 연속돼야 해제한다 (= 마지막 N일 중 하나라도 ON 이면 유지).
    마지막 봉까지의 데이터만 본다 — 구조상 미래를 참조할 수 없다
    (backtest_bear_exposure.check() 가 '데이터를 t 에서 잘라도 신호 불변'을 매 실행 검증한다).

    현행 market_regime(60일 수익률 < -3%) 을 쓰지 않는 이유: 그 신호는 28년 중 ON 인 날만
    모으면 지수가 **+41%** 다 (방어할 게 없는 구간에 켜진다). 이 신호는 -55% 다.
    """
    if BEAR_EXPOSURE_PCT is None or not bear_data_ok(df):
        return None
    c = df["close"]
    dd = (c / c.rolling(252).max() - 1) * 100
    on = bool((dd <= -BEAR_DD_PCT).tail(max(BEAR_OFF_DAYS, 1)).any())
    return BEAR_EXPOSURE_PCT / 100 if on else None


def _vol_target_cap(verbose=False):
    """변동성 타겟 상한(0~1). 자산 이력이 모자라면 None (기능 비활성).

    ★ 계좌 변동성에는 **이미 현금 비중이 섞여 있다.** target/account_vol 을 그대로 절대
      상한으로 쓰면 되먹임이 뒤집힌다: 현금이 많은 날일수록 계좌 변동성이 낮아 상한이
      커지고, 노출을 키우면 다시 변동성이 뛰어 상한이 줄어드는 진동이 생긴다.
      (변동성 60% 자산·목표 30%: 50% 투자 → 계좌 30% → 상한 100% → 다음엔 계좌 60% →
       상한 50% → …)
      그래서 계좌 변동성을 같은 창의 **평균 주식 비중**으로 나눠 주식 자체의 변동성을
      복원한 뒤 목표와 비교한다. 위 예에서는 상한이 50% 로 고정되어 진동하지 않는다.
      (엄밀하게는 목표 포트폴리오 w 의 sqrt(w'Σw) 를 써야 하지만, 계좌 이력만으로는
       Σ 를 복원할 수 없다. 단일 자산 근사임을 명시한다.)

    입출금은 equity.cashflow 에서 빼고 계산한다 — 안 채워 두면 입금이 그대로 '수익률'로
    잡혀 변동성이 부풀고 노출이 근거 없이 잘린다. 큰 점프는 경고를 남긴다.
    """
    if VOL_TARGET_PCT <= 0:
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = list(conn.execute(
                "SELECT date, total_value, stock_value, COALESCE(cashflow, 0) FROM "
                "(SELECT date, total_value, stock_value, cashflow FROM equity "
                "ORDER BY date DESC LIMIT ?) ORDER BY date", (VOL_WINDOW + 1,)))
    except sqlite3.OperationalError:      # 첫 실행 — equity 테이블이 아직 없다
        return None
    if len(rows) < VOL_WINDOW + 1:
        return None
    rets = []
    for (_, v0, _, _), (d1, v1, _, cf1) in zip(rows, rows[1:]):
        if not v0:
            continue
        r = (v1 - cf1) / v0 - 1
        if abs(r) > 0.25:                 # 하루 ±25% — 입출금·환전 미기재를 의심한다
            log.warning("자산 이력 %s: 하루 %.1f%% 변동. 입출금이면 equity.cashflow 에 적어야 "
                        "변동성이 오염되지 않는다", d1, r * 100)
        rets.append(r)
    if len(rets) < 2:
        return None
    vol = (sum((r - sum(rets) / len(rets)) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
    vol_ann = vol * (252 ** 0.5) * 100
    if vol_ann <= 0:
        return None
    ws = [sv / tv for _, tv, sv, _ in rows if tv and sv is not None]
    if not ws:                            # stock_value 를 안 남기던 옛 행뿐 — 계좌=주식 가정
        log.warning("변동성 타겟: 주식 비중 이력이 없어 계좌 변동성을 그대로 쓴다 (상한이 느슨해짐)")
        w = 1.0
    else:
        w = max(0.05, min(1.0, sum(ws) / len(ws)))   # 0 근처에서 상한이 발산하지 않게 하한
    asset_vol = vol_ann / w               # 현금 희석을 되돌린 '주식 부분'의 변동성
    cap = min(1.0, VOL_TARGET_PCT / asset_vol)
    if verbose:
        log.info("변동성 타겟: 계좌 실현 %.1f%% ÷ 평균 주식비중 %.0f%% = 주식 %.1f%% / 목표 %.0f%% "
                 "→ 주식 노출 상한 %.0f%%", vol_ann, w * 100, asset_vol, VOL_TARGET_PCT, cap * 100)
    return cap


def _now():
    return datetime.datetime.now(KST).isoformat(timespec="seconds")


def db_insert(table, row):
    keys = ", ".join(row)
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(f"INSERT INTO {table} ({keys}) VALUES ({', '.join('?' * len(row))})",
                           list(row.values()))
        return cur.lastrowid


def db_update_run(run_id, **fields):
    sets = ", ".join(f"{k}=?" for k in fields)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"UPDATE runs SET {sets} WHERE id=?", [*fields.values(), run_id])


def filled_orders(toss):
    """계좌의 체결 완료 주문 (토스 status=CLOSED). 실패하면 빈 목록."""
    try:
        rows = (toss.orders("CLOSED") or {}).get("orders") or []
    except Exception as e:  # noqa: BLE001
        log.warning("체결 이력 조회 실패: %s", e)
        return []
    return [o for o in rows
            if (o.get("execution") or {}).get("filledQuantity") and o.get("orderedAt")]


def position_entry_map(toss=None):
    """{종목: 현 보유분을 처음 산 시각(ISO)}.

    1순위는 토스 체결 이력이다 — **계좌 기준이라 다른 컴퓨터에서 낸 주문도 보인다.**
    수량을 시간순으로 누적해 0 → 양수로 바뀐 시점을 진입으로 잡으므로 분할 매도도 처리된다.
    토스 조회가 실패하면 이 컴퓨터의 orders 표로 대체한다(로컬 주문만 보인다).
    """
    out = {}
    for o in sorted(filled_orders(toss) if toss else [], key=lambda x: x["orderedAt"]):
        sym = str(o.get("symbol", "")).upper()
        qty, opened = out.get(sym, (0.0, None))
        f = float(o["execution"]["filledQuantity"])
        if str(o.get("side", "")).upper() == "BUY":
            if qty <= 1e-9:
                opened = o["orderedAt"]
            qty += f
        else:
            qty -= f
            if qty <= 1e-9:
                qty, opened = 0.0, None
        out[sym] = (qty, opened)
    entries = {sym: opened for sym, (qty, opened) in out.items() if opened}
    if entries:
        return entries
    with sqlite3.connect(DB_PATH) as conn:       # 대체: 로컬 DB
        rows = conn.execute(
            """SELECT symbol, timestamp, side FROM orders
               WHERE status IN ('submitted','filled') ORDER BY id""").fetchall()
    local = {}
    for sym, ts, side in rows:
        if side == "buy":
            local.setdefault(sym, ts)
        elif side == "sell":
            local.pop(sym, None)
    return local


# 실제 거래일 달력 (미국). run_cycle 이 지수 일봉에서 채운다 — 휴장일·조기폐장이 반영된
# 유일한 출처다. 비어 있으면 주말만 빼는 근사로 떨어진다 (만기가 최대 며칠 일찍 온다).
TRADING_DAYS = []


def trading_days_since(ts, calendar=None):
    """ts(ISO) 이후 지나간 미국 거래일 수.

    calendar 가 있으면(=거래소 일봉이 존재하는 날짜) 공휴일·조기폐장이 그대로 반영된다.
    없으면 주말만 빼는 근사라 휴장일마다 만기가 하루씩 앞당겨진다.

    lot age 정책: '현 보유분을 처음 산 시각'부터 센다(position_entry_map). 추가 매수는
    시계를 리셋하지 않고(=최초 진입 기준), 전량 청산 후 재진입하면 새로 시작한다.
    부분 매도는 남은 수량의 나이를 유지한다.
    """
    if not ts:
        return None
    try:
        start = datetime.datetime.fromisoformat(ts).astimezone(NY).date()
    except ValueError:
        return None
    cal = TRADING_DAYS if calendar is None else calendar
    today = datetime.datetime.now(NY).date()
    if cal:
        return sum(1 for d in cal if start < d <= today)
    days = 0
    cur = start
    while cur < today:
        cur += datetime.timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


def fetch_last_decisions(symbol, num=10):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""SELECT timestamp, decision, percentage, reason,
            stock_balance, usd_balance, avg_buy_price, current_price
            FROM trading_decisions WHERE symbol=? ORDER BY id DESC LIMIT ?""",
            (symbol, num)).fetchall()
    keys = ["timestamp", "decision", "percentage", "reason",
            "stock_balance", "usd_balance", "avg_buy_price", "current_price"]
    return [dict(zip(keys, r)) for r in rows]


# ---------- 토스 데이터 ----------
def candles(toss, symbol, interval, count):
    """토스 캔들 → DataFrame. 200개 넘으면 nextBefore 로 페이지를 잇는다."""
    rows, before = [], None
    while len(rows) < count:
        r = toss.candles(symbol, interval=interval,
                         count=min(200, count - len(rows)), before=before) or {}
        page = r.get("candles") or []
        if not page:
            break
        rows += page
        before = r.get("nextBefore")
        if not before:
            break
    if not rows:
        raise ValueError(f"{symbol}: 캔들 없음")
    df = pd.DataFrame(rows).rename(columns={
        "openPrice": "open", "highPrice": "high", "lowPrice": "low", "closePrice": "close"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _rma(s, n):
    """Wilder 평활 (RSI 용)."""
    return s.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df):
    """SMA10/20/50, EMA10, RSI14, 스토캐스틱(14,3,3), MACD(12,26,9), 볼린저(20,2).
    컬럼명은 pandas_ta 판본과 동일하게 유지한다 (pandas_ta 는 py3.12+ 전용이라 직접 계산)."""
    c, h, l = df["close"], df["high"], df["low"]
    for n in (10, 20, 50):
        df[f"SMA_{n}"] = c.rolling(n).mean()
    df["EMA_10"] = c.ewm(span=10, adjust=False).mean()

    d = c.diff()
    df["RSI_14"] = 100 - 100 / (1 + _rma(d.clip(lower=0), 14) / _rma(-d.clip(upper=0), 14))

    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    df["STOCHk_14_3_3"] = (100 * (c - ll) / (hh - ll)).rolling(3).mean()
    df["STOCHd_14_3_3"] = df["STOCHk_14_3_3"].rolling(3).mean()

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    df["MACD_12_26_9"], df["MACDs_12_26_9"], df["MACDh_12_26_9"] = macd, sig, macd - sig

    mid, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=1)
    lower, upper = mid - 2 * sd, mid + 2 * sd
    df["BBL_20_2.0_2.0"], df["BBM_20_2.0_2.0"], df["BBU_20_2.0_2.0"] = lower, mid, upper
    df["BBP_20_2.0_2.0"] = (c - lower) / (upper - lower)
    return df


def records(df, fmt):
    out = df.round(4).reset_index()
    out["timestamp"] = out["timestamp"].dt.tz_convert(KST).dt.strftime(fmt)
    out = out.astype(object).where(out.notna(), None)   # NaN → null
    return out.to_dict(orient="records")


def fetch_and_prepare_data(toss, symbol):
    """일봉 30개 + 시간봉 24개(1분봉을 묶음). 토스 1분봉은 약 1.3일치만 제공된다."""
    daily = add_indicators(candles(toss, symbol, "1d", 90)).tail(30)
    m1 = candles(toss, symbol, "1m", 2000)
    hourly = m1.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"}).dropna()
    hourly = add_indicators(hourly).tail(24)
    return {"daily_ohlcv": records(daily, "%Y-%m-%d"),
            "hourly_ohlcv": records(hourly, "%Y-%m-%d %H:%M")}


def get_news_data(symbol):
    """SerpApi 구글 뉴스 (SERPAPI_API_KEY 없으면 생략)."""
    key = os.environ.get("SERPAPI_API_KEY")
    if not key:
        return []
    try:
        r = requests.get("https://serpapi.com/search.json", params={
            "engine": "google_news", "q": f"{symbol} stock", "api_key": key}, timeout=15)
        r.raise_for_status()
        return [{"title": n.get("title"), "source": (n.get("source") or {}).get("name"),
                 "date": n.get("date")} for n in r.json().get("news_results", [])[:15]]
    except Exception as e:  # noqa: BLE001
        log.warning("%s 뉴스 수집 실패: %s", symbol, e)
        return []


def account_state(toss):
    """현금·보유·총자산·미체결 주문 종목."""
    items = (toss.holdings() or {}).get("items") or []
    holdings = {}
    for i in items:
        if i.get("currency") != "USD":
            continue
        holdings[i["symbol"]] = {
            "name": i.get("name"), "quantity": float(i["quantity"]),
            "avg_price": float(i["averagePurchasePrice"]), "last_price": float(i["lastPrice"]),
            "market_value": float(i["marketValue"]["amount"]),
            "pnl_pct": round(float(i["profitLoss"]["rate"]) * 100, 2)}
    cash = float(toss.buying_power("USD")["cashBuyingPower"])
    try:
        open_orders = sorted({o.get("symbol") for o in
                              (toss.orders("OPEN") or {}).get("orders", []) if o.get("symbol")})
    except TossError as e:
        log.warning("미체결 조회 실패: %s", e)
        open_orders = []
    return {"cash": round(cash, 2), "holdings": holdings, "open_orders": open_orders,
            "total_value": round(cash + sum(h["market_value"] for h in holdings.values()), 2)}


def market_session(toss):
    """오늘(미국 날짜) 거래 가능 구간 (거래 시작, 정규장 시작, 정규장 종료, 거래 종료) KST. 휴장이면 None.
    PREMARKET 이면 거래 시작 = 프리장 개장, AFTERMARKET 이면 거래 종료 = 애프터장 마감."""
    us_today = datetime.datetime.now(NY).date().isoformat()
    day = (toss.us_market_calendar(date=us_today) or {}).get("today") or {}
    reg = day.get("regularMarket")
    if not reg:
        return None
    def kst(v):
        return datetime.datetime.fromisoformat(v).astimezone(KST)
    start, end = kst(reg["startTime"]), kst(reg["endTime"])
    pre = (day.get("preMarket") or {}).get("startTime")
    post = (day.get("postMarket") or {}).get("endTime")
    return (kst(pre) if PREMARKET and pre else start, start, end,
            kst(post) if AFTERMARKET and post else end)


def fractional_allowed(session):
    """금액 기반 매수·소수점 매도는 정규장 시작 ~ 종료 1시간 전까지만 접수된다."""
    if not session:
        return False
    now = datetime.datetime.now(KST)
    return session[1] <= now <= session[2] - datetime.timedelta(hours=1)


def analysis_lead_min(session):
    """정규장 개장까지 남은 분. 사전 분석 창(개장 전 ANALYSIS_LEAD_MIN 분 안) 밖이면 None."""
    if not session:
        return None
    left = (session[1] - datetime.datetime.now(KST)).total_seconds() / 60
    return left if 0 < left <= ANALYSIS_LEAD_MIN else None


def extended_hours(session):
    """프리장·애프터장 구간이면 True — 시장가가 아니라 지정가로 낸다."""
    if not session:
        return False
    now = datetime.datetime.now(KST)
    return now < session[1] or now > session[2]


def session_block(session):
    """지금 실행하면 안 되는 이유. 거래 시작 5분 전 ~ 거래 종료 사이면 None(실행 가능).
    session[-1] 이라 프리장만 쓰던 3원소 구간도 그대로 받는다."""
    if not session:
        return "휴장일"
    now = datetime.datetime.now(KST)
    if now < session[0] - datetime.timedelta(minutes=5):
        return f"장 시작 전 (개장 {session[0]:%H:%M} KST)"
    if now > session[-1]:
        return f"장 마감 후 (마감 {session[-1]:%H:%M} KST)"
    return None


# ---------- Claude 토큰 집계 ----------
_usage_lock = threading.Lock()
_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def usage_reset():
    with _usage_lock:
        _usage.update(calls=0, input_tokens=0, output_tokens=0)


def usage_add(resp):
    u = resp.usage
    inp = (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) \
        + (getattr(u, "cache_creation_input_tokens", 0) or 0)
    with _usage_lock:
        _usage["calls"] += 1
        _usage["input_tokens"] += inp
        _usage["output_tokens"] += u.output_tokens or 0
    log.info("Claude 호출: 입력 %s 출력 %s 토큰", f"{inp:,}", f"{u.output_tokens:,}")


def usage_snapshot():
    with _usage_lock:
        d = dict(_usage)
    d["cost_usd"] = estimate_cost(MODEL, d["input_tokens"], d["output_tokens"])
    return d


def estimate_cost(model, input_tokens, output_tokens):
    tier = next((p for k, p in PRICES.items() if k in model.lower()), (5, 25))
    return round(input_tokens / 1e6 * tier[0] + output_tokens / 1e6 * tier[1], 4)


# ---------- Claude ----------
def check_schema(out, schema, path="응답"):
    """받은 JSON 이 스키마를 실제로 지키는지 본다 (필수 키 존재 확인만으로는 부족하다).

    게이트웨이(OmniRoute 등) 경유일 때는 서버 측 json_schema 강제를 못 걸어서
    raw_decode 결과가 무검증으로 흘러든다. 임의 필드, 문자열 숫자, NaN/Inf,
    범위 밖 비율, enum 밖 side 를 여기서 막는다.
    """
    t = schema.get("type")
    if t == "object":
        if not isinstance(out, dict):
            raise ValueError(f"{path}: 객체가 아님")
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in out:
                raise ValueError(f"{path}.{k} 없음")
        if schema.get("additionalProperties") is False:
            for k in out:
                if k not in props:
                    raise ValueError(f"{path}.{k}: 스키마에 없는 필드")
        for k, v in out.items():
            if k in props:
                check_schema(v, props[k], f"{path}.{k}")
    elif t == "array":
        if not isinstance(out, list):
            raise ValueError(f"{path}: 배열이 아님")
        for i, v in enumerate(out):
            check_schema(v, schema["items"], f"{path}[{i}]")
    elif t == "string":
        if not isinstance(out, str):
            raise ValueError(f"{path}: 문자열이 아님")
        if "enum" in schema and out not in schema["enum"]:
            raise ValueError(f"{path}: {out!r} 는 {schema['enum']} 중 하나가 아님")
    elif t in ("number", "integer"):
        if isinstance(out, bool) or not isinstance(out, (int, float)):
            raise ValueError(f"{path}: 숫자가 아님 ({out!r})")
        if out != out or out in (float("inf"), float("-inf")):
            raise ValueError(f"{path}: NaN/Inf")
        if t == "integer" and float(out) != int(out):
            raise ValueError(f"{path}: 정수가 아님 ({out!r})")
        if "minimum" in schema and out < schema["minimum"]:
            raise ValueError(f"{path}: {out} < 최소 {schema['minimum']}")
        if "maximum" in schema and out > schema["maximum"]:
            raise ValueError(f"{path}: {out} > 최대 {schema['maximum']}")
    return out


def ask_claude(prompt_file, payload, schema, retries=1):
    """instructions 파일을 시스템 프롬프트로, payload(dict) 를 사용자 메시지로 보내 JSON 을 받는다."""
    client = anthropic.Anthropic(timeout=300, max_retries=2)
    system = (ROOT / prompt_file).read_text(encoding="utf-8")
    user = "\n\n".join(f"## {k}\n{json.dumps(v, ensure_ascii=False)}" for k, v in payload.items())
    user += ("\n\n## 출력 형식\n아래 JSON 스키마를 만족하는 JSON 객체 하나만 출력한다.\n"
             + json.dumps(schema, ensure_ascii=False))
    common = dict(model=MODEL, max_tokens=16000, system=system,
                  messages=[{"role": "user", "content": user}])
    last_err = None
    for attempt in range(retries + 1):
        try:
            if GATEWAY:
                # 게이트웨이는 베타 파라미터(대체 모델·JSON 스키마 강제)를 안 넘길 수 있어 뺀다
                resp = client.messages.create(**common)
            elif "opus" in MODEL or "fable" in MODEL:
                # 안전 분류기 거부 시 서버 측 대체 모델 (Opus/Fable 계열)
                resp = client.beta.messages.create(
                    **common, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                    output_config={"format": {"type": "json_schema", "schema": schema}})
            else:
                resp = client.messages.create(
                    **common, output_config={"format": {"type": "json_schema", "schema": schema}})
            usage_add(resp)
            if resp.stop_reason == "refusal":
                raise RuntimeError(f"Claude 응답 거부: {resp.stop_details}")
            text = next(b.text for b in resp.content if b.type == "text")
            # 첫 JSON 객체만 읽는다 — 뒤에 설명 문장이나 두 번째 블록이 붙어도 무시
            start = text.find("{")
            if start < 0:
                raise ValueError("JSON 객체가 없음")
            out, _ = json.JSONDecoder().raw_decode(text[start:])
            return check_schema(out, schema)
        except (json.JSONDecodeError, ValueError, StopIteration) as e:
            last_err = e
            log.warning("Claude 응답 파싱 실패 (%d/%d): %s", attempt + 1, retries + 1, e)
    raise RuntimeError(f"Claude 응답 파싱 실패: {last_err}")


# ---------- 1단계: 스크리닝 ----------
# 백테스트(backtest.py, 나스닥 101종목 × 3년, 탐색/검증 분할) 결과:
#   · 이평 정배열·MACD·RSI 적정·볼린저 적정 같은 교과서 조건은 기준선과 차이 없음
#   · 20일 수익률 상위 5종목을 매일 뽑아 20일 보유 → 유니버스 대비 +2.7%p(탐색) / +4.7%p(검증)
#   · 20일 수익률 > 20% 구간: 20일 뒤 +4.6% / +7.7% (기준선 +1.7% / +2.0%)
#   · "과열 제외(RSI<70)"와 "눌림 우선"은 모두 성적을 깎았다
# 그래서 선별은 20일 모멘텀 순, 사이즈는 모멘텀 구간별로 코드가 정한다. Claude 는 뉴스·정성 거부권.
MOMENTUM_TIERS = (   # (20일 수익률 하한 %, 포지션 크기 배수, 이름) — 백테스트 20일 뒤 평균으로 나눔
    (20.0, 1.0, "강(>20%: 20일 뒤 +4.6~7.7%)"),
    (10.0, 0.7, "중(10~20%: +2.1~2.6%)"),
    (0.0, 0.4, "약(0~10%: +1.0~1.4%, 기준선 이하)"),
)


def index_daily(toss):
    """국면 판정·거래일 달력에 쓰는 지수 일봉. 한 사이클에 한 번만 받는다.

    300봉을 받는다 — bear_derisk 의 252일 고점이 그만큼 필요하다. market_regime 은
    뒤 61봉만 쓰므로 더 받아도 판정이 달라지지 않는다 (토스 상한은 750봉).
    """
    if not BEAR_INDEX:
        return None
    try:
        return candles(toss, BEAR_INDEX, "1d", 300)
    except Exception as e:  # noqa: BLE001
        log.warning("지수 %s 캔들 실패: %s", BEAR_INDEX, e)
        return None


def market_regime(toss, rows=None, df=None):
    """시장 국면. 지수(기본 QQQ) 60일 수익률이 BEAR_RET60_PCT 미만이면 '하락'.

    QQQ 는 ^NDX 를 사실상 그대로 따라간다 (1999~2026 60일 수익률 상관 0.9996,
    -3% 판정 일치율 99.6%). 지수를 못 받으면 유니버스 60일 수익률 중앙값으로 대신한다
    (지수 판정과 90.7% 일치하지만 성적은 더 낮으므로 어디까지나 대비책이다).
    """
    ret60 = src = None
    if df is None and BEAR_INDEX:
        df = index_daily(toss)
    if df is not None and len(df) >= 61:
        ret60 = (df["close"].iloc[-1] / df["close"].iloc[-61] - 1) * 100
        src = BEAR_INDEX
    if ret60 is None and rows:
        vals = sorted(r["ret_60d_pct"] for r in rows if r.get("ret_60d_pct") is not None)
        if vals:
            ret60 = vals[len(vals) // 2]
            src = "유니버스 중앙값"
    if ret60 is None:
        return {"regime": "보통", "ret_60d_pct": None, "source": "판정 실패 — 기본값"}
    regime = "하락" if ret60 < BEAR_RET60_PCT else "보통"
    return {"regime": regime, "ret_60d_pct": round(float(ret60), 2), "source": src}


def momentum_tier(ret_20d, regime="보통"):
    """포지션 크기 배수. 하락 국면에서는 선별 기준이 모멘텀이 아니라 변동성이므로
    모멘텀 부호로 매수를 막지 않는다 (막으면 저변동성 후보의 약 47% 가 잘려나가고,
    깊은 하락장에서는 전 종목이 잘려 자동으로 현금 100% 가 된다 — 검증에서 가장 나빴던 상태)."""
    if regime == "하락":
        return 1.0, "하락 국면(저변동성 선별 — 모멘텀 무관)"
    for lo, factor, name in MOMENTUM_TIERS:
        if ret_20d is not None and ret_20d >= lo:
            return factor, name
    return 0.0, "음(<0%: 매수 안 함)"


def rule_score(df):
    """(구버전) 교과서 규칙 점수. 백테스트에서 예측력이 없어 선별에는 더 쓰지 않는다. backtest.py 비교용."""
    c = df.iloc[-1]
    s = 0.0
    s += 1 if c["close"] > c["SMA_20"] else 0
    s += 1 if c["SMA_20"] > c["SMA_50"] else 0
    s += 1 if df["close"].iloc[-1] > df["close"].iloc[-21] else 0
    s += 1 if c.get("MACDh_12_26_9", 0) > 0 else 0
    rsi = c["RSI_14"]
    s += 1 if 40 <= rsi <= 65 else (-1 if rsi > 75 else 0)
    bbp = c.get("BBP_20_2.0_2.0")
    if bbp is not None and not pd.isna(bbp):
        s += 1 if 0.2 <= bbp <= 0.85 else (-1 if bbp > 1 else 0)
    vol_ratio = c["volume"] / max(df["volume"].tail(20).mean(), 1)
    s += 0.5 if vol_ratio > 1.2 else 0
    return round(s, 1)


def screen(toss, universe=TICKERS, regime="보통"):
    """전 종목 일봉 요약. 평상시엔 20일 모멘텀 내림차순.

    하락 국면에서는 **일중 변동폭(atr_pct) 오름차순**으로 바꾼다. 하락장에서 모멘텀 상위는
    유니버스 대비 -0.71%p 로 엣지가 뒤집히는 반면, 저변동성은 진입 승률이
    50.2%→58.4%(탐색) / 61.6%→68.8%(검증), -10% 넘는 손실 비율이 21.2%→8.1% / 14.4%→4.0%
    로 개선된다 (backtest_bear.py).
    """
    rows = []
    for sym in universe:
        try:
            df = add_indicators(candles(toss, sym, "1d", 90))
            if len(df) < 60:
                continue
            c = df.iloc[-1]
            ret20 = (c["close"] / df["close"].iloc[-21] - 1) * 100
            factor, tier = momentum_tier(ret20, regime)
            rows.append({
                "symbol": sym, "close": round(c["close"], 2),
                "ret_20d_pct": round(ret20, 2), "momentum_tier": tier, "size_factor": factor,
                "ret_5d_pct": round((c["close"] / df["close"].iloc[-6] - 1) * 100, 2),
                "ret_60d_pct": round((c["close"] / df["close"].iloc[-61] - 1) * 100, 2),
                "vs_sma20_pct": round((c["close"] / c["SMA_20"] - 1) * 100, 2),
                "rsi14": round(c["RSI_14"], 1),
                "bb_pct": round(c.get("BBP_20_2.0_2.0", float("nan")), 2),
                "vol_ratio_20d": round(c["volume"] / max(df["volume"].tail(20).mean(), 1), 2),
                "atr_pct": round((df["high"] - df["low"]).tail(14).mean() / c["close"] * 100, 2)})
        except Exception as e:  # noqa: BLE001
            log.warning("스크리닝 %s 실패: %s", sym, e)
    if regime == "하락":
        rows.sort(key=lambda r: (r["atr_pct"] is None, r["atr_pct"]))
    else:
        rows.sort(key=lambda r: r["ret_20d_pct"], reverse=True)
    return [{k: (None if isinstance(v, float) and pd.isna(v)
                 else float(v) if isinstance(v, float) else v) for k, v in r.items()}
            for r in rows]


def pick_candidates(rows, account, regime="보통"):
    """상위 SCREEN_N 개 표를 주고 Claude 가 TOP_N 개를 고른다 (거부권 + 분산).

    ★ 표의 정렬 기준은 국면마다 다르다(screen() — 평상시 20일 모멘텀 내림차순, 하락 국면
      atr_pct 오름차순). 예전엔 국면과 무관하게 "20일 수익률 내림차순" 이라고 라벨을 붙여
      **하락 국면에 LLM 에 거짓 입력**을 줬다. 국면 판정도 코드(QQQ 60일 < BEAR_RET60_PCT)
      하나로 통일한다 — 프롬프트가 표 중앙값으로 따로 판정하면 두 판정이 엇갈린다.
    """
    table = rows[:SCREEN_N]
    bear = regime == "하락"
    order = "atr_pct(일중 변동폭) 오름차순" if bear else "20일 수익률 내림차순"
    signal = (f"저변동성 우선 (표는 {order}). 하락 국면이므로 20일 수익률 순위는 무시한다"
              if bear else f"20일 수익률 상위 (표는 {order}). 상위권을 이유 없이 빼지 말 것")
    out = ask_claude("instructions_screen.md", {
        "기준 시각 (KST)": _now(),
        "시장 국면": regime,
        "선정 규칙": {"최대 후보 수": TOP_N, "보유 기간": "약 20 거래일",
                  "표 정렬 기준": order, "검증된 신호": signal},
        "현재 보유 종목": sorted(account["holdings"]),
        f"유니버스 ({order})": table,
    }, CANDIDATES_SCHEMA)
    valid = {r["symbol"] for r in table}
    picks = []
    for c in out["candidates"]:
        sym = str(c["symbol"]).upper()
        if sym in valid and sym not in [p["symbol"] for p in picks]:
            picks.append({"symbol": sym, "reason": c["reason"]})
    return picks[:TOP_N]


# ---------- 2단계: 종목별 판단 ----------
def get_current_status(toss, symbol, account):
    """종목 상태. 현금(usd_balance)은 일부러 뺀다 — 종목 판단에 현금 사정이 섞이면 hold 편향이 생긴다."""
    ob = toss.orderbook(symbol) or {}
    price = float(toss.prices(symbol)[0]["lastPrice"])
    held = account["holdings"].get(symbol, {})
    return {"orderbook_timestamp": ob.get("timestamp"),
            "best_bid": (ob.get("bids") or [{}])[0].get("price"),
            "best_ask": (ob.get("asks") or [{}])[0].get("price"),
            "current_price": price,
            "stock_balance": held.get("quantity", 0.0),
            "avg_buy_price": held.get("avg_price", 0.0),
            "pnl_pct": held.get("pnl_pct")}


def stock_decision(toss, symbol, account, why_candidate=None, regime="보통", entries=None):
    status = get_current_status(toss, symbol, account)
    data = fetch_and_prepare_data(toss, symbol)
    d = data["daily_ohlcv"]
    ret20 = (d[-1]["close"] / d[-21]["close"] - 1) * 100 if len(d) >= 21 else None
    factor, tier = momentum_tier(ret20, regime)
    status["ret_20d_pct"] = round(ret20, 2) if ret20 is not None else None
    status["momentum_tier"] = tier
    status["size_factor"] = factor
    held_days = trading_days_since((entries or {}).get(symbol)) if status["stock_balance"] else None
    decision = ask_claude("instructions.md", {
        "시장 국면": regime,
        "보유 거래일수": held_days,
        "만기 청산 기준 (거래일)": MAX_HOLD_DAYS,
        "종목": {"symbol": symbol, "선별 이유": why_candidate,
               "현재 보유 여부": symbol in account["holdings"],
               "20일 수익률 %": status["ret_20d_pct"], "모멘텀 구간": tier},
        "최근 뉴스": get_news_data(symbol),
        "시장 데이터 (일봉·시간봉 + 보조지표)": data,
        "최근 판단 기록 (최신순)": fetch_last_decisions(symbol),
        "현재 종목 상태": {k: v for k, v in status.items() if k != "size_factor"},
    }, DECISION_SCHEMA)
    return {"symbol": symbol, **decision, "status": status}


def decide_all(toss, symbols, account, reasons, regime="보통", entries=None):
    """종목별 판단을 병렬로. 한 종목이 실패해도 나머지는 진행한다."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(stock_decision, toss, s, account, reasons.get(s), regime, entries): s
                for s in symbols}
        for f in concurrent.futures.as_completed(futs):
            s = futs[f]
            try:
                results[s] = f.result()
                log.info("판단 %s: %s %d%% — %s", s, results[s]["decision"],
                         results[s]["percentage"], results[s]["reason"][:80])
            except Exception as e:  # noqa: BLE001
                log.error("판단 %s 실패: %s", s, e)
    return results


# ---------- 3단계: 배분 ----------
def planned_buy_amount(symbol, decisions, account):
    """코드가 정하는 매수 금액 = 종목당 한도(총자산 × MAX_POSITION_PCT − 기존 평가액) × 모멘텀 배수."""
    d = decisions[symbol]
    cap = account["total_value"] * MAX_POSITION_PCT / 100 \
        - account["holdings"].get(symbol, {}).get("market_value", 0.0)
    return max(0.0, cap * d["status"]["size_factor"])


def allocate(decisions, account, session, regime="보통", entries=None):
    cap = exposure_cap()
    rules = {"최대 보유 종목 수": MAX_POSITIONS,
             "주식 노출 상한 %": round(cap * 100, 1) if cap is not None else "미적용",
             "노출 상한을 묶은 제약": ("하락 국면 축소(REGIME_DERISK)"
                              if BEAR_CAP is not None and cap is not None and cap >= BEAR_CAP
                              else "변동성 타겟") if cap is not None else "없음",
             "종목당 최대 비중 % (총자산 대비)": MAX_POSITION_PCT,
             "항상 남길 현금 비중 %": CASH_RESERVE_PCT, "최소 주문 금액 USD": MIN_ORDER_USD,
             "소수점(금액) 주문 가능": fractional_allowed(session),
             "장외(프리·애프터) 여부": extended_hours(session),
             "매수 금액": "코드가 모멘텀 구간으로 정함 (아래 planned_buy_usd). Claude 는 승인/거부만"}
    payload = {
        "기준 시각 (KST)": _now(),
        "시장 국면": regime,
        "만기 청산": f"보유 {MAX_HOLD_DAYS}거래일이 지난 종목은 코드가 자동 매도한다 (0이면 끔)",
        "보유 거래일수": {s: trading_days_since((entries or {}).get(s))
                     for s in account["holdings"]},
        "장 시간 (KST, 거래시작·정규개장·정규마감·거래종료)":
            [s.isoformat(timespec="minutes") for s in session] if session else None,
        "규칙": rules,
        "계좌": {"현금 USD": account["cash"], "총자산 USD": account["total_value"],
               "보유": account["holdings"], "미체결 주문 종목": account["open_orders"]},
        "종목별 판단": [{k: v for k, v in d.items() if k != "status"}
                    | {"current_price": d["status"]["current_price"],
                       "pnl_pct": d["status"]["pnl_pct"],
                       "ret_20d_pct": d["status"]["ret_20d_pct"],
                       "momentum_tier": d["status"]["momentum_tier"],
                       "planned_buy_usd": round(planned_buy_amount(s, decisions, account), 2)}
                    for s, d in decisions.items()],
    }
    return ask_claude("instructions_portfolio.md", payload, ALLOCATION_SCHEMA)


def forced_exits(decisions, account):
    """코드가 강제하는 청산. Claude 판단과 무관하게 나간다 — LLM 이 hold 를 고집하거나
    종목 판단 호출 자체가 실패해도 포지션이 방치되지 않게 하는 안전장치.
      (1) 손절: 평단 대비 -STOP_LOSS_PCT%  (계좌 값만 쓰므로 Claude 없이도 동작)
      (2) 모멘텀 청산: 보유 종목의 20일 수익률이 음수로 꺾임 — 단 갈아탈 후보가 있을 때만
    (2)는 새 신호가 아니라 기존 신호의 대칭 적용이다 — 백테스트에서 20일 수익률 음수 구간이
    가장 못 올랐고, 매수도 size_factor 0 으로 이미 막고 있다.
    다만 백테스트가 보여준 모멘텀 청산의 우위는 '판 자리에 바로 재진입한다'는 가정에서만 나온다.
    갈아탈 후보가 하나도 없으면 회전이 아니라 그냥 저점 매도이므로 청산하지 않는다.
    미체결 주문이 있는 종목은 validate_orders 의 매도 루프가 알아서 걸러낸다."""
    out = []
    # 재진입 후보 = 미보유 종목 중 모멘텀 배수가 살아 있는 것. 실제 1주 살 현금이 되는지까지는 안 본다.
    # ponytail: 후보 존재 여부만 확인. 현금·최소주문까지 보려면 planned_buy_amount 와 현재가가 필요.
    rotate_to = [s for s, d in decisions.items()
                 if s not in account["holdings"] and d["status"]["size_factor"] > 0]
    for sym, h in account["holdings"].items():
        pnl = h.get("pnl_pct")
        if STOP_LOSS_PCT > 0 and pnl is not None and pnl <= -STOP_LOSS_PCT:
            why = f"손절 규칙: 평단 대비 {pnl:.1f}% (기준 -{STOP_LOSS_PCT:.0f}%)"
        else:
            ret20 = ((decisions.get(sym) or {}).get("status") or {}).get("ret_20d_pct")
            if not (MOMENTUM_EXIT and ret20 is not None and ret20 < 0):
                continue
            if not rotate_to:
                log.info("모멘텀 청산 보류 %s (20일 %.1f%%): 갈아탈 후보 없음", sym, ret20)
                continue
            why = f"모멘텀 청산 규칙: 20일 수익률 {ret20:.1f}% (음수), 대체 후보 {len(rotate_to)}개"
        log.warning("강제 청산 %s — %s", sym, why)
        out.append({"symbol": sym, "side": "sell", "sell_pct": 100, "reason": why,
                    "forced": True})
    return out


def validate_orders(plan, decisions, account, session, entries=None, skip_symbols=()):
    """Claude 의 주문 목록을 규칙으로 걸러 실제 낼 주문만 남긴다. 매도 먼저, 매수 나중.

    plan 을 비우고 decisions={} 로 부르면 **LLM 없이도** 손절·만기·노출 축소만 뽑아낼 수
    있다. run_cycle 은 이 형태로 위험 관리 패스를 먼저 돌린다 (P0-3).
    skip_symbols 는 그 패스에서 이미 주문을 낸 종목 — 같은 사이클에서 두 번 건드리지 않는다.
    """
    frac = fractional_allowed(session)
    pre = extended_hours(session)
    holdings = dict(account["holdings"])
    total = account["total_value"]
    cash_left = account["cash"] - total * CASH_RESERVE_PCT / 100
    positions = set(holdings)
    out, skipped = [], []

    def skip(o, why):
        skipped.append({**o, "skipped": why})
        log.info("주문 제외 %s %s: %s", o.get("side"), o.get("symbol"), why)

    # 강제 청산이 먼저. 같은 종목에 대한 Claude 매도는 중복이므로 버린다.
    forced = forced_exits(decisions, account)
    exited = {o["symbol"] for o in forced}
    sells = forced + [o for o in plan["orders"] if o["side"] == "sell"
                      and str(o["symbol"]).upper() not in exited]

    # 변동성 타겟: 주식 노출이 상한을 넘으면 큰 종목부터 줄인다.
    # 백테스트는 전 종목 비례 축소였지만, 실전에서는 최소 주문 금액과 수수료 때문에
    # 큰 종목부터 깎는다 (주문 수가 줄고 집중도도 함께 낮아진다).
    cap = exposure_cap()
    # 어느 제약이 상한을 묶었는지 — 사유 코드로 남겨야 나중에 오버레이 기여를 분리할 수 있다
    src = ("REGIME_DERISK — 하락 국면 노출 상한" if BEAR_CAP is not None and cap >= BEAR_CAP
           else "변동성 타겟 — 주식 노출 상한") if cap is not None else ""
    if cap is not None and total > 0:
        stock_value = sum(h["market_value"] for h in holdings.values())
        excess = stock_value - total * cap
        if excess > MIN_ORDER_USD:
            log.info("노출 축소(%s): 주식 %.1f%% → 상한 %.0f%%, %.2f 달러 줄인다",
                     src.split(" —")[0], stock_value / total * 100, cap * 100, excess)
            named = {str(o["symbol"]).upper() for o in sells}
            for sym, h in sorted(holdings.items(), key=lambda kv: -kv[1]["market_value"]):
                if excess <= MIN_ORDER_USD:
                    break
                if sym in named or sym in account["open_orders"]:
                    continue
                cut = min(excess, h["market_value"])
                pct = min(100.0, cut / h["market_value"] * 100)
                if cut < MIN_ORDER_USD:
                    continue
                sells.append({
                    "symbol": sym, "side": "sell", "sell_pct": round(pct, 2),
                    "reason": f"{src} {cap * 100:.0f}% 초과분 축소",
                    "forced": True})
                named.add(sym)
                excess -= cut

    buys = [o for o in plan["orders"] if o["side"] == "buy"]
    # 만기 청산: 보유 MAX_HOLD_DAYS 거래일이 지난 종목은 Claude 판단과 무관하게 전량 매도한다.
    # 28년 백테스트에서 고정 20거래일 만기 청산의 승률은 56.4%, 하락 5구간 전부 1등이었다.
    # 기존의 "20일 수익률 음전 시 매도" 는 같은 조건에서 승률 42.1% 로 14%p 낮다.
    if MAX_HOLD_DAYS > 0:
        named = {str(o["symbol"]).upper() for o in sells}
        for sym in holdings:
            if sym in named or sym in account["open_orders"]:
                continue
            held = trading_days_since((entries or {}).get(sym))
            if held is not None and held >= MAX_HOLD_DAYS:
                sells.append({"symbol": sym, "side": "sell", "sell_pct": 100, "forced": True,
                              "reason": f"보유 {held}거래일로 만기({MAX_HOLD_DAYS}) 도달 — 코드 강제 청산"})
                log.info("만기 청산 %s: 보유 %d거래일", sym, held)
    done = set()
    for o in sells:
        sym = str(o["symbol"]).upper()
        h = holdings.get(sym)
        if sym in skip_symbols:
            skip(o, "같은 사이클의 위험관리 패스에서 이미 주문함"); continue
        if sym in done:
            skip(o, "같은 종목 매도 중복"); continue
        if not h:
            skip(o, "보유하지 않은 종목"); continue
        cancel_first = False
        if sym in account["open_orders"]:
            # 위험 축소(손절·만기·노출)는 미체결 때문에 미룰 수 없다. 기존 주문을 취소하고,
            # 취소/체결 결과를 대사한 뒤 **실제 매도 가능 수량**으로 낸다 (place_order → free_position).
            if not o.get("forced"):
                skip(o, "미체결 주문 있음"); continue
            cancel_first = True
            log.warning("위험 축소 %s: 미체결 주문을 취소하고 매도한다 — %s", sym, o["reason"])
        pct = min(100.0, max(0.0, float(o.get("sell_pct") or 0)))
        if pct <= 0:
            skip(o, "sell_pct 없음"); continue
        qty = h["quantity"] * pct / 100
        if not frac:
            # 토스는 소수점 수량을 **정규장 시장가 매도**로만 받는다. 장외 지정가에 소수점을 실으면
            # 주문 전체가 400 으로 거절된다 (2026-09-08 WDAY 0.62주 3회 연속). 전량 매도도 예외가 아니다.
            # 지금은 정수 주만 팔고, 소수점 잔량은 다음 정규장 사이클이 판다.
            qty = float(int(qty + 1e-9))
            if qty < 1:
                skip(o, "소수점 잔량은 정규장(마감 1시간 전까지) 시장가로만 매도 가능 — 다음 정규장 사이클"); continue
        if qty <= 0 or (pct < 100 and qty * h["last_price"] < MIN_ORDER_USD):
            skip(o, "최소 주문 금액 미만 (전량 매도는 예외)"); continue
        limit = round(h["last_price"] * (1 - PREMARKET_SLIP / 100), 2) if pre else None
        out.append({"symbol": sym, "side": "sell", "quantity": round(qty, 6),
                    "price": h["last_price"], "limit_price": limit,
                    "amount_usd": round(qty * h["last_price"], 2), "whole": not frac,
                    "cancel_first": cancel_first, "reason": o["reason"]})
        done.add(sym)
        # ★ 슬롯은 여기서 비우지 않는다. 매도는 '제출'했을 뿐 체결이 아니고, 현금·슬롯은
        #   체결로만 생긴다 (P0-4). 자리는 다음 사이클이 계좌를 다시 읽어서 쓴다.
    if cap is not None and total > 0:      # 노출 상한 안에서만 신규 매수
        room = total * cap - sum(h["market_value"] for h in holdings.values())
        cash_left = min(cash_left, max(0.0, room))
    for o in buys:
        sym = str(o["symbol"]).upper()
        d = decisions.get(sym)
        if sym in skip_symbols:
            skip(o, "같은 사이클의 위험관리 패스에서 이미 주문함"); continue
        if sym in exited:
            skip(o, "같은 사이클에서 강제 청산된 종목"); continue
        if sym in done:
            skip(o, "같은 종목 주문 중복"); continue
        if not d:
            skip(o, "판단 대상이 아닌 종목"); continue
        if sym in account["open_orders"]:
            skip(o, "미체결 주문 있음"); continue
        if sym not in positions and len(positions) >= MAX_POSITIONS:
            skip(o, f"최대 보유 종목 수 {MAX_POSITIONS} 초과"); continue
        price = d["status"]["current_price"]
        if d["status"]["size_factor"] <= 0:
            skip(o, f"20일 수익률 {d['status']['ret_20d_pct']}% — 모멘텀 음수는 매수 안 함"); continue
        # 금액은 Claude 가 아니라 코드가 정한다: 종목당 한도 × 모멘텀 배수, 현금 한도 안에서
        amount = min(planned_buy_amount(sym, decisions, account), cash_left)
        if amount < MIN_ORDER_USD:
            skip(o, f"금액 ${amount:.2f} < 최소 ${MIN_ORDER_USD} (비중·현금 한도 적용 후)"); continue
        qty = None
        if not frac:
            qty = int(amount / price)
            if qty < 1:
                skip(o, "정규장 외 시간이라 정수 주 필요, 1주 미만"); continue
            amount = qty * price
        out.append({"symbol": sym, "side": "buy", "quantity": qty, "price": price,
                    "limit_price": round(price * (1 + PREMARKET_SLIP / 100), 2) if pre else None,
                    "amount_usd": round(amount, 2), "reason": o["reason"]})
        cash_left -= amount
        positions.add(sym)
        done.add(sym)
    return out, skipped


def intent_key(o, when=None):
    """주문 의도의 고유 키 (clientOrderId).

    예전엔 run_id 를 썼는데, run_id 는 **머신마다 다른 로컬 시퀀스**라 다른 PC·다른
    프로세스가 같은 매수 의도를 각자 내면 서로 다른 id 가 되어 중복 주문이 그대로 나갔다.
    이제는 의도 자체(미국 날짜 · 30분 슬롯 · 종목 · 방향)로 만든다. 같은 슬롯의 같은
    의도는 어디서 내도 같은 id 가 되고, 거래소가 clientOrderId 중복을 거절해 준다.

    한계(과장 금지): 30분 버킷은 스케줄이 대략 같은 시각에 도는 것을 전제한 근사이고,
    두 프로세스가 정확히 동시에 제출하는 경합은 여기서 못 막는다 — 그건 브로커의 중복
    거절에 의존한다. 단일 머신 락으로 다른 PC 까지 막았다고 말할 수 없다.
    """
    t = when or datetime.datetime.now(NY)
    return f"at{t:%y%m%d}{t.hour:02d}{t.minute // 30 * 30:02d}{o['symbol']}{o['side'][0].upper()}"


def find_order(toss, coid):
    """clientOrderId 로 실제 접수 여부를 확인한다 — 제출 도중 타임아웃·연결 끊김이 나면
    '주문이 안 나갔다'고 단정하지 말고 이걸로 대사한 뒤 재시도할지 정한다."""
    for st in ("OPEN", "CLOSED"):
        try:
            for r in (toss.orders(st) or {}).get("orders") or []:
                if r.get("clientOrderId") == coid:
                    return r
        except Exception as e:  # noqa: BLE001
            log.warning("주문 대사(%s) 실패: %s", st, e)
    return None


def free_position(toss, symbol, timeout=20):
    """미체결 주문을 취소하고 **실제 매도 가능 수량**을 돌려준다 (위험 축소 전용).

    순서: 취소 요청 → 취소/체결 결과 대사 → sellable-quantity 재조회.
    취소와 체결은 경합한다 — 취소가 거절되면 이미 체결된 것이므로, 판단은 항상
    재조회 결과로 한다. 조회 자체가 실패하면 None (호출자는 계획 수량을 그대로 쓴다).
    """
    try:
        opens = (toss.orders("OPEN", symbol=symbol) or {}).get("orders") or []
    except Exception as e:  # noqa: BLE001
        log.warning("%s 미체결 조회 실패: %s", symbol, e)
        opens = []
    for o in opens:
        try:
            toss.cancel_order(o["orderId"])
        except TossError as e:
            log.warning("%s 주문 %s 취소 거절 (이미 체결/취소?): %s", symbol, o.get("orderId"), e)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if not ((toss.orders("OPEN", symbol=symbol) or {}).get("orders") or []):
                break
        except Exception as e:  # noqa: BLE001
            log.warning("%s 취소 대사 실패: %s", symbol, e)
            break
        time.sleep(1)
    else:
        log.error("%s 미체결이 %d초 안에 정리되지 않았다 — 매도 가능한 만큼만 낸다", symbol, timeout)
    try:
        r = toss.sellable_quantity(symbol) or {}
        q = r.get("sellableQuantity", r.get("quantity"))
        return None if q is None else float(q)
    except (TossError, TypeError, ValueError) as e:
        log.warning("%s 매도가능 수량 조회 실패: %s", symbol, e)
        return None


def place_order(toss, o, coid=None):
    """토스 주문. clientOrderId 는 intent_key — 같은 의도의 재시도·중복 제출을 막는다."""
    coid = coid or intent_key(o)
    if o.get("cancel_first"):
        q = free_position(toss, o["symbol"])
        if q is not None:
            if o.get("whole"):              # 장외: 재조회 수량도 정수 주로 (소수점은 거절된다)
                q = float(int(q + 1e-9))
            if q <= 0:
                raise TossError(409, "no-sellable", f"{o['symbol']}: 매도 가능 수량 0")
            o = {**o, "quantity": min(o["quantity"], q)}
    lim = o.get("limit_price")          # 프리장엔 시장가를 못 받아서 지정가로 낸다
    otype = "LIMIT" if lim else "MARKET"
    if o["side"] == "buy":
        if o["quantity"] is None:
            return toss.create_order(o["symbol"], "BUY", otype, price=lim,
                                     order_amount=f"{o['amount_usd']:.2f}", client_order_id=coid)
        return toss.create_order(o["symbol"], "BUY", otype, price=lim,
                                 quantity=str(int(o["quantity"])), client_order_id=coid)
    q = f"{o['quantity']:.6f}".rstrip("0").rstrip(".")
    return toss.create_order(o["symbol"], "SELL", otype, price=lim,
                             quantity=q, client_order_id=coid)


def place_all(toss, run_id, orders, dry, tag=""):
    """주문을 내고 DB 에 남긴다. 의도를 **먼저** 기록하고(durable intent) 결과로 갱신하므로,
    제출 중에 죽어도 다음 실행이 어떤 의도가 떠 있었는지 알 수 있다."""
    for o in orders:
        coid = intent_key(o)
        row_id = db_insert("orders", {
            "run_id": run_id, "timestamp": _now(), "symbol": o["symbol"], "side": o["side"],
            "quantity": o["quantity"], "amount_usd": o["amount_usd"], "price": o["price"],
            "order_id": coid, "status": "dry-run" if dry else "pending", "reason": o["reason"]})
        if dry:
            log.info("[dry-run]%s %s %s $%.2f (%s주)", tag, o["side"], o["symbol"],
                     o["amount_usd"], o["quantity"])
            continue
        try:
            r = place_order(toss, o, coid)
            fields = {"order_id": r.get("orderId"), "status": "submitted"}
            log.info("주문%s %s %s $%.2f → %s", tag, o["side"], o["symbol"], o["amount_usd"],
                     r.get("orderId"))
        except TossError as e:
            found = find_order(toss, coid)      # 정말 안 나갔는지 대사한다
            if found:
                fields = {"order_id": found.get("orderId"), "status": "submitted (대사 확인)"}
                log.warning("주문%s %s %s: 오류(%s)였지만 실제로는 접수됨 — 재시도하지 않는다",
                            tag, o["side"], o["symbol"], e)
            else:
                fields = {"status": f"error {e.code}: {e}"[:300]}
                log.error("주문 실패%s %s %s: %s", tag, o["side"], o["symbol"], e)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("UPDATE orders SET " + ", ".join(f"{k}=?" for k in fields) +
                         " WHERE id=?", [*fields.values(), row_id])


def log_funnel(rows, picks, decisions, plan, risk_orders, orders, skipped, account):
    """의사결정 퍼널 한 줄 + 차단 사유 집계. **주문이 0건이어도 반드시 남긴다.**

    이게 없어서 2026-09-09~16 의 43사이클 무주문을 사후 DB 파싱으로만 알아냈다.
    특히 '매수 판단은 났는데 배분 단계가 주문으로 안 올렸다' 는 경우는 어디에도 흔적이 없었다.
    """
    d = collections.Counter(v["decision"] for v in decisions.values())
    plan_buy = sum(1 for o in plan["orders"] if o.get("side") == "buy")
    plan_sell = sum(1 for o in plan["orders"] if o.get("side") == "sell")
    cash_left = account["cash"] - account["total_value"] * CASH_RESERVE_PCT / 100
    log.info("퍼널: 스크리닝 %d → 후보 %d → 판단 %d(매수%d·매도%d·보유%d) → 배분 %d(매수%d·매도%d) "
             "→ 주문 %d(위험관리 %d) · 제외 %d | 가용현금 $%.2f (현금 $%.2f − 유지선 $%.2f)",
             len(rows), len(picks), len(decisions), d["buy"], d["sell"], d["hold"],
             len(plan["orders"]), plan_buy, plan_sell, len(orders) + len(risk_orders),
             len(risk_orders), len(skipped), cash_left, account["cash"],
             account["total_value"] * CASH_RESERVE_PCT / 100)
    if d["buy"] and not any(o["side"] == "buy" for o in orders):
        log.warning("매수 판단 %d건이 전부 주문이 되지 못했다 — 사유: %s", d["buy"],
                    dict(collections.Counter(o["skipped"] for o in skipped
                                             if o.get("side") == "buy")) or "배분이 안 올림")
    if skipped:
        for why, n in collections.Counter(o["skipped"] for o in skipped).most_common():
            log.info("  차단 %d건: %s", n, why)


def log_skipped(run_id, skipped):
    for o in skipped:
        db_insert("orders", {"run_id": run_id, "timestamp": _now(), "symbol": o.get("symbol"),
                             "side": o.get("side"), "amount_usd": o.get("amount_usd"),
                             "status": "skipped: " + o["skipped"], "reason": o.get("reason")})


# ---------- 한 사이클 ----------
def run_cycle(dry_run=None, force=False):
    """한 사이클. dry_run 을 명시하지 않으면 환경변수 DRY_RUN 을 따른다.
    (대시보드처럼 다른 프로세스·스레드에서 부를 때는 반드시 명시할 것 — 전역에 기대지 않는다)
    force=True 면 장 시작 전·마감 후에도 실행한다 (휴장일은 여전히 건너뜀). 수동 1회 실행용."""
    dry = DRY_RUN if dry_run is None else bool(dry_run)
    run_id = db_insert("runs", {"timestamp": _now(), "dry_run": int(dry), "status": "running",
                                "model": MODEL})
    log.info("=== run %d 시작 (%s, %s) ===", run_id, "DRY_RUN" if dry else "실주문", MODEL)
    usage_reset()
    try:
        toss = shared_client()
        session = market_session(toss)
        why = session_block(session)
        if why and not (force and session):
            log.info("%s — 건너뜀 (강제 실행은 --force 또는 대시보드 1회 실행)", why)
            db_update_run(run_id, status="skipped", summary=why)
            return
        account = account_state(toss)
        db_update_run(run_id, total_value=account["total_value"], cash=account["cash"])
        log.info("계좌: 현금 $%.2f 총자산 $%.2f 보유 %s", account["cash"],
                 account["total_value"], list(account["holdings"]))
        log_equity(account["total_value"],     # 변동성 타겟이 쓰는 일별 자산 이력
                   sum(h["market_value"] for h in account["holdings"].values()))
        if VOL_TARGET_PCT > 0 and _vol_target_cap(verbose=True) is None:
            with sqlite3.connect(DB_PATH) as _c:
                _n = _c.execute("SELECT COUNT(*) FROM equity").fetchone()[0]
            log.info("변동성 타겟 대기: 자산 이력 %d/%d일", _n, VOL_WINDOW + 1)
        entries = position_entry_map(toss)     # 계좌 기준 진입 시각 (다른 PC 주문도 포함)

        # 0) 시장 국면 + 실제 거래일 달력 (지수 일봉 한 번으로 둘 다 해결한다)
        idx = index_daily(toss)
        global TRADING_DAYS, BEAR_CAP
        TRADING_DAYS = ([d.astimezone(NY).date() for d in idx.index] if idx is not None else [])
        # 오버레이 상한은 위험관리 패스(바로 아래)보다 먼저 정해져야 한다 — 축소 매도가 거기서 난다
        BEAR_CAP = bear_derisk(idx)
        if BEAR_CAP is not None:
            log.warning("REGIME_DERISK ON: %s 252일 고점 대비 -%.0f%% 이하 (최근 %d일) "
                        "→ 주식 노출 상한 %.0f%%", BEAR_INDEX, BEAR_DD_PCT, BEAR_OFF_DAYS,
                        BEAR_CAP * 100)
        elif BEAR_EXPOSURE_PCT is not None and not bear_data_ok(idx):
            log.error("REGIME_DERISK 판정 불가 — 지수 %s 일봉 %s (253봉 필요). 신호가 꺼진 게"
                      " 아니라 **평가하지 못했다**. 오버레이 없이(=축소 없이) 진행한다",
                      BEAR_INDEX, "없음" if idx is None else f"{len(idx)}봉")
        elif BEAR_EXPOSURE_PCT is not None:
            log.info("REGIME_DERISK OFF (상한 %.0f%% 설정됐으나 신호 미점등)", BEAR_EXPOSURE_PCT)
        if not TRADING_DAYS:
            log.warning("거래일 달력 없음 — 만기 계산이 주말만 빼는 근사로 떨어진다(휴장일만큼 이르게 청산)")
        log.info("보유 거래일수: %s", {s: trading_days_since(entries.get(s))
                                 for s in account["holdings"]})

        # 0-1) ★ 위험 관리 먼저. 손절·만기·노출 축소는 LLM 스크리닝이 실패하거나
        #      느려도 반드시 나가야 한다 — 그래서 진입 분석보다 앞에서, LLM 없이 돌린다.
        risk_orders, risk_skipped = validate_orders(
            {"orders": [], "summary": ""}, {}, account, session, entries)
        risk_symbols = {o["symbol"] for o in risk_orders}
        if risk_orders:
            log.warning("위험관리 주문 %d건 선제 실행: %s", len(risk_orders),
                        [(o["symbol"], o["reason"][:30]) for o in risk_orders])
            place_all(toss, run_id, risk_orders, dry, tag="[위험관리]")
            log_skipped(run_id, risk_skipped)
            account = account_state(toss)      # 슬롯·현금은 체결로만 생긴다 — 다시 읽는다

        reg = market_regime(toss, df=idx)
        regime = reg["regime"]
        log.info("국면: %s (%s 60일 %s%%, 기준 %s%%)", regime, reg["source"],
                 reg["ret_60d_pct"], BEAR_RET60_PCT)

        # 1) 스크리닝
        t_snapshot = time.time()
        rows = screen(toss, regime=regime)
        if regime == "하락":
            reg = market_regime(toss, rows) if reg["ret_60d_pct"] is None else reg
            log.info("스크리닝 %d종목, 저변동성(atr_pct) 하위: %s", len(rows),
                     [(r["symbol"], r["atr_pct"]) for r in rows[:8]])
        else:
            log.info("스크리닝 %d종목, 20일 수익률 상위: %s", len(rows),
                     [(r["symbol"], r["ret_20d_pct"]) for r in rows[:8]])
        picks = pick_candidates(rows, account, regime)
        reasons = {p["symbol"]: p["reason"] for p in picks}
        log.info("후보: %s", list(reasons))
        db_update_run(run_id, candidates=json.dumps(picks, ensure_ascii=False))

        # 2) 종목별 판단 (후보 + 보유)
        symbols = list(dict.fromkeys(list(reasons) + list(account["holdings"])))
        decisions = decide_all(toss, symbols, account, reasons, regime, entries)
        for d in decisions.values():
            st = d["status"]
            db_insert("trading_decisions", {
                "run_id": run_id, "timestamp": _now(), "symbol": d["symbol"],
                "decision": d["decision"], "percentage": d["percentage"], "reason": d["reason"],
                "stock_balance": st["stock_balance"], "usd_balance": account["cash"],
                "avg_buy_price": st["avg_buy_price"], "current_price": st["current_price"]})
        if not decisions:
            raise RuntimeError("종목별 판단이 하나도 없음")

        # 3) 배분 → 검증 → 주문
        plan = allocate(decisions, account, session, regime, entries)
        log.info("배분 요약: %s", plan["summary"])
        account = account_state(toss)          # LLM 왕복 동안 바뀐 현금·보유를 반영
        stale = (time.time() - t_snapshot) / 60
        if stale > STALE_MAX_MIN:
            log.error("시세 스냅샷이 %.0f분 낡음 (>%.0f) — 신규 매수는 취소하고 매도만 낸다",
                      stale, STALE_MAX_MIN)
            plan["orders"] = [o for o in plan["orders"] if o.get("side") != "buy"]
        orders, skipped = validate_orders(plan, decisions, account, session, entries,
                                          skip_symbols=risk_symbols)
        log_funnel(rows, picks, decisions, plan, risk_orders, orders, skipped, account)
        place_all(toss, run_id, orders, dry)
        log_skipped(run_id, skipped)
        db_update_run(run_id, status="done", summary=plan["summary"])
        log.info("=== run %d 완료: 주문 %d건(위험관리 %d건 포함), 제외 %d건 ===",
                 run_id, len(orders) + len(risk_orders), len(risk_orders), len(skipped))
    except Exception as e:  # noqa: BLE001
        log.exception("run %d 실패: %s", run_id, e)
        db_update_run(run_id, status=f"error: {e}"[:300])
    finally:
        u = usage_snapshot()
        db_update_run(run_id, claude_calls=u["calls"], input_tokens=u["input_tokens"],
                      output_tokens=u["output_tokens"], cost_usd=u["cost_usd"])
        log.info("토큰: Claude %d회, 입력 %s, 출력 %s, 추정 $%.4f (%s API 단가 기준)",
                 u["calls"], f"{u['input_tokens']:,}", f"{u['output_tokens']:,}", u["cost_usd"], MODEL)


def run_analysis():
    """정규장 개장 직전 사전 분석. 주문 없이 분석만 (dry_run) 돌린다."""
    left = analysis_lead_min(market_session(shared_client()))
    if left is None:
        log.info("사전 분석 건너뜀: 휴장일이거나 개장 %d분 전 창 밖", ANALYSIS_LEAD_MIN)
        return
    log.info("=== 사전 분석 (정규장 개장 %.0f분 전, 주문 없음) ===", left)
    run_cycle(dry_run=True, force=True)


if __name__ == "__main__":
    initialize_db()
    log.info("모델 %s @ %s · 후보 %d · 최대 %d종목 · 종목당 %.0f%% · 현금유지 %.0f%% · 청산 %s · 실행 %s%s",
             MODEL, BASE_URL, TOP_N, MAX_POSITIONS, MAX_POSITION_PCT, CASH_RESERVE_PCT,
             (f"손절 -{STOP_LOSS_PCT:.0f}%" if STOP_LOSS_PCT > 0 else "손절 없음")
             + (" + 모멘텀 음수" if MOMENTUM_EXIT else ""),
             ", ".join(TRADE_TIMES) + " · 사전분석 " + ", ".join(ANALYSIS_TIMES),
             " · DRY_RUN" if DRY_RUN else " · 실주문")
    run_cycle(force="--force" in sys.argv or "--once" in sys.argv)   # 수동 실행은 장 시간 무시
    if "--once" in sys.argv:
        sys.exit(0)
    for t in TRADE_TIMES:
        schedule.every().day.at(t).do(run_cycle)
    for t in ANALYSIS_TIMES:
        schedule.every().day.at(t).do(run_analysis)
    while True:
        schedule.run_pending()
        time.sleep(1)
