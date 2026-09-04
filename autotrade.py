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
import pandas_ta as ta
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
# 실행 시각 (KST): 개장 22:30 → 1시간 뒤 23:30 → 00:00 부터 2시간 간격. 정규장 종료(05:00/06:00) 1시간 전까지.
# 겨울(서머타임 해제)엔 개장이 23:30 이라 22:30 실행은 "장 시작 전"으로 자동 건너뛴다.
TRADE_TIMES = [t.strip() for t in
               os.environ.get("TRADE_TIMES", "22:30,23:30,00:00,02:00,04:00").split(",")]

SCREEN_N = int(os.environ.get("SCREEN_N", 40))            # 규칙 점수 상위 몇 개를 Claude 에 보여줄지
TOP_N = int(os.environ.get("TOP_N", 5))                   # Claude 가 고르는 후보 수
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", 5))   # 동시 보유 최대 종목 수
MAX_POSITION_PCT = float(os.environ.get("MAX_POSITION_PCT", 30))   # 종목당 최대 비중 (총자산 대비 %)
CASH_RESERVE_PCT = float(os.environ.get("CASH_RESERVE_PCT", 10))   # 항상 남겨둘 현금 비중 (%)
MIN_ORDER_USD = float(os.environ.get("MIN_ORDER_USD", 5))
WORKERS = int(os.environ.get("WORKERS", 3))               # 종목별 판단 병렬 수 (Claude 속도 제한 고려)

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
            status TEXT, reason TEXT);""")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(trading_decisions)")]
        if "run_id" not in cols:   # 단일 종목 버전 DB 호환
            conn.execute("ALTER TABLE trading_decisions ADD COLUMN run_id INTEGER")
        rcols = [r[1] for r in conn.execute("PRAGMA table_info(runs)")]
        for col, typ in (("model", "TEXT"), ("claude_calls", "INTEGER"),
                         ("input_tokens", "INTEGER"), ("output_tokens", "INTEGER"),
                         ("cost_usd", "REAL")):
            if col not in rcols:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")


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


def add_indicators(df):
    """SMA10/20/50, EMA10, RSI14, 스토캐스틱(14,3,3), MACD(12,26,9), 볼린저(20,2).
    봉이 모자라 pandas_ta 가 None 을 주는 지표는 뺀다."""
    for n in (10, 20, 50):
        df[f"SMA_{n}"] = ta.sma(df["close"], length=n)
    df["EMA_10"] = ta.ema(df["close"], length=10)
    df["RSI_14"] = ta.rsi(df["close"], length=14)
    extra = [ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3),
             ta.macd(df["close"], fast=12, slow=26, signal=9),
             ta.bbands(df["close"], length=20, std=2)]
    return pd.concat([df] + [x for x in extra if x is not None], axis=1)


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
    """오늘(미국 날짜) 정규장 (시작, 종료) KST. 휴장이면 None."""
    us_today = datetime.datetime.now(NY).date().isoformat()
    day = (toss.us_market_calendar(date=us_today) or {}).get("today") or {}
    reg = day.get("regularMarket")
    if not reg:
        return None
    return tuple(datetime.datetime.fromisoformat(reg[k]).astimezone(KST)
                 for k in ("startTime", "endTime"))


def fractional_allowed(session):
    """금액 기반 매수·소수점 매도는 정규장 시작 ~ 종료 1시간 전까지만 접수된다."""
    if not session:
        return False
    now = datetime.datetime.now(KST)
    return session[0] <= now <= session[1] - datetime.timedelta(hours=1)


def session_block(session):
    """지금 실행하면 안 되는 이유. 정규장 시작 5분 전 ~ 종료 사이면 None(실행 가능)."""
    if not session:
        return "휴장일"
    now = datetime.datetime.now(KST)
    if now < session[0] - datetime.timedelta(minutes=5):
        return f"장 시작 전 (개장 {session[0]:%H:%M} KST)"
    if now > session[1]:
        return f"장 마감 후 (마감 {session[1]:%H:%M} KST)"
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
            out, _ = json.JSONDecoder().raw_decode(text[text.find("{"):])
            for k in schema["required"]:
                if k not in out:
                    raise ValueError(f"응답에 {k} 없음")
            return out
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


def momentum_tier(ret_20d):
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


def screen(toss, universe=TICKERS):
    """전 종목 일봉 요약, 20일 모멘텀 내림차순. 지표는 참고 열로만 남긴다."""
    rows = []
    for sym in universe:
        try:
            df = add_indicators(candles(toss, sym, "1d", 90))
            if len(df) < 60:
                continue
            c = df.iloc[-1]
            ret20 = (c["close"] / df["close"].iloc[-21] - 1) * 100
            factor, tier = momentum_tier(ret20)
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
    rows.sort(key=lambda r: r["ret_20d_pct"], reverse=True)
    return [{k: (None if isinstance(v, float) and pd.isna(v)
                 else float(v) if isinstance(v, float) else v) for k, v in r.items()}
            for r in rows]


def pick_candidates(rows, account):
    """20일 모멘텀 상위 SCREEN_N 개 표를 주고 Claude 가 TOP_N 개를 고른다 (거부권 + 분산)."""
    table = rows[:SCREEN_N]
    out = ask_claude("instructions_screen.md", {
        "기준 시각 (KST)": _now(),
        "선정 규칙": {"최대 후보 수": TOP_N, "보유 기간": "약 20 거래일",
                  "검증된 신호": "20일 수익률 상위 (표는 그 순서). 상위권을 이유 없이 빼지 말 것"},
        "현재 보유 종목": sorted(account["holdings"]),
        "유니버스 (20일 수익률 내림차순)": table,
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


def stock_decision(toss, symbol, account, why_candidate=None):
    status = get_current_status(toss, symbol, account)
    data = fetch_and_prepare_data(toss, symbol)
    d = data["daily_ohlcv"]
    ret20 = (d[-1]["close"] / d[-21]["close"] - 1) * 100 if len(d) >= 21 else None
    factor, tier = momentum_tier(ret20)
    status["ret_20d_pct"] = round(ret20, 2) if ret20 is not None else None
    status["momentum_tier"] = tier
    status["size_factor"] = factor
    decision = ask_claude("instructions.md", {
        "종목": {"symbol": symbol, "선별 이유": why_candidate,
               "현재 보유 여부": symbol in account["holdings"],
               "20일 수익률 %": status["ret_20d_pct"], "모멘텀 구간": tier},
        "최근 뉴스": get_news_data(symbol),
        "시장 데이터 (일봉·시간봉 + 보조지표)": data,
        "최근 판단 기록 (최신순)": fetch_last_decisions(symbol),
        "현재 종목 상태": {k: v for k, v in status.items() if k != "size_factor"},
    }, DECISION_SCHEMA)
    return {"symbol": symbol, **decision, "status": status}


def decide_all(toss, symbols, account, reasons):
    """종목별 판단을 병렬로. 한 종목이 실패해도 나머지는 진행한다."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(stock_decision, toss, s, account, reasons.get(s)): s for s in symbols}
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


def allocate(decisions, account, session):
    rules = {"최대 보유 종목 수": MAX_POSITIONS, "종목당 최대 비중 % (총자산 대비)": MAX_POSITION_PCT,
             "항상 남길 현금 비중 %": CASH_RESERVE_PCT, "최소 주문 금액 USD": MIN_ORDER_USD,
             "소수점(금액) 주문 가능": fractional_allowed(session),
             "매수 금액": "코드가 모멘텀 구간으로 정함 (아래 planned_buy_usd). Claude 는 승인/거부만"}
    payload = {
        "기준 시각 (KST)": _now(),
        "정규장 (KST)": [s.isoformat(timespec="minutes") for s in session] if session else None,
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


def validate_orders(plan, decisions, account, session):
    """Claude 의 주문 목록을 규칙으로 걸러 실제 낼 주문만 남긴다. 매도 먼저, 매수 나중."""
    frac = fractional_allowed(session)
    holdings = dict(account["holdings"])
    total = account["total_value"]
    cash_left = account["cash"] - total * CASH_RESERVE_PCT / 100
    positions = set(holdings)
    out, skipped = [], []

    def skip(o, why):
        skipped.append({**o, "skipped": why})
        log.info("주문 제외 %s %s: %s", o.get("side"), o.get("symbol"), why)

    sells = [o for o in plan["orders"] if o["side"] == "sell"]
    buys = [o for o in plan["orders"] if o["side"] == "buy"]
    for o in sells:
        sym = str(o["symbol"]).upper()
        h = holdings.get(sym)
        if not h:
            skip(o, "보유하지 않은 종목"); continue
        if sym in account["open_orders"]:
            skip(o, "미체결 주문 있음"); continue
        pct = min(100.0, max(0.0, float(o.get("sell_pct") or 0)))
        if pct <= 0:
            skip(o, "sell_pct 없음"); continue
        qty = h["quantity"] * pct / 100
        if not frac:
            qty = float(int(qty)) if pct < 100 else h["quantity"]   # 정규장 외엔 정수 주만
        if qty <= 0 or (pct < 100 and qty * h["last_price"] < MIN_ORDER_USD):
            skip(o, "최소 주문 금액 미만 (전량 매도는 예외)"); continue
        out.append({"symbol": sym, "side": "sell", "quantity": round(qty, 6),
                    "price": h["last_price"], "amount_usd": round(qty * h["last_price"], 2),
                    "reason": o["reason"]})
        if pct >= 100:
            positions.discard(sym)
    for o in buys:
        sym = str(o["symbol"]).upper()
        d = decisions.get(sym)
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
                    "amount_usd": round(amount, 2), "reason": o["reason"]})
        cash_left -= amount
        positions.add(sym)
    return out, skipped


def place_order(toss, run_id, o):
    """토스 주문. clientOrderId 로 멱등성 확보 — 같은 run 에서 재시도해도 중복 주문이 안 난다."""
    coid = f"at{run_id}-{o['symbol']}-{o['side']}"
    if o["side"] == "buy":
        if o["quantity"] is None:
            return toss.create_order(o["symbol"], "BUY", "MARKET",
                                     order_amount=f"{o['amount_usd']:.2f}", client_order_id=coid)
        return toss.create_order(o["symbol"], "BUY", "MARKET",
                                 quantity=str(int(o["quantity"])), client_order_id=coid)
    q = f"{o['quantity']:.6f}".rstrip("0").rstrip(".")
    return toss.create_order(o["symbol"], "SELL", "MARKET", quantity=q, client_order_id=coid)


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
        if why and not force:
            log.info("%s — 건너뜀 (강제 실행은 --force 또는 대시보드 1회 실행)", why)
            db_update_run(run_id, status="skipped", summary=why)
            return
        account = account_state(toss)
        db_update_run(run_id, total_value=account["total_value"], cash=account["cash"])
        log.info("계좌: 현금 $%.2f 총자산 $%.2f 보유 %s", account["cash"],
                 account["total_value"], list(account["holdings"]))

        # 1) 스크리닝
        rows = screen(toss)
        log.info("스크리닝 %d종목, 20일 수익률 상위: %s", len(rows),
                 [(r["symbol"], r["ret_20d_pct"]) for r in rows[:8]])
        picks = pick_candidates(rows, account)
        reasons = {p["symbol"]: p["reason"] for p in picks}
        log.info("후보: %s", list(reasons))
        db_update_run(run_id, candidates=json.dumps(picks, ensure_ascii=False))

        # 2) 종목별 판단 (후보 + 보유)
        symbols = list(dict.fromkeys(list(reasons) + list(account["holdings"])))
        decisions = decide_all(toss, symbols, account, reasons)
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
        plan = allocate(decisions, account, session)
        log.info("배분 요약: %s", plan["summary"])
        orders, skipped = validate_orders(plan, decisions, account, session)
        for o in orders:
            row = {"run_id": run_id, "timestamp": _now(), "symbol": o["symbol"], "side": o["side"],
                   "quantity": o["quantity"], "amount_usd": o["amount_usd"], "price": o["price"],
                   "reason": o["reason"]}
            if dry:
                row |= {"status": "dry-run"}
                log.info("[dry-run] %s %s $%.2f (%s주)", o["side"], o["symbol"],
                         o["amount_usd"], o["quantity"])
            else:
                try:
                    r = place_order(toss, run_id, o)
                    row |= {"order_id": r.get("orderId"), "status": "submitted"}
                    log.info("주문 %s %s $%.2f → %s", o["side"], o["symbol"], o["amount_usd"],
                             r.get("orderId"))
                except TossError as e:
                    row |= {"status": f"error {e.code}: {e}"}
                    log.error("주문 실패 %s %s: %s", o["side"], o["symbol"], e)
            db_insert("orders", row)
        for o in skipped:
            db_insert("orders", {"run_id": run_id, "timestamp": _now(), "symbol": o.get("symbol"),
                                 "side": o.get("side"), "amount_usd": o.get("amount_usd"),
                                 "status": "skipped: " + o["skipped"], "reason": o.get("reason")})
        db_update_run(run_id, status="done", summary=plan["summary"])
        log.info("=== run %d 완료: 주문 %d건, 제외 %d건 ===", run_id, len(orders), len(skipped))
    except Exception as e:  # noqa: BLE001
        log.exception("run %d 실패: %s", run_id, e)
        db_update_run(run_id, status=f"error: {e}"[:300])
    finally:
        u = usage_snapshot()
        db_update_run(run_id, claude_calls=u["calls"], input_tokens=u["input_tokens"],
                      output_tokens=u["output_tokens"], cost_usd=u["cost_usd"])
        log.info("토큰: Claude %d회, 입력 %s, 출력 %s, 추정 $%.4f (%s API 단가 기준)",
                 u["calls"], f"{u['input_tokens']:,}", f"{u['output_tokens']:,}", u["cost_usd"], MODEL)


if __name__ == "__main__":
    initialize_db()
    log.info("모델 %s @ %s · 후보 %d · 최대 %d종목 · 종목당 %.0f%% · 현금유지 %.0f%% · 실행 %s%s",
             MODEL, BASE_URL, TOP_N, MAX_POSITIONS, MAX_POSITION_PCT,
             CASH_RESERVE_PCT, ", ".join(TRADE_TIMES), " · DRY_RUN" if DRY_RUN else " · 실주문")
    run_cycle(force="--force" in sys.argv or "--once" in sys.argv)   # 수동 실행은 장 시간 무시
    if "--once" in sys.argv:
        sys.exit(0)
    for t in TRADE_TIMES:
        schedule.every().day.at(t).do(run_cycle)
    while True:
        schedule.run_pending()
        time.sleep(1)
