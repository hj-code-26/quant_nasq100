"""예측 -> 토스 API 매매 계획.

관점은 '데이 트레이딩'이 아니라 **보유 기간**이다. 익일 방향은 노이즈가
지배적이라 라벨로 쓰기에 나쁘고, 매일 신호를 갈아타면 거래비용만 쌓인다.
그래서 두 트랙 모두 보유 기간을 명시적으로 잡는다.

  주간(week):  라벨 = 5거래일 뒤 종가 방향.  보유 ~1주,  주 1회 점검. 예산 10%
  월간(month): 라벨 = 21거래일 뒤 종가 방향. 보유 ~1개월, 월 1회 점검. 예산 25%

매수/매도 적정선은 고정값이 아니라 각 종목의 walk-forward '표본 외' 예측으로
그리드 탐색해 샤프 최대 지점을 고른다. 단, 표본 외 성적이 우연과 구분되지
않으면(이항검정 p >= P_MAX) 그 종목·그 기간은 거래하지 않는다.

horizon > 1 의 두 가지 함정을 모두 처리한다:
  1) 라벨 누수 — 학습 구간 끝 h-1 개 표본의 라벨은 테스트 구간 가격을 참조한다.
     walk_forward(embargo=h-1) 로 그 구간을 버린다.
  2) 표본 겹침 — 이웃 표본이 미래 구간을 공유해 독립이 아니다. 유효 표본은
     대략 n/h 이므로 유의성 검정에서 표본 수를 그만큼 깎는다(effective_n).

사용:
  python -m backend.trader --scan          # 스캔 상위 종목 → 계획 → 체결(dry-run)
  python -m backend.trader --scan --top=3 --loose --live
  python -m backend.trader AAPL            # 단일 종목 계획만 출력 (기본)
  python -m backend.trader AAPL --live     # 실제 주문 실행
  python -m backend.trader --report        # 거래 일지 + 현재가로 손익 확인
"""
import json
import pathlib
import sys
import threading
import time

from scipy import stats

from . import backtest as bt
from . import data as data_mod
from . import nasdaq100
from .toss import TossClient

P_MAX = 0.05            # 이항검정 단측 p-value 상한 (거래 자격 게이트)
MIN_HOLDOUT_TRADES = 3  # 홀드아웃 승률을 믿으려면 최소 이만큼은 매매했어야
MIN_SCAN_ACC = 0.60     # 스캔 롤링 보정 정확도 하한 — 이 밑은 계획에 편성하지 않는다

# 트랙 정의: 라벨 전망 기간, 예산 비중, 점검 주기, walk 수.
#
# walks 가 트랙마다 다른 이유 — 겹침 보정 후 유효 표본이 n/horizon 이라,
# horizon 이 길수록 검증 구간을 늘리지 않으면 검정할 표본이 남지 않는다.
# 예) 6 walks = 126일 검증 → 월간(h=21)은 유효 표본이 6개뿐이라 어떤 성적도
#     유의하다고 말할 수 없다. 24 walks(504일)면 독립 관측 24개가 확보된다.
TRACKS = {
    "week":  {"horizon": 5,  "budget": 0.10, "label": "주간",
              "review": "주 1회", "walks": 12},   # 252일 검증 → 유효 ~50개
    "month": {"horizon": 21, "budget": 0.25, "label": "월간",
              "review": "월 1회", "walks": 24},   # 504일 검증 → 유효 ~24개
}
JOURNAL = pathlib.Path(__file__).with_name("trade_journal.jsonl")
# 산출한 계획의 요약 기록 — 기준일별로 다시 묶어 보기 위한 것. 계획 본문은
# 메모리에만 있고 서버를 내리면 사라지므로, 목록에 필요한 값만 여기 남긴다.
PLAN_LOG = pathlib.Path(__file__).with_name("plan_history.jsonl")

# 계획 산출은 VAE 2회 학습 + 36 walks 라 수 분 걸린다. HTTP 를 막지 않도록
# 백그라운드 스레드로 돌리고 상태만 폴링한다.
_PLANS: dict = {}          # ticker -> {status, message, progress, plan, error}
_plock = threading.Lock()


def get_plan_state(ticker: str) -> dict:
    with _plock:
        return dict(_PLANS.get(ticker.upper(),
                               {"status": "idle", "progress": 0.0, "message": ""}))


def _set_plan(ticker: str, **kw):
    with _plock:
        _PLANS.setdefault(ticker.upper(), {}).update(kw)


def start_plan(ticker: str, force: bool = False) -> dict:
    """계획 산출을 백그라운드로 시작."""
    ticker = ticker.upper()
    st = get_plan_state(ticker)
    if st.get("status") == "running":
        return st
    if not force and st.get("status") == "done":
        return st

    def run():
        try:
            _set_plan(ticker, status="running", progress=0.05,
                      message="주간(5일) 모델 학습 중…", plan=None, error=None)
            df = data_mod.history(ticker, "5y")
            legs = {}
            for i, t in enumerate(("week", "month")):
                _set_plan(ticker, progress=0.1 + 0.45 * i,
                          message=f"{TRACKS[t]['label']}"
                                  f"({TRACKS[t]['horizon']}일) 모델 학습·검증 중…")
                legs[t] = track_plan(ticker, t, df=df)
            plan = {
                "ticker": ticker, "date": str(df.index[-1].date()),
                "last_close": float(df["Close"].iloc[-1]),
                "tradable": any(v["tradable"] for v in legs.values()),
                "win_rate_pct": plan_win_rate(legs),
                "tracks": legs,
            }
            log_plan(plan)
            _set_plan(ticker, status="done", progress=1.0, message="완료",
                      plan=plan)
        except Exception as e:  # noqa: BLE001
            _set_plan(ticker, status="error", progress=1.0,
                      message=f"오류: {e}", error=str(e))

    _set_plan(ticker, status="running", progress=0.01, message="시작…")
    threading.Thread(target=run, daemon=True).start()
    time.sleep(0.05)
    return get_plan_state(ticker)


def scan_row(ticker: str) -> dict:
    """스캔 결과에서 해당 종목 행. 스캔 전이면 빈 dict."""
    return next((r for r in nasdaq100.get_state()["results"]
                 if r.get("ticker") == ticker.upper()), {})


def log_plan(plan: dict):
    """계획 요약을 기준일과 함께 기록."""
    s = scan_row(plan["ticker"])
    row = {"date": plan["date"], "ticker": plan["ticker"], "ts": time.time(),
           "last_close": plan["last_close"], "tradable": plan["tradable"],
           "win_rate_pct": plan.get("win_rate_pct"),
           "rolling_acc": s.get("rolling_acc"), "scan_as_of": s.get("as_of"),
           "actions": {k: v["action"] for k, v in plan["tracks"].items()}}
    with PLAN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def plans_by_date(min_acc: float = MIN_SCAN_ACC) -> dict:
    """산출한 계획을 기준일별로 정리. 스캔 정확도 min_acc 미만은 제외.

    같은 날 같은 종목을 여러 번 돌렸으면 마지막 산출만 남긴다.
    """
    if not PLAN_LOG.exists():
        return {"min_acc": min_acc, "dates": [], "excluded": 0}
    rows = [json.loads(l) for l in
            PLAN_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    latest = {}
    for r in sorted(rows, key=lambda r: r.get("ts", 0)):
        latest[(r["date"], r["ticker"])] = r
    keep, drop = [], 0
    for r in latest.values():
        if (r.get("rolling_acc") or 0) >= min_acc:
            keep.append(r)
        else:
            drop += 1
    by = {}
    for r in keep:
        by.setdefault(r["date"], []).append(r)
    return {
        "min_acc": min_acc, "excluded": drop,
        "dates": [{"date": d,
                   "plans": sorted(by[d], key=lambda r: r.get("win_rate_pct") or 0,
                                   reverse=True)}
                  for d in sorted(by, reverse=True)],
    }


def effective_n(n: int, horizon: int) -> int:
    """겹치는 표본의 유효 개수. horizon 일마다 1개만 독립으로 센다."""
    return max(1, n // max(1, horizon))


def edge_test(per_day, horizon: int = 1):
    """표본 외 예측이 동전 던지기보다 유의하게 나은지 이항검정.

    horizon>1 이면 표본이 겹쳐 독립이 아니므로 유효 표본 수로 깎아서 검정한다.
    이 보정을 빼면 p-value 가 과소평가되어 없는 예측력을 있다고 판정한다.
    """
    n = len(per_day)
    if not n:
        return {"n": 0, "n_eff": 0, "accuracy": float("nan"),
                "p_value": 1.0, "significant": False, "horizon": horizon}
    hits = sum(1 for d in per_day if (d["prob"] >= 0.5) == bool(d["actual"]))
    acc = hits / n
    ne = effective_n(n, horizon)
    he = round(acc * ne)                       # 같은 정확도, 깎인 표본 수
    pv = float(stats.binomtest(he, ne, 0.5, alternative="greater").pvalue)
    return {"n": n, "n_eff": ne, "hits": hits, "hits_eff": he,
            "accuracy": acc, "p_value": pv, "significant": pv < P_MAX,
            "horizon": horizon}


def split_by_walk(per_day, holdout_frac=0.34):
    """walk 단위 시간순 분할 → (보정 구간, 홀드아웃 구간).

    임계값 그리드 탐색은 128가지 조합을 같은 표본에서 훑기 때문에, 탐색에
    쓴 구간의 승률·샤프는 반드시 부풀려진다. 그래서 뒤쪽 walk 들은 탐색에서
    빼두고, 확정된 임계값으로 단 한 번만 채점한다.
    """
    walks = sorted({d.get("walk", 1) for d in per_day})
    if len(walks) < 3:
        return per_day, []
    hold = set(walks[-max(1, round(len(walks) * holdout_frac)):])
    return ([d for d in per_day if d.get("walk", 1) not in hold],
            [d for d in per_day if d.get("walk", 1) in hold])


def holdout_backtest(per_day, closes_by_date, holdout_frac=0.34):
    """앞 구간에서 임계값을 고르고, 뒤 구간에서 그 임계값으로만 채점.

    반환 stats 의 승률·수익률은 임계값 탐색에 한 번도 쓰이지 않은 구간의
    값이라, 종목 간 줄세우기에 쓸 수 있는 유일한 수치다.
    """
    cal, hold = split_by_walk(per_day, holdout_frac)
    th = calibrate_trade_thresholds(cal, closes_by_date)
    r = bt.run_backtest(hold, closes_by_date, buy_th=th["buy_th"],
                        sell_th=th["sell_th"]) if hold else None
    st = dict(r["stats"]) if r else {"win_rate_pct": float("nan"), "trades": 0,
                                     "strategy_return_pct": 0.0,
                                     "buyhold_return_pct": 0.0, "sharpe": 0.0,
                                     "mdd_pct": 0.0, "days": 0}
    st["days_cal"] = len(cal)
    return th, st


def calibrate_trade_thresholds(per_day, closes_by_date, min_trades=2):
    """표본 외 예측 위에서 (buy_th, sell_th) 그리드 탐색 — 샤프 최대.

    주의: 이 함수가 돌려주는 sharpe/return 은 탐색에 쓴 바로 그 표본의
    값이므로 낙관 편향이다. 종목 선별에는 holdout_backtest() 를 써라.
    """
    best = None
    b = 0.50
    while b <= 0.65 + 1e-9:
        s = 0.35
        while s <= b - 0.02 + 1e-9:
            r = bt.run_backtest(per_day, closes_by_date, buy_th=round(b, 2),
                                sell_th=round(s, 2))
            if r and r["stats"]["trades"] >= min_trades:
                if best is None or r["stats"]["sharpe"] > best["sharpe"]:
                    best = {"buy_th": round(b, 2), "sell_th": round(s, 2),
                            "sharpe": r["stats"]["sharpe"],
                            "return_pct": r["stats"]["strategy_return_pct"],
                            "buyhold_pct": r["stats"]["buyhold_return_pct"],
                            "mdd_pct": r["stats"]["mdd_pct"],
                            "trades": r["stats"]["trades"]}
            s += 0.02
        b += 0.02
    return best or {"buy_th": 0.55, "sell_th": 0.45, "sharpe": 0.0,
                    "return_pct": 0.0, "buyhold_pct": 0.0,
                    "mdd_pct": 0.0, "trades": 0}


def track_plan(ticker, track, df=None, period="5y", epochs=12):
    """한 트랙(주간/월간)의 계획. 해당 horizon 으로 모델을 새로 학습한다."""
    cfg = TRACKS[track]
    h, n_walks = cfg["horizon"], cfg["walks"]
    if df is None:
        df = data_mod.history(ticker, period)

    from . import gaf, models, vae
    X, y, dates = gaf.build_dataset(df, 20, horizon=h)
    lab = y >= 0
    dl = [d for d, m in zip(dates, lab) if m]
    m = vae.train_vae(X[lab], epochs=epochs, latent_dim=15)
    Z = vae.extract_latents(m, X)
    wf = models.walk_forward(Z[lab], y[lab], dl, n_walks=n_walks,
                             test_days=21, embargo=h - 1)
    closes = {str(i.date()): float(v) for i, v in df["Close"].items()}

    edge = edge_test(wf["per_day"], horizon=h)
    th, hold = holdout_backtest(wf["per_day"], closes)
    p = float(models.predict_proba(models.fit_classifiers(Z[lab], y[lab]),
                                   Z[-1:])["ensemble"][0])
    last = float(df["Close"].iloc[-1])
    sma = float(df["Close"].rolling(20).mean().iloc[-1])

    # 거래 자격 = 방향 예측이 우연과 구분되고(이항검정),
    #             홀드아웃 구간에서 실제로 돈을 벌었을 것(승률·매매횟수).
    hold_ok = (hold["trades"] >= MIN_HOLDOUT_TRADES
               and hold["win_rate_pct"] > 50.0
               and hold["strategy_return_pct"] > 0)
    tradable = edge["significant"] and hold_ok
    if not tradable:
        action = "HOLD"
    elif p >= th["buy_th"] and last >= sma:      # 추세 필터: 두 트랙 공통
        action = "BUY"
    elif p <= th["sell_th"] or last < sma:
        action = "SELL"
    else:
        action = "HOLD"

    return {
        "track": track, "label": cfg["label"], "horizon_days": h,
        "review": cfg["review"], "budget_frac": cfg["budget"],
        "action": action, "tradable": tradable,
        "p_up": round(p, 4), "edge": edge, "thresholds": th,
        "holdout": hold, "win_rate_pct": hold["win_rate_pct"],
        "last_close": last, "sma20": sma,
        "hold_note": f"{h}거래일(≈{cfg['label']}) 보유 전제",
        "reason": (f"p={p:.3f} vs 매수≥{th['buy_th']}/매도≤{th['sell_th']}, "
                   f"종가 {last:.2f} vs SMA20 {sma:.2f}, "
                   f"홀드아웃 승률 {hold['win_rate_pct']:.1f}% "
                   f"({hold['trades']}회 매매)"),
    }


def make_plan(ticker, tracks=("week", "month")):
    """주간·월간 계획을 함께 산출."""
    df = data_mod.history(ticker, "5y")
    legs = {t: track_plan(ticker, t, df=df) for t in tracks}
    return {
        "ticker": ticker,
        "date": str(df.index[-1].date()),
        "last_close": float(df["Close"].iloc[-1]),
        "tradable": any(l["tradable"] for l in legs.values()),
        "win_rate_pct": plan_win_rate(legs),
        "tracks": legs,
    }


def plan_win_rate(legs) -> float:
    """계획의 대표 승률 = 자격 있는 트랙 중 최고 홀드아웃 승률.

    자격 있는 트랙이 없으면 어차피 주문하지 않으므로 참고용으로 전체 최고값.
    """
    ok = [l for l in legs.values() if l["tradable"]] or list(legs.values())
    return max((l.get("win_rate_pct") or 0.0) for l in ok) if ok else 0.0


def execute(plan, live=False, budget_scale=1.0):
    """계획 실행. live=False 면 계산만(dry-run).

    budget_scale: 여러 종목에 동시 진입할 때 종목당 예산 배분 비율(1/N).
    """
    t = TossClient()
    results = []
    for name, leg in plan["tracks"].items():
        entry = {"track": name, "label": leg["label"], "action": leg["action"],
                 "ticker": plan["ticker"], "date": plan["date"],
                 "horizon_days": leg["horizon_days"], "p_up": leg["p_up"],
                 "live": live, "ts": time.time()}
        if leg["action"] == "BUY" and leg["tradable"]:
            power = float(t.buying_power("USD")["cashBuyingPower"])
            qty = int(power * leg["budget_frac"] * budget_scale
                      / plan["last_close"])
            entry.update(quantity=qty, limit_price=plan["last_close"],
                         budget_scale=budget_scale)
            if qty < 1:
                entry["skipped"] = "예산 부족"
            elif live:
                entry["order"] = t.buy(plan["ticker"], quantity=str(qty),
                                       price=f"{plan['last_close']:.2f}")
        elif leg["action"] == "SELL" and leg["tradable"]:
            items = ((t.holdings(symbol=plan["ticker"]) or {}).get("items")
                     if live else None) or []
            qty = items[0]["quantity"] if items else "0"
            entry["quantity"] = qty
            if live and float(qty) > 0:
                entry["order"] = t.sell(plan["ticker"], quantity=qty)
        results.append(entry)
        with JOURNAL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return results


# ---------- 스캔 → 계획 → 체결 ----------
#
# 종목을 사람이 고르는 대신 스캔 결과에서 뽑는다. 스캔은 고속 모드(2년·3 walks)
# 라 표본이 63개뿐이고 250종목 중 상위를 고르는 다중비교 편향이 있다. 그래서
# 스캔은 '후보 선별'까지만 하고, 실제 거래 자격은 각 종목에서 다시 학습한
# 주간/월간 트랙의 표본 외 이항검정(P_MAX)이 판정한다.
_PORT = {"status": "idle", "progress": 0.0, "message": "",
         "candidates": [], "plans": [], "error": None}
_polock = threading.Lock()


def scan_candidates(top: int = 5, strict: bool = True) -> list:
    """스캔 결과 → 표본 외 백테스트 승률 상위 N.

    1차 컷은 스캔의 롤링 보정 정확도 MIN_SCAN_ACC(60%) 이상.
    (롤링 보정 = 그날 이전 walk 들로만 정한 임계값으로 채점한 값이라
     임계값 탐색 편향이 없다. 스캔표의 'WF 정확도'는 임계값 0.5 고정값이다.)
    통과한 종목들 사이의 순위는 '그 신호로 실제 매매했을 때의 승률'로 매긴다.
    매매가 거의 없었던 종목은 승률이 우연 그 자체이므로 제외한다.

    strict=True 면 방향 예측의 유의성(p<0.05)까지 요구한다. 250종목 중 이
    조건을 넘는 건 한두 개뿐인 게 정상이다(우연 기대치와 같은 수준).
    strict=False 는 그 필터를 빼는 '넓은 후보' 모드 — 어차피 주문 여부는
    종목별 홀드아웃 백테스트와 본페로니 보정 검정이 다시 판정한다.
    """
    rows = nasdaq100.get_state()["results"]
    cand = [r for r in rows
            if r.get("direction") == "상승"
            # 스캔 단계 1차 컷: 롤링 보정 정확도 60% 이상만 편성한다
            and (r.get("rolling_acc") or 0) >= MIN_SCAN_ACC
            # 다수 클래스만 찍는 것보다 못한 모델은 상승 신호도 의미가 없다
            and r.get("accuracy", 0) >= r.get("baseline", 1)
            and (r.get("bt_trades", 0) or 0) >= MIN_HOLDOUT_TRADES
            and (r.get("significant") or not strict)]
    cand.sort(key=lambda r: (r.get("win_rate_pct") or 0.0,
                             r.get("accuracy", 0)), reverse=True)
    return cand[:top]


def get_portfolio_state() -> dict:
    with _polock:
        return dict(_PORT)


def _set_port(**kw):
    with _polock:
        _PORT.update(kw)


def start_portfolio(top: int = 5, force: bool = False,
                    strict: bool = True) -> dict:
    """스캔 상위 종목들에 대해 매매 계획을 순차 산출 (백그라운드)."""
    st = get_portfolio_state()
    if st["status"] == "running":
        return st
    if not force and st["status"] == "done":
        return st
    cand = scan_candidates(top, strict=strict)
    if not cand:
        _set_port(status="error", progress=1.0, plans=[], candidates=[],
                  message="스캔 결과에 조건을 만족하는 종목이 없습니다"
                          + (" (유의 + 상승). strict 를 끄거나" if strict
                             else " (상승).")
                          + " 먼저 스캔을 실행하세요.",
                  error="no_candidates")
        return get_portfolio_state()

    def run():
        plans = []
        try:
            for i, c in enumerate(cand):
                _set_port(status="running", progress=i / len(cand),
                          message=f"{c['ticker']} 계획 산출 중… "
                                  f"({i + 1}/{len(cand)})", plans=list(plans))
                try:
                    plan = make_plan(c["ticker"])
                    log_plan(plan)
                    plans.append(plan | {"scan": c})
                except Exception as e:  # noqa: BLE001
                    plans.append({"ticker": c["ticker"], "error": str(e)[:120],
                                  "tradable": False, "tracks": {}, "scan": c})
            plans = rank_plans(plans)
            ok = sum(1 for p in plans if p.get("tradable"))
            _set_port(status="done", progress=1.0, plans=plans,
                      message=f"{ok}/{len(plans)} 종목 거래 자격 있음"
                              + (f" · 최고 홀드아웃 승률 "
                                 f"{plans[0]['win_rate_pct']:.1f}%" if ok else ""))
        except Exception as e:  # noqa: BLE001
            _set_port(status="error", progress=1.0, plans=plans,
                      message=f"오류: {e}", error=str(e))

    _set_port(status="running", progress=0.01, plans=[], candidates=cand,
              error=None, message="시작…")
    threading.Thread(target=run, daemon=True).start()
    time.sleep(0.05)
    return get_portfolio_state()


def rank_plans(plans: list) -> list:
    """홀드아웃 승률 내림차순 정렬 + 다중비교 보정.

    N개 후보를 훑어 그 중 최고를 고르면, 종목당 p<0.05 를 통과했다는 사실만으로는
    부족하다(N=5면 하나라도 우연히 통과할 확률이 23%). 본페로니로 종목당 기준을
    P_MAX/N 까지 조인다 — 후보를 넓게 볼수록 개별 종목의 증거 요구치가 올라간다.
    """
    n = max(1, len(plans))
    for p in plans:
        legs = list(p.get("tracks", {}).values())
        for l in legs:
            l["tradable"] = l["tradable"] and l["edge"]["p_value"] < P_MAX / n
        if legs:
            p["tradable"] = any(l["tradable"] for l in legs)
            p["win_rate_pct"] = plan_win_rate(p["tracks"])
        p["selection_p_max"] = P_MAX / n
    return sorted(plans, key=lambda p: (p.get("tradable", False),
                                        p.get("win_rate_pct") or 0.0),
                  reverse=True)


def execute_portfolio(live: bool = False) -> list:
    """거래 자격 있는 계획만 체결. 예산은 종목 수로 균등 분할."""
    plans = [p for p in get_portfolio_state()["plans"] if p.get("tradable")]
    if not plans:
        return []
    scale = 1.0 / len(plans)
    out = []
    for p in plans:
        out += execute(p, live=live, budget_scale=scale)
    return out


def report():
    """일지 + 현재가 대조로 결과 확인."""
    if not JOURNAL.exists():
        return "일지 없음"
    entries = [json.loads(l) for l in
               JOURNAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    t = TossClient()
    lines = []
    for e in entries:
        now = float(t.prices(e["ticker"])[0]["lastPrice"])
        ref = e.get("limit_price")
        pnl = (f" 손익 {((now / ref) - 1) * 100:+.2f}%"
               if e["action"] == "BUY" and ref else "")
        lines.append(
            f"{e['date']} [{e.get('label', e.get('leg', '?'))}] {e['action']} "
            f"{e['ticker']} x{e.get('quantity', '-')} @{ref or '-'} "
            f"(현재 {now}){pnl}{' [주문됨]' if e.get('order') else ' [dry-run]'}")
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--report" in args:
        print(report())
        sys.exit(0)
    sys.stdout.reconfigure(encoding="utf-8")
    live = "--live" in args

    if "--scan" in args:
        top = int(next((a.split("=")[1] for a in args
                        if a.startswith("--top=")), 5))
        st = start_portfolio(top, force=True,
                             strict="--loose" not in args)
        while st["status"] == "running":
            print(f"  {st['message']}")
            time.sleep(5)
            st = get_portfolio_state()
        if st.get("error"):
            print(st["message"])
            sys.exit(1)
        for p in st["plans"]:
            if p.get("error"):
                print(f"{p['ticker']}: 오류 {p['error']}")
                continue
            legs = " | ".join(
                f"{l['label']} {l['action']}"
                f"{'' if l['tradable'] else '(자격없음)'}"
                for l in p["tracks"].values())
            print(f"{p['ticker']:6s} ${p['last_close']:8.2f}  "
                  f"홀드아웃 승률 {p.get('win_rate_pct', 0):5.1f}%  {legs}")
        print()
        print(st["message"])
        for r in execute_portfolio(live=live):
            print(("주문 실행: " if live else "dry-run: ")
                  + json.dumps(r, ensure_ascii=False))
        sys.exit(0)

    ticker = next((a for a in args if not a.startswith("-")), "AAPL")
    plan = make_plan(ticker)
    print(f"\n{plan['ticker']}  기준일 {plan['date']}  종가 ${plan['last_close']:.2f}\n")
    for leg in plan["tracks"].values():
        e = leg["edge"]
        print(f"[{leg['label']}] {leg['horizon_days']}거래일 전망 · {leg['review']} 점검 "
              f"· 예산 {leg['budget_frac']:.0%}")
        print(f"   판정: {leg['action']}   ({leg['reason']})")
        print(f"   표본 외 정확도 {e['accuracy']:.1%} "
              f"({e['hits']}/{e['n']}, 겹침 보정 유효표본 {e['n_eff']}개) "
              f"p={e['p_value']:.3f} → "
              f"{'거래 자격 있음' if e['significant'] else '우연과 구분 불가'}")
        th = leg["thresholds"]
        print(f"   임계값 매수≥{th['buy_th']} 매도≤{th['sell_th']} | "
              f"샤프 {th['sharpe']:.2f} 수익 {th['return_pct']:.1f}% "
              f"(보유 {th['buyhold_pct']:.1f}%) 거래 {th['trades']}회\n")
    if not plan["tradable"]:
        print("두 트랙 모두 거래 자격 없음 — 주문하지 않습니다.")
        sys.exit(0)
    for r in execute(plan, live=live):
        print(("주문 실행: " if live else "dry-run: ")
              + json.dumps(r, ensure_ascii=False))
