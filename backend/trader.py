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
  python -m backend.trader AAPL            # 계획만 출력 (주문 안 나감, 기본)
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
from .toss import TossClient

P_MAX = 0.05            # 이항검정 단측 p-value 상한 (거래 자격 게이트)

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
                "tracks": legs,
            }
            _set_plan(ticker, status="done", progress=1.0, message="완료",
                      plan=plan)
        except Exception as e:  # noqa: BLE001
            _set_plan(ticker, status="error", progress=1.0,
                      message=f"오류: {e}", error=str(e))

    _set_plan(ticker, status="running", progress=0.01, message="시작…")
    threading.Thread(target=run, daemon=True).start()
    time.sleep(0.05)
    return get_plan_state(ticker)


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


def calibrate_trade_thresholds(per_day, closes_by_date, min_trades=2):
    """표본 외 예측 위에서 (buy_th, sell_th) 그리드 탐색 — 샤프 최대."""
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
    th = calibrate_trade_thresholds(wf["per_day"], closes)
    p = float(models.predict_proba(models.fit_classifiers(Z[lab], y[lab]),
                                   Z[-1:])["ensemble"][0])
    last = float(df["Close"].iloc[-1])
    sma = float(df["Close"].rolling(20).mean().iloc[-1])

    tradable = edge["significant"]
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
        "last_close": last, "sma20": sma,
        "hold_note": f"{h}거래일(≈{cfg['label']}) 보유 전제",
        "reason": (f"p={p:.3f} vs 매수≥{th['buy_th']}/매도≤{th['sell_th']}, "
                   f"종가 {last:.2f} vs SMA20 {sma:.2f}"),
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
        "tracks": legs,
    }


def execute(plan, live=False):
    """계획 실행. live=False 면 계산만(dry-run)."""
    t = TossClient()
    results = []
    for name, leg in plan["tracks"].items():
        entry = {"track": name, "label": leg["label"], "action": leg["action"],
                 "ticker": plan["ticker"], "date": plan["date"],
                 "horizon_days": leg["horizon_days"], "p_up": leg["p_up"],
                 "live": live, "ts": time.time()}
        if leg["action"] == "BUY" and leg["tradable"]:
            power = float(t.buying_power("USD")["cashBuyingPower"])
            qty = int(power * leg["budget_frac"] / plan["last_close"])
            entry.update(quantity=qty, limit_price=plan["last_close"])
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
    ticker = next((a for a in args if not a.startswith("-")), "AAPL")
    live = "--live" in args
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
