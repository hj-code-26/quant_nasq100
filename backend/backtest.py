"""
백테스트 (개선사항) — Walk-forward 검증의 '표본 외' 예측만으로
전략 수익률을 계산해 시기 분석을 수익 관점으로 확장한다.

전략: 상승확률 >= 임계값+밴드 이면 매수(보유), <= 임계값-밴드 이면 청산,
      중립 구간에서는 직전 포지션 유지. 거래비용 반영.
비교: 단순 보유(Buy & Hold).

임계값은 rolling_calibration() 이 각 walk 에 대해 '그 이전 walk 들만'
으로 정한 값을 쓴다. 전체 표본에서 정확도가 최대가 되는 임계값을 골라
같은 표본으로 채점하면 정확도가 구조적으로 부풀려지기 때문이다.
"""
import math

import numpy as np

from .models import accuracy_stats, majority_baseline

# 편도 거래비용(bp). 토스증권 실계좌 조회값 기준 — 미국 주식 수수료 0.100% = 10bp.
# (GET /api/v1/commissions, 2026-09 확인). 왕복이면 20bp 가 든다.
# 스프레드·환전은 미반영이라 이 값도 낙관적이다.
# 국내(KR) 종목을 넣으려면 매도 거래세(약 18bp)를 별도로 더해야 한다.
US_COST_BP = 10.0

NEUTRAL_BAND = 0.03


def _search_threshold(rows, lo=0.40, hi=0.60, step=0.01):
    """주어진 표본에서 정확도를 최대화하는 임계값 (표본 내 탐색)."""
    if not rows:
        return 0.5, float("nan")
    probs = np.array([r["prob"] for r in rows])
    actual = np.array([r["actual"] for r in rows])
    best_t, best_a = 0.5, -1.0
    t = lo
    while t <= hi + 1e-9:
        acc = float(((probs >= t).astype(int) == actual).mean())
        if acc > best_a:
            best_a, best_t = acc, round(t, 2)
        t += step
    return best_t, best_a


def rolling_calibration(per_day):
    """
    임계값 보정 — 누수 없는 방식 (수정사항).

    walk k 에 적용할 임계값은 walk 1..k-1 의 예측만으로 결정한다.
    첫 walk 은 참고할 과거가 없으므로 0.5 를 쓰고 채점에서 제외한다.
    각 per_day 항목에 그날 실제로 적용된 임계값(threshold)과 그 기준의
    적중 여부(hit_cal)를 기록한다.

    반환:
      threshold        다음(미래) 예측에 쓸 임계값 — 전체 과거로 결정
      accuracy         위 규칙으로 산출한 진짜 표본 외 정확도
      stats            그 정확도의 표본수·신뢰구간·p-value
      fixed            임계값 0.5 고정 기준 정확도(전 구간)
      insample_accuracy 전체 표본에서 임계값을 탐색했을 때의 정확도
                       (비교용. 편향된 값이며 화면 대표값으로 쓰지 않는다)
    """
    if not per_day:
        nan = float("nan")
        return {"threshold": 0.5, "accuracy": nan, "stats": accuracy_stats(0, 0),
                "fixed": accuracy_stats(0, 0), "insample_accuracy": nan,
                "baseline": 0.5,
                "n_eval": 0, "walks_scored": 0}

    order, groups = [], {}
    for d in per_day:
        w = d.get("walk", 1)
        if w not in groups:
            groups[w] = []
            order.append(w)
        groups[w].append(d)

    prior, hits, n_eval, scored = [], 0, 0, 0
    scored_rows = []
    for w in order:
        rows = groups[w]
        if prior:
            th, _ = _search_threshold(prior)
            evaluated = True
            scored += 1
        else:
            th, evaluated = 0.5, False
        for d in rows:
            d["threshold"] = th
            d["hit_cal"] = bool((d["prob"] >= th) == (d["actual"] == 1))
            if evaluated:
                n_eval += 1
                hits += int(d["hit_cal"])
                scored_rows.append(d)
        prior.extend(rows)

    next_th, insample_acc = _search_threshold(per_day)
    fixed_hits = sum(1 for d in per_day if (d["prob"] >= 0.5) == (d["actual"] == 1))
    base_all = majority_baseline(d["actual"] for d in per_day)
    base_eval = majority_baseline(d["actual"] for d in scored_rows)
    return {
        "threshold": next_th,
        "accuracy": (hits / n_eval) if n_eval else float("nan"),
        "stats": accuracy_stats(hits, n_eval, p0=base_eval),
        "fixed": accuracy_stats(fixed_hits, len(per_day), p0=base_all),
        "insample_accuracy": insample_acc,
        "baseline": base_all,
        "n_eval": n_eval,
        "walks_scored": scored,
    }


def run_backtest(per_day, closes_by_date, buy_th=None, sell_th=None,
                 cost_bp=US_COST_BP, band=NEUTRAL_BAND):
    """
    per_day: walk_forward()의 per_day (표본 외 예측: date, prob)
             rolling_calibration() 을 먼저 돌리면 각 항목의 threshold 를
             그대로 사용한다(그날 시점에 알 수 있었던 임계값).
    closes_by_date: {date(str): close}
    반환: 에쿼티 곡선 + 성과 지표 (총수익률, 샤프, MDD, 승률, 매매횟수)
    """
    days = [d for d in per_day if d["date"] in closes_by_date]
    if len(days) < 5:
        return None
    cost = cost_bp / 10000.0
    pos = 0
    equity, bh = 1.0, 1.0
    curve = []
    trades = 0
    strat_rets = []
    prev_close = None
    for d in days:
        c = closes_by_date[d["date"]]
        if prev_close is not None:
            r = c / prev_close - 1.0
            sr = pos * r
            equity *= (1.0 + sr)
            bh *= (1.0 + r)
            strat_rets.append(sr)
        # 그날 시점에 알 수 있었던 임계값 (없으면 인자/기본값)
        th = d.get("threshold")
        hi = buy_th if buy_th is not None else ((th if th is not None else 0.5) + band)
        lo = sell_th if sell_th is not None else ((th if th is not None else 0.5) - band)
        # 종가 기준 신호 -> 다음날부터 포지션 반영
        new_pos = pos
        if d["prob"] >= hi:
            new_pos = 1
        elif d["prob"] <= lo:
            new_pos = 0
        if new_pos != pos:
            equity *= (1.0 - cost)
            trades += 1
            pos = new_pos
        prev_close = c
        curve.append({"date": d["date"], "strategy": equity, "buyhold": bh,
                      "position": pos})
    rets = np.asarray(strat_rets)
    active = rets[rets != 0]
    sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0.0
    peak, mdd = 1.0, 0.0
    for p in curve:
        peak = max(peak, p["strategy"])
        mdd = min(mdd, p["strategy"] / peak - 1.0)
    return {
        "curve": curve,
        "stats": {
            "strategy_return_pct": (equity - 1.0) * 100.0,
            "buyhold_return_pct": (bh - 1.0) * 100.0,
            "sharpe": sharpe,
            "mdd_pct": mdd * 100.0,
            "win_rate_pct": float((active > 0).mean() * 100.0) if len(active) else 0.0,
            "trades": trades,
            "days": len(curve),
            "band": band,
            "cost_bp": cost_bp,
        },
    }
