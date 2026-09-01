"""
백테스트 (개선사항) — Walk-forward 검증의 '표본 외' 예측만으로
전략 수익률을 계산해 시기 분석을 수익 관점으로 확장한다.

전략: 앙상블 상승확률 >= buy_th 이면 매수(보유), <= sell_th 이면 청산,
      중립 구간에서는 직전 포지션 유지. 거래비용 반영.
비교: 단순 보유(Buy & Hold).
"""
import math

import numpy as np


# 편도 거래비용(bp). 토스증권 실계좌 조회값 기준 — 미국 주식 수수료 0.100% = 10bp.
# (GET /api/v1/commissions, 2026-09 확인). 왕복이면 20bp 가 든다.
# 스프레드·환전은 미반영이라 이 값도 낙관적이다.
# 국내(KR) 종목을 넣으려면 매도 거래세(약 18bp)를 별도로 더해야 한다.
US_COST_BP = 10.0


def run_backtest(per_day, closes_by_date, buy_th=0.55, sell_th=0.45,
                 cost_bp=US_COST_BP):
    """
    per_day: walk_forward()의 per_day (표본 외 예측: date, prob)
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
        # 종가 기준 신호 -> 다음날부터 포지션 반영
        new_pos = pos
        if d["prob"] >= buy_th:
            new_pos = 1
        elif d["prob"] <= sell_th:
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
            "buy_th": buy_th,
            "sell_th": sell_th,
            "cost_bp": cost_bp,
        },
    }


def calibrate_threshold(per_day, lo=0.40, hi=0.60, step=0.01):
    """
    결정 임계값 보정 (개선사항) — 논문은 0.5 고정 경계값을 사용하지만,
    표본 외(walk-forward) 예측에서 정확도를 최대화하는 임계값을 탐색한다.
    """
    if not per_day:
        return {"threshold": 0.5, "accuracy": float("nan")}
    probs = np.array([d["prob"] for d in per_day])
    actual = np.array([d["actual"] for d in per_day])
    best_t, best_a = 0.5, -1.0
    t = lo
    while t <= hi + 1e-9:
        acc = float(((probs >= t).astype(int) == actual).mean())
        if acc > best_a:
            best_a, best_t = acc, round(t, 2)
        t += step
    return {"threshold": best_t, "accuracy": best_a}
