"""주도주 모멘텀 + QQQ 200일선 시장 필터 — 결정 규칙 (research/leaders_bt.py 전용). **REJECT 후 폐기 (2026-09-17)** — 운영에 쓰지 않는다.

연구 기록 재현용으로만 남긴다.
  시장   QQQ 종가 > 200일 SMA 일 때만 매매. 아니면 보유 전량 매도 · 신규 매수 없음
  종목   시총 ≥ $50B 중 3개월(63거래일) 상대강도 상위 10% (올림)
         ※ RS = 종목 수익률 / S&P500 수익률. 같은 날엔 분모가 모든 종목에 같아 **순위가 3개월 수익률 순위와 동일**
           → S&P500 자료 없이 수익률로 순위를 매긴다 (선별 결과 동일)
  진입   종가 > 20일선 이고 (a) 종가 > 직전 252일 고가 (52주 신고가 돌파)
                         또는 (b) 최근 3일 안에 저가가 10일선·20일선 중 높은 쪽 이하로 닿은 뒤 오늘 양봉(종가 > 시가)
  청산   진입가 대비 −6% · 종가 20일선 하향 이탈 · 시장 필터 OFF
  슬롯   최대 3종목 · 종목당 총자산 30% (현금 10% 이상 유지)
"""
import math

import pandas as pd

SLOTS = 3
WEIGHT = 0.30
STOP = -0.06
MIN_CAP = 50e9
RS_TOP = 0.10
RS_DAYS = 63
TOUCH_DAYS = 3


def market_on(qqq_close):
    """QQQ 종가 Series (결정일까지) → True = 시스템 ON. 200봉 미만이면 판정 불가 → OFF."""
    sma = qqq_close.rolling(200).mean()
    return bool(qqq_close.iloc[-1] > sma.iloc[-1])      # NaN 비교는 False


def indicators(o, h, l, c):
    """일봉 DataFrame(날짜×종목) → 지표 dict. 전부 후행 창이라 t 행은 t 이하 자료만 쓴다."""
    sma10, sma20 = c.rolling(10).mean(), c.rolling(20).mean()
    touch = l <= sma10.where(sma10 > sma20, sma20)          # 10·20일선 중 높은 쪽까지 내려왔다
    return {
        "close": c, "sma20": sma20,
        "rs": c / c.shift(RS_DAYS) - 1,
        "breakout": c > h.shift(1).rolling(252, min_periods=252).max(),
        "pullback": (touch.astype(float).rolling(TOUCH_DAYS, min_periods=1).max() > 0) & (c > o),
    }


def exits(held, price, sma20, on):
    """held {종목: 진입가}, price/sma20 {종목: 값} → {종목: 사유}. 손절이 20일선보다 먼저다."""
    out = {}
    for s, entry in held.items():
        p = price.get(s)
        if not on:
            out[s] = "QQQ ≤ 200일선 — 전량 현금화"
        elif p is None or not math.isfinite(p):
            continue                                        # 가격 모름 → 판단 보류 (추측으로 팔지 않는다)
        elif p <= entry * (1 + STOP):
            out[s] = f"손절 {p / entry - 1:+.1%}"
        elif p < sma20.get(s, math.nan):
            out[s] = "20일선 하향 이탈"
    return out


def leaders(rs, cap):
    """rs·cap: {종목: 값} (결정일) → 상위 10% 주도주, RS 내림차순."""
    pool = sorted((v, s) for s, v in rs.items()
                  if math.isfinite(v) and math.isfinite(cap.get(s, math.nan)) and cap[s] >= MIN_CAP)
    k = math.ceil(len(pool) * RS_TOP)
    return [s for _, s in reversed(pool[len(pool) - k:])] if k else []


def entries(day, cap, held, on):
    """day: {지표명: {종목: 값}} (결정일) → 살 종목 목록 (빈 슬롯 수만큼, RS 순)."""
    if not on:
        return []
    free = SLOTS - len(held)
    out = []
    for s in leaders(day["rs"], cap):
        if len(out) >= free:
            break
        if s in held or not day["close"][s] > day["sma20"][s]:
            continue
        if day["breakout"][s] or day["pullback"][s]:
            out.append(s)
    return out


if __name__ == "__main__":
    # 자체 점검 — 규칙이 깨지면 실패한다
    import numpy as np
    q = pd.Series(np.r_[np.full(199, 100.0), 110.0])
    assert market_on(q) and not market_on(pd.Series(np.full(200, 100.0))) and not market_on(q.iloc[:150])
    assert exits({"A": 100}, {"A": 94.0}, {"A": 90}, True) == {"A": "손절 -6.0%"}
    assert exits({"A": 100}, {"A": 99.0}, {"A": 100}, True) == {"A": "20일선 하향 이탈"}
    assert exits({"A": 100}, {"A": 120.0}, {"A": 100}, True) == {}
    assert list(exits({"A": 100}, {}, {}, False)) == ["A"]
    rs = {f"S{i}": i / 100 for i in range(25)}
    cap = {s: 60e9 for s in rs} | {"S24": 10e9}               # 최고 RS 이지만 소형 → 제외
    assert leaders(rs, cap) == ["S23", "S22", "S21"]          # 24종목의 10% 올림 = 3
    day = {"rs": rs, "close": {s: 10 for s in rs}, "sma20": {s: 9 for s in rs},
           "breakout": {s: s == "S21" for s in rs}, "pullback": {s: s == "S22" for s in rs}}
    assert entries(day, cap, {}, True) == ["S22", "S21"] and entries(day, cap, {}, False) == []
    assert entries(day, cap, {"X": 1, "Y": 1}, True) == ["S22"]
    o = c = pd.DataFrame({"A": np.linspace(10, 20, 300)})
    ind = indicators(o - 0.1, c, c - 0.1, c)
    assert ind["breakout"]["A"].iloc[-1] and not ind["pullback"]["A"].iloc[-1]
    print("leaders 자체 점검 통과")
