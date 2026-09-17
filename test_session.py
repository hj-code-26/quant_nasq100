"""장 구간 판정 자체 점검 (프리장·애프터장 포함)."""
import datetime
from unittest.mock import patch
import autotrade as a

KST = a.KST
d = datetime.datetime(2026, 9, 8, tzinfo=KST)
nx = d + datetime.timedelta(days=1)
sess = (d.replace(hour=17), d.replace(hour=22, minute=30),   # 프리 개장, 정규 개장
        nx.replace(hour=5), nx.replace(hour=9))              # 정규 마감, 애프터 마감


def at(h, m=0, day=0):
    return patch.object(a.datetime, "datetime", wraps=datetime.datetime,
                        **{"now.return_value": d.replace(hour=h, minute=m)
                           + datetime.timedelta(days=day)})


with at(16, 30):                       # 프리장 개장 30분 전
    assert a.session_block(sess).startswith("장 시작 전")
with at(17, 1):                        # 프리장
    assert a.session_block(sess) is None
    assert a.extended_hours(sess) and not a.fractional_allowed(sess)
with at(23):                           # 정규장
    assert not a.extended_hours(sess) and a.fractional_allowed(sess)
with at(4, 30, day=1):                 # 정규 마감 1시간 전 이후 → 정수 주만
    assert a.session_block(sess) is None and not a.fractional_allowed(sess)
    assert not a.extended_hours(sess)
with at(7, 30, day=1):                 # 애프터장 → 실행 가능, 지정가·정수 주
    assert a.session_block(sess) is None and not a.fractional_allowed(sess)
    assert a.extended_hours(sess)
with at(9, 30, day=1):                 # 애프터장 마감 후
    assert a.session_block(sess).startswith("장 마감 후")
assert a.session_block(None) == "휴장일"

# 사이클 모드: 장 밖은 건너뛰지 않고 주문 없는 사전 분석, 휴장일만 건너뜀
with at(16, 30):                       # 장 시작 전 → 분석
    assert a.cycle_mode(sess)[0] == "analysis"
    assert a.cycle_mode(sess, force=True) == ("trade", None)       # 수동 강제 실행은 그대로
with at(23):                           # 장중 → 매매
    assert a.cycle_mode(sess) == ("trade", None)
with at(9, 30, day=1):                 # 장 마감 후 → 분석
    assert a.cycle_mode(sess)[0] == "analysis"
assert a.cycle_mode(None) == ("skip", "휴장일")
assert a.cycle_mode(None, force=True) == ("skip", "휴장일")        # 강제여도 휴장일은 건너뜀

# 실전 예측의 기준일·목표일: 예측 시각 직전에 끝난 정규장 → 그다음 거래일
_d = datetime.date
_sess = [_d(2026, 9, 10), _d(2026, 9, 11), _d(2026, 9, 14)]          # 목·금·월 (주말 건너뜀)
_ny = lambda *x: datetime.datetime(*x, tzinfo=a.NY)
assert a.prediction_window(_ny(2026, 9, 11, 10, 0), _sess) == (_d(2026, 9, 10), _d(2026, 9, 11))   # 장중 → 어제 종가 기준
assert a.prediction_window(_ny(2026, 9, 11, 16, 30), _sess) == (_d(2026, 9, 11), _d(2026, 9, 14))  # 장 마감 후 → 오늘 기준, 다음은 월요일
assert a.prediction_window(_ny(2026, 9, 14, 17, 0), _sess) == (_d(2026, 9, 14), None)            # 목표일 아직 없음
assert a.prediction_window(_ny(2026, 9, 9, 12, 0), _sess) == (None, None)

# 성적표: 같은 종목·기준일의 여러 예측은 마지막만 센다, 표본이 적으면 판정 보류
import pandas as _pd
_df = _pd.DataFrame([
    {"made_at": "2026-09-11T10:00", "symbol": "A", "base_date": "2026-09-10", "up_prob": 30, "pct": -2, "actual_pct": 1.0},
    {"made_at": "2026-09-11T14:00", "symbol": "A", "base_date": "2026-09-10", "up_prob": 70, "pct": 0.5, "actual_pct": 1.0},
    {"made_at": "2026-09-11T14:00", "symbol": "B", "base_date": "2026-09-10", "up_prob": 65, "pct": 1.0, "actual_pct": -0.5},
    {"made_at": "2026-09-11T14:00", "symbol": "C", "base_date": "2026-09-10", "up_prob": 35, "pct": -3, "actual_pct": None}])
_s = a.prediction_stats(_df)
assert _s["채점 건수"] == 2 and _s["상승 확답(≥60) 건수"] == 2 and _s["상승 확답 적중 %"] == 50.0, _s
assert _s["실제 ≥ 보수적 % 비율"] == 50.0 and _s["판정"].startswith("표본 부족"), _s

# 사전 분석 창: 정규장 개장(22:30) 전 60분 안에서만 연다.
with at(22, 0):                        # 개장 30분 전 → 열림
    assert round(a.analysis_lead_min(sess)) == 30
with at(21, 0):                        # 개장 90분 전 → 아직
    assert a.analysis_lead_min(sess) is None
with at(23, 0):                        # 이미 개장 → 정규 사이클이 맡는다
    assert a.analysis_lead_min(sess) is None
assert a.analysis_lead_min(None) is None          # 휴장일
print("ok")
