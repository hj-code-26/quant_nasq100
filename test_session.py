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

# 사전 분석 창: 정규장 개장(22:30) 전 60분 안에서만 연다.
with at(22, 0):                        # 개장 30분 전 → 열림
    assert round(a.analysis_lead_min(sess)) == 30
with at(21, 0):                        # 개장 90분 전 → 아직
    assert a.analysis_lead_min(sess) is None
with at(23, 0):                        # 이미 개장 → 정규 사이클이 맡는다
    assert a.analysis_lead_min(sess) is None
assert a.analysis_lead_min(None) is None          # 휴장일
print("ok")
