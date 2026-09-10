"""§7-N1 수정안 — 국면지표 캔들의 최신성 판정 (연구 구현. **운영 미적용**).

현행 `autotrade.market_regime` 은 캔들이 얼마나 오래됐는지 보지 않는다. 2020년 캔들로도
"정상 판정"을 낸다(research/test_ops.py::t_stale_index_is_not_detected 로 재현).

여기서는 **거래소 캘린더 + 마지막 완료 세션**을 기준으로 네 상태를 구분한다.

  fresh       마지막 봉의 날짜 == 마지막 **완료된** 세션
  incomplete  마지막 봉이 아직 진행 중인 세션의 것 (미완성 봉) — 값이 확정되지 않았다
  stale       마지막 봉이 마지막 완료 세션보다 오래됨 (지연·정지)
  missing     캔들이 없거나 판정에 필요한 길이(61봉)에 못 미침

  + 중간 결측(gap): 창 안에 캘린더에는 있는데 봉이 없는 거래일 → 상태에 gap_days 로 덧붙인다.

정규장 마감은 16:00 ET, **조기폐장일은 13:00 ET**. DST 는 zoneinfo 가 처리하므로
UTC 오프셋을 손으로 더하지 않는다.

대응은 분리한다 — stale/missing 이라고 청산까지 멈추지 않는다:
  · 신규 진입: 차단 (낡은 국면 판정으로 사지 않는다)
  · 보유분: 대사·손절·만기·노출 축소는 **계속** (가격이 있는 종목은 그대로 관리)
  · 국면 자체가 필요한 규칙: '판정 불가'로 표시하고 폴백으로 조용히 갈아타지 않는다
"""
import datetime
import zoneinfo

NY = zoneinfo.ZoneInfo("America/New_York")
REGULAR_CLOSE = datetime.time(16, 0)
EARLY_CLOSE = datetime.time(13, 0)
MIN_BARS = 61                      # ret60 판정에 필요한 최소 봉 수


def last_completed_session(now, calendar, early_closes=()):
    """지금(NY tz) 기준 **마감이 끝난** 마지막 거래일. 캘린더는 거래일 date 리스트."""
    now = now.astimezone(NY)
    today = now.date()
    for d in sorted(calendar, reverse=True):
        if d > today:
            continue
        if d == today:
            close_t = EARLY_CLOSE if d in set(early_closes) else REGULAR_CLOSE
            if now.time() < close_t:
                continue           # 오늘 장은 아직 안 끝났다
        return d
    return None


def index_state(bar_dates, now, calendar, early_closes=(), min_bars=MIN_BARS):
    """(state, detail) 반환. bar_dates 는 받은 캔들의 date 리스트(오름차순)."""
    bars = sorted(bar_dates or [])
    lcs = last_completed_session(now, calendar, early_closes)
    if not bars:
        return "missing", {"reason": "캔들 없음", "last_completed_session": lcs}
    if len(bars) < min_bars:
        return "missing", {"reason": f"봉 {len(bars)} < 필요 {min_bars}",
                           "last_bar": bars[-1], "last_completed_session": lcs}
    if lcs is None:
        return "missing", {"reason": "캘린더에 완료된 세션이 없다", "last_bar": bars[-1]}

    # 창 안의 중간 결측 (캘린더에는 있는데 봉이 없는 거래일)
    window = [d for d in sorted(calendar) if bars[0] <= d <= min(bars[-1], lcs)]
    gaps = sorted(set(window) - set(bars))

    detail = {"last_bar": bars[-1], "last_completed_session": lcs,
              "gap_days": len(gaps), "gaps": gaps[:5], "bars": len(bars)}
    if bars[-1] > lcs:
        detail["reason"] = "마지막 봉이 아직 진행 중인 세션의 것 (미완성 봉)"
        return "incomplete", detail
    if gaps:
        detail["reason"] = f"창 안에 결측 거래일 {len(gaps)}일"
        return "incomplete", detail
    if bars[-1] < lcs:
        sessions_behind = sum(1 for d in calendar if bars[-1] < d <= lcs)
        detail["sessions_behind"] = sessions_behind
        detail["reason"] = f"마지막 완료 세션보다 {sessions_behind}세션 뒤짐"
        return "stale", detail
    detail["reason"] = "최신"
    return "fresh", detail


def response_policy(state, tolerance_sessions=1):
    """상태별 대응. **정책 변경은 승인 후에만 운영에 적용한다.**"""
    if state == "fresh":
        return {"regime_usable": True, "new_entries": "허용",
                "holdings": "정상 관리", "alert": None}
    if state == "incomplete":
        return {"regime_usable": False, "new_entries": "보류(다음 완료 세션까지)",
                "holdings": "정상 관리", "alert": "미완성 봉 — 판정 보류"}
    if state == "stale":
        return {"regime_usable": False, "new_entries": "차단",
                "holdings": "대사·손절·만기·노출 축소 계속",
                "alert": f"지수 캔들 지연 (허용 {tolerance_sessions}세션 초과) — 재조회"}
    return {"regime_usable": False, "new_entries": "차단",
            "holdings": "대사·손절·만기·노출 축소 계속",
            "alert": "지수 캔들 없음 — 국면 판정 불가. 폴백으로 조용히 갈아타지 않는다"}


# ── 테스트 ────────────────────────────────────────────────────────────────
def _cal(start, n, holidays=()):
    """평일에서 휴장일을 뺀 거래일 리스트."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5 and d not in set(holidays):
            out.append(d)
        d += datetime.timedelta(days=1)
    return out


def _dt(y, m, d, hh, mm=0):
    return datetime.datetime(y, m, d, hh, mm, tzinfo=NY)


PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok   {name}")
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL {name}: {e}")


def t_fresh_after_close():
    cal = _cal(datetime.date(2026, 1, 2), 80)
    now = _dt(2026, 4, 22, 17)                       # 마감 후
    lcs = last_completed_session(now, cal)
    assert lcs == datetime.date(2026, 4, 22), lcs
    st, d = index_state(cal[:cal.index(lcs) + 1], now, cal)
    assert st == "fresh", (st, d)


def t_incomplete_during_session():
    """장중에 오늘 봉이 들어오면 미완성 봉 — fresh 가 아니다."""
    cal = _cal(datetime.date(2026, 1, 2), 80)
    now = _dt(2026, 4, 22, 11)                       # 장중
    lcs = last_completed_session(now, cal)
    assert lcs == datetime.date(2026, 4, 21), lcs    # 어제가 마지막 완료 세션
    bars = cal[:cal.index(datetime.date(2026, 4, 22)) + 1]
    st, d = index_state(bars, now, cal)
    assert st == "incomplete" and "미완성" in d["reason"], (st, d)


def t_weekend_is_not_stale():
    """토요일에는 금요일 봉이 최신이다 — 주말을 지연으로 오해하면 안 된다."""
    cal = _cal(datetime.date(2026, 1, 2), 80)
    fri = datetime.date(2026, 4, 17)
    now = _dt(2026, 4, 18, 10)                       # 토요일
    assert last_completed_session(now, cal) == fri
    st, _ = index_state(cal[:cal.index(fri) + 1], now, cal)
    assert st == "fresh"


def t_holiday_is_not_stale():
    """휴장일(예: 7/3 대체공휴일) 다음 날 아침에도 직전 거래일 봉이 최신이다."""
    hol = datetime.date(2026, 7, 3)
    cal = _cal(datetime.date(2026, 1, 2), 140, holidays=[hol])
    prev = datetime.date(2026, 7, 2)
    now = _dt(2026, 7, 4, 9)                         # 휴장 다음 날(토) 아침
    assert last_completed_session(now, cal) == prev
    st, _ = index_state(cal[:cal.index(prev) + 1], now, cal)
    assert st == "fresh", st


def t_early_close_completes_at_1300():
    """조기폐장일은 13:00 ET 에 마감된다 — 13:30 이면 오늘이 완료 세션이다."""
    cal = _cal(datetime.date(2026, 1, 2), 250)
    day = datetime.date(2026, 11, 27)                # 추수감사절 다음날 조기폐장 가정
    if day not in cal:
        day = next(d for d in cal if d >= day)
    early = [day]
    assert last_completed_session(_dt(day.year, day.month, day.day, 13, 30),
                                  cal, early) == day
    prev = cal[cal.index(day) - 1]
    assert last_completed_session(_dt(day.year, day.month, day.day, 12, 30),
                                  cal, early) == prev
    # 조기폐장을 모르면 16:00 까지 완료로 안 쳐서 하루 뒤진 판정을 한다
    assert last_completed_session(_dt(day.year, day.month, day.day, 13, 30),
                                  cal, ()) == prev


def t_dst_boundary():
    """DST 전환일에도 오프셋을 손으로 더하지 않는다 (zoneinfo 가 처리)."""
    cal = _cal(datetime.date(2026, 1, 2), 80)
    day = datetime.date(2026, 3, 9)                  # 서머타임 시작 다음 거래일
    assert day in cal
    after = _dt(2026, 3, 9, 16, 30)
    assert after.utcoffset() == datetime.timedelta(hours=-4)   # EDT
    assert last_completed_session(after, cal) == day
    before = _dt(2026, 3, 6, 16, 30)
    assert before.utcoffset() == datetime.timedelta(hours=-5)  # EST
    assert last_completed_session(before, cal) == datetime.date(2026, 3, 6)


def t_stale_detected():
    cal = _cal(datetime.date(2026, 1, 2), 120)
    now = _dt(2026, 6, 1, 17)
    lcs = last_completed_session(now, cal)
    old = cal[cal.index(lcs) - 5]
    st, d = index_state(cal[:cal.index(old) + 1], now, cal)
    assert st == "stale" and d["sessions_behind"] == 5, (st, d)


def t_missing_and_short():
    cal = _cal(datetime.date(2026, 1, 2), 120)
    now = _dt(2026, 6, 1, 17)
    assert index_state([], now, cal)[0] == "missing"
    assert index_state(cal[:30], now, cal)[0] == "missing"     # 61봉 미만


def t_mid_gap_detected():
    """창 안에 결측 거래일이 있으면 incomplete — '봉 수만 채웠다'로 통과시키지 않는다."""
    cal = _cal(datetime.date(2026, 1, 2), 120)
    now = _dt(2026, 6, 1, 17)
    lcs = last_completed_session(now, cal)
    bars = cal[:cal.index(lcs) + 1]
    hole = bars.pop(40)
    st, d = index_state(bars, now, cal)
    assert st == "incomplete" and d["gap_days"] == 1 and d["gaps"] == [hole], (st, d)


def t_policy_separates_entry_from_holdings():
    """stale/missing 대응은 신규 진입만 막고 보유분 관리는 유지한다."""
    for st in ("stale", "missing"):
        p = response_policy(st)
        assert p["new_entries"] == "차단" and "계속" in p["holdings"], (st, p)
        assert p["regime_usable"] is False
    assert "폴백" in response_policy("missing")["alert"]
    p = response_policy("fresh")
    assert p["regime_usable"] and p["new_entries"] == "허용"


TESTS = [
    ("마감 후 = fresh", t_fresh_after_close),
    ("장중 오늘 봉 = incomplete (미완성 봉)", t_incomplete_during_session),
    ("주말은 stale 이 아니다", t_weekend_is_not_stale),
    ("공휴일은 stale 이 아니다", t_holiday_is_not_stale),
    ("조기폐장 13:00 마감", t_early_close_completes_at_1300),
    ("DST 경계", t_dst_boundary),
    ("지연 감지 (stale)", t_stale_detected),
    ("캔들 없음·길이 부족 (missing)", t_missing_and_short),
    ("중간 결측 (gap) 감지", t_mid_gap_detected),
    ("대응 분리: 신규 진입 차단 / 보유분 유지", t_policy_separates_entry_from_holdings),
]


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"N1 최신성 판정 테스트 {len(TESTS)}건 (연구 구현 — 운영 미적용)")
    for n, f in TESTS:
        check(n, f)
    print(f"\n통과 {len(PASS)} / 실패 {len(FAIL)}")
    for n, e in FAIL:
        print(f"  FAIL {n}: {e}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
