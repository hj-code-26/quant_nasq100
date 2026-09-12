"""당시 나스닥100 구성종목 (point-in-time) — 백테스트 유니버스 편향 보정.

왜 있나. 백테스트가 **오늘의** 구성종목을 과거에 적용하면 두 가지가 섞인다.
  · 전방 탐색: 급등한 **뒤에** 편입된 종목(PLTR·APP·MSTR…)을 편입 전부터 살 수 있었다
  · 생존 편향: 지수에서 빠져 사라진 회사가 표본에 아예 없다
2015~2026 검증 구간에서 이 둘을 걷어내니 CAGR 26.9%→12.5%, Sharpe 0.985→0.569 로
QQQ 매수보유(18.9%, 0.901)에 진다. → `research/pit_conclusion.md`

한계 (읽는 사람이 반드시 알아야 하는 것)
  · 커버리지는 **2015-01-01 이후**뿐이다. 그 전 구간은 마스크가 전부 True = 옛 방식 그대로다
  · 사라진 회사의 **가격**은 별개 문제다. 가격이 없는 종목은 그냥 빠진다 → 편향 '축소'지
    '제거'가 아니다. 결손 목록은 research/out/pit_missing.csv
  · 실거래(autotrade.py)에는 영향이 없다 — 실전은 애초에 오늘 지수에 있는 종목만 산다.
    여기서 고치는 것은 **평가**이지 실행이 아니다.

출처: jmccarrell/n100tickers (MIT). 오프라인 사본 research/data/pit/n100-YYYY.yaml
끄기: 환경변수 PIT=0 (옛 방식 재현·A/B 비교용)
"""
import datetime
import os
import pathlib
import sys

import numpy as np

DATA = pathlib.Path(__file__).with_name("research") / "data" / "pit"
COVERAGE_START = datetime.date(2015, 1, 1)
ENABLED = os.environ.get("PIT", "1") != "0"
# 티커 변경 — 옛 티커의 지수 소속을 새 티커로 잇는다 (새 티커 이력이 옛 구간까지 이어질 때만)
RENAME = {"FB": "META", "NLOK": "GEN", "CTRP": "TCOM", "WLTW": "WTW",
          "DISCA": "WBD", "DISCK": "WBD"}
_events = None


def events():
    """{날짜: set(티커)} — 구성종목이 바뀐 시점만. YAML 은 BaseLoader 로 읽는다('ON'이 bool 이 되면 안 된다)."""
    global _events
    if _events is None:
        import yaml
        ev = {}
        for f in sorted(DATA.glob("n100-*.yaml")):
            d = yaml.load(f.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
            cur = set(d["tickers_on_Jan_1"])
            ev[datetime.date(int(d["year"]), 1, 1)] = set(cur)
            for ds, ch in sorted((d.get("changes") or {}).items()):
                cur = (cur | set(ch.get("union", []))) - set(ch.get("difference", []))
                ev[datetime.date.fromisoformat(str(ds))] = set(cur)
        _events = dict(sorted(ev.items()))
    return _events


def mask(dates, symbols, quiet=False):
    """(n_days, n_sym) bool. True = 그날 매수 후보로 삼아도 되는 종목.

    커버리지 시작 전(2015-01-01 이전)과 PIT=0 일 때는 전부 True 를 돌려준다 — 즉
    **아무것도 바꾸지 않는다.** 구성종목이 아닌 날은 False 이며, 이미 보유 중인 종목의
    청산까지 막지는 않는다 (호출부가 선택 지표에만 씌운다).
    """
    m = np.ones((len(dates), len(symbols)), bool)
    if not ENABLED:
        return m
    ev = events()
    ix = {s: i for i, s in enumerate(symbols)}
    keys = list(ev)
    k = 0
    cur = None
    covered = 0
    for t, d in enumerate(dates):
        dd = d.date() if hasattr(d, "date") else d
        while k < len(keys) and keys[k] <= dd:
            cur = np.zeros(len(symbols), bool)
            for s in ev[keys[k]]:
                s = RENAME.get(s, s)
                if s in ix:
                    cur[ix[s]] = True
            k += 1
        if cur is not None and dd >= COVERAGE_START:
            m[t] = cur
            covered += 1
    if not quiet:
        print(f"[pit] 당시 구성종목 적용: {covered}/{len(dates)}일 "
              f"(커버리지 {COVERAGE_START} 이후). 하루 평균 후보 {m.sum(1).mean():.0f}종목. "
              f"끄려면 PIT=0", file=sys.stderr)
    return m
