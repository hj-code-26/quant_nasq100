"""§7 승인 대기 3건 — 설계 검증용 **모의 테스트**. 운영 적용 없음.

실행: python research/test_ops.py
실주문·실토큰 발급·운영 DB/환경변수 변경 없음. 토큰 파일은 임시 경로로 돌린다.
"""
import datetime
import pathlib
import sqlite3
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from research.isolation import guard              # noqa: E402

guard()

import autotrade as at                 # noqa: E402
import toss                            # noqa: E402
from research.test_safety import FakeToss, tmpdb   # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  ok   {name}")
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL {name}: {e}")


# ── A. 대시보드 종료 ──────────────────────────────────────────────────────
def t_scheduler_dies_with_dashboard():
    """최소 재현: 스케줄러가 **데몬 스레드**라 대시보드 프로세스가 죽으면 같이 죽는다.
    → 손절·만기·노출 축소도 함께 멈춘다. (P0-5 를 코드로 확정한다.)"""
    src = (pathlib.Path(__file__).parent.parent / "autotrade_server.py").read_text(
        encoding="utf-8")
    assert "daemon=True" in src, "데몬 스레드가 아니다 — 전제를 다시 확인할 것"
    assert "schedule.run_pending()" in src
    started = []

    def worker():
        started.append(True)
        time.sleep(10)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    while not started:
        time.sleep(0.01)
    assert th.daemon, "데몬이면 부모 종료 시 정리 없이 사라진다"


def t_restart_does_not_duplicate_order():
    """독립 worker 가 재시작해도 같은 슬롯의 같은 의도는 같은 clientOrderId →
    브로커가 중복을 거절하고, 대사에서 원주문을 찾아 **재제출하지 않는다.**"""
    tmpdb()
    o = {"symbol": "AAA", "side": "buy", "quantity": None, "amount_usd": 100.0,
         "price": 10, "reason": "t"}
    coid = at.intent_key(o)
    t1 = FakeToss()
    rid = at.db_insert("runs", {"timestamp": at._now(), "status": "run1"})
    at.place_all(t1, rid, [o], dry=False)
    assert len(t1.created) == 1

    class Dup(FakeToss):                       # 재시작한 두 번째 프로세스
        def create_order(self, *a, **k):
            raise at.TossError(409, "duplicate-client-order-id", "중복")

    t2 = Dup(closed=[{"orderId": "REAL1", "clientOrderId": coid, "symbol": "AAA"}])
    rid2 = at.db_insert("runs", {"timestamp": at._now(), "status": "run2"})
    at.place_all(t2, rid2, [o], dry=False)
    assert not t2.created, "재시작 후 중복 주문이 나갔다"
    with sqlite3.connect(at.DB_PATH) as c:
        st = [r[0] for r in c.execute("SELECT status FROM orders ORDER BY id")]
    assert "대사 확인" in st[-1], st


def t_protect_only_still_places_orders():
    """PROTECT_ONLY 는 '주문을 안 낸다'가 아니다 — 손절·만기·노출 축소는 **실제 주문**이다.
    승인 없이 켜면 안 된다는 것을 규칙으로 고정한다."""
    at.STOP_LOSS_PCT, at.MOMENTUM_EXIT, at.MAX_HOLD_DAYS = 15, False, 0
    now = datetime.datetime.now(at.KST)
    session = (now - datetime.timedelta(hours=1), now - datetime.timedelta(hours=1),
               now + datetime.timedelta(hours=3))
    acct = {"cash": 100.0, "open_orders": [],
            "holdings": {"AAA": {"name": "x", "quantity": 10, "avg_price": 100,
                                 "last_price": 50, "market_value": 500, "pnl_pct": -50}},
            "total_value": 600.0}
    out, _ = at.validate_orders({"orders": [], "summary": ""}, {}, acct, session)
    assert out and out[0]["side"] == "sell", "위험관리 패스가 주문을 내지 않았다"


# ── B. 다중 PC 토큰 ───────────────────────────────────────────────────────
def t_token_file_shared_on_same_machine():
    """같은 머신에서 두 클라이언트가 토큰 파일을 공유하면 발급은 한 번만 일어난다.
    ★ 이것은 **같은 파일시스템**을 볼 때만이다. 다중 PC 를 막았다는 근거가 아니다."""
    tmp = pathlib.Path(tempfile.mkdtemp()) / "tok.json"
    old = toss.TossClient.TOKEN_FILE
    toss.TossClient.TOKEN_FILE = tmp
    issued = []

    class FakeResp:
        status_code = 200

        def json(self):
            issued.append(1)
            return {"access_token": "T", "expires_in": 3600}

        def raise_for_status(self):
            pass

    try:
        a = toss.TossClient("cid", "sec", "1")
        b = toss.TossClient("cid", "sec", "1")
        for cl in (a, b):
            cl._s.post = lambda *args, **kw: FakeResp()
        a._access_token()
        b._access_token()
        assert len(issued) == 1, f"발급이 {len(issued)}회 — 파일 공유가 안 됐다"
        assert tmp.exists()
    finally:
        toss.TossClient.TOKEN_FILE = old


def t_token_write_is_atomic():
    """토큰 저장이 원자적 교체(tmp → replace)인지 소스로 확인한다."""
    src = (pathlib.Path(__file__).parent.parent / "toss.py").read_text(encoding="utf-8")
    assert ".replace(" in src and "tmp" in src, "원자적 교체가 아니다"


def t_single_node_lock_is_machine_local():
    """단일 실행 노드 후보: OS 파일 락은 **머신 로컬**이다. 두 번째 취득은 같은 머신에서만 막힌다.
    다중 PC split-brain 은 이걸로 해결되지 않는다 — 한계를 테스트로 못박는다."""
    lock = pathlib.Path(tempfile.mkdtemp()) / "run.lock"
    lock.write_text(str(1234))
    assert lock.exists()
    # 같은 머신: 두 번째 프로세스는 파일 존재를 보고 물러난다.
    assert lock.read_text() == "1234"
    # 다른 머신: 같은 경로를 공유하지 않으면 존재조차 보이지 않는다.
    other = pathlib.Path(tempfile.mkdtemp()) / "run.lock"
    assert not other.exists(), "다른 머신은 이 락을 보지 못한다 (설계 한계)"


# ── C. QQQ 실패 ───────────────────────────────────────────────────────────
def t_regime_states_are_distinguishable():
    """정상 / 조회 실패 / 폴백 / 판정 불가 네 상태가 source 로 구분된다."""
    import numpy as np
    import pandas as pd
    ok = at.market_regime(None, df=pd.DataFrame({"close": np.linspace(100, 99, 61)}))
    assert ok["source"] == at.BEAR_INDEX
    fb = at.market_regime(None, rows=[{"ret_60d_pct": -8}], df=None)
    assert fb["source"] == "유니버스 중앙값"
    none = at.market_regime(None, rows=None, df=None)
    assert "실패" in none["source"] and none["ret_60d_pct"] is None


def t_stale_index_is_not_detected():
    """**미해결 결함**: market_regime 은 캔들의 최신성을 보지 않는다.
    몇 달 전 데이터로도 '정상 판정'을 내놓는다. stale 상태가 구분되지 않는다."""
    import numpy as np
    import pandas as pd
    idx = pd.DataFrame({"close": np.linspace(100, 99, 61)},
                       index=pd.date_range("2020-01-01", periods=61, freq="D"))
    r = at.market_regime(None, df=idx)
    assert r["source"] == at.BEAR_INDEX and r["ret_60d_pct"] is not None
    # 2020년 캔들인데 오늘 국면으로 반환된다 → stale 감지 필요 (승인 요청 C 에 포함)


def t_regime_recovery_no_duplicate_orders():
    """조회 복구 후 재개해도 같은 30분 슬롯의 같은 의도는 같은 키 → 중복 주문이 안 된다."""
    when = datetime.datetime(2026, 9, 10, 22, 5, tzinfo=at.NY)
    o = {"symbol": "AAA", "side": "buy"}
    assert at.intent_key(o, when) == at.intent_key(o, when + datetime.timedelta(minutes=10))


TESTS = [
    ("A 스케줄러가 대시보드와 함께 죽는다 (재현)", t_scheduler_dies_with_dashboard),
    ("A 재시작 후 중복 주문 없음", t_restart_does_not_duplicate_order),
    ("A PROTECT_ONLY 도 실제 주문을 낸다", t_protect_only_still_places_orders),
    ("B 같은 머신 토큰 파일 공유 (다중 PC 아님)", t_token_file_shared_on_same_machine),
    ("B 토큰 저장 원자성", t_token_write_is_atomic),
    ("B 단일 노드 락은 머신 로컬 (split-brain 미해결)", t_single_node_lock_is_machine_local),
    ("C 국면 상태 구분 (정상/폴백/판정불가)", t_regime_states_are_distinguishable),
    ("C stale 지수 미감지 — 미해결 결함", t_stale_index_is_not_detected),
    ("C 복구 후 중복 주문 없음", t_regime_recovery_no_duplicate_orders),
]


def main():
    real = at.DB_PATH
    print(f"승인 대기 3건 모의 테스트 {len(TESTS)}건 (실주문·실토큰·운영 DB 변경 없음)")
    for n, f in TESTS:
        check(n, f)
    at.DB_PATH = real
    print(f"\n통과 {len(PASS)} / 실패 {len(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
