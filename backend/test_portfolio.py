"""스캔 → 계획 → 체결 경로 자체 점검.  python -m backend.test_portfolio"""
import pathlib
import tempfile

from . import nasdaq100, trader

# win_rate_pct = 표본 외 백테스트 승률(선별 1차 기준), bt_trades = 그 표본의 매매 횟수
ROWS = [
    {"ticker": "AAA", "direction": "상승", "significant": True,  "p_up": 0.58,
     "accuracy": 0.60, "baseline": 0.52, "win_rate_pct": 55.0, "bt_trades": 6,
     "rolling_acc": 0.62},
    {"ticker": "BBB", "direction": "상승", "significant": False, "p_up": 0.62,
     "accuracy": 0.55, "baseline": 0.52, "win_rate_pct": 61.0, "bt_trades": 5,
     "rolling_acc": 0.66},
    {"ticker": "CCC", "direction": "하락", "significant": True,  "p_up": 0.30,
     "accuracy": 0.61, "baseline": 0.52, "win_rate_pct": 70.0, "bt_trades": 8,
     "rolling_acc": 0.70},
    {"ticker": "DDD", "direction": "중립", "significant": True,  "p_up": 0.51,
     "accuracy": 0.58, "baseline": 0.52, "win_rate_pct": 70.0, "bt_trades": 8,
     "rolling_acc": 0.70},
    # 기준선(다수 클래스)보다 못한 모델 — 상승이어도 후보에서 빠져야 한다
    {"ticker": "EEE", "direction": "상승", "significant": False, "p_up": 0.70,
     "accuracy": 0.40, "baseline": 0.55, "win_rate_pct": 90.0, "bt_trades": 9,
     "rolling_acc": 0.70},
    # 매매 2회짜리 100% 승률 — 표본이 없어 우연과 구분 불가, 역시 제외
    {"ticker": "FFF", "direction": "상승", "significant": True,  "p_up": 0.60,
     "accuracy": 0.60, "baseline": 0.52, "win_rate_pct": 100.0, "bt_trades": 2,
     "rolling_acc": 0.70},
    # 롤링 보정 정확도 59% — 60% 컷에 걸려 후보에서 빠져야 한다
    {"ticker": "GGG", "direction": "상승", "significant": True,  "p_up": 0.65,
     "accuracy": 0.62, "baseline": 0.52, "win_rate_pct": 88.0, "bt_trades": 7,
     "rolling_acc": 0.59},
]


class FakeToss:
    orders = []

    def buying_power(self, cur):
        return {"cashBuyingPower": "10000"}

    def buy(self, tk, quantity, price):
        FakeToss.orders.append((tk, quantity))
        return {"ok": True}


def _leg(win_rate, p_value, action="BUY"):
    return {"label": "주간", "action": action, "tradable": True,
            "horizon_days": 5, "p_up": 0.6, "budget_frac": 0.10,
            "win_rate_pct": win_rate, "edge": {"p_value": p_value}}


def _plan(ticker, win_rate=60.0, p_value=0.001, tradable=True):
    return {"ticker": ticker, "date": "2026-01-02", "last_close": 100.0,
            "tradable": tradable, "win_rate_pct": win_rate,
            "tracks": {"week": _leg(win_rate, p_value)}}


def main():
    nasdaq100._STATE["results"] = ROWS

    # 1) 후보 선별: 상승 + 기준선 초과 + 매매 표본 확보, 승률 내림차순
    assert [c["ticker"] for c in trader.scan_candidates(5)] == ["AAA"]
    assert "GGG" not in [c["ticker"] for c in trader.scan_candidates(9, strict=False)],         "롤링 정확도 59% 는 60% 컷에 걸려야 한다"
    assert [c["ticker"] for c in trader.scan_candidates(5, strict=False)] \
        == ["BBB", "AAA"], "승률 높은 BBB 가 먼저여야 한다"
    assert len(trader.scan_candidates(1, strict=False)) == 1

    # 2) 홀드아웃 분할: 뒤쪽 walk 는 임계값 탐색에서 빠진다
    per_day = [{"walk": w, "prob": 0.6, "actual": 1, "date": f"d{w}"}
               for w in range(1, 7)]
    cal, hold = trader.split_by_walk(per_day)
    assert [d["walk"] for d in cal] == [1, 2, 3, 4]
    assert [d["walk"] for d in hold] == [5, 6]
    assert not (set(d["walk"] for d in cal) & set(d["walk"] for d in hold))
    # walk 가 너무 적으면 분할하지 않는다 (홀드아웃 없음 → 자격도 없음)
    assert trader.split_by_walk(per_day[:2])[1] == []

    # 3) 순위·다중비교 보정: 5종목을 훑었으면 종목당 기준은 0.05/5 = 0.01
    plans = [_plan("LOW", win_rate=52.0), _plan("HIGH", win_rate=71.0),
             _plan("WEAK", win_rate=80.0, p_value=0.03)]
    ranked = trader.rank_plans(plans + [_plan("X1"), _plan("X2")])
    assert ranked[0]["ticker"] == "HIGH", [p["ticker"] for p in ranked]
    weak = next(p for p in ranked if p["ticker"] == "WEAK")
    assert not weak["tradable"], "p=0.03 은 보정 기준(0.01)을 넘지 못한다"
    assert ranked[-1]["ticker"] == "WEAK", "자격 없는 종목은 뒤로"

    # 4) 예산 분할: 종목 수로 나뉘고, 자격 없는 계획은 아예 빠진다
    trader.TossClient = FakeToss
    trader.JOURNAL = pathlib.Path(tempfile.mkdtemp()) / "j.jsonl"
    trader._PORT["status"] = "done"
    trader._PORT["plans"] = [_plan("AAA"), _plan("BBB"),
                             _plan("CCC", tradable=False)]

    out = trader.execute_portfolio(live=False)
    assert [e["ticker"] for e in out] == ["AAA", "BBB"], out
    # 10000 * 0.10 예산 / 2종목 / 100달러 = 5주
    assert all(e["quantity"] == 5 for e in out), out
    assert not FakeToss.orders, "dry-run 인데 주문이 나갔다"

    # 5) live 는 실제로 주문을 낸다
    FakeToss.orders = []
    trader.execute_portfolio(live=True)
    assert FakeToss.orders == [("AAA", "5"), ("BBB", "5")], FakeToss.orders

    # 6) 자격 있는 계획이 없으면 아무것도 하지 않는다
    trader._PORT["plans"] = [_plan("CCC", tradable=False)]
    assert trader.execute_portfolio(live=True) == []

    # 7) 기준일별 정리: 60% 미만 제외, 같은 날 같은 종목은 마지막 산출만
    trader.PLAN_LOG = pathlib.Path(tempfile.mkdtemp()) / "plans.jsonl"
    nasdaq100._STATE["results"] = ROWS
    trader.log_plan(_plan("AAA", win_rate=55.0) | {"date": "2026-01-02"})
    trader.log_plan(_plan("AAA", win_rate=64.0) | {"date": "2026-01-02"})
    trader.log_plan(_plan("BBB", win_rate=71.0) | {"date": "2026-01-02"})
    trader.log_plan(_plan("GGG", win_rate=99.0) | {"date": "2026-01-02"})
    trader.log_plan(_plan("AAA", win_rate=50.0) | {"date": "2026-01-05"})
    by = trader.plans_by_date()
    assert [g["date"] for g in by["dates"]] == ["2026-01-05", "2026-01-02"], by
    day = by["dates"][1]["plans"]
    assert [r["ticker"] for r in day] == ["BBB", "AAA"], "승률 내림차순"
    assert day[1]["win_rate_pct"] == 64.0, "같은 날 재산출은 마지막 값"
    assert by["excluded"] == 1, "GGG(59%) 는 제외 카운트"

    print("모든 점검 통과")


if __name__ == "__main__":
    main()
