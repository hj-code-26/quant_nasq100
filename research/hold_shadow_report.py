"""만기 연장 섀도 판독 — 실계좌 기록으로 "연장했다면 나았나" 를 본다. 읽기 전용.

    python research/hold_shadow_report.py [--min-events 20]

운영 DB 는 사본만 읽는다(guard). 사후 주가는 yfinance 로 받는다 — 토스 API 는 격리로 막혀 있고,
그래야 이 판독이 운영 자격증명 없이도 돌아간다.

읽는 것: autotrade.log_hold_shadow 가 매 사이클 남긴 hold_shadow 행.
  at_expiry=1  만기 도달 시점의 판정 (연장 vs 청산)
  would_extend 그 시점 선별 상위 N위 안이었는가

판정 규칙 (기록을 보기 전에 정한다):
  · 최소 20건의 만기 이벤트가 쌓이기 전에는 **결론을 내지 않는다.**
  · 비교 지표는 만기 후 20거래일 초과수익(지수 대비). 승률이 아니라 평균과 분포를 본다.
  · 연장 대상이 청산 대상보다 **유의하게** 낫지 않으면 HOLD_EXTEND_TOP 을 켜지 않는다.
"""
import argparse
import pathlib
import shutil
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

OUT.mkdir(parents=True, exist_ok=True)
SNAP = OUT / "hold_shadow_snapshot.db"
shutil.copyfile(ROOT / "trading_decisions.db", SNAP)        # 격리 전에 사본

from research.isolation import guard                        # noqa: E402

guard(allow_network=())                                     # yahoo 는 ALLOW_HOSTS 로 이미 허용

HORIZON = 20                                                # 만기 후 몇 거래일을 볼지
BENCH = "QQQ"


def events():
    c = sqlite3.connect(f"file:{SNAP}?mode=ro", uri=True)
    try:
        df = pd.read_sql("SELECT * FROM hold_shadow ORDER BY id", c)
    except Exception:                                       # noqa: BLE001 — 아직 테이블이 없다
        return pd.DataFrame()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    # 만기 이벤트 = 종목별로 at_expiry 가 처음 1 이 된 시점 (그 뒤 사이클은 같은 사건의 반복)
    exp = df[df["at_expiry"] == 1].sort_values("id")
    return exp.groupby(["symbol", (exp["held_days"] // 40)]).first().reset_index(drop=True)


def forward(ev):
    """만기 시점 이후 HORIZON 거래일 수익률과 지수 대비 초과수익. yfinance 일봉."""
    import yfinance as yf
    syms = sorted(set(ev["symbol"]) | {BENCH})
    px = yf.download(syms, start=(ev["timestamp"].min() - pd.Timedelta(days=10)).date(),
                     progress=False, auto_adjust=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(syms[0])
    px.index = pd.to_datetime(px.index, utc=True)
    out = []
    for _, r in ev.iterrows():
        s = px.get(r["symbol"])
        b = px.get(BENCH)
        if s is None or b is None:
            continue
        after = s.index[s.index > r["timestamp"]]
        if len(after) <= HORIZON:
            out.append({**r, "fwd_pct": np.nan, "excess_pct": np.nan, "미완": True})
            continue
        t0, t1 = after[0], after[HORIZON]
        out.append({**r, "미완": False,
                    "fwd_pct": (s[t1] / s[t0] - 1) * 100,
                    "excess_pct": (s[t1] / s[t0] - b[t1] / b[t0]) * 100})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-events", type=int, default=20)
    a = ap.parse_args()

    ev = events()
    if ev.empty:
        print("만기 섀도 기록이 아직 없다. 봇이 돌면서 hold_shadow 에 쌓인다.\n"
              "보유 종목이 MAX_HOLD_DAYS 에 도달해야 at_expiry=1 행이 생긴다 "
              "(현재 보유분은 2026-10-05 전후부터).")
        return
    print(f"만기 이벤트 {len(ev)}건 (기준 상위 {int(ev['extend_top'].iloc[0])}위)")
    print(ev.groupby("would_extend").size().rename({0: "청산 판정", 1: "연장 판정"}).to_string())

    if len(ev) < a.min_events:
        print(f"\n■ 결론 보류 — 사전에 정한 최소 {a.min_events}건에 미달({len(ev)}건).")
        print("  표본이 작을 때 평균을 보고 켜는 것이 정확히 피하려는 실수다.")
        print(ev[["timestamp", "symbol", "regime", "held_days", "rank",
                  "would_extend", "price"]].to_string(index=False))
        return

    df = forward(ev)
    done = df[~df["미완"]]
    df.to_csv(OUT / "hold_shadow_events.csv", index=False, encoding="utf-8-sig")
    if len(done) < a.min_events:
        print(f"\n■ 결론 보류 — 만기 후 {HORIZON}거래일이 지난 건이 {len(done)}건뿐.")
        return
    g = done.groupby("would_extend")["excess_pct"]
    print(f"\n만기 후 {HORIZON}거래일 초과수익(지수 {BENCH} 대비, %)")
    print(g.agg(["count", "mean", "median", "std"]).round(2).to_string())
    ext = done[done["would_extend"] == 1]["excess_pct"]
    cut = done[done["would_extend"] == 0]["excess_pct"]
    if len(ext) > 1 and len(cut) > 1:
        rng = np.random.default_rng(0)
        d = np.array([rng.choice(ext, len(ext)).mean() - rng.choice(cut, len(cut)).mean()
                      for _ in range(5000)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"\n연장 − 청산 평균 초과수익: {ext.mean() - cut.mean():+.2f}%p "
              f"(부트스트랩 95% CI {lo:+.2f}~{hi:+.2f})")
        print("■ " + ("차이가 유의하지 않다 — HOLD_EXTEND_TOP 을 켜지 않는다." if lo <= 0 <= hi
                     else "CI 가 0 을 제외한다. 다만 표본이 작으니 백테스트 근거와 함께 판단한다."))
    print(f"\n→ {OUT / 'hold_shadow_events.csv'}")


if __name__ == "__main__":
    main()
