"""운영 SOXL 규칙(autotrade.soxl_plan · soxl_blocked) == 백테스트 C2 인가 — 그리고 그 규칙의 성적표.

    python research/soxl_ops_parity.py   # → research/soxl_ops.md

운영 결정 함수로 과거를 하루씩 재생한다 (t 종가 결정 → t+1 종가 체결, 편도 0.12%, 현금은 단기금리).
research/soxl_rules.sim(C2 설정)과 계좌 곡선이 1e-9 안에서 같지 않으면 assert 로 멈춘다.
같다면 아래 성적표는 **운영 코드가 내리는 결정 그대로**의 백테스트다.

운영과 백테스트의 남는 차이 (동등성 검사가 잡지 못하는 것)
  · 시뮬레이터는 결정 비중을 그날 체결 **전** 계좌 가치로 잰다. 운영은 체결 후 실계좌다. 같은 날 체결과 신호가
    겹칠 때 수수료만큼 달라진다 (재생을 실계좌 방식으로 했을 때 합성 32년 누적 0.35%, 실제 2010~ 0). 재생은 시뮬레이터에 맞췄다
  · 체결가: 백테스트는 다음 날 종가, 운영은 정규장 사이클의 시장가
  · 결정 시점: 운영은 사이클마다 현재가로 비중을 다시 본다 (밴드 ±10%p 라 같은 날 반복 매매는 드물다)
  · 소수점·금액 주문은 정규장(마감 1시간 전까지)에만 → 그 밖의 사이클은 대기
  · 해외 레버리지 ETF 기본예탁금·사전교육 요건 미충족 계좌에서는 매수가 거절된다
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import soxl_rules as S       # noqa: E402  (guard() 는 여기서 켜진다)
from research import soxl_rules_v2 as V    # noqa: E402
import autotrade as at                     # noqa: E402

FEE = S.FEE


def replay(px):
    """운영 결정 함수로 재생."""
    c, rf = px["close"].to_numpy(float), px["rf"].to_numpy(float)
    hi = px["close"].rolling(252, min_periods=1).max().to_numpy()
    n = len(c)
    cash, qty, age, trimmed, pending = 1.0, 0.0, 0, False, []
    eq, w, trades = np.ones(n), np.zeros(n), 0
    for i in range(n):
        if i > 0:
            cash *= 1 + rf[i]
        V0 = cash + qty * c[i]                          # 시뮬레이터와 같게: 결정 비중은 체결 전 계좌 가치로
        blocked = at.soxl_blocked(c[i], hi[i])
        todo, pending = pending, []
        for kind, x in todo:
            V_ = cash + qty * c[i]
            if kind == "sell":
                q = qty * min(1.0, x)
                if qty > 0 and x > 0:
                    cash += q * c[i] * (1 - FEE)
                    qty -= q
                    trades += 1
                    if qty < 1e-12:
                        qty = 0.0
            elif not blocked:
                amt = min(x, max(0.0, at.SOXL_MAX_WEIGHT * V_ - qty * c[i]) / (1 + FEE), cash / (1 + FEE))
                if amt > 1e-9:
                    cash -= amt * (1 + FEE)
                    qty += amt / c[i]
                    trades += 1
        weight = qty * c[i] / V0
        for kind, x, _ in at.soxl_plan(weight, age + (1 if qty > 0 else 0), trimmed):
            if kind == "buy_to":
                pending.append(("buy", x * V0 - qty * c[i]))
            else:
                pending.append(("sell", x))
                if x == 0.5 and qty > 0 and age + 1 >= at.SOXL_TRIM_AGE and not trimmed:
                    trimmed = True
        if qty > 0:
            age += 1
        else:
            age, trimmed = 0, False
        eq[i], w[i] = cash + qty * c[i], qty * c[i] / (cash + qty * c[i])
    return pd.Series(eq, px.index), pd.Series(w, px.index), trades


def main():
    act, syn, bench = S.load()
    cfg = V.GRID["C2 리밸런싱 목표 40%"]
    for name, px in (("실제 2010~", act), ("합성 1994~", syn), ("실제 최근 5년", act.loc["2021-09-16":])):
        a, _, _ = S.sim(px, cfg)
        b, _, _ = replay(px)
        gap = float((a - b).abs().max())
        assert gap < 1e-9, f"{name}: 운영 규칙 ≠ 백테스트 (최대 괴리 {gap})"
        print(f"동등성 {name}: 최대 괴리 {gap:.1e}")

    rows = []
    for pname, px in (("실제 2010-03~2026-09", act), ("실제 최근 5년 2021-09~2026-09", act.loc["2021-09-16":]),
                      ("합성 1994-06~2026-09 (닷컴·금융위기 포함)", syn)):
        rf = px["rf"]
        e, w, tr = replay(px)
        yrs = len(px) / 252
        rows.append({"구간": pname, "구성": "운영 SOXL 규칙", **S.stat(e, rf), "총수익%": round((e.iloc[-1] - 1) * 100),
                     "평균 비중%": round(w.mean() * 100, 1), "연 매매": round(tr / yrs, 1)})
        refs = {"SOXL 매수보유": px["close"]}
        if pname.startswith("실제"):
            refs.update({"SOXX 매수보유": bench["SOXX"], "QQQ 매수보유": bench["QQQ"]})
        else:
            refs["^SOX 1배(가격)"] = bench["SOX"]
        for k, s in refs.items():
            s = s.reindex(px.index).ffill()
            e2 = s / s.iloc[0]
            rows.append({"구간": pname, "구성": k, **S.stat(e2, rf), "총수익%": round((e2.iloc[-1] - 1) * 100)})
    tb = pd.DataFrame(rows)
    e, w, _ = replay(act)
    yr = (e.groupby(e.index.year).last() / e.groupby(e.index.year).last().shift(1) - 1) * 100
    sx = bench["SOXX"].reindex(act.index).ffill()
    yr_sx = (sx.groupby(sx.index.year).last() / sx.groupby(sx.index.year).last().shift(1) - 1) * 100
    yt = pd.DataFrame({"운영 규칙 %": yr.round(1), "SOXX %": yr_sx.round(1)})
    dd = e / e.cummax() - 1
    worst = []
    for _ in range(3):
        t = dd.idxmin()
        pk = e.loc[:t].idxmax()
        rec = e.loc[t:][e.loc[t:] >= e[pk]]
        end = rec.index[0] if len(rec) else e.index[-1]
        worst.append({"고점": pk.date(), "저점": t.date(), "낙폭%": round(dd.min() * 100, 1),
                      "회복": rec.index[0].date() if len(rec) else "미회복"})
        dd = dd.copy()
        dd.loc[pk:end] = 0
    md = ["# 운영 SOXL 규칙 — 백테스트 성적표",
          "운영 결정 함수(`autotrade.soxl_plan`·`soxl_blocked`)로 재생한 결과. 백테스트 C2 와 계좌 곡선 **동일**(괴리 < 1e-9) 확인.\n\n"
          "규칙: SOXL 목표 40% (30% 미만 매수·50% 초과 매도) · 매수 후 비중 ≤ 70% · SOXL 252일 고점 대비 −30% 이하면 매수 중단 · "
          "포지션 63거래일에 50% 축소(1회) · 전략 밖 보유는 정리. 편도 0.12%, 다음 날 종가 체결, 현금은 단기금리.",
          "## 성과\n\n" + tb.to_string(index=False),
          "## 연도별 (실제)\n\n" + yt.to_string(),
          "## 최대 낙폭 상위 3 (실제 2010~)\n\n" + pd.DataFrame(worst).to_string(index=False),
          "## 사전등록 판정 (soxl_rules_v2.md)\n\nREJECT — 합성 1994~ MDD −66.4% (기준 −60%), 실제 Sharpe 0.73 < SOXX 0.82. "
          "사용자 결정으로 메인 규칙 채택(2026-09-17). 누적 시도 N=99.",
          "## 운영과 백테스트의 남는 차이\n\n체결가(다음 날 종가 vs 정규장 시장가) · 사이클마다 현재가 재판정 · "
          "결정 비중을 시뮬레이터는 체결 전 계좌로, 운영은 체결 후 실계좌로 잰다(합성 32년 누적 0.35%) · "
          "정규장 밖 대기 · 해외 레버리지 ETF 예탁금·교육 요건 미충족 시 매수 거절"]
    text = "\n\n".join(md)
    (ROOT / "research" / "soxl_ops.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
