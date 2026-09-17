"""A — 시가총액 가중 + 종목별 추세 신호. 동일가중 구조가 QQQ 에 −4.3%p 지는 문제를 직접 겨냥한다.

    python research/mcap_momentum.py   # → research/out/mcap_momentum.csv

전제 (research/pit_mcap.py)
  시총 = SEC 공시 발행주식수(공시일부터) × 분할·배당조정 종가. ADR 7종목·티커재사용 1종목 제외, 커버리지 91.9%.
  **검증 게이트 실패**: 매일 리밸런스 재현 vs ^NDX 상관 0.995 · 추적오차 2.39% 는 통과,
  CAGR 차 +3.17%p (기준 ±2%p) 불통과. 원인: ① 게이트 설계 오류 — 종가는 배당조정인데 ^NDX 는 가격지수
  (배당 포함 QQQ 대비 +2.27%p) ② NDX 는 수정 시총가중(상한)이라 순수 시총가중이 대형주를 더 담는다.
  사용자 결정(2026-09-17): 같은 데이터의 **재현 지수(A0)를 기준선**으로 두고 진행한다.
  → 데이터 편향은 A0 와 전략에 같이 들어가 상쇄된다. QQQ 대비 수치는 약 +2%p 유리하게 치우쳐 있다.

사전 등록 (2026-09-17, 결과 보기 전 커밋)
  유니버스 그날의 PIT 구성종목 중 시총 산출 가능 종목
  공통    20거래일마다 재판정, t−1 종가 정보(시총·신호)로 판정 → t 종가 체결, 사이엔 표류. 편도 10bp
          신호 = 종목 자체 12-1개월 수익률 > 0 (ts_momentum P2 와 같음, 지수 동기화 없음)
  기준선  A0 전 종목 시총가중, 신호 없음 (재현 지수의 20일 리밸런스판) — 시도 아님
  시도    A1 시총 상위10 시총가중, 신호 없음
          A2 상위10 + 신호, 꺼진 몫은 현금
          A3 상위10 + 신호, 꺼진 몫은 켜진 종목에 비례 재배분 (전부 꺼지면 현금)
          A4 상위30 + 신호, 꺼진 몫은 현금
          A5 전 종목 + 신호, 꺼진 몫은 현금
          A6 상위10 + 종가>200일선 신호, 재배분
          → 6 시도. 누적 N = 64 + 6 = 70
  판정    ① A0 대비 ΔSharpe 95% CI 하한 > 0   ② QQQ 대비 ΔSharpe CI 하한 > 0
          ③ 두 반기(2015~20 / 2021~26) Sharpe 가 A0·QQQ 둘 다보다 높다   ④ 25bp Sharpe > A0(25bp)
          ⑤ DSR ≥ 0.95 (A0 대비 초과수익 계열, N=70)
          ①~⑤ 전부 = ADOPT · ③④ = CANDIDATE · 그 외 REJECT. ADOPT 가 아니면 운영 반영 안 함
  불변식  (a) A0 매일 재판정·비용 0 = 전일 시총가중 일수익 직접 계산과 같다 (assert)
          (b) 6 시도 prefix(끝 400일 삭제 후 시총·지표 재계산) 괴리 < 1e-12 (assert)
  서술    (판정 미사용) CAGR·MDD·평균 주식비중·회전율, 재판정 위상 0/5/10/15 Sharpe
  표본 외 없음 — 1999~2014 는 PIT 구성종목·XBRL 이 없다. 이 결과는 2015~2026 한 구간뿐이다
  실계좌   상위10 이면 $570 에서 종목당 약 $57 → 최소주문 $5 문제 없음 (P 계열과 다른 점)
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import exit_study as ES      # noqa: E402  (guard() 가 여기서 켜진다)
from research import corr_limits as C      # noqa: E402
from research import pit_mcap as M         # noqa: E402
from research import ts_momentum as T      # noqa: E402
from backtest_bear_exposure import dsr     # noqa: E402

N_TRIALS = 70
BASE = "A0 전 종목 시총가중 (재현 지수)"
GRID = {
    BASE: dict(always=True),
    "A1 상위10 시총가중": dict(top=10, always=True),
    "A2 상위10 + 12-1>0 · 꺼지면 현금": dict(top=10, sig="mom12_1"),
    "A3 상위10 + 12-1>0 · 재배분": dict(top=10, sig="mom12_1", redistribute=True),
    "A4 상위30 + 12-1>0 · 꺼지면 현금": dict(top=30, sig="mom12_1"),
    "A5 전 종목 + 12-1>0 · 꺼지면 현금": dict(sig="mom12_1"),
    "A6 상위10 + 종가>200일선 · 재배분": dict(top=10, sig="dist200", redistribute=True),
}


def panel(close=None):
    raw_close, _, ndx, qqq = ES.raw()
    close = raw_close if close is None else close
    P = ES.build(close, ndx, qqq, ES.PIT_START)
    cap = M.mcap_panel(close)[0].reindex(P["dates"])[P["cols"]].to_numpy(float)
    return P, cap


def run(P, cap, cfg, **kw):
    return T.ts_sim(P, {**cfg, "cap": cap, **kw})


def check_equiv(P, cap):
    ret, valid = P["ret"], P["valid"]
    exp = [0.0]
    for t in range(2, len(ret)):
        ok = valid[t - 2] & np.isfinite(ret[t - 1]) & np.isfinite(cap[t - 2])
        w = cap[t - 2][ok] / cap[t - 2][ok].sum()
        exp.append(float((w * np.nan_to_num(ret[t][ok])).sum()))
    exp = pd.Series(exp, index=P["dates"][1:])
    r = run(P, cap, GRID[BASE], every=1, cost_bp=0)[0]
    gap = float((r - exp.loc[r.index]).abs().max())
    assert gap < 1e-10, f"시총가중 동치 실패 (괴리 {gap})"
    return gap


def check_prefix(k_back=400):
    close = ES.raw()[0]
    Pf, cf = panel(close)
    Pp, cp = panel(close.iloc[:-k_back])
    worst = 0.0
    for name, cfg in GRID.items():
        full, part = run(Pf, cf, cfg)[0], run(Pp, cp, cfg)[0]
        gap = float((full.loc[part.index] - part).abs().max())
        assert gap < 1e-12, f"{name}: prefix 불변식 실패 (괴리 {gap})"
        worst = max(worst, gap)
    return worst


def main():
    P, cap = panel()
    print(f"불변식 (a) 시총가중 동치 괴리 {check_equiv(P, cap):.1e} · (b) prefix {check_prefix():.1e}")

    q = ES.benchmarks(P)["지수 매수보유"]
    res = {n: {cb: run(P, cap, c, cost_bp=cb) for cb in (ES.COST, ES.STRESS)} for n, c in GRID.items()}
    b10, b25 = res[BASE][ES.COST][0], res[BASE][ES.STRESS][0]
    H2 = pd.Timestamp(ES.H1_END) + pd.Timedelta(days=1)
    ex = {n: (res[n][ES.COST][0] - b10).dropna() for n in GRID if n != BASE}
    var_sr = float(np.var([x.mean() / x.std(ddof=1) for x in ex.values() if x.std() > 0], ddof=1))

    def halves(r):
        return C.stat(r[:ES.H1_END])["sharpe"], C.stat(r[H2:])["sharpe"]

    qa, qb = halves(q)
    ba, bb = halves(b10)
    rows = []
    for n in list(GRID) + ["QQQ 매수보유"]:
        r, tv, _, w = (q, 0.0, None, 1.0) if n == "QQQ 매수보유" else res[n][ES.COST]
        s = C.stat(r)
        h1, h2 = halves(r)
        row = {"설정": n, "CAGR%": round(s["cagr"], 2), "Sharpe": round(s["sharpe"], 3), "MDD%": round(s["mdd"], 1),
               "주식비중%": round(w * 100), "연회전율": round(tv, 2), "Sh 15~20": round(h1, 3), "Sh 21~26": round(h2, 3)}
        if n not in (BASE, "QQQ 매수보유"):
            x, y = b10.align(r, join="inner")
            bs0 = C.paired_block_boot(x.values, y.values)
            x, y = q.align(r, join="inner")
            bsq = C.paired_block_boot(x.values, y.values)
            s25 = C.stat(res[n][ES.STRESS][0])["sharpe"]
            p, _ = dsr(ex[n], N_TRIALS, var_sr)
            c1, c2 = bs0["d_sharpe_lo"] > 0, bsq["d_sharpe_lo"] > 0
            c3 = h1 > max(ba, qa) and h2 > max(bb, qb)
            c4 = s25 > C.stat(b25)["sharpe"]
            c5 = p >= 0.95
            row.update({"Sh 25bp": round(s25, 3),
                        "A0 대비 ΔSh (CI)": f"{bs0['d_sharpe']:+.3f} ({bs0['d_sharpe_lo']:+.3f}~{bs0['d_sharpe_hi']:+.3f})",
                        "QQQ 대비 ΔSh (CI)": f"{bsq['d_sharpe']:+.3f} ({bsq['d_sharpe_lo']:+.3f}~{bsq['d_sharpe_hi']:+.3f})",
                        "DSR": round(p, 3),
                        "위상 0/5/10/15": " / ".join(f"{C.stat(run(P, cap, GRID[n], phase=ph)[0])['sharpe']:.3f}"
                                                  for ph in (0, 5, 10, 15)),
                        "판정": "ADOPT" if all((c1, c2, c3, c4, c5)) else "CANDIDATE" if c3 and c4 else "REJECT"})
        rows.append(row)
    tb = pd.DataFrame(rows)
    tb.to_csv(ES.OUT / "mcap_momentum.csv", index=False, encoding="utf-8-sig")
    print(f"\n### 사전등록 판정 — PIT 2015~2026, 편도 {ES.COST:.0f}bp, 누적 N={N_TRIALS}\n")
    print(tb.to_string(index=False))


if __name__ == "__main__":
    main()
