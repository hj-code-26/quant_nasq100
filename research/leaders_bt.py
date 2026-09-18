"""주도주 모멘텀 + QQQ 200일선 필터 백테스트 (사용자 지정 규칙, 2026-09-17). 규칙 본문은 research/leaders_rules.py. **REJECT → 폐기 (2026-09-17)**.

    python research/leaders_bt.py          # → research/leaders_bt.md, research/out/leaders_bt*.csv, leaders_bt.png

사전 등록 (결과 보기 전 고정 — 결과를 보고 규칙·기준을 바꾸지 않는다)
  표본   PIT 나스닥100 구성종목 OHLCV 2015-01 ~ 2026-09 (research/data/pit/ohlcv_panel.pkl, 옛 구성종목 포함)
  체결   t−1 종가까지로 결정 → t 시가 체결. 손절은 보유 중 **장중** 판정: 시가 ≤ 손절가면 시가, 저가 ≤ 손절가면 손절가
         슬리피지 편도 0.05% + 수수료 편도 0.10%(토스). 스트레스: 슬리피지 0.25%
  시총   SEC 공시 발행주식수(공시일부터) × 분할조정 종가 (research/pit_mcap). 명목 $50B (물가 미조정)
  현금   수익 0% (QQQ 대비 불리한 가정)
  시도   1. 누적 N = 99 + 1 = 100
  기준선 QQQ 매수보유(배당조정) · QQQ 200일선 타이밍(필터만, 종목 선별 없음 — 선별이 보태는 게 있는가)
  판정   ADOPT = 모두 충족
         ① 짝지은 블록 부트스트랩 ΔSharpe(전략−QQQ) 95% CI 하한 > 0
         ② 두 반기(2015~2020 / 2021~2026) 모두 Sharpe > QQQ
         ③ 스트레스 비용에서도 전체 Sharpe > QQQ
         ④ DSR ≥ 0.95 (QQQ 대비 초과수익 계열, N=100, 시도 간 SR 분산 = 1/T 귀무 근사)
         ②③ 만 = CANDIDATE · 그 외 REJECT
  불변식 (a) prefix: 자료 끝 250일을 잘라 돌려도 겹치는 구간 계좌 곡선이 같다 (미래 참조 없음) — assert
         (b) 시총 기준 무한대(매매 없음) → 계좌 곡선 = 1 — assert
         (c) 현금 ≥ 0 · 보유 ≤ 3 — 시뮬 안에서 assert

★ 한계
  · 유니버스가 "미국 $50B 이상 전체"가 아니라 **나스닥100 구성종목 중 $50B 이상**이다 (S&P500 전용 PIT 자료 없음).
    하루 평균 대상 수는 보고서에 적는다. 상위 10% 는 그 안에서의 순위다
  · 가격이 없는 결손 종목(대부분 피인수)은 빠진다 → 생존 편향 '축소'지 '제거'가 아니다
  · ADR 7종목은 시총 산출 불가로 제외 (research/pit_mcap.EXCLUDE)
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import exit_study as ES      # noqa: E402  (guard() 가 여기서 켜진다)
from research import corr_limits as C      # noqa: E402
from research import pit_mcap as M         # noqa: E402
from backtest_bear_exposure import dsr     # noqa: E402
from research import leaders_rules as LD   # noqa: E402
import pit                                 # noqa: E402

START, H1_END = "2015-01-01", "2020-12-31"
SLIP, COMM, STRESS_SLIP = 0.0005, 0.0010, 0.0025
N_TRIALS = 100
OUT = ES.OUT


def load():
    px = pickle.load(open(ROOT / "research" / "data" / "pit" / "ohlcv_panel.pkl", "rb"))
    qqq = pickle.load(open(ROOT / "research" / "data" / "soxl_close.pkl", "rb"))["QQQ"].dropna()
    return px, qqq


def sim(px, qqq, slip=SLIP, min_cap=None):
    O, H, L, Cl = (px[k] for k in ("Open", "High", "Low", "Close"))
    ind = LD.indicators(O, H, L, Cl)
    mark = Cl.ffill(limit=5)
    last = Cl.ffill()
    member = pd.DataFrame(pit.mask(Cl.index, list(Cl.columns), quiet=True), Cl.index, Cl.columns)
    cap = M.mcap_panel(Cl)[0].where(member)
    if min_cap is not None:
        cap = cap * 0 + min_cap - 1                          # 불변식 (b): 아무도 기준을 못 넘는다
    dates = Cl.index
    k0 = int(dates.searchsorted(pd.Timestamp(START)))
    cash, held = 1.0, {}                                     # held: sym -> [qty, entry_px, entry_date]
    eq, trades, onlog, pool = [], [], [], []

    def sell(s, t, fill, why):
        nonlocal cash
        q, e, d0 = held.pop(s)
        got = q * fill * (1 - slip) * (1 - COMM)
        cash += got
        trades.append({"sym": s, "entry": d0, "exit": dates[t], "entry_px": e, "exit_px": fill,
                       "pnl": got / (q * e / (1 - COMM)) - 1, "why": why})

    for t in range(k0, len(dates)):
        d = t - 1
        on = LD.market_on(qqq.loc[:dates[d]])
        onlog.append(on)
        nav = cash + sum(q * mark.iat[d, Cl.columns.get_loc(s)] for s, (q, _, _) in held.items())
        row = lambda k: ind[k].iloc[d].to_dict()               # noqa: E731
        o_t = O.iloc[t]
        # 1) 어제 종가 결정 청산 → 오늘 시가
        for s, why in LD.exits({s: v[1] for s, v in held.items()}, row("close"), row("sma20"), on).items():
            f = o_t[s] if np.isfinite(o_t[s]) else last.iat[t, Cl.columns.get_loc(s)]
            sell(s, t, f, why)
        for s in [s for s in held if not np.isfinite(mark.iat[t, Cl.columns.get_loc(s)])]:
            sell(s, t, last.iat[t, Cl.columns.get_loc(s)], "자료 소멸 (최종가 청산)")
        # 2) 진입 → 오늘 시가
        capd = cap.iloc[d].dropna()
        pool.append(int((capd >= LD.MIN_CAP).sum()))
        day = {k: row(k) for k in ("rs", "close", "sma20", "breakout", "pullback")}
        for s in LD.entries(day, capd.to_dict(), held, on):
            if not np.isfinite(o_t[s]):
                continue
            amt = min(LD.WEIGHT * nav, cash)
            if amt <= 1e-9:
                break
            fill = o_t[s] * (1 + slip)
            held[s] = [amt * (1 - COMM) / fill, fill, dates[t]]
            cash -= amt
        # 3) 장중 손절
        for s in list(held):
            j = Cl.columns.get_loc(s)
            stop = held[s][1] * (1 + LD.STOP)
            if np.isfinite(O.iat[t, j]) and O.iat[t, j] <= stop:
                sell(s, t, O.iat[t, j], "손절 (시가 갭)")
            elif np.isfinite(L.iat[t, j]) and L.iat[t, j] <= stop:
                sell(s, t, stop, "손절 (장중)")
        assert cash >= -1e-12 and len(held) <= LD.SLOTS
        eq.append(cash + sum(q * mark.iat[t, Cl.columns.get_loc(s)] for s, (q, _, _) in held.items()))
    idx = dates[k0:]
    return (pd.Series(eq, idx), pd.DataFrame(trades), pd.Series(onlog, idx), float(np.mean(pool)))


def benchmarks(qqq, on, slip):
    """on = 전략과 같은 시장 필터 (t−1 종가 판정)."""
    r = qqq.pct_change().reindex(on.index).fillna(0)
    sw = on.astype(int).diff().abs().fillna(0) * (slip + COMM)
    return {"QQQ 매수보유": r, "QQQ 200일선 타이밍": r.where(on, 0.0) - sw}


def trade_stats(tr):
    if tr.empty:
        return {"매매 수": 0}
    w, l = tr.loc[tr.pnl > 0, "pnl"], tr.loc[tr.pnl <= 0, "pnl"]
    return {"매매 수": len(tr), "승률%": 100 * len(w) / len(tr),
            "평균 이익%": 100 * w.mean(), "평균 손실%": 100 * l.mean(),
            "평균 손익비": w.mean() / -l.mean() if len(l) and l.mean() else np.nan,
            "Profit Factor": w.sum() / -l.sum() if l.sum() else np.nan,
            "평균 보유일": float(np.mean([np.busday_count(a.date(), b.date()) for a, b in zip(tr.entry, tr.exit)]))}


def row_stats(r, qr):
    s = C.stat(r)
    x, y = r.align(qr, join="inner")
    beta = np.cov(x, y)[0, 1] / y.var()
    return {"누적%": s["ret"], "CAGR%": s["cagr"], "변동성%": s["vol"], "Sharpe": s["sharpe"], "MDD%": s["mdd"],
            "Alpha CAGR차%p": s["cagr"] - C.stat(qr)["cagr"], "Jensen α%/년": (x.mean() - beta * y.mean()) * 252 * 100,
            "β": beta}


def checks(px, qqq, eq):
    cut = {k: v.iloc[:-250] for k, v in px.items()}
    e2 = sim(cut, qqq.loc[:cut["Close"].index[-1]])[0]
    gap = float((eq.loc[e2.index] - e2).abs().max())
    assert gap < 1e-9, f"prefix 불변식 실패 {gap}"
    flat = sim(px, qqq, min_cap=LD.MIN_CAP)[0]
    assert float((flat - 1).abs().max()) < 1e-12, "매매 없음 불변식 실패"
    return gap


def plot(curves, on, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[2.2, 1],
                                 facecolor="#fcfcfb")
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.fill_between(on.index, 0, 1, where=~on.to_numpy(), color="#f0efec", transform=ax.get_xaxis_transform(),
                        lw=0, label="_nolegend_")
    for (name, r), c in zip(curves.items(), colors):
        e = (1 + r).cumprod()
        a1.plot(e.index, e, color=c, lw=2 if name.startswith("전략") else 1.4, label=name)
        a1.annotate(f"{(e.iloc[-1] - 1) * 100:+.0f}%", (e.index[-1], e.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color="#52514e")
        a2.plot(e.index, (e / e.cummax() - 1) * 100, color=c, lw=1.2)
    a1.set_yscale("log")
    a1.set_ylabel("누적 자산 (시작=1, 로그)", color="#52514e")
    a1.legend(frameon=False, loc="upper left")
    a1.set_title("주도주 모멘텀 + QQQ 200일선 필터 (회색 = 시스템 OFF)", loc="left", color="#0b0b0b")
    a2.set_ylabel("낙폭 %", color="#52514e")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    px, qqq = load()
    eq, tr, on, pool = sim(px, qqq)
    r = eq.pct_change().fillna(eq.iloc[0] - 1)
    gap = checks(px, qqq, eq)
    eq_s, tr_s, _, _ = sim(px, qqq, slip=STRESS_SLIP)
    r_s = eq_s.pct_change().fillna(eq_s.iloc[0] - 1)
    b = benchmarks(qqq, on, SLIP)
    q = b["QQQ 매수보유"]
    curves = {"전략": r, **b}

    tab = pd.DataFrame({n: {**row_stats(x, q), **(trade_stats(tr) if n == "전략" else {})}
                        for n, x in {**curves, "전략 (슬리피지 0.25%)": r_s}.items()}).T
    tab.loc["전략 (슬리피지 0.25%)", list(trade_stats(tr_s))] = list(trade_stats(tr_s).values())
    halves = pd.DataFrame({n: {h: C.stat(x.loc[a:z])["sharpe"] for h, (a, z) in
                               {"2015~2020": (START, H1_END), "2021~2026": ("2021-01-01", None)}.items()}
                           for n, x in curves.items()}).T
    years = pd.DataFrame({n: (1 + x).groupby(x.index.year).prod() - 1 for n, x in curves.items()}) * 100
    bs = C.paired_block_boot(q.values, r.values)
    p_dsr, _ = dsr(r - q, N_TRIALS, 1 / len(r))
    crit = {"① ΔSharpe CI 하한 > 0": bs["d_sharpe_lo"] > 0,
            "② 두 반기 Sharpe > QQQ": bool((halves.loc["전략"] > halves.loc["QQQ 매수보유"]).all()),
            "③ 스트레스 Sharpe > QQQ": C.stat(r_s)["sharpe"] > C.stat(q)["sharpe"],
            "④ DSR ≥ 0.95": p_dsr >= 0.95}
    verdict = "ADOPT" if all(crit.values()) else ("CANDIDATE" if crit["② 두 반기 Sharpe > QQQ"] and crit["③ 스트레스 Sharpe > QQQ"] else "REJECT")
    why = tr.groupby("why").pnl.agg(["count", "mean"]).assign(mean=lambda x: x["mean"] * 100) if len(tr) else None

    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 200
    body = "\n\n".join([
        f"# 주도주 모멘텀 + QQQ 200일선 필터 (사전등록, 누적 N={N_TRIALS})",
        f"기간 {r.index[0].date()} ~ {r.index[-1].date()} ({len(r) / 252:.1f}년) · 슬리피지 {SLIP:.2%} + 수수료 {COMM:.2%} (편도) · "
        f"하루 평균 $50B 이상 대상 {pool:.1f}종목 → 주도주 {np.ceil(pool * LD.RS_TOP):.0f}종목 안팎 · 시스템 ON {on.mean():.0%}",
        "## 성과\n\n```\n" + tab.to_string() + "\n```",
        f"## 사전등록 판정: **{verdict}**\n\n```\n" + "\n".join(f"{k}: {v}" for k, v in crit.items()) +
        f"\nΔSharpe {bs['d_sharpe']:+.3f} (95% CI {bs['d_sharpe_lo']:+.3f} ~ {bs['d_sharpe_hi']:+.3f}) · DSR {p_dsr:.3f}\n```",
        "## 반기 Sharpe\n\n```\n" + halves.to_string() + "\n```",
        "## 청산 사유별\n\n```\n" + (why.to_string() if why is not None else "-") + "\n```",
        "## 연도별 수익률 %\n\n```\n" + years.to_string() + "\n```",
        f"불변식: prefix 괴리 {gap:.1e} · 매매 없음 = 현금 · 현금 ≥ 0 · 보유 ≤ {LD.SLOTS} — 통과",
    ])
    print(body)
    (ROOT / "research" / "leaders_bt.md").write_text(body + "\n\n![](out/leaders_bt.png)\n", encoding="utf-8")
    tab.to_csv(OUT / "leaders_bt.csv", encoding="utf-8-sig")
    tr.to_csv(OUT / "leaders_bt_trades.csv", index=False, encoding="utf-8-sig")
    plot(curves, on, OUT / "leaders_bt.png")


if __name__ == "__main__":
    main()
