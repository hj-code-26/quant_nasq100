"""G-3 — '백테스트로 고르기' 자체를 표본 외에서 재고, QQQ 와 섞어 본다.

    python research/wf_select.py

g_conclusion 의 후보(P1 등)는 2015~2026 PIT 결과를 **보고** 골랐다. DSR 이 그 낙관을 대략 깎지만,
더 직접적인 질문은 "그때그때 과거 자료로만 골랐다면 L0·QQQ 를 이겼나" 다.

① 걷는 선택 — 사전등록 그리드 32개(exit_study.GRID) 안에서만 고른다. P1 은 사후 조합이라 뺀다
   · 고정-편향안: 1999~2014 편향 패널 Sharpe 1위를 2015~2026 내내 쓴다 (2015년 연구자의 선택)
   · 확장창안  : 매년 1월, PIT 2015-01 ~ (전년 말 − 20거래일) Sharpe 1위를 그 해에 쓴다.
                 2015년은 편향 패널로 고른다. 설정이 바뀌면 첫날에 2×편도비용(전량 교체 가정)을 물린다.
   ★ 설정 전환은 연속 곡선을 이어붙인 근사다 (실제로는 보유가 서서히 바뀐다).
② QQQ 혼합 — QQQ 비중 0/50/75%, 월초 리밸런싱, 옮긴 금액에 편도비용.
③ QQQ 회귀 — 일별 α(연율)·β·α 의 t (자기상관 무시 — 참고용).
"""
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from research import exit_study as ES      # noqa: E402  (guard() 는 거기서 켜진다)
from research import corr_limits as C      # noqa: E402

P1 = "P1 126일 모멘텀+연장20 (사후)"
EMBARGO = 20


def walk_select(res, names, cost):
    """반환: (고정-편향 이름, 확장창 수익률, 연도별 선택표)."""
    r = {n: res[("PIT", n, cost)][0] for n in names}
    bias = {n: C.stat(res[("BIAS", n, ES.COST)][0])["sharpe"] for n in names}
    fixed = max(bias, key=bias.get)
    idx = r[ES.BASE].index
    fee = 2 * cost / 1e4
    pieces, rows, prev = [], [], None
    for y in sorted(set(idx.year)):
        past = idx[idx < pd.Timestamp(f"{y}-01-01")]
        if len(past) <= EMBARGO + 60:
            pick, basis = fixed, "편향 1999~2014"
        else:
            end = past[-EMBARGO - 1]
            sh = {n: C.stat(r[n][:end])["sharpe"] for n in names}
            pick, basis = max(sh, key=sh.get), f"PIT 2015-01~{end.date()}"
        seg = r[pick][str(y)].copy()
        if prev is not None and pick != prev:
            seg.iloc[0] -= fee
        pieces.append(seg)
        prev = pick
        rows.append({"연도": y, "선택": pick, "근거": basis,
                     "선택 수익%": round(((1 + seg).prod() - 1) * 100, 1),
                     "L0 수익%": round(((1 + r[ES.BASE][str(y)]).prod() - 1) * 100, 1)})
    return fixed, pd.concat(pieces), pd.DataFrame(rows)


def blend(ra, rq, wq, cost=ES.COST):
    """전략 (1-wq) + QQQ wq, 월초에 목표 비중으로 되돌린다."""
    a, q = ra.align(rq, join="inner")
    fee = cost / 1e4
    va, vq, month, out = 1 - wq, wq, None, []
    for d, x, y in zip(a.index, a.to_numpy(), q.to_numpy()):
        if month is not None and d.month != month:
            v = va + vq
            moved = abs(va - v * (1 - wq))
            va, vq = v * (1 - wq), v * wq
            va -= moved * fee
            vq -= moved * fee
        month = d.month
        va *= 1 + x
        vq *= 1 + y
        out.append(va + vq)
    s = pd.Series(out, index=a.index)
    return s.pct_change().fillna(s.iloc[0] - 1)


def vs_qqq(r, q):
    a, b = r.align(q, join="inner")
    x, y = b.to_numpy(), a.to_numpy()
    beta = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
    alpha = y.mean() - beta * x.mean()
    resid = y - alpha - beta * x
    s = resid.std(ddof=2)
    se = s * np.sqrt(1 / len(x) + x.mean() ** 2 / ((x - x.mean()) ** 2).sum())
    return {"β": round(beta, 3), "α 연율%": round(alpha * 252 * 100, 2), "α t": round(alpha / se, 2),
            "상관": round(np.corrcoef(x, y)[0, 1], 3),
            "추적오차%": round((y - x).std(ddof=1) * np.sqrt(252) * 100, 1)}


def row(name, r, base, q):
    m = C.stat(r)
    out = {"설정": name, "CAGR%": round(m["cagr"], 2), "Sharpe": round(m["sharpe"], 3),
           "MDD%": round(m["mdd"], 1), "Calmar": round(m["calmar"], 2)}
    for tag, ref in (("L0", base), ("QQQ", q)):
        if ref is r:
            continue
        x, y = ref.align(r, join="inner")
        bs = C.paired_block_boot(x.values, y.values)
        out[f"{tag} 대비 ΔSh"] = round(bs["d_sharpe"], 3)
        out[f"{tag} CI"] = f"{bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f}"
    return out


def main():
    names = list(ES.GRID)
    grid = dict(ES.GRID)
    grid[P1] = ES.v(rank="mom126", extend=20)
    with mp.Pool(min(11, mp.cpu_count() - 1), initializer=ES._init) as pool:
        res = ES.run_grid(grid, pool)
    P = ES.panels()["PIT"]
    q = ES.benchmarks(P)["지수 매수보유"]
    base = res[("PIT", ES.BASE, ES.COST)][0]

    # ① 걷는 선택
    rows = []
    for cost in (ES.COST, ES.STRESS):
        fixed, wf, picks = walk_select(res, names, cost)
        b = res[("PIT", ES.BASE, cost)][0]
        if cost == ES.COST:
            picks.to_csv(ES.OUT / "g3_picks.csv", index=False, encoding="utf-8-sig")
            print("### ① 확장창 연도별 선택 (편도 10bp) — 그 해 이전 자료로만 골랐다")
            print(picks.to_string(index=False))
            print(f"  고정-편향안(1999~2014 Sharpe 1위): {fixed}")
        for name, r in ((f"고정-편향 선택 [{fixed.split()[0]}]", res[("PIT", fixed, cost)][0]),
                        ("확장창 선택", wf), ("L0 현행", b), ("QQQ 매수보유", q)):
            rows.append({"비용bp": cost, **row(name, r, b, q)})
    sel = pd.DataFrame(rows)
    sel.to_csv(ES.OUT / "g3_select.csv", index=False, encoding="utf-8-sig")
    print("\n### ① 걷는 선택의 표본 외 성적 (2015~2026)")
    print(sel.to_string(index=False))

    # ② QQQ 혼합 · ③ 회귀
    _, wf10, _ = walk_select(res, names, ES.COST)
    series = {"L0 현행": base, "X14 만기연장": res[("PIT", "X14 만기 때 상위20이면 연장", ES.COST)][0],
              "E2 126일": res[("PIT", "E2 순위=126일 모멘텀", ES.COST)][0],
              P1: res[("PIT", P1, ES.COST)][0], "확장창 선택": wf10}
    brows, arows = [], []
    for name, r in series.items():
        arows.append({"설정": name, **vs_qqq(r, q)})
        for wq in (0.0, 0.5, 0.75):
            br = blend(r, q, wq)
            brows.append({"QQQ비중%": int(wq * 100), **row(name, br, base, q)})
    brows.append({"QQQ비중%": 100, **row("QQQ 매수보유", q, base, q)})
    bl = pd.DataFrame(brows)
    bl.to_csv(ES.OUT / "g3_blend.csv", index=False, encoding="utf-8-sig")
    print("\n### ② QQQ 혼합 (월초 리밸런싱, 편도 10bp)")
    print(bl.to_string(index=False))
    al = pd.DataFrame(arows)
    al.to_csv(ES.OUT / "g3_alpha.csv", index=False, encoding="utf-8-sig")
    print("\n### ③ QQQ 에 대한 회귀 (일별, 2015~2026)")
    print(al.to_string(index=False))


if __name__ == "__main__":
    main()
