"""F — 상관군(correlation group) 한도 연구 + nested walk-forward.

    python research/corr_limits.py
      → research/out/corr_grid.csv, corr_nested.csv, corr_summary.md, corr_bootstrap.csv

무엇을 재는가
  종목당 한도(기존 15%)와 **별개로**, 같은 상관군에 슬롯을 몇 개까지 허용할지를 제한한다.
  상관군은 **매 시점 과거 자료로만** 만든다 — t 일의 군집은 t-1 까지의 252일 수익률로
  계산하고, 21거래일마다 갱신한다. 미래 상관은 쓰지 않는다 (assert 로 확인).

동일 노출 vs 동일 위험 (프롬프트 요구 — 분리해서 읽을 것)
  여기 구현은 **동일 노출**이다. 상관군 한도에 걸린 후보는 건너뛰고 순위에서 더
  아래로 내려가 슬롯을 채운다. 즉 현금 비중은 기준선과 같고 **구성만** 바뀐다.
  '한도에 걸리면 현금으로 남긴다'(=노출 축소) 는 다른 실험이며 여기서 하지 않았다.
  동일 노출이 동일 위험을 뜻하지 않는다 — 그래서 변동성·MDD 를 따로 낸다.

★ 생존 편향: 유니버스가 **오늘의** 나스닥100 이다 (PIT 구성종목 없음 — data_gap D1).
  절대 수익률은 물론 규칙 간 상대 비교도 왜곡된다. 이 파일의 결과는 **탐색 전용**이다.
★ 섹터 한도는 하지 않았다 — PIT 섹터 매핑이 없다 (data_gap D10). 오늘 기준 섹터를
  과거에 적용하면 상관군보다 더 심한 전방 편향이 들어간다.
★ Sharpe 는 표준 정의(초과 일수익 평균/표준편차×√252, rf=0)를 쓴다.
  backtest_rules.metrics 의 'sharpe' 는 CAGR/변동성이라 **다른 값**이다.
"""
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from research.isolation import guard        # noqa: E402

guard()

import backtest_slots as BS                 # noqa: E402
from backtest_rules import features         # noqa: E402

OUT = pathlib.Path(__file__).with_name("out")
SEED = 20260912
N_BOOT = 2000
BLOCK = 63                # 거래일. 20일 보유가 겹치는 것을 덮고도 남는 길이
CONF = 0.95

# 운영 설정 (autotrade.py): 슬롯 10, 만기 20거래일
SLOTS, HOLD = 10, 20
CORR_WIN, REGROUP = 252, 21       # 군집 추정 창 / 갱신 주기 (거래일)
COST_BPS = 5.0                    # 기본 편도 비용. 0/5/10/20 민감도는 아래에서 따로
# 사전 등록 격자 — 이게 전부다. 확장하지 않는다 (시도 수를 DSR 에 그대로 쓴다)
RHOS = (0.50, 0.60, 0.70)
GCAPS = (2, 3, 4)
GRID = [None] + list(itertools.product(RHOS, GCAPS))     # None = 기준선(한도 없음)


# ---------- 상관군 (point-in-time) ----------
def cluster_ids(corr, rho):
    """탐욕적 seed 군집. corr[i, seed] >= rho 인 첫 군에 붙이고, 없으면 새 군을 연다.

    종목 순서는 컬럼 순(알파벳)으로 고정 — 수익률·미래 정보에 의존하지 않는다.
    ponytail: 탐욕 seed 군집. 군집 품질이 결론을 바꾸면 average-linkage 로 올린다.
    """
    n = corr.shape[0]
    ids = np.full(n, -1, np.int16)
    seeds = []
    for i in range(n):
        for k, s in enumerate(seeds):
            if corr[i, s] >= rho:
                ids[i] = k
                break
        else:
            seeds.append(i)
            ids[i] = len(seeds) - 1
    return ids


def group_matrix(ret, rho, win=CORR_WIN, step=REGROUP):
    """(n_days, n_sym) 군집 id. t 행은 **t-1 까지**의 win 일 수익률만 본다.

    관측이 모자란 종목(상장 전·결측)은 -1 = 무소속 → 한도의 적용을 받지 않는다.
    """
    n_days, n_sym = ret.shape
    g = np.full((n_days, n_sym), -1, np.int16)
    cur = np.full(n_sym, -1, np.int16)
    for t in range(n_days):
        if t >= win + 1 and (t - win - 1) % step == 0:
            w = ret[t - win:t]                       # ★ t 행 제외 = 미래 없음
            ok = np.isfinite(w).all(axis=0)
            cur = np.full(n_sym, -1, np.int16)
            if ok.sum() >= 2:
                c = np.corrcoef(np.nan_to_num(w[:, ok]), rowvar=False)
                cur[ok] = cluster_ids(np.nan_to_num(c), rho)
        g[t] = cur
    return g


# ---------- 시뮬레이터 (backtest_slots.simulate 에 상관군 한도·비용만 얹음) ----------
def simulate_g(N, hold, rank, ret, dates, groups=None, gcap=None, cost_bps=0.0):
    """반환: (일별 수익률, 최대 단일종목 비중, 연 turnover 배수).

    groups/gcap 가 None 이고 cost_bps=0 이면 backtest_slots.simulate 와 **완전히** 같다
    (아래 check() 가 매 실행 assert 한다).
    """
    n_days, n_sym = ret.shape
    fee = cost_bps / 1e4
    pos = np.zeros(n_sym)
    day = np.full(n_sym, -1)
    cash, maxw, traded = 1.0, 0.0, 0.0
    eq = np.empty(n_days)
    for t in range(n_days):
        held = pos > 0
        if held.any():
            pos[held] *= 1 + np.nan_to_num(ret[t][held])
            gone = held & (t - day >= hold)
            if gone.any():
                amt = pos[gone].sum()
                traded += amt
                cash += amt * (1 - fee)
                pos[gone] = 0.0
                day[gone] = -1
        held = pos > 0
        free = N - int(held.sum())
        if free > 0 and cash > 1e-12:
            rk = rank[t]
            order = np.argsort(rk, kind="stable")
            if gcap is None:
                picks = [c for c in order[:N + n_sym // 4]
                         if np.isfinite(rk[c]) and not held[c]][:free]
            else:
                g = groups[t]
                cnt = {}
                for c in np.flatnonzero(held):
                    if g[c] >= 0:
                        cnt[g[c]] = cnt.get(g[c], 0) + 1
                picks = []
                for c in order:                       # 한도에 걸리면 더 아래로 = 동일 노출
                    if len(picks) >= free:
                        break
                    if not np.isfinite(rk[c]) or held[c]:
                        continue
                    gid = int(g[c])
                    if gid >= 0 and cnt.get(gid, 0) >= gcap:
                        continue
                    picks.append(c)
                    cnt[gid] = cnt.get(gid, 0) + 1
            if picks:
                per = cash / len(picks)
                traded += per * len(picks)
                for c in picks:
                    pos[c] = per * (1 - fee)
                    day[c] = t
                cash -= per * len(picks)
        v = cash + pos.sum()
        eq[t] = v
        if v > 0 and pos.max() / v > maxw:
            maxw = pos.max() / v
    curve = pd.Series(eq, index=dates)
    turnover = traded / 2 / max(np.mean(eq), 1e-12) / (n_days / 252)
    return curve.pct_change().dropna(), maxw, turnover


# ---------- 지표 ----------
def stat(r):
    """표준 정의. sharpe = 평균/표준편차×√252 (rf=0 — 무위험수익률 자료 없음, 명시)."""
    if len(r) < 60:
        return {k: np.nan for k in ("cagr", "vol", "sharpe", "mdd", "calmar", "ret")}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = (eq.iloc[-1] ** (1 / yrs) - 1) * 100
    vol = r.std() * np.sqrt(252) * 100
    mdd = (eq / eq.cummax() - 1).min() * 100
    return {"ret": (eq.iloc[-1] - 1) * 100, "cagr": cagr, "vol": vol,
            "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() else 0.0,
            "mdd": mdd, "calmar": cagr / abs(mdd) if mdd else np.nan}


def paired_block_boot(ra, rb, block=BLOCK, reps=N_BOOT, seed=SEED):
    """같은 날짜 짝을 유지한 원형 블록 재표집. 누적수익 차·Sharpe 차의 CI."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(ra, float), np.asarray(rb, float)
    n = len(a)
    k = int(np.ceil(n / block))
    idx = ((rng.integers(0, n, size=(reps, k))[:, :, None] + np.arange(block)[None, None, :])
           .reshape(reps, k * block) % n)[:, :n]
    A, B = a[idx], b[idx]
    d_cum = np.prod(1 + B, axis=1) - np.prod(1 + A, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_sh = ((B.mean(1) / B.std(1) - A.mean(1) / A.std(1)) * np.sqrt(252))
    lo, hi = (1 - CONF) / 2 * 100, (1 + CONF) / 2 * 100
    return {"d_cum": float(np.prod(1 + b) - np.prod(1 + a)),
            "d_cum_lo": float(np.percentile(d_cum, lo)),
            "d_cum_hi": float(np.percentile(d_cum, hi)),
            "d_sharpe": float((b.mean() / b.std() - a.mean() / a.std()) * np.sqrt(252)),
            "d_sharpe_lo": float(np.nanpercentile(d_sh, lo)),
            "d_sharpe_hi": float(np.nanpercentile(d_sh, hi)),
            "frac_gt0": float((d_cum > 0).mean())}


# ---------- 불변식 (매 실행 확인) ----------
def check(rank, ret, dates, groups):
    """세 가지를 확인한다. 하나라도 깨지면 결과를 내지 않는다."""
    r0, _ = BS.simulate(SLOTS, HOLD, rank, ret, dates)
    r1, _, _ = simulate_g(SLOTS, HOLD, rank, ret, dates, cost_bps=0.0)
    gap = float((r0 - r1).abs().max())
    assert gap == 0.0, f"기준선 동치 실패 (괴리 {gap})"          # ① 같은 저울

    t = len(dates) - 200                                        # ② 미래 상관 참조 없음
    for rho in RHOS:
        cut = group_matrix(ret[:t + 1], rho)[t]
        assert np.array_equal(groups[rho][t], cut), f"군집이 미래를 본다 (rho={rho})"

    g = groups[0.60]                                            # ③ 한도가 실제로 구속한다
    rc, _, _ = simulate_g(SLOTS, HOLD, rank, ret, dates, g, gcap=2, cost_bps=0.0)
    assert float((rc - r0).abs().max()) > 0, "한도가 아무것도 바꾸지 못했다"
    bind = binding_days(SLOTS, HOLD, rank, ret, g, gcap=2)
    assert bind > 0
    print(f"불변식 통과 — 기준선 동치 0.0, 미래참조 없음(rho {len(RHOS)}개), "
          f"기준선이 한도(2)를 넘긴 날 {bind}일 → 제약이 실제로 구속한다")


def binding_days(N, hold, rank, ret, groups, gcap):
    """**기준선**이 같은 상관군에 gcap 개를 넘겨 들고 있던 날수. 제약의 유효성 근거."""
    n_days, n_sym = ret.shape
    pos, day, cash, bad = np.zeros(n_sym), np.full(n_sym, -1), 1.0, 0
    for t in range(n_days):
        held = pos > 0
        if held.any():
            pos[held] *= 1 + np.nan_to_num(ret[t][held])
            gone = held & (t - day >= hold)
            cash += pos[gone].sum()
            pos[gone], day[gone] = 0.0, -1
        held = pos > 0
        free = N - int(held.sum())
        if free > 0 and cash > 1e-12:
            rk = rank[t]
            picks = [c for c in np.argsort(rk, kind="stable")[:N + n_sym // 4]
                     if np.isfinite(rk[c]) and not held[c]][:free]
            if picks:
                per = cash / len(picks)
                for c in picks:
                    pos[c], day[c] = per, t
                cash -= per * len(picks)
        g, cnt = groups[t], {}
        for c in np.flatnonzero(pos > 0):
            if g[c] >= 0:
                cnt[g[c]] = cnt.get(g[c], 0) + 1
        if cnt and max(cnt.values()) > gcap:
            bad += 1
    return bad


# ---------- nested walk-forward ----------
EMBARGO, VAL, FOLDS, WARM, TRAIN_MIN = 20, 504, 6, 300, 1260


def nested_walkforward(rank, ret, dates, groups, cost=COST_BPS):
    """바깥 fold 의 시험구간은 선택에 절대 쓰이지 않는다.

    선택은 fold 시작 **EMBARGO(20거래일=보유기간) 이전**에서 끝나는 VAL 구간에서만 한다.
    20일 보유가 fold 경계를 넘어 겹치는 것을 그만큼 잘라낸다(purge/embargo).
    각 fold 는 현금 1 로 새로 시작한다 — 기준선도 같은 조건이라 비교는 공정하지만,
    fold 앞 20일은 슬롯을 채우는 중이라 그 구간의 절대 수익은 읽지 말 것.
    """
    n = len(dates)
    first = WARM + TRAIN_MIN
    edges = np.linspace(first, n, FOLDS + 1).astype(int)
    rows, te_base, te_sel = [], [], []
    for f in range(FOLDS):
        ts, tend = int(edges[f]), int(edges[f + 1])
        ve, vs = ts - EMBARGO, ts - EMBARGO - VAL
        best, best_sh = None, -np.inf
        for cfg in GRID:
            g = None if cfg is None else groups[cfg[0]]
            cap = None if cfg is None else cfg[1]
            r, _, _ = simulate_g(SLOTS, HOLD, rank[vs:ve], ret[vs:ve], dates[vs:ve],
                                 None if g is None else g[vs:ve], cap, cost)
            sh = stat(r)["sharpe"]
            if sh > best_sh:
                best, best_sh = cfg, sh
        g = None if best is None else groups[best[0]]
        cap = None if best is None else best[1]
        rb, _, tb = simulate_g(SLOTS, HOLD, rank[ts:tend], ret[ts:tend], dates[ts:tend],
                               None, None, cost)
        rs, _, tsv = simulate_g(SLOTS, HOLD, rank[ts:tend], ret[ts:tend], dates[ts:tend],
                                None if g is None else g[ts:tend], cap, cost)
        te_base.append(rb)
        te_sel.append(rs)
        mb, ms = stat(rb), stat(rs)
        rows.append({
            "fold": f, "test_start": dates[ts].date(), "test_end": dates[tend - 1].date(),
            "n_days": tend - ts, "val_start": dates[vs].date(), "val_end": dates[ve - 1].date(),
            "embargo_days": EMBARGO, "trials": len(GRID),
            "selected": "기준선(한도없음)" if best is None else f"rho{best[0]:.2f}/cap{best[1]}",
            "val_sharpe": round(best_sh, 3),
            "base_ret_pct": round(mb["ret"], 2), "sel_ret_pct": round(ms["ret"], 2),
            "base_sharpe": round(mb["sharpe"], 3), "sel_sharpe": round(ms["sharpe"], 3),
            "base_mdd_pct": round(mb["mdd"], 2), "sel_mdd_pct": round(ms["mdd"], 2),
            "base_vol_pct": round(mb["vol"], 2), "sel_vol_pct": round(ms["vol"], 2),
            "base_turnover": round(tb, 1), "sel_turnover": round(tsv, 1)})
    return pd.DataFrame(rows), pd.concat(te_base), pd.concat(te_sel)


def main():
    f, _lab = features(False)
    close, mom, ret1 = f["close"], f["mom"], f["ret1"]
    rank = np.array(mom.rank(axis=1, ascending=False, method="first").to_numpy(float), copy=True)
    rank[np.isnan(rank)] = np.inf
    ret = np.array(ret1.to_numpy(float), copy=True)
    dates = close.index
    print(f"종목 {close.shape[1]}개 · {dates.min().date()}~{dates.max().date()} "
          f"({len(dates)}거래일) · 슬롯 {SLOTS} · 만기 {HOLD}일 · 편도 {COST_BPS}bp")
    print("★ 오늘의 나스닥100 = 생존 편향. 탐색 전용이다.\n")

    groups = {rho: group_matrix(ret, rho) for rho in RHOS}
    for rho in RHOS:
        k = [len(set(groups[rho][t][groups[rho][t] >= 0])) for t in (len(dates) // 2, -1)]
        print(f"  rho {rho:.2f}: 군집 수 (중간/최근) {k[0]} / {k[1]}")
    check(rank, ret, dates, groups)

    # ① 전체표본 격자 — **선택에 쓰인 표본 안**이다. 채택 근거로 쓰지 않는다
    grid_rows = []
    for cfg in GRID:
        g = None if cfg is None else groups[cfg[0]]
        cap = None if cfg is None else cfg[1]
        r, maxw, tv = simulate_g(SLOTS, HOLD, rank, ret, dates, g, cap, COST_BPS)
        a, b = stat(r[:"2015-01-01"]), stat(r["2015-01-01":])
        grid_rows.append({
            "config": "기준선(한도없음)" if cfg is None else f"rho{cfg[0]:.2f}/cap{cfg[1]}",
            "탐_cagr": round(a["cagr"], 2), "탐_sharpe": round(a["sharpe"], 3),
            "탐_mdd": round(a["mdd"], 2), "탐_calmar": round(a["calmar"], 2),
            "검_cagr": round(b["cagr"], 2), "검_sharpe": round(b["sharpe"], 3),
            "검_mdd": round(b["mdd"], 2), "검_calmar": round(b["calmar"], 2),
            "최대단일비중": round(maxw * 100, 1), "연turnover": round(tv, 1)})
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "corr_grid.csv", index=False, encoding="utf-8-sig")
    print("\n### ① 전체표본 격자 (탐 1999~2014 / 검 2015~2026) — 표본 내, 채택 근거 아님")
    print(grid.to_string(index=False))

    # ② nested walk-forward — 선택과 평가를 분리
    wf, rb, rs = nested_walkforward(rank, ret, dates, groups)
    wf.to_csv(OUT / "corr_nested.csv", index=False, encoding="utf-8-sig")
    print(f"\n### ② nested walk-forward ({FOLDS} fold · 선택 시도 {len(GRID)}개/fold "
          f"· embargo {EMBARGO}일)")
    print(wf[["fold", "test_start", "test_end", "selected", "val_sharpe", "base_ret_pct",
              "sel_ret_pct", "base_sharpe", "sel_sharpe", "base_mdd_pct",
              "sel_mdd_pct"]].to_string(index=False))
    mb, ms = stat(rb), stat(rs)
    print(f"\n  fold 시험구간 이어붙임: 기준선 Sharpe {mb['sharpe']:.3f} MDD {mb['mdd']:.1f}% "
          f"| 상관군한도 Sharpe {ms['sharpe']:.3f} MDD {ms['mdd']:.1f}%")

    # ③ 차이의 불확실성 — 짝지은 블록 부트스트랩 (시험구간만)
    bs = paired_block_boot(rb.values, rs.values)
    bs.update({"block_days": BLOCK, "reps": N_BOOT, "seed": SEED, "conf": CONF,
               "표본": "nested walk-forward 시험구간 일별 수익률만"})
    pd.DataFrame([bs]).to_csv(OUT / "corr_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(f"  누적수익 차 {bs['d_cum']:+.4f} (95% CI {bs['d_cum_lo']:+.3f}~{bs['d_cum_hi']:+.3f}), "
          f"Sharpe 차 {bs['d_sharpe']:+.3f} "
          f"(CI {bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f})")
    zero_in = bs["d_cum_lo"] <= 0 <= bs["d_cum_hi"]
    print(f"  → CI 가 0 을 {'포함한다 (부호를 가릴 수 없다)' if zero_in else '제외한다'}")

    # ③-b fold 하나를 빼도 부호가 남는가 (leave-one-fold-out)
    lofo = []
    for f_ in range(FOLDS):
        keep = [i for i in range(FOLDS) if i != f_]
        a = pd.concat([rb[wf.loc[i, "test_start"]:wf.loc[i, "test_end"]] for i in keep])
        b = pd.concat([rs[wf.loc[i, "test_start"]:wf.loc[i, "test_end"]] for i in keep])
        x = paired_block_boot(a.values, b.values)
        lofo.append({"제외_fold": f_, "n_days": len(a),
                     "d_sharpe": round(x["d_sharpe"], 3),
                     "lo": round(x["d_sharpe_lo"], 3), "hi": round(x["d_sharpe_hi"], 3),
                     "CI_0_제외": bool(x["d_sharpe_lo"] > 0 or x["d_sharpe_hi"] < 0)})
    lf = pd.DataFrame(lofo)
    lf.to_csv(OUT / "corr_lofo.csv", index=False, encoding="utf-8-sig")
    print("\n### ③-b fold 하나를 빼면 (Sharpe 차) — 한 구간이 결론을 끌고 가는지 본다")
    print(lf.to_string(index=False))

    # ④ 비용 민감도 — 0/5/10/20bp (실제 비용이라고 주장하지 않는다)
    cost_rows = []
    for bps in (0, 5, 10, 20):
        for cfg in (None, (0.60, 3)):
            g = None if cfg is None else groups[cfg[0]]
            cap = None if cfg is None else cfg[1]
            r, _, tv = simulate_g(SLOTS, HOLD, rank, ret, dates, g, cap, float(bps))
            m = stat(r)
            cost_rows.append({"편도bps": bps,
                              "config": "기준선" if cfg is None else f"rho{cfg[0]}/cap{cfg[1]}",
                              "cagr": round(m["cagr"], 2), "sharpe": round(m["sharpe"], 3),
                              "mdd": round(m["mdd"], 2), "연turnover": round(tv, 1)})
    cs = pd.DataFrame(cost_rows)
    cs.to_csv(OUT / "corr_cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\n### ④ 편도 비용 민감도 (전체표본. 0/5/10/20bp 는 스트레스 격자지 실제 비용이 아니다)")
    print(cs.to_string(index=False))
    print("\n산출물: corr_grid.csv, corr_nested.csv, corr_bootstrap.csv, corr_lofo.csv, corr_cost_sensitivity.csv")


if __name__ == "__main__":
    main()
