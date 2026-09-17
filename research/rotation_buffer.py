"""약한 종목 매도 · 강한 종목 매수 교체 (월 1회 · 순위 버퍼) — 사용자 제안의 회전율 억제판.

    python research/rotation_buffer.py     # → research/out/rotation_buffer.csv, rotation_buffer_vs.csv

동기 (2026-09-17 대화)
  사용자: "모멘텀이 약한 걸 팔고 강한 걸 사면?" — 기존 교체형(L0·주도주)은 회전율·비용으로 QQQ 에 졌다.
  그래서 매일 교체 대신 월 1회 · 버퍼로 교체 빈도를 줄인 형태만 잰다.

사전 등록 (2026-09-17, 결과 보기 전 커밋)
  유니버스 그날의 PIT 나스닥100 구성종목 (exit_study 패널 그대로 — 사후 선정 종목 목록 쓰지 않음)
  재판정  21거래일마다. t−1 종가 정보로 순위, t 종가 체결. 사이에는 가격 따라 표류. 편도 10bp.
          가격 5일 없으면 청산 (ts_momentum 과 같음). 체결 불가(당일 가격 없음) 보유는 그대로 두고 슬롯 차지
  보유    5 슬롯 동일가중. 재판정일마다 남는 종목 포함 전부 1/5 로 다시 맞춘다. 현금 몫 없음 · 시장 필터 없음
  교체    보유 종목 순위가 BUF 위 밖(또는 구성종목 이탈)이면 매도. 빈 슬롯은 미보유 중 순위 높은 순으로 채움
  시도    B0 순위=12-1개월 수익률, 버퍼 없음 (BUF=5 — 원래 제안 그대로: 상위 5 밖이면 교체)
          B1 순위=12-1개월 수익률, BUF=10
          B2 순위=20일 수익률 상위 20 중 id20(꾸준함) 낮은 순, BUF=10  (residual_mom F3 의 순위를 옮김)
          → 3 시도. 누적 N = 100(leaders_bt 까지) + 3 = 103
  판정    ① exit_study.evaluate 그대로 (L0 대비: 두 반기·25bp·CI·DSR)
          ② **QQQ 대비** 짝지은 블록 부트스트랩 ΔSharpe CI 하한 > 0
          ADOPT = ① ADOPT **그리고** ②. 아니면 운영 반영 안 함
  서술    (판정 미사용) 운영 SOXL 규칙(C2)과 같은 기간 CAGR·Sharpe·MDD, SOXL 대비 ΔSharpe CI ·
          재판정 위상 0/7/14 Sharpe
  불변식  (a) 슬롯 ≥ 유니버스 · 매일 재판정 · 비용 0 이면 동일가중 유니버스 일수익과 같다 (assert)
          (b) 3 설정 prefix(끝 400일 삭제 후 재계산) 괴리 < 1e-12 (assert)
  하지 않음 슬롯 수·버퍼·주기·창 길이 이웃 탐색, 시장 필터 추가, 조합 — 하면 새 시도
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

N_TRIALS = 103
GRID = {
    "B0 12-1개월 · 버퍼 없음(상위5)": dict(rank="mom12_1", buf=5),
    "B1 12-1개월 · 버퍼 상위10": dict(rank="mom12_1", buf=10),
    "B2 20일 상위20 중 꾸준함 · 버퍼 10": dict(rank="fip20", buf=10),
}


def order(F, u, ok, rank):
    """오늘 체결 가능한 구성종목의 순위 리스트 (좋은 순, 종목 인덱스)."""
    if rank == "all":                               # 불변식 (a) 전용: 순위 없이 전부
        return list(np.flatnonzero(ok))
    if rank == "fip20":
        s = np.where(ok, F["mom20"][u], np.nan)
        cand = [i for i in np.argsort(-np.nan_to_num(s, nan=-np.inf)) if np.isfinite(s[i])][:20]
        return sorted(cand, key=lambda i: np.nan_to_num(F["id20"][u][i], nan=np.inf))
    s = np.where(ok, F[rank][u], np.nan)
    return [i for i in np.argsort(-np.nan_to_num(s, nan=-np.inf), kind="stable") if np.isfinite(s[i])]


def rot_sim(P, cfg):
    """반환 형식은 exit_study.sim 과 같다: (일별수익률, 연turnover, 청산기록[], 평균주식비중)."""
    ret, valid, F = P["ret"], P["valid"], P["F"]
    n, m = ret.shape
    fee = cfg.get("cost_bp", ES.COST) / 1e4
    every, phase = cfg.get("every", 21), cfg.get("phase", 0)
    slots, buf = cfg.get("slots", 5), cfg["buf"]
    pos, nanrun = np.zeros(m), np.zeros(m, int)
    cash, traded = 1.0, 0.0
    eq, sw = np.ones(n), np.zeros(n)
    for t in range(1, n):
        u = t - 1
        held = pos > 0
        if held.any():
            rr = ret[t, held]
            pos[held] *= 1 + np.nan_to_num(rr)
            nanrun[held] = np.where(np.isnan(rr), nanrun[held] + 1, 0)
        gone = held & (nanrun >= 5)
        if gone.any():
            traded += pos[gone].sum()
            cash += pos[gone].sum() * (1 - fee)
            pos[gone], nanrun[gone] = 0.0, 0
        if (t - 1 - phase) % every == 0:
            V = cash + pos.sum()
            ok = valid[u] & np.isfinite(ret[t])
            rk = order(F, u, ok, cfg["rank"])
            top_buf = set(rk[:buf])
            stuck = [i for i in np.flatnonzero(pos > 0) if not np.isfinite(ret[t, i])]
            keep = [i for i in np.flatnonzero(pos > 0) if i in top_buf]
            pick = list(keep)
            for i in rk:
                if len(pick) + len(stuck) >= slots:
                    break
                if i not in pick:
                    pick.append(i)
            tgt = pos.copy() if stuck else np.zeros(m)
            tradable = np.ones(m, bool)
            tradable[stuck] = False
            tgt[tradable] = 0.0
            if pick:
                tgt[pick] = (V - pos[stuck].sum()) / slots if slots < m else (V - pos[stuck].sum()) / len(pick)
            d = np.where(tradable, tgt - pos, 0.0)
            sell = np.clip(-d, 0, None)
            buy = np.clip(d, 0, None)
            traded += sell.sum() + buy.sum()
            cash += sell.sum() * (1 - fee) - buy.sum()
            pos = pos - sell + buy * (1 - fee)
            nanrun[pos <= 0] = 0
        V = cash + pos.sum()
        eq[t] = V
        sw[t] = pos.sum() / V
    r = pd.Series(eq[1:], index=P["dates"][1:]).pct_change().dropna()
    return r, traded / 2 / max(eq[1:].mean(), 1e-12) / ((n - 1) / 252), [], float(sw[1:].mean())


def check_ew(P):
    ret, valid = P["ret"], P["valid"]
    ew = pd.Series([0.0] + [np.nan_to_num(ret[t])[valid[t - 2] & np.isfinite(ret[t - 1])].mean()
                            for t in range(2, len(ret))], index=P["dates"][1:])
    r = rot_sim(P, dict(rank="all", buf=10**6, slots=10**6, every=1, cost_bp=0))[0]
    gap = float((r - ew.loc[r.index]).abs().max())
    assert gap < 1e-10, f"동일가중 불변식 실패 (괴리 {gap})"
    return gap


def check_prefix(k_back=400):
    close, _, ndx, qqq = ES.raw()
    full_p = ES.build(close, ndx, qqq, ES.PIT_START)
    part_p = ES.build(close.iloc[:-k_back], ndx, qqq, ES.PIT_START)
    worst = 0.0
    for name, cfg in GRID.items():
        full, part = rot_sim(full_p, cfg)[0], rot_sim(part_p, cfg)[0]
        gap = float((full.loc[part.index] - part).abs().max())
        assert gap < 1e-12, f"{name}: prefix 불변식 실패 (괴리 {gap})"
        worst = max(worst, gap)
    return worst


def soxl_rule():
    """운영 SOXL 규칙(C2) 일수익 — 서술용."""
    from research import soxl_rules as S
    from research import soxl_rules_v2 as V2
    act, _, _ = S.load()
    e, _, _ = S.sim(act, V2.GRID["C2 리밸런싱 목표 40%"])
    return e.pct_change().dropna()


def main():
    panels = ES.panels()
    P = panels["PIT"]
    print(f"불변식 (a) 동일가중 괴리 {check_ew(P):.1e} · (b) prefix 3설정 최대 괴리 {check_prefix():.1e}")

    res = {}
    for pname in ("PIT", "BIAS"):
        for cb in ((ES.COST, ES.STRESS) if pname == "PIT" else (ES.COST,)):
            res[(pname, ES.BASE, cb)] = ES.sim(panels[pname], {**ES.L0, "cost_bp": cb})
            for name, cfg in GRID.items():
                res[(pname, name, cb)] = rot_sim(panels[pname], {**cfg, "cost_bp": cb})
    names = [ES.BASE, *GRID]
    tb = ES.evaluate(res, names, N_TRIALS)
    tb.to_csv(ES.OUT / "rotation_buffer.csv", index=False, encoding="utf-8-sig")
    print(f"\n### ① L0 대비 사전등록 판정 — PIT 2015~2026, 편도 {ES.COST:.0f}bp, 누적 N={N_TRIALS}\n")
    print(tb.to_string(index=False))

    q = ES.benchmarks(P)["지수 매수보유"]
    sx = soxl_rule()
    rows = []
    for name in names:
        r = res[("PIT", name, ES.COST)][0]
        s = C.stat(r)
        bq = C.paired_block_boot(*[a.values for a in q.align(r, join="inner")])
        bx = C.paired_block_boot(*[a.values for a in sx.align(r, join="inner")])
        judge = tb.loc[tb["설정"] == name, "판정"].iloc[0] if name != ES.BASE else "기준"
        rows.append({"설정": name, "CAGR%": round(s["cagr"], 2), "Sharpe": round(s["sharpe"], 3),
                     "MDD%": round(s["mdd"], 1),
                     "QQQ 대비 ΔSh (CI)": f"{bq['d_sharpe']:+.3f} ({bq['d_sharpe_lo']:+.3f}~{bq['d_sharpe_hi']:+.3f})",
                     "SOXL규칙 대비 ΔSh (CI)": f"{bx['d_sharpe']:+.3f} ({bx['d_sharpe_lo']:+.3f}~{bx['d_sharpe_hi']:+.3f})",
                     "위상 0/7/14 Sharpe": "" if name == ES.BASE else " / ".join(
                         f"{C.stat(rot_sim(P, {**GRID[name], 'phase': ph})[0])['sharpe']:.3f}" for ph in (0, 7, 14)),
                     "최종": ("기준" if name == ES.BASE else
                            "ADOPT" if judge == "ADOPT" and bq["d_sharpe_lo"] > 0 else "운영 반영 안 함")})
    for label, r in (("QQQ 매수보유", q), ("운영 SOXL 규칙(C2)", sx.loc[q.index[0]:q.index[-1]])):
        s = C.stat(r)
        rows.append({"설정": label, "CAGR%": round(s["cagr"], 2), "Sharpe": round(s["sharpe"], 3),
                     "MDD%": round(s["mdd"], 1)})
    vs = pd.DataFrame(rows)
    vs.to_csv(ES.OUT / "rotation_buffer_vs.csv", index=False, encoding="utf-8-sig")
    print("\n### ② QQQ 대비 (판정) · SOXL 규칙 대비 · 위상 (서술)\n")
    print(vs.to_string(index=False))


if __name__ == "__main__":
    main()
