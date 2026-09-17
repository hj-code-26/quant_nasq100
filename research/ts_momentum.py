"""종목별 독립 판단 (시계열 모멘텀) — 순위 경쟁·10슬롯 대신 종목마다 자기 추세로 보유 여부를 정한다.

    python research/ts_momentum.py     # → research/out/ts_momentum.csv, ts_momentum_vs_qqq.csv

동기 (2026-09-17 격차 분해, PIT 2015~2026)
  QQQ 18.86% → 동일가중 유니버스 14.55% (−4.3%p) → 10종목 모멘텀 13.42% (−1.1%p, 선별 효과 ≈ 0)
  → 등급·상한 사이징 10.27% (−3.2%p) → 현금10%·변동성타겟 8.09% (−2.2%p) → 비용 7.63%
  사용자 방향: 지수 기준 순위를 버리고 각 종목의 모멘텀을 각각 판단, 지수 동기화는 부분적으로만.

사전 등록 (2026-09-17, 결과 보기 전 커밋)
  유니버스 그날의 PIT 나스닥100 구성종목 전체 (과거 시총 자료가 없어 '시총 상위' 로 자르지 않는다)
  판단    종목마다 독립: 신호 on 이면 보유, off 면 그 몫은 현금. **순위·슬롯 없음. 규칙은 전 종목 공통**
          (종목별 파라미터 최적화는 하지 않는다 — 종목 수만큼 시도가 늘어 과최적화가 확정적이다)
  기본 몫  1/K (K = 그날 유효 구성종목 수). 총노출 = 신호 on 비율
  재판정  20거래일마다 (L0 보유기간과 같음. 매일 판정은 exit_study 에서 회전율로 무너졌다).
          t−1 종가 정보로 판정, t 종가 체결. 사이에는 비중이 가격 따라 표류. 편도 10bp. 가격 5일 없으면 청산
  시도    P1 신호 = 20일 수익률 > 0
          P2 신호 = 12-1개월 수익률 > 0            (Moskowitz·Ooi·Pedersen 계열 표준)
          P3 신호 = 종가 > 200일 이동평균
          P4 P2 + 몫을 종목 변동성 역가중 (1/vol20 을 유효 종목 합으로 정규화)
          P5 P2 + 부분 지수 동기화: 재판정일에 지수 60일 < −3% (운영 하락 판정) 이면 목표 노출 × 0.5
          P6 P1 + 부분 지수 동기화 (P5 와 같은 방식)
          → 6 시도. 누적 N = 58 + 6 = 64
  판정    ① exit_study.evaluate 그대로 (L0 대비: 두 반기·25bp·CI·DSR)
          ② **QQQ 대비** 짝지은 블록 부트스트랩 ΔSharpe CI (구조 문제를 푸는지가 본 질문)
          ADOPT 조건 = ① ADOPT **그리고** ② CI 하한 > 0. 둘 중 하나라도 못 넘으면 운영 반영 안 함
  불변식  (a) 신호 항상 on · 매일 재판정 · 비용 0 이면 동일가중 유니버스 일수익과 같다 (assert)
          (b) 6 설정 모두 prefix(끝 400일 삭제 후 재계산) 괴리 < 1e-12 (assert)
  서술    (판정 미사용) 재판정 시작일 위상 0/5/10/15 일의 Sharpe — 달력 운에 얼마나 기대는지
  하지 않음 조합, 파라미터 이웃 탐색(창 길이·노출 배수) — 하면 새 시도
  한계    실계좌 $570 에서 몫 1/K ≈ $6 → 최소주문 $5 에 걸린다. 하락 동기화(×0.5)면 $3 로 전부 막힌다.
          백테스트는 최소주문 0.1% 가정 — 결과가 좋아도 계좌 규모 문제는 별도로 풀어야 한다.
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

N_TRIALS = 64
TS = {
    "P1 종목별 20일수익>0": dict(sig="mom20"),
    "P2 종목별 12-1개월>0": dict(sig="mom12_1"),
    "P3 종목별 종가>200일선": dict(sig="dist200"),
    "P4 P2 + 변동성 역가중": dict(sig="mom12_1", invvol=True),
    "P5 P2 + 지수 하락 시 노출×0.5": dict(sig="mom12_1", bear=0.5),
    "P6 P1 + 지수 하락 시 노출×0.5": dict(sig="mom20", bear=0.5),
}


def ts_sim(P, cfg):
    """반환 형식은 exit_study.sim 과 같다: (일별수익률, 연turnover, 청산기록[], 평균주식비중)."""
    ret, valid, F, R = P["ret"], P["valid"], P["F"], P["R"]
    n, m = ret.shape
    fee = cfg.get("cost_bp", ES.COST) / 1e4
    every, phase = cfg.get("every", 20), cfg.get("phase", 0)
    sig, always = cfg.get("sig"), cfg.get("always", False)
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
            ok = valid[u] & np.isfinite(ret[t])            # 오늘 체결 가능한 구성종목
            cap = cfg.get("cap")                           # mcap_momentum: 전일 시총 가중 · 상위 top 개
            if cap is not None:
                ok &= np.isfinite(cap[u])
                if cfg.get("top") and ok.sum() > cfg["top"]:
                    cut = np.sort(cap[u][ok])[-cfg["top"]]
                    ok &= cap[u] >= cut
            base = np.zeros(m)
            if ok.any():
                if cap is not None:
                    base = np.where(ok, cap[u], 0.0) / cap[u][ok].sum()
                elif cfg.get("invvol"):
                    iv = np.where(ok, 1 / np.maximum(-F["lowvol"][u], 1e-6), 0.0)
                    iv = np.where(np.isfinite(iv), iv, 0.0)
                    base = iv / iv.sum()
                else:
                    base = ok / ok.sum()
            on = ok if always else ok & (np.nan_to_num(F[sig][u], nan=-1.0) > 0)
            tgt = np.where(on, base, 0.0)
            if cfg.get("redistribute") and tgt.sum() > 0:  # 꺼진 몫을 켜진 종목에 비례 재배분 (전부 꺼지면 현금)
                tgt = tgt / tgt.sum()
            tgt = tgt * V
            if cfg.get("bear") and R["live_bear"][u]:
                tgt *= cfg["bear"]
            d = tgt - pos
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
    # t 일 수익 = t−1 종가에 맞춘 균등 비중(판정 u=t−2, 체결 가능 = t−1 가격 있음) × t 일 수익
    ew = pd.Series([0.0] + [np.nan_to_num(ret[t])[valid[t - 2] & np.isfinite(ret[t - 1])].mean()
                            for t in range(2, len(ret))], index=P["dates"][1:])
    r = ts_sim(P, dict(always=True, every=1, cost_bp=0))[0]
    gap = float((r - ew.loc[r.index]).abs().max())
    assert gap < 1e-10, f"동일가중 불변식 실패 (괴리 {gap})"
    return gap


def check_prefix(k_back=400):
    close, _, ndx, qqq = ES.raw()
    full_p = ES.build(close, ndx, qqq, ES.PIT_START)
    part_p = ES.build(close.iloc[:-k_back], ndx, qqq, ES.PIT_START)
    for name, cfg in TS.items():
        full, part = ts_sim(full_p, cfg)[0], ts_sim(part_p, cfg)[0]
        gap = float((full.loc[part.index] - part).abs().max())
        assert gap < 1e-12, f"{name}: prefix 불변식 실패 (괴리 {gap})"
    return gap


def main():
    panels = ES.panels()
    P = panels["PIT"]
    print(f"불변식 (a) 동일가중 괴리 {check_ew(P):.1e} · (b) prefix 6설정 통과 {check_prefix():.1e}")

    res = {}
    for pname in ("PIT", "BIAS"):
        for cb in ((ES.COST, ES.STRESS) if pname == "PIT" else (ES.COST,)):
            res[(pname, ES.BASE, cb)] = ES.sim(panels[pname], {**ES.L0, "cost_bp": cb})
            for name, cfg in TS.items():
                res[(pname, name, cb)] = ts_sim(panels[pname], {**cfg, "cost_bp": cb})
    names = [ES.BASE, *TS]
    tb = ES.evaluate(res, names, N_TRIALS)
    tb.to_csv(ES.OUT / "ts_momentum.csv", index=False, encoding="utf-8-sig")
    print(f"\n### ① L0 대비 사전등록 판정 — PIT 2015~2026, 편도 {ES.COST:.0f}bp, 누적 N={N_TRIALS}\n")
    print(tb.to_string(index=False))

    q = ES.benchmarks(P)["지수 매수보유"]
    rows = []
    for name in names:
        r = res[("PIT", name, ES.COST)][0]
        x, y = q.align(r, join="inner")
        bs = C.paired_block_boot(x.values, y.values)
        s = C.stat(r)
        phases = "" if name == ES.BASE else " / ".join(
            f"{C.stat(ts_sim(P, {**TS[name], 'phase': ph})[0])['sharpe']:.3f}" for ph in (0, 5, 10, 15))
        judge = tb.loc[tb["설정"] == name, "판정"].iloc[0] if name != ES.BASE else "기준"
        rows.append({"설정": name, "CAGR%": round(s["cagr"], 2), "Sharpe": round(s["sharpe"], 3),
                     "MDD%": round(s["mdd"], 1), "QQQ 대비 ΔSh": round(bs["d_sharpe"], 3),
                     "CI": f"{bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f}",
                     "위상 0/5/10/15 Sharpe": phases,
                     "최종": ("ADOPT" if judge == "ADOPT" and bs["d_sharpe_lo"] > 0 else
                            "기준" if name == ES.BASE else "운영 반영 안 함")})
    s = C.stat(q)
    rows.append({"설정": "QQQ 매수보유", "CAGR%": round(s["cagr"], 2), "Sharpe": round(s["sharpe"], 3),
                 "MDD%": round(s["mdd"], 1)})
    vq = pd.DataFrame(rows)
    vq.to_csv(ES.OUT / "ts_momentum_vs_qqq.csv", index=False, encoding="utf-8-sig")
    print("\n### ② QQQ 대비 (본 질문) + 위상 민감도(서술)\n")
    print(vq.to_string(index=False))


if __name__ == "__main__":
    main()
