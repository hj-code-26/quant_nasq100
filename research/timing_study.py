"""현행 L0 를 거래 단위로 다시 본다 — 우민철(2025) 지수옵션 HFT 논문의 분석 틀을 가져오되 약점은 고친다.

    python research/timing_study.py     # → research/out/timing_study.md, timing_study_trades.csv

가져온 것 → 고친 것
  T1 거래 단위 성과 분포(평균·중앙·사분위·t)   → t 를 **진입일 클러스터**로도 낸다 (논문은 131만 건을 독립으로 봄)
  T2 'HFT 대 일반 계좌' 비교                   → 원화가 아니라 수익률로, 대조군 = **같은 날 진입 가능했던 전 종목**
  T3 Price Ratio 매수/매도 타이밍              → 매수·매도에 **같은 기준**(±10일 종가 평균)을 쓰고,
                                                 추세가 만드는 기계적 부호는 같은 날 대조군을 빼서 제거
  T4 기간별 재분석                             → 사전등록 반기(2015~20/2021~26) + 진입 국면(평상/하락)
  T5 손익 요인 회귀                            → 종속변수는 초과수익 그대로(부호·로그 변환 안 함), SE 는 진입일 클러스터

사전 등록 (결과 보기 전 고정)
  표본  PIT 2015-01~2026-09, 현행 L0(편도 10bp). L0 청산은 전부 '만기' → 진입일 = 청산일 − 20 (assert)
  수익  종목 20일 수익률 − 왕복 20bp. 초과 = − 같은 구간 ^NDX 수익률
  타이밍 BPR = 진입가 / 진입일±10 종가평균 − 1 (<0 이면 싸게 샀다)
         SPR = 청산일±10 종가평균 / 청산가 − 1 (<0 이면 비싸게 팔았다)   ← 미래 자료 사용, 평가 전용
         '능력' 판정은 대조군을 뺀 값의 클러스터 t 가 |t|>2 일 때만
  한계  종가만 있어 장중 VWAP 타이밍(논문의 주 지표)은 잴 수 없다. 실계좌 체결은 표본이 작아 제외
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import exit_study as ES      # noqa: E402  (guard() 가 여기서 켜진다)

OUT = ROOT / "research" / "out"
H, W, RT = 20, 10, 2 * ES.COST / 1e4


def cmean_t(x, g):
    """평균의 naive t 와 클러스터 t (군집합 기반 샌드위치)."""
    x, g = np.asarray(x, float), np.asarray(g)
    e = x - x.mean()
    naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    s = pd.Series(e).groupby(g).sum().to_numpy()
    G = len(s)
    se = np.sqrt((s ** 2).sum() * G / (G - 1)) / len(x)
    return naive, x.mean() / se


def ratios(px, t, c):
    win = px[t - W:t + W + 1, c]
    return (np.nanmean(win) if np.isfinite(win).sum() >= W else np.nan), px[t, c]


def trades(P, exits):
    px, ix, valid, F, R = P["px"], P["ixpx"], P["valid"], P["F"], P["R"]
    n = len(px)
    rows, ctrl = [], {}
    for t1, c, why in exits:
        assert why == "만기", why
        t0 = t1 - H
        if t0 - W < 0 or t1 + W >= n:
            continue
        ref0, p0 = ratios(px, t0, c)
        ref1, p1 = ratios(px, t1, c)
        mret = ix[t1] / ix[t0] - 1
        u = t0 - 1
        if t0 not in ctrl:                      # 대조군: 같은 날 진입 가능했던 전 종목, 같은 20일
            ok = valid[u] & np.isfinite(px[t0]) & np.isfinite(px[t1])
            cc = np.flatnonzero(ok)
            r = px[t1, cc] / px[t0, cc] - 1 - RT
            m0 = np.array([np.nanmean(px[t0 - W:t0 + W + 1, k]) for k in cc]) / px[t0, cc] - 1
            m1 = np.array([np.nanmean(px[t1 - W:t1 + W + 1, k]) for k in cc])
            ctrl[t0] = (r.mean(), np.nanmean(m0), np.nanmean(m1 / px[t1, cc] - 1))
        cr, cb, cs = ctrl[t0]
        ret = p1 / p0 - 1 - RT
        rows.append({
            "진입일": P["dates"][t0], "종목": P["cols"][c],
            "국면": "하락(저변동성)" if R["live_bear"][u] else "평상(모멘텀)",
            "반기": "2015~2020" if P["dates"][t0] <= pd.Timestamp(ES.H1_END) else "2021~2026",
            "수익": ret, "초과": ret - mret, "대조군대비": ret - cr,
            "BPR": p0 / ref0 - 1, "BPR_대조": p0 / ref0 - 1 - cb,
            "SPR": ref1 / p1 - 1, "SPR_대조": ref1 / p1 - 1 - cs,
            "mom20": F["mom20"][u, c], "ret5": F["ret5"][u, c], "vol20": -F["lowvol"][u, c],
            "dist200": F["dist200"][u, c], "지수60일": ix[u] / ix[u - 60] - 1})
    return pd.DataFrame(rows)


def dist(df, col, label):
    x = df[col]
    nt, ct = cmean_t(x, df["진입일"])
    _, mt = cmean_t(x, df["진입일"].dt.to_period("M"))     # 20일 보유가 겹치므로 월 단위가 더 정직하다
    return {"구분": label, "지표": col, "건수": len(x), "평균%": round(x.mean() * 100, 2),
            "중앙%": round(x.median() * 100, 2), "Q1%": round(x.quantile(.25) * 100, 2),
            "Q3%": round(x.quantile(.75) * 100, 2), "양수%": round((x > 0).mean() * 100, 1),
            "naive t": round(nt, 2), "클러스터 t": round(ct, 2), "월클러스터 t": round(mt, 2), "진입일 수": df["진입일"].nunique()}


def main():
    P = ES.panels()["PIT"]
    _, _, exits, _ = ES.sim(P, ES.L0)
    df = trades(P, exits)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "timing_study_trades.csv", index=False, encoding="utf-8-sig")

    groups = [("전체", df)] + [(k, g) for k, g in df.groupby("반기")] + [(k, g) for k, g in df.groupby("국면")]
    md = [f"# 현행 L0 거래 단위 재평가 (PIT, 편도 10bp) — 거래 {len(df)} · 진입일 {df['진입일'].nunique()}"]

    t12 = pd.DataFrame([dist(g, col, k) for col in ("수익", "초과", "대조군대비") for k, g in groups])
    md.append("## T1·T2·T4 성과 분포 — 수익 / 지수 대비 / 같은 날 전 종목 대비\n\n" + t12.to_string(index=False))

    t3 = pd.DataFrame([dist(g, col, k) for col in ("BPR", "BPR_대조", "SPR", "SPR_대조") for k, g in groups])
    md.append("## T3 매수·매도 타이밍 (<0 이 유리, _대조 = 같은 날 전 종목 평균을 뺀 값)\n\n" + t3.to_string(index=False))

    d = df.dropna(subset=["초과", "mom20", "ret5", "vol20", "dist200", "지수60일"])
    X = sm.add_constant(d[["mom20", "ret5", "vol20", "dist200", "지수60일"]].astype(float))
    fit = sm.OLS(d["초과"].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["진입일"].dt.to_period("M"))[0]})
    t5 = pd.DataFrame({"계수": fit.params.round(4), "월클러스터 t": fit.tvalues.round(2)})
    md.append(f"## T5 초과수익 요인 (진입 전일 값, 월 클러스터 SE) — adj R² {fit.rsquared_adj:.4f}\n\n"
              + t5.to_string())

    text = "\n\n".join(md)
    (OUT / "timing_study.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
