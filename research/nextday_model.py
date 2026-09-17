"""다음 날 상승·하락 + 보수적 % 예측 모델 — 정확도 극대화, 집중 매수. 목표: 표본 외 적중 70%.

    python research/nextday_model.py          # → research/out/nextday_*.csv, nextday_model.md
    python research/nextday_model.py --check  # 불변식만

사용자 요구 (2026-09-17)
  · 다음 날 오를지·내릴지를 **보수적인 %** 로 예측 ("다음날 +2% 오른다" 면 실행)
  · 분산보다 **가장 높은 종목에 한 번에 집중**
  · 장기 보유 아님. 백테스트로 **정확도 70% 이상**

사전 등록 (결과 보기 전 커밋)
  표본    PIT 나스닥100 구성종목 2015-01~2026-09 (당시 지수 종목만, 옛 구성종목 포함). 종가만 사용
  특징    전부 t 종가까지 (종목 38 + 시장 8 + 횡단면 순위 4):
          수익률 1·2·3·5·10·20·60일, 변동성 5·20·60일, 1일 수익 z(20일 변동성 대비), RSI2·RSI14,
          이동평균 5·20·50·200 대비 거리, 5·20일 고점 대비 낙폭, 20일 최고점 대비, 연속 상승/하락 일수,
          지수 대비 1·5·20일 초과수익, 지수(^NDX) 수익 1·5·20일·변동성 20일·RSI2, VIX 수준·5일 변화, 요일,
          당일 횡단면 순위(1일·5일 수익, RSI2, 1일 z)
  라벨    다음 날 종가/오늘 종가 − 1 (체결 가정: 오늘 종가 매수 → 다음 날 종가 매도)
  모델    sklearn HistGradientBoosting — ① 분류(상승 확률) ② 분위 회귀 α=0.3 (보수적 % 예측:
          "70% 확률로 이 값 이상")  하이퍼파라미터 고정: lr 0.05 · 300회 · 잎 31 · 최소 표본 200 · 조기종료
  검증    연 단위 확장 창 walk-forward: Y년 예측 모델은 Y−1년 12월 말까지로만 학습 (마지막 하루 라벨은
          Y년 첫날을 보므로 제거). 표본 외 = 2017~2026
  임계값  τ(상승 확률 문턱) 는 **학습 구간 안에서만** 정한다: 학습 마지막 1년을 검증용으로 떼어 먼저 학습 →
          검증 연도에서 '일별 1위 종목 적중률 ≥ 70% 이고 신호 ≥ 20일' 을 만족하는 최소 τ (0.50~0.80, 0.01 간격).
          없으면 그 해는 거래하지 않는다. 그 뒤 전체 학습 구간으로 다시 학습해 Y년에 적용
  전략    S1 매일 상승 확률 1위 1종목 전액 (확률 ≥ τ 인 날만)
          S2 매일 보수적 % 예측 1위 1종목 전액 (예측 > 0 인 날만)
          S3 매일 상승 확률 상위 3종목 균등 (1위 확률 ≥ τ 인 날만)
          → 3 시도. 누적 N = 76 + 3 = 79
  비용    왕복 0.24% (토스 0.1%×2 + 슬리피지) / 참고 0.05%
  판정    **PASS** = S1 표본 외 적중률(다음 날 수익 > 0) ≥ 70% **그리고** 연평균 신호 ≥ 50일
          **그리고** 0.24% 비용 후 누적 수익 > 0 **그리고** 두 반기(2017~21 / 2022~26) 모두 적중률 ≥ 65%.
          하나라도 못 넘으면 FAIL. 기준을 결과 보고 바꾸지 않는다
  서술    전 종목·일 적중률 vs 기본 확률, AUC, 확률 구간별 적중률, 보수적 % 예측의 실제 달성률
          (전체 / 예측 > 0 인 경우 / 예측 ≥ +1%·+2%), 연도별 성과
  불변식  (a) 특징 prefix: 끝 300일을 잘라 다시 계산해도 남은 구간 특징이 같다 (미래 참조 없음) — assert
          (b) 학습 데이터 최대 날짜 < 예측 연도 첫날 (라벨 누설 없음) — 매 연도 assert
  한계    종가 체결 가정(토스 장 마감 직전 체결로 근사). 시가·장중 자료가 옛 구성종목에 없어 쓰지 않는다.
          뉴스·실적 일정 등 가격 밖 정보 없음
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import exit_study as ES      # noqa: E402  (guard() 가 여기서 켜진다)
import pit                                 # noqa: E402

OUT = ES.OUT
N_TRIALS = 79
COST, COST_LOW = 0.0024, 0.0005
YEARS = list(range(2017, 2027))
HP = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=200,
          early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=0)


def rsi(px, n):
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def streak(ret):
    s = np.sign(ret.fillna(0)).to_numpy()
    out = np.zeros_like(s)
    for j in range(s.shape[1]):
        run = 0.0
        for i in range(s.shape[0]):
            v = s[i, j]
            run = run + v if v != 0 and np.sign(run) == v else v
            out[i, j] = run
    return pd.DataFrame(out, index=ret.index, columns=ret.columns)


def features(close, ndx, vix):
    """long 형태 (date, sym) 특징 + 라벨. 모든 창은 후행 — t 행은 t 이하 자료만 본다."""
    px = close.ffill(limit=5)
    ret = px.pct_change(fill_method=None)
    ix = ndx.reindex(close.index).ffill()
    ixr = ix.pct_change()
    vx = vix.reindex(close.index).ffill()
    F = {}
    for n in (1, 2, 3, 5, 10, 20, 60):
        F[f"ret{n}"] = px / px.shift(n) - 1
    for n in (5, 20, 60):
        F[f"vol{n}"] = ret.rolling(n).std()
    F["z1"] = ret / F["vol20"]
    F["rsi2"], F["rsi14"] = rsi(px, 2), rsi(px, 14)
    for n in (5, 20, 50, 200):
        F[f"ma{n}"] = px / px.rolling(n).mean() - 1
    F["dd5"] = px / px.rolling(5).max() - 1
    F["dd20"] = px / px.rolling(20).max() - 1
    F["up20"] = px / px.rolling(20).min() - 1
    F["streak"] = streak(ret)
    for n in (1, 5, 20):
        F[f"xs_ix{n}"] = F[f"ret{n}"].sub(ix / ix.shift(n) - 1, axis=0)
    rank = {k: F[k].rank(axis=1, pct=True) for k in ("ret1", "ret5", "rsi2", "z1")}
    for k, v in rank.items():
        F[f"rk_{k}"] = v
    mkt = pd.DataFrame({"ix1": ixr, "ix5": ix / ix.shift(5) - 1, "ix20": ix / ix.shift(20) - 1,
                        "ixvol20": ixr.rolling(20).std(), "ixrsi2": rsi(ix, 2),
                        "vix": vx, "vix5": vx / vx.shift(5) - 1, "dow": close.index.dayofweek})
    y = ret.shift(-1)                                         # 라벨: 다음 날 수익
    member = pd.DataFrame(pit.mask(close.index, list(close.columns), quiet=True),
                          index=close.index, columns=close.columns)
    long = pd.concat({k: v.stack(future_stack=True) for k, v in F.items()}, axis=1)
    long["y"] = y.stack(future_stack=True)
    long["member"] = member.stack(future_stack=True)
    long.index.names = ["date", "sym"]
    long = long.join(mkt, on="date")
    return long


def load():
    close, _, ndx, _ = ES.raw()
    vix = pickle.load(open(ROOT / "research" / "data" / "pit" / "VIX.pkl", "rb"))
    return close, ndx, vix


def check_prefix(k_back=300):
    close, ndx, vix = load()
    close = close.loc["2020-01-01":]                          # 계산량 절약 — 불변식 확인에는 충분
    full = features(close, ndx, vix)
    cut = features(close.iloc[:-k_back], ndx, vix)
    last = close.index[-k_back - 1]
    cols = [c for c in full.columns if c not in ("y",)]
    a = full.loc[:last, cols].sort_index()
    b = cut.loc[:last, cols].sort_index()
    a, b = a.align(b, join="inner")
    gap = float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float))))
    same_nan = bool((a.isna().to_numpy() == b.isna().to_numpy()).all())
    assert gap < 1e-9 and same_nan, f"특징 prefix 불변식 실패 (괴리 {gap}, NaN 일치 {same_nan})"
    return gap


def top1_hit(dates, p, y, tau):
    df = pd.DataFrame({"d": dates, "p": p, "y": y})
    best = df.loc[df.groupby("d")["p"].idxmax()]
    best = best[best["p"] >= tau]
    return (best["y"] > 0).mean() if len(best) else np.nan, len(best)


def pick_tau(tr, feats):
    """학습 구간의 마지막 1년을 검증으로 떼어 τ 를 정한다 (예측 연도는 보지 않는다)."""
    vy = tr.index.get_level_values("date").max().year
    inner = tr[tr.index.get_level_values("date").year < vy]
    val = tr[tr.index.get_level_values("date").year == vy]
    inner = inner.iloc[:-1] if len(inner) else inner
    clf = HistGradientBoostingClassifier(**HP).fit(inner[feats], inner["y"] > 0)
    p = clf.predict_proba(val[feats])[:, 1]
    d = val.index.get_level_values("date")
    for tau in np.round(np.arange(0.50, 0.805, 0.01), 2):
        hit, n = top1_hit(d, p, val["y"].to_numpy(), tau)
        if n >= 20 and hit >= 0.70:
            return float(tau), hit, n
    return None, np.nan, 0


def main():
    if "--check" in sys.argv:
        print(f"불변식 (a) 특징 prefix 괴리 {check_prefix():.1e}")
        return
    print(f"불변식 (a) 특징 prefix 괴리 {check_prefix():.1e}")
    close, ndx, vix = load()
    data = features(close, ndx, vix)
    data = data[data["member"].astype(bool) & data["y"].notna() & data["ret1"].notna()]
    data = data.loc["2014-10-01":]
    feats = [c for c in data.columns if c not in ("y", "member")]
    dates = data.index.get_level_values("date")
    print(f"표본 {len(data):,} 종목-일 · 특징 {len(feats)}개 · {dates.min().date()}~{dates.max().date()}")

    preds, taus = [], []
    for Y in YEARS:
        tr = data[dates.year < Y]
        te = data[dates.year == Y]
        if not len(te):
            continue
        tr = tr[tr.index.get_level_values("date") < tr.index.get_level_values("date").max()]  # 마지막 날 라벨 = Y년 첫날
        assert tr.index.get_level_values("date").max() < te.index.get_level_values("date").min()
        tau, vhit, vn = pick_tau(tr, feats)
        clf = HistGradientBoostingClassifier(**HP).fit(tr[feats], tr["y"] > 0)
        q30 = HistGradientBoostingRegressor(loss="quantile", quantile=0.3, **HP).fit(tr[feats], tr["y"])
        preds.append(pd.DataFrame({"p": clf.predict_proba(te[feats])[:, 1], "q30": q30.predict(te[feats]),
                                   "y": te["y"].to_numpy()}, index=te.index))
        taus.append({"연도": Y, "τ": tau, "검증 적중%": None if tau is None else round(vhit * 100, 1), "검증 신호일": vn})
        print(f"  {Y}: 학습 {len(tr):,} · τ={tau} (검증 적중 {vhit if tau is None else round(vhit*100,1)}%, {vn}일)")
    P = pd.concat(preds)
    P.to_csv(OUT / "nextday_preds.csv", encoding="utf-8-sig")
    tau_by = {t["연도"]: t["τ"] for t in taus}
    pd.DataFrame(taus).to_csv(OUT / "nextday_tau.csv", index=False, encoding="utf-8-sig")
    report(P, tau_by)


def report(P, tau_by):
    d = P.index.get_level_values("date")
    up = P["y"] > 0
    md = ["# 다음 날 예측 모델 — 표본 외 결과 (2017~2026)"]
    md.append(f"## 전 종목·일\n\n기본 확률(다음 날 상승) {up.mean():.1%} · 모델 적중률(확률>0.5 면 상승) "
              f"{((P['p'] > 0.5) == up).mean():.1%} · AUC {roc_auc_score(up, P['p']):.3f} · 건수 {len(P):,}")
    b = pd.qcut(P["p"], 10, duplicates="drop")
    dec = P.groupby(b, observed=True).agg(건수=("y", "size"), 평균확률=("p", "mean"),
                                          실제상승률=("y", lambda s: (s > 0).mean()), 평균수익=("y", "mean"))
    md.append("## 상승 확률 구간별 실제\n\n" + (dec.assign(평균확률=lambda x: (x.평균확률 * 100).round(1),
                                                  실제상승률=lambda x: (x.실제상승률 * 100).round(1),
                                                  평균수익=lambda x: (x.평균수익 * 100).round(3))).to_string())
    rows = []
    for lab, m in (("전체", P["q30"] > -9), ("예측 > 0", P["q30"] > 0), ("예측 ≥ +1%", P["q30"] >= 0.01),
                   ("예측 ≥ +2%", P["q30"] >= 0.02)):
        s = P[m]
        rows.append({"조건": lab, "건수": len(s), "실제 ≥ 예측 %": round((s["y"] >= s["q30"]).mean() * 100, 1) if len(s) else None,
                     "실제 > 0 %": round((s["y"] > 0).mean() * 100, 1) if len(s) else None,
                     "평균 예측 %": round(s["q30"].mean() * 100, 2) if len(s) else None,
                     "평균 실제 %": round(s["y"].mean() * 100, 2) if len(s) else None})
    md.append("## 보수적 % 예측(30% 분위)의 실제 달성률 — 목표 70%\n\n" + pd.DataFrame(rows).to_string(index=False))

    strat = {}
    g = P.assign(d=d)
    tau_row = d.year.map(tau_by)
    g["tau"] = tau_row
    top = g.loc[g.groupby("d")["p"].idxmax()]
    s1 = top[top["tau"].notna() & (top["p"] >= top["tau"])]
    top3 = g.sort_values("p", ascending=False).groupby("d").head(3)
    ok3 = set(s1["d"])
    s3 = top3[top3["d"].isin(ok3)].groupby("d")["y"].mean()
    tq = g.loc[g.groupby("d")["q30"].idxmax()]
    s2 = tq[tq["q30"] > 0]
    strat["S1 확률 1위 전액 (≥τ)"] = s1.set_index("d")["y"]
    strat["S2 보수적% 1위 전액 (>0)"] = s2.set_index("d")["y"]
    strat["S3 확률 상위3 균등 (1위≥τ)"] = s3
    allday = pd.Index(sorted(set(d)))
    rows, verdict = [], {}
    for n, r in strat.items():
        for lab, c in (("0.24%", COST), ("0.05%", COST_LOW)):
            daily = pd.Series(0.0, index=allday)
            daily.loc[r.index] = r.to_numpy() - c
            eq = (1 + daily).cumprod()
            yrs = len(allday) / 252
            h1 = (r[r.index.year <= 2021] > 0).mean() if (r.index.year <= 2021).any() else np.nan
            h2 = (r[r.index.year >= 2022] > 0).mean() if (r.index.year >= 2022).any() else np.nan
            rows.append({"전략": n, "비용": lab, "신호일": len(r), "연평균 신호": round(len(r) / yrs, 1),
                         "적중%": round((r > 0).mean() * 100, 1) if len(r) else None,
                         "적중% 17~21": round(h1 * 100, 1) if h1 == h1 else None,
                         "적중% 22~26": round(h2 * 100, 1) if h2 == h2 else None,
                         "평균 수익%": round(r.mean() * 100, 3) if len(r) else None,
                         "누적 수익%": round((eq.iloc[-1] - 1) * 100, 1), "MDD%": round((eq / eq.cummax() - 1).min() * 100, 1)})
            if n.startswith("S1") and lab == "0.24%":
                verdict = dict(hit=(r > 0).mean() if len(r) else 0, per_year=len(r) / yrs,
                               net=eq.iloc[-1] - 1, h1=h1, h2=h2)
    st = pd.DataFrame(rows)
    st.to_csv(OUT / "nextday_strategies.csv", index=False, encoding="utf-8-sig")
    md.append("## 집중 매수 전략 (표본 외)\n\n" + st.to_string(index=False))
    v = verdict
    ok = (v["hit"] >= 0.70 and v["per_year"] >= 50 and v["net"] > 0
          and (v["h1"] >= 0.65 if v["h1"] == v["h1"] else False) and (v["h2"] >= 0.65 if v["h2"] == v["h2"] else False))
    md.append(f"## 사전등록 판정: **{'PASS' if ok else 'FAIL'}**\n\nS1 적중 {v['hit']:.1%} (기준 70%) · "
              f"연평균 신호 {v['per_year']:.1f}일 (기준 50) · 0.24% 비용 후 누적 {v['net']:+.1%} (기준 > 0) · "
              f"반기 적중 {v['h1']:.1%} / {v['h2']:.1%} (기준 각 65%)")
    yr = pd.DataFrame({"S1 신호일": s1.groupby(s1["d"].dt.year).size(),
                       "S1 적중%": (s1.groupby(s1["d"].dt.year)["y"].apply(lambda s: (s > 0).mean()) * 100).round(1),
                       "전 종목 기본 상승률%": (g.groupby(g["d"].dt.year)["y"].apply(lambda s: (s > 0).mean()) * 100).round(1)})
    md.append("## 연도별\n\n" + yr.to_string())
    text = "\n\n".join(md)
    (ROOT / "research" / "nextday_model.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
