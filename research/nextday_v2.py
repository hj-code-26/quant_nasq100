"""다음 날 예측 v2 — 가격 밖 정보(실적·서프라이즈·시가 갭·거래량·반도체 지수) + 확률 보정. 목표: 60% 확답.

    python research/nextday_v2.py          # → research/nextday_v2.md, research/out/nextday_v2_*.csv
    python research/nextday_v2.py --check  # 불변식만

v1(nextday_model.py) FAIL: 가격만으로 AUC 0.515, 1위 종목 적중 44.7%. 사용자 요구(2026-09-17):
가격 밖 정보를 넣고, **다음 날 방향에 대해 60% 이상 확답**을 주는 모델로 다시 세팅.

사전 등록 (결과 보기 전 커밋)
  자료    yfinance 조정 OHLCV 2014~ (PIT 패널 179종목 중 180/182 시가 확보) · 실적 발표 시각·EPS 예상·실적·서프라이즈%
          (176종목, 상장폐지 3종목 결측) · QQQ OHLC · ^SOX · ^NDX · VIX. 구성종목은 pit.mask(당일 지수 편입)
  판단 시점 두 가지를 따로 평가한다
    T1 종가 결정: t 종가 매수 → t+1 종가 매도. 라벨 C[t+1]/C[t]−1.
       실적은 **t 16:00(뉴욕) 전에 발표된 것만** 안다. 오늘 장 후 발표 예정 여부는 사전 공지 일정이라 쓴다
    T2 시가 결정: t+1 시가 매수 → t+1 종가 매도. 라벨 C[t+1]/O[t+1]−1.
       실적은 **t+1 09:30 전 발표까지** 안다(장 후·장 전 서프라이즈 포함). t+1 시가 갭·QQQ 시가 갭을 안다
  특징    v1 가격 특징 36개 + 거래량 z(20일)·거래대금 순위 · 당일 갭·장중 수익·고저 폭·종가 위치(CLV)·ATR14 ·
          ^SOX 1·5일 · 실적: 직전 발표 후 거래일 수(60 상한)·직전 서프라이즈%(±100 절단)·오늘 장 후 발표 예정(T1)·
          밤사이 발표 여부와 그 서프라이즈%(T2) · T2 전용: 종목 시가 갭·갭 z·QQQ 시가 갭·갭−QQQ 갭
  모델    HistGradientBoosting 분류 (v1 과 같은 고정 하이퍼파라미터) + **등위 보정(isotonic)**:
          학습 구간 마지막 1년을 떼어 내부 모델로 예측 → 그 연도에서 보정 함수 학습 → 전체 학습 모델 출력에 적용.
          보수적 % 는 분위 회귀 α=0.3 (서술용)
  검증    연 단위 확장 창 walk-forward 2017~2026. 학습 마지막 날 제거(라벨이 예측 연도를 봄), 매년 날짜 assert
  확답    보정 확률 ≥ 0.60 → '상승 확답', ≤ 0.40 → '하락 확답'. 그 사이는 답하지 않는다
  전략    (T1·T2 각각) top1 = 확답 중 확률 1위 1종목 전액 · top3 = 상위 3종목 균등 (1위가 확답일 때)
          → 4 시도. 누적 N = 79 + 4 = 83
  비용    왕복 0.24% / 참고 0.05%
  판정    T1·T2 각각 **PASS** = 모두 충족
          ① 상승 확답 전체 적중률 ≥ 60% 그리고 연평균 상승 확답 ≥ 250 종목-일
          ② top1: 적중 ≥ 60% · 연평균 ≥ 50일 · 0.24% 비용 후 누적 > 0
          ③ 두 반기(2017~21 / 2022~26) 상승 확답 적중률 각각 ≥ 57%
          하나라도 못 넘으면 FAIL. 기준은 결과를 보고 바꾸지 않는다
  서술    AUC, 기본 확률, 보정 확률 구간별 실제 상승률, 하락 확답 적중률, 보수적 % 달성률, 연도별
  불변식  (a) 특징 prefix: 가격·실적 표를 끝 300일 전에서 잘라 다시 만들어도 남은 구간 특징이 같다 — assert
          (b) 학습 최대 날짜 < 예측 연도 첫날 — 매 연도 assert
  한계    실적 표는 오늘의 티커로 조회 — 티커가 재사용된 옛 종목은 다른 회사 실적이 붙을 수 있다.
          종가·시가 체결 가정(경매가 근사). 뉴스·장중 자료 없음(뉴스는 실전 예측 기록으로만 잰다)
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research import exit_study as ES      # noqa: E402  (guard() 가 여기서 켜진다)
from research import nextday_model as V1   # noqa: E402
import pit                                 # noqa: E402

DATA = ROOT / "research" / "data" / "pit"
OUT = ES.OUT
NY = "America/New_York"
N_TRIALS = 83
COST, COST_LOW = 0.0024, 0.0005
YEARS = V1.YEARS
HP = V1.HP
IDX = ("QQQ", "^SOX", "^NDX")


def load():
    px = pickle.load(open(DATA / "ohlcv_panel.pkl", "rb"))
    earn = pickle.load(open(DATA / "earnings.pkl", "rb"))
    vix = pickle.load(open(DATA / "VIX.pkl", "rb"))
    return px, earn, vix


def earn_table(earn, dates):
    """실적 행을 (종목, 발표시각 뉴욕, 서프라이즈) 로. 발표 예정(실적 없음) 행도 남긴다."""
    rows = []
    for sym, df in earn.items():
        for ts, r in df.iterrows():
            rows.append({"sym": sym, "ts": ts.tz_convert(NY).tz_localize(None),
                         "surp": np.clip(r.get("Surprise(%)"), -100, 100) if pd.notna(r.get("Surprise(%)")) else np.nan,
                         "reported": pd.notna(r.get("Reported EPS"))})
    e = pd.DataFrame(rows)
    # 같은 종목·같은 시각 중복 행(SMCI 2025-02 등, 서프라이즈 부호까지 다름)은 평균으로 합친다 —
    # 어느 행을 고르느냐가 정렬 순서에 달리면 prefix 불변식이 깨진다
    e = (e.groupby(["sym", "ts"], as_index=False).agg(surp=("surp", "mean"), reported=("reported", "max"))
         .sort_values(["ts", "sym"]))
    return e[e["ts"] <= dates.max() + pd.Timedelta(days=2)]


def earn_features(e, dates, syms):
    """판단 시각 이전에 **발표된** 실적만 본다 (merge_asof, 같은 시각 제외).
    T1 컷오프 = t 16:00, T2 컷오프 = 다음 거래일 09:30."""
    nxt = pd.Series(dates[1:].tolist() + [dates[-1] + pd.Timedelta(days=1)], index=dates)
    grid = pd.MultiIndex.from_product([dates, syms], names=["date", "sym"]).to_frame(index=False)
    grid["c1"] = grid["date"] + pd.Timedelta(hours=16)
    grid["c2"] = grid["date"].map(nxt) + pd.Timedelta(hours=9, minutes=30)
    rep = e[e["reported"]][["sym", "ts", "surp"]].sort_values("ts")
    pos = pd.Series(np.arange(len(dates)), index=dates)
    out = grid[["date", "sym"]].copy()
    for tag, col in (("1", "c1"), ("2", "c2")):
        g = grid.sort_values(col)
        m = pd.merge_asof(g, rep.rename(columns={"ts": "last_ts", "surp": f"surp{tag}"}), left_on=col,
                          right_on="last_ts", by="sym", direction="backward", allow_exact_matches=False)
        m = m.sort_values(["date", "sym"]).reset_index(drop=True)
        ld = m["last_ts"].dt.normalize()
        # 반응 첫날: 장 후(16시 이후) 발표면 다음 거래일
        react = np.where(m["last_ts"].dt.hour >= 16, ld + pd.Timedelta(days=1), ld)
        rpos = np.searchsorted(dates.values, pd.to_datetime(react).values.astype("datetime64[ns]"))
        since = pos.reindex(m["date"]).to_numpy() - rpos
        out[f"esince{tag}"] = np.where(m["last_ts"].notna(), np.clip(since, 0, 60), np.nan)
        out[f"surp{tag}"] = m[f"surp{tag}"].to_numpy()
        if tag == "2":                                   # 밤사이(t 16:00 ~ t+1 09:30) 발표
            night = m["last_ts"].notna() & (m["last_ts"] >= grid.sort_values(["date", "sym"])["c1"].to_numpy())
            out["enight2"] = night.astype(float).to_numpy()
            out["enight_surp2"] = np.where(night, m["surp2"], np.nan)
    sched = e[["sym", "ts"]].copy()                      # 사전 공지된 일정: 오늘 장 후 발표 예정
    sched["date"] = sched["ts"].dt.normalize()
    tonight = sched[sched["ts"].dt.hour >= 16].drop_duplicates(["date", "sym"]).assign(etonight1=1.0)
    out = out.merge(tonight[["date", "sym", "etonight1"]], on=["date", "sym"], how="left")
    out["etonight1"] = out["etonight1"].fillna(0.0)
    return out.set_index(["date", "sym"])


def build(px, earn, vix, end=None):
    O, H, L, C, V = (px[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    if end is not None:
        O, H, L, C, V = (x.loc[:end] for x in (O, H, L, C, V))
    stocks = [c for c in C.columns if c not in IDX]
    close = C[stocks]
    base = V1.features(close, C["^NDX"], vix)            # v1 가격 특징 + 라벨(y = C[t+1]/C[t]−1) + member
    o, h, l_, v = O[stocks], H[stocks], L[stocks], V[stocks]
    prev = close.shift(1)
    ret = close.pct_change(fill_method=None)
    vol20 = ret.rolling(20).std()
    tr = pd.concat({"a": h - l_, "b": (h - prev).abs(), "c": (l_ - prev).abs()}).groupby(level=1).max()
    X = {"gap0": o / prev - 1, "intra0": close / o - 1, "range0": (h - l_) / prev,
         "clv0": (2 * close - h - l_) / (h - l_).replace(0, np.nan), "atr14": tr.rolling(14).mean() / close,
         "vz20": np.log(v.replace(0, np.nan) / v.rolling(20).mean()), "dv_rank": (v * close).rank(axis=1, pct=True)}
    o1 = o.shift(-1)                                     # t+1 시가 (T2 결정 시점에 안다)
    qgap = O["QQQ"].shift(-1) / C["QQQ"] - 1
    X["gap1"] = o1 / close - 1
    X["gapz1"] = X["gap1"] / vol20
    X["gapx1"] = X["gap1"].sub(qgap, axis=0)
    long = pd.concat({k: val.stack(future_stack=True) for k, val in X.items()}, axis=1)
    long.index.names = ["date", "sym"]
    base = base.join(long)
    base["qgap1"] = base.index.get_level_values("date").map(qgap)
    sox = C["^SOX"]
    base["sox1"] = base.index.get_level_values("date").map(sox.pct_change())
    base["sox5"] = base.index.get_level_values("date").map(sox / sox.shift(5) - 1)
    base["y2"] = (close.shift(-1) / o1 - 1).stack(future_stack=True).reindex(base.index)
    dates = close.index
    ef = earn_features(earn_table(earn, dates) if end is None else
                       earn_table(earn, dates).pipe(lambda e: e[e["ts"] < pd.Timestamp(end) + pd.Timedelta(days=1)]),
                       dates, stocks)
    return base.join(ef)


T1_EXTRA = ["gap0", "intra0", "range0", "clv0", "atr14", "vz20", "dv_rank", "sox1", "sox5",
            "esince1", "surp1", "etonight1"]
T2_EXTRA = T1_EXTRA[:-3] + ["esince2", "surp2", "enight2", "enight_surp2", "gap1", "gapz1", "gapx1", "qgap1"]


def check_prefix(k_back=300):
    px, earn, vix = load()
    px = {k: v.loc["2021-01-01":] for k, v in px.items()}
    full = build(px, earn, vix)
    end = px["Close"].index[-k_back]
    cut = build(px, earn, vix, end=end)
    last = px["Close"].index[-k_back - 2]                # T2 특징은 t+1 시가를 쓰므로 하루 더 앞까지만 비교
    cols = [c for c in full.columns if c not in ("y", "y2", "member")]
    a = full.loc[:last, cols].sort_index()
    b = cut.loc[:last, cols].sort_index()
    a, b = a.align(b, join="inner")
    av, bv = a.to_numpy(float), b.to_numpy(float)
    gap = float(np.nanmax(np.abs(av - bv))) if np.isfinite(av).any() else 0.0
    assert gap < 1e-9 and (np.isnan(av) == np.isnan(bv)).all(), f"prefix 불변식 실패 (괴리 {gap})"
    return gap


def fit_predict(tr, te, feats, label):
    ytr = tr[label] > 0
    vy = tr.index.get_level_values("date").max().year
    din = tr.index.get_level_values("date")
    inner, val = tr[din.year < vy], tr[din.year == vy]
    inner = inner[inner.index.get_level_values("date") < inner.index.get_level_values("date").max()]
    m_in = HistGradientBoostingClassifier(**HP).fit(inner[feats], inner[label] > 0)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        m_in.predict_proba(val[feats])[:, 1], (val[label] > 0).astype(float))
    clf = HistGradientBoostingClassifier(**HP).fit(tr[feats], ytr)
    raw = clf.predict_proba(te[feats])[:, 1]
    q30 = HistGradientBoostingRegressor(loss="quantile", quantile=0.3, **HP).fit(tr[feats], tr[label])
    return pd.DataFrame({"raw": raw, "p": iso.predict(raw), "q30": q30.predict(te[feats]),
                         "y": te[label].to_numpy()}, index=te.index)


def evaluate(P, name):
    d = P.index.get_level_values("date")
    up, H2 = P["y"] > 0, pd.Timestamp("2022-01-01")
    yrs = len(pd.Index(d).unique()) / 252
    sig_up, sig_dn = P[P["p"] >= 0.60], P[P["p"] <= 0.40]
    md = [f"## {name}", f"건수 {len(P):,} · 기본 상승률 {up.mean():.1%} · AUC(원 확률) {roc_auc_score(up, P['raw']):.3f}"]
    b = pd.cut(P["p"], [0, .3, .4, .45, .5, .55, .6, .65, .7, 1.0], include_lowest=True)
    t = P.groupby(b, observed=True).agg(건수=("y", "size"), 실제상승률=("y", lambda s: round((s > 0).mean() * 100, 1)),
                                         평균수익bp=("y", lambda s: round(s.mean() * 1e4, 1)))
    md.append("보정 확률 구간별 실제\n\n" + t.to_string())
    hu = (sig_up["y"] > 0).mean() if len(sig_up) else np.nan
    hd = (sig_dn["y"] < 0).mean() if len(sig_dn) else np.nan
    su = sig_up.index.get_level_values("date")
    h1 = (sig_up[su < H2]["y"] > 0).mean() if (su < H2).any() else np.nan
    h2 = (sig_up[su >= H2]["y"] > 0).mean() if (su >= H2).any() else np.nan
    md.append(f"상승 확답(p≥0.60): {len(sig_up):,}건 (연 {len(sig_up)/yrs:,.0f}) · 적중 {hu:.1%} · 반기 {h1:.1%} / {h2:.1%}\n\n"
              f"하락 확답(p≤0.40): {len(sig_dn):,}건 · 적중(실제 하락) {hd:.1%}")
    rows = []
    for lab, m in (("예측 > 0", P["q30"] > 0), ("예측 ≥ +1%", P["q30"] >= 0.01), ("예측 ≥ +2%", P["q30"] >= 0.02)):
        s = P[m]
        rows.append({"조건": lab, "건수": len(s), "실제 ≥ 예측 %": round((s["y"] >= s["q30"]).mean() * 100, 1) if len(s) else None})
    md.append("보수적 % 예측 달성률 (목표 70%)\n\n" + pd.DataFrame(rows).to_string(index=False))

    g = sig_up.assign(d=su).sort_values("p", ascending=False)
    top1 = g.groupby("d").head(1).set_index("d")["y"]
    top3 = g.groupby("d").head(3).groupby("d")["y"].mean()
    top3 = top3[top3.index.isin(top1.index)]
    allday = pd.Index(sorted(set(d)))
    srows, v = [], {}
    for sn, r in (("top1 전액", top1), ("top3 균등", top3)):
        for cl, c in (("0.24%", COST), ("0.05%", COST_LOW)):
            daily = pd.Series(0.0, index=allday)
            daily.loc[r.index] = r.to_numpy() - c
            eq = (1 + daily).cumprod()
            srows.append({"전략": sn, "비용": cl, "신호일": len(r), "연평균": round(len(r) / yrs, 1),
                          "적중%": round((r > 0).mean() * 100, 1) if len(r) else None,
                          "평균bp": round(r.mean() * 1e4, 1) if len(r) else None,
                          "누적%": round((eq.iloc[-1] - 1) * 100, 1), "MDD%": round((eq / eq.cummax() - 1).min() * 100, 1)})
            if sn == "top1 전액" and cl == "0.24%":
                v = dict(hit=(r > 0).mean() if len(r) else 0.0, n=len(r) / yrs, net=eq.iloc[-1] - 1)
    md.append("집중 매수\n\n" + pd.DataFrame(srows).to_string(index=False))
    ok = (hu >= 0.60 and len(sig_up) / yrs >= 250 and v["hit"] >= 0.60 and v["n"] >= 50 and v["net"] > 0
          and h1 >= 0.57 and h2 >= 0.57)
    md.append(f"**판정 {name}: {'PASS' if ok else 'FAIL'}** — ① 확답 적중 {hu:.1%}(≥60%)·연 {len(sig_up)/yrs:,.0f}건(≥250) "
              f"② top1 적중 {v['hit']:.1%}(≥60%)·연 {v['n']:.0f}일(≥50)·비용 후 {v['net']:+.1%}(>0) ③ 반기 {h1:.1%}/{h2:.1%}(≥57%)")
    yr = sig_up.groupby(su.year)["y"].agg(확답건수="size", 적중=lambda s: round((s > 0).mean() * 100, 1))
    yr["기본상승률"] = P.groupby(d.year)["y"].apply(lambda s: round((s > 0).mean() * 100, 1))
    md.append("연도별 상승 확답\n\n" + yr.to_string())
    return "\n\n".join(md), ok


def main():
    gap = check_prefix()
    print(f"불변식 (a) 특징 prefix 괴리 {gap:.1e}")
    if "--check" in sys.argv:
        return
    px, earn, vix = load()
    data = build(px, earn, vix)
    data = data[data["member"].astype(bool) & data["ret1"].notna()].loc["2014-10-01":]
    feats1 = [c for c in V1_FEATS(data)] + T1_EXTRA
    feats2 = [c for c in V1_FEATS(data)] + T2_EXTRA
    dates = data.index.get_level_values("date")
    print(f"표본 {len(data):,} 종목-일 · T1 특징 {len(feats1)} · T2 특징 {len(feats2)}")
    out, verdicts = ["# 다음 날 예측 v2 — 표본 외 2017~2026"], {}
    for name, feats, label in (("T1 종가→다음 날 종가", feats1, "y"), ("T2 다음 날 시가→종가", feats2, "y2")):
        preds = []
        for Y in YEARS:
            sub = data[data[label].notna()]
            sd = sub.index.get_level_values("date")
            tr, te = sub[sd.year < Y], sub[sd.year == Y]
            if not len(te):
                continue
            tr = tr[tr.index.get_level_values("date") < tr.index.get_level_values("date").max()]
            assert tr.index.get_level_values("date").max() < te.index.get_level_values("date").min()
            preds.append(fit_predict(tr, te, feats, label))
            print(f"  {name} {Y}: 학습 {len(tr):,}")
        P = pd.concat(preds)
        P.to_csv(OUT / f"nextday_v2_{label}.csv", encoding="utf-8-sig")
        text, ok = evaluate(P, name)
        verdicts[name] = ok
        out.append(text)
    out.append("## 사전등록 판정 요약\n\n" + "\n".join(f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in verdicts.items()))
    text = "\n\n".join(out)
    (ROOT / "research" / "nextday_v2.md").write_text(text, encoding="utf-8")
    print(text)


def V1_FEATS(data):
    skip = {"y", "y2", "member"} | set(T1_EXTRA) | set(T2_EXTRA) | {"esince2", "surp2", "enight2", "enight_surp2"}
    return [c for c in data.columns if c not in skip]


if __name__ == "__main__":
    main()
