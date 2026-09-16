"""G — 현행 설정보다 나은 설정이 있는가 (청산·시장 국면·진입·보조지표) — PIT 패널 기준.

    python research/exit_study.py            # 전체: 불변식 → 보조지표 관계 → 그리드 → 판정
    python research/exit_study.py --quick    # 불변식 + 기준선만

왜 다시 재나
  기존 결론("손절은 해롭다", "모멘텀 청산은 만기보다 못하다")은 전부 **오늘의 나스닥100**
  패널에서 나왔다. 그 패널은 0 으로 간 회사가 없고 **회복한 종목만** 남아 있으므로
  떨어지는 종목을 들고 버티는 규칙이 구조적으로 유리하다. PIT 패널에서 다시 잰다.

사전 등록 (결과를 보기 전에 고정 — 바꾸면 시도 수에 합산)
  주 표본  PIT 구성종목 2015-01 ~ 2026-09 (당시 지수 종목만 매수 후보, 옛 구성종목 80개 복원)
  보조 표본 1999~2014 오늘의 유니버스 — 생존 편향. **방향 확인용**이고 손절류에 불리하다
  체결     t-1 종가까지의 정보로 결정 → t 종가 체결. 만기만 사전 확정 달력이라 지연 없음
  비용     편도 10bp (기본) / 25bp (스트레스 — 토스 수수료 수준)
  기준선   L0 = 운영 근사: 슬롯10·만기20·종목당15%×모멘텀등급·현금10%·20일수익률>0 진입
           ·지수60일<-3% 이면 저변동성 선별·계좌 변동성 타겟 30%. **LLM 재량은 모사 불가**
  판정     ① 두 반기(2015~2020 / 2021~2026) 모두 Sharpe 가 L0 보다 높다 (10bp)
           ② 25bp 에서도 전체 Sharpe 가 L0 보다 높다
           ③ 짝지은 블록 부트스트랩 ΔSharpe 95% CI 가 0 을 제외한다
           ④ DSR ≥ 0.95 (L0 대비 초과수익 계열, N = 이 파일의 전체 시도 수)
           ①~④ 전부 = ADOPT · ①② = CANDIDATE(섀도 대상) · 그 외 = REJECT
  조합     그룹(X 청산 / M 시장 / E 진입 / R 현행규칙 제거)마다 CANDIDATE 중 ΔSharpe 최고 1개만
           합쳐 1회 더 잰다. 그 1회도 N 에 더한다.

★ 한계
  · 결손 31종목(대부분 피인수 — 프리미엄 상승 후 소멸)은 가격이 없어 빠진다 → 편향 '축소'
  · 옛 구성종목은 종가만 있어 ATR 대신 20일 수익률 표준편차로 저변동성을 잰다 (운영은 ATR%)
  · 국면 판정 지수는 ^NDX (QQQ 이력이 2014-12 부터라 200일선 워밍업이 안 된다. 상관 0.9996)
  · 현금 수익률 0% (M4 제외). 방어형 규칙에 불리한 가정이다
"""
import multiprocessing as mp
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from research.isolation import guard        # noqa: E402

guard()

import pit                                  # noqa: E402
from backtest_bear_exposure import dsr      # noqa: E402
from research import corr_limits as C       # noqa: E402  (simulate_g · stat · paired_block_boot)

OUT = ROOT / "research" / "out"
PITDIR = ROOT / "research" / "data" / "pit"
PIT_START, H1_END, BIAS_START, BIAS_END = "2015-01-01", "2020-12-31", "1999-01-01", "2014-12-31"
COST, STRESS = 10.0, 25.0
FEATS = ("mom20", "mom60", "mom126", "mom12_1", "mom20_vadj", "nh52", "lowvol",
         "rsi14", "dist50", "dist200", "ret5", "bbp", "macdh")


# ---------- 패널 ----------
def build(close, idx, park, start):
    """지표는 전부 후행 창(rolling/ewm adjust=False)이라 t 행은 t 이하 자료만 본다 — prefix 불변식으로 검증."""
    dates = close.index
    px = close.ffill(limit=5)                       # 거래정지 며칠은 잇고, 소멸 종목은 5일 뒤 끊는다
    ret = px.pct_change(fill_method=None)
    f = {}
    f["mom20"] = close / close.shift(20) - 1
    f["mom60"] = close / close.shift(60) - 1
    f["mom126"] = close / close.shift(126) - 1
    f["mom12_1"] = close.shift(21) / close.shift(252) - 1
    vol20 = ret.rolling(20).std()
    f["lowvol"] = -vol20
    f["mom20_vadj"] = f["mom20"] / (vol20 * np.sqrt(20))
    f["nh52"] = close / close.rolling(252, min_periods=200).max()
    sma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    f["bbp"] = (close - (sma20 - 2 * sd20)) / (4 * sd20)
    f["dist50"] = close / close.rolling(50).mean() - 1
    f["dist200"] = close / close.rolling(200).mean() - 1
    f["ret5"] = close / close.shift(5) - 1
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi14"] = 100 - 100 / (1 + up / dn)
    macd = px.ewm(span=12, adjust=False).mean() - px.ewm(span=26, adjust=False).mean()
    f["macdh"] = (macd - macd.ewm(span=9, adjust=False).mean()) / px

    ix = idx.reindex(dates).ffill()
    dd = (ix / ix.rolling(252).max() - 1) * 100
    reg = {"live_bear": (ix / ix.shift(60) - 1) * 100 < -3,          # 운영 market_regime
           "below200": ix < ix.rolling(200).mean(),
           "dd15": (dd <= -15).astype(float).rolling(3, min_periods=1).max() > 0}  # 운영 bear_derisk 3일 지연
    valid = pit.mask(dates, list(close.columns), quiet=True) & close.notna().to_numpy()

    k0 = max(int(dates.searchsorted(pd.Timestamp(start))) - 1, 0)   # 하루 앞: t=1 이 첫 거래일
    sl = slice(k0, None)
    return {"dates": dates[sl], "cols": list(close.columns),
            "px": px.to_numpy(float)[sl], "ret": ret.to_numpy(float)[sl], "valid": valid[sl],
            "F": {k: v.to_numpy(float)[sl] for k, v in f.items()},
            "R": {k: v.to_numpy(bool)[sl] for k, v in reg.items()},
            "park": park.reindex(dates).ffill().pct_change(fill_method=None).fillna(0).to_numpy(float)[sl],
            "ixpx": ix.to_numpy(float)[sl]}


def raw():
    d = pickle.load(open(ROOT / "data_cache" / "yf_ohlcv.pkl", "rb"))
    base = d["close"]
    ndx = base["^NDX"]
    base = base.drop(columns=["^NDX"])
    old = pickle.load(open(PITDIR / "pit_prices.pkl", "rb"))
    qqq = pickle.load(open(PITDIR / "QQQ_2015.pkl", "rb")).squeeze()
    add = [c for c in old.columns if c not in base.columns]
    close = base.join(old[add], how="outer").loc["2014-01-01":base.index[-1]]
    close = close.loc[:, close.notna().sum() > 60]
    b = base.loc[:BIAS_END]
    b = b.loc[:, b.notna().sum() > 300]
    return close, b, ndx, qqq


def panels():
    close, b, ndx, qqq = raw()
    return {"PIT": build(close, ndx, qqq, PIT_START), "BIAS": build(b, ndx, ndx, BIAS_START)}


# ---------- 시뮬레이터 ----------
L0 = dict(slots=10, hold=20, cost_bp=COST, lag=True, rank="mom20", entry_pos=True, tiers=True,
          reserve=0.10, maxpos=0.15, live_regime=True, vt=30.0)


def sim(P, cfg):
    """운영 구조 슬롯 시뮬. 반환 (일별수익률, 연turnover, 청산기록[(t, 종목열, 사유)], 평균주식비중).

    lag=True: t 일 결정은 t-1 종가 정보(F[t-1]·보유 종목의 t-1 종가 기준 낙폭)로 하고 t 종가에 체결.
    lag=False + 부가규칙 전부 끔 + stale=False + min_frac=0 이면 corr_limits.simulate_g 와 같다 (check).
    """
    ret, valid, F, R = P["ret"], P["valid"], P["F"], P["R"]
    n, m = ret.shape
    N, H = cfg.get("slots", 10), cfg.get("hold", 20)
    fee = cfg.get("cost_bp", COST) / 1e4
    lag = cfg.get("lag", True)
    rank_key = cfg.get("rank", "mom20")
    tiers, reserve, maxpos = cfg.get("tiers", False), cfg.get("reserve", 0.0), cfg.get("maxpos", 1.0)
    live_regime, vt, entry_pos = cfg.get("live_regime", False), cfg.get("vt"), cfg.get("entry_pos", False)
    trail, stop = cfg.get("trail"), cfg.get("stop")
    mom_exit, sma_exit, extend = cfg.get("mom_exit"), cfg.get("sma50_exit"), cfg.get("extend")
    mkt, bear_cap, park = cfg.get("mkt"), cfg.get("bear_cap"), cfg.get("park", False)
    above200, rsi_max = cfg.get("above200"), cfg.get("rsi_max")
    stale, min_frac = cfg.get("stale", True), cfg.get("min_frac", 1e-3)
    # 라이브 진단용 opt-in (기본 꺼짐 — 주면 안 주나 결과가 같다):
    #   bear_noentry: 하락 국면에서 신규 진입을 아예 막는다. 운영 2단계 프롬프트가
    #     "약 구간(20일<10%)이면 hold" 라 저변동성 후보를 사실상 전부 거부하는 현상의 근사.
    #   probe: dict 를 주면 진입 차단 일수를 센다 (현금유지·최소주문에 막힌 날).
    bear_noentry = cfg.get("bear_noentry", False)
    probe = cfg.get("probe")
    #   extend_daily: 만기 연장 시 보유 시계를 **리셋하지 않는다**. X14 는 연장하면 H일을
    #     더 주고 그때 다시 보지만, 운영은 진입 시각을 브로커 체결 이력에서 읽으므로
    #     리셋할 수단이 없다 — 만기 이후 **매 사이클** 재확인하고 순위에서 빠지는 날 판다.
    #     운영에 옮길 수 있는 형태가 어느 쪽인지 재려고 둘을 분리했다.
    extend_daily = cfg.get("extend_daily", False)
    feeq = fee if park else 0.0                     # 대기 현금을 QQQ 로 둘 때 드나드는 비용

    pos = np.zeros(m)
    day = np.full(m, -1)
    grow, peak = np.ones(m), np.ones(m)             # 진입 이후 가격 배수 / 그 최고치
    nanrun = np.zeros(m, int)
    cash, traded = 1.0, 0.0
    eq, sw = np.ones(n), np.zeros(n)
    exits = []

    def rule_flags(u):
        held = pos > 0
        if not held.any():
            return []
        out = []
        if trail:
            out.append(("추적손절", held & (grow / peak - 1 <= -trail / 100)))
        if stop:
            out.append(("손절", held & (grow - 1 <= -stop / 100)))
        if mom_exit:
            out.append(("모멘텀음전", held & (F["mom20"][u] < 0)))
        if sma_exit:
            out.append(("50일선이탈", held & (F["dist50"][u] < 0)))
        return out

    def close_out(mask):
        pos[mask], day[mask], grow[mask], peak[mask], nanrun[mask] = 0.0, -1, 1.0, 1.0, 0

    for t in range(1, n):
        u = t - 1 if lag else t
        fl = rule_flags(u) if lag else None         # 어제 종가 기준 판정 → 오늘 종가 체결
        if park:
            cash *= 1 + P["park"][t]
        held = pos > 0
        if held.any():
            rr = ret[t, held]
            g = 1 + np.nan_to_num(rr)
            pos[held] *= g
            grow[held] *= g
            peak[held] = np.maximum(peak[held], grow[held])
            nanrun[held] = np.where(np.isnan(rr), nanrun[held] + 1, 0)
        if not lag:
            fl = rule_flags(u)
        bear = bool(live_regime and R["live_bear"][u])
        score = F["lowvol"][u] if bear else F[rank_key][u]

        # 1) 청산: 만기 → 규칙 → 소멸(가격 5일 없음)
        held = pos > 0
        if held.any():
            gone = np.zeros(m, bool)
            if H:
                exp = held & (t - day >= H)
                if extend and exp.any():            # 만기지만 오늘도 진입 자격 상위권이면 연장(거래 없음)
                    ok = valid[u] & np.isfinite(score)
                    if entry_pos and not bear:
                        ok &= F["mom20"][u] > 0
                    if ok.sum() >= extend:
                        keep = exp & ok & (score >= np.partition(score[ok], -extend)[-extend])
                        if not extend_daily:
                            day[keep] = t          # 시계 리셋 → H일 뒤에 다시 본다
                        exp &= ~keep               # extend_daily 면 내일 또 만기라 매일 재확인
                exits += [(t, c, "만기") for c in np.flatnonzero(exp)]
                gone |= exp
            for why, f in fl:
                f = f & held & ~gone
                exits += [(t, c, why) for c in np.flatnonzero(f)]
                gone |= f
            if stale:
                gone |= held & (nanrun >= 5)
            if gone.any():
                amt = pos[gone].sum()
                traded += amt
                cash += amt * (1 - fee) * (1 - feeq)
                close_out(gone)

        # 2) 노출 상한: 변동성 타겟(운영 _vol_target_cap) · 252일고점 오버레이 · 200일선 전량현금
        V = cash + pos.sum()
        cap = 1.0
        if vt and t > 61:
            e = eq[t - 61:t]
            vol = (e[1:] / e[:-1] - 1).std(ddof=1) * np.sqrt(252) * 100
            w = min(1.0, max(0.05, sw[t - 61:t].mean()))
            if vol > 0:
                cap = min(cap, vt / (vol / w))
        if bear_cap is not None and R["dd15"][u]:
            cap = min(cap, bear_cap)
        if mkt == "exit" and R["below200"][u]:
            cap = 0.0
        excess = pos.sum() - V * cap
        if excess > 1e-4 * V:
            for c in np.argsort(-pos):              # 큰 종목부터 (운영 validate_orders 순서)
                if excess <= 1e-4 * V or pos[c] <= 0:
                    break
                cut = min(excess, pos[c])
                pos[c] -= cut
                excess -= cut
                traded += cut
                cash += cut * (1 - fee) * (1 - feeq)
                if pos[c] <= 1e-12:
                    exits.append((t, c, "노출축소"))
                    close_out(np.arange(m) == c)

        # 3) 진입
        if not (mkt == "nonew" and R["below200"][u]) and not (bear_noentry and bear):
            held = pos > 0
            free = N - int(held.sum())
            budget = min(cash - reserve * V, V * cap - pos.sum())
            if probe is not None and free > 0:
                probe["빈슬롯일"] = probe.get("빈슬롯일", 0) + 1
                if budget <= 1e-12:
                    probe["현금유지에 막힘"] = probe.get("현금유지에 막힘", 0) + 1
                elif budget < min_frac * V:
                    probe["최소주문에 막힘"] = probe.get("최소주문에 막힘", 0) + 1
            if free > 0 and budget > 1e-12:
                ok = valid[u] & np.isfinite(score) & ~held
                if entry_pos and not bear:
                    ok &= F["mom20"][u] > 0
                if above200:
                    ok &= F["dist200"][u] > 0
                if rsi_max:
                    ok &= F["rsi14"][u] < rsi_max
                cand = np.flatnonzero(ok)
                if cand.size:
                    picks = cand[np.argsort(-score[cand], kind="stable")][:free]
                    if tiers:                       # 운영 MOMENTUM_TIERS × MAX_POSITION_PCT
                        mo = F["mom20"][u]
                        amts = [maxpos * V * (1.0 if bear or mo[c] >= .2 else .7 if mo[c] >= .1
                                              else .4 if mo[c] >= 0 else 0.0) for c in picks]
                    else:
                        amts = [min(budget / len(picks), maxpos * V)] * len(picks)
                    for c, a in zip(picks, amts):
                        a = min(a, budget)
                        if a <= 0 or a < min_frac * V:
                            continue
                        pos[c] = a * (1 - fee)
                        cash -= a * (1 + feeq)
                        budget -= a
                        traded += a
                        day[c], grow[c], peak[c], nanrun[c] = t, 1.0, 1.0, 0
        V = cash + pos.sum()
        eq[t] = V
        sw[t] = pos.sum() / V if V > 0 else 0.0
    r = pd.Series(eq[1:], index=P["dates"][1:]).pct_change().dropna()
    return r, traded / 2 / max(eq[1:].mean(), 1e-12) / ((n - 1) / 252), exits, float(sw[1:].mean())


def benchmarks(P, cost_bp=COST):
    d = P["dates"]
    q = pd.Series(P["park"], index=d).iloc[2:]
    on = ~P["R"]["below200"][:-1]                   # 어제 종가가 200일선 위면 오늘 보유
    sw = np.r_[0, np.abs(np.diff(on.astype(int)))] * cost_bp / 1e4
    qs = pd.Series(np.where(on, P["park"][1:], 0.0) - sw, index=d[1:]).iloc[1:]
    with np.errstate(all="ignore"):
        ew = np.array([np.nanmean(np.where(P["valid"][t - 1], P["ret"][t], np.nan))
                       for t in range(1, len(d))])
    return {"지수 매수보유": q, "지수 200일선 타이밍": qs,
            "유니버스 동일가중(비용0)": pd.Series(np.nan_to_num(ew), index=d[1:]).iloc[1:]}


# ---------- 불변식 ----------
def check(P):
    ret, valid, mom = P["ret"], P["valid"], P["F"]["mom20"]
    rk = pd.DataFrame(np.where(valid, mom, np.nan)).rank(axis=1, ascending=False, method="first")
    rk = np.nan_to_num(rk.to_numpy(float), nan=np.inf)
    a, _, _ = C.simulate_g(10, 20, rk[1:], ret[1:], P["dates"][1:], None, None, 5.0)
    b, _, _, _ = sim(P, dict(slots=10, hold=20, cost_bp=5.0, lag=False, stale=False, min_frac=0.0))
    gap = float((a - b).abs().max())
    assert gap < 1e-9, f"엔진 동치 실패 (괴리 {gap})"
    return gap


def check_prefix(k_back=400):
    """데이터를 t 에서 잘라 지표를 **다시 계산**해도 t 까지의 일별 수익률이 같아야 한다 (미래 참조 없음)."""
    close, _, ndx, qqq = raw()
    cfg = {**L0, "trail": 15, "mkt": "nonew", "extend": 20, "bear_cap": 0.7, "rank": "mom12_1",
           "rsi_max": 80}
    full = sim(build(close, ndx, qqq, PIT_START), cfg)[0]
    cut = close.iloc[:-k_back]
    part = sim(build(cut, ndx, qqq, PIT_START), cfg)[0]
    gap = float((full.loc[part.index] - part).abs().max())
    assert gap < 1e-12, f"prefix 불변식 실패 — 미래 참조 (괴리 {gap})"
    return gap, len(part)


# ---------- 보조지표 관계 ----------
def fwd(px, h):
    """t 행 = t+1 종가 매수 → t+1+h 종가 수익률 (결정 t 종가, 체결 t+1)."""
    out = np.full_like(px, np.nan)
    out[:-(h + 1)] = px[h + 1:] / px[1:-h] - 1
    return out


def ic_table(P, lo, hi, h=20):
    d, valid = P["dates"], P["valid"]
    fr = fwd(P["px"], h)
    ts = [t for t in range(0, len(d) - h - 1, h) if pd.Timestamp(lo) <= d[t] <= pd.Timestamp(hi)]
    rows = []
    for k in FEATS:
        ics, tops, bots = [], [], []
        for t in ts:
            x, y = P["F"][k][t], fr[t]
            ok = valid[t] & np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 30:
                continue
            x, y = x[ok], y[ok]
            ics.append(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
            tops.append(y[x >= np.quantile(x, .9)].mean() - y.mean())
            bots.append(y[x <= np.quantile(x, .1)].mean() - y.mean())
        ics = np.array(ics)
        rows.append({"지표": k, "표본일": len(ics), "평균IC": round(ics.mean(), 4),
                     "t": round(ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics)), 2),
                     "IC>0비율%": round((ics > 0).mean() * 100, 1),
                     "상위10%초과%": round(np.mean(tops) * 100, 2),
                     "하위10%초과%": round(np.mean(bots) * 100, 2)})
    return pd.DataFrame(rows)


def cohort_table(P, lo, hi, back=10, h=10, top=10):
    """보유 중 조건: t-back 에 모멘텀 상위 top 이라 산 종목들이 t 에 조건을 만족하면
    남은 h 일(t+1 체결 → t+1+h) 동안 유니버스 대비 얼마나 벌었나. 날짜별 차이의 평균·t (같은 날 종목끼리 상관)."""
    d, px, valid, F = P["dates"], P["px"], P["valid"], P["F"]
    fr = fwd(px, h)
    ts = [t for t in range(back + 1, len(d) - h - 1, h) if pd.Timestamp(lo) <= d[t] <= pd.Timestamp(hi)]
    conds = {
        "고점대비 -10% 이하": lambda t, c, e: px[t, c] / np.nanmax(px[e + 1:t + 1, c]) - 1 <= -.10,
        "고점대비 -15% 이하": lambda t, c, e: px[t, c] / np.nanmax(px[e + 1:t + 1, c]) - 1 <= -.15,
        "매수가대비 -10% 이하": lambda t, c, e: px[t, c] / px[e + 1, c] - 1 <= -.10,
        "20일수익률 음전": lambda t, c, e: F["mom20"][t, c] < 0,
        "50일선 하회": lambda t, c, e: F["dist50"][t, c] < 0,
        "RSI14 > 80": lambda t, c, e: F["rsi14"][t, c] > 80,
        "RSI14 < 40": lambda t, c, e: F["rsi14"][t, c] < 40,
        "볼린저%b > 1": lambda t, c, e: F["bbp"][t, c] > 1,
        "MACD히스토 < 0": lambda t, c, e: F["macdh"][t, c] < 0,
    }
    acc = {k: {"diff": [], "yes": [], "no": []} for k in conds}
    for t in ts:
        e = t - back
        s = np.where(valid[e] & (F["mom20"][e] > 0), F["mom20"][e], np.nan)
        ok = np.isfinite(s)
        if ok.sum() < top:
            continue
        coh = [c for c in np.argsort(-np.nan_to_num(s, nan=-np.inf), kind="stable")[:top]
               if np.isfinite(px[e + 1, c]) and np.isfinite(px[t, c]) and np.isfinite(fr[t, c])]
        um = np.nanmean(np.where(valid[t], fr[t], np.nan))
        for k, fn in conds.items():
            with np.errstate(all="ignore"):
                flag = np.array([bool(fn(t, c, e)) for c in coh])
            ex = np.array([fr[t, c] - um for c in coh])
            acc[k]["yes"] += list(ex[flag])
            acc[k]["no"] += list(ex[~flag])
            if flag.any() and (~flag).any():
                acc[k]["diff"].append(ex[flag].mean() - ex[~flag].mean())
    rows = []
    for k, a in acc.items():
        df = np.array(a["diff"])
        rows.append({"조건(보유 중)": k, "해당건수": len(a["yes"]),
                     "해당 초과%": round(np.mean(a["yes"]) * 100, 2) if a["yes"] else np.nan,
                     "비해당 초과%": round(np.mean(a["no"]) * 100, 2) if a["no"] else np.nan,
                     "날짜별차이%": round(df.mean() * 100, 2) if len(df) > 2 else np.nan,
                     "t": round(df.mean() / df.std(ddof=1) * np.sqrt(len(df)), 2) if len(df) > 2 else np.nan,
                     "비교일": len(df)})
    return pd.DataFrame(rows)


def after_exits(P, exits, hs=(10, 20)):
    """팔고 난 뒤 그 종목이 어떻게 됐나 — '버텼으면' 의 사후 확인 (평가 전용)."""
    px, ix, n = P["px"], P["ixpx"], len(P["px"])
    rows = []
    for t, c, why in exits:
        rec = {"사유": why}
        for h in hs:
            if t + h < n and np.isfinite(px[t, c]) and np.isfinite(px[t + h, c]):
                rec[f"매도후{h}일%"] = (px[t + h, c] / px[t, c] - 1) * 100
                rec[f"지수대비{h}일%"] = rec[f"매도후{h}일%"] - (ix[t + h] / ix[t] - 1) * 100
        rows.append(rec)
    df = pd.DataFrame(rows)
    g = df.groupby("사유")
    out = g.mean().round(2)
    out.insert(0, "건수", g.size())
    out["매도후20일 상승비율%"] = g[f"매도후{hs[-1]}일%"].apply(lambda x: round((x > 0).mean() * 100, 1))
    return out


# ---------- 그리드 (사전 등록) ----------
def v(**kw):
    return {**L0, **kw}


GRID = {
    "L0 현행(운영 근사)": L0,
    "R1 국면전환(저변동성 선별) 제거": v(live_regime=False),
    "R2 변동성타겟 제거": v(vt=None),
    "R3 모멘텀등급 사이징 제거(균등)": v(tiers=False),
    "R4 부가규칙 전부 제거(단순 슬롯)": v(live_regime=False, vt=None, tiers=False, entry_pos=False,
                                  reserve=0.0, maxpos=1.0),
    "X1 추적손절 10%": v(trail=10),
    "X2 추적손절 15%": v(trail=15),
    "X3 추적손절 20%": v(trail=20),
    "X4 추적손절 25%": v(trail=25),
    "X5 추적손절 15%·만기없음": v(trail=15, hold=None),
    "X6 추적손절 25%·만기없음": v(trail=25, hold=None),
    "X7 손절 10%": v(stop=10),
    "X8 손절 20%": v(stop=20),
    "X9 모멘텀 음전 청산": v(mom_exit=True),
    "X10 50일선 이탈 청산": v(sma50_exit=True),
    "X11 만기 10일": v(hold=10),
    "X12 만기 40일": v(hold=40),
    "X13 만기 60일": v(hold=60),
    "X14 만기 때 상위20이면 연장": v(extend=20),
    "M1 지수<200일선: 신규매수 중단": v(mkt="nonew"),
    "M2 지수<200일선: 전량 현금": v(mkt="exit"),
    "M3 REGIME_DERISK 70%": v(bear_cap=0.70),
    "M4 대기현금 QQQ 보유": v(park=True),
    "E1 순위=60일 모멘텀": v(rank="mom60"),
    "E2 순위=126일 모멘텀": v(rank="mom126"),
    "E3 순위=12-1개월 모멘텀": v(rank="mom12_1"),
    "E4 순위=변동성조정 20일": v(rank="mom20_vadj"),
    "E5 순위=52주고점 근접": v(rank="nh52"),
    "E6 종목 200일선 위만 진입": v(above200=True),
    "E7 RSI14≥80 진입 제외": v(rsi_max=80),
    "E8 슬롯5·종목당20%": v(slots=5, maxpos=0.20),
    "E9 슬롯15·종목당10%": v(slots=15, maxpos=0.10),
}
BASE = "L0 현행(운영 근사)"

_P = None


def _init():
    global _P
    _P = panels()


def _run(job):
    pname, name, cfg = job
    r, tv, ex, w = sim(_P[pname], cfg)
    keep = pname == "PIT" and cfg["cost_bp"] == COST
    return pname, name, cfg["cost_bp"], r, tv, (ex if keep else None), w


def run_grid(grid, pool):
    jobs = [(p, n, {**c, "cost_bp": cb}) for n, c in grid.items()
            for p, cb in (("PIT", COST), ("PIT", STRESS), ("BIAS", COST))]
    res = {}
    for pname, name, cb, r, tv, ex, w in pool.imap_unordered(_run, jobs):
        res[(pname, name, cb)] = (r, tv, ex, w)
    return res


def evaluate(res, names, n_trials):
    b10 = res[("PIT", BASE, COST)][0]
    b25 = res[("PIT", BASE, STRESS)][0]
    bb = res[("BIAS", BASE, COST)][0]
    ex = {n: (res[("PIT", n, COST)][0] - b10).dropna() for n in names if n != BASE}
    var_sr = float(np.var([x.mean() / x.std(ddof=1) for x in ex.values() if x.std() > 0], ddof=1))
    H2 = pd.Timestamp(H1_END) + pd.Timedelta(days=1)
    rows = []
    for n in names:
        r10, tv, _, w = res[("PIT", n, COST)]
        r25, rb = res[("PIT", n, STRESS)][0], res[("BIAS", n, COST)][0]
        m, m1, m2 = C.stat(r10), C.stat(r10[:H1_END]), C.stat(r10[H2:])
        s25, sb = C.stat(r25), C.stat(rb)
        row = {"설정": n, "CAGR%": round(m["cagr"], 2), "Sharpe": round(m["sharpe"], 3),
               "MDD%": round(m["mdd"], 1), "Calmar": round(m["calmar"], 2),
               "평균주식비중%": round(w * 100), "연turnover": round(tv, 1),
               "Sh 15~20": round(m1["sharpe"], 3), "Sh 21~26": round(m2["sharpe"], 3),
               "Sh 25bp": round(s25["sharpe"], 3), "Sh 편향99~14": round(sb["sharpe"], 3)}
        if n != BASE:
            a1, a2 = C.stat(b10[:H1_END])["sharpe"], C.stat(b10[H2:])["sharpe"]
            c1 = m1["sharpe"] > a1 and m2["sharpe"] > a2
            c2 = s25["sharpe"] > C.stat(b25)["sharpe"]
            x, y = b10.align(r10, join="inner")
            bs = C.paired_block_boot(x.values, y.values)
            c3 = bs["d_sharpe_lo"] > 0
            p, _ = dsr(ex[n], n_trials, var_sr)
            c4 = p >= 0.95
            worse = m1["sharpe"] < a1 and m2["sharpe"] < a2 and bs["d_sharpe_hi"] < 0
            row.update({"ΔSh": round(bs["d_sharpe"], 3),
                        "ΔSh CI": f"{bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f}",
                        "DSR": round(p, 3),
                        "편향구간 방향": "같음" if (sb["sharpe"] > C.stat(bb)["sharpe"]) == (bs["d_sharpe"] > 0) else "반대",
                        "판정": ("ADOPT" if c1 and c2 and c3 and c4 else "CANDIDATE" if c1 and c2
                               else "HARMFUL" if worse else "REJECT")})
        rows.append(row)
    return pd.DataFrame(rows)


def combo(tb):
    """그룹마다 CANDIDATE 이상 중 ΔSharpe 최고 1개를 L0 위에 합친다 (사전 등록 규칙)."""
    picked = []
    for g in ("R", "X", "M", "E"):
        sub = tb[tb["설정"].str.startswith(g) & tb["판정"].isin(["ADOPT", "CANDIDATE"])]
        if len(sub):
            picked.append(sub.sort_values("ΔSh").iloc[-1]["설정"])
    if len(picked) < 2:
        return None, picked
    cfg = dict(L0)
    for p in picked:
        cfg.update({k: val for k, val in GRID[p].items() if L0.get(k, "∅") != val})
    return cfg, picked


def show(title, df):
    print(f"\n### {title}")
    print(df.to_string(index=False) if isinstance(df, pd.DataFrame) and df.index.name is None
          else df.to_string())


def main():
    t0 = time.time()
    P = panels()
    pp = P["PIT"]
    print(f"PIT 패널 {pp['dates'][1].date()}~{pp['dates'][-1].date()} · {len(pp['cols'])}종목 · "
          f"하루 평균 후보 {pp['valid'].sum(1).mean():.0f} | 편향 패널 {P['BIAS']['dates'][1].date()}~"
          f"{P['BIAS']['dates'][-1].date()} · {len(P['BIAS']['cols'])}종목")
    print(f"불변식: simulate_g 동치 괴리 {check(pp):.1e}", end=" · ")
    gap, nd = check_prefix()
    print(f"prefix(끝 400일 삭제 후 재계산) 괴리 {gap:.1e} ({nd}일)")

    if "--quick" in sys.argv:
        r = sim(pp, L0)[0]
        print(C.stat(r))
        return

    # ① 보조지표 ↔ 향후 20일 수익률 (횡단면)
    for lab, p, lo, hi in (("PIT 2015~2020", pp, PIT_START, H1_END), ("PIT 2021~2026", pp, "2021-01-01", "2030"),
                           ("편향 1999~2014", P["BIAS"], BIAS_START, BIAS_END)):
        t = ic_table(p, lo, hi)
        t.to_csv(OUT / f"g_ic_{lab.split()[0]}_{lab.split()[1][:4]}.csv", index=False, encoding="utf-8-sig")
        show(f"① 보조지표 순위상관(IC) — {lab}, 비중첩 20일", t)

    # ② 보유 중 조건 → 남은 10일
    for lab, p, lo, hi in (("PIT 2015~2020", pp, PIT_START, H1_END), ("PIT 2021~2026", pp, "2021-01-01", "2030"),
                           ("편향 1999~2014", P["BIAS"], BIAS_START, BIAS_END)):
        t = cohort_table(p, lo, hi)
        t.to_csv(OUT / f"g_cohort_{lab.split()[0]}_{lab.split()[1][:4]}.csv", index=False, encoding="utf-8-sig")
        show(f"② 모멘텀 상위10 매수 10일 뒤 조건별 → 남은 10일 유니버스 대비 — {lab}", t)

    # ③ 그리드
    n_trials = len(GRID) - 1 + 1                    # +1 = 조합 1회 (사전 예약)
    with mp.Pool(min(11, mp.cpu_count() - 1), initializer=_init) as pool:
        res = run_grid(GRID, pool)
        tb = evaluate(res, list(GRID), n_trials)
        cfg, picked = combo(tb)
        if cfg:
            GRID["C 조합: " + " + ".join(x.split()[0] for x in picked)] = cfg
            res.update(run_grid({k: GRID[k] for k in list(GRID)[-1:]}, pool))
            tb = evaluate(res, list(GRID), n_trials)
    tb.to_csv(OUT / "g_grid.csv", index=False, encoding="utf-8-sig")
    show(f"③ 그리드 — PIT 2015~2026 (편도 {COST:.0f}bp, 시도 N={n_trials}), 조합 후보 {picked or '없음'}", tb)

    bench = benchmarks(pp)
    base = res[("PIT", BASE, COST)][0]
    rows = []
    for k, r in bench.items():
        m = C.stat(r)
        x, y = base.align(r, join="inner")
        bs = C.paired_block_boot(x.values, y.values)
        rows.append({"기준": k, "CAGR%": round(m["cagr"], 2), "Sharpe": round(m["sharpe"], 3),
                     "MDD%": round(m["mdd"], 1), "L0 대비 ΔSh": round(bs["d_sharpe"], 3),
                     "CI": f"{bs['d_sharpe_lo']:+.3f}~{bs['d_sharpe_hi']:+.3f}"})
    bt = pd.DataFrame(rows)
    bt.to_csv(OUT / "g_bench.csv", index=False, encoding="utf-8-sig")
    show("④ 벤치마크 (PIT 구간, QQQ 총수익)", bt)

    # ⑤ 판 뒤에 어떻게 됐나
    for n in (BASE, "X2 추적손절 15%", "X9 모멘텀 음전 청산", "X10 50일선 이탈 청산", "X7 손절 10%"):
        ae = after_exits(pp, res[("PIT", n, COST)][2])
        ae.to_csv(OUT / f"g_after_{n.split()[0]}.csv", encoding="utf-8-sig")
        show(f"⑤ 매도 후 경로 — {n}", ae)

    yr = pd.DataFrame({n.split()[0]: (1 + res[("PIT", n, COST)][0]).resample("YE").prod() - 1
                       for n in list(GRID) if n.split()[0] in ("L0", "X2", "X9", "M1", "M2", "M4", "E3")
                       or n.startswith("C ")})
    yr["QQQ"] = (1 + bench["지수 매수보유"]).resample("YE").prod() - 1
    yr.index = yr.index.year
    show("⑥ 연도별 수익률 %", (yr * 100).round(1).reset_index())
    print(f"\n산출물 research/out/g_*.csv · {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
