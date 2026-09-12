"""추가 알고리즘 규칙 후보 탐색 — 포트폴리오 수익 곡선으로 평가.

⑧⑩ 은 "건당 수익률"로 규칙을 비교했다. 그건 두 가지를 못 잰다.
  · 안 사는 규칙(국면 필터)의 가치 — 현금으로 쉰 날이 표본에서 빠져버린다
  · 위험 — 같은 수익이어도 변동성·낙폭이 다르면 다른 전략이다
그래서 여기서는 **실제 자본 곡선**을 만든다.

포트폴리오 구성 (겹치는 트랜치)
  · 매일 자본의 1/HOLD 를 그 날 규칙이 고른 TOP 종목에 동일 비중으로 넣고 HOLD 일 뒤 뺀다
  · 따라서 항상 HOLD 개 트랜치 × TOP 종목 = HOLD*TOP 칸이 돌아간다
  · 규칙이 그 날 아무것도 안 고르면 그 트랜치는 현금(수익 0) — 국면 필터가 제대로 평가된다
  · 진입은 그 날 종가, 수익은 다음 날부터 → 미래 참조 없음 (holding = 선택행렬을 20일 롤링합 후 1일 시프트)

평가
  · 탐색 1999~2014 / 검증 2015~2026 으로 나눈다. 두 구간 모두 하락장을 포함한다
    (탐색: 닷컴·금융위기 / 검증: 2018 4분기·코로나·2022)
  · **두 구간에서 모두 기준선을 이겨야** 채택 후보로 본다. 한쪽만 이기면 과적합으로 본다

사용:  python backtest_rules.py            (전체)
       python backtest_rules.py --curve    (채택 후보의 연도별 수익률까지)

★ 생존 편향: 유니버스가 오늘의 나스닥 100. 절대 CAGR 은 크게 부풀려져 있다.
  **기준선 대비 차이(Δ)로만** 읽을 것. 모든 규칙이 같은 편향을 공유한다.
"""
import sys

import numpy as np
import pandas as pd

from backtest_volume import DIR_BAND, VR_HI
from market_data import INDEX, load_ohlcv

TOP = 5
HOLD = 20
SPLIT = "2015-01-01"
POOL = 20          # "모멘텀 상위 POOL 중에서 다시 고른다" 규칙들의 1차 풀


# ---------- 피처 ----------
def features(refresh=False):
    d = load_ohlcv(refresh)
    idx = d["close"][INDEX].dropna()
    close = d["close"].drop(columns=[INDEX])
    vol = d["volume"].drop(columns=[INDEX])
    close = close.loc[:, close.notna().sum() > 300]

    f = {}
    f["close"] = close
    f["ret1"] = close.pct_change()
    f["mom"] = (close / close.shift(20) - 1) * 100
    f["mom60"] = (close / close.shift(60) - 1) * 100
    f["ret5"] = (close / close.shift(5) - 1) * 100
    f["vol20"] = f["ret1"].rolling(20).std() * 100
    f["slope"] = (f["mom"] - f["mom"].shift(5)) / 5
    f["nh52"] = close / close.rolling(252).max()
    f["vratio"] = vol / vol.rolling(20).mean()
    # 정보 이산성 (Da·Gurun·Warachka): 같은 수익률이어도 잔잔하게 오른 쪽이 낫다는 가설
    # ID = sign(모멘텀) × (음봉비율 − 양봉비율). 낮을수록 '부드러운' 상승
    pos = (f["ret1"] > 0).rolling(20).mean()
    neg = (f["ret1"] < 0).rolling(20).mean()
    f["idisc"] = np.sign(f["mom"]) * (neg - pos)
    # 변동성 조정 모멘텀
    f["mom_adj"] = f["mom"] / f["vol20"].replace(0, np.nan)

    # ★ 당시 구성종목만 후보로 둔다 (2015~). 선택에 쓰는 지표에만 씌우고 close·ret1 은
    #   그대로 둔다 — 이미 보유한 종목의 수익·청산은 지수에서 빠져도 계속돼야 한다.
    #   PIT=0 이면 전부 True 라 옛 방식이 그대로 재현된다.
    import pit
    _m = pd.DataFrame(pit.mask(close.index, list(close.columns)),
                      index=close.index, columns=close.columns)
    for _k in f:
        if _k not in ("close", "ret1"):
            f[_k] = f[_k].where(_m)

    # 시장 국면 (backtest_volume.py 와 같은 정의)
    dv = (close * vol).sum(axis=1, min_count=1)
    vr = dv.rolling(20).mean() / dv.rolling(250).mean()
    dir60 = (idx / idx.shift(60) - 1) * 100
    lab = pd.DataFrame({"vr": vr, "dir60": dir60}).dropna()
    lab["방향"] = np.where(lab.dir60 > DIR_BAND, "상승",
                         np.where(lab.dir60 < -DIR_BAND, "하락", "횡보"))
    lab["거래량"] = np.where(lab.vr > VR_HI, "실림", "마름")
    lab["국면"] = lab["거래량"] + "·" + lab["방향"]
    return f, lab


def top_n(score, n):
    return score.rank(axis=1, ascending=False, method="first") <= n


def refine(f, key, n=TOP, pool=POOL, ascending=False):
    """모멘텀 상위 pool 로 좁힌 뒤 key 기준 상위(또는 하위) n 개."""
    inpool = top_n(f["mom"], pool)
    s = f[key].where(inpool)
    return s.rank(axis=1, ascending=ascending, method="first") <= n


# ---------- 규칙 후보 ----------
def rules(f, lab):
    R = {}
    R["A 기준선: 20일 모멘텀 상위 5"] = top_n(f["mom"], TOP)
    R["B 변동성 조정 모멘텀 상위 5"] = top_n(f["mom_adj"], TOP)
    R["C 52주 신고가 근접 상위 5"] = top_n(f["nh52"], TOP)
    R["D 풀20 → 부드러운 모멘텀 5"] = refine(f, "idisc", ascending=True)
    R["E 풀20 → 기울기 급락 5 (눌림목)"] = refine(f, "slope", ascending=True)
    R["F 풀20 → |기울기| 큰 5 (평탄 회피)"] = refine({**f, "abs_slope": f["slope"].abs()}, "abs_slope")
    R["G 풀20 → 5일 수익률 낮은 5"] = refine(f, "ret5", ascending=True)
    R["H 풀20 → 거래량비 높은 5"] = refine(f, "vratio")
    R["I 풀20 → 저변동성 5"] = refine(f, "vol20", ascending=True)
    R["J 기준선 + 마름·하락 국면 미보유"] = _regime_off(R["A 기준선: 20일 모멘텀 상위 5"], lab, ["마름·하락"])
    R["K 기준선 + 하락장 전체 미보유"] = _regime_off(R["A 기준선: 20일 모멘텀 상위 5"], lab,
                                            ["마름·하락", "실림·하락"])
    return R


def _regime_off(sel, lab, off):
    bad = lab.index[lab["국면"].isin(off)]
    out = sel.copy()
    out.loc[out.index.isin(bad)] = False
    return out


def combo(f, lab, base):
    """살아남은 규칙을 합친 안. base 는 rules() 결과."""
    return {}


# ---------- 포트폴리오 ----------
def curve(sel, ret1, hold=HOLD, top=TOP):
    """겹치는 트랜치 포트폴리오의 일별 수익률. 안 고른 트랜치는 현금(0)."""
    s = sel.reindex(ret1.index).fillna(False).astype(float)
    holding = s.rolling(hold).sum().shift(1)          # 그 날 이 종목을 쥐고 있는 트랜치 수
    w = holding / (hold * top)                        # 나머지는 현금
    r = (w * ret1.reindex(columns=w.columns)).sum(axis=1, min_count=1)
    return r.dropna()


def metrics(r, lab=None):
    if len(r) < 250:
        return None
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = (eq.iloc[-1] ** (1 / yrs) - 1) * 100
    vol = r.std() * np.sqrt(252) * 100
    mdd = ((eq / eq.cummax() - 1).min()) * 100
    out = {"cagr": cagr, "vol": vol, "sharpe": cagr / vol if vol else 0, "mdd": mdd}
    if lab is not None:
        d = lab["방향"].reindex(r.index)
        for k in ("상승", "횡보", "하락"):
            x = r[d == k]
            out[k] = ((1 + x).prod() ** (252 / max(len(x), 1)) - 1) * 100 if len(x) > 30 else np.nan
    return out


def report(R, f, lab):
    ret1 = f["ret1"]
    curves = {n: curve(s, ret1) for n, s in R.items()}
    base_name = "A 기준선: 20일 모멘텀 상위 5"
    print(f"\n{'규칙':36}" + "".join(f"{h:>9}" for h in
          ("탐CAGR", "탐Δ", "탐MDD", "탐Sh", "|", "검CAGR", "검Δ", "검MDD", "검Sh")))
    base = {}
    for label, lo, hi in (("탐", "1999-01-01", SPLIT), ("검", SPLIT, "2030-01-01")):
        base[label] = metrics(curves[base_name][lo:hi])
    for n, c in curves.items():
        cells = ""
        ok = 0
        for label, lo, hi in (("탐", "1999-01-01", SPLIT), ("검", SPLIT, "2030-01-01")):
            m = metrics(c[lo:hi])
            if m is None:
                cells += f"{'-':>36}"; continue
            dl = m["cagr"] - base[label]["cagr"]
            ok += dl > 0
            cells += f"{m['cagr']:+8.1f}%{dl:+9.1f}{m['mdd']:8.1f}%{m['sharpe']:9.2f}"
            if label == "탐":
                cells += f"{'|':>9}"
        mark = " ★" if ok == 2 and n != base_name else ""
        print(f"{n:36}{cells}{mark}")
    print("\n  ★ = 탐색·검증 두 구간 모두에서 기준선보다 CAGR 이 높은 규칙")
    print("  Δ = 기준선 대비 CAGR 차이(%p).  Sh = CAGR/변동성 (무위험수익률 미차감)")
    print("  ★ 생존 편향으로 절대 CAGR 은 크게 부풀려져 있다. Δ 로만 비교할 것.")

    print(f"\n### 국면별 연율 수익률 (전 구간)")
    print(f"{'규칙':36}{'전체':>9}{'상승장':>9}{'횡보장':>9}{'하락장':>9}{'MDD':>9}")
    for n, c in curves.items():
        m = metrics(c, lab)
        if m:
            print(f"{n:36}{m['cagr']:+8.1f}%{m['상승']:+8.1f}%{m['횡보']:+8.1f}%"
                  f"{m['하락']:+8.1f}%{m['mdd']:8.1f}%")
    return curves


def yearly(curves, names):
    print("\n### 연도별 수익률 (%)")
    yr = pd.DataFrame({n: (1 + curves[n]).resample("YE").prod() - 1 for n in names}) * 100
    yr.index = yr.index.year
    print(yr.round(1).to_string())


def main():
    f, lab = features("--refresh" in sys.argv)
    print(f"종목 {f['close'].shape[1]}개, 기간 {f['close'].index.min().date()} ~ "
          f"{f['close'].index.max().date()}, 탐색/검증 경계 {SPLIT}")
    R = rules(f, lab)
    curves = report(R, f, lab)
    if "--curve" in sys.argv:
        yearly(curves, list(R)[:4])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
