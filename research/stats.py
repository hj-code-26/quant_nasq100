"""§4.3 통계 모듈 — 짝지은 block bootstrap 과 walk-forward 실행 가능성 판정.

python research/stats.py  → research/out/bootstrap_summary.csv, walkforward_results.csv

원칙:
- A 와 B 를 **독립 resample 하지 않는다.** 같은 날짜의 일별 순수익을 짝으로 묶어
  동일한 블록 인덱스로 함께 뽑는다(joint circular block bootstrap).
- 겹치는 20일 보유 거래를 독립 표본으로 쓰지 않는다. 표본 단위는 **일별 포트폴리오 수익률**이다.
- bootstrap 은 이 편향된 표본 안의 불확실성만 다룬다. 생존 편향·미래 누수·선택 편향은
  전혀 제거하지 않는다.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from research.isolation import guard        # noqa: E402

guard()

import backtest as bt                       # noqa: E402
from research import engine as E            # noqa: E402
from research.ab_audit import RULES         # noqa: E402

OUT = pathlib.Path(__file__).with_name("out")
SEED = 20260910
N_BOOT = 2000
BLOCKS = (10, 20, 40)                        # 거래일. 20 = 보유기간과 같은 길이
PAIRS = (("A", "B"), ("B", "C"), ("A", "C"))
CONF = 0.95
UNITS = {"cum": "누적수익 배수 차 (0.1442 = +14.42%p)",
         "cagr": "연복리수익률 차 (0.05 = +5%p)",
         "sharpe": "연율화 Sharpe 차 (무차원)",
         "mdd": "최대낙폭 차 (-0.05 = 낙폭이 5%p 더 깊음)"}
INTERP = ("frac_resamples_gt0 = 재표집 2000회 중 차이가 양수로 나온 비율. "
          "'진짜 우위일 확률'도 '실전 성공 확률'도 p-value 도 아니다. "
          "CI 가 0 을 포함한다는 것은 '이 표본으로는 부호를 가릴 수 없다'는 뜻이며 "
          "동등성·무효과의 증거가 아니다 — 동등성을 주장하려면 사전에 정한 "
          "동등성 마진과 TOST 같은 절차가 따로 필요하다.")


def circular_blocks(rng, n, L, reps):
    """길이 n 을 덮는 블록 시작점 인덱스 (원형). 반환 shape = (reps, ceil(n/L)*L)[:n]"""
    k = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(reps, k))
    off = np.arange(L)
    idx = (starts[:, :, None] + off[None, None, :]).reshape(reps, k * L) % n
    return idx[:, :n]


def paired_bootstrap(ra, rb, L, reps=N_BOOT, seed=SEED):
    """같은 날짜 짝을 유지한 채 블록 재표집. 누적수익·CAGR·Sharpe·MDD 차이 분포."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(ra, float), np.asarray(rb, float)
    n = len(a)
    idx = circular_blocks(rng, n, L, reps)
    out = {}
    for tag, r in (("A", a), ("B", b)):
        s = r[idx]                                   # (reps, n)
        cum = np.prod(1 + s, axis=1) - 1
        yrs = n / 252
        out[tag] = dict(
            cum=cum, cagr=(1 + cum) ** (1 / yrs) - 1,
            sharpe=s.mean(axis=1) / s.std(axis=1, ddof=1) * np.sqrt(252),
            mdd=_mdd(np.cumprod(1 + s, axis=1)))
    return out


def _mdd(path):
    peak = np.maximum.accumulate(path, axis=1)
    return (path / peak - 1).min(axis=1)


def ci(x, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def run_bootstrap(navs):
    rows = []
    for rule in RULES:
        r = {m: navs[f"{rule}|{m}"].pct_change() for m in E.MODES}
        for x, y in PAIRS:
            pair = pd.concat({x: r[x], y: r[y]}, axis=1).dropna()
            for L in BLOCKS:
                bo = paired_bootstrap(pair[x], pair[y], L)
                for metric in ("cum", "cagr", "sharpe", "mdd"):
                    d = bo["B"][metric] - bo["A"][metric]
                    lo, hi = ci(d)
                    rows.append(dict(
                        rule=rule, comparison=f"{y}-{x}", metric=metric,
                        unit=UNITS[metric], block_len_days=L, n_days=len(pair),
                        n_boot=N_BOOT, seed=SEED, confidence_level=CONF,
                        ci_method="percentile (2.5 / 97.5)",
                        point_diff=round(float(_point(pair[y], pair[x], metric)), 4),
                        boot_mean_diff=round(float(d.mean()), 4),
                        ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                        frac_resamples_gt0=round(float((d > 0).mean()), 4),
                        ci_excludes_zero_95=bool(lo > 0 or hi < 0),
                        interpretation=INTERP,
                        note=("탐색 지표 — 블록 연결/경로 재구성에 민감"
                              if metric == "mdd" else "")))
    return pd.DataFrame(rows)


def _point(ry, rx, metric):
    def f(r):
        r = np.asarray(r, float)
        cum = np.prod(1 + r) - 1
        if metric == "cum":
            return cum
        if metric == "cagr":
            return (1 + cum) ** (252 / len(r)) - 1
        if metric == "sharpe":
            return r.mean() / r.std(ddof=1) * np.sqrt(252)
        return _mdd(np.cumprod(1 + r)[None, :])[0]
    return f(ry) - f(rx)


# ── walk-forward ──────────────────────────────────────────────────────────
WARMUP = 20        # ret20 워밍업
MIN_EVAL = 63      # 평가 구간 최소 거래일(약 3개월). 이보다 짧으면 평가 불가로 반환


def walk_forward(close, opens, ret20, rule, cfg, n_folds=4, embargo=20):
    """고정 규칙 평가라 내부 파라미터 학습이 없다 → **nested 선택 불필요**.
    (파라미터를 고르는 순간 내부 학습/외부 평가를 나눠야 한다. 여기서는 고르지 않는다.)

    폴드 경계: 각 평가 구간 앞에 워밍업 WARMUP 일을 붙여 지표를 인과적으로 계산한다.
    경계에 걸친 보유 포지션은 **사전 정의된 종료 처리**(평가 구간 시작 시 무포지션 출발)로
    통일한다 — carry-over 를 흉내내면 폴드 간 독립성이 더 나빠진다.
    라벨 중첩이 없으므로(라벨을 학습에 쓰지 않는다) purge 는 불필요하고,
    분할 방식상 필요한 embargo 만 평가 구간 사이에 둔다. '20일 보유'라는 이유로
    무조건 20일을 삭제하지 않는다 — embargo 는 폴드 경계에만 적용한다.
    """
    n = len(close)
    usable = n - WARMUP
    fold = usable // n_folds
    rows = []
    for k in range(n_folds):
        e0 = WARMUP + k * fold + (embargo if k else 0)
        e1 = WARMUP + (k + 1) * fold if k < n_folds - 1 else n
        if e1 - e0 < MIN_EVAL:
            rows.append(dict(rule=rule, fold=k, status="평가불가",
                             reason=f"평가 거래일 {e1-e0} < 최소 {MIN_EVAL}",
                             start=None, end=None))
            continue
        s0 = max(0, e0 - WARMUP)
        sub = close.iloc[s0:e1]
        res = {m: E.sim(sub, ret20.loc[sub.index], cfg,
                        opens=opens.loc[sub.index], mode=m) for m in E.MODES}
        row = dict(rule=rule, fold=k, status="평가",
                   reason="고정 규칙 — 내부 파라미터 선택 없음(nested 불필요)",
                   start=str(sub.index[0].date()), end=str(sub.index[-1].date()),
                   n_days=len(sub), embargo_days=embargo if k else 0)
        for m in E.MODES:
            row[f"ret_{m}_pct"] = round(res[m]["총수익"] * 100, 2)
            row[f"sharpe_{m}"] = round(res[m]["Sharpe"], 3)
        rows.append(row)
    return rows


def main():
    close, opens, ret20 = bt.daily_panel()
    navs = {}
    for rule, cfg in RULES.items():
        for m in E.MODES:
            navs[f"{rule}|{m}"] = E.sim(close, ret20, cfg, opens=opens, mode=m)["eq"]

    bs = run_bootstrap(navs)
    bs.to_csv(OUT / "bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    wf = []
    for rule, cfg in RULES.items():
        wf += walk_forward(close, opens, ret20, rule, cfg)
    wfd = pd.DataFrame(wf)
    wfd.to_csv(OUT / "walkforward_results.csv", index=False, encoding="utf-8-sig")

    print(f"### 짝지은 circular block bootstrap — 신뢰수준 {CONF:.0%}, percentile CI, "
          f"재표집 {N_BOOT}회, seed {SEED}")
    print(f"단위: {UNITS['cum']}")
    v = bs[(bs.metric == "cum")]
    print(v[["rule", "comparison", "block_len_days", "point_diff", "ci_lo", "ci_hi",
             "frac_resamples_gt0", "ci_excludes_zero_95"]].to_string(index=False))
    print(f"\n### Sharpe 차이 — 단위: {UNITS['sharpe']}")
    v = bs[(bs.metric == "sharpe")]
    print(v[["rule", "comparison", "block_len_days", "point_diff", "ci_lo", "ci_hi",
             "ci_excludes_zero_95"]].to_string(index=False))
    print("\n" + INTERP)
    print("\n### walk-forward")
    print(wfd.to_string(index=False))
    print("\n주의: 이 CI 는 **같은 편향 표본 안의** 불확실성이다. 생존 편향·PIT 부재·"
          "선택 편향은 전혀 반영하지 않는다. 2015~2026 이 아니라 3년 표본이며 "
          "이미 열람한 구간이므로 새로운 최종 OOS 가 아니다.")


if __name__ == "__main__":
    main()
