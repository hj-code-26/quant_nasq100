"""A 전략(레짐 기반 비중 축소) 검증 — 하락 국면에 주식 노출 상한을 걸면 남는 게 있나.

왜 이걸 재는가.
  운영 코드는 국면을 판정하면서도(autotrade.py:653) 그 결과를 **비중에 연결하지 않는다.**
  momentum_tier() 가 하락 국면에서 무조건 1.0 을 반환해 '무엇을 사는지'만 바뀌고
  '얼마나 사는지'는 그대로다. 유일한 축소 장치인 변동성 타겟은 60일 실현 변동성 기반이라
  급락을 후행한다. 여기서는 국면 신호를 노출 상한에 직접 물렸을 때의 손익을 잰다.

구조 (backtest_slots.simulate 를 그대로 확장)
  · 슬롯 N 개. 빈 슬롯이 생기면 보유하지 않은 종목 중 모멘텀 1위부터 채운다
  · 청산 = 만기 hold 거래일. 리밸런스 없음 (운영 코드가 그렇다)
  · [추가] 매일 주식 노출을 cap 이하로 유지한다. 초과분은 **큰 종목부터** 판다
    (운영 validate_orders 의 노출 축소 블록과 같은 순서)
  · [추가] 매매 금액에 편도 비용을 물린다. 오버레이는 회전율을 늘리므로
    비용 0 은 채택 쪽으로 편향된다 — 반드시 넣어야 한다

신호 — 2단 관문 (미래 참조 없음)
  1차: 후보 4개를 **사전 등록**하고, 포트폴리오 백테스트 전에 신호 품질로 먼저 거른다.
       "ON 인 날만 이어붙인 지수 수익률이 SIG_GATE 이하" 를 통과해야 한다.
       운영이 지금 쓰는 60일<-3% 는 이 값이 **+41%** 라 여기서 탈락한다 —
       방어할 게 없는 구간에 켜진다는 뜻이고, 실제로 12조합 전부 REJECT 였다(커밋 aa037bf).
  2차: 통과한 신호만 상한 그리드에 올린다. 집행은 t-1 종가 신호를 t 에 (shift(1)).
       같은 봉에서 신호와 체결을 동시에 하지 않는다.

★ 한계 (결론을 읽기 전에)
  · 생존 편향: 유니버스가 **오늘의** 나스닥 100. 절대 CAGR 은 크게 부풀려져 있고
    하락장에서 특히 심하다. 기준선 대비 Δ 로만 읽을 것
  · 현금 수익률 0% 로 둔다. 실제로는 단기 국채 금리가 붙으므로 방어안에 **불리한** 가정이다
  · 폴드가 탐색/검증 2개뿐이다. 프롬프트가 요구하는 롤링 walk-forward 는 아직 안 했다
  · 다중검정 N 은 이번 16 + 이전 12 = 28 로 잡았다. 신호 선별 12회는 N 에 안 들어가
    있으므로 DSR 은 **낙관 쪽으로 치우친 값**이다 (그래도 기준 미달이다)

사용:  python backtest_bear_exposure.py              (관문 + 그리드 + 구간별 + 강건성)
       python backtest_bear_exposure.py --cost2      (비용 2배 시나리오)
       python backtest_bear_exposure.py --signal     (1차 관문만)

결론: 252일고점 -15% / 복귀지연 3일 / 상한 70% → **ADOPT_LIMITED**.
      핵심 기준(방어 효과·보험료·일관성)은 통과, 통계적 유의성(DSR 0.086)과
      파라미터 강건성은 미달. autotrade.py 에 기본값 OFF 로 넣었다 (BEAR_EXPOSURE_PCT).
"""
import pathlib
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from backtest_regimes import PERIODS
from backtest_rules import SPLIT, features, metrics
from market_data import INDEX, load_ohlcv

SLOTS = 10              # 운영 MAX_POSITIONS
HOLD = 20               # 운영 MAX_HOLD_DAYS
BEAR_RET60 = -3.0       # 운영 BEAR_RET60_PCT
CAPS = (70, 50, 30, 0)  # 하락 국면 주식 노출 상한 (%). 100% = 기준선이라 따로 안 돈다
DELAYS = (0, 3, 5)      # 복귀 지연 — 신호 OFF 가 N일 연속돼야 상한 해제
COST_BP = 10.0          # 매매 대금당 편도 비용 (수수료+슬리피지, bp)


# ---------- 신호 ----------
# 사전 등록한 후보 4개. **결과를 보고 고르는 게 아니라** 아래 기준으로 먼저 거른다.
# 매매에 쓰려면 최소한 "ON 인 날만 모았을 때 지수가 내려가 있어야" 한다 (SIG_GATE).
# 이 관문을 통과 못 한 신호는 포트폴리오 백테스트를 돌릴 가치조차 없다.
SIG_GATE = -10.0        # ON 중 지수 누적수익률(%) 상한. 이보다 높으면 탈락
EULER = 0.5772156649015329      # DSR 의 오일러-마스케로니 상수
PRIOR_TRIALS = 12       # 이전 커밋(aa037bf)에서 이미 돌린 현행 신호 12조합. N 에 합산한다


def sig_ret60(idx, th=-3.0):
    """[현행] 지수 60일 수익률 < -3%. autotrade.py 가 지금 쓰는 정의."""
    return (idx / idx.shift(60) - 1) * 100 < th


def sig_sma200(idx, days=5):
    """200일선을 N일 연속 하회. 지속 조건이 휩소를 직접 겨냥한다."""
    below = idx < idx.rolling(200).mean()
    return below.rolling(days).sum() >= days


def sig_dd252(idx, th=15.0):
    """252일 고점 대비 -15% 이상. 닷컴·금융위기는 잡고 얕은 조정은 거른다."""
    return (idx / idx.rolling(252).max() - 1) * 100 <= -th


def sig_rvol(idx, q=0.8, win=20, look=252):
    """20일 실현변동성이 직전 1년 분포의 80분위 이상. 급락 국면에만 반응한다."""
    rv = idx.pct_change().rolling(win).std()
    return rv >= rv.rolling(look).quantile(q)


SIGNALS = {
    "현행 60일<-3%": sig_ret60,
    "200일선 5일연속하회": sig_sma200,
    "252일고점 -15%": sig_dd252,
    "실현변동성 80분위": sig_rvol,
}


def hold_off(raw, off_days):
    """복귀 지연 — 신호 OFF 가 off_days 연속돼야 해제한다.

    "마지막 off_days 일 중 하나라도 ON 이면 유지"와 같은 말이다 (OFF 가 off_days 연속이면
    그 창에 ON 이 하나도 없다). 그래서 rolling max 한 줄이면 된다 — 상태 루프가 필요 없고,
    운영 코드에는 `.tail(off_days).any()` 로 그대로 옮겨진다. 동치성은 check() 에서 검증한다.
    """
    return raw.rolling(max(off_days, 1), min_periods=1).max().astype(bool)


def _hold_off_loop(raw, off_days):
    """hold_off 의 참조 구현 (상태 루프). check() 의 동치 비교에만 쓴다."""
    if off_days <= 0:
        return raw
    on, cur, noff = np.zeros(len(raw), bool), False, 0
    for i, v in enumerate(raw.to_numpy()):
        if v:
            cur, noff = True, 0
        elif cur:
            noff += 1
            if noff >= off_days:
                cur = False
        on[i] = cur
    return pd.Series(on, index=raw.index)


def bear_signal(idx, name, off_days=0):
    """하락 국면 여부. t 종가까지만 쓴다 (집행 시점 shift 는 호출부에서)."""
    return hold_off(SIGNALS[name](idx).fillna(False), off_days)


def signal_quality(idx):
    """휩소·적중률·지연. 사전 신호를 매매에 쓰기 전에 확인해야 하는 것들.

    반환: 관문(SIG_GATE)을 통과한 (신호명, 지연) 목록.
    """
    # 앞으로 1년(252거래일) 안의 최저가 대비 낙폭 — 평가 전용, 매매에 쓰지 않는다
    fwd_min = idx.shift(-1)[::-1].rolling(252, min_periods=1).min()[::-1]
    fwd_dd = (fwd_min / idx - 1) * 100
    print(f"\n### 하락 신호 품질 ({INDEX}) — 포트폴리오 백테스트 이전의 1차 관문")
    print(f"{'신호':22}{'지연':>5}{'ON일수':>8}{'ON비율':>8}{'전환':>6}{'평균ON':>7}"
          f"{'ON후1년-20%':>12}{'ON중지수':>10}{'':>4}")
    passed = []
    for name in SIGNALS:
        for d in DELAYS:
            on = bear_signal(idx, name, d).dropna()
            flips = int((on != on.shift(1)).sum())
            starts = on & ~on.shift(1, fill_value=False)
            hit = fwd_dd.reindex(on.index)[starts].dropna()
            runs = on.ne(on.shift()).cumsum()[on]
            avg = runs.value_counts().mean() if len(runs) else float("nan")
            r = idx.reindex(on.index).pct_change()
            on_ret = ((1 + r[on.shift(1, fill_value=False)].fillna(0)).prod() - 1) * 100
            pct = (hit <= -20).mean() * 100 if len(hit) else float("nan")
            ok = on_ret <= SIG_GATE
            if ok:
                passed.append((name, d))
            print(f"{name:22}{d:>5}{int(on.sum()):>8,}{on.mean() * 100:>7.1f}%{flips:>6}"
                  f"{avg:>7.0f}{pct:>11.0f}%{on_ret:>9.0f}%{'  통과' if ok else '  탈락':>6}")
    print(f"  'ON중지수' = 신호 ON 인 날만 이어붙인 지수 누적수익률. {SIG_GATE:+.0f}% 이하라야 통과시킨다 —")
    print("  이게 양수면 '방어할 게 없는 구간에 켜졌다'는 뜻이라 노출을 줄일수록 손해만 난다.")
    print("  전환 = ON↔OFF 가 바뀐 횟수(휩소 대리). 'ON후1년-20%' = 켜진 뒤 1년 내 지수 -20% 도달 비율.")
    return passed


# ---------- 포트폴리오 ----------
def simulate(N, hold, rank, ret, dates, cap=None, cost_bp=COST_BP):
    """슬롯 N개·만기 hold. cap[t] = 그 날 허용 주식 비중(0~1). None 이면 기준선.

    반환: (일별수익률, 연환산 회전율, 최대단일비중)
    """
    n_days, n_sym = ret.shape
    if cap is None:
        cap = np.ones(n_days)
    cost = cost_bp / 1e4
    pos = np.zeros(n_sym)
    day = np.full(n_sym, -1)
    cash, maxw, traded = 1.0, 0.0, 0.0
    eq = np.empty(n_days)
    for t in range(n_days):
        held = pos > 0
        if held.any():                                    # 1) 평가액 갱신
            pos[held] *= 1 + np.nan_to_num(ret[t][held])
            gone = held & (t - day >= hold)               # 2) 만기 청산
            if gone.any():
                amt = pos[gone].sum()
                traded += amt
                cash += amt * (1 - cost)
                pos[gone] = 0.0
                day[gone] = -1
        v = cash + pos.sum()
        # 3) 노출 상한 초과분 축소 — 큰 종목부터 (운영 validate_orders 와 같은 순서)
        excess = pos.sum() - v * cap[t]
        if excess > 1e-12:
            for c in np.argsort(-pos):
                if excess <= 1e-12 or pos[c] <= 0:
                    break
                cut = min(excess, pos[c])
                pos[c] -= cut
                traded += cut
                cash += cut * (1 - cost)
                excess -= cut
                if pos[c] <= 1e-12:
                    pos[c] = 0.0
                    day[c] = -1
        # 4) 빈 슬롯 채우기 — 단 상한이 허용하는 만큼만
        held = pos > 0
        free = N - int(held.sum())
        budget = min(cash, v * cap[t] - pos.sum())
        if free > 0 and budget > 1e-12:
            rk = rank[t]
            order = np.argsort(rk, kind="stable")
            picks = [c for c in order[:N + n_sym // 4]
                     if np.isfinite(rk[c]) and not held[c]][:free]
            if picks:
                per = budget / len(picks)
                for c in picks:
                    pos[c] = per * (1 - cost)
                    day[c] = t
                traded += per * len(picks)
                cash -= per * len(picks)
        v = cash + pos.sum()
        eq[t] = v
        if v > 0 and pos.max() / v > maxw:
            maxw = pos.max() / v
    curve = pd.Series(eq, index=dates)
    turn = traded / max(curve.mean(), 1e-12) / (n_days / 252)
    return curve.pct_change().dropna(), turn, maxw


def calmar(m):
    return m["cagr"] / abs(m["mdd"]) if m and m["mdd"] else float("nan")


# ---------- 강건성 (프롬프트 4-3) ----------
BOOT = 1000
BLOCK = 63              # 분기(63거래일). 하락 국면이 수개월 단위로 이어지므로 그 지속성을 보존한다
SEED = 20260911


def _stats(x):
    """일별 수익률 배열 → (CAGR%, MDD%, Calmar)."""
    eq = np.cumprod(1 + x)
    cagr = (eq[-1] ** (252 / len(x)) - 1) * 100
    mdd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    return cagr, mdd, cagr / abs(mdd) if mdd else np.nan


def block_bootstrap(base, ov, n_boot=BOOT, block=BLOCK, seed=SEED):
    """순환 블록 부트스트랩. 기준선과 오버레이를 **같은 블록**으로 뽑아 동시성을 보존한다.

    반환: {지표: (하위2.5%, 중앙값, 상위97.5%, 개선이 양수인 비율)} — 개선폭의 신뢰구간.
    """
    a, b = base.align(ov, join="inner")
    A, B = a.to_numpy(float), b.to_numpy(float)
    n = len(A)
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    diffs = np.empty((n_boot, 3))
    for i in range(n_boot):
        starts = rng.integers(0, n, nb)
        ix = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        diffs[i] = np.subtract(_stats(B[ix]), _stats(A[ix]))
    out = {}
    for j, k in enumerate(("ΔCAGR", "ΔMDD", "ΔCalmar")):
        d = diffs[:, j]
        out[k] = (*np.percentile(d, [2.5, 50, 97.5]), (d > 0).mean() * 100)
    return out


def dsr(diff, n_trials, var_sr):
    """Deflated Sharpe Ratio (Bailey·López de Prado 2014).

    기준선 대비 **초과수익 계열**에 적용한다 — "보정 후에도 개선이 유의한가"가 질문이므로.
    수익률의 왜도·첨도와 시도 횟수 N 을 함께 반영한다. 정규분포는 stdlib 로 충분하다.
    """
    x = diff.to_numpy(float)
    x = x[np.isfinite(x)]
    T = len(x)
    sd = x.std(ddof=1)
    if T < 100 or sd == 0:
        return float("nan"), float("nan")
    sr = x.mean() / sd
    z = NormalDist()
    g3 = float(((x - x.mean()) ** 3).mean() / sd ** 3)
    g4 = float(((x - x.mean()) ** 4).mean() / sd ** 4)
    # 무작위 시도 N 번 중 최대 SR 의 기대값 — 이걸 넘어야 실력이다
    sr0 = np.sqrt(var_sr) * ((1 - EULER) * z.inv_cdf(1 - 1 / n_trials)
                             + EULER * z.inv_cdf(1 - 1 / (n_trials * np.e)))
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(z.cdf((sr - sr0) * np.sqrt(T - 1) / denom)), float(sr * np.sqrt(252))


def split(r):
    return metrics(r["1999-01-01":SPLIT]), metrics(r[SPLIT:"2030-01-01"])


def check(rank, ret, dates, idx):
    """불변식 3개. 하나라도 깨지면 아래 결과를 믿으면 안 된다."""
    # 1) cap 없음 + 비용 0 이면 기존 엔진(backtest_slots)과 **완전히** 같아야 한다.
    #    깨지면 오버레이와 기준선을 다른 저울로 잰 것이라 비교 전체가 무효다.
    from backtest_slots import simulate as sim_old
    a, _ = sim_old(SLOTS, HOLD, rank, ret, dates)
    b, _, _ = simulate(SLOTS, HOLD, rank, ret, dates, None, 0.0)
    diff = (a - b).abs().max()
    assert len(a) == len(b) and diff < 1e-12, f"기준선이 기존 엔진과 다르다 (최대 괴리 {diff:.2e})"

    # 2) rolling max 로 바꾼 hold_off 가 상태 루프와 같은가
    for name in SIGNALS:
        raw = SIGNALS[name](idx).fillna(False)
        for d in DELAYS:
            assert hold_off(raw, d).equals(_hold_off_loop(raw, d)), f"hold_off 불일치: {name}/{d}"

    # 3) ★ 미래 참조 없음 — 데이터를 t 에서 잘라도 t 의 신호가 그대로여야 한다.
    #    이게 깨지면 백테스트 성과 전체가 누설이다.
    for cut in (0.5, 0.7, 0.9):
        k = int(len(idx) * cut)
        for name in SIGNALS:
            for d in DELAYS:
                full = bear_signal(idx, name, d).iloc[:k]
                trunc = bear_signal(idx.iloc[:k], name, d)
                assert full.equals(trunc), f"미래 참조 발견: {name}/{d} (컷 {cut})"
    print(f"[불변식 OK] 기준선 일치(괴리 {diff:.1e}) · hold_off 동치 · "
          f"미래 참조 없음({len(SIGNALS)}신호 × {len(DELAYS)}지연 × 3컷)")


def main():
    cost = COST_BP * (2 if "--cost2" in sys.argv else 1)
    f, lab = features("--refresh" in sys.argv)
    close, mom, ret1 = f["close"], f["mom"], f["ret1"]
    idx = load_ohlcv()["close"][INDEX].dropna()

    if "--signal" in sys.argv:
        signal_quality(idx)
        return

    rank = np.array(mom.rank(axis=1, ascending=False, method="first").to_numpy(float), copy=True)
    rank[np.isnan(rank)] = np.inf
    ret = np.array(ret1.to_numpy(float), copy=True)
    dates = close.index

    print(f"종목 {close.shape[1]}개, 기간 {dates.min().date()} ~ {dates.max().date()}, "
          f"탐색/검증 경계 {SPLIT}")
    print(f"구조: 슬롯 {SLOTS}개 · 만기 {HOLD}일 · 편도 비용 {cost:.0f}bp"
          f"{'  [비용 2배 시나리오]' if '--cost2' in sys.argv else ''}")
    print(f"테스트한 파라미터 조합 수: {len(CAPS) * len(DELAYS)} "
          f"(상한 {len(CAPS)} × 복귀지연 {len(DELAYS)})")
    print("★ 생존 편향으로 절대 CAGR 은 부풀려져 있다. 기준선 대비 Δ 로만 읽을 것.")

    check(rank, ret, dates, idx)
    passed = signal_quality(idx)
    if not passed:
        print("\n### 결론: 1차 관문을 통과한 신호가 없다. 포트폴리오 백테스트를 돌리지 않는다.")
        return
    print(f"\n통과 {len(passed)}개 → 포트폴리오 조합 {len(passed) * len(CAPS)}개를 돌린다 "
          f"(신호 선별 {len(SIGNALS) * len(DELAYS)}회 포함해 다중검정 보정에 반영할 것)")

    base, base_turn, _ = simulate(SLOTS, HOLD, rank, ret, dates, None, cost)
    ba, bb = split(base)
    BASE = "기준선 (상한 없음)"
    results = {BASE: (base, base_turn)}

    print("\n### 그리드 — 통과 신호 × 노출 상한")
    print(f"{'설정':30}{'탐CAGR':>9}{'탐Δ':>8}{'탐MDD':>8}{'탐Cal':>7}  |"
          f"{'검CAGR':>9}{'검Δ':>8}{'검MDD':>8}{'검Cal':>7}{'회전율':>8}")
    print(f"{BASE:28}{ba['cagr']:+8.1f}%{0.0:+8.1f}{ba['mdd']:7.1f}%"
          f"{calmar(ba):7.2f}  |{bb['cagr']:+8.1f}%{0.0:+8.1f}{bb['mdd']:7.1f}%"
          f"{calmar(bb):7.2f}{base_turn:7.1f}x")
    for sig, d in passed:
        on = bear_signal(idx, sig, d).reindex(dates).fillna(False)
        on = on.shift(1, fill_value=False).to_numpy()     # t-1 신호를 t 에 집행
        for c in CAPS:
            arr = np.where(on, c / 100.0, 1.0)
            r, turn, _ = simulate(SLOTS, HOLD, rank, ret, dates, arr, cost)
            a, b = split(r)
            key = f"{sig}/{d}일 · 상한{c:>3}%"
            results[key] = (r, turn)
            print(f"{key:28}{a['cagr']:+8.1f}%{a['cagr'] - ba['cagr']:+8.1f}{a['mdd']:7.1f}%"
                  f"{calmar(a):7.2f}  |{b['cagr']:+8.1f}%{b['cagr'] - bb['cagr']:+8.1f}"
                  f"{b['mdd']:7.1f}%{calmar(b):7.2f}{turn:7.1f}x")
    print("  Δ = 기준선 대비 CAGR 차이(%p).  Cal = Calmar (CAGR/|MDD|).  회전율 = 연환산 매매대금/평잔")
    print("  채택 후보는 **탐색·검증 두 구간 모두** 에서 Calmar 가 기준선 이상이어야 한다.")

    # 두 구간 동시 우위 — 한쪽만 이기면 과적합으로 본다 (backtest_rules.py 와 같은 규칙)
    both = [k for k, (r, _) in results.items() if k != BASE
            and calmar(split(r)[0]) >= calmar(ba) and calmar(split(r)[1]) >= calmar(bb)]
    print(f"\n  탐색·검증 동시 Calmar 우위: {len(both)}개" + (f" — {', '.join(both)}" if both else ""))

    show = [BASE] + (both[:3] if both else sorted(
        (k for k in results if k != BASE),
        key=lambda k: -calmar(split(results[k][0])[1]))[:3])

    print("\n### 이름 붙은 구간별 수익률 (%) — 하락 구간의 방어와 상승 구간의 보험료")
    print(f"{'구간':32}" + "".join(f"{k.replace(BASE, '기준선')[-13:]:>15}" for k in show))
    for name, a, b, kind in PERIODS:
        cells = []
        for k in show:
            x = results[k][0][a:b]
            cells.append(f"{((1 + x).prod() - 1) * 100:+14.0f}%" if len(x) > 5 else f"{'-':>15}")
        print(f"{name[:24]:26}[{kind}]" + "".join(cells))

    # 프롬프트 2단계 A 필수 분석 — 방어 전략이 수익을 깎는 경로를 직접 보여준다
    print("\n### 기준선 상승 상위 거래일 중 신호가 ON 이던 비율")
    print("  (ON 이면 노출이 잘려 그 날 상승을 통째로 혹은 일부 놓친다)")
    print(f"{'신호':22}{'지연':>5}{'상위10일':>10}{'상위20일':>10}{'상위50일':>10}{'전체ON':>9}")
    for sig, d in passed:
        on = bear_signal(idx, sig, d).reindex(dates).fillna(False)
        on = on.shift(1, fill_value=False).reindex(base.index).fillna(False)
        cells = [on[base.nlargest(n).index].mean() * 100 for n in (10, 20, 50)]
        print(f"{sig:22}{d:>5}" + "".join(f"{c:>9.0f}%" for c in cells)
              + f"{on.mean() * 100:>8.0f}%")
    print("  전체 ON 비율보다 상위일 비율이 높으면, 이 신호는 반등일을 골라서 놓치고 있다는 뜻이다.")

    # ---- 강건성 (프롬프트 4-3) — 동시 우위 후보만 ----
    if both:
        n_trials = len(results) - 1 + PRIOR_TRIALS
        diffs = {k: (results[k][0] - base).dropna() for k in results if k != BASE}
        var_sr = float(np.var([d.mean() / d.std(ddof=1) for d in diffs.values()], ddof=1))
        print(f"\n### 강건성 — 블록 부트스트랩 {BOOT}회 (블록 {BLOCK}일) + 다중검정 보정")
        print(f"  시도 횟수 N = {n_trials} (이번 {len(results) - 1} + 이전 {PRIOR_TRIALS}), "
              f"시도간 SR 분산 {var_sr:.2e}, 시드 {SEED}")
        print(f"{'후보':30}{'지표':>9}{'하위2.5%':>10}{'중앙값':>9}{'상위97.5%':>11}{'개선>0':>8}")
        for k in both:
            ci = block_bootstrap(base, results[k][0])
            for i, (m, (lo, md, hi, pos)) in enumerate(ci.items()):
                print(f"{k if i == 0 else '':30}{m:>9}{lo:>+10.1f}{md:>+9.1f}{hi:>+11.1f}{pos:>7.0f}%")
            p, ir = dsr(diffs[k], n_trials, var_sr)
            print(f"{'':30}{'DSR':>9}{p:>10.3f}   (초과수익 연환산 SR {ir:+.2f}, 기준 0.95)")
        print("  ΔMDD 는 **양수가 개선**이다 (낙폭이 덜 깊다). 신뢰구간이 0 을 걸치면 유의하지 않다.")
        print("  DSR 은 기준선 대비 초과수익 계열에 적용했다 — 질문이 '개선이 유의한가'이기 때문이다.")
        print(f"  비용 2배 시나리오는 `python {pathlib.Path(__file__).name} --cost2` 로 따로 확인할 것.")

    print("\n### 국면별 연율 수익률 + 전체 (전 구간)")
    print(f"{'설정':30}{'상승장':>9}{'횡보장':>9}{'하락장':>9}{'전체CAGR':>10}{'전체MDD':>9}{'Sharpe':>8}")
    for k, (r, _) in results.items():
        m = metrics(r, lab)
        print(f"{k:28}{m['상승']:+8.1f}%{m['횡보']:+8.1f}%{m['하락']:+8.1f}%"
              f"{m['cagr']:+9.1f}%{m['mdd']:8.1f}%{m['sharpe']:8.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
