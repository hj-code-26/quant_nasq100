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

신호 (운영과 동일 정의, 미래 참조 없음)
  · 지수 60일 수익률 < BEAR_RET60 → 하락. 운영은 QQQ, 여기는 ^NDX
    (autotrade.py 주석: 1999~2026 상관 0.9996, -3% 판정 일치율 99.6%)
  · t-1 종가까지의 신호를 t 에 집행한다 (shift(1)). 같은 봉 신호·집행 금지

★ 한계 (결론을 읽기 전에)
  · 생존 편향: 유니버스가 **오늘의** 나스닥 100. 절대 CAGR 은 크게 부풀려져 있고
    하락장에서 특히 심하다. 기준선 대비 Δ 로만 읽을 것
  · 현금 수익률 0% 로 둔다. 실제로는 단기 국채 금리가 붙으므로 방어안에 **불리한** 가정이다
  · 파라미터 조합 12개(상한 4 × 복귀지연 3)를 테스트했다. 다중검정 보정에 쓸 것

사용:  python backtest_bear_exposure.py              (그리드 + 구간별)
       python backtest_bear_exposure.py --cost2      (비용 2배 시나리오)
       python backtest_bear_exposure.py --signal     (신호 품질만)
"""
import sys

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
def bear_signal(idx, ret60=BEAR_RET60, off_days=0):
    """하락 국면 여부. t 종가까지만 쓴다 (집행 시점 shift 는 호출부에서)."""
    raw = ((idx / idx.shift(60) - 1) * 100 < ret60)
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


def signal_quality(idx):
    """휩소·적중률·지연. 사전 신호를 매매에 쓰기 전에 확인해야 하는 것들."""
    # 앞으로 1년(252거래일) 안의 최저가 대비 낙폭 — 평가 전용, 매매에 쓰지 않는다
    fwd_min = idx.shift(-1)[::-1].rolling(252, min_periods=1).min()[::-1]
    fwd_dd = (fwd_min / idx - 1) * 100
    print(f"\n### 하락 신호 품질 — 지수 60일 수익률 < {BEAR_RET60}% ({INDEX})")
    print(f"{'복귀지연':>8}{'ON일수':>9}{'ON비율':>8}{'전환횟수':>9}{'평균ON길이':>11}"
          f"{'ON후1년내-20%':>14}{'ON중지수':>11}")
    for d in DELAYS:
        on = bear_signal(idx, off_days=d).dropna()
        flips = int((on != on.shift(1)).sum())
        starts = on & ~on.shift(1, fill_value=False)
        hit = fwd_dd.reindex(on.index)[starts].dropna()
        runs = on.ne(on.shift()).cumsum()[on]
        avg = runs.value_counts().mean() if len(runs) else float("nan")
        r = idx.reindex(on.index).pct_change()
        held = on.shift(1, fill_value=False)
        on_ret = ((1 + r[held].fillna(0)).prod() - 1) * 100
        pct = (hit <= -20).mean() * 100 if len(hit) else float("nan")
        print(f"{d:>8}{int(on.sum()):>9,}{on.mean() * 100:>7.1f}%{flips:>9}{avg:>11.0f}"
              f"{pct:>13.0f}%{on_ret:>10.0f}%")
    print("  전환횟수 = ON↔OFF 가 바뀐 횟수 (휩소 대리 지표). 'ON후1년내-20%' = 신호가 켜진 뒤")
    print("  1년 안에 지수가 그 시점 대비 -20% 이상 빠진 비율 — 낮으면 헛경보가 많다는 뜻이다.")
    print("  'ON중지수' = 신호 ON 인 날만 이어붙인 지수 누적수익률. 크게 음수여야 신호가 쓸모 있다.")


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


def split(r):
    return metrics(r["1999-01-01":SPLIT]), metrics(r[SPLIT:"2030-01-01"])


def check(rank, ret, dates):
    """불변식: cap 없음 + 비용 0 이면 기존 엔진(backtest_slots)과 **완전히** 같아야 한다.

    이게 깨지면 오버레이와 기준선을 다른 저울로 잰 것이므로 비교 결과 전체가 무효다.
    """
    from backtest_slots import simulate as sim_old
    a, _ = sim_old(SLOTS, HOLD, rank, ret, dates)
    b, _, _ = simulate(SLOTS, HOLD, rank, ret, dates, None, 0.0)
    diff = (a - b).abs().max()
    assert len(a) == len(b) and diff < 1e-12, f"기준선이 기존 엔진과 다르다 (최대 괴리 {diff:.2e})"
    print(f"[불변식 OK] cap=None·비용0 → backtest_slots.simulate 와 일치 (최대 괴리 {diff:.1e})")


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

    check(rank, ret, dates)
    signal_quality(idx)

    base, base_turn, _ = simulate(SLOTS, HOLD, rank, ret, dates, None, cost)
    ba, bb = split(base)
    results = {"기준선 (상한 없음)": (base, base_turn)}

    print("\n### 그리드 — 하락 국면 주식 노출 상한 × 복귀 지연")
    print(f"{'설정':22}{'탐CAGR':>9}{'탐Δ':>8}{'탐MDD':>8}{'탐Cal':>7}  |"
          f"{'검CAGR':>9}{'검Δ':>8}{'검MDD':>8}{'검Cal':>7}{'회전율':>8}")
    print(f"{'기준선 (상한 없음)':18}{ba['cagr']:+8.1f}%{0.0:+8.1f}{ba['mdd']:7.1f}%"
          f"{calmar(ba):7.2f}  |{bb['cagr']:+8.1f}%{0.0:+8.1f}{bb['mdd']:7.1f}%"
          f"{calmar(bb):7.2f}{base_turn:7.1f}x")
    for d in DELAYS:
        on = bear_signal(idx, off_days=d).reindex(dates).fillna(False)
        on = on.shift(1, fill_value=False).to_numpy()     # t-1 신호를 t 에 집행
        for c in CAPS:
            arr = np.where(on, c / 100.0, 1.0)
            r, turn, _ = simulate(SLOTS, HOLD, rank, ret, dates, arr, cost)
            a, b = split(r)
            name = f"상한 {c:>3}% · 지연 {d}일"
            results[name] = (r, turn)
            print(f"{name:20}{a['cagr']:+8.1f}%{a['cagr'] - ba['cagr']:+8.1f}{a['mdd']:7.1f}%"
                  f"{calmar(a):7.2f}  |{b['cagr']:+8.1f}%{b['cagr'] - bb['cagr']:+8.1f}"
                  f"{b['mdd']:7.1f}%{calmar(b):7.2f}{turn:7.1f}x")
    print("  Δ = 기준선 대비 CAGR 차이(%p).  Cal = Calmar (CAGR/|MDD|).  회전율 = 연환산 매매대금/평잔")

    print("\n### 이름 붙은 구간별 수익률 (%) — 하락 구간의 방어와 상승 구간의 보험료")
    keys = ["기준선 (상한 없음)"] + [k for k in results
                                if k.startswith(("상한  50", "상한  30", "상한   0"))
                                and "지연 3" in k]
    hdr = [k.replace("기준선 (상한 없음)", "기준선").replace("상한 ", "")
            .replace(" · 지연 ", "/") for k in keys]
    print(f"{'구간':32}" + "".join(f"{h:>13}" for h in hdr))
    for name, a, b, kind in PERIODS:
        cells = []
        for k in keys:
            x = results[k][0][a:b]
            cells.append(f"{((1 + x).prod() - 1) * 100:+12.0f}%" if len(x) > 5 else f"{'-':>13}")
        print(f"{name[:24]:26}[{kind}]" + "".join(cells))

    # 프롬프트 2단계 A 필수 분석 — 방어 전략이 수익을 깎는 경로를 직접 보여준다
    print("\n### 기준선 상승 상위 거래일 중 신호가 ON 이던 비율")
    print("  (ON 이면 노출이 잘려 그 날 상승을 통째로 혹은 일부 놓친다)")
    print(f"{'복귀지연':>8}{'상위10일':>10}{'상위20일':>10}{'상위50일':>10}{'전체ON비율':>12}")
    for d in DELAYS:
        on = bear_signal(idx, off_days=d).reindex(dates).fillna(False)
        on = on.shift(1, fill_value=False).reindex(base.index).fillna(False)
        cells = [on[base.nlargest(n).index].mean() * 100 for n in (10, 20, 50)]
        print(f"{d:>8}" + "".join(f"{c:>9.0f}%" for c in cells) + f"{on.mean() * 100:>11.0f}%")
    print("  전체 ON 비율보다 상위일 비율이 높으면, 이 신호는 반등일을 골라서 놓치고 있다는 뜻이다.")

    print("\n### 국면별 연율 수익률 + 전체 (전 구간)")
    print(f"{'설정':22}{'상승장':>9}{'횡보장':>9}{'하락장':>9}{'전체CAGR':>10}{'전체MDD':>9}{'Sharpe':>8}")
    for k, (r, _) in results.items():
        m = metrics(r, lab)
        print(f"{k:20}{m['상승']:+8.1f}%{m['횡보']:+8.1f}%{m['하락']:+8.1f}%"
              f"{m['cagr']:+9.1f}%{m['mdd']:8.1f}%{m['sharpe']:8.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
