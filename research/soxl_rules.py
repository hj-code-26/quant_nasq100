"""SOXL 규칙 백테스트 — 무한매수법 · RSI/이동평균 스윙 · 20/80 리밸런싱 + 리스크 규칙 (사용자 지정, 2026-09-17).

    python research/soxl_rules.py          # → research/soxl_rules.md, research/out/soxl_rules.csv
    python research/soxl_rules.py --check  # 불변식만

자료 (research/data/, gitignore — 재현은 yfinance 재다운로드)
  실제   SOXL 조정 OHLC 2010-03-11 ~ 2026-09-16
  합성   ^SOX 가격지수 × 3 − 2×단기금리(^IRX) − 보수 0.9%/년, 1994-06 ~ (실제와 일수익 상관 0.9965)
         합성에는 시가·저가가 없다 → 양봉 = 종가 > 전일 종가, 200일선 터치 = 종가 ≤ 200일선

사전 등록 (결과 보기 전 커밋)
  공통   계좌 = SOXL + 현금. 현금은 단기금리를 받는다. 매매 비용 편도 0.12% (토스 0.1% + 슬리피지)
         신호는 t 종가로 계산 → **t+1 종가 체결** (무한매수법 LOC 는 예외: 가격 조건을 t−1 정보로 정하고 t 종가 체결)
  M1 무한매수법  시드 = 사이클 시작 시 계좌 × 시드비율, 1회분 u = 시드/40.
         포지션 없음 → 그날 종가에 u 매수로 사이클 시작.
         이후 매일 LOC 두 건: ① 0.5u, 한도 = 평단가 ② 0.5u, 한도 = 전일 종가. 종가 ≤ 한도면 종가 체결.
         종가 ≥ 평단 × 1.10 → 전량 매도, 사이클 리셋. 시드를 다 쓰면 익절까지 보유만.
  M2 스윙  진입 A: RSI14 ≤ 30 / 진입 B: 저가 ≤ 200일선 이고 양봉·종가 > 200일선.
         진입 신호 → 신호 시점 현금의 60% 를 4회(연속 4거래일) 균등 매수. 진행 중 신호는 무시.
         청산: RSI14 ≥ 70 → 보유 50% 매도(포지션당 1회) / 20일선 **하향 교차**(전일 ≥, 당일 <) → 잔량 전량.
  M3 리밸런싱  목표 SOXL 20%. 비중 < 10% → 20% 까지 매수, > 30% → 20% 까지 매도 (종가 기준 판정, 다음 날 체결)
  리스크 규칙
    R1 현금 30%: 모든 매수는 매수 후 SOXL 비중 ≤ 70% 로 잘린다
    R2 낙폭 관망: 낙폭 ≤ −30% 인 날은 **신규 매수 전부 중단** (매도는 계속). 두 정의를 따로 잰다
        R2p = SOXL 가격의 252일 고점 대비 / R2a = 계좌 평가액의 역대 고점 대비
    R3 보유 3개월: 포지션 나이(첫 매수부터, 전량 청산 시 리셋) 63거래일 도달 시 보유 50% 매도 (포지션당 1회)
  시도   M1·M2·M3 × {리스크 끔, R1+R2p+R3, R1+R2a+R3} = 9. M1 시드비율 = 리스크 끔 1.0 / 켬 0.7.
         누적 N = 83 + 9 = 92
  기준선 (시도 아님) SOXL 매수보유 · SOXX 매수보유(실제) / ^SOX 1배(합성) · QQQ 매수보유 · 현금(단기금리)
  판정   **ADOPT 후보** = 모두 충족
         ① 합성 1994~ MDD > −60% (닷컴·금융위기에서 살아남는가)
         ② 실제 2010~ Sharpe > SOXX 매수보유 Sharpe
         ③ 합성 1994~2009 · 실제 2010~2026 두 구간 CAGR > 현금(단기금리) CAGR
         하나라도 못 넘으면 REJECT. 결과를 보고 파라미터·해석을 바꾸지 않는다
  서술   CAGR·변동성·Sharpe·MDD·평균 SOXL 비중·연 매매 횟수·익절 사이클 수, 연도별(실제)
  불변식 (a) prefix: 자료 끝 250일을 잘라 돌려도 남은 구간 계좌 곡선이 같다 (미래 참조 없음) — 9 설정 assert
         (b) 매매 없음(현금만) 설정은 단기금리 복리와 같다 — assert
         (c) 모든 날 현금 ≥ −1e-9, 보유 수량 ≥ 0 — 시뮬 안에서 assert
"""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research.isolation import guard       # noqa: E402

guard()

DATA = ROOT / "research" / "data"
FEE = 0.0012
N_TRIALS = 92


def load():
    close = pickle.load(open(DATA / "soxl_close.pkl", "rb"))
    ohlc = pickle.load(open(DATA / "soxl_ohlc.pkl", "rb"))
    rf = (close["^IRX"].ffill() / 100 / 252)
    act = pd.DataFrame({k.lower(): ohlc[k]["SOXL"] for k in ("Open", "High", "Low", "Close")}).dropna()
    act["rf"] = rf.reindex(act.index).ffill().fillna(0)
    sox = close["^SOX"].dropna()
    r = 3 * sox.pct_change() - 2 * rf.reindex(sox.index).ffill().fillna(0) - 0.009 / 252
    c = (1 + r.fillna(0)).cumprod() * 100
    syn = pd.DataFrame({"open": np.nan, "high": np.nan, "low": np.nan, "close": c})
    syn["rf"] = rf.reindex(syn.index).ffill().fillna(0)
    syn = syn.loc["1994-06-01":]
    bench = {"SOXX": close["SOXX"].dropna(), "QQQ": close["QQQ"].dropna(), "SOX": sox}
    return act, syn, bench


def indicators(px):
    c = px["close"]
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out = pd.DataFrame(index=px.index)
    out["rsi"] = 100 - 100 / (1 + up / dn)
    out["ma20"], out["ma50"], out["ma200"] = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    out["hi252"] = c.rolling(252, min_periods=1).max()
    has_ohlc = px["open"].notna().all()
    bull = (c > px["open"]) if has_ohlc else (c > c.shift(1))
    low = px["low"] if has_ohlc else c
    out["touch200"] = (low <= out["ma200"]) & bull & (c > out["ma200"])
    out["xdown20"] = (c.shift(1) >= out["ma20"].shift(1)) & (c < out["ma20"])
    out["xdown50"] = (c.shift(1) >= out["ma50"].shift(1)) & (c < out["ma50"])
    return out


def sim(px, cfg):
    """일별 루프. 반환: (계좌 곡선, 비중 곡선, 통계 dict)."""
    ind = indicators(px)
    c, rf = px["close"].to_numpy(float), px["rf"].to_numpy(float)
    rsi, ma200 = ind["rsi"].to_numpy(), ind["ma200"].to_numpy()
    touch = ind["touch200"].to_numpy()
    xdown = ind["xdown50" if cfg.get("exit_ma") == 50 else "xdown20"].to_numpy()
    dd_lim = cfg.get("dd", -0.30)                      # v2 A2: 관망 기준 완화
    quarter, trend = cfg.get("quarter", False), cfg.get("trend", False)
    target, band, dyn = cfg.get("target", 0.20), cfg.get("band", 0.10), cfg.get("dyn", False)
    hi252 = ind["hi252"].to_numpy()
    n = len(c)
    mod, risk = cfg["module"], cfg.get("risk")          # risk: None | "price" | "account"
    cash, qty, cost = 1.0, 0.0, 0.0                     # cost = 보유분 매입 원가 합 (평단 = cost/qty)
    eq, w = np.ones(n), np.zeros(n)
    peak, age, trades, cycles = 1.0, 0, 0, 0
    trimmed_age = trimmed_rsi = False
    pending = []                                        # 다음 날 종가 체결 주문 [(side, 금액 또는 비율)]
    seed = unit = spent = 0.0
    tranche_left, tranche_amt = 0, 0.0

    r1_cap = cfg.get("r1_cap", 0.70)

    def buy(amount, i):
        nonlocal cash, qty, cost, trades
        V = cash + qty * c[i]
        if risk:                                        # R1: 매수 후 SOXL 비중 ≤ r1_cap
            amount = min(amount, max(0.0, r1_cap * V - qty * c[i]) / (1 + FEE))
        amount = min(amount, cash / (1 + FEE))
        if amount <= 1e-9:
            return 0.0
        cash -= amount * (1 + FEE)
        qty += amount / c[i]
        cost += amount
        trades += 1
        return amount

    def sell(frac, i):
        nonlocal cash, qty, cost, trades
        if qty <= 0 or frac <= 0:
            return
        q = qty * min(1.0, frac)
        cash += q * c[i] * (1 - FEE)
        cost *= (1 - q / qty)
        qty -= q
        trades += 1
        if qty < 1e-12:
            qty, cost = 0.0, 0.0

    for i in range(n):
        if i > 0:
            cash *= 1 + rf[i]
        V = cash + qty * c[i]
        # 오늘 체결: 어제 신호의 주문 (낙폭 관망은 체결일 기준으로 다시 본다)
        dd_blocked = False
        if risk == "price":
            dd_blocked = c[i] / hi252[i] - 1 <= dd_lim
        elif risk == "account":
            dd_blocked = V / peak - 1 <= dd_lim
        todo, pending = pending, []
        for side, x in todo:
            if side == "sell":
                sell(x, i)
            elif not dd_blocked:
                buy(x, i)
        if mod == "cash":
            pass
        elif mod == "M1":
            if qty > 0 and c[i] >= cost / qty * 1.10:
                sell(1.0, i)
                cycles += 1
            elif qty == 0 and i > 0:
                seed = V * (0.7 if risk else 1.0)
                unit, spent = seed / 40, 0.0
                if not dd_blocked:
                    spent += buy(unit, i)
            elif qty > 0 and quarter and spent >= seed - 1e-9 and c[i] < cost / qty:
                sell(0.25, i)                           # v2 A1 쿼터손절: 소진 후 손실이면 1/4 매도, 10회분 재사용
                spent -= 10 * unit
            elif qty > 0 and spent < seed - 1e-9 and not dd_blocked:
                avg = cost / qty
                for lim in (avg, c[i - 1]):             # LOC: 한도는 어제까지 정보, 체결은 오늘 종가
                    if c[i] <= lim and spent < seed - 1e-9:
                        spent += buy(min(0.5 * unit, seed - spent), i)
        elif mod == "M2":
            if qty > 0 and not np.isnan(rsi[i]):
                if rsi[i] >= 70 and not trimmed_rsi:
                    pending.append(("sell", 0.5))
                    trimmed_rsi = True
                if xdown[i]:
                    pending.append(("sell", 1.0))
                    tranche_left = 0
            if tranche_left > 0:
                pending.append(("buy", tranche_amt))
                tranche_left -= 1
            elif ((rsi[i] <= 30 and (not trend or c[i] > ma200[i])) or touch[i]) and not xdown[i]:
                tranche_amt, tranche_left = cash * 0.60 / 4, 3
                pending.append(("buy", tranche_amt))
        elif mod == "M3":
            wt = qty * c[i] / V
            tg = (0.30 if c[i] > ma200[i] else 0.10) if dyn and not np.isnan(ma200[i]) else target
            if wt < tg - band:
                pending.append(("buy", tg * V - qty * c[i]))
            elif wt > tg + band:
                pending.append(("sell", 1 - tg / wt))
        # R3: 포지션 나이 63거래일 → 50% 축소 (포지션당 1회)
        if qty > 0:
            age += 1
            if risk and not cfg.get("no_r3") and age >= 63 and not trimmed_age:
                pending.append(("sell", 0.5))
                trimmed_age = True
        else:
            age, trimmed_age, trimmed_rsi = 0, False, False
        assert cash >= -1e-9 and qty >= 0, (i, cash, qty)
        V = cash + qty * c[i]
        peak = max(peak, V)
        eq[i], w[i] = V, qty * c[i] / V
    idx = px.index
    return pd.Series(eq, idx), pd.Series(w, idx), {"trades": trades, "cycles": cycles}


def stat(eq, rf):
    r = eq.pct_change().dropna()
    yrs = len(r) / 252
    ex = r - rf.reindex(r.index).fillna(0)
    return {"CAGR%": round(((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100, 1),
            "변동성%": round(r.std() * np.sqrt(252) * 100, 1),
            "Sharpe": round(ex.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else 0.0,
            "MDD%": round((eq / eq.cummax() - 1).min() * 100, 1)}


GRID = {f"{m} · {rl}": {"module": m, "risk": rk}
        for m in ("M1", "M2", "M3")
        for rl, rk in (("리스크 끔", None), ("R1+R2p+R3", "price"), ("R1+R2a+R3", "account"))}


def check(act, grid=None):
    cut = act.iloc[:-250]
    for name, cfg in (grid or GRID).items():
        a, _, _ = sim(act, cfg)
        b, _, _ = sim(cut, cfg)
        gap = float((a.loc[b.index] - b).abs().max())
        assert gap < 1e-9, f"{name}: prefix 불변식 실패 ({gap})"
    e, _, _ = sim(act, {"module": "cash"})
    ref = (1 + act["rf"].where(np.arange(len(act)) > 0, 0)).cumprod()
    assert float((e - ref).abs().max()) < 1e-9, "현금 불변식 실패"
    return True


def main(grid=None, tag="soxl_rules", title="SOXL 규칙 백테스트 (사전등록, 누적 N=92)"):
    GRID = grid or globals()["GRID"]
    act, syn, bench = load()
    check(act, GRID)
    print("불변식 (a) prefix 9설정 · (b) 현금 = 단기금리 복리 · (c) 현금·수량 음수 없음 — 통과")
    if "--check" in sys.argv:
        return
    rows, curves = [], {}
    periods = {"실제 2010~2026": act, "합성 1994~2026": syn, "합성 1994~2009": syn.loc[:"2009-12-31"]}
    for pname, px in periods.items():
        rf = px["rf"]
        yrs = len(px) / 252
        cash_eq, _, _ = sim(px, {"module": "cash"})
        cash_cagr = stat(cash_eq, rf)["CAGR%"]
        ref = {"SOXL(또는 합성) 매수보유": px["close"] / px["close"].iloc[0], "현금(단기금리)": cash_eq}
        if pname.startswith("실제"):
            for b in ("SOXX", "QQQ"):
                s = bench[b].reindex(px.index).ffill()
                ref[f"{b} 매수보유"] = s / s.iloc[0]
        else:
            s = bench["SOX"].reindex(px.index).ffill()
            ref["^SOX 1배 매수보유(가격)"] = s / s.iloc[0]
        for k, e in ref.items():
            rows.append({"구간": pname, "설정": k, **stat(e, rf), "평균 SOXL 비중%": None, "연 매매": None, "익절 사이클": None})
        for name, cfg in GRID.items():
            e, w, info = sim(px, cfg)
            curves[(pname, name)] = e
            rows.append({"구간": pname, "설정": name, **stat(e, rf), "평균 SOXL 비중%": round(w.mean() * 100, 1),
                         "연 매매": round(info["trades"] / yrs, 1), "익절 사이클": info["cycles"] if cfg["module"] == "M1" else None})
        rows.append({"구간": pname, "설정": "(현금 CAGR 기준)", "CAGR%": cash_cagr})
    tb = pd.DataFrame(rows)
    tb.to_csv(ROOT / "research" / "out" / f"{tag}.csv", index=False, encoding="utf-8-sig")

    def get(p, s, col):
        return tb[(tb["구간"] == p) & (tb["설정"] == s)][col].iloc[0]

    soxx_sh = get("실제 2010~2026", "SOXX 매수보유", "Sharpe")
    verdict = []
    for name in GRID:
        c1 = get("합성 1994~2026", name, "MDD%") > -60
        c2 = get("실제 2010~2026", name, "Sharpe") > soxx_sh
        c3 = (get("합성 1994~2009", name, "CAGR%") > get("합성 1994~2009", "현금(단기금리)", "CAGR%")
              and get("실제 2010~2026", name, "CAGR%") > get("실제 2010~2026", "현금(단기금리)", "CAGR%"))
        verdict.append({"설정": name, "① 합성 MDD > −60%": c1, "② 실제 Sharpe > SOXX": c2,
                        "③ 두 구간 CAGR > 현금": c3, "판정": "ADOPT 후보" if c1 and c2 and c3 else "REJECT"})
    vt = pd.DataFrame(verdict)
    yr = pd.DataFrame({n: (curves[("실제 2010~2026", n)].resample("YE").last().pct_change() * 100).round(0)
                       for n in GRID})
    yr.index = yr.index.year
    md = [f"# {title}",
          "## 성과\n\n" + tb.to_string(index=False),
          "## 사전등록 판정\n\n" + vt.to_string(index=False),
          "## 연도별 수익률 % (실제 2010~2026)\n\n" + yr.to_string()]
    text = "\n\n".join(md)
    (ROOT / "research" / f"{tag}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
