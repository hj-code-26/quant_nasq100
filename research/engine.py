"""A/B/C 체결 가정 감사용 계측 엔진.

backtest.daily_sim 을 고치지 않고 **재현**한다(운영/기존 산출물 보존).
`assert_equivalent()` 가 A·B 모드에서 daily_sim 과 자산곡선이 동일함을 확인한다.

세 가지 모드 — 바뀌는 것은 **판정가·체결가·신호 시각뿐**이고 나머지 규칙은 동일하다:

  A  판정=t 종가, 체결=t 종가, 모멘텀=t          (원본. 같은 종가 체결 = 비현실적 진단 대조군)
  B  판정=t 종가, 체결=t 시가, 모멘텀=t-1        (현재 저장소의 '수정된' 엔진)
  C  판정=t-1 종가, 체결=t 시가, 모멘텀=t-1      (엄밀 인과. 결정 시각 <= 주문 시각 <= 체결 시각)

B 는 진입 신호의 하루 선행은 제거했지만 **청산 판정과 주문 수량 산정에 t 일 종가를 쓴다.**
즉 "오늘 종가가 손절선을 깼다"를 보고 "오늘 시가"에 판다 — 여전히 미래 정보다.
C 는 그 잔여 누수까지 제거한다. A→B 상승분의 원인 분해는 B 와 C 의 차이로 본다.

NAV 평가는 세 모드 모두 t 일 종가로 mark-to-market 한다(회계 기준 통일).
"""
import numpy as np
import pandas as pd

MODES = ("A", "B", "C")

# ── 모드 C 의 정보 가용 시각 (available_at) ──────────────────────────────
# 일괄 shift 가 아니다. 각 입력이 실제로 언제 알려지는지, 그리고 **주문 종류**가
# 그 정보만으로 성립하는지를 따로 정한다.
#   decision_time = t-1 세션 종료 직후,  order_time = t 개장,  fill_time = t 개장
AVAILABILITY = {
    "momentum_ret20":  ("t-1 close", "신호. 결정 시각에 확정"),
    "expiry_count":    ("t-1 close", "거래일 달력은 사전 확정"),
    "stop_loss_price": ("t-1 close", "판정가. B 는 여기서 t 종가를 써서 누수가 났다"),
    "positions_qty":   ("t-1 close", "보유 수량"),
    "cash":            ("t-1 close", "현금. 실전은 개장 전 조회로 더 정확하나 보수적으로 t-1"),
    "nav_for_limits":  ("t-1 close", "15% 종목한도·10% 현금유지 산정 기준"),
    "cov_matrix":      ("t-1 close", "C1 공분산. 창은 t-1 까지"),
    "regime_ret60":    ("t-1 close", "C3 국면 판정"),
    "order_amount":    ("t-1 close", "금액 주문(정규장 소수점) — 수량은 체결가로 결정된다"),
    "order_quantity":  ("t-1 close", "수량 주문(정수 주) — 수량을 t-1 종가로 산정해야 한다"),
    "fill_price":      ("t open",    "체결가. **결정에 쓰면 안 된다**"),
    "nav_mark":        ("t close",   "회계 평가 전용. 결정에 쓰지 않는다"),
}
# ★ 모드 C 는 **연구 기준 엔진**이다. 하루 1회 개장 시각 시장가 체결을 가정한다.
#   운영은 하루 여러 번 사이클을 돌고 프리/애프터장 지정가와 LLM 재량이 들어간다.
#   따라서 모드 C 가 실전 다회 실행과 동등하다고 말하지 않는다.
MAX_EXT = 2        # C2 사전 정의: 만기 연장 최대 2회 (=최대 보유 60거래일)
SHRINK = 0.2       # C1 사전 정의: 공분산 축소 강도 (대각으로)
COV_WIN, COV_MIN = 60, 40   # C1 사전 정의: 공분산 창 / 최소 관측


def sim(close, ret20, cfg, cash0=10000.0, opens=None, mode="A", ledger=None):
    """계좌 단위 일별 시뮬. ledger 가 list 면 거래별 원장 행을 append 한다.

    §5 실험 옵션(전부 기본 꺼짐 — 켜지 않으면 감사 엔진과 바이트 동일):
      cfg["risk_target"]  C1. 목표 바스켓 위험(연 %)에 맞춘 총자산 대비 주식 상한.
      cfg["trim"]         "prop"(비례) | "largest"(평가액 큰 순). 노출 축소 방식.
      cfg["regime"]       C3. (진입 임계, 해제 임계). 단일 경계면 두 값을 같게 준다.
      cfg["expiry_net"]   C2. 만기에 전량매도 대신 자격 유지 시 목표 수량 차이만 거래.
    """
    top = cfg.get("top", 5)
    max_pos_pct = cfg.get("max_pos_pct", 30)
    reserve = cfg.get("reserve_pct", 10)
    min_usd = cfg.get("min_usd", 5)
    cost = cfg.get("cost_pct", 0.25) / 100
    sl = cfg.get("sl")
    mom_exit = cfg.get("mom_exit", True)
    mom_th = cfg.get("mom_th", 0)
    hold_days = cfg.get("hold_days")
    entry_th = cfg.get("entry_th", 0)
    rotate_guard = cfg.get("rotate_guard", False)
    frac = cfg.get("frac", True)
    rebal = cfg.get("rebal", 1)
    risk_target = cfg.get("risk_target")          # C1
    # 단순 저노출·현금 혼합 기준선 (C1 대조군). 총자산 대비 주식 상한을 상수로 고정.
    static_w = cfg.get("static_weight")
    trim = cfg.get("trim", "prop")
    regime_th = cfg.get("regime")                 # C3 (enter, exit) — 음수 %
    # 국면지표 출처. None 이면 유니버스 60일 수익률 중앙값(운영 **폴백** 경로).
    # Series 를 주면 그것(예: QQQ 60일 수익률 = 운영 **기본** 지표)을 쓴다.
    regime_series = cfg.get("regime_series")
    expiry_net = cfg.get("expiry_net", False)     # C2
    rets = (close.pct_change()
            if (risk_target or regime_th or cfg.get("static_weight")) else None)
    ret60 = (close.pct_change(60) * 100
             if (regime_th and cfg.get("regime_series") is None) else None)
    vol20 = (rets.rolling(20).std() * np.sqrt(252) * 100) if regime_th else None
    if mode != "A" and opens is None:
        raise ValueError("B/C 모드는 opens 가 필요하다")

    cash, pos = cash0, {}          # pos: sym -> [qty, avg, entry_n, entry_date]
    equity, trades, holds = [], [], []
    bear, regime_log, cap_log, expiry_log = False, [], [], []
    ext, resize = {}, set()        # C2: 종목별 연장 횟수 / 목표수량 재조정 대상
    dates = close.index[20:]
    for n, t in enumerate(dates):
        if mode == "A":
            judge = fill = close.loc[t]
            mo = ret20.loc[t]
            sig_t = t
        else:
            if n == 0:
                equity.append(cash)
                continue
            prev = dates[n - 1]
            mo = ret20.loc[prev]
            sig_t = prev
            judge = close.loc[t] if mode == "B" else close.loc[prev]
            fill = opens.loc[t]
            fill = fill.where(fill.notna(), close.loc[t])
        mark = close.loc[t]        # NAV 평가는 항상 당일 종가

        # ── C3 국면 (판정은 sig_t 까지의 자료만 쓴다) ──────────────────
        if regime_th:
            med = (regime_series.loc[sig_t] if regime_series is not None
                   else ret60.loc[sig_t].median())
            enter_th, exit_th = regime_th
            if bear:
                bear = not (pd.notna(med) and med > exit_th)
            else:
                bear = bool(pd.notna(med) and med < enter_th)
            regime_log.append((t, float(med) if pd.notna(med) else np.nan, bear))

        if regime_th and bear:      # 하락 국면: 모멘텀 부호 무시, 저변동성 우선 (운영 규칙)
            v = vol20.loc[sig_t].dropna()
            cands = [s for s in v.sort_values().index
                     if s not in pos and pd.notna(fill.get(s))]
        else:
            cands = [s for s in mo.dropna().sort_values(ascending=False).index
                     if mo[s] > entry_th and s not in pos and pd.notna(fill.get(s))]
        for sym in list(pos):
            p, f = judge.get(sym), fill.get(sym)
            if pd.isna(p) or pd.isna(f):
                continue
            qty, avg, since, edate = pos[sym]
            why = None
            if hold_days and n - since >= hold_days:
                why = "만기"
                if expiry_net:
                    # 유지 자격: 정상 국면이면 모멘텀 문턱 통과, 하락 국면이면 저변동성 상위 top.
                    ok = (sym in (list(vol20.loc[sig_t].dropna().sort_values().index[:top])
                                  if (regime_th and bear) else [])
                          or (not (regime_th and bear)
                              and pd.notna(mo.get(sym)) and mo[sym] > entry_th))
                    if ok and ext.get(sym, 0) < MAX_EXT:
                        ext[sym] = ext.get(sym, 0) + 1
                        pos[sym][2] = n                      # 기준일 재설정
                        resize.add(sym)
                        expiry_log.append((t, sym, "연장", ext[sym]))
                        continue
                    expiry_log.append((t, sym, "청산", ext.get(sym, 0)))
            elif sl and p <= avg * (1 - sl / 100):
                why = "손절"
            elif mom_exit and pd.notna(mo.get(sym)) and mo[sym] < mom_th:
                why = None if (rotate_guard and not cands) else "모멘텀"
            if why:
                proceeds = qty * f * (1 - cost)
                cash += proceeds
                trades.append((why, (f / avg - 1) * 100))
                holds.append(n - since)
                if ledger is not None:
                    ledger.append(dict(mode=mode, side="SELL", symbol=sym, reason=why,
                                       signal_date=sig_t, order_date=t, fill_date=t,
                                       entry_date=edate, qty=qty, fill_px=f, avg_px=avg,
                                       gross=qty * f, cost_usd=qty * f * cost,
                                       net_pnl=proceeds - qty * avg,
                                       hold_days=n - since, feasible=True))
                del pos[sym]
        # ── C1 위험 타겟: 목표 바스켓 위험 기반 총자산 대비 주식 상한 ──
        room = None
        if risk_target or static_w:
            hist = (rets.loc[:sig_t].tail(COV_WIN) if risk_target
                    else pd.DataFrame(index=[], columns=close.columns))
            syms = [x for x in pos if x in hist.columns and hist[x].count() >= COV_MIN]
            cap, warm = (static_w, False) if static_w else (cfg.get("warmup_cap", 1.0), True)
            if syms and risk_target:
                mv = np.array([pos[x][0] * judge.get(x, pos[x][1]) for x in syms], float)
                if mv.sum() > 0:
                    w = mv / mv.sum()
                    S = hist[syms].cov().values * 252
                    S = (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S))
                    bvol = float(np.sqrt(max(w @ S @ w, 0.0))) * 100
                    if bvol > 0:
                        cap, warm = min(1.0, risk_target / bvol), False
            total_now = cash + sum(q * judge.get(x, a) for x, (q, a, _, _) in pos.items())
            sv = sum(q * judge.get(x, a) for x, (q, a, _, _) in pos.items())
            cap_log.append((t, cap, warm, sv / total_now if total_now else 0.0))
            excess = sv - total_now * cap
            if excess > min_usd and sv > 0:
                order = (sorted(pos, key=lambda x: -pos[x][0] * judge.get(x, pos[x][1]))
                         if trim == "largest" else list(pos))
                sv0 = sv
                for x in order:
                    if excess <= min_usd:
                        break
                    q, avg, since, edate = pos[x]
                    f, jp = fill.get(x), judge.get(x)
                    if pd.isna(f) or pd.isna(jp) or jp <= 0:
                        continue
                    v = q * jp
                    cut = min(excess, v) if trim == "largest" else v * (excess / sv0)
                    dq = min(q, cut / jp)
                    if dq * f < min_usd:
                        continue
                    cash += dq * f * (1 - cost)
                    excess -= dq * jp
                    if dq >= q - 1e-12:
                        trades.append(("노출축소", (f / avg - 1) * 100))
                        holds.append(n - since)
                        del pos[x]
                    else:
                        pos[x][0] = q - dq
                    if ledger is not None:
                        ledger.append(dict(mode=mode, side="SELL", symbol=x,
                                           reason="노출축소", signal_date=sig_t,
                                           order_date=t, fill_date=t, entry_date=edate,
                                           qty=dq, fill_px=f, avg_px=avg, gross=dq * f,
                                           cost_usd=dq * f * cost,
                                           net_pnl=dq * f * (1 - cost) - dq * avg,
                                           hold_days=n - since, feasible=True))
                sv = sum(q * judge.get(x, a) for x, (q, a, _, _) in pos.items())
            room = max(0.0, total_now * cap - sv)

        # ── C2 만기 연장분: 목표 수량 차이만 거래 ──────────────────────
        for sym in list(resize):
            resize.discard(sym)
            if sym not in pos:
                continue
            q, avg, since, edate = pos[sym]
            f, jp = fill.get(sym), judge.get(sym)
            if pd.isna(f) or pd.isna(jp):
                continue
            total_now = cash + sum(x[0] * judge.get(k, x[1]) for k, x in pos.items())
            budget = min(total_now * max_pos_pct / 100,
                         cash - total_now * reserve / 100 + q * jp)
            if room is not None:
                budget = min(budget, room + q * jp)
            tgt = max(0.0, budget) / f / (1 + cost)
            if not frac:
                tgt = float(int(tgt))
            dq = tgt - q
            if abs(dq) * f < min_usd:
                continue
            if dq > 0:
                cash -= dq * f * (1 + cost)
                pos[sym][1] = (q * avg + dq * f) / (q + dq)
            else:
                cash += (-dq) * f * (1 - cost)
            pos[sym][0] = tgt
            if room is not None:
                room = max(0.0, room - dq * f)
            if ledger is not None:
                ledger.append(dict(mode=mode, side="BUY" if dq > 0 else "SELL",
                                   symbol=sym, reason="만기재평가-순주문",
                                   signal_date=sig_t, order_date=t, fill_date=t,
                                   entry_date=edate, qty=abs(dq), fill_px=f,
                                   avg_px=pos[sym][1], gross=abs(dq) * f,
                                   cost_usd=abs(dq) * f * cost, net_pnl=0.0,
                                   hold_days=0, feasible=True))

        if n % rebal:
            equity.append(cash + sum(q * mark.get(s, a) for s, (q, a, _, _) in pos.items()))
            continue
        for sym in cands:
            if len(pos) >= top:
                break
            total = cash + sum(q * judge.get(s, a) for s, (q, a, _, _) in pos.items())
            budget = min(total * max_pos_pct / 100, cash - total * reserve / 100)
            if room is not None:
                budget = min(budget, room)
            p = fill.get(sym)
            # 주문 종류별 수량 산정 (available_at 정합성):
            #  · 금액 주문(정규장 소수점): 달러 금액만 정하고 수량은 체결가로 결정된다 → 누수 없음
            #  · 수량 주문(정수 주, 장외): 수량을 **주문 시각에 아는 가격**(t-1 종가)으로 산정해야
            #    한다. t 시가로 정수 수량을 정하면 미래 정보다.
            size_px = p if (frac or mode == "A") else judge.get(sym)
            if pd.isna(p) or pd.isna(size_px):
                continue
            qty = (budget / p / (1 + cost) if frac
                   else float(int(budget / size_px / (1 + cost))))
            if qty <= 0 or qty * p < min_usd:
                continue
            cash -= qty * p * (1 + cost)
            if room is not None:
                room -= qty * p
            pos[sym] = [qty, p, n, t]
            if ledger is not None:
                ledger.append(dict(mode=mode, side="BUY", symbol=sym, reason="진입",
                                   signal_date=sig_t, order_date=t, fill_date=t,
                                   entry_date=t, qty=qty, fill_px=p, avg_px=p,
                                   gross=qty * p, cost_usd=qty * p * cost,
                                   net_pnl=0.0, hold_days=0, feasible=True))
        equity.append(cash + sum(q * mark.get(s, a) for s, (q, a, _, _) in pos.items()))

    eq = pd.Series(equity, index=dates)
    m = _metrics(eq, cash0, trades, holds, pos, cash)
    m["regime_log"], m["cap_log"], m["expiry_log"] = regime_log, cap_log, expiry_log
    return m


def _metrics(eq, cash0, trades, holds, pos, cash):
    r = eq.pct_change().dropna()
    years = len(eq) / 252
    cagr = (eq.iloc[-1] / cash0) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1
    kinds = pd.Series([w for w, _ in trades])
    return {"eq": eq, "총수익": eq.iloc[-1] / cash0 - 1, "CAGR": cagr,
            "MDD": (eq / eq.cummax() - 1).min(),
            "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0,
            "거래": len(trades), "평균보유일": float(np.mean(holds)) if holds else 0,
            "손절%": (kinds == "손절").mean() * 100 if len(kinds) else 0,
            "승률": float(np.mean([p > 0 for _, p in trades]) * 100) if trades else 0,
            "미청산": len(pos), "종료현금": cash, "일수익률": r}


def frozen_intent(close, opens, ret20, cfg, cash0=10000.0):
    """(1) Frozen-intent 진단 — A 의 의도(어느 날 어느 종목을 사고 판다)를 고정하고
    체결가만 **다음 세션 시가**로 바꾼다. 경로(현금·슬롯)는 재계산하지 않는다.
    A 의 의도는 t 일 종가에 확정되므로 실현 가능한 가장 빠른 체결은 t+1 시가다.
    현금이 모자라 실행 불가능한 주문은 feasible=False 로 표시만 하고 자금을 넣지 않는다.
    ★ 이것은 자체 실행 가능한 전략 성과가 아니라 가격 효과만 떼어 보는 진단이다."""
    led = []
    sim(close, ret20, cfg, cash0, mode="A", ledger=led)
    cost = cfg.get("cost_pct", 0.25) / 100
    nxt = {d: close.index[i + 1] for i, d in enumerate(close.index[:-1])}
    cash, pos, rows = cash0, {}, []
    for r in led:
        s = r["symbol"]
        d = nxt.get(r["order_date"])
        if d is None:                                # 표본 마지막 날 의도 → 체결 불가
            rows.append(dict(r, frozen_fill_px=np.nan, feasible=False))
            continue
        f = opens.loc[d].get(s)
        if pd.isna(f):
            f = close.loc[d].get(s)
        if r["side"] == "BUY":
            need = r["qty"] * f * (1 + cost)
            ok = need <= cash + 1e-9
            if ok:
                cash -= need
                pos[s] = (r["qty"], f)
        else:
            ok = s in pos
            if ok:
                q, avg = pos.pop(s)
                cash += q * f * (1 - cost)
        rows.append(dict(r, frozen_fill_date=d, frozen_fill_px=f, feasible=bool(ok)))
    nav = cash + sum(q * close.iloc[-1].get(s, a) for s, (q, a) in pos.items())
    return {"총수익": nav / cash0 - 1, "불가주문": sum(1 for x in rows if not x["feasible"]),
            "주문수": len(rows), "rows": rows}


def assert_equivalent(close, opens, ret20, rules):
    """계측 엔진의 A/B 가 기존 backtest.daily_sim 과 자산곡선까지 동일한지 확인."""
    import backtest as bt
    for name, cfg in rules.items():
        for mode, op in (("A", None), ("B", opens)):
            mine = sim(close, ret20, cfg, opens=opens, mode=mode)["eq"]
            theirs = bt.daily_sim(close, ret20, cfg, opens=op)["eq"]
            assert mine.index.equals(theirs.index), (name, mode)
            assert np.allclose(mine.values, theirs.values), (name, mode,
                                                             float(abs(mine - theirs).max()))
    return True
