"""§2·§3 — A/B 를 원장 수준으로 재현하고 상승 원인을 분해한다.

python research/ab_audit.py     → research/out/*.csv, manifest.json

산출: manifest.json, ab_summary.csv, ab_trade_diff.csv, daily_nav.csv,
      cost_sensitivity.csv, ab_decomposition.txt
"""
import hashlib
import json
import pathlib
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from research.isolation import guard, verify           # noqa: E402

guard()

import backtest as bt                                   # noqa: E402
from research import engine as E                        # noqa: E402

OUT = pathlib.Path(__file__).with_name("out")
OUT.mkdir(exist_ok=True)
CACHE = pathlib.Path(__file__).parent.parent / "data_cache"

RULES = {
    "⓪운영설정": dict(sl=None, mom_exit=False, hold_days=20, top=10, max_pos_pct=15),
    "①손절15+모멘텀0": dict(sl=15, mom_exit=True),
    "④손절25+모멘텀0+회전가드": dict(sl=25, mom_exit=True, rotate_guard=True),
}
COST_BP = (0, 5, 10, 20, 25)      # 편도 bp. 25bp(0.25%) 가 저장소 기본값.


def manifest(close, opens):
    h = hashlib.sha256()
    for p in sorted(CACHE.glob("*_1d.pkl")):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True).stdout.strip()
    return {
        "git_commit": git,
        "git_dirty_files": [l[3:] for l in dirty.splitlines()] if dirty else [],
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "data_source": "토스증권 Open API 일봉 (data_cache/*.pkl, 오프라인 캐시)",
        "data_files": len(list(CACHE.glob("*_1d.pkl"))),
        "data_sha256": h.hexdigest(),
        "universe": "nasdaq100.TICKERS — **오늘(2026-09) 기준 구성종목**. point-in-time 아님 → 생존 편향 잔존",
        "universe_asof": "2026-09 (편입/편출 이력 없음)",
        "panel_symbols": int(close.shape[1]),
        "panel_start": str(close.index[0].date()), "panel_end": str(close.index[-1].date()),
        "panel_rows": int(close.shape[0]),
        "open_missing_pct": round(float(opens.isna().mean().mean()) * 100, 3),
        "warmup_rows": 20,
        "seed": 20260910,
        "cash0": 10000.0, "currency": "USD (원화 환산 없음, 단리 아님·복리 누적수익)",
        "cost_default_bp_oneway": 25,
        "regime_indicator": {
            "⓪운영설정": "사용 안 함", "①손절15+모멘텀0": "사용 안 함",
            "④손절25+모멘텀0+회전가드": "사용 안 함",
            "BASE(결정적 기준선)": "사용 안 함", "C1": "사용 안 함", "C2": "사용 안 함",
            "C3": "QQQ 60일 수익률(운영 기본) + 유니버스 60일 중앙값(운영 폴백) 병행",
            "note": "국면 판정을 쓰는 실험은 C3 뿐이다 — 나머지는 QQQ 부재의 영향을 받지 않는다",
            "source_file": "research/out/regime_source.csv"},
        "qqq_history": "research/data/QQQ_1d.pkl (Yahoo Finance via yfinance, 무료·비상업. "
                       "유료 구매·브로커 API·신규 토큰 발급 없음). meta: research/data/QQQ_1d.meta.json",
        "availability_model": "research/engine.py AVAILABILITY — 입력별 available_at. "
                              "일괄 shift 아님. 정수 주문은 t-1 종가로 수량 산정",
        "isolation": "research/isolation.py — 자격증명 제거·운영 경로 쓰기 차단·"
                     "브로커 요청 차단·로그 리다이렉트·WAL 포함 지문 검증",
        "mode_c_scope": "연구 기준 엔진. 하루 1회 개장 시장가 체결 가정. "
                        "실전 다회 실행(프리/애프터장 지정가·LLM 재량)과 동등하지 않다",
        "modes": {"A": "판정=t종가, 체결=t종가 (원본·비현실적 진단 대조군)",
                  "B": "판정=t종가, 체결=t시가, 신호=t-1 (현 저장소 엔진, 잔여 누수 있음)",
                  "C": "판정=t-1종가, 체결=t시가, 신호=t-1 (엄밀 인과)"},
    }


def roundtrips(led):
    """원장 → 왕복거래. 종목당 동시 1로트라 BUY→SELL 짝짓기가 일의적이다."""
    open_, rows = {}, []
    for r in led:
        s = r["symbol"]
        if r["side"] == "BUY":
            open_[s] = r
        else:
            b = open_.pop(s, None)
            if b is None:
                continue
            rows.append(dict(
                symbol=s, entry_date=b["order_date"], exit_date=r["order_date"],
                signal_date_in=b["signal_date"], signal_date_out=r["signal_date"],
                qty=b["qty"], entry_px=b["fill_px"], exit_px=r["fill_px"],
                reason=r["reason"], hold_days=r["hold_days"],
                buy_cost=b["cost_usd"], sell_cost=r["cost_usd"],
                net_pnl=r["qty"] * r["fill_px"] * (1 - b["cost_usd"] / b["gross"])
                - b["qty"] * b["fill_px"] * (1 + b["cost_usd"] / b["gross"]),
                ret_pct=(r["fill_px"] / b["fill_px"] - 1) * 100))
    for s, b in open_.items():                       # 표본 종료 시 미청산
        rows.append(dict(symbol=s, entry_date=b["order_date"], exit_date=pd.NaT,
                         signal_date_in=b["signal_date"], signal_date_out=pd.NaT,
                         qty=b["qty"], entry_px=b["fill_px"], exit_px=np.nan,
                         reason="미청산(표본종료)", hold_days=np.nan,
                         buy_cost=b["cost_usd"], sell_cost=0.0,
                         net_pnl=np.nan, ret_pct=np.nan))
    return pd.DataFrame(rows)


def identity_check(led, res, close, cash0):
    """금액 항등식: 초기현금 − Σ매수지출 + Σ매도수취 + 미청산 평가액 == 최종 NAV"""
    buy = sum(r["gross"] + r["cost_usd"] for r in led if r["side"] == "BUY")
    sell = sum(r["gross"] - r["cost_usd"] for r in led if r["side"] == "SELL")
    open_ = {}
    for r in led:
        open_[r["symbol"]] = r if r["side"] == "BUY" else None
    mv = sum(r["qty"] * close.iloc[-1].get(s, r["fill_px"])
             for s, r in open_.items() if r)
    lhs = cash0 - buy + sell + mv
    nav = float(res["eq"].iloc[-1])
    return lhs, nav, abs(lhs - nav)


def main():
    close, opens, ret20 = bt.daily_panel()
    assert E.assert_equivalent(close, opens, ret20, RULES), "기존 daily_sim 과 불일치"

    mf = manifest(close, opens)
    mf["equivalence_vs_backtest_daily_sim"] = "PASS (A·B 자산곡선 완전 일치)"
    mf["isolation_verify"] = verify() if (OUT / "isolation_baseline.json").exists() else         "baseline 없음 — python research/isolation.py 를 먼저 실행"
    with open(OUT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(mf, f, ensure_ascii=False, indent=2)

    summary, navs, ledgers, ident = [], {}, {}, []
    for name, cfg in RULES.items():
        for m in E.MODES:
            led = []
            r = E.sim(close, ret20, cfg, opens=opens, mode=m, ledger=led)
            navs[f"{name}|{m}"] = r["eq"]
            ledgers[(name, m)] = led
            lhs, nav, err = identity_check(led, r, close, 10000.0)
            ident.append(dict(rule=name, mode=m, ledger_nav=round(lhs, 6),
                              engine_nav=round(nav, 6), abs_err=round(err, 9),
                              ok=err < 1e-6))
            summary.append(dict(
                rule=name, mode=m, total_return_pct=round(r["총수익"] * 100, 2),
                cagr_pct=round(r["CAGR"] * 100, 2), mdd_pct=round(r["MDD"] * 100, 2),
                sharpe=round(r["Sharpe"], 3), trades=r["거래"],
                win_pct=round(r["승률"], 1), avg_hold=round(r["평균보유일"], 1),
                stop_pct=round(r["손절%"], 1), open_at_end=r["미청산"],
                start=str(r["eq"].index[0].date()), end=str(r["eq"].index[-1].date()),
                years=round(len(r["eq"]) / 252, 3)))
    pd.DataFrame(summary).to_csv(OUT / "ab_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ident).to_csv(OUT / "accounting_identity.csv", index=False,
                               encoding="utf-8-sig")
    pd.DataFrame(navs).to_csv(OUT / "daily_nav.csv", encoding="utf-8-sig")

    # ── 거래별 변화 원장 ───────────────────────────────────────────────
    diff_rows = []
    for name in RULES:
        rt = {m: roundtrips(ledgers[(name, m)]) for m in E.MODES}
        for m in E.MODES:
            rt[m]["key"] = rt[m]["symbol"] + "@" + rt[m]["entry_date"].astype(str)
        keys = {m: set(rt[m]["key"]) for m in E.MODES}
        for m in E.MODES:
            d = rt[m].copy()
            d["rule"], d["mode"] = name, m
            d["presence"] = ["공통" if all(k in keys[x] for x in E.MODES)
                             else "+".join(x for x in E.MODES if k in keys[x])
                             for k in d["key"]]
            diff_rows.append(d)
    diff = pd.concat(diff_rows, ignore_index=True)
    diff.to_csv(OUT / "ab_trade_diff.csv", index=False, encoding="utf-8-sig")

    # ── 비용 민감도: 각 비용에서 엔진 전체 재실행 ────────────────────
    cost_rows = []
    for name, cfg in RULES.items():
        for bp in COST_BP:
            for m in E.MODES:
                c = dict(cfg, cost_pct=bp / 100.0)
                r = E.sim(close, ret20, c, opens=opens, mode=m)
                cost_rows.append(dict(rule=name, mode=m, cost_bp_oneway=bp,
                                      total_return_pct=round(r["총수익"] * 100, 2),
                                      cagr_pct=round(r["CAGR"] * 100, 2),
                                      sharpe=round(r["Sharpe"], 3), trades=r["거래"]))
    cs = pd.DataFrame(cost_rows)
    cs.to_csv(OUT / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")

    # ── 분해 리포트 ───────────────────────────────────────────────────
    L = []
    P = L.append
    P("# A/B 상승 원인 분해 — 실행 결과")
    P("")
    P("## 0. 재현성 (Gate 1)")
    P(f"- 기존 `backtest.daily_sim` 과 A·B 자산곡선 **완전 일치** (max abs diff = 0).")
    P(f"- 데이터 해시 {mf['data_sha256'][:16]}…, 종목 {mf['panel_symbols']}, "
      f"{mf['panel_start']}~{mf['panel_end']} ({mf['panel_rows']}행)")
    P("- 금액 항등식(초기현금−매수+매도+미청산평가 = NAV):")
    for r in ident:
        P(f"  - {r['rule']} {r['mode']}: err={r['abs_err']:.2e} {'OK' if r['ok'] else 'FAIL'}")
    P("")
    P("## 1. 모드별 성적 (편도 25bp)")
    P("| 규칙 | A 총수익 | B 총수익 | C 총수익 | A→B | B→C |")
    P("|---|---|---|---|---|---|")
    S = pd.DataFrame(summary).set_index(["rule", "mode"])
    for name in RULES:
        a, b, c = (S.loc[(name, m), "total_return_pct"] for m in E.MODES)
        P(f"| {name} | {a:+.1f}% | {b:+.1f}% | {c:+.1f}% | {b-a:+.1f}%p | {c-b:+.1f}%p |")
    P("")
    P("모두 **누적수익률**이다. CAGR 차이도 연간 알파도 아니다.")
    P("")
    P("## 2. 가격 효과 vs 경로 의존 효과")
    P("Frozen-intent: A 의 의도(날짜·종목·수량)를 고정하고 체결가만 다음-시가로 교체.")
    P("경로(현금·슬롯·재진입)를 재계산하지 않는 **진단**이며 실행 가능한 전략이 아니다.")
    P("")
    P("| 규칙 | A(전체) | Frozen-intent(가격만) | B(전체경로) | 가격효과 | 경로효과 | 실행불가 주문 |")
    P("|---|---|---|---|---|---|---|")
    for name, cfg in RULES.items():
        fz = E.frozen_intent(close, opens, ret20, cfg)
        a = S.loc[(name, "A"), "total_return_pct"]
        b = S.loc[(name, "B"), "total_return_pct"]
        f = fz["총수익"] * 100
        P(f"| {name} | {a:+.1f}% | {f:+.1f}% | {b:+.1f}% | {f-a:+.1f}%p | {b-f:+.1f}%p | "
          f"{fz['불가주문']}/{fz['주문수']} |")
    P("")
    P("## 3. B 의 잔여 미래 정보 (B→C)")
    P("B 는 진입 신호만 t-1 로 밀었을 뿐, **청산 판정(손절·만기)과 주문 수량 산정에는")
    P("t 일 종가**를 쓴다. 즉 '오늘 종가가 손절선을 깼다'를 보고 '오늘 시가'에 판다.")
    P("C 는 판정까지 t-1 종가로 밀어 이 잔여 누수를 제거한다.")
    P("")
    for name in RULES:
        b, c = (S.loc[(name, m), "total_return_pct"] for m in ("B", "C"))
        P(f"- {name}: B {b:+.1f}% → C {c:+.1f}% = **{c-b:+.1f}%p** (잔여 누수 기여분)")
    P("")
    P("## 4. 상위 기여 거래 / 연도 / 갭")
    for name in RULES:
        d = diff[diff.rule == name]
        pv = d.pivot_table(index="key", columns="mode", values="net_pnl", aggfunc="sum")
        if {"A", "B"} <= set(pv.columns):
            pv["dBA"] = pv["B"].fillna(0) - pv["A"].fillna(0)
            top = pv.reindex(pv.dBA.abs().sort_values(ascending=False).index).head(5)
            P(f"\n### {name} — B−A 기여 상위 5 (금액 PnL, $)")
            P("| 거래 | A | B | Δ |")
            P("|---|---|---|---|")
            for k, row in top.iterrows():
                P(f"| {k} | {row.get('A', float('nan')):.1f} | "
                  f"{row.get('B', float('nan')):.1f} | {row['dBA']:+.1f} |")
            P("> 상위 거래 제외는 사후 취약성 진단이지 대체 전략 성과가 아니다.")
        d2 = d.dropna(subset=["exit_date"]).copy()
        d2["year"] = pd.to_datetime(d2["entry_date"]).dt.year
        yr = d2.pivot_table(index="year", columns="mode", values="net_pnl", aggfunc="sum")
        P(f"\n### {name} — 연도별 실현 PnL ($)")
        P("```"); P(yr.round(1).to_string()); P("```")
    P("")
    P("## 5. 비용 민감도 (편도 bp, 각 비용에서 엔진 전체 재실행 — 회전율 근사 아님)")
    P("임의 스트레스 격자다. 실제 토스 수수료·환전비용 자료로 확인한 값이 아니다.")
    P("")
    piv = cs.pivot_table(index=["rule", "mode"], columns="cost_bp_oneway",
                         values="total_return_pct")
    P("```"); P(piv.round(1).to_string()); P("```")
    P("")
    P("일봉만 있으므로 스프레드·시장충격·분봉 지연은 재현하지 못한다.")
    (OUT / "ab_decomposition.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
