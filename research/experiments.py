"""§4.1 결정적 정량 기준선 + §5 후보 실험 (C1/C2/C3) + 국면지표 출처 추적.

실행: python research/experiments.py
  → out/experiment_registry.csv, experiment_results.csv, risk_reduction_compare.csv,
    regime_source.csv, c1_vs_cashmix.csv, experiments.md

원칙 (변경 없음):
- 기준선은 mode C(엄밀 인과)·결정적 규칙. 실제 LLM 을 제거한 별도 실험이며 운영 변경이 아니다.
- 사전등록을 먼저 저장하고 실행한다. **새 후보나 파라미터 탐색을 늘리지 않는다.**
- PIT 부재이므로 어떤 후보도 '실전 우위'로 승격하지 않는다.

이번 추가:
- 국면지표 출처를 실험별로 기록한다(regime_source.csv). ⓪①④·C1·C2 는 국면을 **쓰지 않는다**.
- C3 는 운영 기본 지표 **QQQ** 로 재실행하고, 폴백(유니버스 중앙값)과 나란히 낸다.
- C1 을 단순 저노출·현금 혼합 기준선과 비교한다(실제 위험 지표로).
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
from research.fetch_index import aligned_ret60   # noqa: E402
from research.stats import paired_bootstrap      # noqa: E402

OUT = pathlib.Path(__file__).with_name("out")
OUT.mkdir(exist_ok=True)

BASE = dict(sl=None, mom_exit=False, hold_days=20, top=10, max_pos_pct=15,
            reserve_pct=10, cost_pct=0.25)

REGISTRY = [
    dict(exp_id="C1", name="위험 타겟 방식",
         hypothesis="계좌 이력 단일자산 근사 대신 보유 바스켓의 ex-ante 위험 sqrt(w'Σw) 로 "
                    "총자산 대비 주식 상한을 정하면 같은 목표 위험에서 낙폭이 줄어든다",
         primary_metric="MDD (부차: 수익 감소 대비 위험 감소가 단순 비중 축소보다 나은가)",
         allowed_risk_change="목표 30%. 현금 10%·종목당 15%·비레버리지 유지",
         drop_condition="단순 현금 혼합 기준선보다 낫지 않거나 CI 가 0 을 포함하면 미확정",
         n_attempts=3, variants="warmup_cap 1.0/0.5, trim prop/largest"),
    dict(exp_id="C2", name="만기 재평가·순주문",
         hypothesis="만기에 전량 매도 대신 유지 자격이 있으면 목표 수량 차이만 거래하면 "
                    "회전율이 줄고 성과가 나빠지지 않는다",
         primary_metric="누적수익 (부차: 회전율, 평균 보유일, 꼬리손실)",
         allowed_risk_change="연장 최대 2회(=최대 60거래일), 기준일 재설정",
         drop_condition="만기 당일 동일 종목 재매수 빈도 0 이면 비용 절감 주장 취소",
         n_attempts=1, variants="expiry_net=True"),
    dict(exp_id="C3", name="국면 전환 완충",
         hypothesis="단일 -3% 경계 대신 -4% 진입 / -2% 해제 완충을 두면 국면 전환 횟수가 "
                    "줄고 그로 인한 회전이 준다",
         primary_metric="국면 전환 횟수 (부차: 누적수익, 하락기 손익)",
         allowed_risk_change="없음 — 국면 판정 임계만 변경",
         drop_condition="전환 횟수가 줄지 않거나 성과가 유의하게 나빠지면 탈락",
         n_attempts=2,
         variants="지표 QQQ(운영 기본) 와 유니버스 중앙값(폴백) 각각에서 (-3,-3) vs (-4,-2)"),
]
CAVEATS = ("생존 편향 유니버스(오늘의 나스닥100) · PIT 부재 · 3년 표본 · 이미 열람한 구간")


# ── 위험 지표 ─────────────────────────────────────────────────────────────
def risk_metrics(m):
    """총노출이 같아도 위험은 같지 않다 — 실제 위험 지표로 잰다."""
    r = m["eq"].pct_change().dropna()
    dn = r[r < 0]
    es5 = float(r[r <= np.percentile(r, 5)].mean()) if len(r) > 20 else np.nan
    cl = pd.DataFrame(m["cap_log"], columns=["date", "cap", "warmup", "stock_w"]) \
        if m["cap_log"] else pd.DataFrame(columns=["cap", "warmup", "stock_w"])
    return dict(
        vol_ann_pct=round(float(r.std() * np.sqrt(252) * 100), 2),
        downside_dev_pct=round(float(dn.std() * np.sqrt(252) * 100), 2) if len(dn) else np.nan,
        mdd_pct=round(float(m["MDD"] * 100), 2),
        es5_daily_pct=round(es5 * 100, 3),
        worst_day_pct=round(float(r.min() * 100), 2),
        mean_stock_weight_pct=(round(float(cl.stock_w.mean() * 100), 1)
                               if len(cl) else np.nan),
        warmup_days=int(cl.warmup.sum()) if len(cl) else 0)


def hhi(m, close):
    """평균 집중도(HHI). 축소 방식이 위험 구조를 어떻게 바꾸는지 — 총노출과 별개 지표."""
    return np.nan   # 보유 비중 시계열을 남기지 않으므로 미측정. (거래 수·최대비중으로 대신함)


def turnover(m):
    return m["거래"] / (len(m["eq"]) / 252)


def run(cfg, close, opens, ret20, mode="C"):
    return E.sim(close, ret20, dict(BASE, **cfg), opens=opens, mode=mode)


def boot_ci(a_eq, b_eq, L=20):
    pair = pd.concat({"a": a_eq.pct_change(), "b": b_eq.pct_change()}, axis=1).dropna()
    bo = paired_bootstrap(pair["a"], pair["b"], L)
    out = {}
    for k in ("cum", "sharpe", "mdd"):
        d = bo["B"][k] - bo["A"][k]
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        out[k] = (round(float(lo), 4), round(float(hi), 4), bool(lo > 0 or hi < 0))
    return out


def main():
    pd.DataFrame(REGISTRY).assign(caveats=CAVEATS, registered_before_run=True).to_csv(
        OUT / "experiment_registry.csv", index=False, encoding="utf-8-sig")

    close, opens, ret20 = bt.daily_panel()
    close = close.reindex(sorted(close.columns), axis=1)      # 동점 처리 고정
    opens = opens.reindex(close.columns, axis=1)
    ret20 = ret20.reindex(close.columns, axis=1)
    qqq = aligned_ret60(close.index)                          # 운영 기본 국면지표

    base = run({}, close, opens, ret20)
    rows = [dict(exp_id="BASE", variant="결정적 정량 기준선 (운영 파라미터 proxy, mode C)",
                 regime_source="없음(국면 미사용)",
                 total_return_pct=round(base["총수익"] * 100, 2),
                 cagr_pct=round(base["CAGR"] * 100, 2),
                 sharpe=round(base["Sharpe"], 3), trades=base["거래"],
                 turnover_per_yr=round(turnover(base), 1),
                 **risk_metrics(base),
                 ci_cum="", ci_sharpe="", ci_mdd="", ci_excludes_zero_95="",
                 note="LLM 제거 · 운영 변경 아님")]

    variants = [
        ("C1", "위험타겟30% · 비례축소 · warmup_cap=1.0", "없음(국면 미사용)",
         dict(risk_target=30, trim="prop", warmup_cap=1.0)),
        ("C1", "위험타겟30% · 비례축소 · warmup_cap=0.5(보수적)", "없음(국면 미사용)",
         dict(risk_target=30, trim="prop", warmup_cap=0.5)),
        ("C1", "위험타겟30% · 평가액 큰 순 축소(실전 방식)", "없음(국면 미사용)",
         dict(risk_target=30, trim="largest", warmup_cap=1.0)),
        ("C1-대조", "단순 현금혼합 주식 75% 고정 (사전 지정)", "없음(국면 미사용)",
         dict(static_weight=0.75)),
        ("C2", "만기 재평가·순주문 (연장 최대 2회)", "없음(국면 미사용)",
         dict(expiry_net=True)),
        ("C3", "QQQ · 단일경계 -3%", "QQQ (운영 기본, yfinance)",
         dict(regime=(-3, -3), regime_series=qqq)),
        ("C3", "QQQ · 완충 -4%/-2%", "QQQ (운영 기본, yfinance)",
         dict(regime=(-4, -2), regime_series=qqq)),
        ("C3-폴백", "유니버스중앙값 · 단일경계 -3%", "유니버스 60일 중앙값 (운영 폴백)",
         dict(regime=(-3, -3))),
        ("C3-폴백", "유니버스중앙값 · 완충 -4%/-2%", "유니버스 60일 중앙값 (운영 폴백)",
         dict(regime=(-4, -2))),
    ]
    res = {}
    for eid, label, src, cfg in variants:
        m = run(cfg, close, opens, ret20)
        res[label] = m
        ci = boot_ci(base["eq"], m["eq"])
        rows.append(dict(
            exp_id=eid, variant=label, regime_source=src,
            total_return_pct=round(m["총수익"] * 100, 2),
            cagr_pct=round(m["CAGR"] * 100, 2), sharpe=round(m["Sharpe"], 3),
            trades=m["거래"], turnover_per_yr=round(turnover(m), 1),
            **risk_metrics(m),
            ci_cum=f"[{ci['cum'][0]:+.2f}, {ci['cum'][1]:+.2f}]",
            ci_sharpe=f"[{ci['sharpe'][0]:+.2f}, {ci['sharpe'][1]:+.2f}]",
            ci_mdd=f"[{ci['mdd'][0]:+.3f}, {ci['mdd'][1]:+.3f}]",
            ci_excludes_zero_95=any(x[2] for x in ci.values()),
            note="기준선 대비 짝지은 block bootstrap L=20, 2000회, 95% percentile CI"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "experiment_results.csv", index=False, encoding="utf-8-sig")

    # ── 국면지표 출처 추적표 ───────────────────────────────────────────
    src_rows = [
        dict(experiment="⓪운영설정", uses_regime=False, regime_source="해당 없음",
             evidence="research/ab_audit.py RULES — cfg 에 regime 키 없음",
             affected_by_qqq_gap=False),
        dict(experiment="①손절15+모멘텀0", uses_regime=False, regime_source="해당 없음",
             evidence="research/ab_audit.py RULES", affected_by_qqq_gap=False),
        dict(experiment="④손절25+모멘텀0+회전가드", uses_regime=False, regime_source="해당 없음",
             evidence="research/ab_audit.py RULES", affected_by_qqq_gap=False),
        dict(experiment="BASE(결정적 기준선)", uses_regime=False, regime_source="해당 없음",
             evidence="research/experiments.py BASE", affected_by_qqq_gap=False),
        dict(experiment="C1(위험타겟)", uses_regime=False, regime_source="해당 없음",
             evidence="cfg=risk_target/trim/warmup_cap 만", affected_by_qqq_gap=False),
        dict(experiment="C2(만기 재평가)", uses_regime=False, regime_source="해당 없음",
             evidence="cfg=expiry_net 만", affected_by_qqq_gap=False),
        dict(experiment="C3(국면 완충)", uses_regime=True,
             regime_source="QQQ 60일 수익률 (운영 기본) + 유니버스 중앙값(폴백) 병행",
             evidence="research/fetch_index.py / experiments.py C3 변형 4개",
             affected_by_qqq_gap=True),
    ]
    pd.DataFrame(src_rows).to_csv(OUT / "regime_source.csv", index=False,
                                  encoding="utf-8-sig")

    # ── C1 vs 단순 현금혼합 ────────────────────────────────────────────
    c1 = res["위험타겟30% · 비례축소 · warmup_cap=1.0"]
    c1w = risk_metrics(c1)["mean_stock_weight_pct"] / 100
    matched = run(dict(static_weight=round(c1w, 4)), close, opens, ret20)   # 사후 진단
    fixed75 = res["단순 현금혼합 주식 75% 고정 (사전 지정)"]
    cmp_rows = []
    for label, m, kind in (("기준선(노출 제한 없음)", base, "기준"),
                           ("C1 위험타겟30%(비례)", c1, "후보"),
                           ("단순 현금혼합 75% (사전 지정)", fixed75, "대조·사전"),
                           (f"단순 현금혼합 {c1w * 100:.0f}% (C1 실현 노출에 맞춤)", matched,
                            "대조·**사후 진단**")):
        rm = risk_metrics(m)
        d_ret = (m["총수익"] - base["총수익"]) * 100
        d_vol = rm["vol_ann_pct"] - risk_metrics(base)["vol_ann_pct"]
        cmp_rows.append(dict(
            variant=label, kind=kind,
            total_return_pct=round(m["총수익"] * 100, 2), **rm,
            sharpe=round(m["Sharpe"], 3),
            d_return_pp=round(d_ret, 2), d_vol_pp=round(d_vol, 2),
            return_given_up_per_vol_pp=(round(-d_ret / -d_vol, 2)
                                        if d_vol < -1e-9 else np.nan)))
    cc = pd.DataFrame(cmp_rows)
    cc.to_csv(OUT / "c1_vs_cashmix.csv", index=False, encoding="utf-8-sig")

    # ── 비례 vs 큰 순 축소 (실제 위험 지표) ─────────────────────────────
    cmp2 = []
    for label in ("위험타겟30% · 비례축소 · warmup_cap=1.0",
                  "위험타겟30% · 평가액 큰 순 축소(실전 방식)"):
        m = res[label]
        cmp2.append(dict(trim=label, total_return_pct=round(m["총수익"] * 100, 2),
                         sharpe=round(m["Sharpe"], 3), trades=m["거래"],
                         **risk_metrics(m)))
    ci_trim = boot_ci(res["위험타겟30% · 비례축소 · warmup_cap=1.0"]["eq"],
                      res["위험타겟30% · 평가액 큰 순 축소(실전 방식)"]["eq"])
    pd.DataFrame(cmp2).to_csv(OUT / "risk_reduction_compare.csv", index=False,
                              encoding="utf-8-sig")

    # ── C2 사전 확인 ───────────────────────────────────────────────────
    led = []
    E.sim(close, ret20, BASE, opens=opens, mode="C", ledger=led)
    by_day = {}
    for r in led:
        by_day.setdefault((r["order_date"], r["symbol"]), set()).add(r["side"])
    same_day = sum(1 for v in by_day.values() if v == {"BUY", "SELL"})
    exp_sells = sum(1 for r in led if r["reason"] == "만기")

    # ── C3 전환 횟수 / 국면별 손익 ──────────────────────────────────────
    c3 = {}
    for label in [x[1] for x in variants if x[0].startswith("C3")]:
        lg = res[label]["regime_log"]
        sw = sum(1 for a, b in zip(lg, lg[1:]) if a[2] != b[2])
        c3[label] = (sw, sum(1 for x in lg if x[2]), len(lg))

    L = ["# §4.1 기준선 + §5 후보 실험 — 실행 결과", "",
         f"엔진: mode C (엄밀 인과, 연구 기준 엔진). 한계: {CAVEATS}", "",
         "## 국면지표 출처 (모든 실험 추적)", "```",
         pd.DataFrame(src_rows).to_string(index=False), "```", "",
         "**⓪·①·④·BASE·C1·C2 는 국면 판정을 아예 쓰지 않는다.** 따라서 QQQ 부재의 영향을",
         "받은 실험은 C3 뿐이었고, 이번에 QQQ 를 확보해 운영 기본 지표로 재실행했다.",
         "'운영설정 성과'를 폴백 기반 proxy 로 재표시할 필요는 없다 — 애초에 국면을 안 쓴다.",
         "",
         "## 결과 (기준선 대비 95% percentile CI)", "```",
         df.to_string(index=False), "```", "",
         "## C1 vs 단순 저노출·현금 혼합", "```", cc.to_string(index=False), "```",
         "- `return_given_up_per_vol_pp` = 포기한 수익(%p) ÷ 줄인 변동성(%p). 작을수록 좋다.",
         "- '실현 노출에 맞춤' 행은 **전체 표본의 사후 실현값으로 배율을 맞춘 사후 진단**이다. "
         "사전에 알 수 없는 값이므로 실행 가능한 대안이 아니다.", "",
         "## 비례 축소 vs 평가액 큰 순 축소 (실제 위험 지표)", "```",
         pd.DataFrame(cmp2).to_string(index=False), "```",
         f"- 두 방식 차이의 95% CI: 누적수익 {ci_trim['cum'][:2]}, "
         f"Sharpe {ci_trim['sharpe'][:2]}, MDD {ci_trim['mdd'][:2]} — "
         f"0 을 {'제외' if any(x[2] for x in ci_trim.values()) else '포함'}한다.", "",
         "## C2 사전 확인 — 만기 당일 동일 종목 재매수",
         f"- 만기 청산 {exp_sells}건 중 같은 날 같은 종목 재매수 **{same_day}건**.",
         ("- 0 이므로 '만기 즉시 재매수 비용 절감' 주장은 취소한다." if same_day == 0
          else "- 재매수가 존재하므로 순주문의 비용 절감 여지가 있다."), "",
         "## C3 국면 전환 횟수", "```"]
    for k, (sw, days, n) in c3.items():
        L.append(f"{k:34} 전환 {sw:2d}회, 하락 판정일 {days:3d}/{n}일")
    L += ["```", ""]
    for src in ("QQQ", "유니버스중앙값"):
        try:
            s1 = next(k for k in c3 if src in k and "단일경계" in k)
            s2 = next(k for k in c3 if src in k and "완충" in k)
        except StopIteration:
            continue
        ci3 = boot_ci(res[s1]["eq"], res[s2]["eq"])
        L.append(f"### {src} — 완충 − 단일경계 (짝지은 block bootstrap L=20, 95% CI)")
        L.append(f"- 누적수익 {ci3['cum'][:2]} · Sharpe {ci3['sharpe'][:2]} · "
                 f"MDD {ci3['mdd'][:2]} → 0 을 "
                 f"{'제외' if any(x[2] for x in ci3.values()) else '포함'}한다")
        L.append(f"- 점추정 완충 {res[s2]['총수익'] * 100:+.2f}% vs "
                 f"단일경계 {res[s1]['총수익'] * 100:+.2f}%")
        for label, key in (("단일경계", s1), ("완충", s2)):
            m = res[key]
            lg = pd.DataFrame(m["regime_log"], columns=["date", "ind", "bear"])
            r = m["eq"].pct_change().reindex(lg["date"]).fillna(0).values
            b = lg["bear"].values
            L.append(f"  · {label} 하락일 {b.sum():3d}일 누적 {np.prod(1 + r[b]) - 1:+7.2%} | "
                     f"정상일 {(~b).sum():3d}일 누적 {np.prod(1 + r[~b]) - 1:+8.2%}")
        L.append("")
    L += ["※ -4/-2 는 최적값이 아니라 사전 지정한 단일 후보다. 그리드 탐색을 하지 않았다.",
          "※ 'V자 회복 손실'은 회복 구간의 사전 정의가 없으면 사후 곡선 맞추기가 된다.", "",
          "## 판정",
          "CI 가 0 을 포함하면 **미확정**이다(동등성 선언이 아니다). PIT 부재이므로 유의해도",
          "'채택 후보'까지이고 실전 우위로 승격하지 않는다."]
    (OUT / "experiments.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
