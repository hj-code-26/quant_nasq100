"""§4 A/B 차이의 **완전 분해** — 잔여분을 '종목 교체'로 뭉뚱그리지 않는다.

실행: python research/ab_attribution.py → research/out/ab_attribution.csv, ab_attribution.md

두 층으로 나눈다. 둘 다 **반올림 전** 값으로 합이 전체 차이와 정확히 일치하는지 검증한다.

1) 현금흐름 항등식 분해 (정확)
   NAV − cash0 = −Σ(매수 명목) − Σ(매수 비용) + Σ(매도 명목) − Σ(매도 비용) + 미청산 평가액
   → ΔNAV 를 이 다섯 항의 차이로 정확히 나눈다.

2) 거래 단위 기여 분해 (정확 — 모든 달러가 매수/매도이므로 빠짐이 없다)
   왕복거래 c 의 기여 = qty × [ 청산가×(1−비용) − 진입가×(1+비용) ]
   미청산은 청산가 대신 마지막 종가.  NAV − cash0 = Σ_c 기여.
   A/B 의 왕복거래를 **같은 종목·진입일 ±SHIFT_MAX 거래일** 로 짝지어 다음 네 갈래로 분류:
     · 시프트-가격/체결 : 같은 종목이 하루 밀려 진입가·청산가가 달라진 몫
     · 시프트-사이징    : 같은 종목인데 수량이 달라진 몫 (현금·평가액 경로의 결과)
     · 시기 변경        : 같은 종목이지만 짝지을 수 없는 다른 시기의 거래
     · 구성종목 변경    : 한쪽에만 등장한 종목
   ★ '구성종목 변경'만 실제 종목 교체다. 시프트는 **같은 종목의 하루 차이**다.

3) 재투자·회전 규모, 비용 총액, 체결 불가는 별도 지표로 낸다 (위 분해 안에 이미 녹아
   있으므로 더하지 않는다 — 중복 계상 금지).
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
CASH0 = 10000.0
SHIFT_MAX = 3          # 사전 정의: 진입일이 이 거래일 수 이내면 '같은 거래가 밀린 것'으로 본다
TOL = 1e-7


def contributions(led, last_close, date_pos):
    """왕복거래(미청산 포함) 단위 기여. Σ 기여 = NAV − cash0 (정확)."""
    open_, rows = {}, []
    for r in led:
        s = r["symbol"]
        if r["side"] == "BUY":
            if s in open_:                      # 추가 매수/재조정 — 평균단가로 합친다
                b = open_[s]
                q = b["qty"] + r["qty"]
                b["entry_cash"] = b["entry_cash"] + r["gross"] + r["cost_usd"]
                b["qty"] = q
            else:
                open_[s] = {"symbol": s, "qty": r["qty"], "entry_date": r["order_date"],
                            "entry_px": r["fill_px"],
                            "entry_cash": r["gross"] + r["cost_usd"]}
        else:
            b = open_.get(s)
            if b is None:
                continue
            proceeds = r["gross"] - r["cost_usd"]
            if r["qty"] < b["qty"] - 1e-12:     # 부분 매도 — 비례 배분
                frac = r["qty"] / b["qty"]
                rows.append(dict(symbol=s, entry_date=b["entry_date"], qty=r["qty"],
                                 entry_px=b["entry_px"], exit_px=r["fill_px"],
                                 exit_date=r["order_date"],
                                 entry_cash=b["entry_cash"] * frac, proceeds=proceeds,
                                 closed=True))
                b["entry_cash"] *= (1 - frac)
                b["qty"] -= r["qty"]
            else:
                rows.append(dict(symbol=s, entry_date=b["entry_date"], qty=b["qty"],
                                 entry_px=b["entry_px"], exit_px=r["fill_px"],
                                 exit_date=r["order_date"],
                                 entry_cash=b["entry_cash"], proceeds=proceeds,
                                 closed=True))
                open_.pop(s)
    for s, b in open_.items():
        rows.append(dict(symbol=s, entry_date=b["entry_date"], qty=b["qty"],
                         entry_px=b["entry_px"], exit_px=float(last_close.get(s, np.nan)),
                         exit_date=pd.NaT, entry_cash=b["entry_cash"],
                         proceeds=b["qty"] * float(last_close.get(s, np.nan)),
                         closed=False))
    for r in rows:
        r["contrib"] = r["proceeds"] - r["entry_cash"]
        r["entry_pos"] = date_pos[r["entry_date"]]
    return rows


def cashflow_split(led, last_close):
    buy_gross = sum(r["gross"] for r in led if r["side"] == "BUY")
    buy_cost = sum(r["cost_usd"] for r in led if r["side"] == "BUY")
    sell_gross = sum(r["gross"] for r in led if r["side"] == "SELL")
    sell_cost = sum(r["cost_usd"] for r in led if r["side"] == "SELL")
    held = {}
    for r in led:
        q = held.get(r["symbol"], 0.0)
        held[r["symbol"]] = q + r["qty"] if r["side"] == "BUY" else q - r["qty"]
    mv = sum(q * float(last_close.get(s, 0.0)) for s, q in held.items() if q > 1e-12)
    return {"매수 명목(−)": -buy_gross, "매수 비용(−)": -buy_cost,
            "매도 명목(+)": sell_gross, "매도 비용(−)": -sell_cost,
            "미청산 평가액(+)": mv}


def match(a_rows, b_rows):
    """같은 종목·진입일 ±SHIFT_MAX 거래일 이내를 가장 가까운 것끼리 짝짓는다(탐욕적)."""
    pairs, used_b = [], set()
    by_sym = {}
    for j, r in enumerate(b_rows):
        by_sym.setdefault(r["symbol"], []).append(j)
    for i, ra in enumerate(a_rows):
        cands = [j for j in by_sym.get(ra["symbol"], [])
                 if j not in used_b
                 and abs(b_rows[j]["entry_pos"] - ra["entry_pos"]) <= SHIFT_MAX]
        if not cands:
            continue
        j = min(cands, key=lambda j: abs(b_rows[j]["entry_pos"] - ra["entry_pos"]))
        used_b.add(j)
        pairs.append((i, j))
    return pairs, used_b


def attribute(a_rows, b_rows):
    """정확한 4갈래 분해. 합 = Σ contrib_B − Σ contrib_A."""
    pairs, used_b = match(a_rows, b_rows)
    matched_a = {i for i, _ in pairs}
    out = {"시프트-가격/체결": 0.0, "시프트-사이징": 0.0,
           "시기 변경(A만)": 0.0, "시기 변경(B만)": 0.0,
           "구성종목 변경(A만)": 0.0, "구성종목 변경(B만)": 0.0}
    detail = []
    for i, j in pairs:
        ra, rb = a_rows[i], b_rows[j]
        # 1주당 기여 r = 청산수취/수량 − 진입지출/수량
        pa = ra["proceeds"] / ra["qty"] - ra["entry_cash"] / ra["qty"]
        pb = rb["proceeds"] / rb["qty"] - rb["entry_cash"] / rb["qty"]
        sizing = (rb["qty"] - ra["qty"]) * pa
        price = rb["qty"] * (pb - pa)
        out["시프트-사이징"] += sizing
        out["시프트-가격/체결"] += price
        detail.append(dict(symbol=ra["symbol"], a_entry=ra["entry_date"],
                           b_entry=rb["entry_date"],
                           shift_days=rb["entry_pos"] - ra["entry_pos"],
                           d_contrib=rb["contrib"] - ra["contrib"],
                           sizing=sizing, price=price))
    b_syms = {r["symbol"] for r in b_rows}
    a_syms = {r["symbol"] for r in a_rows}
    for i, ra in enumerate(a_rows):
        if i in matched_a:
            continue
        key = "시기 변경(A만)" if ra["symbol"] in b_syms else "구성종목 변경(A만)"
        out[key] -= ra["contrib"]
    for j, rb in enumerate(b_rows):
        if j in used_b:
            continue
        key = "시기 변경(B만)" if rb["symbol"] in a_syms else "구성종목 변경(B만)"
        out[key] += rb["contrib"]
    return out, detail


def main():
    close, opens, ret20 = bt.daily_panel()
    last = close.iloc[-1]
    date_pos = {d: i for i, d in enumerate(close.index)}
    rows, md = [], []
    md.append("# A/B 차이의 완전 분해 (반올림 전 합계 검증 포함)")
    md.append("")
    md.append(f"짝짓기 기준: 같은 종목 · 진입일 차이 ≤ {SHIFT_MAX} 거래일 (사전 정의).")
    md.append("**'구성종목 변경'만 실제 종목 교체다.** 시프트는 같은 종목의 하루 차이다.")
    for rule, cfg in RULES.items():
        led = {}
        nav = {}
        for m in E.MODES:
            L = []
            r = E.sim(close, ret20, cfg, opens=opens, mode=m, ledger=L)
            led[m], nav[m] = L, float(r["eq"].iloc[-1])
        for x, y in (("A", "B"), ("B", "C"), ("A", "C")):
            dnav = nav[y] - nav[x]
            # (1) 현금흐름 항등식
            cx, cy = (cashflow_split(led[k], last) for k in (x, y))
            cf = {k: cy[k] - cx[k] for k in cx}
            assert abs(sum(cx.values()) + CASH0 - nav[x]) < TOL, (rule, x)
            assert abs(sum(cf.values()) - dnav) < TOL, (rule, x, y, sum(cf.values()), dnav)
            # (2) 거래 단위 4갈래
            ar = contributions(led[x], last, date_pos)
            br = contributions(led[y], last, date_pos)
            assert abs(sum(r["contrib"] for r in ar) + CASH0 - nav[x]) < TOL, (rule, x)
            assert abs(sum(r["contrib"] for r in br) + CASH0 - nav[y]) < TOL, (rule, y)
            att, detail = attribute(ar, br)
            assert abs(sum(att.values()) - dnav) < TOL, \
                (rule, x, y, sum(att.values()), dnav)
            base = dict(rule=rule, comparison=f"{y}-{x}",
                        nav_x=round(nav[x], 6), nav_y=round(nav[y], 6),
                        delta_nav_usd=round(dnav, 6),
                        delta_total_return_pct=round(dnav / CASH0 * 100, 4),
                        exact_sum_check_usd=round(sum(att.values()) - dnav, 12))
            rows.append({**base, **{f"cf_{k}": round(v, 6) for k, v in cf.items()},
                         **{f"tr_{k}": round(v, 6) for k, v in att.items()},
                         "matched_pairs": len(detail),
                         "roundtrips_x": len(ar), "roundtrips_y": len(br),
                         "capital_deployed_x": round(sum(r["entry_cash"] for r in ar), 2),
                         "capital_deployed_y": round(sum(r["entry_cash"] for r in br), 2),
                         "cost_total_x": round(sum(r["cost_usd"] for r in led[x]), 2),
                         "cost_total_y": round(sum(r["cost_usd"] for r in led[y]), 2)})
            if (x, y) == ("A", "B"):
                md.append("")
                md.append(f"## {rule} — B − A = {dnav:+,.2f} USD "
                          f"({dnav / CASH0 * 100:+.2f}%p, 초기 ${CASH0:,.0f} 기준)")
                md.append("")
                md.append("### (1) 현금흐름 항등식 분해 (정확)")
                md.append("```")
                for k, v in cf.items():
                    md.append(f"  {k:16} {v:+12,.2f}")
                md.append(f"  {'합계':16} {sum(cf.values()):+12,.2f}  "
                          f"(ΔNAV {dnav:+,.2f}, 오차 {sum(cf.values()) - dnav:.2e})")
                md.append("```")
                md.append("### (2) 거래 단위 4갈래 분해 (정확)")
                md.append("```")
                for k, v in att.items():
                    md.append(f"  {k:20} {v:+12,.2f}")
                md.append(f"  {'합계':20} {sum(att.values()):+12,.2f}  "
                          f"(오차 {sum(att.values()) - dnav:.2e})")
                md.append("```")
                shift = att["시프트-가격/체결"] + att["시프트-사이징"]
                swap = att["구성종목 변경(A만)"] + att["구성종목 변경(B만)"]
                tim = att["시기 변경(A만)"] + att["시기 변경(B만)"]
                md.append(f"- 같은 종목이 하루 밀린 몫: **{shift:+,.2f}** "
                          f"(짝지은 거래 {len(detail)}쌍)")
                md.append(f"- 같은 종목·다른 시기: **{tim:+,.2f}**")
                md.append(f"- 실제 구성종목이 달라진 몫: **{swap:+,.2f}**")
                md.append(f"- 투입 자본(재투자 규모): A ${sum(r['entry_cash'] for r in ar):,.0f} "
                          f"→ B ${sum(r['entry_cash'] for r in br):,.0f}")
                md.append(f"- 총 비용: A ${sum(r['cost_usd'] for r in led[x]):,.2f} "
                          f"→ B ${sum(r['cost_usd'] for r in led[y]):,.2f} "
                          f"(위 분해에 이미 포함 — 따로 더하지 않는다)")
                d = pd.DataFrame(detail)
                if len(d):
                    top = d.reindex(d.d_contrib.abs().sort_values(ascending=False).index)
                    md.append("")
                    md.append("상위 5쌍 (같은 종목, 진입일 시프트):")
                    md.append("```")
                    md.append(top.head(5)[["symbol", "shift_days", "d_contrib",
                                           "sizing", "price"]].round(2).to_string(index=False))
                    md.append("```")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ab_attribution.csv", index=False, encoding="utf-8-sig")
    md.append("")
    md.append("## 체결 불가")
    md.append("Full-path A/B/C 에는 체결 불가 주문이 없다 — 엔진이 현금·슬롯 한도 안에서만 "
              "주문을 만들기 때문이다. 체결 불가가 나오는 것은 의도를 고정한 frozen-intent "
              "진단뿐이며(⓪ 2/646, ① 2/345, ④ 0/325), 그 2건은 표본 마지막 날 의도라 "
              "다음 세션이 없어서 체결하지 못한 것이다. → ab_decomposition.md §2")
    (OUT / "ab_attribution.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
