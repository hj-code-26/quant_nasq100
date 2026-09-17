"""과거 시점(PIT) 시가총액 — SEC 공시 발행주식수 × 분할조정 종가.

    SEC_CONTACT=<email> python research/pit_mcap.py --fetch   # SEC companyfacts + yfinance 분할 이력 (1회)
    python research/pit_mcap.py                               # 검증 게이트: 시총가중 재현 vs ^NDX

설계
  발행주식수  dei:EntityCommonStockSharesOutstanding (공시 표지). 없으면 us-gaap:CommonStockSharesOutstanding.
             같은 공시(accn)·같은 기준일의 여러 값은 **주식 종류별**이라 합산한다 (GOOGL A·B·C 등).
  시점       값은 **공시일(filed)부터** 쓴다 — 기준일(end)이 아니다. 그 전에는 알 수 없었다.
  분할       종가가 분할조정돼 있으므로 주식수를 오늘 기준으로 맞춘다:
             조정주식수 = 공시값 × (기준일 이후 분할 비율의 곱). 시총 = 조정주식수 × 분할조정 종가.
  같은 회사   CIK 가 같은 티커(GOOG/GOOGL, FOX/FOXA, NWS/NWSA)는 회사 시총을 한 번만 센다 —
             알파벳 순 첫 티커 하나만 남긴다 (고정 규칙).
  연락처     SEC 는 User-Agent 에 연락처를 요구한다. 환경변수 SEC_CONTACT 로만 받고 파일에 남기지 않는다.

검증 게이트 (사전 등록 — 통과 못 하면 시총 기반 전략을 돌리지 않는다)
  재현 = 그날 PIT 구성종목을 전일 시총 비중으로 보유 (매일 리밸런스, 비용 0)
  비교 = ^NDX 일수익 (가격지수. NDX 는 수정 시총가중이라 완전 일치는 불가)
  통과 = 일수익 상관 ≥ 0.98 · 연 추적오차 ≤ 4% · CAGR 차 |Δ| ≤ 2%p · 구성종목-일 시총 커버리지 ≥ 90%
"""
import os
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PITDIR = ROOT / "research" / "data" / "pit"
SHARES, SPLITS = PITDIR / "sec_shares.pkl", PITDIR / "splits.pkl"
MANUAL_CIK = {"ESRX": 1532063, "SPLS": 791519}          # 회사 티커 목록에서 빠진 상장폐지 종목
EXCLUDE = {"VIP": "티커 재사용 — SEC 매핑이 VimpelCom 이 아니라 Vulcan Infrastructure 를 준다"}
# ADR: SEC 값은 보통주 수, 종가는 ADS 가격. ADS 비율이 기간마다 바뀌어(NTES 25→5, TCOM 1/8→1 등)
# 추정하면 오차가 더 크다 → 시총 산출에서 뺀다 (결손으로 센다). 1:1 상장(ASML·ARM·CCEP)은 남긴다.
EXCLUDE.update({t: "ADR 비율" for t in ("NTES", "PDD", "BIDU", "JD", "TCOM", "VOD", "AZN")})
EXTRA_CIK = {"GOOG": [1288776], "GOOGL": [1288776]}     # 2015 지주사 전환 이전 Google Inc.
TAGS = (("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"))   # 우선순위 순 (마지막은 대체값)


def fetch():
    from research.isolation import guard
    guard(allow_network=("www.sec.gov", "data.sec.gov"))
    import requests
    import yfinance as yf
    from research import exit_study as ES

    contact = os.environ.get("SEC_CONTACT")
    assert contact, "SEC_CONTACT 환경변수가 필요하다 (SEC User-Agent 요구사항)"
    H = {"User-Agent": f"quant_nasq100 research {contact}"}
    close, _, _, _ = ES.raw()
    cols = list(close.columns)
    m = {v["ticker"]: v["cik_str"] for v in
         requests.get("https://www.sec.gov/files/company_tickers.json", headers=H, timeout=30).json().values()}
    out = {}
    for t in cols:
        cik = m.get(t.replace(".", "-"), MANUAL_CIK.get(t))
        if cik is None:
            out[t] = None
            continue
        rows, name = [], None
        for k in [cik, *EXTRA_CIK.get(t, [])]:
            time.sleep(0.15)                              # SEC 10 req/s 제한
            r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(k):010d}.json",
                             headers=H, timeout=60)
            if r.status_code != 200:
                continue
            j = r.json()
            name = name or j.get("entityName")
            for ns, tag in TAGS:
                for u in j.get("facts", {}).get(ns, {}).get(tag, {}).get("units", {}).get("shares", []):
                    rows.append({"src": tag, "end": u["end"], "start": u.get("start"), "filed": u["filed"],
                                 "accn": u["accn"], "val": u["val"]})
        out[t] = {"cik": cik, "name": name, "rows": rows}
        print(f"{t:6s} CIK {cik:<8} {(name or '')[:40]:40s} {len(rows)} 행")
    pickle.dump(out, open(SHARES, "wb"))

    splits = {}
    for t in cols:
        try:
            s = yf.Ticker(t.replace(".", "-")).splits
            splits[t] = s[s > 0] if len(s) else pd.Series(dtype=float)
        except Exception as e:                            # 폐지 종목은 이력이 없을 수 있다 — 결손으로 기록
            splits[t] = None
            print(f"{t} 분할 이력 실패: {type(e).__name__}")
    pickle.dump(splits, open(SPLITS, "wb"))
    print(f"→ {SHARES.name} · {SPLITS.name}")


def shares_series(rec, splits, dates):
    """공시일부터 유효한 분할조정 주식수 (일별, 앞으로 채움)."""
    if not rec or not rec["rows"]:
        return None
    df = pd.DataFrame(rec["rows"])
    df["prio"] = df["src"].map({tag: i for i, (_, tag) in enumerate(TAGS)})
    df = df[df["prio"] == df.groupby("accn")["prio"].transform("min")]         # 공시마다 가장 나은 태그 하나
    df = df[df["end"] == df.groupby("accn")["end"].transform("max")]           # 그 공시의 가장 최근 기준일
    if df["src"].str.startswith("Weighted").any():                             # 가중평균: 가장 짧은(최근 분기) 기간만
        st = pd.to_datetime(df["start"])
        df = df[st.isna() | (st == st.groupby([df["accn"], df["end"]]).transform("max"))]
    df = df.drop_duplicates(["accn", "end", "start", "val"])
    df = df.groupby(["accn", "end", "filed"], as_index=False)["val"].sum()      # 주식 종류 합산
    df["end"], df["filed"] = pd.to_datetime(df["end"]), pd.to_datetime(df["filed"])
    if splits is not None and len(splits):
        sp = splits.copy()
        sp.index = pd.to_datetime(sp.index).tz_localize(None)
        df["val"] = [v * float(np.prod(sp[sp.index > e].values)) for v, e in zip(df["val"], df["end"])]
    df = df.sort_values(["filed", "end"]).groupby("filed")["val"].last()        # 같은 날 여러 공시면 최신 기준일
    med = df.rolling(5, min_periods=1).median()                                 # 후행 창 — 이후 공시를 보지 않는다
    df = df[(df / med).between(1 / 3, 3)]                                       # 단위 오기(×1000 등) 제거
    return df.reindex(dates.union(df.index)).ffill().reindex(dates)


def mcap_panel(close):
    sh, spl = pickle.load(open(SHARES, "rb")), pickle.load(open(SPLITS, "rb"))
    dates = close.index
    cap = pd.DataFrame(index=dates, columns=close.columns, dtype=float)
    for t in close.columns:
        s = None if t in EXCLUDE else shares_series(sh.get(t), spl.get(t), dates)
        if s is not None:
            cap[t] = s * close[t]
    by_cik = {}
    for t in close.columns:                                # 같은 회사는 티커 하나만
        rec = sh.get(t)
        if rec:
            by_cik.setdefault(rec["cik"], []).append(t)
    dup = []
    for cik, ts in by_cik.items():
        if len(ts) > 1:
            keep = min(ts)                                 # 고정 규칙 (자료 길이에 따라 바뀌면 prefix 불변식이 깨진다)
            dup += [x for x in ts if x != keep]
    cap[dup] = np.nan
    return cap, dup


def validate():
    from research import exit_study as ES
    from research import corr_limits as C
    import pit

    close, _, ndx, _ = ES.raw()
    close = close.loc["2014-06-01":]
    cap, dup = mcap_panel(close)
    member = pd.DataFrame(pit.mask(close.index, list(close.columns), quiet=True), index=close.index,
                          columns=close.columns)
    ret = close.ffill(limit=5).pct_change(fill_method=None)
    w = cap.shift(1).where(member.shift(1, fill_value=False))
    cover = (w.notna() & member.shift(1, fill_value=False)).sum(1) / member.shift(1, fill_value=False).sum(1)
    rep = (w.div(w.sum(1), axis=0) * ret.fillna(0)).sum(1).loc[ES.PIT_START:]
    ix = ndx.reindex(close.index).ffill().pct_change().loc[ES.PIT_START:]
    a, b = rep.align(ix, join="inner")
    corr = float(np.corrcoef(a, b)[0, 1])
    te = float((a - b).std() * np.sqrt(252) * 100)
    dc = C.stat(a)["cagr"] - C.stat(b)["cagr"]
    cov = float(cover.loc[ES.PIT_START:].mean() * 100)
    ok = corr >= 0.98 and te <= 4 and abs(dc) <= 2 and cov >= 90
    print(f"중복 CIK 로 뺀 티커: {dup}")
    print(f"시총 커버리지(구성종목-일) {cov:.1f}% · 상관 {corr:.4f} · 연 추적오차 {te:.2f}% · "
          f"CAGR 재현 {C.stat(a)['cagr']:.2f}% vs ^NDX {C.stat(b)['cagr']:.2f}% (Δ {dc:+.2f}%p)")
    yr = pd.DataFrame({"재현": (1 + a).groupby(a.index.year).prod() - 1, "^NDX": (1 + b).groupby(b.index.year).prod() - 1})
    print((yr * 100).round(1).to_string())
    print(f"\n검증 게이트: {'통과' if ok else '실패 — 시총 기반 전략을 돌리지 않는다'}")
    return ok


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        validate()
