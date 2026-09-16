"""20일 모멘텀이 음수인 종목이 양전할 빈도 — 과거 빈도다. 예측이 아니다.

    python research/mom_flip.py

두 가지를 **따로** 센다. 섞으면 답이 틀린다.
  ① 양전 확률   P(mom20 > 0 at t+h | mom20 < 0 at t)
     mom20 = P_t / P_{t-20} - 1 이라 **20일 전의 나쁜 날이 창 밖으로 빠지기만 해도 양전한다.**
     주가가 그대로여도 일어난다. 즉 이건 '회복'이 아니라 달력 효과를 상당 부분 포함한다.
  ② 실제 수익   P(t+h 종가 > t 종가) 와 지수 대비 초과수익 — 돈이 되는지는 이쪽이다.

관측이 겹치므로(같은 날 여러 종목, 이웃 날짜끼리 중복) 날짜별 횡단면 비율을 먼저 내고
날짜에 대해 원형 블록 부트스트랩으로 CI 를 낸다. 종목 단위 t 검정은 쓰지 않는다.

출력: research/out/mom_flip.csv
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from research import exit_study as ES       # noqa: E402  (guard() 는 여기서 켜진다)

HORIZONS = (5, 10, 20, 40, 60)
# ★ F["mom20"] 은 **비율**이다 (0.2 = +20%). 퍼센트로 쓰면 전 구간이 한 칸에 들어간다.
BANDS = ((-0.05, 0.0, "얕음 (-5~0%)"), (-0.10, -0.05, "중간 (-10~-5%)"),
         (-0.20, -0.10, "깊음 (-20~-10%)"), (-1e9, -0.20, "매우 깊음 (<-20%)"),
         (-1e9, 0.0, "── 음수 전체 ──"))
BLOCK, REPS, SEED = 21, 3000, 0


def boot_ci(daily, w):
    """날짜별 비율의 가중평균에 원형 블록 부트스트랩 CI. daily/w 는 같은 길이의 배열."""
    ok = np.isfinite(daily) & (w > 0)
    d, ww = daily[ok], w[ok]
    if len(d) < BLOCK * 3:
        return np.nan, np.nan
    rng = np.random.default_rng(SEED)
    n, k = len(d), int(np.ceil(len(d) / BLOCK))
    idx = ((rng.integers(0, n, size=(REPS, k))[:, :, None] + np.arange(BLOCK)[None, None, :])
           .reshape(REPS, k * BLOCK) % n)[:, :n]
    s = np.average(d[idx], axis=1, weights=ww[idx])
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def measure(P, label, bear_only=None):
    mom, valid, px = P["F"]["mom20"], P["valid"], np.cumprod(1 + np.nan_to_num(P["ret"]), axis=0)
    ix = P["ixpx"]
    bear = P["R"]["live_bear"]
    n = len(P["dates"])
    rows = []
    for lo, hi, bname in BANDS:
        for h in HORIZONS:
            flip, up, exc, wts = [], [], [], []
            for t in range(n - h):
                if bear_only is not None and bool(bear[t]) != bear_only:
                    continue
                sel = valid[t] & np.isfinite(mom[t]) & (mom[t] > lo) & (mom[t] <= hi)
                sel &= np.isfinite(px[t]) & np.isfinite(px[t + h]) & (px[t] > 0)
                m = int(sel.sum())
                if m < 3:
                    continue
                r = px[t + h, sel] / px[t, sel] - 1
                flip.append(float(np.nanmean(mom[t + h, sel] > 0)))
                up.append(float(np.nanmean(r > 0)))
                exc.append(float(np.nanmean(r - (ix[t + h] / ix[t] - 1))) * 100)
                wts.append(m)
            if len(flip) < BLOCK * 3:
                continue
            f, u, e, w = (np.array(x) for x in (flip, up, exc, wts))
            lo_f, hi_f = boot_ci(f, w)
            lo_u, hi_u = boot_ci(u, w)
            lo_e, hi_e = boot_ci(e, w)
            rows.append({
                "표본": label, "구간": bname, "기간(거래일)": h,
                "관측일": len(f), "평균 종목수": round(w.mean(), 1),
                "① 양전%": round(np.average(f, weights=w) * 100, 1),
                "① 95%CI": f"{lo_f * 100:.1f}~{hi_f * 100:.1f}",
                "② 상승%": round(np.average(u, weights=w) * 100, 1),
                "② 95%CI": f"{lo_u * 100:.1f}~{hi_u * 100:.1f}",
                "③ 지수대비%p": round(np.average(e, weights=w), 2),
                "③ 95%CI": f"{lo_e:+.2f}~{hi_e:+.2f}"})
    return rows


def main():
    panels = ES.panels()
    rows = measure(panels["PIT"], "PIT 2015~2026")
    rows += measure(panels["PIT"], "PIT · 하락 국면만", bear_only=True)
    rows += measure(panels["BIAS"], "편향 1999~2014")
    tb = pd.DataFrame(rows)
    tb.to_csv(ES.OUT / "mom_flip.csv", index=False, encoding="utf-8-sig")
    for lab in tb["표본"].unique():
        print(f"\n### {lab}")
        print(tb[tb["표본"] == lab].drop(columns="표본").to_string(index=False))

    # 대조군 — 전체(음수·양수 무관) 기준선. ②·③ 를 이것과 비교해야 의미가 있다
    P = panels["PIT"]
    px = np.cumprod(1 + np.nan_to_num(P["ret"]), axis=0)
    ix, n = P["ixpx"], len(P["dates"])
    for h in (20, 60):
        u = [np.nanmean(px[t + h, P["valid"][t]] / px[t, P["valid"][t]] - 1 > 0)
             for t in range(n - h) if P["valid"][t].sum() > 3]
        print(f"\n[대조군] PIT 전체 종목 {h}거래일 상승 비율: {np.nanmean(u) * 100:.1f}%")
    print(f"\n→ {ES.OUT / 'mom_flip.csv'}")


if __name__ == "__main__":
    main()
