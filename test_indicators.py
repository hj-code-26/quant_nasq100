"""add_indicators 자체 점검 — 알려진 값과 대조."""
import numpy as np, pandas as pd
from autotrade import add_indicators

n = 120
rng = np.random.default_rng(0)
c = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6})
df.index = pd.date_range("2024-01-01", periods=n, tz="UTC")
d = add_indicators(df.copy())

assert np.isclose(d["SMA_20"].iloc[-1], c.iloc[-20:].mean())
assert d["SMA_50"].iloc[:49].isna().all() and d["SMA_50"].notna().iloc[49]
assert np.isclose(d["MACD_12_26_9"].iloc[-1],
                  (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()).iloc[-1])
assert np.isclose(d["MACDh_12_26_9"].iloc[-1],
                  d["MACD_12_26_9"].iloc[-1] - d["MACDs_12_26_9"].iloc[-1])
assert 0 <= d["RSI_14"].iloc[-1] <= 100
assert 0 <= d["STOCHk_14_3_3"].iloc[-1] <= 100
mid, sd = c.iloc[-20:].mean(), c.iloc[-20:].std(ddof=1)
assert np.isclose(d["BBU_20_2.0_2.0"].iloc[-1], mid + 2 * sd)
assert np.isclose(d["BBP_20_2.0_2.0"].iloc[-1], (c.iloc[-1] - (mid - 2 * sd)) / (4 * sd))

# 단조 상승이면 RSI=100, 단조 하락이면 RSI=0
up = pd.Series(np.arange(60, dtype=float) + 100)
u = add_indicators(pd.DataFrame({"open": up, "high": up, "low": up, "close": up, "volume": 1.0}))
assert np.isclose(u["RSI_14"].iloc[-1], 100), u["RSI_14"].iloc[-1]
dn = pd.Series(200 - np.arange(60, dtype=float))
w = add_indicators(pd.DataFrame({"open": dn, "high": dn, "low": dn, "close": dn, "volume": 1.0}))
assert np.isclose(w["RSI_14"].iloc[-1], 0), w["RSI_14"].iloc[-1]
print("indicators ok")
