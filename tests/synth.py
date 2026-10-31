import numpy as np, pandas as pd
def synth(n=3800, seed=1):
    rng = np.random.default_rng(seed); idx = pd.bdate_range("2008-01-01", periods=n)
    reg = np.zeros(n, int)
    for t in range(1, n): reg[t] = reg[t-1] if rng.random() > .02 else rng.integers(0, 3)
    vol = np.array([.007, .012, .03])[reg]; mu = np.array([.0007, 0, -.002])[reg]
    d = {}
    base = rng.normal(mu, vol)
    for k, (nm, beta) in enumerate({"S&P 500": 1, "NASDAQ": 1.2, "Dow Jones": .9, "Russell 2000": 1.1, "Gold": .1,
                                   "WTI Oil": .5, "Bitcoin": .6, "EUR/USD": .2, "DXY": -.2, "US 10Y": .15}.items()):
        d[nm] = 100 * np.exp(np.cumsum(beta * base + rng.normal(0, .006, n)))
    d["VIX"] = 15 + 400 * pd.Series(vol, idx).rolling(5, min_periods=1).mean().values + rng.normal(0, 1, n)
    return pd.DataFrame(d, index=idx)
