"""Multi-scale market, volatility, distributional, cross-asset and macro features.
All rolling windows are backward-looking only (no look-ahead)."""
from __future__ import annotations
import numpy as np, pandas as pd


def pct_rank(s: pd.Series, win: int = 756, minp: int = 120) -> pd.Series:
    """Causal percentile of today's value vs the previous `win` observations."""
    return s.rolling(win, min_periods=minp).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)


def _conc(R: pd.DataFrame, w: int) -> pd.Series:
    """Average pairwise correlation proxy: rho ~ (n*var(mean z)-1)/(n-1)."""
    z = R / R.rolling(w).std()
    n = z.notna().sum(axis=1)
    v = z.mean(axis=1).rolling(w).var()
    return ((n * v - 1) / (n - 1).clip(lower=1)).clip(-1, 1)


def build_features(panel: pd.DataFrame, target: str, macro):
    panel = panel.ffill()
    panel = panel.loc[panel[target].dropna().index]
    p = panel[target]; r = np.log(p).diff(); ann = np.sqrt(252)
    f = pd.DataFrame(index=p.index)
    f["ret1"] = r
    for n in (5, 20, 60):
        f[f"ret{n}"] = p.pct_change(n)
    for n in (5, 20, 60):
        f[f"vol{n}"] = r.rolling(n).std() * ann
    f["vol_accel"] = f.vol5 / f.vol20 - 1
    f["vol_trend"] = f.vol20 / f.vol60 - 1
    f["dd"] = p / p.rolling(252, min_periods=20).max() - 1
    f["skew60"] = r.rolling(60).skew(); f["kurt60"] = r.rolling(60).kurt()
    f["cvar60"] = r.rolling(60).apply(lambda x: x[x <= np.quantile(x, .05)].mean(), raw=True)
    f["tail_freq20"] = (r < -0.02).astype(float).rolling(20).mean()
    d = p.diff(); g = d.clip(lower=0).rolling(14).mean(); l = (-d.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + g / (l + 1e-12))
    macd = p.ewm(span=12).mean() - p.ewm(span=26).mean()
    f["macd_h"] = (macd - macd.ewm(span=9).mean()) / p
    ma, sd = p.rolling(20).mean(), p.rolling(20).std()
    f["bb_pct"] = (p - (ma - 2 * sd)) / (4 * sd + 1e-12)
    f["mom20"] = p / ma - 1
    f["ma_gap200"] = p / p.rolling(200, min_periods=50).mean() - 1
    f["sharpe60"] = r.rolling(60).mean() / (r.rolling(60).std() + 1e-12) * ann
    if "VIX" in panel and target != "VIX":
        v = panel["VIX"]; f["vix"] = v; f["vix_chg5"] = v.pct_change(5)
    others = [c for c in panel.columns if c != "VIX" and panel[c].notna().mean() > .5]
    R = np.log(panel[others].where(panel[others] > 0)).diff()
    f["corr20"], f["corr60"] = _conc(R, 20), _conc(R, 60)
    f["corr_break"] = f.corr20 - f.corr60
    for nm in ("Gold", "WTI Oil", "DXY", "US 10Y"):
        if nm in panel and nm != target:
            f[f"{nm}_mom20"] = panel[nm].pct_change(20)
    if macro is not None and len(macro) and "curve" in macro:
        f["curve"] = macro["curve"].reindex(f.index, method="ffill")
    groups = {
        "Market": ["ret1", "ret5", "ret20", "ret60", "dd", "skew60", "kurt60", "cvar60", "tail_freq20",
                   "rsi14", "macd_h", "bb_pct", "mom20", "ma_gap200", "sharpe60"],
        "Volatility": ["vol5", "vol20", "vol60", "vol_accel", "vol_trend", "vix", "vix_chg5"],
        "Cross-Asset": ["corr20", "corr60", "corr_break", "Gold_mom20", "WTI Oil_mom20"],
        "Macro": ["curve", "DXY_mom20", "US 10Y_mom20"],
    }
    groups = {k: [c for c in v if c in f.columns] for k, v in groups.items()}
    return f.replace([np.inf, -np.inf], np.nan), groups
