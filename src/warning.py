"""Stress index, forward-event models, adaptive fusion, early-warning score,
uncertainty, drift detection and explainability."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from config import HORIZONS, EW_LEVELS, SENSITIVITY
from src.features import pct_rank

STRESS_W = {"vol": .20, "dd": .20, "tail": .15, "vix": .20, "corr": .15, "curve": .10}


def rank_frame(f: pd.DataFrame) -> pd.DataFrame:
    src = {"vol": f.vol20, "dd": -f.dd, "tail": -f.cvar60, "corr": f.corr60,
           "ret_neg": -f.ret20, "vacc": f.vol_accel}
    if "vix" in f: src["vix"] = f.vix
    if "curve" in f: src["curve"] = -f.curve          # inversion = stress
    return pd.DataFrame({k: pct_rank(v) for k, v in src.items()})


def stress_index(f: pd.DataFrame):
    """0-100 Market Stress Index: weighted causal percentiles (weights renormalised if a part is missing).
    Liquidity proxy is not available from free daily data and is therefore omitted."""
    pr = rank_frame(f); cols = [c for c in STRESS_W if c in pr]
    w = pd.Series({c: STRESS_W[c] for c in cols})
    num = (pr[cols] * w).sum(axis=1); den = (pr[cols].notna() * w).sum(axis=1)
    return 100 * num / den.replace(0, np.nan), pr


def label_ew(score: float):
    for hi, name, col in EW_LEVELS:
        if score <= hi: return name, col
    return EW_LEVELS[-1][1], EW_LEVELS[-1][2]


def make_labels(price: pd.Series, h: int, thr: float) -> pd.Series:
    """1 if the minimum close over t+1..t+h is <= thr below close_t. NaN for the last h rows."""
    fmin = price[::-1].rolling(h).min()[::-1].shift(-1)
    return ((fmin / price - 1) <= thr).astype(float).where(fmin.notna())


def new_rf(n: int = 150):
    return RandomForestClassifier(n_estimators=n, max_depth=6, min_samples_leaf=30,
                                  max_features="sqrt", n_jobs=-1, random_state=0)


def train_event_models(X: pd.DataFrame, price: pd.Series) -> dict:
    out = {}
    for h, thr in HORIZONS.items():
        y = make_labels(price, h, thr).reindex(X.index); m = y.notna()
        out[h] = dict(model=new_rf().fit(X[m], y[m]), X=X[m], y=y[m], base=float(y[m].mean()), thr=thr)
    return out


def tree_probs(rf, x: pd.DataFrame) -> tuple[float, float]:
    """Mean and across-tree std of P(event) -> epistemic-style uncertainty."""
    ps = []
    for t in rf.estimators_:
        pp = t.predict_proba(x.values)[0]
        ps.append(pp[list(t.classes_).index(1.0)] if 1.0 in t.classes_ else 0.0)
    return float(np.mean(ps)), float(np.std(ps))


def _nm(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def fuse(pr_last: pd.Series, news: dict | None, credit_rank: float | None, macro_fresh: float, user_w: dict):
    """Adaptive modality fusion. Each modality yields a 0-100 risk score; reliability-adjusted weights =
    user prior x availability x shock-salience (0.5 + |score-50|/50), then normalised."""
    g = lambda k: pr_last.get(k, np.nan)
    news_comp = (1 - news["sentiment"]) / 2 if news else None
    s = {"Market": 100 * _nm([g("dd"), g("ret_neg")]),
         "Volatility": 100 * _nm([g("vol"), g("vacc"), g("vix")]),
         "Cross-Asset": 100 * _nm([g("corr")]),
         "Macro": 100 * _nm([g("curve"), credit_rank]),
         "News": news["risk"] if news else np.nan,
         "Sentiment": 100 * _nm([g("vix"), g("ret_neg"), news_comp])}
    raw = {}
    for m, v in s.items():
        if not np.isfinite(v): raw[m] = 0.0; continue
        fresh = macro_fresh if m == "Macro" else 1.0
        raw[m] = user_w.get(m, 1.0) * fresh * (0.5 + abs(v - 50) / 50)
    tot = sum(raw.values()) or 1.0
    w = {m: raw[m] / tot for m in s}
    fused = sum(w[m] * s[m] for m in s if np.isfinite(s[m]))
    return s, w, float(fused)


def early_warning(stress: float, probs: dict, fused: float, sens: str) -> float:
    pm = np.mean([probs[h] for h in (3, 5, 10)])
    prob_score = 100 * min(1.0, pm / 0.5)
    return float(np.clip((0.35 * stress + 0.35 * prob_score + 0.30 * fused) * SENSITIVITY[sens], 0, 100))


def confidence(regime_entropy: float, tree_std: float, drift: bool, avail: float):
    c = 0.5 * (1 - regime_entropy) + 0.5 * (1 - min(1.0, tree_std * 4))
    c *= avail * (0.7 if drift else 1.0)
    return float(c), ("HIGH" if c >= .65 else "MEDIUM" if c >= .4 else "LOW")


def detect_drift(X: pd.DataFrame, recent: int = 60, ref: int = 750):
    """Two-sample KS on key features: last 60 days vs preceding history. Daily data is autocorrelated,
    so p-values are optimistic; a strict cutoff and majority rule are used."""
    cols = [c for c in ("vol20", "ret20", "vix", "corr60", "skew60", "dd") if c in X]
    A, B = X[cols].iloc[-recent:], X[cols].iloc[-recent - ref:-recent]
    flagged = [c for c in cols if ks_2samp(A[c], B[c]).pvalue < 1e-3]
    return len(flagged) >= max(2, len(cols) // 2 + 1), flagged


def explain(rf, x: pd.DataFrame, Xtr: pd.DataFrame, ytr: pd.Series):
    """Per-feature contribution in probability points. SHAP TreeExplainer if installed, otherwise an
    approximation: importance x signed z-score (sign from feature/label correlation)."""
    try:
        import shap
        sv = shap.TreeExplainer(rf).shap_values(x)
        sv = np.array(sv[1] if isinstance(sv, list) else sv)
        sv = sv[0, :, 1] if sv.ndim == 3 else sv[0]
        return pd.Series(sv * 100, index=x.columns), "SHAP"
    except Exception:
        z = (x.iloc[0] - Xtr.mean()) / (Xtr.std() + 1e-9)
        sign = np.sign(Xtr.apply(lambda c: np.corrcoef(c, ytr)[0, 1] if c.std() > 0 else 0))
        c = pd.Series(rf.feature_importances_, index=x.columns) * z * sign
        return c * 100 / (c.abs().sum() + 1e-9) * 10, "approx (importance x z-score)"


def similar_days(X: pd.DataFrame, price: pd.Series, k: int = 5, exclude: int = 60):
    cols = [c for c in ("ret20", "vol20", "dd", "skew60", "vix", "corr60", "vol_accel") if c in X]
    Z = ((X[cols] - X[cols].mean()) / (X[cols].std() + 1e-9))
    d = np.sqrt(((Z.iloc[:-exclude] - Z.iloc[-1]) ** 2).sum(axis=1)).sort_values()
    picks = []
    for dt, dist in d.items():
        if all(abs((dt - q).days) > 30 for q, _, _ in picks):
            i = price.index.get_loc(dt); fut = price.iloc[i + 1:i + 11]
            picks.append((dt, dist, float(fut.min() / price.iloc[i] - 1) if len(fut) else np.nan))
        if len(picks) == k: break
    return pd.DataFrame(picks, columns=["date", "distance", "worst_10d_move"]).set_index("date")
