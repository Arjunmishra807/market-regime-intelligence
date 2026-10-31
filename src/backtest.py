"""Walk-forward (expanding window, purged by horizon) evaluation. No random splits."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.warning import make_labels, new_rf
from config import HORIZONS

EPISODES = [("GFC", "2008-09-01", "2009-03-09"), ("2011 selloff", "2011-07-22", "2011-10-03"),
            ("2015-16 China/oil", "2015-08-10", "2016-02-11"), ("2018 Q4", "2018-10-03", "2018-12-24"),
            ("COVID crash", "2020-02-19", "2020-03-23"), ("2022 bear", "2022-01-03", "2022-10-12")]


def walk_forward(X, y, make_model, h, min_train=750, step=250):
    oof = pd.Series(np.nan, index=X.index)
    for s in range(min_train, len(X), step):
        ytr = y.iloc[:s - h]; m = ytr.notna()      # purge h rows so no label peeks past s
        if m.sum() < 200 or ytr[m].nunique() < 2: continue
        mdl = make_model().fit(X.iloc[:s - h][m.values], ytr[m])
        oof.iloc[s:s + step] = mdl.predict_proba(X.iloc[s:s + step])[:, 1]
    return oof


def metrics(y, p, thr):
    m = y.notna() & p.notna(); y, p = y[m], p[m]
    if len(y) < 50 or y.nunique() < 2: return {}
    yhat = (p >= thr).astype(int); fp = ((yhat == 1) & (y == 0)).sum(); tn = ((yhat == 0) & (y == 0)).sum()
    return {"n_test": len(y), "base_rate": y.mean(), "ROC-AUC": roc_auc_score(y, p),
            "PR-AUC": average_precision_score(y, p), "Brier": brier_score_loss(y, p),
            "Precision": precision_score(y, yhat, zero_division=0), "Recall": recall_score(y, yhat, zero_division=0),
            "F1": f1_score(y, yhat, zero_division=0), "False-alarm rate": fp / max(fp + tn, 1),
            "Accuracy": (yhat == y).mean()}


def episodes(p: pd.Series, price: pd.Series, thr: float):
    rows = []
    for name, a, b in EPISODES:
        a, b = pd.Timestamp(a), pd.Timestamp(b); seg = price[a:b]
        pp = p[a - pd.Timedelta(days=15):b].dropna()
        if seg.empty or pp.empty:
            rows.append(dict(episode=name, status="no out-of-sample coverage")); continue
        trough = seg.idxmin(); pre = pp[:trough]; hit = pre[pre >= thr]
        lead = len(price[hit.index[0]:trough]) - 1 if len(hit) else None
        rows.append(dict(episode=name, status="detected" if len(hit) else "missed", trough=trough.date(),
                         first_alarm=hit.index[0].date() if len(hit) else None, lead_trading_days=lead,
                         peak_prob=float(pre.max()) if len(pre) else np.nan))
    return pd.DataFrame(rows)


def evaluate_all(X, price, groups, regime_filtered, h, thr):
    y = make_labels(price, h, HORIZONS[h]).reindex(X.index)
    mk = [c for g in ("Market", "Volatility") for c in groups[g]]
    vo = [c for c in ("vol20", "vol_accel", "vix") if c in X]
    models = {
        "Vol-only logistic (baseline)": (vo, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))),
        "RF market+vol": (mk, new_rf), "RF full (multi-source)": (list(X.columns), new_rf),
        "HistGB full": (list(X.columns), lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=.05, max_iter=150)),
    }
    oof = {n: walk_forward(X[c], y, mf, h) for n, (c, mf) in models.items()}
    risk = regime_filtered[["HIGH VOLATILITY", "BEAR", "CRISIS"]].sum(axis=1).reindex(X.index)
    oof["GMM regime risk (in-sample fit)"] = risk.where(oof["RF full (multi-source)"].notna())
    table = pd.DataFrame({n: metrics(y, p, thr) for n, p in oof.items()}).T
    return oof, table, y
