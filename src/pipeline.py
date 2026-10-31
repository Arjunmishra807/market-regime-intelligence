"""Assemble features -> regimes -> stress -> event models (called once, cached by the app)."""
from __future__ import annotations
import numpy as np, pandas as pd
from src.features import build_features
from src.regimes import RegimeEngine
from src.warning import stress_index, train_event_models


def build_system(panel: pd.DataFrame, macro, target: str) -> dict:
    f_raw, groups = build_features(panel, target, macro)
    core = [c for c in ("ret60", "vol60", "skew60", "cvar60", "corr60") if c in f_raw]
    f = f_raw.dropna(subset=core)
    X = f.ffill().fillna(0.0)
    price = panel[target].ffill().reindex(X.index)
    stress, pr = stress_index(f)
    eng = RegimeEngine().fit(X)
    models = train_event_models(X, price)
    return dict(f=f, X=X, groups=groups, price=price, stress=stress, pr=pr, eng=eng, models=models)
