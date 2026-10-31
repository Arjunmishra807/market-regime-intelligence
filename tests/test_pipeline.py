"""Run: python -m tests.test_pipeline  (offline, synthetic data - checks plumbing, NOT market validity)"""
import numpy as np, pandas as pd
from tests.synth import synth
from src.pipeline import build_system
from src.warning import *
from src.backtest import evaluate_all, episodes

panel = synth(); S = build_system(panel, None, "S&P 500")
cur = S["eng"].current(); print("regime", cur["regime"], round(cur["prob"], 2), "dur", cur["duration"], "leave", round(cur["p_leave"], 2))
print(S["eng"].stats.round(3))
x = S["X"].iloc[[-1]]; probs = {}; stds = {}
for h, m in S["models"].items(): probs[h], stds[h] = tree_probs(m["model"], x)
print("probs", {h: round(p, 2) for h, p in probs.items()}, "base", {h: round(m['base'], 2) for h, m in S["models"].items()})
last = S["pr"].iloc[-1]; sc, w, fused = fuse(last, None, None, 1.0, {"Market": 1, "News": 1, "Macro": 1, "Cross-Asset": 1, "Sentiment": 1, "Volatility": 1})
print({k: round(v, 1) for k, v in sc.items()}, {k: round(v, 2) for k, v in w.items()}, round(fused, 1))
st = float(S["stress"].iloc[-1]); ew = early_warning(st, probs, fused, "Balanced"); print("stress", round(st, 1), "EW", round(ew, 1), label_ew(ew))
print("drift", detect_drift(S["X"])); c, lab = explain(S["models"][5]["model"], x, S["models"][5]["X"], S["models"][5]["y"])
print(lab); print(c.sort_values().tail(3)); print(similar_days(S["X"], S["price"]).head(3))
oof, table, y = evaluate_all(S["X"], S["price"], S["groups"], S["eng"].filtered, 5, .3)
print(table.round(3)); print(episodes(oof["RF full (multi-source)"], S["price"], .3))
assert not np.isnan(ew); print("PIPELINE OK")
