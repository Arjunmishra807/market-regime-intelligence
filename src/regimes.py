"""Hybrid regime engine: Gaussian Mixture (latent states) + Markov transition
matrix + Bayesian forward filter (particle-filter-inspired belief update)."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from config import REGIMES, REGIME_COLS


class RegimeEngine:
    def __init__(self, seed: int = 0):
        self.k = len(REGIMES); self.seed = seed

    def fit(self, X: pd.DataFrame):
        cols = [c for c in REGIME_COLS if c in X.columns]
        A = X[cols]
        self.sc = StandardScaler().fit(A)
        self.gmm = GaussianMixture(self.k, covariance_type="full", n_init=3, reg_covar=1e-3,
                                   random_state=self.seed).fit(self.sc.transform(A))
        P = self.gmm.predict_proba(self.sc.transform(A))
        # Map latent states -> names by ordering their risk profile (data-driven; see stats table)
        st = pd.DataFrame({c: (P * A[c].values[:, None]).sum(0) / P.sum(0) for c in ["ret20", "vol20", "dd"]})
        z = (st - st.mean()) / (st.std() + 1e-9)
        order = (z.vol20 - z.ret20 - z.dd).sort_values().index.tolist()
        raw = pd.DataFrame(P[:, order], index=A.index, columns=REGIMES)
        hard = raw.rolling(3, min_periods=1).mean().values.argmax(1)
        T = np.full((self.k, self.k), 0.5)
        for a, b in zip(hard[:-1], hard[1:]):
            T[a, b] += 1
        T = T / T.sum(1, keepdims=True)
        pi = raw.mean().values + 1e-6
        b = raw.iloc[0].values; out = []
        for row in raw.values:
            b = (b @ T) * (row + 1e-6) / pi; b = b / b.sum(); out.append(b)
        self.filtered = pd.DataFrame(out, index=raw.index, columns=REGIMES)
        self.raw, self.T = raw, pd.DataFrame(T, index=REGIMES, columns=REGIMES)
        self.labels = self.filtered.idxmax(axis=1)
        s = X.loc[raw.index]
        self.stats = pd.DataFrame({
            r: {"share_%": 100 * (self.labels == r).mean(),
                **{c: s.loc[self.labels == r, c].mean() for c in
                   ["ret20", "vol20", "dd", "skew60", "kurt60", "vix"] if c in s}}
            for r in REGIMES}).T
        return self

    def current(self) -> dict:
        lab, F = self.labels, self.filtered
        cur = lab.iloc[-1]; diff = lab != cur
        run = int((diff)[::-1].values.argmax()) if diff.any() else len(lab)
        prev = lab[diff].iloc[-1] if diff.any() else "n/a"
        nxt = F.iloc[-1].values @ self.T.values
        ci = REGIMES.index(cur)
        dest = REGIMES[int(np.argmax(np.where(np.arange(self.k) == ci, -1, nxt)))]
        rk = F @ np.arange(self.k) / (self.k - 1)
        d5 = float(rk.iloc[-1] - rk.iloc[-6]) if len(rk) > 6 else 0.0
        p = F.iloc[-1]; ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(self.k))
        return dict(regime=cur, prob=float(p[cur]), prev=prev, duration=run, p_leave=float(1 - nxt[ci]),
                    dest=dest, risk_shift_5d=d5, entropy=ent,
                    transition=bool(run <= 5 or abs(d5) > 0.15), probs=p)
