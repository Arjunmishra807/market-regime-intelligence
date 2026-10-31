"""Plotly figures."""
from __future__ import annotations
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import REGIMES, REGIME_COLORS, EW_LEVELS

LAYOUT = dict(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
              margin=dict(l=10, r=10, t=40, b=10), font=dict(size=12))


def _fig(fig, h=320, title=None):
    fig.update_layout(**LAYOUT, height=h, title=title); return fig


def gauge(score: float, title="Early Warning Score"):
    steps, lo = [], 0
    for hi, _, col in EW_LEVELS:
        steps.append(dict(range=[lo, hi], color=col)); lo = hi
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, title=dict(text=title),
                                 gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="white", thickness=.25), steps=steps)))
    return _fig(fig, 260)


def overview(price, vix, vol, dd, labels, name):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.03, row_heights=[.4, .2, .2, .2])
    fig.add_scatter(x=price.index, y=price, name=name, line=dict(color="#4da3ff"), row=1, col=1)
    if vix is not None: fig.add_scatter(x=vix.index, y=vix, name="VIX", line=dict(color="#e0b23a"), row=2, col=1)
    fig.add_scatter(x=vol.index, y=vol * 100, name="Realised vol 20D %", line=dict(color="#c77dff"), row=3, col=1)
    fig.add_scatter(x=dd.index, y=dd * 100, name="Drawdown %", fill="tozeroy", line=dict(color="#e07b39"), row=4, col=1)
    fig.update_layout(showlegend=False)
    return _fig(fig, 620)


def candles(df, name):
    fig = go.Figure(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close))
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _fig(fig, 320, f"{name} - candlestick (6M)")


def regime_timeline(labels: pd.Series):
    fig = go.Figure()
    for r in REGIMES:
        s = labels[labels == r]
        fig.add_scatter(x=s.index, y=[r] * len(s), mode="markers", name=r,
                        marker=dict(color=REGIME_COLORS[r], symbol="square", size=5))
    fig.update_yaxes(categoryorder="array", categoryarray=REGIMES[::-1])
    return _fig(fig, 280, "Regime timeline")


def prob_area(F: pd.DataFrame):
    fig = go.Figure()
    for r in REGIMES:
        fig.add_scatter(x=F.index, y=F[r], stackgroup="one", name=r, line=dict(width=0), fillcolor=REGIME_COLORS[r])
    fig.update_yaxes(range=[0, 1])
    return _fig(fig, 300, "Regime probabilities (filtered)")


def heat(df: pd.DataFrame, title, fmt=".2f"):
    fig = go.Figure(go.Heatmap(z=df.values, x=list(df.columns), y=list(df.index), colorscale="RdBu_r",
                               zmid=0 if df.values.min() < 0 else None, text=np.round(df.values, 2),
                               texttemplate="%{text}"))
    return _fig(fig, 380, title)


def stress_chart(stress: pd.Series):
    fig = go.Figure(); fig.add_scatter(x=stress.index, y=stress, line=dict(color="#e07b39"), name="Stress")
    for lvl, col in ((40, "#c9a227"), (60, "#e08a1e"), (80, "#d9534f")):
        fig.add_hline(y=lvl, line_dash="dot", line_color=col)
    fig.update_yaxes(range=[0, 100])
    return _fig(fig, 300, "Market Stress Index")


def bars(d: dict, title, pct=True, color="#4da3ff"):
    k, v = list(d.keys()), list(d.values())
    fig = go.Figure(go.Bar(x=v, y=k, orientation="h", marker_color=color,
                           text=[f"{x*100:.0f}%" if pct else f"{x:.1f}" for x in v], textposition="auto"))
    fig.update_yaxes(autorange="reversed")
    return _fig(fig, 280, title)


def contrib_bar(c: pd.Series, title):
    c = c.reindex(c.abs().sort_values(ascending=False).index)[:12][::-1]
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h",
                           marker_color=["#d9534f" if v > 0 else "#2e8b57" for v in c.values]))
    return _fig(fig, 380, title)


def calib(y, p, bins=8):
    m = y.notna() & p.notna(); d = pd.DataFrame({"y": y[m], "p": p[m]})
    d["b"] = pd.qcut(d.p, bins, duplicates="drop"); g = d.groupby("b", observed=True).mean()
    fig = go.Figure(); fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot"), name="perfect")
    fig.add_scatter(x=g.p, y=g.y, mode="lines+markers", name="model")
    fig.update_layout(xaxis_title="predicted", yaxis_title="observed frequency")
    return _fig(fig, 320, "Calibration (out-of-sample)")


def network(corr: pd.DataFrame, vols: pd.Series, delta: pd.DataFrame, thr=0.3):
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(corr.columns)
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            if abs(corr.loc[a, b]) >= thr: G.add_edge(a, b, w=float(corr.loc[a, b]))
    pos = nx.circular_layout(G); fig = go.Figure()
    for a, b, d in G.edges(data=True):
        broke = abs(delta.loc[a, b]) > 0.25
        fig.add_scatter(x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines", hoverinfo="skip",
                        line=dict(width=1 + 6 * abs(d["w"]), color="#ffd166" if broke else ("#4da3ff" if d["w"] > 0 else "#ef476f")))
    strength = pd.Series({n: sum(abs(d["w"]) for _, _, d in G.edges(n, data=True)) for n in G.nodes})
    vs = (vols / vols.max()).reindex(list(G.nodes)).fillna(.2)
    fig.add_scatter(x=[pos[n][0] for n in G], y=[pos[n][1] for n in G], mode="markers+text", text=list(G.nodes),
                    textposition="top center", hovertext=[f"{n}: strength {strength[n]:.2f}" for n in G],
                    marker=dict(size=20 + 40 * vs.values, color=strength.reindex(list(G.nodes)).values, colorscale="Viridis"))
    fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return _fig(fig, 520, "Cross-market network (blue +corr, red -corr, gold = correlation shift)"), strength
