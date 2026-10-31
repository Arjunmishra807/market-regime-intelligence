"""Adaptive Multimodal Real-Time Market Regime Intelligence & Early-Warning System."""
import logging
import numpy as np, pandas as pd, streamlit as st
import streamlit.components.v1 as components
from config import *
from src import viz
from src.data_loader import load_prices, load_macro, load_ohlc
from src.news import fetch_news
from src.pipeline import build_system
from src.warning import (fuse, early_warning, label_ew, tree_probs, confidence, detect_drift,
                         explain, similar_days)
from src.backtest import evaluate_all, episodes

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Market Regime Intelligence", page_icon="📡", layout="wide")

# ---------------- sidebar ----------------
sb = st.sidebar; sb.title("Controls")
names = dict(ASSETS)
target = sb.selectbox("Asset / Index", [n for n in names if n != "VIX"], 0)
custom = sb.text_input("Add custom ticker (optional)", "").strip().upper()
if custom:
    names[custom] = custom; target = custom
window = sb.selectbox("Time period (display)", ["6M", "1Y", "3Y", "5Y", "10Y", "Max"], 2)
horizon = sb.selectbox("Forecast horizon (days)", list(HORIZONS), 2)
sens = sb.radio("Warning mode", list(SENSITIVITY), 1)
thr_alert = sb.slider("Alert threshold", 20, 90, 60)
sb.caption("Modality weight priors (adapted dynamically)")
uw = {"News": sb.slider("News", 0.0, 2.0, 1.0, .1), "Market": sb.slider("Market", 0.0, 2.0, 1.0, .1),
      "Macro": sb.slider("Macro", 0.0, 2.0, 1.0, .1), "Cross-Asset": sb.slider("Cross-Asset", 0.0, 2.0, 1.0, .1),
      "Sentiment": 1.0, "Volatility": 1.0}
auto = sb.checkbox("Auto refresh (5 min)")
if sb.button("Manual refresh / retrain"):
    st.cache_data.clear(); st.cache_resource.clear()
if auto:
    components.html("<script>setTimeout(()=>window.parent.location.reload(),300000)</script>", height=0)

# ---------------- data (cached) ----------------
@st.cache_data(ttl=900, show_spinner="Loading market data...")
def get_prices(items): return load_prices(dict(items))
@st.cache_data(ttl=3600, show_spinner="Loading macro data...")
def get_macro(): return load_macro()
@st.cache_data(ttl=600)
def get_news(): return fetch_news()
@st.cache_data(ttl=900)
def get_ohlc(t): return load_ohlc(t)
@st.cache_resource(show_spinner="Fitting regime + early-warning models (first load only)...")
def get_system(_panel, _macro, tgt, last_date, n): return build_system(_panel, _macro, tgt)

panel, p_status, p_ts = get_prices(tuple(names.items()))
macro, m_status, m_last = get_macro()
news_df, news, n_status = get_news()
if panel.empty or target not in panel:
    st.error("Live data temporarily unavailable and no cached data exists. Nothing is displayed rather than fabricating values.")
    st.stop()
S = get_system(panel, macro, target, str(panel.index[-1].date()), len(panel))
f, X, price, eng = S["f"], S["X"], S["price"], S["eng"]
cur = eng.current()

# ---------------- live computation ----------------
x = X.iloc[[-1]]; probs, stds = {}, {}
for h, m in S["models"].items():
    probs[h], stds[h] = tree_probs(m["model"], x)
stress = S["stress"].dropna(); st_now = float(stress.iloc[-1]); st_prev = float(stress.iloc[-2]); st_5 = float(stress.iloc[-6])
credit_rank = None
if len(macro) and "credit" in macro and macro["credit"].notna().sum() > 120:
    c = macro["credit"].dropna(); credit_rank = float((c.iloc[:-1] < c.iloc[-1]).mean())
macro_fresh = 1.0 if m_status != "UNAVAILABLE" else 0.0
mscores, mweights, fused = fuse(S["pr"].iloc[-1], news, credit_rank, macro_fresh, uw)
ew = early_warning(st_now, probs, fused, sens); ew_lab, ew_col = label_ew(ew)
drift, drift_cols = detect_drift(X)
avail = np.mean([np.isfinite(v) for v in mscores.values()])
conf, conf_lab = confidence(cur["entropy"], stds[horizon], drift, 0.5 + 0.5 * avail)
status_txt = {"DELAYED": "DELAYED (free feed)", "CACHED": "CACHED", "UNAVAILABLE": "UNAVAILABLE"}
days = {"6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "10Y": 2520, "Max": len(X)}[window]

# ---------------- header ----------------
st.title("📡 Market Regime Intelligence & Early-Warning")
st.caption(f"Prices: **{status_txt[p_status]}** · last updated {p_ts} · last bar {panel.index[-1].date()} | "
           f"Macro: **{m_status}** (latest available obs: {m_last or 'n/a'}) | "
           f"News: **{n_status}**" + ("" if news else " - Live news unavailable — using latest cached/available data."))
if p_status != "DELAYED": st.warning("Live data temporarily unavailable. Showing cached/latest available data.")
if drift: st.warning(f"MODEL DRIFT DETECTED in {', '.join(drift_cols)} - confidence reduced. Consider retraining (Manual refresh).")
if ew > 80: st.error(f"🚨 CRITICAL MARKET STRESS - score {ew:.0f}. Confidence {conf_lab}. See Explainability for drivers.")
elif ew > thr_alert: st.warning(f"⚠️ EARLY WARNING - score {ew:.0f} ({ew_lab}). Confidence {conf_lab}. See Explainability for drivers.")

tabs = st.tabs(["Overview", "Regimes", "Early Warning", "Multimodal", "News & Events", "Network", "Explainability", "Backtest"])

# ---------------- 1 overview ----------------
with tabs[0]:
    c = st.columns(6)
    c[0].metric("Regime", cur["regime"]); c[1].metric("Regime prob.", f"{cur['prob']:.0%}")
    c[2].metric("Stress Index", f"{st_now:.0f}/100", f"{st_now - st_prev:+.1f}")
    c[3].metric("Early Warning", f"{ew:.0f}", ew_lab, delta_color="off")
    c[4].metric("Risk level", ew_lab); c[5].metric("Confidence", conf_lab, f"uncertainty {100*(1-conf):.0f}%", delta_color="off")
    s5 = st_now - st_5
    st.write(f"**Stress status:** {'RISING STRESS' if s5 > 5 else 'FALLING STRESS' if s5 < -5 else 'STABLE'} "
             f"(5-day change {s5:+.1f}; previous day {st_prev:.0f}). Previous regime: {cur['prev']}, in current regime {cur['duration']} days.")
    sl = slice(-days, None)
    st.plotly_chart(viz.overview(price.iloc[sl], panel["VIX"].reindex(X.index).iloc[sl] if "VIX" in panel else None,
                                 f.vol20.iloc[sl], f.dd.iloc[sl], eng.labels.iloc[sl], target), use_container_width=True)
    st.plotly_chart(viz.regime_timeline(eng.labels.iloc[sl]), use_container_width=True)
    with st.expander("Candlestick (6M)"):
        o = get_ohlc(names.get(target, target))
        st.plotly_chart(viz.candles(o, target), use_container_width=True) if len(o) else st.info("OHLC unavailable.")

# ---------------- 2 regimes ----------------
with tabs[1]:
    st.write(f"**{cur['regime']}** ({cur['prob']:.0%}) · previous {cur['prev']} · duration {cur['duration']}d · "
             f"P(leave next day) {cur['p_leave']:.0%} · most likely destination **{cur['dest']}** · "
             f"transition flag: {'YES' if cur['transition'] else 'NO'} (5-day risk shift {cur['risk_shift_5d']:+.2f}). "
             "Warning lead time: *Not yet validated* (see Backtest).")
    st.plotly_chart(viz.prob_area(eng.filtered.iloc[-days:]), use_container_width=True)
    st.plotly_chart(viz.regime_timeline(eng.labels.iloc[-days:]), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(viz.heat(eng.T, "Transition matrix (empirical, daily)"), use_container_width=True)
    b.write("**Regime profile (validation of latent-state naming)**"); b.dataframe(eng.stats.round(3))
    st.caption("States come from a 6-component Gaussian mixture fit on return/vol/drawdown/skew/VIX/momentum; names are assigned by "
               "ordering each state's risk profile - check the table above that profiles are sensible. Fit uses the full history window (in-sample).")

# ---------------- 3 early warning ----------------
with tabs[2]:
    a, b = st.columns([1, 2]); a.plotly_chart(viz.gauge(ew), use_container_width=True)
    with b:
        cc = st.columns(4)
        for col, (h, m) in zip(cc, S["models"].items()):
            col.metric(f"{h}D warning", f"{probs[h]:.0%}", f"base {m['base']:.0%} · ±{stds[h]:.0%}", delta_color="off")
        st.caption("Warning probability = model P(close falls ≥ threshold at some point in the next h days): " +
                   ", ".join(f"{h}D ≤ {t:.1%}" for h, t in HORIZONS.items()) + ". Compare with the historical base rate shown under each value. "
                   "Probabilities are uncalibrated unless the Backtest calibration plot says otherwise.")
        st.write(f"Prediction: **{ew_lab}** · Confidence **{conf_lab}** · Uncertainty **{100*(1-conf):.0f}%**")
    st.plotly_chart(viz.stress_chart(stress.iloc[-days:]), use_container_width=True)
    st.write("Historical false-alarm statistics and lead times are computed on the **Backtest** tab (walk-forward).")

# ---------------- 4 multimodal ----------------
with tabs[3]:
    a, b = st.columns(2)
    a.plotly_chart(viz.bars(mweights, "Adaptive modality weights"), use_container_width=True)
    b.plotly_chart(viz.bars({k: v for k, v in mscores.items() if np.isfinite(v)}, "Modality risk scores (0-100)", pct=False, color="#e07b39"), use_container_width=True)
    st.write(f"Fused multimodal risk score: **{fused:.1f}/100**")
    miss = [k for k, v in mscores.items() if not np.isfinite(v)]
    if miss: st.info(f"Unavailable right now (weight set to 0, not fabricated): {', '.join(miss)}")
    st.caption("Weight = user prior × availability/freshness × signal salience (0.5 + |score−50|/50), normalised. "
               "Heuristic reliability model, not yet validated against a learned gating network.")

# ---------------- 5 news ----------------
with tabs[4]:
    if news is None: st.warning("Live news unavailable — using latest cached/available data.")
    else:
        c = st.columns(5)
        c[0].metric("News sentiment", f"{news['sentiment']:+.2f}"); c[1].metric("Volume", news["volume"])
        c[2].metric("Negative ratio", f"{news['neg_ratio']:.0%}"); c[3].metric("Positive ratio", f"{news['pos_ratio']:.0%}")
        c[4].metric("Max event severity", f"{news['max_sev']}/5")
        st.dataframe(news_df[["time", "source", "headline", "sentiment", "event", "severity"]], use_container_width=True, hide_index=True)
        import plotly.express as px
        fig = px.scatter(news_df, x="time", y="severity", color="event", hover_name="headline")
        st.plotly_chart(viz._fig(fig, 300, "Event timeline"), use_container_width=True)
        st.caption("Lexicon-based sentiment and keyword event tagging on headlines only: coarse, not validated, live-only (no historical news in the trained models).")

# ---------------- 6 network ----------------
with tabs[5]:
    cols = [c for c in panel.columns if panel[c].notna().mean() > .5]
    R = np.log(panel[cols].ffill().where(panel[cols] > 0)).diff().dropna(how="all")
    now_c, old_c = R.iloc[-60:].corr(), R.iloc[-310:-60].corr()
    fig, strength = viz.network(now_c, R.iloc[-20:].std() * np.sqrt(252), now_c - old_c)
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    a.write("**Stress transmitters** (connection strength × volatility)")
    a.dataframe((strength * R.iloc[-20:].std().reindex(strength.index)).sort_values(ascending=False).round(4).rename("score"))
    d = (now_c - old_c).where(np.triu(np.ones(now_c.shape), 1).astype(bool)).stack()
    b.write("**Largest correlation shifts (60D vs prior 250D)**")
    b.dataframe(d.reindex(d.abs().sort_values(ascending=False).index).head(8).round(2).rename("Δcorr"))
    st.plotly_chart(viz.heat(now_c, "Rolling 60D correlation"), use_container_width=True)
    st.caption(f"Average pairwise correlation (20D): {f.corr20.iloc[-1]:.2f} vs 60D: {f.corr60.iloc[-1]:.2f}.")

# ---------------- 7 explainability ----------------
with tabs[6]:
    M = S["models"][horizon]
    contrib, method = explain(M["model"], x, M["X"], M["y"])
    st.plotly_chart(viz.contrib_bar(contrib, f"Feature contributions to {horizon}D warning - {method}"), use_container_width=True)
    gmap = {c: g for g, cs in S["groups"].items() for c in cs}
    mc = contrib.abs().groupby(contrib.index.map(lambda c: gmap.get(c, "Other"))).sum()
    st.plotly_chart(viz.bars((mc / mc.sum()).to_dict(), "Modality contribution (model features)"), use_container_width=True)
    up, dn = contrib.sort_values(ascending=False), contrib.sort_values()
    st.subheader("WHY IS THE SYSTEM WARNING ME?")
    st.write(f"Early Warning Score **{ew:.0f}** ({ew_lab}). Strongest risk-raising factors: "
             f"{', '.join(up.index[:3])}. Factors reducing risk: {', '.join(dn.index[:2])}. "
             f"Stress index {st_now:.0f}/100; highest-weighted modality: **{max(mweights, key=mweights.get)}**. "
             f"The {horizon}D event probability is {probs[horizon]:.0%} versus a historical base rate of {M['base']:.0%}. "
             "This describes model behaviour, not a certain outcome.")
    st.write("**Historical days most similar to today** (nearest neighbours in standardised feature space):")
    st.dataframe(similar_days(X, price).round(3))

# ---------------- 8 backtest ----------------
with tabs[7]:
    st.write("Walk-forward, expanding window, purged by the horizon length. No random splits. Results are computed on your data when you click - until then: **Not yet validated.**")
    bh = st.selectbox("Horizon", list(HORIZONS), 2, key="bh"); bt = st.slider("Alarm threshold", .1, .8, .3, .05)
    @st.cache_resource(show_spinner="Running walk-forward evaluation...")
    def run_bt(_S, tgt, last, n, h, thr): return evaluate_all(_S["X"], _S["price"], _S["groups"], _S["eng"].filtered, h, thr)
    if st.button("Run walk-forward evaluation"):
        st.session_state["bt"] = (bh, bt, run_bt(S, target, str(panel.index[-1].date()), len(panel), bh, bt))
    if "bt" in st.session_state:
        h_, t_, (oof, table, y) = st.session_state["bt"]
        st.write(f"Event: close falls ≥ {abs(HORIZONS[h_]):.1%} within {h_} trading days · threshold {t_:.2f}")
        st.dataframe(table.round(3), use_container_width=True)
        st.plotly_chart(viz.calib(y, oof["RF full (multi-source)"]), use_container_width=True)
        st.write("**Historical stress episodes (RF full, out-of-sample only)**")
        st.dataframe(episodes(oof["RF full (multi-source)"], price, t_), hide_index=True)
        st.caption("Notes: XGBoost/LSTM/Transformer variants are not included (HistGB is the boosting stand-in). The GMM row is fit in-sample and is optimistic. "
                   "News/macro-surprise modalities have no history, so the 'multi-source' model uses market, volatility, cross-asset and yield-curve features. "
                   "Dot-com is outside the free data window. Episode lead times are single-episode anecdotes, not statistical proof.")

st.divider(); st.caption(DISCLAIMER)
