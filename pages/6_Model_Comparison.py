import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import helpers as H
from core import data as D
from core.models import MODEL_BLURBS, MODELS
from core.optimize import InfeasibleError, annualised_inputs, portfolio_stats
from core.performance import portfolio_returns, summarise
from core.risk import var_cvar_historical
from core.robustness import bootstrap_weights, stability_summary

H.page_setup("Model comparison", "⚖️")
st.title("⚖️ Model comparison & the estimation-noise lab")
st.write("Six ways to build a portfolio from the same data. They agree on the big picture and differ on how much they trust the numbers.")

cfg = H.sidebar_controls("mc")
chosen = st.sidebar.multiselect("Models", list(MODELS), default=list(MODELS))
if len(cfg["tickers"]) < 3 or not chosen:
    st.warning("Pick at least 3 stocks and one model.")
    st.stop()
try:
    mk = H.load_market(cfg["tickers"], cfg["mode"], cfg["demo"])
except D.DataError as exc:
    st.error(f"{exc}  Try Demo mode.")
    st.stop()
H.badge(mk["source"], mk["as_of"])
rf, tr, te = mk["rf"], mk["rets_train"], mk["rets_test"]
mu, cov = annualised_inputs(tr, cfg["shrink"], cfg["lw"])

W, errs = {}, []
for m in chosen:
    try:
        W[m] = MODELS[m](tr, cap=cfg["cap"], rf=rf, shrink=cfg["shrink"], ledoit_wolf=cfg["lw"])
    except InfeasibleError as exc:
        errs.append(str(exc))
if errs:
    st.error(errs[0])
    st.stop()
Wdf = pd.DataFrame(W)

st.markdown("### Weights")
st.dataframe(Wdf.style.format("{:.1%}").background_gradient(cmap="Blues"), use_container_width=True)
fig = go.Figure()
for m in Wdf:
    fig.add_bar(x=Wdf.index, y=Wdf[m], name=m)
fig.update_layout(barmode="group", yaxis_tickformat=".0%", height=380)
st.plotly_chart(fig, use_container_width=True)
st.caption("Cap handling: Max Sharpe, Min variance and Min CVaR respect the cap **exactly**. Risk parity, HRP and 1/N use clip-and-redistribute, a heuristic.")

st.markdown("### How they compare")
rows = {}
for m in Wdf:
    s = portfolio_stats(Wdf[m], mu, cov, rf)
    in_r = portfolio_returns(tr, Wdf[m], "daily")
    row = {"Volatility (fit period)": s.vol, "Worst-1% day (CVaR 99%)": var_cvar_historical(in_r, 0.99)[1], "Effective positions": s.eff_n,
           "Largest position": float(Wdf[m].max())}
    if len(te) > 10:
        o = summarise(portfolio_returns(te, Wdf[m], "none"), rf)
        row["Test-period return"], row["Test-period max drawdown"] = o["Total return"], o["Max drawdown"]
    rows[m] = row
cmp_ = pd.DataFrame(rows).T
fm = {c: "{:.1%}" for c in cmp_.columns}
fm["Effective positions"] = "{:.1f}"
st.dataframe(cmp_.style.format(fm), use_container_width=True)
st.caption("Test-period numbers cover only ~3 months and are not evidence of skill. For a real test use the Walk-forward back-test.")
with st.expander("What each model is for"):
    for m in Wdf:
        st.markdown(f"**{m}**: {MODEL_BLURBS[m]}")

st.markdown("---")
st.markdown("### 🧪 Estimation-noise lab")
st.write("Resample the history many times, re-run **Max Sharpe** each time, and watch how much the 'optimal' weights move. "
         "Then repeat with return-shrinkage and a stabilised covariance.")
c1, c2 = st.columns(2)
n_boot = c1.slider("Resamples", 20, 120, 40, 10)
shr = c2.slider("Shrinkage for the stabilised version", 0.0, 1.0, 0.8, 0.1)
if st.button("▶ Run noise lab", type="primary"):
    with st.spinner("Re-optimising on resampled histories…"):
        raw = bootstrap_weights(tr, rf, cfg["cap"], n_boot, 0.0, False, seed=1)
        stab = bootstrap_weights(tr, rf, cfg["cap"], n_boot, shr, True, seed=1)
    st.session_state["noise"] = (raw, stab)
if "noise" in st.session_state:
    raw, stab = st.session_state["noise"]
    if list(raw.columns) != list(tr.columns):
        st.info("Stock selection changed. Re-run the lab.")
    else:
        sr, ss = stability_summary(raw), stability_summary(stab)
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Raw estimates", "Shrunk + Ledoit-Wolf"))
        for col in raw.columns:
            fig.add_box(y=raw[col], name=col, row=1, col=1, showlegend=False)
            fig.add_box(y=stab[col], name=col, row=1, col=2, showlegend=False)
        fig.update_yaxes(tickformat=".0%", title_text="Optimal weight across resamples", col=1)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        k = st.columns(2)
        H.metric(k[0], "Average weight swing (raw)", H.pct(sr["avg_weight_swing"], 0))
        H.metric(k[1], "Average weight swing (stabilised)", H.pct(ss["avg_weight_swing"], 0), delta=f"{ss['avg_weight_swing'] - sr['avg_weight_swing']:+.0%}", )
        st.info("A *weight swing* of 50% means that, on average, half the portfolio would be replaced if you had seen a slightly different sample of history. "
                "Stabilised inputs reduce that churn, which is why they usually produce better out-of-sample behaviour and lower trading costs.")

H.quiz("mc", [
    ("Which method ignores expected returns entirely?", ["Max Sharpe", "Min variance", "Both"], 1, "Min variance uses only the covariance matrix, so it avoids return-estimation noise."),
    ("Wide boxes in the noise lab mean…", ["The stock is risky", "The optimiser's choice is unstable", "The cap is too tight"], 1, "The same data, resampled, gives very different 'optimal' weights."),
])
H.footer()
