import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import helpers as H
from core import data as D
from core.factors import factor_correlation, factor_summary, is_india_portfolio, run_ff5, run_market_model
from core.insights import factor_story
from core.optimize import InfeasibleError, annualised_inputs, max_sharpe
from core.performance import portfolio_returns

H.page_setup("Factor Analysis", "🔬")
st.title("🔬 Factor Analysis: what drives the portfolio?")
st.write("Stocks don't move randomly. Decades of research show a handful of *factors* explain much of their behaviour. "
         "This page measures how much of each factor your portfolio carries (its **exposure** or **beta**).")

cfg = H.sidebar_controls("fac")
freq = st.sidebar.radio("Data frequency", ["Daily", "Weekly"], help="Weekly smooths noise but has fewer observations.")
train_frac = st.sidebar.slider("Share of data used for training", 0.5, 0.9, 0.7, 0.05)

with st.expander("📖 The five factors in plain English", expanded=False):
    st.markdown(
        """
| Factor | Plain meaning | Positive beta means… |
|---|---|---|
| **Market (Mkt-RF)** | Return of the whole US stock market above cash | Moves with the market (1.0 = same as market) |
| **SMB** (Small minus Big) | Small companies vs large | Leans toward small companies |
| **HML** (High minus Low) | Cheap ('value') stocks vs expensive ('growth') | Leans toward cheap, out-of-favour stocks |
| **RMW** (Robust minus Weak) | Highly profitable firms vs weakly profitable | Leans toward high-quality, profitable firms |
| **CMA** (Conservative minus Aggressive) | Firms that invest cautiously vs aggressively | Leans toward conservative reinvestors |
        """
    )

if len(cfg["tickers"]) < 3:
    st.warning("Pick at least 3 stocks.")
    st.stop()
try:
    mk = H.load_market(cfg["tickers"], cfg["mode"], cfg["demo"])
except D.DataError as exc:
    st.error(f"{exc}  Try Demo mode.")
    st.stop()
H.badge(mk["source"], mk["as_of"])
india = is_india_portfolio(cfg["tickers"])
if mk["ff"] is None and not india:
    st.error("Fama-French factor data is unavailable right now. " + " ".join(mk["notes"]))
    st.stop()

try:
    mu, cov = annualised_inputs(mk["rets_train"], cfg["shrink"], cfg["lw"])
    w = max_sharpe(mu, cov, mk["rf"], cfg["cap"])
except InfeasibleError as exc:
    st.error(str(exc))
    st.stop()
port = portfolio_returns(mk["rets_train"], w, "daily")

if india:
    st.info("🇮🇳 Most of your holdings are Indian-listed, so the **US** Fama-French factors would be misleading. "
            "Showing a single-index market model against the **Nifty 50** instead (beta, alpha, R²). "
            "India-specific size/value/profitability factors are not included.")
    a_, b_, c_, d_ = mk["windows"]
    try:
        nifty = mk["bench"] if cfg["demo"] else H.series_cached("^NSEI", a_, d_)[0]
        rf_in = st.sidebar.number_input("Indian risk-free rate (annual %)", 0.0, 15.0, 6.8, 0.1) / 100
        mm = run_market_model(port, nifty.pct_change().dropna(), rf_in, train_frac)
    except (D.DataError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    t = pd.DataFrame({"OLS": mm["ols"].coef, "OLS p-value": mm["ols"].p, "Robust (Huber)": mm["robust"].coef}).rename(index={"const": "Alpha (daily)", "MKT_RF": "Beta to Nifty 50"})
    st.dataframe(t.style.format("{:.3f}"), use_container_width=True)
    k = st.columns(3)
    H.metric(k[0], "Beta", f"{mm['ols'].loc['MKT_RF', 'coef']:.2f}", "Beta")
    H.metric(k[1], "R² (in sample)", H.pct(mm["r2"], 0), "R-squared")
    H.metric(k[2], "R² on unseen data", H.pct(mm["r2_oos"], 0))
    st.caption(f"{mm['n_obs']} daily observations, {mm['start']:%d %b %Y} → {mm['end']:%d %b %Y}. Chronological {train_frac:.0%}/{1-train_frac:.0%} split.")
    H.footer()
    st.stop()

try:
    res = run_ff5(port, mk["ff"], "D" if freq == "Daily" else "W", train_frac)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"Portfolio = max-Sharpe mix with a {cfg['cap']:.0%} cap · {res['n_obs']} {freq.lower()} observations "
           f"({res['start']:%d %b %Y} → {res['end']:%d %b %Y}). Factor data is published with a delay, so the most recent weeks may be missing.")

st.markdown("### 1 · How the factors relate to each other")
ff_win = mk["ff"].loc[res["start"]:res["end"]]
c1, c2 = st.columns([3, 2])
with c1:
    fig = px.imshow(factor_correlation(ff_win), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=380, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    cm = factor_correlation(ff_win)
    pairs = cm.where(np.triu(np.ones(cm.shape, dtype=bool), 1)).stack()
    top = pairs.abs().sort_values(ascending=False).head(2).index
    st.markdown("**Reading it.** Values near 0 mean the factors carry different information, which is what we want in a regression. "
                "Strong pairs (|ρ| > 0.5) can make individual betas harder to separate.")
    for i, j in top:
        st.markdown(f"- Strongest link: **{i} & {j}** (ρ = {cm.loc[i, j]:+.2f})")
st.dataframe(factor_summary(ff_win).style.format("{:.3f}"), use_container_width=True)

st.markdown("### 2 · Exposure results")
ols, rob = res["ols"], res["robust"]
tbl = pd.DataFrame({
    "OLS beta": ols.coef, "OLS p-value": ols.p, "Robust (Huber) beta": rob.coef, "Robust p-value": rob.p,
}).rename(index={"const": "Alpha (daily)" if freq == "Daily" else "Alpha (weekly)"})
st.dataframe(tbl.style.format({"OLS beta": "{:.3f}", "OLS p-value": "{:.3f}", "Robust (Huber) beta": "{:.3f}", "Robust p-value": "{:.3f}"}), use_container_width=True)
fb = go.Figure()
f5 = ols.drop("const")
fb.add_bar(x=f5.index, y=f5.coef, error_y=dict(type="data", array=1.96 * f5.std_err), name="OLS ±95% CI")
fb.add_bar(x=rob.drop("const").index, y=rob.drop("const").coef, name="Robust")
fb.update_layout(barmode="group", yaxis_title="Beta", height=360)
st.plotly_chart(fb, use_container_width=True)

k1, k2, k3, k4 = st.columns(4)
H.metric(k1, "R² (in sample)", H.pct(res["r2"], 0), "R-squared")
H.metric(k2, "R² on unseen test data", H.pct(res["oos_ols"]["R2_oos"], 0))
H.metric(k3, "Annualised alpha", H.pct(res["alpha_annual"]), "Alpha")
H.metric(k4, "Stock-specific risk (annual)", H.pct(res["resid_vol_annual"]))

st.markdown("### 3 · In plain words")
for line in factor_story(res):
    st.markdown(f"- {line}")

with st.expander("🧪 How the data were split (training vs testing)"):
    st.markdown(
        f"""
Observations are ordered in time. The **first {train_frac:.0%} ({res['n_train']} obs, up to {res['train_end']:%d %b %Y})** train the model;
the **last {1-train_frac:.0%} ({res['n_test']} obs, from {res['test_start']:%d %b %Y})** test it. The split is chronological, never random,
because a random split would let future information leak into training.
Out-of-sample R² compares the model's errors with simply predicting the *training-period average*; it can be negative if the model generalises badly.

| | OLS | Robust (Huber) |
|---|---|---|
| Test R² | {res['oos_ols']['R2_oos']:.3f} | {res['oos_robust']['R2_oos']:.3f} |
| Test RMSE | {res['oos_ols']['RMSE']:.5f} | {res['oos_robust']['RMSE']:.5f} |

OLS uses Newey-West (HAC, 5 lags) standard errors so p-values stay valid despite autocorrelation and changing volatility.
If OLS and Huber betas differ a lot, a few extreme days are driving the OLS result.
        """
    )
    st.dataframe(pd.concat({"OLS (train)": res["ols_train"].coef, "Robust (train)": res["robust_train"].coef}, axis=1).style.format("{:.3f}"))
H.quiz("fac", [
    ("A market beta of 0.7 means…", ["It moves 70% as much as the market", "It earns 70% a year", "70% of its risk is market risk"], 0, "Beta is sensitivity to market moves."),
    ("Why a chronological train/test split?", ["It is faster", "A random split would leak future information", "Regulators require it"], 1, "Returns are time-ordered; random splits let the model peek ahead."),
])
H.footer()
