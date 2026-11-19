from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import helpers as H
from core import data as D
from core.models import MODELS, risk_contributions
from core.optimize import InfeasibleError, annualised_inputs
from core.performance import portfolio_returns
from core.risk import (
    STRESS_EVENTS,
    drawdown_series,
    portfolio_beta,
    replay_event,
    rolling_vol,
    var_cvar_historical,
    var_cvar_parametric,
)

H.page_setup("Risk & Stress", "🛡️")
st.title("🛡️ Risk & stress tests")
st.write("Averages hide the bad days. This page shows how much you could lose, which holdings drive the risk, and how the portfolio would have "
         "fared in real market crises.")

cfg = H.sidebar_controls("rk", window=False)
which = st.sidebar.radio("Portfolio to analyse", ["Max Sharpe (capped)", "Min variance (capped)", "Equal weight (1/N)"])
if len(cfg["tickers"]) < 3:
    st.warning("Pick at least 3 stocks.")
    st.stop()

try:
    hist = H.load_history(cfg["tickers"], date(2018, 1, 2), date.today(), cfg["demo"])
except D.DataError as exc:
    st.error(f"{exc}  Try Demo mode.")
    st.stop()
H.badge(hist["source"], hist["as_of"])
prices = hist["prices"]
rets = prices.pct_change().dropna(how="all")
common = rets.dropna()
if len(common) < 400:
    st.error("Too little shared history for these stocks. Remove recently listed names.")
    st.stop()

name = {"Max Sharpe (capped)": "Max Sharpe", "Min variance (capped)": "Min variance", "Equal weight (1/N)": "Equal weight (1/N)"}[which]
try:
    w = MODELS[name](common.iloc[-504:], cap=cfg["cap"], rf=hist["rf"], shrink=cfg["shrink"], ledoit_wolf=cfg["lw"])
except InfeasibleError as exc:
    st.error(str(exc))
    st.stop()
port = portfolio_returns(common, w, "daily")
amt = cfg["amount"]

st.markdown("### 1 · How much could I lose?")
rows = []
for level in (0.95, 0.99):
    for h in (1, 10):
        hv, hc = var_cvar_historical(port, level, h)
        pv, pc = var_cvar_parametric(port, level, h)
        rows.append({"Confidence": f"{level:.0%}", "Horizon": f"{h} day{'s' if h > 1 else ''}", "VaR (historical)": hv, "CVaR (historical)": hc,
                     "VaR (normal)": pv, "CVaR (normal)": pc})
tbl = pd.DataFrame(rows).set_index(["Confidence", "Horizon"])
st.dataframe(tbl.style.format("{:.2%}"), use_container_width=True)
hv, hc = var_cvar_historical(port, 0.99, 1)
st.info(f"On the worst 1% of days the portfolio lost at least **{hv:.1%}** ({H.money(hv * amt)} on {H.money(amt)}), and **{hc:.1%}** ({H.money(hc * amt)}) on average. "
        "Compare the *normal* and *historical* columns: when historical is larger, real markets have fatter tails than the bell curve assumes.")
st.caption("VaR = the loss exceeded only on the worst days; CVaR = average loss on those days. 10-day figures use overlapping historical windows (historical) or √time scaling (normal).")

st.markdown("### 2 · Drawdowns and changing volatility")
c1, c2 = st.columns(2)
dd = drawdown_series(port)
with c1:
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown"))
    fig.update_layout(title=f"Drawdown (worst {dd.min():.0%})", yaxis_tickformat=".0%", height=320)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    rv = rolling_vol(port)
    fig = go.Figure(go.Scatter(x=rv.index, y=rv.values, name="63-day volatility"))
    fig.update_layout(title="Rolling volatility (annualised)", yaxis_tickformat=".0%", height=320)
    st.plotly_chart(fig, use_container_width=True)
st.caption(H.money(abs(dd.min()) * amt) + f" is what the worst peak-to-trough fall would have cost on {H.money(amt)}.")

st.markdown("### 3 · What actually drives the risk?")
_, cov = annualised_inputs(common.iloc[-504:])
rc = risk_contributions(w, cov)
comp = pd.DataFrame({"Share of money": w, "Share of risk": rc})
comp = comp[(comp > 1e-4).any(axis=1)].sort_values("Share of risk", ascending=False)
fig = go.Figure()
for col in comp:
    fig.add_bar(x=comp.index, y=comp[col], name=col)
fig.update_layout(barmode="group", yaxis_tickformat=".0%", height=340)
st.plotly_chart(fig, use_container_width=True)
big = comp.index[0]
st.caption(f"**{big}** is {comp.loc[big, 'Share of money']:.0%} of the money but {comp.loc[big, 'Share of risk']:.0%} of the risk. "
           "Volatile or highly correlated holdings punch above their weight.")
corr = common.iloc[-504:].corr()
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
fig.update_layout(title="Correlations (last 2 years)", height=420)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 4 · Stress tests")
st.markdown("**Replay history.** Buy-and-hold this portfolio through real market episodes.")
ev_rows = []
for label, (s, e) in STRESS_EVENTS.items():
    out = replay_event(prices, w, s, e, hist["bench"])
    if out is None:
        ev_rows.append({"Episode": label, "Portfolio": None, "S&P 500": None, "Coverage of holdings": None})
    else:
        ev_rows.append({"Episode": label, "Portfolio": out["portfolio"], "S&P 500": out.get("benchmark"), "Coverage of holdings": out["coverage"]})
ev = pd.DataFrame(ev_rows).set_index("Episode")
st.dataframe(ev.style.format("{:.1%}", na_rep="n/a"), use_container_width=True)
st.caption("Coverage < 100% means some holdings didn't exist then; the rest are re-scaled. Past crises are not a catalogue of future ones.")

st.markdown("**What if the market falls?** Uses the portfolio's measured sensitivity (beta) to the S&P 500.")
if hist["bench"] is not None:
    beta = portfolio_beta(common, w, hist["bench"].pct_change().dropna())
    shock = st.slider("Market move", -50, 10, -20, 1, format="%d%%") / 100
    impact = beta * shock
    k = st.columns(3)
    H.metric(k[0], "Portfolio beta", f"{beta:.2f}", "Beta")
    H.metric(k[1], "Estimated portfolio move", H.pct(impact))
    H.metric(k[2], f"On {H.money(amt)}", H.money(impact * amt))
    st.caption("A linear approximation. Real crashes also raise correlations and gaps, so treat this as a floor on the damage, not a ceiling.")
else:
    st.info("Benchmark data unavailable, so the beta shock is skipped.")

H.quiz("rk", [
    ("CVaR is…", ["The loss exceeded on the worst days", "The average loss on those worst days", "Daily volatility"], 1, "CVaR (expected shortfall) averages the tail beyond VaR."),
    ("A holding with 10% of the money but 25% of the risk is…", ["Safe", "Volatile or highly correlated with others", "Mis-measured"], 1, "Risk share depends on volatility and correlation, not only weight."),
])
H.footer()
