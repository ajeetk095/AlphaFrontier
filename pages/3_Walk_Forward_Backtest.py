from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import helpers as H
from core import data as D
from core.backtest import BTConfig, backtest_table, rolling_sharpe, run_backtest, sharpe_diff_test
from core.insights import backtest_verdict
from core.models import MODEL_BLURBS, MODELS
from core.risk import drawdown_series

H.page_setup("Walk-forward back-test", "⏱️")
st.title("⏱️ Walk-forward back-test")
st.write("The honest test: at every rebalance date the model sees **only the past**, picks weights, pays trading costs, "
         "and is judged on what happens next. Does clever optimisation beat just splitting money equally?")

cfg = H.sidebar_controls("bt", default_cap=0.15, window=False)
sb = st.sidebar
years = sb.slider("Years of history", 4, 12, 8, help="Longer = more market regimes, but fewer stocks have that much data.")
lookback_y = sb.select_slider("Look-back for each estimate (years)", [1, 2, 3], value=2)
rebalance = sb.radio("Rebalance", ["Monthly", "Quarterly"])
cost = sb.slider("Trading cost (bps per unit of turnover)", 0, 50, 10, help="1 bp = 0.01%. 10 bps is realistic for liquid US large caps.")
strategies = sb.multiselect("Strategies", list(MODELS), default=["Max Sharpe", "Min variance", "Risk parity", "HRP", "Equal weight (1/N)"])

if len(cfg["tickers"]) < 3 or not strategies:
    st.warning("Pick at least 3 stocks and one strategy.")
    st.stop()

with st.expander("How each strategy works"):
    for s in strategies:
        st.markdown(f"**{s}**: {MODEL_BLURBS[s]}")


@st.cache_data(show_spinner=False)
def _bt(rets, strats, lookback, rebal, cost_bps, cap, shrink, lw, rf):
    c = BTConfig(lookback=lookback, rebalance=rebal, cost_bps=cost_bps, cap=cap, rf=rf, shrink=shrink, ledoit_wolf=lw)
    return run_backtest(rets, list(strats), c)


if st.button("▶ Run walk-forward back-test", type="primary"):
    end = date.today()
    start = end - timedelta(days=int(365.25 * years))
    try:
        hist = H.load_history(cfg["tickers"], start, end, cfg["demo"])
    except D.DataError as exc:
        st.error(f"{exc}  Try Demo mode.")
        st.stop()
    rets = hist["prices"].pct_change().dropna()
    lb = 252 * lookback_y
    if len(rets) < lb + 252:
        st.error(f"Only {len(rets)} common trading days across these stocks; need at least {lb + 252}. Shorten the look-back or drop recently listed stocks.")
        st.stop()
    with st.spinner("Re-estimating and rebalancing through history…"):
        try:
            res = _bt(rets, tuple(strategies), lb, "M" if rebalance == "Monthly" else "Q", cost, cfg["cap"], cfg["shrink"], cfg["lw"], hist["rf"])
        except Exception as exc:
            st.error(f"Back-test failed: {exc}")
            st.stop()
    st.session_state["bt"] = (res, hist, cost, cfg["cap"], rebalance)

if "bt" not in st.session_state:
    st.info("Choose strategies in the sidebar and press **Run walk-forward back-test**.")
    H.footer()
    st.stop()

res, hist, cost_, cap_, reb_ = st.session_state["bt"]
H.badge(hist["source"], hist["as_of"])
first = next(iter(res.values()))["returns"].index
st.caption(f"Evaluation period {first[0]:%d %b %Y} → {first[-1]:%d %b %Y} · {reb_.lower()} rebalancing · cap {cap_:.0%} · costs {cost_} bps")

tab = backtest_table(res, hist["rf"], cost_, hist["bench"])
fmt = tab.copy().astype(object)
for col in tab.columns:
    fmt[col] = [("n/a" if pd.isna(v) else (f"{v:.2f}" if col in ("Sharpe", "Sortino") else f"{v:.1%}")) for v in tab[col]]
st.markdown("### Results after trading costs")
st.dataframe(fmt, use_container_width=True)

curves = {k: (1 + v["returns"]).cumprod() * cfg["amount"] for k, v in res.items()}
if hist["bench"] is not None:
    b = hist["bench"].pct_change().dropna().reindex(first).dropna()
    curves["S&P 500"] = (1 + b).cumprod() * cfg["amount"]
logy = st.toggle("Log scale", value=False)
fig = go.Figure()
for k, v in curves.items():
    fig.add_scatter(x=v.index, y=v.values, name=k)
fig.update_layout(yaxis_title=f"Value of {H.money(cfg['amount'])}", yaxis_type="log" if logy else "linear", height=440)
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    for k, v in res.items():
        fig.add_scatter(x=v["returns"].index, y=drawdown_series(v["returns"]).values, name=k)
    fig.update_layout(title="Drawdowns", yaxis_tickformat=".0%", height=330)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = go.Figure()
    for k, v in res.items():
        rs = rolling_sharpe(v["returns"], hist["rf"])
        fig.add_scatter(x=rs.index, y=rs.values, name=k)
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="Rolling 1-year Sharpe ratio (it moves a lot)", height=330)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Is any difference real, or luck?")
base = "Equal weight (1/N)"
if base in res and len(res) > 1:
    rows, sigs = [], {}
    for k, v in res.items():
        if k == base:
            continue
        t = sharpe_diff_test(v["returns"], res[base]["returns"], hist["rf"])
        sigs[k] = t
        rows.append({"Strategy": k, "Sharpe vs 1/N": f"{t['diff']:+.2f}", "90% interval": f"{t['lo']:+.2f} to {t['hi']:+.2f}",
                     "Distinguishable from luck?": "Yes" if t["significant"] else "No"})
    st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)
    best = tab.drop(index=[i for i in tab.index if "benchmark" in i], errors="ignore")["Sharpe"].idxmax()
    if best in sigs:
        st.info(backtest_verdict(tab, sigs[best], best, base))
    st.caption("Method: paired block bootstrap (21-day blocks, 2,000 resamples) of the Sharpe difference.")
else:
    st.info("Include **Equal weight (1/N)** and at least one other strategy to see the significance test.")

focus = st.selectbox("Inspect weights over time", list(res))
wh = res[focus]["weights"]
fig = go.Figure()
for col in wh.columns:
    fig.add_scatter(x=wh.index, y=wh[col], stackgroup="one", name=col)
fig.update_layout(yaxis_tickformat=".0%", height=340, yaxis_title="Weight")
st.plotly_chart(fig, use_container_width=True)

st.warning("**Read with care.** (1) *Hindsight bias*: today's mega-cap names are in the test universe because they already succeeded. "
           "(2) A single historical path is one draw from many possible ones. (3) Costs are a simple proportional model; real costs include spread and market impact. "
           "(4) Taxes are ignored.")
H.quiz("bt", [
    ("Why do we require weights at day t to use only data before t?", ["Speed", "To avoid look-ahead bias", "Regulation"], 1,
     "Using the future would make any strategy look brilliant, and unrealisable."),
    ("If the 90% interval for a Sharpe difference includes zero, we should…", ["Declare a winner anyway", "Treat the ranking as inconclusive", "Add more stocks"], 1,
     "The data can't distinguish the strategies from noise."),
])
H.footer()
