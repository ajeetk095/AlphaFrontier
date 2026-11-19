import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import helpers as H
from core import data as D
from core.insights import DISCLAIMER, money_caption, portfolio_story
from core.optimize import (
    InfeasibleError,
    annualised_inputs,
    efficient_frontier,
    max_sharpe,
    min_variance,
    portfolio_stats,
)
from core.pdf_report import build_pdf
from core.performance import portfolio_returns, scenario_paths, summarise

H.page_setup("Portfolio Optimizer", "🎯")
st.title("🎯 Portfolio Optimizer")
st.write("Find the mix of stocks with the best return for the risk taken, with and without a limit per stock.")
cfg = H.sidebar_controls("opt")
objective = st.sidebar.radio("Objective", ["Best return per unit of risk (max Sharpe)", "Lowest risk (min variance)"])

if len(cfg["tickers"]) < 3:
    st.warning("Pick at least 3 stocks.")
    st.stop()
try:
    mk = H.load_market(cfg["tickers"], cfg["mode"], cfg["demo"])
except D.DataError as exc:
    st.error(f"{exc}  Try Demo mode in the sidebar.")
    st.stop()

a, b, c, d = mk["windows"]
rf = mk["rf"]
H.badge(mk["source"], mk["as_of"])
mu, cov = annualised_inputs(mk["rets_train"], cfg["shrink"], cfg["lw"])
st.caption(f"Estimation window {a:%d %b %Y} → {b:%d %b %Y} ({len(mk['rets_train'])} trading days) · "
           f"test window {c:%d %b %Y} → {d:%d %b %Y} ({len(mk['rets_test'])} days) · risk-free {rf:.2%}")
for n in mk["notes"]:
    st.caption(f"ℹ️ {n}")


def solve(cap):
    return min_variance(mu, cov, cap) if objective.startswith("Lowest") else max_sharpe(mu, cov, rf, cap)


try:
    w_free, w_cap = solve(1.0), solve(cfg["cap"])
except InfeasibleError as exc:
    st.error(str(exc))
    st.stop()
s_free, s_cap = portfolio_stats(w_free, mu, cov, rf), portfolio_stats(w_cap, mu, cov, rf)
cap_label = f"Max {cfg['cap']:.0%} each"

tab1, tab2, tab3, tab4 = st.tabs(["① Suggested mix", "② Risk vs reward curve", "③ Recent back-test", "④ What could happen next"])

with tab1:
    wt = pd.DataFrame({"No cap": w_free, cap_label: w_cap})
    wt["Change"] = wt[cap_label] - wt["No cap"]
    st.dataframe(wt.style.format("{:.1%}").background_gradient(subset=["No cap", cap_label], cmap="Blues"), use_container_width=True)
    k = st.columns(5)
    H.metric(k[0], "Return (historical)", H.pct(s_cap.ret))
    H.metric(k[1], "Volatility", H.pct(s_cap.vol), "Volatility")
    H.metric(k[2], "Sharpe ratio", f"{s_cap.sharpe:.2f}", "Sharpe ratio")
    H.metric(k[3], "Holdings", f"{s_cap.n_hold}")
    H.metric(k[4], "Effective positions", f"{s_cap.eff_n:.1f}", "Effective N")
    st.markdown("**In plain words:**")
    story = portfolio_story(w_cap, s_cap, cfg["cap"], rf)
    for line in story:
        st.markdown(f"- {line}")
    st.markdown(f"- 💬 On {H.money(cfg['amount'])}, a typical year's swing (1 volatility) is about **{H.money(s_cap.vol * cfg['amount'])}** either way.")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Weights (CSV)", wt.to_csv().encode(), "portfolio_weights.csv", "text/csv")

with tab2:
    try:
        f1, f2 = efficient_frontier(mu, cov, 1.0), efficient_frontier(mu, cov, cfg["cap"])
        fig = go.Figure()
        fig.add_scatter(x=f1.vol, y=f1.ret, name="No cap", mode="lines")
        fig.add_scatter(x=f2.vol, y=f2.ret, name=cap_label, mode="lines")
        fig.add_scatter(x=np.sqrt(np.diag(cov)), y=mu, mode="markers+text", text=mu.index, textposition="top center",
                        name="Single stocks", marker=dict(size=8, color="grey"))
        for nm, s_ in (("No cap", s_free), ("Capped", s_cap)):
            fig.add_scatter(x=[s_.vol], y=[s_.ret], mode="markers", name=f"{nm} pick", marker=dict(size=13, symbol="star"))
        fig.update_layout(xaxis_title="Risk (annual volatility)", yaxis_title="Expected annual return",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=520)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**How to read it.** Each point on a curve is the least risky mix for a given return. The capped curve can never sit "
                    "*above/left* of the no-cap curve because a cap only removes options. The gap is widest at the high-return end, "
                    "where the no-cap mix piles into a few winners.")
    except InfeasibleError as exc:
        st.error(str(exc))

with tab3:
    if len(mk["rets_test"]) < 10:
        st.info("Not enough test-period data yet.")
    else:
        rows, curves = {}, {}
        for nm, w_ in (("No cap", w_free), (f"Max {cfg['cap']:.0%}", w_cap)):
            r = portfolio_returns(mk["rets_test"], w_, "none")
            rows[nm], curves[nm] = summarise(r, rf), (1 + r).cumprod()
        if mk["bench"] is not None:
            br = mk["bench"].pct_change().dropna().loc[pd.Timestamp(c):]
            rows["S&P 500"], curves["S&P 500"] = summarise(br, rf), (1 + br).cumprod()
        res = pd.DataFrame(rows)
        fmt = res.copy().astype(object)
        for k_ in res.index:
            fmt.loc[k_] = [f"{v:.2f}" if k_ in ("Sharpe ratio", "Sortino ratio") else (f"{int(v)}" if k_ == "Trading days" else f"{v:.1%}") for v in res.loc[k_]]
        st.dataframe(fmt, use_container_width=True)
        fig = go.Figure()
        for k_, v in curves.items():
            fig.add_scatter(x=v.index, y=v.values * cfg["amount"], name=k_)
        fig.update_layout(yaxis_title=f"Value of {H.money(cfg['amount'])} invested (buy & hold)", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.warning("A ~3-month test is **very short**. Annualised figures from ~60 days are noisy; use it to spot gross problems, "
                   "not to prove the strategy works. For a proper test, see **Walk-forward back-test**.")

with tab4:
    st.write("Bootstrapped one-year scenarios built from the capped portfolio's own history. **Scenarios, not forecasts.**")
    haircut = st.slider("Discount past returns by", 0.0, 1.0, 0.5, 0.05, help="Drift haircut. " + "Optimised portfolios look best on the data they were fitted to.")
    sc = scenario_paths(portfolio_returns(mk["rets_train"], w_cap, "daily"), drift_haircut=haircut)
    k = st.columns(4)
    H.metric(k[0], "Bad year (5th pct)", H.pct(sc["p05"]), "VaR")
    H.metric(k[1], "Typical (median)", H.pct(sc["p50"]))
    H.metric(k[2], "Good year (95th pct)", H.pct(sc["p95"]))
    H.metric(k[3], "Chance of a loss", H.pct(sc["prob_loss"], 0))
    st.caption(money_caption(sc["p05"], cfg["amount"], cfg["ccy"], "In a bad year (worst 5%)"))
    fig = go.Figure(go.Histogram(x=sc["terminal"], nbinsx=60))
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(xaxis_title="1-year return", xaxis_tickformat=".0%", yaxis_title="Scenarios", height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"In the worst 5% of scenarios the average loss is {abs(sc['cvar95']):.0%} (expected shortfall).")

# ---- PDF summary
pdf = build_pdf(
    "Portfolio Lab summary", f"{len(cfg['tickers'])} stocks · cap {cfg['cap']:.0%} · estimation {a:%d %b %Y} to {b:%d %b %Y}",
    [("Suggested weights", [wt.map(lambda v: f"{v:.1%}")]),
     ("In plain words", [s.replace("**", "") for s in story]),
     ("Scenarios (1 year)", [f"Bad year (5th percentile): {sc['p05']:.1%}; median {sc['p50']:.1%}; good year {sc['p95']:.1%}; "
                             f"chance of a loss {sc['prob_loss']:.0%} (drift haircut {haircut:.0%})."])],
    DISCLAIMER)
c2.download_button("⬇️ One-page PDF summary", pdf, "portfolio_summary.pdf", "application/pdf")

H.quiz("opt", [
    ("The capped efficient frontier can sit above the uncapped one. True?", ["True", "False"], 1,
     "A cap only removes feasible portfolios, so the capped frontier is never better."),
    ("The historical return shown for the optimal mix is best read as…", ["A forecast", "A description of the fitted past"], 1,
     "The weights were chosen because they looked good on this exact data."),
])
H.footer()
