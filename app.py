from datetime import date, timedelta

import plotly.express as px
import streamlit as st

import helpers as H
from core import data as D
from core.insights import audience_guidance, portfolio_story, vol_regime
from core.optimize import annualised_inputs, max_sharpe, portfolio_stats
from core.performance import portfolio_returns, scenario_paths

H.page_setup("Home", "📈")

st.markdown(
    """<div class="pl-hero"><h1>📈 Portfolio Lab</h1>
<b>Turn market data into a clear, risk-aware stock portfolio, and learn why it works (or doesn't).</b><br>
Live data · walk-forward back-tests · Black-Litterman · risk &amp; stress tests · factor analysis · no finance degree needed.</div>""",
    unsafe_allow_html=True,
)

demo = st.toggle("Demo mode (synthetic data; use if Yahoo Finance is unreachable)", value=False)

# ---------------- market snapshot ----------------
vix_val, sp3m, source, as_of = None, None, "demo", None
if not demo:
    try:
        end, start = date.today(), date.today() - timedelta(days=200)
        sp, source, as_of = H.series_cached("^GSPC", start, end)
        sp = sp.dropna()
        sp3m = float(sp.iloc[-1] / sp.iloc[max(0, len(sp) - 64)] - 1)
        vix_val = float(H.series_cached("^VIX", start, end)[0].dropna().iloc[-1])
    except D.DataError as exc:
        st.warning(f"Live market snapshot unavailable ({exc}). Switch on Demo mode to explore the app.")
H.badge(source if not demo else "demo", as_of)
regime, regime_text = vol_regime(vix_val)
c1, c2, c3 = st.columns(3)
H.metric(c1, "S&P 500, last ~3 months", H.pct(sp3m) if sp3m is not None else "n/a")
H.metric(c2, "VIX (market fear gauge)", f"{vix_val:.1f}" if vix_val else "n/a")
H.metric(c3, "Market mood", regime.title())
st.info(regime_text)

# ---------------- headline portfolio ----------------
st.markdown("### Headline portfolio: 10 large US stocks, max 15% each")
stats = None
try:
    mk = H.load_market(D.DEFAULT_TICKERS, "Live (latest 2 years)", demo)
    mu, cov = annualised_inputs(mk["rets_train"])
    w = max_sharpe(mu, cov, mk["rf"], 0.15)
    stats = portfolio_stats(w, mu, cov, mk["rf"])
    sc = scenario_paths(portfolio_returns(mk["rets_train"], w, "daily"), n_paths=2000)
    a, b, _, _ = mk["windows"]
    st.caption(f"Estimated on {a:%d %b %Y} – {b:%d %b %Y}. Risk-free rate {mk['rf']:.2%}.")
    k1, k2, k3, k4 = st.columns(4)
    H.metric(k1, "Sharpe ratio (historical)", f"{stats.sharpe:.2f}", "Sharpe ratio")
    H.metric(k2, "Effective positions", f"{stats.eff_n:.1f}", "Effective N")
    H.metric(k3, "Bad year (5th pct scenario)", H.pct(sc["p05"]), "VaR")
    H.metric(k4, "Chance of a losing year", H.pct(sc["prob_loss"], 0))
    left, right = st.columns([3, 2])
    with left:
        wd = w[w > 1e-4].sort_values()
        fig = px.bar(wd, orientation="h", labels={"value": "Weight", "index": ""}, text=wd.map(lambda v: f"{v:.1%}"))
        fig.update_layout(showlegend=False, height=360, xaxis_tickformat=".0%", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        for line in portfolio_story(w, stats, 0.15, mk["rf"]):
            st.markdown(f"- {line}")
    for n in mk["notes"]:
        st.caption(f"ℹ️ {n}")
except Exception as exc:
    st.error(f"Could not build the headline portfolio ({exc}). Turn on Demo mode above to explore.")

# ---------------- navigation cards ----------------
st.markdown("### Explore")
cards = [
    ("pages/1_Learn_and_Quiz.py", "🎓 Learn & Quiz", "Interactive lessons, a risk-profile quiz, and a self-test. Start here if finance is new."),
    ("pages/2_Portfolio_Optimizer.py", "🎯 Optimizer", "Best-mix weights, risk/reward curves, back-test, and one-year scenarios."),
    ("pages/3_Walk_Forward_Backtest.py", "⏱️ Walk-forward back-test", "Does the optimizer beat 1/N after costs, with a statistical test?"),
    ("pages/4_Black_Litterman.py", "🧭 Black-Litterman", "Blend your own views with the market's. See how confidence moves the weights."),
    ("pages/5_Risk_and_Stress.py", "🛡️ Risk & stress", "VaR/CVaR, drawdowns, risk contributions, COVID and 2022 replays."),
    ("pages/6_Model_Comparison.py", "⚖️ Model comparison", "Six construction methods side by side, plus the estimation-noise lab."),
    ("pages/7_Factor_Analysis.py", "🔬 Factor analysis", "What drives returns: market, size, value, profitability, investment."),
    ("pages/8_Simulation_vs_Optimization.py", "🎲 Simulation vs optimization", "How many random portfolios match the exact answer?"),
]
cols = st.columns(4)
for i, (path, title, text) in enumerate(cards):
    with cols[i % 4]:
        st.markdown(f'<div class="pl-card"><b>{title}</b><br><span style="font-size:.9rem">{text}</span></div>', unsafe_allow_html=True)
        st.page_link(path, label="Open →")

# ---------------- recommendations ----------------
st.markdown("### What this means for you")
guide = audience_guidance(stats, 0.15, regime, None)
for tab, (_who, tips) in zip(st.tabs(list(guide)), guide.items()):
    with tab:
        for t in tips:
            st.markdown(f"- {t}")

st.markdown(
    """
### Four principles that hold up in any market
1. **Diversify on purpose.** Caps and many holdings protect you from the stock you didn't see coming.
2. **Don't trust past returns.** Optimisers reward what *already* did well; markets rarely repeat that.
3. **Judge by the bad year, not the average year.** Check the scenario range before you invest.
4. **Review on a schedule, and count costs.** Rebalance quarterly or when a holding drifts far from target.
"""
)
H.footer()
