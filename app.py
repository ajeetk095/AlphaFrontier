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
    """<div class="pl-hero"><h1>📈 AlphaFrontier</h1>
<b>Build a clearer view of portfolio risk, diversification, and performance.</b><br>
Live market data · portfolio research · walk-forward testing · Fama–French factors<br>
<span>Interactive, educational analysis — not investment advice.</span></div>""",
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
        if len(sp) > 1:
            sp3m = float(sp.iloc[-1] / sp.iloc[max(0, len(sp) - 64)] - 1)
        vix = H.series_cached("^VIX", start, end)[0].dropna()
        if not vix.empty:
            vix_val = float(vix.iloc[-1])
    except D.DataError as exc:
        st.warning(f"Live market snapshot unavailable ({exc}). Switch on Demo mode to explore the app.")
H.badge(source if not demo else "demo", as_of)
regime, regime_text = vol_regime(vix_val)
c1, c2, c3 = st.columns(3)
H.metric(c1, "S&P 500 · ~3 months", H.pct(sp3m) if sp3m is not None else "n/a")
H.metric(c2, "VIX · market fear gauge", f"{vix_val:.1f}" if vix_val is not None else "n/a")
H.metric(c3, "Market mood", regime.title())
st.info(regime_text)

# ---------------- headline portfolio ----------------
st.markdown("### ✨ Portfolio snapshot")
st.caption("Illustrative large-cap US basket · max 15% per holding. Change tickers and assumptions on an analysis page.")
stats = None
try:
    mk = H.load_market(D.DEFAULT_TICKERS, "Live (latest 2 years)", demo)
    mu, cov = annualised_inputs(mk["rets_train"])
    w = max_sharpe(mu, cov, mk["rf"], 0.15)
    stats = portfolio_stats(w, mu, cov, mk["rf"])
    sc = scenario_paths(portfolio_returns(mk["rets_train"], w, "daily"), n_paths=1200)
    a, b, _, _ = mk["windows"]
    st.caption(f"Estimated on {a:%d %b %Y} – {b:%d %b %Y} · risk-free assumption {mk['rf']:.2%}.")
    k1, k2, k3, k4 = st.columns(4)
    H.metric(k1, "Historical Sharpe", f"{stats.sharpe:.2f}", "Sharpe ratio")
    H.metric(k2, "Effective positions", f"{stats.eff_n:.1f}", "Effective N")
    H.metric(k3, "Scenario · 5th percentile", H.pct(sc["p05"]), "VaR")
    H.metric(k4, "Scenario · chance of loss", H.pct(sc["prob_loss"], 0))
    left, right = st.columns([3, 2])
    with left:
        wd = w[w > 1e-4].sort_values()
        fig = px.bar(wd, orientation="h", labels={"value": "Weight", "index": "Holding"},
                     text=wd.map(lambda v: f"{v:.1%}"), color=wd.index, color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(showlegend=False, height=max(300, min(520, 32 * len(wd))), xaxis_tickformat=".0%", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### At a glance")
        for line in portfolio_story(w, stats, 0.15, mk["rf"]):
            st.markdown(f"- {line}")
    for note in mk["notes"]:
        st.caption(f"ℹ️ {note}")
except Exception as exc:
    st.error(f"Could not build the headline portfolio ({exc}). Turn on Demo mode above to explore.")

# ---------------- full feature directory ----------------
st.markdown("### 🧭 Explore the toolkit")
st.caption("Choose a research area below. Every feature is listed here and in the navigation menu.")
cards = [
    ("pages/1_Learn_and_Quiz.py", "🎓 Learn & Quiz", "Interactive lessons, a risk-profile quiz, and a self-test."),
    ("pages/2_Portfolio_Optimizer.py", "🎯 Portfolio Optimizer", "Compare max-Sharpe and min-variance allocations, efficient frontiers, tests, and scenarios."),
    ("pages/3_Walk_Forward_Backtest.py", "⏱️ Walk-forward back-test", "Test rolling strategies with scheduled rebalancing, costs, and benchmarks."),
    ("pages/4_Black_Litterman.py", "🧭 Black–Litterman", "Blend market-implied returns with your own views and confidence levels."),
    ("pages/5_Risk_and_Stress.py", "🛡️ Risk & stress", "Explore VaR/CVaR, drawdowns, factor exposures, and historical stress periods."),
    ("pages/6_Model_Comparison.py", "⚖️ Model comparison", "Compare six construction methods and test sensitivity to estimation noise."),
    ("pages/7_Factor_Analysis.py", "🔬 Factor analysis", "Analyze Fama–French five-factor exposures or the Nifty market model."),
    ("pages/8_Simulation_vs_Optimization.py", "🎲 Simulation vs optimization", "Compare random portfolio search with convex optimization."),
    ("pages/9_Report_and_Methodology.py", "📘 Report & methodology", "Read the research background, methodology, assumptions, and limitations."),
    ("pages/10_About the Developer.py", "👋 About & contact", "Project background, developer information, and contact links."),
]
for offset in range(0, len(cards), 2):
    col_a, col_b = st.columns(2)
    for col, (path, title, description) in zip((col_a, col_b), cards[offset:offset + 2]):
        with col:
            st.markdown(f'<div class="pl-card"><b>{title}</b><br><span>{description}</span></div>', unsafe_allow_html=True)
            st.page_link(path, label="Explore →", use_container_width=True)

# ---------------- recommendations ----------------
st.markdown("### 💡 What this means for you")
guide = audience_guidance(stats, 0.15, regime, None)
for tab, (_who, tips) in zip(st.tabs(list(guide)), guide.items()):
    with tab:
        for item in tips:
            st.markdown(f"- {item}")

st.markdown(
    """
### Four principles that hold up in any market
1. **Diversify on purpose.** Caps and many holdings can reduce single-stock concentration.
2. **Treat past returns carefully.** Optimizers can overreact to noisy historical estimates.
3. **Inspect downside risk.** Scenario ranges and stress tests describe possibilities, not guarantees.
4. **Account for costs.** Turnover, fees, and taxes can materially change realized results.
"""
)
H.footer()
