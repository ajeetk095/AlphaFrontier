import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import helpers as H
from core import data as D
from core.blacklitterman import implied_returns, posterior
from core.glossary import tip
from core.optimize import InfeasibleError, annualised_inputs, max_sharpe, portfolio_stats

H.page_setup("Black-Litterman", "🧭")
st.title("🧭 Black-Litterman: blend your views with the market's")
st.write("Plain optimisation trusts noisy past returns. Black-Litterman starts from what the **market collectively implies**, "
         "then nudges it toward **your own views**, as much as your confidence justifies.")

with st.expander("📖 The idea in four lines", expanded=False):
    st.markdown("""
1. **Start with the market portfolio** (weights ∝ company size). Assume it's reasonable.
2. **Reverse-engineer** the returns that would make that portfolio optimal: these are the *implied returns*.
3. **Add your views** ("I expect Apple to return 15%") with a **confidence** level.
4. **Blend** them: more confidence pulls the expected return closer to your view. Optimise on the blend.

With *no* views you get the market portfolio back, which is a sensible, stable default.
""")

cfg = H.sidebar_controls("bl", default_cap=1.0)
delta = st.sidebar.slider("Market risk aversion (δ)", 1.0, 5.0, 2.5, 0.1, help="Higher δ → market implies higher required returns. 2.5 is a common default.")
tau = st.sidebar.slider("Uncertainty in the prior (τ)", 0.01, 0.2, 0.05, 0.01, help="Higher τ → views carry more weight relative to the market prior.")
prior_kind = st.sidebar.radio("Prior weights", ["Market-cap weights (live)", "Equal weights"])

if len(cfg["tickers"]) < 3:
    st.warning("Pick at least 3 stocks.")
    st.stop()
try:
    mk = H.load_market(cfg["tickers"], cfg["mode"], cfg["demo"])
except D.DataError as exc:
    st.error(f"{exc}  Try Demo mode.")
    st.stop()
H.badge(mk["source"], mk["as_of"])
rf = mk["rf"]
mu_hist, cov = annualised_inputs(mk["rets_train"], cfg["shrink"], cfg["lw"])
tick = list(cov.index)

w_prior = pd.Series(1 / len(tick), index=tick)
if prior_kind.startswith("Market") and not cfg["demo"]:
    caps = H.caps_cached(tuple(tick))
    if caps is not None:
        w_prior = caps / caps.sum()
    else:
        st.warning("Live market caps unavailable; using equal-weight prior instead.")
elif prior_kind.startswith("Market"):
    st.caption("Demo mode has no market caps; using equal weights.")
w_prior = w_prior.reindex(tick)

pi = implied_returns(cov, w_prior, delta)  # excess returns
st.markdown("### Step 1 · Your views (optional)")
with_view = st.multiselect("Stocks you have an opinion on", tick, help="Leave empty to see the pure market-implied result.")
views = {}
for t in with_view:
    c1, c2 = st.columns(2)
    default = int(round(100 * (pi[t] + rf)))
    q = c1.slider(f"{t}: your expected annual return (%)", -20, 60, min(60, max(-20, default + 5)), 1, key=f"bl_q_{t}")
    conf = c2.slider(f"{t}: confidence (%)", 5, 95, 50, 5, key=f"bl_c_{t}", help=tip("Confidence (Black-Litterman)"))
    views[t] = (q / 100 - rf, conf / 100)

mu_post, cov_post = posterior(cov, pi, views, tau)
try:
    w_bl = max_sharpe(mu_post + rf, cov_post, rf, cfg["cap"])
    w_mvo = max_sharpe(mu_hist, cov, rf, cfg["cap"])
except InfeasibleError as exc:
    st.error(str(exc))
    st.stop()

st.markdown("### Step 2 · What the blend does to expected returns")
ret_df = pd.DataFrame({"Historical average": mu_hist, "Market-implied": pi + rf, "Blended (BL)": mu_post + rf})
fig = go.Figure()
for col in ret_df:
    fig.add_bar(x=ret_df.index, y=ret_df[col], name=col)
fig.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_title="Expected annual return", height=360)
st.plotly_chart(fig, use_container_width=True)
st.caption("Historical averages are jagged (noisy). Market-implied returns are smooth and consistent with the risk each stock carries. "
           "Your views move the blended bars toward your numbers.")

st.markdown("### Step 3 · Resulting portfolios")
w_df = pd.DataFrame({"Market prior": w_prior, "Black-Litterman": w_bl, "Plain optimisation (historical means)": w_mvo})
fig = go.Figure()
for col in w_df:
    fig.add_bar(x=w_df.index, y=w_df[col], name=col)
fig.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_title="Weight", height=380)
st.plotly_chart(fig, use_container_width=True)

rows = {}
for name, w_ in (("Market prior", w_prior), ("Black-Litterman", w_bl), ("Plain optimisation", w_mvo)):
    s = portfolio_stats(w_, mu_hist, cov, rf)
    rows[name] = {"Effective positions": f"{s.eff_n:.1f}", "Largest position": f"{w_.max():.0%}", "Volatility": f"{s.vol:.1%}"}
st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

turn = float((w_bl - w_prior).abs().sum())
if not views:
    st.success("No views entered: Black-Litterman returns the market portfolio (when the cap is loose). "
               "Add a view above and watch the weights move.")
else:
    top = (w_bl - w_prior).abs().sort_values(ascending=False).index[0]
    st.info(f"Your views moved **{turn:.0%}** of the portfolio away from the market weights; the biggest change is in **{top}** "
            f"({w_prior[top]:.1%} → {w_bl[top]:.1%}). Try raising a view's confidence: the tilt grows. "
            "Notice the tilt is far gentler than plain optimisation would make it.")
st.caption("Implementation: absolute views, Omega_ii = (1/confidence − 1)·τ·(PΣPᵀ)_ii, posterior returns via the He-Litterman formula, then long-only max-Sharpe with the cap from the sidebar.")
H.quiz("bl", [
    ("With no views, Black-Litterman returns…", ["Equal weights", "The prior (market) portfolio", "The highest-return stock"], 1,
     "Without views, the posterior equals the market-implied returns, which are optimal at the prior weights."),
    ("Raising your confidence in a view…", ["Moves expected return toward your view", "Has no effect", "Moves it toward the market"], 0,
     "Higher confidence shrinks the view's uncertainty, so it dominates the prior for that asset."),
])
H.footer()
