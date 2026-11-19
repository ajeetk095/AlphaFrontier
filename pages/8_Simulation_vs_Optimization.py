import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import helpers as H
from core import data as D
from core.optimize import InfeasibleError, annualised_inputs, max_sharpe, portfolio_stats
from core.simulation import convergence_study, simulations_needed

H.page_setup("Simulation vs Optimization", "🎲")
st.title("🎲 Simulation vs Optimization")
st.write("Can we skip the maths and just try thousands of random portfolios? This page tests that, and shows why caps make it harder.")

cfg = H.sidebar_controls("sim")
five = st.sidebar.multiselect("Assets for the experiment (3–8)", cfg["tickers"],
                              default=[t for t in ["TSLA", "WMT", "BAC", "AAPL", "XOM"] if t in cfg["tickers"]])
cap = st.sidebar.select_slider("Position cap", [None, 0.5, 0.4, 0.3, 0.25, 0.2], value=0.3,
                               format_func=lambda v: "No cap" if v is None else f"{v:.0%}")
cap = cap or 1.0
alphas = (1.0, 0.3)
method = st.sidebar.radio("Handling the cap", ["Reject invalid draws", "Project draws onto the cap"],
                          help="Reject: throw away random portfolios above the cap (wasteful). "
                          "Project: clip at the cap and redistribute the excess (all draws valid, but biased toward corners).")
sizes_all = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000]
sizes = st.sidebar.multiselect("Simulation budgets", sizes_all, default=sizes_all[:6])
reps = st.sidebar.slider("Repeats per budget", 3, 30, 10, help="Averages out luck so the trend is reliable.")
tol = st.sidebar.slider("Close-enough tolerance (pp)", 0.5, 5.0, 1.0, 0.5, help="Largest allowed gap on any single weight, in percentage points.")

if not (3 <= len(five) <= 8):
    st.warning("Choose between 3 and 8 assets.")
    st.stop()
if st.button("▶ Run experiment", type="primary"):
    try:
        mk = H.load_market(five, cfg["mode"], cfg["demo"])
        H.badge(mk["source"], mk["as_of"])
        mu, cov = annualised_inputs(mk["rets_train"], cfg["shrink"], cfg["lw"])
        w_opt = max_sharpe(mu, cov, mk["rf"], cap)
        s_opt = portfolio_stats(w_opt, mu, cov, mk["rf"]).sharpe
    except (D.DataError, InfeasibleError) as exc:
        st.error(str(exc))
        st.stop()
    prog = st.progress(0.0, "Simulating…")
    df = convergence_study(mu.values, cov.values, mk["rf"], w_opt.values, s_opt, cap, sorted(sizes), reps,
                           "rejection" if method.startswith("Reject") else "projection", alphas=alphas)
    prog.empty()
    st.session_state["sim"] = (w_opt, s_opt, df, five, cap, method)

if "sim" not in st.session_state:
    st.info("Choose assets and a cap in the sidebar, then press **Run experiment**.")
    st.stop()

w_opt, s_opt, df, five_, cap_, method_ = st.session_state["sim"]
st.markdown(f"### Exact answer (optimizer) · cap: {'none' if cap_ >= 1 else f'{cap_:.0%}'}")
st.dataframe(pd.DataFrame({"Optimal weight": w_opt}).style.format("{:.2%}"), use_container_width=False)
st.caption(f"Optimal Sharpe ratio: {s_opt:.4f}")

st.markdown("### How close does random search get?")
st.dataframe(df.style.format({"Simulations": "{:,.0f}", "Valid draws (%)": "{:.1f}", "Sharpe gap (median)": "{:.4f}",
                              "Sharpe gap (worst)": "{:.4f}", "Max weight error, pp (median)": "{:.2f}",
                              "Total weight error, pp (median)": "{:.2f}"}), use_container_width=True)
fig = go.Figure()
fig.add_scatter(x=df["Simulations"], y=df["Max weight error, pp (median)"], mode="lines+markers", name="Median worst-weight error")
fig.add_hline(y=tol, line_dash="dash", annotation_text=f"tolerance {tol} pp")
fig.update_xaxes(type="log", title="Number of random portfolios")
fig.update_yaxes(title="Error vs optimizer (percentage points)")
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

n_star = simulations_needed(df, tol)
if n_star:
    st.success(f"With this setup, **~{n_star:,} simulations** bring the median worst-case weight error under {tol} pp. "
               "The optimizer gets there instantly and exactly.")
else:
    st.warning(f"Even at {int(df['Simulations'].max()):,} simulations the error stays above {tol} pp. Random search struggles here.")
st.markdown(
    """
**Why random search struggles**
- The best portfolio often sits on an *edge* (some weights exactly 0 or exactly at the cap). Random portfolios almost never land exactly there.
- With a cap, most random draws break the rule and are thrown away: the more assets and the tighter the cap, the worse it gets.
- Needed simulations grow quickly with the number of assets (the "curse of dimensionality").
- *Projection* makes every draw valid and finds corner solutions fast, but it over-samples corners, so it can look better than it should when the true optimum is a corner and worse when it isn't. Treat it as a heuristic, not a proof.

**Takeaway:** use exact optimisation for decisions; use simulation to *explore* uncertainty (scenarios, stress tests).
"""
)
H.quiz("sim", [
    ("Why does a 30% cap make random search slower?", ["Maths gets harder", "Most random portfolios break the cap and are discarded", "It doesn't"], 1,
     "For 5 assets only ~5.5% of uniform random portfolios satisfy a 30% cap."),
    ("The best portfolio often has some weights exactly 0. Random search…", ["Finds these easily", "Almost never lands exactly on them", "Is exact"], 1,
     "Edges of the weight space have zero probability under continuous random draws."),
])
H.footer()
