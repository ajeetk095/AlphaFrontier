import numpy as np
import plotly.graph_objects as go
import streamlit as st

import helpers as H

H.page_setup("Learn & Quiz", "🎓")
st.title("🎓 Learn portfolio theory by playing with it")
st.write("Five short, interactive lessons. Drag the sliders: each one makes a core idea visible.")

t1, t2, t3, t4, t5 = st.tabs(["1 · Diversification", "2 · Reward per risk", "3 · Noisy estimates", "4 · Your risk profile", "5 · Test yourself"])

# ------------------------------------------------------------------ 1
with t1:
    st.markdown("**Two stocks. One slider.** Mixing stocks that don't move in lockstep lowers risk *without* lowering average return.")
    c1, c2, c3 = st.columns(3)
    va = c1.slider("Stock A volatility", 5, 60, 20, key="va") / 100
    vb = c2.slider("Stock B volatility", 5, 60, 30, key="vb") / 100
    rho = c3.slider("Correlation between them", -1.0, 1.0, 0.2, 0.05, key="rho",
                    help="+1 = move together perfectly; 0 = unrelated; -1 = move opposite.")
    w = np.linspace(0, 1, 101)

    def pvol(r):
        return np.sqrt((w * va) ** 2 + ((1 - w) * vb) ** 2 + 2 * w * (1 - w) * r * va * vb)

    fig = go.Figure()
    fig.add_scatter(x=w, y=pvol(1.0), name="No diversification (ρ = +1)", line=dict(dash="dash"))
    fig.add_scatter(x=w, y=pvol(rho), name=f"Your correlation (ρ = {rho:+.2f})")
    i = int(np.argmin(pvol(rho)))
    fig.add_scatter(x=[w[i]], y=[pvol(rho)[i]], mode="markers", marker=dict(size=13, symbol="star"), name="Lowest-risk mix")
    fig.update_layout(xaxis_title="Share of money in Stock A", yaxis_title="Portfolio volatility", xaxis_tickformat=".0%",
                      yaxis_tickformat=".0%", height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Best mix: **{w[i]:.0%} in A**, giving **{pvol(rho)[i]:.1%}** volatility. That's below the safer stock alone ({min(va, vb):.0%}) "
               f"{'and the straight-line average would have been ' + format(w[i] * va + (1 - w[i]) * vb, '.1%') if rho < 1 else ''}.")
    st.caption("Try ρ = +1: the benefit disappears. Try ρ = −0.5: risk can fall dramatically. This is the free lunch Markowitz formalised.")

# ------------------------------------------------------------------ 2
with t2:
    st.markdown("**The Sharpe ratio** asks: how much extra return do I get for each unit of risk I take?")
    rf = st.slider("Cash return (risk-free rate)", 0.0, 8.0, 4.0, 0.25, key="rf_l") / 100
    ca, cb = st.columns(2)
    ra = ca.slider("Portfolio A: return", 0.0, 30.0, 12.0, 0.5, key="ra") / 100
    sa = ca.slider("Portfolio A: volatility", 3.0, 50.0, 15.0, 0.5, key="sa") / 100
    rb = cb.slider("Portfolio B: return", 0.0, 30.0, 18.0, 0.5, key="rb") / 100
    sb_ = cb.slider("Portfolio B: volatility", 3.0, 50.0, 30.0, 0.5, key="sb") / 100
    shA, shB = (ra - rf) / sa, (rb - rf) / sb_
    fig = go.Figure(go.Bar(x=["A", "B"], y=[shA, shB], text=[f"{shA:.2f}", f"{shB:.2f}"], textposition="auto"))
    fig.update_layout(yaxis_title="Sharpe ratio", height=300)
    st.plotly_chart(fig, use_container_width=True)
    better = "A" if shA > shB else "B"
    st.info(f"**Portfolio {better}** pays more per unit of risk. B's higher headline return isn't automatically better if it needed far more risk. "
            "Rule of thumb: <0.5 mediocre · ~1 good · >2 very rare and rarely persists.")

# ------------------------------------------------------------------ 3
with t3:
    st.markdown("**Why optimisers are fragile.** Here two stocks are *truly identical*, so the perfect answer is 50/50. "
                "We sample history 300 times and let the optimiser choose each time.")
    years = st.slider("Years of data the optimiser sees", 1, 15, 2, key="yrs")
    rng = np.random.default_rng(0)
    T, trials = 252 * years, 300
    mu_d, sd_d, rho_t, rf_d = 0.10 / 252, 0.20 / np.sqrt(252), 0.3, 0.03 / 252
    L = np.linalg.cholesky(np.array([[1, rho_t], [rho_t, 1]]))
    ws = []
    for _ in range(trials):
        x = (rng.standard_normal((T, 2)) @ L.T) * sd_d + mu_d
        m, S = x.mean(0) - rf_d, np.cov(x.T)
        z = np.linalg.solve(S, m)
        z = np.clip(z, 0, None)
        ws.append(0.5 if z.sum() == 0 else z[0] / z.sum())
    ws = np.array(ws)
    fig = go.Figure(go.Histogram(x=ws, nbinsx=25, xbins=dict(start=0, end=1, size=0.04)))
    fig.add_vline(x=0.5, line_dash="dash", annotation_text="true answer")
    fig.update_layout(xaxis_title="Optimiser's weight in Stock A", xaxis_tickformat=".0%", yaxis_title="Out of 300 samples", height=340)
    st.plotly_chart(fig, use_container_width=True)
    st.warning(f"With {years} year(s) of data, the optimiser's weight ranged from **{np.percentile(ws, 5):.0%} to {np.percentile(ws, 95):.0%}** "
               "(5th–95th percentile), when the right answer is 50%. That's why this app offers shrinkage, caps, and alternatives to pure optimisation.")

# ------------------------------------------------------------------ 4
with t4:
    st.markdown("**Five questions to suggest a sensible position cap.** This is a learning aid, not a regulated suitability assessment.")
    qs = [
        ("If your portfolio fell 25% in a month, you would…", ["Sell everything", "Sell some", "Hold", "Buy more"]),
        ("When will you need this money?", ["< 2 years", "2–5 years", "5–10 years", "10+ years"]),
        ("Your income is…", ["Unstable", "Fairly stable", "Stable", "Stable with large savings"]),
        ("Your investing experience…", ["None", "A little", "Several years", "Extensive"]),
        ("Which describes your goal best?", ["Protect my money", "Beat inflation", "Grow steadily", "Maximise growth"]),
    ]
    score, answered = 0, 0
    for i, (q, opts) in enumerate(qs):
        a = st.radio(q, opts, index=None, key=f"prof_{i}", horizontal=True)
        if a is not None:
            answered += 1
            score += opts.index(a)
    if answered == len(qs):
        label, cap, text = (
            ("Conservative", 0.10, "Prioritise stability. Spread widely, hold a cushion in cash/bonds outside this tool.") if score <= 4 else
            ("Balanced", 0.15, "Moderate risk. A 15% cap keeps any single stock from dominating.") if score <= 9 else
            ("Growth", 0.20, "Comfortable with swings for higher long-run returns. Review the bad-year scenarios.") if score <= 14 else
            ("Aggressive", 0.30, "High risk tolerance. Be especially careful about concentration and costs."))
        st.success(f"**{label}** profile → suggested max weight per stock: **{cap:.0%}**. {text}")
        if st.button("Use this profile across the app"):
            st.session_state["profile"] = {"label": label, "cap": cap}
            st.toast("Saved. The cap slider on other pages will start at your suggested value.")

# ------------------------------------------------------------------ 5
with t5:
    H.quiz("learn_final", [
        ("Two stocks with correlation +1: what does mixing them do for risk?", ["Cuts it a lot", "Nothing beyond averaging", "Doubles it"], 1,
         "Perfectly correlated stocks give no diversification benefit."),
        ("A Sharpe ratio of 2.5 on 2 years of data means…", ["You'll keep earning that", "It was strong historically, but may not persist", "The stock is risk-free"], 1,
         "Short, strong periods flatter Sharpe ratios. Expect regression toward lower values."),
        ("Why cap each position at 15%?", ["It always raises returns", "It limits the damage from one bad holding", "Regulators require it everywhere"], 1,
         "Caps trade a little theoretical efficiency for protection against concentration."),
        ("An optimiser's weights change wildly with a few months of new data. That signals…", ["A bug", "Estimation noise", "A better model"], 1,
         "Mean-variance optimisation amplifies noise in expected-return estimates."),
        ("Max drawdown measures…", ["Average yearly loss", "Worst peak-to-trough fall", "Daily volatility"], 1,
         "It's the deepest hole you would have sat in."),
        ("Which is a *scenario*, not a forecast?", ["'The portfolio will return 12%'", "'In 5% of simulated years, it lost more than 20%'", "'The Sharpe will stay above 2'"], 1,
         "Scenarios describe a range of possibilities built from history; they make no promise."),
    ], "Final quiz (6 questions)")
H.footer()
