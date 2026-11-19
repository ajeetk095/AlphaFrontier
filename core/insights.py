"""Plain-language insights and audience-specific guidance generated from live results.

Everything here is educational. It is rule-based text on top of numbers, not personal advice.
"""
from __future__ import annotations

import pandas as pd

DISCLAIMER = (
    "Educational tool, not investment advice. Past performance does not predict future results. "
    "Optimised weights rely on historical estimates that are noisy; real investing involves "
    "costs, taxes and risk of loss. Consult a licensed adviser (e.g. SEBI-registered in India) "
    "before acting."
)


def vol_regime(vix: float | None) -> tuple[str, str]:
    if vix is None:
        return "unknown", "Market-fear data (VIX) is unavailable right now."
    if vix < 15:
        return "calm", f"VIX is {vix:.1f}: markets are calm. Complacency can build quietly, so keep risk limits in place."
    if vix < 25:
        return "normal", f"VIX is {vix:.1f}: normal conditions. Standard diversification rules apply."
    return "stressed", f"VIX is {vix:.1f}: markets are nervous. Expect bigger daily swings; avoid adding risk in a rush."


def portfolio_story(weights: pd.Series, stats, cap: float, rf: float) -> list[str]:
    top = weights.sort_values(ascending=False)
    lines = [
        f"The model puts the most money in **{top.index[0]} ({top.iloc[0]:.0%})**"
        + (f" and **{top.index[1]} ({top.iloc[1]:.0%})**." if top.iloc[1] > 0.01 else "."),
        f"It holds **{stats.n_hold} of {len(weights)}** stocks, which behaves like about "
        f"**{stats.eff_n:.1f} equal-sized positions** (a higher number means better spread).",
    ]
    if stats.eff_n < 4:
        lines.append("⚠️ This is a concentrated bet. One bad surprise in a top holding would hurt a lot.")
    if top.iloc[0] >= cap - 1e-6 and cap < 1:
        lines.append(f"The {cap:.0%} limit is actively binding on {top.index[0]}: the model wanted more.")
    lines.append(
        f"Based on the chosen history, the mix earned about **{stats.ret:.1%} a year** with "
        f"**{stats.vol:.1%} volatility** (Sharpe {stats.sharpe:.2f}). Treat the return as a "
        "description of the past, not a promise."
    )
    return lines


def factor_story(res: dict) -> list[str]:
    t = res["ols"]
    names = {"MKT_RF": "the overall market", "SMB": "small-company stocks (size)",
             "HML": "cheap 'value' stocks", "RMW": "highly profitable companies",
             "CMA": "conservative-investing companies"}
    lines = []
    beta = t.loc["MKT_RF", "coef"]
    if beta < 0.9:
        lines.append(f"Market sensitivity is **{beta:.2f}**: the portfolio tends to move *less* than the market.")
    elif beta > 1.1:
        lines.append(f"Market sensitivity is **{beta:.2f}**: the portfolio tends to move *more* than the market, in both directions.")
    else:
        lines.append(f"Market sensitivity is **{beta:.2f}**: roughly in line with the market.")
    for f in ["SMB", "HML", "RMW", "CMA"]:
        c, p = t.loc[f, "coef"], t.loc[f, "p"]
        if p < 0.05:
            lean = "tilts toward" if c > 0 else "tilts away from"
            lines.append(f"Statistically clear tilt: the portfolio {lean} {names[f]} (beta {c:+.2f}).")
    if len(lines) == 1:
        lines.append("No other style tilt is statistically distinguishable from zero in this sample.")
    lines.append(
        f"The five factors explain **{res['r2']:.0%}** of day-to-day movements; the rest is stock-specific. "
        f"On unseen data the model explains **{res['oos_ols']['R2_oos']:.0%}**, which is the more honest number."
    )
    return lines


def audience_guidance(stats, cap: float, regime: str, beta: float | None) -> dict[str, list[str]]:
    retail = [
        "Spread across many holdings. If the model's top pick looks too big, lower the cap (10–15% is common).",
        "Invest gradually (e.g. monthly) rather than all at once; it reduces the risk of a bad entry point.",
        "Ignore the headline expected return; focus on the 'bad year' range in the scenario chart.",
    ]
    adviser = [
        "Use the capped portfolio as a starting point, then adjust for client goals, taxes and liquidity needs.",
        "Show clients the scenario range (5th/50th/95th percentile) alongside any expected return.",
        "Re-run quarterly; weights from a 2-year window can shift materially as new data arrives.",
    ]
    institution = [
        "Position caps cost little Sharpe but remove tail concentration; document the cap in the mandate.",
        "Monitor factor betas against the benchmark so unintended value/size/quality bets are visible.",
        "Replace raw sample means with shrinkage estimates and a shrunk covariance for production use.",
    ]
    if stats is not None and stats.eff_n < 4:
        retail.insert(0, "Right now the optimiser's mix is concentrated. Prefer the capped version.")
    if regime == "stressed":
        retail.append("Volatility is elevated: size positions smaller than usual.")
        institution.append("Stress regime: review liquidity and consider a lower risk budget.")
    if beta is not None and beta > 1.1:
        adviser.append(f"Portfolio market beta is {beta:.2f}; clients with low risk tolerance may want a lower beta.")
    return {"Retail investors": retail, "Advisers & wealth managers": adviser, "Companies & institutions": institution}


def money_caption(loss_frac: float, amount: float, ccy: str, what: str) -> str:
    """Make a percentage concrete: 'a -18% fall ≈ losing ₹18,000 of ₹1,00,000'."""
    return f"{what}: {loss_frac:.1%} ≈ {ccy}{abs(loss_frac) * amount:,.0f} on {ccy}{amount:,.0f}."


def backtest_verdict(tab, sig: dict | None, focus: str, base: str = "Equal weight (1/N)") -> str:
    """Honest one-paragraph reading of a walk-forward back-test."""
    if focus not in tab.index or base not in tab.index:
        return ""
    d = tab.loc[focus, "Sharpe"] - tab.loc[base, "Sharpe"]
    msg = f"**{focus}** had a Sharpe ratio of {tab.loc[focus, 'Sharpe']:.2f} vs {tab.loc[base, 'Sharpe']:.2f} for **{base}** ({d:+.2f}) after costs. "
    if sig is None:
        return msg
    if sig["significant"]:
        msg += f"The 90% bootstrap interval for the gap ({sig['lo']:+.2f} to {sig['hi']:+.2f}) excludes zero, so it is unlikely to be luck alone."
    else:
        msg += (f"But the 90% interval for the gap ({sig['lo']:+.2f} to {sig['hi']:+.2f}) includes zero: "
                "the data cannot tell these strategies apart. Don't read the ranking as proof.")
    return msg
