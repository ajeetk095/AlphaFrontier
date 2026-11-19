"""Plain-language definitions used in tooltips and the glossary."""

GLOSSARY = {
    "Sharpe ratio": "Reward per unit of risk: (return - cash return) / volatility. Around 0.5 is typical for stocks, above 1 is strong, above 2 rarely lasts.",
    "Volatility": "How bumpy the ride is: the typical yearly swing. 15% means a normal year lands within about +/-15% of the average.",
    "Max drawdown": "The worst peak-to-trough fall. -30% means at some point you were down 30% from your high.",
    "CAGR": "Compound annual growth rate: the steady yearly return that would give the same final result.",
    "Sortino ratio": "Like Sharpe but only counts downside swings as risk.",
    "VaR": "Value at Risk: a loss you expect to exceed only on the worst 5% (or 1%) of days.",
    "CVaR": "Expected shortfall: the average loss on those worst days. Always at least as bad as VaR.",
    "Beta": "How much the portfolio moves when the market moves 1%. 1.2 means about 1.2% (in the same direction).",
    "Alpha": "Return left over after accounting for factor exposures. Hard to find, easy to fake in short samples.",
    "R-squared": "Share of the portfolio's ups and downs explained by the model. 60% means 40% is stock-specific.",
    "HHI": "Concentration score: sum of squared weights. 1 = everything in one stock; 0.1 = ten equal positions.",
    "Effective N": "How many equal-sized positions the portfolio behaves like (1/HHI). Higher is better diversified.",
    "Turnover": "How much of the portfolio is traded when rebalancing. 100% a year means the whole portfolio is replaced once.",
    "Cost drag": "Return lost to trading costs each year.",
    "Position cap": "A rule that no single stock may exceed a set share of the portfolio.",
    "Efficient frontier": "The best achievable return for each level of risk. Anything below the curve is inefficient.",
    "Risk-free rate": "What you'd earn in safe short-term government bills. The yardstick the Sharpe ratio beats.",
    "Drift haircut": "Discount applied to past average returns before simulating the future, because optimised portfolios look best on the data they were fitted to.",
    "Turnover cost": "Trading costs in basis points (1 bp = 0.01%) per unit of turnover.",
    "Confidence (Black-Litterman)": "How strongly you trust your own view versus the market's. 50% weighs them equally.",
    "Prior weights": "The starting portfolio before your views, usually proportional to company size (market cap).",
}


def tip(term: str) -> str:
    return GLOSSARY.get(term, "")
