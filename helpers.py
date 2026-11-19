"""Shared Streamlit helpers: cached data access, windows, layout, sidebar, teaching widgets."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from core import data as D
from core.glossary import GLOSSARY, tip
from core.insights import DISCLAIMER

REPORT_WINDOWS = {
    "train": (date(2023, 7, 1), date(2025, 6, 30)),
    "test": (date(2025, 7, 1), date(2025, 9, 30)),
}

CSS = """
<style>
.block-container{padding-top:2rem;max-width:1200px}
div[data-testid='stMetricValue']{font-size:1.55rem}
.pl-hero{padding:1.4rem 1.6rem;border-radius:16px;background:linear-gradient(135deg,rgba(37,99,235,.18),rgba(16,185,129,.14));
  border:1px solid rgba(128,128,128,.25);margin-bottom:1rem}
.pl-hero h1{margin:0 0 .3rem 0}
.pl-card{padding:1rem 1.1rem;border-radius:14px;border:1px solid rgba(128,128,128,.25);background:rgba(128,128,128,.07);height:100%}
.pl-pill{display:inline-block;padding:.1rem .6rem;border-radius:999px;font-size:.8rem;border:1px solid rgba(128,128,128,.35);margin-right:.3rem}
</style>
"""

LIGHT_THEME_CSS = """
<style>
.stApp{background-color:#ffffff;color:#262730}
[data-testid="stSidebar"]{background-color:#f0f2f6}
[data-testid="stHeader"]{background-color:rgba(255,255,255,.9)}
.stApp :is(h1,h2,h3,h4,h5,h6,p,li,label,small,.stMarkdown,.stCaption){color:#262730}
.stApp a{color:#2563eb}
.stApp input,.stApp textarea,[data-baseweb="select"]>div,[data-baseweb="input"]>div{background-color:#ffffff;color:#262730}
[data-testid="stExpander"],[data-testid="stAlert"]{background-color:#ffffff}
</style>
"""

DARK_THEME_CSS = """
<style>
.stApp{background-color:#0e1117;color:#fafafa}
[data-testid="stSidebar"]{background-color:#161b22}
[data-testid="stHeader"]{background-color:rgba(14,17,23,.9)}
.stApp :is(h1,h2,h3,h4,h5,h6,p,li,label,small,.stMarkdown,.stCaption){color:#fafafa}
.stApp a{color:#79aaff}
.stApp input,.stApp textarea,[data-baseweb="select"]>div,[data-baseweb="input"]>div{background-color:#222831;color:#fafafa}
[data-testid="stExpander"],[data-testid="stAlert"]{background-color:#161b22;color:#fafafa}
[data-testid="stMetric"]{background-color:rgba(255,255,255,.04);border-radius:8px;padding:.5rem}
[data-testid="stPlotlyChart"]{background-color:transparent}
</style>
"""


def _save_appearance():
    """Keep the chosen theme outside the page-scoped widget state."""
    st.session_state["app_appearance"] = st.session_state["_appearance_selector"]


def page_setup(title: str, icon: str = "📈"):
    st.set_page_config(page_title=f"{title} · Portfolio Lab", page_icon=icon, layout="wide")
    # Streamlit resets page-scoped widgets during navigation. Keep the actual
    # preference in a separate session-state key and seed the widget per page.
    st.session_state.setdefault("app_appearance", "Light")
    st.session_state["_appearance_selector"] = st.session_state["app_appearance"]
    st.sidebar.selectbox(
        "Appearance",
        ("Light", "Dark"),
        key="_appearance_selector",
        on_change=_save_appearance,
    )
    theme_css = DARK_THEME_CSS if st.session_state["app_appearance"] == "Dark" else LIGHT_THEME_CSS
    st.markdown(CSS + theme_css, unsafe_allow_html=True)


def footer():
    st.divider()
    st.caption(DISCLAIMER)
    st.caption("Built by Ajeet Kumar · Quant research & financial engineering · see the *About & Hire* page.")


# ---------------- cached data access ----------------
@st.cache_data(ttl=3600, show_spinner="Fetching prices…")
def prices_cached(tickers: tuple, start, end):
    return D.get_prices(list(tickers), start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def series_cached(symbol: str, start, end):
    return D.get_series(symbol, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def rf_cached(start, end):
    return D.fetch_rf_annual(start, end)


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching Fama-French factors…")
def ff_cached():
    return D.get_ff5()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def caps_cached(tickers: tuple):
    from core.blacklitterman import fetch_market_caps

    return fetch_market_caps(list(tickers))


@st.cache_data(show_spinner=False)
def demo_cached(tickers: tuple, start, end):
    return D.make_demo_data(list(tickers), start, end)


def badge(source: str, as_of=None):
    """Data provenance stamp shown on every analysis page."""
    stamp = f" · latest data point {pd.Timestamp(as_of):%d %b %Y}" if as_of is not None else ""
    if source == "live":
        st.caption(f"🟢 **Live** data from Yahoo Finance{stamp}")
    elif source == "snapshot":
        st.caption(f"🟡 **Snapshot** data (Yahoo was unreachable; using the last saved copy){stamp}")
    else:
        st.caption("🔴 **Demo** data: synthetic, not real markets")


# ---------------- windows & loading ----------------
def get_windows(mode: str, today: date | None = None):
    """Return (train_start, train_end, test_start, test_end)."""
    if mode.startswith("Replicate"):
        (a, b), (c, d) = REPORT_WINDOWS["train"], REPORT_WINDOWS["test"]
        return a, b, c, d
    today = today or date.today()
    test_start = today - timedelta(days=91)
    train_end = test_start - timedelta(days=1)
    train_start = train_end - timedelta(days=730)
    return train_start, train_end, test_start, today


def load_market(tickers, mode: str, use_demo: bool):
    """dict(rets_train, rets_test, bench, ff, rf, source, as_of, windows, notes)."""
    a, b, c, d = get_windows(mode)
    notes: list[str] = []
    if use_demo:
        prices, bench, ff = demo_cached(tuple(tickers), a, d)
        rf, source, as_of = 0.045, "demo", prices.index.max()
        notes.append("DEMO MODE: synthetic data, not real markets.")
    else:
        prices, source, as_of = prices_cached(tuple(tickers), a, d)
        try:
            bench = series_cached("^GSPC", a, d)[0]
        except D.DataError:
            bench = None
            notes.append("S&P 500 benchmark unavailable.")
        try:
            ff = ff_cached()[0]
        except D.DataError as exc:
            ff = None
            notes.append(str(exc))
        rf, rf_ok = rf_cached(a, b)
        if not rf_ok:
            notes.append("Risk-free rate unavailable; using a 4.0% assumption.")
    rets = prices.pct_change().dropna()
    return {"rets_train": rets.loc[: pd.Timestamp(b)], "rets_test": rets.loc[pd.Timestamp(c):], "bench": bench,
            "ff": ff, "rf": rf, "source": source, "as_of": as_of, "windows": (a, b, c, d), "notes": notes,
            "live": source != "demo"}


def load_history(tickers, start, end, use_demo: bool):
    """Long history for back-tests and stress tests: dict(prices, bench, rf, source, as_of, notes)."""
    notes: list[str] = []
    if use_demo:
        prices, bench, _ = demo_cached(tuple(tickers), start, end)
        return {"prices": prices, "bench": bench, "rf": 0.04, "source": "demo", "as_of": prices.index.max(),
                "notes": ["DEMO MODE: synthetic data, not real markets."]}
    prices, source, as_of = prices_cached(tuple(tickers), start, end)
    try:
        bench = series_cached("^GSPC", start, end)[0]
    except D.DataError:
        bench = None
        notes.append("S&P 500 benchmark unavailable.")
    rf, ok = rf_cached(start, end)
    if not ok:
        notes.append("Risk-free rate unavailable; using a 4.0% assumption.")
    return {"prices": prices, "bench": bench, "rf": rf, "source": source, "as_of": as_of, "notes": notes}


# ---------------- small UI helpers ----------------
def pct(x, d=1):
    return f"{x:.{d}%}"


def money(x: float) -> str:
    ccy = st.session_state.get("ccy", "$")
    return f"{ccy}{x:,.0f}"


def metric(container, label: str, value: str, term: str | None = None, delta: str | None = None):
    container.metric(label, value, delta, help=tip(term) if term else None)


def quiz(key: str, items: list[tuple], title: str = "Check your understanding"):
    """items: [(question, [options], correct_index, explanation)]"""
    with st.expander(f"🧠 {title}"):
        right = 0
        for i, (q, opts, ans, why) in enumerate(items):
            choice = st.radio(q, opts, index=None, key=f"{key}_q{i}")
            if choice is not None:
                if opts.index(choice) == ans:
                    right += 1
                    st.success(f"Correct. {why}")
                else:
                    st.error(f"Not quite. {why}")
        if len(items) > 1:
            st.caption(f"Score so far: {right}/{len(items)}")


def sidebar_controls(key: str = "main", default_cap: float = 0.15, window: bool = True):
    """Shared sidebar. Returns config dict. Syncs tickers/cap to the URL so results are shareable."""
    sb = st.sidebar
    sb.header("Settings")
    prof = st.session_state.get("profile")
    if prof:
        default_cap = prof["cap"]
        sb.caption(f"Your risk profile: **{prof['label']}** (suggested cap {prof['cap']:.0%})")
    qp = st.query_params
    qp_tk = [t for t in str(qp.get("tk", "")).split(",") if t]
    try:
        default_cap = min(1.0, max(0.05, float(qp.get("cap", default_cap))))
    except ValueError:
        pass
    mode = "Live (latest 2 years)"
    if window:
        mode = sb.radio(
            "Data window", ["Live (latest 2 years)", "Replicate report (Jul 2023 – Sep 2025)"], key=f"{key}_mode",
            help="Live trains on the 2 years ending 3 months ago and tests on the last 3 months. "
            "Replicate uses the exact windows from the group report.")
    options = list(dict.fromkeys(D.DEFAULT_TICKERS + D.EXTRA_TICKERS + qp_tk))
    tickers = sb.multiselect("Stocks", options, default=qp_tk or D.DEFAULT_TICKERS, key=f"{key}_tk",
                             help="Any Yahoo Finance ticker works (type below). Add .NS for NSE-listed Indian stocks.")
    extra = sb.text_input("Add other tickers (comma-separated)", key=f"{key}_extra")
    tickers = list(dict.fromkeys(tickers + [t.strip().upper() for t in extra.split(",") if t.strip()]))
    cap = sb.slider("Max weight per stock", 0.05, 1.0, float(default_cap), 0.01, key=f"{key}_cap", help=tip("Position cap"))
    shrink = sb.slider("Tame noisy return estimates", 0.0, 1.0, 0.0, 0.05, key=f"{key}_shr",
                       help="Pulls each stock's historical return toward the group average, reducing reliance on past winners.")
    lw = sb.checkbox("Shrink covariance (Ledoit-Wolf)", key=f"{key}_lw", help="A more stable risk estimate for short histories.")
    ccy = sb.selectbox("Currency symbol for examples", ["$", "₹", "€", "£"], key=f"{key}_ccy",
                       help="Label only: no FX conversion is applied.")
    amount = sb.number_input("Example investment amount", 1000, 100_000_000, 100_000, 10_000, key=f"{key}_amt")
    st.session_state["ccy"], st.session_state["amount"] = ccy, float(amount)
    demo = sb.checkbox("Demo mode (synthetic data, works offline)", key=f"{key}_demo")
    with sb.expander("📖 Glossary"):
        for term, text in GLOSSARY.items():
            st.markdown(f"**{term}**: {text}")
    try:
        st.query_params["tk"] = ",".join(tickers)
        st.query_params["cap"] = f"{cap:.2f}"
    except Exception:
        pass
    return {"mode": mode, "tickers": tickers, "cap": cap, "shrink": shrink, "lw": lw, "demo": demo,
            "ccy": ccy, "amount": float(amount)}
