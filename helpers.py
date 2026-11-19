"""Shared Streamlit helpers: cached data access, layout, sidebar, and teaching widgets."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.io as pio
import streamlit as st

from core import data as D
from core.glossary import GLOSSARY, tip
from core.insights import DISCLAIMER

REPORT_WINDOWS = {
    "train": (date(2023, 7, 1), date(2025, 6, 30)),
    "test": (date(2025, 7, 1), date(2025, 9, 30)),
}

# A dark-first design system shared by the landing page and every multipage view.
CSS = """
<style>
:root{color-scheme:dark;--pl-bg:#080d19;--pl-panel:#111a2b;--pl-panel-raised:#17233a;--pl-border:#273650;
  --pl-text:#e8eefb;--pl-muted:#9aabc4;--pl-indigo:#8b83ff;--pl-cyan:#48d9e8;--pl-green:#38d9a9}
.stApp{background:radial-gradient(ellipse at 2% 0%,rgba(28,114,170,.20),transparent 34%),
  radial-gradient(ellipse at 98% 4%,rgba(107,73,190,.18),transparent 32%),var(--pl-bg);color:var(--pl-text)}
[data-testid="stAppViewContainer"],section[data-testid="stMain"]{background:transparent}
.block-container{width:min(100% - 2rem,1440px);max-width:1440px;padding:1.35rem 1rem 3.2rem}
[data-testid="stHeader"]{background:rgba(8,13,25,.82);backdrop-filter:blur(14px)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#101a2c,#15152b 72%,#102329);border-right:1px solid var(--pl-border)}
.stApp :is(h1,h2,h3,h4,h5,h6,p,li,label,small,[data-testid="stMarkdownContainer"]){color:var(--pl-text)}
.stApp [data-testid="stCaptionContainer"],.stApp .stCaption{color:var(--pl-muted)}
.stApp a{color:#7edff5}.stApp a:hover{color:#c3a9ff}
.stApp input,.stApp textarea,[data-baseweb="select"]>div,[data-baseweb="input"]>div{
  color:var(--pl-text)!important;background:#101a2b!important;border-color:#34445f!important;border-radius:11px!important}
.stApp input::placeholder,.stApp textarea::placeholder{color:#8b9bb3!important}
.stApp [role="listbox"],.stApp [role="option"]{background:#121d30;color:var(--pl-text)}
[data-testid="stMetric"]{height:100%;padding:.9rem 1rem;border-radius:17px;border:1px solid rgba(139,131,255,.25);
  background:linear-gradient(145deg,rgba(92,88,190,.18),rgba(17,26,43,.86));box-shadow:0 10px 28px rgba(0,0,0,.16)}
[data-testid="stMetricLabel"]{color:#b6c4d9;font-weight:650}[data-testid="stMetricValue"]{color:#d8d1ff;font-size:clamp(1.25rem,3vw,1.8rem);font-weight:750}
[data-testid="stMetricDelta"]{color:#7de6bd}
.pl-hero{position:relative;isolation:isolate;overflow:hidden;padding:clamp(1.35rem,4vw,2.7rem);border-radius:26px;
  background:linear-gradient(115deg,rgba(28,166,196,.20),rgba(93,78,190,.25) 55%,rgba(18,143,124,.15));
  border:1px solid rgba(139,131,255,.36);margin:0 0 1.4rem;box-shadow:0 20px 60px rgba(0,0,0,.24)}
.pl-hero:after{content:"";position:absolute;z-index:-1;width:290px;height:290px;right:-80px;top:-145px;border-radius:50%;
  background:radial-gradient(circle,rgba(72,217,232,.28),transparent 70%);pointer-events:none}
.pl-hero h1{margin:0 0 .55rem;font-size:clamp(2rem,6vw,3.55rem);letter-spacing:-.045em;line-height:1.08}
.pl-hero b{font-size:clamp(1rem,2.4vw,1.15rem);line-height:1.55}.pl-hero p{color:#bdcbe1}
.pl-card{height:100%;min-height:145px;padding:1.15rem;border-radius:18px;border:1px solid rgba(139,131,255,.23);
  background:linear-gradient(145deg,rgba(83,77,165,.18),rgba(17,26,43,.90));line-height:1.6;
  box-shadow:0 12px 30px rgba(0,0,0,.16);transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease}
.pl-card:hover{transform:translateY(-3px);border-color:rgba(72,217,232,.62);box-shadow:0 16px 34px rgba(21,177,200,.12)}
.pl-card b{font-size:1.04rem;color:#f1efff}.pl-card span{color:#b6c4d9}
.stButton button,.stDownloadButton button,.stLinkButton a,[data-testid="stPageLink"] a{
  min-height:2.65rem;border-radius:12px!important;font-weight:700!important;transition:all .18s ease!important}
.stButton button:hover,.stDownloadButton button:hover,.stLinkButton a:hover,[data-testid="stPageLink"] a:hover{
  transform:translateY(-1px);border-color:var(--pl-cyan)!important;box-shadow:0 6px 20px rgba(72,217,232,.16)!important}
[data-testid="stTabs"] button{font-weight:650;color:#aab8cf}
[data-testid="stTabs"] button[aria-selected="true"]{color:#c8c0ff;border-color:var(--pl-indigo)}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid var(--pl-border);border-radius:13px;overflow:auto;background:#0f1828}
[data-testid="stExpander"],[data-testid="stAlert"]{border-radius:15px;border-color:var(--pl-border);background:rgba(17,26,43,.88);color:var(--pl-text)}
[data-testid="stAlert"] :is(p,span){color:inherit}
[data-testid="stPlotlyChart"]{background:transparent}
.stApp hr{border-color:var(--pl-border)}
@media (max-width:760px){
  .block-container{width:100%;padding:.85rem .75rem 2rem}
  [data-testid="stHorizontalBlock"]{flex-direction:column;gap:.65rem}
  [data-testid="stHorizontalBlock"]>[data-testid="column"]{width:100%!important;flex:1 1 100%!important;min-width:100%!important}
  .pl-hero{border-radius:20px;margin-bottom:1rem}.pl-card{min-height:0;margin-bottom:.25rem}
  [data-testid="stMetric"]{padding:.75rem .85rem}
  [data-testid="stTabs"] [role="tablist"]{gap:.15rem;overflow-x:auto;flex-wrap:nowrap}
  [data-testid="stTabs"] button{white-space:nowrap;font-size:.85rem}
  .stButton button,.stDownloadButton button,.stLinkButton a{width:100%}
  [data-testid="stDataFrame"]{max-width:100%;overflow-x:auto}
}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;transition:none!important;animation:none!important}}
</style>
"""


def page_setup(title: str, icon: str = "📈"):
    st.set_page_config(page_title=f"{title} · AlphaFrontier", page_icon=icon, layout="wide")
    # Transparent plot backgrounds inherit the accessible dark app palette.
    pio.templates.default = "plotly_dark"
    st.markdown(CSS, unsafe_allow_html=True)


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
    """Shared sidebar. Syncs tickers/cap to the URL so results are shareable."""
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
