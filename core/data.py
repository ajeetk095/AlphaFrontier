"""Data access: Yahoo Finance (with retry + committed snapshot fallback), Kenneth French factors, demo data."""
from __future__ import annotations

import io
import time
import zipfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["TSLA", "WMT", "BAC", "GS", "LLY", "MRK", "GOOG", "META", "AAPL", "XOM"]
EXTRA_TICKERS = ["MSFT", "AMZN", "NVDA", "JPM", "JNJ", "PG", "KO",
                 "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
INDEX_SYMBOLS = ["^GSPC", "^VIX", "^IRX", "^NSEI"]
SNAPSHOT_TICKERS = DEFAULT_TICKERS + EXTRA_TICKERS + INDEX_SYMBOLS
FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
TRADING_DAYS = 252
FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_PRICES = SNAPSHOT_DIR / "snapshot_prices.csv.gz"
SNAPSHOT_FF = SNAPSHOT_DIR / "snapshot_ff5.csv.gz"
RETRIES = 3
RETRY_SLEEP = 1.5  # seconds, doubled each retry


class DataError(RuntimeError):
    """Raised when market data cannot be retrieved or is unusable."""


def _retry(fn):
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            return fn()
        except DataError as exc:
            last = exc
            if attempt < RETRIES - 1:
                time.sleep(RETRY_SLEEP * 2 ** attempt)
    raise last  # type: ignore[misc]


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        raise DataError("Yahoo Finance returned no data. Check tickers/dates or try again later.")
    close = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.dropna(axis=1, how="all")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def yf_close(tickers: list[str], start, end) -> pd.DataFrame:
    """Raw Yahoo download with retry. May miss tickers; callers decide how to handle that."""
    import yfinance as yf

    def attempt():
        try:
            raw = yf.download(tickers, start=pd.Timestamp(start), end=pd.Timestamp(end) + timedelta(days=1),
                              auto_adjust=True, progress=False)
        except Exception as exc:  # network, rate limit, parsing
            raise DataError(f"Yahoo Finance request failed: {exc}") from exc
        return _extract_close(raw, tickers)

    return _retry(attempt)


def download_prices(tickers: list[str], start, end) -> pd.DataFrame:
    """Adjusted close prices from Yahoo (end inclusive). Raises DataError if any ticker is missing."""
    prices = yf_close(tickers, start, end)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise DataError(f"No data returned for: {', '.join(missing)}")
    return prices[tickers]


def read_snapshot() -> pd.DataFrame | None:
    """Committed daily snapshot (refreshed by the GitHub Action), if present."""
    if not SNAPSHOT_PRICES.exists():
        return None
    df = pd.read_csv(SNAPSHOT_PRICES, index_col=0, parse_dates=True)
    return df.sort_index()


def get_prices(tickers: list[str], start, end) -> tuple[pd.DataFrame, str, pd.Timestamp]:
    """Prices with graceful degradation. Returns (prices, source, as_of); source is 'live' or 'snapshot'."""
    try:
        px = download_prices(tickers, start, end)
        return px, "live", px.index.max()
    except DataError as live_err:
        snap = read_snapshot()
        if snap is None:
            raise DataError(f"{live_err} No snapshot file is available either.") from live_err
        missing = [t for t in tickers if t not in snap.columns]
        if missing:
            raise DataError(f"{live_err} Snapshot lacks: {', '.join(missing)}.") from live_err
        sub = snap[tickers].loc[pd.Timestamp(start):pd.Timestamp(end)].dropna(how="all")
        if sub.empty:
            raise DataError(f"{live_err} Snapshot has no rows in the requested range.") from live_err
        return sub, "snapshot", sub.index.max()


def get_series(symbol: str, start, end) -> tuple[pd.Series, str, pd.Timestamp]:
    px, src, as_of = get_prices([symbol], start, end)
    return px.iloc[:, 0].rename(symbol), src, as_of


def fetch_rf_annual(start, end, fallback: float = 0.04) -> tuple[float, bool]:
    """Average 13-week T-bill yield (^IRX) as an annual decimal. Returns (value, is_available)."""
    try:
        irx, _, _ = get_series("^IRX", start, end)
        return float(irx.dropna().mean() / 100.0), True
    except DataError:
        return fallback, False


def fetch_ff5_daily() -> pd.DataFrame:
    """Daily Fama-French 5 factors + RF from Kenneth French's library, as decimals."""
    import requests

    def attempt():
        try:
            resp = requests.get(FRENCH_URL, timeout=30)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                return zf.read(zf.namelist()[0]).decode("latin-1")
        except Exception as exc:
            raise DataError(f"Could not download Fama-French factors: {exc}") from exc

    text = _retry(attempt)
    lines = text.splitlines()
    header = next((i for i, ln in enumerate(lines) if "Mkt-RF" in ln), None)
    if header is None:
        raise DataError("Unexpected Fama-French file format.")
    rows = []
    for ln in lines[header + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 7 or not parts[0].isdigit():
            break
        rows.append(parts[:7])
    df = pd.DataFrame(rows, columns=["date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"])
    df.index = pd.to_datetime(df.pop("date"), format="%Y%m%d")
    return df.astype(float) / 100.0


def get_ff5() -> tuple[pd.DataFrame, str]:
    try:
        return fetch_ff5_daily(), "live"
    except DataError as live_err:
        if SNAPSHOT_FF.exists():
            return pd.read_csv(SNAPSHOT_FF, index_col=0, parse_dates=True), "snapshot"
        raise live_err


def make_demo_data(tickers: list[str], start, end, seed: int = 7):
    """Synthetic prices + factors for offline demos and tests. NOT real market data."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    T, n = len(idx), len(tickers)
    vol = np.array([0.0100, 0.0055, 0.0059, 0.0040, 0.0038])
    corr = np.array(
        [[1, -0.05, -0.17, -0.21, -0.35],
         [-0.05, 1, 0.15, -0.28, 0.02],
         [-0.17, 0.15, 1, 0.06, 0.54],
         [-0.21, -0.28, 0.06, 1, 0.12],
         [-0.35, 0.02, 0.54, 0.12, 1]]
    )
    F = rng.standard_normal((T, 5)) @ np.linalg.cholesky(corr).T * vol
    F += np.array([0.0004, 0.0001, 0.0001, 0.0001, 0.0001])
    ff = pd.DataFrame(F, index=idx, columns=FACTORS)
    ff["RF"] = 0.00018
    betas = rng.normal([1.1, 0, 0, 0, 0], [0.3, 0.4, 0.4, 0.3, 0.3], size=(n, 5))
    idio = rng.standard_normal((T, n)) * rng.uniform(0.008, 0.02, size=n)
    alpha = rng.normal(0.0003, 0.0004, size=n)
    R = F @ betas.T + idio + alpha + 0.00018
    prices = pd.DataFrame(100 * np.cumprod(1 + R, axis=0), index=idx, columns=tickers)
    bench = pd.Series(100 * np.cumprod(1 + F[:, 0] + 0.00018), index=idx, name="^GSPC")
    return prices, bench, ff
