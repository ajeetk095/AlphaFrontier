"""Refresh the committed data snapshot (run by the GitHub Action, or manually: python scripts/refresh_snapshot.py)."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import data as D  # noqa: E402


def main() -> int:
    D.SNAPSHOT_DIR.mkdir(exist_ok=True)
    px = D.yf_close(D.SNAPSHOT_TICKERS, "2015-01-01", date.today())
    missing = [t for t in D.SNAPSHOT_TICKERS if t not in px.columns]
    if missing:
        print("Warning: no data for", missing)
    if len(px) < 1000:
        print("Too few rows; refusing to overwrite snapshot.")
        return 1
    px.round(4).to_csv(D.SNAPSHOT_PRICES)
    print(f"prices: {px.shape} through {px.index.max():%Y-%m-%d}")
    try:
        ff = D.fetch_ff5_daily()
        ff.loc["2015-01-01":].to_csv(D.SNAPSHOT_FF)
        print(f"factors: {ff.shape} through {ff.index.max():%Y-%m-%d}")
    except D.DataError as exc:
        print("Factor refresh skipped:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
