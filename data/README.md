# Data snapshot

`snapshot_prices.csv.gz` and `snapshot_ff5.csv.gz` are written by `scripts/refresh_snapshot.py`
(GitHub Action *Refresh data snapshot*, weekdays). The app uses them **only if Yahoo Finance or the
Fama-French site is unreachable**, and says so on every page.

They don't exist until the first run: trigger the workflow once from the Actions tab
(or run `python scripts/refresh_snapshot.py` locally) and commit the files.
