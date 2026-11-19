# 📈 AlphaFrontier

![CI](https://github.com/ajeetk095/AlphaFrontier/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**An interactive, teaching-first portfolio-construction lab.** Explore portfolio optimization, walk-forward testing, Black–Litterman, risk analysis, Fama–French factors, and a REST API. Results are educational analyses—not investment advice.

## Run locally

```bash
git clone https://github.com/ajeetk095/AlphaFrontier.git
cd AlphaFrontier
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To explore without market data access, turn on **Demo mode** in the app. It uses synthetic data and is clearly labeled.

## REST API

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload
```

API documentation is available at `http://localhost:8000/docs`. `POST /optimize` fetches market prices; `POST /optimize/from-moments` accepts annualized expected returns and covariance. The API has input bounds to limit optimization size; deploy behind an appropriate gateway/rate limiter if exposing it publicly.

## Docker

```bash
docker compose up --build
```

Streamlit is served on port 8501 and the API on port 8000.

## Tests and development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q --cov=core --cov=api
```

The tests cover optimization constraints, back-test timing, risk/model calculations, API validation, data fallbacks, and headless rendering of the app pages.

## Data and limitations

Market data is obtained from Yahoo Finance and the Kenneth R. French Data Library. The app can use committed snapshots if live data providers are unavailable; verify the displayed provenance and as-of date. Estimates are sensitive to historical windows and assumptions; back-tests and simulated scenarios do not predict future returns.

> **Disclaimer:** Educational software, not investment advice. Historical estimates are noisy; past performance does not predict future results.

© 2026 Ajeet Kumar · MIT License · [Documentation](docs/DOCUMENTATION.md) · [Deployment](docs/DEPLOYMENT.md) · [Report review](docs/REPORT_REVIEW.md)
