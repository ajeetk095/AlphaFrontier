"""Data fallback, API, PDF and plumbing tests."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import data as D
from core.blacklitterman import implied_returns
from core.optimize import annualised_inputs, max_sharpe
from core.pdf_report import build_pdf


def test_retry_then_success(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(D.time, "sleep", lambda s: None)

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise D.DataError("boom")
        return "ok"

    assert D._retry(flaky) == "ok" and calls["n"] == 3


def test_retry_gives_up(monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)
    with pytest.raises(D.DataError):
        D._retry(lambda: (_ for _ in ()).throw(D.DataError("down")))


def test_snapshot_fallback(monkeypatch, tmp_path):
    idx = pd.bdate_range("2024-01-01", periods=50)
    snap = pd.DataFrame(np.linspace(100, 120, 50)[:, None] * np.ones((1, 3)), index=idx, columns=["AAA", "BBB", "CCC"])
    f = tmp_path / "snap.csv.gz"
    snap.to_csv(f)
    monkeypatch.setattr(D, "SNAPSHOT_PRICES", f)
    monkeypatch.setattr(D, "download_prices", lambda *a, **k: (_ for _ in ()).throw(D.DataError("offline")))
    px, source, as_of = D.get_prices(["AAA", "BBB"], "2024-01-01", "2024-12-31")
    assert source == "snapshot" and list(px.columns) == ["AAA", "BBB"] and as_of == idx[-1]
    with pytest.raises(D.DataError, match="Snapshot lacks"):
        D.get_prices(["ZZZ"], "2024-01-01", "2024-12-31")


def test_no_snapshot_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "SNAPSHOT_PRICES", tmp_path / "missing.csv.gz")
    monkeypatch.setattr(D, "download_prices", lambda *a, **k: (_ for _ in ()).throw(D.DataError("offline")))
    with pytest.raises(D.DataError, match="No snapshot"):
        D.get_prices(["AAA"], "2024-01-01", "2024-12-31")


def test_bl_no_views_max_sharpe_equals_prior():
    prices, _, _ = D.make_demo_data(D.DEFAULT_TICKERS, "2023-01-02", "2025-06-30")
    _, cov = annualised_inputs(prices.pct_change().dropna())
    prior = pd.Series(np.random.default_rng(3).dirichlet(np.ones(10) * 3), index=cov.index)
    pi = implied_returns(cov, prior, 2.5)
    w = max_sharpe(pi + 0.04, cov, 0.04, 1.0)
    assert np.allclose(w.values, prior.values, atol=1e-4)


def test_pdf_builds():
    out = build_pdf("T", "s", [("H", ["text **bold** & <x>", pd.DataFrame({"a": ["1"]}, index=["r"])])], "disc")
    assert out.startswith(b"%PDF") and len(out) > 1000


client = TestClient(app)


def test_api_health_models():
    assert client.get("/health").json() == {"status": "ok"}
    assert "HRP" in client.get("/models").json()["models"]


def test_api_from_moments():
    body = {"names": ["A", "B", "C"], "mu": [0.10, 0.07, 0.12], "cov": [[0.04, 0.01, 0.0], [0.01, 0.02, 0.0], [0.0, 0.0, 0.09]],
            "rf": 0.03, "cap": 0.5}
    r = client.post("/optimize/from-moments", json=body)
    assert r.status_code == 200
    w = r.json()["weights"]
    assert sum(w.values()) == pytest.approx(1, abs=1e-4) and max(w.values()) <= 0.5 + 1e-6
    bad = client.post("/optimize/from-moments", json={**body, "cap": 0.2})
    assert bad.status_code == 422
    asym = client.post("/optimize/from-moments", json={**body, "cov": [[0.04, 0.5, 0], [0.01, 0.02, 0], [0, 0, 0.09]]})
    assert asym.status_code == 422


def test_api_optimize_uses_data_layer(monkeypatch):
    prices, _, _ = D.make_demo_data(["AAPL", "XOM", "WMT", "BAC"], "2023-01-02", "2025-06-30")
    monkeypatch.setattr(D, "get_prices", lambda *a, **k: (prices, "snapshot", prices.index.max()))
    monkeypatch.setattr(D, "fetch_rf_annual", lambda *a, **k: (0.04, True))
    r = client.post("/optimize", json={"tickers": ["AAPL", "XOM", "WMT", "BAC"], "model": "HRP", "cap": 0.5})
    assert r.status_code == 200, r.text
    assert r.json()["data_source"] == "snapshot"
    assert client.post("/optimize", json={"tickers": ["A", "B"]}).status_code == 422
