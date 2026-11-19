"""Headless smoke tests: every Streamlit page renders in demo mode without exceptions."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = [
    "pages/1_Learn_and_Quiz.py",
    "pages/2_Portfolio_Optimizer.py",
    "pages/4_Black_Litterman.py",
    "pages/5_Risk_and_Stress.py",
    "pages/6_Model_Comparison.py",
    "pages/7_Factor_Analysis.py",
    "pages/9_Report_and_Methodology.py",
    "pages/10_About the Developer.py",
]


def _demo(at):
    for cb in at.sidebar.checkbox:
        if "Demo mode" in cb.label:
            cb.set_value(True)
    return at


def _click(at, label):
    for b in at.button:
        if label in b.label:
            return b.click()
    raise AssertionError(f"button {label!r} not found")


def test_home_demo():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    at.toggle[0].set_value(True).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", PAGES)
def test_pages_demo(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=120).run()
    _demo(at).run()
    assert not at.exception, at.exception


def test_backtest_runs():
    at = AppTest.from_file(str(ROOT / "pages/3_Walk_Forward_Backtest.py"), default_timeout=300).run()
    _demo(at).run()
    _click(at, "Run walk-forward").run()
    assert not at.exception, at.exception
    assert any("Results after trading costs" in m.value for m in at.markdown)


def test_simulation_runs():
    at = AppTest.from_file(str(ROOT / "pages/8_Simulation_vs_Optimization.py"), default_timeout=300).run()
    _demo(at).run()
    _click(at, "Run experiment").run()
    assert not at.exception, at.exception


def test_noise_lab_runs():
    at = AppTest.from_file(str(ROOT / "pages/6_Model_Comparison.py"), default_timeout=300).run()
    _demo(at).run()
    _click(at, "Run noise lab").run()
    assert not at.exception, at.exception


def test_black_litterman_with_view():
    at = AppTest.from_file(str(ROOT / "pages/4_Black_Litterman.py"), default_timeout=120).run()
    _demo(at).run()
    ms = [m for m in at.multiselect if "opinion" in m.label][0]
    ms.set_value(["AAPL"]).run()
    assert not at.exception, at.exception
