from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components

import helpers as H

H.page_setup("Report & Methodology", "📘")
st.title("📘 Research report, explained simply")
st.caption(
    "MScFE 652 Portfolio Management · Group Work Project 1 · WorldQuant University · "
    "Group 13321 (Ajeet Kumar, Rajkumar S, Yijing Hong)"
)

REPO_URL = "https://github.com/ajeetk095/AlphaFrontier"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
FILES = {
    "final_report": "MScFE_652_PMP1_Final Report_13321.pdf",
    "assignment": "MScFE 652 PM GWP1 Set E (09.30.25).pdf",
    "notebook": "PMP1.ipynb",
    "review": "REPORT_REVIEW.md",
    "technical": "DOCUMENTATION.md",
}


def repo_file_url(filename: str) -> str:
    return f"{REPO_URL}/blob/main/docs/{quote(filename)}"


def show_pdf(filename: str, heading: str, description: str) -> None:
    """Offer a PDF preview when available, plus reliable open/download links."""
    path = DOCS_DIR / filename
    st.markdown(f"### {heading}")
    st.caption(description)
    link_col, download_col = st.columns(2)
    with link_col:
        st.link_button("Open PDF on GitHub", repo_file_url(filename), use_container_width=True)
    if path.is_file():
        data = path.read_bytes()
        with download_col:
            st.download_button(
                "Download PDF",
                data=data,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
                key=f"download_{filename}",
            )
        encoded = base64.b64encode(data).decode("ascii")
        components.html(
            f'<iframe src="data:application/pdf;base64,{encoded}" '
            'style="width:100%;height:760px;border:1px solid #d6d9df;border-radius:8px" '
            'title="PDF preview"></iframe>',
            height=775,
            scrolling=True,
        )
    else:
        st.info("The PDF preview is unavailable in this deployment. Open the file on GitHub instead.")


st.markdown(
    """
## The project in a nutshell

This research studies how to combine **10 large US stocks** into portfolios with different
risk and return trade-offs. It compares mathematical optimisation with random portfolio
simulation, tests portfolio results on a later period, and uses the Fama–French five-factor
model to describe which broad investment characteristics were associated with returns.

The original analysis uses prices from **1 July 2023 to 30 June 2025** and an out-of-sample
test period from **1 July to 30 September 2025**. The stocks are Tesla, Walmart, Bank of
America, Goldman Sachs, Eli Lilly, Merck, Alphabet, Meta, Apple and ExxonMobil.
"""
)

with st.expander("📄 Source documents and analysis files", expanded=True):
    st.markdown(
        "The original submission, assignment instructions, and executable notebook are included "
        "below. The notebook preview is rendered in the next tab; use its download button to run it locally."
    )
    doc_tabs = st.tabs(["Final research report (PDF)", "Project instructions (PDF)", "Review notes"])
    with doc_tabs[0]:
        show_pdf(
            FILES["final_report"],
            "Final report submitted for the project",
            "Original academic report. See the review tab and the explanation below for important methodology caveats.",
        )
    with doc_tabs[1]:
        show_pdf(
            FILES["assignment"],
            "Project assignment and instructions",
            "The course project brief used to define the analysis and deliverables.",
        )
    with doc_tabs[2]:
        st.markdown("### Report-versus-notebook review")
        st.warning(
            "The submitted report and notebook contain differences in some methods and reported values. "
            "Treat the PDF as the original submission, not as a fully reconciled reproduction."
        )
        st.markdown(
            "The review flags an objective mismatch, possible reuse of in-sample values in the "
            "out-of-sample table, differences in factor-regression windows, an inconsistent F-statistic, "
            "and discrepancies in simulation assumptions and some table values."
        )
        st.link_button("Read full report review", repo_file_url(FILES["review"]))
        st.link_button("Read technical documentation", repo_file_url(FILES["technical"]))

st.markdown("## Results explained")
result_tabs = st.tabs(["Portfolio construction", "Factor analysis", "Simulation", "What to conclude"])
with result_tabs[0]:
    st.markdown(
        """
### Why does the optimiser concentrate?

The unconstrained portfolio in the report places large weights in a small number of stocks
(about 38% Apple and 21% ExxonMobil). An optimiser responds to estimated returns and risk; it
does not automatically know that investors may want to limit company-specific exposure.

A **15% per-stock cap** forces a more even allocation. That constraint removes some choices,
so its efficient frontier cannot be mathematically better than the unconstrained frontier.
The practical benefit is reduced reliance on any single stock. That can be worthwhile even
when it comes with a modest estimated return/risk trade-off.

The report's fixed-period figures are historical estimates—not promises about future performance.
"""
    )
with result_tabs[1]:
    st.markdown(
        """
### What do the five factors mean?

The model compares portfolio returns with the market and four stock-style factors:

- **Market (MKT):** broad market movement.
- **SMB:** small companies versus large companies.
- **HML:** value stocks versus growth stocks.
- **RMW:** more profitable versus less profitable companies.
- **CMA:** conservative versus aggressive investment by companies.

The report describes market exposure as the largest driver (beta around **0.67**) and reports
a positive value tilt (HML around **0.34**). It did not find clear evidence for the other three
style exposures. These are associations in the chosen historical sample, not proof of cause or
reliable forecasts.
"""
    )
with result_tabs[2]:
    st.markdown(
        """
### Can random portfolios replace optimisation?

The project generates random weights and compares their results with an exact optimiser.
Random search can approach the optimum after many trials, but a position cap means many
random allocations must be rejected or adjusted. This slows the search.

**Takeaway:** use an optimiser when you need a solution to a defined objective; use simulation
to explore possibilities, uncertainty, and scenarios.
"""
    )
with result_tabs[3]:
    st.markdown(
        """
### How should a visitor interpret these results?

1. Diversification limits can reduce concentration risk, even if they restrict the optimiser.
2. Market movement explains more of this portfolio's historical behaviour than the reported style factors.
3. Optimised portfolios are sensitive to estimated returns and can overemphasise recent winners.
4. The sample is short and the out-of-sample period is only three months. Neither is sufficient to establish future performance.
5. The analysis is educational and is not investment advice.
"""
    )

st.markdown("## Methodology, without the jargon")
method_tabs = st.tabs(["Optimisation", "Testing", "Factor model", "Limits & caveats"])
with method_tabs[0]:
    st.markdown(
        """
The analysis estimates each stock's average return and how its returns move with the others.
It then chooses weights that satisfy the portfolio rules (weights sum to 100%; long-only, with
optional per-stock limits) and optimises the selected risk/return objective. The application
also includes minimum-variance and minimum-CVaR methods. See the [technical methods]("""
        + repo_file_url(FILES["technical"])
        + ")."
    )
with method_tabs[1]:
    st.markdown(
        """
A chronological split keeps the later observations out of the training data. For an
out-of-sample test, weights are estimated on the earlier period and applied to returns in the
later period. The current app computes test-period metrics from test-period returns; that is
important because the original notebook's report table may reuse in-sample return and Sharpe
values. The app also includes a walk-forward back-test with rebalancing and trading-cost assumptions.
"""
    )
with method_tabs[2]:
    st.markdown(
        r"""
The factor regression models portfolio **excess return** (portfolio return minus the risk-free
rate) as a combination of the five factors. OLS estimates the average relationships; Newey–West
(HAC) errors account for some time-series dependence in the uncertainty estimates. Huber
regression reduces the influence of unusually large observations. Neither method removes the
limitations of a short or unusual sample.
"""
    )
with method_tabs[3]:
    st.markdown(
        """
- Two years is a short period for estimating expected returns and risk.
- Historical winners can dominate an optimiser; this is a form of hindsight/selection risk.
- The original report and notebook have documented inconsistencies; consult the review notes above.
- Market data may have gaps or revisions, and factor data is published with a lag.
- Back-tests and scenarios describe historical or modelled outcomes. They do not predict the market.
- The application is an educational tool, not investment advice.
"""
    )

st.markdown("## Explore the original notebook")
notebook_path = DOCS_DIR / FILES["notebook"]
notebook_col, notebook_download_col = st.columns([1, 1])
with notebook_col:
    st.link_button("Open notebook on GitHub", repo_file_url(FILES["notebook"]), use_container_width=True)
with notebook_download_col:
    if notebook_path.is_file():
        notebook_bytes = notebook_path.read_bytes()
        st.download_button(
            "Download PMP1.ipynb",
            data=notebook_bytes,
            file_name=FILES["notebook"],
            mime="application/x-ipynb+json",
            use_container_width=True,
        )
    else:
        st.info("Notebook download is unavailable in this deployment; use the GitHub link.")

if notebook_path.is_file():
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = notebook.get("cells", [])
        st.caption(f"Notebook preview · {len(cells)} cells · Python 3")
        with st.expander("Show notebook cells", expanded=False):
            for index, cell in enumerate(cells, start=1):
                cell_type = cell.get("cell_type", "cell")
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                with st.expander(f"Cell {index} · {cell_type.title()}", expanded=(index == 1)):
                    if cell_type == "markdown":
                        st.markdown(source)
                    elif source.strip():
                        st.code(source, language="python")
                    for output in cell.get("outputs", []):
                        text = output.get("text", "")
                        if isinstance(text, list):
                            text = "".join(text)
                        if text:
                            st.code(text[:6000], language="text")
                        data = output.get("data", {})
                        plain = data.get("text/plain", "")
                        if isinstance(plain, list):
                            plain = "".join(plain)
                        if plain:
                            st.code(plain[:6000], language="text")
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        st.warning(f"Could not render a notebook preview ({exc}). Download or open the original notebook instead.")
else:
    st.info("Notebook preview is unavailable in this deployment. Open the notebook on GitHub instead.")

st.markdown("## Further reading")
st.link_button("Technical documentation and detailed formulas", repo_file_url(FILES["technical"]))
H.footer()
