from urllib.parse import quote

import streamlit as st

import helpers as H

H.page_setup("About & Hire", "👤")

EMAIL = "ajeetk095@gmail.com"
GITHUB = "https://github.com/ajeetk095"
LINKEDIN = "https://www.linkedin.com/in/ajeek095/"  # verify this handle before publishing


def mailto(subject: str, body: str) -> str:
    return f"mailto:{EMAIL}?subject={quote(subject)}&body={quote(body)}"


st.title("👤 About the developer")
st.markdown("## Ajeet Kumar")
st.markdown("### Bridging code & capital.")
st.markdown(
    "Quantitative finance specialist with an **M.Sc. in Financial Engineering** and a **B.Tech in IT**. "
    "I turn complex data into alpha using **Python, C++, SQL and ML**: from Markowitz MVO and Black-Litterman to Monte Carlo and deep learning."
)
c1, c2, c3 = st.columns(3)
c1.metric("🥇 Hackathon", "1st Place", "IIT Kanpur ML Hackathon")
c2.metric("🏅 Recognition", "Felicitated", "by the President of India")
c3.metric("🤖 AI systems built at", "Algo8.ai · Amul")

st.markdown(
    """
**Seeking full-time Quant Research / Financial Engineering roles** to solve high-stakes problems where finance meets engineering.

**Background:** M.Sc. Financial Engineering (WorldQuant University) · M.Tech Computer Science (NSUT Delhi) · B.Tech IT (AKTU).
**Core stack:** Python · C++ · SQL · machine learning · convex optimisation · time-series & factor models.
"""
)

st.markdown("## What this app demonstrates")
st.markdown(
    """
| Skill | Where to see it |
|---|---|
| Convex optimisation, constraints | Optimizer, Model comparison |
| Walk-forward testing, transaction costs, statistical testing | Walk-forward back-test |
| Bayesian portfolio construction | Black-Litterman |
| Risk management (VaR/CVaR, stress tests) | Risk & stress |
| Econometrics (HAC, robust regression, OOS validation) | Factor analysis |
| Monte Carlo & numerical methods | Simulation vs optimization |
| Software engineering: tests, CI, API, Docker, data fallbacks | The repository |
"""
)

st.markdown("## 🤝 Hire or work with me")
st.write("Pick the option that fits and your email app opens with a ready-to-send message.")
a, b, c = st.columns(3)
with a:
    st.markdown("#### 💼 Full-time role")
    st.caption("Quant research, financial engineering, data science, AI/ML.")
    st.link_button("Email about a role", mailto("Full-time opportunity: Quant Research / Financial Engineering",
        "Hi Ajeet,\n\nI came across Portfolio Lab and would like to discuss a role at our company.\n\nCompany:\nRole:\nLocation / remote:\n\nBest,\n"), use_container_width=True)
with b:
    st.markdown("#### 🛠️ Freelance / project")
    st.caption("Dashboards, backtests, risk tools, ML prototypes.")
    st.link_button("Email about a project", mailto("Project enquiry: quantitative tooling",
        "Hi Ajeet,\n\nI'd like to discuss a project.\n\nWhat we need:\nTimeline:\nBudget range:\n\nBest,\n"), use_container_width=True)
with c:
    st.markdown("#### 📚 Collaboration / talk")
    st.caption("Research, mentoring, guest sessions.")
    st.link_button("Email to collaborate", mailto("Collaboration idea",
        "Hi Ajeet,\n\nI'd love to explore working together on:\n\n\nBest,\n"), use_container_width=True)

d, e, f = st.columns(3)
d.link_button("🔗 LinkedIn", LINKEDIN, use_container_width=True)
e.link_button("💻 GitHub", GITHUB, use_container_width=True)
f.link_button("✉️ ajeetk095@gmail.com", f"mailto:{EMAIL}", use_container_width=True)

st.markdown("## About this project")
st.markdown(
    "Portfolio Lab began as **MScFE 652 Group Work Project 1** at WorldQuant University (Group 13321: Ajeet Kumar, Rajkumar S, Yijing Hong) "
    "and was turned into a live, open-source app by Ajeet Kumar. Data: Yahoo Finance and Kenneth R. French's Data Library."
)
H.footer()
