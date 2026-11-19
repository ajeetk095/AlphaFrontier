# Deploy to GitHub and Streamlit Community Cloud (free)

## 1 · Put the code on GitHub
1. Create an account at github.com (you already have `ajeetk095`).
2. Click **New repository** → name it `portfolio-lab` → **Public** → don't add a README (we have one) → **Create**.
3. In a terminal, inside the unzipped `portfolio-lab` folder:
```bash
git init
git add .
git commit -m "Portfolio Lab: initial release"
git branch -M main
git remote add origin https://github.com/ajeetk095/portfolio-lab.git
git push -u origin main
```
If GitHub asks for a password, use a **Personal Access Token** (GitHub → Settings → Developer settings → Tokens), or sign in with the GitHub CLI (`gh auth login`).
4. Open the repo's **Actions** tab: the CI workflow runs the tests automatically on every push.

## 2 · Make it live on Streamlit
1. Go to **share.streamlit.io** and sign in with GitHub (authorise access to your repos).
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Repository: `ajeetk095/portfolio-lab` · Branch: `main` · Main file path: `app.py`.
4. Open **Advanced settings** → choose **Python 3.11** (3.12 also works).
5. (Optional) pick a custom subdomain, e.g. `portfolio-lab` → `https://portfolio-lab.streamlit.app`.
6. Click **Deploy**. The first build takes a few minutes (installing cvxpy, statsmodels, etc.).
7. Every `git push` to `main` redeploys automatically.

## 3 · After it's live
- Add the live URL to the repo's **About** box, README, LinkedIn and resume.
- Update `LINKEDIN` in `pages/5_About_and_Hire.py` if your real handle differs (**double-check it**: the bio you gave says `ajeek095`, which may be a typo for `ajeetk095`).
- Optional: add a screenshot to the README.

## 4 · Turn on the data fallback (recommended, 2 minutes)
Yahoo sometimes blocks shared cloud IPs. The repo ships a GitHub Action that saves a daily copy of the data:
1. On GitHub open **Actions → Refresh data snapshot → Run workflow**. It writes `data/snapshot_prices.csv.gz` (and the factor file) and commits them.
2. From then on the app silently falls back to that copy if Yahoo fails, and shows a yellow "Snapshot" badge.
Repo **Settings → Actions → General → Workflow permissions** must allow *Read and write*.

## 5 · Optional: API and Docker
- API: `pip install -r requirements-api.txt && uvicorn api.main:app` (docs at `/docs`). Deploy on Render/Railway/Fly.io with `Dockerfile.api`.
- Both services: `docker compose up --build`. (The Dockerfiles were written but not built in the authoring sandbox. Do a first local build to confirm.)

## Troubleshooting
| Symptom | Fix |
|---|---|
| "Yahoo Finance returned no data" | Yahoo sometimes rate-limits shared cloud IPs. Wait a minute and reload (data is cached 1 h), or tick **Demo mode**. |
| Factor page says factors unavailable | Dartmouth's server was unreachable; retry later. Factors are published with a delay of about 1-2 months, so the newest weeks may be absent by design. |
| Build fails on `cvxpy` | In Advanced settings choose Python 3.11, then **Reboot app**. |
| App "went to sleep" | Free apps sleep after inactivity; click **Wake up**. |
| Want to hide nothing sensitive | No secrets are needed. Never commit `.streamlit/secrets.toml`. |

## Alternatives
Hugging Face Spaces (Streamlit SDK), Render, Railway, or Docker on any VPS (`streamlit run app.py --server.port 8501`).
