# Deploying the EquiBirth AI dashboard

The dashboard is a single Streamlit page (`src/equibirth/app/dashboard.py`). It is fully self-contained:
every number and figure is read from committed aggregates in `reports/dashboard_data.json`. It needs **no
raw data** (the gitignored ~2 GB natality files are never required) and **no machine-learning libraries**,
so it deploys cleanly on a free tier.

Only four packages are needed at runtime: `streamlit`, `plotly`, `pandas`, `numpy` (see
[`requirements-app.txt`](requirements-app.txt)).

---

## Run it locally first (sanity check)

```bash
PYTHONPATH=src ./.venv/bin/streamlit run src/equibirth/app/dashboard.py
# opens http://localhost:8501
```

If you change any analysis upstream, regenerate the committed aggregates before deploying:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/build_dashboard_data.py
```

---

## Option A (recommended): Streamlit Community Cloud, free shareable URL

Streamlit Community Cloud installs dependencies from the **repository-root `requirements.txt`**. The
project's root file is the heavy full-stack dev environment (torch, xgboost, shap, ...), which is far more
than the dashboard needs and can exceed the free-tier build limit. The clean fix is a small **deploy
branch** whose `requirements.txt` is the slim app list.

1. Create a deploy branch with the slim requirements as the root file:

   ```bash
   git checkout -b deploy
   cp requirements-app.txt requirements.txt
   git commit -am "Deploy: slim requirements for the Streamlit dashboard"
   git push -u origin deploy
   ```

2. Go to **https://share.streamlit.io**, sign in with GitHub, and authorize access to this repository
   (public repos work out of the box; for a private repo, grant Streamlit access).

3. Click **Create app** and fill in:
   - **Repository:** `ParshvCrafts/EquiBirth_AI`
   - **Branch:** `deploy`
   - **Main file path:** `src/equibirth/app/dashboard.py`

4. Click **Deploy**. After the build you get a public URL (for example
   `https://equibirth-ai.streamlit.app`) that anyone can open. Readers can scroll and interact
   (tabs, the race multiselect, the COVID toggle) with no setup.

To update the live app later: merge new work into `deploy` (re-running `build_dashboard_data.py` first if
the numbers changed) and push; Community Cloud redeploys automatically.

> **Privacy note.** Only small aggregates are committed; the raw row-level natality data stay gitignored,
> so the repository and the deployed app are safe to make public.

---

## Option B: deploy from `main` as-is

You can point Streamlit Cloud at `main` directly, but it will then install the full heavy
`requirements.txt` (the whole modelling stack). The build is slow and may hit the free-tier resource
limit. Prefer Option A unless you have a reason to install the full stack.

---

## Alternatives

- **Hugging Face Spaces (Streamlit template):** create a Space, set the SDK to Streamlit, push this repo
  (or just `src/equibirth/app/dashboard.py`, `reports/dashboard_data.json`, and `requirements-app.txt`
  renamed to `requirements.txt`), and it builds the same app.
- **Render / Railway:** start command
  `streamlit run src/equibirth/app/dashboard.py --server.port $PORT --server.address 0.0.0.0`, with
  `requirements-app.txt` as the dependency file.

---

*Author: Parshv Patel ([LinkedIn](https://www.linkedin.com/in/parshv-patel-65a90326b/)). Educational and
research demonstration. Not a clinical tool.*
