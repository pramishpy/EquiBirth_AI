# HANDOFF.md

Written on 2026-08-19, before switching laptops. Read this first on the new machine.

Purpose: get you from a bare `git clone` to a working setup, and record the things that cannot live
in a git repo so nothing is silently lost.

---

## 1. Where the project stands

**The project is complete.** Nine weeks of planned work are finished and reported. This is not a
half-built pipeline waiting to be resumed; it is a finished deliverable that could be extended.

| Area | State |
|---|---|
| Analysis, weeks 1-9 | Complete. Findings F1-F27 in [`FINDINGS.md`](FINDINGS.md) |
| Decisions | 48 logged in [`DECISION_LOG.md`](DECISION_LOG.md) |
| Data track | Full CDC/NCHS 2018-2024, 24,920,407 singleton births. The earlier ~2M-row sample is **superseded and must not be cited** |
| Reports | Per-week `.md` + `.json`, model card, datasheet, all in `reports/` |
| Notebooks | 13, weeks 1-9, in `notebooks/` |
| Streamlit app | Built and guard-tested, `src/equibirth/app/dashboard.py` |
| Final presentation | 13-slide `.pptx` + word-for-word speaker script, in `presentation/` |
| Future ideas | [`FUTURE_WORK.md`](FUTURE_WORK.md) |

Everything tracked is pushed to `origin/main` at `github.com/ParshvCrafts/EquiBirth_AI`.

### The one thing to know before answering questions about it

The headline is deliberately *not* the best score. The leaky post-delivery model reaches AUC 0.917;
the honest pre-delivery model reaches 0.6561. Removing three reverse-causal conditions (eclampsia,
gestational hypertension, gestational diabetes) cost about 3 AUC points on purpose (D26). The
project's argument is that the weaker number is the true one.

---

## 2. What is NOT in the repo, and what that costs you

Deliberate exclusions. None of it is lost work, but you cannot regenerate analysis without the
first item.

| Missing | Size | Consequence | How to get it back |
|---|---|---|---|
| `data/raw/` natality CSVs | ~13 GB | Cannot re-run any pipeline stage | `bash scripts/week1_download_full.sh` (hours; mirror throttles) |
| `data/interim/`, `data/processed/` | tens of GB | Cannot re-run modeling | Rebuilt by the week1-week2 scripts |
| `.venv/` | - | No environment at all | Section 3 below |
| `.venv-deck/` | - | Cannot rebuild the presentation | Section 3 below |
| Model artifacts | - | Nothing to load | By design: every script retrains from scratch, so reproduction is fully determined by data plus seed 42 |
| `presentation/preview/` | - | No layout renders | Regenerate with `make presentation-check` |

**Nothing else is missing.** This folder held no private notes, no `CLAUDE.md`, no local-only
documents. I verified with `git status --ignored`: the only ignored files present were caches,
`.DS_Store`, the two venvs, and the regenerable preview renders.

**All reported numbers survive without the data.** Every figure quoted anywhere traces to a tracked
JSON in `reports/`. You can answer any question about results, rebuild the deck, and run the
dashboard on a fresh clone with no download at all.

---

## 3. New-laptop setup

### 3.1 Prerequisite: Python 3.11 or newer

This matters. The old laptop's system Python was **3.9.6**, while the project declares
`requires-python = ">=3.11"` and ruff/mypy target py311. Check first:

```bash
python3 --version
```

If it is below 3.11, install a newer one before continuing. On macOS the least painful route:

```bash
brew install python@3.11
```

Then use `python3.11` explicitly in the commands below wherever you see `python3`.

### 3.2 Clone and build the main environment

```bash
git clone https://github.com/ParshvCrafts/EquiBirth_AI.git
cd EquiBirth_AI

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # was broken until 2026-08-19; see section 5
```

Verify:

```bash
make test        # expect ~137 passed, 4 skipped (the 4 skips need the gitignored raw data)
make lint        # ruff, should be clean
```

If tests crash rather than fail, jump to section 5 item 2 (xgboost segfault).

### 3.3 Run the dashboard (no data needed)

The fastest way to confirm the clone is healthy. Reads committed aggregates only, and needs just
four packages.

```bash
pip install -r requirements-app.txt
PYTHONPATH=src streamlit run src/equibirth/app/dashboard.py
# http://localhost:8501
```

See [`DEPLOY.md`](DEPLOY.md) for putting it back on Streamlit Community Cloud.

### 3.4 Rebuild the presentation (no data needed)

The deck lives in its own tiny environment on purpose, so you never need the ML stack to edit slides.

```bash
python3.11 -m venv .venv-deck
./.venv-deck/bin/pip install python-pptx matplotlib pillow ruff

make presentation       PY=./.venv-deck/bin/python    # figures, then the .pptx
make presentation-check PY=./.venv-deck/bin/python    # layout audit + claim verify + timing
```

`presentation-check` must print all three passes: no layout problems, 57 numeric claims verified,
and the script fitting the 10-minute slot. If a claim check fails, a slide number no longer matches
`reports/` and the slide is wrong, not the checker.

### 3.5 Only if you need to re-run the analysis

Requires the 13 GB download first. Full ordered command list is in the
"Reproducing the analysis from a clean clone" section of [`README.md`](README.md). Two constraints
that are easy to trip over:

- **Run heavy training in the foreground.** Backgrounded Python jobs got their child process killed
  in the previous environment (D40). Memory stays around 4-6 GB with the built-in caps.
- **Run weeks 5-8 in order.** They read tuned hyperparameters out of
  `reports/week4_boosting_fairness.json` instead of re-tuning.

---

## 4. Things only you can do (not automatable, not in the repo)

- **Streamlit Community Cloud.** If the app was deployed, the hosting account and its link to this
  GitHub repo are outside version control. Re-check the app still builds after the dependency change
  in section 5. Deploy steps are in `DEPLOY.md`.
- **GitHub credentials on the new machine.** A fresh clone needs auth to push (SSH key or a PAT).
- **The `.pptx` itself.** Editable in PowerPoint, Google Slides, Keynote, or Canva. If you hand-edit
  slides, re-running `make presentation` **overwrites your edits**, because the deck is generated.
  Change the build script instead, or accept that the file is the final artifact.
- **AI4ALL submission artifacts.** Anything uploaded to a course portal is not tracked here.

---

## 5. Known issues, and their current status

**1. `pip install -r requirements.txt` used to fail outright. Fixed on 2026-08-19.**

`rtdl 0.0.13` (FT-Transformer) pins `torch>=1.7,<2` and `numpy<2`, which cannot coexist with the
project's `torch>=2.3` or with `samplics` needing `numpy>=2.1`. The set was unsatisfiable, so the
install failed and **CI was red at the install step** on commit `8798572`. Fixed by moving `rtdl`
into [`requirements-deep.txt`](requirements-deep.txt) as an isolated optional env.

Nothing in the pipeline depends on it: `rtdl` is imported lazily inside `_FTWrapper.__init__`
(`src/equibirth/models/deep.py`) and the deep tests use `pytest.importorskip`, so the suite skips
cleanly. The FT-Transformer benchmark is already complete with results archived in
`reports/week5_models.json`.

**Verify CI is green on the new machine before trusting the install.** If it is still red, read the
Actions log rather than assuming this fix was sufficient.

**2. xgboost segfaulted on the old laptop (macOS).** Crashed the interpreter on import, which
blocked 5 test files and left the suite at 117 passed instead of 137. Reproduced on xgboost 3.2.0
and 2.1.4, under both numpy 1.26.4 and 2.4.6, so it was environment-specific rather than a code
defect. Most likely the OpenMP runtime; try `brew install libomp`. **Check whether it reproduces at
all on the new machine before spending time on it.**

**3. All dependency versions are floors, not pins.** Two clean clones months apart will resolve
differently. Once the new environment is verified good, `pip freeze > requirements.lock.txt` and
commit it. Related: `shap>=0.52` now requires Python 3.12+, so a 3.11 environment quietly resolves
to an older shap.

---

## 6. If you want to keep going

[`FUTURE_WORK.md`](FUTURE_WORK.md) has nine items with blockers and costs. Short version:

- **Cheapest real result:** the multiclass severity target. Designed, never run, blocked only on
  retaining `oegest_comb` in the modeling Parquet (~30 min of preprocessing).
- **Most important result:** external data the certificate lacks (neighborhood, environment, care
  access). Blocked by NCHS geography suppression post-2005, so it means aggregate-level analysis or
  a restricted-use file application.
- **Do not** spend time tuning the existing model. F16 and F27 already show the tree field clusters
  within 0.008 AUC and more rows buy 0.003. That avenue is closed by our own evidence.

---

## 7. Sanity checklist for the new machine

```
[ ] python3 --version reports 3.11 or newer
[ ] git clone succeeded and `git log -1` shows the handoff commit
[ ] pip install -r requirements.txt completes without a resolver error
[ ] make test         -> ~137 passed, 4 skipped
[ ] make lint         -> clean
[ ] streamlit app opens and shows the disclaimer banner
[ ] make presentation-check PY=./.venv-deck/bin/python -> all three checks pass
[ ] GitHub Actions CI is green on the latest commit
```

If every line passes, you have lost nothing.
