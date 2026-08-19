# FUTURE_WORK.md

Everything considered but not built, with enough context to pick it up cold. Each item says what it
is, why it matters, what blocks it, and roughly what it costs.

Ordered by how much it would change the project's conclusions, not by how easy it is.

The framing rules in [`README.md`](README.md) apply to every item here: race is a marker of exposure
to structural conditions and racism, never a biological cause; nothing here becomes a clinical tool;
fairness is never reduced to one number.

---

## Tier 1: follows directly from what we already found

### 1. Bring in data the birth certificate does not contain

**Why this is first.** The central result is that the ceiling is in the data, not the model. The
entire tree field plus two ensembles clustered inside 0.008 AUC (F16), and training rows past 250K
buy 0.003 AUC-PR (F27). More modeling on the same 103 features is the one thing the results say will
not help. The way forward is different features, not better fitting.

What the structural reading predicts would matter, none of which is on the certificate: neighborhood
deprivation, residential segregation, environmental exposure, distance to obstetric care, housing
instability, experienced discrimination.

**The blocker is real and must be stated.** NCHS suppresses state and county identifiers in
public-use files after 2005 (see `reports/model_card.md` limitations). There is no record-level
geographic key, so a row-level join to ADI, County Health Rankings, or EPA data is not possible with
the public file. Three honest paths:

- **Aggregate-level (free, doable now).** Pull county preterm rates from CDC WONDER, join to
  county-level deprivation indices, and analyze at the county level. Ecological inference only, and
  it must be labeled as such: county associations do not license individual claims.
- **Restricted-use NCHS files (free but slow).** NCHS has a research data centre process for
  geographic detail. Requires a proposal and approval, and results can only leave under their rules.
- **State-level partnership.** Some states release linked birth-and-Medicaid data under DUA.

**Cost:** aggregate path is days. Restricted-use path is months of lead time.

### 2. Run the multiclass severity target

Already designed as a protocol and deliberately not run (`DECISION_LOG` D45,
`reports/week9_external_context.json` field `multiclass_protocol`). Four classes: very preterm
(<28w), moderate (28-31w), late preterm (32-36w), term (37w+).

**Why it matters.** A 27-week birth and a 36-week birth are not the same event clinically or
morally, and binary framing hides that. The disparity may well concentrate in the severe tail, which
would make the equity finding *worse* than currently reported, and late-preterm cases dominate the
current positive class.

**The only blocker is plumbing.** The exact-week gestation column (`oegest_comb`) is not retained in
the modeling Parquet; only the binary indicator survives. Reconstructing it means re-running the
preprocessing pipeline, estimated at 30+ minutes of compute, not a research problem.

**Cheapest genuine win available.** Start here if you want a result quickly.

### 3. Give deep learning a fair fight

The project was careful never to claim trees beat deep learning, because the comparison was
compute-capped: the nets saw 30K rows for 3 epochs against the boosters' 500K, with hand-set configs
and no hyperparameter search (D46, F20). The honest follow-up is to actually test it.

**Why it could move**, unlike more trees: F27's plateau is an *XGBoost* property. Deep tabular
models are the archetype that keeps improving with data and tuning, so this is the one modeling
avenue not already closed by our own evidence.

**Blockers.** The FT-Transformer's O(n) CPU forward pass was the binding constraint (about 14 min at
d_token=32). This is a GPU-hours problem, not a science problem. Also see the dependency mess in
[`requirements-deep.txt`](requirements-deep.txt): `rtdl` pins torch<2, so it needs its own env, and
a modern rewrite would be better than reviving it.

**Do it properly or not at all:** matched training rows, real hyperparameter search, same held-out
partition, same calibration step. If trees still win, that becomes a claim you can actually make.

---

## Tier 2: methodological depth on findings already in hand

### 4. Redo the decomposition without the linearity assumption

The 19-43% explained range (F24) comes from g-computation standardization over a **logistic** model,
linear in covariates. The SHAP dependence plots from the same week show the real risk surface is
nonlinear and interacted. So the range partly reflects functional form, not just covariate choice,
and the report says so.

**What to do.** Re-estimate with a flexible outcome model (gradient-boosted g-computation, or the
Fairlie nonlinear decomposition that D16 named but which was implemented as g-computation). If the
explained share moves materially, the honest range widens. If it holds, the finding hardens.

Keep the mediation framing (D30, F24). These covariates are downstream of structural racism, so
"explained" means transmitted through measured mediators, and the residual stays a **lower bound**.

### 5. Intersectional fairness beyond race alone

The pre-registered comparison family in `FINDINGS.md` included education and payment strata, but the
mitigation table (`reports/week6_mitigation.md`) is race-by-reference-group only.

**Question worth answering.** Does the 2.74x flagging rate for NH Black mothers concentrate in
specific intersections, for instance Medicaid-insured *and* low education, or is it flat across
them? That changes who a screening tool would actually burden.

Watch the multiplicity: intersectional cells multiply comparisons fast. Extend the existing
Benjamini-Hochberg family rather than testing freely, and label anything added now as post-hoc, per
the pre-registration rule already in `FINDINGS.md`.

### 6. Stabilize the small groups

Every wobbly number in the project comes from the same two groups. NH NHOPI has n=801 with 94
positives in the fairness subsample; NH AIAN has n=2,169 with 226. Consequences already observed:

- Reweighting fixed NH Black (+0.389 to +0.017) but drove the **worst** EO gap up to 0.504, entirely
  from NHOPI (F22). The mitigation looked good on average and broke the smallest group.
- AIAN and NHOPI also carry the largest race-misclassification error in the source data, so their
  point estimates are biased *and* noisy.

**What to do.** Hierarchical / partial-pooling estimators that borrow strength across groups instead
of treating each independently. This is the correct fix for the project's weakest evidence, and it
would let the mitigation comparison be judged on worst-group performance without the noise
dominating.

---

## Tier 3: what doing this for real would require

### 7. External validation on a clinical cohort

`reports/model_card.md` already says: do not deploy prospectively without this. Worth being precise
about why the existing evidence does not cover it. F25's temporal stability (AUC-ROC 0.653-0.655
across the COVID regime change) is *within* the NCHS system, same instrument and coding across
2018-2024. It says nothing about transporting to an EHR, a different population, or a care setting
with different documentation practice.

### 8. Annual re-validation as a standing job

Already specified in the model card recommendations: re-run the drift table on each new NCHS annual
release. Trigger reference is the 2018-2024 band (AUC-ROC 0.6556-0.6693); degradation beyond 0.02
AUC-ROC warrants retraining. This is a scheduled job, not research.

### 9. Things deliberately ruled out, and why

Recorded so nobody re-opens them without new information.

| Ruled out | Why | Where |
|---|---|---|
| Custom neural network | Untested code, leakage and calibration risk, no expected gain | D41 |
| SMOTE by default | ~9% prevalence is mild imbalance; harms calibration without helping AUC | D3 |
| Linear Oaxaca-Blinder decomposition | Invalid for a logistic outcome | D14, D16 |
| Aequitas | Heavy dependency, redundant with the tested hand-rolled metrics plus Fairlearn cross-check | D33 |
| Shipping ThresholdOptimizer | Requires race-specific decision thresholds, i.e. disparate treatment | D43 |
| Any individual risk calculator in the app | 82% of predictions sit within 5pp of the base rate | D31, F18 |
| Microdata linkage to external sources | Not legal or ethical with public-use birth records | D44 |

---

## Engineering debt worth clearing

Small, unglamorous, and it will bite whoever picks this up.

1. **`requirements.txt` could not resolve, now fixed.** `rtdl` pinned torch<2 and numpy<2 against
   the project's torch>=2.3 and samplics' numpy>=2.1. This broke `pip install -r requirements.txt`
   outright and was failing CI at the install step. Fixed by moving `rtdl` to
   [`requirements-deep.txt`](requirements-deep.txt). Confirm CI is green before trusting the install.
2. **Pin exact versions.** The file is all floors (`>=`), so two clean clones months apart get
   different resolutions. `pip freeze` into a lock file once an environment is known good. Note
   `shap>=0.52` now needs Python 3.12+, so a 3.11 env silently resolves to an older shap.
3. **xgboost C-extension segfault (previous laptop, macOS).** Crashed the interpreter on import,
   blocking 5 test files; reproduced on xgboost 3.2.0 and 2.1.4 under both numpy 1.26.4 and 2.4.6.
   Environment-specific, not a code defect. Suspect the OpenMP runtime (`brew install libomp`).
   Check whether it reproduces on the new machine before spending time on it.
4. **Descoped multiclass plumbing.** Retaining `oegest_comb` in the modeling Parquet unblocks item 2
   above and costs one preprocessing change.

---

## How to talk about this in 20 seconds

If asked what is next, lead with the item that follows from the headline finding:

> "The ceiling result tells us what to do next: stop modeling this form and go get data it does not
> contain, like neighborhood and environment. The honest complication is that NCHS suppresses
> geography after 2005, so record-level linkage is not available to us. We would work at the
> aggregate level or apply for restricted-use files."

Second choice, if pushed:

> "Split preterm birth by severity. A birth at 27 weeks and one at 36 weeks are very different
> events, and the disparity may be worse at the dangerous end."

And if asked whether you would improve the model itself, the strong answer is no: more modeling on
these 103 features is the one thing our own results say will not help.
