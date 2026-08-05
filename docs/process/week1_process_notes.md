# Notes: Data Acquisition, Exploration, EDA, and Visualization

Author: Parshv

This is my working write-up for the first week of the EquiBirth project.

This week was about getting and understanding the data.

---

## 1. Problem Statement

The project is about preterm birth, which is birth before 37 weeks. It is the leading cause of newborn
death and it has one of the most stubborn racial gaps in U.S. public health. We are exploring 2
questions:

1. Can we predict preterm birth from things known before delivery?
2. How unequal is the outcome across race, and can the usual socioeconomic explanations account for it?

We cannot answer question 1 this week because that needs models, and modeling is later work. So the
real goal for the week was to get the data, make sure it is the right data, and understand it well
enough that everything we build later stands on solid ground.

That means acquisition, a provenance check, a data-quality audit, exploratory analysis, and clear
visualizations.

The three notebooks for the week are:

- `notebooks/week1_01_problem_and_acquisition.ipynb` - framing the problem and the honest acquisition story
- `notebooks/week1_02_exploration.ipynb` - provenance, shape, missingness, the field dictionary, known biases
- `notebooks/week1_03_eda_and_visualization.ipynb` - the actual exploratory analysis and the charts

And there is a standalone dashboard, `reports/week1_dashboard.html`, that pulls the highlights into one
scrollable page you can open in the browser.

---

## 2. Where the data comes from, and the year window

The data is the CDC/NCHS Natality public-use files, which I pulled from the NBER mirror. They are
basically a census: nearly every birth in the country generates a certificate, and the clinical and
demographic fields come off hospital records and a worksheet the mother fills in.

The window is 2018 to 2024 (7 years). Every state adopted the 2003 revision of the birth certificate by
January 1, 2016, so 2016, 2017, and 2018 are all on the exact same form as 2019 onward. That means the
same form does not justify starting in 2019; if anything it argues for going back further.

The only thing that would justify dropping 2018 is a genuine schema or coding break, and the one I cared
about most is the mother's race/ethnicity record, because a silent change there would quietly corrupt
the whole equity analysis. I downloaded the headers and a sample of rows from the 2018 and 2019 files
and diffed them directly: 238 columns in each, zero column differences, and the race recodes
(`mracehisp` 1-8, `mrace6` 1-6) use identical code sets. Every field I derive the target, the race
label, and the singleton filter from is the same across the two years. So 2018 goes in, and it earns its
place by giving me 2 clean pre-COVID baseline years (2018 and 2019) instead of one.

One framing thing: the "10.4% of U.S. births are preterm" number we see everywhere is for all births.
This project is singletons only, because multiples (twins, triplets) are preterm about 60% of the time
and would completely dominate the signal. So I expected our singleton rate to come out somewhere below
10.4%, and it did (about 8.6% across roughly 24.9 million singleton births). That match was my first sign
that the data was correct.

---

## 3. The acquisition headache, and how I actually solved it

Each year's full CSV from NBER is around 1.7 to 2.3 GB (13.6 GB across 7 years). There is no good
scientific reason to subsample here: the files are free and, once converted to a columnar format, they
fit on a laptop. Sampling is for when data will not fit or costs money, and neither is true. So I
corrected it and pulled the complete files.

I used a resumable download. `scripts/week1_download_full.sh` pulls each year, which continues from the
last byte already on disk after any drop, retries on transient errors, and aborts a stalled socket so
the retry can resume. It runs one connection per year in parallel, because the throttle is
per-connection. After each file lands I check its byte size against the size the server reports, so a
truncated file is caught instead of silently analyzed.

- It is the complete population, not a sample. Coverage lands right around 100% of published singleton
  births every year (the audit table reads 100 to 101 percent, slightly over because the 96.8% singleton
  share is a round figure and 2024 births are still provisional, so the denominator is approximate). We
  hold essentially every singleton record; the only ones set aside are births with a missing obstetric
  gestational estimate (null target) and the handful removed by the singleton filter. There is no
  sampling step left to bias anything.
- Counts are comparable across years now, because every year is the full population. I still default to
  rates for cross-year comparisons, because rates are the right unit for a public-health question and are
  robust to the tiny coverage differences from missing-target records. Good habit to keep even when
  counts would also be fair.
- I still validate against published numbers. The per-race preterm rates I compute later reproduce the
  published NVSS ordering and magnitudes, which is the independent check that the ingest pipeline
  (de-sentinelling, target, race mapping, singleton filter) did the right thing.

---

## 4. What I checked before trusting anything (exploration and quality)

- Shape and types. 17 fields, all numeric except the race label, and the whole thing fits in memory
  comfortably. Good, I can explore interactively without reaching for Spark or chunking.
- Missingness. I mapped where the holes are, because in administrative data missingness is never random.
  The big one is marital status (`married`), blank for about 12% of rows. Some reporting areas
  historically did not collect it the same way, so it is missing not at random and correlated with place.
  Everything else is under 2%.
- A field dictionary. I wrote down what every field means, where it sits on the certificate, and crucially
  whether it is known before delivery. That last column is going to matter enormously later (see the
  leakage note in section 8). I deliberately only gathered pre-delivery fields plus the target. The
  tempting at-delivery fields like birthweight and Apgar score are not in this table on purpose.
- Known biases: Birth certificates are a gold standard but they are not flawless. I named three I am
  carrying forward: race misclassification is worse for the smaller groups (AIAN, NHOPI); BMI is
  self-reported and is the least-audited field; and the risk-factor checkboxes are known to be
  undercounted, which means any weak association we find from them is really a floor.
- Group sizes: I listed sample sizes per race group, smallest first, so we know exactly where to be
  careful. Even in the full data NHOPI and AIAN are the smallest groups (roughly 67k and 180k rows,
  against millions for the large groups), so I trust the White, Black, Hispanic, and Asian estimates most.

---

## 5. The EDA, and the order I actually explored in

This is notebook 03. I tried to do the basic exploration but more can be done here.

About 8.6% of singleton births are preterm, stable year to year. That imbalance already tells us that
accuracy will be a garbage metric when we model (a model that always says "not preterm" scores 91%), so
we will need to lean on rate-based and calibration metrics instead.

I ran `describe()` on the numeric fields and immediately two max values jumped out: BMI maxes at 99.9 and
cigarettes-per-day at 98. Those are not real measurements. They are NCHS "not stated" sentinel codes
pretending to be numbers.

Instead of eyeballing, I counted them. BMI >= 70 is about 2.2% of rows, the cigarette sentinels are tiny.
I then made a deliberate call for each weird value: the sentinels get excluded from any quantitative
summary (and flagged so the future feature pipeline nulls them per-field), but the genuinely extreme but
real values (a 13-year-old mother, a 49-year-old mother, a mother with 12 prior births) are kept, because
those people exist and they are clinically high-risk.

Then bivariate, one predictor at a time. I checked the things I expected to matter:

- Prior preterm birth more than triples the rate (about a 3.4x risk ratio). Good, that is the strongest
  known clinical risk factor, so the data passing this check.
- I tried an idea: sum the four pre-pregnancy risk flags into a 0 to 4 count and see if there is a
  dose-response. There is a clean one, climbing toward about 61% at four flags. That is worth engineering
  into a proper feature later.
- Smoking adds about 4 points to the rate.

The thing I keep noting is that every one of these is a group-level gradient, not a clean individual
separator. That tension comes back hard in the correlation section.

---

## 6. The finding I did not go looking for

I plotted the preterm rate by race and there is a roughly 4.7-point gap between NH Black mothers (about
12.3%) and NH White mothers (about 7.5%). I refused to just eyeball it, so I wrote out a two-proportion
z-test. The z came out around 280 (it is that large mostly because the full national data gives millions
of mothers per group, which shrinks the standard error to almost nothing, so the substantive number is
the 4.7-point gap, not the z itself). Either way, the p-value is very close to zero. That gap is not
noise or just by chance.

My honest first reaction was the same one most people have: "this is probably really a socioeconomic gap,
and it would close if I compared mothers with the same education." So I tried to falsify the racial-gap
story by stratifying on education and drawing a line per race. If it was right, the Black and White lines
should converge as education goes up.

They do not converge. Every Black line sits above every White line at every education level. And the
single starkest comparison: a college-educated Black mother has a higher preterm rate (about 10.2%) than
a White mother who never finished high school (about 9.1%). Education, the supposed great equalizer, does
not equalize this.

That falsification is the finding. It says the gap is not simply socioeconomic, and it points toward
something structural that a birth certificate does not record. Being careful with language here:

Race in this project is a marker of exposure to racism and structural barriers, not a biological cause.

It is a strong, real pattern, and it is the thread we most want to pull on in the coming weeks.

I also explored a COVID question because the window covers 2018 to 2024. The annual rate barely moves,
which at first looks like "no COVID effect." But the published effect was a sharp dip over just a few
months in spring 2020, and averaging that into a yearly national number dilutes it into nothing. So the
real lesson there is about temporal granularity: a real effect at one time scale can disappear at another.
I noted it as a caution rather than a finding.

---

## 7. The statistical habit this week hammered home

The last EDA piece is correlations. I correlated every numeric field with the target, and the strongest
one (prior preterm birth) reaches only about +0.13. Everything else is weaker. There is no dominant
signal hiding in these fields.

To make the significance vs importance point concrete, I ran a t-test on BMI between preterm and term
births. Across tens of millions of rows it returns a p-value near 0, which looks impressive until we see
the actual difference is about 1 BMI unit, which no clinician would ever act on. At this sample size,
p-values detect trivia. Significance is cheap; effect size is what matters, and the effect sizes here are
all small.

It strongly suggests that when we get to modeling, the honest performance ceiling will be modest, because
the real causes of preterm birth mostly live outside what a birth certificate records. Knowing that now
will keep us from chasing a high accuracy number that would only be achievable by cheating (see leakage
below). I would much rather know the ceiling is low and report it honestly than be surprised later.

---

## 8. What comes next

This week is acquisition, exploration, quality auditing, EDA, and visualization. That is it. No models,
no fairness metrics with confidence intervals, no explainability. Those are planned, and the EDA is
exactly what tells us how to do them well. Here is the plan, in the order we would tackle it:

Feature engineering with a strict, tested leakage gate. Birth certificates record a lot of fields at
delivery (birthweight, Apgar, NICU admission, labor interventions) that correlate almost perfectly with
prematurity but are completely useless as predictors, because they do not exist at the moment a
prediction would actually be made. Given how weak the legitimate pre-delivery signals turned out to be,
the temptation to let a leaky field sneak in will be strong. I think this will be the single most
important engineering decision in the whole project.

Calibrated baseline and tree models: Logistic regression and tree-based models, judged on rate-aware
metrics and calibration rather than accuracy, because of the class imbalance. We should expect a modest
ceiling and we want to report it honestly.

A formal fairness audit: Turn this week's disparity finding into rigorous metrics: demographic parity,
equal opportunity, calibration within each group, each with a confidence interval and a proper
significance test.

Explainability and a decomposition of the gap: Verify what any model actually leans on, and then try to
answer the real question: how much of the racial gap can measured factors account for, and how much is
left over? Given that education did not close the gap this week, we suspect the answer is "measured
factors explain only a part of it," but we want to quantify that carefully.

Stretch goals if time allow: State-level context would be great, but the public-use files carry no
geography, so that would need separate CDC WONDER aggregates. And external sanity checks against a second
data source like NHANES would strengthen the representativeness argument.

---

## 9. Limitations so far

- It is the complete public-use files, validated against published national rates. The remaining caveat
  is not sample size but that public-use natality data carries no geography (no state or county), so any
  state-level analysis later will need separate CDC WONDER aggregates.
- The smallest race groups (AIAN, NHOPI) have real misclassification and, even in the full data, the
  smallest counts, so their numbers are read with caution.
- BMI is self-reported and the least-audited field, with sentinel codes which we had to filter.
- Everything this week is descriptive and associational. No causal claim is made, and nothing here is a
  clinical tool. It is an educational research project.

---

## 10. How to reproduce this week's work

```bash
# 1. download the COMPLETE per-year files from NBER (resumable; safe to re-run to resume)
bash scripts/week1_download_full.sh

# 2. build the pooled full dataset + the tidy 17-field week-1 exploration table (singletons, 2018-2024)
PYTHONPATH=src ./.venv/bin/python scripts/week1_build_full_dataset.py

# 3. precompute the dashboard data bundle (small aggregates only)
PYTHONPATH=src ./.venv/bin/python scripts/week1_build_data.py

# 4. (re)generate the three week-1 notebooks
PYTHONPATH=src ./.venv/bin/python scripts/week1_build_notebooks.py

# 5. execute them top to bottom
./.venv/bin/jupyter nbconvert --execute --to notebook --inplace notebooks/week1_*.ipynb

# 6. (re)build and open the standalone dashboard in any browser
./.venv/bin/python scripts/week1_build_dashboard.py && open reports/week1_dashboard.html
```

Every number in the notebooks and the dashboard is computed live from the complete data, so nothing is
taken on faith. That was the whole point of the week.
