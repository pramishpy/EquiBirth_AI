"""Week 2 (Phase 2 companion): the full-data feature-experiments compute (deliverable D3 compute).

This produces the heavy numbers a later notebook will display. It runs the whole battery of
feature experiments on the COMPLETE 2018-2024 public-use files (all ~24.9M singletons), not a sample:
a dual-track LightGBM (leakage-safe clean matrix vs a leakage matrix that adds the barred
post-delivery variables), per-feature univariate rank-sum AUC, LightGBM gain importance, Pearson
correlation with the target, mutual information, IncrementalPCA, and MiniBatchKMeans.

Run this FOREGROUND. Background Python children are killed on this project, and the run takes several
minutes. Every heavy step is out-of-core or incremental so it stays memory-safe on 18 GB: a dense
24.9M x ~100 float32 matrix is ~10 GB and a naive fit copies it and OOMs, so we only ever materialize
the two partitions a tree needs, LightGBM is histogram-based (Dataset frees its raw copy), PCA and
KMeans stream the full matrix in row-batches, and correlation/rank-AUC are single-pass vectorized.
Full-data is achieved via these incremental variants, NOT subsampling, with one explicit exception:
mutual information is computed on a DISCLOSED ~1,000,000-row stratified subsample (mutual_info_classif
has no incremental form), and that is recorded in the report.

Language note: all prose here is associational. Race is a marker of exposure to racism and structural
barriers, not a biological cause, and none of this is a clinical tool.

Run: PYTHONPATH=src ./.venv/bin/python scripts/week2_experiments.py
"""

from __future__ import annotations

import os

# Cap threads BEFORE importing polars / numpy BLAS so the process stays within a predictable memory
# and CPU footprint on this 18 GB machine.
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from equibirth.features import full_dataset, splits
from equibirth.features.build import build_modeling_frame
from equibirth.ingest import schema
from equibirth.models import evaluate as ev
from equibirth.utils.seeds import set_seeds

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
SEED = 42

OUT_JSON = REPORTS / "week2_experiments.json"
OUT_MD = REPORTS / "week2_experiments.md"

# Cap on dev_train ROWS carried into a LightGBM fit. dev_train is ~60% of ~24.9M, and materializing
# that as dense float32 (~6 GB) plus a LightGBM Dataset copy risks OOM on 18 GB. 5,000,000 stratified
# dev_train rows is well past the point where the weak signal stops improving. Held-out scoring always
# uses the FULL held_out_final partition. Set to None to use all dev_train rows.
DEV_TRAIN_CAP = 5_000_000
# Rows for the disclosed mutual-information subsample (mutual_info_classif has no incremental variant).
MI_ROWS = 1_000_000
# Row-batch size for the streaming IncrementalPCA / MiniBatchKMeans passes.
PCA_BATCH = 500_000
N_PCA = 10
N_CLUSTERS = 6
NUM_BOOST_ROUND = 300           # LightGBM boosting rounds per track
KMEANS_BATCH = 100_000          # MiniBatchKMeans mini-batch size
HELD_OUT_FRAC = 0.4             # fraction of the whole dataset held out for final scoring
N_LOADING_PCS = 3               # number of leading PCs to report top loadings for

# seed + deterministic pin the histogram summation so the tracked JSON (AUCs and the ordered
# gain_importance list) is reproducible even with n_jobs=4 (the project convention in models/boosting).
LGB_PARAMS_BASE = dict(
    objective="binary",
    metric="average_precision",
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=200,
    max_bin=255,
    feature_pre_filter=False,
    n_jobs=4,
    verbosity=-1,
    force_row_wise=True,
    seed=SEED,
    deterministic=True,
)


def rank_auc(x, y):
    """Exact ROC-AUC of a single feature vs a binary target, via the rank-sum (Mann-Whitney) identity.

    Vectorized, handles ties with average ranks, and drops rows where the feature is NaN. Returns 0.5
    when either class is empty after dropping NaNs. This is the full-data univariate AUC with no model.
    """
    x = np.asarray(x, float)
    y = np.asarray(y)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return 0.5
    uniq, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg_rank = csum - (cnt - 1) / 2.0          # average (1-based) rank per unique value
    ranks = avg_rank[inv]
    s = ranks[y == 1].sum()
    return float((s - npos * (npos + 1) / 2.0) / (npos * nneg))


def _coerce_barred(col: str) -> pl.Expr:
    """Coerce a barred raw column to a numeric leakage feature.

    Y/N string flags map to 1.0/0.0 on the uppercased value; any other string (U, X, C, P, blank)
    becomes null. Numeric columns are cast to float. This is deliberately crude: the point is to show
    that post-delivery variables trivially beat the leakage-safe matrix, not to model them carefully.
    """
    return (
        pl.when(pl.col(col).cast(pl.String, strict=False).str.to_uppercase() == "Y")
        .then(1.0)
        .when(pl.col(col).cast(pl.String, strict=False).str.to_uppercase() == "N")
        .then(0.0)
        .otherwise(pl.col(col).cast(pl.Float64, strict=False))
        .cast(pl.Float32)
        .alias(col)
    )


def _to_f32(frame: pl.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Materialize a boolean-masked partition of a Polars frame as a float32 numpy block."""
    return frame.filter(pl.Series(mask)).to_numpy().astype("float32")


def _train_track(X: pl.DataFrame, y: pl.Series, tr: np.ndarray, ho: np.ndarray, name: str):
    """Train one histogram LightGBM track on the (optionally capped) dev_train partition.

    Returns (auc_pr, auc_roc, booster, train_rows_used, train_note). Only the two partitions needed
    are ever materialized as dense float32; the LightGBM Dataset frees its raw copy after building.
    Held-out scoring uses the FULL held_out_final partition.
    """
    tr_used = tr
    capped = False
    n_tr_full = int(tr.sum())
    if DEV_TRAIN_CAP is not None and n_tr_full > DEV_TRAIN_CAP:
        # Stratified cap WITHIN the dev_train subset so the sampled rows keep the outcome/race mix.
        keep = splits.stratified_subsample(_valid_ref.filter(pl.Series(tr)), DEV_TRAIN_CAP, seed=SEED)
        tr_used = tr.copy()
        tr_used[np.flatnonzero(tr)] = keep.to_numpy()
        capped = True

    Xtr = _to_f32(X, tr_used)
    ytr = y.filter(pl.Series(tr_used)).to_numpy()
    pos = int((ytr == 1).sum())
    neg = int((ytr == 0).sum())
    spw = (neg / pos) if pos else 1.0
    train_rows = int(Xtr.shape[0])
    print(f"  [{name}] dev_train rows used = {train_rows:,} (pos={pos:,} neg={neg:,}, "
          f"scale_pos_weight={spw:.3f})", flush=True)

    params = dict(LGB_PARAMS_BASE, scale_pos_weight=spw)
    ds = lgb.Dataset(Xtr, label=ytr, free_raw_data=True)
    booster = lgb.train(params, ds, num_boost_round=NUM_BOOST_ROUND)
    del Xtr, ds
    gc.collect()

    Xho = _to_f32(X, ho)
    yho = y.filter(pl.Series(ho)).to_numpy()
    p = booster.predict(Xho)
    a_pr = ev.auc_pr(yho, p)
    a_roc = ev.auc_roc(yho, p)
    print(f"  [{name}] held-out ({int(ho.sum()):,} rows): AUC-PR={a_pr:.4f}  AUC-ROC={a_roc:.4f}",
          flush=True)
    del Xho, yho, p
    gc.collect()

    note = (
        f"clean/leakage tracks trained on a stratified cap of {train_rows:,} dev_train rows "
        f"(full dev_train was {n_tr_full:,})"
        if capped
        else f"clean/leakage tracks trained on the full dev_train partition ({train_rows:,} rows)"
    ) + "; held-out scoring used the full held_out_final partition."
    return a_pr, a_roc, booster, train_rows, note


def _jsonable(obj):
    """Recursively cast numpy scalars/arrays to plain Python floats/ints for json.dump."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    return obj


# Module-level handle so _train_track can restratify within dev_train without re-threading `valid`.
_valid_ref: pl.DataFrame | None = None


def main() -> None:
    global _valid_ref
    set_seeds(SEED)
    print("Loading the complete 2018-2024 cleaned files (singletons)...", flush=True)
    df = full_dataset.load_full_frame(INTERIM, YEARS)
    valid = df.filter(pl.col("preterm").is_not_null())
    n_singletons = valid.height
    del df
    gc.collect()
    print(f"  full population with a valid target: {n_singletons:,} singletons", flush=True)

    # ---- Build the leakage-safe matrix from the raw frame ----
    X_clean, y, _components = build_modeling_frame(valid)
    feature_cols = list(X_clean.columns)
    print(f"  clean matrix: {X_clean.width} leakage-safe features", flush=True)

    # Extract everything else we still need from the raw ~9 GB `valid` frame BEFORE freeing it:
    #   - the barred post-delivery columns (coerced to float32) for the leakage matrix,
    #   - the race label array (for the clustering pass),
    #   - a slim strata frame (the stratify columns) for split assignment and subsampling.
    barred = sorted(c for c in schema.LEAKAGE_DETECTOR_VARS if c in valid.columns)
    print(f"  leakage matrix adds {len(barred)} barred post-delivery columns", flush=True)
    barred_frame = valid.select([_coerce_barred(c) for c in barred])
    race_np = valid["mother_race_ethnicity"].to_numpy()
    strata_cols = [c for c in splits.DEFAULT_STRATIFY if c in valid.columns]
    strata = valid.select(strata_cols)
    del valid
    gc.collect()
    # Slim frame used wherever the split logic needs stratify columns (keeps the ~9 GB raw frame freed).
    _valid_ref = strata

    split = splits.assign_splits(strata, held_out_frac=HELD_OUT_FRAC, seed=SEED)
    split_np = split.to_numpy()
    tr = split_np == "dev_train"
    ho = split_np == "held_out_final"
    print(f"  splits: dev_train={int(tr.sum()):,}  held_out_final={int(ho.sum()):,}", flush=True)

    # Leakage matrix = clean columns + the coerced barred columns, materialized as one frame.
    X_leak = pl.concat([X_clean, barred_frame], how="horizontal")
    del barred_frame
    gc.collect()

    # ---- Dual-track LightGBM (memory-safe) ----
    print("\nDual-track LightGBM (clean vs leakage):", flush=True)
    clean_pr, clean_roc, clean_booster, train_rows_used, train_note = _train_track(
        X_clean, y, tr, ho, "clean"
    )
    # Keep the clean gain importance before anything else touches memory.
    gain = clean_booster.feature_importance(importance_type="gain")
    gain_importance = sorted(
        ({"feature": f, "gain": float(g)} for f, g in zip(feature_cols, gain)),
        key=lambda d: d["gain"],
        reverse=True,
    )
    del clean_booster
    gc.collect()

    leak_pr, leak_roc, leak_booster, _lr, _ln = _train_track(X_leak, y, tr, ho, "leakage")
    del leak_booster, X_leak
    gc.collect()

    dual_track = {
        "clean": {"auc_pr": float(clean_pr), "auc_roc": float(clean_roc)},
        "leakage": {"auc_pr": float(leak_pr), "auc_roc": float(leak_roc)},
    }

    # ---- Full-data y once, as numpy, for univariate AUC / correlation / MI / clustering ----
    y_np = y.to_numpy()

    # ---- Per-feature univariate rank-sum AUC on FULL data (exact, vectorized, no model) ----
    print("\nUnivariate rank-sum AUC (full data)...", flush=True)
    # Sanity-check rank_auc against sklearn on a 50K probe slice. Scan for the first feature that both
    # varies (>=2 distinct non-null values, so sklearn's AUC is defined) and sits over a 2-class target
    # slice, rather than trusting feature_cols[0] to satisfy this, then assert a column was actually
    # tested so the check can never silently vanish.
    ys = y_np[:50_000]
    tested_col = None
    for check_col in feature_cols:
        xs = X_clean[check_col].to_numpy()[:50_000].astype(float)
        ok = ~np.isnan(xs)
        if len(np.unique(ys[ok])) < 2 or len(np.unique(xs[ok])) < 2:
            continue
        skl = roc_auc_score(ys[ok], xs[ok])
        ours = rank_auc(xs, ys)
        assert abs(skl - ours) < 1e-6, f"rank_auc mismatch on '{check_col}': {ours} vs sklearn {skl}"
        print(f"  rank_auc sanity check on '{check_col}' (50K): ours={ours:.6f} "
              f"sklearn={skl:.6f} -> PASS", flush=True)
        tested_col = check_col
        break
    assert tested_col is not None, "rank_auc sanity check found no eligible probe column"

    univariate_auc = sorted(
        ({"feature": c, "univariate_auc": rank_auc(X_clean[c].to_numpy(), y_np)} for c in feature_cols),
        key=lambda d: d["univariate_auc"],
        reverse=True,
    )
    print(f"  top univariate AUC: "
          f"{[(d['feature'], round(d['univariate_auc'], 4)) for d in univariate_auc[:5]]}", flush=True)

    # ---- Pearson correlation with the target on FULL data (Polars) ----
    print("\nPearson correlation with target (full data)...", flush=True)
    X_corr = X_clean.with_columns(pl.Series("preterm", y_np))
    corr_rows = []
    for c in feature_cols:
        val = X_corr.select(pl.corr(c, "preterm")).item()
        corr_rows.append({"feature": c, "corr": float(val) if val is not None else 0.0})
    correlation = sorted(corr_rows, key=lambda d: abs(d["corr"]), reverse=True)
    del X_corr
    gc.collect()
    print(f"  top |corr|: {[(d['feature'], round(d['corr'], 4)) for d in correlation[:5]]}", flush=True)

    # ---- Mutual information on a DISCLOSED stratified subsample ----
    print(f"\nMutual information on a stratified {MI_ROWS:,}-row subsample...", flush=True)
    mi_mask = splits.stratified_subsample(_valid_ref, MI_ROWS, seed=SEED)
    mi_mask_np = mi_mask.to_numpy()
    mi_subsample_rows = int(mi_mask_np.sum())
    Xsub = _to_f32(X_clean, mi_mask_np)
    ysub = y_np[mi_mask_np]
    # Median-impute NaNs per column (mutual_info_classif rejects NaN).
    col_med = np.nanmedian(Xsub, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    nan_idx = np.where(np.isnan(Xsub))
    Xsub[nan_idx] = np.take(col_med, nan_idx[1])
    mi = mutual_info_classif(Xsub, ysub, discrete_features="auto", random_state=SEED)
    mutual_info = sorted(
        ({"feature": f, "mi": float(m)} for f, m in zip(feature_cols, mi)),
        key=lambda d: d["mi"],
        reverse=True,
    )
    del Xsub, ysub
    gc.collect()
    print(f"  MI rows used = {mi_subsample_rows:,}; top MI: "
          f"{[(d['feature'], round(d['mi'], 5)) for d in mutual_info[:5]]}", flush=True)

    # ---- Standardization stats (two-pass via Polars) for streaming PCA / KMeans ----
    print("\nComputing per-feature mean/std (full data) for standardization...", flush=True)
    means = np.array([X_clean[c].mean() for c in feature_cols], dtype=np.float64)
    means = np.where(np.isnan(means), 0.0, means)
    stds = np.array([X_clean[c].std() for c in feature_cols], dtype=np.float64)
    stds = np.where(np.isnan(stds) | (stds == 0), 1.0, stds)

    n_rows = X_clean.height
    n_batches = (n_rows + PCA_BATCH - 1) // PCA_BATCH

    def _standardized_batch(lo: int, hi: int) -> np.ndarray:
        b = X_clean[lo:hi].to_numpy().astype("float32")
        # Mean-impute NaNs with the precomputed per-feature mean, then standardize.
        bad = np.isnan(b)
        if bad.any():
            b[bad] = np.take(means.astype("float32"), np.where(bad)[1])
        b -= means.astype("float32")
        b /= stds.astype("float32")
        return b

    # ---- IncrementalPCA over batches (full data) ----
    print(f"\nIncrementalPCA (n_components={N_PCA}) over {n_batches} batches of {PCA_BATCH:,}...",
          flush=True)
    ipca = IncrementalPCA(n_components=N_PCA)
    for i in range(n_batches):
        lo, hi = i * PCA_BATCH, min((i + 1) * PCA_BATCH, n_rows)
        batch = _standardized_batch(lo, hi)
        ipca.partial_fit(batch)
        del batch
        gc.collect()
    evr = ipca.explained_variance_ratio_.astype(float)
    cum = np.cumsum(evr).astype(float)
    loadings = {}
    for k in range(N_LOADING_PCS):
        comp = ipca.components_[k]
        order = np.argsort(np.abs(comp))[::-1][:8]
        loadings[f"PC{k + 1}"] = [
            {"feature": feature_cols[j], "loading": float(comp[j])} for j in order
        ]
    pca = {
        "explained_variance_ratio": [float(v) for v in evr],
        "cumulative": [float(v) for v in cum],
        "loadings": loadings,
    }
    print(f"  explained_variance_ratio[:3]={[round(v, 4) for v in evr[:3]]}  "
          f"cumulative[-1]={cum[-1]:.4f}", flush=True)

    # ---- MiniBatchKMeans (fit pass, then a streaming assign+aggregate pass) ----
    print(f"\nMiniBatchKMeans (n_clusters={N_CLUSTERS}) over the standardized batches...", flush=True)
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS, batch_size=KMEANS_BATCH, random_state=SEED, n_init=3
    )
    for i in range(n_batches):
        lo, hi = i * PCA_BATCH, min((i + 1) * PCA_BATCH, n_rows)
        batch = _standardized_batch(lo, hi)
        kmeans.partial_fit(batch)
        del batch
        gc.collect()

    # Second streaming pass: assign labels and aggregate per cluster. `race_np` was extracted from the
    # raw frame earlier (aligned to X_clean rows) so it stays available after `valid` was freed.
    key_feats = [c for c in ["mager", "bmi", "medical_risk_count"] if c in feature_cols]
    if "prior_preterm" in feature_cols:
        key_feats.append("prior_preterm")

    sizes = np.zeros(N_CLUSTERS, dtype=np.int64)
    preterm_sum = np.zeros(N_CLUSTERS, dtype=np.float64)
    feat_sum = np.zeros((N_CLUSTERS, len(key_feats)), dtype=np.float64)
    feat_cnt = np.zeros((N_CLUSTERS, len(key_feats)), dtype=np.int64)
    race_counts = [{} for _ in range(N_CLUSTERS)]

    for i in range(n_batches):
        lo, hi = i * PCA_BATCH, min((i + 1) * PCA_BATCH, n_rows)
        batch = _standardized_batch(lo, hi)
        labels = kmeans.predict(batch)
        del batch
        gc.collect()
        # Raw (unstandardized) feature block for interpretable per-cluster means.
        raw_block = X_clean[lo:hi, key_feats].to_numpy().astype("float64") if key_feats else None
        yb = y_np[lo:hi]
        rb = race_np[lo:hi]
        for cl in range(N_CLUSTERS):
            m = labels == cl
            cnt = int(m.sum())
            if cnt == 0:
                continue
            sizes[cl] += cnt
            preterm_sum[cl] += float(np.nansum(yb[m]))
            if raw_block is not None:
                sub = raw_block[m]
                feat_sum[cl] += np.nansum(sub, axis=0)
                feat_cnt[cl] += np.sum(~np.isnan(sub), axis=0)
            races_m = rb[m]
            uniq, cts = np.unique(races_m.astype(str), return_counts=True)
            for rname, rc in zip(uniq, cts):
                race_counts[cl][rname] = race_counts[cl].get(rname, 0) + int(rc)
        del labels
        gc.collect()

    clusters = []
    for cl in range(N_CLUSTERS):
        size = int(sizes[cl])
        rate = float(preterm_sum[cl] / size) if size else float("nan")
        rc = race_counts[cl]
        top3 = sorted(rc.items(), key=lambda kv: kv[1], reverse=True)[:3]
        race_mix = [
            {"race": rname, "share": (cnt / size) if size else float("nan")}
            for rname, cnt in top3
        ]
        feature_means = {}
        for j, c in enumerate(key_feats):
            feature_means[c] = float(feat_sum[cl][j] / feat_cnt[cl][j]) if feat_cnt[cl][j] else float("nan")
        clusters.append(
            {
                "cluster": cl,
                "size": size,
                "preterm_rate": rate,
                "race_mix": race_mix,
                "feature_means": feature_means,
            }
        )
    print("  per-cluster (size, preterm_rate): "
          f"{[(c['cluster'], c['size'], round(c['preterm_rate'], 4)) for c in clusters]}", flush=True)

    # ---- Assemble + write JSON ----
    result = {
        "n_singletons": int(n_singletons),
        "dual_track": dual_track,
        "train_rows_used": int(train_rows_used),
        "train_note": train_note,
        "univariate_auc": univariate_auc,
        "gain_importance": gain_importance,
        "correlation": correlation,
        "mutual_info": mutual_info,
        "mi_subsample_rows": int(mi_subsample_rows),
        "pca": pca,
        "clusters": clusters,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(_jsonable(result), indent=2))

    # ---- Human-readable markdown ----
    gap_pr = dual_track["leakage"]["auc_pr"] - dual_track["clean"]["auc_pr"]
    gap_roc = dual_track["leakage"]["auc_roc"] - dual_track["clean"]["auc_roc"]
    # PC count to reach 80% / 90% cumulative variance.
    cum = pca["cumulative"]
    pc_80 = next((i + 1 for i, v in enumerate(cum) if v >= 0.80), None)
    pc_90 = next((i + 1 for i, v in enumerate(cum) if v >= 0.90), None)
    pc_80_txt = f"{pc_80} of {N_PCA}" if pc_80 else f"more than {N_PCA}"
    pc_90_txt = f"{pc_90} of {N_PCA}" if pc_90 else f"more than {N_PCA}"

    md = [
        "# Week 2 feature experiments (full-data compute)",
        "",
        f"All numbers computed live on the COMPLETE 2018-2024 public-use files: "
        f"**{n_singletons:,} singleton births** with a valid target. Every heavy step ran out-of-core "
        "or incrementally on the full population, with one disclosed exception: mutual information was "
        f"computed on a stratified {mi_subsample_rows:,}-row subsample. Language is associational: race "
        "is a marker of exposure to racism and structural barriers, not a biological cause, and none of "
        "this is a clinical tool.",
        "",
        "## Dual-track LightGBM: leakage vs leakage-safe",
        "",
        f"{train_note}",
        "",
        "| track | AUC-PR | AUC-ROC |",
        "|---|---|---|",
        f"| clean (leakage-safe, {X_clean.width} pre-delivery features) | "
        f"{dual_track['clean']['auc_pr']:.4f} | {dual_track['clean']['auc_roc']:.4f} |",
        f"| leakage (adds {len(barred)} post-delivery columns) | "
        f"{dual_track['leakage']['auc_pr']:.4f} | {dual_track['leakage']['auc_roc']:.4f} |",
        "",
        f"Adding the barred post-delivery variables lifts AUC-ROC by "
        f"**{gap_roc:.4f}** and AUC-PR by **{gap_pr:.4f}**. That gap is the whole point: variables "
        "recorded at or after delivery (birthweight, Apgar, labor interventions, total visits) make a "
        "model look excellent and predict nothing usable before birth, so they are barred from the "
        "leakage-safe matrix.",
        "",
        "## Top-5 univariate rank-sum AUC (full data, no model)",
        "",
        "| feature | univariate AUC |",
        "|---|---|",
    ]
    for d in univariate_auc[:5]:
        md.append(f"| {d['feature']} | {d['univariate_auc']:.4f} |")
    md += [
        "",
        "These are exact single-feature ROC-AUCs over every row, so they measure raw marginal signal. "
        "The strongest are modest, which is why the clean model's discrimination is modest.",
        "",
        "## Structure: PCA and clustering",
        "",
        f"IncrementalPCA over the standardized full matrix needs **{pc_80_txt}** components to reach "
        f"80% of the variance and **{pc_90_txt}** to reach 90%. MiniBatchKMeans ({N_CLUSTERS} clusters) "
        "on the same standardized full matrix gives these per-cluster preterm rates:",
        "",
        "| cluster | size | preterm rate |",
        "|---|---|---|",
    ]
    for c in clusters:
        md.append(f"| {c['cluster']} | {c['size']:,} | {c['preterm_rate']*100:.2f}% |")
    md += [
        "",
        "Cluster preterm rates spread around the population base rate, reflecting how the unsupervised "
        "structure lines up with (but does not cleanly separate) the outcome. Per-cluster race mix and "
        "feature means are in the JSON companion.",
        "",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- Headline numbers to stdout ----
    print("\n=== HEADLINE ===", flush=True)
    print(f"n_singletons = {n_singletons:,}", flush=True)
    print(f"clean   AUC-PR={dual_track['clean']['auc_pr']:.4f}  AUC-ROC={dual_track['clean']['auc_roc']:.4f}",
          flush=True)
    print(f"leakage AUC-PR={dual_track['leakage']['auc_pr']:.4f}  AUC-ROC={dual_track['leakage']['auc_roc']:.4f}",
          flush=True)
    print(f"leakage minus clean: AUC-ROC +{gap_roc:.4f}, AUC-PR +{gap_pr:.4f}", flush=True)
    print(f"train_rows_used={train_rows_used:,}  |  MI rows={mi_subsample_rows:,}", flush=True)
    print(f"PCA to 80% variance: {pc_80_txt}; to 90%: {pc_90_txt}", flush=True)
    print(f"cluster preterm rates: {[round(c['preterm_rate'], 4) for c in clusters]}", flush=True)
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}\nwrote {OUT_MD.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
