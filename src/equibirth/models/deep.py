"""Deep tabular benchmarks: TabNet and FT-Transformer behind the boosting-style contract (D41).

These exist as HONEST BENCHMARKS expected to lose to the gradient-boosted trees on this
weak-signal administrative data (max |corr| with the target ~0.13, prevalence ~8.6%; Grinsztajn
et al., NeurIPS 2022). No custom architecture (DECISION_LOG D41): proven libraries, small honest
configs, deterministic seeds, CPU-safe. The interpretability side-product (TabNet feature masks)
is the real deliverable of the deep-learning week, not a leaderboard win.

Contract mirrors ``equibirth.models.boosting.fit_predict``:
``fit_predict(name, X, y, tr, val, ho, ...) -> held-out probabilities``. Predictions are
isotonic-recalibrated on ``val`` (same calibration-first discipline as every other model here).

NaN policy: unlike the boosters, neural nets cannot ingest NaN; features are median-imputed
(fit on train only) and standardized. That is disclosed wherever these models are reported.
"""

from __future__ import annotations

import os

import numpy as np

VALID = ("tabnet", "ft_transformer")


def _set_torch_determinism(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Torch defaults to a single intra-op thread for small models on some macOS builds; give it the
    # cores we allow the rest of the pipeline so the CPU benchmark completes in reasonable wall time.
    try:
        torch.set_num_threads(max(1, int(os.environ.get("EQUIBIRTH_TORCH_THREADS", "4"))))
    except Exception:  # pragma: no cover - torch thread config is best-effort
        pass


def _prep(X: np.ndarray, y: np.ndarray, tr: np.ndarray):
    """Median-impute (train-fit) + standardize (train-fit). Returns transformed X (float32)."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imp = SimpleImputer(strategy="median").fit(X[tr])
    sc = StandardScaler().fit(imp.transform(X[tr]))
    return sc.transform(imp.transform(X)).astype(np.float32)


def _fit_tabnet(Xp, y, tr, val, *, max_epochs: int, seed: int, verbose: int = 0):
    from pytorch_tabnet.tab_model import TabNetClassifier

    clf = TabNetClassifier(
        n_d=16, n_a=16, n_steps=3, gamma=1.5, seed=seed, verbose=verbose,
        optimizer_params={"lr": 2e-2}, device_name="cpu",
    )
    clf.fit(
        Xp[tr], y[tr], eval_set=[(Xp[val], y[val])], eval_metric=["auc"],
        max_epochs=max_epochs, patience=max(3, max_epochs // 4),
        batch_size=4096, virtual_batch_size=512, num_workers=0, drop_last=False,
        weights=1,  # inverse-frequency class weighting for the ~9% imbalance
    )
    return clf


class _FTWrapper:
    """Minimal rtdl FT-Transformer binary classifier with an sklearn-ish predict_proba."""

    def __init__(self, n_features: int, seed: int):
        try:
            import rtdl
        except ImportError as exc:  # rtdl pins torch<2/numpy<2, so it lives in its own env
            raise ImportError(
                "rtdl is not installed. It cannot coexist with the main requirements "
                "(it pins torch<2 and numpy<2), so the FT-Transformer runs in a separate "
                "environment: see requirements-deep.txt. The archived benchmark numbers are "
                "already in reports/week5_models.json, so you only need this to re-run it."
            ) from exc
        import torch

        _set_torch_determinism(seed)
        self.torch = torch
        self.seed = seed
        self.model = rtdl.FTTransformer.make_baseline(
            n_num_features=n_features, cat_cardinalities=None,
            d_token=32, n_blocks=2, attention_dropout=0.1,
            ffn_d_hidden=64, ffn_dropout=0.1, residual_dropout=0.0, d_out=1,
        )
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def fit(self, X, y, X_val, y_val, *, max_epochs: int, lr: float = 1e-3,
            batch_size: int = 1024, pos_weight: float = 1.0):
        torch = self.torch
        opt = self.model.make_default_optimizer()
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
        Xt = torch.tensor(X)
        yt = torch.tensor(y, dtype=torch.float32)
        n = len(Xt)
        # Explicit generator so the shuffle sequence is reproducible regardless of how much global
        # torch RNG was consumed during model construction.
        gen = torch.Generator().manual_seed(int(self.seed))
        best_val, best_state, patience_left = -np.inf, None, max(3, max_epochs // 4)
        for _ in range(max_epochs):
            self.model.train()
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                out = self.model(Xt[idx], None).squeeze(-1)
                loss_fn(out, yt[idx]).backward()
                opt.step()
            val_auc = self._val_auc(X_val, y_val)
            if val_auc > best_val:
                best_val, patience_left = val_auc, max(3, max_epochs // 4)
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _val_auc(self, X_val, y_val) -> float:
        from sklearn.metrics import roc_auc_score

        p = self.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, p)) if len(np.unique(y_val)) > 1 else 0.0

    def predict_proba(self, X) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        outs = []
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(X, dtype=np.float32))
            for i in range(0, len(Xt), 8192):
                logits = self.model(Xt[i:i + 8192], None).squeeze(-1)
                outs.append(torch.sigmoid(logits).numpy())
        p1 = np.concatenate(outs)
        return np.column_stack([1 - p1, p1])


def fit(name: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray, val: np.ndarray, *,
        max_epochs: int = 20, seed: int = 42):
    """Fit a deep tabular model on ``tr`` (early-stopping on ``val``).

    Returns ``(est, Xp)``: the fitted estimator plus the imputed+standardized full matrix
    (prep fit on train only), so callers can score any partition once via ``predict`` without
    refitting. ``fit_predict`` below composes this with isotonic calibration.
    """
    name = name.lower()
    if name not in VALID:
        raise ValueError(f"Unknown deep model {name!r}; expected one of {VALID}.")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)
    tr = np.asarray(tr)
    val = np.asarray(val)
    _set_torch_determinism(seed)
    np.random.seed(seed)

    Xp = _prep(X, y, tr)
    if name == "tabnet":
        est = _fit_tabnet(Xp, y, tr, val, max_epochs=max_epochs, seed=seed)
    else:
        pos = max(int(y[tr].sum()), 1)
        spw = float((len(y[tr]) - pos) / pos)
        est = _FTWrapper(Xp.shape[1], seed).fit(
            Xp[tr], y[tr], Xp[val], y[val], max_epochs=max_epochs, pos_weight=spw)
    return est, Xp


def predict(est, Xp: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Raw (uncalibrated) positive-class probabilities on ``Xp[mask]`` (all rows if None)."""
    Z = Xp if mask is None else Xp[np.asarray(mask)]
    return est.predict_proba(Z)[:, 1]


def fit_predict(name: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray, val: np.ndarray,
                ho: np.ndarray, *, max_epochs: int = 20, seed: int = 42,
                calibrate: bool = True) -> np.ndarray:
    """Fit a deep tabular model on ``tr`` (early-stop on ``val``), isotonic-recalibrate on
    ``val``, and return held-out probabilities on ``ho``."""
    y = np.asarray(y).astype(int)
    val = np.asarray(val)
    ho = np.asarray(ho)
    est, Xp = fit(name, X, y, tr, val, max_epochs=max_epochs, seed=seed)
    raw_val = predict(est, Xp, val)
    raw_ho = predict(est, Xp, ho)

    if not calibrate or len(np.unique(y[val])) < 2:
        return raw_ho
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_val, y[val])
    return np.asarray(iso.predict(raw_ho))


def tabnet_feature_masks(est, Xp: np.ndarray, feature_names: list) -> list:
    """Global TabNet importance from its built-in explain masks (associational, leakage-check
    before reporting). Returns [(feature, weight)] descending."""
    imp = np.asarray(est.feature_importances_, dtype=float)
    order = np.argsort(imp)[::-1]
    return [(feature_names[i], float(imp[i])) for i in order]
