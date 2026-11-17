"""Panel specification, history features and learners shared by estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..interventions import HANDOFF_TYPES
from ..sim.simulator import PANEL_BASELINE

Learner = Literal["hgb", "linear", "stack"]


@dataclass(frozen=True)
class CovariateSpec:
    """How a time-varying covariate is modelled and simulated.

    Attributes
    ----------
    name:
        Column in the panel.
    kind:
        ``"continuous"`` (regression plus resampled residuals) or
        ``"binary"`` (classifier plus a Bernoulli draw).
    log1p:
        Model a non-negative continuous covariate on the ``log1p`` scale.
    """

    name: str
    kind: Literal["continuous", "binary"] = "continuous"
    log1p: bool = False


HANDOFF_COVARIATES: tuple[CovariateSpec, ...] = (
    CovariateSpec("queue_length"),
    CovariateSpec("wait", log1p=True),
    CovariateSpec("shift_time_left"),
    CovariateSpec("skill_mismatch", kind="binary"),
    CovariateSpec("with_specialist", kind="binary"),
    CovariateSpec("sentiment"),
    CovariateSpec("idle", log1p=True),
)
"""Default time-varying covariates for the chat panel, in within-turn order."""


@dataclass(frozen=True)
class PanelSpec:
    """Column roles in a long panel with one row per unit and time step.

    The defaults match the panel produced by :class:`gmethods.ChatSimulator`.
    """

    id_col: str = "chat_id"
    time_col: str = "turn"
    treatment: str = "handoff"
    levels: tuple[str, ...] = HANDOFF_TYPES
    outcome: str = "complaint"
    competing: str | None = "ended"
    baseline: tuple[str, ...] = tuple(PANEL_BASELINE)
    covariates: tuple[CovariateSpec, ...] = HANDOFF_COVARIATES
    cluster_col: str | None = "customer_id"

    @property
    def covariate_names(self) -> list[str]:
        return [c.name for c in self.covariates]

    def history_names(self) -> list[str]:
        lv = self.levels[1:]
        return (
            [f"prev_{x}" for x in lv]
            + [f"cum_{x}" for x in lv]
            + ["n_treated", "since_treated", "t"]
            + [f"lag_{c}" for c in self.covariate_names]
        )

    def treatment_codes(self, panel: pd.DataFrame) -> np.ndarray:
        col = panel[self.treatment]
        if isinstance(col.dtype, pd.CategoricalDtype):
            col = col.astype(str)
        mapping = {lv: i for i, lv in enumerate(self.levels)}
        codes = col.map(mapping)
        if codes.isna().any():
            bad = sorted(set(col[codes.isna()]))
            raise ValueError(f"unknown treatment values {bad}")
        return codes.to_numpy(dtype=int)

    def check(self, panel: pd.DataFrame) -> None:
        need = [
            self.id_col,
            self.time_col,
            self.treatment,
            self.outcome,
            *self.baseline,
            *self.covariate_names,
        ]
        if self.competing:
            need.append(self.competing)
        missing = [c for c in need if c not in panel.columns]
        if missing:
            raise KeyError(f"panel is missing columns {missing}")


@dataclass
class BaselineEncoder:
    """Integer codes for categorical baseline columns, fixed at fit time."""

    categories: dict[str, list] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, cols) -> BaselineEncoder:
        for c in cols:
            s = frame[c]
            if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
                self.categories[c] = sorted(map(str, s.astype(str).unique()))
        return self

    def transform(self, frame: pd.DataFrame, cols) -> dict[str, np.ndarray]:
        out = {}
        for c in cols:
            if c in self.categories:
                cats = self.categories[c]
                codes = pd.Categorical(frame[c].astype(str), categories=cats).codes
                out[c] = codes.astype(float)
            else:
                out[c] = frame[c].to_numpy(dtype=float)
        return out

    def is_categorical(self, name: str) -> bool:
        return name in self.categories


def panel_history(panel: pd.DataFrame, spec: PanelSpec, a: np.ndarray) -> dict:
    """History features for each panel row, from earlier rows of its unit."""
    ids = panel[spec.id_col].to_numpy()
    first = np.r_[True, ids[1:] != ids[:-1]]
    prev = np.r_[0, a[:-1]]
    prev[first] = 0
    n_lv = len(spec.levels)
    onehot_prev = np.eye(n_lv)[prev]
    grp = panel[spec.id_col]
    cum = pd.DataFrame(onehot_prev).groupby(grp.to_numpy()).cumsum().to_numpy()
    block = pd.Series(prev != 0).groupby(grp.to_numpy()).cumsum().to_numpy()
    since = (
        pd.Series(np.ones(len(a)))
        .groupby([grp.to_numpy(), block])
        .cumcount()
        .to_numpy()
    )
    feats = {}
    for k, lv in enumerate(spec.levels[1:], start=1):
        feats[f"prev_{lv}"] = onehot_prev[:, k]
        feats[f"cum_{lv}"] = cum[:, k]
    feats["n_treated"] = cum[:, 1:].sum(axis=1)
    feats["since_treated"] = since.astype(float)
    feats["t"] = panel[spec.time_col].to_numpy(dtype=float)
    for c in spec.covariate_names:
        v = panel[c].to_numpy(dtype=float)
        lag = np.r_[0.0, v[:-1]]
        lag[first] = 0.0
        feats[f"lag_{c}"] = lag
    return feats


def treatment_onehot(a: np.ndarray, levels) -> dict:
    return {f"a_{lv}": (a == k).astype(float) for k, lv in enumerate(levels) if k > 0}


def make_learner(
    learner: Learner,
    task: str,
    cat_mask: list[bool],
    seed: int = 0,
    params: dict | None = None,
):
    """A fresh sklearn estimator for a classification or regression task."""
    if learner == "hgb":
        # No early stopping: on rare outcomes it stops after a few rounds and
        # leaves hazards near the pooled mean, which biases the risk upward.
        kw = dict(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            categorical_features=np.array(cat_mask) if any(cat_mask) else None,
            early_stopping=False,
            random_state=seed,
        )
        kw.update(params or {})
        if task == "classify":
            return HistGradientBoostingClassifier(**kw)
        return HistGradientBoostingRegressor(**kw)
    if learner == "linear":
        cat_idx = [i for i, m in enumerate(cat_mask) if m]
        num_idx = [i for i, m in enumerate(cat_mask) if not m]
        prep = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx),
                ("num", StandardScaler(), num_idx),
            ]
        )
        model = (
            LogisticRegression(max_iter=2000)
            if task == "classify"
            else LinearRegression()
        )
        return Pipeline([("prep", prep), ("model", model)])
    if learner == "stack":
        from ._stack import StackedClassifier, StackedRegressor

        members = [
            make_spline_learner(task, cat_mask),
            make_learner("hgb", task, cat_mask, seed, params),
        ]
        if task == "classify":
            return StackedClassifier(members, random_state=seed)
        return StackedRegressor(members, random_state=seed)
    raise ValueError(f"unknown learner {learner!r}")


def make_spline_learner(task: str, cat_mask: list[bool], n_knots: int = 5):
    """Additive spline model: B-splines on numeric columns, one-hot categories.

    Columns with few distinct values (indicators, small counts) pass through
    scaled, since a spline basis on them adds nothing. A ridge penalty keeps
    the basis stable. This is the smooth member of the default stack.
    """
    from ._stack import SplineFeatures

    model = (
        LogisticRegression(C=1.0, max_iter=3000)
        if task == "classify"
        else Ridge(alpha=1.0)
    )
    return Pipeline(
        [("prep", SplineFeatures(cat_mask=cat_mask, n_knots=n_knots)), ("model", model)]
    )


def fit_learner(est, X, y, sample_weight=None):
    est = clone(est)
    if sample_weight is None:
        est.fit(X, y)
    elif isinstance(est, Pipeline):
        est.fit(X, y, **{f"{est.steps[-1][0]}__sample_weight": sample_weight})
    else:
        est.fit(X, y, sample_weight=sample_weight)
    return est


def predict_proba_full(est, X, n_classes: int) -> np.ndarray:
    """Class probabilities with a column for every class code, seen or not."""
    p = est.predict_proba(X)
    classes = est.classes_ if hasattr(est, "classes_") else est[-1].classes_
    out = np.zeros((X.shape[0], n_classes))
    out[:, np.asarray(classes, dtype=int)] = p
    return out


def stack(feats: dict, names: list[str]) -> np.ndarray:
    return np.column_stack([np.asarray(feats[n], dtype=float) for n in names])
