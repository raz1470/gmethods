"""Stacked learners: a smooth spline model blended with gradient boosting.

The blend is chosen on out-of-fold predictions, which is the super learner
idea (van der Laan, Polley and Hubbard, 2007) with two members. Splines fit
smooth additive structure, such as a hazard that rises with frustration or
with the turn index. Boosting picks up interactions and thresholds. On a
rare outcome boosting alone is pulled toward the pooled mean; the spline
member is not, and the blend weights decide how much of each to use.

Both stackers accept ``sample_weight``, which the sequential regressions in
:mod:`gmethods.estimators.ltmle` need for fractional targets.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


class SplineFeatures(TransformerMixin, BaseEstimator):
    """B-splines on continuous columns, one-hot on categorical, scaled otherwise."""

    def __init__(self, cat_mask=None, n_knots: int = 5, min_unique: int = 12):
        self.cat_mask = cat_mask
        self.n_knots = n_knots
        self.min_unique = min_unique

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        mask = (
            list(self.cat_mask) if self.cat_mask is not None else [False] * X.shape[1]
        )
        self.cat_idx_ = [j for j, m in enumerate(mask) if m]
        num = [j for j, m in enumerate(mask) if not m]
        self.spline_idx_ = [
            j for j in num if len(np.unique(X[:, j])) >= self.min_unique
        ]
        self.plain_idx_ = [j for j in num if j not in self.spline_idx_]
        if self.cat_idx_:
            self.onehot_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.onehot_.fit(X[:, self.cat_idx_])
        if self.spline_idx_:
            self.spline_ = SplineTransformer(
                n_knots=self.n_knots, degree=3, extrapolation="linear"
            )
            self.spline_.fit(X[:, self.spline_idx_])
        self.scale_ = StandardScaler().fit(self._raw(X))
        return self

    def _raw(self, X):
        parts = []
        if self.cat_idx_:
            parts.append(self.onehot_.transform(X[:, self.cat_idx_]))
        if self.spline_idx_:
            parts.append(self.spline_.transform(X[:, self.spline_idx_]))
        if self.plain_idx_:
            parts.append(X[:, self.plain_idx_])
        return np.hstack(parts)

    def transform(self, X):
        return self.scale_.transform(self._raw(np.asarray(X, dtype=float)))


def _fit(est, X, y, w):
    from ._features import fit_learner

    return fit_learner(est, X, y, sample_weight=w)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class StackedClassifier(ClassifierMixin, BaseEstimator):
    """Blend member class probabilities with a multinomial logistic regression.

    The blender sees each member's out-of-fold log-probabilities, so a member
    that overfits gets little weight.
    """

    def __init__(self, members, cv: int = 2, random_state: int = 0):
        self.members = members
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None, groups=None):
        """Fit members and the blender.

        ``groups`` keeps rows of one group in the same fold. Pass it when a
        row is duplicated (a fractional outcome coded as a weighted 1 and a
        weighted 0): otherwise a copy in the training fold leaks its twin's
        label into the out-of-fold predictions and the blend is biased.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        oof = np.zeros((len(y), len(self.members) * len(self.classes_)))
        for tr, te in self._splits(X, y, groups):
            for m, est in enumerate(self.members):
                fitted = _fit(est, X[tr], y[tr], None if w is None else w[tr])
                oof[te, self._cols(m)] = self._member_logp(fitted, X[te])
        self.blender_ = LogisticRegression(C=10.0, max_iter=3000)
        if len(self.classes_) > 1:
            self.blender_.fit(oof, y, sample_weight=w)
        self.members_ = [_fit(est, X, y, w) for est in self.members]
        return self

    def _splits(self, X, y, groups):
        if groups is None:
            folds = StratifiedKFold(
                self.cv, shuffle=True, random_state=self.random_state
            )
            return list(folds.split(X, y))
        uniq = np.unique(groups)
        rng = np.random.default_rng(self.random_state)
        fold_of = dict(zip(uniq, rng.integers(0, self.cv, len(uniq)), strict=True))
        f = np.array([fold_of[g] for g in groups])
        return [(np.nonzero(f != k)[0], np.nonzero(f == k)[0]) for k in range(self.cv)]

    def _cols(self, m):
        k = len(self.classes_)
        return slice(m * k, (m + 1) * k)

    def _member_logp(self, fitted, X):
        p = np.full((len(X), len(self.classes_)), 1e-6)
        seen = np.searchsorted(
            self.classes_,
            fitted.classes_ if hasattr(fitted, "classes_") else fitted[-1].classes_,
        )
        p[:, seen] = np.maximum(fitted.predict_proba(X), 1e-6)
        return np.log(p / p.sum(axis=1, keepdims=True))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if len(self.classes_) == 1:
            return np.ones((len(X), 1))
        feats = np.hstack([self._member_logp(f, X) for f in self.members_])
        return self.blender_.predict_proba(feats)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def weights_(self):
        """Blender coefficients, one block per member."""
        return self.blender_.coef_


class StackedRegressor(RegressorMixin, BaseEstimator):
    """Non-negative linear blend of member predictions, fitted out of fold."""

    def __init__(self, members, cv: int = 2, random_state: int = 0):
        self.members = members
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        oof = np.zeros((len(y), len(self.members)))
        folds = KFold(self.cv, shuffle=True, random_state=self.random_state)
        for tr, te in folds.split(X):
            for m, est in enumerate(self.members):
                fitted = _fit(est, X[tr], y[tr], None if w is None else w[tr])
                oof[te, m] = fitted.predict(X[te])
        self.blender_ = LinearRegression(positive=True).fit(oof, y, sample_weight=w)
        self.members_ = [_fit(est, X, y, w) for est in self.members]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        feats = np.column_stack([f.predict(X) for f in self.members_])
        return self.blender_.predict(feats)

    @property
    def weights_(self):
        return self.blender_.coef_
