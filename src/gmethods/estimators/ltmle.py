"""Sequential regression (ICE) and targeted (LTMLE) estimators.

Both estimate the risk of the outcome by the end of follow-up under a
modified treatment policy: at each turn the natural treatment is drawn, then
the intervention maps it to the treatment given (Díaz, Williams, Hoffman and
Schenck, 2021). "Never k" and "scale k" in :mod:`gmethods.interventions`
are such policies.

Iterated conditional expectations (ICE) work backwards from the last turn.
At turn t, among units still at risk, the pseudo-outcome is

    Z_t = Y_t + (1 - Y_t)(1 - D_t) Q_{t+1}

where Y_t is the outcome, D_t the competing event and Q_{t+1} the value
already computed for the unit's next turn (0 after the last turn). A
regression of Z_t on the history and the treatment gives Qbar_t(h, a). The
value under the policy averages it over the policy's treatment given the
natural one: Q_t = sum_a' q(a' | A_t) Qbar_t(h, a'). The estimate is the
mean of Q_1. ICE needs no model for the time-varying covariates, which
removes most of the parametric g-formula's chances of misspecification.

LTMLE adds a targeting step after each regression (van der Laan and Gruber,
2012). Qbar_t is fluctuated on the logit scale by one parameter, fitted by a
logistic regression of Z_t with the cumulative density-ratio weights
W_t = prod_{s<=t} p*(A_s | H_s) / p(A_s | H_s). The result is doubly robust:
consistent if either the outcome regressions or the treatment model are
right. Its influence function gives standard errors without a bootstrap.

Nuisance models are cross-fitted by cluster (customers by default), so a
unit's own data never enters the predictions used for it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..interventions import Intervention
from ..results import REFERENCE, RiskTable
from ._features import (
    BaselineEncoder,
    Learner,
    PanelSpec,
    fit_learner,
    make_learner,
    panel_history,
    predict_proba_full,
    stack,
    treatment_onehot,
)


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _fluctuate(z, qbar, w, max_iter: int = 50, min_events: float = 5.0) -> float:
    """Weighted logistic fit of an intercept with offset logit(qbar).

    Returns the fluctuation parameter epsilon (Newton-Raphson on the
    quasi-binomial score, which allows fractional z). Turns with almost no
    outcome mass among the weighted rows are left untargeted: there the
    maximum-likelihood epsilon runs off to minus infinity and would set the
    risk of every unit at that turn to zero.
    """
    keep = w > 0
    if (
        not keep.any()
        or np.sum(z[keep]) < min_events * np.mean(z[keep] > 0)
        or (np.sum(z[keep] > 0) < min_events)
    ):
        return 0.0
    z, off, w = z[keep], _logit(qbar[keep]), w[keep]
    eps = 0.0
    for _ in range(max_iter):
        p = _expit(off + eps)
        grad = np.sum(w * (z - p))
        hess = np.sum(w * p * (1 - p))
        if hess <= 0:
            break
        step = grad / hess
        eps += step
        if abs(step) < 1e-10:
            break
    return float(eps)


class LTMLE:
    """Longitudinal targeted maximum likelihood (or ICE when ``target=False``).

    Parameters
    ----------
    spec:
        Column roles in the panel. Defaults match the chat simulator.
    learner:
        Learner for the outcome regressions: ``"linear"`` (logistic, the
        default), ``"stack"`` (spline + boosting super learner) or ``"hgb"``.
        The targeting step corrects a misspecified outcome model when the
        treatment model is good, so a stable, well-calibrated logistic model
        is the default. Boosting alone shrinks rare-outcome regressions,
        and the error compounds across turns.
    treatment_learner:
        Learner for the treatment model: ``"hgb"`` (default), ``"stack"``
        or ``"linear"``. Shift changes are near-deterministic in shift time
        left, a threshold a logistic model cannot represent.
    folds:
        Number of cross-fitting folds, split by cluster.
    target:
        Run the targeting step. ``False`` gives the ICE g-formula.
    truncate:
        Cap cumulative density-ratio weights at this quantile of their
        positive values. Only used when targeting.
    min_rows:
        Turns with fewer rows at risk than this pool their regression with
        the next earlier turn's rows (the turn index is a feature).
    """

    def __init__(
        self,
        spec: PanelSpec | None = None,
        learner: Learner = "linear",
        treatment_learner: Learner = "hgb",
        folds: int = 2,
        target: bool = True,
        truncate: float | None = 0.999,
        min_rows: int = 500,
        random_state: int = 0,
        learner_params: dict | None = None,
        prob_floor: float = 1e-4,
    ):
        self.spec = spec or PanelSpec()
        self.learner = learner
        self.treatment_learner = treatment_learner
        self.folds = folds
        self.target = target
        self.truncate = truncate
        self.min_rows = min_rows
        self.random_state = random_state
        self.learner_params = learner_params or {"max_iter": 100, "learning_rate": 0.1}
        self.prob_floor = prob_floor

    @property
    def label(self) -> str:
        return "ltmle" if self.target else "ice"

    # ------------------------------------------------------------------ fit
    def fit(self, panel: pd.DataFrame) -> LTMLE:
        """Build features, assign folds and cross-fit the treatment model."""
        spec = self.spec
        spec.check(panel)
        work = panel.sort_values([spec.id_col, spec.time_col], kind="stable")
        work = work.reset_index(drop=True).copy()
        for c in spec.covariates:
            if c.log1p:
                work[c.name] = np.log1p(np.maximum(work[c.name].to_numpy(float), 0.0))
        a = spec.treatment_codes(work)
        enc = BaselineEncoder().fit(work, spec.baseline)
        feats = enc.transform(work, spec.baseline)
        feats.update(panel_history(work, spec, a))
        for c in spec.covariate_names:
            feats[c] = work[c].to_numpy(dtype=float)
        self.h_names_ = (
            list(spec.baseline) + spec.history_names() + spec.covariate_names
        )
        self.a_names_ = [f"a_{lv}" for lv in spec.levels[1:]]
        self.cat_mask_ = [enc.is_categorical(n) for n in self.h_names_] + [False] * len(
            self.a_names_
        )
        self.H_ = stack(feats, self.h_names_)

        cluster = spec.cluster_col or spec.id_col
        clusters = work[cluster].unique()
        rng = np.random.default_rng(self.random_state)
        fold_of = pd.Series(rng.integers(0, self.folds, len(clusters)), index=clusters)
        self.fold_ = fold_of.loc[work[cluster]].to_numpy()
        self.cluster_ = work[cluster].to_numpy()

        n_lv = len(spec.levels)
        p = np.zeros((len(work), n_lv))
        mask = self.cat_mask_[: len(self.h_names_)]
        for k in range(self.folds):
            tr, te = self.fold_ != k, self.fold_ == k
            m = fit_learner(
                make_learner(
                    self.treatment_learner,
                    "classify",
                    mask,
                    self.random_state,
                    self.learner_params,
                ),
                self.H_[tr],
                a[tr],
            )
            p[te] = predict_proba_full(m, self.H_[te], n_lv)
        p = np.maximum(p, self.prob_floor)
        self.probs_ = p / p.sum(axis=1, keepdims=True)
        self.a_ = a
        self.work_ = work
        return self

    # ------------------------------------------------------------- estimate
    def _X(self, rows: np.ndarray, a: np.ndarray) -> np.ndarray:
        onehot = treatment_onehot(a, self.spec.levels)
        return np.column_stack([self.H_[rows], *[onehot[n] for n in self.a_names_]])

    def _weights(self, iv: Intervention) -> np.ndarray:
        spec = self.spec
        p = self.probs_
        p_star = iv.transform_probs(p, spec.levels)
        rows = np.arange(len(p))
        num, den = p_star[rows, self.a_], p[rows, self.a_]
        ratio = np.where((num == den) & (num > 0), 1.0, num / np.maximum(den, 1e-12))
        ids = self.work_[spec.id_col].to_numpy()
        logw = pd.Series(np.log(np.maximum(ratio, 1e-300))).groupby(ids).cumsum()
        w = np.where(ratio > 0, np.exp(logw.to_numpy()), 0.0)
        dead = pd.Series(ratio == 0).groupby(ids).cummax().to_numpy()
        w[dead] = 0.0
        if self.truncate is not None and (w > 0).any():
            w = np.minimum(w, np.quantile(w[w > 0], self.truncate))
        return w

    def _policy(self, iv: Intervention) -> np.ndarray:
        """Distribution of the treatment given under the policy, per row."""
        n_lv = len(self.spec.levels)
        return iv.transform_probs(np.eye(n_lv)[self.a_], self.spec.levels)

    def _one(self, iv: Intervention):
        spec = self.spec
        work = self.work_
        t_all = work[spec.time_col].to_numpy()
        ids = work[spec.id_col].to_numpy()
        y = work[spec.outcome].to_numpy(dtype=float)
        d = (
            work[spec.competing].to_numpy(dtype=float)
            if spec.competing
            else np.zeros(len(work))
        )
        n_lv = len(spec.levels)
        q_pol = self._policy(iv)
        w = self._weights(iv) if self.target else None

        # Index of each row's next row in the same unit, or -1.
        nxt = np.r_[np.arange(1, len(work)), -1]
        nxt[np.r_[ids[1:] != ids[:-1], True]] = -1

        q_val = np.zeros(len(work))  # Q_t under the policy, per row
        z_all = np.zeros(len(work))  # pseudo-outcome Z_t, per row
        eif = np.zeros(len(work))
        turns = np.sort(np.unique(t_all))[::-1]
        counts = pd.Series(t_all).value_counts()
        for t in turns:
            sub = np.nonzero(t_all == t)[0]
            nx = nxt[sub]
            q_next = np.where(nx >= 0, q_val[np.maximum(nx, 0)], 0.0)
            z_all[sub] = y[sub] + (1 - y[sub]) * (1 - d[sub]) * q_next
            # Sparse late turns borrow rows from the turns just after them,
            # whose pseudo-outcomes are already known. The turn index is a
            # feature, so the pooled regression can still differ by turn.
            upto, n_rows = t, counts[t]
            while n_rows < self.min_rows and upto < turns[0]:
                upto += 1
                n_rows += counts.get(upto, 0)
            train = np.nonzero((t_all >= t) & (t_all <= upto))[0]
            self._solve(sub, train, z_all, q_val, q_pol, w, eif, n_lv)

        first = np.r_[True, ids[1:] != ids[:-1]]
        q1 = q_val[first]
        psi = float(q1.mean())
        unit_eif = pd.Series(eif).groupby(ids).sum().to_numpy() + q1 - psi
        return psi, unit_eif, ids[first]

    def _solve(self, sub, train, z_all, q_val, q_pol, w, eif, n_lv):
        """Fit Qbar on ``train`` rows (cross-fitted), predict ``sub``, target."""
        qbar_all = np.zeros((len(sub), n_lv))
        a_sub = self.a_[sub]
        z = z_all[sub]
        for k in range(self.folds):
            te_mask = self.fold_[sub] == k
            if not te_mask.any():
                continue
            tr = train[self.fold_[train] != k]
            qbar_all[te_mask] = self._fit_predict(tr, z_all[tr], sub[te_mask])
        qbar_obs = qbar_all[np.arange(len(sub)), a_sub]
        if self.target:
            eps = _fluctuate(z, qbar_obs, w[sub])
            qbar_all = _expit(_logit(qbar_all) + eps)
            qbar_obs = qbar_all[np.arange(len(sub)), a_sub]
            eif[sub] = w[sub] * (z - qbar_obs)
        q_val[sub] = np.sum(q_pol[sub] * qbar_all, axis=1)

    def _fit_predict(self, tr, z_tr, te) -> np.ndarray:
        """Regress fractional Z on history and treatment; predict every level."""
        n_lv = len(self.spec.levels)
        if len(tr) < 20 or np.sum(z_tr > 0) < 10 or np.sum(z_tr < 1) < 10:
            return np.full((len(te), n_lv), float(np.mean(z_tr)) if len(tr) else 0.0)
        # A fractional outcome as a weighted binary one: each row appears as
        # a 1 with weight z and as a 0 with weight 1 - z (quasi-binomial).
        X = self._X(tr, self.a_[tr])
        x_dup = np.vstack([X, X])
        yd = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
        wd = np.r_[z_tr, 1 - z_tr]
        keep = wd > 0
        model = fit_learner(
            make_learner(
                self.learner,
                "classify",
                self.cat_mask_,
                self.random_state,
                self.learner_params,
            ),
            x_dup[keep],
            yd[keep].astype(int),
            sample_weight=wd[keep],
        )
        out = np.zeros((len(te), n_lv))
        for k in range(n_lv):
            out[:, k] = predict_proba_full(model, self._X(te, np.full(len(te), k)), 2)[
                :, 1
            ]
        return out

    def estimate(
        self,
        interventions: Sequence[Intervention],
        level: float = 0.95,
        reference: str = REFERENCE,
    ) -> RiskTable:
        """Risk under each regime, with influence-function intervals for the RD.

        Intervals are only reported when targeting; ICE has no valid
        influence-function variance.
        """
        names = [iv.name for iv in interventions]
        if len(set(names)) != len(names):
            raise ValueError("intervention names must be unique")
        risks, eifs = {}, {}
        for iv in interventions:
            psi, ic, unit_ids = self._one(iv)
            risks[iv.name] = psi
            eifs[iv.name] = ic
        ci, se = {}, {}
        if self.target and reference in eifs:
            cl = (
                pd.Series(self.cluster_, index=self.work_[self.spec.id_col])
                .groupby(level=0)
                .first()
                .loc[unit_ids]
                .to_numpy()
            )
            n = len(unit_ids)
            zq = norm.ppf(0.5 + level / 2)
            for name, ic in eifs.items():
                diff = ic - eifs[reference]
                s = (
                    np.sqrt(np.sum(pd.Series(diff).groupby(cl).sum().to_numpy() ** 2))
                    / n
                )
                rd = risks[name] - risks[reference]
                se[name] = float(s)
                ci[name] = (rd - zq * s, rd + zq * s)
        self.se_ = se
        return RiskTable(risks=risks, ci=ci, label=self.label)


class ICE(LTMLE):
    """Iterated conditional expectation g-formula: :class:`LTMLE` without targeting."""

    def __init__(
        self,
        spec: PanelSpec | None = None,
        learner: Learner = "stack",
        folds: int = 2,
        **kw,
    ):
        super().__init__(spec=spec, learner=learner, folds=folds, target=False, **kw)
