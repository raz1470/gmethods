"""Parametric g-formula for a categorical time-varying treatment.

The g-formula fits a model for every time-varying variable in temporal
order, then simulates units forward under each intervention:

1. Covariates at turn t, each given baseline, history and the covariates
   before it in the same turn.
2. Treatment at turn t, given everything before it. Its natural draw is
   then changed by the intervention.
3. Outcome hazard at turn t, given everything above.
4. Competing-event hazard at turn t, among units without the outcome.

Risk under an intervention is the cumulative incidence accumulated from the
predicted hazards. That carries less Monte Carlo noise than simulating
events. Because later covariates are drawn from their models under the
intervened treatment history, effects that run through later confounders
(a handoff raising frustration, which drives a later escalation) are part
of the estimate. That is what conditioning on those confounders in a single
regression cannot do.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..interventions import NONE, Intervention, Natural
from ..results import RiskTable
from ..sim.simulator import cumulative_incidence
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


@dataclass
class _ResidualSampler:
    """Empirical residuals grouped by bins of the fitted value."""

    edges: np.ndarray
    resid: np.ndarray
    offsets: np.ndarray
    counts: np.ndarray
    lo: float
    hi: float

    @classmethod
    def fit(cls, fitted, resid, target, n_bins: int):
        qs = np.unique(np.quantile(fitted, np.linspace(0, 1, n_bins + 1)))
        edges = qs[1:-1]
        b = np.searchsorted(edges, fitted, side="right")
        order = np.argsort(b, kind="stable")
        counts = np.bincount(b, minlength=len(edges) + 1)
        offsets = np.r_[0, np.cumsum(counts)[:-1]]
        return cls(
            edges,
            resid[order],
            offsets,
            counts,
            float(np.min(target)),
            float(np.max(target)),
        )

    def draw(self, mean: np.ndarray, u: np.ndarray) -> np.ndarray:
        b = np.searchsorted(self.edges, mean, side="right")
        cnt = np.maximum(self.counts[b], 1)
        idx = self.offsets[b] + np.minimum((u * cnt).astype(int), cnt - 1)
        return np.clip(mean + self.resid[idx], self.lo, self.hi)


class GFormula:
    """Parametric g-formula with flexible learners.

    Parameters
    ----------
    spec:
        Column roles in the panel. Defaults match the chat simulator.
    learner:
        ``"hgb"`` (histogram gradient boosting) or ``"linear"`` (logistic and
        linear regression: the classic parametric g-formula). A dict sets
        the learner per model role: ``"covariates"``, ``"treatment"``,
        ``"outcome"`` and ``"competing"``. Missing roles use ``"hgb"``.
    n_residual_bins:
        Continuous covariates are drawn as fitted mean plus a residual
        resampled from training rows with a similar fitted mean.
    random_state:
        Seed for the learners.
    learner_params:
        Extra keyword arguments for every learner, such as
        ``{"max_iter": 50}`` for a quicker fit with ``"hgb"``.

    Examples
    --------
    >>> gf = GFormula().fit(data.panel)  # doctest: +SKIP
    >>> table = gf.estimate([iv.natural(), iv.never("escalation")])  # doctest: +SKIP
    """

    def __init__(
        self,
        spec: PanelSpec | None = None,
        learner: Learner | dict[str, Learner] = "hgb",
        n_residual_bins: int = 20,
        random_state: int = 0,
        learner_params: dict | None = None,
    ):
        self.spec = spec or PanelSpec()
        self.learner_params = learner_params
        self.learner = learner
        self.n_residual_bins = n_residual_bins
        self.random_state = random_state

    # ----------------------------------------------------------------- fit
    def _prepare(self, panel: pd.DataFrame) -> pd.DataFrame:
        spec = self.spec
        spec.check(panel)
        work = panel.sort_values([spec.id_col, spec.time_col], kind="stable")
        work = work.reset_index(drop=True).copy()
        for c in spec.covariates:
            v = work[c.name].to_numpy(dtype=float)
            work[c.name] = np.log1p(np.maximum(v, 0.0)) if c.log1p else v
        return work

    def fit(self, panel: pd.DataFrame) -> GFormula:
        """Fit covariate, treatment, outcome and competing-event models."""
        spec = self.spec
        work = self._prepare(panel)
        a = spec.treatment_codes(work)
        self.encoder_ = BaselineEncoder().fit(work, spec.baseline)
        feats = self.encoder_.transform(work, spec.baseline)
        feats.update(panel_history(work, spec, a))
        for c in spec.covariate_names:
            feats[c] = work[c].to_numpy(dtype=float)
        feats.update(treatment_onehot(a, spec.levels))

        base_hist = list(spec.baseline) + spec.history_names()
        cov_names = spec.covariate_names
        self.names_ = {}
        self.models_ = {}
        self.samplers_ = {}
        for j, c in enumerate(spec.covariates):
            names = base_hist + cov_names[:j]
            X = stack(feats, names)
            y = feats[c.name]
            if c.kind == "binary":
                m = fit_learner(self._learner("classify", names), X, y.astype(int))
            else:
                m = fit_learner(self._learner("regress", names), X, y)
                fitted = m.predict(X)
                self.samplers_[c.name] = _ResidualSampler.fit(
                    fitted, y - fitted, y, self.n_residual_bins
                )
            self.names_[c.name] = names
            self.models_[c.name] = m

        names = base_hist + cov_names
        self.names_["_treatment"] = names
        self.models_["_treatment"] = fit_learner(
            self._learner("classify", names, "treatment"), stack(feats, names), a
        )
        names = names + [f"a_{lv}" for lv in spec.levels[1:]]
        y = work[spec.outcome].to_numpy(dtype=int)
        self.names_["_outcome"] = names
        self.models_["_outcome"] = fit_learner(
            self._learner("classify", names, "outcome"), stack(feats, names), y
        )
        if spec.competing:
            keep = y == 0
            d = work[spec.competing].to_numpy(dtype=int)[keep]
            X = stack(feats, names)[keep]
            self.names_["_competing"] = names
            self.models_["_competing"] = fit_learner(
                self._learner("classify", names, "competing"), X, d
            )

        first = work.groupby(spec.id_col, sort=False).head(1)
        self.baseline_rows_ = first.reset_index(drop=True)
        self.t_max_ = int(work[spec.time_col].max())
        self.observed_curve_ = _observed_curve(work, spec, self.t_max_)
        self.observed_means_ = panel.groupby(spec.time_col)[spec.covariate_names].mean()
        return self

    def _learner(self, task: str, names: list[str], role: str = "covariates"):
        mask = [self.encoder_.is_categorical(n) for n in names]
        kind = (
            self.learner.get(role, "hgb")
            if isinstance(self.learner, dict)
            else self.learner
        )
        params = self.learner_params if kind == "hgb" else None
        return make_learner(kind, task, mask, self.random_state, params)

    # ------------------------------------------------------------ estimate
    def estimate(
        self,
        interventions: Sequence[Intervention],
        n_mc: int = 50_000,
        seed: int = 0,
        t_max: int | None = None,
        tol: float = 1e-4,
    ) -> RiskTable:
        """Risk of the outcome by the end of follow-up under each regime.

        Every regime is simulated on the same resampled baseline rows with
        the same random numbers, so differences between regimes carry little
        Monte Carlo noise. Simulation stops early once fewer than ``tol`` of
        the units are still at risk, since later turns cannot move the risk.
        """
        if not hasattr(self, "models_"):
            raise RuntimeError("call fit() first")
        names = [iv.name for iv in interventions]
        if len(set(names)) != len(names):
            raise ValueError("intervention names must be unique")
        spec = self.spec
        T = t_max or self.t_max_
        rng = np.random.default_rng([seed, 0])
        rows = self.baseline_rows_.iloc[
            rng.integers(0, len(self.baseline_rows_), n_mc)
        ].reset_index(drop=True)
        base = self.encoder_.transform(rows, spec.baseline)

        curves, means = {}, {}
        for iv in interventions:
            h_y, h_d, m = self._simulate(base, n_mc, iv, T, seed, tol)
            curves[iv.name] = cumulative_incidence(h_y, h_d).mean(axis=1)
            if isinstance(iv, Natural):
                means[iv.name] = m
        if means:
            key = next(iter(means))
            self.natural_course_means_ = means[key]
            self.natural_curve_ = curves[key]
        return RiskTable(
            risks={k: float(v[-1]) for k, v in curves.items()},
            curves=curves,
            label="gformula",
        )

    def _simulate(
        self, base: dict, n: int, iv: Intervention, T: int, seed: int, tol: float = 0.0
    ):
        spec = self.spec
        n_lv = len(spec.levels)
        rng = np.random.default_rng([seed, 1])
        prev = np.zeros(n, dtype=int)
        cum = np.zeros((n, n_lv))
        since = np.zeros(n)
        lags = {c: np.zeros(n) for c in spec.covariate_names}
        surv = np.ones(n)
        h_y = np.zeros((T, n))
        h_d = np.zeros((T, n))
        mean_rows = []
        for t in range(T):
            feats = dict(base)
            onehot_prev = np.eye(n_lv)[prev]
            for k, lv in enumerate(spec.levels[1:], start=1):
                feats[f"prev_{lv}"] = onehot_prev[:, k]
                feats[f"cum_{lv}"] = cum[:, k]
            feats["n_treated"] = cum[:, 1:].sum(axis=1)
            feats["since_treated"] = since
            feats["t"] = np.full(n, t + 1.0)
            for c in spec.covariate_names:
                feats[f"lag_{c}"] = lags[c]

            u_cov = rng.random((len(spec.covariates), n))
            u_a, u_iv = rng.random(n), rng.random(n)
            for j, c in enumerate(spec.covariates):
                X = stack(feats, self.names_[c.name])
                m = self.models_[c.name]
                if c.kind == "binary":
                    p = predict_proba_full(m, X, 2)[:, 1]
                    feats[c.name] = (u_cov[j] < p).astype(float)
                else:
                    feats[c.name] = self.samplers_[c.name].draw(m.predict(X), u_cov[j])

            X = stack(feats, self.names_["_treatment"])
            p = predict_proba_full(self.models_["_treatment"], X, n_lv)
            cdf = np.cumsum(p, axis=1)
            a_nat = np.minimum((u_a[:, None] > cdf).sum(axis=1), n_lv - 1)
            a = iv.apply(a_nat, u_iv, spec.levels, t=t + 1)
            feats.update(treatment_onehot(a, spec.levels))

            X = stack(feats, self.names_["_outcome"])
            h_y[t] = predict_proba_full(self.models_["_outcome"], X, 2)[:, 1]
            if spec.competing:
                h_d[t] = predict_proba_full(self.models_["_competing"], X, 2)[:, 1]

            w = surv / max(surv.sum(), 1e-12)
            row = {}
            for c in spec.covariates:
                v = feats[c.name]
                v = np.expm1(v) if c.log1p else v
                row[c.name] = float(np.sum(w * v))
            mean_rows.append(row)
            surv = surv * (1 - h_y[t]) * (1 - h_d[t])
            if surv.mean() < tol:
                break

            cum[np.arange(n), a] += 1
            since = np.where(a != NONE, 0.0, since + 1.0)
            prev = a
            for c in spec.covariate_names:
                lags[c] = feats[c]
        means = pd.DataFrame(
            mean_rows, index=pd.RangeIndex(1, len(mean_rows) + 1, name=spec.time_col)
        )
        return h_y, h_d, means


def _observed_curve(work: pd.DataFrame, spec: PanelSpec, T: int) -> np.ndarray:
    """Nonparametric cumulative incidence of the outcome by time step."""
    g = work.groupby(spec.time_col)
    at_risk = g.size().reindex(range(1, T + 1), fill_value=0).to_numpy(dtype=float)
    events = g[spec.outcome].sum().reindex(range(1, T + 1), fill_value=0).to_numpy()
    if spec.competing:
        comp = g[spec.competing].sum().reindex(range(1, T + 1), fill_value=0).to_numpy()
    else:
        comp = np.zeros(T)
    with np.errstate(invalid="ignore", divide="ignore"):
        h_y = np.where(at_risk > 0, events / at_risk, 0.0)
        h_d = np.where(at_risk > events, comp / np.maximum(at_risk - events, 1), 0.0)
    return cumulative_incidence(h_y[:, None], h_d[:, None])[:, 0]
