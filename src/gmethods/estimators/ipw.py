"""Inverse probability weighting for interventions on a time-varying treatment.

Each unit-turn gets the weight ``p*(a) / p(a)``: the probability of the
observed treatment under the intervention over its probability under the
natural course. ``p`` comes from a fitted treatment model and ``p*`` from
:meth:`Intervention.transform_probs`. Weights multiply over turns. For
"never k" a turn with treatment k gets weight 0, which is the
clone-censor-weight idea: units stop contributing once they deviate.

Hazards of the outcome and the competing event are then weighted averages
at each turn, and the risk is the cumulative incidence built from them.

Only the treatment model is needed, so this check relies on different
modelling assumptions from the g-formula. Agreement between the two is
evidence the result is not driven by misspecification of either.

The regime here is the same as everywhere in this package: a natural draw
of a removed level becomes the reference level. The textbook
clone-censor-weight estimator with weight ``1 / P(not k)`` targets a
different regime, in which the freed probability is spread across all other
levels in proportion.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..interventions import Intervention
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
)


class IPW:
    """Inverse probability weighted risk under each intervention.

    Parameters
    ----------
    spec:
        Column roles in the panel. Defaults match the chat simulator.
    learner:
        ``"hgb"`` or ``"linear"`` for the treatment model.
    truncate:
        Cap cumulative weights at this quantile of their positive values
        (for example 0.99). ``None`` leaves them as they are.
    prob_floor:
        Floor on predicted treatment probabilities before renormalising.
        Boosted trees can push probabilities to exactly 0 for near-certain
        events (a shift ending), which would make weights infinite.
    """

    def __init__(
        self,
        spec: PanelSpec | None = None,
        learner: Learner = "hgb",
        truncate: float | None = None,
        random_state: int = 0,
        learner_params: dict | None = None,
        prob_floor: float = 1e-4,
    ):
        self.prob_floor = prob_floor
        self.spec = spec or PanelSpec()
        self.learner = learner
        self.truncate = truncate
        self.random_state = random_state
        self.learner_params = learner_params

    def fit(self, panel: pd.DataFrame) -> IPW:
        """Fit the treatment model and keep the panel for weighting."""
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
        names = list(spec.baseline) + spec.history_names() + spec.covariate_names
        mask = [enc.is_categorical(n) for n in names]
        model = fit_learner(
            make_learner(
                self.learner, "classify", mask, self.random_state, self.learner_params
            ),
            stack(feats, names),
            a,
        )
        self.model_ = model
        p = predict_proba_full(model, stack(feats, names), len(spec.levels))
        p = np.maximum(p, self.prob_floor)
        self.probs_ = p / p.sum(axis=1, keepdims=True)
        self.a_ = a
        self.work_ = work
        return self

    def weights(self, intervention: Intervention) -> np.ndarray:
        """Cumulative weight of each panel row, through its own treatment."""
        spec = self.spec
        p = self.probs_
        p_star = intervention.transform_probs(p, spec.levels)
        rows = np.arange(len(p))
        num, den = p_star[rows, self.a_], p[rows, self.a_]
        # Rows the intervention leaves alone get weight 1 even when the
        # model gives the observed treatment a near-zero probability.
        ratio = np.where((num == den) & (num > 0), 1.0, num / np.maximum(den, 1e-12))
        ratio = np.where(num == 0, 0.0, ratio)
        ids = self.work_[spec.id_col].to_numpy()
        logw = pd.Series(np.log(np.maximum(ratio, 1e-300))).groupby(ids).cumsum()
        w = np.where(ratio > 0, np.exp(logw.to_numpy()), 0.0)
        # A zero ratio censors the unit for the rest of follow-up.
        dead = pd.Series(ratio == 0).groupby(ids).cummax().to_numpy()
        w[dead] = 0.0
        if self.truncate is not None and (w > 0).any():
            cap = np.quantile(w[w > 0], self.truncate)
            w = np.minimum(w, cap)
        return w

    def estimate(self, interventions: Sequence[Intervention]) -> RiskTable:
        """Weighted cumulative incidence under each intervention."""
        spec = self.spec
        work = self.work_
        t = work[spec.time_col].to_numpy()
        T = int(t.max())
        y = work[spec.outcome].to_numpy(dtype=float)
        d = (
            work[spec.competing].to_numpy(dtype=float)
            if spec.competing
            else np.zeros(len(work))
        )
        curves = {}
        self.weight_summary_ = {}
        for iv in interventions:
            w = self.weights(iv)
            at_risk = np.bincount(t - 1, weights=w, minlength=T)
            ev = np.bincount(t - 1, weights=w * y, minlength=T)
            no_y = w * (1 - y)
            at_risk_d = np.bincount(t - 1, weights=no_y, minlength=T)
            ev_d = np.bincount(t - 1, weights=no_y * d, minlength=T)
            h_y = np.divide(ev, at_risk, out=np.zeros(T), where=at_risk > 0)
            h_d = np.divide(ev_d, at_risk_d, out=np.zeros(T), where=at_risk_d > 0)
            curves[iv.name] = cumulative_incidence(h_y[:, None], h_d[:, None])[:, 0]
            pos = w[w > 0]
            self.weight_summary_[iv.name] = {
                "mean": float(w.mean()),
                "max": float(w.max()),
                "share_censored": float((w == 0).mean()),
                "ess": float(pos.sum() ** 2 / (pos**2).sum()) if len(pos) else 0.0,
            }
        return RiskTable(
            risks={k: float(v[-1]) for k, v in curves.items()},
            curves=curves,
            label="ipw",
        )
