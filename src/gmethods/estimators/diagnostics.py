"""Diagnostics for g-methods: positivity, natural course and weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .gformula import GFormula
from .ipw import IPW


def positivity_table(
    ipw: IPW,
    by: str | list[str] = "task_type",
    threshold: float = 0.01,
) -> pd.DataFrame:
    """How often each treatment level is near-impossible or near-certain.

    Uses the fitted treatment model of an :class:`IPW` estimator. For each
    group and level: the mean predicted probability, the share of turns
    where it is below ``threshold``, and the share where the reference level
    ("none") is below ``threshold``. A removal regime needs "none" to be
    possible at every turn, so the last column is the one to watch.
    """
    spec = ipw.spec
    work = ipw.work_
    p = ipw.probs_
    keys = [by] if isinstance(by, str) else list(by)
    frame = work[keys].copy()
    for k, lv in enumerate(spec.levels):
        frame[f"p_{lv}"] = p[:, k]
    out = []
    for key, g in frame.groupby(keys, observed=True):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        row["turns"] = len(g)
        for lv in spec.levels[1:]:
            row[f"mean_p_{lv}"] = g[f"p_{lv}"].mean()
            row[f"share_p_{lv}_lt_{threshold:g}"] = (g[f"p_{lv}"] < threshold).mean()
        row[f"share_p_none_lt_{threshold:g}"] = (
            g[f"p_{spec.levels[0]}"] < threshold
        ).mean()
        out.append(row)
    return pd.DataFrame(out).set_index(keys)


def natural_course_check(gf: GFormula, n_turns: int = 10) -> pd.DataFrame:
    """Observed vs g-formula natural course, by turn.

    Needs :meth:`GFormula.estimate` to have been run with the natural course
    among the regimes. Returns the cumulative incidence of the outcome and
    the mean of each covariate among units still at risk, both observed and
    simulated. Large gaps point to a misspecified model.
    """
    if not hasattr(gf, "natural_course_means_"):
        raise RuntimeError("run estimate() with the natural course first")
    sim = gf.natural_course_means_.head(n_turns).add_suffix("_sim")
    obs = gf.observed_means_.head(n_turns).add_suffix("_obs")
    out = pd.concat([obs, sim], axis=1)
    cols = []
    for c in gf.spec.covariate_names:
        cols += [f"{c}_obs", f"{c}_sim"]
    out = out[cols]
    t = np.arange(1, n_turns + 1)
    out.insert(0, "risk_obs", gf.observed_curve_[t - 1])
    curve = getattr(gf, "natural_curve_", None)
    if curve is not None:
        out.insert(1, "risk_sim", curve[np.minimum(t, len(curve)) - 1])
    return out


def weight_summary(ipw: IPW) -> pd.DataFrame:
    """Mean, max, censored share and effective sample size of IPW weights."""
    if not hasattr(ipw, "weight_summary_"):
        raise RuntimeError("run estimate() first")
    return pd.DataFrame(ipw.weight_summary_).T
