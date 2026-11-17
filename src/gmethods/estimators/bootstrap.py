"""Nonparametric bootstrap that resamples clusters (customers by default)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..results import REFERENCE, RiskTable


def resample_clusters(
    panel: pd.DataFrame,
    rng: np.random.Generator,
    cluster_col: str = "customer_id",
    id_col: str = "chat_id",
) -> pd.DataFrame:
    """Draw clusters with replacement and give each copy fresh unit ids."""
    clusters = panel[cluster_col].unique()
    draw = rng.choice(clusters, size=len(clusters), replace=True)
    groups = panel.groupby(cluster_col, sort=False).indices
    parts, new_ids = [], []
    offset = int(panel[id_col].max()) + 1
    for copy, c in enumerate(draw):
        idx = groups[c]
        parts.append(idx)
        new_ids.append(panel[id_col].to_numpy()[idx] + copy * offset)
    out = panel.iloc[np.concatenate(parts)].copy()
    out[id_col] = np.concatenate(new_ids)
    return out.reset_index(drop=True)


def bootstrap(
    fit_estimate: Callable[[pd.DataFrame], RiskTable],
    panel: pd.DataFrame,
    n_boot: int = 200,
    level: float = 0.95,
    cluster_col: str = "customer_id",
    id_col: str = "chat_id",
    seed: int = 0,
    reference: str = REFERENCE,
) -> tuple[RiskTable, pd.DataFrame]:
    """Percentile intervals for risk differences, refitting every model.

    Parameters
    ----------
    fit_estimate:
        Takes a panel and returns a :class:`RiskTable`. It must fit every
        model from scratch, so the interval reflects model-fitting noise.
    panel:
        The analysis panel. Clusters are resampled whole, because one
        customer can have several chats.

    Returns
    -------
    The point estimate on the full panel with ``ci`` filled in, and a frame
    of bootstrap risk differences (one row per replicate).
    """
    point = fit_estimate(panel)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        table = fit_estimate(resample_clusters(panel, rng, cluster_col, id_col))
        rows.append({r: table.rd(r, reference) for r in table.regimes})
    reps = pd.DataFrame(rows)
    alpha = (1 - level) / 2
    ci = {
        r: (float(reps[r].quantile(alpha)), float(reps[r].quantile(1 - alpha)))
        for r in reps.columns
    }
    point.ci = ci
    return point, reps
