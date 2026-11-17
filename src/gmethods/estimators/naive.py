"""Naive estimators that a first analysis of handoff data would try.

Each one returns a :class:`~gmethods.RiskTable` for the same regimes as the
g-formula, so its bias against the truth can be read off directly.

- :class:`CountRegression` regresses a chat's complaint on its handoff
  counts. With ``window="all"`` the counts include handoffs logged after the
  complaint (reverse causality). With ``window="before_complaint_or_all"``
  complainers count handoffs before the complaint and everyone else counts
  all of them, so the two groups have exposure windows of different length.
  ``adjust_sentiment=True`` adds the chat's mean sentiment, which is partly
  caused by the handoffs and so adjusts part of their effect away.
- :class:`PooledHazard` fits a turn-level complaint hazard on current
  covariates and handoffs, then switches handoffs off with every covariate
  held at its observed value. It recovers the direct effect at best: the
  paths through later frustration, waits and chat length are held fixed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..interventions import HANDOFF_TYPES, Intervention, Natural, Never, Scale
from ..results import RiskTable
from ..sim.simulator import PANEL_BASELINE, PANEL_TIME_VARYING

LEVELS = HANDOFF_TYPES[1:]


def _design(
    frame: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> pd.DataFrame:
    parts = [frame[numeric].astype(float)]
    for c in categorical:
        parts.append(
            pd.get_dummies(frame[c].astype(str), prefix=c, drop_first=True, dtype=float)
        )
    return pd.concat(parts, axis=1)


def _counts_under(counts: pd.DataFrame, iv: Intervention) -> pd.DataFrame:
    out = counts.copy()
    if isinstance(iv, Natural):
        return out
    if isinstance(iv, Never):
        for lv in iv.levels:
            out[lv] = 0.0
        return out
    if isinstance(iv, Scale):
        for lv in iv.levels:
            out[lv] = out[lv] * iv.factor
        return out
    raise TypeError(f"naive estimators do not support {type(iv).__name__}")


class CountRegression:
    """Logistic regression of a chat's complaint on its handoff counts.

    Parameters
    ----------
    window:
        ``"all"`` counts every logged handoff. ``"before_complaint_or_all"``
        counts handoffs before the complaint for complainers and all
        handoffs otherwise.
    adjust_sentiment:
        Add the chat's mean sentiment as a covariate.
    """

    CATEGORICAL = ("task_type", "first_agent_team")
    NUMERIC = (
        "friction_history",
        "start_hour",
        "n_products",
        "age_band",
        "income_band",
    )

    def __init__(self, window: str = "all", adjust_sentiment: bool = False):
        if window not in ("all", "before_complaint_or_all"):
            raise ValueError("window must be 'all' or 'before_complaint_or_all'")
        self.window = window
        self.adjust_sentiment = adjust_sentiment

    @property
    def label(self) -> str:
        lab = f"count_{self.window}"
        return lab + "_sentiment" if self.adjust_sentiment else lab

    def _counts(self, chats: pd.DataFrame) -> pd.DataFrame:
        suffix = "" if self.window == "all" else "_before_complaint"
        return pd.DataFrame(
            {lv: chats[f"n_{lv}{suffix}"].astype(float) for lv in LEVELS}
        )

    def fit(self, chats: pd.DataFrame) -> CountRegression:
        """Fit on the one-row-per-chat summary."""
        numeric = list(self.NUMERIC) + (
            ["mean_sentiment"] if self.adjust_sentiment else []
        )
        self.numeric_ = numeric
        self.chats_ = chats.reset_index(drop=True)
        X = self._X(self.chats_, self._counts(self.chats_))
        self.columns_ = X.columns
        self.model_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        self.model_.fit(X.to_numpy(), self.chats_["complained"].to_numpy())
        return self

    def _X(self, chats, counts):
        X = _design(chats, self.numeric_, list(self.CATEGORICAL))
        return pd.concat([counts.add_prefix("n_"), X], axis=1)

    def odds_ratios(self) -> pd.Series:
        """Odds ratio per extra handoff of each type."""
        coef = self.model_[-1].coef_[0] / self.model_[0].scale_
        s = pd.Series(np.exp(coef), index=self.columns_)
        return s[[f"n_{lv}" for lv in LEVELS]]

    def estimate(self, interventions: Sequence[Intervention]) -> RiskTable:
        """Mean predicted complaint risk with counts set by each regime."""
        counts = self._counts(self.chats_)
        risks = {}
        for iv in interventions:
            X = self._X(self.chats_, _counts_under(counts, iv)).reindex(
                columns=self.columns_, fill_value=0.0
            )
            risks[iv.name] = float(self.model_.predict_proba(X.to_numpy())[:, 1].mean())
        return RiskTable(risks=risks, label=self.label)


class PooledHazard:
    """Turn-level complaint hazard conditional on current covariates.

    The model uses baseline covariates, the current time-varying covariates
    (including sentiment), cumulative handoffs before the turn and the
    handoff at the turn. Under a regime the handoffs and cumulative counts
    are changed and every covariate stays as observed. Each chat keeps its
    observed number of turns.
    """

    label = "pooled_hazard"

    def fit(self, panel: pd.DataFrame) -> PooledHazard:
        self.panel_ = panel.sort_values(["chat_id", "turn"]).reset_index(drop=True)
        a = self.panel_["handoff"].astype(str)
        self.a_ = a
        X = self._X(self.panel_, a)
        self.columns_ = X.columns
        self.model_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        self.model_.fit(X.to_numpy(), self.panel_["complaint"].to_numpy())
        return self

    def _X(self, panel, a: pd.Series) -> pd.DataFrame:
        cur = pd.DataFrame({f"a_{lv}": (a == lv).astype(float) for lv in LEVELS})
        prev = cur.groupby(panel["chat_id"].to_numpy()).shift(1, fill_value=0.0)
        cum = prev.groupby(panel["chat_id"].to_numpy()).cumsum()
        cum.columns = [c.replace("a_", "cum_") for c in cum.columns]
        numeric = [c for c in PANEL_BASELINE if c not in CountRegression.CATEGORICAL]
        base = _design(
            panel,
            numeric + PANEL_TIME_VARYING + ["turn"],
            list(CountRegression.CATEGORICAL),
        )
        return pd.concat([base, cum, cur], axis=1)

    def estimate(self, interventions: Sequence[Intervention]) -> RiskTable:
        risks = {}
        ids = self.panel_["chat_id"].to_numpy()
        rng = np.random.default_rng(0)
        u = rng.random(len(self.panel_))
        codes = self.a_.map({lv: i for i, lv in enumerate(HANDOFF_TYPES)}).to_numpy()
        for iv in interventions:
            new = iv.apply(codes, u, HANDOFF_TYPES)
            a = pd.Series(np.array(HANDOFF_TYPES)[new])
            X = self._X(self.panel_, a).reindex(columns=self.columns_, fill_value=0.0)
            h = self.model_.predict_proba(X.to_numpy())[:, 1]
            log_s = pd.Series(np.log1p(-h)).groupby(ids).sum()
            risks[iv.name] = float((1 - np.exp(log_s)).mean())
        return RiskTable(risks=risks, label=self.label)
