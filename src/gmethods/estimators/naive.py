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


CATEGORICAL = ("task_type", "first_agent_team")
NUMERIC = (
    "friction_history",
    "start_hour",
    "n_products",
    "age_band",
    "income_band",
)


class CountRegression:
    """Logistic regression of a chat's complaint on its handoff counts.

    Parameters
    ----------
    window:
        ``"all"`` counts every logged handoff. ``"before_complaint_or_all"``
        counts handoffs before the complaint for complainers and all
        handoffs otherwise.
    adjust_sentiment:
        Add the chat's mean sentiment (or the summary's ``proxy`` column) as
        a covariate.
    levels, numeric, categorical, outcome, before_suffix, proxy:
        Column roles in the one-row-per-unit summary. Defaults match the
        chat simulator; :meth:`for_simulator` fills them for the weekly
        simulators.
    """

    def __init__(
        self,
        window: str = "all",
        adjust_sentiment: bool = False,
        levels: Sequence[str] = LEVELS,
        numeric: Sequence[str] = NUMERIC,
        categorical: Sequence[str] = CATEGORICAL,
        outcome: str = "complained",
        before_suffix: str = "_before_complaint",
        proxy: str = "mean_sentiment",
        all_levels: Sequence[str] = HANDOFF_TYPES,
    ):
        if window not in ("all", "before_complaint_or_all"):
            raise ValueError("window must be 'all' or 'before_complaint_or_all'")
        self.window = window
        self.adjust_sentiment = adjust_sentiment
        self.levels = tuple(levels)
        self.numeric = tuple(numeric)
        self.categorical = tuple(categorical)
        self.outcome = outcome
        self.before_suffix = before_suffix
        self.proxy = proxy
        self.all_levels = tuple(all_levels)

    @classmethod
    def for_simulator(cls, sim, window: str = "all", adjust_sentiment: bool = False):
        """Configured for a weekly simulator's customer summary."""
        return cls(
            window=window,
            adjust_sentiment=adjust_sentiment,
            levels=sim.LEVELS[1:],
            numeric=sim.BASELINE,
            categorical=(),
            outcome=sim.OUTCOME,
            before_suffix="_before_event",
            proxy=f"mean_{sim.PROXY}",
            all_levels=sim.LEVELS,
        )

    @property
    def label(self) -> str:
        lab = f"count_{self.window}"
        return lab + "_sentiment" if self.adjust_sentiment else lab

    def _counts(self, chats: pd.DataFrame) -> pd.DataFrame:
        suffix = "" if self.window == "all" else self.before_suffix
        return pd.DataFrame(
            {lv: chats[f"n_{lv}{suffix}"].astype(float) for lv in self.levels}
        )

    def fit(self, chats: pd.DataFrame) -> CountRegression:
        """Fit on the one-row-per-chat summary."""
        numeric = list(self.numeric) + ([self.proxy] if self.adjust_sentiment else [])
        self.numeric_ = numeric
        self.chats_ = chats.reset_index(drop=True)
        X = self._X(self.chats_, self._counts(self.chats_))
        self.columns_ = X.columns
        self.model_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        self.model_.fit(X.to_numpy(), self.chats_[self.outcome].to_numpy())
        return self

    def _X(self, chats, counts):
        X = _design(chats, self.numeric_, list(self.categorical))
        return pd.concat([counts.add_prefix("n_"), X], axis=1)

    def odds_ratios(self) -> pd.Series:
        """Odds ratio per extra handoff of each type."""
        coef = self.model_[-1].coef_[0] / self.model_[0].scale_
        s = pd.Series(np.exp(coef), index=self.columns_)
        return s[[f"n_{lv}" for lv in self.levels]]

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
    """Time-step outcome hazard conditional on current covariates.

    The model uses baseline covariates, the current time-varying covariates,
    cumulative treatments before the step and the treatment at the step.
    Under a regime the treatments and cumulative counts are changed and
    every covariate stays as observed. Each unit keeps its observed length.

    Parameters
    ----------
    spec:
        Column roles (a :class:`~gmethods.estimators.PanelSpec`). Defaults
        match the chat simulator.
    """

    label = "pooled_hazard"

    def __init__(self, spec=None):
        from ._features import PanelSpec

        self.spec = spec or PanelSpec()

    def fit(self, panel: pd.DataFrame) -> PooledHazard:
        sp = self.spec
        self.panel_ = panel.sort_values([sp.id_col, sp.time_col]).reset_index(drop=True)
        a = self.panel_[sp.treatment].astype(str)
        self.a_ = a
        self.categorical_ = [
            c
            for c in sp.baseline
            if isinstance(self.panel_[c].dtype, pd.CategoricalDtype)
            or self.panel_[c].dtype == object
        ]
        X = self._X(self.panel_, a)
        self.columns_ = X.columns
        self.model_ = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        self.model_.fit(X.to_numpy(), self.panel_[sp.outcome].to_numpy())
        return self

    def _X(self, panel, a: pd.Series) -> pd.DataFrame:
        sp = self.spec
        ids = panel[sp.id_col].to_numpy()
        cur = pd.DataFrame({f"a_{lv}": (a == lv).astype(float) for lv in sp.levels[1:]})
        prev = cur.groupby(ids).shift(1, fill_value=0.0)
        cum = prev.groupby(ids).cumsum()
        cum.columns = [c.replace("a_", "cum_") for c in cum.columns]
        numeric = [c for c in sp.baseline if c not in self.categorical_]
        base = _design(
            panel,
            numeric + sp.covariate_names + [sp.time_col],
            self.categorical_,
        )
        return pd.concat([base, cum, cur], axis=1)

    def estimate(self, interventions: Sequence[Intervention]) -> RiskTable:
        sp = self.spec
        risks = {}
        ids = self.panel_[sp.id_col].to_numpy()
        rng = np.random.default_rng(0)
        u = rng.random(len(self.panel_))
        levels = list(sp.levels)
        codes = self.a_.map({lv: i for i, lv in enumerate(levels)}).to_numpy()
        for iv in interventions:
            new = iv.apply(codes, u, levels, t=self.panel_[sp.time_col].to_numpy())
            a = pd.Series(np.array(levels)[new])
            X = self._X(self.panel_, a).reindex(columns=self.columns_, fill_value=0.0)
            h = self.model_.predict_proba(X.to_numpy())[:, 1]
            log_s = pd.Series(np.log1p(-h)).groupby(ids).sum()
            risks[iv.name] = float((1 - np.exp(log_s)).mean())
        return RiskTable(risks=risks, label=self.label)


class ShapDrivers:
    """A churn (or take-up) model explained with SHAP: the common driver analysis.

    Fits gradient boosting to the one-row-per-unit summary (treatment counts
    from the raw log, baseline covariates and the mean of the observed
    proxy) and reports SHAP values for each treatment count. This is what
    "build a model, then read the drivers off SHAP" produces. SHAP explains
    the model's predictions: it credits whatever predicts the outcome,
    including confounders' footprints and treatments logged after the event,
    so it is not a causal attribution.

    Needs the ``shap`` package.
    """

    label = "shap"

    def __init__(self, levels, features, outcome: str, random_state: int = 0):
        self.levels = tuple(levels)
        self.features = tuple(features)
        self.outcome = outcome
        self.random_state = random_state

    @classmethod
    def for_simulator(cls, sim) -> ShapDrivers:
        counts = [f"n_{lv}" for lv in sim.LEVELS[1:]]
        return cls(
            levels=sim.LEVELS[1:],
            features=(*counts, *sim.BASELINE, f"mean_{sim.PROXY}"),
            outcome=sim.OUTCOME,
        )

    def fit(self, units: pd.DataFrame) -> ShapDrivers:
        from sklearn.ensemble import HistGradientBoostingClassifier

        X = units[list(self.features)].astype(float)
        self.model_ = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, random_state=self.random_state
        ).fit(X, units[self.outcome])
        self.X_ = X
        return self

    def drivers(self, sample: int = 5_000) -> pd.DataFrame:
        """Per treatment type: mean |SHAP|, its share and the mean SHAP when present.

        ``share`` is the type's share of mean |SHAP| across the treatment
        count features, the number a driver analysis would report.
        ``mean_shap_when_present`` is the average log-odds contribution among
        units with at least one such event: its sign is the direction the
        model credits.
        """
        import shap

        X = self.X_.sample(min(sample, len(self.X_)), random_state=self.random_state)
        values = shap.TreeExplainer(self.model_).shap_values(X)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., 1]
        sv = pd.DataFrame(values, columns=X.columns, index=X.index)
        rows = {}
        for lv in self.levels:
            col = f"n_{lv}"
            present = X[col] > 0
            rows[lv] = {
                "mean_abs_shap": float(sv[col].abs().mean()),
                "mean_shap_when_present": float(sv.loc[present, col].mean())
                if present.any()
                else np.nan,
            }
        out = pd.DataFrame(rows).T
        out["share"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum()
        return out
