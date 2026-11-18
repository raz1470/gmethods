"""Shared engine for the weekly customer simulators.

A subclass supplies baseline covariates, the per-step random draws and one
pass through the structural equations. This base turns that into observed
data (panel, event log, customer summary) and into ground-truth risks under
any intervention, with common random numbers across regimes.

Within each week, in order: time-varying covariates update, the treatment
happens (at most one level per week), the outcome may happen, then the
competing event may happen. The panel stops at the first outcome or
competing event.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..interventions import NONE, Intervention, natural
from ..results import RiskTable
from .simulator import cumulative_incidence


@dataclass
class CustomerData:
    """Output of a weekly simulator's ``simulate``.

    Attributes
    ----------
    panel:
        One row per customer-week while the customer is at risk.
    event_log:
        Timestamped treatment and outcome events as a platform would log
        them, including treatment events logged after the outcome.
    customers:
        One row per customer with treatment counts. What a naive analysis uses.
    latent:
        Unobserved variables by customer-week, for teaching and checks.
    """

    panel: pd.DataFrame
    event_log: pd.DataFrame
    customers: pd.DataFrame
    latent: pd.DataFrame
    scenario: object


class WeeklySimulator:
    """Base class: subclasses define the structural equations.

    Subclasses set ``LEVELS``, ``BASELINE``, ``TIME_VARYING``, ``OUTCOME``
    (panel column name of the outcome), ``COMPETING`` (or ``None``),
    ``PROXY`` (time-varying column averaged in the customer summary) and
    ``EVENT_TIME`` (``"week"``), and implement ``_baseline``, ``_draws`` and
    ``_run``.
    """

    LEVELS: tuple[str, ...] = ()
    BASELINE: tuple[str, ...] = ()
    TIME_VARYING: tuple[str, ...] = ()
    LOG1P: tuple[str, ...] = ()
    BINARY: tuple[str, ...] = ()
    OUTCOME = "outcome"
    COMPETING: str | None = None
    PROXY = ""
    ARTEFACT_LEVEL = ""
    """Treatment level logged after the outcome by the raw-log artefact."""

    def __init__(self, scenario):
        self.scenario = scenario

    # ------------------------------------------------------------ subclass
    def _baseline(self, rng: np.random.Generator, n: int, id_offset: int) -> dict:
        raise NotImplementedError

    def _draws(self, rng: np.random.Generator, n: int) -> dict:
        raise NotImplementedError

    def _run(self, base: dict, draws: dict, iv: Intervention, record: bool):
        """Return hazards ``h_y``, ``h_d`` (shape weeks x units) and records."""
        raise NotImplementedError

    # --------------------------------------------------------------- shared
    def _batch(self, n: int, batch: int, seed: int):
        ss = np.random.SeedSequence([seed, batch])
        base_rng, draw_rng, event_rng = (np.random.default_rng(s) for s in ss.spawn(3))
        base = self._baseline(base_rng, n, 0)
        draws = self._draws(draw_rng, n)
        draws["u_y"] = draw_rng.random((self.scenario.t_max, n))
        draws["u_d"] = draw_rng.random((self.scenario.t_max, n))
        return base, draws, event_rng

    def panel_spec(self):
        """The :class:`~gmethods.estimators.PanelSpec` for this simulator's panel."""
        from ..estimators._features import CovariateSpec, PanelSpec

        covs = tuple(
            CovariateSpec(
                c,
                kind="binary" if c in self.BINARY else "continuous",
                log1p=c in self.LOG1P,
            )
            for c in self.TIME_VARYING
        )
        return PanelSpec(
            id_col="customer_id",
            time_col="week",
            treatment="treatment",
            levels=self.LEVELS,
            outcome=self.OUTCOME,
            competing=self.COMPETING,
            baseline=self.BASELINE,
            covariates=covs,
            cluster_col="customer_id",
        )

    def simulate(
        self,
        n: int | None = None,
        seed: int | None = None,
        intervention: Intervention | None = None,
    ) -> CustomerData:
        """Simulate observed data, or data under an intervention."""
        sc = self.scenario
        n = n or sc.n_customers
        seed = sc.seed if seed is None else seed
        base, draws, event_rng = self._batch(n, 0, seed)
        h_y, h_d, recs = self._run(base, draws, intervention or natural(), record=True)
        y = draws["u_y"] < h_y
        d = (~y) & (draws["u_d"] < h_d)
        return self._assemble(base, recs, y, d, event_rng)

    def true_risks(
        self,
        interventions: Sequence[Intervention],
        n_mc: int = 200_000,
        batch_size: int = 50_000,
        seed: int = 12345,
    ) -> RiskTable:
        """Ground-truth risk of the outcome by the horizon under each regime."""
        sc = self.scenario
        names = [iv.name for iv in interventions]
        if len(set(names)) != len(names):
            raise ValueError("intervention names must be unique")
        totals = {nm: np.zeros(sc.t_max) for nm in names}
        done, batch = 0, 0
        while done < n_mc:
            n = min(batch_size, n_mc - done)
            base, draws, _ = self._batch(n, batch, seed)
            for iv in interventions:
                h_y, h_d, _ = self._run(base, draws, iv, record=False)
                totals[iv.name] += cumulative_incidence(h_y, h_d).sum(axis=1)
            done += n
            batch += 1
        curves = {nm: tot / done for nm, tot in totals.items()}
        return RiskTable(
            risks={nm: float(c[-1]) for nm, c in curves.items()},
            curves=curves,
            label="truth",
        )

    # ------------------------------------------------------------ assembly
    def _assemble(self, base, recs, y, d, event_rng) -> CustomerData:
        T, n = y.shape
        stop = y | d
        any_stop = stop.any(axis=0)
        last = np.where(any_stop, stop.argmax(axis=0), T - 1)
        had_y = y[last, np.arange(n)] & any_stop
        had_d = d[last, np.arange(n)] & any_stop
        keep = np.arange(T)[:, None] <= last[None, :]
        tt, ii = np.nonzero(keep)

        def col(name):
            return np.stack([r[name] for r in recs])[tt, ii]

        bframe = pd.DataFrame({"customer_id": base["customer_id"]})
        for c in self.BASELINE:
            bframe[c] = base[c]
        panel = bframe.iloc[ii].reset_index(drop=True)
        panel.insert(1, "week", tt + 1)
        for c in self.TIME_VARYING:
            panel[c] = col(c)
        a = col("treatment")
        prev = col("prev_treatment")
        cum = np.stack([r["cum"] for r in recs])[tt, ii]
        panel["prev_treatment"] = pd.Categorical.from_codes(prev, self.LEVELS)
        for k, lv in enumerate(self.LEVELS[1:], start=1):
            panel[f"cum_{lv}"] = cum[:, k]
        panel["treatment"] = pd.Categorical.from_codes(a, self.LEVELS)
        panel[self.OUTCOME] = y[tt, ii].astype(int)
        if self.COMPETING:
            panel[self.COMPETING] = (d[tt, ii] & ~y[tt, ii]).astype(int)
        panel = panel.sort_values(["customer_id", "week"], kind="stable")
        panel = panel.reset_index(drop=True)

        latent = pd.DataFrame({"customer_id": base["customer_id"][ii], "week": tt + 1})
        for name in recs[0].get("_latent", []):
            latent[name] = col(name)
        latent = latent.sort_values(["customer_id", "week"], kind="stable")

        log = self._event_log(base, recs, last, had_y, had_d, event_rng)
        customers = self._summary(bframe, panel, log, last, had_y, had_d)
        return CustomerData(
            panel, log, customers, latent.reset_index(drop=True), self.scenario
        )

    def _event_log(self, base, recs, last, had_y, had_d, rng) -> pd.DataFrame:
        n = len(base["customer_id"])
        ids, events, kinds, weeks, days = [], [], [], [], []

        def add(mask, event, kind, week, day):
            idx = np.nonzero(mask)[0]
            ids.append(base["customer_id"][idx])
            events.append(np.full(len(idx), event, dtype=object))
            kinds.append(
                np.asarray(kind, dtype=object)[idx]
                if np.ndim(kind)
                else np.full(len(idx), kind, dtype=object)
            )
            weeks.append(np.broadcast_to(week, n)[idx])
            days.append(np.broadcast_to(day, n)[idx])

        levels = np.array(self.LEVELS, dtype=object)
        u_day = rng.random((len(recs), n))
        for t, r in enumerate(recs):
            alive = t <= last
            treated = alive & (r["treatment"] != NONE)
            # Treatment happens during the week, before any outcome that week.
            add(
                treated,
                "treatment",
                levels[r["treatment"]],
                t + 1,
                7 * t + 1 + 3 * u_day[t],
            )
        add(had_y, self.OUTCOME, None, last + 1, 7 * last + 5.0)
        if self.COMPETING:
            add(had_d, self.COMPETING, None, last + 1, 7 * last + 5.0)
        post = had_y & (rng.random(n) < self.scenario.p_post)
        add(post, "treatment", self.ARTEFACT_LEVEL, last + 1, 7 * last + 6.0)
        log = pd.DataFrame(
            {
                "customer_id": np.concatenate(ids),
                "event": np.concatenate(events),
                "treatment_type": np.concatenate(kinds),
                "week": np.concatenate(weeks).astype(int),
                "day": np.concatenate(days),
            }
        )
        log["timestamp"] = np.datetime64("2026-01-05") + pd.to_timedelta(
            log["day"], unit="D"
        )
        return (
            log.sort_values(["customer_id", "day"], kind="stable")
            .drop(columns="day")
            .reset_index(drop=True)
        )

    def _summary(self, bframe, panel, log, last, had_y, had_d) -> pd.DataFrame:
        out = bframe.copy()
        out["n_weeks"] = last + 1
        out[self.OUTCOME] = had_y.astype(int)
        if self.COMPETING:
            out[self.COMPETING] = had_d.astype(int)
        tr = log[log["event"] == "treatment"]
        event_time = log[log["event"] == self.OUTCOME].set_index("customer_id")[
            "timestamp"
        ]
        after = tr["timestamp"] > tr["customer_id"].map(event_time)
        ids = out["customer_id"]
        for lv in self.LEVELS[1:]:
            sub = tr[tr["treatment_type"] == lv]
            out[f"n_{lv}"] = (
                sub.groupby("customer_id").size().reindex(ids, fill_value=0).to_numpy()
            )
            out[f"n_{lv}_before_event"] = (
                sub[~after.loc[sub.index]]
                .groupby("customer_id")
                .size()
                .reindex(ids, fill_value=0)
                .to_numpy()
            )
        if self.PROXY:
            out[f"mean_{self.PROXY}"] = (
                panel.groupby("customer_id")[self.PROXY].mean().reindex(ids).to_numpy()
            )
        return out
