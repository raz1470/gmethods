"""Product friction and churn: a weekly simulator with known effects.

Each week a customer may hit one kind of friction: a failed payment, a
card decline, an app error or a re-verification check. Friction lowers
engagement the following week, and engagement drives both later friction
and churn. Exposure to friction grows with usage, and heavy users churn
less, so friction can look protective in a naive analysis.

Within each week, in order:

1. Covariates update: engagement (latent) and its observed signals (card
   transactions, app sessions), balance, support contacts after last week's
   friction and weeks until the next scheduled re-verification.
2. Friction: a re-verification fires when the schedule runs out unless it
   is skipped (``p_skip``); otherwise a softmax over none, payment failure,
   card decline and app error.
3. Churn.

"Never k" means the friction does not happen and the customer's action goes
through (the payment succeeds, the app works).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np

from ._base import WeeklySimulator

LEVELS = ("none", "payment_failure", "card_decline", "app_error", "reverification")
PF, CD, AE, RV = 1, 2, 3, 4


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _per_level(*values):
    return field(default_factory=lambda: dict(zip(LEVELS, values, strict=True)))


@dataclass(frozen=True)
class FrictionScenario:
    """Knobs of the friction-and-churn process. Defaults are the baseline."""

    n_customers: int = 50_000
    t_max: int = 26
    seed: int = 0

    # Engagement E_t (latent), AR(1)
    rho: float = 0.8
    e_salary: float = 0.5
    e_products: float = 0.15
    e_tenure: float = 0.1
    e_sd: float = 0.35
    gamma: dict = _per_level(0.0, 0.5, 0.3, 0.15, 0.5)
    """Drop in engagement the week after each friction type."""
    contact_drag: float = 0.1
    """Further drop in engagement per support contact."""
    lambda_e: float = 0.0
    """Unmeasured switching intent U on engagement (negative effect)."""
    lambda_y: float = 0.0
    """Unmeasured switching intent U on churn."""

    # Observed signals
    tx_base: float = 2.0
    tx_engagement: float = 0.5
    sessions_base: float = 1.0
    sessions_engagement: float = 0.6
    contact_rate: dict = _per_level(0.05, 0.6, 0.4, 0.3, 0.5)
    """Mean support contacts in the week after each friction type."""

    # Friction hazard (multinomial logit against none)
    c_pf: float = -6.2
    b_pf_tx: float = 0.45
    b_pf_balance: float = -0.6
    c_cd: float = -6.4
    b_cd_tx: float = 0.55
    c_ae: float = -4.6
    b_ae_sessions: float = 0.35
    b_ae_android: float = 0.7
    reverify_every: int = 52
    p_skip: float = 0.1
    """Probability a due re-verification is skipped this week."""
    c_rv_spike: float = -6.0
    """Unscheduled re-verification after a balance spike."""

    # Churn hazard
    alpha_0: float = -5.4
    alpha_engagement: float = 0.9
    alpha_salary: float = 0.8
    alpha_tenure: float = 0.15
    delta: dict = _per_level(0.0, 0.6, 0.4, 0.15, 0.6)
    """Direct effect of each friction type on the churn logit that week."""

    p_post: float = 0.4
    """Probability a churner's raw log shows a card decline after they left."""

    PRESETS: ClassVar[tuple[str, ...]] = ("baseline", "null", "unmeasured_intent")

    @classmethod
    def preset(cls, name: str, **overrides) -> FrictionScenario:
        base = cls()
        if name == "baseline":
            sc = base
        elif name == "null":
            zero = dict.fromkeys(LEVELS, 0.0)
            sc = replace(base, gamma=zero, delta=dict(zero), contact_drag=0.0)
        elif name == "unmeasured_intent":
            sc = replace(base, lambda_e=0.4, lambda_y=0.6)
        else:
            raise ValueError(f"unknown preset {name!r}")
        return replace(sc, **overrides)


class FrictionSimulator(WeeklySimulator):
    """Simulate weekly friction and churn, observed or under an intervention."""

    LEVELS = LEVELS
    BASELINE = (
        "tenure_years",
        "salary_paid_in",
        "n_products",
        "age_band",
        "income_band",
        "android",
    )
    TIME_VARYING = (
        "weeks_to_reverify",
        "support_contacts",
        "balance",
        "transactions",
        "app_sessions",
    )
    LOG1P = ("support_contacts", "transactions", "app_sessions")
    OUTCOME = "churned"
    COMPETING = None
    PROXY = "app_sessions"
    ARTEFACT_LEVEL = "card_decline"

    def __init__(self, scenario: FrictionScenario | None = None):
        super().__init__(scenario or FrictionScenario())
        sc = self.scenario
        self._gamma = np.array([sc.gamma[k] for k in LEVELS])
        self._delta = np.array([sc.delta[k] for k in LEVELS])
        self._contacts = np.array([sc.contact_rate[k] for k in LEVELS])

    def _baseline(self, rng, n, id_offset):
        income = rng.choice(4, size=n, p=[0.25, 0.35, 0.25, 0.15])
        salary = rng.random(n) < 0.25 + 0.1 * income
        return {
            "customer_id": np.arange(n) + id_offset,
            "tenure_years": rng.gamma(2.0, 1.2, n).round(1),
            "salary_paid_in": salary.astype(int),
            "n_products": 1 + rng.binomial(3, 0.25 + 0.05 * income),
            "age_band": rng.choice(5, size=n, p=[0.2, 0.3, 0.25, 0.15, 0.1]),
            "income_band": income,
            "android": (rng.random(n) < 0.45).astype(int),
            "U": rng.standard_normal(n),
            "reverify_offset": rng.integers(1, self.scenario.reverify_every + 1, n),
            "balance0": rng.normal(6.5 + 0.4 * income, 1.0),
        }

    def _draws(self, rng, n):
        T = self.scenario.t_max
        return {
            "z_e": rng.standard_normal((T, n)),
            "u_tx": rng.random((T, n)),
            "u_sess": rng.random((T, n)),
            "u_cont": rng.random((T, n)),
            "z_bal": rng.standard_normal((T, n)),
            "u_a": rng.random((T, n)),
            "u_iv": rng.random((T, n)),
        }

    @staticmethod
    def _poisson(mean, u):
        """Poisson draw by inverting the CDF with a fixed uniform."""
        from scipy.stats import poisson

        return poisson.ppf(u, np.maximum(mean, 1e-9)).astype(float)

    def _run(self, base, draws, iv, record):
        sc = self.scenario
        n = len(base["customer_id"])
        T = sc.t_max
        mu_e = (
            sc.e_salary * base["salary_paid_in"]
            + sc.e_products * (base["n_products"] - 1)
            + sc.e_tenure * np.minimum(base["tenure_years"], 5)
            - 0.5
            - sc.lambda_e * base["U"]
        )
        eng = mu_e.copy()
        balance = base["balance0"].copy()
        prev = np.zeros(n, dtype=int)
        cum = np.zeros((n, len(LEVELS)), dtype=int)
        to_reverify = base["reverify_offset"].astype(float)
        contacts = np.zeros(n)
        h_y = np.empty((T, n))
        h_d = np.zeros((T, n))
        recs = [] if record else None
        for t in range(T):
            if t > 0:
                contacts = self._poisson(self._contacts[prev], draws["u_cont"][t])
                eng = (
                    sc.rho * eng
                    + (1 - sc.rho) * mu_e
                    - self._gamma[prev]
                    - sc.contact_drag * contacts
                    + sc.e_sd * draws["z_e"][t]
                )
                to_reverify = np.where(prev == RV, sc.reverify_every, to_reverify - 1)
            tx = self._poisson(
                np.exp(sc.tx_base + sc.tx_engagement * eng), draws["u_tx"][t]
            )
            sessions = self._poisson(
                np.exp(sc.sessions_base + sc.sessions_engagement * eng),
                draws["u_sess"][t],
            )
            balance = (
                0.85 * balance
                + 0.15 * (base["balance0"] + 0.3 * eng)
                + 0.25 * draws["z_bal"][t]
            )
            spike = balance - base["balance0"]

            due = to_reverify <= 0
            eta = np.stack(
                [
                    np.zeros(n),
                    sc.c_pf
                    + sc.b_pf_tx * np.log1p(tx)
                    + sc.b_pf_balance * (balance - 6.5),
                    sc.c_cd + sc.b_cd_tx * np.log1p(tx),
                    sc.c_ae
                    + sc.b_ae_sessions * np.log1p(sessions)
                    + sc.b_ae_android * base["android"],
                    sc.c_rv_spike + 1.5 * np.maximum(spike, 0),
                ]
            )
            eta -= eta.max(axis=0)
            p = np.exp(eta)
            p /= p.sum(axis=0)
            p_rv_due = 1 - sc.p_skip
            p = np.where(due, (1 - p_rv_due) * p, p)
            p[RV] = np.where(due, p[RV] + p_rv_due, p[RV])
            cdf = np.cumsum(p, axis=0)
            a_nat = np.minimum(
                (draws["u_a"][t][None, :] > cdf).sum(axis=0), len(LEVELS) - 1
            )
            a = iv.apply(a_nat, draws["u_iv"][t], LEVELS, t=t + 1)

            eta_y = (
                sc.alpha_0
                - sc.alpha_engagement * eng
                - sc.alpha_salary * base["salary_paid_in"]
                - sc.alpha_tenure * np.minimum(base["tenure_years"], 5)
                + self._delta[a]
                + sc.lambda_y * base["U"]
            )
            h_y[t] = _sigmoid(eta_y)
            if record:
                recs.append(
                    {
                        "weeks_to_reverify": to_reverify.copy(),
                        "support_contacts": contacts.copy(),
                        "balance": balance.copy(),
                        "transactions": tx,
                        "app_sessions": sessions,
                        "treatment": a.copy(),
                        "prev_treatment": prev.copy(),
                        "cum": cum.copy(),
                        "engagement": eng.copy(),
                        "_latent": ["engagement"],
                    }
                )
            cum[np.arange(n), a] += 1
            prev = a
        return h_y, h_d, recs


__all__ = ["LEVELS", "FrictionScenario", "FrictionSimulator"]
