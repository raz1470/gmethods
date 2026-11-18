"""Customer campaigns and product take-up: a weekly simulator with known effects.

Each week a customer may get one campaign: an information email, an offer
email or a push notification. A CRM rule picks who gets what from an
engagement score, so the campaigns go to customers whose engagement has
moved. Campaigns raise interest in the product and also build fatigue, and
both feed the next week's engagement score and the next send. The outcome
is taking up the product (opening a savings pot); churn is a competing
event that fatigue makes more likely.

Within each week, in order:

1. Covariates update: interest and fatigue (latent), app sessions and email
   opens (observed signals), the CRM engagement score built from them,
   balance and weeks since the last send.
2. Campaign: softmax over none, information email, offer email and push,
   driven by the engagement score, sessions and weeks since the last send
   (a soft frequency cap).
3. Take-up, then churn.

"Never k" means the CRM does not send campaign k that week. A confirmation
email sent after take-up appears in the raw log as an information email,
the artefact a naive count picks up.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np

from ._base import WeeklySimulator

LEVELS = ("none", "email_info", "email_offer", "push")
INFO, OFFER, PUSH = 1, 2, 3


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _per_level(*values):
    return field(default_factory=lambda: dict(zip(LEVELS, values, strict=True)))


@dataclass(frozen=True)
class CampaignScenario:
    """Knobs of the campaign-and-take-up process. Defaults are the baseline."""

    n_customers: int = 50_000
    t_max: int = 26
    seed: int = 0

    # Interest I_t and fatigue F_t (latent)
    rho_i: float = 0.85
    i_balance: float = 0.3
    i_age: float = 0.1
    i_sd: float = 0.3
    gamma_interest: dict = _per_level(0.0, 0.25, 0.45, 0.2)
    """Rise in interest the week after each campaign."""
    rho_f: float = 0.7
    gamma_fatigue: dict = _per_level(0.0, 0.3, 0.3, 0.45)
    """Rise in fatigue the week after each campaign."""
    lambda_i: float = 0.0
    """Unmeasured saving intent U on interest."""
    lambda_y: float = 0.0
    """Unmeasured saving intent U on take-up."""

    # Observed signals
    sessions_base: float = 1.0
    sessions_interest: float = 0.3
    sessions_fatigue: float = -0.3
    open_base: float = -1.0
    open_interest: float = 0.6
    open_fatigue: float = -0.8

    # CRM targeting (multinomial logit against none)
    c_info: float = -2.6
    b_info_score: float = 0.5
    c_offer: float = -3.0
    b_offer_score: float = -0.6
    """Offers go to customers whose engagement score has dropped."""
    c_push: float = -3.2
    b_push_sessions: float = 0.4
    b_recent: float = -1.5
    """Soft frequency cap: lower odds of any send the week after a send."""

    # Take-up and churn hazards
    alpha_0: float = -5.6
    alpha_interest: float = 0.9
    alpha_fatigue: float = -0.4
    alpha_balance: float = 0.3
    delta: dict = _per_level(0.0, 0.2, 0.6, 0.15)
    """Direct effect of each campaign on the take-up logit that week."""
    d_0: float = -6.2
    d_fatigue: float = 0.6

    p_post: float = 0.7
    """Probability a taker's raw log shows a confirmation email afterwards."""

    PRESETS: ClassVar[tuple[str, ...]] = ("baseline", "null", "unmeasured_intent")

    @classmethod
    def preset(cls, name: str, **overrides) -> CampaignScenario:
        base = cls()
        if name == "baseline":
            sc = base
        elif name == "null":
            zero = dict.fromkeys(LEVELS, 0.0)
            sc = replace(
                base, gamma_interest=zero, gamma_fatigue=dict(zero), delta=dict(zero)
            )
        elif name == "unmeasured_intent":
            sc = replace(base, lambda_i=0.4, lambda_y=0.6)
        else:
            raise ValueError(f"unknown preset {name!r}")
        return replace(sc, **overrides)


class CampaignSimulator(WeeklySimulator):
    """Simulate weekly campaigns and product take-up."""

    LEVELS = LEVELS
    BASELINE = ("tenure_years", "n_products", "age_band", "income_band")
    TIME_VARYING = (
        "weeks_since_send",
        "balance",
        "app_sessions",
        "emails_opened",
        "engagement_score",
    )
    LOG1P = ("app_sessions",)
    BINARY = ("emails_opened",)
    OUTCOME = "took_up"
    COMPETING = "churned"
    PROXY = "engagement_score"
    ARTEFACT_LEVEL = "email_info"

    def __init__(self, scenario: CampaignScenario | None = None):
        super().__init__(scenario or CampaignScenario())
        sc = self.scenario
        self._gi = np.array([sc.gamma_interest[k] for k in LEVELS])
        self._gf = np.array([sc.gamma_fatigue[k] for k in LEVELS])
        self._delta = np.array([sc.delta[k] for k in LEVELS])

    def _baseline(self, rng, n, id_offset):
        income = rng.choice(4, size=n, p=[0.25, 0.35, 0.25, 0.15])
        return {
            "customer_id": np.arange(n) + id_offset,
            "tenure_years": rng.gamma(2.0, 1.2, n).round(1),
            "n_products": 1 + rng.binomial(3, 0.25 + 0.05 * income),
            "age_band": rng.choice(5, size=n, p=[0.2, 0.3, 0.25, 0.15, 0.1]),
            "income_band": income,
            "U": rng.standard_normal(n),
            "balance0": rng.normal(6.5 + 0.4 * income, 1.0),
        }

    def _draws(self, rng, n):
        T = self.scenario.t_max
        return {
            "z_i": rng.standard_normal((T, n)),
            "u_sess": rng.random((T, n)),
            "u_open": rng.random((T, n)),
            "z_bal": rng.standard_normal((T, n)),
            "u_a": rng.random((T, n)),
            "u_iv": rng.random((T, n)),
        }

    def _run(self, base, draws, iv, record):
        from scipy.stats import poisson

        sc = self.scenario
        n = len(base["customer_id"])
        T = sc.t_max
        mu_i = (
            sc.i_balance * (base["balance0"] - 6.5)
            + sc.i_age * (base["age_band"] - 2)
            - 0.5
            + sc.lambda_i * base["U"]
        )
        interest = mu_i.copy()
        fatigue = np.zeros(n)
        balance = base["balance0"].copy()
        since = np.full(n, 4.0)
        prev = np.zeros(n, dtype=int)
        cum = np.zeros((n, len(LEVELS)), dtype=int)
        h_y = np.empty((T, n))
        h_d = np.empty((T, n))
        recs = [] if record else None
        for t in range(T):
            if t > 0:
                interest = (
                    sc.rho_i * interest
                    + (1 - sc.rho_i) * mu_i
                    + self._gi[prev]
                    + sc.i_sd * draws["z_i"][t]
                )
                fatigue = sc.rho_f * fatigue + self._gf[prev]
                since = np.where(prev != 0, 1.0, since + 1.0)
            balance = 0.9 * balance + 0.1 * base["balance0"] + 0.2 * draws["z_bal"][t]
            sessions = poisson.ppf(
                draws["u_sess"][t],
                np.exp(
                    sc.sessions_base
                    + sc.sessions_interest * interest
                    + sc.sessions_fatigue * fatigue
                ),
            ).astype(float)
            opened = (
                draws["u_open"][t]
                < _sigmoid(
                    sc.open_base
                    + sc.open_interest * interest
                    + sc.open_fatigue * fatigue
                )
            ).astype(float)
            score = 0.5 * np.log1p(sessions) + 0.8 * opened - 0.1 * np.minimum(since, 8)

            recent = since <= 1
            eta = np.stack(
                [
                    np.zeros(n),
                    sc.c_info + sc.b_info_score * score + sc.b_recent * recent,
                    sc.c_offer + sc.b_offer_score * score + sc.b_recent * recent,
                    sc.c_push
                    + sc.b_push_sessions * np.log1p(sessions)
                    + sc.b_recent * recent,
                ]
            )
            eta -= eta.max(axis=0)
            p = np.exp(eta)
            p /= p.sum(axis=0)
            cdf = np.cumsum(p, axis=0)
            a_nat = np.minimum(
                (draws["u_a"][t][None, :] > cdf).sum(axis=0), len(LEVELS) - 1
            )
            a = iv.apply(a_nat, draws["u_iv"][t], LEVELS, t=t + 1)

            h_y[t] = _sigmoid(
                sc.alpha_0
                + sc.alpha_interest * interest
                + sc.alpha_fatigue * fatigue
                + sc.alpha_balance * (balance - 6.5)
                + self._delta[a]
                + sc.lambda_y * base["U"]
            )
            h_d[t] = _sigmoid(sc.d_0 + sc.d_fatigue * fatigue)
            if record:
                recs.append(
                    {
                        "weeks_since_send": since.copy(),
                        "balance": balance.copy(),
                        "app_sessions": sessions,
                        "emails_opened": opened,
                        "engagement_score": score,
                        "treatment": a.copy(),
                        "prev_treatment": prev.copy(),
                        "cum": cum.copy(),
                        "interest": interest.copy(),
                        "fatigue": fatigue.copy(),
                        "_latent": ["interest", "fatigue"],
                    }
                )
            cum[np.arange(n), a] += 1
            prev = a
        return h_y, h_d, recs


__all__ = ["LEVELS", "CampaignScenario", "CampaignSimulator"]
