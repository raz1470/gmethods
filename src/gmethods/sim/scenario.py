"""Scenario settings for the chat handoff simulator.

Every structural coefficient lives on one :class:`Scenario`. Named presets
switch individual sources of bias on and off so each can be shown alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import ClassVar

import numpy as np

TASK_TYPES: tuple[str, ...] = (
    "card",
    "payments",
    "fraud",
    "account_access",
    "lending",
    "savings",
    "disputes",
    "general",
)
"""Task types a chat can be about."""

TEAMS: tuple[str, ...] = ("general", "payments", "fraud", "lending", "specialist")
"""Agent teams. ``specialist`` only receives escalations."""


def _per_handoff(none, transfer, escalation, requeue, shift_change):
    return field(
        default_factory=lambda: {
            "none": none,
            "transfer": transfer,
            "escalation": escalation,
            "requeue": requeue,
            "shift_change": shift_change,
        }
    )


@dataclass(frozen=True)
class Scenario:
    """All knobs of the data-generating process.

    Defaults give the ``baseline`` preset. Coefficients are on the logit
    scale for binary and multinomial hazards and on the natural scale for
    frustration, which is a latent continuous variable with unit-scale noise.
    """

    # Size and horizon
    n_chats: int = 50_000
    t_max: int = 40
    seed: int = 0
    repeat_rate: float = 0.3
    """Mean extra chats per customer (Poisson). Customers share W and U."""

    # Baseline mix
    task_mix: tuple[float, ...] = (0.22, 0.18, 0.10, 0.12, 0.08, 0.08, 0.10, 0.12)
    task_complexity: tuple[float, ...] = (0.2, 0.4, 0.9, 0.5, 0.7, 0.2, 0.8, 0.1)
    task_home_team: tuple[int, ...] = (0, 1, 2, 0, 3, 0, 1, 0)
    """Index into :data:`TEAMS` of the team that can resolve each task."""
    misroute_rate: tuple[float, ...] = (0.10, 0.25, 0.35, 0.15, 0.30, 0.10, 0.30, 0.05)
    """Probability the first agent is from the wrong team, by task."""
    friction_rate: tuple[float, ...] = (0.4, 0.6, 1.0, 0.8, 0.5, 0.2, 1.2, 0.3)
    """Poisson mean of past complaints and contacts, by task."""

    # Unmeasured temperament U ~ N(0, 1)
    lambda_f: float = 0.0
    """Effect of U on frustration. 0 switches unmeasured confounding off."""
    lambda_y: float = 0.0
    """Effect of U on the complaint logit."""

    # Queue and waits (minutes)
    queue_mean: float = 12.0
    queue_peak: float = 0.6
    """Relative size of the daytime peak in queue length."""
    queue_ar: float = 0.7
    base_wait_per_queue: float = 0.08
    queue_time: dict = _per_handoff(0.0, 5.0, 9.0, 14.0, 1.0)
    """Mean extra wait at the next turn after each handoff type."""
    wait_shape: float = 2.0

    # Frustration F_t = rho F_{t-1} + ... + noise
    rho: float = 0.75
    beta_wait: float = 0.35
    """Effect of log(1 + wait) on frustration."""
    gamma: dict = _per_handoff(0.0, 0.5, 0.4, 0.5, 0.15)
    """Jolt to frustration at the turn after each handoff type."""
    beta_complexity_f: float = 0.6
    beta_friction_f: float = 0.15
    frustration_sd: float = 0.5
    sentiment_noise: float = 0.0
    """SD of measurement error when sentiment proxies frustration."""

    # Customer idle time (minutes)
    idle_mean: float = 2.5
    idle_frustration: float = 0.25
    idle_age: float = 0.15
    idle_shape: float = 1.2

    # Agent replies and shifts (minutes)
    reply_mean: float = 2.0
    shift_left_range: tuple[float, float] = (5.0, 300.0)
    """Uniform range of shift time left when a new agent picks up a chat."""
    p_stay: float = 0.1
    """Probability an agent finishes the chat past the end of their shift."""
    shift_softness: float = 1.5
    """Logistic scale (minutes) of the shift-end trigger."""

    # Handoff hazard (multinomial logit against "none")
    c_transfer: float = -5.0
    b_mismatch: float = 3.5
    b_transfer_early: float = 0.8
    c_escalation: float = -6.0
    b_frustration_esc: float = 0.6
    b_complexity_esc: float = 1.2
    b_friction_esc: float = 0.2
    b_specialist_esc: float = -3.0
    """Specialists rarely escalate further."""
    escalation_determinism: float = 0.0
    """Extra escalation logit for the ``fraud`` task. Large values break positivity."""
    c_requeue: float = -3.6
    b_idle_requeue: float = 2.0
    idle_threshold: float = 5.0
    b_queue_requeue: float = 0.05

    # Complaint hazard
    alpha_0: float = -7.7
    alpha_frustration: float = 0.6
    alpha_friction: float = 0.25
    alpha_products: float = 0.1
    alpha_income: float = -0.1
    delta: dict = _per_handoff(0.0, 0.4, 0.5, 0.45, 0.15)
    """Direct effect of each handoff on the complaint logit at the same turn."""
    delta_agents: float = 0.1
    """Effect of each agent already involved before this turn."""

    # Chat end (resolution or abandonment)
    d_0: float = -3.4
    d_progress: float = 0.3
    d_frustration: float = 0.2
    progress_gain_match: float = 1.0
    progress_gain_mismatch: float = 0.2
    progress_keep: dict = _per_handoff(1.0, 0.3, 0.5, 0.6, 0.8)
    """Share of resolution progress kept after each handoff type."""
    handoff_blocks_end: bool = True
    """A chat cannot end at a turn with a handoff (a new agent has it)."""

    # Raw-log artefact
    p_post: float = 0.3
    """Probability of an escalation logged after a complaint."""

    @classmethod
    def preset(cls, name: str, **overrides) -> Scenario:
        """Build a named preset, then apply keyword overrides.

        Presets:

        - ``baseline``: time-varying confounding and feedback, sentiment
          measures frustration exactly, no unmeasured confounding.
        - ``null``: no handoff affects complaints by any path.
        - ``strong_feedback``: larger handoff-to-frustration jolts.
        - ``noisy_sentiment``: sentiment measures frustration with error.
        - ``unmeasured_confounding``: temperament U drives frustration and
          complaints and is not in the data.
        - ``poor_positivity``: fraud chats are nearly always escalated and
          agents rarely stay past shift end.
        """
        base = cls()
        if name == "baseline":
            sc = base
        elif name == "null":
            zero = dict.fromkeys(base.delta, 0.0)
            sc = replace(
                base,
                delta=zero,
                gamma=dict(zero),
                queue_time=dict(zero),
                delta_agents=0.0,
                progress_keep=dict.fromkeys(base.progress_keep, 1.0),
                progress_gain_mismatch=base.progress_gain_match,
                handoff_blocks_end=False,
            )
        elif name == "strong_feedback":
            sc = replace(base, gamma={k: 2.0 * v for k, v in base.gamma.items()})
        elif name == "noisy_sentiment":
            sc = replace(base, sentiment_noise=0.6)
        elif name == "unmeasured_confounding":
            sc = replace(base, lambda_f=0.5, lambda_y=0.5)
        elif name == "poor_positivity":
            sc = replace(base, escalation_determinism=3.0, p_stay=0.01)
        else:
            raise ValueError(f"unknown preset {name!r}")
        return replace(sc, **overrides)

    PRESETS: ClassVar[tuple[str, ...]] = (
        "baseline",
        "null",
        "strong_feedback",
        "noisy_sentiment",
        "unmeasured_confounding",
        "poor_positivity",
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self):
        n = len(TASK_TYPES)
        for name in (
            "task_mix",
            "task_complexity",
            "task_home_team",
            "misroute_rate",
            "friction_rate",
        ):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} needs {n} entries, one per task type")
        if not np.isclose(sum(self.task_mix), 1.0):
            raise ValueError("task_mix must sum to 1")
        if self.t_max < 1 or self.n_chats < 1:
            raise ValueError("t_max and n_chats must be positive")
