"""Interventions on a categorical time-varying treatment.

An intervention says how the treatment is set at every turn. The same
objects drive the simulator's ground truth and every estimator, so a
regime cannot mean one thing in the truth and another in an estimate.

Treatment levels are integer codes. Code 0 is always the reference level
("none"): when an intervention removes a treatment, the unit gets the
reference level instead. For chat handoffs that means the same agent
carries on.

Every intervention is defined relative to the natural course. At each turn
the natural treatment is drawn first. The intervention then maps it to the
treatment actually given. This makes each regime a stochastic regime that
depends on the natural value, which is the definition the g-formula,
inverse probability weighting and the simulator all share.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

NONE = 0
"""Integer code of the reference treatment level."""

HANDOFF_TYPES: tuple[str, ...] = (
    "none",
    "transfer",
    "escalation",
    "requeue",
    "shift_change",
)
"""Treatment levels used by the chat simulator, in code order."""


def _codes(levels: Sequence[str] | str, names: Sequence[str]) -> tuple[int, ...]:
    if isinstance(levels, str):
        levels = (levels,)
    out = []
    for level in levels:
        if level not in names:
            raise ValueError(f"unknown treatment level {level!r}; known: {names}")
        code = names.index(level)
        if code == NONE:
            raise ValueError("the reference level cannot be removed")
        out.append(code)
    return tuple(sorted(set(out)))


@dataclass(frozen=True)
class Intervention:
    """Base class. Subclasses implement :meth:`apply` and :meth:`transform_probs`.

    Attributes
    ----------
    name:
        Label used in result tables.
    """

    name: str

    def apply(
        self, a: np.ndarray, u: np.ndarray, names: Sequence[str], t=None
    ) -> np.ndarray:
        """Map natural treatment draws ``a`` to the treatment given.

        ``u`` holds one uniform draw per unit. Stochastic interventions use
        it so that results are reproducible under common random numbers.
        ``t`` is the time step (1-based), a scalar or one value per unit.
        Only time-dependent interventions use it.
        """
        raise NotImplementedError

    def transform_probs(
        self, p: np.ndarray, names: Sequence[str], t=None
    ) -> np.ndarray:
        """Treatment probabilities under the intervention.

        ``p`` has one row per unit and one column per level, holding the
        natural-course probabilities. The result is the distribution of the
        treatment after :meth:`apply`. Inverse probability weights are the
        ratio of the two.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Natural(Intervention):
    """No intervention: the treatment follows its natural course."""

    name: str = "natural_course"

    def apply(self, a, u, names, t=None):
        return a

    def transform_probs(self, p, names, t=None):
        return p


@dataclass(frozen=True)
class Never(Intervention):
    """Remove the listed treatment levels at every turn.

    A natural draw of a removed level becomes the reference level. All other
    levels are left as drawn.
    """

    levels: tuple[str, ...] = ()

    def apply(self, a, u, names, t=None):
        out = a.copy()
        out[np.isin(out, _codes(self.levels, names))] = NONE
        return out

    def transform_probs(self, p, names, t=None):
        codes = list(_codes(self.levels, names))
        out = p.copy()
        out[:, NONE] += out[:, codes].sum(axis=1)
        out[:, codes] = 0.0
        return out


@dataclass(frozen=True)
class Scale(Intervention):
    """Keep each natural draw of the listed levels with probability ``factor``.

    ``Scale(levels=("transfer",), factor=0.5)`` halves the probability of a
    transfer at every turn. The freed probability goes to the reference
    level. This needs weaker positivity than removing the level entirely.
    """

    levels: tuple[str, ...] = ()
    factor: float = 0.5

    def __post_init__(self):
        if not 0.0 <= self.factor <= 1.0:
            raise ValueError("factor must be in [0, 1]")

    def apply(self, a, u, names, t=None):
        out = a.copy()
        drop = np.isin(out, _codes(self.levels, names)) & (u >= self.factor)
        out[drop] = NONE
        return out

    def transform_probs(self, p, names, t=None):
        codes = list(_codes(self.levels, names))
        out = p.copy()
        freed = out[:, codes].sum(axis=1) * (1.0 - self.factor)
        out[:, codes] *= self.factor
        out[:, NONE] += freed
        return out


@dataclass(frozen=True)
class Windowed(Intervention):
    """Remove treatment levels only at some time steps.

    ``windows`` is a tuple of ``(first, last, levels)``: at time steps
    ``first`` to ``last`` (inclusive, 1-based) a natural draw of any of
    ``levels`` becomes the reference level. Outside every window the
    treatment follows its natural course. Used for the time-ordered
    attribution, where handoffs are removed up to a turn and left natural
    after it.
    """

    windows: tuple[tuple[int, int, tuple[str, ...]], ...] = ()

    def _mask(self, a_or_shape, names, t):
        if t is None:
            raise ValueError(f"{self.name} needs the time step t")
        t = np.broadcast_to(np.asarray(t), np.shape(a_or_shape)[:1])
        masks = []
        for first, last, levels in self.windows:
            masks.append(((t >= first) & (t <= last), _codes(levels, names)))
        return masks

    def apply(self, a, u, names, t=None):
        out = a.copy()
        for in_window, codes in self._mask(a, names, t):
            out[in_window & np.isin(out, codes)] = NONE
        return out

    def transform_probs(self, p, names, t=None):
        out = p.copy()
        for in_window, codes in self._mask(p, names, t):
            codes = list(codes)
            moved = out[:, codes].sum(axis=1) * in_window
            out[:, NONE] += moved
            out[np.ix_(in_window, codes)] = 0.0
        return out


def natural() -> Natural:
    """The natural course."""
    return Natural()


def never(*levels: str, name: str | None = None) -> Never:
    """Remove one or more treatment levels at every turn."""
    if not levels:
        raise ValueError("give at least one level; use never_any() for all")
    label = name or "never_" + "_".join(levels)
    return Never(name=label, levels=tuple(levels))


def never_any(names: Sequence[str] = HANDOFF_TYPES) -> Never:
    """Remove every non-reference level: no handoffs at all."""
    return Never(name="never_any", levels=tuple(names[1:]))


def scale(*levels: str, factor: float = 0.5, name: str | None = None) -> Scale:
    """Multiply the probability of one or more levels by ``factor``."""
    if not levels:
        raise ValueError("give at least one level")
    label = name or f"scale_{'_'.join(levels)}_{factor:g}"
    return Scale(name=label, levels=tuple(levels), factor=factor)


def standard_set(names: Sequence[str] = HANDOFF_TYPES) -> list[Intervention]:
    """Natural course, never each level in turn, and never any."""
    return [natural(), *(never(k) for k in names[1:]), never_any(names)]


def scale_set(
    names: Sequence[str] = HANDOFF_TYPES, factor: float = 0.5
) -> list[Intervention]:
    """Natural course, scale each level in turn, and scale all of them.

    These regimes need far weaker positivity than removing a level: only
    the natural course has to be supported, not a world where the level
    never happens.
    """
    every = Scale(name=f"scale_any_{factor:g}", levels=tuple(names[1:]), factor=factor)
    return [natural(), *(scale(k, factor=factor) for k in names[1:]), every]
