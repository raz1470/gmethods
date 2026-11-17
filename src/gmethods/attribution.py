"""Time-ordered attribution of complaints to handoff types.

Handoffs cause only part of all complaints. The part they cause is the
difference between the natural course and "no handoffs at all"; as a share
of the natural-course risk it is the population attributable fraction. This
module splits that part across handoff types so the shares add up to it
exactly.

The split works forward through time. Let g_j be the policy "no handoffs in
the first j turn buckets, natural course after". Then g_0 is the natural
course and the last g_j is "no handoffs at all", so

    R(natural) - R(no handoffs) = sum_j [R(g_{j-1}) - R(g_j)]

and bucket j's credit is what removing its handoffs adds, with every earlier
handoff already removed and every later one left to happen naturally. An
earlier handoff therefore gets credit for the later handoffs it triggers: a
transfer that frustrates the customer into an escalation keeps the credit
for that escalation.

Within a single turn a chat has at most one handoff, so removing type k at
that turn only changes chats whose natural handoff there was k. The type
credits at a one-turn bucket add up to the bucket credit. A bucket spanning
several turns can leave a small remainder (removing one type changes later
handoffs of other types within the bucket); it is shared across types in
proportion to their credits and reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import pandas as pd

from .interventions import HANDOFF_TYPES, Intervention, Windowed, natural

DEFAULT_BUCKETS: tuple[tuple[int, int], ...] = (
    *((t, t) for t in range(1, 26)),
    (26, 10_000),
)
"""One bucket per turn for turns 1 to 25, then one bucket for the rest.

Single-turn buckets make the type split exact. The tail bucket holds few
chats (about 0.05% of chats reach turn 26).
"""


def _base_name(j: int) -> str:
    return f"none_through_bucket_{j}"


def attribution_regimes(
    buckets: Sequence[tuple[int, int]] = DEFAULT_BUCKETS,
    names: Sequence[str] = HANDOFF_TYPES,
) -> list[Intervention]:
    """Every regime the time-ordered split needs.

    The natural course, ``none_through_bucket_j`` for each bucket, and
    ``bucket_j_no_<type>``: no handoffs before bucket j, no ``type`` within
    it, natural after. The last base regime is "no handoffs at all".
    """
    if buckets[0][0] != 1 or buckets[-1][1] < 10_000:
        raise ValueError("buckets must start at turn 1 and run to the end")
    if any(b[0] != a[1] + 1 for a, b in pairwise(buckets)):
        raise ValueError("buckets must be contiguous")
    levels = tuple(names[1:])
    regimes: list[Intervention] = [natural()]
    for j, (first, last) in enumerate(buckets):
        before = ((1, first - 1, levels),) if first > 1 else ()
        for k in levels:
            regimes.append(
                Windowed(
                    name=f"bucket_{j}_no_{k}",
                    windows=(*before, (first, last, (k,))),
                )
            )
        regimes.append(Windowed(name=_base_name(j), windows=((1, last, levels),)))
    return regimes


@dataclass
class Attribution:
    """Time-ordered split of complaints caused by handoffs.

    Attributes
    ----------
    by_bucket:
        Credit (risk avoided) per turn bucket and handoff type.
    risk_natural, risk_no_handoffs:
        Risk under the natural course and with no handoffs at all.
    remainder:
        Per bucket, bucket credit minus the sum of its type credits, before
        it was shared out. Near zero for one-turn buckets.
    se:
        Optional standard errors of each type's total credit and of the
        attributable risk.
    """

    by_bucket: pd.DataFrame
    risk_natural: float
    risk_no_handoffs: float
    remainder: pd.Series
    label: str = ""
    se: dict = field(default_factory=dict)

    @property
    def attributable_risk(self) -> float:
        """Complaint risk caused by handoffs: natural minus no handoffs."""
        return self.risk_natural - self.risk_no_handoffs

    @property
    def attributable_fraction(self) -> float:
        """Share of all complaints caused by handoffs."""
        return self.attributable_risk / self.risk_natural

    def summary(self) -> pd.DataFrame:
        """Per handoff type: complaints per 1,000 chats and shares.

        ``share_of_handoff_complaints`` adds up to 1 over the types.
        ``share_of_all_complaints`` adds up to the attributable fraction;
        the rest of all complaints would happen without any handoff.
        """
        credit = self.by_bucket.sum(axis=0)
        out = pd.DataFrame(
            {
                "per_1000": credit * 1000,
                "share_of_handoff_complaints": credit / self.attributable_risk,
                "share_of_all_complaints": credit / self.risk_natural,
            }
        )
        if self.se:
            out["se_per_1000"] = pd.Series(
                {k: v * 1000 for k, v in self.se.items() if k in out.index}
            )
        out.loc["all handoffs"] = [
            self.attributable_risk * 1000,
            1.0,
            self.attributable_fraction,
            *(
                [self.se["all handoffs"] * 1000]
                if "all handoffs" in self.se
                else ([np.nan] if self.se else [])
            ),
        ]
        out.loc["not caused by handoffs"] = [
            self.risk_no_handoffs * 1000,
            np.nan,
            1 - self.attributable_fraction,
            *([np.nan] if self.se else []),
        ]
        return out


def attribute(
    risks: dict[str, float],
    buckets: Sequence[tuple[int, int]] = DEFAULT_BUCKETS,
    names: Sequence[str] = HANDOFF_TYPES,
    label: str = "",
) -> Attribution:
    """Build the split from regime risks (truth or any estimator)."""
    levels = list(names[1:])
    rows, remainder = [], {}
    prev = risks["natural_course"]
    for j, _ in enumerate(buckets):
        credit = {k: prev - risks[f"bucket_{j}_no_{k}"] for k in levels}
        base = risks[_base_name(j)]
        total = prev - base
        raw_sum = sum(credit.values())
        remainder[j] = total - raw_sum
        if raw_sum != 0:
            credit = {k: v * total / raw_sum for k, v in credit.items()}
        rows.append(credit)
        prev = base
    index = pd.Index(
        [
            f"{a}" if a == b else f"{a}+" if b >= 10_000 else f"{a}-{b}"
            for a, b in buckets
        ],
        name="turns",
    )
    return Attribution(
        by_bucket=pd.DataFrame(rows, index=index),
        risk_natural=risks["natural_course"],
        risk_no_handoffs=prev,
        remainder=pd.Series(remainder, index=range(len(buckets))).set_axis(index),
        label=label,
    )


def attribute_ltmle(
    model,
    buckets: Sequence[tuple[int, int]] = DEFAULT_BUCKETS,
    names: Sequence[str] = HANDOFF_TYPES,
) -> Attribution:
    """Estimate the split with a fitted :class:`~gmethods.LTMLE`.

    Standard errors for each type's total credit and for the attributable
    risk come from the influence functions. They ignore the proportional
    sharing of multi-turn remainders, which is small.
    """
    regimes = attribution_regimes(buckets, names)
    table = model.estimate(regimes)
    out = attribute(dict(table.risks), buckets, names, label=model.label)
    if model.target:
        levels = list(names[1:])
        se = {}
        for k in levels:
            coefs: dict[str, float] = {}
            prev = "natural_course"
            for j, _ in enumerate(buckets):
                coefs[prev] = coefs.get(prev, 0.0) + 1.0
                key = f"bucket_{j}_no_{k}"
                coefs[key] = coefs.get(key, 0.0) - 1.0
                prev = _base_name(j)
            se[k] = model.contrast_se(coefs)
        last = _base_name(len(buckets) - 1)
        se["all handoffs"] = model.contrast_se({"natural_course": 1.0, last: -1.0})
        out.se = se
    return out
