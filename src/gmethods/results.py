"""Result tables shared by the simulator's ground truth and the estimators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REFERENCE = "natural_course"


@dataclass
class RiskTable:
    """Risk of the outcome by the end of follow-up under each regime.

    Attributes
    ----------
    risks:
        Regime name to risk.
    curves:
        Optional regime name to cumulative incidence by turn.
    label:
        Who produced the table, such as ``"truth"`` or ``"gformula"``.
    ci:
        Optional regime name to a ``(low, high)`` interval for the risk
        difference against the natural course.
    """

    risks: Mapping[str, float]
    curves: Mapping[str, np.ndarray] = field(default_factory=dict)
    label: str = ""
    ci: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def regimes(self) -> list[str]:
        return list(self.risks)

    def to_frame(self, reference: str = REFERENCE) -> pd.DataFrame:
        """One row per regime: risk, RD, RR and complaints avoided per 1,000.

        The reference regime is the natural course unless another is named.
        """
        if reference not in self.risks:
            raise KeyError(f"reference regime {reference!r} not in table")
        ref = self.risks[reference]
        rows = []
        for name, risk in self.risks.items():
            row = {
                "regime": name,
                "risk": risk,
                "rd": risk - ref,
                "rr": risk / ref if ref > 0 else np.nan,
                "avoided_per_1000": (ref - risk) * 1000.0,
            }
            if name in self.ci:
                row["rd_low"], row["rd_high"] = self.ci[name]
            rows.append(row)
        return pd.DataFrame(rows).set_index("regime")

    def rd(self, regime: str, reference: str = REFERENCE) -> float:
        """Risk difference of ``regime`` against ``reference``."""
        return float(self.risks[regime] - self.risks[reference])

    def compare(self, truth: RiskTable, reference: str = REFERENCE) -> pd.DataFrame:
        """Side-by-side RD of this table and a ground-truth table.

        Only regimes present in both tables are kept. ``error`` is the
        estimate minus the truth.
        """
        common = [r for r in self.risks if r in truth.risks]
        rows = []
        for name in common:
            est = self.rd(name, reference)
            true = truth.rd(name, reference)
            rows.append(
                {"regime": name, "rd_true": true, "rd_est": est, "error": est - true}
            )
        return pd.DataFrame(rows).set_index("regime")

    def __repr__(self) -> str:
        label = f" ({self.label})" if self.label else ""
        return f"RiskTable{label}\n{self.to_frame().round(5).to_string()}"
