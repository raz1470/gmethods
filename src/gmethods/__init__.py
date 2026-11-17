"""G-methods for time-varying confounding.

The package has two parts:

- A simulator of chat handoff data with known causal effects.
- Estimators that recover those effects, led by the parametric g-formula.
"""

from . import interventions
from .estimators import IPW, CountRegression, GFormula, PanelSpec, PooledHazard
from .interventions import HANDOFF_TYPES
from .results import RiskTable
from .sim import ChatSimulator, Scenario, SimData

__version__ = "0.1.0"

__all__ = [
    "HANDOFF_TYPES",
    "IPW",
    "ChatSimulator",
    "CountRegression",
    "GFormula",
    "PanelSpec",
    "PooledHazard",
    "RiskTable",
    "Scenario",
    "SimData",
    "__version__",
    "interventions",
]
