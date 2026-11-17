# gmethods

[![CI](https://github.com/raz1470/gmethods/actions/workflows/ci.yml/badge.svg)](https://github.com/raz1470/gmethods/actions/workflows/ci.yml)

**G-methods for time-varying confounding, tested on simulated support chats where the true effect of every handoff is known.**

- A handoff (transfer, escalation, requeue, shift change) makes the customer wait and raises their frustration. Frustration then drives later escalations and complaints. The confounder is caused by earlier treatment, so regression cannot both adjust for it and keep the effect that runs through it.
- The parametric g-formula models every step of the chat and simulates it forward with handoffs switched off. IPW reweights chats instead. Both target the same interventions as the simulator's ground truth.
- Naive analyses get the size and even the ranking of the handoff effects wrong. Counting handoffs from the raw log doubles the escalation effect, because escalations are logged after complaints.

![Estimated effect of removing each handoff type, by estimator, against the truth.](https://raw.githubusercontent.com/raz1470/gmethods/main/assets/hero.png)

## Quick start

```bash
git clone https://github.com/raz1470/gmethods
cd gmethods
uv sync
uv run pytest
```

```python
from gmethods import IPW, ChatSimulator, GFormula, Scenario
from gmethods import interventions as iv

sim = ChatSimulator(Scenario.preset("baseline"))
data = sim.simulate(n_chats=50_000, seed=1)       # .panel, .event_log, .chats
regimes = iv.standard_set()                        # natural, never each type, never any

truth = sim.true_risks(regimes, n_mc=500_000)      # the known answer
est = GFormula().fit(data.panel).estimate(regimes)
print(est.compare(truth))
```

## What is in the package

| Module | What it does |
| --- | --- |
| `gmethods.sim` | `Scenario` (every knob, six presets) and `ChatSimulator` (data and ground truth) |
| `gmethods.interventions` | `natural()`, `never(*types)`, `never_any()`, `scale(*types, factor)` |
| `gmethods.estimators` | `GFormula`, `IPW`, naive baselines, cluster `bootstrap`, diagnostics |
| `gmethods.results` | `RiskTable`: risk, RD, RR and complaints avoided per 1,000 chats |

The same intervention objects drive the truth and every estimator, so a
regime cannot mean one thing in the truth and another in an estimate.

## Recovery study

Twenty datasets of 50,000 chats from the `baseline` preset. Mean estimate as
a share of the true effect (100% is unbiased), with RMSE per 1,000 chats in
brackets. Truth from one million simulated chats per regime.

| Regime | True effect per 1,000 | g-formula | IPW | Count, all handoffs | Count, before complaint or all | Pooled hazard |
| --- | --- | --- | --- | --- | --- | --- |
| never escalation | −4.09 | 93% (0.48) | 109% (0.58) | 218% (4.83) | 79% (0.89) | 26% (3.09) |
| never requeue | −3.09 | 95% (0.36) | 99% (0.40) | 22% (2.44) | 65% (1.11) | 19% (2.52) |
| never shift change | −1.32 | 81% (0.36) | 73% (0.49) | 3% (1.30) | 22% (1.06) | 15% (1.15) |
| never transfer | −0.45 | 153% (0.59) | 222% (1.68) | 487% (1.79) | 625% (2.40) | 207% (0.62) |
| never any | −5.69 | 83% (1.42) | 114% (3.66) | 181% (4.67) | 122% (1.37) | 47% (3.11) |

- The g-formula has the lowest RMSE for every regime. It still shrinks the
  larger effects by 5% to 20%: the boosted hazard models are pulled toward
  the mean on a 2% outcome.
- Transfers are near-certain once a chat is with the wrong team, so "never
  transfer" asks about chats the data barely contains. Every estimator
  overstates its small effect; IPW weights reach the hundreds.
- Counting every logged handoff doubles the escalation effect: escalations
  logged after a complaint look like causes of it.
- The pooled hazard holds frustration and waits at their observed values, so
  it misses most of each effect.

Regenerate with `uv run python tools/recovery.py --preset baseline --reps 20`
(about five minutes per replicate on one core).

## Notebooks

1. [`01_simulator`](notebooks/01_simulator.ipynb): the data, one chat end to end, the true effects.
2. [`02_naive_vs_gformula`](notebooks/02_naive_vs_gformula.ipynb): every estimator on one dataset, with diagnostics.
3. [`03_recovery`](notebooks/03_recovery.ipynb): bias and RMSE across repeated datasets.

## Docs

- [The simulator](docs/simulator.md): time grid, variables, structural equations, presets.
- [Estimators](docs/estimators.md): g-formula, IPW, naive baselines, bootstrap, diagnostics, using your own panel.

## Development

```bash
uv run --no-default-groups --group test ruff format .
uv run --no-default-groups --group test ruff check . --fix
uv run --no-default-groups --group test pytest -q
```

Notebooks are built by `tools/build_notebooks.py` and the recovery study by
`tools/recovery.py`.

## Licence

MIT
