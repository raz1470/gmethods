"""Build the demo notebooks from source cells.

    uv run python tools/build_notebooks.py          # all, or e.g. "04"
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

Notebook 03 reads results/recovery_baseline.csv, which comes from
tools/recovery.py.
"""

from pathlib import Path

import nbformat as nbf

OUT = Path("notebooks")

STYLE = """import matplotlib.pyplot as plt

# Reference categorical palette, fixed order. Truth is drawn in ink.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "legend.frameon": False,
})"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def nb01():
    return [
        md("""
# 01. The chat handoff simulator

A chat runs for up to 40 turns. At each turn the confounders update, a handoff
may happen, a complaint may happen and the chat may end. Handoffs feed back into
later confounders: a transfer adds queue time at the next turn, raises
frustration and resets resolution progress. Frustration then drives later
escalations and complaints. That feedback is time-varying confounding affected
by earlier treatment.

The simulator knows the true effect of every intervention, because it can rerun
the same chats with handoffs switched off.
"""),
        code("""
import numpy as np
import pandas as pd

from gmethods import ChatSimulator, Scenario
from gmethods import interventions as iv

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)
"""),
        code(STYLE),
        md("""
## Scenario and data

Every knob lives on `Scenario`. Presets switch sources of bias on and off:
`baseline`, `null`, `strong_feedback`, `noisy_sentiment`,
`unmeasured_confounding` and `poor_positivity`.
"""),
        code("""
sim = ChatSimulator(Scenario.preset("baseline"))
data = sim.simulate(n_chats=20_000, seed=0)
chats = data.chats
print(f"chats: {len(chats):,}  chat-turns: {len(data.panel):,}")
print(f"complaint rate: {chats['complained'].mean():.2%}")
print(f"mean turns: {chats['n_turns'].mean():.1f}")
chats[[f"n_{k}" for k in ["transfer", "escalation", "requeue", "shift_change"]]].mean().round(3)
"""),
        md("""
## Three views of the same data

- `panel`: one row per chat-turn while the chat is open and before any complaint.
  This is what the g-methods use.
- `event_log`: timestamped events, as a support platform logs them. It includes
  escalations logged *after* a complaint, the reverse-causality artefact.
- `chats`: one row per chat with counts. This is what a naive analysis uses.
"""),
        code("""
one = chats.loc[(chats["complained"] == 1) & (chats["total_agents"] >= 3), "chat_id"].iloc[0]
cols = ["turn", "queue_length", "wait", "sentiment", "idle", "skill_mismatch",
        "with_specialist", "handoff", "complaint", "ended"]
data.panel.loc[data.panel["chat_id"] == one, cols].round(2)
"""),
        code("""
data.event_log[data.event_log["chat_id"] == one]
"""),
        md("""
Read the wait column against the handoff column one row up. Wait at turn t is
the wait *going into* turn t, so it carries the queue time caused by the handoff
at turn t-1. It can confound the handoff at turn t but is never caused by it.
"""),
        md("""
## The truth

`true_risks` reruns the structural equations under each intervention. Every
regime sees the same chats and random numbers, so the differences between
regimes carry little Monte Carlo noise. "Never k" means a draw of k becomes no
handoff: the same agent carries on, and whatever triggered the handoff is still
there.
"""),
        code("""
regimes = [*iv.standard_set(), iv.scale("escalation", factor=0.5)]
truth = sim.true_risks(regimes, n_mc=200_000)
truth.to_frame().round(4)
"""),
        code("""
fig, ax = plt.subplots(figsize=(7, 3.6))
show = ["natural_course", "never_transfer", "never_escalation", "never_requeue",
        "never_shift_change", "never_any"]
for i, name in enumerate(show):
    curve = truth.curves[name] * 1000
    ax.plot(np.arange(1, len(curve) + 1), curve, lw=2, color=INK if i == 0 else SERIES[i - 1],
            label=name.replace("_", " "))
ax.set_xlim(1, 25)
ax.set_xlabel("turn")
ax.set_ylabel("complaints per 1,000 chats")
ax.set_title("True cumulative complaints by intervention", loc="left")
ax.legend(ncol=2, fontsize=9)
plt.tight_layout()
"""),
        md("""
The single-type effects sum to more than "never any". Each one includes
complaints that also run through other handoffs (a transfer that raises
frustration and so triggers a later escalation), so summing them counts the
shared routes twice.
"""),
        code("""
f = truth.to_frame()
parts = f.loc[["never_transfer", "never_escalation", "never_requeue", "never_shift_change"], "rd"]
print(f"sum of single-type RDs: {parts.sum():.4f}")
print(f"never_any RD:          {f.loc['never_any', 'rd']:.4f}")
"""),
        md("""
## Null check

With every handoff effect set to zero, every true risk difference is zero.
"""),
        code("""
null = ChatSimulator(Scenario.preset("null")).true_risks(iv.standard_set(), n_mc=50_000)
null.to_frame()["rd"].round(6)
"""),
    ]


def nb02():
    return [
        md("""
# 02. Naive estimates vs g-methods

One simulated dataset, every estimator, compared with the truth.

- **Count regression, all handoffs.** Complaint on handoff counts from the raw
  log. Escalations logged after a complaint inflate the escalation effect.
- **Count regression, "before complaint, or all".** Fixes the reverse causality
  but gives complainers a shorter exposure window than everyone else.
- **Count regression plus sentiment.** Adjusts for frustration, which handoffs
  partly cause.
- **Pooled hazard.** Turn-level model on current covariates. Holds later
  frustration and waits at their observed values, so it misses the effect that
  runs through them.
- **Parametric g-formula.** Models every time-varying variable and simulates
  forward under each intervention.
- **IPW.** Weights by the treatment model only. A cross-check that relies on
  different modelling assumptions.
- **LTMLE.** Sequential outcome regressions, targeted with the IPW weights.
  Doubly robust, with influence-function intervals. The recommended estimator.
"""),
        code("""
import pandas as pd

from gmethods import (
    IPW, LTMLE, ChatSimulator, CountRegression, GFormula, PooledHazard, Scenario,
)
from gmethods import interventions as iv
from gmethods.estimators import natural_course_check, positivity_table, weight_summary

pd.set_option("display.width", 120)
"""),
        code(STYLE),
        code("""
sim = ChatSimulator(Scenario.preset("baseline"))
data = sim.simulate(n_chats=50_000, seed=1)
regimes = iv.standard_set()
halving = iv.scale_set()
truth = sim.true_risks(regimes + halving[1:], n_mc=500_000)
"""),
        code("""
fast = {"max_iter": 100, "learning_rate": 0.1}
gf = GFormula(learner_params=fast).fit(data.panel)
ipw = IPW(learner_params=fast).fit(data.panel)
ltmle = LTMLE().fit(data.panel)
tables = {
    "count, all handoffs": CountRegression("all").fit(data.chats).estimate(regimes),
    "count, before complaint or all": CountRegression("before_complaint_or_all")
        .fit(data.chats).estimate(regimes),
    "count + sentiment": CountRegression("before_complaint_or_all", adjust_sentiment=True)
        .fit(data.chats).estimate(regimes),
    "pooled hazard": PooledHazard().fit(data.panel).estimate(regimes),
    "IPW": ipw.estimate(regimes),
    "g-formula": gf.estimate(regimes, n_mc=20_000),
    "LTMLE": ltmle.estimate(regimes),
}
rd = pd.DataFrame({k: t.to_frame()["rd"] for k, t in tables.items()})
rd.insert(0, "truth", truth.to_frame()["rd"].loc[rd.index])
(rd.drop(index="natural_course") * 1000).round(2)
"""),
        md("Risk differences per 1,000 chats against the natural course."),
        code("""
import numpy as np

shown = [r for r in rd.index if r != "natural_course"]
# Six estimators keep the categorical palette within its validated range.
est = ["count, all handoffs", "count, before complaint or all", "pooled hazard",
       "IPW", "g-formula", "LTMLE"]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
y = np.arange(len(shown))
step = 0.11
for j, name in enumerate(est):
    off = (j - (len(est) - 1) / 2) * step
    ax.scatter(rd.loc[shown, name] * 1000, y + off, s=30, color=SERIES[j], label=name,
               edgecolor="white", linewidth=1, zorder=3)
ax.scatter(rd.loc[shown, "truth"] * 1000, y, marker="|", s=500, color=INK, lw=2.5, zorder=4)
ax.set_yticks(y, [s.replace("_", " ") for s in shown])
ax.invert_yaxis()
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel("change in complaints per 1,000 chats")
ax.set_title("Estimated effect of removing each handoff type vs the truth", loc="left")
handles, labels = ax.get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=INK, lw=2.5))
labels.append("truth")
ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
plt.tight_layout()
"""),
        md("""
## LTMLE intervals, and halving instead of removing

"Never transfer" asks about chats that stay with the wrong team, which the
data barely contains. Halving each handoff type needs far weaker positivity.
LTMLE gives a 95% interval for every regime from its influence function.
"""),
        code("""
table = ltmle.estimate(regimes + halving[1:]).to_frame()
table["truth"] = truth.to_frame()["rd"]
(table[["rd", "rd_low", "rd_high", "truth"]].drop(index="natural_course") * 1000).round(2)
"""),
        md("""
## Diagnostics

The g-formula's natural course should reproduce the observed data: the
cumulative complaint risk and the mean of each confounder by turn.
"""),
        code("""
natural_course_check(gf, n_turns=8).round(3)
"""),
        md("""
Positivity: removal regimes need "no handoff" to be possible at every turn.
IPW weights show where it is thin. Transfers are close to certain for misrouted
chats, which is why IPW struggles with `never_transfer`.
"""),
        code("""
weight_summary(ipw).round(3)
"""),
        code("""
positivity_table(ipw, by="skill_mismatch").round(3)
"""),
    ]


def nb03():
    return [
        md("""
# 03. Recovery study

Repeated datasets of 50,000 chats from the `baseline` preset. Each estimator is
fitted to each dataset. The truth comes from one million simulated chats per
regime. Regenerate with:

```bash
uv run python tools/recovery.py --preset baseline --reps 20
```
"""),
        code("""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("..").resolve() / "tools"))
from recovery import summarise

pd.set_option("display.width", 120)
"""),
        code(STYLE),
        code("""
files = sorted(Path("../results").glob("recovery_baseline_*.csv"))
full = summarise(files, Path("../results/truth_baseline.csv"))
summary = full.xs(50_000, level="n_chats")
print(summary["reps"].groupby("estimator").max().to_string())
(summary[["rd_true", "rd_mean", "bias", "sd", "rmse"]] * 1000).round(2)
"""),
        md("Root mean squared error per 1,000 chats, by estimator and regime."),
        code("""
rmse = (summary["rmse"] * 1000).unstack("estimator")
# Six estimators keep the categorical palette within its validated range.
shown_est = ["ltmle", "gformula", "ipw", "ice", "pooled_hazard", "count_all"]
order = rmse[shown_est].mean().sort_values().index
rmse = rmse[order]
fig, ax = plt.subplots(figsize=(9.5, 4))
y = np.arange(len(rmse.index))
step = 0.11
for j, name in enumerate(order):
    off = (j - (len(order) - 1) / 2) * step
    ax.scatter(rmse[name], y + off, s=30, color=SERIES[j], label=name,
               edgecolor="white", linewidth=1, zorder=3)
ax.set_yticks(y, [s.replace("_", " ") for s in rmse.index])
ax.invert_yaxis()
ax.set_xlim(left=0)
ax.set_xlabel("RMSE of the risk difference, per 1,000 chats")
ax.set_title("Error against the truth, 50,000 chats per dataset", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
plt.tight_layout()
"""),
        md("""
Relative bias: the mean estimate as a share of the true effect. 100% means
unbiased on average.
"""),
        code("""
rel = (summary["rd_mean"] / summary["rd_true"] * 100).unstack("estimator")
rel.round(0)
"""),
        md("""
Coverage of LTMLE's 95% influence-function intervals: the share of datasets
whose interval contains the true effect.
"""),
        code("""
summary.xs("ltmle", level="estimator")[["coverage"]].round(2)
"""),
        md("""
## Larger samples

LTMLE and the g-formula at 200,000 chats. Bias that stays as the sample
grows comes from the method; bias that shrinks came from small samples.
"""),
        code("""
sizes = sorted(full.index.get_level_values("n_chats").unique())
if len(sizes) > 1:
    keep = full.loc[full.index.get_level_values("estimator").isin(["gformula", "ltmle"])]
    out = (keep["rd_mean"] / keep["rd_true"] * 100).unstack("n_chats").round(0)
    out["rmse_ratio"] = (keep["rmse"].unstack("n_chats")[sizes[-1]]
                         / keep["rmse"].unstack("n_chats")[sizes[0]]).round(2)
    display(out)
"""),
    ]


def nb04():
    return [
        md("""
# 04. Attributing complaints to handoff types

Two numbers answer "how much do handoffs drive complaints?":

1. **The headline:** the share of complaints handoffs cause, from LTMLE's
   "no handoffs at all" regime. The rest would happen anyway.
2. **The split:** complaints each handoff type would save if it were 10%
   rarer than today. The type effects add up to the joint effect, up to a
   small interaction that is reported.

The time-ordered split is shown at the end for comparison.
"""),
        code("""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gmethods import LTMLE, ChatSimulator, Scenario
from gmethods import interventions as iv
from gmethods.attribution import attribute, attribute_marginal, marginal_from_risks

sys.path.insert(0, str(Path("..").resolve() / "tools"))
from recovery import summarise

pd.set_option("display.width", 120)
"""),
        code(STYLE),
        code("""
sim = ChatSimulator(Scenario.preset("baseline"))
data = sim.simulate(n_chats=50_000, seed=1)
model = LTMLE().fit(data.panel)
truth = pd.read_csv("../results/truth_baseline.csv", index_col="regime")
"""),
        md("## The headline"),
        code("""
table = model.estimate([iv.natural(), iv.never_any()])
risk, rd = table.risks["natural_course"], -table.rd("never_any")
low, high = (-x for x in table.ci["never_any"][::-1])
true_paf = -truth.loc["never_any", "rd"] / truth.loc["natural_course", "risk"]
print(f"complaints per 1,000 chats: {risk * 1000:.1f}")
print(f"caused by handoffs: {rd * 1000:.1f} (95% CI {low * 1000:.1f} to {high * 1000:.1f})")
print(f"share of all complaints: {rd / risk:.0%} (truth {true_paf:.0%})")
"""),
        md("## The marginal split"),
        code("""
est = attribute_marginal(model, factor=0.9)
true = marginal_from_risks(truth["risk"].to_dict(), factor=0.9)
true_cols = true[["per_1000", "share"]].add_suffix("_true")
split = est[["per_1000", "se_per_1000", "share"]].join(true_cols)
split.round(3)
"""),
        code("""
types = ["escalation", "requeue", "shift_change", "transfer"]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
y = np.arange(len(types))
ax.errorbar(est.loc[types, "per_1000"], y, xerr=1.96 * est.loc[types, "se_per_1000"],
            fmt="o", color=SERIES[0], ms=7, lw=2, capsize=0, label="LTMLE, 95% CI")
ax.scatter(true.loc[types, "per_1000"], y, marker="|", s=400, color=INK, lw=2.5,
           label="truth", zorder=4)
ax.set_yticks(y, [t.replace("_", " ") for t in types])
ax.invert_yaxis()
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel("complaints saved per 1,000 chats, type made 10% rarer")
ax.set_title("Which handoff type to reduce first", loc="left")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
"""),
        md("""
## Recovery across datasets

LTMLE's marginal split across 20 datasets of 50,000 chats. Regenerate with
`uv run python tools/recovery.py --regimes marginal --estimators ltmle --tag marginal`.
"""),
        code("""
files = sorted(Path("../results").glob("recovery_baseline_n50000_marginal_*.csv"))
rec = summarise(files, Path("../results/truth_baseline.csv")).xs(50_000, level="n_chats")
rec = rec.xs("ltmle", level="estimator")
out = rec[["rd_true", "rd_mean", "sd", "rmse"]] * -1000
out[["sd", "rmse"]] *= -1
out["coverage"] = rec["coverage"]
out["reps"] = rec["reps"]
out.round(3)
"""),
        md("""
## For comparison: the time-ordered split

Credit goes turn by turn, each turn valued with every earlier handoff
already removed. It splits the whole attributable risk, but later types are
valued in a world where chats run longer than they do today, and LTMLE
estimates it far less precisely. Truth from one million simulated chats:
"""),
        code("""
risks = pd.read_csv("../results/attribution_truth_baseline.csv", index_col="regime")
time_ordered = attribute(risks["risk"].to_dict(), label="truth").summary()
time_ordered[["per_1000", "share_of_handoff_complaints"]].round(3)
"""),
    ]


def main(only: list[str]):
    """Build every notebook, or only those whose name starts with ``only``."""
    OUT.mkdir(exist_ok=True)
    for name, cells in [
        ("01_simulator", nb01()),
        ("02_estimators", nb02()),
        ("03_recovery", nb03()),
        ("04_attribution", nb04()),
    ]:
        if only and not any(name.startswith(o) for o in only):
            continue
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        nb["metadata"]["kernelspec"] = {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
        nbf.write(nb, OUT / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
