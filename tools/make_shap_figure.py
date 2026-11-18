"""Figure: SHAP driver shares against the true marginal split, three problems.

For each simulator, one dataset of 50,000 units: a boosted model of the
outcome on the unit summary read with SHAP, against the true share of the
effect of making each treatment type 10% rarer.

    uv run python tools/make_shap_figure.py   # writes assets/ and results/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gmethods import (
    CampaignScenario,
    CampaignSimulator,
    ChatSimulator,
    FrictionScenario,
    FrictionSimulator,
    Scenario,
)
from gmethods.attribution import marginal_from_risks
from gmethods.estimators import ShapDrivers
from gmethods.interventions import HANDOFF_TYPES

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TRUE_C, SHAP_C = "#2a78d6", "#eb6834"


def rows() -> pd.DataFrame:
    out = []
    problems = [
        (
            "Friction and churn",
            FrictionSimulator(FrictionScenario.preset("baseline")),
            "truth_friction.csv",
            "churn",
        ),
        (
            "Campaigns and take-up",
            CampaignSimulator(CampaignScenario.preset("baseline")),
            "truth_campaign.csv",
            "take-up",
        ),
        ("Chat handoffs and complaints", None, "truth_baseline.csv", "complaints"),
    ]
    for title, sim, truth_file, outcome in problems:
        risks = pd.read_csv(ROOT / "results" / truth_file, index_col="regime")["risk"]
        if sim is None:
            levels = HANDOFF_TYPES
            data = ChatSimulator(Scenario.preset("baseline")).simulate(
                n_chats=50_000, seed=1
            )
            shap = ShapDrivers.for_chats().fit(data.chats).drivers()
        else:
            levels = sim.LEVELS
            data = sim.simulate(n=50_000, seed=1)
            shap = ShapDrivers.for_simulator(sim).fit(data.customers).drivers()
        true = marginal_from_risks(risks.to_dict(), factor=0.9, names=levels)
        for lv in levels[1:]:
            out.append(
                {
                    "problem": title,
                    "outcome": outcome,
                    "type": lv,
                    "true_share": true.loc[lv, "share"],
                    "shap_share": shap.loc[lv, "share"],
                    "shap_direction": "raises"
                    if shap.loc[lv, "mean_shap_when_present"] > 0
                    else "lowers",
                }
            )
    return pd.DataFrame(out)


def draw(df: pd.DataFrame, path: Path) -> None:
    problems = list(dict.fromkeys(df["problem"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for ax, prob in zip(axes, problems, strict=True):
        d = df[df["problem"] == prob].sort_values("true_share", ascending=False)
        y = np.arange(len(d))
        h = 0.36
        ax.barh(
            y - h / 2,
            d["true_share"] * 100,
            height=h,
            color=TRUE_C,
            label="true share of the effect",
            edgecolor="white",
            linewidth=2,
        )
        ax.barh(
            y + h / 2,
            d["shap_share"] * 100,
            height=h,
            color=SHAP_C,
            label="SHAP share",
            edgecolor="white",
            linewidth=2,
        )
        for yi, (_, r) in zip(y, d.iterrows(), strict=True):
            if r["shap_direction"] == "lowers":
                ax.text(
                    r["shap_share"] * 100 + 1.5,
                    yi + h / 2,
                    f"SHAP: lowers {r['outcome']}",
                    va="center",
                    fontsize=8,
                    color=MUTED,
                )
        ax.set_yticks(y, [t.replace("_", " ") for t in d["type"]])
        ax.invert_yaxis()
        ax.set_title(prob, loc="left", fontsize=10, color=INK)
        ax.set_xlim(0, 100)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel("% of the treatment effect", color=MUTED, fontsize=9)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(
        "SHAP drivers against the true effect of making each type 10% rarer",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    df = rows()
    df.to_csv(ROOT / "results" / "shap_vs_truth.csv", index=False)
    draw(df, ROOT / "assets" / "shap_vs_truth.png")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
