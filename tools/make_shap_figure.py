"""Figure: SHAP driver shares against the true marginal split, three problems.

For each simulator and each of ``--reps`` datasets of 50,000 units: a
boosted model of the outcome on the unit summary read with SHAP. The figure
shows the mean SHAP share (whiskers: 10th to 90th percentile across
datasets) against the true share of the effect of making each treatment
type 10% rarer. Resumable: datasets already in the results file are kept.

    uv run python tools/make_shap_figure.py --reps 10   # writes assets/ and results/
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


PROBLEMS = [
    ("Friction and churn", "friction", "churn"),
    ("Campaigns and take-up", "campaign", "take-up"),
    ("Chat handoffs and complaints", "chat", "complaints"),
]
RAW = ROOT / "results" / "shap_robustness.csv"


def one(key: str, seed: int) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if key == "chat":
        data = ChatSimulator(Scenario.preset("baseline")).simulate(
            n_chats=50_000, seed=seed
        )
        return ShapDrivers.for_chats().fit(data.chats).drivers(), HANDOFF_TYPES
    sim = (
        FrictionSimulator(FrictionScenario.preset("baseline"))
        if key == "friction"
        else CampaignSimulator(CampaignScenario.preset("baseline"))
    )
    data = sim.simulate(n=50_000, seed=seed)
    return ShapDrivers.for_simulator(sim).fit(data.customers).drivers(), sim.LEVELS


def collect(reps: int) -> pd.DataFrame:
    old = pd.read_csv(RAW) if RAW.exists() else pd.DataFrame(columns=["key", "rep"])
    for _, key, _ in PROBLEMS:
        done = set(old.loc[old["key"] == key, "rep"])
        for rep in range(reps):
            if rep in done:
                continue
            shap, _ = one(key, seed=1 + rep)
            frame = shap.reset_index(names="type")
            frame.insert(0, "rep", rep)
            frame.insert(0, "key", key)
            frame.to_csv(RAW, mode="a", header=not RAW.exists(), index=False)
            print(f"{key} rep {rep}", flush=True)
    return pd.read_csv(RAW)


def rows(raw: pd.DataFrame) -> pd.DataFrame:
    truth_files = {
        "friction": "truth_friction.csv",
        "campaign": "truth_campaign.csv",
        "chat": "truth_baseline.csv",
    }
    levels_of = {
        "friction": FrictionSimulator.LEVELS,
        "campaign": CampaignSimulator.LEVELS,
        "chat": HANDOFF_TYPES,
    }
    out = []
    for title, key, outcome in PROBLEMS:
        risks = pd.read_csv(ROOT / "results" / truth_files[key], index_col="regime")[
            "risk"
        ]
        true = marginal_from_risks(risks.to_dict(), factor=0.9, names=levels_of[key])
        sub = raw[raw["key"] == key]
        top = sub.loc[sub.groupby("rep")["share"].idxmax(), "type"].value_counts()
        for lv in levels_of[key][1:]:
            s = sub[sub["type"] == lv]
            out.append(
                {
                    "problem": title,
                    "outcome": outcome,
                    "type": lv,
                    "true_share": true.loc[lv, "share"],
                    "shap_share": s["share"].mean(),
                    "shap_p10": s["share"].quantile(0.1),
                    "shap_p90": s["share"].quantile(0.9),
                    "share_of_reps_lowers": float(
                        (s["mean_shap_when_present"] < 0).mean()
                    ),
                    "shap_direction": "lowers"
                    if (s["mean_shap_when_present"] < 0).mean() > 0.5
                    else "raises",
                    "reps_ranked_first": int(top.get(lv, 0)),
                    "reps": s["rep"].nunique(),
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
            label="SHAP share (whisker: 10th to 90th percentile)",
            edgecolor="white",
            linewidth=2,
        )
        mid = d["shap_share"] * 100
        ax.errorbar(
            mid,
            y + h / 2,
            xerr=[mid - d["shap_p10"] * 100, d["shap_p90"] * 100 - mid],
            fmt="none",
            ecolor=INK,
            elinewidth=1,
            capsize=2,
        )
        for yi, (_, r) in zip(y, d.iterrows(), strict=True):
            if r["shap_direction"] == "lowers":
                ax.text(
                    r["shap_p90"] * 100 + 1.5,
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
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(
        "SHAP drivers against the true effect of making each type 10% rarer"
        f" ({int(df['reps'].iloc[0])} datasets per problem)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    df = rows(collect(args.reps))
    df.to_csv(ROOT / "results" / "shap_vs_truth.csv", index=False)
    draw(df, ROOT / "assets" / "shap_vs_truth.png")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
