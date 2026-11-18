"""Article figures other than the SHAP figure (see make_shap_figure.py).

    uv run python tools/make_figures.py            # all
    uv run python tools/make_figures.py walkthrough diagram

Writes PNGs to assets/. The estimate-vs-truth figure reads the recovery
results in results/ (tools/recovery.py and tools/recovery_weekly.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def estimate_vs_truth(path: Path) -> None:
    """Mean estimate as % of the truth, by method, for each "never k" policy."""
    import recovery
    import recovery_weekly

    chat = recovery.summarise(
        sorted((ROOT / "results").glob("recovery_baseline_n50000_*.csv")),
        ROOT / "results" / "truth_baseline.csv",
    ).xs(50_000, level="n_chats")
    panels = [
        ("Friction and churn", recovery_weekly.summarise("friction")),
        ("Campaigns and take-up", recovery_weekly.summarise("campaign")),
        ("Chat handoffs and complaints", chat),
    ]
    methods = [
        ("count_window", "count regression"),
        ("pooled_hazard", "pooled hazard"),
        ("ltmle", "LTMLE"),
    ]
    lo, hi = -60, 220
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, (title, s) in zip(axes, panels, strict=True):
        regimes = [
            r
            for r in s.index.get_level_values("regime").unique()
            if r.startswith("never_") and r != "never_any"
        ]
        y = np.arange(len(regimes))
        for j, (key, label) in enumerate(methods):
            rel = (s.loc[key, "rd_mean"] / s.loc[key, "rd_true"]).reindex(regimes) * 100
            yy = y + (j - 1) * 0.22
            shown = rel.clip(lo, hi)
            ax.scatter(
                shown,
                yy,
                s=40,
                color=SERIES[j],
                edgecolor="white",
                linewidth=1,
                label=label,
                zorder=3,
            )
            for yi, v in zip(yy, rel, strict=True):
                if v < lo:
                    ax.annotate(
                        f"wrong sign ({v:.0f}%)",
                        (lo, yi),
                        xytext=(6, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=7.5,
                        color=MUTED,
                    )
                elif v > hi:
                    ax.annotate(
                        f"{v:.0f}% off scale",
                        (hi, yi),
                        xytext=(-6, 0),
                        textcoords="offset points",
                        va="center",
                        ha="right",
                        fontsize=7.5,
                        color=MUTED,
                    )
        ax.axvline(100, color=INK, lw=1.5)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, [r.removeprefix("never_").replace("_", " ") for r in regimes])
        ax.invert_yaxis()
        ax.set_xlim(lo - 5, hi)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("mean estimate, % of the true effect", fontsize=9, color=MUTED)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=8.5, frameon=False)
    fig.suptitle(
        "Removing each treatment type: estimates across repeated datasets "
        "(100% = the truth)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def walkthrough(path: Path) -> None:
    """One simulated chat: wait and sentiment by turn, with its handoffs."""
    from gmethods import ChatSimulator, Scenario

    data = ChatSimulator(Scenario.preset("baseline")).simulate(n_chats=20_000, seed=1)
    p = data.panel
    # A complained chat with a transfer, then an escalation on a later turn
    # before the complaint, choosing the largest wait after the transfer.
    best, best_jump = None, -1.0
    for cid, g in p[p["chat_id"].isin(p.loc[p["complaint"] == 1, "chat_id"])].groupby(
        "chat_id"
    ):
        g = g.set_index("turn")
        h = g["handoff"].astype(str)
        tr = h.index[h == "transfer"]
        es = h.index[h == "escalation"]
        last = g.index.max()
        if len(tr) and len(es) and tr[0] < es[0] < last and tr[0] + 1 in g.index:
            jump = g.loc[tr[0] + 1, "wait"] - g.loc[tr[0], "wait"]
            if jump > best_jump:
                best, best_jump = cid, jump
    c = p[p["chat_id"] == best].set_index("turn")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )
    ax1.bar(c.index, c["wait"], width=0.5, color=SERIES[0])
    ax1.set_ylabel("wait going in\n(minutes)", fontsize=9, color=MUTED)
    ax2.plot(c.index, c["sentiment"], color=SERIES[1], lw=2, marker="o", ms=6)
    ax2.set_ylabel("frustration\n(sentiment)", fontsize=9, color=MUTED)
    ax2.set_xlabel("turn", fontsize=9, color=MUTED)
    top = c["sentiment"].max()
    for t, row in c.iterrows():
        if row["handoff"] != "none":
            for ax in (ax1, ax2):
                ax.axvline(t + 0.45, color=MUTED, lw=0.8, ls="--")
            ax2.annotate(
                str(row["handoff"]).replace("_", " "),
                (t + 0.45, top),
                xytext=(3, 0),
                textcoords="offset points",
                fontsize=8,
                color=INK,
                va="top",
            )
        if row["complaint"] == 1:
            ax2.annotate(
                "complaint",
                (t, row["sentiment"]),
                xytext=(0, -16),
                textcoords="offset points",
                fontsize=8,
                color=INK,
                ha="center",
            )
    for ax in (ax1, ax2):
        _style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.grid(axis="x", visible=False)
    ax1.set_title(
        "One chat: each handoff adds wait at the next turn, frustration builds, "
        "the next handoff follows",
        loc="left",
        fontsize=10,
        color=INK,
    )
    ax2.set_xticks(c.index)
    ax2.set_xlim(c.index.min() - 0.6, c.index.max() + 0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def diagram(path: Path) -> None:
    """Causal diagram: treatment and state feeding each other over two steps."""
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    nodes = {
        "W": (0.8, 2.1, "customer\nat start"),
        "L1": (2.8, 3.2, "state at t\n(engagement, wait)"),
        "A1": (4.8, 3.2, "treatment at t\n(friction, handoff)"),
        "L2": (4.8, 1.0, "state at t+1"),
        "A2": (7.0, 1.0, "treatment at t+1"),
        "Y": (9.0, 2.1, "outcome\n(churn, complaint)"),
    }
    boxes = {}
    for key, (x, y, label) in nodes.items():
        boxes[key] = ax.add_patch(
            FancyBboxPatch(
                (x - 0.85, y - 0.42),
                1.7,
                0.84,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                fc="white",
                ec=INK,
                lw=1.2,
                zorder=2,
            )
        )
        ax.text(
            x, y, label, ha="center", va="center", fontsize=8.5, color=INK, zorder=3
        )

    def arrow(a, b, color=MUTED, lw=1.2, rad=0.0):
        # Clip each arrow at the two boxes' edges so it never crosses a label.
        (x1, y1, _), (x2, y2, _) = nodes[a], nodes[b]
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                color=color,
                lw=lw,
                patchA=boxes[a],
                patchB=boxes[b],
                shrinkA=2,
                shrinkB=2,
                connectionstyle=f"arc3,rad={rad}",
                zorder=1,
            )
        )

    for a, b in [
        ("W", "L1"),
        ("L1", "A1"),
        ("L2", "A2"),
        ("W", "L2"),
        ("L1", "L2"),
        ("A1", "Y"),
        ("A2", "Y"),
        ("L2", "Y"),
    ]:
        arrow(a, b)
    arrow("A1", "L2", color=SERIES[1], lw=2.2)
    ax.text(
        5.05,
        2.1,
        "the treatment moves\nthe next state",
        fontsize=8.5,
        color=SERIES[1],
        va="center",
    )
    ax.set_title(
        "Adjusting for the state at t+1 blocks part of the effect of treatment "
        "at t; leaving it out confounds treatment at t+1",
        loc="left",
        fontsize=9.5,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def weights(path: Path) -> None:
    """Effective sample share of each removal policy's LTMLE weights."""
    from gmethods import LTMLE, ChatSimulator, Scenario
    from gmethods import interventions as iv
    from gmethods.estimators import ltmle_weight_summary

    data = ChatSimulator(Scenario.preset("baseline")).simulate(n_chats=50_000, seed=1)
    model = LTMLE().fit(data.panel)
    regimes = [*iv.standard_set()[1:-1], iv.scale("shift_change", factor=0.5)]
    w = ltmle_weight_summary(model, regimes).sort_values("ess_share")
    w.to_csv(ROOT / "results" / "weights_chat.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    y = np.arange(len(w))
    ax.barh(y, w["ess_share"] * 100, color=SERIES[0], height=0.55)
    for yi, (_, r) in zip(y, w.iterrows(), strict=True):
        ax.text(
            r["ess_share"] * 100 + 1.5,
            yi,
            f"max weight {r['max']:,.0f}",
            va="center",
            fontsize=8,
            color=MUTED,
        )
    labels = [
        n.replace("never_", "never ")
        .replace("scale_", "halve ")
        .replace("_0.5", "")
        .replace("_", " ")
        for n in w.index
    ]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("effective sample, % of chat-turns followed", fontsize=9, color=MUTED)
    ax.set_title(
        "Rule-forced handoffs leave almost no data on the alternative",
        loc="left",
        fontsize=10,
        color=INK,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


FIGURES = {
    "estimate_vs_truth": estimate_vs_truth,
    "walkthrough": walkthrough,
    "diagram": diagram,
    "weights": weights,
}


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    for name in wanted:
        FIGURES[name](ROOT / "assets" / f"fig_{name}.png")
        print("wrote", name)
