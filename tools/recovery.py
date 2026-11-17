"""Recovery study: how close does each estimator get to the true effects?

Simulates repeated datasets from one scenario, fits every estimator and
records its risk difference against the natural course for each regime.
The ground truth is computed once from the structural equations.

Resumable: replicates already in the output file are skipped, and the run
stops cleanly after ``--max-minutes`` so it can be re-invoked.

    uv run python tools/recovery.py --preset baseline --reps 20
    uv run python tools/recovery.py --summarise results/recovery_baseline.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from gmethods import (
    IPW,
    ChatSimulator,
    CountRegression,
    GFormula,
    PooledHazard,
    Scenario,
)
from gmethods import interventions as iv

FAST = {"max_iter": 100, "learning_rate": 0.1}


def truth_table(preset: str, path: Path, n_mc: int) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, index_col="regime")
    sim = ChatSimulator(Scenario.preset(preset))
    frame = sim.true_risks(iv.standard_set(), n_mc=n_mc).to_frame()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return frame


def one_rep(preset: str, rep: int, n_chats: int, n_mc: int) -> list[dict]:
    regimes = iv.standard_set()
    sim = ChatSimulator(Scenario.preset(preset))
    data = sim.simulate(n_chats=n_chats, seed=1000 + rep)
    tables = [
        GFormula(learner_params=FAST).fit(data.panel).estimate(regimes, n_mc=n_mc),
        IPW(learner_params=FAST).fit(data.panel).estimate(regimes),
        IPW(learner_params=FAST, truncate=0.99).fit(data.panel).estimate(regimes),
        CountRegression("all").fit(data.chats).estimate(regimes),
        CountRegression("before_complaint_or_all").fit(data.chats).estimate(regimes),
        CountRegression("before_complaint_or_all", adjust_sentiment=True)
        .fit(data.chats)
        .estimate(regimes),
        PooledHazard().fit(data.panel).estimate(regimes),
    ]
    labels = [
        "gformula",
        "ipw",
        "ipw_trunc99",
        "count_all",
        "count_window",
        "count_window_sentiment",
        "pooled_hazard",
    ]
    rows = []
    for label, table in zip(labels, tables, strict=True):
        for regime in table.regimes:
            rows.append(
                {
                    "preset": preset,
                    "rep": rep,
                    "estimator": label,
                    "regime": regime,
                    "risk": table.risks[regime],
                    "rd": table.rd(regime),
                }
            )
    return rows


def summarise(path: Path, truth_path: Path) -> pd.DataFrame:
    res = pd.read_csv(path)
    truth = pd.read_csv(truth_path, index_col="regime")["rd"]
    res["rd_true"] = res["regime"].map(truth)
    res["error"] = res["rd"] - res["rd_true"]
    g = res[res["regime"] != "natural_course"].groupby(["estimator", "regime"])
    out = g.agg(
        rd_true=("rd_true", "first"),
        rd_mean=("rd", "mean"),
        bias=("error", "mean"),
        sd=("rd", "std"),
        reps=("rep", "nunique"),
    )
    out["rmse"] = g["error"].apply(lambda e: (e**2).mean() ** 0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="baseline")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n-chats", type=int, default=50_000)
    ap.add_argument("--n-mc", type=int, default=10_000)
    ap.add_argument("--truth-mc", type=int, default=1_000_000)
    ap.add_argument("--max-minutes", type=float, default=9.0)
    ap.add_argument("--summarise", type=Path)
    args = ap.parse_args()
    out = Path(f"results/recovery_{args.preset}.csv")
    truth_path = Path(f"results/truth_{args.preset}.csv")
    if args.summarise:
        print(summarise(args.summarise, truth_path).round(5).to_string())
        return
    start = time.time()
    truth_table(args.preset, truth_path, args.truth_mc)
    done = set(pd.read_csv(out)["rep"]) if out.exists() else set()
    for rep in range(args.reps):
        if rep in done:
            continue
        if (time.time() - start) / 60 > args.max_minutes:
            print("time budget reached; re-run to continue")
            return
        rows = pd.DataFrame(one_rep(args.preset, rep, args.n_chats, args.n_mc))
        rows.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"rep {rep} done at {(time.time() - start) / 60:.1f} min", flush=True)
    print("all replicates done")


if __name__ == "__main__":
    main()
