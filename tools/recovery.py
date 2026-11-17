"""Recovery study: how close does each estimator get to the true effects?

Simulates repeated datasets from one scenario, fits each estimator and
records its risk difference against the natural course for every regime.
The ground truth is computed once from the structural equations.

Resumable: replicates already in the output file are skipped, and a run
stops cleanly after ``--max-minutes`` so it can be re-invoked. Workers can
split the replicates with ``--start`` and ``--stop``; each writes its own
file and ``--summarise`` reads them all.

    uv run python tools/recovery.py --preset baseline --reps 20
    uv run python tools/recovery.py --preset baseline --summarise
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from gmethods import (
    IPW,
    LTMLE,
    ChatSimulator,
    CountRegression,
    GFormula,
    PooledHazard,
    Scenario,
)
from gmethods import interventions as iv

FAST = {"max_iter": 100, "learning_rate": 0.1}
REGIMES = iv.standard_set() + iv.scale_set()[1:]
ALL = [
    "gformula",
    "ice",
    "ltmle",
    "ipw",
    "count_all",
    "count_window",
    "count_window_sentiment",
    "pooled_hazard",
]


def truth_table(preset: str, path: Path, n_mc: int) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, index_col="regime")
    sim = ChatSimulator(Scenario.preset(preset))
    frame = sim.true_risks(REGIMES, n_mc=n_mc).to_frame()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return frame


def fit_all(data, estimators: list[str], n_mc: int) -> dict:
    out = {}
    if "gformula" in estimators:
        gf = GFormula(learner_params=FAST).fit(data.panel)
        out["gformula"] = gf.estimate(REGIMES, n_mc=n_mc)
    if "ice" in estimators or "ltmle" in estimators:
        m = LTMLE(learner="linear", treatment_learner="hgb").fit(data.panel)
        if "ice" in estimators:
            m.target = False
            out["ice"] = m.estimate(REGIMES)
        if "ltmle" in estimators:
            m.target = True
            out["ltmle"] = m.estimate(REGIMES)
    if "ipw" in estimators:
        out["ipw"] = IPW(learner_params=FAST).fit(data.panel).estimate(REGIMES)
    naive = {
        "count_all": lambda: CountRegression("all"),
        "count_window": lambda: CountRegression("before_complaint_or_all"),
        "count_window_sentiment": lambda: CountRegression(
            "before_complaint_or_all", adjust_sentiment=True
        ),
    }
    for name, make in naive.items():
        if name in estimators:
            out[name] = make().fit(data.chats).estimate(REGIMES)
    if "pooled_hazard" in estimators:
        out["pooled_hazard"] = PooledHazard().fit(data.panel).estimate(REGIMES)
    return out


def one_rep(args, rep: int) -> list[dict]:
    sim = ChatSimulator(Scenario.preset(args.preset))
    data = sim.simulate(n_chats=args.n_chats, seed=1000 + rep)
    rows = []
    for label, table in fit_all(data, args.estimators, args.n_mc).items():
        for regime in table.regimes:
            low, high = table.ci.get(regime, (float("nan"), float("nan")))
            rows.append(
                {
                    "preset": args.preset,
                    "n_chats": args.n_chats,
                    "rep": rep,
                    "estimator": label,
                    "regime": regime,
                    "risk": table.risks[regime],
                    "rd": table.rd(regime),
                    "rd_low": low,
                    "rd_high": high,
                }
            )
    return rows


def summarise(files: list[Path], truth_path: Path) -> pd.DataFrame:
    res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    truth = pd.read_csv(truth_path, index_col="regime")["rd"]
    res["rd_true"] = res["regime"].map(truth)
    res["error"] = res["rd"] - res["rd_true"]
    covered = (res["rd_low"] <= res["rd_true"]) & (res["rd_true"] <= res["rd_high"])
    res["covered"] = covered.astype(float).where(res["rd_low"].notna())
    res = res[res["regime"] != "natural_course"]
    g = res.groupby(["n_chats", "estimator", "regime"])
    out = g.agg(
        rd_true=("rd_true", "first"),
        rd_mean=("rd", "mean"),
        bias=("error", "mean"),
        sd=("rd", "std"),
        reps=("rep", "nunique"),
        coverage=("covered", "mean"),
    )
    out["rmse"] = g["error"].apply(lambda e: (e**2).mean() ** 0.5)
    out["rel_bias"] = out["rd_mean"] / out["rd_true"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="baseline")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--n-chats", type=int, default=50_000)
    ap.add_argument("--n-mc", type=int, default=10_000)
    ap.add_argument("--truth-mc", type=int, default=1_000_000)
    ap.add_argument("--estimators", nargs="+", default=ALL)
    ap.add_argument("--max-minutes", type=float, default=9.0)
    ap.add_argument("--tag", default="all", help="Suffix for the output file")
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    truth_path = Path(f"results/truth_{args.preset}.csv")
    if args.summarise:
        files = sorted(Path("results").glob(f"recovery_{args.preset}_*.csv"))
        print(summarise(files, truth_path).round(5).to_string())
        return
    stop = args.stop if args.stop is not None else args.reps
    name = f"recovery_{args.preset}_n{args.n_chats}_{args.tag}_{args.start}-{stop}.csv"
    out = Path("results") / name
    start = time.time()
    truth_table(args.preset, truth_path, args.truth_mc)
    done = set(pd.read_csv(out)["rep"]) if out.exists() else set()
    for rep in range(args.start, stop):
        if rep in done:
            continue
        if (time.time() - start) / 60 > args.max_minutes:
            print("time budget reached; re-run to continue")
            return
        rows = pd.DataFrame(one_rep(args, rep))
        rows.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"rep {rep} done at {(time.time() - start) / 60:.1f} min", flush=True)
    print("all replicates done")


if __name__ == "__main__":
    main()
