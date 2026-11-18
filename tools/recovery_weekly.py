"""Recovery study for the friction and campaign simulators.

LTMLE, the pooled hazard and the "before event, or all" count regression on
repeated datasets, against the truth. Resumable like tools/recovery.py.

    uv run python tools/recovery_weekly.py --sim friction --reps 10
    uv run python tools/recovery_weekly.py --sim campaign --summarise
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from gmethods import (
    LTMLE,
    CampaignScenario,
    CampaignSimulator,
    CountRegression,
    FrictionScenario,
    FrictionSimulator,
    PooledHazard,
)
from gmethods import interventions as iv

RESULTS = Path(__file__).resolve().parent.parent / "results"

SIMS = {
    "friction": (FrictionSimulator, FrictionScenario),
    "campaign": (CampaignSimulator, CampaignScenario),
}


def setup(name: str):
    sim_cls, sc_cls = SIMS[name]
    sim = sim_cls(sc_cls.preset("baseline"))
    regimes = iv.standard_set(sim.LEVELS) + iv.scale_set(sim.LEVELS, factor=0.9)[1:]
    return sim, regimes


def truth(name: str, n_mc: int) -> pd.Series:
    path = RESULTS / f"truth_{name}.csv"
    if path.exists():
        return pd.read_csv(path, index_col="regime")["rd"]
    sim, regimes = setup(name)
    frame = sim.true_risks(regimes, n_mc=n_mc).to_frame()
    frame.to_csv(path)
    return frame["rd"]


def one_rep(name: str, rep: int, n: int) -> list[dict]:
    sim, regimes = setup(name)
    data = sim.simulate(n=n, seed=3000 + rep)
    spec = sim.panel_spec()
    tables = {
        "ltmle": LTMLE(spec).fit(data.panel).estimate(regimes),
        "pooled_hazard": PooledHazard(spec).fit(data.panel).estimate(regimes),
        "count_window": CountRegression.for_simulator(sim, "before_complaint_or_all")
        .fit(data.customers)
        .estimate(regimes),
    }
    rows = []
    for label, table in tables.items():
        for r in table.regimes:
            low, high = table.ci.get(r, (float("nan"), float("nan")))
            rows.append(
                {
                    "sim": name,
                    "rep": rep,
                    "estimator": label,
                    "regime": r,
                    "rd": table.rd(r),
                    "rd_low": low,
                    "rd_high": high,
                }
            )
    return rows


def summarise(name: str) -> pd.DataFrame:
    res = pd.read_csv(RESULTS / f"recovery_{name}.csv")
    res["rd_true"] = res["regime"].map(truth(name, 0))
    res = res[res["regime"] != "natural_course"]
    covered = (res["rd_low"] <= res["rd_true"]) & (res["rd_true"] <= res["rd_high"])
    res["covered"] = covered.astype(float).where(res["rd_low"].notna())
    g = res.groupby(["estimator", "regime"])
    out = g.agg(
        rd_true=("rd_true", "first"),
        rd_mean=("rd", "mean"),
        sd=("rd", "std"),
        coverage=("covered", "mean"),
        reps=("rep", "nunique"),
    )
    out["rel_bias"] = out["rd_mean"] / out["rd_true"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=sorted(SIMS), required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--truth-mc", type=int, default=1_000_000)
    ap.add_argument("--max-minutes", type=float, default=600)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    if args.summarise:
        print(summarise(args.sim).to_string())
        return
    start = time.time()
    truth(args.sim, args.truth_mc)
    out = RESULTS / f"recovery_{args.sim}.csv"
    done = set(pd.read_csv(out)["rep"]) if out.exists() else set()
    for rep in range(args.reps):
        if rep in done:
            continue
        if (time.time() - start) / 60 > args.max_minutes:
            print("time budget reached; re-run to continue")
            return
        pd.DataFrame(one_rep(args.sim, rep, args.n)).to_csv(
            out, mode="a", header=not out.exists(), index=False
        )
        print(
            f"{args.sim} rep {rep} at {(time.time() - start) / 60:.1f} min", flush=True
        )
    print("all replicates done")


if __name__ == "__main__":
    main()
