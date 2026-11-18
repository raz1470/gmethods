"""How positivity for shift changes affects LTMLE.

Shift changes are forced by the rota unless the agent stays past the end of
their shift (probability ``p_stay``). The smaller ``p_stay``, the rarer a
chat that carries on with the same agent after a shift ends, and the less
the data says about "never shift change". This sweep refits LTMLE on
repeated datasets at several values of ``p_stay`` and records the bias and
interval coverage for removing and for halving shift changes.

    uv run python tools/positivity_shift.py --p-stay 0.01 0.1 0.3 --reps 10
    uv run python tools/positivity_shift.py --summarise
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gmethods import LTMLE, ChatSimulator, Scenario
from gmethods import interventions as iv

REGIMES = [iv.natural(), iv.never("shift_change"), iv.scale("shift_change", factor=0.5)]
OUT = Path("results/positivity_shift.csv")
TRUTH = Path("results/positivity_shift_truth.csv")


def truth(p_stay: float) -> pd.Series:
    old = pd.read_csv(TRUTH) if TRUTH.exists() else pd.DataFrame(columns=["p_stay"])
    hit = old[np.isclose(old["p_stay"].astype(float), p_stay)]
    if len(hit):
        return hit.set_index("regime")["rd"]
    sim = ChatSimulator(Scenario.preset("baseline", p_stay=p_stay))
    frame = sim.true_risks(REGIMES, n_mc=1_000_000).to_frame().reset_index()
    frame["p_stay"] = p_stay
    frame.to_csv(TRUTH, mode="a", header=not TRUTH.exists(), index=False)
    return frame.set_index("regime")["rd"]


def run(p_stay: float, reps: int, max_minutes: float, start: float) -> bool:
    done = set()
    if OUT.exists():
        old = pd.read_csv(OUT)
        done = set(old.loc[np.isclose(old["p_stay"], p_stay), "rep"])
    true = truth(p_stay)
    sim = ChatSimulator(Scenario.preset("baseline", p_stay=p_stay))
    for rep in range(reps):
        if rep in done:
            continue
        if (time.time() - start) / 60 > max_minutes:
            return False
        data = sim.simulate(n_chats=50_000, seed=2000 + rep)
        model = LTMLE().fit(data.panel)
        table = model.estimate(REGIMES)
        rows = []
        for r in REGIMES[1:]:
            low, high = table.ci[r.name]
            rows.append(
                {
                    "p_stay": p_stay,
                    "rep": rep,
                    "regime": r.name,
                    "rd": table.rd(r.name),
                    "rd_true": true[r.name],
                    "rd_low": low,
                    "rd_high": high,
                }
            )
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(
            f"p_stay {p_stay} rep {rep} at {(time.time() - start) / 60:.1f} min",
            flush=True,
        )
    return True


def summarise() -> pd.DataFrame:
    res = pd.read_csv(OUT)
    res["covered"] = (res["rd_low"] <= res["rd_true"]) & (
        res["rd_true"] <= res["rd_high"]
    )
    g = res.groupby(["regime", "p_stay"])
    return g.agg(
        rd_true=("rd_true", "first"),
        rd_mean=("rd", "mean"),
        sd=("rd", "std"),
        coverage=("covered", "mean"),
        reps=("rep", "nunique"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-stay", type=float, nargs="+", default=[0.01, 0.1, 0.3])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=600)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    if args.summarise:
        print(summarise().to_string())
        return
    start = time.time()
    for p in args.p_stay:
        if not run(p, args.reps, args.max_minutes, start):
            print("time budget reached; re-run to continue")
            return
    print("all replicates done")


if __name__ == "__main__":
    main()
