"""Benchmark the linear GFormula against pygformula on the same panel.

pygformula (Harvard CAUSALab) fits a parametric g-formula with
statsmodels GLMs. It pins older pandas, so it runs in its own environment:

    uv venv /tmp/pygf && uv pip install --python /tmp/pygf "pygformula==1.1.6"
    uv run python tools/benchmark_pygformula.py write results/bench
    MPLBACKEND=Agg /tmp/pygf/bin/python tools/benchmark_pygformula.py pygf results/bench
    uv run python tools/benchmark_pygformula.py compare results/bench

The five-level handoff is coded as four binary treatments simulated in
sequence with restrictions (a later one is 0 when an earlier one is 1), which
factorises the same multinomial. "Never k" sets the natural draw of k to 0
after it is simulated, the same policy as gmethods.interventions.never.
pygformula's plot module selects the Qt backend on import; with no display,
replace that line in pygformula/plot.py or import fails.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

A = ["a_shift_change", "a_transfer", "a_escalation", "a_requeue"]
L = ["queue", "lwait", "shift_left", "mismatch", "specialist", "sent", "lidle"]
NAMES = [
    "never_shift_change",
    "never_transfer",
    "never_escalation",
    "never_requeue",
    "never_any",
]


def write(prefix: str, n: int = 20_000) -> None:
    from gmethods import ChatSimulator, Scenario

    p = ChatSimulator(Scenario.preset("baseline")).simulate(n_chats=n, seed=1).panel
    df = pd.DataFrame({"chat_id": p.chat_id, "t0": p.turn - 1})
    for c in ["task_type", "first_agent_team"]:
        df[c] = p[c].astype(str)
    for c in [
        "friction_history",
        "n_products",
        "age_band",
        "income_band",
        "start_hour",
    ]:
        df[c] = p[c].astype(float)
    df["queue"] = p.queue_length
    df["lwait"] = np.log1p(p.wait)
    df["shift_left"] = p.shift_time_left
    df["mismatch"] = p.skill_mismatch
    df["specialist"] = p.with_specialist
    df["sent"] = p.sentiment
    df["lidle"] = np.log1p(p.idle)
    for k in ["transfer", "escalation", "requeue", "shift_change"]:
        df["a_" + k] = (p.handoff.astype(str) == k).astype(int)
    df["complaint"], df["ended"] = p.complaint, p.ended
    df.to_csv(prefix + "_panel.csv", index=False)
    p.to_pickle(prefix + "_raw.pkl")


def run_pygformula(prefix: str, n_simul: int = 5_000) -> None:
    from pygformula import ParametricGformula

    df = pd.read_csv(prefix + "_panel.csv")
    dums = pd.get_dummies(
        df[["task_type", "first_agent_team"]], drop_first=True, dtype=float
    )
    df = pd.concat([df.drop(columns=["task_type", "first_agent_team"]), dums], axis=1)
    base = " + ".join(
        [
            *dums.columns,
            "friction_history",
            "n_products",
            "age_band",
            "income_band",
            "start_hour",
        ]
    )
    hist = " + ".join(f"lag1_{a} + lag_cumavg1_{a}" for a in A) + " + t0"
    lags = " + ".join(f"lag1_{c}" for c in L)
    covnames = L + A
    covtypes = [
        "normal",
        "normal",
        "normal",
        "binary",
        "binary",
        "normal",
        "normal",
        "binary",
        "binary",
        "binary",
        "binary",
    ]
    covmodels = []
    for i, c in enumerate(covnames):
        rhs = " + ".join(x for x in [base, hist, lags, " + ".join(covnames[:i])] if x)
        covmodels.append(f"{c} ~ {rhs}")
    rhs = f"{base} + {hist} + {' + '.join(covnames)}"
    restrictions = [
        ["a_transfer", {"a_shift_change": lambda x: x == 0}, 0],
        [
            "a_escalation",
            {"a_shift_change": lambda x: x == 0, "a_transfer": lambda x: x == 0},
            0,
        ],
        [
            "a_requeue",
            {
                "a_shift_change": lambda x: x == 0,
                "a_transfer": lambda x: x == 0,
                "a_escalation": lambda x: x == 0,
            },
            0,
        ],
    ]

    def zero(new_df, pool, int_var, time_name, t):
        new_df[int_var] = 0

    ints = {f"Intervention{i}_{a}": [zero] for i, a in enumerate(A, start=1)}
    ints.update({f"Intervention5_{a}": [zero] for a in A})
    start = time.time()
    g = ParametricGformula(
        obs_data=df,
        id="chat_id",
        time_name="t0",
        time_points=int(df.t0.max()) + 1,
        covnames=covnames,
        covtypes=covtypes,
        covmodels=covmodels,
        basecovs=[
            *dums.columns,
            "friction_history",
            "n_products",
            "age_band",
            "income_band",
            "start_hour",
        ],
        outcome_name="complaint",
        ymodel=f"complaint ~ {rhs}",
        outcome_type="survival",
        compevent_name="ended",
        compevent_model=f"ended ~ {rhs}",
        compevent_cens=False,
        restrictions=restrictions,
        int_descript=NAMES,
        nsamples=0,
        seed=1,
        n_simul=n_simul,
        **ints,
    )
    g.fit()
    res = {
        nm: float(np.asarray(r)[-1])
        for nm, r in zip(["natural_course", *NAMES], g.g_results, strict=True)
    }
    res["seconds"] = round(time.time() - start)
    with open(prefix + "_pygformula.json", "w") as f:
        json.dump(res, f)


def compare(prefix: str) -> None:
    from gmethods import GFormula
    from gmethods import interventions as iv

    panel = pd.read_pickle(prefix + "_raw.pkl")
    regimes = iv.standard_set()
    py = json.load(open(prefix + "_pygformula.json"))
    out = pd.DataFrame(
        {
            "truth": pd.read_csv("results/truth_baseline.csv", index_col=0)["rd"],
            "gmethods_linear": GFormula(learner="linear")
            .fit(panel)
            .estimate(regimes, n_mc=50_000)
            .to_frame()["rd"],
            "pygformula": pd.Series(
                {k: py[k] - py["natural_course"] for k in ["natural_course", *NAMES]}
            ),
        }
    ).loc[[r.name for r in regimes]]
    print((out * 1000).round(2).to_string())


if __name__ == "__main__":
    {"write": write, "pygf": run_pygformula, "compare": compare}[sys.argv[1]](
        sys.argv[2]
    )
