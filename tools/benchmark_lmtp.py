"""Cross-check LTMLE against R's lmtp on the same panel.

lmtp (Williams and Díaz) implements TMLE for modified treatment policies,
the method gmethods.LTMLE follows. Both estimate the risk of a complaint by
turn ``tau`` under "never k" policies, with chat end as a competing event.

    Rscript -e 'install.packages("lmtp")'   # once; tested with lmtp 1.5.4
    uv run python tools/benchmark_lmtp.py write results/lmtp 20000 10
    Rscript tools/benchmark_lmtp.R results/lmtp 10
    uv run python tools/benchmark_lmtp.py compare results/lmtp 10

lmtp wants one row per chat. Turn t holds the time-varying covariates, the
handoff, a censoring indicator (always 1: nobody is lost to follow-up
before turn tau), then the complaint and chat-end indicators. After a
complaint or a chat end, later treatments and covariates are missing,
complaints are carried forward as 1 and chat ends as 1.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

L = [
    "queue_length",
    "log_wait",
    "shift_time_left",
    "skill_mismatch",
    "with_specialist",
    "sentiment",
    "log_idle",
    "cum_transfer",
    "cum_escalation",
    "cum_requeue",
    "cum_shift_change",
]
REGIMES = ["natural_course", "never_escalation", "never_requeue", "never_any"]


def write(prefix: str, n: int, tau: int) -> None:
    from gmethods import ChatSimulator, Scenario

    data = ChatSimulator(Scenario.preset("baseline")).simulate(n_chats=n, seed=1)
    p = data.panel[data.panel["turn"] <= tau].copy()
    p["log_wait"] = np.log1p(p["wait"])
    p["log_idle"] = np.log1p(p["idle"])
    base = p.groupby("chat_id").head(1).set_index("chat_id")
    wide = pd.get_dummies(
        base[["task_type", "first_agent_team"]], drop_first=True, dtype=float
    )
    for c in [
        "friction_history",
        "n_products",
        "age_band",
        "income_band",
        "start_hour",
    ]:
        wide[c] = base[c].astype(float)
    wide.columns = [c.replace(" ", "_") for c in wide.columns]
    baseline = list(wide.columns)
    cols = {}
    for t in range(1, tau + 1):
        rows = p[p["turn"] == t].set_index("chat_id").reindex(wide.index)
        for c in L:
            cols[f"{c}_{t}"] = rows[c]
        cols[f"A_{t}"] = rows["handoff"].astype(object)
        cols[f"C_{t}"] = np.where(rows["complaint"].notna(), 1.0, np.nan)
        y, d = rows["complaint"].to_numpy(float), rows["ended"].to_numpy(float)
        if t > 1:
            y_prev, d_prev = cols[f"Y_{t - 1}"], cols[f"D_{t - 1}"]
            y = np.where(y_prev == 1, 1.0, np.where(d_prev == 1, 0.0, y))
            d = np.where(d_prev == 1, 1.0, np.where(y_prev == 1, np.nan, d))
        cols[f"Y_{t}"], cols[f"D_{t}"] = y, d
    wide = pd.concat([wide, pd.DataFrame(cols, index=wide.index)], axis=1)
    wide.reset_index().to_csv(prefix + "_wide.csv", index=False)
    pd.Series(baseline).to_csv(prefix + "_baseline.txt", index=False, header=False)
    data.panel.to_pickle(prefix + "_panel.pkl")


def compare(prefix: str, tau: int) -> None:
    from gmethods import LTMLE, ChatSimulator, Scenario
    from gmethods import interventions as iv

    regimes = [
        iv.natural(),
        iv.never("escalation"),
        iv.never("requeue"),
        iv.never_any(),
    ]
    panel = pd.read_pickle(prefix + "_panel.pkl")
    panel = panel[panel["turn"] <= tau]
    truth = ChatSimulator(Scenario.preset("baseline")).true_risks(
        regimes, n_mc=1_000_000
    )
    rows = {
        "truth": {
            r: truth.curves[r][tau - 1] - truth.curves["natural_course"][tau - 1]
            for r in REGIMES
        }
    }
    for label, kw in [
        ("gmethods_glm", {"treatment_learner": "linear"}),
        ("gmethods_default", {}),
    ]:
        table = LTMLE(**kw).fit(panel).estimate(regimes)
        rows[label] = {r: table.rd(r) for r in REGIMES}
        rows[label + "_se"] = {
            r: (table.ci[r][1] - table.ci[r][0]) / 3.92 for r in REGIMES
        }
    r_out = pd.read_csv(prefix + "_lmtp.csv", index_col="regime")
    rows["lmtp_glm"] = (r_out["risk"] - r_out.loc["natural_course", "risk"]).to_dict()
    rows["lmtp_glm_se"] = r_out["rd_se"].to_dict()
    out = pd.DataFrame(rows).loc[REGIMES] * 1000
    out.to_csv(prefix + "_compare.csv")
    print(out.round(3).to_string())


if __name__ == "__main__":
    cmd, prefix = sys.argv[1], sys.argv[2]
    if cmd == "write":
        write(prefix, int(sys.argv[3]), int(sys.argv[4]))
    else:
        compare(prefix, int(sys.argv[3]))
