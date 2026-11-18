"""Descriptive numbers for the article: why each naive analysis goes wrong.

One simulated dataset per problem (50,000 customers or chats, seed 1).
Writes results/article_mechanisms.json, read by tools/build_article.py.

    OMP_NUM_THREADS=1 uv run python tools/article_mechanisms.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gmethods import (
    CampaignScenario,
    CampaignSimulator,
    ChatSimulator,
    FrictionScenario,
    FrictionSimulator,
    Scenario,
)

ROOT = Path(__file__).resolve().parents[1]
N, SEED = 50_000, 1


def friction() -> dict:
    d = FrictionSimulator(FrictionScenario.preset("baseline")).simulate(n=N, seed=SEED)
    p = d.panel.merge(d.latent, on=["customer_id", "week"])
    p["q"] = pd.qcut(p["engagement"], 5, labels=False) + 1
    p["friction"] = p["treatment"] != "none"
    by_q = p.groupby("q").agg(friction=("friction", "mean"), churn=("churned", "mean"))
    p = p.sort_values(["customer_id", "week"])
    p["eng_next"] = p.groupby("customer_id")["engagement"].shift(-1)
    sd = float(p["engagement"].std())
    change = ((p["eng_next"] - p["engagement"]) / sd).groupby(p["treatment"]).mean()
    cu = d.customers
    churners = cu[cu["churned"] == 1]
    decl_all = churners["n_card_decline"].sum()
    decl_before = churners["n_card_decline_before_event"].sum()
    return {
        "friction_by_quintile": by_q["friction"].tolist(),
        "churn_by_quintile": by_q["churn"].tolist(),
        "engagement_change_sd": {k: float(v) for k, v in change.items()},
        "weeks_churners": float(churners["n_weeks"].mean()),
        "weeks_stayers": float(cu.loc[cu["churned"] == 0, "n_weeks"].mean()),
        "churner_declines_after": float(1 - decl_before / decl_all),
        "churn_rate": float(cu["churned"].mean()),
    }


def campaign() -> dict:
    d = CampaignSimulator(CampaignScenario.preset("baseline")).simulate(n=N, seed=SEED)
    cu, p = d.customers, d.panel
    p["q"] = pd.qcut(p["engagement_score"], 5, labels=False) + 1
    offer = p.groupby("q")["treatment"].apply(lambda s: (s == "email_offer").mean())
    sends = ["email_info", "email_offer", "push"]
    cu["sends"] = cu[[f"n_{k}" for k in sends]].sum(axis=1)
    cu["sends_before"] = cu[[f"n_{k}_before_event" for k in sends]].sum(axis=1)
    cu["per_week"] = cu["sends_before"] / cu["n_weeks"]
    g = cu.groupby("took_up")
    info_after = 1 - cu["n_email_info_before_event"].sum() / cu["n_email_info"].sum()
    return {
        "offer_by_quintile": offer.tolist(),
        "weeks_takers": float(g["n_weeks"].mean()[1]),
        "weeks_others": float(g["n_weeks"].mean()[0]),
        "sends_takers": float(g["sends"].mean()[1]),
        "sends_others": float(g["sends"].mean()[0]),
        "per_week_takers": float(g["per_week"].mean()[1]),
        "per_week_others": float(g["per_week"].mean()[0]),
        "info_after_takeup": float(info_after),
        "takeup_rate": float(cu["took_up"].mean()),
    }


def chat() -> dict:
    d = ChatSimulator(Scenario.preset("baseline")).simulate(n_chats=N, seed=SEED)
    p = d.panel.sort_values(["chat_id", "turn"]).copy()
    nxt = p.groupby("chat_id")
    p["next_wait"] = nxt["wait"].shift(-1)
    p["sent_change"] = nxt["sentiment"].shift(-1) - p["sentiment"]
    p["after_transfer"] = p["cum_transfer"] > 0
    esc = p.groupby("after_transfer")["handoff"].apply(
        lambda s: (s == "escalation").mean()
    )
    ch = d.chats
    return {
        "next_wait": {
            k: float(v) for k, v in p.groupby("handoff")["next_wait"].mean().items()
        },
        "sentiment_change": {
            k: float(v) for k, v in p.groupby("handoff")["sent_change"].mean().items()
        },
        "escalation_after_transfer": float(esc[True]),
        "escalation_no_transfer": float(esc[False]),
        "escalations_after_complaint": float(
            1 - ch["n_escalation_before_complaint"].sum() / ch["n_escalation"].sum()
        ),
        "complaint_rate": float(ch["complained"].mean()),
    }


def main() -> None:
    out = {"friction": friction(), "campaign": campaign(), "chat": chat()}
    (ROOT / "results" / "article_mechanisms.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
