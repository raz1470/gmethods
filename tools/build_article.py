"""Build article/index.html from the results/ folder.

Every number and chart in the article comes from here, so re-running the
experiments and this script regenerates the whole page.

    uv run python tools/build_article.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg import paired_hbar_panel, swatch_legend

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CSS = (ROOT / "tools" / "article.css").read_text()
REPO = "https://github.com/raz1470/gmethods"
TITLE = "How to Fix Biased Attribution in SHAP With G-Methods"
TAGLINE = (
    "Simulations with a known answer test SHAP driver analysis against LTMLE on "
    "churn, campaign and support data where treatments repeat over time."
)

BLUE, ORANGE, AQUA, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#111827", "#6b7280"

NAMES = {
    "payment_failure": "payment failure",
    "card_decline": "card decline",
    "app_error": "app error",
    "reverification": "re-verification",
    "email_info": "info email",
    "email_offer": "offer email",
    "push": "push",
    "transfer": "transfer",
    "escalation": "escalation",
    "requeue": "requeue",
    "shift_change": "shift change",
}


def link(text: str, url: str) -> str:
    return f'<a href="{url}" target="_blank" rel="noopener">{text}</a>'


def pct(v: float) -> str:
    return f"{100 * v:.0f}%"


# ======================================================================= data
shap = pd.read_csv(RES / "shap_vs_truth.csv")
N_SHAP = int(shap["reps"].iloc[0])
S = shap.set_index("type")


def share(t: str, col: str = "shap_share") -> str:
    return pct(S.loc[t, col])


# =================================================================== figure 1
fig1_panels = []
for prob, outcome in [
    ("Friction and churn", "churn"),
    ("Campaigns and take-up", "take-up"),
    ("Chat handoffs and complaints", "complaints"),
]:
    d = shap[shap["problem"] == prob].sort_values("true_share", ascending=False)
    notes = [
        f"says it lowers {outcome}" if r.shap_direction == "lowers" else ""
        for r in d.itertuples()
    ]
    fig1_panels.append(
        paired_hbar_panel(
            [NAMES[t] for t in d["type"]],
            list(d["true_share"] * 100),
            list(d["shap_share"] * 100),
            list(d["shap_p10"] * 100),
            list(d["shap_p90"] * 100),
            notes,
            title=prob,
            xmax=100,
            w=680,
        )
    )
LEG1 = swatch_legend(
    [
        ("True share of the effect", BLUE),
        ("SHAP share (whisker: 10th to 90th percentile)", ORANGE),
    ]
)


# ====================================================== data for sections 2-6
import glob  # noqa: E402
import json  # noqa: E402

import recovery  # noqa: E402
import recovery_weekly  # noqa: E402
from svg import (  # noqa: E402
    bar_panel,
    dot_legend,
    dot_panel,
    hbar_panel,
    interval_panel,
    legend,
)

MECH = json.loads((RES / "article_mechanisms.json").read_text())
MF, MC, MH = MECH["friction"], MECH["campaign"], MECH["chat"]

chat_sum = recovery.summarise(
    sorted(RES.glob("recovery_baseline_n50000_*.csv")), RES / "truth_baseline.csv"
).xs(50_000, level="n_chats")
fric_sum = recovery_weekly.summarise("friction")
camp_sum = recovery_weekly.summarise("campaign")


def rel(s: pd.DataFrame, est: str, regime: str) -> float:
    return float(
        s.loc[(est, regime), "rd_mean"] / s.loc[(est, regime), "rd_true"] * 100
    )


def cov(s: pd.DataFrame, regime: str) -> float:
    return float(s.loc[("ltmle", regime), "coverage"] * 100)


def rng(vals) -> str:
    lo, hi = min(vals), max(vals)
    return f"{lo:.0f}% to {hi:.0f}%"


FRIC = ["payment_failure", "card_decline", "app_error", "reverification"]
CAMP = ["email_info", "email_offer", "push"]
CHAT = ["transfer", "escalation", "requeue", "shift_change"]
N_REP_W = int(fric_sum["reps"].max())
N_REP_C = int(chat_sum.loc["ltmle", "reps"].max())

# ------------------------------------------------------------------ figure 2
q = [f"Q{i}" for i in range(1, 6)]
fig2 = "".join(
    [
        bar_panel(
            q,
            [100 * v for v in MF["friction_by_quintile"]],
            "Weeks with friction",
            "% of customer-weeks, by engagement quintile",
            color=ORANGE,
            fmt=lambda v: f"{v:.1f}%",
            xlabel="Engagement quintile (Q5 = most engaged)",
        ),
        bar_panel(
            q,
            [100 * v for v in MF["churn_by_quintile"]],
            "Churn",
            "% of customer-weeks, by engagement quintile",
            color=BLUE,
            fmt=lambda v: f"{v:.2f}%",
            xlabel="Engagement quintile (Q5 = most engaged)",
        ),
    ]
)
fq, cq = MF["friction_by_quintile"], MF["churn_by_quintile"]
eng_drop = MF["engagement_change_sd"]

# ------------------------------------------------------------------ table 3
one = pd.read_csv(RES / "weekly_sims_one_dataset.csv")
one = one[one["simulator"] == "campaign"].set_index("regime")
rows3 = [
    ("Truth", "truth"),
    ("Count regression, every send in the log", "count_all"),
    ("Count regression, sends before take-up", "count_window"),
    ("… plus the engagement score", "count_window_proxy"),
    ("Pooled hazard (week by week)", "pooled"),
    ("LTMLE", "ltmle"),
]


def cell3(v: float, truth: float, col: str) -> str:
    cls = ""
    if col != "truth":
        cls = "bad" if v * truth < 0 else ("good" if abs(v / truth - 1) < 0.2 else "")
    return f"<td class='num {cls}'>{v:+.1f}".replace("-", "−") + "</td>"


body3 = "".join(
    "<tr><td>"
    + lab
    + "</td>"
    + "".join(
        cell3(one.loc[f"never_{k}", col], one.loc[f"never_{k}", "truth"], col)
        for k in CAMP
    )
    + "</tr>"
    for lab, col in rows3
)
TABLE3 = (
    "<div class='dtable-wrap'><table class='dtable'><caption>Change in take-ups per 1,000 "
    "customers if each campaign stopped. One simulated dataset of 50,000 customers. Red: "
    "wrong sign. Green: within 20% of the truth.</caption><thead><tr><th>Method</th>"
    + "".join(f"<th class='num'>{NAMES[k]}</th>" for k in CAMP)
    + "</tr></thead><tbody>"
    + body3
    + "</tbody></table></div>"
)

# ------------------------------------------------------------------ figure 4
marg = pd.concat(
    pd.read_csv(f)
    for f in sorted(glob.glob(str(RES / "recovery_baseline_n50000_marginal_*.csv")))
)
truth_chat = pd.read_csv(RES / "truth_baseline.csv").set_index("regime")
marg = marg[marg["regime"].str.startswith("scale_")]
marg["truth"] = marg["regime"].map(truth_chat["rd"])
marg["covers"] = (marg["rd_low"] <= marg["truth"]) & (marg["truth"] <= marg["rd_high"])
mg = marg.groupby("regime").agg(
    est=("rd", "mean"),
    lo=("rd_low", "mean"),
    hi=("rd_high", "mean"),
    truth=("truth", "first"),
    covers=("covers", "mean"),
)
N_MARG = int(marg["rep"].nunique())
order4 = ["escalation", "requeue", "shift_change", "transfer"]
reg4 = [f"scale_{k}_0.9" for k in order4]
fig4 = interval_panel(
    [NAMES[k] for k in order4] + ["all four"],
    [-1000 * mg.loc[r, "truth"] for r in reg4 + ["scale_any_0.9"]],
    [-1000 * mg.loc[r, "est"] for r in reg4 + ["scale_any_0.9"]],
    [-1000 * mg.loc[r, "hi"] for r in reg4 + ["scale_any_0.9"]],
    [-1000 * mg.loc[r, "lo"] for r in reg4 + ["scale_any_0.9"]],
    xlabel="Complaints avoided per 1,000 chats",
    xmax=1.1,
)
LEG4 = legend(
    [("Truth", INK, ""), ("LTMLE: mean estimate and mean 95% interval", AQUA, "")]
)
tc = {k: -1000 * mg.loc[f"scale_{k}_0.9", "truth"] for k in order4}
t_all = -1000 * mg.loc["scale_any_0.9", "truth"]
t_sum = sum(tc.values())
af = -truth_chat.loc["never_any", "rd"] / truth_chat.loc["natural_course", "risk"]
af_est = (
    -chat_sum.loc[("ltmle", "never_any"), "rd_mean"]
    / truth_chat.loc["natural_course", "risk"]
)
cov4 = [100 * mg.loc[r, "covers"] for r in reg4]
rel4 = [100 * mg.loc[r, "est"] / mg.loc[r, "truth"] for r in reg4]

# ------------------------------------------------------------------ figure 5
methods5 = [
    ("count_window", "Count regression", ORANGE),
    ("pooled_hazard", "Pooled hazard", GREY),
    ("ltmle", "LTMLE", AQUA),
]
fig5 = []
for title, s, keys in [
    ("Friction and churn", fric_sum, FRIC),
    ("Campaigns and take-up", camp_sum, CAMP),
    ("Chat handoffs and complaints", chat_sum, CHAT),
]:
    fig5.append(
        dot_panel(
            [NAMES[k] for k in keys],
            [
                (lab, col, [rel(s, est, f"never_{k}") for k in keys])
                for est, lab, col in methods5
            ],
            title,
        )
    )
LEG5 = dot_legend([(lab, col) for _, lab, col in methods5])
lt_f = [rel(fric_sum, "ltmle", f"never_{k}") for k in FRIC]
lt_c = [rel(camp_sum, "ltmle", f"never_{k}") for k in CAMP]
lt_h = [rel(chat_sum, "ltmle", f"never_{k}") for k in CHAT]
ph_f = [rel(fric_sum, "pooled_hazard", f"never_{k}") for k in FRIC]
ph_c = [rel(camp_sum, "pooled_hazard", f"never_{k}") for k in CAMP]
ph_h = [
    rel(chat_sum, "pooled_hazard", f"never_{k}")
    for k in ["escalation", "requeue", "shift_change"]
]
cw_h = {k: rel(chat_sum, "count_window", f"never_{k}") for k in CHAT}
gf_h = {k: rel(chat_sum, "gformula", f"never_{k}") for k in CHAT}
ipw_h = {k: rel(chat_sum, "ipw", f"never_{k}") for k in CHAT}
cov_w = [cov(fric_sum, f"never_{k}") for k in FRIC] + [
    cov(camp_sum, f"never_{k}") for k in CAMP
]
cov_h = [cov(chat_sum, f"never_{k}") for k in CHAT]

lm = pd.read_csv(RES / "benchmark_lmtp.csv", index_col=0)
pg = pd.read_csv(RES / "benchmark_pygformula.csv").set_index("regime")


def num(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}".replace("-", "−")


TABLE5A = (
    "<div class='dtable-wrap'><table class='dtable'><caption>LTMLE in gmethods against R's lmtp "
    "on the same 20,000 chats: risk of a complaint by turn 10, change per 1,000 chats, logistic "
    "models in both. Standard errors in brackets.</caption><thead><tr><th>Policy</th>"
    "<th class='num'>Truth</th><th class='num'>gmethods</th><th class='num'>lmtp</th></tr>"
    "</thead><tbody>"
    + "".join(
        f"<tr><td>{r.replace('_', ' ')}</td><td class='num'>{num(lm.loc[r, 'truth'])}</td>"
        f"<td class='num'>{num(lm.loc[r, 'gmethods_glm'])} ({num(lm.loc[r, 'gmethods_glm_se'])})</td>"
        f"<td class='num'>{num(lm.loc[r, 'lmtp_glm'])} ({num(lm.loc[r, 'lmtp_glm_se'])})</td></tr>"
        for r in ["never_escalation", "never_requeue", "never_any"]
    )
    + "</tbody></table></div>"
)
TABLE5B = (
    "<div class='dtable-wrap'><table class='dtable'><caption>The parametric g-formula in gmethods "
    "against pygformula on the same chats, logistic models in both: change in complaints per "
    "1,000 chats.</caption><thead><tr><th>Policy</th><th class='num'>Truth</th>"
    "<th class='num'>gmethods</th><th class='num'>pygformula</th></tr></thead><tbody>"
    + "".join(
        f"<tr><td>{r.replace('_', ' ')}</td><td class='num'>{num(pg.loc[r, 'truth'])}</td>"
        f"<td class='num'>{num(pg.loc[r, 'gmethods_linear'])}</td>"
        f"<td class='num'>{num(pg.loc[r, 'pygformula'])}</td></tr>"
        for r in [
            "never_escalation",
            "never_requeue",
            "never_shift_change",
            "never_transfer",
            "never_any",
        ]
    )
    + "</tbody></table></div>"
)

# ------------------------------------------------------------------ figure 6
wt = pd.read_csv(RES / "weights_chat.csv", index_col=0).sort_values(
    "ess_share", ascending=False
)
lab6 = {
    "never_requeue": "never requeue",
    "never_escalation": "never escalation",
    "never_transfer": "never transfer",
    "scale_shift_change_0.5": "halve shift change",
    "never_shift_change": "never shift change",
}
fig6 = hbar_panel(
    [lab6[r] for r in wt.index],
    list(wt["ess_share"] * 100),
    [f"max weight {v:,.0f}" for v in wt["max"]],
    color=BLUE,
    xlabel="Effective sample, % of chat-turns",
    fmt=lambda v: f"{v:.1f}%" if v < 1 else f"{v:.0f}%",
)
ps = pd.read_csv(RES / "positivity_shift.csv")
ps["rel"] = ps["rd"] / ps["rd_true"] * 100
ps_rel = ps[ps["regime"] == "never_shift_change"].groupby("p_stay")["rel"].mean()
reverif_scale = rel(fric_sum, "ltmle", "scale_reverification_0.9")

# ==================================================================== prose
REST = f"""
<!-- 2 -->
<section id="state">
  <h2>2. The treatment follows the state</h2>

  <p>Product friction is a good example of a treatment nobody chooses on purpose: a payment that fails, a card that's declined, an app error, a request to verify identity again. Who meets it isn't random. Customers who use the product more make more payments and open the app more often, so they hit more friction. They are also the customers least likely to leave. Friction then makes them use the product less, and using it less is what leads to churn.</p>

  <p>The simulation:</p>
  <ul class="pts">
    <li>50,000 customers over 26 weeks. Each week a customer can meet one of four frictions: payment failure, card decline, app error or re-verification.</li>
    <li>An unobserved engagement level drives transactions and app sessions, which drive the chance of friction.</li>
    <li>Friction lowers next week's engagement. Churn rises as engagement falls, and also with that week's friction.</li>
    <li>Re-verification fires on a schedule, which matters in section 6.</li>
  </ul>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">The most engaged customers meet the most friction and churn the least</div>
      <div class="fig-sub">One simulated dataset of 50,000 customers, 26 weeks each</div>
    </div>
    <div class="fig-body"><div class="panels">{fig2}</div></div>
    <p class="fig-cap">Customer-weeks grouped by the customer's engagement that week, from least (Q1) to most (Q5) engaged. Left: share of weeks with any friction. Right: share of weeks in which the customer churned.</p>
  </div>

  <p>The most engaged fifth of customers meets friction in {100 * fq[4]:.1f}% of weeks, against {100 * fq[0]:.1f}% for the least engaged, and churns in {100 * cq[4]:.2f}% of weeks against {100 * cq[0]:.2f}%. Across customers, more friction goes with less churn. A payment failure then lowers next week's engagement by {-eng_drop["payment_failure"]:.2f} standard deviations, and a card decline by {-eng_drop["card_decline"]:.2f}. So the state that hides the effect is also the path the effect travels along.</p>

  <p>That creates a trap for the obvious fix, which is to control for engagement. The SHAP model already includes the customer's average app sessions over the whole period, weeks after the friction included, so part of what friction did is credited to that feature instead. A model fitted week by week (a pooled hazard: churn this week on this week's state and treatment) removes the confounding, because each treatment is compared within a state. But it holds engagement fixed, so it misses the part of the effect that runs through engagement. Averaged over {N_REP_W} datasets, the pooled hazard recovers {rng(ph_f)} of the true effect of each friction. LTMLE recovers {rng(lt_f)}, because it lets engagement respond to the friction, as step 2 in section 1 described.</p>
</section>

<!-- 3 -->
<section id="exposure">
  <h2>3. Exposure time and the event log</h2>

  <p>The table behind most driver analyses is built from an event log: one row per customer, with a count of each treatment. Two things that have nothing to do with the treatment's effect change those counts. The first is how long the customer was there to be treated. The second is whether the log keeps recording treatments after the outcome.</p>

  <p>The simulation:</p>
  <ul class="pts">
    <li>50,000 customers over 26 weeks. Each week the CRM can send an information email, an offer email or a push notification.</li>
    <li>Offers go to customers whose engagement score has dropped. The score is built from app sessions and email opens, which the campaigns themselves move.</li>
    <li>The outcome is taking up a product (opening a savings pot), with churn as a competing event. Customers who take up stop receiving campaigns.</li>
    <li>After a take-up, a confirmation email is logged as an information email.</li>
  </ul>

  <p>Customers who take up are observed for {MC["weeks_takers"]:.1f} weeks on average, against {MC["weeks_others"]:.1f} for those who don't. Before taking up they receive campaigns at almost the same rate ({MC["per_week_takers"]:.2f} a week against {MC["per_week_others"]:.2f}), but over the whole period they receive fewer ({MC["sends_takers"]:.1f} against {MC["sends_others"]:.1f}). A high count marks a customer who hasn't taken up yet, so every campaign looks as if it prevents take-up. On top of that, {100 * MC["info_after_takeup"]:.0f}% of information emails are confirmations sent after the take-up. The same thing happens with friction: {100 * MF["churner_declines_after"]:.0f}% of the card declines logged for churners come in or after the week they churn, as they move their money out. That is why SHAP ranks card declines first in figure 1.</p>

  {TABLE3}

  <p>Counting every send in the log gets the sign wrong for all three campaigns. Counting only sends before take-up removes the confirmation emails but not the exposure time, and the signs stay wrong. Adding the engagement score makes them larger, because offers go to customers whose score fell, so the model reads a low score plus an offer as the profile of a non-taker. The pooled hazard fixes the exposure problem, since it compares customers week by week while they are still at risk, but it holds the score fixed and recovers {rng(ph_c)} of each effect across {N_REP_W} datasets. LTMLE recovers {rng(lt_c)}.</p>
</section>

<!-- 4 -->
<section id="chain">
  <h2>4. Earlier treatments cause later ones</h2>

  <p>In a support chat, one handoff often leads to another. A transfer puts the customer back in a queue, the wait frustrates them, and a frustrated customer is more likely to be escalated. When the outcome is a complaint, the question is how much of it to attribute to each handoff type, and the escalation at the end of the chain is the easiest one to blame.</p>

  <p>The simulation:</p>
  <ul class="pts">
    <li>50,000 chats of up to 40 turns. At each turn the chat can be transferred, escalated, requeued or handed over at the end of the agent's shift.</li>
    <li>Transfers follow misrouting, escalations follow frustration, requeues follow customer idle time, and shift changes follow the agent's rota.</li>
    <li>Every handoff adds waiting time and frustration. Frustration is observed with noise, as a sentiment score.</li>
    <li>The outcome is a complaint during the chat. The chat ending without one is a competing event. As in real logs, some escalations are recorded after the complaint.</li>
  </ul>

  <p>After a transfer the next turn's wait averages {MH["next_wait"]["transfer"]:.1f} minutes, against {MH["next_wait"]["none"]:.1f} after a turn with no handoff, and after a requeue {MH["next_wait"]["requeue"]:.1f} minutes. Frustration rises {MH["sentiment_change"]["transfer"] / MH["sentiment_change"]["none"]:.0f} times as much in the turn after a transfer as after a turn without one. The chance of an escalation per turn goes from {100 * MH["escalation_no_transfer"]:.1f}% to {100 * MH["escalation_after_transfer"]:.1f}% once a chat has been transferred.</p>

  <p>SHAP gives escalations {share("escalation")} of the credit against a true {share("escalation", "true_share")}, and requeues {share("requeue")} against {share("requeue", "true_share")}. Escalations collect credit from three places: the transfers that led to them, the frustration that predicts both escalation and complaint, and the {100 * MH["escalations_after_complaint"]:.0f}% of escalations logged after the complaint. Requeues lose it because they work through waiting and frustration, and the model's average sentiment feature absorbs that path.</p>

  <h3>Splitting the credit</h3>

  <p>Attribution needs two numbers. The first is how much of the outcome the treatments cause at all. If no chat were ever handed off, complaints would fall by {100 * af:.0f}% (LTMLE's estimate averages {100 * af_est:.0f}%). The other {100 - 100 * af:.0f}% come from things such as the task and the customer's history, which aren't levers, so they aren't split. The second number is how to share the {100 * af:.0f}% across types. We use the effect of making each type 10% rarer, from how things run today:</p>

  <ul class="pts">
    <li><strong>It adds up.</strong> With a small cut, a chat rarely loses two handoffs at once, so the four effects sum to {t_sum:.2f} complaints per 1,000 chats against {t_all:.2f} for cutting all four together.</li>
    <li><strong>Knock-on effects go to the cause.</strong> Fewer transfers means fewer escalations later, and the transfer gets that credit.</li>
    <li><strong>The data supports it.</strong> A 10% cut only needs chats like today's, so the weights from step 1 stay close to one.</li>
    <li><strong>It answers a decision.</strong> Which handoff type to reduce first.</li>
  </ul>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Complaints avoided by making each handoff type 10% rarer</div>
      <div class="fig-sub">LTMLE across {N_MARG} datasets of 50,000 chats, against the truth</div>
    </div>
    <div class="fig-body">{LEG4}{fig4}</div>
    <p class="fig-cap">Black tick: the true effect. Green: LTMLE's estimate averaged over {N_MARG} datasets, with the average 95% interval. True shares of the effect: escalation {share("escalation", "true_share")}, requeue {share("requeue", "true_share")}, shift change {share("shift_change", "true_share")}, transfer {share("transfer", "true_share")}.</p>
  </div>

  <p>LTMLE's estimates average {rng(rel4)} of the truth for every type, and its intervals contain the truth in {rng(cov4)} of datasets. Escalations and requeues are clearly separated from zero. Shift changes and transfers aren't: their effects are about a tenth the size, and 50,000 chats can't tell them apart from zero.</p>

  <p>Why not split with Shapley values over the handoff types? That needs the complaint rate under every combination of types removed entirely: with four types, 16 policies, such as "only escalations and requeues exist". Most of those worlds are far from anything in the data (section 6 shows how few chats look like "never transfer"), and Shapley values average over every order of removing types, when in the data the order is fixed by time. The package has a time-ordered split for readers who want one, but its standard errors were about 70% of the effects, against 11% for the 10% cut.</p>
</section>

<!-- 5 -->
<section id="recovery">
  <h2>5. Does LTMLE recover the truth?</h2>

  <p>The sections so far used single datasets to show mechanisms. This one repeats each simulation and compares the methods on the policies they are most often used for: removing one treatment type entirely.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">Mean estimate as a share of the true effect of removing each type</div>
      <div class="fig-sub">{N_REP_W} datasets each for friction and campaigns, {N_REP_C} for chats</div>
    </div>
    <div class="fig-body">{LEG5}<div class="stack">{"".join(fig5)}</div></div>
    <p class="fig-cap">100% is the truth. Count regression: a regression of the outcome on counts of each treatment before the outcome, plus customer features. Pooled hazard: the outcome this week (or turn) on this week's state and treatment. Estimates beyond the axis are written next to the edge.</p>
  </div>

  <p>The count regression gets the sign wrong for all seven weekly treatments, for the exposure-time reason in section 3. On chats it gets the signs right but not the sizes: {cw_h["escalation"]:.0f}% of the true effect for escalations, {cw_h["requeue"]:.0f}% for requeues and {cw_h["transfer"]:.0f}% for transfers. The pooled hazard is biased towards zero for every type except transfers, because it holds the state fixed. LTMLE lands at {rng(lt_f + lt_c)} on the weekly problems and {rng([lt_h[1], lt_h[2]])} on escalations and requeues, with 95% intervals that contain the truth in {rng(cov_w + cov_h[1:3])} of datasets. Its two weak spots are transfers ({lt_h[0]:.0f}%) and shift changes ({lt_h[3]:.0f}%), which section 6 explains.</p>

  <p>The two older g-methods, run on the same chats, agree on the large effects. The parametric g-formula, which models every confounder and simulates chats forward, lands at {gf_h["escalation"]:.0f}% for escalations and {gf_h["requeue"]:.0f}% for requeues. With 200,000 chats its error doesn't shrink, because it comes from modelling every confounder rather than from sample size. Inverse probability weighting lands at {ipw_h["escalation"]:.0f}% and {ipw_h["requeue"]:.0f}%, but at {ipw_h["transfer"]:.0f}% for transfers, where the weights get large.</p>

  <h3>How does this compare with existing packages?</h3>

  <p>Two packages implement these methods: R's {link("lmtp", "https://github.com/nt-williams/lmtp")} (Williams and Díaz), which implements the same LTMLE, and Python's {link("pygformula", "https://github.com/CausalInference/pygformula")} (Harvard CAUSALab), which implements the parametric g-formula. We ran each on the same simulated chats as our own version, with the same kind of model.</p>

  {TABLE5A}

  <p>LTMLE and lmtp agree within one standard error, and their standard errors are close. Both miss the requeue effect by the same amount on this dataset, which points to sampling noise rather than to either implementation.</p>

  {TABLE5B}

  <p>The two g-formula implementations are within about one complaint per 1,000 chats of each other, and both overstate the effect of transfers by the same margin. That shared error comes from the logistic models, not from the code.</p>
</section>

<!-- 6 -->
<section id="rules">
  <h2>6. When a rule forces the treatment</h2>

  <p>Some treatments are set by a rule rather than by anyone's judgement. A chat is handed over when the agent's shift ends. A customer is asked to verify their identity again when the schedule says so. "Never hand over at shift end" is a reasonable question, but the data contains almost no chats where an agent stayed on past the end of their shift, so there is very little to learn from.</p>

  <p>LTMLE shows this in its weights. Step 1 gives each row the ratio of its probability under the policy to its observed probability, multiplied over the weeks or turns so far. If the policy needs histories that almost never happen, a handful of rows get enormous weights and the rest count for little. The effective sample (the number of equally weighted rows that would give the same precision) shows how much of the data is doing the work.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">How much of the data each chat policy can use</div>
      <div class="fig-sub">LTMLE weights before truncation, one dataset of 50,000 chats</div>
    </div>
    <div class="fig-body">{fig6}</div>
    <p class="fig-cap">Effective sample: (Σw)² / Σw² as a share of chat-turns, using the weights before truncation. Max weight: the largest single weight.</p>
  </div>

  <p>"Never requeue" and "never escalation" keep almost all of the data. "Never transfer" keeps {100 * wt.loc["never_transfer", "ess_share"]:.0f}%: misrouted chats are transferred within a few turns, so long spells with the wrong team are rare. "Never shift change" keeps {100 * wt.loc["never_shift_change", "ess_share"]:.1f}%, with one chat-turn weighted {wt.loc["never_shift_change", "max"]:,.0f} times.</p>

  <p>Truncating the largest weights doesn't fix it. At the default (the 99.9th percentile), "never shift change" lands at {rng(ps_rel.tolist())} of the truth whether agents stay past their shift in 1%, 10% or 30% of cases. Without truncation the estimate becomes unstable. The friction simulation has the same problem with scheduled re-verification: cutting it by 10% lands at {reverif_scale:.0f}% of the truth.</p>

  <p>So check the weights before reporting a policy (<code>ltmle_weight_summary</code> in the package does this). If the effective sample is small, ask a question the data can answer about the rule itself, such as how long agents may run past their shift, rather than about the treatment disappearing.</p>
</section>

<!-- FAQ -->
<section id="faq">
  <h2>Frequently asked questions</h2>

  <details class="faq"><summary>Why not add the state as a feature in the SHAP model?</summary>
  <p>Because the state is both a confounder and part of the effect. The SHAP models here already include the average engagement signal. In the campaign simulation, adding the engagement score to the count regression made the wrong-signed effects larger (section 3). In the chats, adding sentiment changed almost nothing.</p></details>

  <details class="faq"><summary>What if I fit the model on the weekly panel instead of one row per customer?</summary>
  <p>That fixes exposure time and the post-outcome logs, and it is the pooled hazard in section 5. It still holds the state fixed, so it misses the part of each effect that runs through the state: across the three problems it recovered {rng(ph_f + ph_c + ph_h)} of the true effects, transfers aside.</p></details>

  <details class="faq"><summary>Can double machine learning or causal forests do this?</summary>
  <p>They estimate the effect of a single treatment given the covariates at that time. Applied to one week, that is the effect of this week's treatment with everything else held fixed, which leaves out what the treatment does to later weeks. That is the same gap the pooled hazard has. The g-methods exist to follow the effect through those later weeks.</p></details>

  <details class="faq"><summary>How much data do I need?</summary>
  <p>The simulations use 50,000 customers or chats. With a complaint rate of {100 * MH["complaint_rate"]:.0f}%, effects of about 0.36 complaints per 1,000 chats had standard errors of about 11% of their size. Effects a tenth that size could not be told apart from zero. Rare outcomes, rare treatments and policies far from today all need more.</p></details>

  <details class="faq"><summary>What if the state that drives the treatment isn't recorded?</summary>
  <p>Then no method on this page removes the confounding: LTMLE assumes that, given the recorded history, who gets treated is as good as random. The simulators have presets with an unmeasured confounder, which can be used to see how far the estimates move. The engagement in these simulations is never observed directly, only through transactions, app sessions and sentiment, which are noisy, and that was enough here.</p></details>

  <details class="faq"><summary>Is SHAP wrong?</summary>
  <p>No. It does what it is designed to do: explain the model's predictions. A churn model with SHAP is a good way to see which customers are at risk and why the model thinks so. The problem is reading those explanations as the effect of changing a treatment.</p></details>
</section>

<!-- conclusion -->
<section id="together">
  <h2>Conclusion</h2>

  <p>When treatments repeat over time and react to the customer's state, a model of the outcome explained with SHAP answers a different question from the one attribution asks:</p>

  <ul class="pts">
    <li><strong>The treatment follows the state.</strong> The customers who meet the most friction churn the least. Controlling for the state removes that confounding but also removes the part of the effect that runs through the state.</li>
    <li><strong>Exposure time and the log.</strong> Customers who leave or convert early have fewer weeks to be treated, and logs keep recording treatments after the outcome. Both can flip the sign of an effect.</li>
    <li><strong>Treatments cause treatments.</strong> The last treatment in a chain collects the credit for the ones that led to it.</li>
    <li><strong>The fix.</strong> Ask a policy question (what if this happened 10% less?) and answer it with a g-method. LTMLE recovered every well-supported effect in three simulations to within about 15%.</li>
    <li><strong>Check the weights.</strong> For treatments a rule forces, the data can't say what happens without them. Ask about the rule instead.</li>
  </ul>

  <p>The simulations use one data-generating process per problem, a known set of confounders and samples of 50,000. They don't show how often these problems occur in a given dataset. The questions to ask are the same, though: does the treatment depend on a state that changes over time, does the treatment change that state, how long was each customer observed, and does the log record anything after the outcome?</p>

  <p>The package, {link("gmethods", REPO)}, has the three simulators, LTMLE, the g-formula and IPW, the attribution functions and the script that builds every chart and number in this article.</p>

  <h3>References</h3>
  <ul class="refs">
    <li>Díaz, I., Williams, N., Hoffman, K. L. and Schenck, E. J. (2021). Nonparametric causal effects based on longitudinal modified treatment policies. <em>Journal of the American Statistical Association</em>, 118(542), 846–857.</li>
    <li>Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. <em>Advances in Neural Information Processing Systems</em>, 30.</li>
    <li>Robins, J. (1986). A new approach to causal inference in mortality studies with a sustained exposure period. <em>Mathematical Modelling</em>, 7(9–12), 1393–1512.</li>
    <li>van der Laan, M. J. and Gruber, S. (2012). Targeted minimum loss based estimation of causal effects of multiple time point interventions. <em>The International Journal of Biostatistics</em>, 8(1).</li>
    <li>Williams, N. T. and Díaz, I. (2023). lmtp: an R package for estimating the causal effects of modified treatment policies. <em>Observational Studies</em>, 9(2), 103–122.</li>
  </ul>
</section>
"""


LUNDBERG = link("Lundberg and Lee 2017", "https://arxiv.org/abs/1705.07874")
ROBINS = link("Robins 1986", "https://doi.org/10.1016/0270-0255(86)90088-6")
VDL = link(
    "van der Laan and Gruber 2012",
    "https://doi.org/10.1515/1557-4679.1370",
)
DIAZ = link("Díaz et al. 2021", "https://doi.org/10.1080/01621459.2021.1955691")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>
{CSS}
.flow {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; margin: 1.5rem 0; }}
@media (max-width: 599px) {{ .flow {{ grid-template-columns: 1fr; }} }}
.step {{ border: 1px solid var(--border); border-radius: 8px; padding: .9rem 1rem; background: var(--bg); }}
.step .n {{ font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }}
.step .t {{ font-weight: 700; margin: .15rem 0 .35rem; }}
.step p {{ font-size: .86rem; margin: 0; line-height: 1.5; }}
.stack > div {{ margin-bottom: .9rem; }}
.draft {{ border: 1px dashed var(--light); border-radius: 8px; padding: 1rem 1.2rem; color: var(--muted); font-size: .9rem; }}
</style>
</head>
<body>

<header class="hdr">
  <h1>{TITLE}</h1>
  <p class="tagline">{TAGLINE}</p>
  <p class="byline">Ryan O'Sullivan · Code on <a href="{REPO}" style="color:#cbd5e1">GitHub</a></p>
</header>

<nav class="snav">
  <a href="#measure" class="active">1 · SHAP vs attribution</a>
  <a href="#state">2 · Treatment follows state</a>
  <a href="#exposure">3 · Exposure and logging</a>
  <a href="#chain">4 · Treatments cause treatments</a>
  <a href="#recovery">5 · Does it work?</a>
  <a href="#rules">6 · Rule-forced treatments</a>
  <a href="#faq">Questions</a>
  <a href="#together">Conclusion</a>
</nav>

<main>

<p>Many of the things a business does to a customer happen more than once: payment failures and app errors, marketing emails and push notifications, transfers and escalations in a support chat. A common question is which of them drive an outcome such as churn, product take-up or complaints, and by how much. A common answer is to fit a gradient-boosted model of the outcome on counts of each treatment, plus customer features, and read the drivers off SHAP values ({LUNDBERG}). When treatments repeat over time and react to the customer's state, that answer has three problems:</p>

<ul class="pts">
  <li><strong>The treatment follows the state.</strong> Customers get a treatment because of how they behave (heavy users meet more friction, disengaged customers get more offers), and the treatment then changes that behaviour. A model of one row per customer can't separate the two.</li>
  <li><strong>Counts depend on how long a customer was there.</strong> Customers who churn or take up early have fewer weeks in which to be treated, and raw logs record some treatments after the outcome. Both can make a harmful treatment look protective.</li>
  <li><strong>Earlier treatments cause later ones.</strong> A transfer leads to a long wait, and the wait leads to an escalation. SHAP gives the escalation credit for some of the transfer's effect.</li>
</ul>

<p>This article runs one simulation per problem, each with a known answer, and compares SHAP with LTMLE, a g-method built for treatments that repeat over time. It then checks LTMLE against two existing packages, and shows where it struggles: treatments that a business rule forces.</p>

<!-- 1 -->
<section id="measure">
  <h2>1. What SHAP measures and what attribution needs</h2>

  <p>Let's start with what SHAP returns. A SHAP value splits one prediction of the model across its features, so that the pieces add up to the prediction. Averaged over customers, the absolute SHAP value of each treatment count is the usual measure of a driver's importance. It is faithful to the model, and the model is faithful to the data: it says which counts predict churn. It doesn't say what would happen to churn if a count changed.</p>

  <p>The question a team needs answered is about a change it could make: if payment failures were 10% rarer, how many fewer customers would churn? This is the effect of a policy, a rule for how often a treatment happens. Running the same 10% cut for each treatment type and dividing each effect by their total gives each type's share:</p>

  <div class="eqn">share<sub>k</sub> = effect of 10% fewer k / Σ<sub>j</sub> effect of 10% fewer j</div>

  <p>The shares split the part of churn the treatments cause, not all of churn: customers also leave for reasons that have nothing to do with friction. Section 4 explains why we split this way rather than with Shapley values.</p>

  <p>To check SHAP against these shares we need data where they are known. Each simulator below generates customers week by week (or chats turn by turn) from a set of equations, so the true effect of a policy can be computed by rerunning the same customers, with the same random draws, under the policy. The SHAP model is the one most teams would build: gradient-boosted trees on one row per customer, with treatment counts from the raw event log, customer features and the average of the observed engagement signal.</p>

  <div class="fig">
    <div class="fig-hdr">
      <div class="fig-title">SHAP's share of the credit against the true share of the effect</div>
      <div class="fig-sub">Three simulated problems, {N_SHAP} datasets of 50,000 customers or chats each</div>
    </div>
    <div class="fig-body">{LEG1}<div class="stack">{"".join(fig1_panels)}</div></div>
    <p class="fig-cap">Blue: each treatment type's true share of the effect of making every type 10% rarer. Orange: its share of the mean absolute SHAP value, averaged over {N_SHAP} datasets, with a whisker from the 10th to the 90th percentile across datasets. "Says it lowers" marks types whose average SHAP value, among customers who had that treatment, is negative. Every type raises the outcome in the simulation.</p>
  </div>

  <p>SHAP's ranking is wrong in all three problems, and the same way in every dataset. It ranks card declines first for churn ({share("card_decline")} of the credit against a true {share("card_decline", "true_share")}) and push notifications first for take-up ({share("push")} against {share("push", "true_share")}). It gives requeues {share("requeue")} of the credit for complaints when they cause {share("requeue", "true_share")} of the effect. For churn and take-up it also gets the direction wrong: it says three of the four frictions lower churn, and that offer emails and pushes lower take-up. All five raise them. The narrow whiskers matter: more data would not fix this. Sections 2 to 4 show why it happens, one problem at a time.</p>

  <p>The g-methods ({ROBINS}) answer the policy question directly. We use LTMLE (longitudinal targeted maximum likelihood estimation; {VDL}), in the version for policies that change how often a treatment happens rather than switching it on or off ({DIAZ}). It works on a panel with one row per customer per week, up to the week of their outcome, and takes three steps:</p>

  <div class="flow">
    <div class="step"><div class="n">Step 1</div><div class="t">Model who gets which treatment</div><p>For each week, the probability of each treatment given the customer's history. The policy changes these probabilities (10% fewer payment failures), and the ratio of the policy's probability to the observed one gives each row a weight.</p></div>
    <div class="step"><div class="n">Step 2</div><div class="t">Work backwards from the last week</div><p>Regress the outcome on the history, predict it with the treatment drawn from the policy, and pass that prediction back as the outcome for the week before. Correct each step with the weighted residuals from step 1.</p></div>
    <div class="step"><div class="n">Step 3</div><div class="t">Average and compare</div><p>The average prediction at week 1 is the risk under the policy. Its difference from the risk with no policy is the effect, with a standard error from the same correction.</p></div>
  </div>

  <p>Step 2 is where the feedback is handled. Each week's model conditions on the customer's state that week, so the treatment is compared between customers in the same state. The prediction it passes back already includes how that week's treatment changed the next week's state. A single regression has to choose between adjusting for the state and keeping the effect that runs through it; working one week at a time gets both.</p>

  <p>The correction in step 2 makes LTMLE doubly robust: if either the outcome models or the treatment models are right, the estimate is unbiased. Both are fitted with cross-fitting (each customer's predictions come from models trained on other customers), so flexible machine learning models can be used without biasing the effect.<sup>1</sup> The package also has the two older g-methods, the parametric g-formula and inverse probability weighting; section 5 compares all three.</p>

  <p class="fig-cap" style="padding:0"><sup>1</sup> For a policy that multiplies the probability of treatment k by 0.9 and moves the rest to no treatment, the week-t weight is r<sub>t</sub> = p<sup>policy</sup>(A<sub>t</sub> | H<sub>t</sub>) / p(A<sub>t</sub> | H<sub>t</sub>), where H<sub>t</sub> is the history up to week t. The correction at week t weights each row by the product of these ratios from week 1 to week t, so a policy that changes many weeks gives some customers large weights (section 6). R's {link("lmtp", "https://github.com/nt-williams/lmtp")} implements the same estimator.</p>
</section>

{REST}

</main>

<footer>Simulated data throughout. Every chart and number on this page is generated by <code>tools/build_article.py</code> in the gmethods repo from the outputs of the experiment scripts.</footer>

<script>
(function () {{
  const links = document.querySelectorAll('.snav a');
  const secs = [...links].map(a => document.querySelector(a.getAttribute('href')));
  addEventListener('scroll', () => {{
    let k = 0; secs.forEach((s, i) => {{ if (s && s.getBoundingClientRect().top < 120) k = i; }});
    links.forEach((a, i) => a.classList.toggle('active', i === k));
  }}, {{ passive: true }});
}})();
</script>
</body>
</html>
"""
(ROOT / "article").mkdir(exist_ok=True)
(ROOT / "article" / "index.html").write_text(html)
print("wrote article/index.html", len(html) // 1024, "KB")
