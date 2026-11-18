# Friction and campaign simulators

Two weekly customer simulators reuse the same estimators and attribution as
the chat simulator. Both have the feature that makes regression fail: the
treatment reacts to a customer's state and changes it.

```python
from gmethods import FrictionSimulator, FrictionScenario, LTMLE
from gmethods import interventions as iv

sim = FrictionSimulator(FrictionScenario.preset("baseline"))
data = sim.simulate(n=50_000, seed=1)          # .panel, .event_log, .customers
spec = sim.panel_spec()                         # column roles for the estimators
regimes = iv.standard_set(sim.LEVELS)
LTMLE(spec).fit(data.panel).estimate(regimes)
```

## Product friction and churn (`FrictionSimulator`)

Weekly, 26 weeks. Each week a customer may hit one kind of friction:
payment failure, card decline, app error or re-verification. The outcome is
churn.

| Week step | What happens |
| --- | --- |
| Covariates | Engagement (latent) moves with last week's friction and support contacts; transactions and app sessions follow engagement; balance; weeks until scheduled re-verification |
| Friction | Re-verification fires when the schedule runs out (skipped with `p_skip`); otherwise a softmax driven by transactions, low balance, app sessions and platform |
| Churn | Lower engagement, no salary paid in, short tenure and that week's friction raise it |

Why a naive analysis fails here:

- **Exposure follows usage.** Heavy users meet more friction and churn less,
  so friction looks protective.
- **Exposure follows time.** Customers who churn early have fewer weeks in
  which to meet friction. Counting friction per customer turns that into a
  protective "effect" too.
- **Raw logs show friction after churn.** Customers moving money out hit
  card declines after they have decided to leave (`p_post`).

"Never k" means the friction does not happen and the customer's action goes
through.

## Campaigns and product take-up (`CampaignSimulator`)

Weekly, 26 weeks. Each week the CRM may send an information email, an offer
email or a push notification. The outcome is taking up a product (opening a
savings pot); churn is a competing event.

| Week step | What happens |
| --- | --- |
| Covariates | Interest and fatigue (latent) move with last week's campaign; app sessions and email opens follow them; the CRM engagement score is built from those signals; balance; weeks since the last send |
| Campaign | A softmax driven by the engagement score (offers go to customers whose score dropped), app sessions and a soft frequency cap |
| Take-up, then churn | Interest, balance and that week's campaign raise take-up; fatigue lowers it and raises churn |

Why a naive analysis fails here:

- **Takers stop receiving campaigns.** Customers who take up early leave
  the eligible audience, so counts per customer make campaigns look as if
  they prevent take-up.
- **Targeting follows the score.** The CRM sends offers to customers whose
  engagement dropped.
- **Confirmation emails are logged after take-up** as information emails
  (`p_post`).

## Results on one dataset of 50,000 customers

Risk difference per 1,000 customers against the natural course
(`results/weekly_sims_one_dataset.csv`).

**Friction (natural churn 86.7 per 1,000):**

| Remove | Truth | Count, before event or all | Pooled hazard | g-formula | LTMLE |
| --- | --- | --- | --- | --- | --- |
| payment failures | −1.16 | +3.15 | −0.72 | −0.92 | −1.28 |
| card declines | −0.68 | +4.32 | −0.27 | −0.49 | −0.71 |
| app errors | −1.36 | +21.80 | −0.27 | −1.22 | −1.15 |
| re-verification | −5.54 | +30.68 | −2.55 | +2.12 | −6.10 |
| all friction | −8.04 | +72.15 | −3.77 | −0.38 | −9.20 |

**Campaigns (natural take-up 115.1 per 1,000):**

| Remove | Truth | Count, before event or all | Pooled hazard | g-formula | LTMLE |
| --- | --- | --- | --- | --- | --- |
| information emails | −5.72 | +88.63 | −4.52 | −5.05 | −5.29 |
| offer emails | −8.27 | +47.60 | −5.38 | −5.28 | −6.69 |
| push | −1.95 | +73.82 | −2.18 | −2.48 | −1.85 |
| all campaigns | −17.03 | +314.04 | −11.69 | −16.26 | −13.86 |

- The count regression gets the sign wrong in both: it says friction
  prevents churn and campaigns prevent take-up.
- The pooled hazard gets the sign right and most of the size wrong.
- LTMLE is closest overall. Re-verification is scheduled, so it is the
  rule-forced case again (like shift changes): check `ltmle_weight_summary`
  before reporting it.

## Recovery across ten datasets

LTMLE's mean estimate as a share of the truth, and how often its 95%
intervals cover the truth (`tools/recovery_weekly.py`):

| Friction policy | Truth per 1,000 | LTMLE | Coverage |
| --- | --- | --- | --- |
| never payment failure | −1.16 | 104% | 1.0 |
| never card decline | −0.68 | 104% | 1.0 |
| never app error | −1.36 | 99% | 0.9 |
| never re-verification | −5.54 | 102% | 0.9 |
| never any friction | −8.04 | 110% | 0.5 |
| 10% fewer re-verifications | −0.10 | 188% | 0.9 |

| Campaign policy | Truth per 1,000 | LTMLE | Coverage |
| --- | --- | --- | --- |
| never information email | −5.70 | 113% | 1.0 |
| never offer email | −8.28 | 96% | 1.0 |
| never push | −1.95 | 109% | 1.0 |
| never any campaign | −17.02 | 90% | 1.0 |
| 10% fewer of each, together | −1.59 | 100% | 0.9 |

The count regression has the wrong sign in every regime of both
simulators. The pooled hazard gets the sign right but recovers only 45% to
90% of each effect. Re-verification is scheduled, so policies involving it
lean on a few customers who skipped a due check: "never any friction"
intervals cover only half the time, and the 10% reduction is noisy.

## A churn model read with SHAP

The usual driver analysis fits boosted trees to the customer summary
(treatment counts from the raw log, customer covariates, mean engagement
signal) and ranks drivers by mean |SHAP| (`ShapDrivers`, needs `shap`).
Shares of the treatment effect, one dataset of 50,000 customers:

| Friction type | True marginal share | SHAP share | Direction SHAP credits |
| --- | --- | --- | --- |
| payment failure | 28% | 5% | lowers churn |
| card decline | 16% | 38% | raises churn |
| app error | 33% | 27% | lowers churn |
| re-verification | 23% | 30% | lowers churn |

| Campaign | True marginal share | SHAP share | Direction SHAP credits |
| --- | --- | --- | --- |
| information email | 37% | 18% | raises take-up |
| offer email | 51% | 38% | lowers take-up |
| push | 12% | 44% | lowers take-up |

- SHAP's top friction driver is card declines, because churners' raw logs
  carry declines from moving money out after they left. It credits the
  other three frictions with *lowering* churn.
- SHAP's top campaign driver is push, the smallest true effect, and it
  says offers and pushes lower take-up. Customers who take up early stop
  receiving campaigns, so heavy campaign counts mark the customers who
  have not taken up yet.
- LTMLE's own split on this single dataset is close for campaigns (36%,
  49%, 14%) but noisy for friction, where re-verification takes 57%: that
  is the rule-forced policy again. Averaged over ten datasets its type
  effects are within 4% of the truth (table above).
- SHAP is faithful to the model. The model is faithful to the data. Neither
  answers what would happen if a friction type or campaign were reduced.
