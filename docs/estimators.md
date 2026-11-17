# Estimators

Use `LTMLE` for results. The parametric `GFormula` is the teaching
baseline; `IPW` and `ICE` are cross-checks.

All estimators return a `RiskTable`: the risk of a complaint by chat end
under each intervention, with risk differences against the natural course.
They take the same intervention objects as the simulator's truth engine.

## Parametric g-formula (`GFormula`)

```python
gf = GFormula().fit(data.panel)
table = gf.estimate(iv.standard_set(), n_mc=50_000)
```

1. Fit pooled-over-turn models in temporal order: each time-varying
   covariate given baseline, history and the covariates before it in the
   turn; the handoff (multinomial); the complaint hazard; the chat-end hazard
   among chats without a complaint.
2. Resample baseline rows. For each turn, draw the covariates from their
   models, draw the natural handoff and apply the intervention, then predict
   the two hazards.
3. Accumulate cumulative incidence from the hazards.

History features: the previous handoff, cumulative handoffs by type, turns
since the last handoff, the turn index and each covariate's lag.
Continuous covariates are drawn as the fitted mean plus a residual resampled
from training rows with a similar fitted mean.

Learners: `"hgb"` (histogram gradient boosting, default) or `"linear"` (the
classic parametric version). Early stopping is off on purpose: on rare
outcomes it stops after a few rounds and leaves hazards near the pooled mean.

## LTMLE and the ICE g-formula (`LTMLE`, `ICE`)

```python
model = LTMLE().fit(data.panel)          # logistic outcome models, boosted treatment model
table = model.estimate(iv.standard_set() + iv.scale_set()[1:])
table.ci["never_escalation"]             # 95% interval from the influence function
```

The doubly robust estimator, and the recommended one. It is consistent if
either the outcome regressions or the treatment model are right, and its
influence function gives intervals without a bootstrap.

- **ICE** (iterated conditional expectations) works backwards from the last
  turn. At turn t the pseudo-outcome is the complaint at t, or, if the chat
  carries on, the value already computed for its next turn. A regression on
  history and handoff gives Qbar_t(h, a); the value under the policy averages
  it over the handoff the policy gives. No confounder is ever simulated, so
  there is nothing to extrapolate when the policy creates histories the data
  rarely contains (a chat that stays with the wrong team).
- **Targeting** fluctuates each Qbar_t by one parameter on the logit scale,
  fitted with the cumulative density-ratio weights p*(a)/p(a) (the same
  weights as `IPW`). This is LTMLE for modified treatment policies (Díaz,
  Williams, Hoffman and Schenck, 2021), the method in R's `lmtp`. "Never k"
  and "scale k" are such policies.
- **Cross-fitting** by customer: every nuisance prediction for a chat comes
  from models fitted without that customer.
- **Sparse late turns** borrow rows from the turns just after them for the
  regression, and are left untargeted when they hold almost no complaints
  (the fluctuation would otherwise run off to minus infinity and zero out
  the risk at that turn).
- Intervals use the influence function summed within customers.

Defaults: logistic outcome regressions (`learner="linear"`) and a boosted
treatment model (`treatment_learner="hgb"`). Shift changes are near-certain
once a shift runs out, a threshold a logistic treatment model cannot
represent. A boosted or stacked outcome model in ICE compounds small
calibration errors over 40 backward steps; see the README for what was
tried.

## Learners

Every estimator takes `learner=`:

- `"hgb"`: histogram gradient boosting. Early stopping is off: on a 2%
  outcome it stopped after about 20 rounds and left hazards near the
  pooled mean.
- `"linear"`: logistic or linear regression with one-hot categories.
- `"stack"`: a super learner (van der Laan, Polley and Hubbard, 2007) of an
  additive spline model and boosting, blended on out-of-fold predictions.
  `GFormula(learner="stack")` fixes most of the boosted g-formula's
  shrinkage on escalation, requeue and shift change.

`GFormula` also takes a dict to set the learner per model role
(`{"outcome": "linear"}`).

## Inverse probability weighting (`IPW`)

```python
ipw = IPW(truncate=0.99).fit(data.panel)
table = ipw.estimate(iv.standard_set())
```

Each chat-turn is weighted by p*(a)/p(a): the probability of the observed
handoff under the intervention over its probability under the natural
course. Weights multiply over turns. Under "never k" a turn with handoff k
gets weight 0 and the chat stops contributing (clone-censor-weight). Only the
treatment model is needed, so it is a cross-check on the g-formula's
outcome-side models. `weight_summary()` reports mean, max, censored share
and effective sample size.

The regime is the one used everywhere in the package: a removed handoff
becomes no handoff. The textbook weight 1 / P(not k) targets a different
regime, where the freed probability is spread over all other levels.

## Naive baselines

- `CountRegression("all")`: complaint on handoff counts from the raw log.
  Escalations logged after a complaint inflate the escalation effect.
- `CountRegression("before_complaint_or_all")`: counts before the complaint
  for complainers and all handoffs otherwise. Unequal exposure windows.
- `CountRegression(..., adjust_sentiment=True)`: adds mean chat sentiment,
  which handoffs partly cause.
- `PooledHazard`: turn-level hazard on current covariates, with handoffs
  switched off and covariates held at observed values. Misses the effect
  that runs through later frustration, waits and chat length.

## Inference and diagnostics

- `bootstrap(fit_estimate, panel, n_boot=200)`: resamples customers (whole
  clusters), refits every model and returns percentile intervals for the
  risk differences.
- `natural_course_check(gf)`: observed vs simulated cumulative risk and
  covariate means by turn.
- `positivity_table(ipw, by=...)`: mean handoff probabilities and the share
  of turns where "no handoff" has probability below 0.01.

## Using your own panel

`PanelSpec` names the columns: unit id, time, treatment and its levels,
outcome, competing event, baseline covariates and time-varying covariates
(each continuous or binary, optionally on the log1p scale). The defaults
match the simulator's panel.

```python
spec = PanelSpec(
    id_col="chat_id", time_col="turn", treatment="handoff",
    baseline=("task_type", "friction_history"),
    covariates=(CovariateSpec("wait", log1p=True), CovariateSpec("sentiment")),
)
GFormula(spec=spec).fit(panel)
```

Time-varying covariates must be measured before the treatment at the same
time step. Anything caused by that step's treatment belongs to the next
row.
