# The chat handoff simulator

`ChatSimulator` generates support-chat data where the true effect of every
handoff intervention is known. It exists to check that estimators get the
right answer, and to show why simpler methods do not.

## Time grid

The unit is a chat × turn. A turn is one customer-agent exchange. Chats run
for at most `t_max` turns (default 40). Within each turn, in this order:

1. **Confounders update.** Queue length, then the wait going into the turn
   (including queue time from the previous turn's handoff), the current
   agent's shift time left, frustration and its sentiment proxy, and the
   customer's idle time.
2. **Handoff** A_t ∈ {none, transfer, escalation, requeue, shift_change}.
3. **Complaint** Y_t.
4. **Chat end** D_t (resolved or abandoned), a competing event. A chat cannot
   end at a turn with a handoff, because a new agent has just picked it up.

Wait is set at the start of the turn. It can confound the handoff at turn t
but is never caused by it. A handoff's cost in waiting shows up one turn
later.

## Variables

**Fixed from the start (W).** Drawn once per chat. Every turn's models still
condition on them.

| Variable | Drives |
| --- | --- |
| `task_type` | Transfer (via mismatch), escalation (complexity), complaint |
| `first_agent_team` | Transfer, via mismatch with the task |
| `friction_history` | Escalation, complaint |
| `start_hour`, `weekday` | Queue load and waits |
| `n_products`, `age_band`, `income_band` | Complaint, idle time |
| temperament U (unmeasured) | Frustration, complaint. Off unless `lambda_f`, `lambda_y` > 0 |

Customers can have several chats (`repeat_rate`). Chats of one customer share
the customer traits and U, which is why the bootstrap resamples customers.

**Change turn by turn (L_t).**

| Variable | Changes because |
| --- | --- |
| `queue_length` | Time of day, AR(1) noise |
| `wait` | Base wait from the queue, plus queue time after last turn's handoff |
| `shift_time_left` | Runs down each turn; a new agent starts a fresh shift window |
| `skill_mismatch` | The current agent's team cannot resolve the task |
| `with_specialist` | The chat has been escalated |
| `sentiment` | Frustration plus optional measurement noise |
| `idle` | Customer response time; rises with frustration |
| frustration F_t (latent) | AR(1) plus waits, recent handoffs, task, friction and U |
| resolution progress (latent) | Builds each turn; a handoff resets part of it |

## Structural equations

- **Frustration:** F_t = ρ F_{t−1} + β_w log(1 + wait_t) + γ_{A_{t−1}} +
  β_c complexity + β_f friction + λ_f U + ε_t.
- **Handoff:** a shift change fires when the agent's shift runs out (soft
  threshold), unless the agent stays (`p_stay`). Otherwise a softmax over
  none, transfer (skill mismatch, early turns), escalation (frustration,
  complexity, friction) and requeue (customer idle, busy queue).
- **Complaint:** logit h_Y = α_0 + α_F F_t + α_W W + δ_{A_t} +
  δ_n (agents so far − 1) + λ_y U.
- **End:** logit h_D = d_0 + d_p progress_t + d_F F_t.

Each handoff type therefore affects complaints three ways: directly (δ), by
raising frustration (γ) and by adding wait at the next turn. It also changes
how long the chat lasts. The total effect of an intervention includes all of
them.

## Outputs

`simulate()` returns a `SimData` with:

- `panel`: one row per chat-turn while the chat is open and before any
  complaint. This is the analysis panel.
- `event_log`: timestamped chat starts, agent replies, handoffs, complaints
  and chat ends. After a complaint an escalation is logged with probability
  `p_post`: the reverse-causality artefact found in real logs.
- `chats`: one row per chat with handoff counts, both all logged and before
  the complaint.
- `latent`: frustration, progress and temperament, for teaching only.

## Ground truth

`true_risks(interventions, n_mc)` reruns the structural equations with the
handoff set by each intervention. All regimes share the same chats and random
numbers (common random numbers). Risks are cumulative incidence accumulated
from the hazards, which carries less noise than counting events.

An intervention maps the natural handoff draw to the handoff given:

- `iv.never("transfer")`: a transfer becomes no handoff. The trigger (the
  mismatch, the frustration) is still there; the same agent carries on.
- `iv.never_any()`: no handoffs at all.
- `iv.scale("escalation", factor=0.5)`: each natural escalation is kept with
  probability 0.5.

## Presets

| Preset | What changes |
| --- | --- |
| `baseline` | Time-varying confounding with feedback; sentiment measures frustration exactly |
| `null` | No handoff affects complaints or chat length by any path |
| `strong_feedback` | Handoff-to-frustration jolts doubled |
| `noisy_sentiment` | Sentiment measures frustration with error (SD 0.6) |
| `unmeasured_confounding` | Temperament U drives frustration and complaints |
| `poor_positivity` | Fraud chats are nearly always escalated; agents rarely stay past shift end |

Any field can be overridden: `Scenario.preset("baseline", p_stay=0.3)`.
