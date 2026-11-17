# Attributing complaints to handoff types

Handoffs cause only part of all complaints: 27% in the `baseline`
scenario. Report that share as the headline, from LTMLE's "no handoffs at
all" regime. Then split it across handoff types with the marginal split.

## Marginal split (recommended)

```python
from gmethods import LTMLE
from gmethods.attribution import attribute_marginal

model = LTMLE().fit(data.panel)
attribute_marginal(model, factor=0.9)   # each type made 10% rarer
```

Starting from how things run today, it asks how many complaints each
handoff type would save if it happened 10% less often.

- Each type's effect includes the later handoffs it triggers, because those
  respond naturally. A transfer that would have led to an escalation gets
  that credit, and the escalation still gets credit for its own direct harm.
- For a small reduction the types barely overlap (a chat would need two of
  its handoffs removed at once), so the effects add up. The leftover
  `interaction` is reported: 4% of the joint effect in `baseline`.
- Only today's handoff pattern has to be in the data. The weights stay near
  one, so LTMLE estimates are stable: in `baseline` with 50,000 chats the
  escalation and requeue effects have standard errors of about 11% of
  their size, against 70% for the time-ordered split.
- It is a business question that can be acted on: which type to reduce
  first. It is also the start of the Aumann-Shapley path from today towards
  no handoffs.

Truth in `baseline`, 10% fewer of each type, complaints saved per 1,000
chats: escalation 0.36 (45%), requeue 0.36 (46%), shift change 0.04 (5%),
transfer 0.03 (4%); all together 0.75.

The split describes the margin today. It does not say what share of the
27% each type accounts for if handoffs were removed entirely: that path
crosses histories the data rarely contains.

## Time-ordered split (for comparison)

### What adds up to what

| Quantity | Definition |
| --- | --- |
| Attributable risk | R(natural course) − R(no handoffs at all) |
| Attributable fraction | Attributable risk / R(natural course) |
| Type credits | Add up to the attributable risk |

The rest of the complaints would happen without any handoff. They are not
attributed to anything: task type or friction history are not things to
switch off, so there is no causal question to split them by.

### How it works

Let g_j be "no handoffs in turn buckets 1 to j, natural course after". The
attributable risk telescopes over these policies:

    R(natural) − R(no handoffs) = Σ_j [R(g_{j−1}) − R(g_j)]

Bucket j's credit is what removing its handoffs adds, with every earlier
handoff already removed and every later one left to happen naturally.

- An earlier handoff keeps the credit for later handoffs it triggers. A
  transfer that frustrates the customer into an escalation is credited with
  the escalation's complaints.
- A chat has at most one handoff per turn, so with one-turn buckets the
  credit at each turn splits exactly by type. The default uses one bucket
  per turn up to turn 25 and one bucket for the rest. The small remainder
  in the tail bucket is shared across types in proportion and reported in
  `att.remainder`.
- Each type's credit is valued in a world where earlier handoffs have
  already been removed, not in the world that happened. A later handoff
  type can therefore get more credit than its own "never k" effect, if
  removing earlier handoffs makes chats run longer and expose more of it.

`attribute_ltmle` runs LTMLE on all the policies (1 + 5 × buckets) and
gives influence-function standard errors for each type's total and for the
attributable risk. `attribute(risks)` builds the same split from any table
of regime risks, for example `ChatSimulator.true_risks` for the truth.

## Not used: Shapley values

Shapley values average a type's contribution over every order of removal.
That needs worlds where a later handoff happens without the earlier
handoff that caused it, and it splits shared credit evenly between cause and
consequence. Restricting Shapley to the handoffs a chat actually had does
not help: it asks whether this customer would have complained without one
of their handoffs, an individual counterfactual the data cannot identify.
