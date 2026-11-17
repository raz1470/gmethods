# Attributing complaints to handoff types

Handoffs cause only part of all complaints. `gmethods.attribution` splits
that part across handoff types so the shares add up to it exactly.

```python
from gmethods import LTMLE
from gmethods.attribution import attribute_ltmle

model = LTMLE().fit(data.panel)
att = attribute_ltmle(model)
att.summary()          # per type: per 1,000 chats, share of handoff complaints, share of all
att.by_bucket          # credit by turn and type
```

## What adds up to what

| Quantity | Definition |
| --- | --- |
| Attributable risk | R(natural course) − R(no handoffs at all) |
| Attributable fraction | Attributable risk / R(natural course) |
| Type credits | Add up to the attributable risk |

The rest of the complaints would happen without any handoff. They are not
attributed to anything: task type or friction history are not things to
switch off, so there is no causal question to split them by.

## The time-ordered split

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
