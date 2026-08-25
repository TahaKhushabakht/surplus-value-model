# Counterfactual Passing Value (CPV) — Working Definition

## Core idea

A pass isn't just "completed" or "not." What matters for evaluating the *decision*
is how good the chosen pass was **relative to the other options available at that
moment** — not just whether it worked out. CPV measures decision quality by
comparing the expected value of the pass a player actually chose against the
expected value of the alternatives they had.

This separates two things that are usually conflated:
- **Decision quality** — was this a good choice, given what was knowable at the time?
- **Execution quality** — did it work out better or worse than expected (luck/skill
  in execution)?

## Formal definition

At the moment of a pass, define the game state `s`: ball location, positions/velocities
of all players (from tracking data), possession team, score, time remaining.

**Choice set `A(s)`** — the set of feasible pass targets available to the passer at
that instant (teammates, or target locations on a grid). "Feasible" needs an explicit
rule (e.g., unobstructed passing lane, within realistic pass-speed/distance range).
This is a modeling choice, not given by the data — **see open question 1 below.**

For each candidate action `a ∈ A(s)`:

- `P(complete | a, s)` — probability the pass succeeds. Modeled from pressure,
  distance, angle, and (ideally) the pitch-control surface at the target location.
- `V(receive | a, s)` — value of possession if the pass is completed at the target
  location, from an **Expected Possession Value (EPV)** surface: a model of
  P(this possession ends in a goal within the next N actions | location, game state).
  Start simple: use a static zone-based value grid like Expected Threat (xT). Upgrade
  later to a continuous, state-conditioned EPV model if tracking data is available.
- `V(turnover | a, s)` — value to the *opponent* if the pass fails and possession is
  lost at the interception point (mirror of the EPV surface, opponent's perspective).

**Expected value of an action:**

```
EV(a) = P(complete | a, s) * V(receive | a, s) + (1 - P(complete | a, s)) * V(turnover | a, s)
```

**Chosen action** `a*` = what the player actually did.

**Counterfactual Passing Value:**

```
CPV(a*) = EV(a*) - baseline(A(s))
```

Where `baseline(A(s))` is one of (see open question 2):
- `max_{a ∈ A(s)} EV(a)` — "optimal decision" baseline (harsh, assumes perfect info)
- weighted average of `EV(a)` over `A(s)`, weighted by a behavioral model of how likely
  a typical player is to choose each option (soft, doesn't punish reasonable risk-taking)
- percentile rank of `EV(a*)` within the distribution of `EV(A(s))`

## Decision quality vs. execution quality (decomposition)

Because CPV is computed on *expected* value, not realized outcome, it measures
decision quality only. Execution quality can be captured separately:

```
Decision quality  = EV(a*) - baseline(A(s))          <- CPV, ex-ante
Execution quality = realized_value(a*) - EV(a*)       <- luck/skill in outcome
```

This mirrors decision/execution decompositions used in other sports analytics (e.g.,
shot quality vs. finishing in basketball/hockey). Worth keeping both numbers — a
player can be a great decision-maker with poor execution or vice versa, and they
probably carry different implications for value/scarcity.

## Open design decisions (revisit before building)

1. **Choice set definition** — how many alternatives count, and how to exclude
   unrealistic options. This materially changes CPV: too narrow a set makes every
   pass look "optimal" by default; too broad makes normal passes look bad.
2. **Baseline choice** — max vs. weighted-average vs. percentile. Max is adversarial
   (penalizes any non-optimal pass); weighted-average is more forgiving and arguably
   more honest about what's "reasonably expected" of a player.
3. **EPV model fidelity** — static xT grid (fast to build, coarse) vs. full
   state-conditioned EPV trained on tracking data (accurate, data-hungry).
4. **Data tier** — StatsBomb 360 (freeze-frame snapshot at the pass moment, broad
   match coverage) vs. Metrica tracking (full continuous tracking, only a handful of
   free matches). This is the single biggest constraint on how sophisticated the
   pitch-control/completion-probability model can be. Recommendation: prototype on
   Metrica's small tracking sample to get the full method working, then evaluate
   whether a cruder StatsBomb-360-only version is "good enough" to scale to more matches.
5. **Turnover value asymmetry** — losing the ball in your own third is much worse than
   losing it in the opponent's third; make sure `V(turnover)` reflects location, not
   a flat penalty.

## Generalizing beyond passing

The same `EV(a) = P(success) * V(success) + P(fail) * V(fail)`, compared against a
baseline over `A(s)`, generalizes to any on-ball decision — not just passes:

- Shoot vs. pass vs. dribble (choice of *action type*, not just pass target) — this is
  the direction VAEP-style all-action valuation takes.
- Press/tackle engagement vs. covering space (defensive counterfactual).
- Off-ball run value (needs tracking data to know what runs were "available" —
  significantly harder, likely a stretch goal).

See `NOTES.md` for the running list of these and how they might feed back into the
financial/surplus-value side of the project.
