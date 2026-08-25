# Prototype Validation Criteria & Gap Mitigation Options

Two things in this doc: (1) the finish line for the Metrica prototype phase — what
"validated, ready to scale to StatsBomb" concretely means; (2) options for each known
gap, with a recommended pick. Decisions here are provisional until building starts.

---

## Part 1: Validation criteria (the finish line)

The prototype is DONE when all five pass. If a criterion proves impossible on 3
matches, document why and either relax it explicitly or move it to the StatsBomb
phase — don't silently tune forever.

### V1. Pitch control sanity
Rendered pitch-control surfaces for ~10 randomly sampled frames look physically
sensible on visual inspection: control near a player belongs to that player's team,
contested zones sit between opposing players, no team "controls" space it obviously
can't reach.
*Pass/fail: eyeball test, but on randomly sampled frames — no cherry-picking.*

### V2. Completion-model calibration
Reliability curve for P(complete) stays within ±10 percentage points of the diagonal
across all probability bins containing ≥30 attempted passes, on held-out data
(leave-one-match-out across the 3 matches). Report Brier score alongside.
*Note: 3 matches ≈ 2–3k passes total — bins will be coarse. That's fine; the ±10pp
tolerance reflects prototype-scale data, and tightens at StatsBomb scale.*

**RESULT (revised after the target-leakage fix) — PASSED, max gap 9.4pp.**
Brier 0.0900 vs 0.1117 baseline. The earlier reported 0.046-vs-0.119 was inflated by
target leakage (features evaluated at the ball's realized endpoint, which for a failed
pass is a defender's position). Honest improvement over baseline is ~19%, not ~61%.
Known residual issue, below the qualifying threshold so not blocking: bins 0–3 (n=68
pooled) are systematically UNDERconfident — predicted ~0.23, empirical ~0.46. The model
is too pessimistic about the riskiest passes, which is precisely the regime where
counterfactual alternatives live. Worth re-checking at StatsBomb scale, where those
bins will have enough samples to qualify.

### V3. Value-surface face validity
The EPV/xT surface is monotonically increasing toward the opponent's goal along the
pitch's long axis, highest in the central box, and near-zero deep in one's own half.
Any violation must be explainable (e.g., corner-flag quirks from sparse data) or fixed.

### V4. Action-level eye test
Pull the 10 highest-CPV and 10 lowest-CPV passes across the 3 matches and inspect
each against the tracking animation. At least 8/10 of each must be defensible to a
knowledgeable viewer ("yes, that was a smart/poor choice"). Log every failure case —
failures are the most informative output of the prototype.

### V5. Sensitivity stability
Compute per-player CPV rank under (a) max baseline vs. weighted-average baseline, and
(b) two reasonable choice-set widths. Rank correlation (Spearman) across variants
≥ 0.7. If rankings flip wildly on arbitrary knobs, the metric isn't measuring
something real yet.

**RESULT (revised after the target-leakage fix) — V5b PASSED, V5a resolved by algebra.**

V5b (unrestricted vs. 40m-limited choice set): rho = **0.999**. The EV computation is
robust to this design choice — reassuring, since it's the more consequential knob.

V5a (cpv_max vs cpv_avg per-player ranking): rho = **−0.260**. Note this went from
+0.295 (pre-fix) to negative *after* removing leakage — the leakage had been masking
the real relationship by saturating P(complete) and compressing option spread.

The cause is not statistical, it is definitional. By construction:

    cpv_max = EV(chosen) − max EV      cpv_avg = EV(chosen) − mean EV
    ⟹  cpv_max = cpv_avg − (max EV − mean EV) = cpv_avg − option_spread

Verified exactly in code (max abs error 1.1e-16). So the two metrics differ by
**option spread**, which is a property of the *situations a player finds themselves in*,
not of their decisions. Measured per-player: rho(cpv_max, spread) = **−0.915**, i.e.
~84% of the variance in a player's mean cpv_max is explained by situation type alone.
An attacker who repeatedly faces one standout option among poor ones has large spread
and therefore a poor cpv_max, largely regardless of what they choose.

**Conclusion — cpv_max must not be aggregated to player level.** It is meaningful
per-action ("was this the best available option in this moment") and is retained for
that use, including the V4 eye test. It is *not* a player-evaluation statistic, and the
earlier framing of it as a "secondary indicator" that merely happened to disagree with
cpv_avg was too generous. **cpv_avg is the sole per-player decision-quality metric.**

V5's baseline-choice axis is therefore retired as a sensitivity check: it was testing
whether two quantities that differ by a known situational term produce the same
ranking, which they provably cannot. V5b (choice-set width) remains the meaningful
stability test, and it passes decisively.

**Explicitly NOT prototype goals:** stable per-player estimates (3 matches can't give
that), price linkage (data is anonymized), novelty of results. The prototype validates
*machinery*, not conclusions.

---

## Part 2: Options for each gap

### Gap A: Selection bias (unchosen options are off-support)

| Option | Idea | Cost | Verdict |
|---|---|---|---|
| A1. Physics-first completion model | Derive P(complete) mainly from pitch control (physics valid everywhere), use data-driven correction only as a small learned adjustment | Medium | **Recommended** — attacks the root cause |
| A2. Restrict choice set to "plausible" options | Only count alternatives similar to passes actually observed from similar states (propensity trimming) | Low | Good complement to A1; narrows the claim honestly |
| A3. Uncertainty-aware EV | Penalize/flag EV estimates far from training support (e.g., distance to nearest observed pass in feature space) | Medium | Nice-to-have; adds honesty, defer to StatsBomb phase |
| A4. Ignore, disclose only | State the limitation, change nothing | Free | Insufficient alone, but the disclosure itself is mandatory regardless |

**Pick: A1 + A2 now, A3 later, A4's disclosure always.**

### Gap B: Small-sample instability

| Option | Idea | Verdict |
|---|---|---|
| B1. Empirical-Bayes shrinkage to positional mean | Standard, well-understood, easy | **Recommended** at StatsBomb phase |
| B2. Hard minimum-minutes threshold only | Simple filter (e.g., 900+ min) | Use *with* B1, not instead — thresholds alone still leave noise just above the cutoff |
| B3. Hierarchical Bayesian model (player within position within team) | Most principled; gives full posteriors | Upgrade path if B1 feels too crude; more work |

**Pick: B1 + B2. Not a prototype concern (no player claims made at prototype scale).**

### Gap C: Team-context confounding

| Option | Idea | Verdict |
|---|---|---|
| C1. Position-adjusted baselines | Compare CPV within functional role only | **Minimum viable** — do this first |
| C2. Team fixed effects in the aggregation | Regress out team identity from per-action CPV | **Recommended addition** — cheap once data is tabular |
| C3. Opponent adjustment | Also control for opponent strength per action | Defer — diminishing returns until the basics work |
| C4. With-vs-without-you network deltas | Measure how team CPV/network shifts when player is absent | Interesting but data-hungry (needs many matches per player); park it |

**Pick: C1 + C2 at StatsBomb phase.**

### Gap D: Anonymized prototype data
Not really a gap to "solve" — a scoping fact. One genuine option worth noting:
Metrica Game 3 is the same match released by other providers in some public datasets;
de-anonymization has been done by hobbyists. **Not worth the effort** — the prototype
doesn't need identities. Accept and move on.

### Gap E: Price-side hygiene (inflation, market shifts)

| Option | Idea | Verdict |
|---|---|---|
| E1. Season fixed effects in the surplus model | Let the model absorb per-season market level | **Recommended** — simplest and robust |
| E2. Deflate fees by a transfer-inflation index | Explicit index (e.g., median fee per season) | Fine alternative; slightly more assumptions |
| E3. Model log(fee) | Log transform handles the heavy right tail of fees regardless of E1/E2 | **Do this too** — fees are log-normal-ish; also consider Tobit-style handling for free transfers/loans |

**Pick: E1 + E3. Far-future concern; recorded so it isn't re-derived later.**
