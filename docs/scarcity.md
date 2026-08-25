# Role Scarcity — Working Definition

The third leg of the pipeline (impact → **scarcity** → price). Impact is
docs/counterfactual-passing-value.md; this defines scarcity.

## What scarcity is for

The surplus-value model predicts market value from performance. Scarcity is the
economic half of that story: a player commands a premium partly because few others can
do what they do. Without it, the model treats two players with identical impact as
interchangeable even when one occupies a role a club cannot easily re-staff.

## Why "role rarity" and not "market rarity"

Two readings were on the table:

1. **Role rarity** — few players have this functional profile.
2. **Market rarity** — few available replacements at this price tier.

Reading 2 is **disqualified on circularity grounds**, not taste. The surplus model
regresses price on impact and scarcity; if scarcity is itself defined in terms of price
tiers, price appears on both sides and the residual (our undervaluation signal) is no
longer interpretable. This is the same failure mode that invalidated inferring failed
passes' targets from the interception ray (HISTORY.md §14): a selection rule correlated
with the quantity being measured. Having paid for that lesson once, we take reading 1.

Reading 1 also carries the more original claim: the market prices players largely by
*listed position*, and if functional roles are scarcer or more abundant than position
labels suggest, that mispricing is exactly what the residual should expose.

## Core design principle: role ⟂ quality

> **Role space describes HOW a player plays. Quality describes HOW WELL.**

Features that encode quality (completion %, CPV, progressive passes completed) must NOT
enter role space. If they do, "scarce" degenerates into "good", scarcity becomes
collinear with impact in the regression, and it contributes nothing that CPV did not
already. Role features must be *stylistic and volume-normalized*: direction, distance,
territory, network position — never success.

This orthogonality is testable, and S3 below tests it.

## Formal definition

Each player `p` gets a role vector `r(p) ∈ R^d` (features below), standardized across
the population. Let `N` be the population size.

### Primary metric — role density scarcity (quality-independent)

Rarity is how isolated a player is in role space. Use a kernel density estimate rather
than hard clusters (no arbitrary K, no boundary artifacts, continuous):

```
density(p) = (1 / (N-1)) * Σ_{q ≠ p} K( ‖r(q) − r(p)‖ / h )
scarcity(p) = −log( density(p) )
```

with `K` a Gaussian kernel and bandwidth `h` (the one knob — see S4). Higher = rarer.
This is quality-independent by construction, so it enters the regression as genuinely
new information alongside CPV.

### Secondary variant — replacement-pool scarcity (economically direct)

The club's actual question is "how many players could replace this one *at this
level*":

```
pool(p; τ) = { q ≠ p : ‖r(q) − r(p)‖ ≤ τ  AND  quality(q) ≥ quality(p) }
scarcity_pool(p) = −log( (|pool(p;τ)| + 1) / N )
```

This is the football analogue of baseball's replacement level (VORP). It is more
interpretable but **deliberately entangled with quality** — the best player in any
neighbourhood has an empty pool by construction, so this variant partly re-encodes
quality rank.

**Both are computed, but they are not interchangeable.** Explicit lesson from V5a
(HISTORY.md §15), where cpv_max and cpv_avg were wrongly assumed to be two estimates of
one ranking when they differed by a known situational term: before treating these two
scarcity variants as alternatives, test whether they rank alike, and if they do not,
derive *why* algebraically rather than tuning until they agree. Prior expectation:
they will NOT agree, because the pool variant contains a quality-rank component the
density variant excludes by design. If so, the density variant is primary (it is what
the regression needs) and the pool variant is a reporting/interpretation aid.

Note the regression can reconstruct the pool variant's content anyway: `density`,
`quality`, and their interaction span it, without the collinearity.

## Role space features (all stylistic, none success-based)

**Passing-network position** (from the directed weighted pass graph; see
docs/math-concepts.md §9):
- ~~betweenness centrality~~ — **DEMOTED, see S1/S2 results below.** Football passing
  networks are far too dense (measured 0.64–0.74, i.e. ~70% complete) for betweenness
  to discriminate: observed values span only [0, 0.10] of a possible [0, 1], so rank
  order is dominated by noise. Unlike every other feature, its stability does NOT
  improve with more data. Retain only if computed on a *thresholded* graph (drop weak
  edges to sparsify), otherwise exclude.
- eigenvector / PageRank centrality — connected to well-connected players?
- in-degree ÷ out-degree ratio — receiver vs. distributor
- local clustering coefficient — plays in tight triangles vs. spans distant units

**Territory** (from tracking / event locations):
- mean action location (x, y) in oriented coordinates
- dispersion of action locations (roamer vs. fixed station)
- share of touches in each pitch third

**Action style** (distributions, not success rates):
- pass direction distribution (forward / lateral / backward share)
- mean and variance of pass distance
- share of passes that are long (> 30m)

Deliberately excluded: completion %, CPV, xT added, progressive passes *completed* —
all quality, not style.

## Normalization: team context

Raw centrality is contaminated by team style and possession volume — a metronome in a
possession-dominant side scores higher than an identical player in a counter-attacking
one. Role features must be normalized within team-match (z-score or percentile rank)
before pooling across teams, so the vector encodes *role within the team's structure*
rather than the team's overall behaviour. This is the same team-context confounding
already flagged for CPV in NOTES.md, and it bites harder here.

## Open design decisions

1. **Bandwidth `h` / radius `τ`** — the main knob. S4 tests stability.
2. **Feature weighting** — all role features currently equally weighted after
   standardization. Alternatives: PCA to decorrelate first (network centralities are
   mutually correlated and would otherwise triple-count "connectedness"), or supervised
   weighting. PCA-first is the likely default; decide with data.
3. **Minutes threshold** — network centrality is unstable for players with few touches;
   needs a floor, same as CPV's.
4. **Population definition** — scarcity is *relative to whom*? Same league?
   Same position group? All players? This materially changes the metric and is a
   modelling choice to state explicitly, not a detail.
5. **Positional conditioning** — should a goalkeeper be "scarce" simply for being
   unlike outfielders? Almost certainly the population should be conditioned or
   goalkeepers excluded, or they dominate the rarity ranking trivially.

## Validation criteria

- **S1 — role-space face validity.** Players with known/inferable positions must land
  sensibly: goalkeepers isolated, centre-backs adjacent to each other, full-backs
  distinct from centre-backs, forwards far from defenders. Checkable on Metrica
  (positions inferable from mean pitch location despite anonymization).
- **S2 — stability.** The same player, split across two halves of their available data,
  must land in a similar region of role space (split-half rank correlation ≥ 0.7).
  Roles are supposed to be traits, not noise.
- **S3 — orthogonality to quality.** |correlation(scarcity_density, mean CPV)| should be
  low (< ~0.3). If it is high, role space has leaked quality features and the metric is
  redundant with impact. This is the criterion that protects the core design principle.
- **S4 — bandwidth sensitivity.** Player scarcity ranking under two reasonable
  bandwidths must correlate ≥ 0.7 (same standard as CPV's V5b).

## S1 / S2 results (2026-07, Metrica, `src/validate_scarcity.py`)

**S1 — face validity: PASSED, but the signal is not where the design assumed.**
Constructed to avoid tautology: positions were inferred from *territory* (mean pitch
location), then predicted from *network + style* features only — disjoint feature sets.
Leave-one-out kNN reached 62.2% vs a 53.3% majority-class baseline (n=45): above chance,
but modest. The Spearman correlations against the position ordinal (GK<DEF<MID<FWD)
show why:

| feature | rho vs position |
|---|---|
| in_out_ratio | **+0.777** |
| fwd_share | **−0.770** |
| long_share | −0.415 |
| pagerank | +0.205 |
| betweenness | −0.044 |

Style features carry nearly all the positional signal; network centralities carry
little. The weak kNN accuracy is consistent with diluting two strong features across
ten mostly-noisy dimensions on n=45 — which is itself an argument for the PCA/feature-
weighting step in open decision 2.

Sanity check that the machinery works: goalkeepers were recovered with no position
labels anywhere in the pipeline — deepest mean_x (−42 to −46m), lowest dispersion,
`fwd_share` ≈ 0.93–1.00 (they only pass forward), highest `long_share`. Emergent, not
encoded.

**S2 — split-half stability: FAILED at match scale.** Median split-half rho = 0.353;
only 1/10 features reached the 0.7 bar (back_share, 0.864). The centralities were the
*worst*: betweenness 0.331, pagerank 0.352, clustering 0.237.

Diagnosed rather than accepted, by measuring stability at two data volumes
(quarter-match vs half-match networks):

| feature group | quarter-match | half-match | change |
|---|---|---|---|
| STYLE | 0.153 | 0.545 | **+0.392** |
| NETWORK | 0.296 | 0.485 | +0.189 |

Doubling data roughly triples style-feature stability and materially improves network
features — so the S2 failure is predominantly a **sample-size artifact**, not a
conceptual defect. Extrapolating, season-scale networks (thousands of passes vs ~30 per
half here) should clear the bar. This is untested extrapolation and must be re-run at
StatsBomb scale before any player-level scarcity claim.

**The one genuine exception is betweenness** (−0.029: flat, alone among all features).
Investigated: network density is 0.64–0.74, so almost every pair of players is directly
connected and few shortest paths route through a third player. Observed betweenness
spans only [0, 0.10] of the possible [0, 1] — a compressed range where small absolute
noise produces large rank swings. More data makes the graph *denser*, not sparser, so
this does not improve with volume. **Structural, not small-sample** — hence the demotion
above.

Net design change: **style features are the backbone of role space; centralities are
supporting features pending season-scale re-validation; betweenness is excluded unless
computed on a sparsified graph.**

## Scarcity computed (2026-07, StatsBomb, Bayer Leverkusen 2023/24 + 17 opponents)

The blocking limitation below is resolved. `src/scarcity.py` implements both metrics;
`docs/... ` — results and validation:

**Population**: 263 identified players (min 15 passes), 18 teams. Composition stated
explicitly, not smoothed over: Leverkusen supplies ~25 players at up to season volume
(2000+ passes); the 17 opponents supply ~2 matches each (16-180 passes). Non-Leverkusen
estimates are noisier (S2's k=2 stability was ~0.58-0.62 vs ~0.85 at season volume).

**Feature/population fixes made along the way**:
- `SCARCITY_FEATURES` = RECOMMENDED_NETWORK_FEATURES + TERRITORY_FEATURES + STYLE_FEATURES
  (betweenness excluded per its §18/§22 demotion), PCA-decorrelated to 8 components
  (92.4% variance) before distance computation — several features are compositional and
  collinear by construction (the third-of-pitch and pass-direction triples each sum to 1).
- **GK exclusion took three iterations to get right** (HISTORY.md §27) — the original
  absolute rule and a first relative-rule fix each failed differently. Final rule
  (`mean_x < -30`, in `role_space.infer_position`) verified against a clean, total gap
  across all 18 teams: every goalkeeper sits below -36.8, no outfield player deeper than
  -27.2.

**S3 (orthogonality to quality): PASSED.** corr(scarcity_density, quality) = -0.007,
well inside the |.|<0.3 bar. Role space has not leaked quality information.

**S4 (bandwidth sensitivity): PASSED.** rho = 0.980 across narrow/wide bandwidths.

**Density vs. pool variant: rho = 0.517** — moderate, not high, exactly as predicted in
this doc before either was computed ("expected NOT to rank alike"). Confirms scarcity_pool
is a distinct, quality-entangled lens rather than a redundant restatement of
scarcity_density.

**Face validity (reliable subset, n_passes>=100, 37 players)**: the rarest roles are
Eric Dier, Jonathan Tah, Edmond Tapsoba, Min-jae Kim, Nico Schlotterbeck, Mats Hummels —
overwhelmingly modern ball-playing centre-backs, plus Dier's well-documented positional
reinvention as a deep-lying build-up midfielder under Tuchel that season. This pattern
was not hand-specified; it emerged from passing-network + territory + style features.
Confirms the design: scarcity_density and n_passes are essentially uncorrelated
(-0.005), so this is not simply "low-data players look rare."

**On Leverkusen specifically**: Tah and Tapsoba are the squad's two most distinctive
roles; Wirtz, Grimaldo, and Frimpong sit toward the LOW end of scarcity (0.40-0.50).
Read together with CPV: those attackers are elite QUALITY in a comparatively common
ROLE, while the centre-back pair occupies a comparatively rare role at more moderate
quality — the intended reading of impact and scarcity as separate, complementary axes.

Full results: `output/sb_scarcity.pkl` (scarcity_density, scarcity_pool, quality,
inferred_position, n_passes per player).

## BLOCKING LIMITATION: scarcity needs a population

Unlike CPV, **scarcity is not prototypable on Metrica**. It is inherently a
cross-player statistic — "rare" is meaningless without a population to be rare within.
Metrica gives ~28 distinct player-slots per match, anonymized, with no identity linkage
across matches, so players cannot even be pooled across the two games. A density
estimate over ~28 anonymous points is not a scarcity measurement.

Consequence — the work splits cleanly:

- **Buildable and validatable now on Metrica**: the *role characterization machinery*
  — passing-network construction, centrality computation, territory/style features,
  normalization. S1 (face validity) and partially S2 (stability) can be checked.
- **Deferred to StatsBomb scale**: the *rarity statistic itself*, plus S3 and S4, which
  require a real population of identified players across many matches.

This is the same two-tier pattern as pitch control (validate the method on small data,
compute the conclusions on large data), and it is a scoping fact, not a defect.
