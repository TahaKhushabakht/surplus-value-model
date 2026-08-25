# Math Concepts Used in This Project

Brief reference: each concept, what it is, and where it shows up in the pipeline.
Ordered roughly by when you'll need it.

## 1. Conditional probability & expectation
The core CPV formula is just expected value:
`EV(a) = P(complete|a,s)·V(receive) + (1−P(complete|a,s))·V(turnover)`.
Everything else in the project is machinery for estimating the pieces of this equation.
**Used for:** the definition of action value itself.

## 2. Kinematics + Gaussian uncertainty (pitch control)
Pitch control asks: if the ball went to location x, who gets there first? Modeled from
physics — player max speed, reaction time, ball travel time — with Gaussian noise on
arrival times, converted to a probability each team controls each location.
(Reference: Spearman's pitch control model.)
**Used for:** `P(complete|a,s)` — a pass to a location your team controls is likely to
succeed. Also the mitigation for selection bias (see NOTES): physics-based models are
valid at locations players never pass to, where data-driven models must extrapolate.

## 3. Probabilistic classification & maximum likelihood
Logistic regression / gradient-boosted trees, fit by maximum likelihood, to predict
binary outcomes (pass completes or not) from features (distance, angle, pressure,
pitch control at target).
**Used for:** completion-probability models; later, the shot model if we extend to
shooting decisions.

## 4. Calibration & proper scoring rules
A probability model is *calibrated* if events it calls 70% happen 70% of the time.
Checked with reliability curves; scored with Brier score or log-loss (proper scoring
rules — metrics that reward honest probabilities, unlike accuracy).
**Used for:** the trust story. Every probabilistic sub-model must show calibration
evidence, or downstream EV numbers are systematically biased.

## 5. Markov reward processes & the Bellman equation
Model possession as a chain of states (ball location + context) with transition
probabilities estimated from data, and rewards at terminal states (goal, turnover).
The value of a state satisfies the Bellman equation and is solved by value iteration:
`V(s) = Σ_a π(a|s)[R(s,a) + Σ_s' P(s'|s,a)·V(s')]`.
**Used for:** the EPV/xT value surface — `V(receive)` and `V(turnover)` in the EV
formula. This is what makes the value grid principled rather than hand-tuned.

## 6. Cross-validation & out-of-sample testing
Fit on some matches, evaluate on held-out matches. Prevents the models from
memorizing rather than generalizing.
**Used for:** every fitted model in the pipeline.

## 7. Bootstrap resampling
Resample matches/actions with replacement, recompute the metric many times, read off
the spread. Gives confidence intervals without distributional assumptions.
**Used for:** uncertainty on per-player CPV — a point estimate without an interval
is how sports analytics overclaims.

## 8. Empirical Bayes / shrinkage
Small samples produce extreme estimates by luck. Shrinkage pulls each player's
estimate toward the group (e.g., positional) mean, proportionally to how little data
they have.
**Used for:** stabilizing per-player CPV before ranking anyone; low-minutes players
otherwise dominate both ends of the leaderboard.

## 9. Graph theory: centrality & community detection
Team as weighted directed graph (players = nodes, passes = edges). Betweenness /
eigenvector centrality find structurally important players; community detection finds
tactical sub-units.
**Used for:** the scarcity leg — defining a player's functional role from network
structure rather than listed position.

## 10. Clustering (k-means, Gaussian mixtures)
Group players by feature vectors (network role + CPV profile + action tendencies)
into data-driven role archetypes; scarcity = how rare a player's archetype/quality
combination is.
**Used for:** the scarcity metric.

## 11. Regression & residual analysis
Predict market value from age, league, position, standard stats, plus our custom
metrics (CPV, centrality). The *residual* — actual price minus predicted — is the
surplus-value signal: negative residual with strong metrics = potentially underpriced.
**Used for:** the final surplus value model; also the test of whether our metrics add
predictive power beyond standard stats (compare model fit with vs. without them).

## 12. Counterfactual reasoning & its limits (off-support extrapolation)
CPV compares chosen actions to unchosen alternatives — but training data only contains
chosen actions, which players selected precisely because they looked good. Estimates
for rarely-chosen options are extrapolations (the "off-support" problem in causal
inference). This is a *limitation to manage and disclose*, not a solvable nuisance:
lean on physics-based components where possible, and be explicit that CPV is a
model-based counterfactual, not a causally identified effect.
**Used for:** intellectual honesty — this is the first thing a technical skeptic
will probe.
