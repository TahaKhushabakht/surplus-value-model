# Literature Log

Papers/articles relevant to this project, logged as they're found. Each entry: title,
authors, and the summary as provided — not re-derived or expanded, just recorded.
Add a **Relevance** line only when it's clear how the paper connects to something
already in this project (link with `[[name]]` to the doc it bears on).

---

<!-- Template for new entries:

## {{Title}}
**Authors:** {{names}}
**Summary:** {{summary as provided}}
**Relevance:** {{optional — which part of the project this bears on, e.g. links to
docs/counterfactual-passing-value.md or the parked scarcity/shooting ideas}}

-->

## Machine Learning Value Prediction based on a Soccer Player's Performance on the Field
**Authors:** Yunzhe Chen
**Date:** version created December 10, 2025
**Summary:** This research develops a machine learning approach to predict professional
football player market values based on their on-field performance data. The study
collects comprehensive performance statistics from the five major European leagues
during the 2023-2024 season using web crawler technology to extract data from a
professional football platform. The dataset includes information on 1,934 players with
36 different fields covering offensive, organizational, and defensive attributes.

The researchers employ rigorous data preprocessing techniques to prepare the dataset
for modeling. Missing values stemming from positional requirements are handled by
setting them to zero, while structurally present but occasionally unobserved features
use minimum value imputation. Feature selection is conducted through correlation
analysis and domain expertise, resulting in the engineering of five new features
including average goals per game and average assists per game. The target variable of
player market value undergoes logarithmic transformation to address right-skewed
distribution and improve model performance.

Three ensemble learning models are developed and compared: XGBoost, Gradient Boosted
Decision Trees (GBDT), and Random Forest (RF). Each model employs decision tree-based
approaches with different mechanisms. GBDT iteratively trains weak learners by fitting
negative gradients of the loss function. XGBoost enhances this framework through
regularization terms and second-order Taylor expansion optimization. Random Forest uses
bootstrap sampling and random feature selection to reduce variance and improve
robustness. Model hyperparameters are optimized using RandomizedSearchCV to achieve
optimal performance.

The evaluation uses three metrics: the coefficient of determination (R-squared), mean
absolute error (MAE), and root mean square error (RMSE). Results demonstrate that GBDT
achieves the best fitting performance with the smallest errors on both training and
test sets. XGBoost shows strong generalization ability with reasonable accuracy, while
RF exhibits larger prediction errors. Feature importance analysis reveals that age,
player rating, frontcourt pass success rate, and average ball recoveries per game are
the most influential factors across all models.

Empirical analysis on 54 players from a different season shows that sixteen players
have prediction errors below twenty percent of their actual value, indicating accurate
assessment. However, twenty-one players classified as outliers display prediction
errors exceeding fifty percent. Investigation of these outliers identifies three
primary error sources: international reputation of star players not captured by
performance data alone, rapid value changes in young players during growth phases, and
sudden value drops for declining players. The study concludes that while machine
learning provides quantitative tools for player valuation and club decision-making,
incorporating non-game factors like reputation and historical value trajectories could
improve future model accuracy and practical applicability.

**Relevance:** Closest existing precedent to the surplus-value regression step
([[scarcity]] / the impact->price leg) — same core mechanic (predict market value from
performance stats, treat large residuals as the signal of interest), but without a
counterfactual/decision-quality layer or a scarcity term; useful as a benchmark once
our own price-side data exists. Their outlier diagnosis maps directly onto two open
items already in NOTES.md: reputation as an omitted variable (something CPV/scarcity
don't capture either) and age-driven value trajectories (the parked "age-adjusted value
cliff" idea from the original financial-analysis brainstorm). GBDT beating XGBoost and
RF on tabular data of similar size also matches the finding in
[[counterfactual-passing-value]]'s validation work, where gradient boosting outperformed
a linear model once sample size supported it (HISTORY.md §23).

---

## Soccer Action Spotting with Multimodal Fusion and Markov-Guided Transformers
**Authors:** Kumakura, Takane; Orihara, Ryohei; Tahara, Yasuyuki; Ohsuga, Akihiko; Sei,
Yuichi; van den Herik, H. Jaap; Rocha, Ana Paula; Steels, Luc
**Summary:** In automated analysis of soccer match videos, conventional approaches
primarily rely on visual information, but detecting "invisible actions"—events not
captured in the video due to replays or camera angles—remains a major challenge. This
study proposes ASPERA (Action SPotting thrEe-modal Recognition Architecture), a
multimodal framework that integrates video, audio, and commentary text streams. We
further enhance ASPERA through three extensions: ASPERA_srnd incorporates surrounding
commentary context, ASPERA_cln filters out irrelevant background information, and
ASPERA_MC introduces prior knowledge of action transitions based on Markov chains.
Experiments on the SoccerNet-v2 dataset demonstrate that the proposed methods
significantly improve detection performance, particularly for invisible actions. The
framework contributes to advancing soccer video analysis, enabling real-time analytics
and automated highlight generation.

**Relevance:** Three separate touchpoints with this project. (1) "Invisible actions" —
events absent because of camera framing/replays — is the same underlying failure mode
documented for StatsBomb 360 freeze frames (HISTORY.md §20): players outside
`visible_area` are simply missing from a frame, not flagged as unknown, because the
data is broadcast-camera-derived. Different layer (action detection vs. positional
tracking) but the same root cause. (2) ASPERA_MC's use of Markov chains as a transition
prior over actions is the same mathematical tool as the value-iteration/Markov reward
process behind xT ([[math-concepts]] §5, [[counterfactual-passing-value]]) — worth
reading for how they estimate/validate transition priors, since our own T(z->z')
estimation is currently a simple empirical count with no smoothing. (3) The
transformer/multimodal architecture is a concrete instance of the
"spatiotemporal transformer" upgrade path already parked in NOTES.md — this one fuses
video+audio+text rather than tracking positions, so it's adjacent rather than directly
applicable, but confirms transformer-based approaches are the active frontier for this
class of problem. Also notes SoccerNet-v2 as a distinct open dataset from StatsBomb/
Metrica, video-native rather than event/tracking-native.
