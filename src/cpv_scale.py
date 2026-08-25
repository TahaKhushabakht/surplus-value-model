"""CPV rebuilt at season scale on StatsBomb 360 — now INCLUDING failed passes.

This is what the whole scale-up was for. On Metrica, CPV was confined to completed
passes because a failed pass's intended target had to be inferred from the interception
ray, and that inference was circular with CPV itself (HISTORY.md §14). StatsBomb records
`pass.recipient` independently of outcome, so failed decisions can finally be scored.

TARGET MATCHING AND ITS FILTER. 360 freeze frames are anonymous (positions only), so the
recorded recipient must be located by proximity to `pass.end_location`. That match is
reliable for short passes and unreliable for long balls. The filter
`target_match_dist <= MATCH_DIST_M` is applied to BOTH outcome classes, so the rule is
outcome-independent and cannot by itself manufacture a completed/failed asymmetry. It is
not free of risk: it preferentially drops long ambitious failures, which could flatter
long-ball passers. `match_filter_sensitivity()` sweeps the threshold to measure that.

NO VELOCITIES. Freeze frames are static, so pitch control runs position-only
(see pitch_control.control_from_positions). Absolute CPV values are therefore NOT
comparable to the Metrica prototype's; only within-StatsBomb comparisons are valid.

Reused unchanged from the prototype: the EV formula, the mirrored-xT turnover term, the
cpv_avg / cpv_max definitions, and the finding that cpv_max is per-action only and must
not be aggregated to player level (HISTORY.md §15).

Run: python src/cpv_scale.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from completion_model import reliability_table
from pitch_control import PitchControlParams, control_from_positions

OUTPUT_DIR = Path(__file__).parent.parent / "output"
FEATURES = ["control_target", "distance_m", "forward_m", "pressure_m"]
MATCH_DIST_M = 6.0
PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0


class XTLookup:
    """Season-scale xT grid with metric-coordinate lookup and opponent mirroring."""

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.ny, self.nx = grid.shape

    def _cell(self, x_norm: float, y_norm: float) -> float:
        ix = min(int(np.clip(x_norm, 0, 0.999999) * self.nx), self.nx - 1)
        iy = min(int(np.clip(y_norm, 0, 0.999999) * self.ny), self.ny - 1)
        return float(self.grid[iy, ix])

    def value(self, x_m, y_m):
        xn, yn = np.asarray(x_m) / PITCH_LENGTH_M + 0.5, 0.5 - np.asarray(y_m) / PITCH_WIDTH_M
        return np.array([self._cell(a, b) for a, b in zip(np.atleast_1d(xn), np.atleast_1d(yn))])

    def turnover_value(self, x_m, y_m):
        """Value to the passing team of losing it here: negative of the opponent's
        value at the 180-degree-rotated location (same logic as the prototype)."""
        xn, yn = np.asarray(x_m) / PITCH_LENGTH_M + 0.5, 0.5 - np.asarray(y_m) / PITCH_WIDTH_M
        return -np.array([self._cell(1 - a, 1 - b) for a, b in zip(np.atleast_1d(xn), np.atleast_1d(yn))])


def build_features(passes: pd.DataFrame, params: PitchControlParams = PitchControlParams()) -> pd.DataFrame:
    """Per-pass features for the CHOSEN action, evaluated at the matched target."""
    rows = []
    for r in passes.itertuples(index=False):
        att, opp = np.asarray(r.teammates, float), np.asarray(r.opponents, float)
        if len(att) == 0 or len(opp) == 0:
            continue
        start = np.array([r.start_x, r.start_y])
        target = np.array([[r.target_x, r.target_y]])
        control = control_from_positions(att, opp, target, params)[0]
        rows.append(
            {
                "match_id": r.match_id, "team": r.team,
                "passer_id": r.passer_id, "passer": r.passer,
                "completed": r.completed, "target_match_dist": r.target_match_dist,
                "control_target": control,
                "distance_m": float(np.hypot(r.target_x - r.start_x, r.target_y - r.start_y)),
                "forward_m": float(r.target_x - r.start_x),
                "pressure_m": float(np.min(np.linalg.norm(opp - start, axis=1))),
            }
        )
    return pd.DataFrame(rows)


def _make_model(kind: str):
    """Logistic is the prototype's model; gradient boosting is the flexible alternative.

    At Metrica scale a 4-feature linear logit passed V2, but only because the risky
    low-probability bins held too few passes to qualify. At 20k passes those bins
    qualify and the linear model fails badly (predicts 0.17-0.35 where reality is
    0.44-0.52): completion is a strongly non-linear function of these features, and a
    linear logit underfits the extremes. Sample size now supports a flexible model.
    """
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, min_samples_leaf=60, random_state=0)
    raise ValueError(kind)


def fit_and_validate_completion(feat: pd.DataFrame, kind: str = "gbm", n_folds: int = 5, seed: int = 0, verbose: bool = True):
    """V2 at scale: grouped K-fold by match so no match appears in train and test."""
    data = feat.dropna(subset=FEATURES).reset_index(drop=True)
    matches = np.array(sorted(data["match_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(matches)
    folds = np.array_split(matches, n_folds)

    preds = np.full(len(data), np.nan)
    for fold in folds:
        test = data["match_id"].isin(fold).to_numpy()
        model = _make_model(kind)
        model.fit(data.loc[~test, FEATURES], data.loc[~test, "completed"])
        preds[test] = model.predict_proba(data.loc[test, FEATURES])[:, 1]

    y = data["completed"].to_numpy()
    base = float(y.mean())
    rel = reliability_table(y, preds, n_bins=10, min_bin=30)
    worst = rel[rel["counts_for_v2"]]["abs_gap"].max()
    brier = brier_score_loss(y, preds)

    if verbose:
        print(f"  n={len(y):,}  base completion rate={base:.3f}")
        print(f"  brier={brier:.4f}   baseline={brier_score_loss(y, np.full_like(preds, base)):.4f}")
        print(rel.to_string(index=False))
        print(f"  V2: max |gap| in qualifying bins = {worst*100:.1f}pp (need <= 10pp) -> {'PASSED' if worst <= 0.10 else 'FAILED'}")

    final = _make_model(kind)
    final.fit(data[FEATURES], data["completed"])
    return final, data, preds, {"brier": brier, "worst_gap": worst, "passed": bool(worst <= 0.10)}


def compute_cpv(
    passes: pd.DataFrame,
    model,
    xt: XTLookup,
    params: PitchControlParams = PitchControlParams(),
    turnover_lookup=None,
) -> pd.DataFrame:
    """EV over the full choice set, then cpv_avg / cpv_max per pass.

    turnover_lookup: if given, supplies V(turnover) empirically (see
    turnover_value.TurnoverLookup) instead of mirroring the xT surface. The mirrored-xT
    approximation charged ~nothing for losing the ball outside one's own defensive
    third, collapsing EV into `p * V(receive)`; the empirical surface fixes that.
    """
    rows = []
    for r in passes.itertuples(index=False):
        att, opp = np.asarray(r.teammates, float), np.asarray(r.opponents, float)
        if len(att) < 2 or len(opp) == 0:
            continue
        start = np.array([r.start_x, r.start_y])
        target = np.array([r.target_x, r.target_y])

        # Choice set: every visible teammate. The chosen action is scored by exactly the
        # same pipeline, appended so it is guaranteed to be a member of A(s).
        cands = np.vstack([att, target[None, :]])
        control = control_from_positions(att, opp, cands, params)
        pressure = float(np.min(np.linalg.norm(opp - start, axis=1)))
        feat = pd.DataFrame(
            {
                "control_target": control,
                "distance_m": np.linalg.norm(cands - start, axis=1),
                "forward_m": cands[:, 0] - start[0],
                "pressure_m": pressure,
            }
        )
        if feat[FEATURES].isna().any().any():
            continue
        p = model.predict_proba(feat[FEATURES])[:, 1]
        v_rec = xt.value(cands[:, 0], cands[:, 1])
        v_to = (
            turnover_lookup.value(cands[:, 0], cands[:, 1])
            if turnover_lookup is not None
            else xt.turnover_value(cands[:, 0], cands[:, 1])
        )
        ev = p * v_rec + (1 - p) * v_to

        ev_chosen = float(ev[-1])
        spread = float(ev.std())
        # Context-normalized variants. Both judge the chosen action against ITS OWN
        # choice set, so they self-adjust for where on the pitch the pass was made --
        # a fullback deep in their own half is compared with the options they actually
        # had there, not with the same player's options in the final third. This is
        # action-level normalization; a position LABEL cannot do this because one
        # player spans many contexts. cpv_rank is scale-free (immune to the fact that
        # option EVs span a far wider range near the opponent goal); cpv_z keeps
        # magnitude, with a floor so near-degenerate choice sets cannot explode it.
        rows.append(
            {
                "match_id": r.match_id, "team": r.team,
                "passer_id": r.passer_id, "passer": r.passer,
                "completed": r.completed, "target_match_dist": r.target_match_dist,
                "n_candidates": len(cands),
                "start_x": r.start_x, "start_y": r.start_y,
                "target_x": r.target_x, "target_y": r.target_y,
                "ev_chosen": ev_chosen,
                "ev_mean": float(ev.mean()), "ev_max": float(ev.max()), "ev_spread": spread,
                "p_complete_chosen": float(p[-1]),
                "v_receive_chosen": float(v_rec[-1]),
                "v_turnover_chosen": float(v_to[-1]),
                "cpv_avg": ev_chosen - float(ev.mean()),
                "cpv_max": ev_chosen - float(ev.max()),
                "cpv_z": (ev_chosen - float(ev.mean())) / max(spread, 1e-4),
                "cpv_rank": float((ev < ev_chosen).sum()) / max(len(ev) - 1, 1),
            }
        )
    return pd.DataFrame(rows)


def match_filter_sensitivity(cpv: pd.DataFrame, thresholds=(4.0, 6.0, 8.0, 12.0, 1e9), min_passes: int = 150):
    """Does the match-distance threshold change WHO ranks well? (V5b-style check.)"""
    from scipy.stats import spearmanr

    rankings = {}
    for t in thresholds:
        sub = cpv[cpv["target_match_dist"] <= t]
        g = sub.groupby("passer_id")["cpv_avg"].agg(["mean", "size"])
        rankings[t] = g[g["size"] >= min_passes]["mean"]

    ref = rankings[6.0]
    print(f"  {'threshold':<12}{'n_passes':>10}{'n_players':>11}{'rho vs 6m':>12}")
    for t, series in rankings.items():
        common = ref.index.intersection(series.index)
        rho = spearmanr(ref[common], series[common]).statistic if len(common) > 3 else float("nan")
        label = "no filter" if t > 1e8 else f"<= {t:.0f}m"
        print(f"  {label:<12}{int((cpv['target_match_dist'] <= t).sum()):>10,}{len(series):>11}{rho:>12.3f}")


if __name__ == "__main__":
    passes = pd.read_pickle(OUTPUT_DIR / "sb_passes.pkl")
    xt = XTLookup(np.load(OUTPUT_DIR / "sb_xt.npy"))
    print(f"loaded {len(passes):,} passes with 360 frames")

    kept = passes[passes["target_match_dist"] <= MATCH_DIST_M]
    print(f"training set after symmetric target-match filter (<= {MATCH_DIST_M:.0f}m): {len(kept):,} "
          f"({kept.completed.sum():,} completed, {(1-kept.completed).sum():,} failed)")
    print(f"  retention: completed {kept.completed.sum()/passes.completed.sum():.0%}, "
          f"failed {(1-kept.completed).sum()/(1-passes.completed).sum():.0%}")

    feat = build_features(kept)

    print()
    print("=== V2 at scale: model comparison (grouped 5-fold by match) ===")
    scores = {}
    for kind in ("logistic", "gbm"):
        print(f"\n-- {kind} --")
        model, _, _, s = fit_and_validate_completion(feat, kind=kind)
        scores[kind] = (model, s)
    best = min(scores, key=lambda k: scores[k][1]["worst_gap"])
    model = scores[best][0]
    print(f"\n  selected: {best} (worst gap {scores[best][1]['worst_gap']*100:.1f}pp, "
          f"brier {scores[best][1]['brier']:.4f})")

    # CPV is computed on ALL passes so the match-distance sweep below compares real
    # alternatives. Filtering before this point made the sweep vacuous.
    print()
    print("=== CPV over the full choice set (computed on ALL passes, filtered after) ===")
    cpv = compute_cpv(passes, model, xt)
    cpv.to_pickle(OUTPUT_DIR / "sb_cpv.pkl")
    main = cpv[cpv["target_match_dist"] <= MATCH_DIST_M]
    print(f"  {len(cpv):,} scored total; {len(main):,} inside the <= {MATCH_DIST_M:.0f}m filter")
    print(f"  mean choice set = {cpv.n_candidates.mean():.1f}")
    print(main.groupby("completed")[["ev_chosen", "cpv_avg", "cpv_max"]].mean().round(5).to_string())

    print()
    print("=== Match-distance threshold sensitivity (does it change WHO ranks well?) ===")
    match_filter_sensitivity(cpv)
