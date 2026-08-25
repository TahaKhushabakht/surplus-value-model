"""Pass completion probability model — P(complete | pass, state).

The third ingredient of CPV (docs/counterfactual-passing-value.md): weighs
V(receive) against V(turnover) in the EV formula. Validation criterion V2 in
docs/prototype-validation.md (leave-one-game-out calibration within +/-10pp).

DESIGN (Gap A pick, docs/prototype-validation.md): physics-first. The core feature
is pitch control at the pass target, computed from tracking at the moment the pass
starts. A deliberately small logistic regression layers a few geometric features on
top. Rationale: the physics model is valid at any target location — including
locations players never actually pass to — which is exactly where CPV's
counterfactual alternatives live. A purely data-driven model would have to
extrapolate off-support there (see docs/math-concepts.md §12).

LABELS (documented, with caveat):
  1 = Type PASS (Metrica logs these only when completed)
  0 = Type BALL LOST with an INTERCEPTION-family subtype
Excluded as ambiguous: THEFT (dribble dispossession, not a pass), BALL OUT (cannot
distinguish failed pass from deliberate clearance), FORCED / unlabeled losses.
Consequence: P(complete) here is conditional on "completed or intercepted", so it is
slightly optimistic as an absolute number. Fine for the prototype; revisit at
StatsBomb scale where pass outcomes are labeled explicitly.

INTENDED TARGET, NOT REALIZED ENDPOINT (fixes a target-leakage bug; see HISTORY.md).
Features are evaluated at the INTENDED receiver's position at the pass's start frame,
never at the event's End X/Y. The earlier version used End X/Y, which leaked the label:
measured on game 1, a completed pass's end location sat a median 2.45m from the nearest
teammate, while an intercepted pass's end location sat 4.73m from the nearest OPPONENT
(6.25m from the nearest teammate) — i.e. "where the ball ended up" is a defender's
position precisely when the pass failed, so the feature partly encoded the outcome.
Evaluating at the intended target instead makes the training-time feature the same
object as the CPV-time counterfactual feature (a teammate's position), which is the
whole point: the model must score options the passer had, not outcomes already known.

Metrica does not record `To` for BALL LOST events, so the intended receiver of a failed
pass is inferred (infer_intended_receiver). That inference is validated directly by
running it on completed passes, where the true receiver IS known — see
validate_receiver_inference().

FEATURES per attempt (all computed from the tracking frame at Start Frame):
  control_target  pitch control of passing team at the intended receiver's position
  distance_m      distance from passer to intended receiver, in meters
  forward_m       signed progress toward the opponent goal in meters
  pressure_m      distance from passer to the nearest opponent (pressure proxy)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from expected_threat import _attack_directions  # shared team/period orientation logic
from pitch_control import PITCH_LENGTH_M, PITCH_WIDTH_M, PitchControlParams, pitch_control_at_point

FAILURE_SUBSTR = "INTERCEPTION"

FEATURES = ["control_target", "distance_m", "forward_m", "pressure_m"]


def event_player_to_id(label) -> str | None:
    """'Player26' -> '26'. Returns None for missing/non-string labels."""
    if not isinstance(label, str):
        return None
    return label.replace("Player", "").strip()


def attempt_masks(events: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(completed_pass_mask, intercepted_pass_mask) — the two labeled classes."""
    is_pass = events["Type"] == "PASS"
    is_fail = (events["Type"] == "BALL LOST") & events["Subtype"].fillna("").str.contains(FAILURE_SUBSTR)
    return is_pass, is_fail


def norm_to_metric(x: float, y: float) -> tuple[float, float]:
    """Event coords are normalized [0,1]; tracking is meters centered at pitch center."""
    return (x - 0.5) * PITCH_LENGTH_M, (y - 0.5) * PITCH_WIDTH_M


def metric_to_norm(x_m: float, y_m: float) -> tuple[float, float]:
    """Inverse of norm_to_metric. Used to look up tracking (metric) positions in the
    normalized-coordinate xT grid."""
    return x_m / PITCH_LENGTH_M + 0.5, y_m / PITCH_WIDTH_M + 0.5


def nearest_opponent_dist(row: pd.Series, team: str, player_ids: dict[str, list[str]], point_m: tuple[float, float]) -> float:
    opp = "away" if team == "home" else "home"
    dists = []
    for pid in player_ids[opp]:
        ox, oy = row.get(f"{opp}_{pid}_x"), row.get(f"{opp}_{pid}_y")
        if pd.isna(ox) or pd.isna(oy):
            continue
        dists.append(np.hypot(ox - point_m[0], oy - point_m[1]))
    return float(min(dists)) if dists else float("nan")


def infer_intended_receiver(
    row: pd.Series,
    team: str,
    player_ids: dict[str, list[str]],
    start_m: tuple[float, float],
    end_m: tuple[float, float],
    passer_id: str,
    max_angle_deg: float = 40.0,
) -> str | None:
    """Infer which teammate an intercepted pass was aimed at.

    Metrica leaves `To` empty on BALL LOST events, so the intended target is inferred
    geometrically: the ball travelled from where the pass was struck toward where it was
    intercepted, so take that ray and pick the teammate whose bearing from the passer
    deviates least from it. Returns None if no teammate lies within `max_angle_deg` of
    the ray (the pass was likely aimed into space, or the geometry is degenerate) —
    those attempts are dropped rather than guessed at.

    Reliability of this inference is measured, not assumed: validate_receiver_inference()
    runs it on completed passes where the true receiver is known.
    """
    d = np.array([end_m[0] - start_m[0], end_m[1] - start_m[1]], dtype=float)
    norm = float(np.linalg.norm(d))
    if norm < 1e-6:
        return None
    d /= norm

    best_id, best_angle = None, np.inf
    for pid in player_ids[team]:
        if pid == passer_id:
            continue
        x, y = row.get(f"{team}_{pid}_x"), row.get(f"{team}_{pid}_y")
        if pd.isna(x) or pd.isna(y):
            continue
        v = np.array([x - start_m[0], y - start_m[1]], dtype=float)
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-6:
            continue
        angle = float(np.degrees(np.arccos(np.clip(float(np.dot(v / v_norm, d)), -1.0, 1.0))))
        if angle < best_angle:
            best_angle, best_id = angle, pid

    if best_id is None or best_angle > max_angle_deg:
        return None
    return best_id


def validate_receiver_inference(
    tracking: pd.DataFrame,
    events: pd.DataFrame,
    player_ids: dict[str, list[str]],
    max_angle_deg: float = 40.0,
) -> dict:
    """Accuracy of infer_intended_receiver on COMPLETED passes, where truth is known.

    This is the honest check on the inference used for failed passes: same geometry,
    same code path, but here we can compare against the recorded `To`.
    """
    is_pass, _ = attempt_masks(events)
    frames = tracking.set_index("Frame")
    n_total = n_resolved = n_correct = 0

    for _, ev in events[is_pass].iterrows():
        if pd.isna(ev["Start X"]) or pd.isna(ev["End X"]):
            continue
        frame = int(ev["Start Frame"])
        if frame not in frames.index:
            continue
        passer_id, true_id = event_player_to_id(ev["From"]), event_player_to_id(ev["To"])
        if passer_id is None or true_id is None:
            continue
        team = ev["Team"].lower()
        n_total += 1
        guess = infer_intended_receiver(
            frames.loc[frame], team, player_ids,
            norm_to_metric(ev["Start X"], ev["Start Y"]),
            norm_to_metric(ev["End X"], ev["End Y"]),
            passer_id, max_angle_deg,
        )
        if guess is not None:
            n_resolved += 1
            n_correct += int(guess == true_id)

    return {
        "n_total": n_total,
        "n_resolved": n_resolved,
        "resolve_rate": n_resolved / n_total if n_total else float("nan"),
        "accuracy_when_resolved": n_correct / n_resolved if n_resolved else float("nan"),
    }


def build_pass_dataset(
    tracking: pd.DataFrame,
    events: pd.DataFrame,
    player_ids: dict[str, list[str]],
    params: PitchControlParams = PitchControlParams(),
    max_angle_deg: float = 40.0,
) -> pd.DataFrame:
    """One row per pass attempt (completed or intercepted) with features + label.

    Features are evaluated at the INTENDED receiver's position at the pass's start
    frame — see module docstring on target leakage. For completed passes the receiver
    is read from `To`; for intercepted passes it is inferred.

    `tracking` must already have velocity columns (compute_velocities) and metric
    coordinates. `events` must be raw/unoriented (frames must match tracking).
    """
    directions = _attack_directions(events)
    is_pass, is_fail = attempt_masks(events)
    attempts = events[is_pass | is_fail].copy()
    attempts["completed"] = is_pass[is_pass | is_fail].astype(int)

    frames = tracking.set_index("Frame")

    rows = []
    for _, ev in attempts.iterrows():
        if pd.isna(ev["Start X"]) or pd.isna(ev["Start Y"]) or pd.isna(ev["End X"]) or pd.isna(ev["End Y"]):
            continue
        frame = int(ev["Start Frame"])
        if frame not in frames.index:
            continue
        trow = frames.loc[frame]

        team = ev["Team"].lower()
        passer_id = event_player_to_id(ev["From"])
        if passer_id is None:
            continue
        start_m = norm_to_metric(ev["Start X"], ev["Start Y"])
        end_m = norm_to_metric(ev["End X"], ev["End Y"])

        if ev["completed"]:
            target_id = event_player_to_id(ev["To"])
        else:
            target_id = infer_intended_receiver(trow, team, player_ids, start_m, end_m, passer_id, max_angle_deg)
        if target_id is None:
            continue

        tx, ty = trow.get(f"{team}_{target_id}_x"), trow.get(f"{team}_{target_id}_y")
        if tx is None or ty is None or pd.isna(tx) or pd.isna(ty):
            continue
        target_m = (float(tx), float(ty))

        control_target = pitch_control_at_point(trow, team, player_ids, target_m, params)
        if np.isnan(control_target):
            continue

        direction = directions.get((ev["Team"], int(ev["Period"])), 1)
        rows.append(
            {
                "game_pass_id": ev.name,
                "team": team,
                "period": int(ev["Period"]),
                "frame": frame,
                "passer_id": passer_id,
                "target_id": target_id,
                "completed": int(ev["completed"]),
                "control_target": control_target,
                "distance_m": float(np.hypot(target_m[0] - start_m[0], target_m[1] - start_m[1])),
                "forward_m": float((target_m[0] - start_m[0]) * direction),
                "pressure_m": nearest_opponent_dist(trow, team, player_ids, start_m),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class CompletionModel:
    model: object  # sklearn Pipeline(StandardScaler, LogisticRegression)
    features: list[str]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features])[:, 1]


def fit_completion_model(dataset: pd.DataFrame) -> CompletionModel:
    """Standardize features before fitting L2-regularized logistic regression.

    Without standardization, sklearn's L2 penalty (which punishes coefficient
    magnitude, not effect size) unfairly favors features with a small natural range
    (e.g. control_end in [0,1], needs a large coefficient) over features with a large
    range (e.g. forward_m spanning tens of meters, needs only a tiny coefficient for
    the same effect). Diagnosed via V2 calibration failure: unstandardized fit
    under-weighted forward_m (progressive, riskier passes) relative to control_end,
    overpredicting completion for forward passes specifically. See HISTORY.md.
    """
    df = dataset.dropna(subset=FEATURES)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(df[FEATURES], df["completed"])
    return CompletionModel(model=model, features=FEATURES)


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, min_bin: int = 30) -> pd.DataFrame:
    """Reliability curve data: per probability bin, mean prediction vs. empirical rate.

    Only bins with >= min_bin samples count toward V2 (docs/prototype-validation.md).
    """
    bins = np.clip((y_prob * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "bin": b,
                "n": n,
                "mean_predicted": float(y_prob[mask].mean()),
                "empirical_rate": float(y_true[mask].mean()),
                "counts_for_v2": n >= min_bin,
            }
        )
    df = pd.DataFrame(rows)
    df["abs_gap"] = (df["mean_predicted"] - df["empirical_rate"]).abs()
    return df
