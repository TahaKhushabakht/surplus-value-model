"""Counterfactual Passing Value (CPV) assembly.

Combines the three built pieces into the formula from
docs/counterfactual-passing-value.md:

    EV(a) = P(complete|a,s)*V(receive|a,s) + (1-P(complete|a,s))*V(turnover|a,s)
    CPV(a*) = EV(a*) - baseline(A(s))

  P(complete)  <- completion_model.py (physics-first, pitch control at target)
  V(receive)   <- expected_threat.py, looked up at the target location
  V(turnover)  <- expected_threat.py, MIRRORED to the opponent's perspective (see below)
  A(s)         <- every teammate present on the pitch at the pass's start frame
                  (their actual position at that frame stands in for "pass to them")

SCOPE — completed passes only on Metrica (`include_failed=False` by default).
A decision-quality metric that never sees failed decisions cannot penalize
recklessness, so including intercepted passes was attempted and MEASURED TO FAIL on
this data. Metrica leaves `To` empty on BALL LOST events, so a failed pass's intended
target must be inferred from the ray toward the interception point. That ray points
upfield, so the most-aligned teammate is preferentially the most ADVANCED one — who by
construction sits in the highest-xT zone. Measured consequence (2 games):

    target source          chosen == best option      mean xT of target
    recorded (completed)            10.3%                   0.0102
    inferred (intercepted)          24.7%                   0.0457

The inference rule is circular with the quantity CPV measures: targets are selected by
a criterion correlated with high EV, then scored on whether they had high EV. With
failed passes included, the five highest-cpv_avg passes in the dataset were ALL
interceptions and V5a's rank correlation went negative (-0.224). This is a limitation
of Metrica's event schema, not a tuning problem: the schema records where a pass ended
up, never where it was aimed.

`include_failed=True` is retained so the finding stays reproducible, but unbiased
inclusion of failed passes requires event data that records the intended target (or
intended end location) independently of the outcome. Confirming whether StatsBomb's
schema does so is a REQUIRED gate before the surplus-value model — CPV on completions
only measures decision quality *conditional on success*, which is not the quantity
player valuation needs. See NOTES.md.

CHOICE SET (docs/prototype-validation.md Gap A/A2): candidates are the passer's actual
teammates at their real positions when the pass was struck -- not an arbitrary grid of
pitch locations. This is a form of propensity trimming: it keeps the counterfactual
close to what was really available, rather than asking the model to score passes to
empty space no one occupied. The chosen action is evaluated at the intended receiver's
position at that same frame, so it is by construction one member of the choice set,
scored by exactly the same feature pipeline as its alternatives.

V(turnover) mirroring: the xT grid was built by orienting EVERY action (both teams,
oriented per event via `orient_events`/`_attack_directions`) toward x=1 before pooling
-- so it's a team-agnostic function of "value of having the ball at this position,
given your OWN attacking direction." If team A loses the ball at oriented location
(x, y), team B gains it -- and B's own oriented frame is a 180-degree rotation, so the
same location reads as (1-x, 1-y) to B. Turnover value to A is therefore the negative
of the grid evaluated at the mirrored point.

KNOWN APPROXIMATION (open, deliberate): V(turnover) is evaluated at the candidate's
TARGET location, whereas docs/counterfactual-passing-value.md specifies the
interception point. For counterfactual alternatives the interception point is
unknowable, and scoring the chosen action with a real interception point while scoring
its alternatives with an approximation would bias the very comparison CPV exists to
make. Consistency across the choice set is worth more than fidelity on one member, so
all candidates use the target location. This understates turnover cost for long
forward passes and remains an open item.

BASELINES (both computed):
  cpv_max: EV(a*) - max_{a in A(s)} EV(a)     -- harsh: 0 iff the chosen pass was the
                                                  best modeled option, negative otherwise
  cpv_avg: EV(a*) - mean_{a in A(s)} EV(a)    -- soft: compares against the average
                                                  teammate option (unweighted; a full
                                                  behavioral weighting by how often
                                                  players choose each option type is
                                                  future work, see counterfactual-
                                                  passing-value.md open question 2)
cpv_avg is the primary decision-quality metric (docs/prototype-validation.md V5);
cpv_max is a secondary "missed the clearly-best option" indicator.

DECOMPOSITION (decision vs execution quality, per the CPV doc): execution_quality =
realized_value - EV(a*), where realized_value is what the action actually returned --
for a completed pass, V(receive) at the true end location; for an intercepted one, the
turnover value at the true interception point. Because both outcomes are now included,
this is no longer biased positive by only observing successes. Note it still mixes
"pass was more/less accurate than expected" with "the receiver moved between release
and arrival", since EV(a*) is anchored to the receiver's start-frame position.

LIMITATION (documented, not hidden): for alternative candidates, "if passed to instead"
uses the teammate's actual position at the pass's start frame -- not where they might
have run had they been targeted. This is the same off-support limitation flagged in
docs/math-concepts.md §12: we cannot observe a counterfactual world where the pass
decision was different.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from completion_model import (
    FEATURES,
    CompletionModel,
    attempt_masks,
    event_player_to_id,
    infer_intended_receiver,
    metric_to_norm,
    nearest_opponent_dist,
    norm_to_metric,
)
from expected_threat import XTGrid, _attack_directions
from pitch_control import PitchControlParams, pitch_control_at_point


@dataclass
class PassCPV:
    pass_id: object
    team: str
    period: int
    frame: int
    from_player: str
    to_player: str
    completed: int
    ev_chosen: float
    ev_baseline_max: float
    ev_baseline_avg: float
    cpv_max: float
    cpv_avg: float
    execution_quality: float
    n_candidates: int
    candidates: list[dict] = field(default_factory=list)  # kept for V4 inspection/plotting


def _xt_value(xt: XTGrid, x_m: float, y_m: float, direction: int) -> float:
    x_n, y_n = metric_to_norm(x_m, y_m)
    if direction == -1:
        x_n, y_n = 1 - x_n, 1 - y_n
    return xt.value_at(x_n, y_n)


def _mirror(x_m: float, y_m: float, direction: int) -> tuple[float, float]:
    x_n, y_n = metric_to_norm(x_m, y_m)
    if direction == -1:
        x_n, y_n = 1 - x_n, 1 - y_n
    return 1 - x_n, 1 - y_n


def _candidate_positions(row: pd.Series, team: str, player_ids: list[str], exclude_id: str) -> list[tuple[str, float, float]]:
    out = []
    for pid in player_ids:
        if pid == exclude_id:
            continue
        x, y = row.get(f"{team}_{pid}_x"), row.get(f"{team}_{pid}_y")
        if pd.isna(x) or pd.isna(y):
            continue
        out.append((pid, float(x), float(y)))
    return out


def evaluate_pass(
    row: pd.Series,
    ev: pd.Series,
    team: str,
    player_ids: dict[str, list[str]],
    completion_model: CompletionModel,
    xt: XTGrid,
    direction: int,
    completed: int,
    control_params: PitchControlParams = PitchControlParams(),
    max_choice_range_m: float | None = None,
    max_angle_deg: float = 40.0,
) -> PassCPV | None:
    """Evaluate one pass attempt (completed or intercepted): chosen action + alternatives.

    max_choice_range_m: if set, candidates farther than this from the passer are
    excluded (the "choice-set width" knob for V5 sensitivity). The chosen action is
    always retained regardless, so CPV always compares like with like.
    """
    passer_id = event_player_to_id(ev["From"])
    if passer_id is None:
        return None

    start_m = norm_to_metric(ev["Start X"], ev["Start Y"])
    actual_end_m = norm_to_metric(ev["End X"], ev["End Y"])

    # Intended target: recorded for completed passes, inferred for intercepted ones.
    if completed:
        receiver_id = event_player_to_id(ev["To"])
    else:
        receiver_id = infer_intended_receiver(row, team, player_ids, start_m, actual_end_m, passer_id, max_angle_deg)
    if receiver_id is None:
        return None

    pressure_m = nearest_opponent_dist(row, team, player_ids, start_m)

    candidates = _candidate_positions(row, team, player_ids[team], passer_id)
    if max_choice_range_m is not None:
        kept = [c for c in candidates if np.hypot(c[1] - start_m[0], c[2] - start_m[1]) <= max_choice_range_m]
        # The chosen action must remain in A(s) even if the width knob would exclude it.
        if receiver_id not in [c[0] for c in kept]:
            chosen = next((c for c in candidates if c[0] == receiver_id), None)
            if chosen is not None:
                kept.append(chosen)
        candidates = kept
    if len(candidates) == 0:
        return None

    ids = np.array([c[0] for c in candidates])
    xs = np.array([c[1] for c in candidates])
    ys = np.array([c[2] for c in candidates])

    control_target = np.array([pitch_control_at_point(row, team, player_ids, (x, y), control_params) for x, y in zip(xs, ys)])
    valid = ~np.isnan(control_target)
    if valid.sum() == 0:
        return None
    ids, xs, ys, control_target = ids[valid], xs[valid], ys[valid], control_target[valid]

    # The chosen action must be scored by the same pipeline as its alternatives.
    chosen_mask = ids == receiver_id
    if chosen_mask.sum() == 0:
        return None
    chosen_idx = int(np.argmax(chosen_mask))

    distance_m = np.hypot(xs - start_m[0], ys - start_m[1])
    forward_m = (xs - start_m[0]) * direction
    feat_df = pd.DataFrame(
        {"control_target": control_target, "distance_m": distance_m, "forward_m": forward_m, "pressure_m": pressure_m}
    )
    p_complete = completion_model.predict_proba(feat_df[FEATURES])

    v_receive = np.array([_xt_value(xt, x, y, direction) for x, y in zip(xs, ys)])
    v_turnover = -np.array([xt.value_at(*_mirror(x, y, direction)) for x, y in zip(xs, ys)])
    ev_all = p_complete * v_receive + (1 - p_complete) * v_turnover

    candidate_records = [
        {"player_id": pid, "x": x, "y": y, "p_complete": p, "v_receive": vr, "v_turnover": vt, "ev": e}
        for pid, x, y, p, vr, vt, e in zip(ids, xs, ys, p_complete, v_receive, v_turnover, ev_all)
    ]

    ev_chosen = float(ev_all[chosen_idx])

    # Realized value: what the action actually returned. Completed -> value of the
    # receiving location; intercepted -> turnover value at the true interception point.
    if completed:
        realized_value = _xt_value(xt, actual_end_m[0], actual_end_m[1], direction)
    else:
        realized_value = -xt.value_at(*_mirror(actual_end_m[0], actual_end_m[1], direction))

    return PassCPV(
        pass_id=ev.name,
        team=team,
        period=int(ev["Period"]),
        frame=int(ev["Start Frame"]),
        from_player=passer_id,
        to_player=receiver_id,
        completed=int(completed),
        ev_chosen=ev_chosen,
        ev_baseline_max=float(ev_all.max()),
        ev_baseline_avg=float(ev_all.mean()),
        cpv_max=ev_chosen - float(ev_all.max()),
        cpv_avg=ev_chosen - float(ev_all.mean()),
        execution_quality=realized_value - ev_chosen,
        n_candidates=len(ids),
        candidates=candidate_records,
    )


def build_cpv_dataset(
    tracking: pd.DataFrame,
    events: pd.DataFrame,
    player_ids: dict[str, list[str]],
    completion_model: CompletionModel,
    xt: XTGrid,
    control_params: PitchControlParams = PitchControlParams(),
    max_choice_range_m: float | None = None,
    max_angle_deg: float = 40.0,
    include_failed: bool = False,
) -> tuple[pd.DataFrame, list[PassCPV]]:
    """CPV for pass attempts in one game.

    include_failed: include intercepted passes as chosen actions. Default False on
    Metrica — their intended target must be inferred, and that inference is biased
    toward high-xT targets in a way that is circular with CPV itself (see module
    docstring for the measured effect). Retained as a flag so the finding is
    reproducible, and for use with providers that record intended targets directly.
    """
    directions = _attack_directions(events)
    frames = tracking.set_index("Frame")

    is_pass, is_fail = attempt_masks(events)
    mask = (is_pass | is_fail) if include_failed else is_pass
    attempts = events[mask].copy()
    attempts["completed"] = is_pass[mask].astype(int)

    results: list[PassCPV] = []
    for _, ev in attempts.iterrows():
        if pd.isna(ev["Start X"]) or pd.isna(ev["End X"]):
            continue
        team = ev["Team"].lower()
        frame = int(ev["Start Frame"])
        if frame not in frames.index:
            continue
        direction = directions.get((ev["Team"], int(ev["Period"])), 1)
        result = evaluate_pass(
            frames.loc[frame], ev, team, player_ids, completion_model, xt, direction,
            int(ev["completed"]), control_params, max_choice_range_m, max_angle_deg,
        )
        if result is not None:
            results.append(result)

    summary = pd.DataFrame(
        [
            {
                "pass_id": r.pass_id,
                "team": r.team,
                "period": r.period,
                "frame": r.frame,
                "from_player": r.from_player,
                "to_player": r.to_player,
                "completed": r.completed,
                "ev_chosen": r.ev_chosen,
                "ev_baseline_max": r.ev_baseline_max,
                "ev_baseline_avg": r.ev_baseline_avg,
                "cpv_max": r.cpv_max,
                "cpv_avg": r.cpv_avg,
                "execution_quality": r.execution_quality,
                "n_candidates": r.n_candidates,
            }
            for r in results
        ]
    )
    return summary, results
