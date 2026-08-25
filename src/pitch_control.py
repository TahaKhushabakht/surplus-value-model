"""Pitch control model (V1 — simplified, physics-based).

Estimates, for a grid of pitch locations, the probability the attacking team would
control the ball if it arrived there right now. Two players (their positions and
current velocities) race to a location; whoever gets there first, with some
uncertainty, is more likely to control it.

Simplification vs. the full model (Spearman 2017, "Physics-based modeling of pass
probabilities in soccer"): this version compares only each team's FASTEST player to
reach a location (not a full multi-player stochastic time-integration). It is
intentionally the simpler of the two — documented in
docs/prototype-validation.md V1 as the thing to visually sanity-check before deciding
whether the added complexity of the full time-integrated model is worth it. See
docs/math-concepts.md §2.

Assumptions (documented, not hidden):
- max_speed = 5.0 m/s: a "sustained/controlled" speed, not a sprint top speed
  (players rarely move at top speed while shaping to receive/control a pass).
- reaction_time = 0.7s: time spent continuing at current velocity before reacting.
- sigma = 0.45s: controls how sharply a time advantage converts into a control
  probability (larger sigma = softer, more uncertain handoff between players).
These are standard literature defaults, not fit to this data — fitting/validating
them is future work, not part of V1.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


@dataclass
class PitchControlParams:
    max_speed: float = 5.0  # m/s
    reaction_time: float = 0.7  # s
    sigma: float = 0.45  # s


def player_ids_for_team(columns, team: str) -> list[str]:
    prefix = f"{team}_"
    ids = []
    for c in columns:
        if c.startswith(prefix) and c.endswith("_x"):
            ids.append(c[len(prefix) : -len("_x")])
    return ids


def compute_velocities(
    tracking: pd.DataFrame, player_ids: dict[str, list[str]], dt: float, smoothing_window: int = 7, max_speed_clip: float = 12.0
) -> pd.DataFrame:
    """Add {team}_{id}_vx / _vy columns via smoothed central-difference velocity.

    player_ids: {"home": [...], "away": [...]} jersey-id lists (from player_ids_for_team).
    Velocity is computed within each Period separately (no smoothing across the
    half-time gap) and clipped to max_speed_clip to suppress tracking-noise spikes.
    """
    df = tracking.copy()
    for team, ids in player_ids.items():
        for pid in ids:
            x_col, y_col = f"{team}_{pid}_x", f"{team}_{pid}_y"
            vx_col, vy_col = f"{team}_{pid}_vx", f"{team}_{pid}_vy"
            vx_parts, vy_parts = [], []
            for _, period_df in df.groupby("Period"):
                vx = period_df[x_col].diff().rolling(smoothing_window, center=True, min_periods=1).mean() / dt
                vy = period_df[y_col].diff().rolling(smoothing_window, center=True, min_periods=1).mean() / dt
                vx_parts.append(vx)
                vy_parts.append(vy)
            df[vx_col] = pd.concat(vx_parts).clip(-max_speed_clip, max_speed_clip)
            df[vy_col] = pd.concat(vy_parts).clip(-max_speed_clip, max_speed_clip)
    return df


def _team_positions_velocities(row: pd.Series, team: str, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Positions/velocities (N_on_pitch, 2) for players of `team` present (non-NaN) at this frame."""
    positions, velocities = [], []
    for pid in ids:
        x, y = row.get(f"{team}_{pid}_x"), row.get(f"{team}_{pid}_y")
        if pd.isna(x) or pd.isna(y):
            continue
        vx, vy = row.get(f"{team}_{pid}_vx", 0.0), row.get(f"{team}_{pid}_vy", 0.0)
        vx = 0.0 if pd.isna(vx) else vx
        vy = 0.0 if pd.isna(vy) else vy
        positions.append((x, y))
        velocities.append((vx, vy))
    return np.array(positions), np.array(velocities)


def _time_to_intercept(positions: np.ndarray, velocities: np.ndarray, grid: np.ndarray, params: PitchControlParams) -> np.ndarray:
    """Time for each player to reach each grid point. Returns shape (n_grid, n_players)."""
    reaction_pos = positions + velocities * params.reaction_time  # (n_players, 2)
    diff = grid[:, None, :] - reaction_pos[None, :, :]  # (n_grid, n_players, 2)
    dist = np.linalg.norm(diff, axis=-1)  # (n_grid, n_players)
    return params.reaction_time + dist / params.max_speed


def control_from_positions(
    att_pos: np.ndarray,
    def_pos: np.ndarray,
    targets: np.ndarray,
    params: PitchControlParams = PitchControlParams(),
    att_vel: np.ndarray | None = None,
    def_vel: np.ndarray | None = None,
) -> np.ndarray:
    """Attacking-team control at each target, from raw position arrays.

    Provider-agnostic entry point. StatsBomb 360 freeze frames are static snapshots
    with NO velocity information, so att_vel/def_vel default to zero there — the model
    degrades to pure distance-plus-reaction-time. That is a real fidelity loss versus
    Metrica tracking (a defender sprinting away from a location is treated as if
    stationary), and it is why StatsBomb CPV values are not comparable to Metrica ones.

    att_pos/def_pos: (n_players, 2). targets: (n_targets, 2). Returns (n_targets,).
    """
    att_pos, def_pos, targets = np.asarray(att_pos, float), np.asarray(def_pos, float), np.asarray(targets, float)
    if len(att_pos) == 0 or len(def_pos) == 0 or len(targets) == 0:
        return np.full(len(targets), np.nan)
    att_vel = np.zeros_like(att_pos) if att_vel is None else np.asarray(att_vel, float)
    def_vel = np.zeros_like(def_pos) if def_vel is None else np.asarray(def_vel, float)

    tau_att = _time_to_intercept(att_pos, att_vel, targets, params).min(axis=1)
    tau_def = _time_to_intercept(def_pos, def_vel, targets, params).min(axis=1)
    return 1.0 / (1.0 + np.exp(-(tau_def - tau_att) / params.sigma))


def pitch_control_at_point(
    row: pd.Series,
    attacking_team: str,
    player_ids: dict[str, list[str]],
    target_xy: tuple[float, float],
    params: PitchControlParams = PitchControlParams(),
) -> float:
    """P(attacking team controls a single target location), same model as the grid.

    Used by the completion model: evaluated at a pass's end location using the
    tracking frame at the pass's start. Cheaper than computing the full grid.
    """
    defending_team = "away" if attacking_team == "home" else "home"
    att_pos, att_vel = _team_positions_velocities(row, attacking_team, player_ids[attacking_team])
    def_pos, def_vel = _team_positions_velocities(row, defending_team, player_ids[defending_team])
    if len(att_pos) == 0 or len(def_pos) == 0:
        return float("nan")

    grid = np.asarray([target_xy], dtype=float)  # (1, 2)
    tau_att = _time_to_intercept(att_pos, att_vel, grid, params).min(axis=1)[0]
    tau_def = _time_to_intercept(def_pos, def_vel, grid, params).min(axis=1)[0]
    return float(1.0 / (1.0 + np.exp(-(tau_def - tau_att) / params.sigma)))


def pitch_control_grid(
    row: pd.Series,
    attacking_team: str,
    player_ids: dict[str, list[str]],
    params: PitchControlParams = PitchControlParams(),
    nx: int = 50,
    ny: int = 34,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attacking-team control probability over a grid covering the pitch.

    Returns (grid_x, grid_y, control) where control has shape (ny, nx), values in
    [0, 1] = P(attacking team controls that location).

    xlim/ylim: restrict the grid to a sub-region (metric coords). Used for zoomed
    renders — the same nx/ny spread over a smaller area gives finer contours, so a
    zoomed panel isn't blocky. Defaults to the full pitch.
    """
    defending_team = "away" if attacking_team == "home" else "home"

    att_pos, att_vel = _team_positions_velocities(row, attacking_team, player_ids[attacking_team])
    def_pos, def_vel = _team_positions_velocities(row, defending_team, player_ids[defending_team])

    x0, x1 = xlim if xlim is not None else (-PITCH_LENGTH_M / 2, PITCH_LENGTH_M / 2)
    y0, y1 = ylim if ylim is not None else (-PITCH_WIDTH_M / 2, PITCH_WIDTH_M / 2)
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (nx*ny, 2)

    tau_att = _time_to_intercept(att_pos, att_vel, grid, params).min(axis=1)
    tau_def = _time_to_intercept(def_pos, def_vel, grid, params).min(axis=1)

    control_att = 1.0 / (1.0 + np.exp(-(tau_def - tau_att) / params.sigma))
    return grid_x, grid_y, control_att.reshape(ny, nx)
