"""Expected Threat (xT) value surface.

Produces V(location) = probability that possession starting at a location leads to a
goal soon. This is the value surface the CPV formula needs for V(receive) and
V(turnover). See docs/math-concepts.md §5 (Markov reward process / Bellman) and
docs/prototype-validation.md V3 for the validation criterion.

Model (Karun Singh's Expected Threat, with an explicit possession-loss term). In each
grid zone z a player with the ball either shoots, moves (passes) and keeps possession,
or loses possession:

    xT(z) = s(z)*g(z)  +  m(z) * sum_z' T(z->z') * xT(z')     [ + l(z)*0 ]

where
    s(z) = P(action in z is a shot),      g(z) = P(shot from z scores),
    m(z) = P(action in z is a completed move -> possession retained),
    l(z) = P(action in z loses possession) = 1 - s(z) - m(z)   (reward 0),
    T(z->z') = P(a move from z ends in z').

The possession-loss term l(z) is essential. Without it (m = 1 - s), passes make the
ball effectively immortal: value iteration smears the goal-reward almost uniformly
across the whole pitch, because from anywhere you can eventually pass your way to a
shot. The l(z) leak means deep zones -- many risky passes from goal -- attenuate toward
zero, while zones near goal keep most of their value. That leak is what produces the
gradient a real xT surface has.

The equation is recursive (a zone's value depends on the values of zones you can pass
to), so it is solved by value iteration: initialise xT=0 and repeatedly apply the
right-hand side until it converges. l(z) > 0 guarantees convergence (the move operator
now strictly attenuates each step).

PROTOTYPE SCOPE:
- Coarse 12x8 grid: with only 2 parseable Metrica games (~1.6k passes) a finer grid
  would leave too few actions per zone to estimate T(z->z') reliably.
- Moves = PASS events, shots = SHOT events. Metrica has no explicit carry/dribble
  event, so ball carries are not modelled as separate moves here.
- Estimates are pooled across games AFTER orienting every attack toward x=1.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class XTGrid:
    nx: int
    ny: int
    xt: np.ndarray          # (ny, nx) value surface, normalized coords, attack -> +x
    shot_prob: np.ndarray   # s(z)
    goal_prob: np.ndarray   # g(z)
    move_prob: np.ndarray   # m(z)
    loss_prob: np.ndarray   # l(z) = 1 - s(z) - m(z)
    n_actions: np.ndarray   # action counts per zone (for diagnostics / sparsity)

    def value_at(self, x: float, y: float) -> float:
        """xT at a normalized (x, y) location, coords oriented so attack is toward x=1."""
        ix = min(int(np.clip(x, 0, 0.999999) * self.nx), self.nx - 1)
        iy = min(int(np.clip(y, 0, 0.999999) * self.ny), self.ny - 1)
        return float(self.xt[iy, ix])


def _attack_directions(events: pd.DataFrame) -> dict[tuple[str, int], int]:
    """For each (team, period), infer which goal they attack from their shots.

    Returns +1 if the team attacks toward x=1 (no flip needed) or -1 if toward x=0
    (coordinates must be flipped). Falls back to the opposite of the team's other
    period, then to +1, when a (team, period) has no shots.
    """
    shots = events[events["Type"] == "SHOT"]
    directions: dict[tuple[str, int], int] = {}
    for (team, period), grp in shots.groupby(["Team", "Period"]):
        # Shooting from x>0.5 means attacking the x=1 goal.
        directions[(team, period)] = 1 if grp["Start X"].mean() > 0.5 else -1

    teams = events["Team"].dropna().unique()
    for team in teams:
        for period in (1, 2):
            if (team, period) in directions:
                continue
            other = 2 if period == 1 else 1
            directions[(team, period)] = -directions.get((team, other), -1)  # teams switch ends
    return directions


def orient_events(events: pd.DataFrame) -> pd.DataFrame:
    """Flip coordinates so every attacking action points toward x=1.

    Rotating 180 degrees means x -> 1-x and y -> 1-y for the periods/teams that attack
    the x=0 goal. Must be applied per game (each game's teams switch ends independently).
    """
    directions = _attack_directions(events)
    df = events.copy()
    for (team, period), d in directions.items():
        if d == 1:
            continue
        mask = (df["Team"] == team) & (df["Period"] == period)
        for col in ("Start X", "End X"):
            df.loc[mask, col] = 1 - df.loc[mask, col]
        for col in ("Start Y", "End Y"):
            df.loc[mask, col] = 1 - df.loc[mask, col]
    return df


def _zone_index(x: float, y: float, nx: int, ny: int) -> tuple[int, int]:
    ix = min(int(np.clip(x, 0, 0.999999) * nx), nx - 1)
    iy = min(int(np.clip(y, 0, 0.999999) * ny), ny - 1)
    return ix, iy


def build_xt(events: pd.DataFrame, nx: int = 12, ny: int = 8, max_iter: int = 200, tol: float = 1e-7) -> XTGrid:
    """Estimate s, g, T from oriented events and solve for xT by value iteration.

    `events` should be raw (this function orients them internally). Pool multiple games
    by concatenating their raw events first, but orient per game -- so call this once
    per game and average, OR pass a frame already oriented per game. Here we orient
    inside, which is correct only if `events` is a single game. For pooling, use
    build_xt_pooled().
    """
    ev = orient_events(events)
    return _build_xt_from_oriented(ev, nx, ny, max_iter, tol)


def build_xt_pooled(events_by_game: list[pd.DataFrame], nx: int = 12, ny: int = 8, max_iter: int = 200, tol: float = 1e-7) -> XTGrid:
    """Build xT from several games, orienting each game separately then pooling actions."""
    oriented = [orient_events(ev) for ev in events_by_game]
    pooled = pd.concat(oriented, ignore_index=True)
    return _build_xt_from_oriented(pooled, nx, ny, max_iter, tol)


def _build_xt_from_oriented(ev: pd.DataFrame, nx: int, ny: int, max_iter: int, tol: float) -> XTGrid:
    n_shots = np.zeros((ny, nx))
    n_goals = np.zeros((ny, nx))
    n_moves = np.zeros((ny, nx))
    n_losses = np.zeros((ny, nx))
    # Transition tensor T[iy, ix, jy, jx] = count of moves from (ix,iy) to (jx,jy).
    trans = np.zeros((ny, nx, ny, nx))

    shots = ev[ev["Type"] == "SHOT"]
    for _, s in shots.iterrows():
        if pd.isna(s["Start X"]) or pd.isna(s["Start Y"]):
            continue
        ix, iy = _zone_index(s["Start X"], s["Start Y"], nx, ny)
        n_shots[iy, ix] += 1
        if isinstance(s["Subtype"], str) and "GOAL" in s["Subtype"].upper():
            n_goals[iy, ix] += 1

    passes = ev[ev["Type"] == "PASS"]
    for _, p in passes.iterrows():
        if pd.isna(p["Start X"]) or pd.isna(p["End X"]):
            continue
        ix, iy = _zone_index(p["Start X"], p["Start Y"], nx, ny)
        jx, jy = _zone_index(p["End X"], p["End Y"], nx, ny)
        n_moves[iy, ix] += 1
        trans[iy, ix, jy, jx] += 1

    # Possession-loss events (turnovers): reward 0, absorbing. This is the leakage term.
    losses = ev[ev["Type"] == "BALL LOST"]
    for _, x in losses.iterrows():
        if pd.isna(x["Start X"]) or pd.isna(x["Start Y"]):
            continue
        ix, iy = _zone_index(x["Start X"], x["Start Y"], nx, ny)
        n_losses[iy, ix] += 1

    n_actions = n_shots + n_moves + n_losses
    with np.errstate(divide="ignore", invalid="ignore"):
        shot_prob = np.where(n_actions > 0, n_shots / n_actions, 0.0)
        move_prob = np.where(n_actions > 0, n_moves / n_actions, 0.0)
        loss_prob = np.where(n_actions > 0, n_losses / n_actions, 0.0)
        goal_prob = np.where(n_shots > 0, n_goals / n_shots, 0.0)

    # Normalize each origin zone's transition counts into a probability distribution.
    trans_prob = np.zeros_like(trans)
    for iy in range(ny):
        for ix in range(nx):
            total = trans[iy, ix].sum()
            if total > 0:
                trans_prob[iy, ix] = trans[iy, ix] / total

    # Value iteration: xT = s*g + m * (T @ xT)
    xt = np.zeros((ny, nx))
    shoot_reward = shot_prob * goal_prob
    for _ in range(max_iter):
        move_value = np.tensordot(trans_prob, xt, axes=([2, 3], [0, 1]))  # (ny, nx)
        new_xt = shoot_reward + move_prob * move_value
        if np.max(np.abs(new_xt - xt)) < tol:
            xt = new_xt
            break
        xt = new_xt

    return XTGrid(
        nx=nx, ny=ny, xt=xt, shot_prob=shot_prob, goal_prob=goal_prob,
        move_prob=move_prob, loss_prob=loss_prob, n_actions=n_actions,
    )
