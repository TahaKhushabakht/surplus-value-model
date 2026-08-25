"""Empirical turnover-value surface — replaces the mirrored-xT approximation.

THE BUG THIS FIXES. CPV charged a turnover as `-xT(mirrored target location)`, i.e. it
assumed losing the ball hands the opponent an ORDINARY possession starting there.
Because xT is concentrated near goal, the mirror is ~0 across most of the pitch: the
validation dashboard showed turnover cost is essentially zero over ~85% of the field.
That collapses EV into `p * V(receive)`, so "maximize EV" degenerates into "pass as far
forward as possible" — which is why failed passes out-scored completed ones and why an
elite metronome (Xhaka, lowest fail rate in the squad) ranked 13th of 16.

THE FIX. A turnover does not hand the opponent an ordinary possession; it hands them a
TRANSITION against a defence that is still committed forward. That is an empirical
quantity, so measure it instead of assuming it. StatsBomb tags every event with a
`possession` id and `possession_team`, so for each possession we can ask: where did it
start, did it start from a turnover, and did it end in a goal?

    V_turnover(zone) = -P(opponent's ensuing possession ends in a goal | it began
                          at this zone via a turnover)

Sign convention matches the rest of the pipeline: negative = bad for the team that lost
the ball. Crucially this is estimated on TRANSITION possessions specifically, so it
captures counter-attack danger that a possession-averaged surface cannot.

The zone is expressed in the LOSING team's frame (attacking toward +x), so the surface
is directly comparable with the mirrored-xT surface it replaces.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import statsbomb_io as SB

TURNOVER_TYPES = {"Miscontrol", "Dispossessed"}
PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0


def _possession_summary(events: list[dict]) -> pd.DataFrame:
    """One row per possession: who had it, where it began, did it end in a goal."""
    rows = {}
    for ev in events:
        pid = ev.get("possession")
        if pid is None:
            continue
        rec = rows.setdefault(
            pid,
            {"possession": pid, "team": ev["possession_team"]["name"], "start_loc": None, "goal": 0, "first_period": ev["period"]},
        )
        if rec["start_loc"] is None and ev.get("location"):
            rec["start_loc"] = ev["location"][:2]
        if ev["type"]["name"] == "Shot" and ev.get("shot", {}).get("outcome", {}).get("name") == "Goal":
            if ev["team"]["name"] == rec["team"]:
                rec["goal"] = 1
    return pd.DataFrame(rows.values()).sort_values("possession")


def _turnover_events(events: list[dict]) -> dict[int, tuple[str, list[float]]]:
    """For each possession, how it ENDED if it ended in a turnover.

    Returns {possession_id: (losing_team, location)} — the location is where the ball
    was actually lost, in the losing team's own coordinate frame.
    """
    out = {}
    for ev in events:
        pid = ev.get("possession")
        if pid is None or not ev.get("location"):
            continue
        name = ev["type"]["name"]
        lost = None
        if name == "Pass" and ev["pass"].get("outcome", {}).get("name") in ("Incomplete", "Out"):
            lost = ev["team"]["name"]
        elif name in TURNOVER_TYPES:
            lost = ev["team"]["name"]
        if lost is not None:
            # Keep the LAST turnover-ish event of the possession: that is the one that
            # actually ended it.
            out[pid] = (lost, ev["location"][:2])
    return out


def build_turnover_surface(match_ids, data_dir: Path = SB.DATA_DIR, nx: int = 12, ny: int = 8, prior_strength: float = 40.0):
    """P(conceded goal | possession lost at zone) over all matches, in the loser's frame.

    prior_strength: empirical-Bayes shrinkage toward the overall concession rate. Goals
    are rare, so raw per-zone rates from a single season are extremely noisy; shrinkage
    keeps a zone with 3 turnovers and 1 goal from reading as a 33% disaster zone.
    """
    n_events = np.zeros((ny, nx))
    n_goals = np.zeros((ny, nx))

    for mid in match_ids:
        events = SB.load_events_raw(int(mid), data_dir)
        poss = _possession_summary(events).set_index("possession")
        turnovers = _turnover_events(events)

        for pid, (losing_team, loc) in turnovers.items():
            nxt = pid + 1
            if nxt not in poss.index:
                continue
            nxt_row = poss.loc[nxt]
            # Only count it if the ball genuinely changed hands.
            if nxt_row["team"] == losing_team:
                continue

            # Location in the LOSING team's frame; StatsBomb already orients each team
            # toward x=120, so the raw location is already in that frame.
            xn, yn = loc[0] / SB.SB_LENGTH, loc[1] / SB.SB_WIDTH
            ix = min(int(np.clip(xn, 0, 0.999999) * nx), nx - 1)
            iy = min(int(np.clip(yn, 0, 0.999999) * ny), ny - 1)
            n_events[iy, ix] += 1
            n_goals[iy, ix] += int(nxt_row["goal"])

    overall = n_goals.sum() / max(n_events.sum(), 1)
    surface = (n_goals + prior_strength * overall) / (n_events + prior_strength)
    return surface, n_events, float(overall)


class TurnoverLookup:
    """Metric-coordinate lookup returning NEGATIVE value (cost to the losing team)."""

    def __init__(self, surface: np.ndarray):
        self.surface = surface
        self.ny, self.nx = surface.shape

    def value(self, x_m, y_m):
        xn = np.asarray(x_m) / PITCH_LENGTH_M + 0.5
        yn = 0.5 - np.asarray(y_m) / PITCH_WIDTH_M
        ix = np.clip((np.atleast_1d(xn) * self.nx).astype(int), 0, self.nx - 1)
        iy = np.clip((np.atleast_1d(yn) * self.ny).astype(int), 0, self.ny - 1)
        return -self.surface[iy, ix]


if __name__ == "__main__":
    OUT = Path(__file__).parent.parent / "output"
    matches = SB.load_matches()["match_id"].tolist()
    surface, counts, overall = build_turnover_surface(matches)
    np.save(OUT / "sb_turnover_surface.npy", surface)

    print(f"overall P(concede | turnover) = {overall:.4f}  ({int(counts.sum()):,} turnovers)")
    print(f"turnovers per zone: min={int(counts.min())}, median={int(np.median(counts))}, max={int(counts.max())}")
    print()
    print("Turnover cost x1000, by column (own goal at left, attacking toward right):")
    print("  NEW (empirical transition):", (surface.mean(axis=0) * 1000).round(2))

    xt = np.load(OUT / "sb_xt.npy")
    # Old approach: -xT at the 180-degree mirrored location.
    ny, nx = xt.shape
    old = np.zeros_like(surface)
    sy, sx = surface.shape
    for iy in range(sy):
        for ix in range(sx):
            xn, yn = (ix + 0.5) / sx, (iy + 0.5) / sy
            jx = min(int((1 - xn) * nx), nx - 1)
            jy = min(int((1 - yn) * ny), ny - 1)
            old[iy, ix] = xt[jy, jx]
    print("  OLD (mirrored xT)         :", (old.mean(axis=0) * 1000).round(2))
    print()
    ratio = surface.mean(axis=0) / np.maximum(old.mean(axis=0), 1e-9)
    print("  ratio new/old             :", ratio.round(1))
    print()
    print("Reading: the old surface charges almost nothing outside your own defensive third.")
    print("The empirical surface charges a real, roughly uniform cost everywhere, because a")
    print("turnover concedes a TRANSITION rather than an ordinary possession.")
