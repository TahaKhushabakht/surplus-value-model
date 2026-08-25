"""StatsBomb open-data loader — events + 360 freeze frames.

Converts StatsBomb's JSON into the tabular shape the existing pipeline expects, so
expected_threat / completion_model / cpv / role_space port over with minimal change.

KEY DIFFERENCES FROM METRICA (all verified against real files, not assumed):

1. Coordinates are 120 x 80 with the origin at a corner. Converted here to the same
   metric, centre-origin frame the pipeline already uses (105 x 68, origin at centre)
   so no downstream module needs to know which provider the data came from.

2. Attacking direction is ALREADY NORMALIZED — every team attacks toward x=120 in both
   periods (verified: shot x-means are ~100-110 for both teams in both halves). The
   whole per-team/per-period flipping apparatus that Metrica required
   (`orient_events` / `_attack_directions`) is unnecessary here, and applying it would
   actively corrupt the data.

3. Pass outcome and intended target are recorded SEPARATELY. `pass.recipient` is
   populated for ~85% of INCOMPLETE passes, so a failed pass's intended target is
   known rather than inferred. This clears the blocking gate that forced CPV to
   completed-passes-only on Metrica (HISTORY.md §14).

4. 360 freeze frames are ANONYMOUS: each entry has only `teammate` / `actor` /
   `keeper` / `location` — no player id. Consequences:
     - Candidate positions (what CPV needs) are fine; identity is irrelevant there.
     - The CHOSEN target's position must be matched to a freeze-frame teammate by
       proximity to `pass.end_location`. `end_location` is outcome-contaminated for
       incomplete passes (it sits nearer an opponent than a teammate), so this match
       carries a bias risk of exactly the kind that invalidated Metrica's inference.
       `target_xt_bias_check()` measures it; do not use failed passes in CPV until
       that check passes.

5. Freeze frames cover ~90% of passes but only players inside the BROADCAST camera
   view: median 18-19 of 22 players, ~34% of pass frames below 18. Players outside
   `visible_area` are absent, not flagged — so pitch control will read off-camera
   defenders as empty space. `n_visible` is carried on every row so this can be
   controlled for rather than silently absorbed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "statsbomb"

SB_LENGTH, SB_WIDTH = 120.0, 80.0
PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0


def sb_to_metric(x: float, y: float) -> tuple[float, float]:
    """StatsBomb (0-120, 0-80, corner origin) -> metric (105 x 68, centre origin).

    y is flipped so that positive y is the same side as the rest of the pipeline
    (StatsBomb y increases downward).
    """
    mx = (x / SB_LENGTH - 0.5) * PITCH_LENGTH_M
    my = (0.5 - y / SB_WIDTH) * PITCH_WIDTH_M
    return mx, my


def load_matches(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    matches = json.loads((Path(data_dir) / "matches.json").read_text(encoding="utf-8"))
    return pd.DataFrame(
        [
            {
                "match_id": m["match_id"],
                "date": m.get("match_date"),
                "home_team": m["home_team"]["home_team_name"],
                "away_team": m["away_team"]["away_team_name"],
            }
            for m in matches
        ]
    )


def load_events_raw(match_id: int, data_dir: Path = DATA_DIR) -> list[dict]:
    return json.loads((Path(data_dir) / "events" / f"{match_id}.json").read_text(encoding="utf-8"))


def load_frames_raw(match_id: int, data_dir: Path = DATA_DIR) -> dict[str, dict]:
    path = Path(data_dir) / "three-sixty" / f"{match_id}.json"
    if not path.exists():
        return {}
    return {fr["event_uuid"]: fr for fr in json.loads(path.read_text(encoding="utf-8"))}


def _match_target_position(frame: dict, end_xy: tuple[float, float]) -> tuple[tuple[float, float] | None, float]:
    """Freeze-frame teammate nearest the pass end location, in metric coords.

    Freeze frames carry no identities, so this proximity match is how the recorded
    recipient is located on the pitch. Returns (position, distance) with distance
    exposed so callers can filter implausible matches instead of trusting all of them.
    """
    best, best_d = None, np.inf
    for player in frame["freeze_frame"]:
        if not player["teammate"] or player.get("actor"):
            continue
        px, py = sb_to_metric(*player["location"][:2])
        d = float(np.hypot(px - end_xy[0], py - end_xy[1]))
        if d < best_d:
            best, best_d = (px, py), d
    return best, best_d


def possession_goal_map(events: list[dict]) -> dict[int, int]:
    """{possession_id: 1 if the possession team scored during it}.

    Lets a pass be linked to what its possession actually produced, which is what
    makes V(receive) empirically estimable rather than assumed.
    """
    team_of, scored = {}, {}
    for ev in events:
        pid = ev.get("possession")
        if pid is None:
            continue
        team_of.setdefault(pid, ev["possession_team"]["name"])
        if ev["type"]["name"] == "Shot" and ev.get("shot", {}).get("outcome", {}).get("name") == "Goal":
            if ev["team"]["name"] == team_of[pid]:
                scored[pid] = 1
    return {pid: scored.get(pid, 0) for pid in team_of}


def build_pass_table(match_id: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per pass that has a 360 frame, with candidate positions attached.

    Candidate positions are ALL freeze-frame teammates (excluding the passer), which is
    the choice set A(s). The chosen target is matched by proximity — see module note 4.
    """
    events = load_events_raw(match_id, data_dir)
    frames = load_frames_raw(match_id, data_dir)
    if not frames:
        return pd.DataFrame()
    goal_map = possession_goal_map(events)

    rows = []
    for ev in events:
        if ev["type"]["name"] != "Pass" or "location" not in ev:
            continue
        frame = frames.get(ev["id"])
        if frame is None:
            continue
        p = ev["pass"]
        outcome = p.get("outcome", {}).get("name")
        # "Incomplete" = intercepted/cut out. Out/Offside/Injury Clearance are excluded:
        # they are not decisions between teammates in the same sense.
        if outcome is not None and outcome != "Incomplete":
            continue
        completed = int(outcome is None)

        start = sb_to_metric(*ev["location"][:2])
        end = sb_to_metric(*p["end_location"][:2])

        teammates, opponents = [], []
        for pl in frame["freeze_frame"]:
            pos = sb_to_metric(*pl["location"][:2])
            if pl.get("actor"):
                continue
            (teammates if pl["teammate"] else opponents).append(pos)
        if not teammates or not opponents:
            continue

        target, target_dist = _match_target_position(frame, end)
        if target is None:
            continue

        rows.append(
            {
                "match_id": match_id,
                "event_id": ev["id"],
                "period": ev["period"],
                "minute": ev["minute"],
                "team": ev["team"]["name"],
                "passer_id": ev["player"]["id"],
                "passer": ev["player"]["name"],
                "recipient_id": p.get("recipient", {}).get("id"),
                "recipient": p.get("recipient", {}).get("name"),
                "completed": completed,
                "start_x": start[0], "start_y": start[1],
                "end_x": end[0], "end_y": end[1],
                "target_x": target[0], "target_y": target[1],
                "target_match_dist": target_dist,
                "teammates": teammates,
                "opponents": opponents,
                "n_visible": len(frame["freeze_frame"]),
                "possession": ev.get("possession"),
                "possession_scored": goal_map.get(ev.get("possession"), 0),
            }
        )
    return pd.DataFrame(rows)


def load_all_passes(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Every pass in the season, with NO 360-frame requirement.

    Role features (passing network, territory, style) need only event data, so gating
    them on freeze-frame availability would discard ~10-15% of passes for nothing.
    Coordinates are metric and already attack-oriented (+x), per module note 2.
    """
    rows = []
    for mid in load_matches(data_dir)["match_id"]:
        for ev in load_events_raw(int(mid), data_dir):
            if ev["type"]["name"] != "Pass" or not ev.get("location"):
                continue
            p = ev["pass"]
            outcome = p.get("outcome", {}).get("name")
            sx, sy = sb_to_metric(*ev["location"][:2])
            ex, ey = sb_to_metric(*p["end_location"][:2])
            rows.append(
                {
                    "match_id": int(mid),
                    "team": ev["team"]["name"],
                    "passer_id": ev["player"]["id"],
                    "passer": ev["player"]["name"],
                    "recipient_id": p.get("recipient", {}).get("id"),
                    "completed": int(outcome is None),
                    "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
                }
            )
    return pd.DataFrame(rows)


def frame_alignment(match_id: int, data_dir: Path = DATA_DIR) -> dict:
    """How many of a match's events actually have a matching 360 frame.

    Some open-data 360 files are mismatched upstream: the file exists and is full of
    frames, but its `event_uuid`s belong to a DIFFERENT match, so overlap is exactly
    zero. Verified on 3 of Leverkusen's 34 matches (3895158, 3895266, 3895309). Without
    this check the loader silently yields nothing for those matches, which looks like a
    parsing bug rather than bad input.
    """
    frames = load_frames_raw(match_id, data_dir)
    event_ids = {e["id"] for e in load_events_raw(match_id, data_dir)}
    overlap = len(event_ids & set(frames))
    return {
        "match_id": match_id,
        "n_events": len(event_ids),
        "n_frames": len(frames),
        "overlap": overlap,
        "mismatched": bool(frames) and overlap == 0,
    }


def load_season_passes(data_dir: Path = DATA_DIR, limit: int | None = None, verbose: bool = True) -> pd.DataFrame:
    matches = load_matches(data_dir)
    if limit:
        matches = matches.head(limit)

    tables, skipped = [], []
    for mid in matches["match_id"]:
        table = build_pass_table(int(mid), data_dir)
        if len(table):
            tables.append(table)
        else:
            skipped.append(frame_alignment(int(mid), data_dir))

    if verbose and skipped:
        for s in skipped:
            reason = "360 file belongs to another match (upstream mismatch)" if s["mismatched"] else "no usable frames"
            print(f"  skipped {s['match_id']}: {reason} ({s['n_frames']} frames, {s['overlap']} overlap)")

    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def to_xt_event_frame(match_ids: list[int], data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Adapt StatsBomb events into the normalized-coordinate frame `expected_threat`
    expects, so the validated value-iteration code is reused rather than rewritten.

    Coordinates are emitted already oriented toward x=1 (StatsBomb normalizes attacking
    direction), so callers must use `_build_xt_from_oriented` directly and must NOT
    call `orient_events` — flipping here would corrupt the data.

    Possession-loss events (the l(z) leak term, without which value iteration smears xT
    uniformly — HISTORY.md §10) come from incomplete passes, miscontrols and
    dispossessions.
    """
    rows = []
    for mid in match_ids:
        for ev in load_events_raw(int(mid), data_dir):
            name = ev["type"]["name"]
            if "location" not in ev or not ev["location"]:
                continue
            sx, sy = ev["location"][0] / SB_LENGTH, ev["location"][1] / SB_WIDTH

            if name == "Shot":
                is_goal = ev.get("shot", {}).get("outcome", {}).get("name") == "Goal"
                rows.append({"Type": "SHOT", "Subtype": "GOAL" if is_goal else "", "Team": ev["team"]["name"],
                             "Period": ev["period"], "Start X": sx, "Start Y": sy, "End X": np.nan, "End Y": np.nan})
            elif name == "Pass":
                end = ev["pass"]["end_location"]
                ex, ey = end[0] / SB_LENGTH, end[1] / SB_WIDTH
                if ev["pass"].get("outcome") is None:
                    rows.append({"Type": "PASS", "Subtype": "", "Team": ev["team"]["name"],
                                 "Period": ev["period"], "Start X": sx, "Start Y": sy, "End X": ex, "End Y": ey})
                elif ev["pass"]["outcome"]["name"] in ("Incomplete", "Out"):
                    rows.append({"Type": "BALL LOST", "Subtype": "INTERCEPTION", "Team": ev["team"]["name"],
                                 "Period": ev["period"], "Start X": sx, "Start Y": sy, "End X": ex, "End Y": ey})
            elif name in ("Miscontrol", "Dispossessed"):
                rows.append({"Type": "BALL LOST", "Subtype": "", "Team": ev["team"]["name"],
                             "Period": ev["period"], "Start X": sx, "Start Y": sy, "End X": np.nan, "End Y": np.nan})
    return pd.DataFrame(rows)


POSITION_GROUP_MAP = {
    "Goalkeeper": "GK",
    "Center Back": "DEF", "Left Center Back": "DEF", "Right Center Back": "DEF",
    "Left Back": "DEF", "Right Back": "DEF", "Left Wing Back": "DEF", "Right Wing Back": "DEF",
    "Center Defensive Midfield": "MID", "Left Defensive Midfield": "MID", "Right Defensive Midfield": "MID",
    "Left Center Midfield": "MID", "Right Center Midfield": "MID",
    "Left Midfield": "MID", "Right Midfield": "MID",
    "Center Attacking Midfield": "MID", "Left Attacking Midfield": "MID", "Right Attacking Midfield": "MID",
    "Center Forward": "FWD", "Left Center Forward": "FWD", "Right Center Forward": "FWD",
    "Left Wing": "FWD", "Right Wing": "FWD",
}


def player_position_labels(match_ids, data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Real position label per player (GK/DEF/MID/FWD), from StatsBomb's own
    event-level `position` tags -- NOT inferred from spatial heuristics.

    Superseded a spatial approach (role_space.infer_position, mean_x thresholds) that
    failed on this exact dataset: Bayer Leverkusen's high-line system pushes their
    centre-backs' average pass origin (-8m to -2m) into the range other teams' central
    midfielders occupy, so no threshold -- absolute OR per-team-relative-tertile --
    could separate DEF from MID correctly for both Leverkusen and everyone else at
    once (HISTORY.md #29). StatsBomb tags the actual tactical position on every event
    for identified players, which sidesteps the problem rather than refining around it.

    Uses each player's MODAL tagged position across all their events in these matches
    (a player may be tagged at several granular positions -- e.g. "Left Center Back"
    vs "Center Back" -- across a season; the mode is stable and maps cleanly via
    POSITION_GROUP_MAP). Returns {} for players with no position-tagged events.
    """
    from collections import Counter

    tallies: dict[str, Counter] = {}
    for mid in match_ids:
        for ev in load_events_raw(int(mid), data_dir):
            if "position" not in ev or "player" not in ev:
                continue
            pid = str(ev["player"]["id"])
            tallies.setdefault(pid, Counter())[ev["position"]["name"]] += 1

    out = {}
    for pid, counts in tallies.items():
        modal_name = counts.most_common(1)[0][0]
        out[pid] = POSITION_GROUP_MAP.get(modal_name, "UNK")
    return out


def target_xt_bias_check(passes: pd.DataFrame, xt_value_fn) -> pd.DataFrame:
    """Does proximity-matching pick systematically higher-value targets when a pass FAILED?

    This is the check that Metrica's inference failed catastrophically (inferred targets
    had 4.5x the xT of recorded ones, because the selection rule pointed upfield). If
    the completed/incomplete gap here is comparable, failed passes must stay out of CPV.
    """
    out = passes.copy()
    out["target_xt"] = [xt_value_fn(x, y) for x, y in zip(out["target_x"], out["target_y"])]
    grouped = out.groupby("completed").agg(
        n=("target_xt", "size"),
        mean_target_xt=("target_xt", "mean"),
        median_match_dist=("target_match_dist", "median"),
        mean_n_visible=("n_visible", "mean"),
    )
    grouped.index = grouped.index.map({0: "incomplete", 1: "completed"})
    return grouped
