"""Parser for Metrica Sports' open sample tracking/event data (CSV format).

Covers Sample_Game_1 and Sample_Game_2, which share the standard Metrica raw-CSV
layout: a 3-row header (team name, jersey number, column label), then one row per
frame with Period/Frame/Time and (x, y) pairs per player and the ball, normalized
to [0, 1] (x: left->right, y: top->bottom of the pitch as broadcast).

Sample_Game_3 ships in a different format (tracking as fixed-width text + JSON
events) and is NOT handled here.
"""

from pathlib import Path

import pandas as pd

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def _build_columns(filepath: Path, teamname: str) -> list[str]:
    with open(filepath) as f:
        f.readline()  # team-name row (redundant with `teamname` arg)
        f.readline()  # jersey-number row (redundant, we pull numbers from labels below)
        label_row = f.readline().strip().split(",")

    columns = list(label_row[:3])  # Period, Frame, Time [s]
    i = 3
    while i < len(label_row) and label_row[i] != "":
        label = label_row[i]
        # .strip(): some Metrica headers carry a stray space ("Player 26"), which would
        # otherwise produce a tracking id (" 26") that never matches the event id ("26").
        tag = "ball" if label == "Ball" else f"{teamname}_{label.replace('Player', '').strip()}"
        columns.append(f"{tag}_x")
        columns.append(f"{tag}_y")
        i += 2
    return columns


def read_tracking_data(filepath: Path, teamname: str) -> pd.DataFrame:
    """Read one team's raw tracking CSV into a tidy DataFrame.

    Columns: Period, Frame, Time [s], {teamname}_{jersey}_x/_y for each player,
    ball_x, ball_y. Coordinates are still normalized [0, 1] at this point.
    """
    columns = _build_columns(filepath, teamname)
    df = pd.read_csv(filepath, skiprows=3, header=None, names=columns, usecols=range(len(columns)))
    return df


def merge_tracking_data(home_df: pd.DataFrame, away_df: pd.DataFrame) -> pd.DataFrame:
    """Join home and away tracking on frame; ball position is taken from home_df
    (both files record the same ball trajectory)."""
    away = away_df.drop(columns=["ball_x", "ball_y"])
    return home_df.merge(away, on=["Period", "Frame", "Time [s]"], how="inner")


def to_metric_coordinates(
    df: pd.DataFrame, pitch_length: float = PITCH_LENGTH_M, pitch_width: float = PITCH_WIDTH_M
) -> pd.DataFrame:
    """Convert normalized [0, 1] coordinates to meters, centered at the pitch center
    (x in [-length/2, length/2], y in [-width/2, width/2])."""
    df = df.copy()
    x_cols = [c for c in df.columns if c.endswith("_x")]
    y_cols = [c for c in df.columns if c.endswith("_y")]
    df[x_cols] = (df[x_cols] - 0.5) * pitch_length
    df[y_cols] = (df[y_cols] - 0.5) * pitch_width
    return df


def read_events(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath)


def load_game(data_dir: Path, game_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load tracking (merged, metric coordinates) and events for Sample_Game_{game_id}.

    Only game_id in {1, 2} is supported (see module docstring).
    """
    game_dir = Path(data_dir) / f"Sample_Game_{game_id}"
    prefix = f"Sample_Game_{game_id}"

    home = read_tracking_data(game_dir / f"{prefix}_RawTrackingData_Home_Team.csv", "home")
    away = read_tracking_data(game_dir / f"{prefix}_RawTrackingData_Away_Team.csv", "away")
    tracking = merge_tracking_data(home, away)
    tracking = to_metric_coordinates(tracking)

    events = read_events(game_dir / f"{prefix}_RawEventsData.csv")

    return tracking, events
