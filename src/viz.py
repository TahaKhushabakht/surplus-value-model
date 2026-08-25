"""Visualization helpers for pitch control and tracking frames."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Rectangle

from pitch_control import PITCH_LENGTH_M, PITCH_WIDTH_M

# Project-level output dir for figures. Gitignored (see .gitignore: output/).
FIGURES_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"


def save_figure(fig, name: str, dpi: int = 95) -> Path:
    """Save a figure to output/figures/<name>, creating the dir if needed."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def draw_pitch(ax, color: str = "black", lw: float = 1.0, set_limits: bool = True):
    """Draw full pitch markings in metric coords (origin at center, 105x68m).

    Penalty boxes and goals matter more than they look: on a zoomed panel they are the
    only cue for *where* on the pitch the action is. `set_limits=False` leaves the view
    window alone so zoom_to_action can control it.
    """
    hl, hw = PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2
    ax.plot([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw], color=color, lw=lw, zorder=1)
    ax.plot([0, 0], [-hw, hw], color=color, lw=lw, zorder=1)
    ax.add_patch(plt.Circle((0, 0), 9.15, color=color, fill=False, lw=lw, zorder=1))
    ax.plot([0], [0], marker="o", ms=2, color=color, zorder=1)

    for sign in (-1, 1):
        goal_line = sign * hl
        # Penalty area: 16.5m deep, 40.32m wide. Six-yard box: 5.5m deep, 18.32m wide.
        for depth, half_width in ((16.5, 20.16), (5.5, 9.16)):
            inner = goal_line - sign * depth
            ax.plot(
                [goal_line, inner, inner, goal_line],
                [-half_width, -half_width, half_width, half_width],
                color=color, lw=lw, zorder=1,
            )
        spot = goal_line - sign * 11.0
        ax.plot([spot], [0], marker="o", ms=2, color=color, zorder=1)
        # Penalty arc: only the portion outside the box is drawn in real markings.
        theta = 127.0 if sign > 0 else -53.0
        ax.add_patch(
            Arc((spot, 0), 2 * 9.15, 2 * 9.15, angle=theta, theta1=0, theta2=106, color=color, lw=lw, zorder=1)
        )
        # Goal mouth (7.32m wide), drawn just outside the goal line.
        ax.plot(
            [goal_line, goal_line + sign * 2.0, goal_line + sign * 2.0, goal_line],
            [-3.66, -3.66, 3.66, 3.66],
            color=color, lw=lw * 1.4, zorder=1,
        )

    if set_limits:
        ax.set_xlim(-hl - 3, hl + 3)
        ax.set_ylim(-hw - 3, hw + 3)
    ax.set_aspect("equal")


def zoom_to_action(
    ax,
    xs,
    ys,
    padding: float = 9.0,
    min_width: float = 38.0,
    aspect: float = 1.5,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Frame the view on the relevant players instead of the whole pitch.

    xs/ys: coordinates that must stay visible (passer, candidates, ball).
    Returns the (xlim, ylim) actually applied, so callers can reuse it for the
    pitch-control grid and the minimap rectangle.

    Keeps aspect ratio fixed (so the pitch never looks stretched) and enforces a
    minimum width, otherwise a tight cluster of players zooms in so far the view
    loses all pitch context. The window is SHIFTED to stay near the pitch rather than
    shrunk, which would break the aspect ratio.
    """
    xs = np.asarray([v for v in xs if v is not None and not np.isnan(v)], dtype=float)
    ys = np.asarray([v for v in ys if v is not None and not np.isnan(v)], dtype=float)
    hl, hw = PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2

    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    width = max(xs.max() - xs.min() + 2 * padding, min_width)
    height = max(ys.max() - ys.min() + 2 * padding, width / aspect)
    width = max(width, height * aspect)  # lock aspect after both minimums applied

    # Never zoom out past the pitch itself.
    width, height = min(width, PITCH_LENGTH_M + 6), min(height, PITCH_WIDTH_M + 6)
    # Shift (don't shrink) so the window stays over the pitch.
    cx = float(np.clip(cx, -hl - 3 + width / 2, hl + 3 - width / 2))
    cy = float(np.clip(cy, -hw - 3 + height / 2, hw + 3 - height / 2))

    xlim = (cx - width / 2, cx + width / 2)
    ylim = (cy - height / 2, cy + height / 2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    return xlim, ylim


def add_minimap(ax, xlim, ylim, loc=(0.72, 0.72, 0.26, 0.26)):
    """Inset showing where the zoomed window sits on the full pitch.

    Zooming trades away the reader's sense of position; this gives it back cheaply.
    """
    inset = ax.inset_axes(loc)
    draw_pitch(inset, color="gray", lw=0.6, set_limits=True)
    inset.add_patch(
        Rectangle(
            (xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
            fill=True, facecolor="yellow", alpha=0.35, edgecolor="black", lw=1.0, zorder=3,
        )
    )
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("gray")
        spine.set_linewidth(0.6)
    # Opaque: the pitch-control heatmap underneath would otherwise bleed through and
    # make the minimap unreadable.
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    inset.set_zorder(10)
    return inset


def add_attack_arrow(ax, xlim, ylim, direction: int = 1, label: str = "attacking"):
    """Small arrow marking which way the attacking team is going.

    Coordinates are direction-oriented throughout the pipeline, so without this a
    reader cannot tell which goal matters.
    """
    x0, x1 = xlim
    y0, y1 = ylim
    span = x1 - x0
    ax.annotate(
        "",
        xy=(x0 + span * (0.72 if direction > 0 else 0.28), y0 + (y1 - y0) * 0.06),
        xytext=(x0 + span * (0.28 if direction > 0 else 0.72), y0 + (y1 - y0) * 0.06),
        arrowprops=dict(arrowstyle="-|>", color="dimgray", lw=1.6),
        zorder=7,
    )
    ax.text(
        x0 + span * 0.5, y0 + (y1 - y0) * 0.10, label,
        ha="center", va="bottom", fontsize=7, color="dimgray", zorder=7,
    )


def plot_pitch_control(
    row: pd.Series,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    control: np.ndarray,
    player_ids: dict[str, list[str]],
    attacking_team: str = "home",
    ax=None,
):
    """Render a pitch-control surface with players and ball overlaid.

    Fill uses 'bwr': high attacking-team control -> red, contested -> white,
    defending control -> blue. Home dots are drawn red and away dots blue, so the
    fill color matches the attacking team's dot color (avoids the bwr_r inversion
    that made the first V1 render look wrong).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10.5, 7))

    # Orient so the attacking team's control reads as red regardless of which team attacks.
    field = control if attacking_team == "home" else 1 - control
    cf = ax.contourf(grid_x, grid_y, field, levels=20, cmap="bwr", vmin=0, vmax=1, alpha=0.7)
    ax.contour(grid_x, grid_y, field, levels=[0.5], colors="black", linewidths=1)

    # Color dots by attacking/defending role (not home/away) so red dots always sit
    # in the red (attacking-control) fill regardless of which team is attacking.
    defending_team = "away" if attacking_team == "home" else "home"
    for team, color, label in [(attacking_team, "red", "attacking"), (defending_team, "blue", "defending")]:
        xs = [row.get(f"{team}_{pid}_x") for pid in player_ids[team]]
        ys = [row.get(f"{team}_{pid}_y") for pid in player_ids[team]]
        ax.scatter(xs, ys, c=color, edgecolors="black", s=80, zorder=5, label=label)

    ax.scatter([row["ball_x"]], [row["ball_y"]], c="yellow", edgecolors="black", s=60, zorder=6, label="ball")
    draw_pitch(ax)
    ax.legend(loc="upper right")
    return ax
