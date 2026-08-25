"""Validation visuals for CPV at scale.

Each panel is built to expose a specific suspected failure, not to look informative:

  1. Zone dependence   — is cpv_avg systematically larger in some pitch areas? This is
     the geographic confound directly: a metric whose scale depends on WHERE the pass
     was made cannot be averaged over players who occupy different areas. Compares the
     raw metric against the two context-normalized variants; a good variant should
     flatten the map.
  2. Turnover cost map — what V(turnover) currently charges at each location. The
     approximation under test evaluates turnover at the TARGET, so an ambitious forward
     pass is charged the cheap upfield cost. This panel shows how cheap.
  3. EV anatomy        — for one real pass, every option broken into p_complete,
     V(receive), V(turnover) and resulting EV, so the arithmetic behind a CPV value is
     inspectable rather than trusted.
  4. Player CIs        — bootstrap confidence intervals on per-player CPV. Without
     these a leaderboard implies precision it may not have.

Run: python src/viz_validation.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from cpv_scale import PITCH_LENGTH_M, PITCH_WIDTH_M, XTLookup
from viz import draw_pitch, save_figure

OUTPUT_DIR = Path(__file__).parent.parent / "output"
FOCUS_TEAM = "Bayer Leverkusen"


def zone_means(cpv: pd.DataFrame, column: str, nx: int = 8, ny: int = 5) -> np.ndarray:
    """Mean of `column` per pitch zone, by pass START location."""
    xs = np.clip((cpv["start_x"] / PITCH_LENGTH_M + 0.5) * nx, 0, nx - 1e-9).astype(int)
    ys = np.clip((0.5 - cpv["start_y"] / PITCH_WIDTH_M) * ny, 0, ny - 1e-9).astype(int)
    grid = np.full((ny, nx), np.nan)
    for iy in range(ny):
        for ix in range(nx):
            vals = cpv.loc[(xs == ix) & (ys == iy), column]
            if len(vals) >= 25:
                grid[iy, ix] = vals.mean()
    return grid


def panel_zone_dependence(ax_row, cpv):
    """Does the metric's scale depend on where the pass was taken from?"""
    for ax, col, title in zip(
        ax_row,
        ["cpv_avg", "cpv_z", "cpv_rank"],
        ["raw cpv_avg\n(scale varies by zone = confounded)", "cpv_z\n(z-scored within choice set)", "cpv_rank\n(percentile within choice set)"],
    ):
        grid = zone_means(cpv, col)
        # Diverging around each metric's own neutral point.
        centre = 0.5 if col == "cpv_rank" else 0.0
        lim = np.nanmax(np.abs(grid - centre))
        im = ax.imshow(grid, origin="lower", extent=[-52.5, 52.5, -34, 34], aspect="auto",
                       cmap="RdBu_r", vmin=centre - lim, vmax=centre + lim)
        draw_pitch(ax, color="black", lw=0.8, set_limits=False)
        ax.set_xlim(-53, 53); ax.set_ylim(-35, 35)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.035)
        # Quantify the confound: spread of zone means, relative to the metric's own sd.
        spread = np.nanstd(grid) / (cpv[col].std() + 1e-12)
        ax.set_xlabel(f"zone-mean spread = {spread:.3f} of metric sd\n(lower = less zone-confounded)", fontsize=7.5)


def panel_turnover_map(ax, xt: XTLookup):
    """How much does losing the ball cost, by location? (attack toward +x)"""
    nx, ny = 60, 40
    xs = np.linspace(-52.5, 52.5, nx)
    ys = np.linspace(-34, 34, ny)
    gx, gy = np.meshgrid(xs, ys)
    cost = -xt.turnover_value(gx.ravel(), gy.ravel()).reshape(ny, nx)  # positive = expensive

    im = ax.contourf(gx, gy, cost * 1000, levels=24, cmap="magma_r")
    draw_pitch(ax, color="white", lw=1.0, set_limits=False)
    ax.set_xlim(-53, 53); ax.set_ylim(-35, 35)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Cost of losing the ball, by location (x1000)\nattacking toward +x (right)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.035, label="turnover cost")
    ax.annotate("", xy=(38, -30), xytext=(-38, -30), arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6))
    ax.text(0, -27.5, "attacking", ha="center", color="white", fontsize=8)


def panel_ev_anatomy(ax, cpv, xt):
    """Break one pass into its EV components so the arithmetic is inspectable."""
    sample = cpv[(cpv.n_candidates >= 8) & (cpv.start_x.between(-10, 20))].nlargest(1, "ev_spread").iloc[0]
    labels = ["p(complete)", "V(receive)", "V(turnover)", "EV"]
    chosen = [sample.p_complete_chosen, sample.v_receive_chosen, sample.v_turnover_chosen, sample.ev_chosen]
    context = [np.nan, np.nan, np.nan, sample.ev_mean]

    x = np.arange(len(labels))
    ax.bar(x - 0.19, chosen, 0.38, label="chosen pass", color="#2e7d32")
    ax.bar(x + 0.19, context, 0.38, label="mean of options", color="#90a4ae")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(
        f"EV anatomy — {sample.passer[:26]}\n"
        f"cpv_avg={sample.cpv_avg:+.4f}  rank={sample.cpv_rank:.2f}  "
        f"{'completed' if sample.completed else 'FAILED'}",
        fontsize=9,
    )
    ax.legend(fontsize=7)
    # The imbalance this panel is meant to reveal.
    ax.annotate(
        f"V(turnover) is only {abs(sample.v_turnover_chosen)/max(sample.v_receive_chosen,1e-9):.0%}\n"
        f"the size of V(receive) here",
        xy=(2, sample.v_turnover_chosen), xytext=(1.4, max(chosen[:3]) * 0.55),
        fontsize=7.5, color="#b71c1c",
        arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.2),
    )


def panel_player_ci(ax, cpv, column="cpv_avg", min_passes=150, n_boot=400, seed=0):
    """Bootstrap CIs per player — is the leaderboard ordering even distinguishable?"""
    rng = np.random.default_rng(seed)
    d = cpv[cpv.team == FOCUS_TEAM]
    stats = []
    for (pid, name), grp in d.groupby(["passer_id", "passer"]):
        if len(grp) < min_passes:
            continue
        vals = grp[column].to_numpy()
        boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
        stats.append((name, vals.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), len(vals)))

    stats.sort(key=lambda s: s[1])
    names = [s[0][:24] for s in stats]
    means = np.array([s[1] for s in stats])
    lo = np.array([s[2] for s in stats])
    hi = np.array([s[3] for s in stats])
    y = np.arange(len(stats))

    ax.errorbar(means, y, xerr=[means - lo, hi - means], fmt="o", color="#1565c0", ecolor="#90caf9", capsize=3)
    ax.axvline(float(d[column].mean()), color="gray", ls="--", lw=1, label="squad mean")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_title(f"Per-player {column} with 95% bootstrap CI\n(overlapping bars = ordering not distinguishable)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="x", alpha=0.3)


if __name__ == "__main__":
    cpv = pd.read_pickle(OUTPUT_DIR / "sb_cpv.pkl")
    cpv = cpv[cpv.target_match_dist <= 6.0]
    xt = XTLookup(np.load(OUTPUT_DIR / "sb_xt.npy"))
    print(f"{len(cpv):,} passes; {cpv.team.nunique()} teams")

    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.15, 1.25], hspace=0.42, wspace=0.28)

    panel_zone_dependence([fig.add_subplot(gs[0, i]) for i in range(3)], cpv)
    panel_turnover_map(fig.add_subplot(gs[1, 0:2]), xt)
    panel_ev_anatomy(fig.add_subplot(gs[1, 2]), cpv, xt)
    panel_player_ci(fig.add_subplot(gs[2, :]), cpv)

    fig.suptitle(
        "CPV validation dashboard — Bayer Leverkusen 2023/24 (StatsBomb 360)\n"
        "row 1: is the metric zone-confounded?   row 2: turnover cost + EV anatomy   row 3: is the ranking distinguishable?",
        fontsize=12,
    )
    print("saved:", save_figure(fig, "cpv_validation_dashboard.png", dpi=110))

    print()
    print("=== Zone-confounding summary (lower = better) ===")
    for col in ("cpv_avg", "cpv_z", "cpv_rank"):
        g = zone_means(cpv, col)
        print(f"  {col:<10} zone-mean spread / metric sd = {np.nanstd(g)/cpv[col].std():.4f}")
