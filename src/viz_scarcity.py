"""Scarcity visualization — four panels, each built to show a specific claim from
docs/scarcity.md rather than just decorate the leaderboard:

  1. Role-space map    — the actual PCA geometry scarcity_density is computed in.
     Marker size = n_passes (reliability), shape = position, color = scarcity.
     Leverkusen players labeled so the leaderboard entries are traceable to a location
     in role space rather than trusted as a black-box number.
  2. Leverkusen board   — scarcity_density per squad player, annotated with n_passes so
     reliability isn't hidden (the reliable-subset caveat from HISTORY.md §27 stays
     visible rather than living only in a footnote).
  3. S3 visual          — scarcity vs quality across the population. The validation
     doc's orthogonality claim (corr=-0.007) should look like a flat cloud; this makes
     that checkable by eye, not just by a coefficient.
  4. density vs pool    — the predicted-then-confirmed divergence (rho=0.517) between
     the two scarcity variants, shown rather than just stated.

Run: python src/viz_scarcity.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

sys.path.insert(0, str(Path(__file__).parent))

import statsbomb_io as SB
from scarcity import build_population, pca_embedding
from viz import save_figure

OUTPUT_DIR = Path(__file__).parent.parent / "output"
FOCUS_TEAM = "Bayer Leverkusen"
POSITION_MARKERS = {"DEF": "s", "MID": "o", "FWD": "^"}
POSITION_COLORS = {"DEF": "#1565c0", "MID": "#2e7d32", "FWD": "#c62828"}


def panel_role_space(ax, role, Z, names):
    """PCA map of the role-space population, colored by scarcity."""
    for pos, marker in POSITION_MARKERS.items():
        mask = (role.inferred_position == pos).to_numpy()
        sizes = 18 + 2.2 * np.sqrt(role.loc[mask, "n_passes"].to_numpy())
        sc = ax.scatter(
            Z[mask, 0], Z[mask, 1], c=role.loc[mask, "scarcity_density"], cmap="magma",
            vmin=role.scarcity_density.min(), vmax=role.scarcity_density.max(),
            marker=marker, s=sizes, edgecolors="black", linewidths=0.5, alpha=0.85,
            label=pos, zorder=3,
        )
    plt.colorbar(sc, ax=ax, label="scarcity_density", fraction=0.04)

    # Neutral-gray proxy handles for the position legend: the real markers are colored
    # by scarcity (continuous), so letting matplotlib auto-generate legend swatches
    # from that colormap would make it look like color also encodes position, which it
    # doesn't -- shape is the only thing this legend should communicate.
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0], [0], marker=m, color="none", markerfacecolor="#888888",
               markeredgecolor="black", markersize=8, label=pos)
        for pos, m in POSITION_MARKERS.items()
    ]
    leg = ax.legend(handles=proxies, loc="upper left", fontsize=8, title="position", framealpha=0.9)
    ax.add_artist(leg)

    lev_mask = (role.team == FOCUS_TEAM).to_numpy()
    ax.scatter(Z[lev_mask, 0], Z[lev_mask, 1], facecolors="none", edgecolors="#00e5ff",
               s=90, linewidths=1.6, zorder=4, label=FOCUS_TEAM)
    texts = []
    for i in np.where(lev_mask)[0]:
        name = names.get(role.iloc[i]["player_id"], "?").split(" ")[-1]
        texts.append(ax.text(Z[i, 0], Z[i, 1], name, fontsize=7, color="#006064", zorder=5))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#00838f", lw=0.5))

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(
        "Role-space map (PCA of network+territory+style features)\n"
        "marker size = n_passes, shape = position, color = scarcity, cyan ring = Leverkusen",
        fontsize=10,
    )


def panel_leverkusen_board(ax, role, names):
    lev = role[role.team == FOCUS_TEAM].sort_values("scarcity_density")
    y = np.arange(len(lev))
    colors = [POSITION_COLORS[p] for p in lev.inferred_position]
    ax.barh(y, lev.scarcity_density, color=colors, edgecolor="black", linewidth=0.4, height=0.7)

    labels = [names.get(pid, "?") for pid in lev.player_id]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    for yi, (n, val) in enumerate(zip(lev.n_passes, lev.scarcity_density)):
        weight = "bold" if n >= 100 else "normal"
        alpha = 1.0 if n >= 100 else 0.55
        ax.text(val + 0.015, yi, f"n={n}", fontsize=6.5, va="center", weight=weight, alpha=alpha)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in POSITION_COLORS.values()]
    ax.legend(handles, POSITION_COLORS.keys(), fontsize=7, loc="lower right", title="position")
    ax.set_xlabel("scarcity_density")
    ax.set_title(f"{FOCUS_TEAM}: role scarcity (bold n = reliable, >=100 passes)", fontsize=9.5)
    ax.grid(axis="x", alpha=0.3)


def panel_s3_orthogonality(ax, role):
    for pos, marker in POSITION_MARKERS.items():
        mask = (role.inferred_position == pos) & role.quality.notna()
        ax.scatter(role.loc[mask, "quality"], role.loc[mask, "scarcity_density"],
                   marker=marker, s=22, color=POSITION_COLORS[pos], alpha=0.65, label=pos)
    valid = role.quality.notna()
    corr = np.corrcoef(role.loc[valid, "quality"], role.loc[valid, "scarcity_density"])[0, 1]
    z = np.polyfit(role.loc[valid, "quality"], role.loc[valid, "scarcity_density"], 1)
    xs = np.linspace(role.quality.min(), role.quality.max(), 20)
    ax.plot(xs, np.polyval(z, xs), color="black", lw=1.2, ls="--", alpha=0.7)
    ax.set_xlabel("quality (mean cpv_zz)"); ax.set_ylabel("scarcity_density")
    ax.set_title(f"S3: scarcity vs. quality — corr={corr:+.3f}\n(flat cloud = role space hasn't leaked quality)", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def panel_density_vs_pool(ax, role):
    valid = role.quality.notna()
    for pos, marker in POSITION_MARKERS.items():
        mask = valid & (role.inferred_position == pos)
        ax.scatter(role.loc[mask, "scarcity_density"], role.loc[mask, "scarcity_pool"],
                   marker=marker, s=22, color=POSITION_COLORS[pos], alpha=0.65, label=pos)
    from scipy.stats import spearmanr
    rho = spearmanr(role.loc[valid, "scarcity_density"], role.loc[valid, "scarcity_pool"]).statistic
    ax.set_xlabel("scarcity_density (quality-independent)"); ax.set_ylabel("scarcity_pool (quality-entangled)")
    ax.set_title(f"density vs. pool variant — Spearman rho={rho:.3f}\n(moderate, not high = measuring different things, as predicted)", fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    passes = pd.read_pickle(OUTPUT_DIR / "sb_all_passes.pkl")
    role = pd.read_pickle(OUTPUT_DIR / "sb_scarcity.pkl")
    names = passes.drop_duplicates("passer_id").assign(
        pid=lambda d: d.passer_id.astype(int).astype(str)
    ).set_index("pid")["passer"]

    # PCA embedding wasn't persisted; rebuild it deterministically from the same
    # population (build_population/pca_embedding are seeded, so this reproduces
    # exactly what scarcity_density was computed from). Must pass the same real
    # position labels used when sb_scarcity.pkl was built, or GK exclusion (and hence
    # the population itself) can differ from what's saved -- see HISTORY.md #29.
    pos_labels = SB.player_position_labels(SB.load_matches()["match_id"].tolist())
    role_rebuilt = build_population(passes, position_labels=pos_labels)
    Z, pca = pca_embedding(role_rebuilt)
    assert list(role_rebuilt.player_id) == list(role.player_id), "population order mismatch"

    fig = plt.figure(figsize=(19, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], hspace=0.38, wspace=0.32)

    panel_role_space(fig.add_subplot(gs[0, :]), role, Z, names)
    panel_leverkusen_board(fig.add_subplot(gs[1, 0]), role, names)
    panel_s3_orthogonality(fig.add_subplot(gs[1, 1]), role)
    panel_density_vs_pool(fig.add_subplot(gs[1, 2]), role)

    fig.suptitle(
        "Role scarcity — Bundesliga 2023/24 population (18 teams, 263 players)\n"
        "docs/scarcity.md S3/S4 validation shown visually, not just as coefficients",
        fontsize=12,
    )
    print("saved:", save_figure(fig, "scarcity_dashboard.png", dpi=115))
