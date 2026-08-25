"""Surplus-value model dashboard — four panels covering the actual findings, not just
the pipeline's output:

  1. Predicted vs. actual value  — the headline chart. Colored to reveal what turned
     out to be driving the residual (club identity, not individual quality/scarcity).
  2. Central test                — does adding quality+scarcity improve out-of-sample
     R^2 over boring baseline features? This is the question the whole project was
     built to answer; the answer here is no, shown plainly rather than buried.
  3. Residual by club            — the mechanism behind panel 1's color pattern: a
     near-monotonic "prestige gradient" from smallest to biggest club.
  4. Coefficient plot            — what actually predicts value (age curve, playing
     time, reputation) vs. what doesn't (quality, scarcity: CIs cross zero).

Run: python src/viz_surplus_value.py
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

from surplus_value import BASELINE_FEATURES, FULL_FEATURES, build_dataset, cross_validated_r2, fit_full_model
from viz import save_figure

OUTPUT_DIR = Path(__file__).parent.parent / "output"
HIGHLIGHT = {"Bayern Munich": "#c62828", "Bayer Leverkusen": "#00838f"}
OTHER_COLOR = "#9e9e9e"


def panel_predicted_vs_actual(ax, df):
    for team, color in [(None, OTHER_COLOR)] + list(HIGHLIGHT.items()):
        mask = df.team != df.team if team is None else df.team == team
        mask = (~df.team.isin(HIGHLIGHT)) if team is None else (df.team == team)
        ax.scatter(df.loc[mask, "predicted_value_eur"] / 1e6, df.loc[mask, "market_value_eur"] / 1e6,
                   s=26, alpha=0.75 if team else 0.45, color=color, edgecolors="black" if team else "none",
                   linewidths=0.4, label=team or "other clubs", zorder=3 if team else 2)

    lims = [0.15, 130]
    ax.plot(lims, lims, color="black", ls="--", lw=1, alpha=0.6, zorder=1, label="perfect prediction")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("model-predicted value (EURm, log scale)")
    ax.set_ylabel("actual market value (EURm, log scale)")
    ax.set_title("Predicted vs. actual value\nBayern sits above the line (overpriced by characteristics), Leverkusen below", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    biggest = pd.concat([df.nlargest(6, "surplus"), df.nsmallest(6, "surplus")])
    texts = [
        ax.text(row.predicted_value_eur / 1e6, row.market_value_eur / 1e6, row["name"].split(" ")[-1], fontsize=6.5)
        for _, row in biggest.iterrows()
    ]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.4))


def panel_central_test(ax, r2_base, r2_full):
    means = [r2_base.mean(), r2_full.mean()]
    stds = [r2_base.std(), r2_full.std()]
    bars = ax.bar([0, 1], means, yerr=stds, capsize=5, color=["#78909c", "#5c6bc0"], edgecolor="black", width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["baseline\n(age, minutes,\nposition, caps)", "+ quality\n+ scarcity"], fontsize=8)
    ax.set_ylabel("cross-validated R²")
    ax.axhline(0, color="black", lw=0.8)
    delta = means[1] - means[0]
    ax.set_title(f"Central test: does CPV/scarcity help?\nΔR² = {delta:+.3f} (within fold noise — no)", fontsize=9.5)
    for x, m in zip([0, 1], means):
        ax.text(x, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0, max(means) + 0.15)


def panel_residual_by_club(ax, df):
    g = df.groupby("team")["residual"].agg(["mean", "size"])
    g = g[g["size"] >= 5].sort_values("mean")
    colors = [HIGHLIGHT.get(t, "#90a4ae") for t in g.index]
    ax.barh(np.arange(len(g)), g["mean"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(g)))
    ax.set_yticklabels([f"{t} (n={n})" for t, n in zip(g.index, g["size"])], fontsize=7.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("mean residual (log actual − log predicted)\n> 0 = priced above what characteristics predict")
    ax.set_title("Residual by club: a prestige gradient\nBayern/Leipzig overpriced, smaller clubs underpriced by the model", fontsize=9.5)


def panel_coefficients(ax, model):
    params = model.params.drop("const")
    ci = model.conf_int().drop("const")
    order = params.reindex(params.abs().sort_values().index).index
    y = np.arange(len(order))
    vals = params[order]
    lo, hi = ci.loc[order, 0], ci.loc[order, 1]
    significant = (lo > 0) | (hi < 0)
    colors = ["#2e7d32" if s else "#bdbdbd" for s in significant]
    ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="o", color="black", ecolor="black", capsize=3, zorder=3)
    ax.barh(y, vals, color=colors, alpha=0.35, height=0.5, zorder=1)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("coefficient (on log market value)")
    ax.set_title("What predicts value: green = 95% CI excludes zero\nquality & scarcity: CIs cross zero", fontsize=9.5)


if __name__ == "__main__":
    df = build_dataset()
    r2_base = cross_validated_r2(df, BASELINE_FEATURES)
    r2_full = cross_validated_r2(df, FULL_FEATURES)
    model = fit_full_model(df)
    df["predicted_log_value"] = model.predict(__import__("statsmodels.api", fromlist=["add_constant"]).add_constant(df[FULL_FEATURES]))
    df["residual"] = df["log_value"] - df["predicted_log_value"]
    df["surplus"] = -df["residual"]
    df["predicted_value_eur"] = np.exp(df["predicted_log_value"])

    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.15, 1.0], hspace=0.4, wspace=0.28)

    panel_predicted_vs_actual(fig.add_subplot(gs[0, 0]), df)
    panel_central_test(fig.add_subplot(gs[0, 1]), r2_base, r2_full)
    panel_residual_by_club(fig.add_subplot(gs[1, 0]), df)
    panel_coefficients(fig.add_subplot(gs[1, 1]), model)

    fig.suptitle(
        "Surplus-value model — Bundesliga 2023/24 (n=245)\n"
        "Finding: club identity dominates the residual; CPV/scarcity add no detectable predictive power in this sample",
        fontsize=12,
    )
    print("saved:", save_figure(fig, "surplus_value_dashboard.png", dpi=115))
