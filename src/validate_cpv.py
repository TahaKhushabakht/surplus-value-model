"""Reproducible validation suite for the CPV pipeline (docs/prototype-validation.md).

V2: completion-model calibration, leave-one-game-out, reliability within +/-10pp.
V4: render the 5 highest- and 5 lowest-cpv_avg passes so they can be visually
inspected against the tracking data. (10 total, matching the doc's spec.)
V5: per-player mean CPV, ranked, compared across (a) max vs avg baseline and
(b) unrestricted vs range-limited (40m) choice set. Spearman >= 0.7 required on both.

Also reports the accuracy of the intended-receiver inference that failed passes
depend on, measured against known receivers on completed passes.

Run: python src/validate_cpv.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

from completion_model import (
    FEATURES,
    build_pass_dataset,
    fit_completion_model,
    reliability_table,
    validate_receiver_inference,
)
from cpv import build_cpv_dataset
from expected_threat import _attack_directions, build_xt_pooled
from metrica_io import load_game, read_events
from pitch_control import compute_velocities, pitch_control_grid, player_ids_for_team
from viz import add_attack_arrow, add_minimap, draw_pitch, save_figure, zoom_to_action

DATA_DIR = Path(__file__).parent.parent / "data" / "metrica" / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def load_all():
    tracking, events, pids, comp_dataset = {}, {}, {}, {}
    for g in (1, 2):
        trk, evs = load_game(DATA_DIR, g)
        p = {"home": player_ids_for_team(trk.columns, "home"), "away": player_ids_for_team(trk.columns, "away")}
        dt = trk["Time [s]"].diff().median()
        trk = compute_velocities(trk, p, dt=dt)
        tracking[g], events[g], pids[g] = trk, evs, p
        comp_dataset[g] = build_pass_dataset(trk, evs, p).dropna(subset=FEATURES)

    raw1 = read_events(DATA_DIR / "Sample_Game_1" / "Sample_Game_1_RawEventsData.csv")
    raw2 = read_events(DATA_DIR / "Sample_Game_2" / "Sample_Game_2_RawEventsData.csv")
    xt = build_xt_pooled([raw1, raw2], nx=12, ny=8)
    return tracking, events, pids, comp_dataset, xt


def run_v2(comp_dataset):
    """Leave-one-game-out calibration of the completion model."""
    from sklearn.metrics import brier_score_loss

    preds, trues = [], []
    for test_g in (1, 2):
        train_g = 2 if test_g == 1 else 1
        m = fit_completion_model(comp_dataset[train_g])
        preds.append(m.predict_proba(comp_dataset[test_g]))
        trues.append(comp_dataset[test_g]["completed"].to_numpy())
    y_prob, y_true = np.concatenate(preds), np.concatenate(trues)

    base = float(y_true.mean())
    print(f"n={len(y_true)}, base completion rate={base:.3f}")
    print(f"brier={brier_score_loss(y_true, y_prob):.4f}  baseline={brier_score_loss(y_true, np.full_like(y_prob, base)):.4f}")
    rel = reliability_table(y_true, y_prob)
    print(rel.to_string(index=False))
    qualifying = rel[rel["counts_for_v2"]]
    worst = qualifying["abs_gap"].max()
    print(f"V2: max |gap| in qualifying bins = {worst*100:.1f}pp (need <= 10pp) -> {'PASSED' if worst <= 0.10 else 'FAILED'}")
    return y_prob, y_true, rel


def build_cpv_variant(tracking, events, pids, comp_dataset, xt, max_choice_range_m=None):
    summaries, results = [], {}
    for g in (1, 2):
        train_g = 2 if g == 1 else 1
        cm = fit_completion_model(comp_dataset[train_g])
        summary, res = build_cpv_dataset(tracking[g], events[g], pids[g], cm, xt, max_choice_range_m=max_choice_range_m)
        summary["game"] = g
        summaries.append(summary)
        results[g] = res
    return pd.concat(summaries, ignore_index=True), results


def render_v4(cpv_df, results, tracking, pids, events):
    """5 highest + 5 lowest cpv_avg passes, rendered with pitch control background,
    candidate options, and the actual pass drawn as an arrow."""
    eligible = cpv_df[cpv_df["n_candidates"] >= 4]
    top5 = eligible.nlargest(5, "cpv_avg")
    bottom5 = eligible.nsmallest(5, "cpv_avg")
    selected = pd.concat([top5, bottom5])

    # Real attacking direction per (team, period) -- coordinates here are raw tracking
    # coords, so without this the direction indicator would be a guess.
    directions = {g: _attack_directions(events[g]) for g in events}

    # Shared EV color scale across panels (comparability), but capped at the 97th
    # percentile: EV is heavily right-skewed, and a raw max washes every other
    # candidate into the same dark color.
    all_ev = [c["ev"] for _, prow in selected.iterrows()
              for c in next(r for r in results[prow["game"]] if r.pass_id == prow["pass_id"]).candidates]
    ev_vmin, ev_vmax = float(np.percentile(all_ev, 3)), float(np.percentile(all_ev, 97))

    fig, axes = plt.subplots(2, 5, figsize=(27, 12))
    for ax, (_, prow) in zip(axes.flat, selected.iterrows()):
        g = prow["game"]
        result = next(r for r in results[g] if r.pass_id == prow["pass_id"])
        frow = tracking[g][tracking[g]["Frame"] == result.frame].iloc[0]

        passer_x = frow.get(f"{result.team}_{result.from_player}_x")
        passer_y = frow.get(f"{result.team}_{result.from_player}_y")
        cand_x = [c["x"] for c in result.candidates]
        cand_y = [c["y"] for c in result.candidates]
        cand_ev = [c["ev"] for c in result.candidates]
        chosen = next(c for c in result.candidates if c["player_id"] == result.to_player)

        # Frame on the LOCAL action only. The choice set includes every teammate --
        # a deep goalkeeper would otherwise stretch the window across the whole pitch
        # and defeat the zoom. Distant options stay in the choice set and in the
        # metric; they just fall outside the view.
        near = [(x, y) for x, y in zip(cand_x, cand_y) if np.hypot(x - passer_x, y - passer_y) <= 35.0]
        frame_x = [passer_x, chosen["x"]] + [p[0] for p in near]
        frame_y = [passer_y, chosen["y"]] + [p[1] for p in near]

        # Compute pitch control only over the visible window -- same grid resolution
        # over a smaller area gives finer contours.
        xlim, ylim = zoom_to_action(ax, frame_x, frame_y, padding=7.0, min_width=34.0)
        gx, gy, control = pitch_control_grid(frow, result.team, pids[g], nx=90, ny=60, xlim=xlim, ylim=ylim)
        ax.contourf(gx, gy, control, levels=20, cmap="bwr", vmin=0, vmax=1, alpha=0.5, zorder=0)
        ax.contour(gx, gy, control, levels=[0.5], colors="black", linewidths=0.8, alpha=0.6, zorder=2)
        draw_pitch(ax, color="black", lw=1.0, set_limits=False)

        # Dots colored by role (attacking red / defending blue) to match the fill.
        defending = "away" if result.team == "home" else "home"
        for team, color in [(result.team, "red"), (defending, "blue")]:
            xs = [frow.get(f"{team}_{pid}_x") for pid in pids[g][team]]
            ys = [frow.get(f"{team}_{pid}_y") for pid in pids[g][team]]
            ax.scatter(xs, ys, c=color, edgecolors="black", s=55, zorder=4)

        sc = ax.scatter(
            cand_x, cand_y, c=cand_ev, cmap="viridis", vmin=ev_vmin, vmax=ev_vmax,
            s=150, edgecolors="black", zorder=5, marker="s", alpha=0.9,
        )

        ax.annotate(
            "", xy=(chosen["x"], chosen["y"]), xytext=(passer_x, passer_y),
            arrowprops=dict(arrowstyle="-|>", color="lime", lw=2.8), zorder=6,
        )
        ax.scatter([passer_x], [passer_y], facecolors="none", edgecolors="lime", s=320, linewidths=2.5, zorder=6)

        team_label = "Home" if result.team == "home" else "Away"
        direction = directions[g].get((team_label, result.period), 1)
        add_attack_arrow(ax, xlim, ylim, direction=direction)
        add_minimap(ax, xlim, ylim)

        outcome = "COMPLETED" if result.completed else "INTERCEPTED"
        ax.set_title(
            f'g{g} f{result.frame} {result.team} [{outcome}]\ncpv_avg={prow["cpv_avg"]:.4f} cpv_max={prow["cpv_max"]:.4f}',
            fontsize=9,
        )
        ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_ylabel("HIGHEST cpv_avg (top 5)", fontsize=11)
    axes[1, 0].set_ylabel("LOWEST cpv_avg (bottom 5)", fontsize=11)
    fig.suptitle(
        "V4 eye test — zoomed to the action (inset shows window on full pitch)\n"
        "green circle = passer, green arrow = actual pass, squares = pass options (color = EV), "
        "red fill = attacking-team pitch control",
        fontsize=12,
    )
    fig.colorbar(sc, ax=axes, shrink=0.5, label="candidate EV")
    return save_figure(fig, "cpv_v4_eye_test.png"), selected


def run_v5(tracking, events, pids, comp_dataset, xt):
    """Per-player mean CPV rank stability across baseline choice and choice-set width."""
    default_df, _ = build_cpv_variant(tracking, events, pids, comp_dataset, xt, max_choice_range_m=None)
    narrow_df, _ = build_cpv_variant(tracking, events, pids, comp_dataset, xt, max_choice_range_m=40.0)

    def per_player(df, col, min_passes=5):
        g = df.groupby(["game", "from_player"])[col].agg(["mean", "count"])
        g = g[g["count"] >= min_passes]
        g.index = [f"{game}_{p}" for game, p in g.index]
        return g["mean"]

    max_default = per_player(default_df, "cpv_max")
    avg_default = per_player(default_df, "cpv_avg")
    max_narrow = per_player(narrow_df, "cpv_max")
    avg_narrow = per_player(narrow_df, "cpv_avg")

    common_baseline = max_default.index.intersection(avg_default.index)
    rho_baseline, p_baseline = spearmanr(max_default[common_baseline], avg_default[common_baseline])

    common_width = max_default.index.intersection(max_narrow.index)
    rho_width, p_width = spearmanr(max_default[common_width], max_narrow[common_width])

    print(f"V5a (baseline max vs avg), n_players={len(common_baseline)}: rho={rho_baseline:.3f} (need >= 0.7)")
    print(f"V5b (choice-set unrestricted vs 40m-limited), n_players={len(common_width)}: rho={rho_width:.3f} (need >= 0.7)")
    verdict = rho_baseline >= 0.7 and rho_width >= 0.7
    print("V5 " + ("PASSED" if verdict else "FAILED"))
    return {"rho_baseline": rho_baseline, "rho_width": rho_width, "verdict": verdict}


if __name__ == "__main__":
    tracking, events, pids, comp_dataset, xt = load_all()

    print("=== Intended-receiver inference (failed passes depend on it) ===")
    for g in (1, 2):
        r = validate_receiver_inference(tracking[g], events[g], pids[g])
        print(f"  game{g}: resolve_rate={r['resolve_rate']:.3f}  accuracy_when_resolved={r['accuracy_when_resolved']:.3f}")

    print()
    print("=== V2: completion-model calibration ===")
    run_v2(comp_dataset)

    cpv_df, results = build_cpv_variant(tracking, events, pids, comp_dataset, xt)
    OUTPUT_DIR.mkdir(exist_ok=True)
    cpv_df.to_pickle(OUTPUT_DIR / "cpv_df.pkl")
    with open(OUTPUT_DIR / "cpv_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print()
    print("=== CPV coverage ===")
    print(f"  rows={len(cpv_df)}  completed={int(cpv_df['completed'].sum())}  intercepted={int((1-cpv_df['completed']).sum())}")
    print(cpv_df.groupby("completed")[["cpv_avg", "cpv_max", "execution_quality"]].mean().round(5).to_string())

    print()
    print("=== V4: rendering eye-test figure ===")
    path, selected = render_v4(cpv_df, results, tracking, pids, events)
    print("saved:", path)
    print(selected[["game", "frame", "team", "completed", "from_player", "to_player", "cpv_avg", "cpv_max"]].to_string())

    print()
    print("=== V5: sensitivity stability ===")
    run_v5(tracking, events, pids, comp_dataset, xt)
