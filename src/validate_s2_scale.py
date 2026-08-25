"""S2 re-run at season scale — testing the volume extrapolation from HISTORY.md §18.

On Metrica (2 matches) S2 failed: median split-half rho 0.353, and only back_share
cleared 0.7. That was diagnosed as a sample-size artifact rather than a broken concept,
because stability rose sharply from quarter-match to half-match networks. The
prediction — "season-scale data should clear the bar" — was recorded as UNTESTED.

This tests it properly by measuring a stability-vs-volume CURVE rather than a single
number: split each player's matches into two disjoint halves of k matches each, compute
role features independently on each side, and correlate across players. Sweeping k
shows whether stability is still climbing or has plateaued, which a single split cannot.

Betweenness is reported but excluded from the headline: §18 found it structurally
degenerate on dense passing networks (density 0.64-0.74, values confined to [0, 0.10]),
and that is a property of the graph, not of sample size — so it should NOT improve here.
That prediction is itself a check on the §18 diagnosis.

Run: python src/validate_s2_scale.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

import statsbomb_io as SB
from role_space import (
    NETWORK_FEATURES,
    RECOMMENDED_NETWORK_FEATURES,
    STYLE_FEATURES,
    build_role_features_from_passes,
    normalize_within_team,
)

OUTPUT_DIR = Path(__file__).parent.parent / "output"
FOCUS_TEAM = "Bayer Leverkusen"
FEATURES = NETWORK_FEATURES + STYLE_FEATURES


def split_half_stability(passes: pd.DataFrame, k_matches: int, n_reps: int, rng, min_passes: int) -> dict[str, float]:
    """Correlate role features across two disjoint random sets of k matches each."""
    match_ids = np.array(sorted(passes["match_id"].unique()))
    if len(match_ids) < 2 * k_matches:
        return {}

    per_feature: dict[str, list[float]] = {f: [] for f in FEATURES}
    for _ in range(n_reps):
        picked = rng.choice(match_ids, size=2 * k_matches, replace=False)
        a_ids, b_ids = picked[:k_matches], picked[k_matches:]

        a = normalize_within_team(build_role_features_from_passes(passes[passes.match_id.isin(a_ids)], min_passes))
        b = normalize_within_team(build_role_features_from_passes(passes[passes.match_id.isin(b_ids)], min_passes))
        if a.empty or b.empty:
            continue
        merged = a.merge(b, on=["player_id", "team"], suffixes=("_a", "_b"))
        if len(merged) < 6:
            continue
        for f in FEATURES:
            x, y = merged.get(f"{f}_a"), merged.get(f"{f}_b")
            if x is None:
                continue
            valid = x.notna() & y.notna()
            if valid.sum() < 6:
                continue
            rho = spearmanr(x[valid], y[valid]).statistic
            if not np.isnan(rho):
                per_feature[f].append(float(rho))

    return {f: float(np.mean(v)) for f, v in per_feature.items() if v}


if __name__ == "__main__":
    print("loading all season passes (no 360 requirement) ...")
    passes = SB.load_all_passes()
    passes.to_pickle(OUTPUT_DIR / "sb_all_passes.pkl")
    focus = passes[passes["team"] == FOCUS_TEAM]
    print(f"{len(passes):,} passes total; {len(focus):,} for {FOCUS_TEAM} across {focus.match_id.nunique()} matches")
    print()

    rng = np.random.default_rng(0)
    # min_passes scales with volume: a 1-match split cannot demand 100 passes/player.
    schedule = [(1, 12, 10), (2, 10, 20), (4, 8, 40), (8, 6, 80), (15, 5, 150)]

    curve = {}
    for k, reps, min_passes in schedule:
        res = split_half_stability(focus, k_matches=k, n_reps=reps, rng=rng, min_passes=min_passes)
        if not res:
            continue
        curve[k] = res
        net = np.mean([res[f] for f in RECOMMENDED_NETWORK_FEATURES if f in res])
        sty = np.mean([res[f] for f in STYLE_FEATURES if f in res])
        bet = res.get("betweenness", float("nan"))
        print(f"k={k:>2} matches/side (min {min_passes} passes): "
              f"style={sty:+.3f}  network(recommended)={net:+.3f}  betweenness={bet:+.3f}")

    print()
    print("=== Per-feature stability at the largest split (k=15 matches per side) ===")
    largest = curve[max(curve)]
    for f, rho in sorted(largest.items(), key=lambda kv: -kv[1]):
        mark = "PASS" if rho >= 0.7 else ("marginal" if rho >= 0.4 else "UNSTABLE")
        tag = "  [excluded per §18]" if f == "betweenness" else ""
        print(f"  {f:<16} rho = {rho:+.3f}   {mark}{tag}")

    headline = {f: r for f, r in largest.items() if f != "betweenness"}
    n_pass = sum(1 for r in headline.values() if r >= 0.7)
    median = float(np.median(list(headline.values())))
    print()
    print(f"S2 at season scale (excluding betweenness): median rho = {median:.3f}, "
          f"{n_pass}/{len(headline)} features >= 0.7")
    print(f"  Metrica baseline was: median 0.353, 1/10 features >= 0.7")
    print(f"  S2 {'PASSED' if median >= 0.7 else 'IMPROVED but below bar' if median > 0.353 else 'NOT IMPROVED'}")
