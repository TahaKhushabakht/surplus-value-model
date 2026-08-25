"""Validation for the role-space machinery (docs/scarcity.md criteria S1, S2).

S3 (orthogonality to quality) and S4 (bandwidth sensitivity) are NOT run here: both
require the rarity statistic itself, which needs a population of identified players
across many matches. Metrica gives ~28 anonymous slots per match with no cross-match
linkage, so scarcity cannot be measured on it — see docs/scarcity.md, BLOCKING
LIMITATION. Only the machinery is testable now.

S1 is deliberately constructed to avoid tautology: position labels are inferred from
TERRITORY (mean pitch location), then predicted from NETWORK + STYLE features only.
Those feature sets are disjoint, so recovering position from network structure is real
evidence that the passing graph encodes role, not a restatement of the labels.

Run: python src/validate_scarcity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).parent))

from metrica_io import read_events
from role_space import (
    NETWORK_FEATURES,
    STYLE_FEATURES,
    build_role_features,
    infer_position,
    normalize_within_team,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "metrica" / "data"
POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


def load_role_features(game: int, period: int | None = None, min_passes: int = 10) -> pd.DataFrame:
    events = read_events(DATA_DIR / f"Sample_Game_{game}" / f"Sample_Game_{game}_RawEventsData.csv")
    if period is not None:
        events = events[events["Period"] == period]
    role = build_role_features(events, min_passes=min_passes)
    return infer_position(role)


def run_s1(role_by_game: dict[int, pd.DataFrame]) -> dict:
    """Can NETWORK + STYLE features alone recover territory-inferred position?"""
    features = NETWORK_FEATURES + STYLE_FEATURES
    pooled = []
    for game, df in role_by_game.items():
        norm = normalize_within_team(df)
        norm["game"] = game
        pooled.append(norm)
    data = pd.concat(pooled, ignore_index=True).dropna(subset=features)

    X = data[features].to_numpy()
    y = data["inferred_position"].to_numpy()

    # Leave-one-out kNN: tiny sample, so LOO is the honest choice.
    correct = 0
    for i in range(len(X)):
        mask = np.arange(len(X)) != i
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X[mask], y[mask])
        correct += int(knn.predict(X[i : i + 1])[0] == y[i])
    accuracy = correct / len(X)
    majority = pd.Series(y).value_counts(normalize=True).max()

    print(f"  n={len(X)} players, {len(features)} network+style features")
    print(f"  leave-one-out kNN accuracy = {accuracy:.3f}   (majority-class baseline = {majority:.3f})")

    # Monotone relationships against the position ordinal are the interpretable half.
    ordinal = pd.Series(y).map(POSITION_ORDER).to_numpy()
    print("  Spearman rho vs position ordinal (GK<DEF<MID<FWD):")
    rhos = {}
    for feat in ["in_out_ratio", "fwd_share", "betweenness", "long_share", "pagerank"]:
        rho = float(spearmanr(data[feat], ordinal).statistic)
        rhos[feat] = rho
        print(f"    {feat:<16} rho = {rho:+.3f}")

    verdict = accuracy > majority
    print(f"  S1 {'PASSED' if verdict else 'FAILED'} (network structure recovers position better than chance)")
    return {"accuracy": accuracy, "majority": majority, "rhos": rhos, "verdict": verdict}


def run_s2(min_passes: int = 8) -> dict:
    """Split-half stability: does a player land in a similar role region in each half?"""
    features = NETWORK_FEATURES + STYLE_FEATURES
    rows = []
    for game in (1, 2):
        first = normalize_within_team(load_role_features(game, period=1, min_passes=min_passes))
        second = normalize_within_team(load_role_features(game, period=2, min_passes=min_passes))
        merged = first.merge(second, on=["player_id", "team"], suffixes=("_h1", "_h2"))
        merged["game"] = game
        rows.append(merged)
    data = pd.concat(rows, ignore_index=True)

    print(f"  n={len(data)} players present in both halves (min_passes={min_passes} per half)")
    results = {}
    for feat in features:
        a, b = data[f"{feat}_h1"], data[f"{feat}_h2"]
        valid = a.notna() & b.notna()
        if valid.sum() < 5:
            continue
        rho = float(spearmanr(a[valid], b[valid]).statistic)
        results[feat] = rho

    ordered = sorted(results.items(), key=lambda kv: -kv[1])
    for feat, rho in ordered:
        flag = "" if rho >= 0.7 else ("  <- unstable" if rho < 0.4 else "  <- marginal")
        print(f"    {feat:<16} split-half rho = {rho:+.3f}{flag}")

    stable = [f for f, r in results.items() if r >= 0.7]
    median_rho = float(np.median(list(results.values())))
    print(f"  median split-half rho = {median_rho:.3f}; {len(stable)}/{len(results)} features at rho >= 0.7")
    return {"per_feature": results, "median": median_rho, "stable": stable}


if __name__ == "__main__":
    role_by_game = {g: load_role_features(g) for g in (1, 2)}
    for g, df in role_by_game.items():
        counts = df["inferred_position"].value_counts().to_dict()
        print(f"game {g}: {len(df)} players — {counts}")

    print()
    print("=== S1: role-space face validity (network+style -> territory-inferred position) ===")
    run_s1(role_by_game)

    print()
    print("=== S2: split-half stability (period 1 vs period 2) ===")
    run_s2()
