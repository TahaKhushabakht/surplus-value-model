"""Role scarcity — the statistic itself, per docs/scarcity.md.

This was the BLOCKING LIMITATION recorded when scarcity was first defined: rarity
needs a population, and Metrica's ~28 anonymous per-match slots couldn't supply one.
The StatsBomb scale-up (34 Leverkusen matches, 18 distinct teams, 369 identified
players) finally can.

POPULATION (stated explicitly, per open decision #4 in docs/scarcity.md): all players
with >=MIN_PASSES across the 34-match dataset. Composition is uneven by design, not
oversight — Leverkusen supplies ~25 players at up to 34 matches each; the 17 opponents
supply ~2 matches each (both fixtures against Leverkusen). Non-Leverkusen role vectors
are therefore noisier (S2's k=2 stability was ~0.58-0.62, vs ~0.85 at Leverkusen's
season volume). This is disclosed in output, not smoothed over.

FEATURE SET: RECOMMENDED_NETWORK_FEATURES (betweenness excluded — HISTORY.md §18/§22
demonstrated it is structurally degenerate on dense passing networks, and confirmed the
prediction that it would not improve with more data) + TERRITORY_FEATURES + STYLE_FEATURES.
PCA-decorrelated before distance computation (open decision #2): several features are
compositional and collinear by construction (third_def+third_mid+third_att=1,
fwd_share+lat_share+back_share=1), so raw Euclidean distance would silently
double/triple-count those axes.

POSITION HANDLING (open decision #5): goalkeepers are excluded from the population
before computing scarcity. Including them would make "scarce" trivially mean "not an
outfielder" — no interesting signal, just a restatement of a coarse role label already
known. Outfield position is NOT further conditioned (no separate DEF/MID/FWD
sub-populations): PCA-space distance is exactly what should already down-weight
cross-position comparisons if the role features are doing their job, and conditioning
further would leave too few points per sub-group at this population size (~70-90 per
broad position group) for a stable KDE.

TWO METRICS, per docs/scarcity.md:
  scarcity_density: -log(mean Gaussian-kernel similarity to everyone else) --
    quality-independent by construction, the PRIMARY metric.
  scarcity_pool: -log((count of similar-AND-at-least-as-good players + 1) / N) --
    the replacement-pool / VORP-style variant, deliberately quality-entangled.
    Expected NOT to rank players the same as scarcity_density (same lesson as
    cpv_max/cpv_avg, HISTORY.md §15) -- difference is diagnosed, not tuned away.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))

from role_space import (
    RECOMMENDED_NETWORK_FEATURES,
    STYLE_FEATURES,
    TERRITORY_FEATURES,
    build_role_features_from_passes,
    infer_position,
    normalize_within_team,
)

SCARCITY_FEATURES = RECOMMENDED_NETWORK_FEATURES + TERRITORY_FEATURES + STYLE_FEATURES
MIN_PASSES = 15


def build_population(
    passes: pd.DataFrame,
    min_passes: int = MIN_PASSES,
    exclude_gk: bool = True,
    position_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Role vectors for the scarcity population: identified, normalized, position-labeled.

    position_labels: {player_id: "GK"/"DEF"/"MID"/"FWD"} from real position data
    (statsbomb_io.player_position_labels) when available -- strongly preferred, see
    HISTORY.md #29. Falls back to role_space.infer_position's spatial heuristic (raw
    mean_x thresholds) only for players missing from the map, or entirely when no map
    is given (e.g. Metrica, which has no position ground truth at all).
    """
    role = build_role_features_from_passes(passes, min_passes=min_passes)
    role = infer_position(role)  # spatial fallback; must run BEFORE normalization (needs raw mean_x/dispersion)
    if position_labels:
        real = role["player_id"].map(position_labels)
        role["inferred_position"] = real.where(real.notna(), role["inferred_position"])
    if exclude_gk:
        role = role[role["inferred_position"] != "GK"].reset_index(drop=True)
    role = normalize_within_team(role, features=SCARCITY_FEATURES)
    return role


def pca_embedding(role: pd.DataFrame, n_components: float = 0.90) -> tuple[np.ndarray, PCA]:
    """PCA-decorrelated role vectors. n_components<1 keeps enough PCs for that variance share."""
    X = role[SCARCITY_FEATURES].to_numpy()
    pca = PCA(n_components=n_components, random_state=0)
    Z = pca.fit_transform(X)
    return Z, pca


def scarcity_density(Z: np.ndarray, bandwidth: float | None = None) -> tuple[np.ndarray, float]:
    """-log(mean Gaussian-kernel similarity to the rest of the population).

    bandwidth: if None, uses the median pairwise distance (a standard KDE default).
    """
    dist = squareform(pdist(Z, metric="euclidean"))
    if bandwidth is None:
        iu = np.triu_indices_from(dist, k=1)
        bandwidth = float(np.median(dist[iu]))

    kernel = np.exp(-0.5 * (dist / bandwidth) ** 2)
    np.fill_diagonal(kernel, np.nan)
    density = np.nanmean(kernel, axis=1)
    return -np.log(density + 1e-12), bandwidth


def scarcity_pool(Z: np.ndarray, quality: np.ndarray, radius: float | None = None) -> tuple[np.ndarray, float]:
    """-log((|{similar AND >= as good}| + 1) / N) -- the replacement-pool variant.

    radius: if None, uses the 25th percentile pairwise distance ("similar" = closer
    than most pairs in the population).
    """
    dist = squareform(pdist(Z, metric="euclidean"))
    if radius is None:
        iu = np.triu_indices_from(dist, k=1)
        radius = float(np.percentile(dist[iu], 25))

    n = len(Z)
    out = np.full(n, np.nan)
    valid_q = ~np.isnan(quality)
    for i in range(n):
        if not valid_q[i]:
            continue
        similar = (dist[i] <= radius) & (np.arange(n) != i) & valid_q
        as_good = similar & (quality >= quality[i])
        out[i] = -np.log((as_good.sum() + 1) / n)
    return out, radius


def run_s3_orthogonality(scarcity_vals: np.ndarray, quality_vals: np.ndarray) -> float:
    """|correlation(scarcity_density, quality)| should be low -- role space should not
    have leaked quality information. See docs/scarcity.md S3."""
    valid = ~np.isnan(quality_vals)
    return float(np.corrcoef(scarcity_vals[valid], quality_vals[valid])[0, 1])


def run_s4_bandwidth_sensitivity(Z: np.ndarray, base_bandwidth: float) -> float:
    """Rank correlation of scarcity under two reasonable bandwidths. Need >= 0.7."""
    s_narrow, _ = scarcity_density(Z, bandwidth=base_bandwidth * 0.6)
    s_wide, _ = scarcity_density(Z, bandwidth=base_bandwidth * 1.6)
    return float(spearmanr(s_narrow, s_wide).statistic)
