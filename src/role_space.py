"""Role-space feature extraction — the "how a player plays" half of scarcity.

See docs/scarcity.md for the definition this implements. Scarcity itself (the rarity
statistic) is NOT computed here: it needs a population of identified players, which
Metrica cannot provide (~28 anonymous slots per match, no cross-match linkage). This
module builds and validates the *machinery* — passing network, centralities, territory
and style features — which is the part Metrica can support (criteria S1/S2).

CORE DESIGN PRINCIPLE (docs/scarcity.md): role space encodes HOW a player plays, never
HOW WELL. No success-based feature may enter these vectors — no completion %, no CPV,
no xT added. If quality leaks in, "scarce" degenerates into "good" and the metric
becomes collinear with impact in the surplus model, contributing nothing. Criterion S3
tests this, so keep the boundary clean here.

NORMALIZATION: raw centrality conflates a player's role with their team's style and
possession volume. Features are z-scored within team-match before pooling, so a vector
encodes role *within the team's structure* rather than the team's overall behaviour.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from completion_model import attempt_masks, event_player_to_id, norm_to_metric
from expected_threat import _attack_directions

# Features are grouped so callers (and validation) can reason about them separately.
NETWORK_FEATURES = ["betweenness", "pagerank", "in_out_ratio", "clustering"]

# betweenness is computed and returned for reproducibility of the S2 finding, but is
# EXCLUDED from this recommended set: football passing networks are ~70% complete
# (measured density 0.64-0.74), so observed betweenness spans only [0, 0.10] of [0, 1]
# and rank order is noise-dominated. Alone among all features its split-half stability
# does not improve with more data -- structural, not small-sample. See docs/scarcity.md
# S1/S2 results. Re-include only if computed on a thresholded (sparsified) graph.
RECOMMENDED_NETWORK_FEATURES = ["pagerank", "in_out_ratio", "clustering"]
TERRITORY_FEATURES = ["mean_x", "mean_y", "dispersion", "third_def", "third_mid", "third_att"]
STYLE_FEATURES = ["fwd_share", "lat_share", "back_share", "mean_pass_dist", "std_pass_dist", "long_share"]
ROLE_FEATURES = NETWORK_FEATURES + TERRITORY_FEATURES + STYLE_FEATURES

LONG_PASS_M = 30.0


def build_passing_network(events: pd.DataFrame, team: str) -> nx.DiGraph:
    """Directed weighted pass graph for one team: edge u->v weighted by pass count.

    Uses completed passes only — an intercepted pass never reached its target, so it
    carries no information about who connects to whom. This is a structural choice,
    not a quality judgement: the graph describes realized connections.
    """
    is_pass, _ = attempt_masks(events)
    team_passes = events[is_pass & (events["Team"].str.lower() == team)]

    graph = nx.DiGraph()
    for _, ev in team_passes.iterrows():
        passer, receiver = event_player_to_id(ev["From"]), event_player_to_id(ev["To"])
        if passer is None or receiver is None or passer == receiver:
            continue
        if graph.has_edge(passer, receiver):
            graph[passer][receiver]["weight"] += 1
        else:
            graph.add_edge(passer, receiver, weight=1)
    return graph


def network_features(graph: nx.DiGraph) -> pd.DataFrame:
    """Centrality/structure features per player.

    Betweenness uses distance = 1/weight: a frequently-used pass lane is a SHORT path,
    so betweenness measures bridging along the connections the team actually uses.
    """
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["player_id"] + NETWORK_FEATURES)

    dist_graph = graph.copy()
    for _, _, data in dist_graph.edges(data=True):
        data["distance"] = 1.0 / data["weight"]

    betweenness = nx.betweenness_centrality(dist_graph, weight="distance", normalized=True)
    try:
        pagerank = nx.pagerank(graph, weight="weight")
    except nx.PowerIterationFailedConvergence:  # disconnected/degenerate graphs
        pagerank = {n: np.nan for n in graph.nodes}
    # Clustering on the undirected projection: "do my pass partners pass to each other".
    clustering = nx.clustering(graph.to_undirected(), weight="weight")

    rows = []
    for node in graph.nodes:
        in_w = sum(d["weight"] for _, _, d in graph.in_edges(node, data=True))
        out_w = sum(d["weight"] for _, _, d in graph.out_edges(node, data=True))
        rows.append(
            {
                "player_id": node,
                "betweenness": betweenness.get(node, np.nan),
                "pagerank": pagerank.get(node, np.nan),
                # +1 smoothing keeps this finite for pure receivers/distributors.
                "in_out_ratio": (in_w + 1) / (out_w + 1),
                "clustering": clustering.get(node, np.nan),
            }
        )
    return pd.DataFrame(rows)


def territory_style_features(events: pd.DataFrame, team: str) -> pd.DataFrame:
    """Where a player operates and how they distribute — no success information.

    Coordinates are oriented so attacking is toward +x (per-team, per-period), matching
    the xT convention, so mean_x is comparable across teams and halves.
    """
    directions = _attack_directions(events)
    is_pass, _ = attempt_masks(events)
    team_passes = events[is_pass & (events["Team"].str.lower() == team)]

    per_player: dict[str, dict[str, list]] = {}
    for _, ev in team_passes.iterrows():
        passer = event_player_to_id(ev["From"])
        if passer is None or pd.isna(ev["Start X"]) or pd.isna(ev["End X"]):
            continue
        direction = directions.get((ev["Team"], int(ev["Period"])), 1)
        sx, sy = norm_to_metric(ev["Start X"], ev["Start Y"])
        ex, ey = norm_to_metric(ev["End X"], ev["End Y"])
        if direction == -1:
            sx, sy, ex, ey = -sx, -sy, -ex, -ey

        rec = per_player.setdefault(passer, {"x": [], "y": [], "dx": [], "dist": []})
        rec["x"].append(sx)
        rec["y"].append(sy)
        rec["dx"].append(ex - sx)
        rec["dist"].append(float(np.hypot(ex - sx, ey - sy)))

    rows = []
    for pid, rec in per_player.items():
        xs, ys = np.array(rec["x"]), np.array(rec["y"])
        dx, dist = np.array(rec["dx"]), np.array(rec["dist"])
        n = len(xs)
        if n == 0:
            continue
        # Pitch thirds in metric coords (pitch length 105m, centered): [-52.5,-17.5,17.5,52.5]
        rows.append(
            {
                "player_id": pid,
                "n_passes": n,
                "mean_x": float(xs.mean()),
                "mean_y": float(ys.mean()),
                "dispersion": float(np.sqrt(xs.var() + ys.var())),
                "third_def": float(np.mean(xs < -17.5)),
                "third_mid": float(np.mean((xs >= -17.5) & (xs <= 17.5))),
                "third_att": float(np.mean(xs > 17.5)),
                # Direction shares: "forward" = meaningfully progressive, not just >0.
                "fwd_share": float(np.mean(dx > 5)),
                "lat_share": float(np.mean(np.abs(dx) <= 5)),
                "back_share": float(np.mean(dx < -5)),
                "mean_pass_dist": float(dist.mean()),
                "std_pass_dist": float(dist.std()),
                "long_share": float(np.mean(dist > LONG_PASS_M)),
            }
        )
    return pd.DataFrame(rows)


def build_role_features(events: pd.DataFrame, min_passes: int = 10) -> pd.DataFrame:
    """Role vectors for every player in one match (both teams).

    min_passes: centrality and distribution shares are unstable for players with few
    touches (docs/scarcity.md open decision 3). Substitutes typically fall below this.
    """
    frames = []
    for team in ("home", "away"):
        graph = build_passing_network(events, team)
        net = network_features(graph)
        ter = territory_style_features(events, team)
        if net.empty or ter.empty:
            continue
        merged = ter.merge(net, on="player_id", how="left")
        merged["team"] = team
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=["player_id", "team"] + ROLE_FEATURES)
    out = pd.concat(frames, ignore_index=True)
    return out[out["n_passes"] >= min_passes].reset_index(drop=True)


def build_role_features_from_passes(passes: pd.DataFrame, min_passes: int = 10) -> pd.DataFrame:
    """Role vectors from a tidy pass table — provider-agnostic path.

    Expects metric, already-attack-oriented coordinates (+x toward the opponent goal)
    and columns: team, passer_id, recipient_id, completed, start_x/y, end_x/y. The
    StatsBomb loader emits exactly this; Metrica goes through build_role_features()
    instead because it needs per-period orientation first.

    Same feature definitions as the Metrica path — only the input plumbing differs, so
    stability results remain comparable across providers.
    """
    frames = []
    for team, team_passes in passes.groupby("team"):
        completed = team_passes[team_passes["completed"] == 1]

        graph = nx.DiGraph()
        for passer, receiver in zip(completed["passer_id"], completed["recipient_id"]):
            if pd.isna(passer) or pd.isna(receiver) or passer == receiver:
                continue
            passer, receiver = str(int(passer)), str(int(receiver))
            if graph.has_edge(passer, receiver):
                graph[passer][receiver]["weight"] += 1
            else:
                graph.add_edge(passer, receiver, weight=1)
        net = network_features(graph)

        rows = []
        for pid, grp in team_passes.groupby("passer_id"):
            xs, ys = grp["start_x"].to_numpy(), grp["start_y"].to_numpy()
            dx = (grp["end_x"] - grp["start_x"]).to_numpy()
            dist = np.hypot(grp["end_x"] - grp["start_x"], grp["end_y"] - grp["start_y"]).to_numpy()
            rows.append(
                {
                    "player_id": str(int(pid)),
                    "n_passes": len(grp),
                    "mean_x": float(xs.mean()), "mean_y": float(ys.mean()),
                    "dispersion": float(np.sqrt(xs.var() + ys.var())),
                    "third_def": float(np.mean(xs < -17.5)),
                    "third_mid": float(np.mean((xs >= -17.5) & (xs <= 17.5))),
                    "third_att": float(np.mean(xs > 17.5)),
                    "fwd_share": float(np.mean(dx > 5)),
                    "lat_share": float(np.mean(np.abs(dx) <= 5)),
                    "back_share": float(np.mean(dx < -5)),
                    "mean_pass_dist": float(dist.mean()),
                    "std_pass_dist": float(dist.std()),
                    "long_share": float(np.mean(dist > LONG_PASS_M)),
                }
            )
        ter = pd.DataFrame(rows)
        if ter.empty:
            continue
        merged = ter.merge(net, on="player_id", how="left")
        merged["team"] = team
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=["player_id", "team"] + ROLE_FEATURES)
    out = pd.concat(frames, ignore_index=True)
    return out[out["n_passes"] >= min_passes].reset_index(drop=True)


def normalize_within_team(role_df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Z-score role features within each team so vectors encode role-within-structure.

    Without this, a metronome in a possession-dominant side outranks an identical player
    in a counter-attacking one purely on team style (docs/scarcity.md, Normalization).
    """
    features = features or ROLE_FEATURES
    out = role_df.copy()
    for team, group in role_df.groupby("team"):
        for col in features:
            vals = group[col].astype(float)
            std = vals.std()
            out.loc[group.index, col] = 0.0 if std == 0 or np.isnan(std) else (vals - vals.mean()) / std
    return out


@dataclass
class InferredPosition:
    """Coarse position label inferred from territory, for S1 face-validity only.

    Metrica is anonymized, so there are no ground-truth position labels. Mean pitch
    position is a crude but independent proxy — crucially it is NOT one of the network
    features, so checking network structure against it is a real test rather than a
    tautology.
    """

    player_id: str
    team: str
    label: str


def infer_position(role_df: pd.DataFrame) -> pd.DataFrame:
    """Label players GK / DEF / MID / FWD from mean position and dispersion.

    Uses UNNORMALIZED metric coordinates, so call before normalize_within_team.

    GK detection went through two revisions before landing here (HISTORY.md §27):
    (1) the original absolute rule (x < -30 AND dispersion < 12) was tuned against
    full-season Metrica volumes and misclassified two real keepers as DEF at StatsBomb
    scale: with only ~2 matches of data, a couple of long clearances/goal-kicks inflate
    sample dispersion past a fixed cutoff (Neuer: disp=12.2 on 36 passes; Riemann:
    disp=14.7 on 109). (2) Switching to "deepest mean_x per team" fixed that but
    introduced a new failure: it assumes exactly one goalkeeper per team, so any team
    that used a backup keeper across the sampled matches got a real second keeper
    mislabeled as DEF (5 cases: Kovar, Ulreich, Pervan, Omlin, Dahmen).

    The rule that actually holds: checked mean_x for the deepest OUTFIELD player on
    every one of 18 teams against every known goalkeeper's mean_x, and found a clean,
    total gap -- every goalkeeper (starter or backup) sits below -36.8; no outfield
    player on any team goes deeper than -27.2. So a simple ABSOLUTE threshold on mean_x
    alone (no dispersion term, no per-team cardinality assumption) is both simpler and
    correctly justified by the data, and is robust to both prior failure modes: it
    doesn't care about sample-size-driven dispersion noise, and it doesn't assume how
    many keepers a team used.
    """
    out = role_df.copy()
    is_gk = out["mean_x"] < -30

    labels = []
    for gk, x in zip(is_gk, out["mean_x"]):
        if gk:
            labels.append("GK")
        elif x < -12:
            labels.append("DEF")
        elif x < 12:
            labels.append("MID")
        else:
            labels.append("FWD")
    out["inferred_position"] = labels
    return out
