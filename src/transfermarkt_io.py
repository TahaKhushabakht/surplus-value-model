"""Transfermarkt price data — loader and StatsBomb linkage.

PROVENANCE (important, read before extending). This does NOT scrape Transfermarkt.
Their robots.txt explicitly disallows AI agents by name:

    User-agent: ClaudeBot        User-agent: GPTBot
    Disallow: /                  Disallow: /

Instead we use dcaribou/transfermarkt-datasets — an established (461-star), weekly
refreshed, **CC0-1.0** (public-domain-dedication) redistribution published to
Cloudflare R2 / Kaggle / data.world. Besides being the sanctioned route, it is
genuinely the better engineering choice: pre-cleaned, versioned, no scraper fragility,
and it carries `player_valuations` as a TIME SERIES rather than the single current
snapshot a naive scrape of a player page would yield. The time series is what makes the
parked "age-adjusted value cliff" idea (NOTES.md) testable at all.

THE HARD PART IS LINKAGE, NOT ACQUISITION. StatsBomb and Transfermarkt assign unrelated
player IDs, so joining them is a name-matching problem — and name matching silently
produces WRONG rows rather than missing ones, which is the dangerous failure mode for a
valuation model. Accents ("Hincapié"), transliteration ("Kovar"/"Kovář"), name order,
and short/full forms ("Grimaldo" vs "Alejandro Grimaldo García") all differ between
providers. `link_players` therefore:
  - normalizes aggressively (unicode-fold, lowercase, strip punctuation),
  - restricts candidates by club and season BEFORE matching, which collapses the search
    space from ~37k players to ~30 and makes a wrong match far less likely,
  - returns a per-match score and method so low-confidence links can be filtered or
    audited rather than trusted blindly.
Nothing here auto-accepts a fuzzy match; `audit_links()` exists to eyeball them.
"""

import sys
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent.parent / "data" / "transfermarkt"
ZIP_PATH = DATA_DIR / "transfermarkt-datasets.zip"

# Tables we actually need; the archive ships 12.
CORE_TABLES = ["players", "player_valuations", "transfers", "clubs", "appearances"]


def extract(zip_path: Path = ZIP_PATH, dest: Path = DATA_DIR) -> list[str]:
    """Unpack the archive once; returns the table files available.

    The archive ships gzipped CSVs (`players.csv.gz`), not plain ones — pandas reads
    the .gz directly, so they are left compressed rather than double-expanded on disk.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith((".csv", ".csv.gz"))]
        zf.extractall(dest)
    return names


def load(table: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load one table by name, wherever it landed inside the archive."""
    for pattern in (f"{table}.csv.gz", f"{table}.csv"):
        matches = list(Path(data_dir).rglob(pattern))
        if matches:
            return pd.read_csv(matches[0], low_memory=False)
    raise FileNotFoundError(f"{table}.csv[.gz] not found under {data_dir}; run extract() first")


def normalize_name(name: str) -> str:
    """Fold accents/punctuation/case so provider spelling differences don't block a match.

    'Piero Martín Hincapié Reyna' -> 'piero martin hincapie reyna'
    'Edmond Fayçal Tapsoba'       -> 'edmond faycal tapsoba'
    """
    if not isinstance(name, str):
        return ""
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.lower()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in folded).split())


def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap on name tokens — handles 'Grimaldo' vs 'Alejandro Grimaldo García'
    and differing name order, which plain string similarity handles badly."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _surname_anchor(a: str, b: str) -> bool:
    """Do two names share a distinctive surname token?

    Providers often differ on given names rather than surnames: StatsBomb records full
    legal names ('Daniel Olmo Carvajal', 'Dayotchanculle Upamecano') where Transfermarkt
    uses common forms ('Dani Olmo', 'Dayot Upamecano'). Jaccard punishes these because
    the union is large, so they fell below threshold despite being obviously the same
    player. Requiring a shared token of length >= 4 is only safe BECAUSE the candidate
    pool is already restricted to one club's ~30-player squad; it would be reckless
    against all 50k players.
    """
    ta = {t for t in a.split() if len(t) >= 4}
    tb = {t for t in b.split() if len(t) >= 4}
    return bool(ta & tb)


CLUB_STOPWORDS = {
    "fc", "sc", "sv", "vfl", "vfb", "tsg", "spvgg", "bsc", "fsv", "1", "04", "05", "07",
    "98", "96", "1846", "1899", "1900", "borussia", "eintracht", "fussballclub",
    "fusballclub", "club", "de", "of",
}


def club_core_tokens(name: str) -> set[str]:
    """Distinguishing tokens of a club name, with generic football-club words removed.

    'FC Heidenheim' and '1. Fussballclub Heidenheim 1846' share only the generic token
    'fc'-family plus 'heidenheim'; plain token overlap scored that pairing 0.33 — LOWER
    than its overlap with 'FC Augsburg', which silently produced a wrong match. Stripping
    stopwords leaves {'heidenheim'} on both sides, which matches uniquely and correctly.
    """
    return {t for t in normalize_name(name).split() if t not in CLUB_STOPWORDS}


def match_clubs(sb_teams, tm_clubs: dict[int, str]) -> dict[str, int]:
    """Map StatsBomb team names to Transfermarkt club_ids via distinguishing tokens.

    tm_clubs: {club_id: club_name}, ideally already restricted to the relevant season's
    clubs so the search space is ~18 rather than thousands.
    """
    out = {}
    for team in sb_teams:
        core = club_core_tokens(team)
        best, best_score = None, 0.0
        for cid, cname in tm_clubs.items():
            tokens = club_core_tokens(cname)
            if not core or not tokens:
                continue
            score = len(core & tokens) / len(core | tokens)
            if score > best_score:
                best, best_score = cid, score
        out[team] = best
    return out


def link_players(
    sb_players: pd.DataFrame,
    tm_players: pd.DataFrame,
    club_filter: dict[str, list[int]] | None = None,
    min_score: float = 0.34,
) -> pd.DataFrame:
    """Link StatsBomb players to Transfermarkt player_ids by name, scoped by club.

    sb_players: columns [player_id, name, team]
    tm_players: Transfermarkt `players` table
    club_filter: {statsbomb_team_name: [transfermarkt player_id, ...]} -- the set of
        Transfermarkt player_ids eligible for that team (e.g. from actual appearances
        in the observed season, NOT `players.current_club_id`, which reflects each
        player's club as of the dataset's last refresh, not the season being studied).
        Strongly recommended: matching within a ~30-player squad instead of 50k
        players is what makes name matching safe here.
        CAUGHT BUG (HISTORY.md #30): an earlier version filtered via
        `current_club_id.isin(club_filter[team])`, but callers pass PLAYER ids here,
        not club ids -- comparing the two matches almost nothing, so the filter
        silently fell back to the full unfiltered table for every non-exact lookup.
        "Ambiguous: multiple candidates" for names like "Arthur" or "Iago" was this bug
        showing up as a symptom (common names collide across the whole league/eras);
        against the correctly-restricted ~30-player pool each resolved to exactly one
        player. Filtering is now on `player_id` directly, matching what is actually
        passed and what the season-appearances squad list actually enumerates.

    Returns one row per StatsBomb player with the best candidate, its score, and the
    method used, so links can be audited/filtered rather than silently trusted.
    """
    tm = tm_players.copy()
    tm["norm"] = tm["name"].map(normalize_name)

    rows = []
    for r in sb_players.itertuples(index=False):
        sb_norm = normalize_name(r.name)
        pool = tm
        if club_filter and r.team in club_filter:
            pool = tm[tm["player_id"].isin(club_filter[r.team])]
            if pool.empty:
                pool = tm

        exact = pool[pool["norm"] == sb_norm]
        if len(exact):
            best, score, method = exact.iloc[0], 1.0, "exact"
        else:
            scores = pool["norm"].map(lambda x: _token_overlap(sb_norm, x))
            if len(scores) and scores.max() >= min_score:
                best, score, method = pool.loc[scores.idxmax()], float(scores.max()), "token"
            else:
                # Surname-anchored fallback, safe only within a club-restricted pool.
                anchored = pool[pool["norm"].map(lambda x: _surname_anchor(sb_norm, x))]
                if len(anchored) == 1:
                    best, score, method = anchored.iloc[0], float(scores.max()) if len(scores) else 0.0, "surname"
                else:
                    rows.append({"sb_player_id": r.player_id, "sb_name": r.name, "sb_team": r.team,
                                 "tm_player_id": None, "tm_name": None,
                                 "score": float(scores.max()) if len(scores) else 0.0,
                                 "method": "ambiguous" if len(anchored) > 1 else "none"})
                    continue

        rows.append({
            "sb_player_id": r.player_id, "sb_name": r.name, "sb_team": r.team,
            "tm_player_id": int(best["player_id"]), "tm_name": best["name"],
            "score": score, "method": method,
        })
    return pd.DataFrame(rows)


def value_at_date(valuations: pd.DataFrame, tm_player_id: int, date: str) -> float | None:
    """Market value in effect for a player on a given date (most recent prior quote).

    Point-in-time lookup matters: using a player's CURRENT value against their
    2023/24 performance would leak the future into the model — the very thing the
    surplus-value residual is supposed to detect.
    """
    v = valuations[valuations["player_id"] == tm_player_id].copy()
    if v.empty:
        return None
    v["date"] = pd.to_datetime(v["date"])
    prior = v[v["date"] <= pd.Timestamp(date)].sort_values("date")
    if prior.empty:
        return None
    return float(prior.iloc[-1]["market_value_in_eur"])


def audit_links(links: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Lowest-confidence links first — the ones worth eyeballing before trusting."""
    return links.sort_values("score").head(n)[["sb_name", "sb_team", "tm_name", "score", "method"]]
