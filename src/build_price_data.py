"""Link the StatsBomb player population to Transfermarkt prices — the third pipeline leg.

Produces `output/price_data.pkl`: one row per linked player with point-in-time market
value, plus their CPV quality and scarcity, ready for the surplus-value regression.

POINT-IN-TIME VALUES ARE THE WHOLE GAME HERE. Each player's value is looked up as of a
fixed valuation date within the observed season, NOT their current 2026 value. Using
current values against 2023/24 performance would leak the future into the model: a
player who broke out AFTER the observed season would look "underpriced" purely because
the model was shown his later valuation. That is precisely the residual the surplus
model is supposed to discover, so leaking it would make the headline result circular.

Run: python src/build_price_data.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import transfermarkt_io as TM

OUTPUT_DIR = Path(__file__).parent.parent / "output"
SEASON_START, SEASON_END = "2023-08-01", "2024-05-31"
COMPETITION = "L1"
VALUATION_DATE = "2023-08-01"  # start of season: what the market believed BEFORE the observed performance

# Known-bad Transfermarkt links, found by audit (HISTORY.md #30), not by any general
# rule -- the club-restriction and name-matching machinery both worked correctly here;
# the defect is upstream, in Transfermarkt's own `appearances` table. It attributes ~250
# real minutes for Bayer Leverkusen (7 games, Feb-May 2024, competitions L1+DFB) to
# tm_player_id 278359, whose `players` profile (born 1993, senior Spain international,
# current club Celta de Vigo, GBP20M valuation) describes an established veteran striker
# with no independently corroborated Leverkusen spell. Almost certainly two different
# real people sharing the name "Borja Iglesias" were merged under one ID upstream. A
# swept audit of every OTHER non-exact match in this dataset found no comparable case
# (see HISTORY.md #30 for the full audit) -- this is a one-off data defect, not evidence
# the linkage method itself is unreliable.
KNOWN_BAD_LINKS = {
    (278359, "Bayer Leverkusen"): "appearances table conflates the Betis/Celta striker's "
    "profile with a distinct ~250-minute Leverkusen fringe player; not a real price observation",
}


def season_squads(competition: str = COMPETITION):
    """{club_id: [player_id]} from ACTUAL appearances in the observed season.

    Deliberately not `players.current_club_id`, which reflects each player's club today
    (2026) — anyone transferred since would be filtered into the wrong squad.
    """
    app = TM.load("appearances")
    app["date"] = pd.to_datetime(app["date"], errors="coerce")
    season = app[(app.date >= SEASON_START) & (app.date <= SEASON_END) & (app.competition_id == competition)]
    return season.groupby("player_club_id")["player_id"].unique().to_dict()


def build(valuation_date: str = VALUATION_DATE) -> pd.DataFrame:
    role = pd.read_pickle(OUTPUT_DIR / "sb_scarcity.pkl")
    passes = pd.read_pickle(OUTPUT_DIR / "sb_all_passes.pkl")
    names = (
        passes.drop_duplicates("passer_id")
        .assign(pid=lambda d: d.passer_id.astype(int).astype(str))
        .set_index("pid")["passer"]
    )
    role = role.assign(name=role.player_id.map(names))

    squads = season_squads()
    clubs_all = TM.load("clubs").set_index("club_id")["name"].to_dict()
    tm_clubs = {cid: clubs_all[cid] for cid in squads if cid in clubs_all}
    club_map = TM.match_clubs(sorted(role.team.unique()), tm_clubs)
    club_filter = {t: list(squads.get(cid, [])) for t, cid in club_map.items()}

    tm_players = TM.load("players")
    sb_players = role[["player_id", "name", "team"]].rename(columns={"player_id": "player_id"})
    links = TM.link_players(sb_players, tm_players, club_filter=club_filter)

    valuations = TM.load("player_valuations")
    valuations["date"] = pd.to_datetime(valuations["date"])
    vals = []
    for tm_id in links["tm_player_id"]:
        vals.append(None if pd.isna(tm_id) else TM.value_at_date(valuations, int(tm_id), valuation_date))
    links["market_value_eur"] = vals

    for _, row in links.iterrows():
        key = (row["tm_player_id"], row["sb_team"])
        if key in KNOWN_BAD_LINKS:
            print(f"  excluding known-bad link: {row['sb_name']} ({row['sb_team']}) -> "
                  f"tm_id={int(row['tm_player_id'])}: {KNOWN_BAD_LINKS[key]}")
            links.loc[links["sb_player_id"] == row["sb_player_id"], ["tm_player_id", "market_value_eur", "method"]] = [None, None, "excluded"]

    # Join on (player_id, team), not player_id alone: a handful of players (e.g. Amiri,
    # genuinely transferred Leverkusen->Mainz mid-season) appear as two rows sharing one
    # player_id, split by team. Joining on player_id only produced a many-to-many
    # fanout (2 role rows x 2 links rows = 4 output rows for one real person) -- caught
    # via a duplicate-row check before this was trusted. See HISTORY.md #30.
    out = role.merge(
        links[["sb_player_id", "sb_team", "tm_player_id", "tm_name", "score", "method", "market_value_eur"]],
        left_on=["player_id", "team"], right_on=["sb_player_id", "sb_team"], how="left",
    )
    assert len(out) == len(role), f"merge changed row count: {len(role)} -> {len(out)} (join key too loose)"
    return out


if __name__ == "__main__":
    df = build()
    df.to_pickle(OUTPUT_DIR / "price_data.pkl")

    linked = df.tm_player_id.notna()
    priced = df.market_value_eur.notna()
    print(f"population: {len(df)}")
    print(f"  linked to Transfermarkt: {linked.sum()} ({linked.mean():.1%})")
    print(f"  with a value at {VALUATION_DATE}: {priced.sum()} ({priced.mean():.1%})")
    print()
    print("link method breakdown:")
    print(df.method.value_counts().to_string())
    print()
    print("unlinked / ambiguous (need manual review):")
    bad = df[~linked][["name", "team", "score", "method"]]
    print(bad.to_string(index=False) if len(bad) else "  none")
    print()
    print("market value distribution (EUR):")
    print(df.market_value_eur.describe().apply(lambda v: f"{v:,.0f}").to_string())
    print()
    print("=== Leverkusen: value vs. our metrics ===")
    lev = df[(df.team == "Bayer Leverkusen") & priced].sort_values("market_value_eur", ascending=False)
    show = lev[["name", "inferred_position", "market_value_eur", "quality", "scarcity_density", "n_passes"]]
    print(show.assign(market_value_eur=show.market_value_eur.map(lambda v: f"{v/1e6:,.1f}M")).to_string(index=False))
