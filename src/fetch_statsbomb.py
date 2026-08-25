"""Download one StatsBomb open-data competition's events + 360 freeze frames.

Defaults to Bayer Leverkusen's 2023/24 Bundesliga season: 34 matches, all with 360
coverage, and the deepest per-player volume available anywhere in the open dataset
(~25 players at up to 34 matches each). That volume is the point — it is what lets us
re-test the S2 stability prediction that Metrica's 2 matches could not.

Data lands in data/statsbomb/ (gitignored). Roughly 500MB; skips files already present
so re-running is cheap and interruption-safe.

Run: python src/fetch_statsbomb.py
"""

import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DATA_DIR = Path(__file__).parent.parent / "data" / "statsbomb"

# Bundesliga 2023/24 — the Leverkusen season.
COMPETITION_ID = 9
SEASON_ID = 281
FOCUS_TEAM = "Bayer Leverkusen"


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "surplus-value-model/research"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def _download(url: str, dest: Path) -> tuple[bool, int]:
    """Returns (downloaded_now, size_bytes). Skips if the file already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        return False, dest.stat().st_size
    req = urllib.request.Request(url, headers={"User-Agent": "surplus-value-model/research"})
    with urllib.request.urlopen(req) as resp:
        payload = resp.read()
    dest.write_bytes(payload)
    return True, len(payload)


def fetch_season(competition_id: int = COMPETITION_ID, season_id: int = SEASON_ID, focus_team: str | None = FOCUS_TEAM):
    (DATA_DIR / "events").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "three-sixty").mkdir(parents=True, exist_ok=True)

    matches = _get_json(f"{BASE}/matches/{competition_id}/{season_id}.json")
    if focus_team:
        matches = [
            m for m in matches
            if focus_team in (m["home_team"]["home_team_name"], m["away_team"]["away_team_name"])
        ]
    matches.sort(key=lambda m: m.get("match_date", ""))
    (DATA_DIR / "matches.json").write_text(json.dumps(matches, indent=2), encoding="utf-8")
    print(f"{len(matches)} matches for competition={competition_id} season={season_id} team={focus_team}")

    total_bytes = 0
    for i, match in enumerate(matches, 1):
        mid = match["match_id"]
        label = f"{match['home_team']['home_team_name']} v {match['away_team']['away_team_name']}"
        for kind in ("events", "three-sixty"):
            dest = DATA_DIR / kind / f"{mid}.json"
            try:
                fetched, size = _download(f"{BASE}/{kind}/{mid}.json", dest)
            except Exception as exc:  # 360 is missing for a few matches; not fatal
                print(f"  [{i}/{len(matches)}] {mid} {kind}: FAILED ({exc})")
                continue
            total_bytes += size
            if fetched:
                print(f"  [{i}/{len(matches)}] {label[:44]:<44} {kind:<11} {size/1e6:5.1f}MB")

    print(f"done. total on disk: {total_bytes/1e6:.0f}MB in {DATA_DIR}")


if __name__ == "__main__":
    fetch_season()
    sys.exit(0)
