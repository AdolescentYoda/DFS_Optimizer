import csv
import datetime
import json
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo
import requests

# Constants & Paths
API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
REQUEST_TIMEOUT = 10
# Must NOT share a filename with app.py's cache (odds_cache.json): app.py stores
# {player: {expected_fp, ...}} while this script stores raw events + odds.
CACHE_FILE = Path(__file__).parent / "odds_events_cache.json"
CACHE_MAX_AGE_HOURS = 24

# API key pool ordered with your current active keys first, then additional keys
API_KEYS = [
    "820c20a0c8f1690b173ea97be76556b8",
    "9a7da4e97cd144123ed2e84ba3ac147e",
    "d1911c691f3d8cdb7ecef7767fc3bdf8",
    "66b08ba88184aab99e9bb0d1fad83bca",
    "49875f488a183578e045b693cef295ae",
    "9d005ec0eb665c3ce52c0ba109d6030c",
    "9cb2283a9131c1cc6fe3185d25b7c68e","a59d8ff666d442dc0cf4b07fa26014a5",

]

PLAYER_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_pass_interceptions",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_anytime_td",
]
TEAM_MARKETS = ["spreads", "totals"]


def get_api_keys() -> list[str]:
  keys = []
  env_key = os.environ.get("ODDS_API_KEY", "").strip()
  if env_key:
    keys.append(env_key)
  for k in API_KEYS:
    cleaned = k.strip()
    if cleaned and cleaned not in keys:
      keys.append(cleaned)
  return keys


def normalize_name(name: str) -> str:
  if not name:
    return ""
  s = name.strip().split("\n")[0].casefold()
  s = s.replace("’", "'").replace("`", "'").replace(".", "")
  s = re.sub(r"[^a-z0-9 ]+", " ", s)
  s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
  return re.sub(r"\s+", " ", s).strip()


def normalize_team(team: str) -> str:
  if not team:
    return ""
  return team.strip().upper()


def is_main_slate_game(commence_str: str) -> bool:
  if not commence_str:
    return False
  try:
    dt_utc = datetime.datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
    dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
    # FanDuel main slate = Sunday 1:00 PM and 4:05/4:25 PM ET windows only.
    # Excludes the 9:30 AM international game and Sunday night.
    return dt_et.weekday() == 6 and (13 <= dt_et.hour <= 16)
  except Exception:
    return False


def fetch_vegas_odds_data(api_keys: list[str]) -> tuple[list[dict], list[tuple]]:
  if CACHE_FILE.exists():
    file_age_seconds = (
        datetime.datetime.now().timestamp() - CACHE_FILE.stat().st_mtime
    )
    if file_age_seconds < (CACHE_MAX_AGE_HOURS * 3600):
      print(
          f"[+] Loading cached odds data from '{CACHE_FILE.name}' (age:"
          f" {file_age_seconds / 3600:.1f} hrs)"
      )
      try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
          cached_data = json.load(f)
          sunday_events = [
              ev
              for ev in cached_data.get("sunday_events", [])
              if is_main_slate_game(ev.get("commence_time"))
          ]
          events_with_odds = [
              item
              for item in cached_data.get("events_with_odds", [])
              if is_main_slate_game(item[0].get("commence_time"))
          ]
          if events_with_odds:
            return sunday_events, events_with_odds
      except Exception as e:
        print(
            f"[!] Failed to read cache file ({e}). Fetching fresh data from"
            " Odds API..."
        )

  if not api_keys:
    raise RuntimeError(
        "No API keys available in ODDS_API_KEY or API_KEYS list."
    )

  print("[*] Querying Odds API using batched market requests...")
  session = requests.Session()
  session.headers.update({"User-Agent": "NFL-Lineup-Optimizer/1.0"})

  key_idx = 0
  active_key = api_keys[key_idx]
  raw_events = None

  def rotate_key(reason: str) -> bool:
    nonlocal key_idx, active_key
    print(f"[!] Key #{key_idx + 1} {reason}.")
    key_idx += 1
    if key_idx < len(api_keys):
      active_key = api_keys[key_idx]
      print(f"[*] Switching to API Key #{key_idx + 1}...")
      return True
    return False

  url = f"{API_BASE}/sports/{SPORT}/events"
  while True:
    resp = session.get(
        url,
        params={"apiKey": active_key, "dateFormat": "iso"},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code in (401, 403, 429):
      if rotate_key(f"returned HTTP {resp.status_code}"):
        continue
      raise RuntimeError(
          f"All API keys failed on events request ({resp.status_code}):"
          f" {resp.text}"
      )
    if resp.status_code != 200:
      raise RuntimeError(
          f"Odds API events request failed ({resp.status_code}): {resp.text}"
      )
    raw_events = resp.json()
    rem = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    if rem is not None or used is not None:
      print(
          f"[*] Using API Key #{key_idx + 1} (Credits Used: {used or '?'} |"
          f" Remaining: {rem or '?'})"
      )
    break

  sunday_events = [
      ev for ev in raw_events if is_main_slate_game(ev.get("commence_time"))
  ]

  batch_markets_str = ",".join(PLAYER_MARKETS + TEAM_MARKETS)
  events_with_odds = []

  for ev in sunday_events:
    event_id = ev["id"]
    while True:
      params = {
          "apiKey": active_key,
          "regions": "us",
          "markets": batch_markets_str,
          "oddsFormat": "american",
          "dateFormat": "iso",
      }
      mkt_url = f"{API_BASE}/sports/{SPORT}/events/{event_id}/odds"
      try:
        m_resp = session.get(mkt_url, params=params, timeout=REQUEST_TIMEOUT)
        if m_resp.status_code in (401, 403, 429):
          if rotate_key(
              f"hit HTTP {m_resp.status_code} while fetching event {event_id}"
          ):
            continue
          print(
              "[!] All API keys exhausted mid-fetch; stopping further odds"
              " requests."
          )
          break
        if m_resp.status_code == 200:
          data = m_resp.json()
          events_with_odds.append((ev, data))
        else:
          print(
              f"[!] Warning: Event {event_id} returned status"
              f" {m_resp.status_code}"
          )
        break
      except Exception as e:
        print(f"[!] Warning: Failed fetching odds for event {event_id}: {e}")
        break

  if events_with_odds:
    try:
      with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "sunday_events": sunday_events,
                "events_with_odds": events_with_odds,
            },
            f,
            indent=2,
        )
      print(f"[+] Saved fresh odds response to '{CACHE_FILE.name}'")
    except Exception as e:
      print(f"[!] Warning: Could not write cache file: {e}")

  return sunday_events, events_with_odds


def parse_market_for_players(event, bookmaker, market):
  results = []
  market_key = market.get("key")
  for outcome in market.get("outcomes", []):
    name = str(
        outcome.get("description") or outcome.get("participant") or ""
    ).strip()
    if not name:
      continue
    results.append({
        "event_id": event.get("id"),
        "team": normalize_team(outcome.get("team")),
        "market": market_key,
        "description_norm": normalize_name(name),
        "side": str(outcome.get("name", "")).strip().lower(),
        "point": (
            float(outcome["point"]) if outcome.get("point") is not None else None
        ),
        "price": (
            float(outcome["price"]) if outcome.get("price") is not None else None
        ),
    })
  return results


def main():
  try:
    api_keys = get_api_keys()
    if not api_keys and not CACHE_FILE.exists():
      raise SystemExit("No API keys configured and no cache file found.")

    _, events_with_odds = fetch_vegas_odds_data(api_keys)
    rows = []
    for ev, data in events_with_odds:
      for bookmaker in data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
          for row in parse_market_for_players(ev, bookmaker, market):
            row["book"] = bookmaker.get("key")
            rows.append(row)

    if not rows:
      raise SystemExit("No odds returned - check API key / credits / game day.")

    out = Path(__file__).parent / "nfl_player_props.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
      w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
      w.writeheader()
      w.writerows(rows)
    print(f"[+] Wrote {len(rows)} prop rows to {out}")
  except SystemExit as e:
    print(f"[!] {e}")
    input("\nPress Enter to exit...")
    raise
  except Exception as e:
    print(f"[!] Unexpected error: {e}")
    input("\nPress Enter to exit...")
    raise


if __name__ == "__main__":
  main()