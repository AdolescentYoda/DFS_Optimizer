from __future__ import annotations

import argparse
import csv
import datetime
import difflib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
import pandas as pd
import requests

# Force working directory to the script's folder
SCRIPT_DIR = Path(__file__).parent.resolve() if "__file__" in locals() else Path.cwd()
os.chdir(SCRIPT_DIR)

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
API_BASE = "https://api.the-odds-api.com/v4"
API_KEY = "9d005ec0eb665c3ce52c0ba109d6030c"
SPORT = "americanfootball_nfl"
REQUEST_TIMEOUT = 30
FUZZY_MATCH_THRESHOLD = 0.85
ESPN_MATCH_THRESHOLD = 0.90
CACHE_FILE = Path("odds_cache.json")
CACHE_MAX_AGE_HOURS = 24

PLAYER_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_rush_tds",
    "player_pass_interceptions",
    "player_receptions",
    "player_reception_yds",
    "player_anytime_td",
]

TEAM_MARKETS = ["spreads", "totals"]

POSITION_CONFIG = {
    "Quarterbacks (QB)": {
        "name_col": 2,
        "team_col": 3,
        "opponent_col": 4,
        "salary_col": 5,
        "start_row": 5,
        "header_row": 4,
        "headers": {
            6: "Pass Yds Prop",
            7: "Pass TD Odds",
            8: "Rush Yds Prop",
            9: "Rush TD Odds",
            10: "INT Prop",
            11: "Implied Pass TD %",
            12: "Implied Rush TD %",
            13: "Projected Score",
            14: "ESPN Projection",
            15: "Weighted Projection",
            16: "Value",
            17: "Projected Ownership",
        },
        "markets": {
            6: ("player_pass_yds", "line"),
            7: ("player_pass_tds", "over_price_1.5"),
            8: ("player_rush_yds", "line"),
            9: ("player_rush_tds", "yes_price"),
            10: ("player_pass_interceptions", "line"),
        },
        "odds_cols": {7, 9},
        "implied_pass_td_col": 11,
        "implied_rush_td_col": 12,
        "proj_score_col": 13,
        "espn_col": 14,
        "weighted_col": 15,
        "value_col": 16,
        "proj_ownership_col": 17,
    },
    "Running Backs (RB)": {
        "name_col": 2,
        "team_col": 3,
        "opponent_col": 4,
        "salary_col": 5,
        "start_row": 5,
        "header_row": 4,
        "headers": {
            6: "Rush Yds Prop",
            7: "Anytime TD Odds",
            8: "Receptions Prop",
            9: "Rec Yds Prop",
            10: "Implied TD %",
            11: "Projected Score",
            12: "ESPN Projection",
            13: "Weighted Projection",
            14: "Value",
            15: "Projected Ownership",
        },
        "markets": {
            6: ("player_rush_yds", "line"),
            7: ("player_anytime_td", "yes_price"),
            8: ("player_receptions", "line"),
            9: ("player_reception_yds", "line"),
        },
        "odds_cols": {7},
        "implied_td_col": 10,
        "proj_score_col": 11,
        "espn_col": 12,
        "weighted_col": 13,
        "value_col": 14,
        "proj_ownership_col": 15,
    },
    "Wide Receivers (WR)": {
        "name_col": 2,
        "team_col": 3,
        "opponent_col": 4,
        "salary_col": 5,
        "start_row": 5,
        "header_row": 4,
        "headers": {
            6: "Rec Yds Prop",
            7: "Anytime TD Odds",
            8: "Receptions Prop",
            9: "Implied TD %",
            10: "Projected Score",
            11: "ESPN Projection",
            12: "Weighted Projection",
            13: "Value",
            14: "Projected Ownership",
        },
        "markets": {
            6: ("player_reception_yds", "line"),
            7: ("player_anytime_td", "yes_price"),
            8: ("player_receptions", "line"),
        },
        "odds_cols": {7},
        "implied_td_col": 9,
        "proj_score_col": 10,
        "espn_col": 11,
        "weighted_col": 12,
        "value_col": 13,
        "proj_ownership_col": 14,
    },
    "Tight Ends (TE)": {
        "name_col": 2,
        "team_col": 3,
        "opponent_col": 4,
        "salary_col": 5,
        "start_row": 5,
        "header_row": 4,
        "headers": {
            6: "Rec Yds Prop",
            7: "Anytime TD Odds",
            8: "Receptions Prop",
            9: "Implied TD %",
            10: "Projected Score",
            11: "ESPN Projection",
            12: "Weighted Projection",
            13: "Value",
            14: "Projected Ownership",
        },
        "markets": {
            6: ("player_reception_yds", "line"),
            7: ("player_anytime_td", "yes_price"),
            8: ("player_receptions", "line"),
        },
        "odds_cols": {7},
        "implied_td_col": 9,
        "proj_score_col": 10,
        "espn_col": 11,
        "weighted_col": 12,
        "value_col": 13,
        "proj_ownership_col": 14,
    },
    "Defenses (D_ST)": {
        "name_col": 3,
        "team_col": 3,
        "opponent_col": 4,
        "salary_col": 5,
        "start_row": 5,
        "header_row": 4,
        "headers": {
            6: "Opponent Total",
            7: "Projected Score",
            8: "Value",
            9: "Projected Ownership",
        },
        "markets": {6: ("totals", "opponent_line")},
        "odds_cols": set(),
        "proj_score_col": 7,
        "value_col": 8,
        "proj_ownership_col": 9,
    },
}

TEAM_MAP = {
    "SFO": "SF",
    "TAM": "TB",
    "GNB": "GB",
    "NWE": "NE",
    "NOR": "NO",
    "LVR": "LV",
    "KAN": "KC",
    "WAS": "WAS",
    "WSH": "WAS",
    "LAR": "LA",
    "LAC": "LAC",
    "ARI": "ARI",
    "ARZ": "ARI",
    "BAL": "BAL",
    "BLT": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "HOU": "HOU",
    "IND": "IND",
    "JAX": "JAC",
    "JAC": "JAC",
    "MIA": "MIA",
    "MIN": "MIN",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "TEN": "TEN",
}

FULL_TEAM_NAME_MAP = {
    "ARIZONA CARDINALS": "ARI",
    "CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "RAVENS": "BAL",
    "BUFFALO BILLS": "BUF",
    "BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "BROWNS": "CLE",
    "DALLAS COWBOYS": "DAL",
    "COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "PACKERS": "GB",
    "HOUSTON TEXANS": "HOU",
    "TEXANS": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAC",
    "JAGUARS": "JAC",
    "KANSAS CITY CHIEFS": "KC",
    "CHIEFS": "KC",
    "LAS VEGAS RAIDERS": "LV",
    "RAIDERS": "LV",
    "LOS ANGELES CHARGERS": "LAC",
    "CHARGERS": "LAC",
    "LOS ANGELES RAMS": "LA",
    "RAMS": "LA",
    "MIAMI DOLPHINS": "MIA",
    "DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
    "PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO",
    "SAINTS": "NO",
    "NEW YORK GIANTS": "NYG",
    "GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI",
    "EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF",
    "49ERS": "SF",
    "SEATTLE SEAHAWKS": "SEA",
    "SEAHAWKS": "SEA",
    "TAMPA BAY BUCCANEERS": "TB",
    "BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN",
    "TITANS": "TEN",
    "WASHINGTON COMMANDERS": "WAS",
    "COMMANDERS": "WAS",
}


# ==========================================
# SYSTEM AUTOMATION HELPERS
# ==========================================
def open_file_in_default_app(file_path: Path) -> None:
  try:
    resolved_path = file_path.resolve()
    print(f"[*] Opening spreadsheet automatically: {resolved_path.name}")
    if os.name == "nt":
      os.startfile(resolved_path)
    elif sys.platform == "darwin":
      subprocess.run(["open", str(resolved_path)], check=True)
    else:
      subprocess.run(["xdg-open", str(resolved_path)], check=True)
  except Exception as e:
    print(f"[!] Could not automatically open workbook: {e}")


def launch_streamlit_app(app_script: str = "app.py") -> None:
  app_path = Path(app_script)
  if not app_path.exists():
    candidates = sorted(Path.cwd().glob("app*.py"))
    if candidates:
      app_path = candidates[0]
    else:
      print(f"[!] Streamlit app script '{app_script}' not found.")
      return

  print(f"[*] Launching Streamlit dashboard ({app_path.name})...")
  try:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path.resolve())]
    )
  except Exception as e:
    print(f"[!] Failed to launch Streamlit dashboard: {e}")


# ==========================================
# HELPER & FANDUEL SYNC FUNCTIONS
# ==========================================
def normalize_name(value: Any) -> str:
  if value is None:
    return ""
  text = unicodedata.normalize("NFKD", str(value))
  text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
  text = text.replace("’", "'").replace("`", "'").replace("&", "and")
  text = re.sub(r"[.'’]", "", text)
  text = re.sub(r"[^a-z0-9 ]+", " ", text)
  text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
  return re.sub(r"\s+", " ", text).strip()


def normalize_team(value: Any) -> str:
  if value is None:
    return ""
  val_str = str(value).strip().upper()
  if val_str in FULL_TEAM_NAME_MAP:
    return FULL_TEAM_NAME_MAP[val_str]
  raw = re.sub(r"[^A-Z0-9]", "", val_str)
  return TEAM_MAP.get(raw, raw)


def sync_fanduel_csv_to_workbook(wb: openpyxl.Workbook) -> None:
  csv_files = [
      f
      for f in os.listdir(SCRIPT_DIR)
      if f.endswith(".csv") and ("players-list" in f or "fanduel" in f.lower())
  ]
  if not csv_files:
    print(
        "[!] FanDuel players list CSV not found in workspace. Skipping"
        " auto-sync."
    )
    return

  csv_path = sorted(csv_files)[-1]
  print(f"[+] Synchronizing FanDuel roster from CSV: {csv_path}")
  df_fd = pd.read_csv(Path(SCRIPT_DIR) / csv_path)

  pos_map = {
      "QB": "Quarterbacks (QB)",
      "RB": "Running Backs (RB)",
      "WR": "Wide Receivers (WR)",
      "TE": "Tight Ends (TE)",
      "D": "Defenses (D_ST)",
  }

  for fd_pos, sheet_name in pos_map.items():
    if sheet_name not in wb.sheetnames:
      continue
    ws = wb[sheet_name]

    max_row = ws.max_row
    max_col = ws.max_column
    for r in range(5, max_row + 1):
      for c in range(1, max_col + 1):
        ws.cell(row=r, column=c, value=None)

    df_pos = df_fd[df_fd["Position"] == fd_pos].copy()
    if "FPPG" in df_pos.columns:
      df_pos = df_pos.sort_values(by="FPPG", ascending=False)
    elif "Salary" in df_pos.columns:
      df_pos = df_pos.sort_values(by="Salary", ascending=False)

    row_idx = 5
    for _, row in df_pos.iterrows():
      if fd_pos == "D":
        name = str(row.get("Last Name", "")).strip()
        if not name or name == "nan":
          name = str(row.get("Nickname", "")).strip()
      else:
        name = str(row.get("Nickname", "")).strip()
        if not name or name == "nan":
          first = str(row.get("First Name", "")).strip()
          last = str(row.get("Last Name", "")).strip()
          name = f"{first} {last}"

      team = str(row.get("Team", "")).strip().upper()
      raw_opp = str(row.get("Opponent", "")).strip().upper()
      opponent = raw_opp.replace("@", "").strip()

      try:
        salary = float(row.get("Salary", 5000))
      except (ValueError, TypeError):
        salary = 5000

      ws.cell(row=row_idx, column=1, value=row_idx - 4)
      ws.cell(row=row_idx, column=2, value=name)
      ws.cell(row=row_idx, column=3, value=team)
      ws.cell(row=row_idx, column=4, value=opponent)
      ws.cell(row=row_idx, column=5, value=salary)

      row_idx += 1

    if row_idx > 5:
      ws.auto_filter.ref = f"A4:O{row_idx - 1}"


def load_espn_projections_from_file(
    file_path: str | Path | None,
) -> dict[str, float]:
  projections = {}
  if not file_path:
    return projections

  p = Path(file_path)
  if not p.is_file():
    print(f"[!] Warning: ESPN projections file not found at '{file_path}'")
    return projections

  try:
    with open(p, mode="r", encoding="utf-8-sig") as f:
      reader = csv.reader(f)
      for row in reader:
        if not row or len(row) < 6:
          continue
        if "player" in row[0].lower() or "proj" in row[5].lower():
          continue
        raw_cell = row[0].strip()
        raw_name = raw_cell.split("\n")[0].strip()
        norm_name = normalize_name(raw_name)
        try:
          clean_val = row[5].replace('"', "").strip()
          proj_val = float(clean_val)
          if proj_val >= 1.0 and norm_name:
            projections[norm_name] = proj_val
        except (ValueError, IndexError):
          continue
    print(f"[+] Loaded {len(projections)} player projections from ESPN file.")
  except Exception as e:
    print(f"[!] Could not parse ESPN file '{file_path}': {e}")
  return projections


# ==========================================
# ODDS API FUNCTIONS
# ==========================================
def is_main_slate_game(commence_str: str) -> bool:
  if not commence_str:
    return False
  try:
    dt_utc = datetime.datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
    dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
    return dt_et.weekday() == 6 and (12 <= dt_et.hour <= 23)
  except Exception:
    return False


def fetch_vegas_odds_data(api_key: str) -> tuple[list[dict], list[tuple]]:
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
              for ev in cached_data["sunday_events"]
              if is_main_slate_game(ev.get("commence_time"))
          ]
          events_with_odds = [
              item
              for item in cached_data["events_with_odds"]
              if is_main_slate_game(item[0].get("commence_time"))
          ]
          if events_with_odds:
            return sunday_events, events_with_odds
      except Exception as e:
        print(
            f"[!] Failed to read cache file ({e}). Fetching fresh data from"
            " Odds API..."
        )

  print("[*] Querying Odds API using batched market requests...")
  session = requests.Session()
  session.headers.update({"User-Agent": "NFL-Lineup-Optimizer/1.0"})

  url = f"{API_BASE}/sports/{SPORT}/events"
  resp = session.get(
      url,
      params={"apiKey": api_key, "dateFormat": "iso"},
      timeout=REQUEST_TIMEOUT,
  )
  if resp.status_code != 200:
    raise RuntimeError(
        f"Odds API events request failed ({resp.status_code}): {resp.text}"
    )

  raw_events = resp.json()
  sunday_events = [
      ev for ev in raw_events if is_main_slate_game(ev.get("commence_time"))
  ]

  batch_markets_str = ",".join(PLAYER_MARKETS + TEAM_MARKETS)
  events_with_odds = []

  for ev in sunday_events:
    event_id = ev["id"]
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": batch_markets_str,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    mkt_url = f"{API_BASE}/sports/{SPORT}/events/{event_id}/odds"
    try:
      m_resp = session.get(mkt_url, params=params, timeout=REQUEST_TIMEOUT)
      if m_resp.status_code == 200:
        data = m_resp.json()
        events_with_odds.append((ev, data))
    except Exception as e:
      print(f"[!] Warning: Failed fetching odds for event {event_id}: {e}")

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
        "price": outcome.get("price"),
    })
  return results


def build_prop_index(events_with_odds):
  player_props, game_totals = defaultdict(list), {}
  for event, odds_data in events_with_odds:
    event_id = event["id"]
    for bookmaker in odds_data.get("bookmakers", []):
      for market in bookmaker.get("markets", []):
        market_key = market.get("key")
        if market_key in PLAYER_MARKETS:
          for rec in parse_market_for_players(event, bookmaker, market):
            player_props[
                (rec["description_norm"], market_key, rec["team"])
            ].append(rec)
            player_props[(rec["description_norm"], market_key, "")].append(rec)
        elif market_key == "totals":
          for out in market.get("outcomes", []):
            if out.get("point") is not None:
              game_totals[event_id] = float(out["point"])

  return player_props, game_totals


def candidate_records(player, market, player_props):
  exact = list(
      player_props.get((player["name_norm"], market, player["team"]), [])
  )
  if exact:
    return exact
  any_team = list(player_props.get((player["name_norm"], market, ""), []))
  if any_team:
    return any_team

  player_name_parts = player["name_norm"].split()
  player_first = player_name_parts[0] if player_name_parts else ""
  player_last = (
      player_name_parts[-1] if len(player_name_parts) > 1 else player_first
  )

  best_ratio, best_rows = 0.0, []
  for (name_norm, mkt, team), rows in player_props.items():
    if mkt != market:
      continue
    if team and player["team"] and team != player["team"]:
      continue

    prop_parts = name_norm.split()
    prop_first = prop_parts[0] if prop_parts else ""
    prop_last = prop_parts[-1] if len(prop_parts) > 1 else prop_first

    # Last name must match exactly
    if player_last != prop_last:
      continue

    # First names must match closely and avoid cross-matching different first names
    if player_first and prop_first:
      if player_first != prop_first and (
          player_first[0] != prop_first[0]
          or difflib.SequenceMatcher(None, player_first, prop_first).ratio()
          < 0.85
      ):
        continue

    ratio = difflib.SequenceMatcher(None, player["name_norm"], name_norm).ratio()
    if ratio >= FUZZY_MATCH_THRESHOLD and ratio > best_ratio:
      best_ratio, best_rows = ratio, rows
  return best_rows


def choose_main_line(rows, market=""):
  points = [
      r["point"]
      for r in rows
      if r.get("point") is not None and r.get("side") in {"over", "under"}
  ]
  return float(median(points)) if points else None


def choose_side_price(rows, side):
  if side == "over_price_1.5":
    # 1. STRICTLY look for Over at point 1.5
    exact_15 = [
        float(r["price"])
        for r in rows
        if r.get("side") in {"over", "yes"}
        and r.get("point") is not None
        and abs(r.get("point") - 1.5) < 0.01
    ]
    if exact_15:
      return max(exact_15)

    # 2. If the sportsbook only posts a 2.5 main line (or alternate), adjust/estimate the 1.5 alternate price
    # Over 1.5 is heavily juiced compared to Over 2.5; estimate based on available lines or default to -220
    prices_25 = [
        float(r["price"])
        for r in rows
        if r.get("side") in {"over", "yes"}
        and r.get("point") is not None
        and abs(r.get("point") - 2.5) < 0.01
    ]
    if prices_25:
      # If Over 2.5 is around -102, Over 1.5 is roughly -220 to -240
      return -220

    # 3. Fallback: Find any Over/Yes line closest to 1.5
    valid_rows = [
        r
        for r in rows
        if r.get("side") in {"over", "yes"} and r.get("point") is not None
    ]
    if valid_rows:
      closest = min(valid_rows, key=lambda x: abs(x["point"] - 1.5))
      return float(closest["price"])

    # 4. Final fallback estimate if no specific points are found
    return -220

  prices = [
      float(r["price"])
      for r in rows
      if r.get("side") == side
      or r.get("side") == "yes"
      or r.get("side") == "over"
  ]
  return max(prices) if prices else None


def choose_opponent_team(event_list, team):
  for ev in event_list:
    away, home = normalize_team(ev.get("away_team")), normalize_team(
        ev.get("home_team")
    )
    if team == away:
      return home
    if team == home:
      return away
  return None


def read_workbook_players(wb):
  player_rows, defense_rows = {}, {}
  for sheet_name, cfg in POSITION_CONFIG.items():
    if sheet_name not in wb.sheetnames:
      continue
    sheet = wb[sheet_name]
    max_row = sheet.max_row or 300
    records = []
    for row in range(cfg["start_row"], max_row + 1):
      name = sheet.cell(row=row, column=cfg["name_col"]).value
      team = sheet.cell(row=row, column=cfg["team_col"]).value
      if not name or str(name).strip() == "":
        continue
      rec = {
          "sheet": sheet_name,
          "row": row,
          "name": str(name).strip(),
          "name_norm": normalize_name(name),
          "team": normalize_team(team),
      }
      records.append(rec)
      if sheet_name == "Defenses (D_ST)":
        defense_rows[rec["team"]] = rec
    player_rows[sheet_name] = records
  return player_rows, defense_rows


def extract_player_pool(wb) -> dict[str, list[dict[str, Any]]]:
  player_pool = {"QB": [], "RB": [], "WR": [], "TE": [], "DEF": []}
  pos_key_map = {
      "Quarterbacks (QB)": "QB",
      "Running Backs (RB)": "RB",
      "Wide Receivers (WR)": "WR",
      "Tight Ends (TE)": "TE",
      "Defenses (D_ST)": "DEF",
  }
  for sheet_name, pos_code in pos_key_map.items():
    if sheet_name not in wb.sheetnames:
      continue
    sheet = wb[sheet_name]
    cfg = POSITION_CONFIG[sheet_name]
    max_row = sheet.max_row or 300

    for r in range(cfg["start_row"], max_row + 1):
      name = sheet.cell(row=r, column=cfg["name_col"]).value
      team = sheet.cell(row=r, column=cfg["team_col"]).value
      salary = sheet.cell(row=r, column=cfg["salary_col"]).value
      if not name or str(name).strip() == "" or salary is None:
        continue
      try:
        salary = float(salary)
      except (ValueError, TypeError):
        continue

      if salary <= 0:
        continue

      proj_col = cfg.get("weighted_col") or cfg.get("proj_score_col")
      cell_val = sheet.cell(row=r, column=proj_col).value if proj_col else None
      proj_val = 0.0
      if isinstance(cell_val, (int, float)):
        proj_val = float(cell_val)

      if proj_val > 0:
        player_pool[pos_code].append({
            "name": str(name).strip(),
            "team": str(team or "").strip(),
            "pos": pos_code,
            "salary": salary,
            "projection": round(proj_val, 2),
        })

  return player_pool


def calculate_and_apply_ownership(
    wb, player_pool, player_sim_counts=None, num_sims=1500
):
  if player_sim_counts is None:
    player_sim_counts = {}
  target_mass = {"QB": 1.00, "RB": 2.35, "WR": 3.55, "TE": 1.10, "DEF": 1.00}
  pos_floors = {"QB": 8.0, "RB": 4.0, "WR": 4.0, "TE": 3.0, "DEF": 2.0}
  ownership_map = {}
  pos_key_map = {
      "Quarterbacks (QB)": "QB",
      "Running Backs (RB)": "RB",
      "Wide Receivers (WR)": "WR",
      "Tight Ends (TE)": "TE",
      "Defenses (D_ST)": "DEF",
  }
  for sheet_name, pos_code in pos_key_map.items():
    players = player_pool.get(pos_code, [])
    if not players:
      continue
    target = target_mass.get(pos_code, 1.0)
    floor_val = pos_floors.get(pos_code, 3.0)
    attractions, sim_rates = [], []
    for p in players:
      proj = p["projection"]
      salary = p["salary"]
      val = proj / (salary / 1000.0) if salary > 0 else 0.0
      adj_proj = max(0.0, proj - floor_val)
      attractions.append((adj_proj**1.5) * (val**1.3))
      sim_rates.append(
          player_sim_counts.get(p["name"], 0) / float(num_sims)
          if num_sims > 0
          else 0.0
      )

    total_attraction = sum(attractions)
    for idx, p in enumerate(players):
      att_share = (
          (attractions[idx] / total_attraction * target)
          if total_attraction > 0
          else 0.0
      )
      blended_own = (0.60 * sim_rates[idx]) + (0.40 * att_share)
      ownership_map[(pos_code, normalize_name(p["name"]))] = min(
          0.42, max(0.0, blended_own)
      )

  right_align = Alignment(horizontal="right", vertical="center")
  for sheet_name, cfg in POSITION_CONFIG.items():
    if sheet_name not in wb.sheetnames:
      continue
    sheet = wb[sheet_name]
    own_col = cfg.get("proj_ownership_col")
    if not own_col:
      continue
    pos_code = pos_key_map.get(sheet_name)
    max_row = sheet.max_row or 300
    for r in range(cfg["start_row"], max_row + 1):
      name = sheet.cell(row=r, column=cfg["name_col"]).value
      if not name or str(name).strip() == "":
        continue
      norm = normalize_name(name)
      own_val = ownership_map.get((pos_code, norm), 0.08)
      cell = sheet.cell(row=r, column=own_col, value=round(own_val, 4))
      cell.number_format = "0.0%"
      cell.alignment = right_align


def update_workbook_projections(
    wb, player_rows, defense_rows, player_props, game_totals, espn_projs, events
):
  stats = {"updated": 0, "espn_projs_updated": 0}
  header_fill = PatternFill(
      start_color="1F4E78", end_color="1F4E78", fill_type="solid"
  )
  header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
  header_alignment = Alignment(
      horizontal="center", vertical="center", wrap_text=True
  )

  if "Defenses (D_ST)" in wb.sheetnames:
    ws_def = wb["Defenses (D_ST)"]
    if ws_def.cell(row=4, column=8).value == "ESPN Projection":
      ws_def.delete_cols(8, 2)

  qb_stats_by_team = {}

  for sheet_name, cfg in POSITION_CONFIG.items():
    if sheet_name not in wb.sheetnames:
      continue
    sheet = wb[sheet_name]
    h_row = cfg.get("header_row", 4)

    sheet.row_dimensions[h_row].height = 28.8

    for col_idx in range(1, sheet.max_column + 1):
      sheet.cell(row=2, column=col_idx).value = None
      sheet.cell(row=3, column=col_idx).value = None

    if "headers" in cfg:
      for col_idx, header_label in cfg["headers"].items():
        cell = sheet.cell(row=h_row, column=col_idx)
        cell.value = header_label
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = header_alignment

    max_row = sheet.max_row or 300
    for player in player_rows.get(sheet_name, []):
      row = player["row"]

      if sheet_name == "Defenses (D_ST)":
        continue

      for target_col, (market, mode) in cfg["markets"].items():
        rows = candidate_records(player, market, player_props)
        if (
            not rows
            and sheet_name == "Quarterbacks (QB)"
            and market == "player_rush_tds"
        ):
          rows = candidate_records(player, "player_anytime_td", player_props)
        val = (
            choose_main_line(rows, market)
            if mode == "line"
            else choose_side_price(rows, mode)
        )
        sheet.cell(row=row, column=target_col).value = val
        if val is not None:
          stats["updated"] += 1

      # Implied TD % calculations
      implied_td_val = 0.0
      implied_pass_td_prob = 0.0

      if sheet_name == "Quarterbacks (QB)":
        # Col 7 has Pass TD Odds (over price for 1.5)
        pass_td_odds = sheet.cell(row=row, column=7).value
        if pass_td_odds and isinstance(pass_td_odds, (int, float)):
          if pass_td_odds < 0:
            implied_pass_td_prob = -pass_td_odds / (-pass_td_odds + 100.0)
          else:
            implied_pass_td_prob = 100.0 / (pass_td_odds + 100.0)
          sheet.cell(row=row, column=11).value = round(
              implied_pass_td_prob, 4
          )

        # Col 9 has Rush TD Odds
        rush_td_odds = sheet.cell(row=row, column=9).value
        if rush_td_odds and isinstance(rush_td_odds, (int, float)):
          if rush_td_odds < 0:
            implied_td_val = -rush_td_odds / (-rush_td_odds + 100.0)
          else:
            implied_td_val = 100.0 / (rush_td_odds + 100.0)
          sheet.cell(row=row, column=12).value = round(implied_td_val, 4)
      else:
        td_col, odds_col = cfg.get("implied_td_col"), next(
            (c for c, (_, m) in cfg["markets"].items() if m == "yes_price"), None
        )
        if td_col and odds_col:
          odds_val = sheet.cell(row=row, column=odds_col).value
          if odds_val and isinstance(odds_val, (int, float)):
            if odds_val < 0:
              implied_td_val = -odds_val / (-odds_val + 100.0)
            else:
              implied_td_val = 100.0 / (odds_val + 100.0)
            sheet.cell(row=row, column=td_col).value = round(implied_td_val, 4)

      proj_score_val = 0.0
      if sheet_name == "Quarterbacks (QB)":
        pass_yds = float(sheet.cell(row=row, column=6).value or 0)
        rush_yds = float(sheet.cell(row=row, column=8).value or 0)
        interceptions = float(sheet.cell(row=row, column=10).value or 0)
        proj_score_val = (
            (pass_yds * 0.04)
            + (implied_pass_td_prob * 4.0)
            + (rush_yds * 0.1)
            + (implied_td_val * 6.0)
            - (interceptions * 2.0)
        )

        if player["team"]:
          qb_stats_by_team[player["team"]] = {
              "pass_yds": pass_yds,
              "interceptions": interceptions,
          }
      elif sheet_name == "Running Backs (RB)":
        rush_yds = float(sheet.cell(row=row, column=6).value or 0)
        receptions = float(sheet.cell(row=row, column=8).value or 0)
        rec_yds = float(sheet.cell(row=row, column=9).value or 0)
        proj_score_val = (
            (rush_yds * 0.1)
            + (rec_yds * 0.1)
            + (receptions * 0.5)
            + (implied_td_val * 6.0)
        )
      elif sheet_name in ("Wide Receivers (WR)", "Tight Ends (TE)"):
        rec_yds = float(sheet.cell(row=row, column=6).value or 0)
        receptions = float(sheet.cell(row=row, column=8).value or 0)
        proj_score_val = (
            (rec_yds * 0.1) + (receptions * 0.5) + (implied_td_val * 6.0)
        )

      proj_col = cfg.get("proj_score_col")
      if proj_col:
        sheet.cell(row=row, column=proj_col).value = round(proj_score_val, 2)

      espn_val = 0.0
      if "espn_col" in cfg:
        espn_col = cfg["espn_col"]
        if espn_projs:
          raw_espn_val = espn_projs.get(player["name_norm"])
          if raw_espn_val is not None and raw_espn_val >= 1.0:
            espn_val = raw_espn_val
          else:
            matches = difflib.get_close_matches(
                player["name_norm"],
                espn_projs.keys(),
                n=1,
                cutoff=ESPN_MATCH_THRESHOLD,
            )
            if matches and espn_projs[matches[0]] >= 1.0:
              espn_val = espn_projs[matches[0]]

        sheet.cell(row=row, column=espn_col).value = round(espn_val, 2)
        if espn_val >= 1.0:
          stats["espn_projs_updated"] += 1

      weighted_col = cfg.get("weighted_col")
      if weighted_col:
        if proj_score_val > 0 and espn_val > 0:
          weighted_val = (0.6 * proj_score_val) + (0.4 * espn_val)
        elif proj_score_val > 0:
          weighted_val = proj_score_val
        elif espn_val > 0:
          weighted_val = espn_val
        else:
          weighted_val = 0.0

        sheet.cell(row=row, column=weighted_col).value = round(weighted_val, 2)

      val_col = cfg.get("value_col")
      sal_val = sheet.cell(row=row, column=cfg["salary_col"]).value or 5000
      if val_col and sal_val and sal_val > 0:
        sheet.cell(row=row, column=val_col).value = round(
            weighted_val / (sal_val / 1000.0), 2
        )

  if "Defenses (D_ST)" in wb.sheetnames:
    sheet = wb["Defenses (D_ST)"]
    cfg = POSITION_CONFIG["Defenses (D_ST)"]
    for player in player_rows.get("Defenses (D_ST)", []):
      row = player["row"]
      opponent = choose_opponent_team(events, player["team"])
      if opponent:
        sheet.cell(row=row, column=cfg["opponent_col"]).value = opponent

      opp_tot = 22.0
      event_id = None
      for ev in events:
        away = normalize_team(ev.get("away_team"))
        home = normalize_team(ev.get("home_team"))
        if player["team"] in (away, home) and opponent in (away, home):
          event_id = ev["id"]
          game_tot = game_totals.get(event_id, 44.0)
          opp_tot = game_tot / 2.0
          break

      opp_tot_cell = sheet.cell(row=row, column=6, value=round(opp_tot, 2))
      opp_tot_cell.number_format = "0.00"

      implied_pts = float(opp_tot)

      if implied_pts <= 17.0:
        pa_score = 7.5 + (17.0 - implied_pts) * 0.35
      elif implied_pts <= 24.0:
        pa_score = 4.0 - (implied_pts - 17.0) * 0.35
      else:
        pa_score = max(-2.5, 1.5 - (implied_pts - 24.0) * 0.4)

      opp_qb_data = qb_stats_by_team.get(opponent, {})
      opp_pass_yds = opp_qb_data.get("pass_yds", 240.0)
      proj_sacks = max(2.0, (opp_pass_yds / 230.0) * 3.0)

      opp_qb_ints = opp_qb_data.get("interceptions", 0.7)
      proj_ints = max(0.8, opp_qb_ints * 2.2)
      proj_fumbles = 1.0

      proj_dst_tds = 1.5

      vegas_def_proj = max(
          4.0, pa_score + proj_sacks + proj_ints + proj_fumbles + proj_dst_tds
      )
      proj_cell = sheet.cell(
          row=row, column=7, value=round(vegas_def_proj, 2)
      )
      proj_cell.number_format = "0.00"

      sal = sheet.cell(row=row, column=cfg["salary_col"]).value or 5000
      val_cell = sheet.cell(
          row=row, column=8, value=round(vegas_def_proj / (sal / 1000.0), 2)
      )
      val_cell.number_format = "0.00"

  # ==========================================
  # GLOBAL UNIFORM FORMATTING PASS
  # ==========================================
  data_font = Font(name="Calibri", size=11, bold=False, color="000000")
  align_left = Alignment(horizontal="left", vertical="center")
  align_center = Alignment(horizontal="center", vertical="center")
  align_right = Alignment(horizontal="right", vertical="center")

  for sheet_name, cfg in POSITION_CONFIG.items():
    if sheet_name not in wb.sheetnames:
      continue
    sheet = wb[sheet_name]
    max_row = sheet.max_row or 300
    max_col = sheet.max_column or 17
    odds_cols = cfg.get("odds_cols", set())

    for r in range(cfg["start_row"], max_row + 1):
      name_val = sheet.cell(row=r, column=cfg["name_col"]).value
      if not name_val or str(name_val).strip() == "":
        continue

      for c in range(1, max_col + 1):
        cell = sheet.cell(row=r, column=c)
        cell.font = data_font

        if c in {cfg["name_col"], 2}:
          cell.alignment = align_left
        elif c in {cfg["team_col"], cfg["opponent_col"], 3, 4, 1}:
          cell.alignment = align_center
        elif c == cfg["salary_col"]:
          cell.alignment = align_right
          cell.number_format = "$#,##0"
        elif cfg.get("proj_ownership_col") and c == cfg["proj_ownership_col"]:
          cell.alignment = align_right
          cell.number_format = "0.0%"
        elif c in odds_cols:
          cell.alignment = align_right
          if isinstance(cell.value, (int, float)):
            cell.number_format = "0"
        else:
          cell.alignment = align_right
          if isinstance(cell.value, (int, float)):
            cell.number_format = "0.00"

  return stats


# ==========================================
# CLI / MAIN PIPELINE
# ==========================================
def find_default_file(extension: str) -> Path | None:
  cwd = Path.cwd()
  if extension == ".csv":
    espn_file = cwd / "espn_projections.csv"
    if espn_file.exists():
      return espn_file
    candidates = [f for f in cwd.glob("*.csv") if "espn" in f.name.lower()]
    return candidates[0] if candidates else None

  updated_file = cwd / "NFL Current Week Projections.xlsx"
  if updated_file.exists():
    return updated_file

  candidates = [f for f in cwd.glob("*.xlsx") if not f.name.startswith("~$")]
  return candidates[0] if candidates else None


def main():
  parser = argparse.ArgumentParser(
      description="NFL Vegas Odds & Projections Excel Updater (CLI Mode)"
  )
  parser.add_argument(
      "excel_file", nargs="?", help="Path to input Excel workbook (.xlsx)"
  )
  parser.add_argument(
      "espn_file", nargs="?", help="Path to ESPN projections CSV (Optional)"
  )
  parser.add_argument("--api-key", help="The Odds API Key")
  parser.add_argument("-o", "--output", help="Path to save output Excel file")

  args = parser.parse_args()

  api_key = args.api_key or os.getenv("ODDS_API_KEY") or API_KEY
  if not api_key:
    print("[!] ERROR: No Odds API Key provided.")
    sys.exit(1)

  excel_path = (
      Path(args.excel_file)
      if args.excel_file
      else find_default_file(".xlsx")
  )
  if not excel_path or not excel_path.is_file():
    print("[!] ERROR: No valid Excel input file found.")
    sys.exit(1)

  espn_path = (
      Path(args.espn_file) if args.espn_file else find_default_file(".csv")
  )
  output_path = (
      Path(args.output)
      if args.output
      else SCRIPT_DIR / "NFL Current Week Projections.xlsx"
  )

  print("=" * 60)
  print("🏈 NFL VEGAS ODDS & PROJECTION ENGINE (CLI)")
  print("=" * 60)
  print(f"[*] Base Workbook Target : {excel_path.name}")
  print(f"[*] ESPN Projections CSV : {espn_path.name if espn_path else 'None'}")
  print(f"[*] Final Output File    : {output_path.name}")
  print("-" * 60)

  print("[1/5] Loading workbook & syncing latest FanDuel roster data...")
  wb = openpyxl.load_workbook(excel_path)
  sync_fanduel_csv_to_workbook(wb)

  print("[2/5] Checking odds data cache / querying The Odds API...")
  try:
    events, events_with_odds = fetch_vegas_odds_data(api_key)
    print(
        f"[+] Successfully loaded odds for {len(events)} Sunday main slate"
        " games."
    )
  except Exception as e:
    print(f"[!] ERROR fetching odds: {e}")
    sys.exit(1)

  print("[3/5] Parsing player/team prop markets...")
  player_rows, defense_rows = read_workbook_players(wb)
  player_props, game_totals = build_prop_index(events_with_odds)

  espn_projs = load_espn_projections_from_file(espn_path) if espn_path else {}

  print("[4/5] Updating workbook cells & calculating player projections...")
  stats = update_workbook_projections(
      wb, player_rows, defense_rows, player_props, game_totals, espn_projs, events
  )

  player_pool = extract_player_pool(wb)
  calculate_and_apply_ownership(wb, player_pool, {}, num_sims=1500)

  print("[5/5] Saving updated workbook to disk...")
  try:
    wb.save(output_path)
  except PermissionError:
    fallback_path = output_path.parent / (
        output_path.stem + "_updated" + output_path.suffix
    )
    print(f"[!] WARNING: '{output_path.name}' is open in Excel and locked.")
    print(f"[*] Saving output to fallback file: '{fallback_path.name}'")
    wb.save(fallback_path)
    output_path = fallback_path

  print("-" * 60)
  print(
      "✅ SUCCESS! FanDuel roster, clean opponents, projections & ownership"
      " updated successfully."
  )
  print(f"   • Player Props Matched   : {stats['updated']}")
  print(f"   • ESPN Projs Blended     : {stats['espn_projs_updated']}")
  print(f"   • Output Saved To        : {output_path.resolve()}")
  print("=" * 60)

  open_file_in_default_app(output_path)
  launch_streamlit_app("app.py")


if __name__ == "__main__":
  main()