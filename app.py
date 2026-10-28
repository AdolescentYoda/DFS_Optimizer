import csv
import datetime
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, norm, pearsonr
import streamlit as st

try:
  from scipy.optimize import Bounds, LinearConstraint, milp

  HAS_MILP = True
except ImportError:
  HAS_MILP = False

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & PERMANENT LOGO HEADER
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Adolescent Yoda's Optimizer and Simulation Lab",
    page_icon="👽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants & Paths
APP_DIR = (
    Path(__file__).parent.resolve() if "__file__" in locals() else Path.cwd()
)
DEFAULT_VEGAS_FILE = APP_DIR / "NFL Current Week Projections.xlsx"
DEFAULT_FD_FILE = APP_DIR / "FanDuel.csv"
DEFAULT_ESPN_FILE = APP_DIR / "espn_projections.csv"
LOGO_FILE = APP_DIR / "AY logo.jpeg"
CACHE_FILE = APP_DIR / "odds_cache.json"
CACHE_EXPIRY_SECONDS = 24 * 60 * 60  # 24 hours

# -----------------------------------------------------------------------------
# HARDCODED API KEYS ROTATION POOL (Updated with fresh keys)
# -----------------------------------------------------------------------------
FALLBACK_API_KEYS = [
    "9d005ec0eb665c3ce52c0ba109d6030c",
    "9cb2283a9131c1cc6fe3185d25b7c68e",
    "820c20a0c8f1690b173ea97be76556b8",
    "9a7da4e97cd144123ed2e84ba3ac147e",
    "d1911c691f3d8cdb7ecef7767fc3bdf8",
    "66b08ba88184aab99e9bb0d1fad83bca",
    "49875f488a183578e045b693cef295ae",
]

# --- PERMANENT HEADER LAYOUT WITH LOGO ---
col_logo, col_title = st.columns([1, 15])

with col_logo:
  if LOGO_FILE.exists():
    st.image(str(LOGO_FILE), width=95)
  else:
    st.write("")

with col_title:
  st.markdown(
      """
        <h1 style='margin-bottom: 0px; padding-top: 2px; font-size: 2.1rem;'>Adolescent Yoda's Optimizer and Simulation Lab</h1>
        <p style='color: gray; margin-top: 0px;'>Phase 1 Ingestion Pipeline | Live Odds API Consensus | FanDuel Scoring Engine | Simulation-Driven Portfolio Optimizer</p>
    """,
      unsafe_allow_html=True,
  )

st.markdown("---")

# Canonical NFL Team Dict
NFL_TEAMS = {
    "ARI": ["arizona", "cardinals", "ari"],
    "ATL": ["atlanta", "falcons", "atl"],
    "BAL": ["baltimore", "ravens", "bal"],
    "BUF": ["buffalo", "bills", "buf"],
    "CAR": ["carolina", "panthers", "car"],
    "CHI": ["chicago", "bears", "chi"],
    "CIN": ["cincinnati", "bengals", "cin"],
    "CLE": ["cleveland", "browns", "cle"],
    "DAL": ["dallas", "cowboys", "dal"],
    "DEN": ["denver", "broncos", "den"],
    "DET": ["detroit", "lions", "det"],
    "GB": ["green bay", "packers", "gb", "gnb"],
    "HOU": ["houston", "texans", "hou"],
    "IND": ["indianapolis", "colts", "ind"],
    "JAX": ["jacksonville", "jaguars", "jax", "jac"],
    "KC": ["kansas city", "chiefs", "kc", "kan"],
    "LV": ["las vegas", "raiders", "lv", "lvr", "oakland"],
    "LAC": ["los angeles chargers", "chargers", "lac", "sd"],
    "LAR": ["los angeles rams", "rams", "lar", "stl"],
    "MIA": ["miami", "dolphins", "mia"],
    "MIN": ["minnesota", "vikings", "min"],
    "NE": ["new england", "patriots", "ne", "nwe"],
    "NO": ["new orleans", "saints", "no", "nor"],
    "NYG": ["new york giants", "ny giants", "giants", "nyg"],
    "NYJ": ["new york jets", "ny jets", "jets", "nyj"],
    "PHI": ["philadelphia", "eagles", "phi"],
    "PIT": ["pittsburgh", "steelers", "pit"],
    "SF": ["san francisco", "49ers", "niners", "sf", "sfo"],
    "SEA": ["seattle", "seahawks", "sea"],
    "TB": ["tampa bay", "buccaneers", "bucs", "tb", "tam"],
    "TEN": ["tennessee", "titans", "ten"],
    "WAS": [
        "washington",
        "commanders",
        "redskins",
        "football team",
        "was",
        "wsh",
    ],
}


# -----------------------------------------------------------------------------
# 2. CANONICAL SCORING ENGINE & ODDS CONVERSION
# -----------------------------------------------------------------------------
def american_to_implied_prob(price: float) -> float:
  """Converts American or Decimal odds to implied probability."""
  if price == 0 or pd.isna(price):
    return 0.0
  if price > 1.0 and price < 20.0:  # Decimal Odds
    return 1.0 / price
  if price < 0:
    return (-price) / (-price + 100.0)
  return 100.0 / (price + 100.0)


def decimal_to_american(price: float) -> str:
  """Converts decimal odds price to American odds format (+150, -130)."""
  if price <= 1.0 or pd.isna(price):
    return ""
  if price >= 2.0:
    return f"+{int(round((price - 1.0) * 100))}"
  else:
    return f"-{int(round(100.0 / (price - 1.0)))}"


def calculate_fd_expected_points(
    pass_yds=0.0,
    pass_tds=0.0,
    ints=0.0,
    rush_yds=0.0,
    rush_tds=0.0,
    receptions=0.0,
    rec_yds=0.0,
    rec_tds=0.0,
    fumbles_lost=0.0,
) -> float:
  """Canonical FanDuel NFL Fantasy Point Calculation."""
  fp = (
      (pass_yds * 0.04)
      + (pass_tds * 4.0)
      + (ints * -1.0)
      + (rush_yds * 0.10)
      + (rush_tds * 6.0)
      + (receptions * 0.50)
      + (rec_yds * 0.10)
      + (rec_tds * 6.0)
      + (fumbles_lost * -2.0)
  )
  return max(0.0, fp)


# -----------------------------------------------------------------------------
# 3. LIVE ODDS API (24H CLOCK CACHED & STICKY MERGE ROTATION)
# -----------------------------------------------------------------------------
def check_cache_status(filepath):
  """Checks if local odds cache exists and evaluates age against 24-hr clock."""
  if not os.path.exists(filepath):
    return "Missing", 0, False
  file_mtime = os.path.getmtime(filepath)
  file_age_seconds = time.time() - file_mtime
  file_age_hours = file_age_seconds / 3600
  is_fresh = file_age_seconds < CACHE_EXPIRY_SECONDS
  status = "Fresh (< 24h)" if is_fresh else "Expired (> 24h)"
  return status, file_age_hours, is_fresh


@st.cache_data(ttl=86400)
def load_odds_cache_data(filepath):
  """Loads cached odds file from disk to protect API credits."""
  if os.path.exists(filepath):
    try:
      with open(filepath, "r") as f:
        return json.load(f), "Local JSON Cache"
    except Exception:
      return {}, "Error Loading Cache"
  return {}, "Cache File Not Found"


def fetch_odds_api_data(user_api_key: str, force_refresh=False):
  import requests

  status_str, age_hours, is_fresh = check_cache_status(CACHE_FILE)

  # Use local cache if fresh and refresh is not forced
  if is_fresh and not force_refresh and os.path.exists(CACHE_FILE):
    data, label = load_odds_cache_data(CACHE_FILE)
    if data:
      return data

  # Load previous cache into memory for sticky fallback merging
  existing_cache, _ = load_odds_cache_data(CACHE_FILE)

  # Build list of keys to try: user-provided key first, then hardcoded pool
  keys_to_try = []
  if user_api_key and user_api_key.strip():
    keys_to_try.append(user_api_key.strip())

  for k in FALLBACK_API_KEYS:
    if k and k.strip() and k.strip() not in keys_to_try:
      keys_to_try.append(k.strip())

  if not keys_to_try:
    st.sidebar.warning("⚠️ No API keys configured. Loading local cache.")
    return existing_cache

  base_url = (
      "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
  )
  events = []
  active_key = None

  # Rotate through keys until one successfully connects
  for idx, current_key in enumerate(keys_to_try):
    try:
      events_resp = requests.get(
          base_url, params={"apiKey": current_key}, timeout=10
      )

      if events_resp.status_code in [401, 403]:
        st.sidebar.warning(
            f"⚠️ API Key #{idx+1} invalid or unauthorized. Rotating..."
        )
        continue
      elif events_resp.status_code == 429:
        st.sidebar.warning(
            f"⚠️ API Key #{idx+1} credit limit reached (429). Rotating..."
        )
        continue

      events_resp.raise_for_status()
      events = events_resp.json()
      active_key = current_key
      st.sidebar.success(f"⚡ Connected using API Key #{idx+1}")
      break
    except Exception:
      continue

  if not active_key or not events:
    st.sidebar.error(
        "❌ All API keys failed or exhausted quotas. Falling back to local"
        " cache."
    )
    return existing_cache

  try:
    raw_props = {}
    target_markets = (
        "player_pass_yds,"
        "player_pass_tds,"
        "player_pass_interceptions,"
        "player_rush_yds,"
        "player_reception_yds,"
        "player_receptions,"
        "player_anytime_td"
    )

    for event in events:
      event_id = event["id"]
      prop_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
      p_params = {
          "apiKey": active_key,
          "regions": "us",
          "markets": target_markets,
      }
      p_resp = requests.get(prop_url, params=p_params, timeout=10)
      if p_resp.status_code == 200:
        p_data = p_resp.json()
        for bookmaker in p_data.get("bookmakers", []):
          for market in bookmaker.get("markets", []):
            m_key = market["key"]
            for outcome in market.get("outcomes", []):
              p_name = outcome.get("description", outcome.get("name", ""))
              if not p_name:
                continue
              if p_name not in raw_props:
                raw_props[p_name] = {}
              if m_key not in raw_props[p_name]:
                raw_props[p_name][m_key] = []

              if m_key == "player_anytime_td":
                if outcome.get("name", "").lower() == "yes":
                  price = float(outcome.get("price", 0.0))
                  prob = american_to_implied_prob(price)
                  if prob > 0:
                    raw_props[p_name][m_key].append((price, prob))
              else:
                point = float(outcome.get("point", 0.0))
                if point > 0:
                  raw_props[p_name][m_key].append(point)

    consensus_props = {}
    for p_name, markets in raw_props.items():
      p_pass_yds = (
          np.mean(markets["player_pass_yds"])
          if "player_pass_yds" in markets
          else 0.0
      )
      p_pass_tds = (
          np.mean(markets["player_pass_tds"])
          if "player_pass_tds" in markets
          else 0.0
      )
      p_ints = (
          np.mean(markets["player_pass_interceptions"])
          if "player_pass_interceptions" in markets
          else 0.0
      )
      p_rush_yds = (
          np.mean(markets["player_rush_yds"])
          if "player_rush_yds" in markets
          else 0.0
      )
      p_rec_yds = (
          np.mean(markets["player_reception_yds"])
          if "player_reception_yds" in markets
          else 0.0
      )
      p_rec = (
          np.mean(markets["player_receptions"])
          if "player_receptions" in markets
          else 0.0
      )

      td_tuples = markets.get("player_anytime_td", [])
      if td_tuples:
        avg_price = np.mean([t[0] for t in td_tuples])
        td_prob = np.mean([t[1] for t in td_tuples])
      else:
        avg_price = 0.0
        td_prob = 0.0

      calc_fp = calculate_fd_expected_points(
          pass_yds=p_pass_yds,
          pass_tds=p_pass_tds,
          ints=p_ints,
          rush_yds=p_rush_yds,
          rush_tds=td_prob if p_pass_yds == 0 else 0.0,
          receptions=p_rec,
          rec_yds=p_rec_yds,
          rec_tds=td_prob if p_pass_yds > 0 else 0.0,
      )
      calc_fp = min(34.0, calc_fp)
      num_books = max(
          [len(v) for v in markets.values()] if markets else [0]
      )

      consensus_props[p_name] = {
          "expected_fp": calc_fp,
          "consensus_books": num_books,
          "pass_yds": round(p_pass_yds, 1),
          "pass_tds": round(p_pass_tds, 1),
          "ints": round(p_ints, 1),
          "rush_yds": round(p_rush_yds, 1),
          "rec_yds": round(p_rec_yds, 1),
          "receptions": round(p_rec, 1),
          "anytime_td_odds": decimal_to_american(avg_price)
          if avg_price > 1.0
          else "",
          "implied_td_pct": round(td_prob, 2),
      }

    # --- STICKY MERGE WITH PREVIOUS CACHE ---
    # If a player was in the previous cache with valid props but is missing/zeroed in the new pull,
    # carry over their previous props so they don't unexpectedly drop out.
    for prev_name, prev_data in existing_cache.items():
      if prev_name not in consensus_props:
        consensus_props[prev_name] = prev_data
      else:
        # If new pull has 0 props across key categories but old cache had valid numbers, merge selectively
        new_item = consensus_props[prev_name]
        if (
            new_item["pass_yds"] == 0.0
            and new_item["rush_yds"] == 0.0
            and new_item["rec_yds"] == 0.0
            and not new_item["anytime_td_odds"]
        ):
          if (
              prev_data["pass_yds"] > 0
              or prev_data["rush_yds"] > 0
              or prev_data["rec_yds"] > 0
              or prev_data["anytime_td_odds"]
          ):
            consensus_props[prev_name] = prev_data

    with open(CACHE_FILE, "w") as f:
      json.dump(consensus_props, f)

    return consensus_props
  except Exception as e:
    st.sidebar.error(f"API Fetch Error: {e}")
    return existing_cache


@st.cache_data(ttl=3600)
def fetch_nflverse_weekly_stats(season_year: int):
  url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/stats_player_week_{season_year}.csv"
  try:
    df = pd.read_csv(url)
    return df
  except Exception as e:
    st.error(
        f"Could not automatically fetch nflverse stats for {season_year}: {e}"
    )
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# 4. HELPER UTILITIES & DATA PARSERS
# -----------------------------------------------------------------------------
def excel_col_index(col_str: str) -> int:
  col_str = col_str.upper().strip()
  exp, idx = 0, 0
  for char in reversed(col_str):
    idx += (ord(char) - ord("A") + 1) * (26**exp)
    exp += 1
  return idx - 1


def parse_numeric_series(series, default=0.0):
  def clean_val(v):
    if pd.isna(v):
      return default
    s = str(v).strip()
    if s.startswith("="):
      nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
      return float(nums[-1]) if nums else default
    s_clean = s.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
      return float(s_clean)
    except ValueError:
      return default

  return series.apply(clean_val)


def standardize_position(pos):
  if pd.isna(pos) or pos is None:
    return "FLEX"
  p = str(pos).upper().strip()
  if (
      p in ["D", "DEF", "DST", "D/ST", "DEFENSE", "TEAM DEFENSE"]
      or "DEF" in p
      or "DST" in p
  ):
    return "DEF"
  if "QB" in p:
    return "QB"
  if "RB" in p:
    return "RB"
  if "WR" in p:
    return "WR"
  if "TE" in p:
    return "TE"
  return "FLEX"


def normalize_player_key(value):
  if value is None or pd.isna(value):
    return ""
  s = str(value).strip().split("\n")[0].casefold()
  s = s.replace("’", "'").replace("`", "'").replace(".", "")
  s = re.sub(r"[^a-z0-9 ]+", " ", s)
  s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
  s = re.sub(r"\s+", " ", s).strip()
  aliases = {
      "gabriel davis": "gabe davis",
      "mitchell trubisky": "mitch trubisky",
      "gardner minshew ii": "gardner minshew",
      "jeffery wilson": "jeff wilson",
      "christopher olave": "chris olave",
  }
  return aliases.get(s, s)


def resolve_defense_code(text):
  if not text:
    return ""
  s = str(text).lower()
  s_clean = re.sub(r"\b(d/st|dst|def|defense|team defense)\b", " ", s)
  s_clean = re.sub(r"[^a-z0-9 ]+", " ", s_clean)
  words = s_clean.split()
  for code, keywords in NFL_TEAMS.items():
    for kw in keywords:
      if kw in s_clean or any(w == kw for w in words):
        return code
  return ""


def get_canonical_key(name, team="", position=""):
  pos_std = standardize_position(position)
  raw_name = str(name) if name is not None else ""
  is_def = (pos_std == "DEF") or any(
      k in raw_name.upper() for k in ["D/ST", "DST", "DEFENSE"]
  )
  if is_def:
    code = resolve_defense_code(raw_name) or resolve_defense_code(team)
    if code:
      return f"DEF_{code}"
    norm_name = re.sub(
        r"\b(d/st|dst|def|defense)\b", "", normalize_player_key(raw_name)
    ).strip()
    return f"DEF_{norm_name}"
  return normalize_player_key(raw_name)


# -----------------------------------------------------------------------------
# 5. OWNERSHIP MODELING
# -----------------------------------------------------------------------------
def calculate_estimated_ownership(df: pd.DataFrame) -> pd.DataFrame:
  if df.empty:
    return df
  df = df.copy()
  proj_col = "blended_proj" if "blended_proj" in df.columns else "projection"
  if (
      proj_col not in df.columns
      or "salary" not in df.columns
      or "position" not in df.columns
  ):
    if "proj_ownership" not in df.columns:
      df["proj_ownership"] = 8.0
    return df

  POS_TARGETS = {
      "QB": 100.0,
      "RB": 230.0,
      "WR": 340.0,
      "TE": 110.0,
      "DEF": 100.0,
      "FLEX": 100.0,
  }
  POS_FLOORS = {
      "QB": 9.0,
      "RB": 4.0,
      "WR": 4.0,
      "TE": 2.5,
      "DEF": 2.0,
      "FLEX": 3.0,
  }
  POS_WEIGHTS = {
      "QB": (2.2, 1.8),
      "RB": (2.1, 2.0),
      "WR": (2.0, 2.1),
      "TE": (1.8, 1.6),
      "DEF": (1.5, 2.0),
      "FLEX": (2.0, 2.0),
  }

  salaries = pd.to_numeric(df["salary"], errors="coerce").fillna(5000)
  projections = pd.to_numeric(df[proj_col], errors="coerce").fillna(0.0)
  pos_clean_series = df["position"].apply(standardize_position)

  df["_salary_num"] = salaries
  df["_proj_num"] = projections
  df["_pos_clean"] = pos_clean_series

  calculated_groups = []
  for pos_code, pos_df in df.groupby("_pos_clean"):
    pos_df = pos_df.copy()
    target_mass = POS_TARGETS.get(pos_code, 100.0)
    floor_val = POS_FLOORS.get(pos_code, 3.0)
    alpha, beta = POS_WEIGHTS.get(pos_code, (2.0, 2.0))

    val_per_1k = np.where(
        pos_df["_salary_num"] > 0,
        pos_df["_proj_num"] / (pos_df["_salary_num"] / 1000.0),
        0.0,
    )
    adj_proj = (pos_df["_proj_num"] - floor_val).clip(lower=0.01)
    attraction = (adj_proj**alpha) * (val_per_1k**beta)
    total_attraction = attraction.sum()

    if total_attraction > 0:
      raw_own = (attraction / total_attraction) * target_mass
    else:
      raw_own = pd.Series(
          target_mass / max(1, len(pos_df)), index=pos_df.index
      )

    capped_own = np.clip(raw_own, 0.5, 42.0)
    current_sum = capped_own.sum()
    final_own = (
        np.clip(capped_own * (target_mass / current_sum), 0.5, 45.0)
        if current_sum > 0
        else capped_own
    )
    pos_df["proj_ownership"] = np.round(final_own, 1)
    calculated_groups.append(pos_df)

  if calculated_groups:
    result_df = pd.concat(calculated_groups).sort_index()
    return result_df.drop(
        columns=["_salary_num", "_proj_num", "_pos_clean"], errors="ignore"
    )
  return df


# -----------------------------------------------------------------------------
# 6. WORKBOOK & DATA READERS
# -----------------------------------------------------------------------------
PROJECTED_SCORE_COLUMNS = {
    "Quarterbacks (QB)": ("QB", "L"),
    "Running Backs (RB)": ("RB", "K"),
    "Wide Receivers (WR)": ("WR", "J"),
    "Tight Ends (TE)": ("TE", "J"),
    "Defenses (D_ST)": ("DEF", "G"),
    "Defense (D/ST)": ("DEF", "G"),
    "Defense (DEF)": ("DEF", "G"),
    "Defense": ("DEF", "G"),
    "Defenses": ("DEF", "G"),
    "DST": ("DEF", "G"),
}


def load_fd_csv(fd_file=None):
  fd_source = None
  fd_display_name = None

  if fd_file is not None:
    fd_source = BytesIO(fd_file.getvalue())
    fd_display_name = fd_file.name
  elif DEFAULT_FD_FILE.exists():
    fd_source = str(DEFAULT_FD_FILE)
    fd_display_name = DEFAULT_FD_FILE.name
  else:
    candidates = [
        APP_DIR / f
        for f in os.listdir(APP_DIR)
        if f.endswith(".csv")
        and ("players-list" in f.lower() or "fanduel" in f.lower())
    ]
    if candidates:
      candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
      fd_source = candidates[0]
      fd_display_name = candidates[0].name

  if fd_source is None:
    return pd.DataFrame(), None

  try:
    df_fd = pd.read_csv(fd_source)
    id_col = (
        "Id" if "Id" in df_fd.columns else ("ID" if "ID" in df_fd.columns else None)
    )
    name_col = (
        "Nickname"
        if "Nickname" in df_fd.columns
        else ("First Name" if "First Name" in df_fd.columns else None)
    )
    pos_col = "Position" if "Position" in df_fd.columns else None
    team_col = "Team" if "Team" in df_fd.columns else None
    sal_col = "Salary" if "Salary" in df_fd.columns else None
    game_col = "Game" if "Game" in df_fd.columns else None
    opp_col = "Opponent" if "Opponent" in df_fd.columns else None

    if not name_col or not sal_col:
      return pd.DataFrame(), None

    records = []
    for _, row in df_fd.iterrows():
      r_name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
      if not r_name:
        continue
      r_pos = standardize_position(
          str(row[pos_col]) if pos_col and pd.notna(row[pos_col]) else ""
      )
      r_team = (
          str(row[team_col]).strip()
          if team_col and pd.notna(row[team_col])
          else ""
      )
      try:
        r_sal = int(float(str(row[sal_col]).replace("$", "").replace(",", "")))
      except (ValueError, TypeError):
        r_sal = 0

      c_key = get_canonical_key(r_name, r_team, r_pos)
      records.append({
          "player_id": (
              str(row[id_col]).strip()
              if id_col and pd.notna(row[id_col])
              else ""
          ),
          "name": r_name,
          "position": r_pos,
          "team": r_team,
          "salary": r_sal,
          "game": (
              str(row[game_col]).strip()
              if game_col and pd.notna(row[game_col])
              else ""
          ),
          "opponent": (
              str(row[opp_col]).strip()
              if opp_col and pd.notna(row[opp_col])
              else ""
          ),
          "_canon_key": c_key,
      })

    res_df = pd.DataFrame(records)
    return res_df.drop_duplicates(
        subset=["_canon_key"], keep="first"
    ).reset_index(drop=True), fd_display_name
  except Exception as e:
    st.sidebar.error(f"Error reading FanDuel CSV: {e}")
    return pd.DataFrame(), None


def read_vegas_workbook(source):
  try:
    xls = pd.ExcelFile(source)
  except Exception as e:
    st.sidebar.error(f"Could not open Vegas workbook: {e}")
    return pd.DataFrame()

  records = []
  target_sheets = {}
  for sheet_name, (pos_code, col_let) in PROJECTED_SCORE_COLUMNS.items():
    if sheet_name in xls.sheet_names:
      target_sheets[sheet_name] = (pos_code, col_let)

  for sheet_name, (pos_code, score_column) in target_sheets.items():
    try:
      raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    except Exception:
      continue
    if raw.empty:
      continue

    header_row = None
    for i in range(min(20, len(raw))):
      values = [str(v).strip().casefold() for v in raw.iloc[i].tolist()]
      if any(
          h in values
          for h in ["player name", "team", "defense", "dst", "team defense"]
      ):
        header_row = i
        break
    if header_row is None:
      continue

    name_col_idx, team_col_idx = 1, 2
    score_col_idx = excel_col_index(score_column)
    header_vals = [
        str(v).strip().casefold() for v in raw.iloc[header_row].tolist()
    ]
    for idx, h_val in enumerate(header_vals):
      if any(k in h_val for k in ["player", "defense", "team defense"]):
        name_col_idx = idx
      elif "team" in h_val and "defense" not in h_val:
        team_col_idx = idx

    espn_col_idx = None
    for c_i, c_val in enumerate(header_vals):
      if "espn" in c_val:
        espn_col_idx = c_i

    for raw_row_idx in range(header_row + 1, len(raw)):
      row = raw.iloc[raw_row_idx]
      player = row.iloc[name_col_idx] if name_col_idx < len(row) else None
      if player is None or pd.isna(player) or str(player).strip() == "":
        continue

      team = row.iloc[team_col_idx] if team_col_idx < len(row) else ""
      raw_projection = (
          row.iloc[score_col_idx] if score_col_idx < len(row) else 0.0
      )
      projection = parse_numeric_series(
          pd.Series([raw_projection]), default=0.0
      ).iloc[0]

      espn_projection = 0.0
      if espn_col_idx is not None and espn_col_idx < len(row):
        espn_projection = parse_numeric_series(
            pd.Series([row.iloc[espn_col_idx]]), default=0.0
        ).iloc[0]

      records.append({
          "name": str(player).strip(),
          "team": "" if pd.isna(team) else str(team).strip(),
          "position": pos_code,
          "vegas_proj": float(projection),
          "espn_proj": float(espn_projection),
          "vegas_source_sheet": sheet_name,
      })

  if not records:
    return pd.DataFrame()
  df = pd.DataFrame(records)
  df["vegas_proj"] = pd.to_numeric(
      df["vegas_proj"], errors="coerce"
  ).fillna(0.0)
  df["espn_proj"] = pd.to_numeric(df["espn_proj"], errors="coerce").fillna(0.0)
  df["position"] = df["position"].apply(standardize_position)
  df["_canon_key"] = df.apply(
      lambda r: get_canonical_key(r["name"], r["team"], r["position"]), axis=1
  )
  return df


def extract_espn_projections_map(
    espn_file=None, default_espn_path="espn_projections.csv"
):
  proj_map = {}
  sources_to_try = []
  if espn_file is not None:
    sources_to_try.append(("uploaded", espn_file))
  for path in [
      APP_DIR / default_espn_path,
      Path(default_espn_path),
      APP_DIR / "espn_projections.csv",
  ]:
    if path.exists():
      sources_to_try.append(("local", path))

  for src_type, src in sources_to_try:
    try:
      if src_type == "uploaded":
        src.seek(0)
        content = src.getvalue().decode("utf-8-sig", errors="ignore")
        lines = content.splitlines()
      else:
        with open(src, "r", encoding="utf-8-sig", errors="ignore") as f:
          lines = f.readlines()

      reader = csv.reader(lines)
      for row in reader:
        if not row or len(row) <= 5:
          continue
        raw_cell = str(row[0]).strip()
        if not raw_cell or any(
            h in raw_cell.lower() for h in ["player", "athlete"]
        ):
          continue
        parts = [p.strip() for p in raw_cell.split("\n") if p.strip()]
        raw_name = parts[0] if parts else raw_cell
        team = parts[1] if len(parts) > 1 else ""
        pos = (
            "DEF"
            if any(
                k in raw_cell.upper() for k in ["D/ST", "DST", "DEFENSE"]
            )
            else "FLEX"
        )

        k = get_canonical_key(raw_name, team, pos)
        val = parse_numeric_series(pd.Series([row[5]]), default=0.0).iloc[0]
        if k and val > 0:
          proj_map[k] = float(val)

      if proj_map:
        break
    except Exception:
      pass
  return proj_map


def load_combined_data(
    fd_file,
    espn_file,
    vegas_file,
    api_key="",
    force_api_refresh=False,
    default_espn_path="espn_projections.csv",
):
  fd_df, fd_name = load_fd_csv(fd_file)

  vegas_source = None
  if vegas_file is not None:
    vegas_source = BytesIO(vegas_file.getvalue())
  elif DEFAULT_VEGAS_FILE.exists():
    vegas_source = str(DEFAULT_VEGAS_FILE)

  vegas_df = (
      read_vegas_workbook(vegas_source)
      if vegas_source is not None
      else pd.DataFrame()
  )
  espn_map = extract_espn_projections_map(espn_file, default_espn_path)
  api_props_map = fetch_odds_api_data(
      user_api_key=api_key, force_refresh=force_api_refresh
  )

  if not fd_df.empty:
    st.sidebar.success(f"FanDuel Roster Loaded: {fd_name}")
    df = fd_df.copy()

    vegas_proj_map = {}
    vegas_espn_map = {}
    if not vegas_df.empty:
      for _, row in vegas_df.iterrows():
        k = row["_canon_key"]
        vegas_proj_map[k] = row.get("vegas_proj", 0.0)
        vegas_espn_map[k] = row.get("espn_proj", 0.0)

    df["vegas_proj"] = df["_canon_key"].map(vegas_proj_map).fillna(0.0)
    df["espn_proj"] = df["_canon_key"].map(espn_map).fillna(0.0)
    vegas_espn_series = df["_canon_key"].map(vegas_espn_map).fillna(0.0)
    df["espn_proj"] = np.where(
        df["espn_proj"] > 0, df["espn_proj"], vegas_espn_series
    )

    # Initialize Source Lineage Tracking and Raw Prop Columns safely
    df["proj_source_log"] = "Excel (Proj) + ESPN (Status)"
    df["Pass Yds Prop"] = 0.0
    df["Pass TD Prop"] = 0.0
    df["INT Prop"] = 0.0
    df["Rush Yds Prop"] = 0.0
    df["Rec Yds Prop"] = 0.0
    df["Receptions Prop"] = 0.0
    df["Anytime TD Odds"] = None
    df["Implied TD %"] = 0.0
    df["Anytime TD Odds"] = df["Anytime TD Odds"].astype("object")

    if api_props_map:
      api_matched_count = 0
      for idx, row in df.iterrows():
        p_name = row["name"]
        p_key = normalize_player_key(p_name)
        for api_p_name, api_data in api_props_map.items():
          if normalize_player_key(api_p_name) == p_key:
            calc_fp = api_data["expected_fp"]
            if calc_fp > 0:
              df.at[idx, "vegas_proj"] = calc_fp
              df.at[
                  idx, "proj_source_log"
              ] = f"Excel + ESPN + API ({api_data['consensus_books']} Books)"
              df.at[idx, "Pass Yds Prop"] = api_data.get("pass_yds", 0.0)
              df.at[idx, "Pass TD Prop"] = api_data.get("pass_tds", 0.0)
              df.at[idx, "INT Prop"] = api_data.get("ints", 0.0)
              df.at[idx, "Rush Yds Prop"] = api_data.get("rush_yds", 0.0)
              df.at[idx, "Rec Yds Prop"] = api_data.get("rec_yds", 0.0)
              df.at[idx, "Receptions Prop"] = api_data.get("receptions", 0.0)
              df.at[idx, "Anytime TD Odds"] = api_data.get("anytime_td_odds", "")
              df.at[idx, "Implied TD %"] = api_data.get("implied_td_pct", 0.0)
              api_matched_count += 1
            break
      if api_matched_count > 0:
        st.sidebar.success(
            f"⚡ Ingested Consensus Props for {api_matched_count} Players!"
        )

    def_mask = df["position"].apply(standardize_position) == "DEF"
    if def_mask.sum() > 0:
      df.loc[def_mask, "vegas_proj"] = np.where(
          df.loc[def_mask, "vegas_proj"] > 0,
          df.loc[def_mask, "vegas_proj"],
          7.2,
      )
      df.loc[def_mask, "proj_source_log"] = "Calibrated D/ST Model"

    return df
  return pd.DataFrame()


# -----------------------------------------------------------------------------
# 7. SIDEBAR CONTROLS & WEATHER/INACTIVES CONFIG
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Data Sources")
uploaded_fd = st.sidebar.file_uploader(
    "Upload FanDuel Roster (CSV)", type=["csv"]
)
uploaded_espn = st.sidebar.file_uploader(
    "Upload ESPN Projections (CSV)", type=["csv"]
)
uploaded_vegas = st.sidebar.file_uploader(
    "Upload Vegas Workbook (.xlsx)", type=["xlsx"]
)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Live API & Cache Settings")
cache_status_str, cache_age_hours, cache_is_fresh = check_cache_status(CACHE_FILE)
st.sidebar.info(
    f"**Odds Cache Status:** {cache_status_str}\n\n**File Age:**"
    f" {cache_age_hours:.1f} hours"
)

try:
  api_key_default = st.secrets.get("ODDS_API_KEY", "")
except Exception:
  api_key_default = ""

odds_api_key = st.sidebar.text_input(
    "Override Odds API Key (Optional)", value=api_key_default, type="password"
)

force_api_refresh = st.sidebar.button(
    "🔄 Force API Refresh (Burns Credits)",
    help="Bypasses the 24-hour clock and hits The Odds API live.",
)

st.sidebar.markdown("---")
st.sidebar.header("⚙ Blending Weights")
w_vegas = st.sidebar.slider("Consensus Props / Vegas Weight", 0.0, 1.0, 0.60, 0.05)
w_espn = st.sidebar.slider("ESPN Weight", 0.0, 1.0, 0.40, 0.05)

total_w = w_vegas + w_espn
if total_w > 0:
  w_vegas /= total_w
  w_espn /= total_w
else:
  w_vegas, w_espn = 0.5, 0.5

st.sidebar.markdown("---")
st.sidebar.header("🎲 Simulation Volatility")
global_risk_mult = st.sidebar.slider(
    "Volatility Multiplier", 0.5, 2.0, 1.2, 0.1
)

st.sidebar.markdown("---")
st.sidebar.header("🌦️ Stadium Weather Engine")
enable_weather_impact = st.sidebar.checkbox(
    "Enable Per-Game Weather Adjustments", value=True
)

OUTDOOR_STADIUMS = {
    "BUF",
    "CHI",
    "CLE",
    "DEN",
    "GB",
    "KC",
    "NE",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "TEN",
    "WAS",
    "BAL",
    "CIN",
    "TB",
    "MIA",
    "CAR",
    "JAX",
    "SF",
    "LAR",
    "LAC",
    "ARI",
    "LV",
}
STADIUM_COORDS = {
    "ARI": (33.5276, -112.2626),
    "ATL": (33.7554, -84.4010),
    "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5160),
    "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456),
    "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839),
    "LAC": (33.9535, -118.3390),
    "LAR": (33.9535, -118.3390),
    "LV": (36.0909, -115.1833),
    "MIA": (25.9580, -80.2389),
    "MIN": (44.9739, -93.2581),
    "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812),
    "NYG": (40.8135, -74.0744),
    "NYJ": (40.8135, -74.0744),
    "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SF": (37.4032, -121.9698),
    "SEA": (47.5952, -122.3316),
    "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),
    "WAS": (38.9076, -76.8645),
}


@st.cache_data(ttl=3600)
def fetch_stadium_forecast(lat, lon, target_hour_index=13):
  import requests

  try:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "wind_speed_10m",
            "precipitation_probability",
        ],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "forecast_days": 3,
    }
    response = requests.get(url, params=params, timeout=5)
    if response.status_code == 200:
      hourly = response.json().get("hourly", {})
      temps = hourly.get("temperature_2m", [])
      winds = hourly.get("wind_speed_10m", [])
      precips = hourly.get("precipitation_probability", [])
      idx = min(target_hour_index, len(temps) - 1) if temps else 0
      return {
          "temp": round(temps[idx], 1) if temps else 68.0,
          "wind": round(winds[idx], 1) if winds else 8.0,
          "precip": int(precips[idx]) if precips else 0,
      }
  except Exception:
    pass
  return {"temp": 68.0, "wind": 8.0, "precip": 0}


def get_slate_outdoor_games(df):
  if df.empty or "game" not in df.columns:
    return []
  outdoor_games = set()
  for _, row in df.iterrows():
    team = row.get("team", "")
    game = row.get("game", "")
    if team in OUTDOOR_STADIUMS and game:
      outdoor_games.add(game)
  return sorted(list(outdoor_games))


def apply_per_game_weather_adjustments(df, game_weather_settings):
  if df.empty or not game_weather_settings:
    return df
  df = df.copy()
  for idx, row in df.iterrows():
    game = row.get("game", "")
    pos = standardize_position(row.get("position", ""))
    if game not in game_weather_settings:
      continue
    settings = game_weather_settings[game]
    severity = settings["severity"]
    wind = settings["wind"]
    proj = row["blended_proj"]
    sd = row["std_dev"]

    if "High Winds" in severity:
      if pos == "QB":
        df.at[idx, "blended_proj"] = proj * max(
            0.80, 1.0 - max(0, (wind - 15)) * 0.018
        )
        df.at[idx, "std_dev"] = sd * 1.18
      elif pos == "WR":
        df.at[idx, "blended_proj"] = proj * max(
            0.75, 1.0 - max(0, (wind - 15)) * 0.022
        )
        df.at[idx, "std_dev"] = sd * 1.25
      elif pos == "RB":
        df.at[idx, "blended_proj"] = proj * 1.08
    elif "Heavy Rain" in severity:
      if pos in ["QB", "WR", "TE"]:
        df.at[idx, "blended_proj"] = proj * 0.88
        df.at[idx, "std_dev"] = sd * 1.30
      elif pos == "RB":
        df.at[idx, "blended_proj"] = proj * 1.05
    elif "Blizzard" in severity:
      if pos in ["QB", "WR", "TE"]:
        df.at[idx, "blended_proj"] = proj * 0.70
        df.at[idx, "std_dev"] = sd * 1.45
      elif pos == "RB":
        df.at[idx, "blended_proj"] = proj * 1.15
  return df


# -----------------------------------------------------------------------------
# 8. MASTER DATA LOAD & COMPUTATION
# -----------------------------------------------------------------------------
raw_df = load_combined_data(
    uploaded_fd,
    uploaded_espn,
    uploaded_vegas,
    api_key=odds_api_key,
    force_api_refresh=force_api_refresh,
)
if raw_df.empty:
  st.warning(
      "⚠️ No projection data found. Please upload FanDuel CSV or Vegas Excel"
      " file."
  )
  st.stop()


def compute_blended_proj(row, w_v, w_e):
  pos_std = standardize_position(row.get("position", ""))
  if pos_std == "DEF":
    return row["vegas_proj"] if row["vegas_proj"] > 0 else 7.2
  v, e = row["vegas_proj"], row["espn_proj"]
  if v > 0 and e > 0:
    return (v * w_v) + (e * w_e)
  elif v > 0:
    return v
  elif e > 0:
    return e
  return 0.0


raw_df["blended_proj"] = raw_df.apply(
    lambda r: compute_blended_proj(r, w_vegas, w_espn), axis=1
)

# --- SIDEBAR SAVE FINAL DRAFT PROJECTIONS BUTTON ---
st.sidebar.markdown("---")
st.sidebar.header("💾 Archival Controls")
save_week_num = st.sidebar.number_input(
    "Target NFL Week Number", min_value=1, max_value=22, value=8, step=1
)


def generate_calibration_draft_csv(df, week):
  proj_col = (
      df["blended_proj"]
      if "blended_proj" in df.columns
      else df.apply(lambda r: compute_blended_proj(r, 0.6, 0.4), axis=1)
  )
  export_df = pd.DataFrame({
      "week": week,
      "name": df["name"],
      "position": df["position"].apply(standardize_position),
      "salary": df["salary"],
      "projection": np.round(proj_col, 2),
      "actual": 0.0,
  })
  return export_df.to_csv(index=False).encode("utf-8")


if not raw_df.empty:
  st.sidebar.download_button(
      label="💾 Save Final Draft Projections",
      data=generate_calibration_draft_csv(raw_df, save_week_num),
      file_name=f"week_{save_week_num}_final_draft_projections.csv",
      mime="text/csv",
      help=(
          "Export final projections pre-lock in exact format for Walk-Forward"
          " Calibration Lab."
      ),
  )

# --- ROLE-AWARE OPPORTUNITY REDISTRIBUTION & INACTIVE ENGINE ---
st.sidebar.markdown("---")
st.sidebar.header("🚨 Game-Day Inactives & Role Redistribution")

if "inactives_checked" not in st.session_state:
  st.session_state["inactives_checked"] = False
  st.session_state["inactive_players_list"] = []

if st.sidebar.button("🔄 Sync Official Inactives"):
  fetched_inactives = []
  try:
    for _, r in raw_df.iterrows():
      if r.get("vegas_proj", 1.0) == 0.0 and r.get("espn_proj", 1.0) == 0.0:
        fetched_inactives.append({
            "name": r["name"],
            "team": r.get("team", ""),
            "position": r.get("position", ""),
            "status": "Official Out / Ruled Out",
        })
  except Exception as e:
    st.sidebar.error(f"Inactive sync error: {e}")

  st.session_state["inactives_checked"] = True
  st.session_state["inactive_players_list"] = fetched_inactives
  if fetched_inactives:
    st.sidebar.success(
        f"Synced: Found {len(fetched_inactives)} officially ruled-out players."
    )
  else:
    st.sidebar.info("Sync complete: No official game-day inactives detected.")

if st.session_state["inactives_checked"]:
  with st.expander("🚨 Official Inactive Players Report", expanded=True):
    if st.session_state["inactive_players_list"]:
      st.dataframe(
          pd.DataFrame(st.session_state["inactive_players_list"]),
          use_container_width=True,
          hide_index=True,
      )
    else:
      st.info("All rostered players are currently active.")

if st.session_state["inactives_checked"] and st.session_state[
    "inactive_players_list"
]:
  inactive_names = [p["name"] for p in st.session_state["inactive_players_list"]]

  if inactive_names:
    for team_code, team_group in raw_df.groupby("team"):
      if not team_code:
        continue
      inactive_mask = team_group["name"].isin(inactive_names)
      vacated_df = team_group[inactive_mask]
      if vacated_df.empty:
        continue

      active_team_mask = (raw_df["team"] == team_code) & (
          ~raw_df["name"].isin(inactive_names)
      )
      active_team_df = raw_df[active_team_mask]
      if active_team_df.empty:
        continue

      for _, vac_player in vacated_df.iterrows():
        vac_pos = standardize_position(vac_player["position"])
        vac_pts = vac_player["blended_proj"]
        if vac_pts <= 0:
          continue

        if vac_pos == "RB":
          absorbable_pts = vac_pts * 0.80
          rb_actives = active_team_df[
              active_team_df["position"].apply(standardize_position) == "RB"
          ]
          if not rb_actives.empty:
            weights = rb_actives["blended_proj"] + 1.0
            norm_weights = weights / weights.sum()
            for rb_idx, w in norm_weights.items():
              raw_df.at[rb_idx, "blended_proj"] += absorbable_pts * w

        elif vac_pos in ["WR", "TE"]:
          absorbable_pts = vac_pts * 0.92
          pass_actives = active_team_df[
              active_team_df["position"]
              .apply(standardize_position)
              .isin(["WR", "TE"])
          ]
          if not pass_actives.empty:
            weights = pass_actives["blended_proj"] + 1.0
            norm_weights = weights / weights.sum()
            for p_idx, w in norm_weights.items():
              raw_df.at[p_idx, "blended_proj"] += absorbable_pts * w

        elif vac_pos == "QB":
          absorbable_pts = vac_pts * 0.65
          qb_actives = active_team_df[
              active_team_df["position"].apply(standardize_position) == "QB"
          ]
          if not qb_actives.empty:
            for q_idx in qb_actives.index:
              raw_df.at[q_idx, "blended_proj"] += absorbable_pts

    raw_df.loc[raw_df["name"].isin(inactive_names), "blended_proj"] = 0.0
    raw_df = raw_df[~raw_df["name"].isin(inactive_names)].reset_index(drop=True)

# Weather Controls
game_weather_settings = {}
if enable_weather_impact and not raw_df.empty:
  active_outdoor_games = get_slate_outdoor_games(raw_df)
  if active_outdoor_games:
    with st.expander(
        "🌦️ Configure Independent Game Weather Conditions", expanded=False
    ):
      cols = st.columns(min(3, len(active_outdoor_games)))
      for i, g_name in enumerate(active_outdoor_games):
        with cols[i % len(cols)]:
          home_team = g_name.split("@")[-1].strip()
          coords = STADIUM_COORDS.get(home_team, (40.0, -90.0))
          forecast = fetch_stadium_forecast(coords[0], coords[1])
          st.markdown(
              f"**{g_name}**  \n🌡️ `{forecast['temp']}°F` | 💨"
              f" `{forecast['wind']} mph` | 🌧️ `{forecast['precip']}%`"
          )
          g_sev = st.selectbox(
              "Condition Override",
              [
                  "Mild / Normal",
                  "High Winds (18+ mph)",
                  "Heavy Rain / Wet",
                  "Blizzard / Freezing",
              ],
              key=f"sev_{g_name}",
          )
          g_wind = forecast["wind"]
          if "Wind" in g_sev:
            g_wind = st.slider(
                "Custom Wind (mph)",
                10,
                35,
                int(max(10, forecast["wind"])),
                1,
                key=f"wind_{g_name}",
            )
          game_weather_settings[g_name] = {"severity": g_sev, "wind": g_wind}

# Exclusions
st.sidebar.markdown("---")
st.sidebar.header("🚫 Exclusions")
excluded_players = st.sidebar.multiselect(
    "Exclude Players", sorted(raw_df["name"].unique().tolist())
)
if excluded_players:
  raw_df = raw_df[~raw_df["name"].isin(excluded_players)].reset_index(drop=True)

excluded_games = st.sidebar.multiselect(
    "Exclude Games", sorted(raw_df["game"].unique().tolist())
    if "game" in raw_df.columns
    else []
)
if excluded_games and "game" in raw_df.columns:
  raw_df = raw_df[~raw_df["game"].isin(excluded_games)].reset_index(drop=True)

POSITION_VOLATILITY = {
    "QB": 0.38,
    "RB": 0.45,
    "WR": 0.52,
    "TE": 0.55,
    "DEF": 0.65,
    "FLEX": 0.48,
}


def compute_salary_boom_bust(row):
  pos = standardize_position(row.get("position", ""))
  proj = row["blended_proj"]
  sal = float(row.get("salary", 5000))
  std_dev = proj * POSITION_VOLATILITY.get(pos, 0.45) * global_risk_mult
  boom_target, bust_target = (sal / 1000.0) * 4.0, (sal / 1000.0) * 2.0
  if std_dev > 0:
    boom_pct = norm.sf((boom_target - proj) / std_dev) * 100
    bust_pct = norm.cdf((bust_target - proj) / std_dev) * 100
  else:
    boom_pct, bust_pct = 0.0, 0.0
  return pd.Series({
      "std_dev": std_dev,
      "floor": max(0.0, proj - 1.28 * std_dev),
      "ceiling": proj + 1.28 * std_dev,
      "boom_target": round(boom_target, 1),
      "boom_rate": round(boom_pct, 1),
      "bust_rate": round(bust_pct, 1),
  })


metrics_df = raw_df.apply(compute_salary_boom_bust, axis=1)
raw_df[[
    "std_dev",
    "floor",
    "ceiling",
    "boom_target",
    "boom_rate",
    "bust_rate",
]] = metrics_df

if enable_weather_impact and game_weather_settings:
  raw_df = apply_per_game_weather_adjustments(raw_df, game_weather_settings)


# -----------------------------------------------------------------------------
# 9. GAME-SCRIPT CORRELATED MONTE CARLO SIMULATION ENGINE
# -----------------------------------------------------------------------------
def run_game_script_slate_simulations(df, num_sims=3000):
  games = df["game"].fillna("UNKNOWN_GAME").values
  teams = df["team"].values
  base_projs = df["blended_proj"].values
  std_devs = df["std_dev"].values
  positions = df["position"].apply(standardize_position).values
  N = len(df)

  unique_games = sorted(list(set(games)))
  sim_matrix = np.zeros((num_sims, N))
  np.random.seed(42)

  game_to_indices = {g: np.where(games == g)[0] for g in unique_games}

  for s_idx in range(num_sims):
    for g in unique_games:
      g_indices = game_to_indices[g]
      if len(g_indices) == 0:
        continue

      game_pace_shock = np.random.normal(1.0, 0.10)
      shootout_shock = np.random.normal(1.0, 0.12)

      teams_in_game = set(teams[g_indices])
      team_margin_shocks = {}
      if len(teams_in_game) >= 2:
        t_list = list(teams_in_game)
        margin = np.random.normal(0.0, 1.5)
        team_margin_shocks[t_list[0]] = margin
        team_margin_shocks[t_list[1]] = -margin
      else:
        for t in teams_in_game:
          team_margin_shocks[t] = 0.0

      for p_i in g_indices:
        p_team = teams[p_i]
        p_pos = positions[p_i]
        base_p = base_projs[p_i]
        sd = max(0.5, std_devs[p_i])

        margin = team_margin_shocks.get(p_team, 0.0)
        script_multiplier = 1.0
        if p_pos == "RB":
          script_multiplier += margin * 0.06
        elif p_pos in ["QB", "WR", "TE"]:
          script_multiplier -= margin * 0.04

        adjusted_mean = (
            base_p * game_pace_shock * shootout_shock * script_multiplier
        )
        sim_matrix[s_idx, p_i] = max(0.0, np.random.normal(adjusted_mean, sd))

  return (
      np.percentile(sim_matrix, 85, axis=0) * 0.7
      + np.mean(sim_matrix, axis=0) * 0.3
  ), sim_matrix


sim_result_tuple = run_game_script_slate_simulations(raw_df)
raw_df["gpp_upside_score"] = sim_result_tuple[0]
slate_sim_matrix = sim_result_tuple[1]

df_full = calculate_estimated_ownership(raw_df)
df_full["value"] = np.where(
    df_full["salary"] > 0,
    np.round(df_full["blended_proj"] / (df_full["salary"] / 1000.0), 2),
    0.0,
)

ceiling_z = (df_full["ceiling"] - df_full["ceiling"].mean()) / max(
    0.01, df_full["ceiling"].std()
)
own_z = (df_full["proj_ownership"] - df_full["proj_ownership"].mean()) / max(
    0.01, df_full["proj_ownership"].std()
)
df_full["leverage"] = np.round(ceiling_z - own_z, 2)

df_active = df_full[df_full["blended_proj"] > 0].copy().reset_index(drop=True)

# -----------------------------------------------------------------------------
# 10. DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Player Pool & Metrics",
    "📈 Monte Carlo Visualizer",
    "⚡ Lineup Optimizer",
    "🏆 Portfolio Contest Sims",
    "📉 Walk-Forward Calibration Lab",
])

# --- TAB 1 ---
with tab1:
  st.subheader("📊 Master Player Pool & Blended Projections")
  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Active Players", len(df_active))
  c2.metric(
      "Top Ceiling",
      f"{df_active['ceiling'].max():.1f} pts"
      if not df_active.empty
      else "0.0",
  )
  c3.metric(
      "Top Value",
      f"{df_active['value'].max():.2f}x" if not df_active.empty else "0.0",
  )
  c4.metric(
      "Avg Ownership",
      f"{df_active['proj_ownership'].mean():.1f}%"
      if not df_active.empty
      else "0.0%",
  )

  f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
  selected_pos = f_col1.selectbox(
      "Filter Position", ["ALL", "FLEX", "QB", "RB", "WR", "TE", "DEF"]
  )
  selected_team = f_col2.selectbox(
      "Filter Team", ["ALL"] + sorted(df_active["team"].unique().tolist())
  )
  search_term = f_col3.text_input("Search Player", "").strip().lower()

  view_df = df_active.copy()
  if selected_pos == "FLEX":
    view_df = view_df[view_df["position"].isin(["RB", "WR", "TE"])]
  elif selected_pos != "ALL":
    view_df = view_df[view_df["position"] == selected_pos]
  if selected_team != "ALL":
    view_df = view_df[view_df["team"] == selected_team]
  if search_term:
    view_df = view_df[view_df["name"].str.lower().str.contains(search_term)]

  # Position-Aware Display Column Mapping
  if selected_pos == "QB":
    display_cols = [
        "name",
        "position",
        "team",
        "salary",
        "blended_proj",
        "proj_source_log",
        "Pass Yds Prop",
        "Pass TD Prop",
        "INT Prop",
        "Rush Yds Prop",
        "Anytime TD Odds",
        "Implied TD %",
        "ceiling",
        "proj_ownership",
        "leverage",
        "value",
    ]
  elif selected_pos == "RB":
    display_cols = [
        "name",
        "position",
        "team",
        "salary",
        "blended_proj",
        "proj_source_log",
        "Rush Yds Prop",
        "Rec Yds Prop",
        "Receptions Prop",
        "Anytime TD Odds",
        "Implied TD %",
        "ceiling",
        "proj_ownership",
        "leverage",
        "value",
    ]
  elif selected_pos in ["WR", "TE", "FLEX"]:
    display_cols = [
        "name",
        "position",
        "team",
        "salary",
        "blended_proj",
        "proj_source_log",
        "Rec Yds Prop",
        "Receptions Prop",
        "Rush Yds Prop",
        "Anytime TD Odds",
        "Implied TD %",
        "ceiling",
        "proj_ownership",
        "leverage",
        "value",
    ]
  else:
    display_cols = [
        "name",
        "position",
        "team",
        "salary",
        "blended_proj",
        "proj_source_log",
        "Pass Yds Prop",
        "Pass TD Prop",
        "INT Prop",
        "Rush Yds Prop",
        "Rec Yds Prop",
        "Receptions Prop",
        "Anytime TD Odds",
        "Implied TD %",
        "ceiling",
        "proj_ownership",
        "leverage",
        "value",
    ]

  st.dataframe(
      view_df[[c for c in display_cols if c in view_df.columns]]
      .sort_values(by="blended_proj", ascending=False)
      .style.format({
          "blended_proj": "{:.1f}",
          "Pass Yds Prop": "{:.1f}",
          "Pass TD Prop": "{:.1f}",
          "INT Prop": "{:.1f}",
          "Rush Yds Prop": "{:.1f}",
          "Rec Yds Prop": "{:.1f}",
          "Receptions Prop": "{:.1f}",
          "Implied TD %": "{:.2f}",
          "ceiling": "{:.1f}",
          "value": "{:.2f}",
          "leverage": "{:.2f}",
          "proj_ownership": "{:.1f}%",
      }),
      use_container_width=True,
      height=550,
      hide_index=True,
  )

# --- TAB 2 ---
with tab2:
  st.subheader("📈 Monte Carlo Visualizer & Leverage Map")
  top_players = (
      df_active.sort_values(by="blended_proj", ascending=False)["name"]
      .head(10)
      .tolist()
  )
  selected_players = st.multiselect(
      "Select Players",
      df_active["name"].unique().tolist(),
      default=top_players[:4] if len(top_players) >= 4 else top_players,
  )
  if selected_players:
    fig = go.Figure()
    for p_name in selected_players:
      p_row = df_active[df_active["name"] == p_name].iloc[0]
      sim_points = np.random.normal(
          p_row["blended_proj"], max(0.5, p_row["std_dev"]), 5000
      )
      x_range = np.linspace(
          max(0, p_row["blended_proj"] - 3 * p_row["std_dev"]),
          p_row["blended_proj"] + 3 * p_row["std_dev"],
          200,
      )
      fig.add_trace(go.Scatter(
          x=x_range,
          y=gaussian_kde(sim_points)(x_range),
          mode="lines",
          name=p_name,
          fill="tozeroy",
          opacity=0.4,
      ))
    fig.update_layout(
        title="Point Distributions",
        xaxis_title="FP",
        yaxis_title="Density",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

  st.plotly_chart(
      px.scatter(
          df_active,
          x="proj_ownership",
          y="ceiling",
          color="position",
          size="blended_proj",
          hover_data=["name", "team", "salary", "value", "leverage"],
          title="Ceiling vs Ownership Matrix",
      ).update_layout(template="plotly_white"),
      use_container_width=True,
  )

# --- TAB 3 ---
with tab3:
  st.subheader("⚡ Simulation-Driven Portfolio Optimizer")
  st.markdown(
      "Generate a candidate pool of legal lineups, score them across"
      " game-script slates, and optimize the portfolio as a whole to maximize"
      " joint return and diversity."
  )

  if not HAS_MILP:
    st.error("scipy MILP solver missing.")
    st.stop()

  c1, c2, c3, c4, c5 = st.columns(5)
  num_lineups_target = c1.number_input("Target Portfolio Size", 1, 150, 20)
  pool_size_multiplier = c2.number_input(
      "Candidate Pool Multiplier", 3, 10, 5, 1
  )
  max_stack = c3.number_input("Max Stack", 2, 5, 4)
  avoid_weather_stacks = c3.checkbox(
      "Auto-Fade Weather Games in Stacks", value=True
  )
  max_exposure = c4.slider("Max Exposure (%)", 20.0, 100.0, 35.0, 5.0) / 100.0
  gpp_blend_weight = c5.slider("GPP Blend (%)", 0.0, 100.0, 50.0, 5.0) / 100.0

  stack_col1, stack_col2 = st.columns(2)
  qb_stack_count = stack_col1.selectbox("QB Stack Count", [0, 1, 2, 3], index=1)
  force_runback = stack_col2.checkbox("Force Opposing Runback", value=True)

  if st.button("🚀 Run Simulation-Driven Portfolio Optimizer", type="primary"):
    with st.spinner(
        "Generating candidate pool and solving global portfolio optimization..."
    ):
      pool = (
          df_active[df_active["blended_proj"] > 0]
          .copy()
          .reset_index(drop=True)
      )
      N_pool = len(pool)
      if N_pool < 9:
        st.error("Not enough players in active pool.")
        st.stop()

      weather_penalized_games = set()
      if enable_weather_impact and "game_weather_settings" in locals():
        for g_name, g_settings in game_weather_settings.items():
          if any(
              w in g_settings["severity"]
              for w in ["High Winds", "Heavy Rain", "Blizzard"]
          ):
            weather_penalized_games.add(g_name)

      pool["optimizer_score"] = (
          (1.0 - gpp_blend_weight) * pool["blended_proj"]
          + gpp_blend_weight * pool["gpp_upside_score"]
      ) * np.where(pool["game"].isin(weather_penalized_games), 0.85, 1.0)

      pool_pos = pool["position"].apply(standardize_position)
      is_qb = (pool_pos == "QB").astype(int).to_numpy()
      is_rb = (pool_pos == "RB").astype(int).to_numpy()
      is_wr = (pool_pos == "WR").astype(int).to_numpy()
      is_te = (pool_pos == "TE").astype(int).to_numpy()
      is_dst = (pool_pos == "DEF").astype(int).to_numpy()
      is_flex = is_rb | is_wr | is_te
      has_dst = is_dst.sum() > 0
      target_roster_size = 8 + (1 if has_dst else 0)

      salaries, base_scores = pool["salary"].to_numpy(), pool[
          "optimizer_score"
      ].to_numpy()
      candidate_lineups = []
      historical_lineup_indices = []
      total_candidates_to_gen = num_lineups_target * pool_size_multiplier

      for _ in range(total_candidates_to_gen):
        scores = base_scores * (1.0 + np.random.normal(0, 0.12, N_pool))
        A_rows = [salaries, np.ones(N_pool)]
        lb_list, ub_list = [0, target_roster_size], [60000, target_roster_size]

        if is_qb.sum() > 0:
          A_rows.append(is_qb)
          lb_list.append(1)
          ub_list.append(1)
        A_rows.extend([is_rb, is_wr, is_te, is_flex])
        lb_list.extend([2, 3, 1, 7])
        ub_list.extend([3, 4, 2, 7])
        if has_dst:
          A_rows.append(is_dst)
          lb_list.append(1)
          ub_list.append(1)

        for tm in pool["team"].unique():
          if tm:
            A_rows.append((pool["team"] == tm).astype(int).to_numpy())
            lb_list.append(0)
            ub_list.append(max_stack)

        for q_i in np.where(is_qb == 1)[0]:
          q_team, q_opp = pool.iloc[q_i]["team"], pool.iloc[q_i]["opponent"]
          if qb_stack_count > 0 and q_team:
            A_rows.append(
                (qb_stack_count * np.eye(N_pool)[q_i])
                - (
                    (pool["team"] == q_team)
                    & pool["position"].isin(["RB", "WR", "TE"])
                ).astype(int)
            )
            lb_list.append(-N_pool)
            ub_list.append(0)
          if force_runback and q_opp:
            A_rows.append(
                np.eye(N_pool)[q_i]
                - (
                    (pool["team"] == q_opp)
                    & pool["position"].isin(["RB", "WR", "TE"])
                ).astype(int)
            )
            lb_list.append(-N_pool)
            ub_list.append(0)

        for prev in historical_lineup_indices:
          cut_row = np.zeros(N_pool)
          cut_row[prev] = 1.0
          A_rows.append(cut_row)
          lb_list.append(0)
          ub_list.append(len(prev) - 2)

        res = milp(
            c=-scores,
            integrality=np.ones(N_pool),
            bounds=Bounds(lb=0, ub=1),
            constraints=LinearConstraint(
                np.vstack(A_rows), np.array(lb_list), np.array(ub_list)
            ),
        )
        if res.success:
          sel = np.where(res.x > 0.5)[0]
          historical_lineup_indices.append(sel)
          candidate_lineups.append(sel)

      if not candidate_lineups:
        st.error("Could not generate candidate pool. Adjust constraints.")
        st.stop()

      pool_key_to_sim_idx = {
          row["_canon_key"]: i for i, row in pool.iterrows()
      }
      num_sims = slate_sim_matrix.shape[0]
      candidate_scores = np.zeros((len(candidate_lineups), num_sims))

      for c_idx, c_indices in enumerate(candidate_lineups):
        c_canon_keys = pool.iloc[c_indices]["_canon_key"].tolist()
        sim_cols = [
            pool_key_to_sim_idx[k]
            for k in c_canon_keys
            if k in pool_key_to_sim_idx
        ]
        if sim_cols:
          candidate_scores[c_idx, :] = np.sum(
              slate_sim_matrix[:, sim_cols], axis=1
          )

      candidate_mean = np.mean(candidate_scores, axis=1)
      candidate_ceiling = np.percentile(candidate_scores, 85, axis=1)
      candidate_utility = 0.6 * candidate_mean + 0.4 * candidate_ceiling

      M = len(candidate_lineups)
      cand_matrix = np.zeros((N_pool, M))
      for c_idx, c_indices in enumerate(candidate_lineups):
        cand_matrix[c_indices, c_idx] = 1.0

      port_A_rows = [np.ones(M)]
      port_lb = [num_lineups_target]
      port_ub = [num_lineups_target]

      max_allowed_appearances = max(
          1, int(np.floor(max_exposure * num_lineups_target))
      )
      for p_i in range(N_pool):
        port_A_rows.append(cand_matrix[p_i, :])
        port_lb.append(0)
        port_ub.append(max_allowed_appearances)

      port_res = milp(
          c=-candidate_utility,
          integrality=np.ones(M),
          bounds=Bounds(lb=0, ub=1),
          constraints=LinearConstraint(
              np.vstack(port_A_rows), np.array(port_lb), np.array(port_ub)
          ),
      )

      if port_res.success:
        selected_cand_indices = np.where(port_res.x > 0.5)[0]
        final_portfolio = [
            pool.iloc[candidate_lineups[idx]].copy()
            for idx in selected_cand_indices
        ]

        st.session_state["last_generated_portfolio"] = final_portfolio
        st.success(
            f"Successfully optimized global portfolio of {len(final_portfolio)}"
            f" lineups from a candidate pool of {M}!"
        )
      else:
        st.error(
            "Portfolio selection solver could not satisfy global exposure"
            " constraints. Try loosening max exposure."
        )

  if "last_generated_portfolio" in st.session_state:
    st.markdown("---")
    st.subheader("📋 Current Saved Portfolio Lineups")
    for idx, lu in enumerate(st.session_state["last_generated_portfolio"], 1):
      with st.expander(
          f"Lineup #{idx} — Salary: ${lu['salary'].sum():,} | Blended Proj:"
          f" {lu['blended_proj'].sum():.1f}"
      ):
        display_lu_df = lu[
            ["name", "position", "team", "salary", "blended_proj"]
        ]
        st.dataframe(
            display_lu_df, use_container_width=True, hide_index=True
        )

# --- TAB 4 ---
with tab4:
  st.subheader("🏆 Portfolio Field Contest Simulator & Payout Lab")
  st.markdown(
      "Simulate slates with exact contest payout structures, entry fees, and"
      " vectorized field ranking."
  )

  c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns(4)
  sim_count = c_cfg1.slider(
      "Slate Simulations", 1000, 10000, 3000, 1000, key="sim_cnt"
  )
  field_size = c_cfg2.number_input(
      "Field Lineups (Entries)", 500, 20000, 2500, 500, key="f_size"
  )
  entry_fee = c_cfg3.number_input("Entry Fee ($)", 1.0, 550.0, 20.0, 1.0)
  contest_preset = c_cfg4.selectbox(
      "Contest Structure",
      options=[
          "Milly Maker ($20 Entry, 20% to 1st)",
          "Standard GPP ($15 Entry, 10% to 1st)",
          "Single-Entry Qualifier ($50 Entry)",
      ],
      index=0,
  )

  if "last_generated_portfolio" not in st.session_state:
    st.info("💡 Generate your portfolio in Tab 3 first.")
  else:
    portfolio = st.session_state["last_generated_portfolio"]
    st.write(
        f"Ready to simulate **{len(portfolio)} portfolio entries** against"
        f" **{field_size:,} field entries** across **{sim_count:,} slates**."
    )

    if st.button("🎲 Run Vectorized Payout Sims", type="primary"):
      with st.spinner(
          "Running vectorized simulations and payout resolution..."
      ):
        np.random.seed(42)
        pool = (
            df_active[df_active["blended_proj"] > 0]
            .copy()
            .reset_index(drop=True)
        )
        N = len(pool)
        teams = pool["team"].values
        base_projs = pool["blended_proj"].values
        std_devs = pool["std_dev"].values

        pool_pos = pool["position"].apply(standardize_position)
        is_qb = (pool_pos == "QB").to_numpy()
        is_rb = (pool_pos == "RB").to_numpy()
        is_wr = (pool_pos == "WR").to_numpy()
        is_te = (pool_pos == "TE").to_numpy()
        is_def = (pool_pos == "DEF").to_numpy()
        salaries = pool["salary"].to_numpy()


        def safe_sample(df_sub, count):
          weights = df_sub["proj_ownership"].values
          w_sum = weights.sum()
          p = (
              weights / w_sum
              if w_sum > 0
              else np.ones(len(df_sub)) / len(df_sub)
          )
          return df_sub.sample(n=count, replace=False, weights=p).index.tolist()


        field_indices = []
        attempts = 0
        while len(field_indices) < field_size and attempts < field_size * 5:
          attempts += 1
          try:
            chosen = []
            qbs_sub = pool[is_qb]
            rbs_sub = pool[is_rb]
            wrs_sub = pool[is_wr]
            tes_sub = pool[is_te]
            defs_sub = pool[is_def]

            if (
                len(qbs_sub) < 1
                or len(rbs_sub) < 2
                or len(wrs_sub) < 3
                or len(tes_sub) < 1
            ):
              continue

            chosen.extend(safe_sample(qbs_sub, 1))
            chosen.extend(safe_sample(rbs_sub, 2))
            chosen.extend(safe_sample(wrs_sub, 3))
            chosen.extend(safe_sample(tes_sub, 1))
            if not defs_sub.empty:
              chosen.extend(safe_sample(defs_sub, 1))

            rem_flex = pool.loc[
                (is_rb | is_wr | is_te) & (~pool.index.isin(chosen))
            ]
            if rem_flex.empty:
              continue
            chosen.extend(safe_sample(rem_flex, 1))

            if len(chosen) == 9 and salaries[chosen].sum() <= 60000:
              field_indices.append(chosen)
          except Exception:
            pass

        actual_sims = min(sim_count, slate_sim_matrix.shape[0])
        sim_matrix = slate_sim_matrix[:actual_sims, :]
        sim_count = actual_sims

        key_to_idx = {row["_canon_key"]: i for i, row in pool.iterrows()}

        user_scores = np.zeros((len(portfolio), sim_count))
        for lu_idx, lu_df in enumerate(portfolio):
          c_idx = [
              key_to_idx[k] for k in lu_df["_canon_key"] if k in key_to_idx
          ]
          if c_idx:
            user_scores[lu_idx, :] = np.sum(sim_matrix[:, c_idx], axis=1)

        field_scores = np.zeros((len(field_indices), sim_count))
        for f_idx, f_indices in enumerate(field_indices):
          field_scores[f_idx, :] = np.sum(sim_matrix[:, f_indices], axis=1)

        total_entries = len(field_indices) + len(portfolio)
        total_prize_pool = entry_fee * total_entries

        payout_table = np.zeros(total_entries)
        if "Milly Maker" in contest_preset:
          payout_table[0] = total_prize_pool * 0.20
          payout_table[1:5] = total_prize_pool * 0.05
          payout_table[5:20] = total_prize_pool * 0.01
          payout_table[20:100] = (total_prize_pool * 0.15) / 80.0
          payout_table[100 : int(total_entries * 0.20)] = (
              total_prize_pool * 0.25
          ) / max(1, int(total_entries * 0.20) - 100)
        else:
          payout_table[0] = total_prize_pool * 0.10
          payout_table[1:10] = (total_prize_pool * 0.20) / 9.0
          payout_table[10 : int(total_entries * 0.20)] = (
              total_prize_pool * 0.40
          ) / max(1, int(total_entries * 0.20) - 10)

        eval_records = []
        portfolio_total_payout = 0.0
        portfolio_total_cost = len(portfolio) * entry_fee * sim_count

        for lu_idx in range(len(portfolio)):
          u_sc = user_scores[lu_idx, :]
          lu_wins, lu_top1pct, lu_top10pct, lu_payouts = 0, 0, 0, 0.0

          for s_i in range(sim_count):
            combined_scores = np.concatenate(
                [field_scores[:, s_i], [u_sc[s_i]]]
            )
            ranks = (
                total_entries
                - np.argsort(np.argsort(combined_scores, kind="mergesort"))
            )
            u_rank = ranks[-1]

            if u_rank == 1:
              lu_wins += 1
            if u_rank <= total_entries * 0.01:
              lu_top1pct += 1
            if u_rank <= total_entries * 0.10:
              lu_top10pct += 1

            payout = payout_table[u_rank - 1] if u_rank <= total_entries else 0.0
            lu_payouts += payout
            portfolio_total_payout += payout

          win_prob = (lu_wins / sim_count) * 100
          top1_prob = (lu_top1pct / sim_count) * 100
          cash_prob = (lu_top10pct / sim_count) * 100
          lu_roi = (
              (lu_payouts - (entry_fee * sim_count))
              / (entry_fee * sim_count)
          ) * 100

          eval_records.append({
              "Lineup #": lu_idx + 1,
              "Blended Proj": round(
                  portfolio[lu_idx]["blended_proj"].sum(), 1
              ),
              "1st Place Win Prob (%)": round(win_prob, 2),
              "Top 1% Finish (%)": round(top1_prob, 2),
              "Cash Prob (%)": round(cash_prob, 2),
              "Expected ROI (%)": round(lu_roi, 1),
          })

        portfolio_roi = (
            (portfolio_total_payout - portfolio_total_cost)
            / portfolio_total_cost
        ) * 100
        st.session_state["portfolio_eval_df"] = pd.DataFrame(eval_records)
        st.session_state["portfolio_net_roi"] = portfolio_roi

    if "portfolio_eval_df" in st.session_state:
      res_df = st.session_state["portfolio_eval_df"].copy()
      net_portfolio_roi = st.session_state.get("portfolio_net_roi", 0.0)

      st.markdown("---")
      st.subheader("📊 Portfolio Performance Summary Dashboard")
      c_m1, c_m2, c_m3, c_m4 = st.columns(4)
      c_m1.metric("Net Portfolio Payout ROI", f"{net_portfolio_roi:+.1f}%")
      c_m2.metric(
          "Avg 1st Place Win Prob",
          f"{res_df['1st Place Win Prob (%)'].mean():.2f}%",
      )
      c_m3.metric(
          "Avg Top 1% Probability", f"{res_df['Top 1% Finish (%)'].mean():.2f}%"
      )
      c_m4.metric(
          "Avg Cash Probability",
          f"{res_df['Cash Prob (%)'].mean():.1f}%",
      )

      st.markdown("---")
      st.subheader("🏆 Portfolio Lineup Performance & Expected Payout ROI")
      st.dataframe(
          res_df.sort_values(by="Expected ROI (%)", ascending=False),
          use_container_width=True,
          hide_index=True,
      )

      # --- FANDUEL MASS CSV EXPORT W/ PLAYER ID CODES ---
      st.markdown("---")
      st.subheader("📥 Export Lineups for FanDuel Mass CSV Edit")
      st.markdown(
          "Select individual lineups or check 'Select All', then download the"
          " CSV formatted with official FanDuel player ID numeric codes."
      )

      sel_cols_ex1, sel_cols_ex2 = st.columns([1, 3])
      select_all_box = sel_cols_ex1.checkbox("Select All Lineups", value=True)

      selected_lineup_indices = []
      st.write("**Choose Individual Lineups to Export:**")
      chk_cols = st.columns(min(5, len(portfolio)))
      for idx, _ in enumerate(portfolio):
        col_idx = idx % len(chk_cols)
        with chk_cols[col_idx]:
          is_sel = st.checkbox(
              f"Lineup #{idx+1}",
              value=select_all_box,
              key=f"exp_lu_{idx+1}",
          )
          if is_sel:
            selected_lineup_indices.append(idx)

      if selected_lineup_indices:
        fd_export_rows = []
        for l_idx in selected_lineup_indices:
          lu_df = portfolio[l_idx]
          row_dict = {}
          for _, p_row in lu_df.iterrows():
            pos = standardize_position(p_row.get("position", ""))
            p_id = str(p_row.get("player_id", ""))
            if not p_id or p_id.lower() == "nan":
              p_id = str(p_row.get("name", ""))

            if pos == "QB" and "QB" not in row_dict:
              row_dict["QB"] = p_id
            elif pos == "RB":
              if "RB1" not in row_dict:
                row_dict["RB1"] = p_id
              elif "RB2" not in row_dict:
                row_dict["RB2"] = p_id
              else:
                row_dict["FLEX"] = p_id
            elif pos == "WR":
              if "WR1" not in row_dict:
                row_dict["WR1"] = p_id
              elif "WR2" not in row_dict:
                row_dict["WR2"] = p_id
              elif "WR3" not in row_dict:
                row_dict["WR3"] = p_id
              else:
                row_dict["FLEX"] = p_id
            elif pos == "TE" and "TE" not in row_dict:
              row_dict["TE"] = p_id
            elif pos == "DEF" and "DEF" not in row_dict:
              row_dict["DEF"] = p_id
            else:
              row_dict["FLEX"] = p_id

          fd_export_rows.append(row_dict)

        fd_export_df = pd.DataFrame(fd_export_rows)
        csv_data = fd_export_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label=(
                f"📥 Download FanDuel Mass CSV ({len(selected_lineup_indices)}"
                " Lineups)"
            ),
            data=csv_data,
            file_name="fanduel_mass_edit_lineups.csv",
            mime="text/csv",
            type="primary",
        )
      else:
        st.warning("⚠️ Please select at least one lineup to export.")

      # Lineup Roster Inspector
      st.markdown("---")
      st.subheader("🔍 Inspect Portfolio Lineup Rosters")
      inspect_lu_num = st.selectbox(
          "Select Lineup # to View Roster",
          options=res_df["Lineup #"].tolist(),
      )
      if inspect_lu_num:
        target_lu = portfolio[int(inspect_lu_num) - 1]
        st.markdown(
            f"**Lineup #{inspect_lu_num}** — Total Salary:"
            f" **${target_lu['salary'].sum():,}** | Total Blended Proj:"
            f" **{target_lu['blended_proj'].sum():.1f} pts**"
        )
        display_target_lu = target_lu[
            ["name", "position", "team", "salary", "blended_proj"]
        ]
        st.dataframe(
            display_target_lu, use_container_width=True, hide_index=True
        )

# --- TAB 5 ---
with tab5:
  st.subheader(
      "📉 Walk-Forward Projection Calibration & Automated nflverse Lab"
  )
  st.markdown(
      "Upload your historical pre-game draft files. The app will automatically"
      " pull official player box scores from **nflverse**, calculate actual"
      " FanDuel points on the fly, and execute walk-forward out-of-sample"
      " training."
  )

  col_up1, col_up2 = st.columns(2)
  wf_file = col_up1.file_uploader(
      "Upload Historical Pre-Game Draft File(s) (CSV / Excel)",
      type=["csv", "xlsx"],
      key="wf_upload",
  )
  target_season_year = col_up2.number_input(
      "Target nflverse Season Year", min_value=2021, max_value=2026, value=2026
  )

  c_w1, c_w2, c_w3 = st.columns(3)
  is_window_size = c_w1.number_input(
      "In-Sample Training Weeks", min_value=2, max_value=8, value=4
  )
  oos_window_size = c_w2.number_input(
      "Out-of-Sample Test Weeks", min_value=1, max_value=3, value=1
  )
  opt_objective = c_w3.selectbox(
      "Calibration Objective", ["Minimize MAE", "Minimize RMSE"]
  )

  if wf_file is not None:
    try:
      if wf_file.name.endswith(".csv"):
        wf_df = pd.read_csv(wf_file)
      else:
        wf_df = pd.read_excel(wf_file)

      if st.button("🌐 Auto-Fetch nflverse Stats & Compute Actuals"):
        with st.spinner(
            f"Pulling official weekly player stats for {target_season_year}"
            " from nflverse..."
        ):
          nflverse_stats = fetch_nflverse_weekly_stats(target_season_year)
          if not nflverse_stats.empty:
            calc_actuals_map = {}
            for _, r in nflverse_stats.iterrows():
              p_name = r.get("player_display_name", r.get("player_name", ""))
              p_week = r.get("week", 0)
              p_key = normalize_player_key(p_name)

              py = float(r.get("passing_yards", 0.0) or 0.0)
              pt = float(r.get("passing_tds", 0.0) or 0.0)
              pi = float(r.get("interceptions", 0.0) or 0.0)
              ry = float(r.get("rushing_yards", 0.0) or 0.0)
              rt = float(r.get("rushing_tds", 0.0) or 0.0)
              rc = float(r.get("receptions", 0.0) or 0.0)
              rcy = float(r.get("receiving_yards", 0.0) or 0.0)
              rct = float(r.get("receiving_tds", 0.0) or 0.0)
              fl = float(r.get("fumbles_lost", 0.0) or 0.0)

              calc_fp = calculate_fd_expected_points(
                  pass_yds=py,
                  pass_tds=pt,
                  ints=pi,
                  rush_yds=ry,
                  rush_tds=rt,
                  receptions=rc,
                  rec_yds=rcy,
                  rec_tds=rct,
                  fumbles_lost=fl,
              )
              calc_actuals_map[(p_key, int(p_week))] = calc_fp

            updated_actuals = []
            for _, row in wf_df.iterrows():
              wk = int(row.get("week", 1))
              pkey = normalize_player_key(row.get("name", ""))
              matched_actual = calc_actuals_map.get((pkey, wk), row.get("actual", 0.0))
              updated_actuals.append(matched_actual)

            wf_df["actual"] = updated_actuals
            st.session_state["wf_df_processed"] = wf_df
            st.success(
                "Successfully pulled nflverse stats and auto-calculated"
                " actual FD points!"
            )
          else:
            st.error("Failed to retrieve nflverse dataset.")

      active_wf_df = st.session_state.get("wf_df_processed", wf_df)

      required_cols = ["week", "name", "position", "salary", "projection", "actual"]
      missing_cols = [c for c in required_cols if c not in active_wf_df.columns]

      if missing_cols:
        st.warning(
            f"Missing required columns for automated walk-forward: {missing_cols}."
            " Please ensure your file includes: week, name, position, salary,"
            " projection, actual."
        )
      else:
        if st.button("🔄 Run Walk-Forward Calibration Loop", type="primary"):
          with st.spinner("Executing rolling walk-forward optimization..."):
            weeks = sorted(active_wf_df["week"].unique())
            if len(weeks) <= is_window_size:
              st.error(
                  f"Not enough historical weeks ({len(weeks)}) for an"
                  f" in-sample window of {is_window_size}. Please add more weeks"
                  " or reduce training window size."
              )
            else:
              wf_results = []
              for i in range(len(weeks) - is_window_size):
                train_weeks = weeks[i : i + is_window_size]
                test_weeks = weeks[
                    i + is_window_size : i + is_window_size + oos_window_size
                ]
                if not test_weeks:
                  break

                train_data = active_wf_df[active_wf_df["week"].isin(train_weeks)]
                test_data = active_wf_df[active_wf_df["week"].isin(test_weeks)]

                pos_multipliers = {}
                for pos, group in train_data.groupby("position"):
                  act_sum = group["actual"].sum()
                  proj_sum = group["projection"].sum()
                  pos_multipliers[pos] = (
                      act_sum / proj_sum if proj_sum > 0 else 1.0
                  )

                test_data = test_data.copy()
                test_data["calibrated_proj"] = test_data.apply(
                    lambda r: r["projection"]
                    * pos_multipliers.get(r["position"], 1.0),
                    axis=1,
                )

                uncal_mae = np.mean(
                    np.abs(test_data["projection"] - test_data["actual"])
                )
                uncal_rmse = np.sqrt(
                    np.mean(
                        (test_data["projection"] - test_data["actual"]) ** 2
                    )
                )

                cal_mae = np.mean(
                    np.abs(test_data["calibrated_proj"] - test_data["actual"])
                )
                cal_rmse = np.sqrt(
                    np.mean(
                        (test_data["calibrated_proj"] - test_data["actual"])
                        ** 2
                    )
                )

                wf_results.append({
                    "Train Weeks": str(train_weeks),
                    "Test Week(s)": str(test_weeks),
                    "Uncalibrated MAE": round(uncal_mae, 2),
                    "Calibrated MAE": round(cal_mae, 2),
                    "MAE Improvement (%)": round(
                        ((uncal_mae - cal_mae) / uncal_mae) * 100, 1
                    ),
                    "Uncalibrated RMSE": round(uncal_rmse, 2),
                    "Calibrated RMSE": round(cal_rmse, 2),
                    "RMSE Improvement (%)": round(
                        ((uncal_rmse - cal_rmse) / uncal_rmse) * 100, 1
                    ),
                })

              wf_res_df = pd.DataFrame(wf_results)
              st.session_state["wf_res_df"] = wf_res_df

          if "wf_res_df" in st.session_state:
            st.markdown("### 📈 Out-of-Sample Walk-Forward Validation Results")
            res_summary = st.session_state["wf_res_df"]

            c_res1, c_res2 = st.columns(2)
            c_res1.metric(
                "Avg OOS MAE Improvement",
                f"{res_summary['MAE Improvement (%)'].mean():+.1f}%",
            )
            c_res2.metric(
                "Avg OOS RMSE Improvement",
                f"{res_summary['RMSE Improvement (%)'].mean():+.1f}%",
            )

            st.dataframe(res_summary, use_container_width=True, hide_index=True)

    except Exception as e:
      st.error(f"Error processing historical file: {e}")
  else:
    st.info(
        "💡 Upload your historical pre-game draft files to automatically pull"
        " official stats from nflverse and run calibration."
    )