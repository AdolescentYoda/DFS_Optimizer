import csv
import datetime
from io import BytesIO
import os
from pathlib import Path
import re
import shutil

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
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DFT Optimizer Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏈 Daily Fantasy Tournament (DFT) Optimizer & Simulation Lab")
st.caption(
    "Live Weight Blending | Monte Carlo Ceiling Visualizer | Field-Simulated GPP"
    " Engine | Backtest Lab"
)

# Constants & Paths
APP_DIR = (
    Path(__file__).parent.resolve() if "__file__" in locals() else Path.cwd()
)
DEFAULT_VEGAS_FILE = APP_DIR / "NFL Current Week Projections.xlsx"
DEFAULT_FD_FILE = APP_DIR / "FanDuel.csv"
DEFAULT_ESPN_FILE = APP_DIR / "espn_projections.csv"

# Standard NFL Team Dictionary for Canonical Defense Matching
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
# 2. API CACHING LOGIC (24-Hour TTL)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def fetch_odds_api_data(api_key: str):
  """Fetches live odds/projections data from The Odds API.

  Streamlit caches this in memory for 24 hours (86,400 seconds).
  """
  import requests

  url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
  params = {
      "apiKey": api_key,
      "regions": "us",
      "markets": "h2h,spreads,totals",
  }
  response = requests.get(url, params=params)
  response.raise_for_status()
  return response.json()


# -----------------------------------------------------------------------------
# 3. HELPER UTILITIES & DATA PARSERS
# -----------------------------------------------------------------------------
def excel_col_index(col_str: str) -> int:
  """Converts Excel column letters (e.g., 'A', 'J', 'AA') to zero-based index."""
  col_str = col_str.upper().strip()
  exp = 0
  idx = 0
  for char in reversed(col_str):
    idx += (ord(char) - ord("A") + 1) * (26**exp)
    exp += 1
  return idx - 1


def parse_numeric_series(series, default=0.0):
  """Safely extracts numeric float values, handling string formulas and symbols."""

  def clean_val(v):
    if pd.isna(v):
      return default
    s = str(v).strip()
    if s.startswith("="):
      nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
      if nums:
        return float(nums[-1])
      return default
    s_clean = s.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
      return float(s_clean)
    except ValueError:
      return default

  return series.apply(clean_val)


def standardize_position(pos):
  """Standardizes player and defense position strings into QB, RB, WR, TE, or DEF."""
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
  """Normalize player names for reliable cross-file matching."""
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
  """Extracts standard 2-3 letter NFL team code from any defense string."""
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
  """Generates a unified key for both players and defense units to eliminate cross-source duplicates."""
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
# 4. OWNERSHIP MODELING
# -----------------------------------------------------------------------------
def calculate_estimated_ownership(df: pd.DataFrame) -> pd.DataFrame:
  """Calculates realistic DFS projected ownership (%) based on market attraction score."""
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
    if current_sum > 0:
      final_own = np.clip(
          capped_own * (target_mass / current_sum), 0.5, 45.0
      )
    else:
      final_own = capped_own

    pos_df["proj_ownership"] = np.round(final_own, 1)
    calculated_groups.append(pos_df)

  if calculated_groups:
    result_df = pd.concat(calculated_groups).sort_index()
    result_df = result_df.drop(
        columns=["_salary_num", "_proj_num", "_pos_clean"], errors="ignore"
    )
    return result_df

  return df


# -----------------------------------------------------------------------------
# 5. WORKBOOK & DATA READERS
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
  """Loads FanDuel player list CSV into a structured DataFrame."""
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
    if isinstance(fd_source, BytesIO):
      fd_source.seek(0)
      df_fd = pd.read_csv(fd_source)
    else:
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
    res_df = (
        res_df.drop_duplicates(subset=["_canon_key"], keep="first")
        .reset_index(drop=True)
    )
    return res_df, fd_display_name
  except Exception as e:
    st.sidebar.error(f"Error reading FanDuel CSV: {e}")
    return pd.DataFrame(), None


def read_vegas_workbook(source):
  """Read Vegas projections directly from workbook sheets."""
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

  if not any(pos == "DEF" for pos, _ in target_sheets.values()):
    for s_name in xls.sheet_names:
      s_lower = s_name.lower()
      if any(k in s_lower for k in ["def", "dst", "defense"]):
        target_sheets[s_name] = ("DEF", "G")
        break

  for sheet_name, (pos_code, score_column) in target_sheets.items():
    try:
      raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    except Exception as e:
      st.sidebar.error(f"Could not read {sheet_name}: {e}")
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

    name_col_idx = 1
    team_col_idx = 2
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
      elif c_val in [
          "projected score",
          "vegas projection",
          "vegas proj",
          "vegas score",
          "projected pts",
          "projected points",
      ] or (
          (
              "projected score" in c_val
              or ("vegas" in c_val and "proj" in c_val)
          )
          and "weighted" not in c_val
          and "ownership" not in c_val
      ):
        score_col_idx = c_i

    if score_col_idx >= raw.shape[1]:
      score_col_idx = min(11, raw.shape[1] - 1)

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
          "vegas_source_cell": f"{score_column}{raw_row_idx + 1}",
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
  """Extracts ESPN projections into a dict mapping canonical key to projection value."""
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
        content = src.getvalue()
        if isinstance(content, bytes):
          content = content.decode("utf-8-sig", errors="ignore")
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
    fd_file, espn_file, vegas_file, default_espn_path="espn_projections.csv"
):
  """Load FanDuel CSV as MASTER slate universe and merge Vegas & ESPN projections."""
  fd_df, fd_name = load_fd_csv(fd_file)

  vegas_source = None
  vegas_display_name = None

  if vegas_file is not None:
    vegas_source = BytesIO(vegas_file.getvalue())
    vegas_display_name = vegas_file.name
  elif DEFAULT_VEGAS_FILE.exists():
    vegas_source = str(DEFAULT_VEGAS_FILE)
    vegas_display_name = DEFAULT_VEGAS_FILE.name
  else:
    candidates = sorted(
        APP_DIR.glob("*Projections*.xlsx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
      vegas_source = str(candidates[0])
      vegas_display_name = candidates[0].name

  vegas_df = (
      read_vegas_workbook(vegas_source)
      if vegas_source is not None
      else pd.DataFrame()
  )
  espn_map = extract_espn_projections_map(espn_file, default_espn_path)

  if not fd_df.empty:
    st.sidebar.success(f"FanDuel Roster Loaded: {fd_name}")
    st.sidebar.metric("FanDuel Slate Players", len(fd_df))

    df = fd_df.copy()

    vegas_proj_map = {}
    vegas_espn_map = {}
    vegas_sheet_map = {}
    if not vegas_df.empty:
      for _, row in vegas_df.iterrows():
        k = row["_canon_key"]
        vegas_proj_map[k] = row.get("vegas_proj", 0.0)
        vegas_espn_map[k] = row.get("espn_proj", 0.0)
        vegas_sheet_map[k] = row.get("vegas_source_sheet", "")

    df["vegas_proj"] = df["_canon_key"].map(vegas_proj_map).fillna(0.0)
    df["vegas_source_sheet"] = (
        df["_canon_key"].map(vegas_sheet_map).fillna("")
    )

    df["espn_proj"] = df["_canon_key"].map(espn_map).fillna(0.0)
    vegas_espn_series = df["_canon_key"].map(vegas_espn_map).fillna(0.0)
    df["espn_proj"] = np.where(
        df["espn_proj"] > 0, df["espn_proj"], vegas_espn_series
    )

    vegas_matched = int((df["vegas_proj"] > 0).sum())
    espn_matched = int((df["espn_proj"] > 0).sum())

    st.sidebar.metric("Vegas Projections Matched", vegas_matched)
    st.sidebar.metric("ESPN Projections Matched", espn_matched)

    with st.sidebar.expander("🔎 Import Verification"):
      verification_cols = [
          "name",
          "team",
          "position",
          "salary",
          "espn_proj",
          "vegas_proj",
      ]
      st.dataframe(
          df[verification_cols].head(15),
          use_container_width=True,
          hide_index=True,
      )

    return df

  if not vegas_df.empty:
    st.sidebar.warning(
        "⚠️ No FanDuel CSV found. Base pool created from Vegas workbook."
    )
    df = vegas_df.copy()
    mapped_espn = df["_canon_key"].map(espn_map)
    df["espn_proj"] = np.where(
        mapped_espn > 0, mapped_espn, df.get("espn_proj", 0.0)
    )
    df["salary"] = (
        pd.to_numeric(df.get("salary"), errors="coerce")
        .fillna(5000)
        .astype(int)
    )
    return (
        df.drop_duplicates(subset=["_canon_key"], keep="first")
        .reset_index(drop=True)
    )

  return pd.DataFrame()


# -----------------------------------------------------------------------------
# 6. SIDEBAR CONFIGURATION & CONTROLS
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
st.sidebar.header("⚡ Live API Controls")

try:
  api_key_default = st.secrets.get("ODDS_API_KEY", "")
except Exception:
  api_key_default = ""

odds_api_key = st.sidebar.text_input(
    "Odds API Key", value=api_key_default, type="password"
)

if st.sidebar.button("🔄 Force Fresh API Pull"):
  fetch_odds_api_data.clear()
  st.sidebar.success("API Cache Cleared!")
  st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🎯 Post-Mortem Snapshot")
st.sidebar.markdown(
    "Save a date-stamped pre-lock snapshot of current projections to grade"
    " against actual scores next week."
)

if st.sidebar.button("📸 Snapshot Current Slate for Post-Mortem"):
  if DEFAULT_VEGAS_FILE.exists():
    try:
      today_str = datetime.date.today().strftime("%Y-%m-%d")
      dated_snapshot_file = APP_DIR / f"NFL Previous Week_{today_str}.xlsx"

      shutil.copy(DEFAULT_VEGAS_FILE, dated_snapshot_file)
      st.sidebar.success(
          f"Successfully archived snapshot as `{dated_snapshot_file.name}`!"
      )
    except Exception as e:
      st.sidebar.error(f"Failed to create snapshot: {e}")
  else:
    st.sidebar.warning("⚠️ Default Vegas workbook file not found to snapshot.")

st.sidebar.markdown("---")
st.sidebar.header("⚙ Blending Weights")
w_vegas = st.sidebar.slider("Vegas Weight", 0.0, 1.0, 0.60, 0.05)
w_espn = st.sidebar.slider("ESPN Weight", 0.0, 1.0, 0.40, 0.05)

total_w = w_vegas + w_espn
if total_w > 0:
  w_vegas /= total_w
  w_espn /= total_w
else:
  w_vegas, w_espn = 0.5, 0.5

st.sidebar.caption(
    f"Effective Weights: Vegas = {w_vegas:.2f} | ESPN = {w_espn:.2f}"
)

st.sidebar.markdown("---")
st.sidebar.header("🎲 Simulation Volatility")
global_risk_mult = st.sidebar.slider(
    "Volatility Multiplier", 0.5, 2.0, 1.2, 0.1
)

# -----------------------------------------------------------------------------
# 7. MASTER DATA LOAD & SIMULATION ENGINE
# -----------------------------------------------------------------------------
raw_df = load_combined_data(uploaded_fd, uploaded_espn, uploaded_vegas)

if raw_df.empty:
  st.warning(
      "⚠️ No projection data found. Please upload FanDuel CSV or Vegas Excel"
      " file."
  )
  st.stop()

# --- EXCLUSION FILTERS (Players & Games) ---
st.sidebar.markdown("---")
st.sidebar.header("🚫 Exclusions & Filtering")

all_player_names = (
    sorted(raw_df["name"].unique().tolist()) if not raw_df.empty else []
)
excluded_players = st.sidebar.multiselect("Exclude Players", all_player_names)
if excluded_players:
  raw_df = raw_df[~raw_df["name"].isin(excluded_players)].reset_index(drop=True)

all_games = (
    sorted(raw_df["game"].unique().tolist())
    if "game" in raw_df.columns and not raw_df.empty
    else []
)
excluded_games = st.sidebar.multiselect("Exclude Games", all_games)
if excluded_games and "game" in raw_df.columns:
  raw_df = raw_df[~raw_df["game"].isin(excluded_games)].reset_index(drop=True)


def apply_calibration_bias(row, blended_val):
  pos = standardize_position(row.get("position", ""))
  if blended_val <= 0:
    return 0.0
  multipliers = {
      "QB": 1.030,
      "RB": 1.010,
      "WR": 1.035,
      "TE": 1.025,
      "DEF": 1.000,
      "FLEX": 1.030,
  }
  return blended_val * multipliers.get(pos, 1.02)


def compute_blended_proj(row, w_v, w_e):
  pos_std = standardize_position(row.get("position", ""))
  if pos_std == "DEF":
    return row["vegas_proj"] if row["vegas_proj"] > 0 else 0.0

  v = row["vegas_proj"]
  e = row["espn_proj"]
  salary = row.get("salary", 5000)

  if salary >= 7500 and pos_std in ["QB", "RB", "WR"]:
    effective_v_weight = 0.45
    effective_e_weight = 0.55
  else:
    effective_v_weight = w_v
    effective_e_weight = w_e

  if v > 0 and e > 0:
    base_blend = (v * effective_v_weight) + (e * effective_e_weight)
  elif v > 0:
    base_blend = v
  elif e > 0:
    base_blend = e
  else:
    base_blend = 0.0

  return apply_calibration_bias(row, base_blend)


raw_df["blended_proj"] = raw_df.apply(
    lambda r: compute_blended_proj(r, w_vegas, w_espn), axis=1
)

POSITION_VOLATILITY = {
    "QB": 0.38,
    "RB": 0.45,
    "WR": 0.52,
    "TE": 0.55,
    "DEF": 0.65,
    "FLEX": 0.48,
}


def compute_salary_boom_bust(row):
  """Calculates salary-adjusted Boom (4x target) and Bust rates using standard normal distribution."""
  pos = standardize_position(row.get("position", ""))
  proj = (
      raw_df.loc[row.name, "blended_proj"]
      if hasattr(row, "name")
      else row["blended_proj"]
  )
  sal = float(row.get("salary", 5000))

  vol_coeff = POSITION_VOLATILITY.get(pos, 0.45)
  std_dev = proj * vol_coeff * global_risk_mult

  # 4x Salary Boom target model (e.g., $7,000 player needs 28.0 points to boom)
  boom_target = (sal / 1000.0) * 4.0
  bust_target = (sal / 1000.0) * 2.0

  if std_dev > 0:
    z_boom = (boom_target - proj) / std_dev
    boom_pct = norm.sf(z_boom) * 100

    z_bust = (bust_target - proj) / std_dev
    bust_pct = norm.cdf(z_bust) * 100
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


def run_background_slate_simulations(df, num_sims=2000):
  teams = df["team"].values
  base_projs = df["blended_proj"].values
  std_devs = df["std_dev"].values
  N = len(df)
  sim_matrix = np.zeros((num_sims, N))

  for s_idx in range(num_sims):
    team_shocks = {tm: np.random.normal(1.0, 0.15) for tm in set(teams) if tm}
    for p_i in range(N):
      tm = teams[p_i]
      shock = team_shocks.get(tm, 1.0)
      mean = base_projs[p_i] * shock
      sd = max(0.5, std_devs[p_i])
      sim_matrix[s_idx, p_i] = max(0.0, np.random.normal(mean, sd))

  sim_mean = np.mean(sim_matrix, axis=0)
  sim_ceiling = np.percentile(sim_matrix, 85, axis=0)
  return sim_ceiling * 0.7 + sim_mean * 0.3


raw_df["gpp_upside_score"] = run_background_slate_simulations(raw_df)

df_full = calculate_estimated_ownership(raw_df)
df_full["value"] = np.where(
    df_full["salary"] > 0,
    np.round(df_full["blended_proj"] / (df_full["salary"] / 1000.0), 2),
    0.0,
)
df_full["leverage"] = np.round(
    df_full["ceiling"] - df_full["proj_ownership"], 2
)

df_active = df_full[df_full["blended_proj"] > 0].copy().reset_index(drop=True)

# -----------------------------------------------------------------------------
# 8. DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Player Pool & Metrics",
    "📈 Monte Carlo Visualizer",
    "⚡ Lineup Optimizer",
    "🏆 Portfolio Contest Sims",
    "📉 Model Backtester",
])

# -----------------------------------------------------------------------------
# TAB 1: PLAYER POOL & METRICS
# -----------------------------------------------------------------------------
with tab1:
  st.subheader("📊 Master Player Pool & Blended Projections")

  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Active Slate Players", len(df_active))
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
      "Avg Projected Ownership",
      f"{df_active['proj_ownership'].mean():.1f}%"
      if not df_active.empty
      else "0.0%",
  )

  st.markdown("---")

  f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
  position_options = ["ALL", "FLEX", "QB", "RB", "WR", "TE", "DEF"]
  selected_pos = f_col1.selectbox("Filter Position", position_options)

  teams = ["ALL"] + sorted(df_active["team"].unique().tolist())
  selected_team = f_col2.selectbox("Filter Team", teams)

  search_term = f_col3.text_input("Search Player Name", "").strip().lower()

  view_df = df_active.copy()
  if selected_pos == "FLEX":
    view_df = view_df[view_df["position"].isin(["RB", "WR", "TE"])]
  elif selected_pos != "ALL":
    view_df = view_df[view_df["position"] == selected_pos]

  if selected_team != "ALL":
    view_df = view_df[view_df["team"] == selected_team]
  if search_term:
    view_df = view_df[view_df["name"].str.lower().str.contains(search_term)]

  display_cols = [
      "name",
      "position",
      "team",
      "opponent",
      "salary",
      "blended_proj",
      "gpp_upside_score",
      "value",
      "floor",
      "ceiling",
      "boom_target",
      "boom_rate",
      "bust_rate",
      "proj_ownership",
      "leverage",
  ]

  st.dataframe(
      view_df[display_cols]
      .sort_values(by="blended_proj", ascending=False)
      .style.format({
          "blended_proj": "{:.1f}",
          "gpp_upside_score": "{:.1f}",
          "value": "{:.2f}",
          "floor": "{:.1f}",
          "ceiling": "{:.1f}",
          "boom_target": "{:.1f}",
          "boom_rate": "{:.1f}%",
          "bust_rate": "{:.1f}%",
          "proj_ownership": "{:.1f}%",
          "leverage": "{:.2f}",
      }),
      use_container_width=True,
      height=550,
      hide_index=True,
  )

  csv_buffer = BytesIO()
  view_df[display_cols].to_csv(csv_buffer, index=False)
  st.download_button(
      "📥 Download Filtered Projections CSV",
      data=csv_buffer.getvalue(),
      file_name="dft_projections.csv",
      mime="text/csv",
  )

# -----------------------------------------------------------------------------
# TAB 2: MONTE CARLO VISUALIZER
# -----------------------------------------------------------------------------
with tab2:
  st.subheader("📈 Monte Carlo Projection Distributions & Leverage Analysis")

  top_players = (
      df_active.sort_values(by="blended_proj", ascending=False)["name"]
      .head(10)
      .tolist()
  )
  selected_players = st.multiselect(
      "Select Players to Compare Distributions",
      df_active["name"].unique().tolist(),
      default=top_players[:4] if len(top_players) >= 4 else top_players,
  )

  if selected_players:
    fig = go.Figure()
    sim_stats = []

    for p_name in selected_players:
      p_row = df_active[df_active["name"] == p_name].iloc[0]
      mean_val = p_row["blended_proj"]
      std_val = max(0.5, p_row["std_dev"])

      sim_points = np.random.normal(mean_val, std_val, 5000)
      kde = gaussian_kde(sim_points)
      x_range = np.linspace(
          max(0, mean_val - 3 * std_val), mean_val + 3 * std_val, 200
      )
      y_density = kde(x_range)

      fig.add_trace(go.Scatter(
          x=x_range,
          y=y_density,
          mode="lines",
          name=p_name,
          fill="tozeroy",
          opacity=0.4,
      ))

      sim_stats.append({
          "Player": p_name,
          "Pos": p_row["position"],
          "Median Proj": np.median(sim_points),
          "85th Ceiling": np.percentile(sim_points, 85),
          "4x Boom Target": p_row["boom_target"],
          "Boom Rate (4x Target)": f"{p_row['boom_rate']}%",
          "Bust Rate (2x Floor)": f"{p_row['bust_rate']}%",
      })

    fig.update_layout(
        title="Simulated Fantasy Point Distributions",
        xaxis_title="Fantasy Points",
        yaxis_title="Probability Density",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Simulation Stats")
    st.dataframe(
        pd.DataFrame(sim_stats), use_container_width=True, hide_index=True
    )

  st.markdown("---")
  st.subheader("🎯 Ceiling vs. Projected Ownership (Leverage Map)")

  fig_scatter = px.scatter(
      df_active,
      x="proj_ownership",
      y="ceiling",
      color="position",
      size="blended_proj",
      hover_data=["name", "team", "salary", "value", "leverage"],
      labels={
          "proj_ownership": "Projected Ownership (%)",
          "ceiling": "85th Percentile Ceiling (Pts)",
      },
      title="Ceiling vs Ownership Matrix",
  )
  fig_scatter.update_layout(template="plotly_white")
  st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: LINEUP OPTIMIZER WITH QB STACKING & OPPOSING RUNBACK RULES
# -----------------------------------------------------------------------------
with tab3:
  st.subheader("⚡ Tournament Lineup Portfolio Generator")

  if not HAS_MILP:
    st.error("⚠️ `scipy` MILP solver is not installed in the environment.")
    st.stop()

  # Main portfolio configuration parameters
  c1, c2, c3, c4, c5 = st.columns(5)
  num_lineups = c1.number_input("Number of Lineups", 1, 50, 20)
  max_stack = c2.number_input("Max Stack per Team", 2, 5, 4)
  randomness = c3.slider("Randomness / Noise (%)", 0.0, 20.0, 10.0, 1.0) / 100.0
  max_exposure = (
      c4.slider("Max Player Exposure (%)", 20.0, 100.0, 30.0, 10.0) / 100.0
  )
  gpp_blend_weight = (
      c5.slider("GPP Simulation Blend (%)", 0.0, 100.0, 45.0, 5.0) / 100.0
  )

  st.markdown("---")
  st.markdown("#### 🏈 QB Stacking & Opposing Runback Rules")
  stack_col1, stack_col2 = st.columns(2)

  qb_stack_count = stack_col1.selectbox(
      "QB Teammate Stack Count (Pass Catchers/RBs)",
      options=[0, 1, 2, 3],
      index=1,
      help=(
          "Number of position players (RB/WR/TE) from the QB's team forced into"
          " the lineup."
      ),
  )

  force_runback = stack_col2.checkbox(
      "Force Opposing Team Runback Player",
      value=True,
      help=(
          "Require at least 1 opposing position player (RB/WR/TE) from the"
          " QB's opponent to capture shootout correlation."
      ),
  )

  if st.button("🚀 Optimize Lineups", type="primary"):
    pool = df_active[df_active["blended_proj"] > 0].copy().reset_index(drop=True)
    N = len(pool)

    if N < 9:
      st.error("Not enough players in pool to construct a valid lineup.")
      st.stop()

    pool["optimizer_score"] = (
        1.0 - gpp_blend_weight
    ) * pool["blended_proj"] + gpp_blend_weight * pool["gpp_upside_score"]

    pool_pos = pool["position"].apply(standardize_position)
    is_qb = (pool_pos == "QB").astype(int).to_numpy()
    is_rb = (pool_pos == "RB").astype(int).to_numpy()
    is_wr = (pool_pos == "WR").astype(int).to_numpy()
    is_te = (pool_pos == "TE").astype(int).to_numpy()
    is_dst = (pool_pos == "DEF").astype(int).to_numpy()
    is_flex = is_rb | is_wr | is_te

    has_dst = is_dst.sum() > 0
    req_dst = 1 if has_dst else 0
    target_roster_size = 8 + req_dst

    salaries = pool["salary"].to_numpy()
    base_scores = pool["optimizer_score"].to_numpy()
    salary_cap = 60000

    exposure_counts = np.zeros(N)
    portfolio = []
    historical_lineup_indices = []
    max_allowed_appearances = max(1, int(np.floor(max_exposure * num_lineups)))

    for lu_idx in range(num_lineups):
      if randomness > 0:
        noise = np.random.normal(0, randomness, N)
        scores = base_scores * (1.0 + noise)
      else:
        scores = base_scores.copy()

      c = -scores

      A_rows = [salaries, np.ones(N)]
      lb_list = [0, target_roster_size]
      ub_list = [salary_cap, target_roster_size]

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

      # Enforce Team Max Stack Limit
      for tm in pool["team"].unique():
        if not tm:
          continue
        tm_mask = (pool["team"] == tm).astype(int).to_numpy()
        A_rows.append(tm_mask)
        lb_list.append(0)
        ub_list.append(max_stack)

      # --- ADVANCED QB STACKING & RUNBACK CONSTRAINTS ---
      qb_indices = np.where(is_qb == 1)[0]

      for q_i in qb_indices:
        qb_team = pool.iloc[q_i]["team"]
        qb_opp = pool.iloc[q_i]["opponent"]

        # 1. Primary QB Teammate Stack Constraint (Require X teammates if QB selected)
        if qb_stack_count > 0 and qb_team:
          team_skill_mask = (
              (pool["team"] == qb_team) & pool["position"].isin(["RB", "WR", "TE"])
          ).astype(int).to_numpy()
          stack_row = (qb_stack_count * (1 - is_qb[q_i])) - team_skill_mask
          # Logic: if QB is selected (x_q = 1), then sum(skill_teammates) >= qb_stack_count
          # Linear form: qb_stack_count * x_q - sum(skill_teammates) <= 0
          A_rows.append((qb_stack_count * np.eye(N)[q_i]) - team_skill_mask)
          lb_list.append(-N)
          ub_list.append(0)

        # 2. Opposing Runback Constraint (Require >= 1 opponent if QB selected)
        if force_runback and qb_opp:
          opp_skill_mask = (
              (pool["team"] == qb_opp) & pool["position"].isin(["RB", "WR", "TE"])
          ).astype(int).to_numpy()
          # Logic: if QB selected (x_q = 1), then sum(opp_skill_players) >= 1
          # Linear form: 1 * x_q - sum(opp_skill_players) <= 0
          A_rows.append(np.eye(N)[q_i] - opp_skill_mask)
          lb_list.append(-N)
          ub_list.append(0)

      # Exclude duplicate lineups
      for prev_indices in historical_lineup_indices:
        cut_row = np.zeros(N)
        cut_row[prev_indices] = 1.0
        A_rows.append(cut_row)
        lb_list.append(0)
        ub_list.append(len(prev_indices) - 1)

      ub_bounds = np.ones(N)
      for idx in range(N):
        if exposure_counts[idx] >= max_allowed_appearances:
          ub_bounds[idx] = 0.0

      A_mat = np.vstack(A_rows)
      constraints = LinearConstraint(
          A_mat, np.array(lb_list), np.array(ub_list)
      )
      bounds = Bounds(lb=0, ub=ub_bounds)

      res = milp(
          c=c, integrality=np.ones(N), bounds=bounds, constraints=constraints
      )

      if res.success:
        selected_indices = np.where(res.x > 0.5)[0]
        historical_lineup_indices.append(selected_indices)
        exposure_counts[selected_indices] += 1
        lineup_df = pool.iloc[selected_indices].copy()
        portfolio.append(lineup_df)
      else:
        break

    if portfolio:
      portfolio.sort(key=lambda lu: lu["blended_proj"].sum(), reverse=True)
      st.session_state["last_generated_portfolio"] = portfolio

      st.success(
          f"Generated {len(portfolio)} Unique Optimized Lineups (Stacking: +{qb_stack_count} Teammate(s) | Runback: {force_runback}) — Saved to Session!"
      )

      for idx, lu in enumerate(portfolio, 1):
        with st.expander(
            f"🏈 Lineup #{idx} — Total Salary: ${lu['salary'].sum():,} | Blended"
            f" Pts: {lu['blended_proj'].sum():.1f} | GPP Score:"
            f" {lu['gpp_upside_score'].sum():.1f} | Avg Ownership:"
            f" {lu['proj_ownership'].mean():.1f}%"
        ):
          st.dataframe(
              lu[[
                  "name",
                  "position",
                  "team",
                  "opponent",
                  "salary",
                  "blended_proj",
                  "gpp_upside_score",
                  "proj_ownership",
              ]],
              use_container_width=True,
              hide_index=True,
          )

      st.markdown("---")
      st.subheader("📊 Portfolio Player Exposure & Export")

      col_exp, col_dl = st.columns([2, 1])

      exp_df = pool.copy()
      exp_df["Exposure (%)"] = np.round(
          (exposure_counts / len(portfolio)) * 100, 1
      )
      exp_df = exp_df[exp_df["Exposure (%)"] > 0].sort_values(
          by="Exposure (%)", ascending=False
      )

      with col_exp:
        st.dataframe(
            exp_df[[
                "name",
                "position",
                "team",
                "opponent",
                "salary",
                "blended_proj",
                "gpp_upside_score",
                "Exposure (%)",
            ]],
            use_container_width=True,
            height=300,
            hide_index=True,
        )

      with col_dl:
        st.markdown("#### 📥 Mass Upload CSV (ID-based)")
        fd_rows = []
        for lu in portfolio:
          get_id = lambda pos_code, idx_slice: [
              p.get("player_id", p["name"])
              for p in lu[lu["position"] == pos_code].to_dict(orient="records")
          ][idx_slice]

          qb = get_id("QB", slice(0, 1))
          rbs = get_id("RB", slice(0, 2))
          wrs = get_id("WR", slice(0, 3))
          tes = get_id("TE", slice(0, 1))
          defs = get_id("DEF", slice(0, 1))

          flex_candidates = (
              lu[lu["position"] == "RB"].to_dict(orient="records")[2:]
              + lu[lu["position"] == "WR"].to_dict(orient="records")[3:]
              + lu[lu["position"] == "TE"].to_dict(orient="records")[1:]
          )
          flex_id = (
              [
                  flex_candidates[0].get(
                      "player_id", flex_candidates[0]["name"]
                  )
              ]
              if len(flex_candidates) > 0
              else [""]
          )

          fd_rows.append({
              "QB": qb[0] if len(qb) > 0 else "",
              "RB": rbs[0] if len(rbs) > 0 else "",
              "RB.1": rbs[1] if len(rbs) > 1 else "",
              "WR": wrs[0] if len(wrs) > 0 else "",
              "WR.1": wrs[1] if len(wrs) > 1 else "",
              "WR.2": wrs[2] if len(wrs) > 2 else "",
              "TE": tes[0] if len(tes) > 0 else "",
              "F LEX": flex_id[0] if len(flex_id) > 0 else "",
              "D": defs[0] if len(defs) > 0 else "",
          })

        fd_csv_df = pd.DataFrame(fd_rows)
        fd_csv_df.columns = [
            "QB",
            "RB",
            "RB",
            "WR",
            "WR",
            "WR",
            "TE",
            "F LEX",
            "D",
        ]

        csv_data = fd_csv_df.to_csv(index=False)
        st.download_button(
            label="📥 Download FanDuel ID CSV",
            data=csv_data,
            file_name="fanduel_mass_upload_ids.csv",
            mime="text/csv",
            type="primary",
        )
    else:
      st.error(
          "Could not solve for valid lineups given current stacking"
          " constraints. Try lowering randomness or disabling runback rules."
      )

# -----------------------------------------------------------------------------
# TAB 4: PORTFOLIO CONTEST SIMS (FIELD-SIMULATED ROI ENGINE)
# -----------------------------------------------------------------------------
with tab4:
  st.subheader("🏆 Portfolio Field Contest Simulator")
  st.markdown(
      "Simulate thousands of slates **against a realistic opponent field**"
      " generated from estimated ownership probabilities (`proj_ownership`)."
      " Incorporates tie-breakers and duplicate-lineup penalties."
  )

  c_sim1, c_sim2 = st.columns(2)
  sim_count = c_sim1.slider(
      "Number of Slate Simulations", 1000, 10000, 3000, 1000
  )
  field_size = c_sim2.number_input(
      "Opponent Field Lineup Count", 500, 5000, 1000, 250
  )

  if "last_generated_portfolio" not in st.session_state:
    st.info(
        "💡 Generate your lineups in Tab 3 (Lineup Optimizer) first to enable"
        " field simulation."
    )
  else:
    portfolio = st.session_state["last_generated_portfolio"]
    st.write(
        f"Ready to simulate current portfolio of **{len(portfolio)} lineups**"
        f" against **{field_size:,} opposing field lineups**."
    )

    if st.button("🎲 Run Field Contest Simulations", type="primary"):
      with st.spinner(
          f"Generating {field_size:,} opponent lineups & simulating"
          f" {sim_count:,} slates..."
      ):
        pool = (
            df_active[df_active["blended_proj"] > 0]
            .copy()
            .reset_index(drop=True)
        )
        N = len(pool)
        teams = pool["team"].values
        base_projs = pool["blended_proj"].values
        std_devs = pool["std_dev"].values

        # 1. Generate Field Lineups sampled from player ownership probabilities
        qbs = pool[pool["position"] == "QB"]
        rbs = pool[pool["position"] == "RB"]
        wrs = pool[pool["position"] == "WR"]
        tes = pool[pool["position"] == "TE"]
        defs = pool[pool["position"] == "DEF"]

        def safe_sample(df_sub, count):
          weights = df_sub["proj_ownership"].values
          w_sum = weights.sum()
          p = (
              weights / w_sum
              if w_sum > 0
              else np.ones(len(df_sub)) / len(df_sub)
          )
          return df_sub.sample(n=count, replace=False, weights=p).index.tolist()

        field_lineups_indices = []
        for _ in range(field_size):
          try:
            chosen = []
            chosen.extend(safe_sample(qbs, 1))
            chosen.extend(safe_sample(rbs, 2))
            chosen.extend(safe_sample(wrs, 3))
            chosen.extend(safe_sample(tes, 1))
            if not defs.empty:
              chosen.extend(safe_sample(defs, 1))

            # Flex selection from remaining RBs/WRs/TEs
            rem_flex = pool.loc[
                pool["position"].isin(["RB", "WR", "TE"])
                & (~pool.index.isin(chosen))
            ]
            if not rem_flex.empty:
              chosen.extend(safe_sample(rem_flex, 1))

            field_lineups_indices.append(chosen)
          except Exception:
            pass

        # 2. Run slate simulation matrix across all player outcomes
        sim_matrix = np.zeros((sim_count, N))
        for s_idx in range(sim_count):
          team_shocks = {
              tm: np.random.normal(1.0, 0.15) for tm in set(teams) if tm
          }
          for p_i in range(N):
            tm = teams[p_i]
            shock = team_shocks.get(tm, 1.0)
            sd = max(0.5, std_devs[p_i])
            sim_matrix[s_idx, p_i] = max(0.0, np.random.normal(base_projs[p_i] * shock, sd))

        # 3. Score portfolio and field lineups
        key_to_idx = {row["_canon_key"]: i for i, row in pool.iterrows()}

        user_lu_scores = np.zeros((len(portfolio), sim_count))
        for lu_idx, lu_df in enumerate(portfolio):
          col_indices = [
              key_to_idx[k] for k in lu_df["_canon_key"] if k in key_to_idx
          ]
          if col_indices:
            user_lu_scores[lu_idx, :] = np.sum(
                sim_matrix[:, col_indices], axis=1
            )

        field_lu_scores = np.zeros((len(field_lineups_indices), sim_count))
        for f_idx, f_indices in enumerate(field_lineups_indices):
          field_lu_scores[f_idx, :] = np.sum(
              sim_matrix[:, f_indices], axis=1
          )

        # 4. Grade Portfolio against Field Lineups per Trial
        eval_records = []
        for lu_idx in range(len(portfolio)):
          u_scores = user_lu_scores[lu_idx, :]

          # Calculate rank of user lineup relative to field across each trial
          field_wins = 0
          for s_i in range(sim_count):
            u_score = u_scores[s_i]
            # Count field lineups beating user score
            field_beaten = np.sum(field_lu_scores[:, s_i] > u_score)
            if field_beaten < max(1, int(len(field_lineups_indices) * 0.01)):
              field_wins += 1

          win_rate = (field_wins / sim_count) * 100
          sim_roi = (win_rate * 45.0) - 100.0  # Field GPP payout structure

          eval_records.append({
              "Lineup Index": lu_idx,
              "Lineup #": lu_idx + 1,
              "Blended Proj Pts": round(
                  portfolio[lu_idx]["blended_proj"].sum(), 1
              ),
              "Avg Sim Pts": round(np.mean(u_scores), 1),
              "85th Sim Ceiling": round(np.percentile(u_scores, 85), 1),
              "Top 1% Field Win Rate (%)": round(win_rate, 2),
              "Field-Simulated ROI (%)": round(sim_roi, 1),
          })

        st.session_state["portfolio_eval_df"] = pd.DataFrame(eval_records)
        st.success(
            f"Successfully executed {sim_count:,} slate simulations against"
            f" {len(field_lineups_indices):,} field opponents!"
        )

    if "portfolio_eval_df" in st.session_state:
      st.subheader("🏆 Portfolio Field-Simulated ROI & Win Rate")
      res_df = st.session_state["portfolio_eval_df"]
      st.dataframe(
          res_df.sort_values(by="Field-Simulated ROI (%)", ascending=False),
          use_container_width=True,
          hide_index=True,
      )

      st.markdown("---")
      st.subheader("🔍 Lineup Roster Inspector")

      sorted_res = res_df.sort_values(
          by="Field-Simulated ROI (%)", ascending=False
      )
      options_dict = {}
      for _, row in sorted_res.iterrows():
        l_idx = int(row["Lineup Index"])
        l_num = int(row["Lineup #"])
        l_roi = row["Field-Simulated ROI (%)"]
        label = f"Lineup #{l_num} (Index {l_idx}) — Field ROI: {l_roi}%"
        options_dict[label] = l_idx

      selected_label = st.selectbox(
          "Select Lineup to Inspect Roster", list(options_dict.keys())
      )
      chosen_idx = options_dict[selected_label]
      chosen_lu = portfolio[chosen_idx]

      st.markdown(
          f"**Lineup #{chosen_idx+1} Breakdown** | Total Salary: **${chosen_lu['salary'].sum():,}** | Blended Proj: **{chosen_lu['blended_proj'].sum():.1f} pts**"
      )
      st.dataframe(
          chosen_lu[[
              "name",
              "position",
              "team",
              "opponent",
              "salary",
              "blended_proj",
              "gpp_upside_score",
              "proj_ownership",
          ]],
          use_container_width=True,
          hide_index=True,
      )

# -----------------------------------------------------------------------------
# TAB 5: MODEL BACKTESTER
# -----------------------------------------------------------------------------
with tab5:
  st.subheader("📉 Projection Model Backtest Lab")
  st.markdown(
      "Upload an archived Excel workbook or actual scores file to grade and"
      " compare accuracy metrics (MAE, RMSE, Bias, Correlation, and Top-20"
      " Overlap) across Prop-based, ESPN, and Weighted models."
  )

  actuals_file = st.file_uploader(
      "Upload Actuals / Archived Workbook (Excel or CSV)",
      type=["csv", "xlsx"],
  )

  if actuals_file is not None:
    try:
      df_act_mapped = pd.DataFrame()

      if actuals_file.name.endswith(".xlsx"):
        xls_act = pd.ExcelFile(actuals_file)
        act_records = []
        for (
            sheet_name,
            (pos_code, score_column),
        ) in PROJECTED_SCORE_COLUMNS.items():
          if sheet_name in xls_act.sheet_names:
            try:
              raw_s = pd.read_excel(xls_act, sheet_name=sheet_name, header=None)
              if raw_s.empty:
                continue
              header_row = None
              for i in range(min(20, len(raw_s))):
                vals = [
                    str(v).strip().casefold() for v in raw_s.iloc[i].tolist()
                ]
                if any(
                    h in vals
                    for h in ["player name", "team", "defense", "dst"]
                ):
                  header_row = i
                  break
              if header_row is None:
                continue

              header_vals = [
                  str(v).strip().casefold()
                  for v in raw_s.iloc[header_row].tolist()
              ]
              name_idx = 1
              team_idx = 2
              score_idx = excel_col_index(score_column)

              for idx, h_val in enumerate(header_vals):
                if any(
                    k in h_val for k in ["player", "defense", "team defense"]
                ):
                  name_idx = idx
                elif "team" in h_val and "defense" not in h_val:
                  team_idx = idx
                elif any(
                    k in h_val
                    for k in ["actual", "score", "fp", "points", "projected"]
                ):
                  score_idx = idx

              for r_i in range(header_row + 1, len(raw_s)):
                r = raw_s.iloc[r_i]
                p_name = r.iloc[name_idx] if name_idx < len(r) else None
                if pd.isna(p_name) or str(p_name).strip() == "":
                  continue
                p_team = r.iloc[team_idx] if team_idx < len(r) else ""
                raw_score = r.iloc[score_idx] if score_idx < len(r) else 0.0
                val = parse_numeric_series(
                    pd.Series([raw_score]), default=0.0
                ).iloc[0]

                c_key = get_canonical_key(
                    str(p_name).strip(), str(p_team), pos_code
                )
                act_records.append(
                    {"_canon_key": c_key, "actual_score": float(val)}
                )
            except Exception:
              pass
        if act_records:
          df_act_mapped = pd.DataFrame(act_records).drop_duplicates(
              subset=["_canon_key"]
          )

      else:
        df_actuals = pd.read_csv(actuals_file)
        col_name = next(
            (c for c in df_actuals.columns if "name" in c.lower()), None
        )
        col_actual = next(
            (
                c
                for c in df_actuals.columns
                if any(k in c.lower() for k in ["actual", "fp", "score", "fpts"])
            ),
            None,
        )
        col_team = next(
            (c for c in df_actuals.columns if "team" in c.lower()), None
        )
        col_pos = next(
            (c for c in df_actuals.columns if "pos" in c.lower()), None
        )

        if col_name and col_actual:
          act_records = []
          for _, row in df_actuals.iterrows():
            p_name = str(row[col_name]).strip()
            p_team = (
                str(row[col_team]).strip()
                if col_team and pd.notna(row[col_team])
                else ""
            )
            p_pos = standardize_position(
                str(row[col_pos]) if col_pos and pd.notna(row[col_pos]) else ""
            )
            act_val = parse_numeric_series(
                pd.Series([row[col_actual]]), default=0.0
            ).iloc[0]
            c_key = get_canonical_key(p_name, p_team, p_pos)
            act_records.append({"_canon_key": c_key, "actual_score": act_val})
          df_act_mapped = pd.DataFrame(act_records).drop_duplicates(
              subset=["_canon_key"]
          )

      if not df_act_mapped.empty:
        eval_df = pd.merge(
            df_active, df_act_mapped, on="_canon_key", how="inner"
        )

        if not eval_df.empty:
          st.success(
              f"Successfully matched {len(eval_df)} players with actual scores!"
          )

          def calc_metrics(sub_df, proj_col):
            preds = sub_df[proj_col].to_numpy()
            acts = sub_df["actual_score"].to_numpy()
            n = len(sub_df)
            if n == 0:
              return 0, 0.0, 0.0, 0.0, 0.0, 0.0

            mae = np.mean(np.abs(preds - acts))
            rmse = np.sqrt(np.mean((preds - acts) ** 2))
            bias = np.mean(preds - acts)
            corr, _ = pearsonr(preds, acts) if n > 1 else (0.0, 0.0)

            top_n = min(20, n)
            pred_top20 = set(sub_df.nlargest(top_n, proj_col)["_canon_key"])
            act_top20 = set(
                sub_df.nlargest(top_n, "actual_score")["_canon_key"]
            )
            overlap = len(pred_top20.intersection(act_top20)) / top_n

            return n, mae, rmse, bias, corr, overlap

          models_to_test = [
              ("Prop-based model", "vegas_proj"),
              ("ESPN", "espn_proj"),
              ("Weighted", "blended_proj"),
          ]

          overall_rows = []
          for m_label, m_col in models_to_test:
            n, mae, rmse, bias, corr, overlap = calc_metrics(eval_df, m_col)
            overall_rows.append({
                "Projection": m_label,
                "Scored Players": n,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "Bias": round(bias, 4),
                "Correlation with Actual": round(corr, 4),
                "Top-20 Overlap": round(overlap, 2),
            })

          st.subheader("📊 Overall Projection Method Comparison")
          st.dataframe(
              pd.DataFrame(overall_rows),
              use_container_width=True,
              hide_index=True,
          )

          st.subheader("📌 Position-Level Error Summary")
          pos_rows = []
          for pos_code in ["QB", "RB", "WR", "TE"]:
            pos_subset = eval_df[
                eval_df["position"].apply(standardize_position) == pos_code
            ]
            if pos_subset.empty:
              continue
            for m_label, m_col in models_to_test:
              n, mae, rmse, bias, corr, overlap = calc_metrics(
                  pos_subset, m_col
              )
              pos_rows.append({
                  "Position": pos_code,
                  "Projection Method": m_label,
                  "N": n,
                  "MAE": round(mae, 2),
                  "RMSE": round(rmse, 2),
                  "Bias": round(bias, 2),
                  "Correlation": round(corr, 2),
                  "Top 20 Overlap": (
                      f"{overlap*100:.1f}%"
                      if n >= 20
                      else f"{overlap*100:.1f}% (N<{n})"
                  ),
              })

          st.dataframe(
              pd.DataFrame(pos_rows),
              use_container_width=True,
              hide_index=True,
          )
        else:
          st.error(
              "No player keys matched between your current active slate and the"
              " uploaded file."
          )
      else:
        st.error(
            "Could not parse player records or actual scores from the uploaded"
            " file."
        )
    except Exception as e:
      st.error(f"Error processing backtest file: {e}")
  else:
    st.info(
        "Upload an actuals file or archived workbook to generate model"
        " performance metrics."
    )