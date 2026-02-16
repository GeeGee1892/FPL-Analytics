"""
FPL Assistant - Constants Module

All constant values, lookup tables, team mappings, and utility functions
extracted from main.py for modularity.
"""

import math
from datetime import datetime
from typing import Optional, List, Dict

from fpl_assistant.config import MODEL_CONFIG

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_stats = None


# =============================================================================
# SET PIECE SHARE DATA - v4.3.2
# Sources:
# - Penalties: https://www.fantasyfootballpundit.com/premier-league-penalty-takers/
# - FKs/Corners: https://www.benchboost.com/premier-league-set-piece-takers/
#
# Logic: Penalties (~0.76 xG each) and direct FKs are fixture-neutral.
# A penalty vs Liverpool (A) is the same xG as vs Southampton (H).
# This portion of a player's xG should NOT be suppressed by hard fixture multipliers.
#
# set_piece_share = (pen_xG90 + fk_xG90) / total_xG90
# For nailed 1st choice pen taker: ~0.08 xG90 from pens
# For 2nd choice: ~0.02-0.03 xG90 (only takes if 1st choice off/missing)
# For primary FK taker: ~0.025 xG90 from direct FKs
# =============================================================================

SET_PIECE_SHARES = {
    # ============ CONFIRMED 1ST CHOICE PEN TAKERS ============

    # Man United - Bruno Fernandes (1st pens + primary FKs)
    "Fernandes": 0.50,
    "Bruno Fernandes": 0.50,

    # Sunderland - Le Fée (1st pens + FKs, low xG90 base)
    "Le Fée": 0.55,

    # Bournemouth - Tavernier (1st pens + FKs)
    "Tavernier": 0.35,

    # Newcastle - Bruno Guimarães (1st pens + FKs)
    "Bruno Guimarães": 0.30,

    # Everton - Ndiaye (1st pens, lower xG90)
    "Ndiaye": 0.30,

    # Forest - Gibbs-White (1st pens + FKs)
    "Gibbs-White": 0.30,
    "Morgan Gibbs-White": 0.30,

    # Brentford - Igor Thiago (1st pens)
    "Thiago": 0.22,
    "Igor Thiago": 0.22,

    # Fulham - Jiménez (1st pens + FKs)
    "Jiménez": 0.22,
    "Raul Jiménez": 0.22,

    # West Ham - Bowen (1st pens + FKs)
    "Bowen": 0.20,

    # Liverpool - Salah (1st pens)
    "Salah": 0.19,

    # Chelsea - Palmer (1st pens)
    "Palmer": 0.18,

    # Leeds - Calvert-Lewin (1st pens)
    "Calvert-Lewin": 0.18,

    # Aston Villa - Watkins (1st pens)
    "Watkins": 0.18,

    # Arsenal - Saka (1st pens + FKs)
    "Saka": 0.18,

    # Crystal Palace - Mateta (1st pens)
    "Mateta": 0.15,

    # Spurs - Solanke (1st pens)
    "Solanke": 0.15,

    # Wolves - Strand Larsen (1st pens)
    "Strand Larsen": 0.15,

    # Man City - Haaland (1st pens, massive xG90 dilutes)
    "Haaland": 0.10,

    # ============ OTHER SET PIECE TAKERS ============

    # Arsenal
    "Gyökeres": 0.08,       # 2nd pens, high open play
    "Rice": 0.10,           # Primary FKs
    "Ødegaard": 0.05,       # 3rd pens

    # Aston Villa
    "Tielemans": 0.08,      # 2nd pens
    "Rogers": 0.08,         # FKs

    # Bournemouth
    "Semenyo": 0.05,        # 2nd pens
    "Evanilson": 0.05,      # 3rd pens

    # Brentford
    "Mbeumo": 0.10,         # 2nd pens + FKs
    "Schade": 0.05,         # 3rd pens

    # Brighton
    "Milner": 0.50,         # Pens when on, very low xG90 base
    "Welbeck": 0.08,        # 2nd pens + FKs

    # Burnley
    "Edwards": 0.20,        # 1st pens
    "Flemming": 0.12,       # 2nd pens

    # Chelsea
    "Pedro": 0.05,          # 2nd pens
    "Enzo": 0.15,           # Primary FKs

    # Crystal Palace
    "Eze": 0.12,            # 2nd pens + FKs

    # Fulham
    "Wilson": 0.10,         # FKs

    # Leeds
    "Nmecha": 0.08,         # 2nd pens
    "Struijk": 0.10,        # 3rd pens, DEF

    # Liverpool
    "Gakpo": 0.05,          # 2nd pens
    "Mac Allister": 0.03,   # 3rd pens
    "Wirtz": 0.0,           # Corners only

    # Man City
    "Marmoush": 0.05,       # 2nd pens
    "Foden": 0.08,          # Primary FKs

    # Newcastle
    "Isak": 0.08,           # 2nd pens
    "Gordon": 0.05,         # 3rd pens

    # Forest
    "Wood": 0.10,           # 2nd pens
    "Anderson": 0.10,       # FKs

    # Sunderland
    "Roberts": 0.10,        # 2nd pens
    "O'Nien": 0.08,         # 3rd pens

    # Spurs
    "Kudus": 0.10,          # 2nd pens + FKs
    "Son": 0.05,            # 3rd pens

    # West Ham
    "Füllkrug": 0.10,       # 2nd pens
    "Ward-Prowse": 0.20,    # 3rd pens + primary FKs
    "Paquetá": 0.10,        # 4th pens + FKs
}


def get_set_piece_share(player_name: str) -> float:
    """
    Get the set piece share for a player.
    Returns 0.0 for players not on set pieces (full FDR multiplier applies).
    """
    # Try exact match first
    if player_name in SET_PIECE_SHARES:
        return SET_PIECE_SHARES[player_name]

    # Try partial match (for web names like "Bruno" vs "Bruno Fernandes")
    name_lower = player_name.lower()
    for key, value in SET_PIECE_SHARES.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return value

    return 0.0  # Default: no set piece duties


def apply_set_piece_share_to_multiplier(base_mult: float, player_name: str) -> float:
    """
    Adjust attack multiplier to account for set piece share.

    Set pieces (penalties, direct FKs) are fixture-neutral - the xG is the same
    regardless of opponent. Only open play xG should be affected by FDR.

    Formula: effective_mult = sp_share * 1.0 + (1 - sp_share) * base_mult

    Example: Bruno vs Arsenal (A) with base_mult 0.68
    - sp_share = 0.50 (half his xG from pens + FKs)
    - effective = 0.50 * 1.0 + 0.50 * 0.68 = 0.84
    """
    sp_share = get_set_piece_share(player_name)
    if sp_share <= 0:
        return base_mult

    # Set piece portion gets neutral multiplier (1.0)
    # Open play portion gets full FDR multiplier
    return sp_share * 1.0 + (1 - sp_share) * base_mult


def get_current_season() -> str:
    """
    Derive the current FPL season dynamically.
    FPL season runs Aug-May, so:
    - Before August: previous year's season (e.g., Jan 2025 -> "2024")
    - August onwards: current year's season (e.g., Sep 2025 -> "2025")
    """
    now = datetime.now()
    if now.month < 8:  # Before August
        return str(now.year - 1)
    return str(now.year)


def percentileofscore(data: List[float], score: float) -> float:
    """
    Calculate percentile rank of a score relative to a list of scores.
    Uses scipy if available, otherwise falls back to simple implementation.
    """
    if HAS_SCIPY and scipy_stats:
        return scipy_stats.percentileofscore(data, score)

    # Simple fallback implementation
    if not data:
        return 50.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    count_below = sum(1 for x in sorted_data if x < score)
    count_equal = sum(1 for x in sorted_data if x == score)
    return 100.0 * (count_below + 0.5 * count_equal) / n


# ============ CONSTANTS ============

FPL_BASE_URL = "https://fantasy.premierleague.com/api"
MIN_MINUTES_DEFAULT = 400
POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_ID_MAP = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}

# FDR multipliers now come from config (with price-based dampening)
# These are the BASE multipliers - actual multipliers depend on player price
FDR_MULTIPLIERS_10 = MODEL_CONFIG["fdr"].fdr_multipliers

# Fallback: Map old FPL 1-5 scale to our 1-10 scale
FPL_FDR_TO_10 = {1: 2, 2: 4, 3: 5, 4: 7, 5: 9}

# DEFCON thresholds from config
DEFCON_THRESHOLD_DEF = MODEL_CONFIG["xpts"].defcon_threshold_def
DEFCON_THRESHOLD_MID = MODEL_CONFIG["xpts"].defcon_threshold_mid
DEFCON_POINTS = MODEL_CONFIG["xpts"].defcon_points

# Fixture importance weights from exponential decay
def _generate_fixture_weights(half_life: float, n: int) -> List[float]:
    """Generate fixture weights using exponential decay. Normalized to sum to 1.0."""
    decay = math.log(2) / half_life
    raw = [math.exp(-decay * i) for i in range(n)]
    total = sum(raw)
    return [round(w / total, 4) for w in raw]

FIXTURE_WEIGHTS = _generate_fixture_weights(
    MODEL_CONFIG["xpts"].fixture_weight_half_life,
    MODEL_CONFIG["xpts"].fixture_weight_max_horizon
)

# European teams for rotation modeling
EUROPEAN_TEAMS_UCL = {"Arsenal", "Aston Villa", "Liverpool", "Man City"}
EUROPEAN_TEAMS_UEL = {"Man Utd", "Tottenham"}
EUROPEAN_TEAMS_UECL = {"Chelsea"}

# Manager rotation tendencies
ROTATION_MANAGERS = {"Man City": 0.85, "Chelsea": 0.88, "Brighton": 0.90, "Newcastle": 0.92}
STABLE_MANAGERS = {"Arsenal": 1.0, "Liverpool": 0.98, "Fulham": 1.0, "Bournemouth": 1.0, "Brentford": 0.98}

# League averages from config
LEAGUE_AVG_GOALS_PER_GAME = MODEL_CONFIG["clean_sheet"].league_avg_goals_per_game
LEAGUE_AVG_CS_RATE = MODEL_CONFIG["clean_sheet"].league_avg_cs_rate

# Home advantage factors from config - v4.3.11 SEPARATED
# OPPONENT venue adjustments (for FDR/matchup - small effect)
OPP_HOME_BOOST = MODEL_CONFIG["home_away"].opp_home_boost        # 1.05
OPP_AWAY_PENALTY = MODEL_CONFIG["home_away"].opp_away_penalty    # 0.95

# PLAYER venue adjustments (for xPts fallback - large effect)
PLAYER_HOME_BOOST = MODEL_CONFIG["home_away"].player_home_boost      # 1.15
PLAYER_AWAY_PENALTY = MODEL_CONFIG["home_away"].player_away_penalty  # 0.85

# DEFENCE venue adjustments (for CS probability)
DEFENCE_HOME_BOOST = MODEL_CONFIG["home_away"].defence_home_boost      # 0.92
DEFENCE_AWAY_PENALTY = MODEL_CONFIG["home_away"].defence_away_penalty  # 1.08

# Legacy aliases for backwards compatibility (use OPP_ versions for FDR)
HOME_ATTACK_BOOST = OPP_HOME_BOOST
AWAY_ATTACK_PENALTY = OPP_AWAY_PENALTY
HOME_DEFENCE_BOOST = DEFENCE_HOME_BOOST
AWAY_DEFENCE_PENALTY = DEFENCE_AWAY_PENALTY

# Yellow/Red card expected points deduction per 90
YELLOW_CARD_RATE_PENALTY = -0.15  # Avg ~0.15 yellows/90 for most players
RED_CARD_RATE_PENALTY = -0.02  # Very rare

# Own goal probability (very low)
OWN_GOAL_PENALTY = -0.02  # ~2% chance per 90 for defenders

# Penalty takers - xG boost from penalties (~0.3-0.4 xG/90 for main takers)
# Format: player_id -> (penalty_xG_per_90, confidence)
# Updated for 2024/25 season - these are MAIN takers
PENALTY_TAKERS = {
    # Liverpool
    351: (0.35, 1.0),   # Salah - nailed
    # Man City
    355: (0.38, 1.0),   # Haaland - nailed
    # Chelsea
    506: (0.32, 0.95),  # Palmer - main taker
    # Arsenal
    19: (0.28, 0.90),   # Saka - shares with others
    # Spurs
    494: (0.30, 0.85),  # Son - main taker
    # Newcastle
    24: (0.28, 0.85),   # Isak - when on pitch
    # Aston Villa
    495: (0.25, 0.80),  # Watkins
    # Brentford
    231: (0.30, 0.90),  # Mbeumo
    # West Ham
    556: (0.25, 0.85),  # Kudus - secondary
    318: (0.28, 0.85),  # Bowen - main
    # Brighton
    582: (0.30, 0.90),  # Joao Pedro
    # Bournemouth
    322: (0.25, 0.85),  # Kluivert
    # Wolves
    627: (0.25, 0.90),  # Cunha
    # Fulham
    326: (0.25, 0.85),  # Jimenez
    # Crystal Palace
    420: (0.28, 0.85),  # Eze
}

# Horizon regression from config
HORIZON_REGRESSION = MODEL_CONFIG["xpts"].horizon_regression

# v4.3.4: Default team strength data for ALL teams
# Used as baseline when no manual Analytic FPL data is available
# Values from Analytic FPL 25/26 season data (your screenshot)
# attack_fdr = how hard to score against them (based on their xGA)
# defence_fdr = how hard to keep CS against them (based on their xG)
DEFAULT_TEAM_STRENGTH = {
    # v4.3.7 FIXED: Correct 25/26 FPL API Team IDs
    # Team IDs shifted due to Burnley (3), Leeds (11), Sunderland (17) promotion
    # Verified from /api/bootstrap-static/
    # Sorted by defensive strength (xGA low to high)
    1:  {"name": "Arsenal", "xg": 1.84, "xga": 0.81},       # ARS
    12: {"name": "Liverpool", "xg": 1.78, "xga": 1.06},     # LIV
    13: {"name": "Man City", "xg": 1.98, "xga": 1.13},      # MCI
    15: {"name": "Newcastle", "xg": 1.48, "xga": 1.14},     # NEW
    2:  {"name": "Aston Villa", "xg": 1.41, "xga": 1.22},   # AVL
    14: {"name": "Man Utd", "xg": 1.48, "xga": 1.24},       # MUN
    8:  {"name": "Crystal Palace", "xg": 1.41, "xga": 1.33},# CRY
    6:  {"name": "Brighton", "xg": 1.37, "xga": 1.33},      # BHA - ID 6 (was 5)
    5:  {"name": "Brentford", "xg": 1.41, "xga": 1.35},     # BRE - ID 5 (was 4)
    9:  {"name": "Everton", "xg": 1.03, "xga": 1.38},       # EVE - ID 9 (was 8)
    10: {"name": "Fulham", "xg": 1.24, "xga": 1.39},        # FUL - ID 10 (was 9)
    4:  {"name": "Bournemouth", "xg": 1.24, "xga": 1.40},   # BOU - ID 4 (was 3)
    20: {"name": "Wolves", "xg": 1.08, "xga": 1.43},        # WOL
    17: {"name": "Sunderland", "xg": 0.82, "xga": 1.43},    # SUN (promoted) - ID 17
    7:  {"name": "Chelsea", "xg": 1.54, "xga": 1.44},       # CHE - ID 7 (was 6)
    19: {"name": "West Ham", "xg": 1.22, "xga": 1.45},      # WHU
    16: {"name": "Nott'm Forest", "xg": 1.16, "xga": 1.48}, # NFO
    18: {"name": "Tottenham", "xg": 1.44, "xga": 1.63},     # TOT
    11: {"name": "Leeds United", "xg": 1.15, "xga": 1.70},  # LEE (promoted) - ID 11
    3:  {"name": "Burnley", "xg": 0.84, "xga": 1.82},       # BUR (promoted) - ID 3
}

# Legacy alias for backwards compatibility - also with correct 25/26 IDs
PROMOTED_TEAM_DEFAULTS = {
    3:  {"name": "Burnley", "xg": 0.84, "xga": 1.82, "attack_fdr": 2, "defence_fdr": 1},       # BUR - ID 3
    11: {"name": "Leeds United", "xg": 1.15, "xga": 1.70, "attack_fdr": 5, "defence_fdr": 3}, # LEE - ID 11
    17: {"name": "Sunderland", "xg": 0.82, "xga": 1.43, "attack_fdr": 4, "defence_fdr": 1},   # SUN - ID 17
}

# v4.3.3c: League average constants for matchup calculations
LEAGUE_AVG_XG = 1.35   # Average team xG per game
LEAGUE_AVG_XGA = 1.35  # Average team xGA per game (same, by definition)

# Admin manager ID - overrides from this account become global
ADMIN_MANAGER_ID = 616495  # Gerti's manager ID for global overrides

# Base CS probability by team tier (used when FDR data unavailable)
# Based on 2024/25 data - top teams keep ~35% CS, bottom ~15%
TEAM_BASE_CS_RATES = {
    # 25/26 Season - Based on Analytic FPL adjusted xGA
    # CS rate = e^(-xGA) approximately
    "Arsenal": 0.44, "Liverpool": 0.35, "Man City": 0.32, "Chelsea": 0.26,
    "Aston Villa": 0.30, "Newcastle": 0.32, "Brighton": 0.26, "Tottenham": 0.24,
    "Man Utd": 0.29, "Fulham": 0.25, "Bournemouth": 0.23, "West Ham": 0.20,
    "Crystal Palace": 0.24, "Brentford": 0.25, "Wolves": 0.24, "Everton": 0.26,
    "Nott'm Forest": 0.23,
    # Promoted teams 25/26
    "Leeds United": 0.25, "Leeds": 0.25,
    "Sunderland": 0.23,
    "Burnley": 0.16,
}

# Base attack strength by team (xG per game) - 25/26 Season
TEAM_BASE_XG = {
    # From Analytic FPL adjusted xG
    "Man City": 1.98, "Liverpool": 1.78, "Arsenal": 1.84, "Chelsea": 1.54,
    "Tottenham": 1.44, "Newcastle": 1.48, "Aston Villa": 1.41, "Brighton": 1.37,
    "Man Utd": 1.48, "West Ham": 1.22, "Bournemouth": 1.24, "Brentford": 1.41,
    "Fulham": 1.24, "Crystal Palace": 1.41, "Wolves": 1.08, "Everton": 1.03,
    "Nott'm Forest": 1.16,
    # Promoted teams 25/26
    "Leeds United": 1.15, "Leeds": 1.15,
    "Sunderland": 0.82,
    "Burnley": 0.84,
}

# Team name mappings: FPL short_name -> (fpl_id, understat_name)
# v4.3.7 FIXED: Updated for 25/26 FPL API team IDs
# Previous mapping was from 24/25 and caused display issues
TEAM_NAME_MAPPING = {
    "ARS": (1, "Arsenal"),
    "AVL": (2, "Aston Villa"),
    "BUR": (3, "Burnley"),            # Promoted 25/26 - ID 3
    "BOU": (4, "Bournemouth"),        # ID 4 (was 3 in 24/25)
    "BRE": (5, "Brentford"),          # ID 5 (was 4 in 24/25)
    "BHA": (6, "Brighton"),           # ID 6 (was 5 in 24/25)
    "CHE": (7, "Chelsea"),            # ID 7 (was 6 in 24/25)
    "CRY": (8, "Crystal Palace"),     # ID 8 (was 7 in 24/25)
    "EVE": (9, "Everton"),            # ID 9 (was 8 in 24/25)
    "FUL": (10, "Fulham"),            # ID 10 (was 9 in 24/25)
    "LEE": (11, "Leeds United"),      # Promoted 25/26 - ID 11
    "LIV": (12, "Liverpool"),
    "MCI": (13, "Manchester City"),
    "MUN": (14, "Manchester United"),
    "NEW": (15, "Newcastle United"),
    "NFO": (16, "Nottingham Forest"),
    "SUN": (17, "Sunderland"),        # Promoted 25/26 - ID 17
    "TOT": (18, "Tottenham"),
    "WHU": (19, "West Ham"),
    "WOL": (20, "Wolverhampton Wanderers"),
}

UNDERSTAT_TO_FPL_ID = {v[1]: v[0] for v in TEAM_NAME_MAPPING.values()}
FPL_ID_TO_UNDERSTAT = {v[0]: v[1] for v in TEAM_NAME_MAPPING.values()}

# Analytic FPL team names to FPL ID mapping (handles various name formats)
# v4.3.7 FIXED: Updated for 25/26 season - Burnley (3), Leeds (11), Sunderland (17) promoted
ANALYTIC_FPL_TO_ID = {
    "Arsenal": 1, "Aston Villa": 2,
    "Burnley": 3,  # Promoted 25/26 - ID 3
    "Bournemouth": 4,  # ID 4 (was 3 in 24/25)
    "Brentford": 5,    # ID 5 (was 4 in 24/25)
    "Brighton": 6, "Brighton & Hove Albion": 6, "Brighton and Hove Albion": 6,  # ID 6 (was 5)
    "Chelsea": 7,      # ID 7 (was 6)
    "Crystal Palace": 8,  # ID 8 (was 7)
    "Everton": 9,      # ID 9 (was 8)
    "Fulham": 10,      # ID 10 (was 9)
    "Leeds United": 11, "Leeds": 11,  # Promoted 25/26 - ID 11
    "Liverpool": 12,
    "Manchester City": 13, "Man City": 13,
    "Manchester Utd": 14, "Manchester United": 14, "Man Utd": 14, "Man United": 14,
    "Newcastle Utd": 15, "Newcastle United": 15, "Newcastle": 15,
    "Nott'ham Forest": 16, "Nottingham Forest": 16, "Nott'm Forest": 16, "Forest": 16,
    "Sunderland": 17,  # Promoted 25/26 - ID 17
    "Tottenham": 18, "Spurs": 18, "Tottenham Hotspur": 18,
    "West Ham": 19, "West Ham United": 19,
    "Wolves": 20, "Wolverhampton": 20, "Wolverhampton Wanderers": 20,
}

HOME_AWAY_FDR_ADJUSTMENT = {'home': 0.88, 'away': 1.08}
