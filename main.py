"""
FPL Assistant Backend - FastAPI v3.2
Enhanced with:
- Composite FDR model (1-10 scale) using Understat xG data
- Position-specific FDR: attack_fdr for FWD/MID, defence_fdr for DEF/GKP
- DEFCON per 90 for DEF/MID
- Saves per 90 for GKP
- Expected Minutes with override capability
- Shared HTTP client for connection pooling
- Dynamic season detection
- Rate limiting with exponential backoff
"""

import asyncio
import re
import json
import httpx
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import math
from collections import defaultdict
from datetime import datetime
import os
import logging
import random

# scipy is optional - graceful fallback if not available
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_stats = None

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL CONFIGURATION - All calibration constants with documentation
# =============================================================================

@dataclass
class FDRConfig:
    """
    Fixture Difficulty Rating configuration.
    FDR 1-10 scale where 10 = hardest fixture.
    
    Key insight: Premium players (£10m+) are less affected by fixtures
    than budget players because skill overcomes difficulty.
    """
    
    # FDR multipliers - RECALIBRATED: less extreme than before
    fdr_multipliers: Dict[int, float] = field(default_factory=lambda: {
        1: 1.18,   # Very easy - capped because variance exists
        2: 1.12,
        3: 1.06,
        4: 1.02,
        5: 1.00,   # Neutral baseline
        6: 0.97,
        7: 0.93,
        8: 0.88,
        9: 0.82,
        10: 0.75,  # Very hard
    })
    
    # Price-based FDR dampening
    # Formula: effective_mult = 1 + (base_mult - 1) * dampening
    premium_dampening: float = 0.55      # £10m+: 55% of FDR effect
    mid_price_dampening: float = 0.80    # £6-9.9m: 80% of FDR effect
    budget_dampening: float = 1.0        # Below £6m: full effect
    
    premium_price_threshold: float = 10.0
    mid_price_threshold: float = 6.0
    
    # Home/away FDR adjustment
    home_fdr_multiplier: float = 0.88
    away_fdr_multiplier: float = 1.12


@dataclass  
class HomeAwayConfig:
    """Home/Away performance adjustments from PL data analysis."""
    
    home_attack_boost: float = 1.19      # Teams score ~19% more at home
    away_attack_penalty: float = 0.87    # Teams score ~13% less away
    home_defence_boost: float = 1.15     # Teams concede ~15% less at home
    away_defence_penalty: float = 0.90   # Teams concede ~10% more away
    
    # Player-level home/away variance
    min_games_for_split: int = 8         # Min games to trust split
    split_weight: float = 0.60           # 60% split-specific, 40% overall


@dataclass
class XptsConfig:
    """Expected points calculation configuration."""
    
    goal_points: Dict[int, int] = field(default_factory=lambda: {
        1: 6, 2: 6, 3: 5, 4: 4  # GKP, DEF, MID, FWD
    })
    
    cs_points: Dict[int, int] = field(default_factory=lambda: {
        1: 4, 2: 4, 3: 1, 4: 0
    })
    
    assist_points: int = 3
    
    # DEFCON thresholds
    defcon_threshold_def: int = 10
    defcon_threshold_mid: int = 12
    defcon_points: float = 2.0
    
    # Yellow card baselines by position (per 90)
    yellow_baseline: Dict[int, float] = field(default_factory=lambda: {
        1: 0.02, 2: 0.12, 3: 0.10, 4: 0.06
    })
    
    # Own goal probability by position (per 90)
    og_probability: Dict[int, float] = field(default_factory=lambda: {
        1: 0.005, 2: 0.020, 3: 0.008, 4: 0.003
    })
    
    shots_on_target_per_xg: float = 2.8
    league_avg_save_rate: float = 0.70
    
    # Fixture weighting (sum = 1.0)
    fixture_weights: List[float] = field(default_factory=lambda: [
        0.30, 0.22, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02
    ])
    
    # Horizon regression
    horizon_regression: Dict[int, float] = field(default_factory=lambda: {
        1: 1.00, 2: 0.99, 3: 0.98, 4: 0.97,
        5: 0.96, 6: 0.95, 7: 0.94, 8: 0.93
    })


@dataclass
class VarianceConfig:
    """Ceiling/floor variance configuration."""
    
    ceiling_z: float = 0.84              # 80th percentile
    floor_z: float = -0.84               # 20th percentile
    
    # Position-based baseline stdev (per 90)
    position_stdev_baseline: Dict[int, float] = field(default_factory=lambda: {
        1: 2.1,    # GKP - narrow
        2: 2.4,    # DEF
        3: 3.2,    # MID
        4: 3.5,    # FWD - widest
    })
    
    min_games_for_stdev: int = 10
    
    # Fixture ceiling boosts
    fixture_ceiling_boost: Dict[int, float] = field(default_factory=lambda: {
        1: 1.25, 2: 1.18, 3: 1.10, 4: 1.05, 5: 1.00,
        6: 0.97, 7: 0.93, 8: 0.88, 9: 0.82, 10: 0.75
    })
    
    home_ceiling_boost: float = 1.08
    away_ceiling_penalty: float = 0.94


@dataclass
class BonusConfig:
    """Bonus point prediction configuration."""
    
    bps_per_goal: Dict[int, int] = field(default_factory=lambda: {
        1: 6, 2: 12, 3: 24, 4: 24
    })
    bps_per_assist: int = 9
    bps_per_cs: Dict[int, int] = field(default_factory=lambda: {
        1: 12, 2: 12, 3: 6, 4: 0
    })
    base_bps_by_position: Dict[int, int] = field(default_factory=lambda: {
        1: 10, 2: 11, 3: 9, 4: 6
    })
    
    # BPS to bonus conversion
    bps_to_bonus: List[Tuple[int, float]] = field(default_factory=lambda: [
        (40, 2.7), (35, 2.3), (30, 1.9), (26, 1.5),
        (22, 1.1), (18, 0.7), (14, 0.4), (0, 0.15)
    ])
    
    teammate_dilution: float = 0.70


@dataclass
class CleanSheetConfig:
    """Clean sheet probability configuration."""
    
    cs_prob_min: float = 0.03
    cs_prob_max: float = 0.55
    league_avg_goals_per_game: float = 1.35
    league_avg_cs_rate: float = 0.27


@dataclass
class CaptainConfig:
    """Captain selection scoring configuration."""
    
    position_ceiling: Dict[int, float] = field(default_factory=lambda: {
        1: 0.70, 2: 0.85, 3: 1.12, 4: 1.18
    })
    
    elite_ppg_threshold: float = 8.0
    elite_ppg_bonus: float = 1.15
    good_ppg_threshold: float = 7.0
    good_ppg_bonus: float = 1.10
    solid_ppg_threshold: float = 6.0
    solid_ppg_bonus: float = 1.05
    poor_ppg_threshold: float = 4.0
    poor_ppg_penalty: float = 0.85
    
    ownership_tiers: List[Tuple[float, float]] = field(default_factory=lambda: [
        (5, 0.08), (10, 0.05), (20, 0.02),
        (35, -0.01), (50, -0.03), (100, -0.05)
    ])


# Initialize global config
MODEL_CONFIG = {
    "fdr": FDRConfig(),
    "home_away": HomeAwayConfig(),
    "xpts": XptsConfig(),
    "variance": VarianceConfig(),
    "bonus": BonusConfig(),
    "clean_sheet": CleanSheetConfig(),
    "captain": CaptainConfig(),
}


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


# Global HTTP client (initialized in lifespan)
http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Get the shared HTTP client, creating one if needed."""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            headers={"User-Agent": "FPL-Assistant/3.2"}
        )
    return http_client


async def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> httpx.Response:
    """
    Fetch URL with exponential backoff retry logic.
    Handles rate limiting (429) and transient errors.
    """
    client = await get_http_client()
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = await client.get(url)
            
            if response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get("Retry-After", base_delay * (2 ** attempt)))
                logger.warning(f"Rate limited on {url}, waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                continue
            
            response.raise_for_status()
            return response
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (500, 502, 503, 504):
                # Server error - retry with backoff
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Server error {e.response.status_code} on {url}, retry in {delay:.1f}s")
                await asyncio.sleep(delay)
                last_error = e
                continue
            raise
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Connection error on {url}, retry in {delay:.1f}s: {e}")
            await asyncio.sleep(delay)
            last_error = e
            continue
    
    if last_error:
        raise last_error
    raise HTTPException(status_code=503, detail="Failed after max retries")


async def load_import_data_from_file():
    """
    Auto-load team strength and fixture xG data from data/import_data.json on startup.
    
    Looks for the file in:
    1. ./data/import_data.json (relative to working dir)
    2. /app/data/import_data.json (Docker/production)
    3. ../data/import_data.json (development)
    """
    possible_paths = [
        Path("data/import_data.json"),
        Path("/app/data/import_data.json"),
        Path(__file__).parent / "data" / "import_data.json",
        Path(__file__).parent.parent / "data" / "import_data.json",
    ]
    
    import_file = None
    for p in possible_paths:
        if p.exists():
            import_file = p
            break
    
    if not import_file:
        logger.info("No import_data.json found - skipping auto-load. Looked in: data/import_data.json")
        return
    
    try:
        with open(import_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loading import data from {import_file}")
        
        # Load team strength data
        if "team_strength_import" in data:
            ts_data = data["team_strength_import"]
            teams_list = ts_data.get("teams", [])
            loaded_teams = []
            
            for team_data in teams_list:
                team_name = team_data.get("team", "").strip()
                team_id = ANALYTIC_FPL_TO_ID.get(team_name)
                
                if team_id is None:
                    logger.warning(f"Auto-load: Unknown team name: {team_name}")
                    continue
                
                cache.manual_team_strength[team_id] = {
                    "team_name": team_name,
                    "adjxg_for": team_data.get("adjxg_for", 1.5),
                    "adjxg_ag": team_data.get("adjxg_ag", 1.3),
                    "attack_delta": team_data.get("attack_delta", 0.0),
                    "defence_delta": team_data.get("defence_delta", 0.0),
                    "attack_trend": team_data.get("attack_trend", 0.0),
                    "defence_trend": team_data.get("defence_trend", 0.0),
                }
                loaded_teams.append(team_name)
            
            cache.manual_team_strength_last_update = datetime.now()
            logger.info(f"Auto-loaded team strength for {len(loaded_teams)} teams")
        
        # Load fixture xG data (needs FPL data for team name mapping)
        if "fixture_xg_import" in data:
            try:
                fpl_data = await fetch_fpl_data()
                teams_by_name = {}
                for t in fpl_data["teams"]:
                    teams_by_name[t["name"].lower()] = t["id"]
                    teams_by_name[t["short_name"].lower()] = t["id"]
                
                # Add common aliases
                aliases = {
                    "man city": "Manchester City", "man utd": "Manchester United",
                    "spurs": "Tottenham", "wolves": "Wolverhampton Wanderers",
                    "brighton": "Brighton and Hove Albion", "forest": "Nottingham Forest",
                    "nott'm forest": "Nottingham Forest", "nottingham forest": "Nottingham Forest",
                }
                for alias, full_name in aliases.items():
                    if full_name.lower() in teams_by_name:
                        teams_by_name[alias.lower()] = teams_by_name[full_name.lower()]
                
                # Also use ANALYTIC_FPL_TO_ID
                for name, team_id in ANALYTIC_FPL_TO_ID.items():
                    teams_by_name[name.lower()] = team_id
                
                fx_data = data["fixture_xg_import"]
                fixtures_list = fx_data.get("fixtures", [])
                loaded_fixtures = 0
                
                for fix in fixtures_list:
                    home_name = fix.get("home_team", "").strip().lower()
                    away_name = fix.get("away_team", "").strip().lower()
                    
                    home_id = teams_by_name.get(home_name)
                    away_id = teams_by_name.get(away_name)
                    
                    if home_id is None or away_id is None:
                        logger.warning(f"Auto-load fixture: Unknown team - {fix.get('home_team')} vs {fix.get('away_team')}")
                        continue
                    
                    gw = fix.get("gameweek", 0)
                    key = (home_id, away_id, gw)
                    cache.fixture_xg[key] = {
                        "home_xg": fix.get("home_xg", 1.5),
                        "away_xg": fix.get("away_xg", 1.0),
                        "home_team": fix.get("home_team", ""),
                        "away_team": fix.get("away_team", ""),
                        "gw": gw,
                    }
                    loaded_fixtures += 1
                
                cache.fixture_xg_last_update = datetime.now()
                logger.info(f"Auto-loaded {loaded_fixtures} fixture xG projections")
                
            except Exception as e:
                logger.error(f"Failed to load fixture xG (FPL API may be unavailable): {e}")
        
    except Exception as e:
        logger.error(f"Failed to load import_data.json: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global http_client
    
    # Startup - create shared HTTP client
    http_client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        headers={"User-Agent": "FPL-Assistant/3.2"}
    )
    
    # Auto-load import data from file (team strength + fixture xG)
    try:
        await load_import_data_from_file()
        logger.info("Import data auto-load completed")
    except Exception as e:
        logger.error(f"Import data auto-load failed: {e}")
    
    # Initialize FDR data (will incorporate any loaded team strength)
    try:
        await refresh_fdr_data(force=True)
        logger.info("FDR data initialized on startup")
    except Exception as e:
        logger.error(f"FDR startup refresh failed: {e}")
    
    yield
    
    # Shutdown - close HTTP client
    if http_client:
        await http_client.aclose()
        http_client = None


app = FastAPI(title="FPL Assistant API", version="3.2.0", lifespan=lifespan)

# CORS configuration - fixed for security
# In production, replace "*" with specific origins like ["https://yourdomain.com"]
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,  # Set to True only with specific origins, not "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ HELPER FUNCTIONS ============

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

# Fixture importance weights from config
FIXTURE_WEIGHTS = MODEL_CONFIG["xpts"].fixture_weights

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

# Home advantage factors from config
HOME_ATTACK_BOOST = MODEL_CONFIG["home_away"].home_attack_boost
HOME_DEFENCE_BOOST = MODEL_CONFIG["home_away"].home_defence_boost
AWAY_ATTACK_PENALTY = MODEL_CONFIG["home_away"].away_attack_penalty
AWAY_DEFENCE_PENALTY = MODEL_CONFIG["home_away"].away_defence_penalty

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


def get_price_adjusted_fdr_multiplier(fdr: int, player_price: float) -> float:
    """
    Get FDR multiplier with price-based dampening.
    
    Premium players (£10m+) are less affected by fixtures because skill overcomes difficulty.
    Budget players get full FDR effect.
    
    Args:
        fdr: Fixture difficulty 1-10
        player_price: Player price in millions
        
    Returns:
        Adjusted FDR multiplier
    """
    fdr_config = MODEL_CONFIG["fdr"]
    base_mult = fdr_config.fdr_multipliers.get(fdr, 1.0)
    
    if player_price >= fdr_config.premium_price_threshold:
        dampening = fdr_config.premium_dampening
    elif player_price >= fdr_config.mid_price_threshold:
        dampening = fdr_config.mid_price_dampening
    else:
        dampening = fdr_config.budget_dampening
    
    return 1.0 + (base_mult - 1.0) * dampening


def calculate_goals_conceded_penalty(xGA: float) -> float:
    """
    Calculate expected FPL penalty for goals conceded using Poisson distribution.
    
    FPL rules: -1 per 2 goals conceded (for GKP/DEF playing 60+ mins)
    
    Using Poisson: E[penalty] = Σ P(k goals) × floor(k/2)
    
    This is more accurate than linear approximation because:
    - 0-1 goals = 0 penalty (not -0.5 each)
    - The stepped nature means low xGA is less penalized than linear model suggests
    
    Args:
        xGA: Expected goals against for this fixture
    
    Returns:
        Expected penalty (negative number)
    """
    if xGA <= 0:
        return 0.0
    
    penalty = 0.0
    prob = math.exp(-xGA)  # P(0 goals) - Poisson
    
    # Sum over reasonable range of goals (0-12 covers >99.99% of probability)
    for k in range(13):
        fpl_deduction = k // 2  # floor(k/2) = FPL's -1 per 2 goals rule
        penalty += prob * fpl_deduction
        prob *= xGA / (k + 1)  # P(k+1) = P(k) * λ / (k+1)
    
    return penalty


def detect_backup_status(player: Dict, all_players: List[Dict], current_gw: int) -> tuple[bool, float, str]:
    """
    Detect if a player is a backup by comparing to teammates at same position.
    
    Returns: (is_backup, confidence, reason)
    - is_backup: True if clearly a backup
    - confidence: 0.0-1.0 how confident we are
    - reason: explanation string
    """
    player_id = player.get("id")
    team_id = player.get("team")
    position = player.get("element_type")
    
    total_minutes = int(player.get("minutes", 0) or 0)
    starts = int(player.get("starts", 0) or 0)
    ownership = float(player.get("selected_by_percent", 0) or 0)
    
    # Find teammates at same position
    teammates = [
        p for p in all_players 
        if p.get("team") == team_id 
        and p.get("element_type") == position 
        and p.get("id") != player_id
    ]
    
    if not teammates:
        return False, 0.0, "no_teammates"
    
    # Get the "starter" - teammate with most minutes at this position
    starter = max(teammates, key=lambda p: int(p.get("minutes", 0) or 0))
    starter_mins = int(starter.get("minutes", 0) or 0)
    starter_starts = int(starter.get("starts", 0) or 0)
    starter_ownership = float(starter.get("selected_by_percent", 0) or 0)
    
    available_gws = max(current_gw - 1, 1)
    
    # ==================== GKP SPECIAL CASE ====================
    # GKPs are almost always 1st choice vs backup, very binary
    if position == 1:
        # If starter has 5+ starts and this player has 0-1 starts
        if starter_starts >= 5 and starts <= 1:
            return True, 0.95, f"backup_gkp_to_{starter.get('web_name', '?')}"
        
        # If ownership difference is huge
        if starter_ownership > 5 and ownership < 1:
            return True, 0.85, f"low_ownership_gkp"
        
        # If minutes difference is stark
        if starter_mins > 800 and total_minutes < 100:
            return True, 0.90, f"no_minutes_gkp"
    
    # ==================== OUTFIELD PLAYERS ====================
    else:
        # Calculate start rate for both
        player_start_rate = starts / available_gws if available_gws > 0 else 0
        starter_start_rate = starter_starts / available_gws if available_gws > 0 else 0
        
        # If starter has 3x+ the minutes and decent sample
        if starter_mins > 500 and total_minutes < starter_mins * 0.25:
            return True, 0.80, f"low_minutes_vs_{starter.get('web_name', '?')}"
        
        # If starter has 70%+ start rate and this player has <15%
        if starter_start_rate > 0.70 and player_start_rate < 0.15:
            return True, 0.75, f"low_start_rate"
        
        # If ownership is <1% and a teammate has >10%
        if ownership < 1 and starter_ownership > 10:
            return True, 0.70, f"ownership_suggests_backup"
        
        # Very low minutes with decent GWs played
        if current_gw > 10 and total_minutes < 100 and starts <= 1:
            return True, 0.85, f"minimal_involvement"
    
    return False, 0.0, "appears_starter"


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
TEAM_NAME_MAPPING = {
    "ARS": (1, "Arsenal"),
    "AVL": (2, "Aston Villa"),
    "BOU": (3, "Bournemouth"),
    "BRE": (4, "Brentford"),
    "BHA": (5, "Brighton"),
    "CHE": (6, "Chelsea"),
    "CRY": (7, "Crystal Palace"),
    "EVE": (8, "Everton"),
    "FUL": (9, "Fulham"),
    "BUR": (10, "Burnley"),  # Promoted 25/26
    "LEE": (11, "Leeds United"),  # Promoted 25/26
    "LIV": (12, "Liverpool"),
    "MCI": (13, "Manchester City"),
    "MUN": (14, "Manchester United"),
    "NEW": (15, "Newcastle United"),
    "NFO": (16, "Nottingham Forest"),
    "SUN": (17, "Sunderland"),  # Promoted 25/26
    "TOT": (18, "Tottenham"),
    "WHU": (19, "West Ham"),
    "WOL": (20, "Wolverhampton Wanderers"),
}

UNDERSTAT_TO_FPL_ID = {v[1]: v[0] for v in TEAM_NAME_MAPPING.values()}
FPL_ID_TO_UNDERSTAT = {v[0]: v[1] for v in TEAM_NAME_MAPPING.values()}

# Analytic FPL team names to FPL ID mapping (handles various name formats)
# Updated for 25/26 season: Burnley, Leeds, Sunderland promoted
ANALYTIC_FPL_TO_ID = {
    "Arsenal": 1, "Aston Villa": 2, "Bournemouth": 3, "Brentford": 4,
    "Brighton": 5, "Brighton & Hove Albion": 5, "Brighton and Hove Albion": 5,
    "Chelsea": 6, "Crystal Palace": 7, "Everton": 8,
    "Fulham": 9, 
    "Burnley": 10,  # Promoted 25/26
    "Leeds United": 11, "Leeds": 11,  # Promoted 25/26
    "Liverpool": 12, "Manchester City": 13, "Man City": 13,
    "Manchester Utd": 14, "Manchester United": 14, "Man Utd": 14, "Man United": 14,
    "Newcastle Utd": 15, "Newcastle United": 15, "Newcastle": 15,
    "Nott'ham Forest": 16, "Nottingham Forest": 16, "Nott'm Forest": 16, "Forest": 16,
    "Sunderland": 17,  # Promoted 25/26
    "Tottenham": 18, "Spurs": 18, "Tottenham Hotspur": 18,
    "West Ham": 19, "West Ham United": 19,
    "Wolves": 20, "Wolverhampton": 20, "Wolverhampton Wanderers": 20,
}

HOME_AWAY_FDR_ADJUSTMENT = {'home': 0.85, 'away': 1.15}


# ============ ENUMS & MODELS ============

class Position(str, Enum):
    GKP = "GKP"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


class MinutesOverride(BaseModel):
    player_id: int
    expected_minutes: float
    manager_id: Optional[int] = None  # Optional for backwards compatibility


# ============ RESPONSE SCHEMAS ============
# These provide contract stability between frontend and backend

class PlayerRanking(BaseModel):
    """Schema for a player in rankings response."""
    id: int
    name: str
    team: str
    team_id: int
    position: str
    price: float
    ownership: float
    form: float
    total_points: int
    minutes: int
    xpts: float
    xpts_per_90: float
    xpts_per_cost: float
    exp_mins: float
    mins_reason: str
    fixture_ticker: str
    avg_fdr: float
    
    class Config:
        extra = "allow"  # Allow additional fields


class SquadPlayer(BaseModel):
    """Schema for a player in squad response."""
    id: int
    name: str
    team: str
    team_id: int
    position: str
    price: float
    selling_price: float
    is_captain: bool
    is_vice_captain: bool
    multiplier: int
    xpts_single_gw: float
    exp_mins: float
    fixture_ticker: str
    
    class Config:
        extra = "allow"


class TransferAction(BaseModel):
    """Schema for a transfer action in planner."""
    type: str  # "transfer" or "roll"
    player_out: Optional[Dict] = None
    player_in: Optional[Dict] = None
    gain: Optional[float] = None
    is_hit: bool = False


class GwTimeline(BaseModel):
    """Schema for a gameweek in planner timeline."""
    gw: int
    actions: List[Dict]
    chip: Optional[str] = None
    is_dgw: bool = False
    starting_xi: Optional[List[Dict]] = None
    bench: Optional[List[Dict]] = None
    captain: Optional[Dict] = None
    vice_captain: Optional[Dict] = None
    gw_xpts: Optional[float] = None
    
    class Config:
        extra = "allow"


class TransferPlan(BaseModel):
    """Schema for transfer plan response."""
    total_xpts: float
    baseline_xpts: float
    xpts_gain: float
    total_hits: int
    hit_cost: int
    net_xpts_gain: float
    transfers_by_gw: Dict[int, int]
    timeline: List[GwTimeline]
    
    class Config:
        extra = "allow"


class ChipStatus(BaseModel):
    """Schema for chip availability."""
    WC: bool
    FH: bool
    BB: bool
    TC: bool


class PlannerResponse(BaseModel):
    """Full planner API response schema."""
    manager_id: int
    current_gw: int
    planning_horizon: str
    available_chips: Dict[str, bool]
    chip_schedule: Dict[str, int]
    chip_suggestions: List[Dict]
    gw_info: Dict[int, Dict]
    bank: float
    free_transfers: int
    ft_disclaimer: str
    plan: TransferPlan
    
    class Config:
        extra = "allow"


# =============================================================================
# MODEL DATA CLASSES
# =============================================================================

@dataclass
class HomeAwaySplit:
    """Player's home vs away performance statistics."""
    home_xG90: float = 0.0
    away_xG90: float = 0.0
    home_xA90: float = 0.0
    away_xA90: float = 0.0
    home_pts_per_90: float = 0.0
    away_pts_per_90: float = 0.0
    home_games: int = 0
    away_games: int = 0
    has_sufficient_data: bool = False
    
    @property
    def home_xGI90(self) -> float:
        return self.home_xG90 + self.home_xA90
    
    @property
    def away_xGI90(self) -> float:
        return self.away_xG90 + self.away_xA90


@dataclass
class CSProbability:
    """Clean sheet probability for a fixture."""
    cs_prob: float
    expected_goals_against: float
    data_source: str
    opponent_attack_strength: float = 1.0


@dataclass
class AttackingEstimate:
    """Attacking returns estimate for a fixture."""
    xG90: float
    xA90: float
    attack_multiplier: float
    fixture_xG90: float
    fixture_xA90: float
    data_source: str


@dataclass
class BonusEstimate:
    """Bonus points estimate."""
    expected_bonus: float
    estimated_bps: float
    bonus_per_90_historical: float
    teammate_dilution_applied: bool = False


@dataclass
class VarianceEstimate:
    """Ceiling and floor estimates."""
    ceiling: float
    floor: float
    std_dev: float
    data_source: str


@dataclass
class FixtureXpts:
    """xPts breakdown for a single fixture."""
    fixture_gw: int
    opponent_id: int
    opponent_name: str
    is_home: bool
    difficulty: int
    
    appearance_pts: float = 0.0
    goal_pts: float = 0.0
    assist_pts: float = 0.0
    cs_pts: float = 0.0
    bonus_pts: float = 0.0
    save_pts: float = 0.0
    defcon_pts: float = 0.0
    
    yellow_deduction: float = 0.0
    og_deduction: float = 0.0
    gc_deduction: float = 0.0
    
    cs_prob: float = 0.0
    expected_goals_against: float = 0.0
    attack_multiplier: float = 1.0
    
    @property
    def total_xpts(self) -> float:
        return (
            self.appearance_pts + self.goal_pts + self.assist_pts +
            self.cs_pts + self.bonus_pts + self.save_pts + self.defcon_pts -
            self.yellow_deduction - self.og_deduction - self.gc_deduction
        )


# =============================================================================
# MODEL IMPLEMENTATIONS
# =============================================================================

class HomeAwaySplitCalculator:
    """Calculate player's home vs away performance splits from GW history."""
    
    def __init__(self, min_games: int = 8):
        self.min_games = min_games
    
    def calculate_splits(
        self,
        player_history: Optional[Dict],
        player_data: Dict
    ) -> HomeAwaySplit:
        """Calculate home/away performance splits from player history."""
        split = HomeAwaySplit()
        
        if not player_history or not player_history.get("history"):
            return split
        
        history = player_history["history"]
        
        home_stats = {"minutes": 0, "xG": 0, "xA": 0, "points": 0, "games": 0}
        away_stats = {"minutes": 0, "xG": 0, "xA": 0, "points": 0, "games": 0}
        
        for gw in history:
            mins = gw.get("minutes", 0)
            if mins == 0:
                continue
            
            was_home = gw.get("was_home", True)
            stats = home_stats if was_home else away_stats
            
            stats["minutes"] += mins
            stats["xG"] += float(gw.get("expected_goals", 0) or 0)
            stats["xA"] += float(gw.get("expected_assists", 0) or 0)
            stats["points"] += gw.get("total_points", 0)
            stats["games"] += 1
        
        split.home_games = home_stats["games"]
        split.away_games = away_stats["games"]
        
        if home_stats["minutes"] >= 90:
            home_90s = home_stats["minutes"] / 90
            split.home_xG90 = home_stats["xG"] / home_90s
            split.home_xA90 = home_stats["xA"] / home_90s
            split.home_pts_per_90 = home_stats["points"] / home_90s
        
        if away_stats["minutes"] >= 90:
            away_90s = away_stats["minutes"] / 90
            split.away_xG90 = away_stats["xG"] / away_90s
            split.away_xA90 = away_stats["xA"] / away_90s
            split.away_pts_per_90 = away_stats["points"] / away_90s
        
        split.has_sufficient_data = (
            split.home_games >= self.min_games and 
            split.away_games >= self.min_games
        )
        
        return split


class CleanSheetModel:
    """
    Calculate fixture-specific clean sheet probability.
    
    KEY FIX: CS probability now depends on OPPONENT's attacking strength,
    not just your team's defensive baseline.
    """
    
    def __init__(self, config: CleanSheetConfig = None):
        self.config = config or MODEL_CONFIG["clean_sheet"]
    
    def calculate_fixture_cs_probability(
        self,
        team_id: int,
        opponent_id: int,
        is_home: bool,
        fdr_data: Dict[int, Dict],
        fixture_xg_data: Optional[Dict] = None,
        fixture_xg_weight: float = 0.0
    ) -> CSProbability:
        """Calculate CS probability for a specific fixture."""
        ha_config = MODEL_CONFIG["home_away"]
        data_source = "model"
        
        # Get team's baseline xGA
        if fdr_data and team_id in fdr_data:
            team_xga_baseline = fdr_data[team_id].get('blended_xga', 1.3)
        else:
            team_xga_baseline = 1.3
        
        # Get opponent's attacking strength
        if fdr_data and opponent_id in fdr_data:
            opponent_attack = fdr_data[opponent_id].get('blended_xg', self.config.league_avg_goals_per_game)
        else:
            opponent_attack = self.config.league_avg_goals_per_game
        
        opponent_attack_multiplier = opponent_attack / self.config.league_avg_goals_per_game
        
        if fixture_xg_data and fixture_xg_weight > 0:
            # Use fixture-specific projection
            if is_home:
                fixture_xga = fixture_xg_data.get("away_xg", 1.3)
            else:
                fixture_xga = fixture_xg_data.get("home_xg", 1.5)
            
            # Model-based xGA for blending
            if is_home:
                ha_factor = 1.0 / ha_config.home_defence_boost
                opp_mult = opponent_attack_multiplier * ha_config.away_attack_penalty
            else:
                ha_factor = 1.0 / ha_config.away_defence_penalty
                opp_mult = opponent_attack_multiplier * ha_config.home_attack_boost
            
            model_xga = team_xga_baseline * opp_mult * ha_factor
            expected_goals_against = (
                fixture_xg_weight * fixture_xga + 
                (1 - fixture_xg_weight) * model_xga
            )
            data_source = "fixture_xg+model"
        else:
            # Pure model-based
            if is_home:
                ha_factor = 1.0 / ha_config.home_defence_boost
                opp_mult = opponent_attack_multiplier * ha_config.away_attack_penalty
            else:
                ha_factor = 1.0 / ha_config.away_defence_penalty
                opp_mult = opponent_attack_multiplier * ha_config.home_attack_boost
            
            expected_goals_against = team_xga_baseline * opp_mult * ha_factor
            data_source = "team_model"
        
        expected_goals_against = max(0.4, min(3.0, expected_goals_against))
        
        # Poisson CS probability
        cs_prob = math.exp(-expected_goals_against)
        cs_prob = max(self.config.cs_prob_min, min(self.config.cs_prob_max, cs_prob))
        
        return CSProbability(
            cs_prob=round(cs_prob, 4),
            expected_goals_against=round(expected_goals_against, 3),
            data_source=data_source,
            opponent_attack_strength=round(opp_mult, 3)
        )


class AttackingModel:
    """
    Calculate fixture-adjusted attacking returns with home/away splits
    and price-based FDR dampening.
    """
    
    def __init__(self, config: FDRConfig = None):
        self.config = config or MODEL_CONFIG["fdr"]
    
    def get_fdr_multiplier(self, fdr: int, player_price: float) -> float:
        """Get FDR multiplier with price-based dampening."""
        base_mult = self.config.fdr_multipliers.get(fdr, 1.0)
        
        if player_price >= self.config.premium_price_threshold:
            dampening = self.config.premium_dampening
        elif player_price >= self.config.mid_price_threshold:
            dampening = self.config.mid_price_dampening
        else:
            dampening = self.config.budget_dampening
        
        return 1.0 + (base_mult - 1.0) * dampening
    
    def calculate_fixture_attacking(
        self,
        base_xG90: float,
        base_xA90: float,
        opponent_id: int,
        is_home: bool,
        fdr_data: Dict[int, Dict],
        home_away_split: Optional[HomeAwaySplit] = None,
        player_price: float = 5.0
    ) -> AttackingEstimate:
        """Calculate fixture-adjusted xG/xA with home/away splits."""
        ha_config = MODEL_CONFIG["home_away"]
        data_source = "base_rate"
        
        xG90 = base_xG90
        xA90 = base_xA90
        
        # Apply home/away split if available
        if home_away_split and home_away_split.has_sufficient_data:
            if is_home:
                split_xG90 = home_away_split.home_xG90
                split_xA90 = home_away_split.home_xA90
            else:
                split_xG90 = home_away_split.away_xG90
                split_xA90 = home_away_split.away_xA90
            
            # Blend split with overall
            xG90 = ha_config.split_weight * split_xG90 + (1 - ha_config.split_weight) * base_xG90
            xA90 = ha_config.split_weight * split_xA90 + (1 - ha_config.split_weight) * base_xA90
            data_source = "home_away_split"
        
        # Calculate attack multiplier from FDR
        attack_multiplier = 1.0
        
        if fdr_data and opponent_id in fdr_data:
            opp_def_fdr = fdr_data[opponent_id].get("defence_fdr", 5)
            attack_multiplier = self.get_fdr_multiplier(round(opp_def_fdr), player_price)
            data_source += "+fdr"
        
        # Apply home/away adjustment
        if is_home:
            attack_multiplier *= ha_config.home_attack_boost
        else:
            attack_multiplier *= ha_config.away_attack_penalty
        
        # Cap the total boost
        attack_multiplier = max(0.78, min(1.22, attack_multiplier))
        
        fixture_xG90 = xG90 * attack_multiplier
        fixture_xA90 = xA90 * attack_multiplier
        
        return AttackingEstimate(
            xG90=round(xG90, 4),
            xA90=round(xA90, 4),
            attack_multiplier=round(attack_multiplier, 4),
            fixture_xG90=round(fixture_xG90, 4),
            fixture_xA90=round(fixture_xA90, 4),
            data_source=data_source
        )


class VarianceModel:
    """
    Calculate ceiling and floor using historical variance.
    
    KEY FIX: Uses player's actual points std dev when available,
    not flat position multipliers.
    """
    
    def __init__(self, config: VarianceConfig = None):
        self.config = config or MODEL_CONFIG["variance"]
    
    def calculate_variance(
        self,
        player_history: Optional[Dict],
        player_data: Dict,
        position_id: int,
        xpts: float,
        fixture_fdr: int = 5,
        is_home: bool = True
    ) -> VarianceEstimate:
        """Calculate ceiling/floor using historical variance."""
        data_source = "position_baseline"
        std_dev = self.config.position_stdev_baseline.get(position_id, 3.0)
        
        # Try player-specific std dev from history
        if player_history and player_history.get("history"):
            history = player_history["history"]
            
            points_list = [
                h.get("total_points", 0) 
                for h in history 
                if h.get("minutes", 0) >= 45
            ]
            
            if len(points_list) >= self.config.min_games_for_stdev:
                mean_pts = sum(points_list) / len(points_list)
                variance = sum((p - mean_pts) ** 2 for p in points_list) / (len(points_list) - 1)
                player_stdev = math.sqrt(variance)
                
                # Blend with position baseline
                history_weight = min(1.0, len(points_list) / 25)
                std_dev = history_weight * player_stdev + (1 - history_weight) * std_dev
                data_source = "historical"
        
        # Base ceiling and floor
        base_ceiling = xpts + self.config.ceiling_z * std_dev
        base_floor = xpts + self.config.floor_z * std_dev
        
        # Fixture quality adjustment to ceiling
        fixture_boost = self.config.fixture_ceiling_boost.get(fixture_fdr, 1.0)
        ha_boost = self.config.home_ceiling_boost if is_home else self.config.away_ceiling_penalty
        
        ceiling = base_ceiling * fixture_boost * ha_boost
        floor = max(0, base_floor)
        
        return VarianceEstimate(
            ceiling=round(ceiling, 2),
            floor=round(floor, 2),
            std_dev=round(std_dev, 2),
            data_source=data_source
        )


class BonusModel:
    """
    Calculate expected bonus points with teammate competition.
    """
    
    def __init__(self, config: BonusConfig = None):
        self.config = config or MODEL_CONFIG["bonus"]
    
    def calculate_bonus(
        self,
        player_data: Dict,
        position_id: int,
        fixture_xG90: float,
        fixture_xA90: float,
        cs_prob: float,
        teammate_competition: bool = False
    ) -> BonusEstimate:
        """Calculate expected bonus for a fixture."""
        total_minutes = int(player_data.get("minutes", 0) or 0)
        bonus = int(player_data.get("bonus", 0) or 0)
        
        if total_minutes >= 180:
            mins90 = total_minutes / 90.0
            bonus_per_90 = bonus / mins90
        else:
            bonus_per_90 = 0.45
        
        # Estimate BPS
        goal_bps = fixture_xG90 * self.config.bps_per_goal.get(position_id, 24)
        assist_bps = fixture_xA90 * self.config.bps_per_assist
        cs_bps = cs_prob * self.config.bps_per_cs.get(position_id, 0)
        base = self.config.base_bps_by_position.get(position_id, 8)
        
        estimated_bps = base + goal_bps + assist_bps + cs_bps
        
        # Convert BPS to expected bonus
        expected_bonus = 0.15
        for threshold, bonus_val in self.config.bps_to_bonus:
            if estimated_bps >= threshold:
                expected_bonus = bonus_val
                break
        
        # Blend historical and modeled
        if total_minutes >= 1350:
            final_bonus = 0.65 * bonus_per_90 + 0.35 * expected_bonus
        elif total_minutes >= 720:
            final_bonus = 0.50 * bonus_per_90 + 0.50 * expected_bonus
        else:
            final_bonus = 0.30 * bonus_per_90 + 0.70 * expected_bonus
        
        dilution_applied = False
        if teammate_competition:
            final_bonus *= self.config.teammate_dilution
            dilution_applied = True
        
        final_bonus = max(0.1, min(2.5, final_bonus))
        
        return BonusEstimate(
            expected_bonus=round(final_bonus, 3),
            estimated_bps=round(estimated_bps, 1),
            bonus_per_90_historical=round(bonus_per_90, 3),
            teammate_dilution_applied=dilution_applied
        )


# Initialize global model instances
home_away_calculator = HomeAwaySplitCalculator(min_games=MODEL_CONFIG["home_away"].min_games_for_split)
cs_model = CleanSheetModel()
attacking_model = AttackingModel()
variance_model = VarianceModel()
bonus_model = BonusModel()


# ============ CACHE ============

class DataCache:
    def __init__(self):
        self.bootstrap_data: Optional[Dict] = None
        self.fixtures_data: Optional[List] = None
        self.element_summary_cache: Dict[int, Dict] = {}
        self.last_update: Optional[datetime] = None
        self.fixtures_last_update: Optional[datetime] = None
        self.cache_duration = 300
        # Minutes overrides scoped by (manager_id, player_id) - None manager_id for global
        self.minutes_overrides: Dict[Tuple[Optional[int], int], float] = {}
        # FDR cache
        self.fdr_data: Dict[int, Dict] = {}  # team_id -> {attack_fdr, defence_fdr, form_xg, form_xga, ...}
        self.fdr_last_update: Optional[datetime] = None
        # Predicted minutes from external sources (e.g., FPL Review)
        self.predicted_minutes: Dict[int, float] = {}  # player_id -> predicted_minutes
        self.predicted_minutes_last_update: Optional[datetime] = None
        # Manual team strength data (from Analytic FPL)
        self.manual_team_strength: Dict[str, Dict] = {}  # team_name -> {adjxg_for, adjxg_ag}
        self.manual_team_strength_last_update: Optional[datetime] = None
        # Player history cache (reduces API calls significantly)
        self.player_histories: Dict[int, Dict] = {}  # player_id -> history data
        self.player_histories_last_update: Dict[int, datetime] = {}  # player_id -> last update time
        self.player_history_cache_duration = 1800  # 30 minutes
        # Fixture-specific xG projections (from Analytic FPL or similar)
        # Key: (home_team_id, away_team_id, gw) -> {home_xg, away_xg}
        self.fixture_xg: Dict[Tuple[int, int, int], Dict] = {}
        self.fixture_xg_last_update: Optional[datetime] = None
    
    def get_minutes_override(self, player_id: int, manager_id: Optional[int] = None) -> Optional[float]:
        """Get minutes override, checking manager-specific first, then global, then predicted."""
        # Check manager-specific override first
        if manager_id is not None and (manager_id, player_id) in self.minutes_overrides:
            return self.minutes_overrides[(manager_id, player_id)]
        # Fall back to global override
        if (None, player_id) in self.minutes_overrides:
            return self.minutes_overrides[(None, player_id)]
        # Fall back to predicted minutes (from FPL Review etc.)
        if player_id in self.predicted_minutes:
            return self.predicted_minutes[player_id]
        return None
    
    def set_minutes_override(self, player_id: int, minutes: float, manager_id: Optional[int] = None):
        """Set minutes override, scoped by manager_id."""
        self.minutes_overrides[(manager_id, player_id)] = minutes
    
    def delete_minutes_override(self, player_id: int, manager_id: Optional[int] = None) -> bool:
        """Delete minutes override. Returns True if deleted."""
        key = (manager_id, player_id)
        if key in self.minutes_overrides:
            del self.minutes_overrides[key]
            return True
        return False
    
    def predicted_minutes_is_stale(self) -> bool:
        return self.predicted_minutes_last_update is None or (datetime.now() - self.predicted_minutes_last_update).total_seconds() > 3600  # 1 hour

    def is_stale(self) -> bool:
        return self.last_update is None or (datetime.now() - self.last_update).total_seconds() > self.cache_duration

    def fixtures_is_stale(self) -> bool:
        return self.fixtures_last_update is None or (datetime.now() - self.fixtures_last_update).total_seconds() > self.cache_duration
    
    def fdr_is_stale(self) -> bool:
        return self.fdr_last_update is None or (datetime.now() - self.fdr_last_update).total_seconds() > 21600  # 6 hours
    
    def get_player_history(self, player_id: int) -> Optional[Dict]:
        """Get cached player history if not stale."""
        if player_id not in self.player_histories:
            return None
        last_update = self.player_histories_last_update.get(player_id)
        if last_update is None:
            return None
        if (datetime.now() - last_update).total_seconds() > self.player_history_cache_duration:
            return None  # Stale
        return self.player_histories[player_id]
    
    def set_player_history(self, player_id: int, history: Dict):
        """Cache player history."""
        self.player_histories[player_id] = history
        self.player_histories_last_update[player_id] = datetime.now()
    
    def clear_stale_player_histories(self):
        """Remove stale player histories to prevent memory bloat."""
        now = datetime.now()
        stale_ids = [
            pid for pid, last_update in self.player_histories_last_update.items()
            if (now - last_update).total_seconds() > self.player_history_cache_duration * 2
        ]
        for pid in stale_ids:
            self.player_histories.pop(pid, None)
            self.player_histories_last_update.pop(pid, None)


cache = DataCache()


# ============ FPL API FETCHERS ============

async def fetch_fpl_data():
    if not cache.is_stale() and cache.bootstrap_data:
        return cache.bootstrap_data
    try:
        response = await fetch_with_retry(f"{FPL_BASE_URL}/bootstrap-static/")
        cache.bootstrap_data = response.json()
        cache.last_update = datetime.now()
        return cache.bootstrap_data
    except Exception as e:
        if cache.bootstrap_data:
            return cache.bootstrap_data
        raise HTTPException(status_code=503, detail=f"FPL API unavailable: {str(e)}")


async def fetch_fixtures():
    if cache.fixtures_data and not cache.fixtures_is_stale():
        return cache.fixtures_data
    try:
        response = await fetch_with_retry(f"{FPL_BASE_URL}/fixtures/")
        cache.fixtures_data = response.json()
        cache.fixtures_last_update = datetime.now()
        return cache.fixtures_data
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"FPL API unavailable: {str(e)}")


async def fetch_manager_team(manager_id: int, gameweek: int):
    try:
        client = await get_http_client()
        manager_resp = await client.get(f"{FPL_BASE_URL}/entry/{manager_id}/")
        manager_resp.raise_for_status()
        manager_data = manager_resp.json()
        picks_resp = await client.get(f"{FPL_BASE_URL}/entry/{manager_id}/event/{gameweek}/picks/")
        picks_resp.raise_for_status()
        picks_data = picks_resp.json()
        transfers_resp = await client.get(f"{FPL_BASE_URL}/entry/{manager_id}/transfers/")
        transfers_resp.raise_for_status()
        transfers_data = transfers_resp.json()
        return {"manager": manager_data, "picks": picks_data, "transfers": transfers_data}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Manager not found or no team for this gameweek")
        raise HTTPException(status_code=503, detail=f"FPL API error: {str(e)}")


def get_current_gameweek(events: List[Dict]) -> int:
    for event in events:
        if event.get("is_current"):
            return event["id"]
    for event in events:
        if event.get("is_next"):
            return event["id"]
    return 1


def get_next_gameweek(events: List[Dict]) -> int:
    for event in events:
        if event.get("is_next"):
            return event["id"]
    return min(get_current_gameweek(events) + 1, 38)


# ============ UNDERSTAT SCRAPER ============

class UnderstatScraper:
    """Scrape xG data from Understat (embedded JSON in HTML)."""
    
    BASE_URL = "https://understat.com"
    
    def __init__(self, rate_limit_seconds: float = 2.0):
        self.rate_limit = rate_limit_seconds
        self.last_request = 0
    
    async def _fetch_page(self, url: str) -> str:
        now = asyncio.get_event_loop().time()
        wait_time = self.last_request + self.rate_limit - now
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        client = await get_http_client()
        response = await client.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        self.last_request = asyncio.get_event_loop().time()
        return response.text
    
    def _extract_json(self, html: str, var_name: str) -> Optional[list]:
        pattern = rf"var {var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html, re.DOTALL)
        if not match:
            return None
        
        encoded = match.group(1)
        try:
            decoded = encoded.encode('utf-8').decode('unicode_escape')
            return json.loads(decoded)
        except Exception as e:
            logger.error(f"Failed to parse Understat JSON: {e}")
            return None
    
    async def fetch_league_matches(self, season: Optional[str] = None) -> List[dict]:
        """Fetch league matches from Understat. Season derived automatically if not provided."""
        if season is None:
            season = get_current_season()
        url = f"{self.BASE_URL}/league/EPL/{season}"
        try:
            html = await self._fetch_page(url)
            matches = self._extract_json(html, "datesData")
            return matches or []
        except Exception as e:
            logger.error(f"Failed to fetch Understat matches: {e}")
            return []


understat_scraper = UnderstatScraper()


# ============ FDR SERVICE ============

async def refresh_fdr_data(force: bool = False):
    """Fetch Understat data and compute FDR scores for all teams."""
    # Return cached data immediately if available and not forced refresh
    if not force and cache.fdr_data and not cache.fdr_is_stale():
        return cache.fdr_data
    
    # If we have cached data and it's just a refresh attempt (not forced), return cache
    # This prevents blocking requests while waiting for Understat
    if cache.fdr_data and not force:
        # Trigger background refresh if stale but don't wait for it
        asyncio.create_task(_background_fdr_refresh())
        return cache.fdr_data
    
    # No cached data - must wait for fetch
    return await _do_fdr_refresh()


async def _background_fdr_refresh():
    """Background task to refresh FDR data without blocking."""
    try:
        await _do_fdr_refresh()
    except Exception as e:
        logger.error(f"Background FDR refresh failed: {e}")


async def _do_fdr_refresh():
    """Actually perform the FDR data refresh."""
    logger.info("Refreshing FDR data from Understat...")
    
    # Fetch all matches with timeout (using dynamic season)
    try:
        matches = await asyncio.wait_for(
            understat_scraper.fetch_league_matches(),  # Uses get_current_season() internally
            timeout=10.0
        )
    except asyncio.TimeoutError:
        logger.warning("Understat fetch timed out, using cached/fallback data")
        return cache.fdr_data or {}
    
    if not matches:
        logger.warning("No Understat matches fetched, using FPL API fallback")
        # Set a timestamp even on failure to prevent immediate retry
        if not cache.fdr_data:
            cache.fdr_last_update = datetime.now()
        return cache.fdr_data or {}
    
    # Aggregate by team
    team_stats: Dict[int, Dict] = defaultdict(lambda: {
        'matches': [], 'season_xg': 0, 'season_xga': 0
    })
    
    for match in matches:
        if not match.get('isResult'):
            continue
        
        home_team = match['h']['title']
        away_team = match['a']['title']
        home_id = UNDERSTAT_TO_FPL_ID.get(home_team)
        away_id = UNDERSTAT_TO_FPL_ID.get(away_team)
        
        if not home_id or not away_id:
            continue
        
        home_xg = float(match['xG']['h'])
        away_xg = float(match['xG']['a'])
        home_goals = int(match['goals']['h'])
        away_goals = int(match['goals']['a'])
        match_date = match['datetime']
        
        # Home team
        home_pts = 3 if home_goals > away_goals else (1 if home_goals == away_goals else 0)
        team_stats[home_id]['matches'].append({
            'xg': home_xg, 'xga': away_xg, 'pts': home_pts, 'date': match_date
        })
        team_stats[home_id]['season_xg'] += home_xg
        team_stats[home_id]['season_xga'] += away_xg
        
        # Away team
        away_pts = 3 if away_goals > home_goals else (1 if away_goals == home_goals else 0)
        team_stats[away_id]['matches'].append({
            'xg': away_xg, 'xga': home_xg, 'pts': away_pts, 'date': match_date
        })
        team_stats[away_id]['season_xg'] += away_xg
        team_stats[away_id]['season_xga'] += home_xg
    
    # Compute form (last 6) and FDR
    fdr_data = {}
    all_form_xg = []
    all_form_xga = []
    
    for team_id, data in team_stats.items():
        matches_sorted = sorted(data['matches'], key=lambda x: x['date'])
        last_6 = matches_sorted[-6:] if len(matches_sorted) >= 6 else matches_sorted
        
        if last_6:
            form_xg = sum(m['xg'] for m in last_6) / len(last_6)
            form_xga = sum(m['xga'] for m in last_6) / len(last_6)
            form_ppg = sum(m['pts'] for m in last_6) / len(last_6)
        else:
            form_xg = form_xga = form_ppg = 0
        
        # Calculate season averages
        season_matches = len(data['matches'])
        season_xg_per_game = data['season_xg'] / max(1, season_matches)
        season_xga_per_game = data['season_xga'] / max(1, season_matches)
        
        # Check if we have manual team strength data (from Analytic FPL)
        manual_data = cache.manual_team_strength.get(team_id)
        
        if manual_data:
            # ==================== WEIGHTED BLEND WITH TREND ====================
            # Weighting: 55% Analytic FPL baseline, 30% Understat form, 15% trend adjustment
            #
            # The trend data captures teams improving/declining:
            # - attack_delta: Recent change in attacking output
            # - defence_delta: Recent change in defensive solidity
            # - attack_trend: Directional momentum for attack
            # - defence_trend: Directional momentum for defence
            
            manual_xg = manual_data.get('adjxg_for', form_xg)
            manual_xga = manual_data.get('adjxg_ag', form_xga)
            
            # Get trend adjustments (default to 0 if not provided)
            attack_delta = manual_data.get('attack_delta') or 0
            defence_delta = manual_data.get('defence_delta') or 0
            attack_trend = manual_data.get('attack_trend') or 0
            defence_trend = manual_data.get('defence_trend') or 0
            
            # Combine delta and trend into a momentum factor
            # Delta = recent change (more weight), Trend = longer-term direction
            attack_momentum = attack_delta * 0.6 + attack_trend * 0.4
            defence_momentum = defence_delta * 0.6 + defence_trend * 0.4
            
            # Apply momentum to baseline (capped at ±15% adjustment)
            # Positive attack momentum = expect MORE goals
            # Negative defence momentum = expect FEWER goals conceded
            attack_adjustment = max(-0.15, min(0.15, attack_momentum * 0.5))
            defence_adjustment = max(-0.15, min(0.15, defence_momentum * 0.5))
            
            # Final blend: 55% Analytic FPL, 30% Understat form, 15% trend adjustment
            base_xg = 0.55 * manual_xg + 0.30 * form_xg + 0.15 * (manual_xg + attack_adjustment)
            base_xga = 0.55 * manual_xga + 0.30 * form_xga + 0.15 * (manual_xga + defence_adjustment)
            
            # Simplify to: baseline × (1 + momentum_factor × weight)
            blended_xg = (0.55 * manual_xg + 0.30 * form_xg) * (1 + attack_adjustment)
            blended_xga = (0.55 * manual_xga + 0.30 * form_xga) * (1 + defence_adjustment)
            
            data_source = "analytic_fpl+form+trend"
            trend_info = {
                "attack_momentum": round(attack_momentum, 3),
                "defence_momentum": round(defence_momentum, 3),
                "attack_adjustment": round(attack_adjustment, 3),
                "defence_adjustment": round(defence_adjustment, 3),
            }
        else:
            # Fallback to form + season blend (original behavior)
            if season_matches >= 10:
                blended_xga = 0.6 * form_xga + 0.4 * season_xga_per_game
                blended_xg = 0.6 * form_xg + 0.4 * season_xg_per_game
            else:
                blended_xga = form_xga if form_xga > 0 else season_xga_per_game
                blended_xg = form_xg if form_xg > 0 else season_xg_per_game
            data_source = "understat_form+season"
            trend_info = None
        
        # Calculate clean sheet probability using Poisson: P(0 goals) = e^(-xGA)
        cs_probability = math.exp(-blended_xga) if blended_xga > 0 else 0.25
        cs_probability = min(0.50, max(0.08, cs_probability))  # Bound 8%-50%
        
        fdr_data[team_id] = {
            'season_xg': data['season_xg'],
            'season_xga': data['season_xga'],
            'season_matches': season_matches,
            'form_xg': round(form_xg, 2),
            'form_xga': round(form_xga, 2),
            'form_ppg': round(form_ppg, 2),
            'season_xg_per_game': round(season_xg_per_game, 2),
            'season_xga_per_game': round(season_xga_per_game, 2),
            'blended_xg': round(blended_xg, 2),  # Final blended xG for predictions
            'blended_xga': round(blended_xga, 2),  # Final blended xGA for predictions
            'cs_probability': round(cs_probability, 3),  # Team's expected CS rate
            'xg_per_game': round(blended_xg, 2),  # Alias for backwards compatibility
            'data_source': data_source,
            'trend_info': trend_info,  # Momentum adjustments if available
        }
        all_form_xg.append(blended_xg)  # Use blended for FDR percentiles
        all_form_xga.append(blended_xga)
    
    # Compute FDR scores using normalized xG/xGA values
    # This gives more differentiation than percentiles
    if len(all_form_xg) >= 5:
        # Get min/max for normalization
        min_xg, max_xg = min(all_form_xg), max(all_form_xg)
        min_xga, max_xga = min(all_form_xga), max(all_form_xga)
        xg_range = max(max_xg - min_xg, 0.5)  # Avoid division by zero
        xga_range = max(max_xga - min_xga, 0.3)
        
        for team_id, data in fdr_data.items():
            # Attack FDR: How hard to attack this team (based on their blended xGA)
            # Low xGA = hard to score against = HIGH attack_fdr
            # Normalize xGA to 0-1 scale, then invert and scale to 1-10
            xga_norm = (data['blended_xga'] - min_xga) / xga_range  # 0 = best defense, 1 = worst
            attack_fdr = int(round(1 + (1 - xga_norm) * 9))  # 1-10, inverted
            
            # Defence FDR: How hard to defend against this team (based on their blended xG)
            # High xG = dangerous attack = HIGH defence_fdr
            xg_norm = (data['blended_xg'] - min_xg) / xg_range  # 0 = weakest attack, 1 = strongest
            defence_fdr = int(round(1 + xg_norm * 9))  # 1-10
            
            data['attack_fdr'] = max(1, min(10, attack_fdr))
            data['defence_fdr'] = max(1, min(10, defence_fdr))
            data['composite_fdr'] = round((data['attack_fdr'] + data['defence_fdr']) / 2, 1)
    
    cache.fdr_data = fdr_data
    cache.fdr_last_update = datetime.now()
    logger.info(f"FDR data refreshed for {len(fdr_data)} teams")
    
    return fdr_data


def get_fixture_fdr(opponent_id: int, is_home: bool, position_id: int) -> int:
    """
    Get FDR for a fixture based on opponent and player position.
    
    - FWD/MID: Use opponent's attack_fdr (how hard to score against them)
    - DEF/GKP: Use opponent's defence_fdr (how hard to keep clean sheet)
    """
    if not cache.fdr_data or opponent_id not in cache.fdr_data:
        return 5  # Neutral fallback
    
    opponent_data = cache.fdr_data[opponent_id]
    
    # Position-specific FDR
    if position_id in [3, 4]:  # MID, FWD - care about scoring
        base_fdr = opponent_data.get('attack_fdr', 5)
    else:  # GKP, DEF - care about clean sheets
        base_fdr = opponent_data.get('defence_fdr', 5)
    
    # Home/away adjustment
    ha_mult = HOME_AWAY_FDR_ADJUSTMENT['home'] if is_home else HOME_AWAY_FDR_ADJUSTMENT['away']
    adjusted_fdr = int(round(base_fdr * ha_mult))
    
    return max(1, min(10, adjusted_fdr))


# ============ PLAYER STAT CALCULATIONS ============

def get_ownership_tier(ownership: float) -> Tuple[str, str]:
    """
    Categorize player by ownership for rank climbing strategy.
    
    Returns: (tier_name, tier_description)
    
    - template: Must-own players, you lose rank without them
    - popular: Common picks, low differential upside  
    - differential: Can gain rank significantly if they haul
    - punt: High variance, big rank swings possible
    """
    if ownership >= 30:
        return "template", "Essential pick"
    elif ownership >= 15:
        return "popular", "Common pick"
    elif ownership >= 5:
        return "differential", "Rank gainer"
    else:
        return "punt", "High variance"


def calculate_defcon_per_90(player: Dict, position_id: int) -> tuple[float, float, int]:
    """
    Get DEFCON per 90 from FPL API (already calculated) and compute probability.
    DEFCON = Defensive Contributions (tackles, interceptions, blocks, clearances, recoveries)
    """
    total_minutes = int(player.get("minutes", 0) or 0)
    
    # FPL API provides these directly
    defcon_per_90 = float(player.get("defensive_contribution_per_90", 0) or 0)
    defcon_total = int(player.get("defensive_contribution", 0) or 0)
    
    if total_minutes < 90 or position_id not in [2, 3]:
        return 0.0, 0.0, 0
    
    # Position-specific thresholds for DEFCON points (2 pts if >= threshold)
    if position_id == 2:  # DEF
        threshold = DEFCON_THRESHOLD_DEF
    else:  # MID
        threshold = DEFCON_THRESHOLD_MID
    
    # Calculate probability of hitting threshold using sigmoid
    # Steeper curve (scale=1.5) - elite defenders (Rodri, Rice) hit 55-60% 
    # of the time at 14-16 DEFCON/90 vs threshold of 12
    if defcon_per_90 <= 0:
        prob = 0.0
    else:
        scale = 1.5  # Steeper curve
        raw_prob = 1.0 / (1.0 + math.exp(-(defcon_per_90 - threshold) / scale))
        prob = min(raw_prob, 0.60)  # Cap at 60% - realistic for elite defensive mids
    
    return round(defcon_per_90, 2), round(prob, 3), defcon_total


def calculate_saves_per_90(player: Dict) -> tuple[float, float]:
    total_minutes = int(player.get("minutes", 0) or 0)
    saves = int(player.get("saves", 0) or 0)
    
    if total_minutes < 90:
        return 0.0, 0.0
    
    mins90 = total_minutes / 90.0
    saves_per_90 = saves / mins90
    save_pts_per_90 = saves_per_90 / 3.0
    
    return round(saves_per_90, 2), round(save_pts_per_90, 2)


def calculate_expected_bonus(player: Dict, position: int) -> tuple[float, float]:
    """
    Expert-level bonus point prediction.
    
    BPS (Bonus Point System) is heavily driven by:
    - Goals scored (+24 BPS for MID/FWD, +12 for DEF)
    - Assists (+9 BPS)
    - Clean sheets (+12 for DEF, +6 for MID)
    - Key passes, tackles, saves, interceptions
    - Penalties for cards, missed chances, errors
    
    Top 3 BPS in each match get 3, 2, 1 bonus points respectively.
    
    Returns: (bonus_per_90_historical, expected_bonus_per_90)
    """
    total_minutes = int(player.get("minutes", 0) or 0)
    bonus = int(player.get("bonus", 0) or 0)
    bps = int(player.get("bps", 0) or 0)
    goals = int(player.get("goals_scored", 0) or 0)
    assists = int(player.get("assists", 0) or 0)
    cs = int(player.get("clean_sheets", 0) or 0)
    
    xG = float(player.get("expected_goals", 0) or 0)
    xA = float(player.get("expected_assists", 0) or 0)
    
    if total_minutes < 180:  # Need at least 2 full games
        # New/low-minute players - use position baseline
        baseline = {1: 0.30, 2: 0.45, 3: 0.55, 4: 0.65}.get(position, 0.45)
        return baseline, baseline
    
    mins90 = total_minutes / 90.0
    games_played = mins90 / 1  # Approximate
    
    # Historical rates
    bonus_per_90 = bonus / mins90
    bps_per_90 = bps / mins90
    goals_per_90 = goals / mins90
    assists_per_90 = assists / mins90
    cs_per_90 = cs / mins90 if position in [1, 2] else 0
    
    # xGI per 90
    xG90 = xG / mins90
    xA90 = xA / mins90
    xGI90 = xG90 + xA90
    
    # ==================== MODEL BONUS FROM UNDERLYING STATS ====================
    # BPS (Bonus Point System) weights:
    # - Goals: +24 for MID/FWD, +12 for DEF, +6 for GKP
    # - Assists: +9
    # - Clean sheet: +12 for DEF/GKP, +6 for MID
    # - Key passes, tackles, saves contribute smaller amounts
    # 
    # Top 3 BPS in each match get 3, 2, 1 bonus points.
    # Premium attackers in good fixtures can hit 40+ BPS with a brace.
    
    # Estimate BPS contribution from goals/assists/CS
    goal_bps_contrib = goals_per_90 * (24 if position in [3, 4] else (12 if position == 2 else 6))
    assist_bps_contrib = assists_per_90 * 9
    cs_bps_contrib = cs_per_90 * (12 if position in [1, 2] else (6 if position == 3 else 0))
    
    # Base BPS from other actions (tackles, key passes, saves, interceptions)
    # Estimated from typical position ranges
    base_bps = {1: 10, 2: 11, 3: 9, 4: 6}.get(position, 8)
    
    estimated_bps = base_bps + goal_bps_contrib + assist_bps_contrib + cs_bps_contrib
    
    # Convert BPS to expected bonus using smoother curve
    # Premium attackers regularly hit 35+ BPS in good games
    if estimated_bps >= 40:
        estimated_bonus = 2.7  # Elite haul territory (Haaland brace)
    elif estimated_bps >= 35:
        estimated_bonus = 2.3
    elif estimated_bps >= 30:
        estimated_bonus = 1.9
    elif estimated_bps >= 26:
        estimated_bonus = 1.5
    elif estimated_bps >= 22:
        estimated_bonus = 1.1
    elif estimated_bps >= 18:
        estimated_bonus = 0.7
    elif estimated_bps >= 14:
        estimated_bonus = 0.4
    else:
        estimated_bonus = 0.15
    
    # ==================== BLEND HISTORICAL AND MODELED ====================
    # If player has significant history, weight historical more
    # If limited history, trust the model more
    
    if mins90 >= 15:  # ~15+ full games
        # Trust historical rate more
        final_bonus = 0.65 * bonus_per_90 + 0.35 * estimated_bonus
    elif mins90 >= 8:
        # Balanced blend
        final_bonus = 0.50 * bonus_per_90 + 0.50 * estimated_bonus
    else:
        # Limited data - trust model more
        final_bonus = 0.30 * bonus_per_90 + 0.70 * estimated_bonus
    
    # Adjust for xGI vs actual GI (regression to mean)
    # If player is over/underperforming xGI, adjust expectations
    actual_gi_per_90 = goals_per_90 + assists_per_90
    if xGI90 > 0.1:
        performance_ratio = actual_gi_per_90 / xGI90
        if performance_ratio > 1.3:
            # Overperforming - expect some regression
            final_bonus *= 0.92
        elif performance_ratio < 0.7:
            # Underperforming - expect positive regression
            final_bonus *= 1.08
    
    # Cap at realistic maximum
    final_bonus = min(2.5, max(0.1, final_bonus))
    
    return round(bonus_per_90, 2), round(final_bonus, 2)


def calculate_expected_minutes(
    player: Dict,
    current_gw: int,
    team_name: str,
    fixtures: List[Dict],
    events: List[Dict],
    player_history: Optional[Dict] = None,
    override_minutes: Optional[float] = None,
    manager_id: Optional[int] = None,
    all_players: Optional[List[Dict]] = None
) -> tuple[float, str]:
    """
    Expert-level expected minutes prediction.
    
    Priority order:
    1. User override (explicit or from cache)
    2. Rotowire predicted lineup (if fresh)
    3. Team+position backup detection
    4. Historical pattern analysis
    5. Price-based fallback for new players
    
    Returns: (expected_minutes, reason_code)
    """
    player_id = player.get("id")
    
    # ==================== LAYER 1: USER OVERRIDE ====================
    if override_minutes is not None:
        return override_minutes, "user_override"
    
    # Check cache for override (manager-specific or global)
    cached_override = cache.get_minutes_override(player_id, manager_id)
    if cached_override is not None:
        if (manager_id, player_id) in cache.minutes_overrides or (None, player_id) in cache.minutes_overrides:
            return cached_override, "user_override"
        else:
            return cached_override, "predicted"
    
    # Extract player data
    total_minutes = int(player.get("minutes", 0) or 0)
    starts = int(player.get("starts", 0) or 0)
    chance_of_playing = player.get("chance_of_playing_next_round")
    status = player.get("status", "a")
    news = player.get("news", "") or ""
    position = player.get("element_type", 4)
    price = player.get("now_cost", 50) / 10
    
    available_gws = max(current_gw - 1, 1)
    
    # ==================== HANDLE UNAVAILABLE FIRST ====================
    if status == 'u':
        return 0, "unavailable"
    
    if status == 's':
        return 0, "suspended"
    
    if status == 'i':
        if chance_of_playing is not None:
            if chance_of_playing == 0:
                return 0, "injured"
            elif chance_of_playing <= 25:
                return 10, "injury_doubt"
            elif chance_of_playing <= 50:
                return 30, "injury_50_50"
            elif chance_of_playing <= 75:
                return 55, "injury_likely"
        else:
            return 0, "injured"
    
    if status == 'd':
        if chance_of_playing and chance_of_playing >= 75:
            return 75, "minor_doubt"
        elif chance_of_playing and chance_of_playing >= 50:
            return 50, "doubtful"
        else:
            return 25, "major_doubt"
    
    # ==================== LAYER 2: ROTOWIRE PREDICTED LINEUP ====================
    # Only trust if data is fresh (<6 hours)
    if (cache.predicted_minutes and 
        player_id in cache.predicted_minutes and 
        cache.predicted_minutes_last_update and
        (datetime.now() - cache.predicted_minutes_last_update).total_seconds() < 21600):
        
        predicted_mins = cache.predicted_minutes[player_id]
        return predicted_mins, "predicted"
    
    # ==================== LAYER 3: BACKUP DETECTION ====================
    if all_players:
        is_backup, confidence, backup_reason = detect_backup_status(player, all_players, current_gw)
        if is_backup and confidence >= 0.75:
            # High confidence backup - very low minutes
            if position == 1:  # GKP backups get almost nothing
                return 0, f"backup_{backup_reason}"
            else:  # Outfield backups might get cameos
                return 10, f"backup_{backup_reason}"
        elif is_backup and confidence >= 0.5:
            # Medium confidence - some minutes possible
            return 25, f"likely_backup_{backup_reason}"
    
    # ==================== LAYER 4: INSUFFICIENT DATA ====================
    if starts == 0:
        # New signing or hasn't played - use price as proxy
        if price >= 10:
            return 70, "new_premium"  # Expensive = likely starter
        elif price >= 7:
            return 55, "new_mid_price"
        elif price >= 5:
            return 35, "new_budget"
        else:
            return 15, "new_fodder"
    
    if total_minutes < 90:
        return 15, "insufficient_data"
    
    # ==================== LAYER 5: PATTERN DETECTION ====================
    mins_per_start = total_minutes / starts
    start_rate = min(1.0, starts / available_gws)
    
    # ==================== ANALYZE RECENT HISTORY ====================
    recent_mins = []
    if player_history and player_history.get("history"):
        history = player_history["history"]
        recent = sorted(history, key=lambda x: x.get("round", 0), reverse=True)[:10]
        recent_mins = [h.get("minutes", 0) for h in recent]
    
    # ==================== PATTERN DETECTION ====================
    reason = "calculated"
    base_mins = mins_per_start
    confidence = 0.8  # Default confidence
    
    if len(recent_mins) >= 4:
        # Last 4 games analysis
        last_4 = recent_mins[:4]
        last_4_avg = sum(last_4) / 4
        last_4_starts = sum(1 for m in last_4 if m >= 60)
        last_4_zeros = sum(1 for m in last_4 if m == 0)
        last_4_subs = sum(1 for m in last_4 if 0 < m < 60)
        
        # Pattern 1: NAILED STARTER (plays 85+ mins when starting, rarely misses)
        if mins_per_start >= 85 and last_4_starts >= 3:
            base_mins = 88
            reason = "nailed"
            confidence = 0.95
        
        # Pattern 2: ROTATION RISK (inconsistent starts)
        elif last_4_starts <= 1 and (last_4_subs >= 2 or last_4_zeros >= 2):
            base_mins = min(45, last_4_avg * 1.1)  # Slightly optimistic but capped
            reason = "rotation_risk"
            confidence = 0.6
        
        # Pattern 3: REGULAR STARTER (plays most games, sometimes subbed)
        elif last_4_starts >= 2 and mins_per_start >= 75:
            base_mins = last_4_avg * 0.4 + mins_per_start * 0.6  # Blend recent and season
            reason = "regular_starter"
            confidence = 0.85
        
        # Pattern 4: IMPACT SUB (regularly comes off bench)
        elif last_4_subs >= 3:
            avg_sub_mins = sum(m for m in last_4 if 0 < m < 60) / max(1, last_4_subs)
            base_mins = avg_sub_mins * 0.8  # Slight discount
            reason = "impact_sub"
            confidence = 0.7
        
        # Pattern 5: RETURNING FROM ABSENCE
        # If recent games show 0s followed by starts
        if last_4_zeros >= 2 and last_4_starts >= 1:
            # Check if the starts are more recent than the zeros
            first_start_idx = next((i for i, m in enumerate(last_4) if m >= 60), 4)
            if first_start_idx <= 1:  # Recently started again
                # Was this player a regular before absence?
                older_mins = recent_mins[4:] if len(recent_mins) > 4 else []
                if older_mins and sum(1 for m in older_mins if m >= 60) >= 2:
                    base_mins = 80  # Trust the return
                    reason = "returned_starter"
                    confidence = 0.75
        
        # Pattern 6: FIRST CHOICE WHEN FIT
        # High mins/start but not high start rate (injuries)
        if mins_per_start >= 87 and start_rate < 0.8:
            if last_4_starts >= 2:  # Currently fit and playing
                base_mins = 87
                reason = "first_choice"
                confidence = 0.9
    
    # ==================== MANAGER ROTATION ADJUSTMENT ====================
    # Only apply if not already flagged as rotation risk or nailed
    if reason in ["calculated", "regular_starter"] and team_name in ROTATION_MANAGERS:
        rotation_factor = ROTATION_MANAGERS[team_name]
        # Premium players (>£8m) are less likely to be rotated
        if price >= 8:
            rotation_factor = min(1.0, rotation_factor + 0.08)
        base_mins = base_mins * rotation_factor
        reason = f"{reason}_rotation_adj"
    
    # ==================== STABLE MANAGER BOOST ====================
    if reason in ["calculated", "regular_starter", "first_choice"] and team_name in STABLE_MANAGERS:
        stability_factor = STABLE_MANAGERS[team_name]
        base_mins = min(90, base_mins * stability_factor)
    
    # ==================== FIXTURE CONGESTION ====================
    # Check for midweek games (European/cup) that might cause rotation
    # This would need fixture data analysis - simplified here
    if team_name in EUROPEAN_TEAMS_UCL:
        # UCL teams rest players more in "easy" league games
        base_mins = base_mins * 0.95
    
    # ==================== POSITION-SPECIFIC ADJUSTMENTS ====================
    if position == 1:  # GKP
        # Goalkeepers almost never rotate unless injured
        if reason not in ["injured", "doubtful", "unavailable", "suspended"]:
            if starts >= 5 and mins_per_start >= 85:
                base_mins = max(base_mins, 88)
                reason = "gkp_nailed"
    
    # ==================== PRICE-BASED SANITY CHECK ====================
    # Expensive players are bought to play
    if price >= 12 and base_mins < 70 and reason not in ["injured", "unavailable", "suspended", "rotation_risk"]:
        base_mins = max(base_mins, 75)
        reason = f"{reason}_premium"
    
    # ==================== FINAL BOUNDS ====================
    final_mins = round(min(90, max(0, base_mins)), 1)
    
    return final_mins, reason


def get_dgw_adjustment(
    player: Dict,
    team_name: str,
    team_id: int,
    gw: int,
    fixtures: List[Dict],
    base_exp_mins: float
) -> Tuple[float, int, str]:
    """
    For DGWs, calculate adjusted expected minutes accounting for rotation risk.
    
    Returns: (adjusted_mins_per_game, num_games, reason)
    
    European teams and rotation-happy managers rest players in DGWs.
    Nailed starters rotate less than fringe players.
    """
    # Count fixtures this team has in this GW
    gw_fixtures = [
        f for f in fixtures 
        if f.get("event") == gw and (f["team_h"] == team_id or f["team_a"] == team_id)
    ]
    
    num_games = len(gw_fixtures)
    
    if num_games <= 1:
        return base_exp_mins, 1, "sgw"
    
    # === DGW DETECTED ===
    rotation_factor = 1.0
    reason = "dgw"
    
    # Team-level rotation risk
    if team_name in EUROPEAN_TEAMS_UCL:
        rotation_factor = 0.78
        reason = "dgw_ucl_rotation"
    elif team_name in EUROPEAN_TEAMS_UEL:
        rotation_factor = 0.82
        reason = "dgw_uel_rotation"
    elif team_name in EUROPEAN_TEAMS_UECL:
        rotation_factor = 0.85
        reason = "dgw_uecl_rotation"
    elif team_name in ROTATION_MANAGERS:
        rotation_factor = ROTATION_MANAGERS[team_name] - 0.05
        reason = "dgw_manager_rotation"
    else:
        rotation_factor = 0.90
        reason = "dgw_mild_rotation"
    
    # Player-level nailedness adjustment
    total_mins = int(player.get("minutes", 0) or 0)
    starts = int(player.get("starts", 0) or 0)
    
    if starts > 0:
        mins_per_start = total_mins / starts
        if mins_per_start >= 85:
            # Very nailed - reduce rotation penalty
            rotation_factor = min(1.0, rotation_factor + 0.12)
            if "rotation" in reason:
                reason = reason.replace("rotation", "nailed")
        elif mins_per_start >= 78:
            rotation_factor = min(1.0, rotation_factor + 0.06)
    
    # GKPs almost never rotate
    position = player.get("element_type", 4)
    if position == 1 and starts >= 5:
        rotation_factor = min(1.0, rotation_factor + 0.15)
        reason = "dgw_gkp_nailed"
    
    adjusted_mins = base_exp_mins * rotation_factor
    
    return round(adjusted_mins, 1), num_games, reason


def calculate_expected_points(
    player: Dict,
    position: int,
    current_gw: int,
    upcoming_fixtures: List[Dict],
    teams_dict: Dict,
    all_fixtures: List[Dict] = None,
    events: List[Dict] = None,
    player_history: Optional[Dict] = None,
    override_minutes: Optional[float] = None,
    all_players: Optional[List[Dict]] = None
) -> Dict:
    """
    Expert-level expected points calculation.
    
    Components for each fixture:
    1. Appearance points (2 for 60+ mins, 1 for <60)
    2. Clean sheet points (GKP/DEF=4, MID=1) × fixture-specific CS probability
    3. Goal points (GKP/DEF=6, MID=5, FWD=4) × fixture-adjusted xG
    4. Assist points (3) × fixture-adjusted xA
    5. Bonus points (BPS-based prediction)
    6. Save points (GKP only)
    7. DEFCON points (DEF/MID)
    8. Deductions: Yellow cards, own goals
    
    CS probability is calculated per fixture based on:
    - Team's base defensive strength
    - Opponent's attacking strength
    - Home/away adjustment
    
    xGI is adjusted per fixture based on:
    - Opponent's defensive weakness
    - Home/away boost
    """
    total_minutes = int(player.get("minutes", 0) or 0)
    xG = float(player.get("expected_goals", 0) or 0)
    xA = float(player.get("expected_assists", 0) or 0)
    xGC = float(player.get("expected_goals_conceded", 0) or 0)
    points_per_game = float(player.get("points_per_game", 0) or 0)
    yellow_cards = int(player.get("yellow_cards", 0) or 0)
    red_cards = int(player.get("red_cards", 0) or 0)
    own_goals = int(player.get("own_goals", 0) or 0)
    goals_conceded = int(player.get("goals_conceded", 0) or 0)  # Actual goals conceded
    
    team_id = player["team"]
    team_name = teams_dict.get(team_id, {}).get("name", "")
    player_price = float(player.get("now_cost", 50) or 50) / 10  # For price-based FDR dampening
    
    # Calculate home/away split for this player
    home_away_split = home_away_calculator.calculate_splits(player_history, player)
    
    # Get expected minutes
    exp_mins, mins_reason = calculate_expected_minutes(
        player, current_gw, team_name, all_fixtures or [], events or [],
        player_history, override_minutes, all_players=all_players
    )
    
    # Calculate per-90 rates
    mins90 = max(total_minutes / 90.0, 0.1)
    xG90 = xG / mins90
    xA90 = xA / mins90
    xGC90 = xGC / mins90
    og_per_90 = own_goals / mins90 if mins90 > 1 else 0.01
    
    # ==================== PENALTY TAKER BOOST ====================
    # FPL's xG INCLUDES penalty xG, so we don't need to boost for established takers.
    # We only boost when:
    # 1. Player became main penalty taker mid-season (historical xG understates current role)
    # 2. Player's pen share increased significantly
    #
    # The boost is conservative - we're correcting for stale historical data, not adding raw pen xG
    player_id = player.get("id")
    penalty_boost = 0.0
    
    if player_id in PENALTY_TAKERS:
        pen_xg_per_90, confidence = PENALTY_TAKERS[player_id]
        
        # Get player's actual penalty stats from FPL
        penalties_missed = int(player.get("penalties_missed", 0) or 0)
        penalties_saved = int(player.get("penalties_saved", 0) or 0)  # For GKPs
        
        # Estimate penalties taken (goals from pens + misses)
        # FPL doesn't give pen goals directly, but high pen takers have consistent rates
        # We look at the ratio of their xG to expected pen xG contribution
        
        # If player has low minutes (new to team/role), boost more aggressively
        if mins90 < 8:
            # Limited sample - trust the PENALTY_TAKERS designation
            # Apply partial boost weighted by confidence
            penalty_boost = pen_xg_per_90 * confidence * 0.3
        elif mins90 < 15:
            # Medium sample - smaller boost for emerging takers
            expected_pen_contribution = pen_xg_per_90 * 0.76  # 76% pen conversion
            if xG90 < expected_pen_contribution * 0.7:
                # Their xG seems too low for a main pen taker - boost
                penalty_boost = (expected_pen_contribution - xG90 * 0.3) * confidence * 0.25
        # else: Full season sample - trust their historical xG already includes pens
        
        penalty_boost = max(0, min(0.12, penalty_boost))  # Cap at 0.12 xG/90 boost
    
    # ==================== YELLOW CARD MODEL ====================
    # Yellow card probability based on:
    # 1. Historical yellow card rate
    # 2. Position (DEF/MID get more yellows than FWD/GKP)
    # 3. Fouls committed per 90 (FPL doesn't provide this directly, estimate from position)
    
    historical_yellow_rate = yellow_cards / mins90 if mins90 > 1 else 0
    
    # Position-based yellow card baseline (from PL averages)
    position_yellow_baseline = {
        1: 0.02,   # GKP - rarely get yellows
        2: 0.12,   # DEF - tackles, fouls
        3: 0.10,   # MID - mixed
        4: 0.06,   # FWD - fewer defensive duties
    }.get(position, 0.08)
    
    # Blend historical with baseline (more games = trust historical more)
    if mins90 >= 15:
        yellow_per_90 = 0.75 * historical_yellow_rate + 0.25 * position_yellow_baseline
    elif mins90 >= 5:
        yellow_per_90 = 0.50 * historical_yellow_rate + 0.50 * position_yellow_baseline
    else:
        yellow_per_90 = position_yellow_baseline
    
    # Calculate component stats
    defcon_per_90, defcon_prob, defcon_pts_total = calculate_defcon_per_90(player, position)
    saves_per_90, save_pts_per_90 = calculate_saves_per_90(player) if position == 1 else (0, 0)
    bonus_per_90, expected_bonus = calculate_expected_bonus(player, position)
    
    # Get team's CS rate - prefer dynamic FDR data, fallback to hardcoded
    if cache.fdr_data and team_id in cache.fdr_data:
        team_base_cs = cache.fdr_data[team_id].get('cs_probability', 0.22)
    else:
        team_base_cs = TEAM_BASE_CS_RATES.get(team_name, 0.22)
    
    # Points per goal/assist by position
    goal_pts = {1: 6, 2: 6, 3: 5, 4: 4}.get(position, 4)
    cs_pts = {1: 4, 2: 4, 3: 1, 4: 0}.get(position, 0)
    
    # ==================== FIXTURE-BY-FIXTURE CALCULATION ====================
    horizon = upcoming_fixtures[:8]
    
    if not horizon:
        # No fixtures - use base rates
        app_pts = 2.0 if exp_mins >= 60 else (1.0 if exp_mins > 0 else 0)
        cs_prob = team_base_cs
        base_xpts = app_pts + (cs_pts * cs_prob) + (goal_pts * xG90) + (3 * xA90) + expected_bonus
        
        # Estimate expected goals against from team CS rate
        est_xg_against = -math.log(max(0.05, team_base_cs))  # Invert Poisson
        
        if position == 1:
            # For GKPs without fixture data, estimate saves
            est_shots = est_xg_against * 2.8
            est_saves = est_shots * 0.70
            base_xpts += est_saves / 3.0
        if position in [2, 3]:
            base_xpts += defcon_prob * DEFCON_POINTS
        
        # Deductions
        base_xpts -= yellow_per_90 * 1.0  # -1 per yellow
        if position in [1, 2]:
            base_xpts -= og_per_90 * 2.0  # -2 per OG
            base_xpts -= calculate_goals_conceded_penalty(est_xg_against)  # Poisson-based
        
        minutes_factor = exp_mins / 90.0
        total_xpts = base_xpts * minutes_factor
        
        return {
            "xpts": round(total_xpts, 2),
            "xpts_per_90": round(base_xpts, 2),
            "expected_minutes": exp_mins,
            "minutes_reason": mins_reason,
            "minutes_factor": round(minutes_factor, 3),
            "defcon_per_90": defcon_per_90,
            "defcon_prob": defcon_prob,
            "defcon_pts_total": defcon_pts_total,
            "saves_per_90": saves_per_90,
            "save_pts_per_90": save_pts_per_90,
            "bonus_per_90": bonus_per_90,
            "expected_bonus": expected_bonus,
            "xGC_per_90": round(xGC90, 2),
            "cs_prob": round(cs_prob, 2),
            "xG_per_90": round(xG90, 2),
            "xA_per_90": round(xA90, 2),
        }
    
    # Calculate weighted xPts across fixtures
    weights_used = FIXTURE_WEIGHTS[:len(horizon)]
    total_weight = sum(weights_used)
    
    total_xpts = 0
    avg_cs_prob = 0
    
    # Pre-calculate team's defensive baseline (needed for both loop and final calc)
    if cache.fdr_data and team_id in cache.fdr_data:
        team_xga_baseline = cache.fdr_data[team_id].get('blended_xga', 1.3)
    else:
        # Fallback: estimate from CS rate using inverse Poisson
        team_xga_baseline = -math.log(max(0.08, team_base_cs))
    
    for i, fix in enumerate(horizon):
        weight = weights_used[i] / total_weight
        opponent_id = fix.get("opponent_id")
        is_home = fix.get("is_home", True)
        opponent_name = teams_dict.get(opponent_id, {}).get("name", "")
        fixture_gw = fix.get("gameweek", 0)
        
        # ==================== CHECK FOR FIXTURE-SPECIFIC xG ====================
        # If we have fixture-level projections (GW+1, GW+2), use them with high weight
        # Weighting: GW+1 = 65% fixture xG, GW+2 = 55% fixture xG
        
        fixture_xg_data = None
        fixture_xg_weight = 0
        
        # Look up fixture xG in cache (try both home/away orientations)
        if is_home:
            fixture_key = (team_id, opponent_id, fixture_gw)
        else:
            fixture_key = (opponent_id, team_id, fixture_gw)
        
        if fixture_key in cache.fixture_xg:
            fixture_xg_data = cache.fixture_xg[fixture_key]
            # Weight based on how far out the fixture is
            if i == 0:  # Next GW
                fixture_xg_weight = 0.65
            elif i == 1:  # GW+2
                fixture_xg_weight = 0.55
            else:
                fixture_xg_weight = 0.40  # GW+3 and beyond (if available)
        
        # ==================== FIXTURE-SPECIFIC CS PROBABILITY ====================
        # CS probability uses Poisson: P(0 goals) = e^(-expected_goals_against)
        
        if fixture_xg_data and fixture_xg_weight > 0:
            # USE FIXTURE-SPECIFIC xG PROJECTION
            # This is the highest-signal data for near-term predictions
            if is_home:
                fixture_specific_xga = fixture_xg_data.get("away_xg", 1.3)  # What away team is expected to score
            else:
                fixture_specific_xga = fixture_xg_data.get("home_xg", 1.5)  # What home team is expected to score
            
            # Blend fixture xG with team strength model
            # fixture_xg_weight% from fixture projection, remainder from team strength
            team_model_xga = team_xga_baseline * (opp_xg_per_game / LEAGUE_AVG_GOALS_PER_GAME)
            if is_home:
                team_model_xga *= 0.85 * AWAY_ATTACK_PENALTY
            else:
                team_model_xga *= 1.18 * HOME_ATTACK_BOOST
            
            expected_goals_against = (fixture_xg_weight * fixture_specific_xga + 
                                     (1 - fixture_xg_weight) * team_model_xga)
            data_source_for_fixture = "fixture_xg+model"
        else:
            # FALLBACK TO TEAM STRENGTH MODEL
            # Note: team_xga_baseline was calculated before the loop
            
            # Get opponent's attacking strength
            if cache.fdr_data and opponent_id in cache.fdr_data:
                opp_xg_per_game = cache.fdr_data[opponent_id].get('blended_xg', LEAGUE_AVG_GOALS_PER_GAME)
            else:
                opp_xg_per_game = TEAM_BASE_XG.get(opponent_name, LEAGUE_AVG_GOALS_PER_GAME)
            
            # Calculate opponent's attacking multiplier relative to league average
            opp_attack_multiplier = opp_xg_per_game / LEAGUE_AVG_GOALS_PER_GAME
            
            # Apply home/away adjustment to the fixture
            # When you're HOME: opponent (away) attacks weaker, you defend stronger
            # When you're AWAY: opponent (home) attacks stronger, you defend weaker
            if is_home:
                ha_defensive_factor = 0.85  # You concede ~15% less at home
                opp_attack_multiplier *= AWAY_ATTACK_PENALTY  # 0.87 - they're away
            else:
                ha_defensive_factor = 1.18  # You concede ~18% more away
                opp_attack_multiplier *= HOME_ATTACK_BOOST  # 1.19 - they're at home
            
            # Final expected goals against for THIS FIXTURE
            # = Your baseline xGA × opponent attack quality × home/away factor
            expected_goals_against = team_xga_baseline * opp_attack_multiplier * ha_defensive_factor
            data_source_for_fixture = "team_model"
        
        # Bound expected goals against to realistic range (0.4 to 3.0)
        expected_goals_against = max(0.4, min(3.0, expected_goals_against))
        
        # CS probability using Poisson
        fixture_cs_prob = math.exp(-expected_goals_against)
        fixture_cs_prob = min(0.55, max(0.03, fixture_cs_prob))  # Bound between 3-55%
        
        avg_cs_prob += fixture_cs_prob * weight
        
        # ==================== FIXTURE-SPECIFIC xGI ====================
        # Adjust player's xG/xA based on opponent's defensive weakness
        # Use fixture xG if available for attack projections too
        
        if fixture_xg_data and fixture_xg_weight > 0:
            # Use fixture-specific attacking projection
            if is_home:
                fixture_specific_xg_for = fixture_xg_data.get("home_xg", 1.5)  # What we're expected to score
            else:
                fixture_specific_xg_for = fixture_xg_data.get("away_xg", 1.0)
            
            # Convert fixture xG to multiplier relative to our baseline
            team_base_xg = cache.fdr_data.get(team_id, {}).get('blended_xg', LEAGUE_AVG_GOALS_PER_GAME)
            if team_base_xg > 0:
                attack_multiplier = fixture_specific_xg_for / team_base_xg
            else:
                attack_multiplier = 1.0
            
            # Blend with FDR-based multiplier
            if cache.fdr_data and opponent_id in cache.fdr_data:
                opp_def_fdr = cache.fdr_data[opponent_id].get("defence_fdr", 5)
                # Use price-adjusted FDR multiplier
                fdr_attack_mult = get_price_adjusted_fdr_multiplier(round(opp_def_fdr), player_price)
                if is_home:
                    fdr_attack_mult *= HOME_ATTACK_BOOST
                else:
                    fdr_attack_mult *= AWAY_ATTACK_PENALTY
                
                # Blend: fixture_xg_weight from fixture, rest from FDR model
                attack_multiplier = fixture_xg_weight * attack_multiplier + (1 - fixture_xg_weight) * fdr_attack_mult
        else:
            # Fallback to FDR-based multiplier
            if cache.fdr_data and opponent_id in cache.fdr_data:
                opp_def_fdr = cache.fdr_data[opponent_id].get("defence_fdr", 5)
            else:
                # Estimate from CS rate (lower CS = weaker defence)
                opp_cs_rate = TEAM_BASE_CS_RATES.get(opponent_name, 0.22)
                # Convert to FDR: 10% CS = weak (FDR 3), 35% CS = strong (FDR 8)
                opp_def_fdr = 3 + (opp_cs_rate / 0.35) * 5
                opp_def_fdr = min(10, max(1, opp_def_fdr))
            
            # FDR multiplier for attacking returns - use price-adjusted version
            attack_multiplier = get_price_adjusted_fdr_multiplier(round(opp_def_fdr), player_price)
            
            # Home/away adjustment for attacking (asymmetric)
            if is_home:
                attack_multiplier *= HOME_ATTACK_BOOST  # 1.19
            else:
                attack_multiplier *= AWAY_ATTACK_PENALTY  # 0.87
        
        # CAP TOTAL FIXTURE BOOST AT 22% - reasonable swing for home vs weak defence
        attack_multiplier = min(1.22, max(0.78, attack_multiplier))
        
        # ==================== HOME/AWAY SPLIT FOR xGI ====================
        # If player has sufficient home/away history, blend with split-specific rates
        fixture_xG90_base = xG90
        fixture_xA90_base = xA90
        
        if home_away_split.has_sufficient_data:
            ha_config = MODEL_CONFIG["home_away"]
            if is_home:
                split_xG90 = home_away_split.home_xG90
                split_xA90 = home_away_split.home_xA90
            else:
                split_xG90 = home_away_split.away_xG90
                split_xA90 = home_away_split.away_xA90
            
            # Blend split with overall (60% split, 40% overall)
            fixture_xG90_base = ha_config.split_weight * split_xG90 + (1 - ha_config.split_weight) * xG90
            fixture_xA90_base = ha_config.split_weight * split_xA90 + (1 - ha_config.split_weight) * xA90
        
        # Apply penalty boost to xG if player is a penalty taker
        fixture_xG90 = (fixture_xG90_base + penalty_boost) * attack_multiplier
        fixture_xA90 = fixture_xA90_base * attack_multiplier
        
        # ==================== FIXTURE-SPECIFIC BONUS ====================
        # Bonus is competitive within each match - doesn't scale as aggressively as xG
        # Good fixtures help but effect is muted (50% of attack multiplier effect)
        bonus_multiplier = 1.0 + (attack_multiplier - 1.0) * 0.5
        fixture_bonus = expected_bonus * bonus_multiplier
        
        # ==================== CALCULATE FIXTURE xPTS ====================
        # Appearance points (assume 60+ mins for simplicity in per-90 calc)
        fixture_app_pts = 2.0
        
        # Goal and assist points
        fixture_goal_pts = goal_pts * fixture_xG90
        fixture_assist_pts = 3 * fixture_xA90
        
        # CS points
        fixture_cs_pts = cs_pts * fixture_cs_prob
        
        # ==================== FIXTURE-SPECIFIC SAVES (GKP) - Calculate First ====================
        # Saves are correlated with xG against - more shots = more saves but LOWER CS
        # Need to calculate saves BEFORE bonus since GKP bonus depends on saves
        fixture_expected_saves = 0
        fixture_save_pts = 0
        
        if position == 1:
            # Constants for save calculation
            SHOTS_ON_TARGET_PER_XG = 2.8  # ~2.8 shots on target per xG (league average)
            
            # Calculate keeper's save rate from their historical data
            if xGC90 > 0.1 and saves_per_90 > 0:
                est_shots_on_target_per_90 = xGC90 * SHOTS_ON_TARGET_PER_XG
                keeper_save_rate = min(0.80, saves_per_90 / max(1, est_shots_on_target_per_90))
            else:
                keeper_save_rate = 0.70  # League average save rate
            
            # Expected shots on target THIS FIXTURE (based on expected_goals_against)
            fixture_shots_on_target = expected_goals_against * SHOTS_ON_TARGET_PER_XG
            
            # Expected saves THIS FIXTURE
            fixture_expected_saves = fixture_shots_on_target * keeper_save_rate
            
            # Save points (1 per 3 saves)
            fixture_save_pts = fixture_expected_saves / 3.0
        
        # ==================== FIXTURE-SPECIFIC BONUS ====================
        # Bonus has two components:
        # 1. Attack-driven (goals/assists) - correlates with opponent defensive weakness
        # 2. CS-driven (GKP/DEF get +12 BPS, MID gets +6 BPS for clean sheet)
        
        # Attack-driven bonus
        attack_bonus = expected_bonus * attack_multiplier
        
        if position == 1:  # GKP
            # When a GKP keeps a CS, bonus depends on saves made in that game
            # Saves in CS games correlate with opponent quality - stronger opponents
            # still create chances even when they don't score
            # 
            # Key insight: expected_goals_against tells us opponent strength.
            # In CS games, the keeper typically faces 60-75% of normal shot volume
            # (some games are 0-0 due to few chances, others are lucky CS)
            
            # Calculate expected saves in THIS fixture's CS scenario
            # More dangerous opponents = more saves even in CS games
            cs_shot_ratio = 0.65 + (expected_goals_against / 4.0) * 0.15  # 0.65-0.80 range
            cs_shot_ratio = min(0.80, max(0.55, cs_shot_ratio))
            
            expected_saves_in_cs = fixture_expected_saves * cs_shot_ratio
            
            # BPS calculation for CS scenario: +12 (CS) + ~2 per save
            cs_bps = 12 + (expected_saves_in_cs * 2)
            
            # Convert BPS to expected bonus
            if cs_bps >= 28:
                bonus_given_cs = 2.5
            elif cs_bps >= 24:
                bonus_given_cs = 2.0
            elif cs_bps >= 20:
                bonus_given_cs = 1.3
            elif cs_bps >= 17:
                bonus_given_cs = 0.8
            else:
                bonus_given_cs = 0.4
            
            # CS bonus: probability × expected bonus if CS happens
            cs_bonus_component = fixture_cs_prob * bonus_given_cs
            
            # Non-CS games: saves still contribute to bonus but much harder to win
            # ~2 BPS per save, need 25+ to compete, so 5+ saves for any bonus chance
            non_cs_save_bonus = (1 - fixture_cs_prob) * max(0, (fixture_expected_saves - 3) * 0.06)
            
            fixture_bonus_pts = cs_bonus_component + non_cs_save_bonus
            
        elif position == 2:  # DEF
            # DEF bonus from CS + attacking returns
            cs_bonus_component = fixture_cs_prob * 0.9  # +12 BPS for CS → ~0.9 expected bonus
            fixture_bonus_pts = attack_bonus * 0.5 + cs_bonus_component
            
        elif position == 3:  # MID
            # MID bonus mostly attack-driven, small CS component
            cs_bonus_component = fixture_cs_prob * 0.4  # +6 BPS for CS
            fixture_bonus_pts = attack_bonus * 0.85 + cs_bonus_component * 0.15
            
        else:  # FWD
            fixture_bonus_pts = attack_bonus  # FWD bonus purely attack-driven
        
        # DEFCON
        fixture_defcon_pts = (defcon_prob * DEFCON_POINTS) if position in [2, 3] else 0
        
        # ==================== DEDUCTIONS ====================
        
        # Yellow cards: -1 per yellow
        fixture_yellow_deduction = yellow_per_90 * 1.0
        
        # Own goals: -2 per OG (GKP/DEF only, but can happen to anyone technically)
        fixture_og_deduction = (og_per_90 * 2.0) if position in [1, 2] else (og_per_90 * 2.0 * 0.2)
        
        # Goals conceded: -1 per 2 goals for GKP/DEF (stepped rule)
        # Use Poisson model to properly calculate expected penalty
        if position in [1, 2]:
            fixture_goals_conceded_deduction = calculate_goals_conceded_penalty(expected_goals_against)
        else:
            fixture_goals_conceded_deduction = 0
        
        # Total for this fixture (per 90)
        fixture_xpts_per_90 = (
            fixture_app_pts +
            fixture_goal_pts +
            fixture_assist_pts +
            fixture_cs_pts +
            fixture_bonus_pts +
            fixture_save_pts +
            fixture_defcon_pts -
            fixture_yellow_deduction -
            fixture_og_deduction -
            fixture_goals_conceded_deduction
        )
        
        # Apply horizon regression - further fixtures have more uncertainty
        # This is the RIGHT place for regression - on future predictions, not baseline xGI
        horizon_regression = HORIZON_REGRESSION.get(i + 1, 0.93)
        fixture_xpts_per_90 *= horizon_regression
        
        # Add to weighted total
        total_xpts += fixture_xpts_per_90 * weight
    
    # Scale by number of fixtures in horizon
    total_xpts_per_90 = total_xpts * len(horizon)
    
    # Calculate clean per-90 rate
    model_per_90 = total_xpts_per_90 / max(1, len(horizon))
    
    # ==================== APPLY MINUTES FACTOR ====================
    # FPL points have discrete and continuous components:
    #
    # DISCRETE (conditional on playing threshold):
    # - Appearance: 2 pts for 60+ mins, 1 pt for 1-59 mins
    # - Clean Sheet: 4/1/0 pts - requires 60+ mins for GKP/DEF, always for MID
    # - Bonus: Competitive BPS system - extremely unlikely without 60+ mins
    #
    # CONTINUOUS (scale with minutes):
    # - Goals, Assists, Saves (proportional to time on pitch)
    # - Yellow/Red cards, Own goals (exposure-based)
    
    minutes_factor = exp_mins / 90.0
    
    # Probability of 60+ mins (logistic curve)
    if exp_mins >= 85:
        prob_60_plus = 0.95
    elif exp_mins >= 75:
        prob_60_plus = 0.85
    elif exp_mins >= 60:
        prob_60_plus = 0.50 + (exp_mins - 60) * 0.014  # 50% at 60, 85% at 85
    elif exp_mins >= 30:
        prob_60_plus = (exp_mins - 30) / 60  # 0% at 30, 50% at 60
    else:
        prob_60_plus = 0.0
    
    # Probability of playing at all (for 1-pt appearance)
    prob_plays = min(1.0, exp_mins / 15) if exp_mins > 0 else 0
    
    # ==================== REBUILD xPts WITH PROPER DISCRETIZATION ====================
    
    # Appearance points (discrete)
    appearance_xpts_per_fix = (prob_60_plus * 2.0) + ((prob_plays - prob_60_plus) * 1.0)
    
    # CS points (discrete for GKP/DEF - requires 60+ mins)
    if position in [1, 2]:
        # CS only counts if you play 60+ mins
        cs_xpts_per_fix = avg_cs_prob * cs_pts * prob_60_plus
    else:
        # MID CS point doesn't require 60+ mins
        cs_xpts_per_fix = avg_cs_prob * cs_pts * prob_plays
    
    # Bonus points (discrete - almost impossible without significant minutes)
    # BPS accumulates during play, so very unlikely to get bonus with <45 mins
    if exp_mins >= 70:
        bonus_discount = 1.0
    elif exp_mins >= 55:
        bonus_discount = 0.85  # Slight discount - might miss bonus cutoff
    elif exp_mins >= 40:
        bonus_discount = 0.50  # Significant discount - sub unlikely to win bonus
    elif exp_mins >= 25:
        bonus_discount = 0.20  # Very unlikely
    else:
        bonus_discount = 0.05  # Almost never
    bonus_xpts_per_fix = expected_bonus * bonus_discount
    
    # Goal and assist points (continuous - scale with minutes)
    xgi_xpts_per_fix = ((goal_pts * (xG90 + penalty_boost)) + (3 * xA90)) * minutes_factor
    
    # Save points (continuous - GKP only)
    save_xpts_per_fix = (save_pts_per_90 * minutes_factor) if position == 1 else 0
    
    # DEFCON points (discrete - requires threshold in 90 mins)
    # Scale down for reduced minutes as less time to accumulate actions
    defcon_xpts_per_fix = (defcon_prob * DEFCON_POINTS * min(1.0, minutes_factor * 1.1)) if position in [2, 3] else 0
    
    # Deductions (continuous - exposure-based)
    yellow_deduction_per_fix = yellow_per_90 * minutes_factor
    og_deduction_per_fix = (og_per_90 * 2.0 * minutes_factor) if position in [1, 2] else (og_per_90 * 2.0 * 0.2 * minutes_factor)
    
    # Goals conceded deduction (uses Poisson, already calculated in fixture loop)
    # Use team's defensive baseline (calculated before loop)
    if position in [1, 2]:
        gc_deduction_per_fix = calculate_goals_conceded_penalty(team_xga_baseline) * prob_60_plus  # Only if play 60+
    else:
        gc_deduction_per_fix = 0
    
    # Sum all components per fixture
    xpts_per_fixture = (
        appearance_xpts_per_fix +
        cs_xpts_per_fix +
        bonus_xpts_per_fix +
        xgi_xpts_per_fix +
        save_xpts_per_fix +
        defcon_xpts_per_fix -
        yellow_deduction_per_fix -
        og_deduction_per_fix -
        gc_deduction_per_fix
    )
    
    # Final xPts = per-fixture rate × number of fixtures
    final_xpts = xpts_per_fixture * len(horizon)
    
    # Calculate ceiling and floor using variance model (historical stdev)
    # This replaces the flat multipliers with proper variance-based calculations
    player_price = float(player.get("now_cost", 50) or 50) / 10
    
    # Get next fixture info for ceiling adjustment
    next_fixture_fdr = 5
    next_fixture_home = True
    if horizon:
        next_fixture_fdr = horizon[0].get("difficulty", 5)
        next_fixture_home = horizon[0].get("is_home", True)
    
    variance_result = variance_model.calculate_variance(
        player_history=player_history,
        player_data=player,
        position_id=position,
        xpts=xpts_per_fixture,  # Per-fixture for variance calc
        fixture_fdr=next_fixture_fdr,
        is_home=next_fixture_home
    )
    
    # Scale ceiling/floor to horizon
    xpts_ceiling = variance_result.ceiling * len(horizon)
    xpts_floor = variance_result.floor * len(horizon)
    
    # Track variance info for debugging
    variance_source = variance_result.data_source
    std_dev = variance_result.std_dev
    
    return {
        "xpts": round(final_xpts, 2),
        "xpts_per_90": round(model_per_90, 2),
        "xpts_ceiling": round(xpts_ceiling, 2),
        "xpts_floor": round(xpts_floor, 2),
        "expected_minutes": exp_mins,
        "minutes_reason": mins_reason,
        "minutes_factor": round(minutes_factor, 3),
        "prob_60_plus": round(prob_60_plus, 3),
        "defcon_per_90": defcon_per_90,
        "defcon_prob": defcon_prob,
        "defcon_pts_total": defcon_pts_total,
        "saves_per_90": saves_per_90,
        "save_pts_per_90": save_pts_per_90,
        "bonus_per_90": bonus_per_90,
        "expected_bonus": expected_bonus,
        "xGC_per_90": round(xGC90, 2),
        "cs_prob": round(avg_cs_prob, 2),
        "xG_per_90": round(xG90, 2),
        "xA_per_90": round(xA90, 2),
        # New fields from refactored models
        "std_dev": round(std_dev, 2),
        "variance_source": variance_source,
        "home_away_split_used": home_away_split.has_sufficient_data,
    }


def calculate_captain_score(
    player: Dict,
    position_id: int,
    base_xpts: float,
    next_fixture: Optional[Dict]
) -> Dict:
    """
    Calculate captaincy score factoring in ceiling potential AND differential value.
    
    For rank climbing, we want:
    1. High ceiling potential (attackers vs weak defences at home)
    2. Differential upside (low ownership = rank gain if they haul)
    
    Returns dict with captain_score, ceiling_mult, and diff_boost.
    """
    ceiling_mult = 1.0
    diff_boost = 0.0
    
    # Position multiplier - attackers have higher point ceilings
    position_mults = {
        1: 0.70,   # GKP - very low ceiling (max ~15 pts realistic)
        2: 0.85,   # DEF - moderate ceiling (CS + goal = 15+ pts)
        3: 1.12,   # MID - high ceiling (5pt goals + CS point + assist potential)
        4: 1.18    # FWD - highest ceiling (brace potential)
    }
    ceiling_mult *= position_mults.get(position_id, 1.0)
    
    # Next fixture analysis for captaincy
    if next_fixture:
        is_home = next_fixture.get("is_home", False)
        opponent_id = next_fixture.get("opponent_id")
        
        # Home advantage increases ceiling
        if is_home:
            ceiling_mult *= 1.10
        
        # Opponent defensive strength affects attacker ceilings
        if position_id in [3, 4] and cache.fdr_data and opponent_id in cache.fdr_data:
            opp_def_fdr = cache.fdr_data[opponent_id].get("defence_fdr", 5)
            if opp_def_fdr <= 3:
                ceiling_mult *= 1.15  # Weak defence = haul potential
            elif opp_def_fdr <= 4:
                ceiling_mult *= 1.08
            elif opp_def_fdr >= 8:
                ceiling_mult *= 0.85  # Strong defence = capped upside
            elif opp_def_fdr >= 7:
                ceiling_mult *= 0.92
    
    # Historical explosiveness - PPG as proxy for haul frequency
    total_points = int(player.get("total_points", 0) or 0)
    starts = int(player.get("starts", 0) or 0)
    if starts >= 5:
        ppg = total_points / starts
        if ppg >= 8.0:
            ceiling_mult *= 1.15  # Elite returner (Haaland, Salah tier)
        elif ppg >= 7.0:
            ceiling_mult *= 1.10
        elif ppg >= 6.0:
            ceiling_mult *= 1.05
        elif ppg >= 5.0:
            ceiling_mult *= 1.0
        elif ppg < 4:
            ceiling_mult *= 0.85  # Historically poor
    
    # ==================== DIFFERENTIAL VALUE ====================
    # In mini-leagues, captaining a differential can swing ranks significantly
    # High EO captain = no rank movement on haul
    # Low EO captain with haul = massive rank gain
    ownership = float(player.get("selected_by_percent", 0) or 0)
    
    if ownership < 5:
        diff_boost = 0.08  # Punt captain - high variance play
    elif ownership < 10:
        diff_boost = 0.05  # Strong differential
    elif ownership < 20:
        diff_boost = 0.02  # Mild differential upside
    elif ownership > 50:
        diff_boost = -0.03  # Template captain - no rank gain, only risk
    elif ownership > 35:
        diff_boost = -0.01  # Popular pick
    
    captain_score = base_xpts * ceiling_mult * (1 + diff_boost)
    
    return {
        "captain_score": round(captain_score, 2),
        "ceiling_mult": round(ceiling_mult, 3),
        "diff_boost": round(diff_boost, 3),
        "ownership": round(ownership, 1),
    }


def get_player_upcoming_fixtures(
    player_team: int, fixtures: List[Dict], current_gw: int, gw_end: int, teams_dict: Dict,
    position_id: int = 4  # Default to FWD perspective for attack FDR
) -> List[Dict]:
    upcoming = []
    for fix in fixtures:
        gw = fix.get("event")
        if gw and current_gw <= gw <= gw_end:
            if fix["team_h"] == player_team:
                opponent_id = fix["team_a"]
                # Use our computed FDR (composite of attack + defence)
                fdr = get_fixture_fdr(opponent_id, True, position_id)
                upcoming.append({
                    "gameweek": gw,
                    "opponent": teams_dict.get(opponent_id, {}).get("short_name", "???"),
                    "opponent_id": opponent_id,
                    "is_home": True,
                    "difficulty": fdr,  # Our computed FDR 1-10
                    "fpl_difficulty": fix.get("team_h_difficulty", 3)  # Original FPL 1-5
                })
            elif fix["team_a"] == player_team:
                opponent_id = fix["team_h"]
                fdr = get_fixture_fdr(opponent_id, False, position_id)
                upcoming.append({
                    "gameweek": gw,
                    "opponent": teams_dict.get(opponent_id, {}).get("short_name", "???"),
                    "opponent_id": opponent_id,
                    "is_home": False,
                    "difficulty": fdr,
                    "fpl_difficulty": fix.get("team_a_difficulty", 3)
                })
    return sorted(upcoming, key=lambda x: x["gameweek"])


# ============ MINUTES OVERRIDE ENDPOINTS ============

@app.post("/api/minutes-override")
async def set_minutes_override_endpoint(override: MinutesOverride):
    """
    Set expected minutes override for a player.
    
    If manager_id is ADMIN_MANAGER_ID (616495), the override becomes GLOBAL 
    and applies to all users. Otherwise, it's scoped to that manager only.
    """
    # Admin overrides are global (manager_id=None in storage)
    if override.manager_id == ADMIN_MANAGER_ID:
        effective_manager_id = None  # Global override
        scope = "global"
    else:
        effective_manager_id = override.manager_id
        scope = "local"
    
    cache.set_minutes_override(
        player_id=override.player_id,
        minutes=override.expected_minutes,
        manager_id=effective_manager_id
    )
    return {
        "status": "ok",
        "player_id": override.player_id,
        "expected_minutes": override.expected_minutes,
        "manager_id": override.manager_id,
        "scope": scope
    }


@app.delete("/api/minutes-override/{player_id}")
async def remove_minutes_override_endpoint(player_id: int, manager_id: Optional[int] = None):
    """Remove minutes override for a player. Admin deletions remove global overrides."""
    # Admin deletions target global overrides
    if manager_id == ADMIN_MANAGER_ID:
        manager_id = None
    
    deleted = cache.delete_minutes_override(player_id, manager_id)
    return {"status": "ok", "player_id": player_id, "deleted": deleted}


@app.get("/api/minutes-overrides")
async def get_minutes_overrides_endpoint(manager_id: Optional[int] = None):
    """Get all minutes overrides, optionally filtered by manager."""
    if manager_id is not None:
        # Return only manager-specific overrides
        overrides = {
            pid: mins for (mid, pid), mins in cache.minutes_overrides.items()
            if mid == manager_id
        }
    else:
        # Return all overrides with their scope
        overrides = {
            f"{mid or 'global'}:{pid}": mins
            for (mid, pid), mins in cache.minutes_overrides.items()
        }
    return {"overrides": overrides}


class BulkMinutesImport(BaseModel):
    """Bulk import predicted minutes from external sources."""
    predictions: Dict[int, float]  # player_id -> predicted_minutes
    source: Optional[str] = "manual"


@app.post("/api/predicted-minutes/bulk")
async def import_predicted_minutes(data: BulkMinutesImport):
    """
    Bulk import predicted minutes from external sources like FPL Review.
    These will be used as fallback when no user override exists.
    """
    count = 0
    for player_id, minutes in data.predictions.items():
        if 0 <= minutes <= 90:
            cache.predicted_minutes[player_id] = minutes
            count += 1
    
    cache.predicted_minutes_last_update = datetime.now()
    
    return {
        "status": "ok",
        "imported": count,
        "source": data.source,
        "updated_at": cache.predicted_minutes_last_update.isoformat()
    }


@app.get("/api/predicted-minutes")
async def get_predicted_minutes():
    """Get all predicted minutes currently cached."""
    return {
        "predictions": cache.predicted_minutes,
        "count": len(cache.predicted_minutes),
        "last_update": cache.predicted_minutes_last_update.isoformat() if cache.predicted_minutes_last_update else None,
        "is_stale": cache.predicted_minutes_is_stale()
    }


@app.delete("/api/predicted-minutes")
async def clear_predicted_minutes():
    """Clear all predicted minutes."""
    cache.predicted_minutes.clear()
    cache.predicted_minutes_last_update = None
    return {"status": "ok", "message": "All predicted minutes cleared"}


# ==================== TEAM STRENGTH ADMIN (Analytic FPL Data) ====================

class TeamStrengthData(BaseModel):
    """
    Team strength data from Analytic FPL.
    
    Attack = Adjusted xG For (opponent-quality adjusted)
    Defence = Adjusted xGA (opponent-quality adjusted)
    Deltas = Recent change in strength
    Trends = Directional momentum
    """
    team: str
    adjxg_for: float  # Attack column
    adjxg_ag: float   # Defence column
    # Optional trend/delta data
    attack_delta: Optional[float] = None   # Attack Δ
    defence_delta: Optional[float] = None  # Defence Δ
    attack_trend: Optional[float] = None   # Att Trend
    defence_trend: Optional[float] = None  # Def Trend


class TeamStrengthBulkImport(BaseModel):
    """Bulk import team strength data."""
    teams: List[TeamStrengthData]
    manager_id: int  # Must be admin


class FixtureXGData(BaseModel):
    """Fixture-specific xG projections for near-term predictions."""
    home_team: str
    away_team: str
    gameweek: int
    home_xg: float  # Projected xG for home team
    away_xg: float  # Projected xG for away team
    

class FixtureXGBulkImport(BaseModel):
    """Bulk import fixture xG projections."""
    fixtures: List[FixtureXGData]
    manager_id: int  # Must be admin
    source: Optional[str] = "analytic_fpl"


@app.post("/api/team-strength")
async def update_team_strength(data: TeamStrengthBulkImport):
    """
    Update team strength data from Analytic FPL.
    Admin only (manager_id must be 616495).
    
    Weighting strategy:
    - 55-60% Analytic FPL adjusted xG (stable baseline)
    - 25-30% Understat recent form (reactive)
    - 10-15% Trend adjustment (momentum)
    
    The trend data (Δ and Trend columns) captures teams improving/declining.
    """
    if data.manager_id != ADMIN_MANAGER_ID:
        raise HTTPException(status_code=403, detail="Admin access required (manager_id=616495)")
    
    updated = []
    skipped = []
    
    for team_data in data.teams:
        team_name = team_data.team.strip()
        team_id = ANALYTIC_FPL_TO_ID.get(team_name)
        
        if team_id is None:
            logger.warning(f"Unknown team name: {team_name}")
            skipped.append(team_name)
            continue
        
        cache.manual_team_strength[team_id] = {
            "team_name": team_name,
            "adjxg_for": team_data.adjxg_for,
            "adjxg_ag": team_data.adjxg_ag,
            # Trend data for momentum adjustments
            "attack_delta": team_data.attack_delta,
            "defence_delta": team_data.defence_delta,
            "attack_trend": team_data.attack_trend,
            "defence_trend": team_data.defence_trend,
        }
        updated.append(team_name)
    
    cache.manual_team_strength_last_update = datetime.now()
    
    # Force FDR refresh to incorporate new data
    cache.fdr_last_update = None
    await refresh_fdr_data(force=True)
    
    return {
        "status": "ok",
        "updated_teams": updated,
        "skipped_teams": skipped,
        "count": len(updated),
        "updated_at": cache.manual_team_strength_last_update.isoformat(),
        "weighting_info": {
            "analytic_fpl_weight": 0.55,
            "understat_form_weight": 0.30,
            "trend_weight": 0.15,
            "note": "Trend data (Δ) applies momentum adjustment to near-term predictions"
        }
    }


@app.post("/api/fixture-xg")
async def import_fixture_xg(data: FixtureXGBulkImport):
    """
    Import fixture-specific xG projections for GW+1 and GW+2.
    Admin only.
    
    This is the highest-signal data for near-term predictions.
    When available, fixture xG is weighted:
    - GW+1: 65% fixture xG, 20% team strength, 15% form
    - GW+2: 55% fixture xG, 25% team strength, 20% form
    """
    if data.manager_id != ADMIN_MANAGER_ID:
        raise HTTPException(status_code=403, detail="Admin access required (manager_id=616495)")
    
    fpl_data = await fetch_fpl_data()
    teams_by_name = {}
    for t in fpl_data["teams"]:
        teams_by_name[t["name"].lower()] = t["id"]
        teams_by_name[t["short_name"].lower()] = t["id"]
    
    # Add common aliases
    aliases = {
        "man city": "Manchester City", "man utd": "Manchester United",
        "spurs": "Tottenham", "wolves": "Wolverhampton Wanderers",
        "brighton": "Brighton and Hove Albion", "forest": "Nottingham Forest",
        "nott'm forest": "Nottingham Forest", "nottingham forest": "Nottingham Forest",
    }
    for alias, full_name in aliases.items():
        if full_name.lower() in teams_by_name:
            teams_by_name[alias.lower()] = teams_by_name[full_name.lower()]
    
    # Also use ANALYTIC_FPL_TO_ID for matching
    for name, team_id in ANALYTIC_FPL_TO_ID.items():
        teams_by_name[name.lower()] = team_id
    
    imported = []
    errors = []
    
    for fix in data.fixtures:
        home_name = fix.home_team.strip().lower()
        away_name = fix.away_team.strip().lower()
        
        home_id = teams_by_name.get(home_name)
        away_id = teams_by_name.get(away_name)
        
        if home_id is None:
            errors.append(f"Unknown home team: {fix.home_team}")
            continue
        if away_id is None:
            errors.append(f"Unknown away team: {fix.away_team}")
            continue
        
        key = (home_id, away_id, fix.gameweek)
        cache.fixture_xg[key] = {
            "home_xg": fix.home_xg,
            "away_xg": fix.away_xg,
            "home_team": fix.home_team,
            "away_team": fix.away_team,
            "gw": fix.gameweek,
        }
        imported.append(f"GW{fix.gameweek}: {fix.home_team} vs {fix.away_team}")
    
    cache.fixture_xg_last_update = datetime.now()
    
    return {
        "status": "ok",
        "imported": len(imported),
        "imported_fixtures": imported,
        "errors": errors if errors else None,
        "source": data.source,
        "updated_at": cache.fixture_xg_last_update.isoformat(),
    }


@app.get("/api/fixture-xg")
async def get_fixture_xg(gw: Optional[int] = None):
    """Get cached fixture xG projections."""
    if gw:
        fixtures = {
            f"{v['home_team']} vs {v['away_team']}": v 
            for k, v in cache.fixture_xg.items() 
            if k[2] == gw
        }
    else:
        fixtures = {
            f"GW{k[2]}: {v['home_team']} vs {v['away_team']}": v 
            for k, v in cache.fixture_xg.items()
        }
    
    return {
        "fixtures": fixtures,
        "count": len(fixtures),
        "last_update": cache.fixture_xg_last_update.isoformat() if cache.fixture_xg_last_update else None,
    }


@app.delete("/api/fixture-xg")
async def clear_fixture_xg(manager_id: int):
    """Clear fixture xG cache (admin only)."""
    if manager_id != ADMIN_MANAGER_ID:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    cache.fixture_xg.clear()
    cache.fixture_xg_last_update = None
    return {"status": "ok", "message": "Fixture xG cache cleared"}


@app.get("/api/team-strength")
async def get_team_strength():
    """Get current team strength data (manual + computed)."""
    await refresh_fdr_data()
    
    result = {}
    for team_id, fdr in cache.fdr_data.items():
        team_name = FPL_ID_TO_UNDERSTAT.get(team_id, f"Team {team_id}")
        manual = cache.manual_team_strength.get(team_id, {})
        
        result[team_id] = {
            "team_name": team_name,
            "manual_adjxg_for": manual.get("adjxg_for"),
            "manual_adjxg_ag": manual.get("adjxg_ag"),
            "form_xg": fdr.get("form_xg"),
            "form_xga": fdr.get("form_xga"),
            "blended_xg": fdr.get("blended_xg"),
            "blended_xga": fdr.get("blended_xga"),
            "cs_probability": fdr.get("cs_probability"),
            "attack_fdr": fdr.get("attack_fdr"),
            "defence_fdr": fdr.get("defence_fdr"),
        }
    
    return {
        "teams": result,
        "manual_last_update": cache.manual_team_strength_last_update.isoformat() if cache.manual_team_strength_last_update else None,
        "fdr_last_update": cache.fdr_last_update.isoformat() if cache.fdr_last_update else None,
    }


@app.delete("/api/team-strength")
async def clear_team_strength(manager_id: int):
    """Clear manual team strength data (admin only)."""
    if manager_id != ADMIN_MANAGER_ID:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    cache.manual_team_strength.clear()
    cache.manual_team_strength_last_update = None
    
    # Force FDR refresh
    cache.fdr_last_update = None
    await refresh_fdr_data(force=True)
    
    return {"status": "ok", "message": "Team strength data cleared"}


@app.post("/api/predicted-minutes/fetch")
async def fetch_predicted_minutes_from_source(source: str = "rotowire"):
    """
    Fetch predicted minutes from external sources.
    Primary source: Rotowire predicted lineups
    """
    from bs4 import BeautifulSoup
    
    data = await fetch_fpl_data()
    elements = {e["id"]: e for e in data["elements"]}
    teams_dict = {t["id"]: t for t in data["teams"]}
    
    # Build comprehensive name lookup
    elements_by_name = {}
    for e in data["elements"]:
        name = e.get("web_name", "").lower().strip()
        first = e.get("first_name", "").lower().strip()
        second = e.get("second_name", "").lower().strip()
        full_name = f"{first} {second}".strip()
        
        elements_by_name[name] = e["id"]
        elements_by_name[full_name] = e["id"]
        elements_by_name[second] = e["id"]  # Last name only
        
        # Without accents
        import unicodedata
        for n in [name, full_name, second]:
            if n:
                ascii_name = unicodedata.normalize('NFKD', n).encode('ASCII', 'ignore').decode().lower()
                if ascii_name and ascii_name != n:
                    elements_by_name[ascii_name] = e["id"]
    
    predictions = {}
    errors = []
    matched_players = []
    unmatched_players = []
    
    if source == "rotowire":
        try:
            client = await get_http_client()
            response = await client.get(
                "https://www.rotowire.com/soccer/lineups.php?league=EPL",
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                },
                timeout=20.0
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Find all lineup cards
                lineup_cards = soup.find_all('div', class_='lineup__box') or soup.find_all('div', class_='lineup')
                
                if not lineup_cards:
                    # Try alternative selectors
                    lineup_cards = soup.find_all('div', {'class': lambda x: x and 'lineup' in x.lower()})
                
                for card in lineup_cards:
                    # Find all player elements within this card
                    # Rotowire uses different classes for starters vs subs
                    
                    # Try multiple selector patterns
                    starters = card.select('.lineup__player') or card.select('[class*="player"]')
                    
                    for player_elem in starters:
                        # Get player name - could be in various places
                        name_elem = player_elem.select_one('.lineup__player-name') or player_elem.select_one('a') or player_elem
                        if not name_elem:
                            continue
                            
                        player_name = name_elem.get_text(strip=True).lower()
                        if not player_name or len(player_name) < 2:
                            continue
                        
                        # Check if this is a starter or sub based on class/position
                        classes = ' '.join(player_elem.get('class', []))
                        is_starter = 'is-starter' in classes or 'starter' in classes or 'lineup__player--' not in classes
                        is_sub = 'is-sub' in classes or 'sub' in classes or 'bench' in classes
                        is_confirmed = 'is-confirmed' in classes or 'confirmed' in classes
                        
                        # Determine minutes
                        if is_sub:
                            mins = 15.0  # Bench player
                        elif is_confirmed:
                            mins = 90.0  # Confirmed starter
                        elif is_starter:
                            mins = 85.0  # Predicted starter
                        else:
                            mins = 45.0  # Unknown
                        
                        # Try to match player
                        matched_id = None
                        
                        # Direct match
                        if player_name in elements_by_name:
                            matched_id = elements_by_name[player_name]
                        else:
                            # Fuzzy match - check if any stored name contains this name or vice versa
                            for stored_name, pid in elements_by_name.items():
                                if len(player_name) >= 4 and len(stored_name) >= 4:
                                    if player_name in stored_name or stored_name in player_name:
                                        matched_id = pid
                                        break
                        
                        if matched_id:
                            predictions[matched_id] = mins
                            matched_players.append(f"{player_name} -> {elements[matched_id]['web_name']} ({mins}m)")
                        else:
                            unmatched_players.append(player_name)
                
                logger.info(f"Rotowire: matched {len(matched_players)}, unmatched {len(unmatched_players)}")
                            
            else:
                errors.append(f"Rotowire HTTP {response.status_code}")
                
        except Exception as e:
            errors.append(f"Rotowire: {str(e)}")
            logger.error(f"Rotowire scrape failed: {e}")
    
    # Apply predictions to cache
    count = 0
    for player_id, minutes in predictions.items():
        if 0 <= minutes <= 90:
            cache.predicted_minutes[player_id] = minutes
            count += 1
    
    if count > 0:
        cache.predicted_minutes_last_update = datetime.now()
    
    return {
        "status": "ok" if count > 0 else "failed",
        "source": source,
        "imported": count,
        "matched_sample": matched_players[:20],
        "unmatched_sample": unmatched_players[:20],
        "errors": errors if errors else None,
        "updated_at": cache.predicted_minutes_last_update.isoformat() if cache.predicted_minutes_last_update else None
    }


@app.get("/api/predicted-minutes/status")
async def get_predicted_minutes_status():
    """Get status of predicted minutes and available sources."""
    return {
        "cached_count": len(cache.predicted_minutes),
        "last_update": cache.predicted_minutes_last_update.isoformat() if cache.predicted_minutes_last_update else None,
        "is_stale": cache.predicted_minutes_is_stale(),
        "available_sources": ["fplreview", "rotowire", "manual"],
        "recommended": "fplreview"
    }


# ============ FDR ENDPOINTS ============

@app.get("/api/fdr/teams")
async def get_fdr_teams():
    """Get FDR metrics for all teams with xG-based form data."""
    fdr_data = await refresh_fdr_data()
    
    teams = []
    for team_id, data in fdr_data.items():
        team_name = FPL_ID_TO_UNDERSTAT.get(team_id, f"Team {team_id}")
        teams.append({
            "team_id": team_id,
            "team_name": team_name,
            "form_xg": data.get("form_xg", 0),
            "form_xga": data.get("form_xga", 0),
            "form_ppg": data.get("form_ppg", 0),
            "attack_fdr": data.get("attack_fdr", 5),
            "defence_fdr": data.get("defence_fdr", 5),
            "composite_fdr": data.get("composite_fdr", 5),
            "season_matches": data.get("season_matches", 0),
        })
    
    teams.sort(key=lambda x: x["composite_fdr"], reverse=True)
    
    data_age = 0
    if cache.fdr_last_update:
        data_age = (datetime.now() - cache.fdr_last_update).total_seconds() / 3600
    
    return {
        "teams": teams,
        "data_age_hours": round(data_age, 1),
    }


@app.post("/api/fdr/refresh")
async def force_fdr_refresh():
    """Force refresh FDR data from Understat."""
    cache.fdr_last_update = None  # Force refresh
    fdr_data = await refresh_fdr_data(force=True)
    return {"status": "ok", "teams_refreshed": len(fdr_data)}


# ============ PLAYER SEARCH ENDPOINT ============

@app.get("/api/players/search")
async def search_players(
    q: str = Query("", description="Search query (name or team)"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Simple player search for autocomplete.
    Returns all players if no query, or filtered by name/team.
    """
    data = await fetch_fpl_data()
    teams = {t["id"]: t for t in data["teams"]}
    
    players = []
    query = q.lower().strip()
    
    for p in data["elements"]:
        name = p.get("web_name", "")
        team_data = teams.get(p["team"], {})
        team_name = team_data.get("short_name", "")
        
        # If query provided, filter
        if query:
            if query not in name.lower() and query not in team_name.lower():
                continue
        
        players.append({
            "id": p["id"],
            "name": name,
            "full_name": f"{p.get('first_name', '')} {p.get('second_name', '')}",
            "team": team_name,
            "team_id": p["team"],
            "position": {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(p["element_type"], "???"),
            "price": p["now_cost"] / 10,
            "minutes": p.get("minutes", 0),
            "ownership": float(p.get("selected_by_percent", 0) or 0),
        })
    
    # Sort by ownership (most owned first for better suggestions)
    players.sort(key=lambda x: x["ownership"], reverse=True)
    
    return {"players": players[:limit]}


# ============ RANKINGS ENDPOINT ============

@app.get("/api/rankings/{position}")
async def get_rankings(
    position: Position,
    gw_start: Optional[int] = Query(None, ge=1, le=38),
    gw_end: Optional[int] = Query(None, ge=1, le=38),
    min_minutes: int = Query(MIN_MINUTES_DEFAULT, ge=0),
    min_price: float = Query(0, ge=0),
    max_price: float = Query(20, ge=0, le=20),
    max_ownership: float = Query(100, ge=0, le=100),
    limit: int = Query(50, ge=1, le=200),
    sort_by: str = Query("xpts", pattern="^(xpts|price|form|ownership|defcon_per_90|saves_per_90|expected_minutes|total_points|xgi_per_90|bonus_per_90)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$")
):
    # Ensure FDR data is loaded
    await refresh_fdr_data()
    
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    
    elements = data["elements"]
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    next_gw = get_next_gameweek(events)
    position_id = POSITION_ID_MAP[position.value]
    
    # Use dynamic defaults based on current GW if not specified
    if gw_start is None:
        gw_start = next_gw if next_gw else min(current_gw + 1, 38)
    if gw_end is None:
        gw_end = min(gw_start + 5, 38)
    
    if gw_end < gw_start:
        raise HTTPException(status_code=400, detail="gw_end must be >= gw_start")
    
    # Filter players first
    filtered_players = []
    for player in elements:
        if player["element_type"] != position_id:
            continue
        
        total_minutes = player.get("minutes", 0)
        price = player["now_cost"] / 10
        ownership = float(player.get("selected_by_percent", 0) or 0)
        
        if total_minutes < min_minutes or price < min_price or price > max_price:
            continue
        
        if ownership > max_ownership:
            continue
        
        filtered_players.append(player)
    
    # Batch fetch player histories for smarter expected minutes
    player_histories = {}
    
    async def fetch_history_safe(pid):
        try:
            return pid, await fetch_player_history(pid)
        except:
            return pid, {"history": []}
    
    # Fetch histories in parallel with semaphore
    sem = asyncio.Semaphore(10)
    async def fetch_with_sem(pid):
        async with sem:
            return await fetch_history_safe(pid)
    
    history_tasks = [fetch_with_sem(p["id"]) for p in filtered_players]
    history_results = await asyncio.gather(*history_tasks)
    player_histories = {pid: hist for pid, hist in history_results}
    
    ranked_players = []
    
    for player in filtered_players:
        price = player["now_cost"] / 10
        
        upcoming = get_player_upcoming_fixtures(player["team"], fixtures, gw_start, gw_end, teams, position_id)
        
        # Pass player history for smart pattern detection
        player_hist = player_histories.get(player["id"])
        
        stats = calculate_expected_points(
            player, position_id, current_gw, upcoming, teams, fixtures, events,
            player_history=player_hist, all_players=elements
        )
        
        team_data = teams.get(player["team"], {})
        ownership_pct = float(player.get("selected_by_percent", 0) or 0)
        ownership_tier, tier_desc = get_ownership_tier(ownership_pct)
        
        ranked_players.append({
            "id": player["id"],
            "name": player["web_name"],
            "full_name": f"{player['first_name']} {player['second_name']}",
            "team": team_data.get("short_name", "???"),
            "team_id": player["team"],
            "position": position.value,
            "price": price,
            "xpts": stats["xpts"],
            "xpts_per_90": stats["xpts_per_90"],
            "xpts_ceiling": stats.get("xpts_ceiling", stats["xpts"] * 1.2),
            "xpts_floor": stats.get("xpts_floor", stats["xpts"] * 0.7),
            "expected_minutes": stats["expected_minutes"],
            "minutes_reason": stats["minutes_reason"],
            "prob_60_plus": stats.get("prob_60_plus", 0.8),
            "form": float(player.get("form", 0) or 0),
            "total_points": player.get("total_points", 0),
            "ownership": ownership_pct,
            "ownership_tier": ownership_tier,
            "ownership_tier_desc": tier_desc,
            "xgi_per_90": round(stats["xG_per_90"] + stats["xA_per_90"], 2),
            "bonus_per_90": stats["bonus_per_90"],
            "expected_bonus": stats["expected_bonus"],
            "defcon_per_90": stats["defcon_per_90"] if position_id in [2, 3] else None,
            "defcon_prob": stats["defcon_prob"] if position_id in [2, 3] else None,
            "defcon_pts_total": stats["defcon_pts_total"] if position_id in [2, 3] else None,
            "saves_per_90": stats["saves_per_90"] if position_id == 1 else None,
            "save_pts_per_90": stats["save_pts_per_90"] if position_id == 1 else None,
            "xGC_per_90": stats["xGC_per_90"],
            "cs_prob": stats["cs_prob"],
            "xG_per_90": stats["xG_per_90"],
            "xA_per_90": stats["xA_per_90"],
            "fixtures": upcoming[:8],
            "news": player.get("news", ""),
            "status": player.get("status", "a"),
        })
    
    reverse = sort_order == "desc"
    ranked_players.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)
    
    return {
        "position": position.value,
        "gw_range": f"GW{gw_start}-GW{gw_end}",
        "current_gw": current_gw,
        "total_players": len(ranked_players),
        "filters": {"min_minutes": min_minutes, "min_price": min_price, "max_price": max_price, "sort_by": sort_by},
        "players": ranked_players[:limit]
    }


# ============ MY TEAM ENDPOINTS ============

@app.get("/api/my-team/{manager_id}")
async def get_my_team(manager_id: int):
    await refresh_fdr_data()
    
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    
    elements = {e["id"]: e for e in data["elements"]}
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    next_gw = get_next_gameweek(events)
    
    team_data = await fetch_manager_team(manager_id, current_gw)
    manager = team_data["manager"]
    picks = team_data["picks"]["picks"]
    entry_history = team_data["picks"].get("entry_history", {})
    
    # Fetch player histories for all squad members (only 15 players)
    player_histories = {}
    async def fetch_hist(pid):
        try:
            return pid, await fetch_player_history(pid)
        except:
            return pid, {"history": []}
    
    hist_tasks = [fetch_hist(p["element"]) for p in picks]
    hist_results = await asyncio.gather(*hist_tasks)
    player_histories = {pid: hist for pid, hist in hist_results}
    
    squad = []
    for pick in picks:
        player = elements.get(pick["element"])
        if not player:
            continue
        
        position_id = player["element_type"]
        player_hist = player_histories.get(player["id"])
        
        upcoming = get_player_upcoming_fixtures(player["team"], fixtures, next_gw, next_gw + 5, teams, position_id)
        stats = calculate_expected_points(player, position_id, current_gw, upcoming, teams, fixtures, events, player_history=player_hist, all_players=list(elements.values()))
        
        single_fixtures = get_player_upcoming_fixtures(player["team"], fixtures, next_gw, next_gw, teams, position_id)
        single_stats = calculate_expected_points(player, position_id, current_gw, single_fixtures, teams, fixtures, events, player_history=player_hist, all_players=list(elements.values()))
        
        squad.append({
            "id": player["id"],
            "name": player["web_name"],
            "team": teams.get(player["team"], {}).get("short_name", "???"),
            "team_id": player["team"],
            "position": POSITION_MAP.get(position_id, "???"),
            "position_id": position_id,
            "price": player["now_cost"] / 10,
            "selling_price": pick.get("selling_price", player["now_cost"]) / 10,
            "is_captain": pick["is_captain"],
            "is_vice_captain": pick["is_vice_captain"],
            "squad_position": pick["position"],
            "xpts": stats["xpts"],
            "xpts_single_gw": single_stats["xpts"],
            "expected_minutes": stats["expected_minutes"],
            "minutes_reason": stats["minutes_reason"],
            "form": float(player.get("form", 0) or 0),
            "total_points": player.get("total_points", 0),
            "defcon_per_90": stats["defcon_per_90"],
            "saves_per_90": stats["saves_per_90"],
            "ownership": float(player.get("selected_by_percent", 0) or 0),
            "fixtures": upcoming[:5],
            "news": player.get("news", ""),
            "status": player.get("status", "a"),
        })
    
    squad.sort(key=lambda x: x["squad_position"])
    starting = [p for p in squad if p["squad_position"] <= 11]
    bench = [p for p in squad if p["squad_position"] > 11]
    
    return {
        "manager": {
            "id": manager["id"],
            "name": f"{manager['player_first_name']} {manager['player_last_name']}",
            "team_name": manager["name"],
            "overall_rank": manager.get("summary_overall_rank"),
            "total_points": manager.get("summary_overall_points"),
        },
        "bank": entry_history.get("bank", 0) / 10,
        "free_transfers": entry_history.get("event_transfers", 1),
        "total_value": entry_history.get("value", 0) / 10,
        "current_gw": current_gw,
        "next_gw": next_gw,
        "squad": squad,
        "starting_xi": starting,
        "bench": bench,
        "captain_picks": sorted(starting, key=lambda x: x["xpts_single_gw"], reverse=True)[:3],
        "optimal_bench_order": sorted(bench, key=lambda x: x["xpts_single_gw"], reverse=True),
    }


@app.get("/api/my-team/{manager_id}/transfers")
async def get_transfer_suggestions(
    manager_id: int,
    horizon: int = Query(5, ge=1, le=10),
    min_price: float = Query(0, ge=0),
    max_price: float = Query(20, ge=0, le=20)
):
    await refresh_fdr_data()
    
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    
    elements = {e["id"]: e for e in data["elements"]}
    all_players = data["elements"]
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    next_gw = get_next_gameweek(events)
    
    team_data = await fetch_manager_team(manager_id, current_gw)
    picks = team_data["picks"]["picks"]
    entry_history = team_data["picks"].get("entry_history", {})
    bank = entry_history.get("bank", 0) / 10
    
    current_squad = {}
    squad_by_position = defaultdict(list)
    team_counts = defaultdict(int)
    
    for pick in picks:
        player = elements.get(pick["element"])
        if not player:
            continue
        
        position_id = player["element_type"]
        pos = POSITION_MAP.get(position_id, "???")
        upcoming = get_player_upcoming_fixtures(player["team"], fixtures, next_gw, next_gw + horizon, teams)
        stats = calculate_expected_points(player, position_id, current_gw, upcoming, teams, fixtures, events)
        selling_price = pick.get("selling_price", player["now_cost"]) / 10
        
        current_squad[player["id"]] = {
            "id": player["id"], "name": player["web_name"],
            "team": teams.get(player["team"], {}).get("short_name", "???"),
            "team_id": player["team"], "position": pos, "position_id": position_id,
            "price": player["now_cost"] / 10, "selling_price": selling_price,
            "xpts": stats["xpts"], "expected_minutes": stats["expected_minutes"],
            "form": float(player.get("form", 0) or 0),
        }
        squad_by_position[pos].append(current_squad[player["id"]])
        team_counts[player["team"]] += 1
    
    single_transfers = []
    
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        pos_id = POSITION_ID_MAP[pos]
        for out_player in squad_by_position[pos]:
            available_budget = bank + out_player["selling_price"]
            
            for player in all_players:
                if player["id"] in current_squad or player["element_type"] != pos_id:
                    continue
                price = player["now_cost"] / 10
                if price > available_budget or price < min_price or price > max_price:
                    continue
                
                player_team = player["team"]
                current_count = team_counts[player_team] - (1 if out_player["team_id"] == player_team else 0)
                if current_count >= 3 or player.get("minutes", 0) < 200:
                    continue
                
                upcoming = get_player_upcoming_fixtures(player["team"], fixtures, next_gw, next_gw + horizon, teams)
                stats = calculate_expected_points(player, pos_id, current_gw, upcoming, teams, fixtures, events)
                xpts_gain = stats["xpts"] - out_player["xpts"]
                
                if xpts_gain > 0.3:
                    single_transfers.append({
                        "out": out_player,
                        "in": {
                            "id": player["id"], "name": player["web_name"],
                            "team": teams.get(player["team"], {}).get("short_name", "???"),
                            "team_id": player["team"], "position": pos, "price": price,
                            "xpts": stats["xpts"], "expected_minutes": stats["expected_minutes"],
                            "form": float(player.get("form", 0) or 0),
                            "ownership": float(player.get("selected_by_percent", 0) or 0),
                        },
                        "xpts_gain": round(xpts_gain, 2),
                        "cost_change": round(price - out_player["selling_price"], 1),
                        "bank_after": round(available_budget - price, 1)
                    })
    
    single_transfers.sort(key=lambda x: x["xpts_gain"], reverse=True)
    
    double_transfers = []
    for first in single_transfers[:10]:
        for second in single_transfers:
            if second["out"]["id"] in [first["out"]["id"], first["in"]["id"]]:
                continue
            if second["in"]["id"] in [first["out"]["id"], first["in"]["id"]]:
                continue
            second_cost = second["in"]["price"] - second["out"]["selling_price"]
            if second_cost > first["bank_after"]:
                continue
            new_counts = team_counts.copy()
            for t in [first, second]:
                new_counts[t["out"]["team_id"]] -= 1
                new_counts[t["in"]["team_id"]] += 1
            if new_counts[second["in"]["team_id"]] > 3:
                continue
            total_gain = first["xpts_gain"] + second["xpts_gain"]
            double_transfers.append({
                "transfers": [first, second],
                "total_xpts_gain": round(total_gain, 2),
                "hit_adjusted_gain": round(total_gain - 4, 2),
                "bank_after": round(first["bank_after"] - second_cost, 1)
            })
    
    double_transfers.sort(key=lambda x: x["hit_adjusted_gain"], reverse=True)
    
    return {
        "bank": bank,
        "horizon": f"GW{next_gw}-GW{next_gw + horizon}",
        "best_single_transfers": single_transfers[:15],
        "best_double_transfers": double_transfers[:10],
    }


# ============ FIXTURE RATINGS ============

@app.get("/api/fixture-ratings")
async def get_fixture_ratings(horizon: int = Query(8, ge=1, le=15)):
    await refresh_fdr_data()
    
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    
    team_ratings = []
    for team_id, team in teams.items():
        upcoming = []
        total_attack_fdr = 0
        total_defence_fdr = 0
        
        for fix in fixtures:
            gw = fix.get("event")
            if gw and current_gw <= gw < current_gw + horizon:
                if fix["team_h"] == team_id:
                    opponent_id = fix["team_a"]
                    is_home = True
                elif fix["team_a"] == team_id:
                    opponent_id = fix["team_h"]
                    is_home = False
                else:
                    continue
                
                attack_fdr = get_fixture_fdr(opponent_id, is_home, 4)  # FWD perspective
                defence_fdr = get_fixture_fdr(opponent_id, is_home, 2)  # DEF perspective
                
                upcoming.append({
                    "gameweek": gw,
                    "opponent": teams.get(opponent_id, {}).get("short_name", "???"),
                    "is_home": is_home,
                    "attack_fdr": attack_fdr,
                    "defence_fdr": defence_fdr,
                    "difficulty": fix.get("team_h_difficulty" if is_home else "team_a_difficulty", 3)
                })
                total_attack_fdr += attack_fdr
                total_defence_fdr += defence_fdr
        
        upcoming.sort(key=lambda x: x["gameweek"])
        
        if upcoming:
            avg_attack = total_attack_fdr / len(upcoming)
            avg_defence = total_defence_fdr / len(upcoming)
            avg_composite = (avg_attack + avg_defence) / 2
        else:
            avg_attack = avg_defence = avg_composite = 5
        
        rating = "EASY" if avg_composite <= 4 else ("MEDIUM" if avg_composite <= 6 else "HARD")
        
        # Get team's xG form from FDR data
        team_fdr = cache.fdr_data.get(team_id, {})
        
        team_ratings.append({
            "team_id": team_id,
            "team_name": team["name"],
            "team_short": team["short_name"],
            "avg_attack_fdr": round(avg_attack, 2),
            "avg_defence_fdr": round(avg_defence, 2),
            "avg_composite_fdr": round(avg_composite, 2),
            "rating": rating,
            "fixtures": upcoming,
            "easy_fixtures": len([f for f in upcoming if (f["attack_fdr"] + f["defence_fdr"]) / 2 <= 4]),
            "hard_fixtures": len([f for f in upcoming if (f["attack_fdr"] + f["defence_fdr"]) / 2 >= 7]),
            "form_xg": team_fdr.get("form_xg", 0),
            "form_xga": team_fdr.get("form_xga", 0),
        })
    
    team_ratings.sort(key=lambda x: x["avg_composite_fdr"])
    return {"horizon": horizon, "current_gw": current_gw, "teams": team_ratings}


@app.get("/api/fdr-grid")
async def get_fdr_grid():
    """Get full FDR grid for all teams and all GWs (for fixture ticker)."""
    await refresh_fdr_data()
    
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    
    # Build fixture lookup: team_id -> gw -> fixture info
    grid = {}
    for team_id, team in teams.items():
        grid[team_id] = {
            "team_name": team["name"],
            "team_short": team["short_name"],
            "fixtures": {}  # gw -> fixture data
        }
    
    for fix in fixtures:
        gw = fix.get("event")
        if not gw:
            continue
        
        home_id = fix["team_h"]
        away_id = fix["team_a"]
        
        # Home team fixture
        if home_id in grid:
            attack_fdr = get_fixture_fdr(away_id, True, 4)  # FWD perspective  
            defence_fdr = get_fixture_fdr(away_id, True, 2)  # DEF perspective
            composite_fdr = round((attack_fdr + defence_fdr) / 2)
            
            if gw not in grid[home_id]["fixtures"]:
                grid[home_id]["fixtures"][gw] = []
            grid[home_id]["fixtures"][gw].append({
                "opponent": teams.get(away_id, {}).get("short_name", "???"),
                "is_home": True,
                "fdr": composite_fdr,
                "attack_fdr": attack_fdr,
                "defence_fdr": defence_fdr
            })
        
        # Away team fixture
        if away_id in grid:
            attack_fdr = get_fixture_fdr(home_id, False, 4)
            defence_fdr = get_fixture_fdr(home_id, False, 2)
            composite_fdr = round((attack_fdr + defence_fdr) / 2)
            
            if gw not in grid[away_id]["fixtures"]:
                grid[away_id]["fixtures"][gw] = []
            grid[away_id]["fixtures"][gw].append({
                "opponent": teams.get(home_id, {}).get("short_name", "???"),
                "is_home": False,
                "fdr": composite_fdr,
                "attack_fdr": attack_fdr,
                "defence_fdr": defence_fdr
            })
    
    # Convert to list sorted by team name
    team_list = []
    for team_id, team_data in grid.items():
        team_list.append({
            "team_id": team_id,
            "team_name": team_data["team_name"],
            "team_short": team_data["team_short"],
            "fixtures": team_data["fixtures"]
        })
    
    team_list.sort(key=lambda x: x["team_name"])
    
    return {
        "current_gw": current_gw,
        "teams": team_list
    }


# ============ DIFFERENTIALS ============

async def fetch_manager_transfers(manager_id: int) -> List[Dict]:
    """Fetch all transfers made by a manager this season."""
    try:
        client = await get_http_client()
        response = await client.get(f"{FPL_BASE_URL}/entry/{manager_id}/transfers/")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch transfers: {e}")
        return []


async def fetch_player_history(player_id: int, use_cache: bool = True) -> Dict:
    """
    Fetch player's GW-by-GW history with caching.
    
    Cache duration: 30 minutes (player histories don't change during GW).
    This significantly reduces API calls for rankings/transfers.
    """
    # Check cache first
    if use_cache:
        cached = cache.get_player_history(player_id)
        if cached is not None:
            return cached
    
    try:
        client = await get_http_client()
        response = await client.get(f"{FPL_BASE_URL}/element-summary/{player_id}/")
        response.raise_for_status()
        history = response.json()
        
        # Cache the result
        cache.set_player_history(player_id, history)
        
        return history
    except Exception:
        return {"history": []}


async def fetch_gw_picks(manager_id: int, gw: int) -> Optional[Dict]:
    """Fetch picks for a specific gameweek."""
    try:
        client = await get_http_client()
        response = await client.get(f"{FPL_BASE_URL}/entry/{manager_id}/event/{gw}/picks/")
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def get_player_gw_points(player_history: Dict, gw: int) -> int:
    """Get points a player scored in a specific GW."""
    for h in player_history.get("history", []):
        if h.get("round") == gw:
            return h.get("total_points", 0)
    return 0


def get_player_points_range(player_history: Dict, start_gw: int, end_gw: int) -> int:
    """Get total points a player scored over a GW range."""
    total = 0
    for h in player_history.get("history", []):
        gw = h.get("round", 0)
        if start_gw <= gw <= end_gw:
            total += h.get("total_points", 0)
    return total


@app.get("/api/my-team/{manager_id}/stats")
async def get_manager_stats(manager_id: int):
    """
    Get comprehensive manager stats:
    - Best GW (highest points + rank for that GW)
    - Best/Worst transfer (actual pts diff next GW)
    - Highest bench score
    - Wildcard net effect (5 GW comparison)
    """
    try:
        data = await fetch_fpl_data()
        history = await fetch_manager_history(manager_id)
        transfers = await fetch_manager_transfers(manager_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch data: {str(e)}")
    
    elements = {e["id"]: e for e in data["elements"]}
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    
    # Process GW history
    gw_history = history.get("current", [])
    
    if not gw_history:
        # Return empty stats if no history
        return {
            "manager_id": manager_id,
            "current_gw": current_gw,
            "best_gw": None,
            "highest_bench": {"gw": 0, "points": 0},
            "or_progression": [],
            "best_transfer": None,
            "worst_transfer": None,
            "wildcard_analysis": None,
            "chips_used": [],
        }
    
    # Find best GW (highest points) - include rank for that GW
    best_gw = max(gw_history, key=lambda x: x.get("points", 0))
    
    # Process chips for easy lookup
    chip_gws = {}
    for chip in history.get("chips", []):
        chip_gws[chip.get("event", 0)] = chip.get("name", "")
    
    # OR progression for graph with annotations
    or_progression = []
    prev_or = None
    biggest_green_arrow = None
    biggest_red_arrow = None
    total_hits = 0
    total_hit_cost = 0
    
    for gw in gw_history:
        gw_num = gw["event"]
        current_or = gw.get("overall_rank", 0)
        gw_points = gw.get("points", 0)
        transfers_cost = gw.get("event_transfers_cost", 0)
        
        # Track hits
        if transfers_cost > 0:
            total_hits += transfers_cost // 4
            total_hit_cost += transfers_cost
        
        # Calculate rank change
        rank_change = 0
        if prev_or and current_or:
            rank_change = prev_or - current_or  # Positive = green arrow
            
            if biggest_green_arrow is None or rank_change > biggest_green_arrow["change"]:
                biggest_green_arrow = {"gw": gw_num, "change": rank_change, "from": prev_or, "to": current_or}
            if biggest_red_arrow is None or rank_change < biggest_red_arrow["change"]:
                biggest_red_arrow = {"gw": gw_num, "change": rank_change, "from": prev_or, "to": current_or}
        
        or_progression.append({
            "gw": gw_num,
            "or": current_or,
            "points": gw_points,
            "rank": gw.get("rank", 0),
            "rank_change": rank_change,
            "chip": chip_gws.get(gw_num, None),
            "hit_cost": transfers_cost,
            "bench_pts": gw.get("points_on_bench", 0),
        })
        
        prev_or = current_or
    
    # Calculate averages
    total_gws = len(gw_history)
    avg_gw_points = sum(gw.get("points", 0) for gw in gw_history) / max(1, total_gws)
    total_points = sum(gw.get("points", 0) for gw in gw_history)
    
    # Calculate highest bench points
    highest_bench = {"gw": 0, "points": 0}
    for gw_data in gw_history:
        bench_pts = gw_data.get("points_on_bench", 0)
        if bench_pts > highest_bench["points"]:
            highest_bench = {"gw": gw_data["event"], "points": bench_pts}
    
    # Process transfers with ACTUAL points difference (next GW)
    # Fetch player histories for all transferred players
    transfer_player_ids = set()
    for t in transfers:
        transfer_player_ids.add(t.get("element_in"))
        transfer_player_ids.add(t.get("element_out"))
    
    # Fetch histories in parallel (limit to avoid too many requests)
    player_histories = {}
    player_ids_to_fetch = [pid for pid in list(transfer_player_ids)[:40] if pid]
    
    # Fetch histories concurrently
    async def fetch_history_safe(pid):
        try:
            return pid, await fetch_player_history(pid)
        except Exception:
            return pid, {"history": []}
    
    if player_ids_to_fetch:
        results = await asyncio.gather(*[fetch_history_safe(pid) for pid in player_ids_to_fetch])
        for pid, hist in results:
            player_histories[pid] = hist
    
    transfer_analysis = []
    for transfer in transfers:
        gw = transfer.get("event", 0)
        player_in_id = transfer.get("element_in")
        player_out_id = transfer.get("element_out")
        
        player_in = elements.get(player_in_id, {})
        player_out = elements.get(player_out_id, {})
        
        if not player_in or not player_out:
            continue
        
        # Get actual points in the NEXT gameweek after transfer
        in_history = player_histories.get(player_in_id, {"history": []})
        out_history = player_histories.get(player_out_id, {"history": []})
        
        in_pts_next_gw = get_player_gw_points(in_history, gw)
        out_pts_next_gw = get_player_gw_points(out_history, gw)
        
        pts_diff = in_pts_next_gw - out_pts_next_gw
        
        transfer_analysis.append({
            "gw": gw,
            "in": {
                "id": player_in_id,
                "name": player_in.get("web_name", "Unknown"),
                "team": teams.get(player_in.get("team", 0), {}).get("short_name", "???"),
                "pts_next_gw": in_pts_next_gw,
            },
            "out": {
                "id": player_out_id,
                "name": player_out.get("web_name", "Unknown"),
                "team": teams.get(player_out.get("team", 0), {}).get("short_name", "???"),
                "pts_next_gw": out_pts_next_gw,
            },
            "pts_diff": pts_diff,
        })
    
    # Find best and worst transfer by actual points diff
    best_transfer = None
    worst_transfer = None
    if transfer_analysis:
        transfer_analysis.sort(key=lambda x: x["pts_diff"], reverse=True)
        best_transfer = transfer_analysis[0]
        worst_transfer = transfer_analysis[-1]
    
    # Wildcard analysis - WC1 (<GW20) and WC2 (>=GW20) separately
    wc1_analysis = None
    wc2_analysis = None
    chips_used = history.get("chips", [])
    
    for chip in chips_used:
        if chip.get("name") == "wildcard":
            wc_gw = chip.get("event", 0)
            is_wc2 = wc_gw >= 20
            
            try:
                # Get picks before WC (GW before) and after WC
                picks_before = await fetch_gw_picks(manager_id, wc_gw - 1) if wc_gw > 1 else None
                picks_after = await fetch_gw_picks(manager_id, wc_gw)
                
                if picks_before and picks_after:
                    # Get player IDs before and after
                    team_before = {p["element"] for p in picks_before.get("picks", [])}
                    team_after = {p["element"] for p in picks_after.get("picks", [])}
                    
                    players_out = team_before - team_after
                    players_in = team_after - team_before
                    
                    # Calculate points over next 5 GWs for both sets
                    end_gw = min(wc_gw + 4, current_gw)
                    
                    pts_old_team = 0
                    pts_new_team = 0
                    
                    # Fetch histories for WC players not already fetched
                    wc_players_to_fetch = []
                    for pid in list(players_out | players_in):
                        if pid not in player_histories:
                            wc_players_to_fetch.append(pid)
                    
                    if wc_players_to_fetch:
                        wc_results = await asyncio.gather(*[fetch_history_safe(pid) for pid in wc_players_to_fetch])
                        for pid, hist in wc_results:
                            player_histories[pid] = hist
                    
                    for pid in players_out:
                        pts_old_team += get_player_points_range(player_histories.get(pid, {"history": []}), wc_gw, end_gw)
                    
                    for pid in players_in:
                        pts_new_team += get_player_points_range(player_histories.get(pid, {"history": []}), wc_gw, end_gw)
                    
                    wc_data = {
                        "gw": wc_gw,
                        "players_out": len(players_out),
                        "players_in": len(players_in),
                        "pts_old_team_5gw": pts_old_team,
                        "pts_new_team_5gw": pts_new_team,
                        "net_pts": pts_new_team - pts_old_team,
                        "gw_range": f"GW{wc_gw}-GW{end_gw}",
                    }
                    
                    if is_wc2:
                        wc2_analysis = wc_data
                    else:
                        wc1_analysis = wc_data
                        
            except Exception as e:
                logger.error(f"Failed to analyze wildcard: {e}")
    
    # Triple Captain Analysis - TC1 (<GW20) and TC2 (>=GW20)
    tc1_analysis = None
    tc2_analysis = None
    tc_uses = [c for c in chips_used if c.get("name") == "3xc"]
    
    for chip in tc_uses:
        tc_gw = chip.get("event", 0)
        is_tc2 = tc_gw >= 20
        
        try:
            tc_picks = await fetch_gw_picks(manager_id, tc_gw)
            if tc_picks:
                # Find the captain
                for pick in tc_picks.get("picks", []):
                    if pick.get("is_captain"):
                        captain_id = pick.get("element")
                        captain = elements.get(captain_id, {})
                        captain_history = player_histories.get(captain_id)
                        
                        if not captain_history:
                            captain_history = await fetch_player_history(captain_id)
                        
                        # Get captain's points that GW
                        captain_pts = get_player_gw_points(captain_history, tc_gw)
                        
                        tc_data = {
                            "gw": tc_gw,
                            "player": captain.get("web_name", "Unknown"),
                            "captain_pts": captain_pts,
                            "extra_pts": captain_pts * 2,  # TC gives 3x, captain gives 2x, so extra = pts * 2
                        }
                        
                        if is_tc2:
                            tc2_analysis = tc_data
                        else:
                            tc1_analysis = tc_data
                        break
        except Exception as e:
            logger.error(f"Failed to analyze TC: {e}")
    
    # Bench Boost Analysis - BB1 (<GW20) and BB2 (>=GW20)
    bb1_analysis = None
    bb2_analysis = None
    bb_uses = [c for c in chips_used if c.get("name") == "bboost"]
    
    for chip in bb_uses:
        bb_gw = chip.get("event", 0)
        is_bb2 = bb_gw >= 20
        
        # Find bench points for that GW
        for gw_data in gw_history:
            if gw_data.get("event") == bb_gw:
                bb_data = {
                    "gw": bb_gw,
                    "bench_pts": gw_data.get("points_on_bench", 0),
                }
                
                if is_bb2:
                    bb2_analysis = bb_data
                else:
                    bb1_analysis = bb_data
                break
    
    # Free Hit Analysis - FH1 (<GW20) and FH2 (>=GW20)
    fh1_analysis = None
    fh2_analysis = None
    fh_uses = [c for c in chips_used if c.get("name") == "freehit"]
    
    for chip in fh_uses:
        fh_gw = chip.get("event", 0)
        is_fh2 = fh_gw >= 20
        
        try:
            # Compare FH team points vs what previous team would have scored
            fh_picks = await fetch_gw_picks(manager_id, fh_gw)
            prev_picks = await fetch_gw_picks(manager_id, fh_gw - 1) if fh_gw > 1 else None
            
            if fh_picks and prev_picks:
                # Get FH GW points (manager's actual points that GW)
                fh_pts = 0
                for gw_data in gw_history:
                    if gw_data.get("event") == fh_gw:
                        fh_pts = gw_data.get("points", 0)
                        break
                
                # Calculate what previous team would have scored
                prev_team_ids = {p["element"] for p in prev_picks.get("picks", [])[:11]}  # Starting XI
                prev_team_pts = 0
                
                for pid in prev_team_ids:
                    hist = player_histories.get(pid)
                    if not hist:
                        try:
                            hist = await fetch_player_history(pid)
                        except:
                            hist = {"history": []}
                    prev_team_pts += get_player_gw_points(hist, fh_gw)
                
                fh_data = {
                    "gw": fh_gw,
                    "fh_pts": fh_pts,
                    "prev_team_pts": prev_team_pts,
                    "net_pts": fh_pts - prev_team_pts,
                }
                
                if is_fh2:
                    fh2_analysis = fh_data
                else:
                    fh1_analysis = fh_data
        except Exception as e:
            logger.error(f"Failed to analyze FH: {e}")
    
    # Best Differential - highest points from lowest-owned player in STARTING XI
    # Only counts if player was in starting 11, not on bench
    best_differential = None
    
    try:
        for gw_data in gw_history:
            gw_num = gw_data.get("event", 0)
            if gw_num < 1:
                continue
            
            # Get the picks for this GW
            gw_picks = await fetch_gw_picks(manager_id, gw_num)
            if not gw_picks:
                continue
            
            picks = gw_picks.get("picks", [])
            for pick in picks:
                # Only consider starting XI (position 1-11), not bench (12-15)
                pick_position = pick.get("position", 0)
                if pick_position > 11:
                    continue
                
                pid = pick.get("element")
                element = elements.get(pid, {})
                
                # Get player history
                player_hist = player_histories.get(pid)
                if not player_hist and pid:
                    try:
                        player_hist = await fetch_player_history(pid)
                        player_histories[pid] = player_hist
                    except:
                        continue
                
                # Find ownership and points for this specific GW
                for ph in player_hist.get("history", []):
                    if ph.get("round") == gw_num:
                        gw_pts = ph.get("total_points", 0)
                        # Use current ownership as approximation (historical not available)
                        ownership = float(element.get("selected_by_percent", 0) or 0)
                        
                        # Differential threshold: below 10% ownership
                        if ownership < 10 and gw_pts > 0:
                            if best_differential is None or gw_pts > best_differential["pts"]:
                                best_differential = {
                                    "player": element.get("web_name", "Unknown"),
                                    "team": teams.get(element.get("team"), {}).get("short_name", "???"),
                                    "gw": gw_num,
                                    "pts": gw_pts,
                                    "ownership": round(ownership, 1),
                                }
                        break
    except Exception as e:
        logger.error(f"Failed to calculate best differential: {e}")
    
    return {
        "manager_id": manager_id,
        "current_gw": current_gw,
        "best_gw": {
            "gw": best_gw.get("event", 0),
            "points": best_gw.get("points", 0),
            "rank": best_gw.get("rank", 0),
        },
        "highest_bench": highest_bench,
        "or_progression": or_progression,
        "best_transfer": best_transfer,
        "worst_transfer": worst_transfer,
        # Fun stats
        "biggest_green_arrow": biggest_green_arrow,
        "biggest_red_arrow": biggest_red_arrow,
        "total_hits": total_hits,
        "total_hit_cost": total_hit_cost,
        "avg_gw_points": round(avg_gw_points, 1),
        "total_points": total_points,
        "total_gws_played": total_gws,
        # Chip analysis
        "wc1_analysis": wc1_analysis,
        "wc2_analysis": wc2_analysis,
        "fh1_analysis": fh1_analysis,
        "fh2_analysis": fh2_analysis,
        "tc1_analysis": tc1_analysis,
        "tc2_analysis": tc2_analysis,
        "bb1_analysis": bb1_analysis,
        "bb2_analysis": bb2_analysis,
        "best_differential": best_differential,
        "chips_used": chips_used,
    }


@app.get("/api/differentials")
async def get_differentials(
    max_ownership: float = Query(10.0, ge=0, le=100),
    min_form: float = Query(4.0, ge=0),
    min_minutes: int = Query(MIN_MINUTES_DEFAULT, ge=0),
    min_price: float = Query(0, ge=0),
    max_price: float = Query(20, ge=0, le=20),
    limit: int = Query(30, ge=1, le=100),
    sort_by: str = Query("diff_score"),
    sort_order: str = Query("desc")
):
    await refresh_fdr_data()
    
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    
    elements = data["elements"]
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    
    differentials = []
    for player in elements:
        ownership = float(player.get("selected_by_percent", 0) or 0)
        form = float(player.get("form", 0) or 0)
        total_minutes = player.get("minutes", 0)
        price = player["now_cost"] / 10
        
        if ownership > max_ownership or form < min_form or total_minutes < min_minutes:
            continue
        if price < min_price or price > max_price:
            continue
        
        mins_per_gw = total_minutes / max(current_gw - 1, 1)
        if mins_per_gw < 60:
            continue
        
        position_id = player["element_type"]
        upcoming = get_player_upcoming_fixtures(player["team"], fixtures, current_gw, current_gw + 5, teams)
        stats = calculate_expected_points(player, position_id, current_gw, upcoming, teams, fixtures, events)
        
        diff_score = (form * 0.4 + stats["xpts"] * 0.6) * (1 + (max_ownership - ownership) / max_ownership)
        
        differentials.append({
            "id": player["id"],
            "name": player["web_name"],
            "team": teams.get(player["team"], {}).get("short_name", "???"),
            "position": POSITION_MAP.get(position_id, "???"),
            "price": price,
            "ownership": ownership,
            "form": form,
            "xpts": stats["xpts"],
            "expected_minutes": stats["expected_minutes"],
            "diff_score": round(diff_score, 2),
            "fixtures": upcoming[:5],
            "defcon_per_90": stats["defcon_per_90"],
        })
    
    reverse = sort_order == "desc"
    differentials.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)
    
    return {
        "max_ownership": max_ownership,
        "current_gw": current_gw,
        "total_found": len(differentials),
        "players": differentials[:limit]
    }


@app.get("/")
async def root():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(path) if os.path.exists(path) else {"message": "FPL Assistant API", "docs": "/docs"}


@app.get("/api/bootstrap")
async def get_bootstrap():
    return await fetch_fpl_data()


# ============ TRANSFER PLANNER / SOLVER ============

ALL_CHIPS = {"wildcard", "freehit", "bboost", "3xc"}  # FPL API chip names
CHIP_DISPLAY = {"wildcard": "WC", "freehit": "FH", "bboost": "BB", "3xc": "TC"}


async def fetch_manager_history(manager_id: int) -> Dict:
    """Fetch manager history including chips used."""
    try:
        client = await get_http_client()
        response = await client.get(f"{FPL_BASE_URL}/entry/{manager_id}/history/")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch manager history: {str(e)}")


def get_available_chips(history: Dict, current_gw: int) -> Dict[str, bool]:
    """Determine which chips are still available."""
    chips_used = {c["name"]: c["event"] for c in history.get("chips", [])}
    
    # In second half of season (GW20+), second set of chips becomes available
    # WC1 used before GW20, WC2 available after
    is_second_half = current_gw >= 20
    
    available = {}
    for chip in ALL_CHIPS:
        if chip == "wildcard":
            # Two wildcards per season
            wc_uses = [c for c in history.get("chips", []) if c["name"] == "wildcard"]
            if is_second_half:
                # Can use WC2 if we haven't used it in GW20+
                wc2_used = any(c["event"] >= 20 for c in wc_uses)
                available["wildcard"] = not wc2_used
            else:
                # Can use WC1 if we haven't used any WC yet
                available["wildcard"] = len(wc_uses) == 0
        else:
            available[chip] = chip not in chips_used
    
    return available


def detect_dgw_bgw(
    fixtures: List[Dict],
    events: List[Dict],
    horizon_start: int,
    horizon_end: int,
    team_ids: Optional[List[int]] = None
) -> Dict[int, Dict]:
    """
    Detect Double and Blank gameweeks in horizon.
    
    Args:
        fixtures: List of fixture data
        events: List of event data
        horizon_start: First GW to check
        horizon_end: Last GW to check
        team_ids: List of team IDs to check for BGW. If None, derives from fixtures.
    """
    # Derive team_ids from fixtures if not provided
    if team_ids is None:
        all_teams = set()
        for fix in fixtures:
            all_teams.add(fix.get("team_h"))
            all_teams.add(fix.get("team_a"))
        team_ids = [t for t in all_teams if t is not None]
    
    gw_info = {}
    
    for gw in range(horizon_start, horizon_end + 1):
        team_fixtures = defaultdict(int)
        for fix in fixtures:
            if fix.get("event") == gw:
                team_fixtures[fix["team_h"]] += 1
                team_fixtures[fix["team_a"]] += 1
        
        dgw_teams = [t for t, count in team_fixtures.items() if count >= 2]
        bgw_teams = [t for t in team_ids if team_fixtures.get(t, 0) == 0]
        
        gw_info[gw] = {
            "is_dgw": len(dgw_teams) > 0,
            "is_bgw": len(bgw_teams) > 0,
            "dgw_teams": dgw_teams,
            "bgw_teams": bgw_teams,
            "dgw_count": len(dgw_teams),
            "bgw_count": len(bgw_teams),
        }
    
    return gw_info


def evaluate_squad_xpts(
    squad: List[Dict],
    gw: int,
    fixtures: List[Dict],
    teams_dict: Dict,
    elements: Dict,
    events: List[Dict],
    chip: Optional[str] = None,
    player_gw_cache: Optional[Dict] = None,  # (player_id, gw) -> xpts
) -> float:
    """
    Calculate expected points for a squad in a single GW.
    Handles chip effects (BB = all 15 play, TC = captain x3).
    
    player_gw_cache: Optional cache to avoid recalculating xpts for players already computed.
    """
    # Get xPts for each player for this specific GW
    player_xpts = []
    
    for p in squad:
        pid = p["id"]
        cache_key = (pid, gw)
        
        # Check cache first
        if player_gw_cache is not None and cache_key in player_gw_cache:
            xpts = player_gw_cache[cache_key]
            position_id = elements.get(pid, {}).get("element_type", p.get("position_id", 3))
        else:
            player = elements.get(pid)
            if not player:
                # Player not found in elements, use data from squad
                xpts = p.get("xpts", 0) / 5  # Rough per-GW estimate
                position_id = p.get("position_id", 3)
            else:
                position_id = player["element_type"]
                upcoming = get_player_upcoming_fixtures(player["team"], fixtures, gw, gw, teams_dict)
                
                if not upcoming:
                    # Player has blank gameweek
                    xpts = 0
                else:
                    stats = calculate_expected_points(
                        player, position_id, gw, upcoming, teams_dict, fixtures, events
                    )
                    # calculate_expected_points already sums xpts across all fixtures in the GW
                    # (including DGW double fixtures) - do NOT divide
                    xpts = stats["xpts"]
            
            # Store in cache
            if player_gw_cache is not None:
                player_gw_cache[cache_key] = xpts
        
        player_xpts.append({
            "id": pid,
            "position_id": position_id,
            "xpts": xpts,
            "is_captain": p.get("is_captain", False),
        })
    
    if not player_xpts:
        return 0.0
    
    # Sort by position for valid formation, then by xPts
    player_xpts.sort(key=lambda x: (x["position_id"], -x["xpts"]))
    
    # Select best XI with valid formation
    selected = select_best_xi(player_xpts)
    
    if not selected:
        # Fallback: use all players if formation selection fails
        selected = sorted(player_xpts, key=lambda x: -x["xpts"])[:11]
    
    if chip == "bboost":
        # Bench boost: all 15 count
        total = sum(p["xpts"] for p in player_xpts)
    else:
        # Normal: only starting XI
        total = sum(p["xpts"] for p in selected)
    
    # Captain bonus
    captain = max(selected, key=lambda x: x["xpts"])
    if chip == "3xc":
        # Triple captain: 3x instead of 2x
        total += captain["xpts"] * 2  # Already counted once, add 2 more
    else:
        total += captain["xpts"]  # Normal captain doubles
    
    return total


def select_best_xi(players: List[Dict]) -> List[Dict]:
    """Select best valid XI from 15 players."""
    by_pos = defaultdict(list)
    for p in players:
        by_pos[p["position_id"]].append(p)
    
    # Sort each position by xPts descending
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: -x["xpts"])
    
    # Valid formations: need 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD, total 11
    best_xi = []
    best_xpts = 0
    
    formations = [
        (1, 3, 4, 3), (1, 3, 5, 2), (1, 4, 3, 3), (1, 4, 4, 2), (1, 4, 5, 1),
        (1, 5, 3, 2), (1, 5, 4, 1),
    ]
    
    for gkp, defs, mids, fwds in formations:
        xi = []
        xi.extend(by_pos[1][:gkp])
        xi.extend(by_pos[2][:defs])
        xi.extend(by_pos[3][:mids])
        xi.extend(by_pos[4][:fwds])
        
        if len(xi) == 11:
            total = sum(p["xpts"] for p in xi)
            if total > best_xpts:
                best_xpts = total
                best_xi = xi
    
    return best_xi


def get_transfer_candidates(
    squad: List[Dict],
    all_players: List[Dict],
    teams_dict: Dict,
    fixtures: List[Dict],
    current_gw: int,
    horizon: int,
    budget: float,
    limit: int = 5,
    elements: Dict = None  # Optional: full player data dict for proper out_player stats
) -> List[Dict]:
    """Get top transfer candidates for each position."""
    squad_ids = {p["id"] for p in squad}
    team_counts = defaultdict(int)
    for p in squad:
        team_counts[p["team_id"]] += 1
    
    # Build elements lookup if provided
    elements_dict = elements if elements else {}
    
    candidates = []
    
    for position in ["GKP", "DEF", "MID", "FWD"]:
        pos_id = POSITION_ID_MAP[position]
        squad_in_pos = [p for p in squad if p["position_id"] == pos_id]
        
        for out_player in squad_in_pos:
            available_budget = budget + out_player["selling_price"]
            
            potential_ins = []
            for player in all_players:
                if player["id"] in squad_ids:
                    continue
                if player["element_type"] != pos_id:
                    continue
                
                price = player["now_cost"] / 10
                if price > available_budget:
                    continue
                
                # Check 3-player-per-team rule
                player_team = player["team"]
                current_count = team_counts[player_team] - (1 if out_player["team_id"] == player_team else 0)
                if current_count >= 3:
                    continue
                
                # Minimum minutes filter
                if player.get("minutes", 0) < 200:
                    continue
                
                # Calculate xPts over horizon
                upcoming = get_player_upcoming_fixtures(
                    player["team"], fixtures, current_gw, current_gw + horizon, teams_dict
                )
                stats = calculate_expected_points(
                    player, pos_id, current_gw, upcoming, teams_dict, fixtures, []
                )
                
                # Calculate out player's xPts using full element data if available
                out_upcoming = get_player_upcoming_fixtures(
                    out_player["team_id"], fixtures, current_gw, current_gw + horizon, teams_dict
                )
                
                # Use full player data from elements if available, otherwise build synthetic
                out_element = elements_dict.get(out_player["id"]) if elements_dict else None
                if out_element:
                    out_stats = calculate_expected_points(
                        out_element, pos_id, current_gw, out_upcoming, teams_dict, fixtures, []
                    )
                else:
                    # Fallback to synthetic dict (less accurate)
                    out_stats = calculate_expected_points(
                        {"id": out_player["id"], "team": out_player["team_id"], 
                         "minutes": out_player.get("minutes", 900), "expected_goals": 0,
                         "expected_assists": 0, "expected_goals_conceded": 0,
                         "points_per_game": out_player.get("form", 4), "status": "a"},
                        pos_id, current_gw, out_upcoming, teams_dict, fixtures, []
                    )
                
                xpts_gain = stats["xpts"] - out_player.get("xpts", out_stats["xpts"])
                
                if xpts_gain > 0:
                    potential_ins.append({
                        "out": out_player,
                        "in": {
                            "id": player["id"],
                            "name": player["web_name"],
                            "team": teams_dict.get(player["team"], {}).get("short_name", "???"),
                            "team_id": player["team"],
                            "position": position,
                            "position_id": pos_id,
                            "price": price,
                            "xpts": stats["xpts"],
                            "form": float(player.get("form", 0) or 0),
                        },
                        "xpts_gain": round(xpts_gain, 2),
                        "cost_change": round(price - out_player["selling_price"], 1),
                    })
            
            potential_ins.sort(key=lambda x: -x["xpts_gain"])
            candidates.extend(potential_ins[:limit])
    
    # Sort all candidates by xPts gain
    candidates.sort(key=lambda x: -x["xpts_gain"])
    return candidates[:20]  # Top 20 overall


class TransferPath:
    """Represents a sequence of transfer decisions across GWs."""
    
    def __init__(self, squad: List[Dict], bank: float, free_transfers: int):
        self.squad = squad.copy()
        self.bank = bank
        self.ft = free_transfers
        self.actions = []  # List of (gw, action_type, details)
        self.total_xpts = 0
        self.hits = 0
        self.gw_xpts = {}  # Cache: gw -> xpts (for incremental updates)
    
    def copy(self):
        new_path = TransferPath(self.squad, self.bank, self.ft)
        new_path.actions = self.actions.copy()
        new_path.total_xpts = self.total_xpts
        new_path.hits = self.hits
        new_path.gw_xpts = self.gw_xpts.copy()  # Copy cached GW xpts
        return new_path
    
    def apply_transfer(self, transfer: Dict, gw: int, is_hit: bool = False):
        """Apply a transfer to the squad."""
        out_id = transfer["out"]["id"]
        self.squad = [p for p in self.squad if p["id"] != out_id]
        self.squad.append({
            "id": transfer["in"]["id"],
            "name": transfer["in"]["name"],
            "team_id": transfer["in"]["team_id"],
            "position_id": transfer["in"]["position_id"],
            "price": transfer["in"]["price"],
            "selling_price": transfer["in"]["price"],
            "xpts": transfer["in"]["xpts"],
            "form": transfer["in"]["form"],
            "minutes": transfer["in"].get("minutes", 0),
        })
        self.bank += transfer["out"]["selling_price"] - transfer["in"]["price"]
        
        # Store full player data for timeline building
        action = {
            "type": "transfer",
            "gw": gw,
            "out": {
                "id": transfer["out"]["id"],
                "name": transfer["out"]["name"],
                "team_id": transfer["out"].get("team_id", 0),
                "position_id": transfer["out"].get("position_id", 0),
                "price": transfer["out"].get("price", 0),
                "selling_price": transfer["out"].get("selling_price", 0),
            },
            "in": {
                "id": transfer["in"]["id"],
                "name": transfer["in"]["name"],
                "team_id": transfer["in"]["team_id"],
                "position_id": transfer["in"]["position_id"],
                "price": transfer["in"]["price"],
                "xpts": transfer["in"]["xpts"],
                "form": transfer["in"]["form"],
                "minutes": transfer["in"].get("minutes", 0),
            },
            "xpts_gain": transfer["xpts_gain"],
            "is_hit": is_hit,
            "is_booked": transfer.get("is_booked", False),
        }
        self.actions.append(action)
        
        if is_hit:
            self.hits += 1
            self.total_xpts -= 4
    
    def roll_transfer(self, gw: int):
        """Roll the free transfer."""
        self.ft = min(self.ft + 1, 5)  # Max 5 FTs now (2024/25 rules)
        self.actions.append({"type": "roll", "gw": gw})
    
    def use_ft(self):
        """Use a free transfer."""
        self.ft = max(self.ft - 1, 0)


def find_player_by_name(name: str, players: List[Dict], squad_ids: set = None) -> Optional[Dict]:
    """
    Find a player by name (case-insensitive, partial match).
    
    Args:
        name: Player name to search for
        players: List of player dicts
        squad_ids: If provided, only match players in this set of IDs
    
    Returns: Player dict or None
    """
    name_lower = name.lower().strip()
    
    # Try exact web_name match first
    for p in players:
        if squad_ids and p["id"] not in squad_ids:
            continue
        if p.get("web_name", "").lower() == name_lower:
            return p
    
    # Try partial web_name match
    for p in players:
        if squad_ids and p["id"] not in squad_ids:
            continue
        if name_lower in p.get("web_name", "").lower():
            return p
    
    # Try full name match
    for p in players:
        if squad_ids and p["id"] not in squad_ids:
            continue
        full_name = f"{p.get('first_name', '')} {p.get('second_name', '')}".lower()
        if name_lower in full_name:
            return p
    
    return None


def solve_transfer_plan(
    squad: List[Dict],
    bank: float,
    free_transfers: int,
    all_players: List[Dict],
    teams_dict: Dict,
    elements: Dict,
    fixtures: List[Dict],
    events: List[Dict],
    start_gw: int,
    horizon: int,
    chip_gws: Dict[str, int],  # {chip_name: gw} - user specified chip GWs
    max_hits: int = 1,
    booked_transfers: List[Dict] = None  # [{gw: int, out: str, in: str}]
) -> Dict:
    """
    Tree-based solver for optimal transfer path.
    
    Explores paths: roll FT, use FT for best transfer, take hit for additional transfer.
    Prunes paths that fall too far behind.
    
    booked_transfers: List of forced transfers by GW. Names are matched to player IDs.
    """
    end_gw = min(start_gw + horizon, 39)
    gw_info = detect_dgw_bgw(fixtures, events, start_gw, end_gw)
    
    # Parse booked transfers - match names to player IDs
    booked_by_gw = {}
    if booked_transfers:
        squad_ids = {p["id"] for p in squad}
        logger.info(f"Processing {len(booked_transfers)} booked transfers")
        for bt in booked_transfers:
            gw = bt.get("gw")
            out_name = bt.get("out", "")
            in_name = bt.get("in", "")
            
            if not gw or not out_name or not in_name:
                logger.warning(f"Invalid booked transfer: {bt}")
                continue
            
            # Find the players
            out_player = find_player_by_name(out_name, list(elements.values()), squad_ids)
            in_player = find_player_by_name(in_name, all_players)
            
            if not out_player:
                logger.warning(f"Could not find out player: '{out_name}' in squad")
            if not in_player:
                logger.warning(f"Could not find in player: '{in_name}'")
            
            if out_player and in_player:
                if gw not in booked_by_gw:
                    booked_by_gw[gw] = []
                booked_by_gw[gw].append({
                    "out": out_player,
                    "in": in_player
                })
                logger.info(f"Booked GW{gw}: {out_player.get('web_name')} -> {in_player.get('web_name')}")
    
    # Track ALL booked-in players across entire horizon - these cannot be transferred out
    all_booked_in_ids = set()
    for gw_booked in booked_by_gw.values():
        for bt in gw_booked:
            all_booked_in_ids.add(bt["in"]["id"])
    
    # Track booked-out players by GW - cannot transfer out BEFORE their booked GW
    booked_out_before_gw = {}  # player_id -> gw they're booked out
    for gw_num, gw_booked in booked_by_gw.items():
        for bt in gw_booked:
            booked_out_before_gw[bt["out"]["id"]] = gw_num
    
    if all_booked_in_ids:
        logger.info(f"Protected booked-in player IDs: {all_booked_in_ids}")
    if booked_out_before_gw:
        logger.info(f"Booked-out players: {booked_out_before_gw}")
    
    # Initialize root path
    root = TransferPath(squad, bank, free_transfers)
    
    # Create shared caches for performance
    player_gw_cache = {}  # (player_id, gw) -> xpts - shared across all evaluations
    gw_to_chip = {v: k for k, v in chip_gws.items()}  # gw -> chip_name lookup
    
    # Helper to recompute xpts from a given GW onwards
    def recompute_from(path: TransferPath, from_gw: int):
        for g in range(from_gw, end_gw):
            chip = gw_to_chip.get(g)
            path.gw_xpts[g] = evaluate_squad_xpts(
                path.squad, g, fixtures, teams_dict, elements, events, chip, player_gw_cache
            )
        path.total_xpts = sum(path.gw_xpts.values()) - (path.hits * 4)
    
    # Calculate baseline xPts for root (populate gw_xpts cache)
    for gw in range(start_gw, end_gw):
        chip = gw_to_chip.get(gw)
        root.gw_xpts[gw] = evaluate_squad_xpts(
            squad, gw, fixtures, teams_dict, elements, events, chip, player_gw_cache
        )
    root.total_xpts = sum(root.gw_xpts.values())
    
    # Save baseline xPts BEFORE we start modifying paths
    baseline_xpts = root.total_xpts
    
    # BFS through decision tree
    paths = [root]
    
    for gw in range(start_gw, end_gw):
        new_paths = []
        
        # Get chip for this GW if specified
        chip_this_gw = None
        for chip_name, chip_gw in chip_gws.items():
            if chip_gw == gw:
                chip_this_gw = chip_name
                break
        
        # Check for booked transfers in this GW
        booked_this_gw = booked_by_gw.get(gw, [])
        
        for path in paths:
            # ALWAYS apply booked transfers first if any exist for this GW
            # Booked transfers are MANDATORY - not optional branches
            working_path = path.copy()
            booked_applied = False
            
            if booked_this_gw:
                for booked in booked_this_gw:
                    out_player = booked["out"]
                    in_player = booked["in"]
                    
                    # Check if out_player is still in squad
                    squad_dict = {p["id"]: p for p in working_path.squad}
                    if out_player["id"] not in squad_dict:
                        logger.warning(f"Booked transfer: {out_player.get('web_name')} not in squad, skipping")
                        continue  # Player already transferred out
                    
                    # Get the squad version of out_player (has selling_price)
                    squad_out_player = squad_dict[out_player["id"]]
                    selling_price = squad_out_player.get("selling_price", squad_out_player.get("price", out_player.get("now_cost", 0) / 10))
                    
                    # Build transfer dict
                    position_id = out_player.get("element_type", in_player.get("element_type", 0))
                    upcoming = get_player_upcoming_fixtures(in_player["team"], fixtures, gw, end_gw, teams_dict)
                    stats = calculate_expected_points(in_player, position_id, gw - 1, upcoming, teams_dict, fixtures, events)
                    
                    transfer = {
                        "out": {
                            "id": out_player["id"],
                            "name": out_player.get("web_name", ""),
                            "team_id": out_player.get("team"),
                            "position_id": out_player.get("element_type"),
                            "price": out_player.get("now_cost", 0) / 10,
                            "selling_price": selling_price,
                        },
                        "in": {
                            "id": in_player["id"],
                            "name": in_player.get("web_name", ""),
                            "team_id": in_player.get("team"),
                            "position_id": in_player.get("element_type"),
                            "price": in_player.get("now_cost", 0) / 10,
                            "xpts": stats["xpts"],
                            "form": float(in_player.get("form", 0) or 0),
                            "minutes": in_player.get("minutes", 0),
                        },
                        "xpts_gain": 0,
                        "is_booked": True
                    }
                    
                    logger.info(f"Applying booked transfer GW{gw}: {out_player.get('web_name')} -> {in_player.get('web_name')}")
                    
                    # Use FT if available, otherwise take hit
                    if working_path.ft > 0:
                        working_path.use_ft()
                        working_path.apply_transfer(transfer, gw, is_hit=False)
                    else:
                        working_path.apply_transfer(transfer, gw, is_hit=True)
                    booked_applied = True
                
                # Recompute after booked transfers
                if booked_applied:
                    recompute_from(working_path, gw)
            
            # Now explore options FROM the working_path (which has booked transfers applied)
            # Option 1: Roll FT (or just continue if no FTs)
            roll_path = working_path.copy()
            if not booked_applied:  # Only roll if we didn't use FT for booked
                roll_path.roll_transfer(gw)
            elif working_path.ft > 0:  # Had FTs left after booked transfer
                roll_path.roll_transfer(gw)
            new_paths.append(roll_path)
            
            # Option 2: Use remaining FT for additional transfer (after booked or as main)
            if working_path.ft > 0:
                # Get candidates - exclude transfers that would:
                # 1. Remove a player we booked in (now or future)
                # 2. Remove a player we plan to book out later (before their booked GW)
                candidates = get_transfer_candidates(
                    working_path.squad, all_players, teams_dict, fixtures, gw, end_gw - gw, 
                    working_path.bank, limit=5, elements=elements
                )
                
                def is_protected(player_id):
                    # Can't transfer out booked-in players
                    if player_id in all_booked_in_ids:
                        return True
                    # Can't transfer out players before their booked-out GW
                    if player_id in booked_out_before_gw and gw < booked_out_before_gw[player_id]:
                        return True
                    return False
                
                candidates = [c for c in candidates if not is_protected(c["out"]["id"])]
                
                for i, transfer in enumerate(candidates[:3]):
                    ft_path = working_path.copy()
                    ft_path.use_ft()
                    ft_path.apply_transfer(transfer, gw, is_hit=False)
                    recompute_from(ft_path, gw)
                    new_paths.append(ft_path)
                    
                    # Option 3: Take hit for second transfer (if allowed)
                    if max_hits > 0 and working_path.hits < max_hits and len(candidates) > 1:
                        for second_transfer in candidates[i+1:i+3]:
                            if second_transfer["out"]["id"] == transfer["in"]["id"]:
                                continue
                            if second_transfer["in"]["id"] == transfer["out"]["id"]:
                                continue
                            
                            hit_path = ft_path.copy()
                            hit_path.apply_transfer(second_transfer, gw, is_hit=True)
                            recompute_from(hit_path, gw)
                            new_paths.append(hit_path)
        
        # Prune: keep top 20 paths by xPts
        new_paths.sort(key=lambda p: -p.total_xpts)
        paths = new_paths[:20]
    
    # Find best path
    best_path = max(paths, key=lambda p: p.total_xpts)
    
    # Build timeline with per-GW squad snapshots
    timeline = []
    current_squad = squad.copy()
    
    for gw in range(start_gw, end_gw):
        gw_actions = [a for a in best_path.actions if a.get("gw") == gw]
        chip = chip_gws.get(gw)
        
        # Apply transfers for this GW to get the squad state
        for action in gw_actions:
            if action["type"] == "transfer":
                # Remove the out player and add the in player
                current_squad = [p for p in current_squad if p["id"] != action["out"]["id"]]
                current_squad.append({
                    "id": action["in"]["id"],
                    "name": action["in"]["name"],
                    "team_id": action["in"].get("team_id", 0),
                    "position_id": action["in"].get("position_id", 0),
                    "price": action["in"].get("price", 0),
                    "selling_price": action["in"].get("price", 0),
                    "xpts": action["in"].get("xpts", 0),
                    "form": action["in"].get("form", 0),
                    "minutes": action["in"].get("minutes", 0),
                })
        
        # Calculate xPts for each player for this specific GW
        squad_with_gw_xpts = []
        for player in current_squad:
            element = elements.get(player["id"], {})
            position_id = player.get("position_id") or element.get("element_type", 0)
            team_id = player.get("team_id") or element.get("team", 0)
            
            upcoming = get_player_upcoming_fixtures(team_id, fixtures, gw, gw, teams_dict)
            if upcoming:
                stats = calculate_expected_points(element, position_id, gw, upcoming, teams_dict, fixtures, events)
                # calculate_expected_points already sums xpts for all fixtures in the GW - do NOT divide
                gw_xpts = stats.get("xpts", 0)
            else:
                gw_xpts = player.get("xpts", 0) / horizon if horizon > 0 else 0
            
            squad_with_gw_xpts.append({
                "id": player["id"],
                "name": player.get("name", element.get("web_name", "?")),
                "team": teams_dict.get(team_id, {}).get("short_name", "???"),
                "position": POSITION_MAP.get(position_id, "?"),
                "position_id": position_id,
                "xpts": round(gw_xpts, 1),
                "price": player.get("price", element.get("now_cost", 0) / 10),
            })
        
        # Sort by position for lineup selection, then by xPts
        squad_with_gw_xpts.sort(key=lambda p: (p["position_id"], -p["xpts"]))
        
        # Select starting XI (best by xPts, respecting position constraints)
        gkps = [p for p in squad_with_gw_xpts if p["position_id"] == 1]
        defs = [p for p in squad_with_gw_xpts if p["position_id"] == 2]
        mids = [p for p in squad_with_gw_xpts if p["position_id"] == 3]
        fwds = [p for p in squad_with_gw_xpts if p["position_id"] == 4]
        
        # Sort each position by xPts
        defs.sort(key=lambda p: -p["xpts"])
        mids.sort(key=lambda p: -p["xpts"])
        fwds.sort(key=lambda p: -p["xpts"])
        
        # Standard formation logic: pick best 11 while respecting min/max constraints
        # Min: 1 GKP, 3 DEF, 2 MID, 1 FWD
        starting_xi = []
        bench = []
        
        # Always 1 GKP
        if gkps:
            starting_xi.append(gkps[0])
            if len(gkps) > 1:
                bench.extend(gkps[1:])
        
        # Pick best outfield players (10 spots)
        outfield = defs + mids + fwds
        outfield.sort(key=lambda p: -p["xpts"])
        
        # Ensure minimums: 3 DEF, 2 MID, 1 FWD
        def_count = mid_count = fwd_count = 0
        remaining = []
        
        for p in outfield:
            if p["position_id"] == 2 and def_count < 3:
                starting_xi.append(p)
                def_count += 1
            elif p["position_id"] == 3 and mid_count < 2:
                starting_xi.append(p)
                mid_count += 1
            elif p["position_id"] == 4 and fwd_count < 1:
                starting_xi.append(p)
                fwd_count += 1
            else:
                remaining.append(p)
        
        # Fill remaining spots (up to 11 total) with best remaining players
        # Respecting max: 5 DEF, 5 MID, 3 FWD
        remaining.sort(key=lambda p: -p["xpts"])
        for p in remaining:
            if len(starting_xi) >= 11:
                bench.append(p)
            else:
                pos_in_xi = len([x for x in starting_xi if x["position_id"] == p["position_id"]])
                max_pos = {2: 5, 3: 5, 4: 3}.get(p["position_id"], 1)
                if pos_in_xi < max_pos:
                    starting_xi.append(p)
                else:
                    bench.append(p)
        
        # Sort starting XI by position for display
        starting_xi.sort(key=lambda p: (p["position_id"], -p["xpts"]))
        
        # Sort bench by xPts (optimal bench order)
        bench.sort(key=lambda p: -p["xpts"])
        
        # Captain selection using ceiling score, not just raw xPts
        if starting_xi:
            for p in starting_xi:
                player_data = elements.get(p["id"], {})
                player_team_id = p.get("team_id") or player_data.get("team")
                
                # Get next fixture for this player
                player_next_fixtures = get_player_upcoming_fixtures(
                    player_team_id, fixtures, gw, gw, teams_dict
                )
                next_fix = player_next_fixtures[0] if player_next_fixtures else None
                
                cap_data = calculate_captain_score(
                    player=player_data,
                    position_id=p["position_id"],
                    base_xpts=p["xpts"],
                    next_fixture=next_fix
                )
                p["captain_score"] = cap_data["captain_score"]
                p["ceiling_mult"] = cap_data["ceiling_mult"]
            
            starting_xi_by_cap = sorted(starting_xi, key=lambda p: -p.get("captain_score", p["xpts"]))
            captain = starting_xi_by_cap[0]
            vice_captain = starting_xi_by_cap[1] if len(starting_xi_by_cap) > 1 else None
        else:
            captain = None
            vice_captain = None
        
        # Calculate total GW xPts
        gw_total_xpts = sum(p["xpts"] for p in starting_xi)
        if captain:
            gw_total_xpts += captain["xpts"]  # Captain counted twice
        
        timeline.append({
            "gw": gw,
            "actions": gw_actions,
            "chip": CHIP_DISPLAY.get(chip) if chip else None,
            "is_dgw": gw_info[gw]["is_dgw"],
            "is_bgw": gw_info[gw]["is_bgw"],
            "dgw_count": gw_info[gw]["dgw_count"],
            "starting_xi": starting_xi,
            "bench": bench,
            "captain": captain,
            "vice_captain": vice_captain,
            "gw_xpts": round(gw_total_xpts, 1),
        })
    
    return {
        "total_xpts": round(best_path.total_xpts, 1),
        "baseline_xpts": round(baseline_xpts, 1),
        "xpts_gain": round(best_path.total_xpts - baseline_xpts, 1),
        "total_hits": best_path.hits,
        "hit_cost": best_path.hits * 4,
        "net_xpts_gain": round(best_path.total_xpts - baseline_xpts, 1),
        "final_bank": round(best_path.bank, 1),
        "timeline": timeline,
        "transfers": [a for a in best_path.actions if a["type"] == "transfer"],
        "final_squad": best_path.squad,
    }


@app.get("/api/planner/{manager_id}")
async def get_transfer_plan(
    manager_id: int,
    horizon: int = Query(3, ge=1, le=6),
    max_hits: int = Query(1, ge=0, le=3),
    wc_gw: Optional[int] = Query(None, ge=1, le=38),
    tc_gw: Optional[int] = Query(None, ge=1, le=38),
    bb_gw: Optional[int] = Query(None, ge=1, le=38),
    fh_gw: Optional[int] = Query(None, ge=1, le=38),
    booked_transfers: Optional[str] = Query(None, description="JSON array of booked transfers [{gw, out, in}]"),
):
    """
    Multi-gameweek transfer planner.
    
    Returns optimal transfer path over horizon GWs.
    User can specify which GWs to use chips (or leave for auto-suggestion).
    booked_transfers: JSON string like '[{"gw":23,"out":"Salah","in":"Palmer"}]'
    """
    try:
        await refresh_fdr_data()
        
        # Parse booked transfers JSON
        parsed_booked = []
        if booked_transfers:
            try:
                parsed_booked = json.loads(booked_transfers)
            except:
                pass  # Ignore invalid JSON
        
        data = await fetch_fpl_data()
        fixtures = await fetch_fixtures()
        history = await fetch_manager_history(manager_id)
        
        elements = {e["id"]: e for e in data["elements"]}
        all_players = data["elements"]
        teams = {t["id"]: t for t in data["teams"]}
        events = data["events"]
        current_gw = get_current_gameweek(events)
        next_gw = get_next_gameweek(events)
        
        # Get current squad
        team_data = await fetch_manager_team(manager_id, current_gw)
        picks = team_data["picks"]["picks"]
        entry_history = team_data["picks"].get("entry_history", {})
        bank = entry_history.get("bank", 0) / 10
        
        # ==================== CALCULATE FREE TRANSFERS ====================
        # FPL rules (2024/25):
        # - Start each GW with 1 FT
        # - If you don't use your FT, it rolls over (max 5 FTs banked)
        # - Making 0 transfers = +1 FT for next GW
        # - Making N transfers where N > FT = (N - FT) × 4 point hit
        #
        # We can estimate FT balance from recent transfer history:
        # - Look at last few GWs of transfers
        # - Count consecutive 0-transfer GWs = rolled FTs
        # - But we can't know exact starting balance without full history
        
        # Get manager's transfer history to estimate FT balance
        manager_transfers = team_data.get("transfers", [])
        manager_history = history.get("current", [])
        
        # Count how many FTs the manager likely has
        free_transfers = 1  # Minimum is always 1
        
        # Look at recent GWs to count rolls
        recent_gws = sorted([h for h in manager_history if h.get("event", 0) <= current_gw], 
                          key=lambda x: x.get("event", 0), reverse=True)[:5]
        
        consecutive_rolls = 0
        for gw_data in recent_gws:
            transfers_made = gw_data.get("event_transfers", 0)
            if transfers_made == 0:
                consecutive_rolls += 1
            else:
                break  # Stop counting once we hit a GW with transfers
        
        # Each consecutive 0-transfer GW = +1 FT (capped at 5)
        # But we need to account for the current GW too
        current_gw_transfers = entry_history.get("event_transfers", 0)
        
        if current_gw_transfers == 0:
            # No transfers made this GW yet - they have their FTs
            free_transfers = min(1 + consecutive_rolls, 5)
        else:
            # They already made transfers this GW
            # We're planning for NEXT GW, so they'll have 1 FT
            free_transfers = 1
        
        ft_estimation_method = "history_analysis"
        
        # Build squad list
        squad = []
        for pick in picks:
            player = elements.get(pick["element"])
            if not player:
                continue
            
            position_id = player["element_type"]
            upcoming = get_player_upcoming_fixtures(player["team"], fixtures, next_gw, next_gw + horizon, teams)
            stats = calculate_expected_points(player, position_id, current_gw, upcoming, teams, fixtures, events)
            
            squad.append({
                "id": player["id"],
                "name": player["web_name"],
                "team_id": player["team"],
                "position_id": position_id,
                "price": player["now_cost"] / 10,
                "selling_price": pick.get("selling_price", player["now_cost"]) / 10,
                "xpts": stats["xpts"],
                "form": float(player.get("form", 0) or 0),
                "minutes": player.get("minutes", 0),
            })
        
        # Get available chips
        available_chips = get_available_chips(history, current_gw)
        
        # Build chip schedule from user input
        chip_gws = {}
        if wc_gw and available_chips.get("wildcard"):
            chip_gws["wildcard"] = wc_gw
        if tc_gw and available_chips.get("3xc"):
            chip_gws["3xc"] = tc_gw
        if bb_gw and available_chips.get("bboost"):
            chip_gws["bboost"] = bb_gw
        if fh_gw and available_chips.get("freehit"):
            chip_gws["freehit"] = fh_gw
        
        # Detect DGW/BGW for chip suggestions
        gw_info = detect_dgw_bgw(fixtures, events, next_gw, next_gw + horizon)
        
        # Auto-suggest chips if not specified
        chip_suggestions = []
        for gw, info in gw_info.items():
            if info["is_dgw"] and info["dgw_count"] >= 5:
                if available_chips.get("bboost") and "bboost" not in chip_gws:
                    chip_suggestions.append({
                        "chip": "BB",
                        "gw": gw,
                        "reason": f"DGW with {info['dgw_count']} teams having doubles",
                    })
                if available_chips.get("3xc") and "3xc" not in chip_gws:
                    chip_suggestions.append({
                        "chip": "TC",
                        "gw": gw,
                        "reason": f"DGW opportunity for captain",
                    })
        
        # Run solver
        plan = solve_transfer_plan(
            squad=squad,
            bank=bank,
            free_transfers=free_transfers,
            all_players=all_players,
            teams_dict=teams,
            elements=elements,
            fixtures=fixtures,
            events=events,
            start_gw=next_gw,
            horizon=horizon,
            chip_gws=chip_gws,
            max_hits=max_hits,
            booked_transfers=parsed_booked,
        )
        
        return {
            "manager_id": manager_id,
            "current_gw": current_gw,
            "planning_horizon": f"GW{next_gw}-GW{next_gw + horizon - 1}",
            "available_chips": {CHIP_DISPLAY.get(k, k): v for k, v in available_chips.items()},
            "chip_schedule": {CHIP_DISPLAY.get(k, k): v for k, v in chip_gws.items()},
            "chip_suggestions": chip_suggestions,
            "booked_transfers": parsed_booked,
            "gw_info": gw_info,
            "bank": bank,
            "free_transfers": free_transfers,
            "ft_disclaimer": "FT count is estimated from recent transfer history. Verify before confirming transfers.",
            "plan": plan,
        }
    except Exception as e:
        logger.error(f"Transfer planner error for manager {manager_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============ MODEL BACKTESTING ============

@app.get("/api/backtest/gw/{gw}")
async def backtest_gameweek(
    gw: int = Query(..., ge=1, le=38),
    position: Optional[Position] = None,
    min_minutes: int = Query(200, ge=0),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Backtest the xPts model against actual GW results.
    
    Compares predicted xPts (calculated BEFORE the GW) vs actual points.
    Useful for model validation and calibration.
    
    Returns:
    - Per-player: predicted xPts, actual points, error
    - Aggregate: MAE, RMSE, correlation
    - Breakdown by position
    """
    await refresh_fdr_data()
    
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    
    elements = data["elements"]
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    
    # Filter by position if specified
    if position:
        position_id = POSITION_ID_MAP[position.value]
        elements = [e for e in elements if e["element_type"] == position_id]
    
    # Filter by minutes
    elements = [e for e in elements if e.get("minutes", 0) >= min_minutes]
    
    results = []
    errors = []
    position_stats = defaultdict(lambda: {"errors": [], "count": 0})
    
    for player in elements:
        # Get player's actual points for this GW
        try:
            history = await fetch_player_history(player["id"])
            gw_data = next((h for h in history.get("history", []) if h.get("round") == gw), None)
            
            if not gw_data:
                continue
            
            actual_points = gw_data.get("total_points", 0)
            actual_minutes = gw_data.get("minutes", 0)
            
            # Skip players who didn't play (can't validate model on 0-minute players)
            if actual_minutes == 0:
                continue
            
            # Calculate what our model would have predicted
            # Use fixtures from that GW
            position_id = player["element_type"]
            upcoming = get_player_upcoming_fixtures(player["team"], fixtures, gw, gw, teams, position_id)
            
            # Calculate xPts for single GW
            stats = calculate_expected_points(
                player, position_id, gw - 1, upcoming, teams, fixtures, events
            )
            
            predicted_xpts = stats["xpts"]
            error = predicted_xpts - actual_points
            
            results.append({
                "id": player["id"],
                "name": player["web_name"],
                "team": teams.get(player["team"], {}).get("short_name", "???"),
                "position": POSITION_MAP.get(position_id, "???"),
                "predicted_xpts": round(predicted_xpts, 2),
                "actual_points": actual_points,
                "error": round(error, 2),
                "abs_error": round(abs(error), 2),
                "minutes": actual_minutes,
            })
            
            errors.append(error)
            pos_name = POSITION_MAP.get(position_id, "???")
            position_stats[pos_name]["errors"].append(error)
            position_stats[pos_name]["count"] += 1
            
        except Exception as e:
            logger.warning(f"Backtest error for player {player['id']}: {e}")
            continue
    
    if not errors:
        return {"error": "No valid data for backtesting", "gw": gw}
    
    # Calculate aggregate stats
    import statistics
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
    
    # Calculate correlation (Pearson)
    if len(results) >= 3:
        predicted = [r["predicted_xpts"] for r in results]
        actual = [r["actual_points"] for r in results]
        
        mean_pred = statistics.mean(predicted)
        mean_actual = statistics.mean(actual)
        
        numerator = sum((p - mean_pred) * (a - mean_actual) for p, a in zip(predicted, actual))
        denom_pred = math.sqrt(sum((p - mean_pred)**2 for p in predicted))
        denom_actual = math.sqrt(sum((a - mean_actual)**2 for a in actual))
        
        if denom_pred > 0 and denom_actual > 0:
            correlation = numerator / (denom_pred * denom_actual)
        else:
            correlation = 0
    else:
        correlation = None
    
    # Position breakdown
    pos_breakdown = {}
    for pos, data in position_stats.items():
        if data["errors"]:
            pos_mae = sum(abs(e) for e in data["errors"]) / len(data["errors"])
            pos_breakdown[pos] = {
                "count": data["count"],
                "mae": round(pos_mae, 2),
                "avg_error": round(statistics.mean(data["errors"]), 2),
            }
    
    # Sort results by absolute error (worst predictions first)
    results.sort(key=lambda x: -x["abs_error"])
    
    return {
        "gw": gw,
        "sample_size": len(results),
        "aggregate_stats": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "correlation": round(correlation, 3) if correlation else None,
            "avg_error": round(statistics.mean(errors), 2),
            "std_error": round(statistics.stdev(errors), 2) if len(errors) > 1 else None,
        },
        "position_breakdown": pos_breakdown,
        "worst_predictions": results[:10],
        "best_predictions": sorted(results, key=lambda x: x["abs_error"])[:10],
        "all_results": results[:limit],
    }


@app.get("/api/backtest/summary")
async def backtest_summary(
    gw_start: int = Query(1, ge=1, le=38),
    gw_end: int = Query(10, ge=1, le=38),
):
    """
    Backtest summary across multiple gameweeks.
    
    Returns aggregate model performance metrics.
    """
    all_errors = []
    gw_stats = []
    
    for gw in range(gw_start, min(gw_end + 1, 39)):
        try:
            result = await backtest_gameweek(gw=gw, limit=200)
            if "error" not in result:
                all_errors.extend([r["error"] for r in result.get("all_results", [])])
                gw_stats.append({
                    "gw": gw,
                    "mae": result["aggregate_stats"]["mae"],
                    "correlation": result["aggregate_stats"]["correlation"],
                    "sample_size": result["sample_size"],
                })
        except Exception as e:
            logger.warning(f"Backtest failed for GW{gw}: {e}")
            continue
    
    if not all_errors:
        return {"error": "No valid backtest data"}
    
    import statistics
    
    return {
        "gw_range": f"GW{gw_start}-GW{gw_end}",
        "total_predictions": len(all_errors),
        "overall_mae": round(sum(abs(e) for e in all_errors) / len(all_errors), 2),
        "overall_rmse": round(math.sqrt(sum(e**2 for e in all_errors) / len(all_errors)), 2),
        "avg_correlation": round(statistics.mean([g["correlation"] for g in gw_stats if g["correlation"]]), 3) if gw_stats else None,
        "gw_breakdown": gw_stats,
    }


# ============ VARIANCE AND HOME/AWAY SPLIT ENDPOINTS ============

@app.get("/api/player/{player_id}/variance")
async def get_player_variance(player_id: int):
    """
    Get variance statistics for a player.
    
    Useful for understanding ceiling/floor calculations and
    validating the variance model.
    """
    data = await fetch_fpl_data()
    player = next((p for p in data["elements"] if p["id"] == player_id), None)
    
    if not player:
        raise HTTPException(404, "Player not found")
    
    history = await fetch_player_history(player_id)
    position_id = player["element_type"]
    
    # Calculate variance
    variance_result = variance_model.calculate_variance(
        player_history=history,
        player_data=player,
        position_id=position_id,
        xpts=5.0,  # Reference xpts for calculation
        fixture_fdr=5,
        is_home=True
    )
    
    # Get historical points distribution
    points_dist = []
    if history and history.get("history"):
        points_dist = [
            h.get("total_points", 0) 
            for h in history["history"] 
            if h.get("minutes", 0) >= 45
        ]
    
    return {
        "player_id": player_id,
        "name": player["web_name"],
        "position": POSITION_MAP.get(position_id),
        "variance": {
            "std_dev": variance_result.std_dev,
            "ceiling_per_fixture": variance_result.ceiling,
            "floor_per_fixture": variance_result.floor,
            "data_source": variance_result.data_source,
        },
        "historical_points": {
            "games_played": len(points_dist),
            "min": min(points_dist) if points_dist else None,
            "max": max(points_dist) if points_dist else None,
            "mean": round(sum(points_dist) / len(points_dist), 2) if points_dist else None,
            "recent_20": points_dist[-20:],
        }
    }


@app.get("/api/player/{player_id}/home-away-split")
async def get_player_home_away_split(player_id: int):
    """
    Get home vs away performance splits for a player.
    
    Shows xG90, xA90, and points per 90 for home vs away games.
    """
    data = await fetch_fpl_data()
    player = next((p for p in data["elements"] if p["id"] == player_id), None)
    
    if not player:
        raise HTTPException(404, "Player not found")
    
    history = await fetch_player_history(player_id)
    
    # Calculate splits
    split = home_away_calculator.calculate_splits(history, player)
    
    # Overall stats for comparison
    total_minutes = int(player.get("minutes", 0) or 0)
    mins90 = max(total_minutes / 90.0, 0.1)
    xG = float(player.get("expected_goals", 0) or 0)
    xA = float(player.get("expected_assists", 0) or 0)
    
    return {
        "player_id": player_id,
        "name": player["web_name"],
        "team": player["team"],
        "position": POSITION_MAP.get(player["element_type"]),
        "overall": {
            "xG_per_90": round(xG / mins90, 3),
            "xA_per_90": round(xA / mins90, 3),
            "xGI_per_90": round((xG + xA) / mins90, 3),
            "total_games": split.home_games + split.away_games,
        },
        "home": {
            "xG_per_90": round(split.home_xG90, 3),
            "xA_per_90": round(split.home_xA90, 3),
            "xGI_per_90": round(split.home_xGI90, 3),
            "pts_per_90": round(split.home_pts_per_90, 2),
            "games": split.home_games,
        },
        "away": {
            "xG_per_90": round(split.away_xG90, 3),
            "xA_per_90": round(split.away_xA90, 3),
            "xGI_per_90": round(split.away_xGI90, 3),
            "pts_per_90": round(split.away_pts_per_90, 2),
            "games": split.away_games,
        },
        "has_sufficient_data": split.has_sufficient_data,
        "min_games_required": MODEL_CONFIG["home_away"].min_games_for_split,
    }


@app.get("/api/config")
async def get_model_config():
    """
    Get current model configuration.
    
    Useful for understanding calibration values and debugging.
    """
    return {
        "fdr": {
            "multipliers": MODEL_CONFIG["fdr"].fdr_multipliers,
            "premium_dampening": MODEL_CONFIG["fdr"].premium_dampening,
            "mid_price_dampening": MODEL_CONFIG["fdr"].mid_price_dampening,
            "premium_threshold": MODEL_CONFIG["fdr"].premium_price_threshold,
        },
        "home_away": {
            "home_attack_boost": MODEL_CONFIG["home_away"].home_attack_boost,
            "away_attack_penalty": MODEL_CONFIG["home_away"].away_attack_penalty,
            "split_weight": MODEL_CONFIG["home_away"].split_weight,
            "min_games_for_split": MODEL_CONFIG["home_away"].min_games_for_split,
        },
        "variance": {
            "position_stdev": MODEL_CONFIG["variance"].position_stdev_baseline,
            "min_games_for_stdev": MODEL_CONFIG["variance"].min_games_for_stdev,
        },
        "xpts": {
            "fixture_weights": MODEL_CONFIG["xpts"].fixture_weights,
            "defcon_threshold_def": MODEL_CONFIG["xpts"].defcon_threshold_def,
            "defcon_threshold_mid": MODEL_CONFIG["xpts"].defcon_threshold_mid,
        }
    }


# ============ HEALTH CHECK ============

@app.get("/api/health")
async def health_check():
    """Health check endpoint with cache status."""
    return {
        "status": "ok",
        "cache": {
            "bootstrap_data": cache.bootstrap_data is not None,
            "fixtures_data": cache.fixtures_data is not None,
            "fdr_data_teams": len(cache.fdr_data),
            "player_histories_cached": len(cache.player_histories),
            "minutes_overrides": len(cache.minutes_overrides),
            "predicted_minutes": len(cache.predicted_minutes),
        },
        "fdr_last_update": cache.fdr_last_update.isoformat() if cache.fdr_last_update else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
