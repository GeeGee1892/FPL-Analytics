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
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
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
    
    try:
        await refresh_fdr_data()
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

# New FDR multipliers (1-10 scale, 10 = hardest)
# Higher FDR = harder fixture = lower expected points
FDR_MULTIPLIERS_10 = {
    1: 1.30,   # Very easy
    2: 1.20,
    3: 1.10,
    4: 1.05,
    5: 1.00,   # Neutral
    6: 0.95,
    7: 0.90,
    8: 0.85,
    9: 0.75,
    10: 0.65,  # Very hard
}

# Fallback: Map old FPL 1-5 scale to our 1-10 scale
FPL_FDR_TO_10 = {1: 2, 2: 4, 3: 5, 4: 7, 5: 9}

# DEFCON thresholds (2025/26 rules)
DEFCON_THRESHOLD_DEF = 10
DEFCON_THRESHOLD_MID = 12
DEFCON_POINTS = 2

# European teams for rotation modeling
EUROPEAN_TEAMS_UCL = {"Arsenal", "Aston Villa", "Liverpool", "Man City"}
EUROPEAN_TEAMS_UEL = {"Man Utd", "Tottenham"}
EUROPEAN_TEAMS_UECL = {"Chelsea"}

# Manager rotation tendencies
ROTATION_MANAGERS = {"Man City": 0.85, "Chelsea": 0.88, "Brighton": 0.90, "Newcastle": 0.92}
STABLE_MANAGERS = {"Arsenal": 1.0, "Liverpool": 0.98, "Fulham": 1.0, "Bournemouth": 1.0, "Brentford": 0.98}

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
    "IPS": (10, "Ipswich"),
    "LEI": (11, "Leicester"),
    "LIV": (12, "Liverpool"),
    "MCI": (13, "Manchester City"),
    "MUN": (14, "Manchester United"),
    "NEW": (15, "Newcastle United"),
    "NFO": (16, "Nottingham Forest"),
    "SOU": (17, "Southampton"),
    "TOT": (18, "Tottenham"),
    "WHU": (19, "West Ham"),
    "WOL": (20, "Wolverhampton Wanderers"),
}

UNDERSTAT_TO_FPL_ID = {v[1]: v[0] for v in TEAM_NAME_MAPPING.values()}
FPL_ID_TO_UNDERSTAT = {v[0]: v[1] for v in TEAM_NAME_MAPPING.values()}

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
        return self.predicted_minutes_last_update is None or (datetime.now() - self.predicted_minutes_last_update).seconds > 3600  # 1 hour

    def is_stale(self) -> bool:
        return self.last_update is None or (datetime.now() - self.last_update).seconds > self.cache_duration

    def fixtures_is_stale(self) -> bool:
        return self.fixtures_last_update is None or (datetime.now() - self.fixtures_last_update).seconds > self.cache_duration
    
    def fdr_is_stale(self) -> bool:
        return self.fdr_last_update is None or (datetime.now() - self.fdr_last_update).seconds > 21600  # 6 hours


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
        
        fdr_data[team_id] = {
            'season_xg': data['season_xg'],
            'season_xga': data['season_xga'],
            'season_matches': len(data['matches']),
            'form_xg': round(form_xg, 2),
            'form_xga': round(form_xga, 2),
            'form_ppg': round(form_ppg, 2),
        }
        all_form_xg.append(form_xg)
        all_form_xga.append(form_xga)
    
    # Compute FDR scores using percentiles
    if len(all_form_xg) >= 5:
        for team_id, data in fdr_data.items():
            # Attack FDR: How hard to attack this team (based on their xGA)
            # Low xGA = hard to score against = HIGH attack_fdr
            xga_percentile = percentileofscore(all_form_xga, data['form_xga'])
            attack_fdr = max(1, min(10, int((100 - xga_percentile) / 10) + 1))
            
            # Defence FDR: How hard to defend against this team (based on their xG)
            # High xG = dangerous attack = HIGH defence_fdr
            xg_percentile = percentileofscore(all_form_xg, data['form_xg'])
            defence_fdr = max(1, min(10, int(xg_percentile / 10) + 1))
            
            data['attack_fdr'] = attack_fdr
            data['defence_fdr'] = defence_fdr
            data['composite_fdr'] = round((attack_fdr + defence_fdr) / 2, 1)
    
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
    if defcon_per_90 <= 0:
        prob = 0.0
    else:
        scale = 2.0
        prob = 1.0 / (1.0 + math.exp(-(defcon_per_90 - threshold) / scale))
    
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


def calculate_expected_minutes(
    player: Dict,
    current_gw: int,
    team_name: str,
    fixtures: List[Dict],
    events: List[Dict],
    player_history: Optional[Dict] = None,
    override_minutes: Optional[float] = None,
    manager_id: Optional[int] = None
) -> tuple[float, str]:
    player_id = player.get("id")
    if override_minutes is not None:
        return override_minutes, "user_override"
    
    # Check cache for override (manager-specific or global)
    cached_override = cache.get_minutes_override(player_id, manager_id)
    if cached_override is not None:
        # Determine if it's a user override or predicted
        if (manager_id, player_id) in cache.minutes_overrides or (None, player_id) in cache.minutes_overrides:
            return cached_override, "user_override"
        else:
            return cached_override, "predicted"
    
    total_minutes = int(player.get("minutes", 0) or 0)
    starts = int(player.get("starts", 0) or 0)
    chance_of_playing = player.get("chance_of_playing_next_round")
    status = player.get("status", "a")
    
    available_gws = max(current_gw - 1, 1)
    
    if starts == 0 or total_minutes < 90:
        base_mins = 10.0
        reason = "insufficient_data"
    else:
        mins_per_start = total_minutes / starts
        start_rate = min(1.0, starts / available_gws)
        base_mins = mins_per_start * start_rate
        reason = "season_average"
    
    if status in ('i', 'u', 's'):
        if chance_of_playing and chance_of_playing > 0:
            base_mins = base_mins * (chance_of_playing / 100) * 0.5
        else:
            base_mins = 0
        reason = "unavailable"
    elif status == 'd' or (chance_of_playing is not None and chance_of_playing < 75):
        cop = chance_of_playing / 100 if chance_of_playing else 0.5
        base_mins = base_mins * cop
        reason = "doubtful"
    
    if team_name in ROTATION_MANAGERS:
        rotation_factor = ROTATION_MANAGERS[team_name]
        base_mins = base_mins * rotation_factor
        if reason == "season_average":
            reason = "rotation_adjusted"
    
    return round(min(90, max(0, base_mins)), 1), reason


def calculate_expected_points(
    player: Dict,
    position: int,
    current_gw: int,
    upcoming_fixtures: List[Dict],
    teams_dict: Dict,
    all_fixtures: List[Dict] = None,
    events: List[Dict] = None,
    player_history: Optional[Dict] = None,
    override_minutes: Optional[float] = None
) -> Dict:
    """
    Enhanced expected points using position-specific FDR from Understat xG data.
    """
    total_minutes = int(player.get("minutes", 0) or 0)
    xG = float(player.get("expected_goals", 0) or 0)
    xA = float(player.get("expected_assists", 0) or 0)
    xGC = float(player.get("expected_goals_conceded", 0) or 0)
    points_per_game = float(player.get("points_per_game", 0) or 0)
    
    team_id = player["team"]
    team_name = teams_dict.get(team_id, {}).get("name", "")
    
    exp_mins, mins_reason = calculate_expected_minutes(
        player, current_gw, team_name, all_fixtures or [], events or [],
        player_history, override_minutes
    )
    
    minutes_factor = exp_mins / 90.0
    
    mins90 = max(total_minutes / 90.0, 0.1)
    xG90 = xG / mins90
    xA90 = xA / mins90
    xGC90 = xGC / mins90
    
    cs_prob = math.exp(-max(0.0, xGC90))
    
    defcon_per_90, defcon_prob, defcon_pts_total = calculate_defcon_per_90(player, position)
    saves_per_90, save_pts_per_90 = calculate_saves_per_90(player) if position == 1 else (0, 0)
    
    app_pts = 2.0
    
    if position == 1:  # GKP
        base90 = app_pts + (4.0 * cs_prob) + save_pts_per_90 + 0.5
    elif position == 2:  # DEF
        defcon_pts = defcon_prob * DEFCON_POINTS
        base90 = app_pts + (4.0 * cs_prob) + (6.0 * xG90) + (3.0 * xA90) + defcon_pts
    elif position == 3:  # MID
        defcon_pts = defcon_prob * DEFCON_POINTS
        base90 = app_pts + (1.0 * cs_prob) + (5.0 * xG90) + (3.0 * xA90) + defcon_pts
    else:  # FWD
        base90 = app_pts + (4.0 * xG90) + (3.0 * xA90)
    
    base90 = 0.6 * base90 + 0.4 * points_per_game
    
    # Apply fixture difficulty using new position-specific FDR
    horizon = upcoming_fixtures[:8]
    if not horizon:
        total = base90
    else:
        total = 0
        for fix in horizon:
            opponent_id = fix.get("opponent_id")
            is_home = fix.get("is_home", True)
            
            # Get position-specific FDR
            if cache.fdr_data and opponent_id in cache.fdr_data:
                fdr = get_fixture_fdr(opponent_id, is_home, position)
            else:
                # Fallback to FPL difficulty mapped to 1-10
                fpl_diff = fix.get("difficulty", 3)
                fdr = FPL_FDR_TO_10.get(fpl_diff, 5)
            
            multiplier = FDR_MULTIPLIERS_10.get(fdr, 1.0)
            total += base90 * multiplier
    
    expected_pts = total * minutes_factor
    
    return {
        "xpts": round(expected_pts, 2),
        "xpts_per_90": round(base90, 2),
        "expected_minutes": exp_mins,
        "minutes_reason": mins_reason,
        "minutes_factor": round(minutes_factor, 3),
        "defcon_per_90": defcon_per_90,
        "defcon_prob": defcon_prob,
        "defcon_pts_total": defcon_pts_total,
        "saves_per_90": saves_per_90,
        "save_pts_per_90": save_pts_per_90,
        "xGC_per_90": round(xGC90, 2),
        "cs_prob": round(cs_prob, 2),
        "xG_per_90": round(xG90, 2),
        "xA_per_90": round(xA90, 2),
    }


def get_player_upcoming_fixtures(
    player_team: int, fixtures: List[Dict], current_gw: int, gw_end: int, teams_dict: Dict
) -> List[Dict]:
    upcoming = []
    for fix in fixtures:
        gw = fix.get("event")
        if gw and current_gw <= gw <= gw_end:
            if fix["team_h"] == player_team:
                opponent_id = fix["team_a"]
                upcoming.append({
                    "gameweek": gw,
                    "opponent": teams_dict.get(opponent_id, {}).get("short_name", "???"),
                    "opponent_id": opponent_id,
                    "is_home": True,
                    "difficulty": fix.get("team_h_difficulty", 3)
                })
            elif fix["team_a"] == player_team:
                opponent_id = fix["team_h"]
                upcoming.append({
                    "gameweek": gw,
                    "opponent": teams_dict.get(opponent_id, {}).get("short_name", "???"),
                    "opponent_id": opponent_id,
                    "is_home": False,
                    "difficulty": fix.get("team_a_difficulty", 3)
                })
    return sorted(upcoming, key=lambda x: x["gameweek"])


# ============ MINUTES OVERRIDE ENDPOINTS ============

@app.post("/api/minutes-override")
async def set_minutes_override_endpoint(override: MinutesOverride):
    """Set expected minutes override for a player. Optionally scoped to a manager."""
    cache.set_minutes_override(
        player_id=override.player_id,
        minutes=override.expected_minutes,
        manager_id=override.manager_id
    )
    return {
        "status": "ok",
        "player_id": override.player_id,
        "expected_minutes": override.expected_minutes,
        "manager_id": override.manager_id
    }


@app.delete("/api/minutes-override/{player_id}")
async def remove_minutes_override_endpoint(player_id: int, manager_id: Optional[int] = None):
    """Remove minutes override for a player."""
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


# ============ RANKINGS ENDPOINT ============

@app.get("/api/rankings/{position}")
async def get_rankings(
    position: Position,
    gw_start: int = Query(1, ge=1, le=38),
    gw_end: int = Query(8, ge=1, le=38),
    min_minutes: int = Query(MIN_MINUTES_DEFAULT, ge=0),
    min_price: float = Query(0, ge=0),
    max_price: float = Query(20, ge=0, le=20),
    max_ownership: float = Query(100, ge=0, le=100),
    limit: int = Query(50, ge=1, le=200),
    sort_by: str = Query("xpts", pattern="^(xpts|price|form|ownership|defcon_per_90|saves_per_90|expected_minutes|total_points)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$")
):
    if gw_end < gw_start:
        raise HTTPException(status_code=400, detail="gw_end must be >= gw_start")
    
    # Ensure FDR data is loaded
    await refresh_fdr_data()
    
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    
    elements = data["elements"]
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    position_id = POSITION_ID_MAP[position.value]
    
    ranked_players = []
    
    for player in elements:
        if player["element_type"] != position_id:
            continue
        
        total_minutes = player.get("minutes", 0)
        price = player["now_cost"] / 10
        ownership = float(player.get("selected_by_percent", 0) or 0)
        
        if total_minutes < min_minutes or price < min_price or price > max_price:
            continue
        
        # Filter by max ownership (for differentials)
        if ownership > max_ownership:
            continue
        
        upcoming = get_player_upcoming_fixtures(player["team"], fixtures, gw_start, gw_end, teams)
        
        stats = calculate_expected_points(
            player, position_id, current_gw, upcoming, teams, fixtures, events
        )
        
        team_data = teams.get(player["team"], {})
        
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
            "expected_minutes": stats["expected_minutes"],
            "minutes_reason": stats["minutes_reason"],
            "form": float(player.get("form", 0) or 0),
            "total_points": player.get("total_points", 0),
            "ownership": float(player.get("selected_by_percent", 0) or 0),
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
    
    squad = []
    for pick in picks:
        player = elements.get(pick["element"])
        if not player:
            continue
        
        position_id = player["element_type"]
        
        upcoming = get_player_upcoming_fixtures(player["team"], fixtures, next_gw, next_gw + 5, teams)
        stats = calculate_expected_points(player, position_id, current_gw, upcoming, teams, fixtures, events)
        
        single_fixtures = get_player_upcoming_fixtures(player["team"], fixtures, next_gw, next_gw, teams)
        single_stats = calculate_expected_points(player, position_id, current_gw, single_fixtures, teams, fixtures, events)
        
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


async def fetch_player_history(player_id: int) -> Dict:
    """Fetch player's GW-by-GW history."""
    try:
        client = await get_http_client()
        response = await client.get(f"{FPL_BASE_URL}/element-summary/{player_id}/")
        response.raise_for_status()
        return response.json()
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
    
    # OR progression for graph
    or_progression = [
        {"gw": gw["event"], "or": gw.get("overall_rank", 0), "points": gw.get("points", 0), "rank": gw.get("rank", 0)}
        for gw in gw_history
    ]
    
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
    
    # Best Differential - highest points from lowest-owned player at time of selection
    # We look at all GWs and find the player who scored most points relative to low ownership
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
                pid = pick.get("element")
                element = elements.get(pid, {})
                
                # Get ownership at GW start (approximation - use current minus GW increase)
                # Better: fetch the player's history to get ownership_gw
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
                        # Use value change as proxy for ownership 
                        # Better: FPL API doesn't give historical ownership, so use current as approximation
                        ownership = float(element.get("selected_by_percent", 0) or 0)
                        
                        # Differential threshold: below 10% ownership
                        if ownership < 10 and gw_pts > 0:
                            diff_score = gw_pts / max(ownership, 0.1)  # Points per ownership %
                            
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
    chip: Optional[str] = None
) -> float:
    """
    Calculate expected points for a squad in a single GW.
    Handles chip effects (BB = all 15 play, TC = captain x3).
    """
    # Get xPts for each player for this specific GW
    player_xpts = []
    
    for p in squad:
        player = elements.get(p["id"])
        if not player:
            # Player not found in elements, use data from squad
            player_xpts.append({
                "id": p["id"],
                "position_id": p.get("position_id", 3),
                "xpts": p.get("xpts", 0) / 5,  # Rough per-GW estimate
                "is_captain": p.get("is_captain", False),
            })
            continue
        
        position_id = player["element_type"]
        upcoming = get_player_upcoming_fixtures(player["team"], fixtures, gw, gw, teams_dict)
        
        if not upcoming:
            # Player has blank gameweek
            xpts = 0
        else:
            stats = calculate_expected_points(
                player, position_id, gw, upcoming, teams_dict, fixtures, events
            )
            # Single GW xPts (base90 * minutes_factor * fixture_multiplier)
            xpts = stats["xpts"] / max(len(upcoming), 1)  # Per fixture if DGW
        
        player_xpts.append({
            "id": p["id"],
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
    limit: int = 5
) -> List[Dict]:
    """Get top transfer candidates for each position."""
    squad_ids = {p["id"] for p in squad}
    team_counts = defaultdict(int)
    for p in squad:
        team_counts[p["team_id"]] += 1
    
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
                
                # Calculate out player's xPts
                out_upcoming = get_player_upcoming_fixtures(
                    out_player["team_id"], fixtures, current_gw, current_gw + horizon, teams_dict
                )
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
    
    def copy(self):
        new_path = TransferPath(self.squad, self.bank, self.ft)
        new_path.actions = self.actions.copy()
        new_path.total_xpts = self.total_xpts
        new_path.hits = self.hits
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
    max_hits: int = 1
) -> Dict:
    """
    Tree-based solver for optimal transfer path.
    
    Explores paths: roll FT, use FT for best transfer, take hit for additional transfer.
    Prunes paths that fall too far behind.
    """
    end_gw = min(start_gw + horizon, 39)
    gw_info = detect_dgw_bgw(fixtures, events, start_gw, end_gw)
    
    # Initialize root path
    root = TransferPath(squad, bank, free_transfers)
    
    # Calculate baseline xPts (no transfers)
    baseline_xpts = 0
    for gw in range(start_gw, end_gw):
        chip = None
        for chip_name, chip_gw in chip_gws.items():
            if chip_gw == gw:
                chip = chip_name
                break
        baseline_xpts += evaluate_squad_xpts(squad, gw, fixtures, teams_dict, elements, events, chip)
    
    root.total_xpts = baseline_xpts
    
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
        
        for path in paths:
            # Option 1: Roll FT (if beneficial or no good transfers)
            roll_path = path.copy()
            roll_path.roll_transfer(gw)
            gw_xpts = evaluate_squad_xpts(roll_path.squad, gw, fixtures, teams_dict, elements, events, chip_this_gw)
            roll_path.total_xpts = sum(
                evaluate_squad_xpts(roll_path.squad, g, fixtures, teams_dict, elements, events,
                                   chip_gws.get(g))
                for g in range(start_gw, end_gw)
            )
            new_paths.append(roll_path)
            
            # Get transfer candidates
            candidates = get_transfer_candidates(
                path.squad, all_players, teams_dict, fixtures, gw, end_gw - gw, path.bank, limit=3
            )
            
            if path.ft > 0 and candidates:
                # Option 2: Use FT for best transfer
                for i, transfer in enumerate(candidates[:3]):  # Top 3 transfers
                    ft_path = path.copy()
                    ft_path.use_ft()
                    ft_path.apply_transfer(transfer, gw, is_hit=False)
                    ft_path.total_xpts = sum(
                        evaluate_squad_xpts(ft_path.squad, g, fixtures, teams_dict, elements, events,
                                           chip_gws.get(g))
                        for g in range(start_gw, end_gw)
                    )
                    new_paths.append(ft_path)
                    
                    # Option 3: Take hit for second transfer (if allowed)
                    if max_hits > 0 and path.hits < max_hits and len(candidates) > 1:
                        for j, second_transfer in enumerate(candidates[i+1:i+3]):  # Next 2
                            if second_transfer["out"]["id"] == transfer["in"]["id"]:
                                continue
                            if second_transfer["in"]["id"] == transfer["out"]["id"]:
                                continue
                            
                            hit_path = ft_path.copy()
                            hit_path.apply_transfer(second_transfer, gw, is_hit=True)
                            hit_path.total_xpts = sum(
                                evaluate_squad_xpts(hit_path.squad, g, fixtures, teams_dict, elements, events,
                                                   chip_gws.get(g))
                                for g in range(start_gw, end_gw)
                            ) - 4  # -4 for hit
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
            
            upcoming = get_player_upcoming_fixtures(team_id, fixtures, gw, gw + 1, teams_dict)
            if upcoming:
                stats = calculate_expected_points(element, position_id, gw, upcoming, teams_dict, fixtures, events)
                gw_xpts = stats.get("xpts", 0) / max(len(upcoming), 1)  # Single GW xPts
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
        
        # Captain is highest xPts in starting XI
        captain = max(starting_xi, key=lambda p: p["xpts"]) if starting_xi else None
        vice_captain = sorted(starting_xi, key=lambda p: -p["xpts"])[1] if len(starting_xi) > 1 else None
        
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
):
    """
    Multi-gameweek transfer planner.
    
    Returns optimal transfer path over horizon GWs.
    User can specify which GWs to use chips (or leave for auto-suggestion).
    """
    await refresh_fdr_data()
    
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
    
    # Determine free transfers (1 base + rolled if applicable)
    # This is simplified - real logic would check previous GW transfers
    free_transfers = 1
    if entry_history.get("event_transfers", 0) == 0:
        free_transfers = min(free_transfers + 1, 5)
    
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
    )
    
    return {
        "manager_id": manager_id,
        "current_gw": current_gw,
        "planning_horizon": f"GW{next_gw}-GW{next_gw + horizon - 1}",
        "available_chips": {CHIP_DISPLAY.get(k, k): v for k, v in available_chips.items()},
        "chip_schedule": {CHIP_DISPLAY.get(k, k): v for k, v in chip_gws.items()},
        "chip_suggestions": chip_suggestions,
        "gw_info": gw_info,
        "bank": bank,
        "free_transfers": free_transfers,
        "ft_disclaimer": "FT count is estimated from recent transfer history. Verify before confirming transfers.",
        "plan": plan,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
