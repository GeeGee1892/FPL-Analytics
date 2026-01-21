"""
FPL Assistant Backend - FastAPI v3.1
Enhanced with:
- Composite FDR model (1-10 scale) using Understat xG data
- Position-specific FDR: attack_fdr for FWD/MID, defence_fdr for DEF/GKP
- DEFCON per 90 for DEF/MID
- Saves per 90 for GKP
- Expected Minutes with override capability
"""

import asyncio
import re
import json
import sqlite3
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
import math
from collections import defaultdict
from datetime import datetime
from scipy import stats
import os
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="FPL Assistant API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# ============ CACHE ============

class DataCache:
    def __init__(self):
        self.bootstrap_data: Optional[Dict] = None
        self.fixtures_data: Optional[List] = None
        self.element_summary_cache: Dict[int, Dict] = {}
        self.last_update: Optional[datetime] = None
        self.fixtures_last_update: Optional[datetime] = None
        self.cache_duration = 300
        self.minutes_overrides: Dict[int, float] = {}
        # FDR cache
        self.fdr_data: Dict[int, Dict] = {}  # team_id -> {attack_fdr, defence_fdr, form_xg, form_xga, ...}
        self.fdr_last_update: Optional[datetime] = None

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
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{FPL_BASE_URL}/bootstrap-static/")
            response.raise_for_status()
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
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{FPL_BASE_URL}/fixtures/")
            response.raise_for_status()
            cache.fixtures_data = response.json()
            cache.fixtures_last_update = datetime.now()
            return cache.fixtures_data
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"FPL API unavailable: {str(e)}")


async def fetch_manager_team(manager_id: int, gameweek: int):
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
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
        
        async with httpx.AsyncClient(timeout=30.0) as client:
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
    
    async def fetch_league_matches(self, season: str = "2024") -> List[dict]:
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

async def refresh_fdr_data():
    """Fetch Understat data and compute FDR scores for all teams."""
    if not cache.fdr_is_stale() and cache.fdr_data:
        return cache.fdr_data
    
    logger.info("Refreshing FDR data from Understat...")
    
    # Fetch all matches
    matches = await understat_scraper.fetch_league_matches("2024")
    
    if not matches:
        logger.warning("No Understat matches fetched, using FPL API fallback")
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
            xga_percentile = stats.percentileofscore(all_form_xga, data['form_xga'])
            attack_fdr = max(1, min(10, int((100 - xga_percentile) / 10) + 1))
            
            # Defence FDR: How hard to defend against this team (based on their xG)
            # High xG = dangerous attack = HIGH defence_fdr
            xg_percentile = stats.percentileofscore(all_form_xg, data['form_xg'])
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
    total_minutes = int(player.get("minutes", 0) or 0)
    dc_points = int(player.get("defensive_contributions", 0) or 0)
    cbi = int(player.get("clearances_blocks_interceptions", 0) or 0)
    
    if total_minutes < 90:
        return 0.0, 0.0, 0
    
    mins90 = total_minutes / 90.0
    defcon_per_90 = cbi / mins90
    
    if position_id == 2:
        threshold = DEFCON_THRESHOLD_DEF
    elif position_id == 3:
        threshold = DEFCON_THRESHOLD_MID
    else:
        return 0.0, 0.0, 0
    
    if defcon_per_90 <= 0:
        prob = 0.0
    else:
        scale = 2.0
        prob = 1.0 / (1.0 + math.exp(-(defcon_per_90 - threshold) / scale))
    
    return round(defcon_per_90, 2), round(prob, 3), dc_points


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
    override_minutes: Optional[float] = None
) -> tuple[float, str]:
    player_id = player.get("id")
    if override_minutes is not None:
        return override_minutes, "user_override"
    
    if player_id in cache.minutes_overrides:
        return cache.minutes_overrides[player_id], "user_override"
    
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
async def set_minutes_override(override: MinutesOverride):
    cache.minutes_overrides[override.player_id] = override.expected_minutes
    return {"status": "ok", "player_id": override.player_id, "expected_minutes": override.expected_minutes}


@app.delete("/api/minutes-override/{player_id}")
async def remove_minutes_override(player_id: int):
    if player_id in cache.minutes_overrides:
        del cache.minutes_overrides[player_id]
    return {"status": "ok", "player_id": player_id}


@app.get("/api/minutes-overrides")
async def get_minutes_overrides():
    return {"overrides": cache.minutes_overrides}


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
    fdr_data = await refresh_fdr_data()
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
        
        if total_minutes < min_minutes or price < min_price or price > max_price:
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


@app.on_event("startup")
async def startup_fdr_refresh():
    """Initialize FDR data on startup."""
    try:
        await refresh_fdr_data()
        logger.info("FDR data initialized on startup")
    except Exception as e:
        logger.error(f"FDR startup refresh failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
