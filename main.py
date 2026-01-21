"""
FPL Analytics Backend - FastAPI
Comprehensive Fantasy Premier League analytics with heavy emphasis on expected minutes
"""

import asyncio
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
import math
from collections import defaultdict
from datetime import datetime
import os

app = FastAPI(title="FPL Analytics API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
MIN_MINUTES_DEFAULT = 400
POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_ID_MAP = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}

# Fixture difficulty weights (affects expected points)
FDR_MULTIPLIERS = {
    1: 1.25,  # Very easy fixture
    2: 1.15,  # Easy fixture
    3: 1.00,  # Medium fixture
    4: 0.85,  # Hard fixture
    5: 0.70,  # Very hard fixture
}

class Position(str, Enum):
    GKP = "GKP"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"

class Formation(str, Enum):
    F343 = "3-4-3"
    F352 = "3-5-2"
    F442 = "4-4-2"
    F433 = "4-3-3"
    F451 = "4-5-1"
    F532 = "5-3-2"
    F541 = "5-4-1"

class TeamOptimizeRequest(BaseModel):
    budget: float = 100.0
    formation: Formation = Formation.F343
    gw_start: int = 1
    gw_end: int = 8
    min_minutes: int = MIN_MINUTES_DEFAULT
    excluded_players: List[int] = []
    required_players: List[int] = []

# Cache for API data
class DataCache:
    def __init__(self):
        self.bootstrap_data: Optional[Dict] = None
        self.fixtures_data: Optional[List] = None
        self.element_summary_cache: Dict[int, Dict] = {}
        self.last_update: Optional[datetime] = None
        self.cache_duration = 300  # 5 minutes

    def is_stale(self) -> bool:
        if self.last_update is None:
            return True
        return (datetime.now() - self.last_update).seconds > self.cache_duration

cache = DataCache()

async def fetch_fpl_data():
    """Fetch and cache FPL bootstrap data"""
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
    """Fetch fixture data"""
    if cache.fixtures_data:
        return cache.fixtures_data

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{FPL_BASE_URL}/fixtures/")
            response.raise_for_status()
            cache.fixtures_data = response.json()
            return cache.fixtures_data
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"FPL API unavailable: {str(e)}")

async def fetch_player_history(player_id: int):
    """Fetch individual player history"""
    if player_id in cache.element_summary_cache:
        return cache.element_summary_cache[player_id]

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{FPL_BASE_URL}/element-summary/{player_id}/")
            response.raise_for_status()
            data = response.json()
            cache.element_summary_cache[player_id] = data
            return data
        except:
            return None

def get_current_gameweek(events: List[Dict]) -> int:
    """Get current gameweek from events"""
    for event in events:
        if event.get("is_current"):
            return event["id"]
    # If no current, find next
    for event in events:
        if event.get("is_next"):
            return event["id"]
    return 1

def calculate_minutes_factor(
    total_minutes: int, 
    starts: int, 
    current_gw: int,
    chance_of_playing: Optional[int],
    status: str
) -> tuple[float, str]:
    """
    Calculate minutes factor - distinguishes ROTATION from INJURY
    
    Key insight: Use minutes per START (not per GW) to identify rotation risk
    - Nailed starter: 80+ mins per start (plays full/most games when selected)
    - Rotation risk: Low mins per start OR player regularly subbed early
    
    Returns: (factor, reason)
    """
    if total_minutes < 200:
        return 0.1, "insufficient_minutes"
    
    if starts == 0:
        return 0.15, "no_starts"
    
    # Minutes per start reveals if player is nailed WHEN SELECTED
    mins_per_start = total_minutes / starts
    
    # Start ratio reveals if player is regularly selected
    available_gws = max(current_gw - 1, 1)
    start_ratio = starts / available_gws
    
    # Check current availability
    is_available = status == 'a' and (chance_of_playing is None or chance_of_playing >= 75)
    is_doubtful = status == 'd' or (chance_of_playing is not None and 25 <= chance_of_playing < 75)
    is_out = status in ('i', 'u', 's') or (chance_of_playing is not None and chance_of_playing < 25)
    
    # NAILED STARTER: High mins per start (80+) = plays full games when selected
    if mins_per_start >= 80:
        if start_ratio >= 0.70:
            # Truly nailed - starts most games, plays full 90 when starting
            if is_out:
                return 0.6, "nailed_currently_injured"
            elif is_doubtful:
                return 0.85, "nailed_doubtful"
            else:
                return 1.0, "nailed_starter"
        elif start_ratio >= 0.45:
            # Nailed when fit but has missed games (injury history this season)
            if is_out:
                return 0.5, "nailed_injury_prone_out"
            elif is_doubtful:
                return 0.8, "nailed_injury_prone_doubtful"
            else:
                return 0.95, "nailed_back_from_injury"
        else:
            # Missed many games - likely long-term injury or new signing
            if is_available:
                return 0.85, "nailed_limited_games"
            else:
                return 0.4, "nailed_long_term_out"
    
    # REGULAR STARTER but sometimes subbed: 65-80 mins per start
    elif mins_per_start >= 65:
        if start_ratio >= 0.65:
            if is_available:
                return 0.9, "regular_starter"
            else:
                return 0.55, "regular_currently_out"
        elif start_ratio >= 0.45:
            return 0.75, "rotation_risk_moderate"
        else:
            return 0.55, "rotation_risk_high"
    
    # ROTATION PLAYER: 50-65 mins per start (often subbed or rotated)
    elif mins_per_start >= 50:
        if start_ratio >= 0.60:
            return 0.65, "squad_rotation"
        else:
            return 0.45, "heavy_rotation"
    
    # BENCH PLAYER: Very low mins per start - mainly comes off bench
    else:
        return 0.25, "bench_player"

def calculate_expected_points(
    player: Dict,
    position: int,
    current_gw: int,
    upcoming_fixtures: List[Dict],
    teams_dict: Dict
) -> tuple[float, float, str]:
    """
    Calculate expected points with REFINED minutes analysis
    Distinguishes between rotation risk and injury absence
    
    Returns: (expected_pts, minutes_factor, minutes_reason)
    """
    # Get player stats
    total_minutes = player.get("minutes", 0)
    starts = player.get("starts", 0)
    chance_of_playing = player.get("chance_of_playing_next_round")
    status = player.get("status", "a")
    
    # Get base stats from FPL API
    xG = float(player.get("expected_goals", 0) or 0)
    xA = float(player.get("expected_assists", 0) or 0)
    xGI = float(player.get("expected_goal_involvements", 0) or 0)
    xGC = float(player.get("expected_goals_conceded", 0) or 0)
    
    form = float(player.get("form", 0) or 0)
    points_per_game = float(player.get("points_per_game", 0) or 0)
    
    # Calculate minutes factor using NEW algorithm
    minutes_factor, minutes_reason = calculate_minutes_factor(
        total_minutes, starts, current_gw, chance_of_playing, status
    )
    
    games_played = max(starts, 1)
    
    # Base expected points per 90
    if position == 1:  # GKP
        base_xpts = (
            2 +  # Appearance
            (max(0, 1 - xGC / max(games_played, 1)) * 4) +  # Clean sheet potential
            0.5  # Save points estimate
        )
    elif position == 2:  # DEF
        base_xpts = (
            2 +  # Appearance
            (max(0, 1 - xGC / max(games_played, 1)) * 4) +  # Clean sheet potential
            xG * 6 / max(games_played, 1) +  # Goals
            xA * 3 / max(games_played, 1)   # Assists
        )
    elif position == 3:  # MID
        base_xpts = (
            2 +  # Appearance
            xG * 5 / max(games_played, 1) +  # Goals
            xA * 3 / max(games_played, 1) +  # Assists
            (max(0, 1 - xGC / max(games_played, 1)) * 1)  # Clean sheet (1 point)
        )
    else:  # FWD
        base_xpts = (
            2 +  # Appearance
            xG * 4 / max(games_played, 1) +  # Goals
            xA * 3 / max(games_played, 1)   # Assists
        )
    
    # Blend with actual performance
    if points_per_game > 0:
        base_xpts = (base_xpts * 0.4 + points_per_game * 0.6)
    
    # Apply fixture difficulty
    if upcoming_fixtures:
        fixture_multiplier = 0
        for fix in upcoming_fixtures[:8]:  # Max 8 GWs
            fdr = fix.get("difficulty", 3)
            fixture_multiplier += FDR_MULTIPLIERS.get(fdr, 1.0)
        fixture_multiplier /= len(upcoming_fixtures[:8])
    else:
        fixture_multiplier = 1.0
    
    # Apply minutes factor
    expected_pts = base_xpts * minutes_factor * fixture_multiplier
    
    # Bonus for truly nailed starters
    if minutes_reason == "nailed_starter":
        expected_pts *= 1.1
    
    return round(expected_pts, 2), minutes_factor, minutes_reason

def get_player_upcoming_fixtures(
    player_team: int,
    fixtures: List[Dict],
    current_gw: int,
    gw_end: int,
    teams_dict: Dict
) -> List[Dict]:
    """Get upcoming fixtures for a player's team"""
    upcoming = []
    for fix in fixtures:
        gw = fix.get("event")
        if gw and current_gw <= gw <= gw_end:
            if fix["team_h"] == player_team:
                opponent_id = fix["team_a"]
                is_home = True
                difficulty = fix.get("team_h_difficulty", 3)
            elif fix["team_a"] == player_team:
                opponent_id = fix["team_h"]
                is_home = False
                difficulty = fix.get("team_a_difficulty", 3)
            else:
                continue
            
            opponent_name = teams_dict.get(opponent_id, {}).get("short_name", "???")
            upcoming.append({
                "gameweek": gw,
                "opponent": opponent_name,
                "is_home": is_home,
                "difficulty": difficulty
            })
    
    return sorted(upcoming, key=lambda x: x["gameweek"])

@app.get("/api")
async def api_root():
    """Simple API heartbeat"""
    return {"message": "FPL Analytics API", "docs": "/docs"}

@app.get("/")
async def root():
    """Serve the frontend (index.html) from the same directory as this file."""
    base_dir = os.path.dirname(__file__)
    index_path = os.path.join(base_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)

    # Fallback if you kept the patched filename in the repo
    patched_path = os.path.join(base_dir, "index_patched.html")
    if os.path.exists(patched_path):
        return FileResponse(patched_path)

    # Frontend not bundled; keep a helpful API response
    return {"message": "FPL Analytics API", "docs": "/docs"}

@app.get("/api/bootstrap")
async def get_bootstrap():
    """Get raw bootstrap data"""
    return await fetch_fpl_data()

@app.get("/api/rankings/{position}")
async def get_rankings(
    position: Position,
    gw_start: int = Query(1, ge=1, le=38),
    gw_end: int = Query(8, ge=1, le=38),
    min_minutes: int = Query(MIN_MINUTES_DEFAULT, ge=0),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Get player rankings by position with expected points calculation
    HEAVILY weights expected minutes - bench players rank low
    """
    if gw_end < gw_start:
        raise HTTPException(status_code=400, detail="gw_end must be >= gw_start")
    if gw_end - gw_start > 8:
        raise HTTPException(status_code=400, detail="Maximum 8 gameweek range")
    
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    
    elements = data["elements"]
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    
    # Adjust gw_start to be at least current gameweek
    effective_gw_start = max(gw_start, current_gw)
    effective_gw_end = min(gw_end, 38)
    
    position_id = POSITION_ID_MAP[position.value]
    
    ranked_players = []
    
    for player in elements:
        if player["element_type"] != position_id:
            continue
        
        total_minutes = player.get("minutes", 0)
        if total_minutes < min_minutes:
            continue
        
        starts = player.get("starts", 0)
        mins_per_gw = total_minutes / max(current_gw - 1, 1)
        
        # Get upcoming fixtures
        upcoming = get_player_upcoming_fixtures(
            player["team"],
            fixtures,
            effective_gw_start,
            effective_gw_end,
            teams
        )
        
        # Calculate expected points with NEW algorithm
        xpts, minutes_factor, minutes_reason = calculate_expected_points(
            player,
            position_id,
            current_gw,
            upcoming,
            teams
        )
        
        team_data = teams.get(player["team"], {})
        
        ranked_players.append({
            "id": player["id"],
            "name": player["web_name"],
            "full_name": f"{player['first_name']} {player['second_name']}",
            "team": team_data.get("short_name", "???"),
            "team_id": player["team"],
            "position": position.value,
            "price": player["now_cost"] / 10,
            "xpts": xpts,
            "xpts_per_gw": round(xpts, 2),
            "form": float(player.get("form", 0) or 0),
            "total_points": player.get("total_points", 0),
            "points_per_game": float(player.get("points_per_game", 0) or 0),
            "mins_per_gw": round(mins_per_gw, 1),
            "total_minutes": total_minutes,
            "starts": starts,
            "mins_per_start": round(total_minutes / starts, 1) if starts > 0 else 0,
            "ownership": float(player.get("selected_by_percent", 0) or 0),
            "xG": float(player.get("expected_goals", 0) or 0),
            "xA": float(player.get("expected_assists", 0) or 0),
            "xGI": float(player.get("expected_goal_involvements", 0) or 0),
            "fixtures": upcoming[:8],
            "news": player.get("news", ""),
            "status": player.get("status", "a"),
            "chance_of_playing": player.get("chance_of_playing_next_round"),
            "minutes_factor": minutes_factor,
            "minutes_reason": minutes_reason
        })
    
    # Sort by expected points (descending)
    ranked_players.sort(key=lambda x: x["xpts"], reverse=True)
    
    return {
        "position": position.value,
        "gw_range": f"GW{effective_gw_start}-GW{effective_gw_end}",
        "current_gw": current_gw,
        "total_players": len(ranked_players),
        "min_minutes_filter": min_minutes,
        "players": ranked_players[:limit]
    }

@app.get("/api/fixture-ratings")
async def get_fixture_ratings(
    horizon: int = Query(8, ge=1, le=15)
):
    """Get fixture difficulty ratings for all teams"""
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    
    team_ratings = []
    
    for team_id, team in teams.items():
        upcoming = []
        total_difficulty = 0
        fixture_count = 0
        
        for fix in fixtures:
            gw = fix.get("event")
            if gw and current_gw <= gw < current_gw + horizon:
                if fix["team_h"] == team_id:
                    opponent_id = fix["team_a"]
                    is_home = True
                    difficulty = fix.get("team_h_difficulty", 3)
                elif fix["team_a"] == team_id:
                    opponent_id = fix["team_h"]
                    is_home = False
                    difficulty = fix.get("team_a_difficulty", 3)
                else:
                    continue
                
                opponent = teams.get(opponent_id, {})
                upcoming.append({
                    "gameweek": gw,
                    "opponent": opponent.get("short_name", "???"),
                    "opponent_full": opponent.get("name", "Unknown"),
                    "is_home": is_home,
                    "difficulty": difficulty
                })
                total_difficulty += difficulty
                fixture_count += 1
        
        upcoming.sort(key=lambda x: x["gameweek"])
        avg_difficulty = total_difficulty / fixture_count if fixture_count > 0 else 3
        
        # Determine rating
        if avg_difficulty <= 2.2:
            rating = "EASY"
        elif avg_difficulty <= 2.8:
            rating = "MEDIUM"
        else:
            rating = "HARD"
        
        team_ratings.append({
            "team_id": team_id,
            "team_name": team["name"],
            "team_short": team["short_name"],
            "avg_difficulty": round(avg_difficulty, 2),
            "rating": rating,
            "fixtures": upcoming,
            "easy_fixtures": len([f for f in upcoming if f["difficulty"] <= 2]),
            "hard_fixtures": len([f for f in upcoming if f["difficulty"] >= 4])
        })
    
    # Sort by average difficulty (easier first)
    team_ratings.sort(key=lambda x: x["avg_difficulty"])
    
    return {
        "horizon": horizon,
        "current_gw": current_gw,
        "teams": team_ratings
    }

@app.get("/api/differentials")
async def get_differentials(
    max_ownership: float = Query(10.0, ge=0, le=100),
    min_form: float = Query(4.0, ge=0),
    min_minutes: int = Query(MIN_MINUTES_DEFAULT, ge=0),
    limit: int = Query(30, ge=1, le=100)
):
    """Find differential players - low ownership + high form/xPts"""
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
        
        if ownership > max_ownership:
            continue
        if form < min_form:
            continue
        if total_minutes < min_minutes:
            continue
        
        mins_per_gw = total_minutes / max(current_gw - 1, 1)
        
        # Skip players with low minutes
        if mins_per_gw < 60:
            continue
        
        position_id = player["element_type"]
        games = max(1, player.get("starts", 1))
        
        # Get upcoming fixtures
        upcoming = get_player_upcoming_fixtures(
            player["team"],
            fixtures,
            current_gw,
            current_gw + 5,
            teams
        )
        
        xpts, minutes_factor, minutes_reason = calculate_expected_points(
            player, position_id, current_gw, upcoming, teams
        )
        
        team_data = teams.get(player["team"], {})
        
        # Calculate differential score (form + xpts weighted, ownership inverse)
        diff_score = (form * 0.4 + xpts * 0.6) * (1 + (max_ownership - ownership) / max_ownership)
        
        differentials.append({
            "id": player["id"],
            "name": player["web_name"],
            "full_name": f"{player['first_name']} {player['second_name']}",
            "team": team_data.get("short_name", "???"),
            "position": POSITION_MAP.get(position_id, "???"),
            "price": player["now_cost"] / 10,
            "ownership": ownership,
            "form": form,
            "xpts": xpts,
            "total_points": player.get("total_points", 0),
            "mins_per_gw": round(mins_per_gw, 1),
            "diff_score": round(diff_score, 2),
            "fixtures": upcoming[:5],
            "xG": float(player.get("expected_goals", 0) or 0),
            "xA": float(player.get("expected_assists", 0) or 0)
        })
    
    # Sort by differential score
    differentials.sort(key=lambda x: x["diff_score"], reverse=True)
    
    return {
        "max_ownership": max_ownership,
        "min_form": min_form,
        "current_gw": current_gw,
        "total_found": len(differentials),
        "players": differentials[:limit]
    }

@app.get("/api/price-changes")
async def get_price_changes(
    limit: int = Query(40, ge=1, le=100)
):
    """Predict price changes based on transfer activity"""
    data = await fetch_fpl_data()
    
    elements = data["elements"]
    teams = {t["id"]: t for t in data["teams"]}
    
    rising = []
    falling = []
    
    for player in elements:
        transfers_in = player.get("transfers_in_event", 0)
        transfers_out = player.get("transfers_out_event", 0)
        net_transfers = transfers_in - transfers_out
        
        # Skip players with minimal activity
        if abs(net_transfers) < 1000:
            continue
        
        team_data = teams.get(player["team"], {})
        position_id = player["element_type"]
        
        player_data = {
            "id": player["id"],
            "name": player["web_name"],
            "team": team_data.get("short_name", "???"),
            "position": POSITION_MAP.get(position_id, "???"),
            "price": player["now_cost"] / 10,
            "transfers_in": transfers_in,
            "transfers_out": transfers_out,
            "net_transfers": net_transfers,
            "ownership": float(player.get("selected_by_percent", 0) or 0),
            "form": float(player.get("form", 0) or 0),
            "news": player.get("news", "")
        }
        
        # Calculate price change likelihood
        # This is simplified - real prediction would need target thresholds
        ownership = float(player.get("selected_by_percent", 0) or 0)
        
        if net_transfers > 0:
            # Rising - threshold depends on ownership
            threshold = 50000 + (ownership * 5000)
            likelihood = min(100, (net_transfers / threshold) * 100)
            player_data["likelihood"] = round(likelihood, 1)
            player_data["direction"] = "UP"
            rising.append(player_data)
        else:
            # Falling
            threshold = 30000 + (ownership * 3000)
            likelihood = min(100, (abs(net_transfers) / threshold) * 100)
            player_data["likelihood"] = round(likelihood, 1)
            player_data["direction"] = "DOWN"
            falling.append(player_data)
    
    # Sort by likelihood
    rising.sort(key=lambda x: x["likelihood"], reverse=True)
    falling.sort(key=lambda x: x["likelihood"], reverse=True)
    
    return {
        "rising": rising[:limit // 2],
        "falling": falling[:limit // 2]
    }

@app.post("/api/optimize-team")
async def optimize_team(request: TeamOptimizeRequest):
    """
    Optimize team selection using linear programming
    Maximize expected points within budget and formation constraints
    """
    try:
        from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PuLP not installed. Run: pip install pulp"
        )
    
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    
    elements = data["elements"]
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    
    # Formation requirements
    formation_map = {
        "3-4-3": {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 3},
        "3-5-2": {"GKP": 1, "DEF": 3, "MID": 5, "FWD": 2},
        "4-4-2": {"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2},
        "4-3-3": {"GKP": 1, "DEF": 4, "MID": 3, "FWD": 3},
        "4-5-1": {"GKP": 1, "DEF": 4, "MID": 5, "FWD": 1},
        "5-3-2": {"GKP": 1, "DEF": 5, "MID": 3, "FWD": 2},
        "5-4-1": {"GKP": 1, "DEF": 5, "MID": 4, "FWD": 1},
    }
    
    formation_req = formation_map.get(request.formation.value, formation_map["3-4-3"])
    
    # Filter and prepare players
    valid_players = []
    
    for player in elements:
        if player["id"] in request.excluded_players:
            continue
        
        total_minutes = player.get("minutes", 0)
        if total_minutes < request.min_minutes:
            continue
        
        # Skip injured/unavailable
        chance = player.get("chance_of_playing_next_round")
        if chance is not None and chance < 50:
            continue
        
        mins_per_gw = total_minutes / max(current_gw - 1, 1)
        position_id = player["element_type"]
        
        # Get fixtures
        upcoming = get_player_upcoming_fixtures(
            player["team"],
            fixtures,
            request.gw_start,
            request.gw_end,
            teams
        )
        
        xpts, minutes_factor, minutes_reason = calculate_expected_points(
            player, position_id, current_gw, upcoming, teams
        )
        
        # Skip very low xpts players
        if xpts < 1.5:
            continue
        
        valid_players.append({
            "id": player["id"],
            "name": player["web_name"],
            "team": teams.get(player["team"], {}).get("short_name", "???"),
            "team_id": player["team"],
            "position": POSITION_MAP.get(position_id, "???"),
            "position_id": position_id,
            "price": player["now_cost"] / 10,
            "xpts": xpts,
            "form": float(player.get("form", 0) or 0),
            "ownership": float(player.get("selected_by_percent", 0) or 0),
            "mins_per_gw": round(mins_per_gw, 1),
            "minutes_reason": minutes_reason
        })
    
    # Create optimization problem
    prob = LpProblem("FPL_Team_Optimizer", LpMaximize)
    
    # Decision variables
    player_vars = {p["id"]: LpVariable(f"player_{p['id']}", cat="Binary") for p in valid_players}
    
    # Objective: Maximize expected points
    prob += lpSum(player_vars[p["id"]] * p["xpts"] for p in valid_players)
    
    # Budget constraint
    prob += lpSum(player_vars[p["id"]] * p["price"] for p in valid_players) <= request.budget
    
    # Position constraints
    for pos, count in formation_req.items():
        pos_id = POSITION_ID_MAP[pos]
        prob += lpSum(
            player_vars[p["id"]] for p in valid_players if p["position_id"] == pos_id
        ) == count
    
    # Max 3 players per team
    team_ids = set(p["team_id"] for p in valid_players)
    for team_id in team_ids:
        prob += lpSum(
            player_vars[p["id"]] for p in valid_players if p["team_id"] == team_id
        ) <= 3
    
    # Required players
    for req_id in request.required_players:
        if req_id in player_vars:
            prob += player_vars[req_id] == 1
    
    # Solve
    prob.solve()
    
    if LpStatus[prob.status] != "Optimal":
        raise HTTPException(
            status_code=400,
            detail=f"Could not find optimal solution. Status: {LpStatus[prob.status]}"
        )
    
    # Extract selected players
    selected = []
    total_xpts = 0
    total_cost = 0
    
    for p in valid_players:
        if player_vars[p["id"]].varValue == 1:
            selected.append(p)
            total_xpts += p["xpts"]
            total_cost += p["price"]
    
    # Sort by position then xpts
    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    selected.sort(key=lambda x: (pos_order.get(x["position"], 4), -x["xpts"]))
    
    return {
        "formation": request.formation.value,
        "budget_used": round(total_cost, 1),
        "budget_remaining": round(request.budget - total_cost, 1),
        "total_xpts": round(total_xpts, 2),
        "gw_range": f"GW{request.gw_start}-GW{request.gw_end}",
        "players": selected,
        "by_position": {
            "GKP": [p for p in selected if p["position"] == "GKP"],
            "DEF": [p for p in selected if p["position"] == "DEF"],
            "MID": [p for p in selected if p["position"] == "MID"],
            "FWD": [p for p in selected if p["position"] == "FWD"]
        }
    }

@app.get("/api/transfer-recommendations")
async def get_transfer_recommendations(
    current_players: str = Query("", description="Comma-separated player IDs"),
    budget: float = Query(0.5, ge=0),
    gw_start: int = Query(1, ge=1, le=38),
    gw_end: int = Query(8, ge=1, le=38)
):
    """Get transfer recommendations based on fixtures and form"""
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    
    elements = data["elements"]
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    
    # Parse current players
    current_ids = []
    if current_players:
        try:
            current_ids = [int(x.strip()) for x in current_players.split(",") if x.strip()]
        except:
            pass
    
    # Get current player data
    current_player_data = {}
    for player in elements:
        if player["id"] in current_ids:
            total_minutes = player.get("minutes", 0)
            mins_per_gw = total_minutes / max(current_gw - 1, 1)
            position_id = player["element_type"]
            
            upcoming = get_player_upcoming_fixtures(
                player["team"], fixtures, gw_start, gw_end, teams
            )
            
            xpts, _, _ = calculate_expected_points(
                player, position_id, current_gw, upcoming, teams
            )
            
            current_player_data[player["id"]] = {
                "id": player["id"],
                "name": player["web_name"],
                "team": teams.get(player["team"], {}).get("short_name", "???"),
                "position": POSITION_MAP.get(position_id, "???"),
                "price": player["now_cost"] / 10,
                "xpts": xpts,
                "form": float(player.get("form", 0) or 0)
            }
    
    # Find potential transfers
    recommendations = []
    
    for player in elements:
        if player["id"] in current_ids:
            continue
        
        total_minutes = player.get("minutes", 0)
        if total_minutes < MIN_MINUTES_DEFAULT:
            continue
        
        starts = player.get("starts", 0)
        if starts == 0:
            continue
        
        # Use mins per start to filter rotation players
        mins_per_start = total_minutes / starts
        if mins_per_start < 70:  # Filter out rotation/sub players
            continue
        
        position_id = player["element_type"]
        
        upcoming = get_player_upcoming_fixtures(
            player["team"], fixtures, gw_start, gw_end, teams
        )
        
        xpts, _, minutes_reason = calculate_expected_points(
            player, position_id, current_gw, upcoming, teams
        )
        
        price = player["now_cost"] / 10
        
        # Find who they could replace
        for curr_id, curr_data in current_player_data.items():
            if curr_data["position"] == POSITION_MAP.get(position_id):
                price_diff = price - curr_data["price"]
                if price_diff <= budget:
                    xpts_gain = xpts - curr_data["xpts"]
                    if xpts_gain > 0.5:  # Meaningful improvement
                        recommendations.append({
                            "in": {
                                "id": player["id"],
                                "name": player["web_name"],
                                "team": teams.get(player["team"], {}).get("short_name", "???"),
                                "position": POSITION_MAP.get(position_id, "???"),
                                "price": price,
                                "xpts": round(xpts, 2),
                                "form": float(player.get("form", 0) or 0),
                                "fixtures": upcoming[:5]
                            },
                            "out": curr_data,
                            "xpts_gain": round(xpts_gain, 2),
                            "cost": round(price_diff, 1)
                        })
    
    # Sort by xpts gain
    recommendations.sort(key=lambda x: x["xpts_gain"], reverse=True)
    
    return {
        "budget_available": budget,
        "recommendations": recommendations[:20]
    }

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
