"""
FPL Assistant - Endpoints Module

FastAPI app initialization, CORS middleware, lifespan handler,
and ALL API endpoint handlers.
"""

import asyncio
import json
import math
import os
import logging
import statistics
import unicodedata
from pathlib import Path as PathlibPath
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fpl_assistant.config import MODEL_CONFIG
from fpl_assistant.constants import (
    FPL_BASE_URL, POSITION_MAP, POSITION_ID_MAP,
    ANALYTIC_FPL_TO_ID, ADMIN_MANAGER_ID,
    FPL_ID_TO_UNDERSTAT, MIN_MINUTES_DEFAULT,
    FIXTURE_WEIGHTS,
)
from fpl_assistant.models import (
    Position, MinutesOverride,
    TransferPlannerRequest,
)
from fpl_assistant.cache import cache
from fpl_assistant.calculators import (
    home_away_calculator, variance_model,
    get_ownership_tier, calculate_defcon_per_90, calculate_saves_per_90,
)
from fpl_assistant.services import (
    http_client, get_http_client,
    fetch_fpl_data, fetch_fixtures, fetch_manager_team, fetch_player_history,
    get_current_gameweek, get_next_gameweek,
    refresh_fdr_data, get_fixture_fdr,
    calculate_expected_points, get_player_upcoming_fixtures,
)
from fpl_assistant.planner import (
    ALL_CHIPS, CHIP_DISPLAY,
    get_available_chips, analyze_squad_health, calculate_fixture_swings,
    get_chip_recommendations, build_strategy_plan,
)
import fpl_assistant.services as services_module


logger = logging.getLogger("fpl_assistant")


# ============ PYDANTIC MODELS (endpoint-specific) ============

class BulkMinutesImport(BaseModel):
    """Bulk import predicted minutes from external sources."""
    predictions: Dict[int, float]  # player_id -> predicted_minutes
    source: Optional[str] = "manual"


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


# ============ HELPER FUNCTIONS (endpoint-specific) ============

async def fetch_manager_history(manager_id: int) -> Dict:
    """Fetch manager history including chips used."""
    try:
        client = await get_http_client()
        response = await client.get(f"{FPL_BASE_URL}/entry/{manager_id}/history/")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch manager history: {str(e)}")


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


# ============ STARTUP DATA LOADER ============

async def load_import_data_from_file():
    """
    Auto-load team strength and fixture xG data from data/import_data.json on startup.

    Looks for the file in:
    1. ./data/import_data.json (relative to working dir)
    2. /app/data/import_data.json (Docker/production)
    3. ../data/import_data.json (development)
    """
    possible_paths = [
        PathlibPath("data/import_data.json"),
        PathlibPath("/app/data/import_data.json"),
        PathlibPath(__file__).parent / "data" / "import_data.json",
        PathlibPath(__file__).parent.parent / "data" / "import_data.json",
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
                for name, tid in ANALYTIC_FPL_TO_ID.items():
                    teams_by_name[name.lower()] = tid

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


# ============ LIFESPAN ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup - create shared HTTP client
    services_module.http_client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        headers={"User-Agent": "FPL-Assistant/4.2"}
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
    if services_module.http_client:
        await services_module.http_client.aclose()
        services_module.http_client = None


# ============ APP INITIALIZATION ============

app = FastAPI(title="FPL Assistant API", version="4.2.0", lifespan=lifespan)

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

# Static files (CSS, JS)
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


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
    team: Optional[str] = Query(None, description="Filter by team short name (e.g., ARS, LIV, MCI)"),
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

    # Build team filter lookup if specified
    team_filter_ids = set()
    if team:
        team_lower = team.lower().strip()
        for t in teams.values():
            team_name = t.get("name", "").lower()
            team_short = t.get("short_name", "").lower()
            if team_lower in team_name or team_lower == team_short:
                team_filter_ids.add(t["id"])

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

        # Apply team filter if specified
        if team_filter_ids and player["team"] not in team_filter_ids:
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
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning(f"Failed to fetch history for player {pid}: {e}")
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

        # Calculate avg_fdr and fixture_ticker from upcoming fixtures
        avg_fdr = 5.0  # Default neutral
        fixture_ticker = ""
        if upcoming:
            fdrs = [f.get("difficulty", 5) for f in upcoming[:5]]
            avg_fdr = round(sum(fdrs) / len(fdrs), 1) if fdrs else 5.0

            # Build fixture ticker: "WOL(H) TOT(A) LIV(A)"
            ticker_parts = []
            for fix in upcoming[:5]:
                opp = fix.get("opponent_short", fix.get("opponent", "???")[:3])
                venue = "H" if fix.get("is_home", True) else "A"
                ticker_parts.append(f"{opp}({venue})")
            fixture_ticker = " ".join(ticker_parts)

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
            "exp_mins": stats["expected_minutes"],  # Alias for schema compatibility
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
            # v5.0: Form adjustment data
            "form_data": stats.get("form_data", {}),
            "xG_per_90_effective": stats.get("xG_per_90_effective", stats["xG_per_90"]),
            "xA_per_90_effective": stats.get("xA_per_90_effective", stats["xA_per_90"]),
            "avg_fdr": avg_fdr,
            "fixture_ticker": fixture_ticker,
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
                    fpl_diff = fix.get("team_h_difficulty", 3)
                elif fix["team_a"] == team_id:
                    opponent_id = fix["team_h"]
                    is_home = False
                    fpl_diff = fix.get("team_a_difficulty", 3)
                else:
                    continue

                # Pass FPL difficulty as fallback when Understat data unavailable
                attack_fdr = get_fixture_fdr(opponent_id, is_home, 4, fpl_difficulty=fpl_diff)  # FWD perspective
                defence_fdr = get_fixture_fdr(opponent_id, is_home, 2, fpl_difficulty=fpl_diff)  # DEF perspective

                upcoming.append({
                    "gameweek": gw,
                    "opponent": teams.get(opponent_id, {}).get("short_name", "???"),
                    "is_home": is_home,
                    "attack_fdr": attack_fdr,
                    "defence_fdr": defence_fdr,
                    # Use composite of our computed FDR (1-10 scale) for consistent color coding
                    "difficulty": round((attack_fdr + defence_fdr) / 2),
                    "fdr": round((attack_fdr + defence_fdr) / 2),  # Alias for frontend compatibility
                    "fpl_difficulty": fpl_diff  # Original FPL 1-5 for reference
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
    next_gw = get_next_gameweek(events)

    # v4.3.8: Determine the first GW to show
    # If current GW is finished (data_checked), start from next GW
    current_event = next((e for e in events if e["id"] == current_gw), None)
    if current_event and current_event.get("finished") and current_event.get("data_checked"):
        start_gw = next_gw
    else:
        start_gw = current_gw

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

        # v4.3.8: Skip finished fixtures - only show from start_gw onwards
        if gw < start_gw:
            continue

        # Also skip individual finished fixtures (handles mid-GW scenarios)
        if fix.get("finished"):
            continue

        home_id = fix["team_h"]
        away_id = fix["team_a"]
        home_fpl_diff = fix.get("team_h_difficulty", 3)
        away_fpl_diff = fix.get("team_a_difficulty", 3)

        # Home team fixture
        # v4.3.3c: Use matchup-based FDR by passing team_id
        if home_id in grid:
            attack_fdr = get_fixture_fdr(away_id, True, 4, fpl_difficulty=home_fpl_diff, team_id=home_id)  # FWD perspective
            defence_fdr = get_fixture_fdr(away_id, True, 2, fpl_difficulty=home_fpl_diff, team_id=home_id)  # DEF perspective
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
            attack_fdr = get_fixture_fdr(home_id, False, 4, fpl_difficulty=away_fpl_diff, team_id=away_id)
            defence_fdr = get_fixture_fdr(home_id, False, 2, fpl_difficulty=away_fpl_diff, team_id=away_id)
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
        "next_gw": next_gw,
        "start_gw": start_gw,
        "teams": team_list
    }


# ============ MANAGER STATS ============

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
                            "extra_pts": captain_pts,  # TC gives 3x, captain gives 2x, so extra = 1x base pts
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

    # Captain Performance - track every captain pick and their points
    captain_data = []
    total_captain_pts = 0
    captain_hits = 0  # GWs where captain scored 6+
    best_captain = None
    worst_captain = None

    try:
        for gw_data in gw_history:
            gw_num = gw_data.get("event", 0)
            if gw_num < 1:
                continue

            gw_picks = await fetch_gw_picks(manager_id, gw_num)
            if not gw_picks:
                continue

            picks = gw_picks.get("picks", [])
            for pick in picks:
                if pick.get("is_captain"):
                    pid = pick.get("element")
                    element = elements.get(pid, {})
                    multiplier = pick.get("multiplier", 2)

                    # Get captain base points
                    player_hist = player_histories.get(pid)
                    if not player_hist and pid:
                        try:
                            player_hist = await fetch_player_history(pid)
                            player_histories[pid] = player_hist
                        except:
                            player_hist = {"history": []}

                    base_pts = get_player_gw_points(player_hist, gw_num)
                    captain_pts_earned = base_pts * multiplier  # 2x or 3x if TC

                    entry = {
                        "gw": gw_num,
                        "player": element.get("web_name", "Unknown"),
                        "player_id": pid,
                        "base_pts": base_pts,
                        "multiplier": multiplier,
                        "total_pts": captain_pts_earned,
                    }
                    captain_data.append(entry)
                    total_captain_pts += captain_pts_earned

                    if base_pts >= 6:
                        captain_hits += 1

                    if best_captain is None or base_pts > best_captain["base_pts"]:
                        best_captain = entry
                    if worst_captain is None or base_pts < worst_captain["base_pts"]:
                        worst_captain = entry
                    break
    except Exception as e:
        logger.error(f"Failed to calculate captain performance: {e}")

    captain_performance = {
        "total_pts": total_captain_pts,
        "hit_rate": round(captain_hits / max(1, len(captain_data)) * 100, 1),
        "hits": captain_hits,
        "total_gws": len(captain_data),
        "best": best_captain,
        "worst": worst_captain,
        "picks": captain_data,
    }

    # Transfer P/L - group by GW for clean chart
    from collections import OrderedDict
    gw_transfers = OrderedDict()
    for ta in transfer_analysis:
        gw = ta["gw"]
        if gw not in gw_transfers:
            gw_transfers[gw] = {"gw": gw, "pts_diff": 0, "transfers": []}
        gw_transfers[gw]["pts_diff"] += ta["pts_diff"]
        gw_transfers[gw]["transfers"].append(
            f"{ta['out']['name']} → {ta['in']['name']} ({'+' if ta['pts_diff'] >= 0 else ''}{ta['pts_diff']})"
        )

    transfer_pl = []
    cumulative = 0
    for gw_data in sorted(gw_transfers.values(), key=lambda x: x["gw"]):
        cumulative += gw_data["pts_diff"]
        transfer_pl.append({
            "gw": gw_data["gw"],
            "pts_diff": gw_data["pts_diff"],
            "cumulative": cumulative,
            "details": gw_data["transfers"],
        })

    # Total bench points across all GWs
    total_bench_pts = sum(gw.get("points_on_bench", 0) for gw in gw_history)

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
        # Captain performance
        "captain_performance": captain_performance,
        # Transfer P/L
        "transfer_pl": transfer_pl,
        # Total bench waste
        "total_bench_pts": total_bench_pts,
    }


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


# ============ ROOT & BOOTSTRAP ============

@app.get("/")
async def root():
    path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    # Also check in the same directory as main.py would have been
    alt_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(alt_path):
        return FileResponse(alt_path)
    return {"message": "FPL Assistant API", "docs": "/docs"}


@app.get("/api/bootstrap")
async def get_bootstrap():
    return await fetch_fpl_data()


# ============ TRANSFER PLANNER / SOLVER ============

@app.post("/api/transfer-planner/{manager_id}")
async def get_transfer_planner(
    manager_id: int,
    request: TransferPlannerRequest = None,
    horizon: int = Query(8, ge=1, le=8)
):
    """
    Comprehensive transfer planner with multi-GW optimization.

    Returns three strategy plans (Safe, Balanced, Risky) with:
    - Squad health analysis
    - Fixture swing detection
    - Transfer recommendations per GW
    - Chip timing suggestions
    - Total expected points

    Accepts booked transfers in request body.
    """
    await refresh_fdr_data()

    # Fetch all required data
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()

    elements = data["elements"]
    elements_dict = {e["id"]: e for e in elements}
    teams_dict = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)
    next_gw = get_next_gameweek(events)
    planning_gw = next_gw if next_gw else min(current_gw + 1, 38)

    # Fetch manager data
    try:
        team_data = await fetch_manager_team(manager_id, current_gw)
        history = await fetch_manager_history(manager_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch manager data: {str(e)}")

    picks = team_data["picks"]["picks"]
    entry_history = team_data["picks"].get("entry_history", {})
    manager = team_data["manager"]
    bank = entry_history.get("bank", 0) / 10

    # Determine free transfers
    # FPL API doesn't directly give FT count, but we can infer from event_transfers
    # After using transfers, FT resets to 1 unless you rolled (then 2)
    event_transfers = entry_history.get("event_transfers", 0)
    event_transfers_cost = entry_history.get("event_transfers_cost", 0)

    # If transfers were made this GW, FT is 1 (unless hit was taken, still 1)
    # If no transfers made, FT could be 1-5 (rolled from previous)
    # This is a simplification - ideally we'd track FT across GWs
    if event_transfers > 0:
        free_transfers = 1
    else:
        # Assume 2 FT if no transfer was made (common case)
        free_transfers = 2

    # Build squad with xPts
    squad = []
    player_histories = {}

    # Fetch player histories
    async def fetch_hist(pid):
        try:
            return pid, await fetch_player_history(pid)
        except:
            return pid, {"history": []}

    hist_tasks = [fetch_hist(p["element"]) for p in picks]
    hist_results = await asyncio.gather(*hist_tasks)
    player_histories = {pid: hist for pid, hist in hist_results}

    for pick in picks:
        player = elements_dict.get(pick["element"])
        if not player:
            continue

        position_id = player["element_type"]
        player_hist = player_histories.get(player["id"])

        upcoming = get_player_upcoming_fixtures(
            player["team"], fixtures, planning_gw, planning_gw + horizon, teams_dict, position_id
        )
        stats = calculate_expected_points(
            player, position_id, current_gw, upcoming, teams_dict, fixtures, events,
            player_history=player_hist, all_players=elements
        )

        squad.append({
            "id": player["id"],
            "name": player["web_name"],
            "team": teams_dict.get(player["team"], {}).get("short_name", "???"),
            "team_id": player["team"],
            "position": POSITION_MAP.get(position_id, "???"),
            "position_id": position_id,
            "price": player["now_cost"] / 10,
            "selling_price": pick.get("selling_price", player["now_cost"]) / 10,
            "is_captain": pick["is_captain"],
            "is_vice_captain": pick["is_vice_captain"],
            "squad_position": pick["position"],
            "xpts": stats["xpts"],
            "xpts_ceiling": stats.get("xpts_ceiling", stats["xpts"] * 1.2),
            "xpts_floor": stats.get("xpts_floor", stats["xpts"] * 0.7),
            "expected_minutes": stats["expected_minutes"],
            "minutes_reason": stats["minutes_reason"],
            "form": float(player.get("form", 0) or 0),
            "ownership": float(player.get("selected_by_percent", 0) or 0),
            "status": player.get("status", "a"),
            "news": player.get("news", ""),
            "fixtures": upcoming[:5],
        })

    squad.sort(key=lambda x: x["squad_position"])

    # Get available chips
    available_chips = get_available_chips(history, current_gw)

    # Analyze squad health
    squad_health = analyze_squad_health(
        squad=squad,
        fixtures=fixtures,
        events=events,
        teams_dict=teams_dict,
        elements_dict=elements_dict,
        current_gw=planning_gw,
        horizon=horizon
    )

    # Calculate fixture swings
    fixture_swings = calculate_fixture_swings(
        fixtures=fixtures,
        teams_dict=teams_dict,
        current_gw=planning_gw,
        horizon=horizon
    )

    # Get chip recommendations
    chip_recommendations = get_chip_recommendations(
        squad=squad,
        available_chips=available_chips,
        fixtures=fixtures,
        events=events,
        teams_dict=teams_dict,
        elements_dict=elements_dict,
        current_gw=planning_gw,
        horizon=horizon,
        squad_health=squad_health
    )

    # Parse booked transfers
    booked_transfers = []
    if request and request.booked_transfers:
        for bt in request.booked_transfers:
            booked_transfers.append({
                "out_id": bt.out_id,
                "in_id": bt.in_id,
                "gw": bt.gw,
                "reason": bt.reason or "",
            })

    # Parse chip overrides
    chip_overrides = []
    if request and request.chip_overrides:
        for co in request.chip_overrides:
            chip_overrides.append({
                "chip": co.chip,
                "action": co.action,
                "gw": co.gw,
            })

    # Build strategy plans
    strategies = {}
    for strategy_name in ["safe", "balanced", "risky"]:
        plan = build_strategy_plan(
            squad=squad,
            bank=bank,
            free_transfers=free_transfers,
            fixtures=fixtures,
            events=events,
            teams_dict=teams_dict,
            elements=elements,
            current_gw=planning_gw,
            horizon=horizon,
            strategy=strategy_name,
            available_chips=available_chips,
            squad_health=squad_health,
            fixture_swings=fixture_swings,
            chip_recommendations=chip_recommendations,
            booked_transfers=booked_transfers,
            chip_overrides=chip_overrides,
        )
        strategies[strategy_name] = {
            "name": plan.name.title(),
            "description": plan.description,
            "headline": plan.headline,
            "total_xpts": plan.total_xpts,
            "hit_cost": plan.hit_cost,
            "transfers_made": plan.transfers_made,
            "risk_score": plan.risk_score,
            "gw_actions": plan.gw_actions,
            "chip_recommendations": [
                {
                    "chip": CHIP_DISPLAY.get(c.chip, c.chip.upper()),
                    "chip_id": c.chip,
                    "recommended_gw": c.recommended_gw,
                    "confidence": c.confidence,
                    "reason": c.reason,
                    "expected_value": round(c.expected_value, 1),
                }
                for c in plan.chip_recommendations
            ],
            "chip_placements": [
                {
                    "chip": CHIP_DISPLAY.get(cp.chip, cp.chip.upper()),
                    "chip_id": cp.chip,
                    "gw": cp.gw,
                    "marginal_xpts": round(cp.marginal_xpts, 1),
                    "reason": cp.reason,
                }
                for cp in plan.chip_placements
            ],
        }

    # Build fixture heatmap data for all teams
    fixture_heatmap = []
    for swing in fixture_swings:
        team_fixtures = []
        for fix in fixtures:
            gw = fix.get("event")
            if not gw or gw < planning_gw or gw > planning_gw + horizon:
                continue

            is_home = fix["team_h"] == swing.team_id
            is_away = fix["team_a"] == swing.team_id

            if is_home:
                opponent_id = fix["team_a"]
                fdr = get_fixture_fdr(opponent_id, True, 4, fpl_difficulty=fix.get("team_h_difficulty", 3))
            elif is_away:
                opponent_id = fix["team_h"]
                fdr = get_fixture_fdr(opponent_id, False, 4, fpl_difficulty=fix.get("team_a_difficulty", 3))
            else:
                continue

            team_fixtures.append({
                "gw": gw,
                "opponent": teams_dict.get(opponent_id, {}).get("short_name", "???"),
                "is_home": is_home,
                "fdr": fdr,
            })

        team_fixtures.sort(key=lambda x: x["gw"])

        fixture_heatmap.append({
            "team_id": swing.team_id,
            "team_name": swing.team_name,
            "team_short": swing.team_short,
            "swing": swing.swing,
            "rating": swing.rating,
            "dgw_gws": swing.dgw_in_horizon,
            "bgw_gws": swing.bgw_in_horizon,
            "fixtures": team_fixtures,
        })

    # Sort by swing (improving first)
    fixture_heatmap.sort(key=lambda x: x["swing"])

    return {
        "manager": {
            "id": manager["id"],
            "name": f"{manager['player_first_name']} {manager['player_last_name']}",
            "team_name": manager["name"],
        },
        "planning_gw": planning_gw,
        "horizon": horizon,
        "horizon_range": f"GW{planning_gw}-GW{min(planning_gw + horizon - 1, 38)}",
        "bank": round(bank, 1),
        "free_transfers": free_transfers,
        "available_chips": {
            CHIP_DISPLAY.get(k, k.upper()): v
            for k, v in available_chips.items()
        },
        "squad": squad,
        "squad_health": {
            "issues": [
                {
                    "player_id": i.player_id,
                    "player_name": i.player_name,
                    "team": i.team,
                    "position": i.position,
                    "issue_type": i.issue_type,
                    "severity": i.severity,
                    "description": i.description,
                    "affected_gws": i.affected_gws,
                    "recommendation": i.recommendation,
                }
                for i in squad_health
            ],
            "critical_count": len([i for i in squad_health if i.severity == "critical"]),
            "warning_count": len([i for i in squad_health if i.severity == "warning"]),
            "minor_count": len([i for i in squad_health if i.severity == "minor"]),
            "health_score": max(0, 100 - len([i for i in squad_health if i.severity == "critical"]) * 20
                               - len([i for i in squad_health if i.severity == "warning"]) * 10
                               - len([i for i in squad_health if i.severity == "minor"]) * 3),
        },
        "fixture_swings": [
            {
                "team_id": s.team_id,
                "team_name": s.team_name,
                "team_short": s.team_short,
                "current_fdr": s.current_fdr,
                "upcoming_fdr": s.upcoming_fdr,
                "swing": s.swing,
                "rating": s.rating,
                "dgw_gws": s.dgw_in_horizon,
                "bgw_gws": s.bgw_in_horizon,
            }
            for s in fixture_swings[:10]  # Top 10 most improving
        ],
        "fixture_heatmap": fixture_heatmap,
        "strategies": strategies,
        "booked_transfers": booked_transfers,
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
            "fixture_weights": FIXTURE_WEIGHTS,
            "fixture_weight_half_life": MODEL_CONFIG["xpts"].fixture_weight_half_life,
            "defcon_threshold_def": MODEL_CONFIG["xpts"].defcon_threshold_def,
            "defcon_threshold_mid": MODEL_CONFIG["xpts"].defcon_threshold_mid,
        }
    }


# ============ DEBUG ENDPOINTS ============

@app.get("/api/debug/team-strength")
async def debug_team_strength():
    """
    Debug endpoint to verify team strength data is loaded correctly.
    Shows data from import_data.json and calculated FDR values.
    """
    await refresh_fdr_data()

    data = await fetch_fpl_data()
    teams = {t["id"]: t["name"] for t in data["teams"]}

    result = []
    for team_id in sorted(teams.keys()):
        team_name = teams[team_id]

        # Get manual data (from import_data.json)
        manual = cache.manual_team_strength.get(team_id, {})

        # Get calculated FDR data
        fdr = cache.fdr_data.get(team_id, {})

        result.append({
            "team_id": team_id,
            "team_name": team_name,
            "manual_import": {
                "adjxg_for": manual.get("adjxg_for"),
                "adjxg_ag": manual.get("adjxg_ag"),
                "attack_delta": manual.get("attack_delta"),
                "defence_delta": manual.get("defence_delta"),
            } if manual else None,
            "calculated_fdr": {
                "blended_xg": fdr.get("blended_xg"),
                "blended_xga": fdr.get("blended_xga"),
                "attack_fdr": fdr.get("attack_fdr"),
                "attack_fdr_home": fdr.get("attack_fdr_home"),
                "attack_fdr_away": fdr.get("attack_fdr_away"),
                "defence_fdr": fdr.get("defence_fdr"),
                "cs_probability": fdr.get("cs_probability"),
                "data_source": fdr.get("data_source"),
            } if fdr else None,
        })

    return {
        "manual_team_strength_count": len(cache.manual_team_strength),
        "fdr_data_count": len(cache.fdr_data),
        "last_update": cache.fdr_last_update.isoformat() if cache.fdr_last_update else None,
        "teams": result
    }


@app.get("/api/debug/fixture-fdr/{team_id}")
async def debug_fixture_fdr(team_id: int):
    """
    Debug endpoint to see FDR calculations for a specific team's upcoming fixtures.
    """
    await refresh_fdr_data()

    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()

    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    next_gw = get_next_gameweek(events)

    team_info = teams.get(team_id)
    if not team_info:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Get team's FDR data
    team_fdr = cache.fdr_data.get(team_id, {})

    # Find upcoming fixtures
    upcoming = []
    for fix in fixtures:
        gw = fix.get("event")
        if not gw or gw < next_gw or fix.get("finished"):
            continue

        if fix["team_h"] == team_id:
            opp_id = fix["team_a"]
            is_home = True
        elif fix["team_a"] == team_id:
            opp_id = fix["team_h"]
            is_home = False
        else:
            continue

        opp_info = teams.get(opp_id, {})
        opp_fdr = cache.fdr_data.get(opp_id, {})

        # Calculate FDR for this fixture
        attack_fdr = get_fixture_fdr(opp_id, is_home, 4, team_id=team_id)
        defence_fdr = get_fixture_fdr(opp_id, is_home, 2, team_id=team_id)

        upcoming.append({
            "gw": gw,
            "opponent": opp_info.get("short_name", "???"),
            "opponent_id": opp_id,
            "is_home": is_home,
            "attack_fdr": attack_fdr,
            "defence_fdr": defence_fdr,
            "composite_fdr": round((attack_fdr + defence_fdr) / 2),
            "opponent_xg": opp_fdr.get("blended_xg"),
            "opponent_xga": opp_fdr.get("blended_xga"),
        })

        if len(upcoming) >= 8:
            break

    return {
        "team_id": team_id,
        "team_name": team_info["name"],
        "team_data": {
            "blended_xg": team_fdr.get("blended_xg"),
            "blended_xga": team_fdr.get("blended_xga"),
            "data_source": team_fdr.get("data_source"),
        },
        "upcoming_fixtures": upcoming
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


# ============ MAIN ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
