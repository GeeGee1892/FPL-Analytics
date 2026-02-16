"""
FPL Assistant - Services Module

HTTP client, circuit breaker, FPL API fetchers, FDR refresh service,
form calculations, expected minutes, expected points, captain scoring,
and fixture utilities.
"""

import asyncio
import math
import random
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import httpx
from fastapi import HTTPException

from fpl_assistant.config import MODEL_CONFIG
from fpl_assistant.constants import (
    FPL_BASE_URL, FIXTURE_WEIGHTS, DEFAULT_TEAM_STRENGTH, PROMOTED_TEAM_DEFAULTS,
    LEAGUE_AVG_XG, LEAGUE_AVG_XGA, PENALTY_TAKERS,
    apply_set_piece_share_to_multiplier,
    HOME_AWAY_FDR_ADJUSTMENT, TEAM_BASE_CS_RATES, TEAM_BASE_XG,
    EUROPEAN_TEAMS_UCL, EUROPEAN_TEAMS_UEL, EUROPEAN_TEAMS_UECL,
    ROTATION_MANAGERS, STABLE_MANAGERS,
    FPL_FDR_TO_10, DEFCON_POINTS, HORIZON_REGRESSION,
    LEAGUE_AVG_GOALS_PER_GAME, PLAYER_HOME_BOOST, PLAYER_AWAY_PENALTY,
    OPP_HOME_BOOST, OPP_AWAY_PENALTY, DEFENCE_HOME_BOOST, DEFENCE_AWAY_PENALTY,
)
from fpl_assistant.models import PlayerFormResult, TeamFormResult
from fpl_assistant.cache import cache
from fpl_assistant.calculators import (
    xga_to_attack_fdr, xg_to_defence_fdr, get_price_adjusted_fdr_multiplier,
    calculate_goals_conceded_penalty, calculate_matchup_attack_multiplier,
    calculate_matchup_cs_probability,
    home_away_calculator, variance_model,
    calculate_defcon_per_90, calculate_saves_per_90, calculate_expected_bonus,
)


logger = logging.getLogger("fpl_assistant")


# ============ HTTP CLIENT & CIRCUIT BREAKER ============

# Global HTTP client (initialized in lifespan)
http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Get the shared HTTP client, creating one if needed."""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            headers={"User-Agent": "FPL-Assistant/4.2"}
        )
    return http_client


# Circuit breaker state for FPL API
_circuit_breaker = {
    "consecutive_failures": 0,
    "open_until": None,  # datetime when circuit can be retried
    "threshold": 3,      # failures before opening circuit
    "cooldown": 60,      # seconds before retrying after circuit opens
}


async def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> httpx.Response:
    """
    Fetch URL with exponential backoff retry logic.
    Handles rate limiting (429) and transient errors.
    Includes circuit breaker: after 3 consecutive failures, fails fast for 60s.
    """
    cb = _circuit_breaker
    now = datetime.now()

    # Circuit breaker: fail fast if open
    if cb["open_until"] and now < cb["open_until"]:
        remaining = (cb["open_until"] - now).seconds
        raise HTTPException(
            status_code=503,
            detail=f"FPL API circuit breaker open, retrying in {remaining}s"
        )

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
            # Success - reset circuit breaker
            cb["consecutive_failures"] = 0
            cb["open_until"] = None
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

    # All retries exhausted - update circuit breaker
    cb["consecutive_failures"] += 1
    if cb["consecutive_failures"] >= cb["threshold"]:
        cb["open_until"] = now + timedelta(seconds=cb["cooldown"])
        logger.error(f"Circuit breaker OPEN after {cb['consecutive_failures']} consecutive failures. Cooldown {cb['cooldown']}s.")

    if last_error:
        raise last_error
    raise HTTPException(status_code=503, detail="Failed after max retries")


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


# ============ MATCHUP FDR & BACKUP DETECTION ============

def calculate_matchup_fdr(
    team_id: int,
    opponent_id: int,
    is_home: bool,
    position: int
) -> int:
    """
    v4.3.9: MATCHUP-BASED FDR with FIXED venue adjustment.

    CRITICAL FIX: Previous version double-applied venue adjustments to BOTH
    team attack AND opponent defence, creating ~30% swings that made
    Man City (H) look easier than Forest (A). This was wrong.

    NEW APPROACH: Calculate pure matchup first, then apply SINGLE venue adjustment.
    This matches real-world data where home advantage is worth ~0.15-0.2 xG, not 30%.

    For MID/FWD (attacking FDR):
        base_xg = (our_xG / avg) × (opp_xGA / avg) × avg
        final_xg = base_xg × venue_multiplier (1.08 home, 0.93 away)
        Higher expected goals → Lower FDR (easier to score)

    For GKP/DEF (defensive FDR):
        base_xga = (opp_xG / avg) × (our_xGA / avg) × avg
        final_xga = base_xga × inverse_venue (0.93 home, 1.08 away)
        Lower expected goals against → Lower FDR (easier CS)

    Args:
        team_id: Player's team ID
        opponent_id: Opponent team ID
        is_home: Is player at home?
        position: Player position (1=GKP, 2=DEF, 3=MID, 4=FWD)

    Returns:
        FDR 1-10 (1=very easy, 10=very hard)
    """
    if not cache.fdr_data:
        return 5  # Neutral fallback

    # Get team data (with fallbacks)
    team_data = cache.fdr_data.get(team_id, {})
    opp_data = cache.fdr_data.get(opponent_id, {})

    # Team strengths (blended = baseline + form)
    team_xg = team_data.get('blended_xg', LEAGUE_AVG_XG)
    team_xga = team_data.get('blended_xga', LEAGUE_AVG_XGA)
    opp_xg = opp_data.get('blended_xg', LEAGUE_AVG_XG)
    opp_xga = opp_data.get('blended_xga', LEAGUE_AVG_XGA)

    # v4.3.11: Use small OPPONENT venue adjustments (~5%)
    # FDR represents OPPONENT quality, not your big home/away performance swing
    # City's defence doesn't get 30% worse when they travel - only ~5%
    # Your big home/away swing is handled in xPts via player splits

    if position in [3, 4]:  # MID, FWD - care about scoring
        # Pure matchup: our attack vs their defence (NO venue adjustment yet)
        base_xg = (team_xg / LEAGUE_AVG_XG) * (opp_xga / LEAGUE_AVG_XGA) * LEAGUE_AVG_XG

        # Apply small venue adjustment for opponent quality difference
        # When HOME: opponent away, their defence slightly worse → you score ~5% more
        # When AWAY: opponent home, their defence slightly better → you score ~5% less
        if is_home:
            expected_goals = base_xg * OPP_HOME_BOOST     # 1.05 - slight home advantage
        else:
            expected_goals = base_xg * OPP_AWAY_PENALTY   # 0.95 - slight away penalty

        # Convert to FDR: more expected goals = easier = lower FDR
        if expected_goals >= 2.50:
            return 1   # Elite attack vs terrible defence
        elif expected_goals >= 2.10:
            return 2   # Very easy
        elif expected_goals >= 1.80:
            return 3   # Easy
        elif expected_goals >= 1.55:
            return 4   # Decent
        elif expected_goals >= 1.35:
            return 5   # Average (league avg ~1.35)
        elif expected_goals >= 1.15:
            return 6   # Below average
        elif expected_goals >= 1.00:
            return 7   # Hard
        elif expected_goals >= 0.85:
            return 8   # Very hard
        elif expected_goals >= 0.70:
            return 9   # Elite defence
        else:
            return 10  # Near impossible

    else:  # GKP, DEF - care about clean sheets
        # Pure matchup: their attack vs our defence (NO venue adjustment yet)
        base_xga = (opp_xg / LEAGUE_AVG_XG) * (team_xga / LEAGUE_AVG_XGA) * LEAGUE_AVG_XG

        # Apply venue adjustment for CS probability
        # When HOME: we defend better, opponent attacks worse → concede ~8% less
        # When AWAY: we defend worse, opponent attacks better → concede ~8% more
        if is_home:
            expected_goals_against = base_xga * DEFENCE_HOME_BOOST    # 0.92 - concede less at home
        else:
            expected_goals_against = base_xga * DEFENCE_AWAY_PENALTY  # 1.08 - concede more away

        # Convert to FDR: fewer expected goals against = easier CS = lower FDR
        if expected_goals_against <= 0.60:
            return 1   # Elite defence vs terrible attack
        elif expected_goals_against <= 0.80:
            return 2   # Very easy CS
        elif expected_goals_against <= 1.00:
            return 3   # Easy CS
        elif expected_goals_against <= 1.15:
            return 4   # Decent CS chance
        elif expected_goals_against <= 1.35:
            return 5   # Average (league avg ~1.35)
        elif expected_goals_against <= 1.55:
            return 6   # Below average
        elif expected_goals_against <= 1.80:
            return 7   # Hard
        elif expected_goals_against <= 2.10:
            return 8   # Very hard
        elif expected_goals_against <= 2.50:
            return 9   # Very weak defence vs elite attack
        else:
            return 10  # Near impossible CS


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


# ============ FDR SERVICE (v4.3.4 - No Understat) ============

async def refresh_fdr_data(force: bool = False):
    """
    v4.3.4: Completely rewritten to remove Understat dependency.

    Data sources (in priority order):
    1. Manual Analytic FPL data (from admin interface)
    2. DEFAULT_TEAM_STRENGTH constants (baseline for all 20 teams)

    This ensures FDR data is ALWAYS available, even on first load.
    """
    # Return cached data if available and not forced
    if not force and cache.fdr_data and not cache.fdr_is_stale():
        return cache.fdr_data

    # Always compute fresh FDR data
    return await _do_fdr_refresh()


async def _background_fdr_refresh():
    """Background task to refresh FDR data without blocking."""
    try:
        await _do_fdr_refresh()
    except Exception as e:
        logger.error(f"Background FDR refresh failed: {e}")


async def _do_fdr_refresh():
    """
    v4.3.4: Compute FDR data from Analytic FPL or defaults.

    NEVER depends on Understat. ALWAYS returns valid FDR data.
    """
    logger.info("Refreshing FDR data (v4.3.4 - Analytic FPL only)...")

    fdr_data = {}

    # Process all 20 teams
    for team_id, defaults in DEFAULT_TEAM_STRENGTH.items():
        # Check if we have manual Analytic FPL data for this team
        manual_data = cache.manual_team_strength.get(team_id) if cache.manual_team_strength else None

        if manual_data:
            # Use manual Analytic FPL data
            base_xg = manual_data.get('adjxg_for', defaults['xg'])
            base_xga = manual_data.get('adjxg_ag', defaults['xga'])
            team_name = manual_data.get('team_name', defaults['name'])

            # Get trend adjustments
            attack_delta = manual_data.get('attack_delta') or 0
            defence_delta = manual_data.get('defence_delta') or 0
            attack_trend = manual_data.get('attack_trend') or 0
            defence_trend = manual_data.get('defence_trend') or 0

            # Combine delta and trend into momentum
            attack_momentum = attack_delta * 0.6 + attack_trend * 0.4
            defence_momentum = defence_delta * 0.6 + defence_trend * 0.4

            # Apply trend as additive adjustment (capped ±0.15)
            attack_adjustment = max(-0.15, min(0.15, attack_momentum * 0.3))
            defence_adjustment = max(-0.15, min(0.15, defence_momentum * 0.3))

            blended_xg = base_xg + attack_adjustment
            blended_xga = base_xga + defence_adjustment
            data_source = 'analytic_fpl_manual'
            trend_info = {
                'attack_momentum': round(attack_momentum, 3),
                'defence_momentum': round(defence_momentum, 3),
                'attack_adjustment': round(attack_adjustment, 3),
                'defence_adjustment': round(defence_adjustment, 3),
            }
        else:
            # Use default team strength
            blended_xg = defaults['xg']
            blended_xga = defaults['xga']
            team_name = defaults['name']
            data_source = 'default_team_strength'
            trend_info = None

        # Estimate home/away splits (typical 15% swing)
        home_xga = blended_xga * 0.85  # Better defence at home
        away_xga = blended_xga * 1.15  # Worse defence away

        # Calculate CS probability using Poisson
        cs_probability = math.exp(-blended_xga) if blended_xga > 0 else 0.25
        cs_probability = min(0.52, max(0.08, cs_probability))

        # Calculate FDR scores using calibrated thresholds
        attack_fdr = xga_to_attack_fdr(blended_xga, is_opponent_home=None)
        attack_fdr_home = xga_to_attack_fdr(home_xga, is_opponent_home=True)
        attack_fdr_away = xga_to_attack_fdr(away_xga, is_opponent_home=False)
        defence_fdr = xg_to_defence_fdr(blended_xg)

        fdr_data[team_id] = {
            'team_name': team_name,
            'season_xg': 0,
            'season_xga': 0,
            'season_matches': 0,
            'form_xg': round(blended_xg, 2),
            'form_xga': round(blended_xga, 2),
            'form_ppg': 0,
            'season_xg_per_game': round(blended_xg, 2),
            'season_xga_per_game': round(blended_xga, 2),
            'blended_xg': round(blended_xg, 2),
            'blended_xga': round(blended_xga, 2),
            'home_xga_per_game': round(home_xga, 2),
            'away_xga_per_game': round(away_xga, 2),
            'cs_probability': round(cs_probability, 3),
            'xg_per_game': round(blended_xg, 2),
            'attack_fdr': max(1, min(10, attack_fdr)),
            'attack_fdr_home': max(1, min(10, attack_fdr_home)),
            'attack_fdr_away': max(1, min(10, attack_fdr_away)),
            'defence_fdr': max(1, min(10, defence_fdr)),
            'composite_fdr': round((attack_fdr + defence_fdr) / 2, 1),
            'data_source': data_source,
            'trend_info': trend_info,
        }

    cache.fdr_data = fdr_data
    cache.fdr_last_update = datetime.now()

    manual_count = sum(1 for d in fdr_data.values() if d['data_source'] == 'analytic_fpl_manual')
    logger.info(f"FDR data refreshed for {len(fdr_data)} teams ({manual_count} from Analytic FPL, {len(fdr_data) - manual_count} from defaults)")

    return fdr_data


def get_fixture_fdr(opponent_id: int, is_home: bool, position_id: int, fpl_difficulty: int = None, team_id: int = None) -> int:
    """
    Get FDR for a fixture based on opponent and player position.

    v4.3.3c: Now supports matchup-based FDR when team_id is provided.
    When team_id is given, calculates FDR considering BOTH teams' strengths.

    - FWD/MID: Uses team attack vs opponent defence matchup
    - DEF/GKP: Uses team defence vs opponent attack matchup

    Args:
        opponent_id: Opponent team ID
        is_home: Is the player at home?
        position_id: Player position (1=GKP, 2=DEF, 3=MID, 4=FWD)
        fpl_difficulty: FPL's built-in difficulty (1-5) for fallback
        team_id: Player's team ID (optional, enables matchup calculation)

    Returns:
        FDR 1-10 (1=very easy, 10=very hard)
    """
    # v4.3.3c: Use matchup-based FDR when team_id is provided
    if team_id is not None and cache.fdr_data:
        return calculate_matchup_fdr(team_id, opponent_id, is_home, position_id)

    # v4.3.3: Check for promoted team defaults first
    if opponent_id in PROMOTED_TEAM_DEFAULTS and (not cache.fdr_data or opponent_id not in cache.fdr_data):
        defaults = PROMOTED_TEAM_DEFAULTS[opponent_id]
        if position_id in [3, 4]:  # MID, FWD - care about scoring
            return defaults['attack_fdr']
        else:  # GKP, DEF - care about clean sheets
            return defaults['defence_fdr']

    if not cache.fdr_data or opponent_id not in cache.fdr_data:
        # Fallback: convert FPL's 1-5 difficulty to our 1-10 scale
        if fpl_difficulty is not None:
            # FPL_FDR_TO_10 = {1: 2, 2: 4, 3: 5, 4: 7, 5: 9}
            fdr_10 = FPL_FDR_TO_10.get(fpl_difficulty, 5)
            # Apply home/away adjustment
            ha_mult = HOME_AWAY_FDR_ADJUSTMENT['home'] if is_home else HOME_AWAY_FDR_ADJUSTMENT['away']
            return max(1, min(10, int(round(fdr_10 * ha_mult))))
        return 5  # Neutral fallback when no data at all

    opponent_data = cache.fdr_data[opponent_id]

    # Position-specific FDR with venue consideration
    # v4.3.3: Venue is already baked into attack_fdr_home/attack_fdr_away
    # - If we're HOME, opponent is AWAY → use their attack_fdr_away (weaker on road)
    # - If we're AWAY, opponent is HOME → use their attack_fdr_home (stronger at home)
    if position_id in [3, 4]:  # MID, FWD - care about scoring
        if is_home:
            base_fdr = opponent_data.get('attack_fdr_away', opponent_data.get('attack_fdr', 5))
        else:
            base_fdr = opponent_data.get('attack_fdr_home', opponent_data.get('attack_fdr', 5))
    else:  # GKP, DEF - care about clean sheets
        base_fdr = opponent_data.get('defence_fdr', 5)

    # v4.3.3: REMOVED the redundant +1/-1 venue adjustment
    # The venue effect is already captured in attack_fdr_home vs attack_fdr_away
    # Adding another adjustment was causing double-counting and pushing everything to 8

    return max(1, min(10, base_fdr))


# =============================================================================
# FORM ADJUSTMENT FUNCTIONS - v5.0
# =============================================================================

def calculate_player_form_factor(
    player: Dict,
    player_history: Optional[Dict],
    position: int
) -> PlayerFormResult:
    """
    v5.0: Calculate player's recent form vs season average.

    Compares last N games to season rates, with regression to mean.

    Example - Haaland slumping:
        Season xG90: 0.88, Season PPG: 7.5
        Last 6 GW: 5.0 PPG (implies ~0.45 xG90)
        pts_form = 5.0 / 7.5 = 0.67
        With 40% weight: combined = 0.4 × 0.67 + 0.6 × 1.0 = 0.87
        Effective xG90 = 0.88 × 0.87 = 0.77

    Example - Gabriel hot:
        Season PPG: 4.0
        Last 6 GW: 8.5 PPG
        pts_form = 8.5 / 4.0 = 2.13 → capped at 1.30
        Combined = 0.4 × 1.30 + 0.6 × 1.0 = 1.12

    Args:
        player: FPL API player data
        player_history: Player's game-by-game history
        position: Player position (1-4)

    Returns:
        PlayerFormResult with form factors and metadata
    """
    form_config = MODEL_CONFIG["form"]
    result = PlayerFormResult()

    # Get season rates
    total_minutes = int(player.get("minutes", 0) or 0)
    if total_minutes < 450:  # Less than 5 full games — need reliable season baseline
        return result  # Not enough data for season baseline

    mins90 = total_minutes / 90.0
    season_xg90 = float(player.get("expected_goals", 0) or 0) / mins90
    season_xa90 = float(player.get("expected_assists", 0) or 0) / mins90
    season_ppg = float(player.get("points_per_game", 0) or 0)

    result.season_ppg = season_ppg

    # Need history for recent form
    if not player_history or not player_history.get("history"):
        return result

    history = player_history["history"]

    # Get recent games (with minimum minutes filter)
    recent_games = []
    for h in reversed(history):  # Most recent first
        if len(recent_games) >= form_config.lookback_games:
            break
        mins = h.get("minutes", 0)
        if mins >= 30:  # Only count games with meaningful minutes
            recent_games.append(h)

    if len(recent_games) < form_config.min_games_for_form:
        return result  # Not enough recent games

    result.games_used = len(recent_games)

    # Calculate recent rates
    recent_minutes = sum(g.get("minutes", 0) for g in recent_games)
    if recent_minutes < form_config.min_minutes_for_form:
        return result

    recent_xg = sum(float(g.get("expected_goals", 0) or 0) for g in recent_games)
    recent_xa = sum(float(g.get("expected_assists", 0) or 0) for g in recent_games)
    recent_pts = sum(g.get("total_points", 0) for g in recent_games)

    recent_xg90 = recent_xg / (recent_minutes / 90.0)
    recent_xa90 = recent_xa / (recent_minutes / 90.0)
    recent_ppg = recent_pts / len(recent_games)

    result.recent_xg90 = round(recent_xg90, 3)
    result.recent_xa90 = round(recent_xa90, 3)
    result.recent_ppg = round(recent_ppg, 2)

    # Calculate raw form factors
    xg_form_raw = recent_xg90 / max(0.01, season_xg90) if season_xg90 > 0.01 else 1.0
    xa_form_raw = recent_xa90 / max(0.01, season_xa90) if season_xa90 > 0.01 else 1.0
    pts_form_raw = recent_ppg / max(1.0, season_ppg) if season_ppg > 1.0 else 1.0

    # Calculate confidence based on sample size
    # More games = higher confidence in form signal
    games_factor = (len(recent_games) - form_config.min_games_for_form) / (form_config.lookback_games - form_config.min_games_for_form)
    games_factor = max(0, min(1, games_factor))

    # v4.3.11: Check if player is premium (higher form sensitivity)
    # Premium players (£8+ MID/FWD, £6+ DEF) have more volatile form
    player_price = float(player.get("now_cost", 0) or 0) / 10.0
    is_premium = False
    if position in [3, 4] and player_price >= form_config.premium_threshold_mid_fwd:
        is_premium = True
    elif position == 2 and player_price >= form_config.premium_threshold_def:
        is_premium = True

    if is_premium:
        # Premium players: higher confidence in form signal
        base_confidence = form_config.base_form_weight
        max_confidence = form_config.premium_form_weight  # 0.55 instead of 0.50
        min_form = form_config.premium_min_form_factor     # 0.55 instead of 0.70
        max_form = form_config.premium_max_form_factor     # 1.50 instead of 1.30
    else:
        base_confidence = form_config.base_form_weight
        max_confidence = form_config.max_form_weight
        min_form = form_config.min_form_factor
        max_form = form_config.max_form_factor

    confidence = base_confidence + games_factor * (max_confidence - base_confidence)

    # Apply position-specific sensitivity
    sensitivity = form_config.position_form_sensitivity.get(position, 0.85)
    confidence *= sensitivity

    result.confidence = round(confidence, 3)

    # Regress raw form factors to mean
    xg_form = confidence * xg_form_raw + (1 - confidence) * 1.0
    xa_form = confidence * xa_form_raw + (1 - confidence) * 1.0
    pts_form = confidence * pts_form_raw + (1 - confidence) * 1.0

    # Cap at config bounds (use premium bounds if applicable)
    xg_form = max(min_form, min(max_form, xg_form))
    xa_form = max(min_form, min(max_form, xa_form))
    pts_form = max(min_form, min(max_form, pts_form))

    result.xg_form = round(xg_form, 3)
    result.xa_form = round(xa_form, 3)
    result.pts_form = round(pts_form, 3)

    # Combined form blends xGI form with PPG form
    # PPG form captures bonus, defensive pts, etc. that xGI misses
    #
    # FWD: Goals are everything, but PPG captures bonus for haulers
    # MID: Mixed - attacking returns + bonus + CS (1 pt)
    # DEF: Mostly CS/defensive, xGI is small part of total - use PPG heavily
    # GKP: Almost all CS/saves, xGI minimal - pure PPG form
    #
    # Rationale: A player with low recent xG but high recent PPG (from bonus, CS)
    # shouldn't be treated as "cold" - they're still delivering FPL value.

    if position == 4:  # FWD - xG dominant but PPG captures bonus
        xgi_form = 0.75 * xg_form + 0.25 * xa_form
        result.combined_form = round(0.60 * xgi_form + 0.40 * pts_form, 3)
    elif position == 3:  # MID - balanced between attacking and total value
        xgi_form = 0.55 * xg_form + 0.45 * xa_form
        result.combined_form = round(0.50 * xgi_form + 0.50 * pts_form, 3)
    elif position == 2:  # DEF - mostly defensive, xGI is small part
        xgi_form = 0.50 * xg_form + 0.50 * xa_form
        result.combined_form = round(0.30 * xgi_form + 0.70 * pts_form, 3)
    else:  # GKP - almost pure PPG form (saves, CS dominate)
        result.combined_form = round(pts_form, 3)

    return result


def calculate_team_form_factor(team_id: int) -> TeamFormResult:
    """
    v5.0: Calculate team's current form vs baseline.

    Uses Analytic FPL momentum data when available:
    - attack_delta = recent xG - baseline xG (+ = scoring more)
    - defence_delta = recent xGA - baseline xGA (- = conceding less = better)

    Example - Man City poor attack form:
        attack_delta = -0.40 (scoring 0.4 less xG than baseline)
        attack_form = 1.0 + (-0.40 × 0.5) = 0.80
        Player xG multiplied by 0.80

    Example - Wolves improving defence:
        defence_delta = -0.25 (conceding 0.25 less xGA)
        defence_form = 1.0 + (-0.25 × 0.5) = 0.875
        For ATTACKING players vs Wolves: harder to score (opp defence improved)

    Args:
        team_id: FPL team ID

    Returns:
        TeamFormResult with attack/defence form multipliers
    """
    form_config = MODEL_CONFIG["form"]
    result = TeamFormResult()

    # Check for manual Analytic FPL data
    if cache.manual_team_strength and team_id in cache.manual_team_strength:
        data = cache.manual_team_strength[team_id]

        # Get momentum data
        attack_delta = data.get('attack_delta', 0) or 0
        defence_delta = data.get('defence_delta', 0) or 0
        attack_trend = data.get('attack_trend', 0) or 0
        defence_trend = data.get('defence_trend', 0) or 0

        result.attack_delta = attack_delta
        result.defence_delta = defence_delta

        # Combine delta and trend (delta = current vs baseline, trend = direction)
        attack_momentum = attack_delta * 0.7 + attack_trend * 0.3
        defence_momentum = defence_delta * 0.7 + defence_trend * 0.3

        # Convert to form multiplier
        # +0.40 attack_momentum = 1.16 attack_form
        # -0.40 attack_momentum = 0.84 attack_form
        attack_form = 1.0 + (attack_momentum * form_config.team_form_weight)
        defence_form = 1.0 + (defence_momentum * form_config.team_form_weight)

        # Cap at config bounds
        result.attack_form = round(
            max(form_config.min_team_form, min(form_config.max_team_form, attack_form)),
            3
        )
        result.defence_form = round(
            max(form_config.min_team_form, min(form_config.max_team_form, defence_form)),
            3
        )
        result.source = "analytic_fpl"

        return result

    # Fallback: Check FDR cache for any form signals
    if cache.fdr_data and team_id in cache.fdr_data:
        fdr = cache.fdr_data[team_id]
        trend_info = fdr.get('trend_info')

        if trend_info:
            attack_momentum = trend_info.get('attack_momentum', 0)
            defence_momentum = trend_info.get('defence_momentum', 0)

            attack_form = 1.0 + (attack_momentum * form_config.team_form_weight)
            defence_form = 1.0 + (defence_momentum * form_config.team_form_weight)

            result.attack_form = round(
                max(form_config.min_team_form, min(form_config.max_team_form, attack_form)),
                3
            )
            result.defence_form = round(
                max(form_config.min_team_form, min(form_config.max_team_form, defence_form)),
                3
            )
            result.source = "fdr_trend"

            return result

    return result  # Default: neutral form


# =============================================================================
# EXPECTED MINUTES
# =============================================================================

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
        if is_backup and confidence >= 0.60:
            # v5.2: Lowered from 0.75 — catch more rotation players
            if position == 1:  # GKP backups get almost nothing
                return 0, f"backup_{backup_reason}"
            else:  # Outfield backups might get cameos
                return 10, f"backup_{backup_reason}"
        elif is_backup and confidence >= 0.40:
            # v5.2: Lowered from 0.50 — flag borderline rotation risks
            return 15, f"likely_backup_{backup_reason}"

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
        # v5.2: Steeper discount — subs have inconsistent opportunities
        elif last_4_subs >= 3:
            avg_sub_mins = sum(m for m in last_4 if 0 < m < 60) / max(1, last_4_subs)
            base_mins = min(25, avg_sub_mins * 0.5)  # Halved + capped at 25 min
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


# =============================================================================
# EXPECTED POINTS
# =============================================================================

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
    player_name = player.get("web_name", "")  # For set piece share lookup (v4.3.2)

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

    # ==================== v5.0: INDIVIDUAL FORM ADJUSTMENT ====================
    # Compare recent performance to season average.
    # Hot players (Gabriel) get boosted; cold players (Haaland slump) get reduced.
    player_form = calculate_player_form_factor(player, player_history, position)

    # Apply form factor to attacking rates
    # This adjusts season xG90/xA90 toward recent performance
    form_adjusted_xG90 = xG90 * player_form.combined_form
    form_adjusted_xA90 = xA90 * player_form.combined_form

    # ==================== v5.0: TEAM FORM ADJUSTMENT ====================
    # City struggling = all City players get reduced output
    # Wolves improving defensively = harder to score against
    team_form = calculate_team_form_factor(team_id)

    # Apply team attack form to player's output
    # If team attack_form = 0.88 (City struggling), reduce xG by 12%
    form_adjusted_xG90 *= team_form.attack_form
    form_adjusted_xA90 *= team_form.attack_form

    # Use form-adjusted rates for calculations
    xG90_effective = form_adjusted_xG90
    xA90_effective = form_adjusted_xA90

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

    # Get team's defensive data for CS rate and DEFCON workload dampening
    # v4.3.3b: Moved up before DEFCON call so we can pass team_xga
    if cache.fdr_data and team_id in cache.fdr_data:
        team_base_cs = cache.fdr_data[team_id].get('cs_probability', 0.22)
        team_xga = cache.fdr_data[team_id].get('blended_xga', 1.35)
    else:
        team_base_cs = TEAM_BASE_CS_RATES.get(team_name, 0.22)
        team_xga = 1.35  # League average fallback

    # Calculate component stats
    # v4.3.3b: Pass team_xga for workload dampening (elite teams have fewer DEFCON opportunities)
    defcon_per_90, defcon_prob, defcon_pts_total = calculate_defcon_per_90(player, position, team_xga)
    saves_per_90, save_pts_per_90 = calculate_saves_per_90(player) if position == 1 else (0, 0)
    bonus_per_90, expected_bonus = calculate_expected_bonus(player, position)

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

    # v4.1 FIX: Accumulate fixture-weighted values for use in final discretization
    # Previously these were calculated in the loop but then discarded
    weighted_xG90 = 0  # Fixture-adjusted xG per 90
    weighted_xA90 = 0  # Fixture-adjusted xA per 90
    weighted_bonus = 0  # Fixture-adjusted expected bonus
    weighted_xGA = 0  # Fixture-specific expected goals against
    weighted_saves = 0  # Fixture-specific expected saves (GKP only)

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
        # v4.3.3c: Use MATCHUP-BASED CS probability considering BOTH teams' strengths
        # This considers: our defence (xGA) vs their attack (xG) with home/away adjustments

        if fixture_xg_data and fixture_xg_weight > 0:
            # USE FIXTURE-SPECIFIC xG PROJECTION
            # This is the highest-signal data for near-term predictions
            if is_home:
                fixture_specific_xga = fixture_xg_data.get("away_xg", 1.3)
            else:
                fixture_specific_xga = fixture_xg_data.get("home_xg", 1.5)

            # Get matchup-based CS probability for blending
            matchup_cs_prob, matchup_xga = calculate_matchup_cs_probability(
                team_id, opponent_id, is_home, team_base_cs
            )

            # Convert fixture xGA to CS prob
            cs_config = MODEL_CONFIG["clean_sheet"]
            fixture_cs_prob_raw = math.exp(-fixture_specific_xga * cs_config.cs_steepness)
            fixture_cs_prob_raw = min(cs_config.cs_prob_max, max(cs_config.cs_prob_min, fixture_cs_prob_raw))

            # Blend: fixture_xg_weight from fixture projection, rest from matchup model
            fixture_cs_prob = fixture_xg_weight * fixture_cs_prob_raw + (1 - fixture_xg_weight) * matchup_cs_prob
            # Also blend expected goals against for goals conceded penalty
            expected_goals_against = fixture_xg_weight * fixture_specific_xga + (1 - fixture_xg_weight) * matchup_xga
            data_source_for_fixture = "fixture_xg+matchup"
        else:
            # v4.3.3c: Use full matchup-based CS probability
            # This considers: our defence (xGA) vs their attack (xG)
            fixture_cs_prob, expected_goals_against = calculate_matchup_cs_probability(
                team_id, opponent_id, is_home, team_base_cs
            )
            data_source_for_fixture = "matchup_model"

        # v5.0: Apply team defence form to CS probability
        # If team is conceding more than baseline (defence_form > 1.0), reduce CS probability
        # This captures teams like West Ham who are underperforming defensively
        if team_form.defence_form != 1.0:
            # Adjust expected goals against by defence form
            adjusted_xga = expected_goals_against * team_form.defence_form
            cs_config = MODEL_CONFIG["clean_sheet"]
            # Recalculate CS probability with adjusted xGA
            adjusted_cs_prob = math.exp(-adjusted_xga * cs_config.cs_steepness)
            adjusted_cs_prob = min(cs_config.cs_prob_max, max(cs_config.cs_prob_min, adjusted_cs_prob))
            # Blend with original (50% weight to avoid over-adjustment)
            fixture_cs_prob = 0.50 * fixture_cs_prob + 0.50 * adjusted_cs_prob

        avg_cs_prob += fixture_cs_prob * weight

        # ==================== FIXTURE-SPECIFIC xGI ====================
        # v4.3.3c: Use MATCHUP-BASED multiplier considering BOTH teams' strengths
        # This replaces the one-sided approach that only looked at opponent's xGA

        if fixture_xg_data and fixture_xg_weight > 0:
            # Use fixture-specific attacking projection (from betting odds etc.)
            if is_home:
                fixture_specific_xg_for = fixture_xg_data.get("home_xg", 1.5)
            else:
                fixture_specific_xg_for = fixture_xg_data.get("away_xg", 1.0)

            # Convert fixture xG to multiplier relative to our baseline
            team_base_xg = cache.fdr_data.get(team_id, {}).get('blended_xg', LEAGUE_AVG_GOALS_PER_GAME)
            if team_base_xg > 0:
                fixture_attack_mult = fixture_specific_xg_for / team_base_xg
            else:
                fixture_attack_mult = 1.0

            # Get matchup-based multiplier for blending
            matchup_attack_mult = calculate_matchup_attack_multiplier(
                team_id, opponent_id, is_home, player_price, position
            )

            # Blend: fixture_xg_weight from fixture projection, rest from matchup model
            attack_multiplier = fixture_xg_weight * fixture_attack_mult + (1 - fixture_xg_weight) * matchup_attack_mult
        else:
            # v4.3.3c: Use full matchup-based multiplier
            # This considers BOTH our team's attack AND opponent's defence
            attack_multiplier = calculate_matchup_attack_multiplier(
                team_id, opponent_id, is_home, player_price, position
            )

        # v4.3: POSITION AND PRICE-SPECIFIC CAPS
        # Premium attackers destroy weak defences; budget players are more volatile
        # FWDs should have WIDER bounds than DEFs (more binary outcomes)
        # v4.3.1: Further widened based on Haaland analysis - caps were binding on easy fixtures
        if position == 4:  # FWD
            if player_price >= 10.0:
                # Premium FWD: Haaland vs Southampton should hit ~1.45 multiplier
                mult_cap_high, mult_cap_low = 1.48, 0.55
            elif player_price >= 7.0:
                mult_cap_high, mult_cap_low = 1.55, 0.50
            else:
                mult_cap_high, mult_cap_low = 1.65, 0.45
        elif position == 3:  # MID
            # v4.3.3: Widened MID caps - they were getting undervalued vs DEFs
            # Elite MIDs (Salah, Wirtz, Palmer) play in advanced positions and should
            # benefit as much from easy fixtures as FWDs
            if player_price >= 10.0:
                # Premium MID: Salah-tier - now matches premium FWD caps
                mult_cap_high, mult_cap_low = 1.50, 0.52
            elif player_price >= 7.0:
                mult_cap_high, mult_cap_low = 1.58, 0.48
            else:
                mult_cap_high, mult_cap_low = 1.65, 0.45
        elif position == 2:  # DEF
            # v4.3.3: Slightly narrowed DEF caps - they were overvalued
            # CS probability already handles fixture variance, attacking returns should be stable
            mult_cap_high, mult_cap_low = 1.25, 0.75
        else:  # GKP
            # GKPs mostly CS-driven, attacking returns minimal
            mult_cap_high, mult_cap_low = 1.20, 0.80

        attack_multiplier = min(mult_cap_high, max(mult_cap_low, attack_multiplier))

        # v4.3.2: Apply set piece share correction
        # Penalties and direct FKs are fixture-neutral - their xG doesn't depend on opponent
        # Only open play xG should be suppressed by hard fixture multipliers
        attack_multiplier = apply_set_piece_share_to_multiplier(attack_multiplier, player_name)

        # ==================== HOME/AWAY SPLIT FOR xGI ====================
        # v4.3.11: Separated venue effects:
        # - attack_multiplier has SMALL opponent venue effect (~5%)
        # - Player splits OR fallback boost handles BIG player venue effect (~15%)
        #
        # If player has sufficient home/away history, use split-specific rates
        # If NOT, apply fallback venue boost to capture their typical home/away swing
        # v5.0: Start with form-adjusted rates instead of raw season rates
        # v5.2: Decay form toward baseline for later fixtures — hot streaks don't persist flat
        # Fixture 0: full form effect, Fixture 7: ~30% form effect
        form_decay = max(0.3, 1.0 - i * 0.1)
        decayed_form = 1.0 + (player_form.combined_form - 1.0) * form_decay
        # Recompute effective rates with decayed form (instead of flat xG90_effective)
        fixture_xG90_decayed = xG90 * decayed_form * team_form.attack_form
        fixture_xA90_decayed = xA90 * decayed_form * team_form.attack_form

        fixture_xG90_base = fixture_xG90_decayed
        fixture_xA90_base = fixture_xA90_decayed

        if home_away_split.has_sufficient_data:
            # USE PLAYER'S ACTUAL HOME/AWAY SPLITS
            ha_config = MODEL_CONFIG["home_away"]
            if is_home:
                split_xG90 = home_away_split.home_xG90
                split_xA90 = home_away_split.home_xA90
            else:
                split_xG90 = home_away_split.away_xG90
                split_xA90 = home_away_split.away_xA90

            # Blend split with form-decayed overall (60% split, 40% overall)
            fixture_xG90_base = ha_config.split_weight * split_xG90 + (1 - ha_config.split_weight) * fixture_xG90_decayed
            fixture_xA90_base = ha_config.split_weight * split_xA90 + (1 - ha_config.split_weight) * fixture_xA90_decayed
        else:
            # NO SPLIT DATA: Apply fallback venue boost (~15%)
            if is_home:
                fixture_xG90_base = fixture_xG90_decayed * PLAYER_HOME_BOOST    # 1.15
                fixture_xA90_base = fixture_xA90_decayed * PLAYER_HOME_BOOST
            else:
                fixture_xG90_base = fixture_xG90_decayed * PLAYER_AWAY_PENALTY  # 0.85
                fixture_xA90_base = fixture_xA90_decayed * PLAYER_AWAY_PENALTY

        # Apply matchup multiplier to xG and calculate fixture xGI
        # v4.3.11: attack_multiplier has SMALL (~5%) opponent venue effect
        # Player's BIG venue effect (~15%) is now in fixture_xG90_base (from splits or fallback)
        #
        # v4.3.11 FIX: Penalty boost should NOT scale with attack_multiplier
        # A penalty is ~0.76 xG regardless of opponent - City or Forest
        # Only open-play xG should scale with matchup quality
        fixture_xG90 = (fixture_xG90_base * attack_multiplier) + penalty_boost
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

            # RECALIBRATED v4.2: GKP competes with 4-5 DEFs for bonus in CS games
            # In a typical CS game, DEFs (Gabriel, VVD, Saliba) often win bonus
            # GKP needs exceptional saves (7+) to beat DEFs who get same +12 CS BPS
            # Convert BPS to expected bonus - further reduced
            if cs_bps >= 32:
                bonus_given_cs = 1.4  # v4.2: Was 1.6 - needs 10+ saves in CS to dominate
            elif cs_bps >= 28:
                bonus_given_cs = 1.0  # v4.2: Was 1.1 - 8+ saves
            elif cs_bps >= 24:
                bonus_given_cs = 0.6  # v4.2: Was 0.7 - 6+ saves
            elif cs_bps >= 20:
                bonus_given_cs = 0.35  # v4.2: Was 0.4 - 4+ saves
            else:
                bonus_given_cs = 0.15  # v4.2: Was 0.2 - minimal saves

            # CS bonus: probability × expected bonus if CS happens
            cs_bonus_component = fixture_cs_prob * bonus_given_cs

            # Non-CS games: saves still contribute to bonus but extremely hard to win
            # ~2 BPS per save, need 25+ BPS to compete, so 7+ saves for any real chance
            # Even then, attacking players usually dominate in high-scoring games
            # RECALIBRATED v4.1: Reduced from 0.05 to 0.03
            non_cs_save_bonus = (1 - fixture_cs_prob) * max(0, (fixture_expected_saves - 4) * 0.03)

            fixture_bonus_pts = cs_bonus_component + non_cs_save_bonus

        elif position == 2:  # DEF
            # DEF bonus from CS + attacking returns
            # RECALIBRATED v4.1: CS and attack bonus are COMPETITIVE, not additive
            #
            # Key insight: When Gabriel scores in a CS game, his bonus comes from:
            # - Goal (+12 BPS) puts him in bonus contention
            # - CS (+12 BPS) is baseline for ALL defenders, doesn't give edge
            #
            # So attack bonus and CS bonus should NOT simply add together.
            # CS bonus is the BASELINE expectation; attacking returns provide UPSIDE.
            #
            # Competition dynamics:
            # - In CS games: 4-5 DEFs + GKP all get +12 BPS, need something extra
            # - Gabriel with goal: dominates (12 goal + 12 CS = 24+ BPS)
            # - Gabriel without goal: competing with VVD, Saliba for ~0.4 expected bonus

            # CS bonus baseline - what you expect just from CS (competing with teammates)
            # v4.2: Reduced from 0.40 to 0.32 - 4-5 DEFs splitting bonus pool
            # In a typical CS game, ~1.2 expected bonus split among 5 DEFs = 0.24 each
            # Plus some probability of getting 2 bonus = ~0.32
            cs_bonus_baseline = fixture_cs_prob * 0.32

            # Attack upside - extra bonus from goal/assist
            # v4.2: Reduced from 0.30 to 0.25 - DEF xGI is very low
            attack_upside = attack_bonus * 0.25

            # Use competitive model: attack bonus can exceed CS baseline, not stack
            # Base case: CS baseline
            # With attacking return: mostly replaces CS baseline (goal scorer wins)
            # Small blending factor (0.12) for cases where DEF gets 2 bonus from CS
            # then attacking player gets 3 bonus
            fixture_bonus_pts = max(cs_bonus_baseline, attack_upside) + min(cs_bonus_baseline, attack_upside) * 0.12

        elif position == 3:  # MID
            # v4.3: HYBRID CONDITIONAL MODEL for MIDs
            # MIDs have mixed profile: some are FWD-like (Salah), some are DEF-like (Rice)
            # Use blend of conditional (goals) and linear (CS, other contributions)

            # Conditional component: Goals
            p_score_mid = 1 - math.exp(-fixture_xG90)
            p_brace_mid = 1 - math.exp(-fixture_xG90) - fixture_xG90 * math.exp(-fixture_xG90)
            p_brace_mid = max(0, p_brace_mid)
            p_single_mid = p_score_mid - p_brace_mid

            # MID goal bonuses (slightly lower than FWD due to more competition)
            bonus_single_mid = 1.9
            bonus_brace_mid = 2.75
            bonus_blank_mid_attack = 0.15  # MIDs can get bonus from assists/key passes without scoring

            conditional_bonus = (
                p_single_mid * bonus_single_mid +
                p_brace_mid * bonus_brace_mid +
                (1 - p_score_mid) * bonus_blank_mid_attack
            )

            # CS component (helps defensive mids)
            cs_bonus_component = fixture_cs_prob * 0.45  # +6 BPS for CS

            # Blend: 75% conditional (attack-driven), 25% CS
            fixture_bonus_pts = conditional_bonus * 0.75 + cs_bonus_component * 0.25

        else:  # FWD
            # v4.3.1: CONDITIONAL BONUS MODEL (simplified)
            # Key insight: FWD bonus is driven by SCORING, not continuous xG
            # - If you score, you likely get bonus (especially with brace)
            # - If you blank, you almost never get bonus
            #
            # Fixture difficulty already captured in fixture_xG90 which affects P(score)
            # The bonus GIVEN scoring is relatively stable

            # Poisson probability of at least one goal
            p_score = 1 - math.exp(-fixture_xG90)

            # Probability of 2+ goals (brace)
            p_brace = 1 - math.exp(-fixture_xG90) - fixture_xG90 * math.exp(-fixture_xG90)
            p_brace = max(0, p_brace)  # Ensure non-negative

            # Probability of hat-trick (3+ goals) - rare but huge
            p_hattrick = 0
            if fixture_xG90 > 0.8:
                # Only worth modeling for high xG fixtures
                # P(3+) = 1 - P(0) - P(1) - P(2)
                p_hattrick = max(0, 1 - math.exp(-fixture_xG90) * (1 + fixture_xG90 + fixture_xG90**2/2))

            # P(exactly 1 goal)
            p_single = p_score - p_brace
            # P(exactly 2 goals)
            p_double = p_brace - p_hattrick

            # Bonus given scoring outcomes (empirically calibrated from Haaland data)
            # Single goal: Usually gets 2, sometimes 3, occasionally 1
            bonus_single = 2.10

            # Brace: Almost always 3 bonus
            bonus_double = 2.90

            # Hat-trick: Always 3 bonus, often with assists too
            bonus_hattrick = 3.00

            # Blank: Very rare to get bonus for FWD
            bonus_blank = 0.05

            fixture_bonus_pts = (
                p_single * bonus_single +
                p_double * bonus_double +
                p_hattrick * bonus_hattrick +
                (1 - p_score) * bonus_blank
            )

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

        # v4.1 FIX: Accumulate fixture-weighted values for final discretization
        # These capture the fixture-specific adjustments that would otherwise be lost
        weighted_xG90 += fixture_xG90 * weight * horizon_regression
        weighted_xA90 += fixture_xA90 * weight * horizon_regression
        weighted_bonus += fixture_bonus_pts * weight * horizon_regression
        weighted_xGA += expected_goals_against * weight
        if position == 1:
            weighted_saves += fixture_expected_saves * weight

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
    # v4.1 FIX: Use fixture-weighted accumulated values from the loop
    # Previously this section used baseline rates, ignoring all the fixture-specific
    # adjustments calculated in the loop (opponent strength, home/away, etc.)

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
    # v4.1 FIX: Use fixture-weighted bonus instead of baseline expected_bonus
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
    bonus_xpts_per_fix = weighted_bonus * bonus_discount  # v4.1: Was expected_bonus

    # Goal and assist points (continuous - scale with minutes)
    # v4.1 FIX: Use fixture-weighted xG90/xA90 which include opponent adjustments
    xgi_xpts_per_fix = ((goal_pts * weighted_xG90) + (3 * weighted_xA90)) * minutes_factor

    # Save points (continuous - GKP only)
    # v4.1 FIX: Use fixture-weighted expected saves
    if position == 1:
        # Convert weighted saves to points (1 pt per 3 saves)
        save_xpts_per_fix = (weighted_saves / 3.0) * minutes_factor
    else:
        save_xpts_per_fix = 0

    # DEFCON points (discrete - requires threshold in 90 mins)
    # Scale down for reduced minutes as less time to accumulate actions
    defcon_xpts_per_fix = (defcon_prob * DEFCON_POINTS * min(1.0, minutes_factor * 1.1)) if position in [2, 3] else 0

    # Deductions (continuous - exposure-based)
    yellow_deduction_per_fix = yellow_per_90 * minutes_factor
    og_deduction_per_fix = (og_per_90 * 2.0 * minutes_factor) if position in [1, 2] else (og_per_90 * 2.0 * 0.2 * minutes_factor)

    # Goals conceded deduction (uses Poisson model)
    # v4.1 FIX: Use fixture-weighted xGA instead of team baseline
    # This means DEF vs Wolves (H) gets less GC penalty than DEF vs Liverpool (A)
    if position in [1, 2]:
        gc_deduction_per_fix = calculate_goals_conceded_penalty(weighted_xGA) * prob_60_plus  # Only if play 60+
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
        # v5.0: Form adjustment data
        "form_data": {
            "individual_form": player_form.combined_form,
            "xg_form": player_form.xg_form,
            "xa_form": player_form.xa_form,
            "pts_form": player_form.pts_form,
            "recent_ppg": player_form.recent_ppg,
            "season_ppg": player_form.season_ppg,
            "games_used": player_form.games_used,
            "confidence": player_form.confidence,
            "team_attack_form": team_form.attack_form,
            "team_defence_form": team_form.defence_form,
            "team_form_source": team_form.source,
        },
        "xG_per_90_effective": round(xG90_effective, 3),
        "xA_per_90_effective": round(xA90_effective, 3),
    }


# =============================================================================
# CAPTAIN & FIXTURE UTILITIES
# =============================================================================

def calculate_captain_score(
    player: Dict,
    position_id: int,
    base_xpts: float,
    ceiling_xpts: float,
    next_fixture: Optional[Dict]
) -> Dict:
    """
    Calculate captaincy score factoring in ceiling potential AND differential value.

    KEY INSIGHT: For captain decisions, we care about the 80th percentile outcome,
    not the expected value. When I captain Haaland vs a weak defence, I'm betting
    on his 15+ ceiling, not his 8.5 xPts.

    For rank climbing, we want:
    1. High ceiling potential (attackers vs weak defences at home)
    2. Differential upside (low ownership = rank gain if they haul)

    Returns dict with captain_score, ceiling_mult, and diff_boost.
    """
    diff_boost = 0.0

    # Use ceiling as the base for captain scoring, not xPts
    # Ceiling is already position-adjusted via variance model
    captain_base = ceiling_xpts

    # Fixture-based ceiling adjustment (smaller multipliers since ceiling already accounts for some of this)
    fixture_mult = 1.0
    if next_fixture:
        is_home = next_fixture.get("is_home", False)
        opponent_id = next_fixture.get("opponent_id")

        # Home advantage increases ceiling slightly
        if is_home:
            fixture_mult *= 1.05

        # Opponent defensive strength affects attacker ceilings
        # v4.3.2 FIX: Use attack_fdr for FWD/MID (how hard to score against opponent)
        if position_id in [3, 4] and cache.fdr_data and opponent_id in cache.fdr_data:
            opp_attack_fdr = cache.fdr_data[opponent_id].get("attack_fdr", 5)
            if opp_attack_fdr <= 3:
                fixture_mult *= 1.10  # Weak defence = haul potential
            elif opp_attack_fdr <= 4:
                fixture_mult *= 1.05
            elif opp_attack_fdr >= 8:
                fixture_mult *= 0.90  # Strong defence = capped upside
            elif opp_attack_fdr >= 7:
                fixture_mult *= 0.95

    # Historical explosiveness - PPG as proxy for haul frequency
    total_points = int(player.get("total_points", 0) or 0)
    starts = int(player.get("starts", 0) or 0)
    if starts >= 5:
        ppg = total_points / starts
        if ppg >= 8.0:
            fixture_mult *= 1.10  # Elite returner (Haaland, Salah tier)
        elif ppg >= 7.0:
            fixture_mult *= 1.05
        elif ppg < 4:
            fixture_mult *= 0.90  # Historically poor

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

    captain_score = captain_base * fixture_mult * (1 + diff_boost)

    return {
        "captain_score": round(captain_score, 2),
        "ceiling_mult": round(fixture_mult, 3),  # Renamed from ceiling_mult to fixture_mult
        "diff_boost": round(diff_boost, 3),
        "ownership": round(ownership, 1),
        "base_ceiling": round(ceiling_xpts, 2),
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
                fpl_diff = fix.get("team_h_difficulty", 3)
                # Use our computed FDR with FPL fallback
                fdr = get_fixture_fdr(opponent_id, True, position_id, fpl_difficulty=fpl_diff)
                upcoming.append({
                    "gameweek": gw,
                    "opponent": teams_dict.get(opponent_id, {}).get("short_name", "???"),
                    "opponent_id": opponent_id,
                    "is_home": True,
                    "difficulty": fdr,  # Our computed FDR 1-10
                    "fdr": fdr,  # Alias for frontend compatibility
                    "fpl_difficulty": fpl_diff  # Original FPL 1-5
                })
            elif fix["team_a"] == player_team:
                opponent_id = fix["team_h"]
                fpl_diff = fix.get("team_a_difficulty", 3)
                fdr = get_fixture_fdr(opponent_id, False, position_id, fpl_difficulty=fpl_diff)
                upcoming.append({
                    "gameweek": gw,
                    "opponent": teams_dict.get(opponent_id, {}).get("short_name", "???"),
                    "opponent_id": opponent_id,
                    "is_home": False,
                    "difficulty": fdr,
                    "fdr": fdr,  # Alias for frontend compatibility
                    "fpl_difficulty": fpl_diff
                })
    return sorted(upcoming, key=lambda x: x["gameweek"])
