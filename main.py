"""
FPL Assistant Backend - FastAPI v2.1
Your personal FPL decision-making assistant with:
- Enhanced minutes modeling
- Bookmaker "to start" proxy using multiple signals
- Congestion-aware projections
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
from datetime import datetime, timedelta
import os
import re

app = FastAPI(title="FPL Assistant API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FPL_BASE_URL = "https://fantasy.premierleague.com/api"
MIN_MINUTES_DEFAULT = 400
POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_ID_MAP = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}

FDR_MULTIPLIERS = {1: 1.25, 2: 1.15, 3: 1.00, 4: 0.85, 5: 0.70}

# European competition teams
EUROPEAN_TEAMS_UCL = {"Arsenal", "Aston Villa", "Liverpool", "Man City"}
EUROPEAN_TEAMS_UEL = {"Man Utd", "Tottenham"}
EUROPEAN_TEAMS_UECL = {"Chelsea"}

# Manager rotation tendencies (lower = more rotation)
ROTATION_MANAGERS = {"Man City": 0.85, "Chelsea": 0.88, "Brighton": 0.90, "Newcastle": 0.92}
STABLE_MANAGERS = {"Arsenal": 1.0, "Liverpool": 0.98, "Fulham": 1.0, "Bournemouth": 1.0, "Brentford": 0.98}


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


class DataCache:
    def __init__(self):
        self.bootstrap_data: Optional[Dict] = None
        self.fixtures_data: Optional[List] = None
        self.element_summary_cache: Dict[int, Dict] = {}
        self.last_update: Optional[datetime] = None
        self.fixtures_last_update: Optional[datetime] = None
        self.cache_duration = 300

    def is_stale(self) -> bool:
        return self.last_update is None or (datetime.now() - self.last_update).seconds > self.cache_duration

    def fixtures_is_stale(self) -> bool:
        return self.fixtures_last_update is None or (datetime.now() - self.fixtures_last_update).seconds > self.cache_duration


cache = DataCache()


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
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"FPL API unavailable: {str(e)}")


async def fetch_player_history(player_id: int):
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


def parse_news_for_injury_info(news: str) -> dict:
    if not news:
        return {"has_news": False, "is_injury": False, "is_illness": False, "is_suspension": False}
    news_lower = news.lower()
    return {
        "has_news": True,
        "is_injury": any(w in news_lower for w in ["injury", "injured", "knock", "hamstring", "muscle", "ankle", "knee", "groin", "calf", "thigh", "back"]),
        "is_illness": any(w in news_lower for w in ["illness", "ill", "sick", "virus", "flu"]),
        "is_suspension": any(w in news_lower for w in ["suspend", "ban", "red card"]),
        "is_doubt": any(w in news_lower for w in ["doubt", "uncertain", "assess", "monitor"]),
    }


def calculate_bookmaker_start_proxy(
    player: Dict, team_name: str, current_gw: int, fixtures: List[Dict],
    events: List[Dict], player_history: Optional[Dict] = None
) -> tuple[float, str, dict]:
    """
    Bookmaker-style start probability proxy using:
    1. Recent starts (last 5) - most important
    2. Minutes per start (nailedness)
    3. Injury/availability status
    4. Manager rotation tendencies
    5. Fixture congestion
    """
    total_minutes = int(player.get("minutes", 0) or 0)
    starts = int(player.get("starts", 0) or 0)
    chance_of_playing = player.get("chance_of_playing_next_round")
    status = player.get("status", "a")
    news = player.get("news", "")

    breakdown = {
        "base_start_rate": 0, "recent_start_rate": 0, "nailedness_score": 0,
        "availability_modifier": 1.0, "rotation_modifier": 1.0, "congestion_modifier": 1.0,
        "news_analysis": parse_news_for_injury_info(news), "final_probability": 0,
    }

    # Unavailable check
    if status in ('i', 'u', 's'):
        prob = (chance_of_playing / 100 * 0.5) if (chance_of_playing and chance_of_playing > 0) else 0.05
        breakdown["final_probability"] = prob
        breakdown["availability_modifier"] = prob
        return prob, "unavailable", breakdown

    # Doubtful
    if status == 'd' or (chance_of_playing is not None and chance_of_playing < 75):
        breakdown["availability_modifier"] = (chance_of_playing / 100) if chance_of_playing else 0.5
    else:
        breakdown["availability_modifier"] = 1.0

    # Base start rate
    available_gws = max(current_gw - 1, 1)
    if starts == 0 or total_minutes < 90:
        breakdown["base_start_rate"] = 0.1
        breakdown["final_probability"] = 0.1 * breakdown["availability_modifier"]
        return breakdown["final_probability"], "insufficient_data", breakdown

    season_start_rate = min(1.0, starts / available_gws)
    breakdown["base_start_rate"] = season_start_rate

    # Recent form (last 5)
    recent_start_rate = season_start_rate
    if player_history and "history" in player_history:
        recent_games = player_history["history"][-5:]
        if recent_games:
            recent_starts = sum(1 for g in recent_games if g.get("minutes", 0) >= 60)
            recent_start_rate = recent_starts / len(recent_games)
            breakdown["recent_start_rate"] = recent_start_rate

    blended_start_rate = 0.6 * recent_start_rate + 0.4 * season_start_rate

    # Nailedness
    mins_per_start = total_minutes / starts if starts > 0 else 0
    if mins_per_start >= 85:
        nailedness = 1.0
    elif mins_per_start >= 75:
        nailedness = 0.95
    elif mins_per_start >= 65:
        nailedness = 0.85
    elif mins_per_start >= 55:
        nailedness = 0.70
    else:
        nailedness = 0.50
    breakdown["nailedness_score"] = nailedness

    # Manager rotation
    rotation_factor = ROTATION_MANAGERS.get(team_name, STABLE_MANAGERS.get(team_name, 0.95))
    breakdown["rotation_modifier"] = rotation_factor

    # Congestion
    team_id = player.get("team")
    upcoming_fixtures = [f for f in fixtures if f.get("event") and current_gw <= f["event"] <= current_gw + 3 and (f["team_h"] == team_id or f["team_a"] == team_id)]
    fixture_count = len(upcoming_fixtures)

    has_ucl = team_name in EUROPEAN_TEAMS_UCL
    has_uel = team_name in EUROPEAN_TEAMS_UEL

    if fixture_count >= 4 and (has_ucl or has_uel):
        congestion_mod = 0.85
    elif fixture_count >= 4:
        congestion_mod = 0.90
    elif has_ucl and blended_start_rate < 0.8:
        congestion_mod = 0.88
    elif has_uel and blended_start_rate < 0.8:
        congestion_mod = 0.90
    else:
        congestion_mod = 1.0

    if nailedness >= 0.95 and blended_start_rate >= 0.85:
        congestion_mod = 1.0 - (1.0 - congestion_mod) * 0.3
    breakdown["congestion_modifier"] = congestion_mod

    # Final
    base_prob = blended_start_rate * nailedness
    final_prob = max(0.05, min(0.98, base_prob * rotation_factor * congestion_mod * breakdown["availability_modifier"]))
    breakdown["final_probability"] = final_prob

    if final_prob >= 0.90:
        reason = "very_likely_starter"
    elif final_prob >= 0.75:
        reason = "likely_starter"
    elif final_prob >= 0.55:
        reason = "rotation_risk"
    elif final_prob >= 0.35:
        reason = "significant_rotation_risk"
    else:
        reason = "unlikely_starter"

    if breakdown["availability_modifier"] < 1.0:
        reason += "_doubtful"

    return final_prob, reason, breakdown


def calculate_expected_points(
    player: Dict, position: int, current_gw: int, upcoming_fixtures: List[Dict],
    teams_dict: Dict, all_fixtures: List[Dict] = None, events: List[Dict] = None,
    player_history: Optional[Dict] = None
) -> tuple[float, float, str, dict]:
    total_minutes = int(player.get("minutes", 0) or 0)
    xG = float(player.get("expected_goals", 0) or 0)
    xA = float(player.get("expected_assists", 0) or 0)
    xGC = float(player.get("expected_goals_conceded", 0) or 0)
    points_per_game = float(player.get("points_per_game", 0) or 0)

    team_id = player["team"]
    team_name = teams_dict.get(team_id, {}).get("name", "")

    start_prob, start_reason, breakdown = calculate_bookmaker_start_proxy(
        player, team_name, current_gw, all_fixtures or [], events or [], player_history
    )

    mins90 = max(total_minutes / 90.0, 0.1)
    xG90, xA90, xGC90 = xG / mins90, xA / mins90, xGC / mins90

    app_pts = 2.0
    cs_prob = math.exp(-max(0.0, xGC90))

    if position == 1:
        base90 = app_pts + 4.0 * cs_prob + 0.5
    elif position == 2:
        base90 = app_pts + 4.0 * cs_prob + (6.0 * xG90) + (3.0 * xA90)
    elif position == 3:
        base90 = app_pts + 1.0 * cs_prob + (5.0 * xG90) + (3.0 * xA90)
    else:
        base90 = app_pts + (4.0 * xG90) + (3.0 * xA90)

    base90 = 0.6 * base90 + 0.4 * points_per_game

    horizon = upcoming_fixtures[:8]
    total = sum(base90 * FDR_MULTIPLIERS.get(f.get("difficulty", 3), 1.0) for f in horizon) if horizon else base90

    return round(total * start_prob, 2), start_prob, start_reason, breakdown


def get_player_upcoming_fixtures(player_team: int, fixtures: List[Dict], current_gw: int, gw_end: int, teams_dict: Dict) -> List[Dict]:
    upcoming = []
    for fix in fixtures:
        gw = fix.get("event")
        if gw and current_gw <= gw <= gw_end:
            if fix["team_h"] == player_team:
                upcoming.append({"gameweek": gw, "opponent": teams_dict.get(fix["team_a"], {}).get("short_name", "???"), "is_home": True, "difficulty": fix.get("team_h_difficulty", 3)})
            elif fix["team_a"] == player_team:
                upcoming.append({"gameweek": gw, "opponent": teams_dict.get(fix["team_h"], {}).get("short_name", "???"), "is_home": False, "difficulty": fix.get("team_a_difficulty", 3)})
    return sorted(upcoming, key=lambda x: x["gameweek"])


# ============ ENDPOINTS ============

@app.get("/api/my-team/{manager_id}")
async def get_my_team(manager_id: int):
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
        player_history = await fetch_player_history(player["id"])
        upcoming = get_player_upcoming_fixtures(player["team"], fixtures, next_gw, next_gw + 5, teams)
        xpts, start_prob, start_reason, breakdown = calculate_expected_points(player, position_id, current_gw, upcoming, teams, fixtures, events, player_history)
        single_gw_fixtures = get_player_upcoming_fixtures(player["team"], fixtures, next_gw, next_gw, teams)
        single_xpts, _, _, _ = calculate_expected_points(player, position_id, current_gw, single_gw_fixtures, teams, fixtures, events, player_history)

        squad.append({
            "id": player["id"], "name": player["web_name"],
            "full_name": f"{player['first_name']} {player['second_name']}",
            "team": teams.get(player["team"], {}).get("short_name", "???"),
            "team_full": teams.get(player["team"], {}).get("name", "???"),
            "team_id": player["team"], "position": POSITION_MAP.get(position_id, "???"),
            "position_id": position_id, "price": player["now_cost"] / 10,
            "selling_price": pick.get("selling_price", player["now_cost"]) / 10,
            "multiplier": pick["multiplier"], "is_captain": pick["is_captain"],
            "is_vice_captain": pick["is_vice_captain"], "squad_position": pick["position"],
            "xpts": xpts, "xpts_single_gw": single_xpts,
            "form": float(player.get("form", 0) or 0),
            "total_points": player.get("total_points", 0),
            "start_probability": round(start_prob * 100, 1), "start_reason": start_reason,
            "ownership": float(player.get("selected_by_percent", 0) or 0),
            "fixtures": upcoming[:5], "news": player.get("news", ""),
            "status": player.get("status", "a"),
            "chance_of_playing": player.get("chance_of_playing_next_round")
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
        "current_gw": current_gw, "next_gw": next_gw, "squad": squad,
        "starting_xi": starting, "bench": bench,
        "captain_picks": sorted(starting, key=lambda x: x["xpts_single_gw"], reverse=True)[:3],
        "optimal_bench_order": sorted(bench, key=lambda x: x["xpts_single_gw"], reverse=True),
    }


@app.get("/api/my-team/{manager_id}/transfers")
async def get_transfer_suggestions(
    manager_id: int,
    horizon: int = Query(5, ge=1, le=10),
    max_price: float = Query(15.0, ge=4.0, le=15.0),
    min_price: float = Query(4.0, ge=4.0, le=15.0)
):
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
        xpts, start_prob, start_reason, _ = calculate_expected_points(player, position_id, current_gw, upcoming, teams, fixtures, events)
        selling_price = pick.get("selling_price", player["now_cost"]) / 10

        current_squad[player["id"]] = {
            "id": player["id"], "name": player["web_name"],
            "team": teams.get(player["team"], {}).get("short_name", "???"),
            "team_id": player["team"], "position": pos, "position_id": position_id,
            "price": player["now_cost"] / 10, "selling_price": selling_price,
            "xpts": xpts, "start_probability": round(start_prob * 100, 1),
            "form": float(player.get("form", 0) or 0), "fixtures": upcoming[:5],
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
                if price > available_budget or price > max_price or price < min_price:
                    continue
                player_team = player["team"]
                current_team_count = team_counts[player_team] - (1 if out_player["team_id"] == player_team else 0)
                if current_team_count >= 3 or player.get("minutes", 0) < 200:
                    continue

                upcoming = get_player_upcoming_fixtures(player["team"], fixtures, next_gw, next_gw + horizon, teams)
                xpts, start_prob, start_reason, _ = calculate_expected_points(player, pos_id, current_gw, upcoming, teams, fixtures, events)
                xpts_gain = xpts - out_player["xpts"]

                if xpts_gain > 0.3:
                    single_transfers.append({
                        "out": out_player,
                        "in": {
                            "id": player["id"], "name": player["web_name"],
                            "team": teams.get(player["team"], {}).get("short_name", "???"),
                            "team_id": player["team"], "position": pos, "price": price,
                            "xpts": xpts, "start_probability": round(start_prob * 100, 1),
                            "form": float(player.get("form", 0) or 0),
                            "ownership": float(player.get("selected_by_percent", 0) or 0),
                            "fixtures": upcoming[:5]
                        },
                        "xpts_gain": round(xpts_gain, 2),
                        "cost_change": round(price - out_player["selling_price"], 1),
                        "bank_after": round(available_budget - price, 1)
                    })

    single_transfers.sort(key=lambda x: x["xpts_gain"], reverse=True)

    double_transfers = []
    for first in single_transfers[:10]:
        for second in single_transfers:
            if second["out"]["id"] in [first["out"]["id"], first["in"]["id"]] or second["in"]["id"] in [first["out"]["id"], first["in"]["id"]]:
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
        "bank": bank, "horizon": f"GW{next_gw}-GW{next_gw + horizon}",
        "price_filter": {"min": min_price, "max": max_price},
        "best_single_transfers": single_transfers[:15],
        "best_double_transfers": double_transfers[:10],
    }


@app.get("/api/rankings/{position}")
async def get_rankings(
    position: Position,
    gw_start: int = Query(1, ge=1, le=38),
    gw_end: int = Query(8, ge=1, le=38),
    min_minutes: int = Query(MIN_MINUTES_DEFAULT, ge=0),
    min_price: float = Query(4.0, ge=3.5, le=15.0),
    max_price: float = Query(15.0, ge=3.5, le=15.0),
    sort_by: str = Query("xpts", regex="^(xpts|price|form|ownership|start_probability|total_points)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    limit: int = Query(50, ge=1, le=200)
):
    if gw_end < gw_start:
        raise HTTPException(status_code=400, detail="gw_end must be >= gw_start")

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

        starts = player.get("starts", 0)
        upcoming = get_player_upcoming_fixtures(player["team"], fixtures, gw_start, gw_end, teams)
        xpts, start_prob, start_reason, _ = calculate_expected_points(player, position_id, current_gw, upcoming, teams, fixtures, events)

        ranked_players.append({
            "id": player["id"], "name": player["web_name"],
            "full_name": f"{player['first_name']} {player['second_name']}",
            "team": teams.get(player["team"], {}).get("short_name", "???"),
            "position": position.value, "price": price, "xpts": xpts,
            "form": float(player.get("form", 0) or 0),
            "total_points": player.get("total_points", 0),
            "mins_per_gw": round(total_minutes / max(current_gw - 1, 1), 1),
            "ownership": float(player.get("selected_by_percent", 0) or 0),
            "xG": float(player.get("expected_goals", 0) or 0),
            "xA": float(player.get("expected_assists", 0) or 0),
            "fixtures": upcoming[:8], "news": player.get("news", ""),
            "status": player.get("status", "a"),
            "start_probability": round(start_prob * 100, 1), "start_reason": start_reason
        })

    ranked_players.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=(sort_order == "desc"))

    return {
        "position": position.value,
        "gw_range": f"GW{gw_start}-GW{gw_end}",
        "current_gw": current_gw,
        "total_players": len(ranked_players),
        "filters": {"min_minutes": min_minutes, "min_price": min_price, "max_price": max_price, "sort_by": sort_by},
        "players": ranked_players[:limit]
    }


@app.get("/api/fixture-ratings")
async def get_fixture_ratings(horizon: int = Query(8, ge=1, le=15)):
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)

    team_ratings = []
    for team_id, team in teams.items():
        upcoming = []
        total_difficulty = 0
        for fix in fixtures:
            gw = fix.get("event")
            if gw and current_gw <= gw < current_gw + horizon:
                if fix["team_h"] == team_id:
                    upcoming.append({"gameweek": gw, "opponent": teams.get(fix["team_a"], {}).get("short_name", "???"), "is_home": True, "difficulty": fix.get("team_h_difficulty", 3)})
                    total_difficulty += fix.get("team_h_difficulty", 3)
                elif fix["team_a"] == team_id:
                    upcoming.append({"gameweek": gw, "opponent": teams.get(fix["team_h"], {}).get("short_name", "???"), "is_home": False, "difficulty": fix.get("team_a_difficulty", 3)})
                    total_difficulty += fix.get("team_a_difficulty", 3)

        upcoming.sort(key=lambda x: x["gameweek"])
        avg_difficulty = total_difficulty / len(upcoming) if upcoming else 3
        rating = "EASY" if avg_difficulty <= 2.2 else ("MEDIUM" if avg_difficulty <= 2.8 else "HARD")

        team_ratings.append({
            "team_id": team_id, "team_name": team["name"], "team_short": team["short_name"],
            "avg_difficulty": round(avg_difficulty, 2), "rating": rating, "fixtures": upcoming,
            "easy_fixtures": len([f for f in upcoming if f["difficulty"] <= 2]),
            "hard_fixtures": len([f for f in upcoming if f["difficulty"] >= 4]),
            "rotation_factor": ROTATION_MANAGERS.get(team["name"], STABLE_MANAGERS.get(team["name"], 0.95))
        })

    team_ratings.sort(key=lambda x: x["avg_difficulty"])
    return {"horizon": horizon, "current_gw": current_gw, "teams": team_ratings}


@app.get("/api/differentials")
async def get_differentials(
    max_ownership: float = Query(10.0, ge=0, le=100),
    min_form: float = Query(4.0, ge=0),
    min_minutes: int = Query(MIN_MINUTES_DEFAULT, ge=0),
    min_price: float = Query(4.0, ge=3.5, le=15.0),
    max_price: float = Query(15.0, ge=3.5, le=15.0),
    sort_by: str = Query("diff_score", regex="^(diff_score|xpts|form|price|ownership)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    limit: int = Query(30, ge=1, le=100)
):
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
        mins_per_gw = total_minutes / max(current_gw - 1, 1)

        if ownership > max_ownership or form < min_form or total_minutes < min_minutes or price < min_price or price > max_price or mins_per_gw < 60:
            continue

        position_id = player["element_type"]
        upcoming = get_player_upcoming_fixtures(player["team"], fixtures, current_gw, current_gw + 5, teams)
        xpts, start_prob, start_reason, _ = calculate_expected_points(player, position_id, current_gw, upcoming, teams, fixtures, events)
        diff_score = (form * 0.4 + xpts * 0.6) * (1 + (max_ownership - ownership) / max_ownership)

        differentials.append({
            "id": player["id"], "name": player["web_name"],
            "team": teams.get(player["team"], {}).get("short_name", "???"),
            "position": POSITION_MAP.get(position_id, "???"),
            "price": price, "ownership": ownership, "form": form, "xpts": xpts,
            "total_points": player.get("total_points", 0),
            "diff_score": round(diff_score, 2), "fixtures": upcoming[:5],
            "start_probability": round(start_prob * 100, 1),
        })

    differentials.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=(sort_order == "desc"))
    return {"max_ownership": max_ownership, "current_gw": current_gw, "total_found": len(differentials), "players": differentials[:limit]}


@app.get("/api/price-changes")
async def get_price_changes(limit: int = Query(40, ge=1, le=100)):
    data = await fetch_fpl_data()
    elements = data["elements"]
    teams = {t["id"]: t for t in data["teams"]}

    rising, falling = [], []
    for player in elements:
        net = player.get("transfers_in_event", 0) - player.get("transfers_out_event", 0)
        owners = int(float(player.get("selected_by_percent", 0)) * 100000)
        if owners == 0:
            continue
        ratio = net / max(owners, 1)
        pdata = {
            "id": player["id"], "name": player["web_name"],
            "team": teams.get(player["team"], {}).get("short_name", "???"),
            "position": POSITION_MAP.get(player["element_type"], "???"),
            "price": player["now_cost"] / 10, "net_transfers": net,
            "likelihood": min(abs(ratio * 1000), 100)
        }
        if net > 5000:
            rising.append(pdata)
        elif net < -5000:
            falling.append(pdata)

    rising.sort(key=lambda x: x["net_transfers"], reverse=True)
    falling.sort(key=lambda x: x["net_transfers"])
    return {"rising": rising[:limit], "falling": falling[:limit]}


try:
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False


@app.post("/api/optimize-team")
async def optimize_team(request: TeamOptimizeRequest):
    if not OPTIMIZER_AVAILABLE:
        raise HTTPException(status_code=501, detail="Optimizer not available. Install PuLP.")

    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()
    elements = data["elements"]
    teams = {t["id"]: t for t in data["teams"]}
    events = data["events"]
    current_gw = get_current_gameweek(events)

    formation_parts = request.formation.value.split("-")
    formation_req = {"GKP": 1, "DEF": int(formation_parts[0]), "MID": int(formation_parts[1]), "FWD": int(formation_parts[2])}

    valid_players = []
    for player in elements:
        if player["id"] in request.excluded_players or player.get("minutes", 0) < request.min_minutes or player.get("starts", 0) == 0:
            continue
        chance = player.get("chance_of_playing_next_round")
        if chance is not None and chance < 50:
            continue

        position_id = player["element_type"]
        upcoming = get_player_upcoming_fixtures(player["team"], fixtures, request.gw_start, request.gw_end, teams)
        xpts, start_prob, start_reason, _ = calculate_expected_points(player, position_id, current_gw, upcoming, teams, fixtures, events)
        if xpts < 1.5:
            continue

        valid_players.append({
            "id": player["id"], "name": player["web_name"],
            "team": teams.get(player["team"], {}).get("short_name", "???"),
            "team_id": player["team"], "position": POSITION_MAP.get(position_id, "???"),
            "position_id": position_id, "price": player["now_cost"] / 10, "xpts": xpts,
            "form": float(player.get("form", 0) or 0),
            "start_probability": round(start_prob * 100, 1),
        })

    prob = LpProblem("FPL_Optimizer", LpMaximize)
    pvars = {p["id"]: LpVariable(f"p_{p['id']}", cat="Binary") for p in valid_players}
    prob += lpSum(pvars[p["id"]] * p["xpts"] for p in valid_players)
    prob += lpSum(pvars[p["id"]] * p["price"] for p in valid_players) <= request.budget

    for pos, count in formation_req.items():
        pos_id = POSITION_ID_MAP[pos]
        prob += lpSum(pvars[p["id"]] for p in valid_players if p["position_id"] == pos_id) == count

    for team_id in set(p["team_id"] for p in valid_players):
        prob += lpSum(pvars[p["id"]] for p in valid_players if p["team_id"] == team_id) <= 3

    for req_id in request.required_players:
        if req_id in pvars:
            prob += pvars[req_id] == 1

    prob.solve()
    if LpStatus[prob.status] != "Optimal":
        raise HTTPException(status_code=400, detail=f"No optimal solution: {LpStatus[prob.status]}")

    selected = [p for p in valid_players if pvars[p["id"]].varValue == 1]
    total_xpts = sum(p["xpts"] for p in selected)
    total_cost = sum(p["price"] for p in selected)
    selected.sort(key=lambda x: ({"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}.get(x["position"], 4), -x["xpts"]))

    return {
        "formation": request.formation.value,
        "budget_used": round(total_cost, 1),
        "budget_remaining": round(request.budget - total_cost, 1),
        "total_xpts": round(total_xpts, 2),
        "gw_range": f"GW{request.gw_start}-GW{request.gw_end}",
        "players": selected,
        "by_position": {pos: [p for p in selected if p["position"] == pos] for pos in ["GKP", "DEF", "MID", "FWD"]}
    }


@app.get("/")
async def root():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(path) if os.path.exists(path) else {"message": "FPL Assistant API", "docs": "/docs"}


@app.get("/api/bootstrap")
async def get_bootstrap():
    return await fetch_fpl_data()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
