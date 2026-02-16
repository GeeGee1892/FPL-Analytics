"""
FPL Assistant - Transfer Planner Module

Transfer planning logic including chip management, squad evaluation,
transfer candidates, fixture swings, and multi-GW strategy building.
Extracted from main.py (~lines 6560-8059).
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from fpl_assistant.constants import POSITION_MAP, POSITION_ID_MAP
from fpl_assistant.models import (
    SquadHealthIssue,
    FixtureSwing,
    TransferRecommendation,
    ChipRecommendation,
    ChipPlacement,
    StrategyPlan,
    TransferPlannerConfig,
    BookedTransfer,
    TransferPlannerRequest,
)

import logging
try:
    import pulp
except ImportError:
    pulp = None

# Import service/calculator functions - try dedicated modules first, fall back to main
try:
    from fpl_assistant.services import (
        calculate_expected_points,
        get_player_upcoming_fixtures,
    )
except ImportError:
    from main import (
        calculate_expected_points,
        get_player_upcoming_fixtures,
    )

try:
    from fpl_assistant.services import get_fixture_fdr
except ImportError:
    from main import get_fixture_fdr


# ============ TRANSFER PLANNER / SOLVER ============

ALL_CHIPS = {"wildcard", "freehit", "bboost", "3xc"}  # FPL API chip names
CHIP_DISPLAY = {"wildcard": "WC", "freehit": "FH", "bboost": "BB", "3xc": "TC"}


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


# ============ CHIP SIMULATION ============

def build_optimal_squad(
    elements: List[Dict],
    fixtures: List[Dict],
    teams_dict: Dict,
    events: List[Dict],
    target_gw: int,
    horizon: int,
    budget: float,
    current_squad_ids: set = None,
    single_gw: bool = False,
    player_gw_cache: Dict = None,
) -> List[Dict]:
    """
    Build optimal 15-player squad using PuLP ILP solver.
    Used by Wildcard (multi-GW) and Free Hit (single GW).

    Returns list of player dicts in the same format as squad entries.
    Falls back to greedy if PuLP unavailable.
    """
    if player_gw_cache is None:
        player_gw_cache = {}

    horizon_end = min(target_gw + horizon - 1, 38)
    eval_horizon = 1 if single_gw else horizon

    # Filter eligible players
    eligible = []
    for p in elements:
        if p.get("status", "a") in ("i", "s", "u"):
            continue
        if p.get("minutes", 0) < 200:
            continue
        eligible.append(p)

    # Calculate xPts for each eligible player
    candidates = []
    for p in eligible:
        pid = p["id"]
        pos = p["element_type"]
        price = p["now_cost"] / 10

        # Sum xPts over the evaluation horizon
        total_xpts = 0
        for gw in range(target_gw, target_gw + eval_horizon):
            if gw > 38:
                break
            cache_key = (pid, gw)
            if cache_key in player_gw_cache:
                total_xpts += player_gw_cache[cache_key]
            else:
                upcoming = get_player_upcoming_fixtures(p["team"], fixtures, gw, gw, teams_dict)
                if upcoming:
                    stats = calculate_expected_points(
                        p, pos, gw, upcoming, teams_dict, fixtures, events
                    )
                    gw_xpts = stats["xpts"]
                else:
                    gw_xpts = 0
                player_gw_cache[cache_key] = gw_xpts
                total_xpts += gw_xpts

        candidates.append({
            "element": p,
            "id": pid,
            "position_id": pos,
            "team": p["team"],
            "price": price,
            "xpts": total_xpts,
        })

    # Pre-filter to top 50 per position for performance
    by_pos = defaultdict(list)
    for c in candidates:
        by_pos[c["position_id"]].append(c)
    filtered = []
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: -x["xpts"])
        filtered.extend(by_pos[pos][:50])

    if pulp is None:
        # Fallback: greedy selection
        return _greedy_squad(filtered, budget, teams_dict, current_squad_ids)

    # PuLP ILP solver
    prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)

    x = {c["id"]: pulp.LpVariable(f"x_{c['id']}", cat="Binary") for c in filtered}

    # Objective: maximize total xPts (with small bias to retain current squad)
    retain_bonus = 0.1  # small tiebreaker bonus for keeping existing players
    prob += pulp.lpSum(
        x[c["id"]] * (c["xpts"] + (retain_bonus if current_squad_ids and c["id"] in current_squad_ids else 0))
        for c in filtered
    )

    # Exactly 15 players
    prob += pulp.lpSum(x[c["id"]] for c in filtered) == 15

    # Position constraints: 2 GKP, 5 DEF, 5 MID, 3 FWD
    for pos_id, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        pos_cands = [c for c in filtered if c["position_id"] == pos_id]
        prob += pulp.lpSum(x[c["id"]] for c in pos_cands) == count

    # Team constraint: max 3 per team
    all_teams = set(c["team"] for c in filtered)
    for team_id in all_teams:
        team_cands = [c for c in filtered if c["team"] == team_id]
        prob += pulp.lpSum(x[c["id"]] for c in team_cands) <= 3

    # Budget constraint
    prob += pulp.lpSum(x[c["id"]] * c["price"] for c in filtered) <= budget

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != 1:
        # Infeasible — fallback to greedy
        logging.warning("PuLP squad optimization infeasible, falling back to greedy")
        return _greedy_squad(filtered, budget, teams_dict, current_squad_ids)

    # Extract selected players
    selected = []
    for c in filtered:
        if x[c["id"]].varValue and x[c["id"]].varValue > 0.5:
            p = c["element"]
            selected.append({
                "id": p["id"],
                "name": p.get("web_name", "Unknown"),
                "team": teams_dict.get(p["team"], {}).get("short_name", "???"),
                "team_id": p["team"],
                "position": POSITION_MAP.get(p["element_type"], "MID"),
                "position_id": p["element_type"],
                "price": c["price"],
                "selling_price": c["price"],
                "xpts": c["xpts"],
                "form": float(p.get("form", 0) or 0),
                "expected_minutes": 90,
            })

    return selected


def _greedy_squad(
    candidates: List[Dict],
    budget: float,
    teams_dict: Dict,
    current_squad_ids: set = None,
) -> List[Dict]:
    """Greedy fallback for squad building when PuLP is unavailable."""
    POSITION_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}
    selected = []
    team_counts = defaultdict(int)
    remaining_budget = budget

    # Sort by xPts/price ratio
    sorted_cands = sorted(candidates, key=lambda x: -x["xpts"] / max(x["price"], 0.1))

    for pos_id, count in POSITION_COUNTS.items():
        pos_cands = [c for c in sorted_cands if c["position_id"] == pos_id]
        picked = 0
        for c in pos_cands:
            if picked >= count:
                break
            if c["price"] > remaining_budget:
                continue
            if team_counts[c["team"]] >= 3:
                continue
            p = c["element"]
            selected.append({
                "id": p["id"],
                "name": p.get("web_name", "Unknown"),
                "team": teams_dict.get(p["team"], {}).get("short_name", "???"),
                "team_id": p["team"],
                "position": POSITION_MAP.get(p["element_type"], "MID"),
                "position_id": p["element_type"],
                "price": c["price"],
                "selling_price": c["price"],
                "xpts": c["xpts"],
                "form": float(p.get("form", 0) or 0),
                "expected_minutes": 90,
            })
            remaining_budget -= c["price"]
            team_counts[c["team"]] += 1
            picked += 1

    return selected


def calculate_chip_marginal_value(
    chip: str,
    gw: int,
    squad: List[Dict],
    bank: float,
    fixtures: List[Dict],
    teams_dict: Dict,
    elements: List[Dict],
    elements_dict: Dict,
    events: List[Dict],
    horizon_end: int,
    player_gw_cache: Dict,
):
    """
    Calculate the marginal xPts of activating a chip on a specific GW.
    Returns (marginal_xpts, new_squad_or_None).
    """
    remaining_horizon = horizon_end - gw + 1

    if chip == "bboost":
        # BB: all 15 score vs best XI
        xpts_with = evaluate_squad_xpts(
            squad, gw, fixtures, teams_dict, elements_dict, events,
            chip="bboost", player_gw_cache=player_gw_cache,
        )
        xpts_without = evaluate_squad_xpts(
            squad, gw, fixtures, teams_dict, elements_dict, events,
            chip=None, player_gw_cache=player_gw_cache,
        )
        return xpts_with - xpts_without, None

    elif chip == "3xc":
        # TC: captain scores 3x vs 2x
        xpts_with = evaluate_squad_xpts(
            squad, gw, fixtures, teams_dict, elements_dict, events,
            chip="3xc", player_gw_cache=player_gw_cache,
        )
        xpts_without = evaluate_squad_xpts(
            squad, gw, fixtures, teams_dict, elements_dict, events,
            chip=None, player_gw_cache=player_gw_cache,
        )
        return xpts_with - xpts_without, None

    elif chip == "freehit":
        # FH: optimal squad for this one GW vs current
        total_budget = bank + sum(p.get("selling_price", p.get("price", 0)) for p in squad)
        fh_squad = build_optimal_squad(
            elements, fixtures, teams_dict, events,
            target_gw=gw, horizon=1, budget=total_budget,
            single_gw=True, player_gw_cache=player_gw_cache,
        )
        if not fh_squad:
            return 0.0, None
        xpts_fh = evaluate_squad_xpts(
            fh_squad, gw, fixtures, teams_dict, elements_dict, events,
            player_gw_cache=player_gw_cache,
        )
        xpts_current = evaluate_squad_xpts(
            squad, gw, fixtures, teams_dict, elements_dict, events,
            player_gw_cache=player_gw_cache,
        )
        return xpts_fh - xpts_current, fh_squad

    elif chip == "wildcard":
        # WC: rebuild squad for remaining horizon
        total_budget = bank + sum(p.get("selling_price", p.get("price", 0)) for p in squad)
        wc_squad = build_optimal_squad(
            elements, fixtures, teams_dict, events,
            target_gw=gw, horizon=remaining_horizon, budget=total_budget,
            current_squad_ids={p["id"] for p in squad},
            player_gw_cache=player_gw_cache,
        )
        if not wc_squad:
            return 0.0, None

        # Compare total remaining xPts with new squad vs current
        wc_total = 0
        current_total = 0
        for eval_gw in range(gw, horizon_end + 1):
            wc_total += evaluate_squad_xpts(
                wc_squad, eval_gw, fixtures, teams_dict, elements_dict, events,
                player_gw_cache=player_gw_cache,
            )
            current_total += evaluate_squad_xpts(
                squad, eval_gw, fixtures, teams_dict, elements_dict, events,
                player_gw_cache=player_gw_cache,
            )
        return wc_total - current_total, wc_squad

    return 0.0, None


def determine_chip_placements(
    available_chips: Dict[str, bool],
    chip_recommendations: List[ChipRecommendation],
    squad: List[Dict],
    bank: float,
    fixtures: List[Dict],
    events: List[Dict],
    teams_dict: Dict,
    elements: List[Dict],
    elements_dict: Dict,
    current_gw: int,
    horizon_end: int,
    strategy_config: Dict,
    player_gw_cache: Dict,
    chip_overrides: List[Dict] = None,
) -> Dict[int, str]:
    """
    Determine which chips to place on which GWs.
    Evaluates in priority order: WC → FH → BB → TC.
    Supports user overrides: "lock" forces a chip to a specific GW,
    "block" prevents a chip from being placed at all.
    Returns {gw: chip_name}.
    """
    min_marginal = strategy_config.get("min_chip_marginal_xpts", 2.5)
    placements = {}  # gw -> chip_name
    placement_details = []  # List[ChipPlacement]
    working_squad = [p.copy() for p in squad]
    working_bank = bank

    # Parse chip overrides
    locked = {}   # chip -> gw (force placement)
    blocked = set()  # chips to skip entirely
    if chip_overrides:
        for ov in chip_overrides:
            chip_id = ov.get("chip", "") if isinstance(ov, dict) else getattr(ov, "chip", "")
            action = ov.get("action", "") if isinstance(ov, dict) else getattr(ov, "action", "")
            gw = ov.get("gw") if isinstance(ov, dict) else getattr(ov, "gw", None)
            if action == "block":
                blocked.add(chip_id)
            elif action == "lock" and gw is not None:
                locked[chip_id] = gw

    # Build recommendation lookup: chip -> [recommended GWs]
    rec_gws = defaultdict(list)
    for rec in chip_recommendations:
        rec_gws[rec.chip].append(rec.recommended_gw)

    # Process in priority order
    for chip in ["wildcard", "freehit", "bboost", "3xc"]:
        if not available_chips.get(chip, False):
            continue

        # Skip blocked chips
        if chip in blocked:
            continue

        # Handle locked chips: force to specific GW
        if chip in locked:
            lock_gw = locked[chip]
            if lock_gw in placements:
                continue  # GW already taken by higher-priority chip
            if lock_gw < current_gw or lock_gw > horizon_end:
                continue  # Out of horizon

            marginal, new_squad = calculate_chip_marginal_value(
                chip, lock_gw, working_squad, working_bank,
                fixtures, teams_dict, elements, elements_dict,
                events, horizon_end, player_gw_cache,
            )
            # Lock always places (skip threshold check)
            placements[lock_gw] = chip
            rec = next((r for r in chip_recommendations if r.chip == chip and r.recommended_gw == lock_gw), None)
            reason = rec.reason if rec else (f"+{marginal:.1f} xPts (locked)" if marginal > 0 else "Locked by user")
            placement_details.append(ChipPlacement(
                chip=chip, gw=lock_gw, marginal_xpts=marginal, reason=reason,
            ))

            if chip == "wildcard" and new_squad:
                total_cost = sum(p.get("price", 0) for p in new_squad)
                total_budget = working_bank + sum(
                    p.get("selling_price", p.get("price", 0)) for p in working_squad
                )
                working_squad = new_squad
                working_bank = total_budget - total_cost
            continue

        # Auto placement: evaluate candidates
        candidate_gws = rec_gws.get(chip, [])
        if not candidate_gws:
            candidate_gws = list(range(current_gw, horizon_end + 1))

        # Filter to GWs not already taken by another chip
        candidate_gws = [gw for gw in candidate_gws if gw not in placements]

        if not candidate_gws:
            continue

        best_gw = None
        best_marginal = 0
        best_squad = None
        best_reason = ""

        for gw in candidate_gws:
            marginal, new_squad = calculate_chip_marginal_value(
                chip, gw, working_squad, working_bank,
                fixtures, teams_dict, elements, elements_dict,
                events, horizon_end, player_gw_cache,
            )
            if marginal > best_marginal:
                best_marginal = marginal
                best_gw = gw
                best_squad = new_squad
                # Get reason from recommendation if available
                rec = next((r for r in chip_recommendations if r.chip == chip and r.recommended_gw == gw), None)
                best_reason = rec.reason if rec else f"+{marginal:.1f} xPts"

        if best_gw is not None and best_marginal >= min_marginal:
            placements[best_gw] = chip
            placement_details.append(ChipPlacement(
                chip=chip, gw=best_gw, marginal_xpts=best_marginal, reason=best_reason,
            ))

            # If WC placed, update working squad for subsequent evaluations
            if chip == "wildcard" and best_squad:
                total_cost = sum(p.get("price", 0) for p in best_squad)
                total_budget = working_bank + sum(
                    p.get("selling_price", p.get("price", 0)) for p in working_squad
                )
                working_squad = best_squad
                working_bank = total_budget - total_cost

    return placements, placement_details


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

                # Minimum minutes filter - need decent sample to trust projection
                # 400 mins = ~4-5 full games minimum
                total_minutes = player.get("minutes", 0)
                if total_minutes < 400:
                    continue

                # Check availability status - skip injured/doubtful
                status = player.get("status", "a")
                if status in ["i", "s", "u"]:  # injured, suspended, unavailable
                    continue

                # Calculate xPts over horizon
                upcoming = get_player_upcoming_fixtures(
                    player["team"], fixtures, current_gw, current_gw + horizon, teams_dict
                )
                stats = calculate_expected_points(
                    player, pos_id, current_gw, upcoming, teams_dict, fixtures, []
                )

                # Skip players with low expected minutes (rotation risks)
                # Expected minutes < 60 per game suggests heavy rotation
                exp_mins = stats.get("expected_minutes", 0)
                if exp_mins < 60:
                    continue

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
        self.squad = [p.copy() for p in squad]
        self.bank = bank
        self.ft = free_transfers
        self.initial_ft = free_transfers
        self.actions = []  # List of (gw, action_type, details)
        self.total_xpts = 0
        self.hits = 0
        self.gw_xpts = {}  # Cache: gw -> xpts (for incremental updates)
        self.transferred_out_ids = set()
        self.transferred_in_ids = set()
        self.pre_fh_squad = None
        self.pre_fh_bank = None

    def copy(self):
        new_path = TransferPath.__new__(TransferPath)
        new_path.squad = [p.copy() for p in self.squad]
        new_path.bank = self.bank
        new_path.ft = self.ft
        new_path.initial_ft = self.initial_ft
        new_path.actions = list(self.actions)
        new_path.total_xpts = self.total_xpts
        new_path.hits = self.hits
        new_path.gw_xpts = dict(self.gw_xpts)
        new_path.transferred_out_ids = set(self.transferred_out_ids)
        new_path.transferred_in_ids = set(self.transferred_in_ids)
        new_path.pre_fh_squad = [p.copy() for p in self.pre_fh_squad] if self.pre_fh_squad else None
        new_path.pre_fh_bank = self.pre_fh_bank
        return new_path

    def squad_key(self) -> tuple:
        """Hashable state for deduplication."""
        ids = frozenset(p["id"] for p in self.squad)
        return (ids, round(self.bank * 2) / 2, self.ft)

    def score(self) -> float:
        """Net score for beam ranking (total_xpts already includes -4 per hit)."""
        return self.total_xpts

    def apply_transfer(self, transfer: Dict, gw: int, is_hit: bool = False):
        """Apply a transfer to the squad."""
        out_id = transfer["out"]["id"]
        self.squad = [p for p in self.squad if p["id"] != out_id]
        in_p = transfer["in"]
        self.squad.append({
            "id": in_p["id"],
            "name": in_p.get("name", "Unknown"),
            "team": in_p.get("team", "???"),
            "team_id": in_p.get("team_id", 0),
            "position": in_p.get("position", "MID"),
            "position_id": in_p.get("position_id", 3),
            "price": in_p.get("price", 0),
            "selling_price": in_p.get("price", 0),
            "xpts": in_p.get("xpts", 0),
            "form": in_p.get("form", 0),
            "expected_minutes": in_p.get("expected_minutes", 90),
        })
        self.bank += transfer["out"].get("selling_price", transfer["out"].get("price", 0)) - in_p.get("price", 0)

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


def candidate_to_transfer_dict(candidate) -> Dict:
    """Convert a TransferRecommendation to the dict format TransferPath.apply_transfer() expects."""
    return {
        "out": candidate.out_player,
        "in": candidate.in_player,
        "xpts_gain": candidate.xpts_gain,
        "reason": candidate.reason,
        "is_booked": False,
    }


def generate_compatible_pairs(
    candidates: list,
    max_pairs: int,
) -> List[tuple]:
    """
    Generate compatible transfer pairs from single candidates.
    Two transfers are compatible if they don't share out/in player or same incoming team.
    Returns top-N pairs ranked by combined xpts_gain.
    """
    pairs = []
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i + 1:]:
            if c1.out_player["id"] == c2.out_player["id"]:
                continue
            if c1.in_player["id"] == c2.in_player["id"]:
                continue
            if c1.in_player.get("team_id") == c2.in_player.get("team_id"):
                continue
            pairs.append((c1, c2, c1.xpts_gain + c2.xpts_gain))
    pairs.sort(key=lambda x: -x[2])
    return [(p[0], p[1]) for p in pairs[:max_pairs]]


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


# ============ COMPREHENSIVE TRANSFER PLANNER ============

def analyze_squad_health(
    squad: List[Dict],
    fixtures: List[Dict],
    events: List[Dict],
    teams_dict: Dict,
    elements_dict: Dict,
    current_gw: int,
    horizon: int = 8
) -> List[SquadHealthIssue]:
    """
    Analyze squad for health issues.

    Returns list of issues sorted by severity.
    """
    issues = []
    horizon_end = min(current_gw + horizon, 38)

    # Build team fixture counts for BGW detection
    team_gw_fixtures = defaultdict(lambda: defaultdict(int))
    for fix in fixtures:
        gw = fix.get("event")
        if gw and current_gw <= gw <= horizon_end:
            team_gw_fixtures[fix["team_h"]][gw] += 1
            team_gw_fixtures[fix["team_a"]][gw] += 1

    for player in squad:
        player_id = player["id"]
        name = player.get("name", player.get("web_name", "Unknown"))
        team = player.get("team", "???")
        team_id = player.get("team_id", 0)
        position = player.get("position", "???")

        # Get full player data for status check
        full_player = elements_dict.get(player_id, {})
        status = full_player.get("status", player.get("status", "a"))
        news = full_player.get("news", player.get("news", ""))
        chance_playing = full_player.get("chance_of_playing_next_round")

        # 1. INJURY FLAGS
        if status == "i":  # Injured
            issues.append(SquadHealthIssue(
                player_id=player_id,
                player_name=name,
                team=team,
                position=position,
                issue_type="injury",
                severity="critical",
                description=f"Injured: {news}" if news else "Injured - no return date",
                affected_gws=list(range(current_gw, horizon_end + 1)),
                recommendation=f"Transfer out immediately - 0 xPts while injured"
            ))
        elif status == "d":  # Doubtful
            severity = "critical" if chance_playing and chance_playing <= 25 else "warning"
            issues.append(SquadHealthIssue(
                player_id=player_id,
                player_name=name,
                team=team,
                position=position,
                issue_type="injury",
                severity=severity,
                description=f"Doubtful ({chance_playing}%): {news}" if news else f"Doubtful ({chance_playing}%)",
                affected_gws=[current_gw],
                recommendation=f"Monitor closely - {chance_playing}% chance of playing"
            ))
        elif status == "s":  # Suspended
            issues.append(SquadHealthIssue(
                player_id=player_id,
                player_name=name,
                team=team,
                position=position,
                issue_type="injury",
                severity="critical",
                description=f"Suspended: {news}" if news else "Suspended",
                affected_gws=[current_gw],
                recommendation="Check suspension length - may need transfer"
            ))

        # 2. BGW EXPOSURE
        bgw_gws = []
        for gw in range(current_gw, horizon_end + 1):
            if team_gw_fixtures[team_id][gw] == 0:
                bgw_gws.append(gw)

        if bgw_gws:
            severity = "critical" if len(bgw_gws) >= 2 or bgw_gws[0] == current_gw else "warning"
            issues.append(SquadHealthIssue(
                player_id=player_id,
                player_name=name,
                team=team,
                position=position,
                issue_type="bgw_exposure",
                severity=severity,
                description=f"Blank in GW{', GW'.join(map(str, bgw_gws))}",
                affected_gws=bgw_gws,
                recommendation="Consider FH if many blanks, or transfer for cover"
            ))

        # 3. ROTATION RISK (based on minutes pattern)
        expected_mins = player.get("expected_minutes", 90)
        mins_reason = player.get("minutes_reason", "")

        if expected_mins < 60:
            issues.append(SquadHealthIssue(
                player_id=player_id,
                player_name=name,
                team=team,
                position=position,
                issue_type="rotation",
                severity="warning",
                description=f"Rotation risk - {expected_mins:.0f} expected mins ({mins_reason})",
                affected_gws=list(range(current_gw, horizon_end + 1)),
                recommendation="Consider upgrade to nailed starter"
            ))

        # 4. FORM CONCERNS
        form = float(player.get("form", 0) or 0)
        xpts = player.get("xpts", 0)

        # If form significantly underperforms xPts projection, flag it
        if form < 3.0 and xpts > 4.0:
            issues.append(SquadHealthIssue(
                player_id=player_id,
                player_name=name,
                team=team,
                position=position,
                issue_type="form_concern",
                severity="minor",
                description=f"Form ({form:.1f}) below expected ({xpts:.1f} xPts)",
                affected_gws=[],
                recommendation="May be due to bad luck (xG > G) - monitor but don't panic"
            ))

    # Sort by severity: critical > warning > minor
    severity_order = {"critical": 0, "warning": 1, "minor": 2}
    issues.sort(key=lambda x: severity_order.get(x.severity, 3))

    return issues


def calculate_fixture_swings(
    fixtures: List[Dict],
    teams_dict: Dict,
    current_gw: int,
    horizon: int = 8
) -> List[FixtureSwing]:
    """
    Calculate fixture swings for all teams.

    Uses actual FDR values to identify teams with improving or worsening fixtures.
    Sorted by average FDR across horizon (lowest = easiest fixtures = top).
    """
    swings = []
    horizon_end = min(current_gw + horizon - 1, 38)

    # Split horizon into "now" (first half) and "later" (second half)
    mid_point = current_gw + (horizon // 2)

    for team_id, team in teams_dict.items():
        now_fdrs = []
        later_fdrs = []
        all_fdrs = []
        dgws = []
        bgws = []

        # Count fixtures per GW for this team
        gw_fixture_count = defaultdict(int)

        for fix in fixtures:
            gw = fix.get("event")
            if not gw:
                continue

            is_home = fix["team_h"] == team_id
            is_away = fix["team_a"] == team_id

            if not (is_home or is_away):
                continue

            if current_gw <= gw <= horizon_end:
                gw_fixture_count[gw] += 1

            opponent_id = fix["team_a"] if is_home else fix["team_h"]

            # Get FDR for this fixture
            fpl_diff = fix.get("team_h_difficulty" if is_away else "team_a_difficulty", 3)
            fdr = get_fixture_fdr(opponent_id, is_home, 4, fpl_difficulty=fpl_diff)

            if current_gw <= gw <= horizon_end:
                all_fdrs.append(fdr)

                if gw < mid_point:
                    now_fdrs.append(fdr)
                else:
                    later_fdrs.append(fdr)

        # Detect DGW and BGW
        for gw in range(current_gw, horizon_end + 1):
            count = gw_fixture_count[gw]
            if count >= 2:
                dgws.append(gw)
            elif count == 0:
                bgws.append(gw)

        # Calculate averages
        current_fdr = sum(now_fdrs) / len(now_fdrs) if now_fdrs else 5.0
        upcoming_fdr = sum(later_fdrs) / len(later_fdrs) if later_fdrs else 5.0
        avg_fdr = sum(all_fdrs) / len(all_fdrs) if all_fdrs else 5.0
        swing = upcoming_fdr - current_fdr  # Negative = fixtures improving (getting easier)

        # Determine rating based on swing direction
        if swing <= -0.8:
            rating = "IMPROVING"  # Fixtures getting easier
        elif swing >= 0.8:
            rating = "WORSENING"  # Fixtures getting harder
        else:
            rating = "NEUTRAL"

        # DGW teams get bonus (even if neutral fixtures, more games = more points)
        if dgws and rating == "NEUTRAL":
            rating = "IMPROVING"

        swings.append(FixtureSwing(
            team_id=team_id,
            team_name=team["name"],
            team_short=team["short_name"],
            current_fdr=round(current_fdr, 2),
            upcoming_fdr=round(upcoming_fdr, 2),
            swing=round(swing, 2),
            dgw_in_horizon=dgws,
            bgw_in_horizon=bgws,
            rating=rating
        ))

    # Sort by AVERAGE FDR across horizon (lowest = easiest fixtures first)
    # This is more intuitive than swing direction
    swings.sort(key=lambda x: (x.current_fdr + x.upcoming_fdr) / 2)

    return swings


def get_chip_recommendations(
    squad: List[Dict],
    available_chips: Dict[str, bool],
    fixtures: List[Dict],
    events: List[Dict],
    teams_dict: Dict,
    elements_dict: Dict,
    current_gw: int,
    horizon: int = 8,
    squad_health: List[SquadHealthIssue] = None
) -> List[ChipRecommendation]:
    """
    Generate chip usage recommendations based on squad state and upcoming fixtures.
    """
    recommendations = []
    horizon_end = min(current_gw + horizon, 38)

    # Detect DGW/BGW in horizon
    gw_info = detect_dgw_bgw(fixtures, events, current_gw, horizon_end)

    # Count squad BGW exposure
    squad_team_ids = {p.get("team_id", 0) for p in squad}

    for gw in range(current_gw, horizon_end + 1):
        info = gw_info.get(gw, {})

        # FREE HIT recommendation
        if available_chips.get("freehit", False):
            bgw_teams = set(info.get("bgw_teams", []))
            squad_blanks = len(squad_team_ids & bgw_teams)

            # Also count players with BGW exposure from health issues
            if squad_health:
                bgw_players = [i for i in squad_health
                              if i.issue_type == "bgw_exposure" and gw in i.affected_gws]
                squad_blanks = max(squad_blanks, len(bgw_players))

            if squad_blanks >= 6:
                recommendations.append(ChipRecommendation(
                    chip="freehit",
                    recommended_gw=gw,
                    confidence="high",
                    reason=f"GW{gw}: {squad_blanks} players blanking - FH value is high",
                    expected_value=squad_blanks * 4.0  # ~4 xPts per blanking player saved
                ))
            elif squad_blanks >= 4:
                recommendations.append(ChipRecommendation(
                    chip="freehit",
                    recommended_gw=gw,
                    confidence="medium",
                    reason=f"GW{gw}: {squad_blanks} players blanking - FH worth considering",
                    expected_value=squad_blanks * 4.0
                ))

        # BENCH BOOST recommendation
        if available_chips.get("bboost", False):
            dgw_teams = set(info.get("dgw_teams", []))
            dgw_count = len(squad_team_ids & dgw_teams)

            if dgw_count >= 10:  # At least 10 DGW players
                recommendations.append(ChipRecommendation(
                    chip="bboost",
                    recommended_gw=gw,
                    confidence="high",
                    reason=f"GW{gw}: {dgw_count} DGW players - excellent BB opportunity",
                    expected_value=dgw_count * 1.5  # Extra bench value in DGW
                ))
            elif dgw_count >= 6:
                recommendations.append(ChipRecommendation(
                    chip="bboost",
                    recommended_gw=gw,
                    confidence="medium",
                    reason=f"GW{gw}: {dgw_count} DGW players - good BB opportunity",
                    expected_value=dgw_count * 1.2
                ))

        # TRIPLE CAPTAIN recommendation
        if available_chips.get("3xc", False):
            dgw_teams = set(info.get("dgw_teams", []))

            # Find best premium in squad with DGW
            premiums = [p for p in squad
                       if p.get("price", 0) >= 10.0 and p.get("team_id", 0) in dgw_teams]

            if premiums:
                best_premium = max(premiums, key=lambda x: x.get("xpts", 0))
                # Check if it's a favorable DGW (home games, weak opponents)
                recommendations.append(ChipRecommendation(
                    chip="3xc",
                    recommended_gw=gw,
                    confidence="medium",
                    reason=f"GW{gw}: {best_premium.get('name', 'Premium')} has DGW - TC candidate",
                    expected_value=best_premium.get("xpts", 0) * 0.5  # Extra captain points
                ))

    # WILDCARD recommendation (based on squad health)
    if available_chips.get("wildcard", False) and squad_health:
        critical_issues = [i for i in squad_health if i.severity == "critical"]
        warning_issues = [i for i in squad_health if i.severity == "warning"]

        if len(critical_issues) >= 3:
            recommendations.append(ChipRecommendation(
                chip="wildcard",
                recommended_gw=current_gw,
                confidence="high",
                reason=f"{len(critical_issues)} critical issues - WC could fix squad structure",
                expected_value=len(critical_issues) * 3.0
            ))
        elif len(critical_issues) + len(warning_issues) >= 5:
            recommendations.append(ChipRecommendation(
                chip="wildcard",
                recommended_gw=current_gw,
                confidence="medium",
                reason=f"Multiple squad issues ({len(critical_issues)} critical, {len(warning_issues)} warnings) - consider WC",
                expected_value=(len(critical_issues) * 3.0 + len(warning_issues) * 1.5)
            ))

    # Sort by expected value
    recommendations.sort(key=lambda x: -x.expected_value)

    return recommendations


def generate_transfer_candidates_for_gw(
    squad: List[Dict],
    elements: List[Dict],
    fixtures: List[Dict],
    teams_dict: Dict,
    events: List[Dict],
    target_gw: int,
    remaining_horizon: int,
    bank: float,
    strategy: str,
    fixture_swings: List[FixtureSwing] = None,
    excluded_out_ids: set = None,
    excluded_in_ids: set = None
) -> List[TransferRecommendation]:
    """
    Generate transfer recommendations for a specific GW.

    Key difference from old version:
    - Calculates xPts over REMAINING horizon from target_gw
    - Respects excluded player IDs
    - Strategy affects player selection (variance/ownership)
    """
    config = TransferPlannerConfig.STRATEGIES[strategy]
    recommendations = []

    # Build lookups
    squad_ids = {p["id"] for p in squad}
    team_counts = defaultdict(int)
    elements_dict = {e["id"]: e for e in elements}

    excluded_out_ids = excluded_out_ids or set()
    excluded_in_ids = excluded_in_ids or set()

    for p in squad:
        team_counts[p.get("team_id", 0)] += 1

    # Fixture swing bonus lookup
    swing_bonus = {}
    if fixture_swings:
        for swing in fixture_swings:
            if swing.rating == "IMPROVING":
                swing_bonus[swing.team_id] = 0.15
            elif swing.rating == "WORSENING":
                swing_bonus[swing.team_id] = -0.10

    # For each squad player, find best replacement
    for out_player in squad:
        out_id = out_player["id"]

        # Skip if already transferred out
        if out_id in excluded_out_ids:
            continue

        position_id = out_player.get("position_id", 3)
        position = POSITION_MAP.get(position_id, "MID")

        available_budget = bank + out_player.get("selling_price", out_player.get("price", 5.0))
        out_xpts = out_player.get("xpts", 0)

        # Recalculate OUT player xPts for remaining horizon
        out_element = elements_dict.get(out_id)
        if out_element:
            out_upcoming = get_player_upcoming_fixtures(
                out_element["team"], fixtures, target_gw, target_gw + remaining_horizon, teams_dict
            )
            out_stats = calculate_expected_points(
                out_element, position_id, target_gw, out_upcoming, teams_dict, fixtures, events
            )
            out_xpts = out_stats["xpts"]

        # Find candidates
        for player in elements:
            if player["id"] in squad_ids or player["id"] in excluded_in_ids:
                continue
            if player["element_type"] != position_id:
                continue

            price = player["now_cost"] / 10
            if price > available_budget:
                continue

            # Team limit check
            player_team = player["team"]
            current_count = team_counts[player_team] - (1 if out_player.get("team_id") == player_team else 0)
            if current_count >= 3:
                continue

            # Minutes filter
            total_minutes = player.get("minutes", 0)
            if total_minutes < 400:
                continue

            # Status filter
            status = player.get("status", "a")
            if status in ["i", "s", "u"]:
                continue

            # Calculate xPts for REMAINING horizon
            upcoming = get_player_upcoming_fixtures(
                player["team"], fixtures, target_gw, target_gw + remaining_horizon, teams_dict
            )
            stats = calculate_expected_points(
                player, position_id, target_gw, upcoming, teams_dict, fixtures, events
            )

            in_xpts = stats["xpts"]
            xpts_gain = in_xpts - out_xpts

            # Apply fixture swing bonus
            swing = swing_bonus.get(player_team, 0)
            effective_gain = xpts_gain * (1 + swing)

            # Strategy-specific adjustments
            ownership = float(player.get("selected_by_percent", 0) or 0)
            ceiling = stats.get("xpts_ceiling", in_xpts * 1.2)
            floor = stats.get("xpts_floor", in_xpts * 0.7)
            variance = ceiling - floor

            # SAFE: Prefer high floor, high ownership
            if strategy == "safe":
                if ownership > 15:  # Template player
                    effective_gain *= 1.08
                if variance < 8:  # Low variance
                    effective_gain *= 1.05
                if variance > 15:  # High variance penalty
                    effective_gain *= 0.90

            # RISKY: Prefer high ceiling, low ownership differentials
            elif strategy == "risky":
                if ownership < 10:  # Differential
                    effective_gain *= 1.15
                if variance > 12:  # High variance bonus
                    effective_gain *= 1.10
                # Use ceiling-weighted value
                effective_gain = effective_gain * 0.6 + (ceiling - out_xpts) * 0.4

            # BALANCED: Mix
            else:
                weighted_xpts = in_xpts * 0.5 + ceiling * 0.3 + floor * 0.2
                effective_gain = weighted_xpts - out_xpts

            # Nailed preference for safe
            exp_mins = stats.get("expected_minutes", 90)
            if config["prefer_nailed"] and exp_mins < 75:
                effective_gain *= 0.85

            if effective_gain > 0.3:  # Minimum threshold to consider
                recommendations.append(TransferRecommendation(
                    out_player={
                        "id": out_id,
                        "name": out_player.get("name", "Unknown"),
                        "team": out_player.get("team", "???"),
                        "team_id": out_player.get("team_id", 0),
                        "position": position,
                        "position_id": position_id,
                        "price": out_player.get("price", 0),
                        "selling_price": out_player.get("selling_price", 0),
                        "xpts": out_xpts,
                    },
                    in_player={
                        "id": player["id"],
                        "name": player["web_name"],
                        "team": teams_dict.get(player_team, {}).get("short_name", "???"),
                        "team_id": player_team,
                        "position": position,
                        "position_id": position_id,
                        "price": price,
                        "xpts": in_xpts,
                        "xpts_ceiling": ceiling,
                        "xpts_floor": floor,
                        "form": float(player.get("form", 0) or 0),
                        "ownership": ownership,
                        "expected_minutes": exp_mins,
                        "variance": variance,
                    },
                    gw=target_gw,
                    xpts_gain=round(effective_gain, 2),
                    reason=f"+{effective_gain:.1f} xPts over {remaining_horizon}GW",
                    is_essential=False,
                    is_hit=False,
                    priority=2
                ))

    # Sort by xPts gain
    recommendations.sort(key=lambda x: -x.xpts_gain)

    return recommendations[:20]


def calculate_roll_value(
    current_ft: int,
    gws_remaining: int,
    has_dgw_ahead: bool,
    has_bgw_ahead: bool,
    critical_issues: int
) -> float:
    """
    Calculate the expected value of rolling a free transfer.

    Factors:
    - Already at 2+ FT = lower roll value (capped)
    - DGW/BGW ahead = higher roll value (want flexibility)
    - Critical issues = lower roll value (need to act)
    - More GWs remaining = higher roll value
    """
    base_roll_value = 0.4  # Base optionality value

    # Already have 2+ FT - diminishing returns
    if current_ft >= 2:
        base_roll_value *= 0.6
    if current_ft >= 3:
        base_roll_value *= 0.5
    if current_ft >= 4:
        base_roll_value *= 0.3

    # DGW/BGW ahead - want flexibility
    if has_dgw_ahead or has_bgw_ahead:
        base_roll_value *= 1.4

    # Critical issues - need to act, not roll
    if critical_issues >= 2:
        base_roll_value *= 0.5
    elif critical_issues >= 1:
        base_roll_value *= 0.7

    # Horizon factor - more value if more GWs to plan
    horizon_factor = min(gws_remaining / 8, 1.0)
    base_roll_value *= (0.7 + 0.3 * horizon_factor)

    return base_roll_value


def beam_search_strategy(
    initial_squad: List[Dict],
    initial_bank: float,
    initial_ft: int,
    current_gw: int,
    horizon_end: int,
    strategy: str,
    config: Dict,
    elements: List[Dict],
    elements_dict: Dict,
    fixtures: List[Dict],
    events: List[Dict],
    teams_dict: Dict,
    fixture_swings: List,
    squad_health: List,
    chip_placements: Dict[int, str],
    booked_by_gw: Dict,
    player_gw_cache: Dict,
    gw_info: Dict,
) -> TransferPath:
    """
    Beam search over transfer decisions across the planning horizon.
    Explores multiple paths (roll, single transfer, pair, triple) at each GW
    and prunes to the top beam_width paths by total xPts.
    Returns the single best TransferPath.
    """
    beam_width = config.get("beam_width", 10)
    n_candidates = config.get("beam_candidates", 5)
    max_pairs = config.get("beam_max_pairs", 3)

    # Identify forced-essential player IDs (injuries)
    injury_ids = {i.player_id for i in squad_health
                  if i.issue_type == "injury" and i.severity == "critical"}
    bgw_issues = [i for i in squad_health if i.issue_type == "bgw_exposure"]

    # Initialize beam with single starting path
    seed = TransferPath(initial_squad, initial_bank, initial_ft)
    beam = [seed]

    prev_gw_chip = None

    for gw in range(current_gw, horizon_end + 1):
        remaining_horizon = horizon_end - gw + 1
        active_chip = chip_placements.get(gw)

        # --- FH revert from previous GW ---
        if prev_gw_chip == "freehit":
            for path in beam:
                if path.pre_fh_squad is not None:
                    path.squad = [p.copy() for p in path.pre_fh_squad]
                    path.bank = path.pre_fh_bank
                    path.pre_fh_squad = None
                    path.pre_fh_bank = None

        # --- Chip GWs: WC / FH — apply uniformly, skip branching ---
        if active_chip in ("wildcard", "freehit"):
            for path in beam:
                if active_chip == "wildcard":
                    total_budget = path.bank + sum(
                        p.get("selling_price", p.get("price", 0)) for p in path.squad
                    )
                    new_squad = build_optimal_squad(
                        elements, fixtures, teams_dict, events,
                        target_gw=gw, horizon=remaining_horizon, budget=total_budget,
                        current_squad_ids={p["id"] for p in path.squad},
                        player_gw_cache=player_gw_cache,
                    )
                    if new_squad:
                        total_cost = sum(p.get("price", 0) for p in new_squad)
                        path.squad = new_squad
                        path.bank = total_budget - total_cost
                    path.ft = 1
                    path.actions.append({
                        "type": "wildcard", "gw": gw,
                        "new_squad": [p.copy() for p in path.squad],
                    })
                elif active_chip == "freehit":
                    path.pre_fh_squad = [p.copy() for p in path.squad]
                    path.pre_fh_bank = path.bank
                    total_budget = path.bank + sum(
                        p.get("selling_price", p.get("price", 0)) for p in path.squad
                    )
                    fh_squad = build_optimal_squad(
                        elements, fixtures, teams_dict, events,
                        target_gw=gw, horizon=1, budget=total_budget,
                        single_gw=True, player_gw_cache=player_gw_cache,
                    )
                    if fh_squad:
                        path.squad = fh_squad
                    path.actions.append({
                        "type": "freehit", "gw": gw,
                        "new_squad": [p.copy() for p in path.squad],
                    })

                gw_xpts = evaluate_squad_xpts(
                    path.squad, gw, fixtures, teams_dict, elements_dict, events,
                    chip=active_chip if active_chip in ("bboost", "3xc") else None,
                    player_gw_cache=player_gw_cache,
                )
                path.total_xpts += gw_xpts
                path.gw_xpts[gw] = gw_xpts

            prev_gw_chip = active_chip
            continue

        # --- Booked transfers: apply to all paths ---
        gw_booked = booked_by_gw.get(gw, [])
        if gw_booked:
            for path in beam:
                for bt in gw_booked:
                    out_player = next((p for p in path.squad if p["id"] == bt["out_id"]), None)
                    in_element = elements_dict.get(bt["in_id"])
                    if out_player and in_element:
                        pos_id = in_element["element_type"]
                        in_price = in_element["now_cost"] / 10
                        transfer_dict = {
                            "out": out_player,
                            "in": {
                                "id": in_element["id"],
                                "name": in_element.get("web_name", "Unknown"),
                                "team": teams_dict.get(in_element["team"], {}).get("short_name", "???"),
                                "team_id": in_element["team"],
                                "position": POSITION_MAP.get(pos_id, "MID"),
                                "position_id": pos_id,
                                "price": in_price,
                                "xpts": 0,
                                "form": float(in_element.get("form", 0) or 0),
                                "expected_minutes": 90,
                            },
                            "xpts_gain": 0,
                            "reason": "Booked transfer",
                            "is_booked": True,
                        }
                        is_hit = path.ft <= 0
                        if not is_hit:
                            path.use_ft()
                        path.apply_transfer(transfer_dict, gw, is_hit=is_hit)
                        path.transferred_out_ids.add(bt["out_id"])
                        path.transferred_in_ids.add(bt["in_id"])

                gw_xpts = evaluate_squad_xpts(
                    path.squad, gw, fixtures, teams_dict, elements_dict, events,
                    chip=active_chip, player_gw_cache=player_gw_cache,
                )
                path.total_xpts += gw_xpts
                path.gw_xpts[gw] = gw_xpts

                # FT accrual: after transfers, next GW gets at least 1
                path.ft = max(1, path.ft)

            prev_gw_chip = active_chip
            continue

        # --- Normal branching ---
        new_branches = []
        candidate_cache = {}  # (squad_key, excluded) -> candidates

        for path in beam:
            # Cache candidate generation by squad state
            cache_key = (path.squad_key(), frozenset(path.transferred_out_ids))
            if cache_key in candidate_cache:
                candidates = candidate_cache[cache_key]
            else:
                candidates = generate_transfer_candidates_for_gw(
                    squad=path.squad, elements=elements, fixtures=fixtures,
                    teams_dict=teams_dict, events=events,
                    target_gw=gw, remaining_horizon=remaining_horizon,
                    bank=path.bank, strategy=strategy,
                    fixture_swings=fixture_swings,
                    excluded_out_ids=path.transferred_out_ids,
                    excluded_in_ids=path.transferred_in_ids,
                )
                candidate_cache[cache_key] = candidates

            # Mark essentials
            for c in candidates:
                if c.out_player["id"] in injury_ids:
                    c.is_essential = True
                    c.priority = 1
                else:
                    for bi in bgw_issues:
                        if bi.player_id == c.out_player["id"] and gw in bi.affected_gws:
                            c.is_essential = True
                            c.priority = 1
                            break

            essentials = [c for c in candidates if c.is_essential]
            non_essentials = [c for c in candidates if not c.is_essential]
            top_singles = (essentials + non_essentials)[:n_candidates]

            has_forced = any(
                c.is_essential and c.out_player["id"] in injury_ids for c in essentials
            )

            # --- Branch 0: ROLL ---
            if not has_forced:
                roll = path.copy()
                roll.roll_transfer(gw)
                gw_xpts = evaluate_squad_xpts(
                    roll.squad, gw, fixtures, teams_dict, elements_dict, events,
                    chip=active_chip, player_gw_cache=player_gw_cache,
                )
                roll.total_xpts += gw_xpts
                roll.gw_xpts[gw] = gw_xpts
                new_branches.append(roll)

            # --- Branch 1..N: SINGLE TRANSFERS ---
            for c in top_singles:
                is_hit = path.ft <= 0
                if is_hit:
                    if path.hits >= config.get("max_hits_per_horizon", 1):
                        continue
                    if c.xpts_gain < config.get("min_xpts_gain_for_hit", 12.0):
                        continue

                branch = path.copy()
                if not is_hit:
                    branch.use_ft()
                td = candidate_to_transfer_dict(c)
                branch.apply_transfer(td, gw, is_hit=is_hit)
                branch.transferred_out_ids.add(c.out_player["id"])
                branch.transferred_in_ids.add(c.in_player["id"])

                gw_xpts = evaluate_squad_xpts(
                    branch.squad, gw, fixtures, teams_dict, elements_dict, events,
                    chip=active_chip, player_gw_cache=player_gw_cache,
                )
                branch.total_xpts += gw_xpts
                branch.gw_xpts[gw] = gw_xpts
                # After transfers, next GW gets at least 1 FT
                branch.ft = max(1, branch.ft)
                new_branches.append(branch)

            # --- Branch N+1..M: TRANSFER PAIRS (FT >= 2) ---
            if path.ft >= 2 and len(top_singles) >= 2:
                pairs = generate_compatible_pairs(top_singles, max_pairs)
                for c1, c2 in pairs:
                    branch = path.copy()
                    branch.use_ft()
                    branch.apply_transfer(candidate_to_transfer_dict(c1), gw, is_hit=False)
                    branch.use_ft()
                    branch.apply_transfer(candidate_to_transfer_dict(c2), gw, is_hit=False)
                    branch.transferred_out_ids.add(c1.out_player["id"])
                    branch.transferred_out_ids.add(c2.out_player["id"])
                    branch.transferred_in_ids.add(c1.in_player["id"])
                    branch.transferred_in_ids.add(c2.in_player["id"])

                    gw_xpts = evaluate_squad_xpts(
                        branch.squad, gw, fixtures, teams_dict, elements_dict, events,
                        chip=active_chip, player_gw_cache=player_gw_cache,
                    )
                    branch.total_xpts += gw_xpts
                    branch.gw_xpts[gw] = gw_xpts
                    branch.ft = max(1, branch.ft)
                    new_branches.append(branch)

            # --- Branch: TRIPLE (FT >= 3) ---
            if path.ft >= 3 and len(top_singles) >= 3:
                # Check that top 3 are mutually compatible
                t3 = top_singles[:3]
                ids_out = {c.out_player["id"] for c in t3}
                ids_in = {c.in_player["id"] for c in t3}
                if len(ids_out) == 3 and len(ids_in) == 3:
                    branch = path.copy()
                    for c in t3:
                        branch.use_ft()
                        branch.apply_transfer(candidate_to_transfer_dict(c), gw, is_hit=False)
                        branch.transferred_out_ids.add(c.out_player["id"])
                        branch.transferred_in_ids.add(c.in_player["id"])
                    gw_xpts = evaluate_squad_xpts(
                        branch.squad, gw, fixtures, teams_dict, elements_dict, events,
                        chip=active_chip, player_gw_cache=player_gw_cache,
                    )
                    branch.total_xpts += gw_xpts
                    branch.gw_xpts[gw] = gw_xpts
                    branch.ft = max(1, branch.ft)
                    new_branches.append(branch)

        # --- PRUNE: deduplicate by squad_key, keep top beam_width ---
        if new_branches:
            seen = {}
            for branch in new_branches:
                key = branch.squad_key()
                if key not in seen or branch.score() > seen[key].score():
                    seen[key] = branch
            beam = sorted(seen.values(), key=lambda p: -p.score())[:beam_width]
        # If no branches (shouldn't happen), keep current beam

        prev_gw_chip = active_chip

    # Return best path
    return beam[0] if beam else TransferPath(initial_squad, initial_bank, initial_ft)


def convert_path_to_gw_actions(
    path: TransferPath,
    current_gw: int,
    horizon_end: int,
    chip_placements: Dict[int, str],
) -> Dict[int, Dict]:
    """Convert a TransferPath's action list into gw_actions format for StrategyPlan."""
    gw_actions = {}

    # Group actions by GW
    actions_by_gw = defaultdict(list)
    for action in path.actions:
        actions_by_gw[action["gw"]].append(action)

    # Track FT for display
    ft = path.initial_ft

    for gw in range(current_gw, horizon_end + 1):
        gw_acts = actions_by_gw.get(gw, [])
        transfers = [a for a in gw_acts if a["type"] == "transfer"]
        is_roll = any(a["type"] == "roll" for a in gw_acts)
        is_wc = any(a["type"] == "wildcard" for a in gw_acts)
        is_fh = any(a["type"] == "freehit" for a in gw_acts)
        chip = chip_placements.get(gw)

        gw_action = {
            "gw": gw,
            "action": "roll",
            "transfers": [],
            "chip": chip,
            "reasoning": "",
            "ft_before": ft,
            "ft_after": ft,
            "bank_before": 0,
            "bank_after": 0,
            "xpts": round(path.gw_xpts.get(gw, 0), 1),
        }

        if is_wc:
            gw_action["action"] = "wildcard"
            gw_action["reasoning"] = "Wildcard active — squad rebuilt"
            ft = 1
            # Include the new squad for display
            wc_act = next((a for a in gw_acts if a["type"] == "wildcard"), None)
            if wc_act and wc_act.get("new_squad"):
                gw_action["wildcard_squad"] = [
                    {
                        "id": p.get("id"),
                        "name": p.get("name", p.get("web_name", "?")),
                        "team": p.get("team", p.get("team_short", "?")),
                        "position": p.get("position", "?"),
                        "price": p.get("price", 0),
                    }
                    for p in wc_act["new_squad"]
                ]
        elif is_fh:
            gw_action["action"] = "freehit"
            gw_action["reasoning"] = "Free Hit — temporary optimal squad"
            # Include the FH squad for display
            fh_act = next((a for a in gw_acts if a["type"] == "freehit"), None)
            if fh_act and fh_act.get("new_squad"):
                gw_action["freehit_squad"] = [
                    {
                        "id": p.get("id"),
                        "name": p.get("name", p.get("web_name", "?")),
                        "team": p.get("team", p.get("team_short", "?")),
                        "position": p.get("position", "?"),
                        "price": p.get("price", 0),
                    }
                    for p in fh_act["new_squad"]
                ]
            # FT unchanged after FH
        elif transfers:
            gw_action["action"] = "transfer"
            for t in transfers:
                gw_action["transfers"].append({
                    "out": t["out"],
                    "in": t["in"],
                    "is_booked": t.get("is_booked", False),
                    "is_hit": t.get("is_hit", False),
                    "xpts_gain": t.get("xpts_gain", 0),
                    "reason": t.get("reason", ""),
                })
            reasons = [t.get("reason", "") for t in transfers[:2]]
            gw_action["reasoning"] = "; ".join(r for r in reasons if r)
            # FT: used some, get at least 1 next week
            ft_used = sum(1 for t in transfers if not t.get("is_hit", False))
            ft = max(1, ft - ft_used)
        else:
            gw_action["action"] = "roll"
            gw_action["reasoning"] = "Roll FT — bank for future"
            ft = min(ft + 1, TransferPlannerConfig.MAX_FT)

        gw_action["ft_after"] = ft
        gw_actions[gw] = gw_action

    return gw_actions


def build_strategy_plan(
    squad: List[Dict],
    bank: float,
    free_transfers: int,
    fixtures: List[Dict],
    events: List[Dict],
    teams_dict: Dict,
    elements: List[Dict],
    current_gw: int,
    horizon: int,
    strategy: str,
    available_chips: Dict[str, bool],
    squad_health: List[SquadHealthIssue],
    fixture_swings: List[FixtureSwing],
    chip_recommendations: List[ChipRecommendation],
    booked_transfers: List[Dict] = None,
    chip_overrides: List[Dict] = None,
) -> StrategyPlan:
    """
    Build a complete strategy plan with proper multi-GW branching tree logic.

    Key principles:
    1. Each GW, decide: transfer or roll based on value comparison
    2. Track transferred players to prevent duplicates
    3. Recalculate xPts for remaining horizon at each GW
    4. Strategy affects both thresholds AND player selection
    """
    config = TransferPlannerConfig.STRATEGIES[strategy]
    elements_dict = {e["id"]: e for e in elements}

    # Initialize plan state
    plan_squad = [p.copy() for p in squad]
    plan_bank = bank
    plan_ft = free_transfers

    # Booked transfers by GW
    booked_by_gw = defaultdict(list)
    if booked_transfers:
        for bt in booked_transfers:
            booked_by_gw[bt.get("gw", current_gw)].append(bt)

    horizon_end = min(current_gw + horizon - 1, 38)

    # Detect DGW/BGW in horizon
    gw_info = detect_dgw_bgw(fixtures, events, current_gw, horizon_end)

    # Pre-compute chip placements
    player_gw_cache = {}
    chip_placements, chip_placement_details = determine_chip_placements(
        available_chips=available_chips,
        chip_recommendations=chip_recommendations,
        squad=plan_squad,
        bank=plan_bank,
        fixtures=fixtures,
        events=events,
        teams_dict=teams_dict,
        elements=elements,
        elements_dict=elements_dict,
        current_gw=current_gw,
        horizon_end=horizon_end,
        strategy_config=config,
        player_gw_cache=player_gw_cache,
        chip_overrides=chip_overrides,
    )

    # Phase 2: Beam search over transfer decisions
    best_path = beam_search_strategy(
        initial_squad=plan_squad,
        initial_bank=plan_bank,
        initial_ft=plan_ft,
        current_gw=current_gw,
        horizon_end=horizon_end,
        strategy=strategy,
        config=config,
        elements=elements,
        elements_dict=elements_dict,
        fixtures=fixtures,
        events=events,
        teams_dict=teams_dict,
        fixture_swings=fixture_swings,
        squad_health=squad_health,
        chip_placements=chip_placements,
        booked_by_gw=booked_by_gw,
        player_gw_cache=player_gw_cache,
        gw_info=gw_info,
    )

    # Phase 3: Convert best path to gw_actions format
    gw_actions = convert_path_to_gw_actions(
        best_path, current_gw, horizon_end, chip_placements
    )
    total_xpts = best_path.total_xpts
    plan_hits = best_path.hits
    plan_transfers = sum(1 for a in best_path.actions if a["type"] == "transfer")

    # Build headline
    chips_used = [CHIP_DISPLAY.get(c, c.upper()) for c in chip_placements.values()]
    chip_text = f" + {', '.join(chips_used)}" if chips_used else ""

    if plan_transfers == 0 and not chips_used:
        headline = f"Roll all transfers - bank FT for flexibility"
    elif plan_hits == 0:
        headline = f"{plan_transfers} transfer{'s' if plan_transfers > 1 else ''}{chip_text}"
    else:
        headline = f"{plan_transfers} transfer{'s' if plan_transfers > 1 else ''} ({plan_hits} hit{'s' if plan_hits > 1 else ''}){chip_text}"

    # total_xpts from beam search already includes -4 per hit
    net_xpts = total_xpts

    return StrategyPlan(
        name=strategy,
        description=config["description"],
        gw_actions=gw_actions,
        total_xpts=round(net_xpts, 1),
        hit_cost=plan_hits * 4,
        transfers_made=plan_transfers,
        chip_recommendations=[c for c in chip_recommendations if c.chip in
                             [k for k, v in available_chips.items() if v]],
        risk_score=config["risk_score"],
        headline=headline,
        chip_placements=chip_placement_details,
    )
