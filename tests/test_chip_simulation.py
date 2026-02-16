"""Tests for chip simulation: squad optimizer, marginal calcs, chip integration."""
import sys
import os
from unittest.mock import patch, MagicMock
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from main import (
    build_optimal_squad, calculate_chip_marginal_value,
    determine_chip_placements, evaluate_squad_xpts, select_best_xi,
    ChipPlacement, ChipRecommendation, StrategyPlan, TransferPlannerConfig,
    DataCache, MODEL_CONFIG, POSITION_MAP,
)
from fpl_assistant.planner import _greedy_squad


# ============ FIXTURES ============

def _teams_dict():
    """Minimal teams dict."""
    return {
        i: {"name": f"Team{i}", "short_name": f"T{i:02d}"}
        for i in range(1, 21)
    }


def _make_element(pid, team, pos, price_tenths, minutes=1800, xg="3.0", xa="2.0", form="5.0"):
    """Create a player element dict matching FPL API shape."""
    return {
        "id": pid,
        "web_name": f"Player{pid}",
        "team": team,
        "element_type": pos,
        "now_cost": price_tenths,
        "minutes": minutes,
        "goals_scored": 5,
        "assists": 3,
        "clean_sheets": 4,
        "bonus": 10,
        "bps": 400,
        "saves": 30 if pos == 1 else 0,
        "expected_goals": xg,
        "expected_assists": xa,
        "expected_goal_involvements": str(float(xg) + float(xa)),
        "points_per_game": "5.0",
        "total_points": 90,
        "yellow_cards": 2,
        "red_cards": 0,
        "own_goals": 0,
        "defensive_contribution_per_90": "8.0",
        "defensive_contribution": 160,
        "starts": 18,
        "status": "a",
        "form": form,
    }


def _make_pool():
    """Create a pool of 80 elements: 8 GKP, 25 DEF, 25 MID, 22 FWD across 10 teams."""
    elements = []
    pid = 1

    # GKPs: 8 across teams 1-8 (prices 4.0-5.5m)
    for team in range(1, 9):
        elements.append(_make_element(pid, team, 1, 40 + team * 2))
        pid += 1

    # DEFs: 25 across teams 1-10 (prices 4.0-6.0m)
    for team in range(1, 11):
        for j in range(2 + (1 if team <= 5 else 0)):
            elements.append(_make_element(pid, team, 2, 40 + j * 10))
            pid += 1

    # MIDs: 25 across teams 1-10 (prices 5.0-8.0m)
    for team in range(1, 11):
        for j in range(2 + (1 if team <= 5 else 0)):
            elements.append(_make_element(pid, team, 3, 50 + j * 15))
            pid += 1

    # FWDs: 22 across teams 1-10 (prices 5.5-8.0m)
    for team in range(1, 11):
        for j in range(2 + (1 if team <= 2 else 0)):
            elements.append(_make_element(pid, team, 4, 55 + j * 12))
            pid += 1

    return elements


def _make_squad():
    """Create a 15-player squad (2 GKP, 5 DEF, 5 MID, 3 FWD)."""
    squad = []
    pid = 1001

    for pos, count, base_price in [(1, 2, 5.0), (2, 5, 5.5), (3, 5, 7.0), (4, 3, 8.0)]:
        for i in range(count):
            squad.append({
                "id": pid,
                "name": f"Squad{pid}",
                "team": (pid % 10) + 1,
                "team_id": (pid % 10) + 1,
                "position": POSITION_MAP.get(pos, "MID"),
                "position_id": pos,
                "price": base_price + i * 0.5,
                "selling_price": base_price + i * 0.5,
                "xpts": 4.0 + i * 0.3,
                "form": 4.0,
                "expected_minutes": 85,
            })
            pid += 1

    return squad


def _make_fixtures():
    """Create basic fixtures for GW24-28."""
    fixtures = []
    fid = 1
    for gw in range(24, 29):
        for home in range(1, 11):
            away = (home % 10) + 1
            fixtures.append({
                "id": fid,
                "event": gw,
                "team_h": home,
                "team_a": away,
                "team_h_difficulty": 3,
                "team_a_difficulty": 3,
                "finished": False,
            })
            fid += 1
    return fixtures


def _make_events():
    """Create basic events for GW24-28."""
    return [{"id": gw, "finished": gw < 24, "deadline_time": f"2025-02-{gw}T11:00:00Z"} for gw in range(1, 39)]


def _make_test_cache():
    """Populate a DataCache with FDR data for testing."""
    cache = DataCache()
    cache.fdr_data = {
        i: {
            "attack_fdr": 5, "defence_fdr": 5,
            "xg": 1.3, "xga": 1.3, "blended_xga": 1.3,
            "cs_probability": 0.25,
            "home_xga": 1.1, "away_xga": 1.5,
        }
        for i in range(1, 21)
    }
    return cache


# ============ TESTS: _greedy_squad ============

class TestGreedySquad:
    def test_valid_structure(self):
        """Greedy squad should return 15 players: 2 GKP, 5 DEF, 5 MID, 3 FWD."""
        elements = _make_pool()
        candidates = [
            {"element": p, "id": p["id"], "position_id": p["element_type"],
             "team": p["team"], "price": p["now_cost"] / 10, "xpts": 5.0}
            for p in elements
        ]
        result = _greedy_squad(candidates, budget=100.0, teams_dict=_teams_dict())

        assert len(result) == 15
        pos_counts = defaultdict(int)
        for p in result:
            pos_counts[p["position_id"]] += 1
        assert pos_counts[1] == 2
        assert pos_counts[2] == 5
        assert pos_counts[3] == 5
        assert pos_counts[4] == 3

    def test_respects_budget(self):
        """Total cost of greedy squad must not exceed budget."""
        elements = _make_pool()
        candidates = [
            {"element": p, "id": p["id"], "position_id": p["element_type"],
             "team": p["team"], "price": p["now_cost"] / 10, "xpts": 5.0}
            for p in elements
        ]
        budget = 85.0
        result = _greedy_squad(candidates, budget=budget, teams_dict=_teams_dict())
        total_cost = sum(p["price"] for p in result)
        assert total_cost <= budget

    def test_max_three_per_team(self):
        """No team should have more than 3 players."""
        elements = _make_pool()
        candidates = [
            {"element": p, "id": p["id"], "position_id": p["element_type"],
             "team": p["team"], "price": p["now_cost"] / 10, "xpts": 5.0}
            for p in elements
        ]
        result = _greedy_squad(candidates, budget=100.0, teams_dict=_teams_dict())
        team_counts = defaultdict(int)
        for p in result:
            team_counts[p["team_id"]] += 1
        for team_id, count in team_counts.items():
            assert count <= 3, f"Team {team_id} has {count} players"


# ============ TESTS: build_optimal_squad (with mocked xPts) ============

class TestBuildOptimalSquad:
    @patch("fpl_assistant.planner.calculate_expected_points")
    @patch("fpl_assistant.planner.get_player_upcoming_fixtures")
    @patch("main.cache", _make_test_cache())
    def test_valid_structure(self, mock_fixtures, mock_xpts):
        """PuLP squad should return valid 2-5-5-3 structure."""
        mock_fixtures.return_value = [{"opponent_id": 2, "is_home": True, "gameweek": 24}]
        mock_xpts.return_value = {"xpts": 5.0}

        result = build_optimal_squad(
            elements=_make_pool(),
            fixtures=_make_fixtures(),
            teams_dict=_teams_dict(),
            events=_make_events(),
            target_gw=24,
            horizon=3,
            budget=100.0,
        )

        assert len(result) == 15
        pos_counts = defaultdict(int)
        for p in result:
            pos_counts[p["position_id"]] += 1
        assert pos_counts[1] == 2, f"Got {pos_counts[1]} GKPs"
        assert pos_counts[2] == 5, f"Got {pos_counts[2]} DEFs"
        assert pos_counts[3] == 5, f"Got {pos_counts[3]} MIDs"
        assert pos_counts[4] == 3, f"Got {pos_counts[4]} FWDs"

    @patch("fpl_assistant.planner.calculate_expected_points")
    @patch("fpl_assistant.planner.get_player_upcoming_fixtures")
    @patch("main.cache", _make_test_cache())
    def test_respects_budget(self, mock_fixtures, mock_xpts):
        """PuLP squad total cost must not exceed budget."""
        mock_fixtures.return_value = [{"opponent_id": 2, "is_home": True, "gameweek": 24}]
        mock_xpts.return_value = {"xpts": 5.0}

        budget = 85.0
        result = build_optimal_squad(
            elements=_make_pool(),
            fixtures=_make_fixtures(),
            teams_dict=_teams_dict(),
            events=_make_events(),
            target_gw=24,
            horizon=3,
            budget=budget,
        )

        total_cost = sum(p["price"] for p in result)
        assert total_cost <= budget + 0.01  # small float tolerance

    @patch("fpl_assistant.planner.calculate_expected_points")
    @patch("fpl_assistant.planner.get_player_upcoming_fixtures")
    @patch("main.cache", _make_test_cache())
    def test_max_three_per_team(self, mock_fixtures, mock_xpts):
        """No team should have more than 3 players in PuLP squad."""
        mock_fixtures.return_value = [{"opponent_id": 2, "is_home": True, "gameweek": 24}]
        mock_xpts.return_value = {"xpts": 5.0}

        result = build_optimal_squad(
            elements=_make_pool(),
            fixtures=_make_fixtures(),
            teams_dict=_teams_dict(),
            events=_make_events(),
            target_gw=24,
            horizon=3,
            budget=100.0,
        )

        team_counts = defaultdict(int)
        for p in result:
            team_counts[p["team_id"]] += 1
        for team_id, count in team_counts.items():
            assert count <= 3, f"Team {team_id} has {count} players"


# ============ TESTS: chip marginal value ============

class TestChipMarginalValue:
    @patch("fpl_assistant.planner.evaluate_squad_xpts")
    def test_bb_marginal_positive(self, mock_eval):
        """BB marginal should be positive when bench has xPts."""
        # With BB: all 15 score → higher; without: best XI only
        mock_eval.side_effect = [45.0, 38.0]  # with BB, without

        marginal, new_squad = calculate_chip_marginal_value(
            chip="bboost", gw=24,
            squad=_make_squad(), bank=2.0,
            fixtures=_make_fixtures(), teams_dict=_teams_dict(),
            elements=_make_pool(), elements_dict={},
            events=_make_events(), horizon_end=28,
            player_gw_cache={},
        )
        assert marginal == pytest.approx(7.0)
        assert new_squad is None

    @patch("fpl_assistant.planner.evaluate_squad_xpts")
    def test_tc_marginal_positive(self, mock_eval):
        """TC marginal should be positive (captain scores 3x vs 2x)."""
        mock_eval.side_effect = [42.0, 38.0]  # with TC, without

        marginal, new_squad = calculate_chip_marginal_value(
            chip="3xc", gw=24,
            squad=_make_squad(), bank=2.0,
            fixtures=_make_fixtures(), teams_dict=_teams_dict(),
            elements=_make_pool(), elements_dict={},
            events=_make_events(), horizon_end=28,
            player_gw_cache={},
        )
        assert marginal == pytest.approx(4.0)
        assert new_squad is None

    @patch("fpl_assistant.planner.evaluate_squad_xpts")
    @patch("fpl_assistant.planner.build_optimal_squad")
    def test_fh_returns_new_squad(self, mock_build, mock_eval):
        """FH should return a new squad and positive marginal."""
        fh_squad = _make_squad()
        mock_build.return_value = fh_squad
        mock_eval.side_effect = [50.0, 38.0]  # FH squad xPts, current xPts

        marginal, new_squad = calculate_chip_marginal_value(
            chip="freehit", gw=24,
            squad=_make_squad(), bank=2.0,
            fixtures=_make_fixtures(), teams_dict=_teams_dict(),
            elements=_make_pool(), elements_dict={},
            events=_make_events(), horizon_end=28,
            player_gw_cache={},
        )
        assert marginal == pytest.approx(12.0)
        assert new_squad is not None


# ============ TESTS: determine_chip_placements ============

class TestDetermineChipPlacements:
    @patch("fpl_assistant.planner.calculate_chip_marginal_value")
    def test_chip_used_once(self, mock_calc):
        """Each chip should appear at most once in placements."""
        mock_calc.return_value = (5.0, None)

        available = {"wildcard": True, "freehit": True, "bboost": True, "3xc": True}
        recs = [
            ChipRecommendation("bboost", 25, "high", "DGW", 5.0),
            ChipRecommendation("bboost", 26, "medium", "DGW", 3.0),
        ]
        config = TransferPlannerConfig.STRATEGIES["balanced"]

        placements, details = determine_chip_placements(
            available, recs, _make_squad(), 2.0,
            _make_fixtures(), _make_events(), _teams_dict(),
            _make_pool(), {}, 24, 28, config, {},
        )

        chip_names = [d.chip for d in details]
        assert len(chip_names) == len(set(chip_names)), "Duplicate chips placed"

    @patch("fpl_assistant.planner.calculate_chip_marginal_value")
    def test_chip_only_if_available(self, mock_calc):
        """Unavailable chips should never be placed."""
        mock_calc.return_value = (10.0, None)

        available = {"wildcard": False, "freehit": False, "bboost": True, "3xc": False}
        recs = [ChipRecommendation("bboost", 25, "high", "DGW", 5.0)]
        config = TransferPlannerConfig.STRATEGIES["balanced"]

        placements, details = determine_chip_placements(
            available, recs, _make_squad(), 2.0,
            _make_fixtures(), _make_events(), _teams_dict(),
            _make_pool(), {}, 24, 28, config, {},
        )

        placed_chips = {d.chip for d in details}
        assert "wildcard" not in placed_chips
        assert "freehit" not in placed_chips
        assert "3xc" not in placed_chips

    @patch("fpl_assistant.planner.calculate_chip_marginal_value")
    def test_chip_respects_threshold(self, mock_calc):
        """Low-marginal chips should be skipped per strategy config."""
        mock_calc.return_value = (1.0, None)  # Below safe threshold of 4.0

        available = {"wildcard": False, "freehit": False, "bboost": True, "3xc": True}
        recs = [
            ChipRecommendation("bboost", 25, "low", "DGW", 1.0),
            ChipRecommendation("3xc", 26, "low", "Single GW", 0.5),
        ]
        config = TransferPlannerConfig.STRATEGIES["safe"]  # min_chip_marginal_xpts = 4.0

        placements, details = determine_chip_placements(
            available, recs, _make_squad(), 2.0,
            _make_fixtures(), _make_events(), _teams_dict(),
            _make_pool(), {}, 24, 28, config, {},
        )

        assert len(details) == 0, "Chips placed despite being below threshold"

    @patch("fpl_assistant.planner.calculate_chip_marginal_value")
    def test_no_two_chips_same_gw(self, mock_calc):
        """Two chips should not be placed on the same GW."""
        mock_calc.return_value = (8.0, None)

        available = {"wildcard": False, "freehit": False, "bboost": True, "3xc": True}
        recs = [
            ChipRecommendation("bboost", 25, "high", "DGW", 8.0),
            ChipRecommendation("3xc", 25, "high", "DGW captain", 8.0),
        ]
        config = TransferPlannerConfig.STRATEGIES["risky"]

        placements, details = determine_chip_placements(
            available, recs, _make_squad(), 2.0,
            _make_fixtures(), _make_events(), _teams_dict(),
            _make_pool(), {}, 24, 28, config, {},
        )

        placed_gws = [d.gw for d in details]
        assert len(placed_gws) == len(set(placed_gws)), "Two chips placed on same GW"


# ============ TESTS: StrategyPlan model ============

class TestStrategyPlanModel:
    def test_chip_placements_field_exists(self):
        """StrategyPlan should have chip_placements field."""
        plan = StrategyPlan(
            name="test", description="test", gw_actions={},
            total_xpts=50.0, hit_cost=0, transfers_made=2,
            chip_recommendations=[], risk_score=5, headline="test",
        )
        assert hasattr(plan, "chip_placements")
        assert plan.chip_placements == []

    def test_chip_placements_populated(self):
        """StrategyPlan should store ChipPlacement objects."""
        cp = ChipPlacement(chip="bboost", gw=26, marginal_xpts=5.2, reason="DGW")
        plan = StrategyPlan(
            name="test", description="test", gw_actions={},
            total_xpts=50.0, hit_cost=0, transfers_made=2,
            chip_recommendations=[], risk_score=5, headline="test",
            chip_placements=[cp],
        )
        assert len(plan.chip_placements) == 1
        assert plan.chip_placements[0].chip == "bboost"
        assert plan.chip_placements[0].gw == 26


# ============ TESTS: TransferPlannerConfig ============

class TestChipConfig:
    def test_all_strategies_have_chip_threshold(self):
        """Every strategy should have min_chip_marginal_xpts."""
        for name, config in TransferPlannerConfig.STRATEGIES.items():
            assert "min_chip_marginal_xpts" in config, f"{name} missing min_chip_marginal_xpts"

    def test_safe_threshold_highest(self):
        """Safe strategy should have the highest chip threshold."""
        safe = TransferPlannerConfig.STRATEGIES["safe"]["min_chip_marginal_xpts"]
        balanced = TransferPlannerConfig.STRATEGIES["balanced"]["min_chip_marginal_xpts"]
        risky = TransferPlannerConfig.STRATEGIES["risky"]["min_chip_marginal_xpts"]
        assert safe > balanced > risky
