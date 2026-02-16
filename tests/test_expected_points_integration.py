"""Integration tests for calculate_expected_points with cache mocking."""
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import calculate_expected_points, DataCache, MODEL_CONFIG

import pytest


def _make_teams_dict():
    """Create a minimal teams dict for testing. Format: {id: {"name": ..., ...}}."""
    return {
        1: {"name": "Arsenal", "short_name": "ARS"},
        2: {"name": "Aston Villa", "short_name": "AVL"},
        3: {"name": "Burnley", "short_name": "BUR"},
        6: {"name": "Brighton", "short_name": "BHA"},
        13: {"name": "Man City", "short_name": "MCI"},
        14: {"name": "Man Utd", "short_name": "MUN"},
    }


def _make_upcoming_fixtures(team_id, is_home=True, opponent_id=3, gw=24):
    """Create a single upcoming fixture with the keys that calculate_expected_points expects."""
    return [{
        "opponent_id": opponent_id,
        "is_home": is_home,
        "gameweek": gw,
        "event": gw,
    }]


def _make_test_cache():
    """Create a real DataCache with populated fdr_data."""
    test_cache = DataCache()
    test_cache.fdr_data = {
        1: {
            "attack_fdr": 2, "defence_fdr": 9,
            "xg": 1.78, "xga": 0.82, "blended_xga": 0.82,
            "cs_probability": 0.42,
            "home_xga": 0.68, "away_xga": 0.96,
        },
        3: {
            "attack_fdr": 8, "defence_fdr": 1,
            "xg": 0.85, "xga": 1.82, "blended_xga": 1.82,
            "cs_probability": 0.10,
            "home_xga": 1.60, "away_xga": 2.04,
        },
        13: {
            "attack_fdr": 2, "defence_fdr": 10,
            "xg": 1.88, "xga": 1.17, "blended_xga": 1.17,
            "cs_probability": 0.28,
            "home_xga": 1.00, "away_xga": 1.34,
        },
    }
    return test_cache


class TestCalculateExpectedPointsIntegration:
    @patch("main.cache", _make_test_cache())
    def test_gk_produces_valid_output(self, make_player):
        """GK xPts should include CS probability and save points."""
        player = make_player(
            id=100, team=1, element_type=1, now_cost=50,
            minutes=1800, goals_scored=0, assists=0,
            clean_sheets=10, saves=60, bonus=8, bps=350,
            expected_goals="0.0", expected_assists="0.1",
            expected_goal_involvements="0.1",
            points_per_game="4.5", defensive_contribution_per_90="0",
            defensive_contribution=0,
        )
        result = calculate_expected_points(
            player=player,
            position=1,
            current_gw=24,
            upcoming_fixtures=_make_upcoming_fixtures(1, is_home=True, opponent_id=3),
            teams_dict=_make_teams_dict(),
            override_minutes=85.0,
        )
        assert "xpts" in result
        assert result["xpts"] >= 0
        assert result["xpts"] <= 15

    @patch("main.cache", _make_test_cache())
    def test_def_produces_valid_output(self, make_player):
        """DEF xPts should include CS, DEFCON, and goal threat."""
        player = make_player(
            id=101, team=1, element_type=2, now_cost=60,
            minutes=1800, goals_scored=3, assists=2,
            clean_sheets=10, saves=0, bonus=15, bps=500,
            expected_goals="2.5", expected_assists="1.8",
            expected_goal_involvements="4.3",
            points_per_game="5.0", defensive_contribution_per_90="12.0",
            defensive_contribution=240,
        )
        result = calculate_expected_points(
            player=player,
            position=2,
            current_gw=24,
            upcoming_fixtures=_make_upcoming_fixtures(1, is_home=True, opponent_id=3),
            teams_dict=_make_teams_dict(),
            override_minutes=85.0,
        )
        assert result["xpts"] >= 0
        assert result["xpts"] <= 15

    @patch("main.cache", _make_test_cache())
    def test_mid_produces_valid_output(self, make_player):
        """MID xPts should include goal, assist, and bonus components."""
        player = make_player(
            id=102, team=1, element_type=3, now_cost=120,
            minutes=1800, goals_scored=10, assists=8,
            clean_sheets=6, saves=0, bonus=25, bps=650,
            expected_goals="9.0", expected_assists="7.0",
            expected_goal_involvements="16.0",
            points_per_game="7.5", defensive_contribution_per_90="6.0",
            defensive_contribution=120,
        )
        result = calculate_expected_points(
            player=player,
            position=3,
            current_gw=24,
            upcoming_fixtures=_make_upcoming_fixtures(1, is_home=True, opponent_id=3),
            teams_dict=_make_teams_dict(),
            override_minutes=85.0,
        )
        assert result["xpts"] >= 0
        assert result["xpts"] <= 15

    @patch("main.cache", _make_test_cache())
    def test_fwd_produces_valid_output(self, make_player):
        """FWD xPts should include goals, assists, and bonus."""
        player = make_player(
            id=103, team=13, element_type=4, now_cost=140,
            minutes=1800, goals_scored=18, assists=3,
            clean_sheets=0, saves=0, bonus=30, bps=700,
            expected_goals="16.0", expected_assists="2.5",
            expected_goal_involvements="18.5",
            points_per_game="8.0", defensive_contribution_per_90="2.0",
            defensive_contribution=40,
        )
        result = calculate_expected_points(
            player=player,
            position=4,
            current_gw=24,
            upcoming_fixtures=_make_upcoming_fixtures(13, is_home=True, opponent_id=3),
            teams_dict=_make_teams_dict(),
            override_minutes=85.0,
        )
        assert result["xpts"] >= 0
        assert result["xpts"] <= 15

    @patch("main.cache", _make_test_cache())
    def test_xpts_nonnegative_for_all_positions(self, make_player):
        """xPts should always be non-negative across all positions."""
        for pos in [1, 2, 3, 4]:
            player = make_player(
                id=200 + pos, team=1, element_type=pos,
                minutes=1800, now_cost=70,
            )
            result = calculate_expected_points(
                player=player,
                position=pos,
                current_gw=24,
                upcoming_fixtures=_make_upcoming_fixtures(1, is_home=True, opponent_id=3),
                teams_dict=_make_teams_dict(),
                override_minutes=85.0,
            )
            assert result["xpts"] >= 0, f"Negative xPts for position {pos}"
