"""Shared fixtures for FPL test suite."""
import pytest


@pytest.fixture
def make_player():
    """Factory for creating player dicts matching FPL API shape."""
    def _make(**overrides):
        base = {
            "id": 1,
            "web_name": "TestPlayer",
            "team": 1,
            "element_type": 3,  # MID
            "now_cost": 70,  # £7.0m
            "minutes": 1800,  # 20 full games
            "goals_scored": 5,
            "assists": 4,
            "clean_sheets": 6,
            "bonus": 12,
            "bps": 450,
            "saves": 0,
            "expected_goals": "4.50",
            "expected_assists": "3.80",
            "expected_goal_involvements": "8.30",
            "points_per_game": "5.2",
            "total_points": 104,
            "yellow_cards": 3,
            "red_cards": 0,
            "own_goals": 0,
            "defensive_contribution_per_90": "8.5",
            "defensive_contribution": 170,
            "starts": 18,
        }
        base.update(overrides)
        return base
    return _make


@pytest.fixture
def make_player_history():
    """Factory for creating player GW history entries."""
    def _make(entries=None):
        if entries is None:
            entries = []
        default_entry = {
            "round": 1,
            "minutes": 90,
            "total_points": 5,
            "goals_scored": 0,
            "assists": 0,
            "clean_sheets": 0,
            "expected_goals": "0.25",
            "expected_assists": "0.20",
            "was_home": True,
        }
        result = []
        for i, entry in enumerate(entries):
            row = dict(default_entry)
            row["round"] = i + 1
            row.update(entry)
            result.append(row)
        return {"history": result}
    return _make


@pytest.fixture
def make_fixture():
    """Factory for creating fixture dicts."""
    def _make(**overrides):
        base = {
            "id": 1,
            "event": 24,
            "team_h": 1,
            "team_a": 2,
            "team_h_difficulty": 3,
            "team_a_difficulty": 3,
            "finished": False,
        }
        base.update(overrides)
        return base
    return _make
