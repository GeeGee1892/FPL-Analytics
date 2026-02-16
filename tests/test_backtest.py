"""Tests for backtest validation pipeline."""
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fpl_assistant.backtest import (
    run_backtest_sync,
    _compute_correlation,
    _compute_segment_stats,
    _compute_component_accuracy,
    backtest_result_to_dict,
)
from fpl_assistant.models import (
    BacktestPrediction, BacktestResult, SegmentStats, ComponentAccuracy,
)
from main import DataCache, POSITION_MAP


# ============ FIXTURES ============

def _make_prediction(
    pid=1, name="Player1", team="ARS", position="MID", pos_id=3,
    price=7.5, gw=5, predicted=4.5, actual=3, minutes=90,
    goals=0, assists=0, cs=0, bonus=0,
    opponent_id=2, opponent="CHE", is_home=True, fdr=5,
    predicted_minutes=85.0,
):
    return BacktestPrediction(
        player_id=pid, player_name=name, team_short=team,
        position=position, position_id=pos_id, price=price,
        gameweek=gw, predicted_xpts=predicted, predicted_minutes=predicted_minutes,
        actual_points=actual, actual_minutes=minutes,
        actual_goals=goals, actual_assists=assists,
        actual_cs=cs, actual_bonus=bonus,
        opponent_id=opponent_id, opponent_name=opponent,
        is_home=is_home, fdr=fdr,
    )


def _make_element(pid, team, pos, price_tenths, minutes=1800, form="5.0"):
    return {
        "id": pid, "web_name": f"Player{pid}", "team": team,
        "element_type": pos, "now_cost": price_tenths,
        "minutes": minutes, "goals_scored": 5, "assists": 3,
        "clean_sheets": 4, "bonus": 10, "bps": 400,
        "saves": 30 if pos == 1 else 0,
        "expected_goals": "3.0", "expected_assists": "2.0",
        "expected_goal_involvements": "5.0",
        "points_per_game": "5.0", "total_points": 90,
        "yellow_cards": 2, "red_cards": 0, "own_goals": 0,
        "defensive_contribution_per_90": "8.0",
        "defensive_contribution": 160,
        "starts": 18, "status": "a", "form": form,
    }


def _make_history(gw_entries):
    """Create player history from list of (gw, points, minutes) tuples."""
    history = []
    for gw, pts, mins in gw_entries:
        history.append({
            "round": gw, "total_points": pts, "minutes": mins,
            "goals_scored": 0, "assists": 0, "clean_sheets": 0,
            "bonus": 0, "saves": 0, "yellow_cards": 0,
            "expected_goals": "0.3", "expected_assists": "0.2",
            "was_home": True, "opponent_team": 2,
        })
    return {"history": history}


def _teams_dict():
    return {i: {"name": f"Team{i}", "short_name": f"T{i:02d}"} for i in range(1, 21)}


def _events(finished_through=5):
    """Create events where GWs 1..finished_through are finished."""
    return [
        {"id": gw, "finished": gw <= finished_through, "is_current": gw == finished_through + 1}
        for gw in range(1, 39)
    ]


def _fixtures():
    """Basic fixtures for GW1-10."""
    fixtures = []
    fid = 1
    for gw in range(1, 11):
        for h in range(1, 11):
            a = h + 10
            fixtures.append({
                "id": fid, "event": gw,
                "team_h": h, "team_a": a,
                "team_h_difficulty": 3, "team_a_difficulty": 3,
                "finished": gw <= 5,
            })
            fid += 1
    return fixtures


def _make_test_cache():
    test_cache = DataCache()
    test_cache.fdr_data = {
        i: {
            "attack_fdr": 5, "defence_fdr": 5,
            "xg": 1.3, "xga": 1.3, "blended_xga": 1.3,
            "cs_probability": 0.30,
            "home_xga": 1.1, "away_xga": 1.5,
        }
        for i in range(1, 21)
    }
    return test_cache


# ============ BacktestPrediction TESTS ============

class TestBacktestPrediction:

    def test_error_positive_overestimate(self):
        p = _make_prediction(predicted=5.0, actual=3)
        assert p.error == pytest.approx(2.0)

    def test_error_negative_underestimate(self):
        p = _make_prediction(predicted=2.0, actual=5)
        assert p.error == pytest.approx(-3.0)

    def test_abs_error(self):
        p = _make_prediction(predicted=2.0, actual=5)
        assert p.abs_error == pytest.approx(3.0)

    def test_price_tier_premium(self):
        p = _make_prediction(price=12.0)
        assert p.price_tier == "premium"

    def test_price_tier_mid(self):
        p = _make_prediction(price=8.0)
        assert p.price_tier == "mid_price"

    def test_price_tier_budget(self):
        p = _make_prediction(price=5.5)
        assert p.price_tier == "budget"

    def test_price_tier_fodder(self):
        p = _make_prediction(price=4.0)
        assert p.price_tier == "fodder"

    def test_fdr_bucket_easy(self):
        p = _make_prediction(fdr=2)
        assert p.fdr_bucket == "easy"

    def test_fdr_bucket_medium(self):
        p = _make_prediction(fdr=5)
        assert p.fdr_bucket == "medium"

    def test_fdr_bucket_hard(self):
        p = _make_prediction(fdr=8)
        assert p.fdr_bucket == "hard"


# ============ CORRELATION TESTS ============

class TestComputeCorrelation:

    def test_perfect_positive(self):
        r = _compute_correlation([1, 2, 3], [1, 2, 3])
        assert r == pytest.approx(1.0)

    def test_perfect_negative(self):
        r = _compute_correlation([1, 2, 3], [3, 2, 1])
        assert r == pytest.approx(-1.0)

    def test_insufficient_data(self):
        r = _compute_correlation([1, 2], [3, 4])
        assert r is None

    def test_constant_values(self):
        r = _compute_correlation([5, 5, 5], [1, 2, 3])
        assert r is None


# ============ SEGMENT STATS TESTS ============

class TestComputeSegmentStats:

    def test_basic_stats(self):
        preds = [
            _make_prediction(predicted=5.0, actual=3),  # error = +2
            _make_prediction(predicted=3.0, actual=5),  # error = -2
            _make_prediction(predicted=4.0, actual=4),  # error = 0
        ]
        s = _compute_segment_stats("test", preds)
        assert s.count == 3
        assert s.mae == pytest.approx(1.33, abs=0.01)
        assert s.avg_error == pytest.approx(0.0, abs=0.01)

    def test_perfect_predictions(self):
        preds = [
            _make_prediction(predicted=3.0, actual=3),
            _make_prediction(predicted=5.0, actual=5),
        ]
        s = _compute_segment_stats("perfect", preds)
        assert s.mae == pytest.approx(0.0)
        assert s.rmse == pytest.approx(0.0)

    def test_positive_bias(self):
        preds = [
            _make_prediction(predicted=6.0, actual=3),
            _make_prediction(predicted=5.0, actual=2),
        ]
        s = _compute_segment_stats("bias", preds)
        assert s.avg_error > 0  # Overestimate

    def test_single_prediction(self):
        preds = [_make_prediction(predicted=5.0, actual=3)]
        s = _compute_segment_stats("single", preds)
        assert s.count == 1
        assert s.mae == pytest.approx(2.0)

    def test_empty_predictions(self):
        s = _compute_segment_stats("empty", [])
        assert s.count == 0
        assert s.mae == 0


# ============ COMPONENT ACCURACY TESTS ============

class TestComputeComponentAccuracy:

    def test_returns_components(self):
        preds = [
            _make_prediction(cs=1, bonus=3, goals=1, assists=0, position="DEF", pos_id=2),
            _make_prediction(cs=0, bonus=0, goals=0, assists=1, position="DEF", pos_id=2),
            _make_prediction(cs=1, bonus=1, goals=0, assists=0, position="DEF", pos_id=2),
        ]
        comps = _compute_component_accuracy(preds)
        names = {c.component for c in comps}
        assert "clean_sheets" in names
        assert "bonus" in names
        assert "goals" in names
        assert "assists" in names

    def test_cs_actual_rate(self):
        preds = [
            _make_prediction(cs=1, position="DEF", pos_id=2),
            _make_prediction(cs=0, position="DEF", pos_id=2),
            _make_prediction(cs=1, position="DEF", pos_id=2),
            _make_prediction(cs=0, position="DEF", pos_id=2),
        ]
        comps = _compute_component_accuracy(preds)
        cs = next(c for c in comps if c.component == "clean_sheets")
        assert cs.actual_rate == pytest.approx(0.5)

    def test_empty_predictions(self):
        comps = _compute_component_accuracy([])
        assert comps == []


# ============ RUN BACKTEST SYNC TESTS ============

class TestRunBacktestSync:

    @patch("fpl_assistant.backtest.calculate_expected_points")
    @patch("fpl_assistant.services.cache", _make_test_cache())
    def test_basic_backtest(self, mock_calc):
        """Basic backtest produces predictions."""
        mock_calc.return_value = {
            "xpts": 4.5, "expected_minutes": 85, "cs_prob": 0.3,
            "xG_per_90": 0.4, "xA_per_90": 0.2, "expected_bonus": 0.8,
        }
        elements = [_make_element(1, 1, 3, 70)]  # MID, team 1
        histories = {1: _make_history([(1, 5, 90), (2, 3, 80), (3, 7, 90)])}

        result = run_backtest_sync(
            elements=elements, fixtures=_fixtures(),
            teams_dict=_teams_dict(), events=_events(finished_through=3),
            player_histories=histories, gw_start=1, gw_end=3,
        )
        assert isinstance(result, BacktestResult)
        assert result.total_predictions == 3
        assert result.overall_mae > 0

    @patch("fpl_assistant.backtest.calculate_expected_points")
    @patch("fpl_assistant.services.cache", _make_test_cache())
    def test_filters_by_position(self, mock_calc):
        mock_calc.return_value = {"xpts": 4.0, "expected_minutes": 80}
        elements = [
            _make_element(1, 1, 2, 55),  # DEF
            _make_element(2, 2, 3, 70),  # MID
        ]
        histories = {
            1: _make_history([(1, 5, 90)]),
            2: _make_history([(1, 3, 90)]),
        }
        result = run_backtest_sync(
            elements=elements, fixtures=_fixtures(),
            teams_dict=_teams_dict(), events=_events(finished_through=1),
            player_histories=histories, gw_start=1, gw_end=1,
            position_filter="DEF",
        )
        assert result.total_predictions == 1
        assert result.predictions[0].position == "DEF"

    @patch("fpl_assistant.backtest.calculate_expected_points")
    @patch("fpl_assistant.services.cache", _make_test_cache())
    def test_filters_by_min_minutes(self, mock_calc):
        mock_calc.return_value = {"xpts": 4.0, "expected_minutes": 80}
        elements = [
            _make_element(1, 1, 3, 70, minutes=500),  # Passes
            _make_element(2, 2, 3, 70, minutes=100),  # Fails min_season_minutes
        ]
        histories = {
            1: _make_history([(1, 5, 90)]),
            2: _make_history([(1, 3, 90)]),
        }
        result = run_backtest_sync(
            elements=elements, fixtures=_fixtures(),
            teams_dict=_teams_dict(), events=_events(finished_through=1),
            player_histories=histories, gw_start=1, gw_end=1,
            min_season_minutes=200,
        )
        assert result.total_predictions == 1

    @patch("fpl_assistant.backtest.calculate_expected_points")
    @patch("fpl_assistant.services.cache", _make_test_cache())
    def test_skips_zero_gw_minutes(self, mock_calc):
        mock_calc.return_value = {"xpts": 4.0, "expected_minutes": 80}
        elements = [_make_element(1, 1, 3, 70)]
        histories = {1: _make_history([(1, 0, 0)])}  # 0 minutes in GW1

        result = run_backtest_sync(
            elements=elements, fixtures=_fixtures(),
            teams_dict=_teams_dict(), events=_events(finished_through=1),
            player_histories=histories, gw_start=1, gw_end=1,
            min_gw_minutes=45,
        )
        assert result.total_predictions == 0

    def test_unfinished_gw_skipped(self):
        elements = [_make_element(1, 1, 3, 70)]
        histories = {1: _make_history([(10, 5, 90)])}

        result = run_backtest_sync(
            elements=elements, fixtures=_fixtures(),
            teams_dict=_teams_dict(), events=_events(finished_through=5),
            player_histories=histories, gw_start=8, gw_end=10,
        )
        assert result.total_predictions == 0

    @patch("fpl_assistant.backtest.calculate_expected_points")
    @patch("fpl_assistant.services.cache", _make_test_cache())
    def test_segmentations_present(self, mock_calc):
        mock_calc.return_value = {"xpts": 4.0, "expected_minutes": 80}
        elements = [
            _make_element(1, 1, 2, 55),   # DEF budget
            _make_element(2, 2, 3, 120),   # MID premium
        ]
        histories = {
            1: _make_history([(1, 5, 90)]),
            2: _make_history([(1, 8, 90)]),
        }
        result = run_backtest_sync(
            elements=elements, fixtures=_fixtures(),
            teams_dict=_teams_dict(), events=_events(finished_through=1),
            player_histories=histories, gw_start=1, gw_end=1,
        )
        assert len(result.by_position) > 0
        assert len(result.by_price_tier) > 0
        assert len(result.by_gameweek) > 0


# ============ SERIALIZATION TESTS ============

class TestBacktestResultToDict:

    def test_all_keys_present(self):
        result = BacktestResult(
            gw_start=1, gw_end=5, total_predictions=100,
            overall_mae=2.5, overall_rmse=3.1, overall_correlation=0.35,
            overall_bias=0.3,
            by_position=[SegmentStats("DEF", 50, 2.0, 2.5, 0.2, 0.4)],
            by_price_tier=[SegmentStats("premium", 30, 2.8, 3.2, 0.5, 0.3)],
            by_fdr=[SegmentStats("easy", 40, 2.1, 2.6, 0.1, 0.45)],
            by_gameweek=[SegmentStats("GW1", 20, 2.3, 2.9, 0.0, 0.38)],
            component_accuracy=[ComponentAccuracy("bonus", 0.8, 0.7, 0.1, 100)],
            predictions=[_make_prediction()],
            look_ahead_bias=True,
            timestamp="2025-01-01T00:00:00Z",
        )
        d = backtest_result_to_dict(result)
        assert "overall" in d
        assert "by_position" in d
        assert "by_price_tier" in d
        assert "by_fdr" in d
        assert "by_gameweek" in d
        assert "component_accuracy" in d
        assert "worst_predictions" in d
        assert "best_predictions" in d
        assert d["total_predictions"] == 100
        assert d["look_ahead_bias"] is True

    def test_worst_sorted_by_abs_error(self):
        preds = [
            _make_prediction(pid=1, predicted=2.0, actual=10),  # error = -8
            _make_prediction(pid=2, predicted=5.0, actual=4),   # error = 1
            _make_prediction(pid=3, predicted=1.0, actual=15),  # error = -14
        ]
        result = BacktestResult(
            gw_start=1, gw_end=1, total_predictions=3,
            overall_mae=0, overall_rmse=0, overall_correlation=None, overall_bias=0,
            by_position=[], by_price_tier=[], by_fdr=[], by_gameweek=[],
            component_accuracy=[], predictions=preds,
            look_ahead_bias=True, timestamp="2025-01-01T00:00:00Z",
        )
        d = backtest_result_to_dict(result)
        worst = d["worst_predictions"]
        assert worst[0]["id"] == 3  # Biggest abs error first
        assert worst[1]["id"] == 1
