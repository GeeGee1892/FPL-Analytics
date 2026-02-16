"""Tests for player stat functions: bonus, DEFCON, saves, form factor."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    calculate_expected_bonus,
    calculate_defcon_per_90,
    calculate_saves_per_90,
    calculate_player_form_factor,
    PlayerFormResult,
)

import pytest


# =============================================================================
# calculate_expected_bonus
# =============================================================================

class TestExpectedBonus:
    def test_fwd_high_xgi(self, make_player):
        """FWD with high goals/assists should have high bonus."""
        player = make_player(
            minutes=1800, goals_scored=12, assists=5, bonus=25, bps=600,
            expected_goals="11.0", expected_assists="4.5",
        )
        hist, pred = calculate_expected_bonus(player, position=4)
        assert pred > 1.0

    def test_mid_with_assists(self, make_player):
        """MID with decent assists should have moderate bonus."""
        player = make_player(
            minutes=1800, goals_scored=3, assists=8, bonus=15, bps=420,
            expected_goals="3.0", expected_assists="7.0",
        )
        hist, pred = calculate_expected_bonus(player, position=3)
        assert 0.5 < pred < 2.0

    def test_def_with_clean_sheets(self, make_player):
        """DEF with CS should get bonus from defensive BPS."""
        player = make_player(
            minutes=1800, goals_scored=1, assists=2, clean_sheets=10,
            bonus=14, bps=500, expected_goals="0.8", expected_assists="1.5",
        )
        hist, pred = calculate_expected_bonus(player, position=2)
        assert pred > 0.3

    def test_gk_baseline(self, make_player):
        """GK with saves should get moderate bonus."""
        player = make_player(
            minutes=1800, goals_scored=0, assists=0, clean_sheets=8,
            bonus=10, bps=380, saves=60,
            expected_goals="0.0", expected_assists="0.1",
        )
        hist, pred = calculate_expected_bonus(player, position=1)
        assert pred > 0.2

    def test_low_minutes_returns_baseline(self, make_player):
        """Player with <180 minutes should get position baseline."""
        player = make_player(minutes=90, goals_scored=0, assists=0, bonus=0, bps=10)
        hist, pred = calculate_expected_bonus(player, position=4)
        assert hist == 0.65  # FWD baseline
        assert pred == 0.65

    def test_zero_minutes(self, make_player):
        player = make_player(minutes=0)
        hist, pred = calculate_expected_bonus(player, position=3)
        assert hist == 0.55  # MID baseline

    def test_capped_at_2_5(self, make_player):
        """Bonus should never exceed 2.5 per 90."""
        player = make_player(
            minutes=1800, goals_scored=20, assists=10, bonus=50, bps=900,
            expected_goals="18.0", expected_assists="9.0",
        )
        _, pred = calculate_expected_bonus(player, position=4)
        assert pred <= 2.5

    def test_minimum_0_1(self, make_player):
        """Bonus should never be below 0.1."""
        player = make_player(
            minutes=1800, goals_scored=0, assists=0, bonus=1, bps=50,
            expected_goals="0.1", expected_assists="0.1",
        )
        _, pred = calculate_expected_bonus(player, position=3)
        assert pred >= 0.1

    def test_overperformer_regression(self, make_player):
        """Player significantly overperforming xGI should get bonus regression."""
        player = make_player(
            minutes=1800, goals_scored=15, assists=8, bonus=30, bps=700,
            expected_goals="8.0", expected_assists="4.0",  # Over by ~2x
        )
        _, pred = calculate_expected_bonus(player, position=4)
        # Should apply 0.92 regression factor
        hist_bonus = 30 / (1800 / 90.0)  # 1.5
        assert pred < hist_bonus * 1.1  # Shouldn't exceed historical by much

    def test_returns_tuple(self, make_player):
        player = make_player()
        result = calculate_expected_bonus(player, position=3)
        assert isinstance(result, tuple)
        assert len(result) == 2


# =============================================================================
# calculate_defcon_per_90
# =============================================================================

class TestDefconPer90:
    def test_def_with_high_defcon(self, make_player):
        """DEF with high DEFCON per 90 should have high probability."""
        player = make_player(
            minutes=1800,
            defensive_contribution_per_90="14.0",
            defensive_contribution=280,
        )
        per90, prob, total = calculate_defcon_per_90(player, position_id=2)
        assert per90 == 14.0
        assert prob > 0.5
        assert total == 280

    def test_mid_higher_threshold(self, make_player):
        """MID needs 12 DEFCON (vs 10 for DEF) so same per90 = lower prob."""
        player = make_player(
            minutes=1800,
            defensive_contribution_per_90="11.0",
            defensive_contribution=220,
        )
        def_per90, def_prob, _ = calculate_defcon_per_90(player, position_id=2)
        mid_per90, mid_prob, _ = calculate_defcon_per_90(player, position_id=3)
        # DEF threshold is 10, MID is 12, so with per90=11, DEF should have higher prob
        assert def_prob > mid_prob

    def test_fwd_returns_zero(self, make_player):
        """FWD (position 4) should return zeros — DEFCON is DEF/MID only."""
        player = make_player(minutes=1800, defensive_contribution_per_90="5.0")
        per90, prob, total = calculate_defcon_per_90(player, position_id=4)
        assert per90 == 0.0
        assert prob == 0.0

    def test_gk_returns_zero(self, make_player):
        """GK (position 1) should return zeros."""
        player = make_player(minutes=1800, defensive_contribution_per_90="5.0")
        per90, prob, total = calculate_defcon_per_90(player, position_id=1)
        assert per90 == 0.0
        assert prob == 0.0

    def test_low_minutes_returns_zero(self, make_player):
        """Player with <90 minutes should return zeros."""
        player = make_player(minutes=60, defensive_contribution_per_90="15.0")
        per90, prob, total = calculate_defcon_per_90(player, position_id=2)
        assert per90 == 0.0
        assert prob == 0.0

    def test_workload_dampening_elite_team(self, make_player):
        """Elite defensive team (low xGA) should dampen DEFCON probability."""
        player = make_player(
            minutes=1800,
            defensive_contribution_per_90="12.0",
            defensive_contribution=240,
        )
        prob_avg = calculate_defcon_per_90(player, position_id=2, team_xga=1.35)[1]
        prob_elite = calculate_defcon_per_90(player, position_id=2, team_xga=0.80)[1]
        assert prob_elite < prob_avg

    def test_workload_dampening_weak_team(self, make_player):
        """Weak defensive team (high xGA) should boost DEFCON probability."""
        player = make_player(
            minutes=1800,
            defensive_contribution_per_90="12.0",
            defensive_contribution=240,
        )
        prob_avg = calculate_defcon_per_90(player, position_id=2, team_xga=1.35)[1]
        prob_weak = calculate_defcon_per_90(player, position_id=2, team_xga=1.80)[1]
        assert prob_weak > prob_avg

    def test_prob_capped(self, make_player):
        """Probability should be capped at 0.85 for DEF."""
        player = make_player(
            minutes=1800,
            defensive_contribution_per_90="25.0",
            defensive_contribution=500,
        )
        _, prob, _ = calculate_defcon_per_90(player, position_id=2)
        assert prob <= 0.85

    def test_prob_has_floor(self, make_player):
        """Even with low DEFCON, probability should have a floor of 0.02."""
        player = make_player(
            minutes=1800,
            defensive_contribution_per_90="3.0",
            defensive_contribution=60,
        )
        _, prob, _ = calculate_defcon_per_90(player, position_id=2)
        assert prob >= 0.02


# =============================================================================
# calculate_saves_per_90
# =============================================================================

class TestSavesPer90:
    def test_gk_with_saves(self, make_player):
        """GK with saves should return positive saves/90 and save points."""
        player = make_player(minutes=1800, saves=60)
        sav_per_90, sav_pts = calculate_saves_per_90(player)
        assert sav_per_90 == 3.0  # 60 / 20 = 3.0
        assert sav_pts == 1.0  # 3.0 / 3 = 1.0

    def test_high_saves(self, make_player):
        player = make_player(minutes=1800, saves=90)
        sav_per_90, sav_pts = calculate_saves_per_90(player)
        assert sav_per_90 == 4.5
        assert sav_pts == 1.5

    def test_low_minutes_returns_zero(self, make_player):
        """<90 minutes should return zeros."""
        player = make_player(minutes=45, saves=10)
        sav_per_90, sav_pts = calculate_saves_per_90(player)
        assert sav_per_90 == 0.0
        assert sav_pts == 0.0

    def test_no_saves_returns_zero(self, make_player):
        player = make_player(minutes=1800, saves=0)
        sav_per_90, sav_pts = calculate_saves_per_90(player)
        assert sav_per_90 == 0.0
        assert sav_pts == 0.0


# =============================================================================
# calculate_player_form_factor
# =============================================================================

class TestPlayerFormFactor:
    def test_no_history_returns_default(self, make_player):
        """No history should return default form (all 1.0)."""
        player = make_player(minutes=1800)
        result = calculate_player_form_factor(player, None, position=3)
        assert result.combined_form == 1.0
        assert result.confidence == 0.0

    def test_empty_history(self, make_player, make_player_history):
        """Empty history should return default."""
        player = make_player(minutes=1800)
        history = make_player_history([])
        result = calculate_player_form_factor(player, history, position=3)
        assert result.combined_form == 1.0

    def test_low_minutes_returns_default(self, make_player):
        """Player with <270 minutes should return default."""
        player = make_player(minutes=180)
        result = calculate_player_form_factor(player, None, position=3)
        assert result.combined_form == 1.0

    def test_trending_up(self, make_player, make_player_history):
        """Player with recent points > season average should have form > 1."""
        player = make_player(
            minutes=1800, points_per_game="5.0",
            expected_goals="4.50", expected_assists="3.60",
        )
        # Recent games: 8 PPG (above 5.0 season avg)
        recent = [
            {"minutes": 90, "total_points": 8, "expected_goals": "0.40", "expected_assists": "0.30"}
            for _ in range(6)
        ]
        history = make_player_history(recent)
        result = calculate_player_form_factor(player, history, position=3)
        assert result.combined_form > 1.0

    def test_trending_down(self, make_player, make_player_history):
        """Player with recent points < season average should have form < 1."""
        player = make_player(
            minutes=1800, points_per_game="6.0",
            expected_goals="5.40", expected_assists="3.60",
        )
        # Recent games: 2 PPG (well below 6.0 season avg)
        recent = [
            {"minutes": 90, "total_points": 2, "expected_goals": "0.10", "expected_assists": "0.05"}
            for _ in range(6)
        ]
        history = make_player_history(recent)
        result = calculate_player_form_factor(player, history, position=3)
        assert result.combined_form < 1.0

    def test_short_history_lower_confidence(self, make_player, make_player_history):
        """3 games (minimum) should have lower confidence than 6 games."""
        player = make_player(
            minutes=1800, points_per_game="5.0",
            expected_goals="4.50", expected_assists="3.60",
        )
        entry = {"minutes": 90, "total_points": 8, "expected_goals": "0.40", "expected_assists": "0.30"}
        hist_3 = make_player_history([entry] * 3)
        hist_6 = make_player_history([entry] * 6)

        result_3 = calculate_player_form_factor(player, hist_3, position=3)
        result_6 = calculate_player_form_factor(player, hist_6, position=3)
        assert result_6.confidence >= result_3.confidence

    def test_fwd_full_sensitivity(self, make_player, make_player_history):
        """FWD has sensitivity 1.0 (highest), so form effect should be strongest."""
        player = make_player(
            minutes=1800, points_per_game="5.0",
            expected_goals="9.0", expected_assists="1.80",
        )
        recent = [
            {"minutes": 90, "total_points": 10, "expected_goals": "0.80", "expected_assists": "0.10"}
            for _ in range(6)
        ]
        history = make_player_history(recent)

        fwd_result = calculate_player_form_factor(player, history, position=4)
        gkp_result = calculate_player_form_factor(player, history, position=1)
        # FWD should have higher confidence due to higher sensitivity
        assert fwd_result.confidence > gkp_result.confidence

    def test_returns_player_form_result(self, make_player):
        player = make_player(minutes=1800)
        result = calculate_player_form_factor(player, None, position=3)
        assert isinstance(result, PlayerFormResult)

    def test_form_bounded(self, make_player, make_player_history):
        """Combined form should stay within configured bounds."""
        player = make_player(
            minutes=1800, points_per_game="5.0",
            expected_goals="4.50", expected_assists="3.60",
        )
        # Extreme hot streak
        recent = [
            {"minutes": 90, "total_points": 20, "expected_goals": "1.50", "expected_assists": "1.00"}
            for _ in range(6)
        ]
        history = make_player_history(recent)
        result = calculate_player_form_factor(player, history, position=3)
        assert result.combined_form <= 1.50  # max for premiums
        assert result.combined_form >= 0.55  # min for premiums
