"""Tests for pure FDR, math, and utility functions in main.py."""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    _generate_fixture_weights,
    xga_to_attack_fdr,
    xg_to_defence_fdr,
    get_price_adjusted_fdr_multiplier,
    calculate_goals_conceded_penalty,
    percentileofscore,
    get_set_piece_share,
    apply_set_piece_share_to_multiplier,
    calculate_roll_value,
    MODEL_CONFIG,
)

import pytest


# =============================================================================
# _generate_fixture_weights
# =============================================================================

class TestGenerateFixtureWeights:
    def test_weights_sum_to_one(self):
        weights = _generate_fixture_weights(2.0, 8)
        assert abs(sum(weights) - 1.0) < 0.01

    def test_monotonically_decreasing(self):
        weights = _generate_fixture_weights(2.0, 8)
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]

    def test_half_life_2_curve(self):
        """With half-life 2, fixture 3 (index 2) should have ~half the weight of fixture 1."""
        weights = _generate_fixture_weights(2.0, 8)
        # Raw ratio before normalization: e^(-ln2/2 * 2) / e^0 = 0.5
        # After normalization the ratio is preserved
        ratio = weights[2] / weights[0]
        assert 0.45 < ratio < 0.55

    def test_single_fixture(self):
        weights = _generate_fixture_weights(2.0, 1)
        assert len(weights) == 1
        assert weights[0] == 1.0

    def test_large_n(self):
        weights = _generate_fixture_weights(2.0, 20)
        assert len(weights) == 20
        assert abs(sum(weights) - 1.0) < 0.01
        assert all(w >= 0 for w in weights)

    def test_different_half_life(self):
        """Shorter half-life = faster decay."""
        fast = _generate_fixture_weights(1.0, 8)
        slow = _generate_fixture_weights(4.0, 8)
        # First weight should be larger for faster decay (more concentrated)
        assert fast[0] > slow[0]
        # Last weight should be smaller for faster decay
        assert fast[-1] < slow[-1]


# =============================================================================
# xga_to_attack_fdr
# =============================================================================

class TestXgaToAttackFdr:
    def test_elite_defence_returns_10(self):
        """Very low xGA = very hard to score against = FDR 10."""
        assert xga_to_attack_fdr(0.80) == 10

    def test_average_overall_returns_mid(self):
        """League avg xGA ~1.35 should be FDR 5-6."""
        fdr = xga_to_attack_fdr(1.35)
        assert 5 <= fdr <= 6

    def test_weak_defence_returns_low(self):
        """High xGA = easy to score against = FDR 1-2."""
        fdr = xga_to_attack_fdr(1.80)
        assert fdr <= 2

    def test_very_high_xga_returns_1(self):
        assert xga_to_attack_fdr(3.0) == 1

    def test_monotonic_overall(self):
        """Higher xGA should always give equal or lower FDR."""
        xga_values = [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        fdrs = [xga_to_attack_fdr(x) for x in xga_values]
        for i in range(len(fdrs) - 1):
            assert fdrs[i] >= fdrs[i + 1]

    def test_home_vs_away_thresholds_differ(self):
        """Home and away use different threshold scales, so same xGA maps differently.

        xGA=1.10 for a home team is above average (home avg ~1.15), so moderate FDR.
        xGA=1.10 for an away team is very good (away avg ~1.55), so high FDR.
        """
        xga = 1.10
        fdr_home = xga_to_attack_fdr(xga, is_opponent_home=True)
        fdr_away = xga_to_attack_fdr(xga, is_opponent_home=False)
        # Away thresholds are wider, so 1.10 away = elite = FDR 9
        # Home thresholds are tighter, so 1.10 home = above avg = FDR 6
        assert fdr_away > fdr_home

    def test_away_thresholds_are_wider(self):
        """xGA 1.5 away should be mid-range (not as bad as 1.5 home)."""
        fdr = xga_to_attack_fdr(1.50, is_opponent_home=False)
        assert 5 <= fdr <= 7

    def test_none_uses_overall(self):
        """None venue = overall thresholds."""
        fdr = xga_to_attack_fdr(1.35, is_opponent_home=None)
        assert 5 <= fdr <= 6

    def test_returns_int(self):
        result = xga_to_attack_fdr(1.0)
        assert isinstance(result, int)

    def test_range_1_to_10(self):
        for xga in [0.3, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0, 3.0]:
            fdr = xga_to_attack_fdr(xga)
            assert 1 <= fdr <= 10


# =============================================================================
# xg_to_defence_fdr
# =============================================================================

class TestXgToDefenceFdr:
    def test_elite_attack_returns_10(self):
        """Very high xG (Man City) = hardest to keep CS."""
        assert xg_to_defence_fdr(2.0) == 10

    def test_average_returns_mid(self):
        """League avg ~1.30 should be FDR 5."""
        fdr = xg_to_defence_fdr(1.30)
        assert fdr == 5

    def test_weak_attack_returns_low(self):
        """Low xG (Burnley/Sunderland) = easy to keep CS."""
        fdr = xg_to_defence_fdr(0.85)
        assert fdr <= 2

    def test_very_low_xg_returns_1(self):
        assert xg_to_defence_fdr(0.50) == 1

    def test_monotonic(self):
        """Higher xG should always give equal or higher defence FDR."""
        xg_values = [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        fdrs = [xg_to_defence_fdr(x) for x in xg_values]
        for i in range(len(fdrs) - 1):
            assert fdrs[i] <= fdrs[i + 1]

    def test_returns_int(self):
        assert isinstance(xg_to_defence_fdr(1.5), int)

    def test_range_1_to_10(self):
        for xg in [0.3, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5]:
            fdr = xg_to_defence_fdr(xg)
            assert 1 <= fdr <= 10


# =============================================================================
# get_price_adjusted_fdr_multiplier
# =============================================================================

class TestPriceAdjustedFdrMultiplier:
    def test_neutral_fdr5_returns_close_to_1(self):
        """FDR 5 (neutral) should return ~1.0 regardless of price/position."""
        mult = get_price_adjusted_fdr_multiplier(5, 7.0, 3)
        assert abs(mult - 1.0) < 0.05

    def test_easy_fixture_above_1(self):
        """FDR 1-2 should return multiplier > 1.0."""
        mult = get_price_adjusted_fdr_multiplier(1, 7.0, 3)
        assert mult > 1.0

    def test_hard_fixture_below_1(self):
        """FDR 9-10 should return multiplier < 1.0."""
        mult = get_price_adjusted_fdr_multiplier(10, 7.0, 3)
        assert mult < 1.0

    def test_premium_player_less_affected(self):
        """£12m player should have smaller deviation from 1.0 than £5m player."""
        premium = get_price_adjusted_fdr_multiplier(1, 12.0, 3)
        budget = get_price_adjusted_fdr_multiplier(1, 5.0, 3)
        # Both above 1, but budget should be further from 1
        assert (budget - 1.0) > (premium - 1.0)

    def test_fwd_amplified_effect(self):
        """FWDs (position 4) should have more extreme multipliers."""
        fwd = get_price_adjusted_fdr_multiplier(1, 7.0, 4)
        mid = get_price_adjusted_fdr_multiplier(1, 7.0, 3)
        # FWD has position_dampening 1.10 vs MID 1.05
        assert (fwd - 1.0) > (mid - 1.0) * 0.95  # FWD effect at least as large

    def test_def_dampened_effect(self):
        """DEFs (position 2) should have less extreme multipliers than FWDs."""
        defender = get_price_adjusted_fdr_multiplier(1, 7.0, 2)
        forward = get_price_adjusted_fdr_multiplier(1, 7.0, 4)
        assert (forward - 1.0) > (defender - 1.0)

    def test_returns_float(self):
        result = get_price_adjusted_fdr_multiplier(5, 7.0, 3)
        assert isinstance(result, float)


# =============================================================================
# calculate_goals_conceded_penalty
# =============================================================================

class TestGoalsConcededPenalty:
    def test_zero_xga_returns_zero(self):
        assert calculate_goals_conceded_penalty(0.0) == 0.0

    def test_negative_xga_returns_zero(self):
        assert calculate_goals_conceded_penalty(-1.0) == 0.0

    def test_positive_xga_returns_positive(self):
        """Penalty is a positive number (expected deductions per the FPL point formula)."""
        result = calculate_goals_conceded_penalty(1.5)
        assert result > 0

    def test_higher_xga_means_higher_penalty(self):
        low = calculate_goals_conceded_penalty(0.8)
        high = calculate_goals_conceded_penalty(2.0)
        assert high > low

    def test_low_xga_small_penalty(self):
        """xGA 0.5 — most probability at 0-1 goals, penalty should be very small."""
        result = calculate_goals_conceded_penalty(0.5)
        assert result < 0.15

    def test_high_xga_substantial_penalty(self):
        """xGA 2.5 — expect significant penalty."""
        result = calculate_goals_conceded_penalty(2.5)
        assert result > 0.5

    def test_poisson_consistency(self):
        """Verify Poisson-based calculation agrees with manual computation for xGA=1."""
        xGA = 1.0
        result = calculate_goals_conceded_penalty(xGA)
        # Manual: P(0)=0.368, P(1)=0.368, P(2)=0.184, P(3)=0.061, P(4)=0.015...
        # Deductions: 0*0.368 + 0*0.368 + 1*0.184 + 1*0.061 + 2*0.015 + ...
        # ≈ 0.184 + 0.061 + 0.030 + ... ≈ 0.28
        assert 0.20 < result < 0.40


# =============================================================================
# percentileofscore
# =============================================================================

class TestPercentileOfScore:
    def test_empty_returns_50(self):
        assert percentileofscore([], 5.0) == 50.0

    def test_single_element_below(self):
        result = percentileofscore([10.0], 5.0)
        assert result == 0.0

    def test_single_element_exact(self):
        result = percentileofscore([5.0], 5.0)
        assert result == 50.0

    def test_single_element_above(self):
        result = percentileofscore([5.0], 10.0)
        assert result == 100.0

    def test_middle_of_list(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = percentileofscore(data, 3.0)
        assert 30 < result < 70

    def test_at_maximum(self):
        data = [1.0, 2.0, 3.0]
        result = percentileofscore(data, 3.0)
        assert result > 50

    def test_below_minimum(self):
        data = [5.0, 10.0, 15.0]
        result = percentileofscore(data, 1.0)
        assert result == 0.0

    def test_returns_float(self):
        result = percentileofscore([1.0, 2.0, 3.0], 2.0)
        assert isinstance(result, float)


# =============================================================================
# get_set_piece_share
# =============================================================================

class TestGetSetPieceShare:
    def test_known_player_exact_match(self):
        """Bruno Fernandes should have set piece share."""
        share = get_set_piece_share("Fernandes")
        assert share == 0.50

    def test_known_player_full_name(self):
        share = get_set_piece_share("Bruno Fernandes")
        assert share == 0.50

    def test_unknown_player_returns_zero(self):
        share = get_set_piece_share("Unknown Player")
        assert share == 0.0

    def test_salah(self):
        share = get_set_piece_share("Salah")
        assert share > 0

    def test_partial_match(self):
        """Partial name matching should work."""
        share = get_set_piece_share("Palmer")
        assert share > 0


class TestApplySetPieceShareToMultiplier:
    def test_no_set_pieces_returns_base(self):
        result = apply_set_piece_share_to_multiplier(0.68, "Unknown Player")
        assert result == 0.68

    def test_with_set_pieces_moves_toward_neutral(self):
        """Set piece share should move multiplier toward 1.0."""
        base = 0.68  # Hard fixture
        result = apply_set_piece_share_to_multiplier(base, "Fernandes")
        assert result > base  # Closer to 1.0

    def test_set_piece_formula(self):
        """sp_share * 1.0 + (1 - sp_share) * base_mult."""
        base = 0.68
        result = apply_set_piece_share_to_multiplier(base, "Fernandes")
        expected = 0.50 * 1.0 + 0.50 * 0.68  # 0.84
        assert abs(result - expected) < 0.01


# =============================================================================
# calculate_roll_value
# =============================================================================

class TestCalculateRollValue:
    def test_1ft_base(self):
        """1 FT, no special circumstances."""
        val = calculate_roll_value(1, 8, False, False, 0)
        assert 0.3 < val < 0.5

    def test_2ft_reduced(self):
        """2+ FT should reduce roll value (diminishing returns)."""
        val_1ft = calculate_roll_value(1, 8, False, False, 0)
        val_2ft = calculate_roll_value(2, 8, False, False, 0)
        assert val_2ft < val_1ft

    def test_3ft_further_reduced(self):
        val_2ft = calculate_roll_value(2, 8, False, False, 0)
        val_3ft = calculate_roll_value(3, 8, False, False, 0)
        assert val_3ft < val_2ft

    def test_dgw_boost(self):
        """DGW ahead should increase roll value."""
        no_dgw = calculate_roll_value(1, 8, False, False, 0)
        dgw = calculate_roll_value(1, 8, True, False, 0)
        assert dgw > no_dgw

    def test_bgw_boost(self):
        """BGW ahead should increase roll value."""
        no_bgw = calculate_roll_value(1, 8, False, False, 0)
        bgw = calculate_roll_value(1, 8, False, True, 0)
        assert bgw > no_bgw

    def test_critical_issues_reduce(self):
        """Critical player issues should reduce roll value."""
        no_issues = calculate_roll_value(1, 8, False, False, 0)
        issues = calculate_roll_value(1, 8, False, False, 2)
        assert issues < no_issues

    def test_more_gws_more_value(self):
        """More GWs remaining = higher roll value."""
        short = calculate_roll_value(1, 2, False, False, 0)
        long_ = calculate_roll_value(1, 8, False, False, 0)
        assert long_ > short

    def test_last_gw_lowest(self):
        """Last GW should have very low roll value."""
        val = calculate_roll_value(1, 1, False, False, 0)
        assert val < 0.4

    def test_always_positive(self):
        """Roll value should always be positive."""
        for ft in [1, 2, 3, 4]:
            for gw in [1, 4, 8]:
                for issues in [0, 1, 3]:
                    val = calculate_roll_value(ft, gw, False, False, issues)
                    assert val > 0

    def test_combined_dgw_and_issues(self):
        """DGW + issues should partially offset each other."""
        base = calculate_roll_value(1, 8, False, False, 0)
        combined = calculate_roll_value(1, 8, True, False, 2)
        # DGW adds 1.4x, issues multiply by 0.5; net < base
        assert combined < base * 1.2
