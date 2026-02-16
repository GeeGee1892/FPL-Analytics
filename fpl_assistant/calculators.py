"""
FPL Assistant - Calculators Module

FDR mapping functions, matchup calculators, model classes (HomeAwaySplit,
CleanSheet, Attacking, Variance, Bonus), and player stat calculation functions.
"""

import math
from typing import Optional, Dict, List, Tuple

from fpl_assistant.config import (
    MODEL_CONFIG, FDRConfig, CleanSheetConfig, BonusConfig, VarianceConfig,
)
from fpl_assistant.constants import (
    DEFCON_THRESHOLD_DEF, DEFCON_THRESHOLD_MID, DEFCON_POINTS,
    LEAGUE_AVG_GOALS_PER_GAME, LEAGUE_AVG_CS_RATE,
    OPP_HOME_BOOST, OPP_AWAY_PENALTY,
    LEAGUE_AVG_XG, LEAGUE_AVG_XGA,
    DEFENCE_HOME_BOOST, DEFENCE_AWAY_PENALTY,
    PROMOTED_TEAM_DEFAULTS,
)
from fpl_assistant.models import (
    HomeAwaySplit, CSProbability, AttackingEstimate,
    BonusEstimate, VarianceEstimate,
)
from fpl_assistant.cache import cache

__all__ = [
    # FDR mapping functions
    "xga_to_attack_fdr",
    "xg_to_defence_fdr",
    "get_price_adjusted_fdr_multiplier",
    # Matchup calculation functions
    "calculate_goals_conceded_penalty",
    "calculate_matchup_attack_multiplier",
    "calculate_matchup_cs_probability",
    # Calculator classes
    "HomeAwaySplitCalculator",
    "CleanSheetModel",
    "AttackingModel",
    "VarianceModel",
    "BonusModel",
    # Global instances
    "home_away_calculator",
    "cs_model",
    "attacking_model",
    "variance_model",
    "bonus_model",
    # Player stat functions
    "get_ownership_tier",
    "calculate_defcon_per_90",
    "calculate_saves_per_90",
    "calculate_expected_bonus",
]


# =============================================================================
# FDR MAPPING FUNCTIONS
# =============================================================================

# v4.3.3: Absolute xGA to FDR conversion thresholds
# This replaces the broken min/max normalization that was creating FDR 8 for everyone
# Based on PL historical data: elite defence ~0.9 xGA, worst ~2.0 xGA
def xga_to_attack_fdr(xga: float, is_opponent_home: bool = None) -> int:
    """
    Convert xGA (goals conceded per game) to attack FDR (1-10 scale).

    Higher xGA = weaker defence = LOWER attack FDR (easier to score against)
    Lower xGA = stronger defence = HIGHER attack FDR (harder to score against)

    v4.3.3c: CRITICAL FIX - Separate thresholds for home vs away xGA!

    Teams concede ~25-30% fewer goals at home than away. Using the same thresholds
    for both was making ALL away fixtures look hard (because opponent's home xGA
    is naturally lower).

    Example: Bournemouth overall xGA = 1.46
    - Home xGA ≈ 1.24 (they concede less at home)
    - Away xGA ≈ 1.68 (they concede more away)

    When Wirtz goes @BOU, we check BOU's HOME xGA (1.24). With unified thresholds,
    this was returning FDR 7 (hard) when BOU's home defence is actually average.

    League averages (approx):
    - HOME xGA: ~1.15 (teams defend better at home)
    - AWAY xGA: ~1.55 (teams leak more on the road)
    - OVERALL: ~1.35

    Args:
        xga: Expected goals against per game
        is_opponent_home: True if this xGA is for opponent's HOME games (they defend better),
                         False if for opponent's AWAY games (they defend worse),
                         None for overall/blended
    """
    if is_opponent_home is True:
        # Opponent is at HOME = they defend better = tougher for us to score
        # Calibrated so league avg home xGA (~1.15) = FDR 5-6
        if xga <= 0.70:
            return 10  # Elite home defence (very rare)
        elif xga <= 0.80:
            return 9   # Top tier (Arsenal home ~0.68)
        elif xga <= 0.90:
            return 8   # Very good (Liverpool, City home)
        elif xga <= 1.00:
            return 7   # Good home defence
        elif xga <= 1.10:
            return 6   # Above average
        elif xga <= 1.20:
            return 5   # Average home defence (NEUTRAL)
        elif xga <= 1.32:
            return 4   # Below average (BOU, Wolves, Forest home)
        elif xga <= 1.45:
            return 3   # Weak home defence
        elif xga <= 1.60:
            return 2   # Very weak
        else:
            return 1   # Bottom tier

    elif is_opponent_home is False:
        # Opponent is AWAY = they defend worse = easier for us to score
        # Calibrated so league avg away xGA (~1.55) = FDR 5-6
        if xga <= 1.00:
            return 10  # Elite away defence (very rare)
        elif xga <= 1.15:
            return 9   # Top tier (Arsenal away)
        elif xga <= 1.30:
            return 8   # Very good away defence
        elif xga <= 1.45:
            return 7   # Good away defence
        elif xga <= 1.55:
            return 6   # Above average
        elif xga <= 1.70:
            return 5   # Average away defence (NEUTRAL)
        elif xga <= 1.85:
            return 4   # Below average (most mid-table away)
        elif xga <= 2.00:
            return 3   # Weak away defence
        elif xga <= 2.20:
            return 2   # Very weak
        else:
            return 1   # Bottom tier (promoted teams away)

    else:
        # Overall/blended xGA - original thresholds
        # Calibrated so league average (~1.35) = FDR 5-6
        if xga <= 0.90:
            return 10  # Elite (Arsenal)
        elif xga <= 1.05:
            return 9   # Top tier (Liverpool)
        elif xga <= 1.15:
            return 8   # Very good (City, Newcastle)
        elif xga <= 1.25:
            return 7   # Good (Villa, Man Utd)
        elif xga <= 1.35:
            return 6   # Above average (Chelsea, Everton, Brighton)
        elif xga <= 1.42:
            return 5   # Average/Neutral (Fulham, Leeds, Brentford)
        elif xga <= 1.50:
            return 4   # Below average (Spurs, Wolves, BOU, Forest)
        elif xga <= 1.60:
            return 3   # Weak (Crystal Palace, West Ham)
        elif xga <= 1.75:
            return 2   # Very weak
        else:
            return 1   # Bottom tier (Burnley)


def xg_to_defence_fdr(xg: float) -> int:
    """
    Convert xG (goals scored per game) to defence FDR (1-10 scale).

    Higher xG = stronger attack = HIGHER defence FDR (harder to keep CS)
    Lower xG = weaker attack = LOWER defence FDR (easier to keep CS)

    v4.3.3 RECALIBRATED: Based on Analytic FPL 25/26 data
    League average xG is ~1.30

    Thresholds:
    - Man City: 1.98 xG -> FDR 10
    - Arsenal: 1.84 xG -> FDR 9
    - Liverpool: 1.78 xG -> FDR 9
    - Chelsea: 1.54 xG -> FDR 7
    - Newcastle/Man Utd: 1.48 xG -> FDR 7
    - Spurs/Villa/Brentford/Palace: 1.41-1.44 xG -> FDR 6
    - Brighton/Fulham/BOU: 1.24-1.37 xG -> FDR 5 (neutral)
    - West Ham/Forest: 1.16-1.22 xG -> FDR 4
    - Leeds: 1.15 xG -> FDR 4
    - Wolves/Everton: 1.03-1.08 xG -> FDR 3
    - Sunderland/Burnley: 0.82-0.84 xG -> FDR 1-2
    """
    if xg >= 1.95:
        return 10  # Elite attack (City)
    elif xg >= 1.75:
        return 9   # Top tier (Arsenal, Liverpool)
    elif xg >= 1.55:
        return 8   # Very good attack
    elif xg >= 1.45:
        return 7   # Good attack (Chelsea, Newcastle, Man Utd)
    elif xg >= 1.38:
        return 6   # Above average (Spurs, Villa, Brentford, Palace)
    elif xg >= 1.25:
        return 5   # Average/Neutral (Brighton, Fulham, BOU)
    elif xg >= 1.12:
        return 4   # Below average (West Ham, Forest, Leeds)
    elif xg >= 1.00:
        return 3   # Weak attack (Wolves, Everton)
    elif xg >= 0.88:
        return 2   # Very weak
    else:
        return 1   # Bottom tier (Sunderland, Burnley)


def get_price_adjusted_fdr_multiplier(fdr: int, player_price: float, position: int = 3) -> float:
    """
    Get FDR multiplier with price and position-based dampening.

    v4.3: Now includes position-specific dampening.
    - FWDs get AMPLIFIED fixture effect (they destroy weak defences)
    - DEFs get dampened effect (more consistent, CS handles variance)

    Args:
        fdr: Fixture difficulty 1-10
        player_price: Player price in millions
        position: Player position (1=GKP, 2=DEF, 3=MID, 4=FWD)

    Returns:
        Adjusted FDR multiplier
    """
    fdr_config = MODEL_CONFIG["fdr"]
    base_mult = fdr_config.fdr_multipliers.get(fdr, 1.0)

    # Price-based dampening
    if player_price >= fdr_config.premium_price_threshold:
        price_dampening = fdr_config.premium_dampening
    elif player_price >= fdr_config.mid_price_threshold:
        price_dampening = fdr_config.mid_price_dampening
    else:
        price_dampening = fdr_config.budget_dampening

    # Position-specific dampening (FWDs amplified, DEFs/GKPs dampened)
    position_dampening = fdr_config.position_dampening.get(position, 1.0)

    # Combined dampening
    total_dampening = price_dampening * position_dampening

    return 1.0 + (base_mult - 1.0) * total_dampening


# =============================================================================
# MATCHUP CALCULATION FUNCTIONS
# =============================================================================

def calculate_goals_conceded_penalty(xGA: float) -> float:
    """
    Calculate expected FPL penalty for goals conceded using Poisson distribution.

    FPL rules: -1 per 2 goals conceded (for GKP/DEF playing 60+ mins)

    Using Poisson: E[penalty] = Σ P(k goals) × floor(k/2)

    This is more accurate than linear approximation because:
    - 0-1 goals = 0 penalty (not -0.5 each)
    - The stepped nature means low xGA is less penalized than linear model suggests

    Args:
        xGA: Expected goals against for this fixture

    Returns:
        Expected penalty (negative number)
    """
    if xGA <= 0:
        return 0.0

    penalty = 0.0
    prob = math.exp(-xGA)  # P(0 goals) - Poisson

    # Sum over reasonable range of goals (0-12 covers >99.99% of probability)
    for k in range(13):
        fpl_deduction = k // 2  # floor(k/2) = FPL's -1 per 2 goals rule
        penalty += prob * fpl_deduction
        prob *= xGA / (k + 1)  # P(k+1) = P(k) * λ / (k+1)

    return penalty


def calculate_matchup_attack_multiplier(
    team_id: int,
    opponent_id: int,
    is_home: bool,
    player_price: float,
    position: int
) -> float:
    """
    v4.3.9: Calculate attack multiplier based on BOTH teams' strengths.

    FIXED: Now uses single venue adjustment at end (was double-applying before).

    This considers:
    - Player's team attack strength (xG)
    - Opponent's defence weakness (xGA)
    - Single home/away adjustment applied to final result

    Example:
    - Salah vs Forest (A): Liverpool (1.76 xG) vs Forest (1.48 xGA)
      Base = (1.76/1.35) * (1.48/1.35) = 1.30 * 1.10 = 1.43
      Away penalty: 1.43 * 0.93 = 1.33 → Good fixture despite away

    - Salah vs City (H): Liverpool (1.76 xG) vs City (1.13 xGA)
      Base = (1.76/1.35) * (1.13/1.35) = 1.30 * 0.84 = 1.09
      Home boost: 1.09 * 1.08 = 1.18 → Harder than Forest away!

    Args:
        team_id: Player's team ID
        opponent_id: Opponent team ID
        is_home: Is player at home?
        player_price: For price-based dampening
        position: Player position (affects dampening)

    Returns:
        Attack multiplier (1.0 = neutral, >1 = easier, <1 = harder)
    """
    # Get team attack strength (NO venue adjustment yet)
    team_attack = LEAGUE_AVG_XG  # Default to average
    if cache.fdr_data and team_id in cache.fdr_data:
        team_data = cache.fdr_data[team_id]
        team_attack = team_data.get('blended_xg', LEAGUE_AVG_XG)

    # Get opponent defence weakness - use blended, NOT venue-specific
    # v4.3.9: Don't use home_xga/away_xga here - that double-applies venue effect
    opp_defence = LEAGUE_AVG_XGA  # Default to average
    if cache.fdr_data and opponent_id in cache.fdr_data:
        opp_data = cache.fdr_data[opponent_id]
        opp_defence = opp_data.get('blended_xga', LEAGUE_AVG_XGA)
    elif opponent_id in PROMOTED_TEAM_DEFAULTS:
        # Promoted team fallback
        defaults = PROMOTED_TEAM_DEFAULTS[opponent_id]
        opp_defence = defaults['xga']

    # Calculate base matchup score (NO venue adjustment)
    team_attack_factor = team_attack / LEAGUE_AVG_XG
    opp_defence_factor = opp_defence / LEAGUE_AVG_XGA

    # Raw matchup: product of both factors
    raw_matchup = team_attack_factor * opp_defence_factor

    # v4.3.11: Apply SMALL opponent venue adjustment (~5%)
    # This reflects opponent quality change, not your big home/away swing
    # Your personal venue effect comes from player splits OR fallback boost below
    if is_home:
        raw_matchup *= OPP_HOME_BOOST   # 1.05 - slight home advantage
    else:
        raw_matchup *= OPP_AWAY_PENALTY  # 0.95 - slight away penalty

    # Apply price and position dampening (premiums less affected by fixtures)
    fdr_config = MODEL_CONFIG["fdr"]

    if player_price >= fdr_config.premium_price_threshold:
        price_dampening = fdr_config.premium_dampening
    elif player_price >= fdr_config.mid_price_threshold:
        price_dampening = fdr_config.mid_price_dampening
    else:
        price_dampening = fdr_config.budget_dampening

    position_dampening = fdr_config.position_dampening.get(position, 1.0)
    total_dampening = price_dampening * position_dampening

    # Apply dampening: move toward 1.0 based on dampening factor
    final_multiplier = 1.0 + (raw_matchup - 1.0) * total_dampening

    # Cap at reasonable bounds
    return max(0.50, min(1.80, final_multiplier))


def calculate_matchup_cs_probability(
    team_id: int,
    opponent_id: int,
    is_home: bool,
    team_base_cs: float
) -> tuple:
    """
    v4.3.9: Calculate CS probability based on BOTH teams' strengths.

    FIXED: Now uses single venue adjustment at end (was double-applying before).

    Uses Poisson model with expected goals against based on:
    - Player's team defence strength (xGA)
    - Opponent's attack strength (xG)
    - Single venue adjustment applied to final xGA

    Example:
    - Gabriel vs City (H): Arsenal (0.81 xGA) vs City (1.95 xG)
      Base xGA = (0.81/1.35) * (1.95/1.35) * 1.35 = 0.87
      Home bonus: 0.87 * 0.93 = 0.81 xGA → CS prob = 44%

    - Gabriel vs Forest (A): Arsenal (0.81 xGA) vs Forest (1.15 xG)
      Base xGA = (0.81/1.35) * (1.15/1.35) * 1.35 = 0.51
      Away penalty: 0.51 * 1.08 = 0.55 xGA → CS prob = 58%

    Args:
        team_id: Player's team ID
        opponent_id: Opponent team ID
        is_home: Is player at home?
        team_base_cs: Fallback CS probability

    Returns:
        Tuple of (cs_probability, expected_goals_against)
    """
    cs_config = MODEL_CONFIG["clean_sheet"]

    # Get team defence strength - use BLENDED, not venue-specific
    # v4.3.9: Don't use home_xga/away_xga here - we apply venue once at the end
    team_defence = LEAGUE_AVG_XGA  # Default to average
    if cache.fdr_data and team_id in cache.fdr_data:
        team_data = cache.fdr_data[team_id]
        team_defence = team_data.get('blended_xga', LEAGUE_AVG_XGA)

    # Get opponent attack strength (xG) - use BLENDED, not venue-adjusted
    opp_attack = LEAGUE_AVG_XG  # Default to average
    if cache.fdr_data and opponent_id in cache.fdr_data:
        opp_data = cache.fdr_data[opponent_id]
        opp_attack = opp_data.get('blended_xg', LEAGUE_AVG_XG)
    elif opponent_id in PROMOTED_TEAM_DEFAULTS:
        # Promoted team fallback
        defaults = PROMOTED_TEAM_DEFAULTS[opponent_id]
        opp_attack = defaults['xg']

    # Calculate base expected goals against (NO venue adjustment yet)
    team_defence_factor = team_defence / LEAGUE_AVG_XGA
    opp_attack_factor = opp_attack / LEAGUE_AVG_XG

    # Base xG against = league average * matchup factors
    base_xga = LEAGUE_AVG_XGA * team_defence_factor * opp_attack_factor

    # v4.3.11: Apply venue adjustment for CS probability (~8%)
    # At home, we defend better and opponent attacks worse → concede less
    # Away, we defend worse and opponent attacks better → concede more
    if is_home:
        expected_xga = base_xga * DEFENCE_HOME_BOOST    # 0.92 (8% fewer goals at home)
    else:
        expected_xga = base_xga * DEFENCE_AWAY_PENALTY  # 1.08 (8% more goals away)

    # Bound to realistic range
    expected_xga = max(0.4, min(3.0, expected_xga))

    # Poisson CS probability
    cs_prob = math.exp(-expected_xga * cs_config.cs_steepness)

    # Apply caps
    cs_prob = min(cs_config.cs_prob_max, max(cs_config.cs_prob_min, cs_prob))

    return cs_prob, expected_xga


# =============================================================================
# CALCULATOR CLASSES
# =============================================================================

class HomeAwaySplitCalculator:
    """Calculate player's home vs away performance splits from GW history."""

    def __init__(self, min_games: int = 8):
        self.min_games = min_games

    def calculate_splits(
        self,
        player_history: Optional[Dict],
        player_data: Dict
    ) -> HomeAwaySplit:
        """Calculate home/away performance splits from player history."""
        split = HomeAwaySplit()

        if not player_history or not player_history.get("history"):
            return split

        history = player_history["history"]

        home_stats = {"minutes": 0, "xG": 0, "xA": 0, "points": 0, "games": 0}
        away_stats = {"minutes": 0, "xG": 0, "xA": 0, "points": 0, "games": 0}

        for gw in history:
            mins = gw.get("minutes", 0)
            if mins == 0:
                continue

            was_home = gw.get("was_home", True)
            stats = home_stats if was_home else away_stats

            stats["minutes"] += mins
            stats["xG"] += float(gw.get("expected_goals", 0) or 0)
            stats["xA"] += float(gw.get("expected_assists", 0) or 0)
            stats["points"] += gw.get("total_points", 0)
            stats["games"] += 1

        split.home_games = home_stats["games"]
        split.away_games = away_stats["games"]

        if home_stats["minutes"] >= 90:
            home_90s = home_stats["minutes"] / 90
            split.home_xG90 = home_stats["xG"] / home_90s
            split.home_xA90 = home_stats["xA"] / home_90s
            split.home_pts_per_90 = home_stats["points"] / home_90s

        if away_stats["minutes"] >= 90:
            away_90s = away_stats["minutes"] / 90
            split.away_xG90 = away_stats["xG"] / away_90s
            split.away_xA90 = away_stats["xA"] / away_90s
            split.away_pts_per_90 = away_stats["points"] / away_90s

        split.has_sufficient_data = (
            split.home_games >= self.min_games and
            split.away_games >= self.min_games
        )

        return split


class CleanSheetModel:
    """
    Calculate fixture-specific clean sheet probability.

    KEY FIX: CS probability now depends on OPPONENT's attacking strength,
    not just your team's defensive baseline.
    """

    def __init__(self, config: CleanSheetConfig = None):
        self.config = config or MODEL_CONFIG["clean_sheet"]

    def calculate_fixture_cs_probability(
        self,
        team_id: int,
        opponent_id: int,
        is_home: bool,
        fdr_data: Dict[int, Dict],
        fixture_xg_data: Optional[Dict] = None,
        fixture_xg_weight: float = 0.0
    ) -> CSProbability:
        """Calculate CS probability for a specific fixture."""
        ha_config = MODEL_CONFIG["home_away"]
        data_source = "model"

        # Get team's baseline xGA
        if fdr_data and team_id in fdr_data:
            team_xga_baseline = fdr_data[team_id].get('blended_xga', 1.3)
        else:
            team_xga_baseline = 1.3

        # Get opponent's attacking strength
        if fdr_data and opponent_id in fdr_data:
            opponent_attack = fdr_data[opponent_id].get('blended_xg', self.config.league_avg_goals_per_game)
        else:
            opponent_attack = self.config.league_avg_goals_per_game

        opponent_attack_multiplier = opponent_attack / self.config.league_avg_goals_per_game

        if fixture_xg_data and fixture_xg_weight > 0:
            # Use fixture-specific projection
            if is_home:
                fixture_xga = fixture_xg_data.get("away_xg", 1.3)
            else:
                fixture_xga = fixture_xg_data.get("home_xg", 1.5)

            # Model-based xGA for blending
            if is_home:
                ha_factor = 1.0 / ha_config.home_defence_boost
                opp_mult = opponent_attack_multiplier * ha_config.away_attack_penalty
            else:
                ha_factor = 1.0 / ha_config.away_defence_penalty
                opp_mult = opponent_attack_multiplier * ha_config.home_attack_boost

            model_xga = team_xga_baseline * opp_mult * ha_factor
            expected_goals_against = (
                fixture_xg_weight * fixture_xga +
                (1 - fixture_xg_weight) * model_xga
            )
            data_source = "fixture_xg+model"
        else:
            # Pure model-based
            if is_home:
                ha_factor = 1.0 / ha_config.home_defence_boost
                opp_mult = opponent_attack_multiplier * ha_config.away_attack_penalty
            else:
                ha_factor = 1.0 / ha_config.away_defence_penalty
                opp_mult = opponent_attack_multiplier * ha_config.home_attack_boost

            expected_goals_against = team_xga_baseline * opp_mult * ha_factor
            data_source = "team_model"

        expected_goals_against = max(0.4, min(3.0, expected_goals_against))

        # v4.3: Poisson CS probability with configurable steepness
        steepness = getattr(self.config, 'cs_steepness', 1.0)
        cs_prob = math.exp(-expected_goals_against * steepness)
        cs_prob = max(self.config.cs_prob_min, min(self.config.cs_prob_max, cs_prob))

        return CSProbability(
            cs_prob=round(cs_prob, 4),
            expected_goals_against=round(expected_goals_against, 3),
            data_source=data_source,
            opponent_attack_strength=round(opp_mult, 3)
        )


class AttackingModel:
    """
    Calculate fixture-adjusted attacking returns with home/away splits
    and price-based FDR dampening.
    """

    def __init__(self, config: FDRConfig = None):
        self.config = config or MODEL_CONFIG["fdr"]

    def get_fdr_multiplier(self, fdr: int, player_price: float) -> float:
        """Get FDR multiplier with price-based dampening."""
        base_mult = self.config.fdr_multipliers.get(fdr, 1.0)

        if player_price >= self.config.premium_price_threshold:
            dampening = self.config.premium_dampening
        elif player_price >= self.config.mid_price_threshold:
            dampening = self.config.mid_price_dampening
        else:
            dampening = self.config.budget_dampening

        return 1.0 + (base_mult - 1.0) * dampening

    def calculate_fixture_attacking(
        self,
        base_xG90: float,
        base_xA90: float,
        opponent_id: int,
        is_home: bool,
        fdr_data: Dict[int, Dict],
        home_away_split: Optional[HomeAwaySplit] = None,
        player_price: float = 5.0
    ) -> AttackingEstimate:
        """Calculate fixture-adjusted xG/xA with home/away splits."""
        ha_config = MODEL_CONFIG["home_away"]
        data_source = "base_rate"

        xG90 = base_xG90
        xA90 = base_xA90

        # Apply home/away split if available
        if home_away_split and home_away_split.has_sufficient_data:
            if is_home:
                split_xG90 = home_away_split.home_xG90
                split_xA90 = home_away_split.home_xA90
            else:
                split_xG90 = home_away_split.away_xG90
                split_xA90 = home_away_split.away_xA90

            # Blend split with overall
            xG90 = ha_config.split_weight * split_xG90 + (1 - ha_config.split_weight) * base_xG90
            xA90 = ha_config.split_weight * split_xA90 + (1 - ha_config.split_weight) * base_xA90
            data_source = "home_away_split"

        # Calculate attack multiplier from FDR
        attack_multiplier = 1.0

        if fdr_data and opponent_id in fdr_data:
            # v4.3.2 FIX: Use attack_fdr (opponent's defensive weakness) not defence_fdr
            # attack_fdr = how easy to score against opponent (based on their xGA)
            # defence_fdr = how dangerous opponent's attack is (based on their xG)
            # v4.3.2: Use venue-specific attack_fdr:
            # - If we're HOME, opponent is AWAY → use their attack_fdr_away
            # - If we're AWAY, opponent is HOME → use their attack_fdr_home
            if is_home:
                opp_attack_fdr = fdr_data[opponent_id].get("attack_fdr_away",
                                 fdr_data[opponent_id].get("attack_fdr", 5))
            else:
                opp_attack_fdr = fdr_data[opponent_id].get("attack_fdr_home",
                                 fdr_data[opponent_id].get("attack_fdr", 5))
            attack_multiplier = self.get_fdr_multiplier(round(opp_attack_fdr), player_price)
            data_source += "+fdr"

        # Apply home/away adjustment
        if is_home:
            attack_multiplier *= ha_config.home_attack_boost
        else:
            attack_multiplier *= ha_config.away_attack_penalty

        # v4.3.2 FIX: Removed hardcoded cap of (0.78, 1.22) that was blocking FDR multipliers
        # Proper price-based caps are applied in calculate_expected_points()
        # Only apply loose sanity bounds here
        attack_multiplier = max(0.50, min(1.70, attack_multiplier))

        fixture_xG90 = xG90 * attack_multiplier
        fixture_xA90 = xA90 * attack_multiplier

        return AttackingEstimate(
            xG90=round(xG90, 4),
            xA90=round(xA90, 4),
            attack_multiplier=round(attack_multiplier, 4),
            fixture_xG90=round(fixture_xG90, 4),
            fixture_xA90=round(fixture_xA90, 4),
            data_source=data_source
        )


class VarianceModel:
    """
    Calculate ceiling and floor using historical variance.

    KEY FIX: Uses player's actual points std dev when available,
    not flat position multipliers.
    """

    def __init__(self, config: VarianceConfig = None):
        self.config = config or MODEL_CONFIG["variance"]

    def calculate_variance(
        self,
        player_history: Optional[Dict],
        player_data: Dict,
        position_id: int,
        xpts: float,
        fixture_fdr: int = 5,
        is_home: bool = True
    ) -> VarianceEstimate:
        """Calculate ceiling/floor using historical variance."""
        data_source = "position_baseline"
        std_dev = self.config.position_stdev_baseline.get(position_id, 3.0)

        # Try player-specific std dev from history
        if player_history and player_history.get("history"):
            history = player_history["history"]

            points_list = [
                h.get("total_points", 0)
                for h in history
                if h.get("minutes", 0) >= 45
            ]

            if len(points_list) >= self.config.min_games_for_stdev:
                mean_pts = sum(points_list) / len(points_list)
                variance = sum((p - mean_pts) ** 2 for p in points_list) / (len(points_list) - 1)
                player_stdev = math.sqrt(variance)

                # Blend with position baseline
                history_weight = min(1.0, len(points_list) / 25)
                std_dev = history_weight * player_stdev + (1 - history_weight) * std_dev
                data_source = "historical"

        # Base ceiling and floor
        base_ceiling = xpts + self.config.ceiling_z * std_dev
        base_floor = xpts + self.config.floor_z * std_dev

        # Fixture quality adjustment to ceiling
        # v4.3.1: Dampen fixture effect for HARD fixtures
        # Rationale: xPts already captures probability reduction
        # Ceiling represents "what if he hauls" - hauls CAN happen anywhere
        # Easy fixtures still get full boost (ceiling goes UP)
        # Hard fixtures get dampened reduction (ceiling doesn't crash as much)
        fixture_boost = self.config.fixture_ceiling_boost.get(fixture_fdr, 1.0)

        if fixture_boost < 1.0:
            # Hard fixture - dampen the reduction
            # sqrt(0.68) = 0.82 instead of 0.68
            fixture_boost = math.sqrt(fixture_boost)

        ha_boost = self.config.home_ceiling_boost if is_home else self.config.away_ceiling_penalty

        ceiling = base_ceiling * fixture_boost * ha_boost
        floor = max(0, base_floor)

        return VarianceEstimate(
            ceiling=round(ceiling, 2),
            floor=round(floor, 2),
            std_dev=round(std_dev, 2),
            data_source=data_source
        )


class BonusModel:
    """
    Calculate expected bonus points with teammate competition.
    """

    def __init__(self, config: BonusConfig = None):
        self.config = config or MODEL_CONFIG["bonus"]

    def calculate_bonus(
        self,
        player_data: Dict,
        position_id: int,
        fixture_xG90: float,
        fixture_xA90: float,
        cs_prob: float,
        teammate_competition: bool = False
    ) -> BonusEstimate:
        """Calculate expected bonus for a fixture."""
        total_minutes = int(player_data.get("minutes", 0) or 0)
        bonus = int(player_data.get("bonus", 0) or 0)

        if total_minutes >= 180:
            mins90 = total_minutes / 90.0
            bonus_per_90 = bonus / mins90
        else:
            bonus_per_90 = 0.45

        # Estimate BPS
        goal_bps = fixture_xG90 * self.config.bps_per_goal.get(position_id, 24)
        assist_bps = fixture_xA90 * self.config.bps_per_assist
        cs_bps = cs_prob * self.config.bps_per_cs.get(position_id, 0)
        base = self.config.base_bps_by_position.get(position_id, 8)

        estimated_bps = base + goal_bps + assist_bps + cs_bps

        # Convert BPS to expected bonus
        expected_bonus = 0.15
        for threshold, bonus_val in self.config.bps_to_bonus:
            if estimated_bps >= threshold:
                expected_bonus = bonus_val
                break

        # Blend historical and modeled
        if total_minutes >= 1350:
            final_bonus = 0.65 * bonus_per_90 + 0.35 * expected_bonus
        elif total_minutes >= 720:
            final_bonus = 0.50 * bonus_per_90 + 0.50 * expected_bonus
        else:
            final_bonus = 0.30 * bonus_per_90 + 0.70 * expected_bonus

        dilution_applied = False
        if teammate_competition:
            final_bonus *= self.config.teammate_dilution
            dilution_applied = True

        final_bonus = max(0.1, min(2.5, final_bonus))

        return BonusEstimate(
            expected_bonus=round(final_bonus, 3),
            estimated_bps=round(estimated_bps, 1),
            bonus_per_90_historical=round(bonus_per_90, 3),
            teammate_dilution_applied=dilution_applied
        )


# Initialize global model instances
home_away_calculator = HomeAwaySplitCalculator(min_games=MODEL_CONFIG["home_away"].min_games_for_split)
cs_model = CleanSheetModel()
attacking_model = AttackingModel()
variance_model = VarianceModel()
bonus_model = BonusModel()


# =============================================================================
# PLAYER STAT FUNCTIONS
# =============================================================================

def get_ownership_tier(ownership: float) -> Tuple[str, str]:
    """
    Categorize player by ownership for rank climbing strategy.

    Returns: (tier_name, tier_description)

    - template: Must-own players, you lose rank without them
    - popular: Common picks, low differential upside
    - differential: Can gain rank significantly if they haul
    - punt: High variance, big rank swings possible
    """
    if ownership >= 30:
        return "template", "Essential pick"
    elif ownership >= 15:
        return "popular", "Common pick"
    elif ownership >= 5:
        return "differential", "Rank gainer"
    else:
        return "punt", "High variance"


def calculate_defcon_per_90(player: Dict, position_id: int, team_xga: float = None) -> tuple[float, float, int]:
    """
    Get DEFCON per 90 from FPL API (already calculated) and compute probability.
    DEFCON = Defensive Contributions (tackles, interceptions, blocks, clearances, recoveries)

    v4.3.2: Using normal distribution approximation.
    v5.0 VALIDATED: Checked against 193 games of actual data:
        - Actual aggregate hit rate: 61.7%
        - Model predicted rate: 62.9%
        - Model is well-calibrated, no changes needed.

    Workload dampening (v4.3.3b) still applies - elite teams face fewer attacks.
    """
    total_minutes = int(player.get("minutes", 0) or 0)

    # FPL API provides these directly
    defcon_per_90 = float(player.get("defensive_contribution_per_90", 0) or 0)
    defcon_total = int(player.get("defensive_contribution", 0) or 0)

    if total_minutes < 90 or position_id not in [2, 3]:
        return 0.0, 0.0, 0

    # Position-specific thresholds for DEFCON points (2 pts if >= threshold)
    if position_id == 2:  # DEF
        threshold = DEFCON_THRESHOLD_DEF  # 10
    else:  # MID
        threshold = DEFCON_THRESHOLD_MID  # 12

    # Calculate probability of hitting threshold
    # Using normal approximation: std ≈ 30% of mean (empirical variance)
    if defcon_per_90 <= 0:
        prob = 0.0
    else:
        # Estimate standard deviation (DEFCON has moderate variance)
        std_dev = max(2.5, defcon_per_90 * 0.30)

        # Z-score: how many std devs is threshold from mean?
        z_score = (threshold - defcon_per_90) / std_dev

        # P(X >= threshold) using normal CDF approximation
        prob = 1.0 / (1.0 + math.exp(z_score * 1.7))

        # Realistic caps: can't hit 100% due to early subs, tactical changes, etc.
        if position_id == 2:  # DEF
            prob = min(prob, 0.85)
        else:  # MID
            prob = min(prob, 0.80)

        # Floor: even low DEFCON players occasionally hit threshold
        prob = max(prob, 0.02)

    # v4.3.3b: Apply workload dampening based on team's defensive quality
    # Elite defensive teams face fewer attacks → fewer DEFCON opportunities
    if team_xga is not None and team_xga > 0:
        LEAGUE_AVG_XGA = 1.35
        workload_factor = team_xga / LEAGUE_AVG_XGA

        # Cap between 0.6 and 1.2 to avoid extreme adjustments
        workload_factor = max(0.60, min(1.20, workload_factor))

        prob = prob * workload_factor
        # Re-apply caps after dampening
        prob = min(prob, 0.85 if position_id == 2 else 0.80)

    return round(defcon_per_90, 2), round(prob, 3), defcon_total


def calculate_saves_per_90(player: Dict) -> tuple[float, float]:
    total_minutes = int(player.get("minutes", 0) or 0)
    saves = int(player.get("saves", 0) or 0)

    if total_minutes < 90:
        return 0.0, 0.0

    mins90 = total_minutes / 90.0
    saves_per_90 = saves / mins90
    save_pts_per_90 = saves_per_90 / 3.0

    return round(saves_per_90, 2), round(save_pts_per_90, 2)


def calculate_expected_bonus(player: Dict, position: int) -> tuple[float, float]:
    """
    Expert-level bonus point prediction.

    BPS (Bonus Point System) is heavily driven by:
    - Goals scored (+24 BPS for MID/FWD, +12 for DEF)
    - Assists (+9 BPS)
    - Clean sheets (+12 for DEF, +6 for MID)
    - Key passes, tackles, saves, interceptions
    - Penalties for cards, missed chances, errors

    Top 3 BPS in each match get 3, 2, 1 bonus points respectively.

    Returns: (bonus_per_90_historical, expected_bonus_per_90)
    """
    total_minutes = int(player.get("minutes", 0) or 0)
    bonus = int(player.get("bonus", 0) or 0)
    bps = int(player.get("bps", 0) or 0)
    goals = int(player.get("goals_scored", 0) or 0)
    assists = int(player.get("assists", 0) or 0)
    cs = int(player.get("clean_sheets", 0) or 0)

    xG = float(player.get("expected_goals", 0) or 0)
    xA = float(player.get("expected_assists", 0) or 0)

    if total_minutes < 180:  # Need at least 2 full games
        # New/low-minute players - use position baseline
        baseline = {1: 0.30, 2: 0.45, 3: 0.55, 4: 0.65}.get(position, 0.45)
        return baseline, baseline

    mins90 = total_minutes / 90.0
    games_played = mins90 / 1  # Approximate

    # Historical rates
    bonus_per_90 = bonus / mins90
    bps_per_90 = bps / mins90
    goals_per_90 = goals / mins90
    assists_per_90 = assists / mins90
    cs_per_90 = cs / mins90 if position in [1, 2] else 0

    # xGI per 90
    xG90 = xG / mins90
    xA90 = xA / mins90
    xGI90 = xG90 + xA90

    # ==================== MODEL BONUS FROM UNDERLYING STATS ====================
    # BPS (Bonus Point System) weights:
    # - Goals: +24 for MID/FWD, +12 for DEF, +6 for GKP
    # - Assists: +9
    # - Clean sheet: +12 for DEF/GKP, +6 for MID
    # - Key passes, tackles, saves contribute smaller amounts
    #
    # Top 3 BPS in each match get 3, 2, 1 bonus points.
    # Premium attackers in good fixtures can hit 40+ BPS with a brace.

    # Estimate BPS contribution from goals/assists/CS
    goal_bps_contrib = goals_per_90 * (24 if position in [3, 4] else (12 if position == 2 else 6))
    assist_bps_contrib = assists_per_90 * 9
    cs_bps_contrib = cs_per_90 * (12 if position in [1, 2] else (6 if position == 3 else 0))

    # Base BPS from other actions (tackles, key passes, saves, interceptions)
    # Estimated from typical position ranges
    base_bps = {1: 10, 2: 11, 3: 9, 4: 6}.get(position, 8)

    estimated_bps = base_bps + goal_bps_contrib + assist_bps_contrib + cs_bps_contrib

    # Convert BPS to expected bonus using smoother curve
    # RECALIBRATED v4.2: Reduced by ~25% - BPS-based model overestimates
    # because it assumes consistent returns, but bonus is binary (you get it or you don't)
    # Haaland scores in ~60% of games, so expected bonus = 0.6 × high + 0.4 × low
    if estimated_bps >= 40:
        estimated_bonus = 2.0  # Was 2.7 - elite haul, but doesn't happen every game
    elif estimated_bps >= 35:
        estimated_bonus = 1.7  # Was 2.3
    elif estimated_bps >= 30:
        estimated_bonus = 1.4  # Was 1.9
    elif estimated_bps >= 26:
        estimated_bonus = 1.1  # Was 1.5
    elif estimated_bps >= 22:
        estimated_bonus = 0.8  # Was 1.1
    elif estimated_bps >= 18:
        estimated_bonus = 0.5  # Was 0.7
    elif estimated_bps >= 14:
        estimated_bonus = 0.3  # Was 0.4
    else:
        estimated_bonus = 0.1  # Was 0.15

    # ==================== BLEND HISTORICAL AND MODELED ====================
    # If player has significant history, weight historical more
    # If limited history, trust the model more

    if mins90 >= 15:  # ~15+ full games
        # Trust historical rate more
        final_bonus = 0.65 * bonus_per_90 + 0.35 * estimated_bonus
    elif mins90 >= 8:
        # Balanced blend
        final_bonus = 0.50 * bonus_per_90 + 0.50 * estimated_bonus
    else:
        # Limited data - trust model more
        final_bonus = 0.30 * bonus_per_90 + 0.70 * estimated_bonus

    # Adjust for xGI vs actual GI (regression to mean)
    # If player is over/underperforming xGI, adjust expectations
    actual_gi_per_90 = goals_per_90 + assists_per_90
    if xGI90 > 0.1:
        performance_ratio = actual_gi_per_90 / xGI90
        if performance_ratio > 1.3:
            # Overperforming - expect some regression
            final_bonus *= 0.92
        elif performance_ratio < 0.7:
            # Underperforming - expect positive regression
            final_bonus *= 1.08

    # Cap at realistic maximum
    final_bonus = min(2.5, max(0.1, final_bonus))

    return round(bonus_per_90, 2), round(final_bonus, 2)
