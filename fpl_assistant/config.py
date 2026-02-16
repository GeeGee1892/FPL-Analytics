from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# =============================================================================
# MODEL CONFIGURATION - All calibration constants with documentation
# =============================================================================

@dataclass
class FDRConfig:
    """
    Fixture Difficulty Rating configuration.
    FDR 1-10 scale where 10 = hardest fixture.

    Key insight: Premium players (£10m+) are less affected by fixtures
    than budget players because skill overcomes difficulty.
    """

    # FDR multipliers - v4.3.1: WIDENED further for FDR 1-2
    # Empirical analysis shows Haaland vs bottom 6 (H) averages ~9.5 PPG
    # vs top 6 (A) averages ~5.0 PPG - that's a 90% swing
    # FDR 1-2 needed additional boost to capture elite fixture upside
    fdr_multipliers: Dict[int, float] = field(default_factory=lambda: {
        1: 1.38,   # v4.3.1: Was 1.30 - elite fixtures need more boost
        2: 1.30,   # v4.3.1: Was 1.22
        3: 1.18,   # v4.3.1: Was 1.14
        4: 1.08,
        5: 1.00,   # Neutral baseline
        6: 0.93,
        7: 0.84,
        8: 0.75,
        9: 0.66,
        10: 0.55,  # v4.3.1: Was 0.60 - hardest fixtures more punishing
    })

    # Price-based FDR dampening - v4.3: REDUCED dampening for more variance
    # Previous values were too aggressive, flattening fixture impact
    # Formula: effective_mult = 1 + (base_mult - 1) * dampening
    premium_dampening: float = 0.70      # £10m+: 70% of FDR effect (was 55%)
    mid_price_dampening: float = 0.88    # £6-9.9m: 88% of FDR effect (was 80%)
    budget_dampening: float = 1.0        # Below £6m: full effect

    # Position-specific dampening multipliers (applied on top of price dampening)
    # Attackers should feel fixtures MORE than defenders
    # FWDs destroy weak defences; DEFs are more consistent
    # v4.3.3: MIDs slightly boosted (many elite MIDs play as wingers/AMs like Salah, Wirtz)
    position_dampening: Dict[int, float] = field(default_factory=lambda: {
        1: 0.85,   # GKP - moderate fixture dependency (CS-driven)
        2: 0.90,   # DEF - moderate (CS + attacking returns)
        3: 1.05,   # MID - slightly boosted (many play advanced positions)
        4: 1.10,   # FWD - AMPLIFIED effect (premiums massacre weak defences)
    })

    premium_price_threshold: float = 10.0
    mid_price_threshold: float = 6.0

    # Home/away FDR adjustment
    home_fdr_multiplier: float = 0.88
    away_fdr_multiplier: float = 1.08


@dataclass
class HomeAwayConfig:
    """
    Home/Away performance adjustments - v4.3.11 SEPARATION OF CONCERNS.

    We need TWO different venue adjustments:

    1. OPPONENT VENUE EFFECT (small, ~5%):
       - Does City's defence get worse when they travel? Slightly.
       - Used in FDR and matchup calculations
       - opp_home_boost/opp_away_penalty

    2. PLAYER VENUE EFFECT (large, ~15%):
       - Does Salah score more at Anfield? Significantly.
       - Used in xPts for players WITHOUT sufficient split data
       - player_home_boost/player_away_penalty
       - Players WITH split data use their actual home/away xG90 instead

    This separation means:
    - FDR shows OPPONENT quality (City = hard everywhere)
    - xPts reflects YOUR performance (Salah better at home)
    """

    # OPPONENT venue adjustments (for FDR/matchup calculations)
    # Small because elite teams stay elite on the road
    opp_home_boost: float = 1.05       # Opponent 5% better at home
    opp_away_penalty: float = 0.95     # Opponent 5% worse away

    # PLAYER venue adjustments (fallback for xPts when no split data)
    # These should give SIMILAR total swing as players WITH split data
    # Players with splits get ~25% inherent swing, +10% from matchup = ~35-40% total
    # So fallback should be ~12% (not 15%) to avoid overcounting with matchup
    player_home_boost: float = 1.12    # You score 12% more at home
    player_away_penalty: float = 0.88  # You score 12% less away

    # Defence venue adjustments (for CS probability)
    # Moderate - teams concede more away but effect is smaller than attack
    defence_home_boost: float = 0.92   # Concede 8% less at home
    defence_away_penalty: float = 1.08 # Concede 8% more away

    # Player-level home/away variance
    min_games_for_split: int = 8       # Min games to trust split
    split_weight: float = 0.60         # 60% split-specific, 40% overall


@dataclass
class XptsConfig:
    """Expected points calculation configuration."""

    goal_points: Dict[int, int] = field(default_factory=lambda: {
        1: 6, 2: 6, 3: 5, 4: 4  # GKP, DEF, MID, FWD
    })

    cs_points: Dict[int, int] = field(default_factory=lambda: {
        1: 4, 2: 4, 3: 1, 4: 0
    })

    assist_points: int = 3

    # DEFCON thresholds
    defcon_threshold_def: int = 10
    defcon_threshold_mid: int = 12
    defcon_points: float = 2.0

    # Yellow card baselines by position (per 90)
    yellow_baseline: Dict[int, float] = field(default_factory=lambda: {
        1: 0.02, 2: 0.12, 3: 0.10, 4: 0.06
    })

    # Own goal probability by position (per 90)
    og_probability: Dict[int, float] = field(default_factory=lambda: {
        1: 0.005, 2: 0.020, 3: 0.008, 4: 0.003
    })

    shots_on_target_per_xg: float = 2.8
    league_avg_save_rate: float = 0.70

    # Fixture weighting: exponential decay with half-life parameter
    # Half-life of 2.0 means fixture 3 has half the weight of fixture 1
    # Weights are auto-normalized to sum to 1.0 at usage time
    fixture_weight_half_life: float = 2.0
    fixture_weight_max_horizon: int = 8

    # Horizon regression
    horizon_regression: Dict[int, float] = field(default_factory=lambda: {
        1: 1.00, 2: 0.99, 3: 0.98, 4: 0.97,
        5: 0.96, 6: 0.95, 7: 0.94, 8: 0.93
    })


@dataclass
class VarianceConfig:
    """Ceiling/floor variance configuration."""

    ceiling_z: float = 0.84              # 80th percentile
    floor_z: float = -0.84               # 20th percentile

    # Position-based baseline stdev (per 90) - v4.3: INCREASED for realistic variance
    # FPL points are right-skewed with long tails; previous values underestimated
    position_stdev_baseline: Dict[int, float] = field(default_factory=lambda: {
        1: 2.5,    # GKP - narrow but not too narrow
        2: 3.0,    # DEF - CS binary nature adds variance
        3: 3.8,    # MID - mixed profile
        4: 4.5,    # FWD - widest (binary goal outcomes)
    })

    min_games_for_stdev: int = 10

    # Fixture ceiling boosts - v4.3.1: Less extreme for hard fixtures
    # Hauls CAN happen in any fixture, just less likely
    # The xPts model captures likelihood; ceiling represents "if he hauls"
    # So hard fixture ceiling shouldn't be crushed as much as xPts
    fixture_ceiling_boost: Dict[int, float] = field(default_factory=lambda: {
        1: 1.45, 2: 1.35, 3: 1.25, 4: 1.12, 5: 1.00,
        6: 0.92, 7: 0.84, 8: 0.76, 9: 0.68, 10: 0.60  # v4.3.1: Was 0.52 for FDR10
    })

    home_ceiling_boost: float = 1.10      # v4.3.1: Slight increase from 1.08
    away_ceiling_penalty: float = 0.92    # v4.3.1: Was 0.94, slightly less harsh


@dataclass
class BonusConfig:
    """Bonus point prediction configuration."""

    bps_per_goal: Dict[int, int] = field(default_factory=lambda: {
        1: 6, 2: 12, 3: 24, 4: 24
    })
    bps_per_assist: int = 9
    bps_per_cs: Dict[int, int] = field(default_factory=lambda: {
        1: 12, 2: 12, 3: 6, 4: 0
    })
    base_bps_by_position: Dict[int, int] = field(default_factory=lambda: {
        1: 10, 2: 11, 3: 9, 4: 6
    })

    # BPS to bonus conversion
    bps_to_bonus: List[Tuple[int, float]] = field(default_factory=lambda: [
        (40, 2.7), (35, 2.3), (30, 1.9), (26, 1.5),
        (22, 1.1), (18, 0.7), (14, 0.4), (0, 0.15)
    ])

    # Legacy flat dilution (still used as fallback)
    teammate_dilution: float = 0.70

    # Position-based teammate dilution - more nuanced
    # Key = (player_position, competitor_position), Value = dilution factor
    # Lower factor = more dilution (harder to win bonus)
    position_dilution: Dict[Tuple[int, int], float] = field(default_factory=lambda: {
        # Two FWDs compete heavily (Jackson vs Palmer scenario)
        (4, 4): 0.65,
        # FWD vs MID - moderate competition
        (4, 3): 0.75,
        (3, 4): 0.75,
        # Two MIDs - lighter competition (different BPS profiles)
        (3, 3): 0.80,
        # DEF vs anyone - DEFs rarely compete with attackers for bonus
        (2, 4): 0.90,
        (2, 3): 0.90,
        (2, 2): 0.85,
        # GKP only competes with DEFs in CS scenarios
        (1, 2): 0.80,
        (1, 1): 0.95,  # Two GKPs never both play
    })


@dataclass
class CleanSheetConfig:
    """Clean sheet probability configuration."""

    # v4.3.3: Lowered CS cap from 52% to 45% - DEFs were overvalued
    # Even Arsenal's elite defence rarely exceeds 45% CS rate
    cs_prob_min: float = 0.05            # v4.3: Raised from 0.03 - very hard fixtures still have small chance
    cs_prob_max: float = 0.52            # v4.3.4: Raised back to 0.52 - Arsenal/Liverpool can hit 50%+ vs weak teams
    league_avg_goals_per_game: float = 1.35
    league_avg_cs_rate: float = 0.27
    # v4.3: Steepness factor for Poisson CS calculation
    # Higher = more dramatic response to xGA differences
    cs_steepness: float = 1.05


@dataclass
class CaptainConfig:
    """Captain selection scoring configuration."""

    position_ceiling: Dict[int, float] = field(default_factory=lambda: {
        1: 0.70, 2: 0.85, 3: 1.12, 4: 1.18
    })

    elite_ppg_threshold: float = 8.0
    elite_ppg_bonus: float = 1.15
    good_ppg_threshold: float = 7.0
    good_ppg_bonus: float = 1.10
    solid_ppg_threshold: float = 6.0
    solid_ppg_bonus: float = 1.05
    poor_ppg_threshold: float = 4.0
    poor_ppg_penalty: float = 0.85

    ownership_tiers: List[Tuple[float, float]] = field(default_factory=lambda: [
        (5, 0.08), (10, 0.05), (20, 0.02),
        (35, -0.01), (50, -0.03), (100, -0.05)
    ])


@dataclass
class FormConfig:
    """
    v5.0: Player and team form adjustment configuration.
    v5.1: Added premium player settings based on backtest data.

    Backtest findings:
    - Premium MIDs (£8+) had 5x variance: 0.43 → 2.09 → 0.84 MAE
    - Budget players consistently lowest MAE (0.30-0.50)
    - Premium hot/cold streaks exceed ±30% cap

    Form adjusts xG90/xA90 based on recent performance vs season average.
    This captures hot/cold streaks that static season rates miss.
    """

    # Lookback period for recent form
    lookback_games: int = 6
    min_games_for_form: int = 3
    min_minutes_for_form: int = 360  # v5.2: Raised from 180 — prevents tiny-sample form inflation

    # Form weight (how much to trust recent vs season)
    # Higher = more reactive to form changes
    base_form_weight: float = 0.35
    max_form_weight: float = 0.50

    # Form factor bounds (prevents over-adjustment)
    min_form_factor: float = 0.70  # Max 30% reduction
    max_form_factor: float = 1.30  # Max 30% increase

    # v5.1: Premium player adjustments (£8+ for MID/FWD, £6+ for DEF)
    # These players have higher variance - need wider form caps
    premium_threshold_mid_fwd: float = 8.0
    premium_threshold_def: float = 6.0
    premium_min_form_factor: float = 0.55  # Max 45% reduction for premiums
    premium_max_form_factor: float = 1.50  # Max 50% increase for premiums
    premium_form_weight: float = 0.70      # Higher weight on recent form (was 0.55)

    # Position-specific form sensitivity
    # FWDs are more streaky (binary goal outcomes)
    # GKPs are more consistent (saves are stable)
    position_form_sensitivity: Dict[int, float] = field(default_factory=lambda: {
        1: 0.60,  # GKP - less affected by form
        2: 0.80,  # DEF - moderate
        3: 0.90,  # MID - high
        4: 1.00,  # FWD - full effect
    })

    # Team form weight
    team_form_weight: float = 0.40
    min_team_form: float = 0.80
    max_team_form: float = 1.20


# Initialize global config
MODEL_CONFIG = {
    "fdr": FDRConfig(),
    "home_away": HomeAwayConfig(),
    "xpts": XptsConfig(),
    "variance": VarianceConfig(),
    "bonus": BonusConfig(),
    "clean_sheet": CleanSheetConfig(),
    "captain": CaptainConfig(),
    "form": FormConfig(),
}
