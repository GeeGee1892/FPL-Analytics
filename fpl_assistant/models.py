from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pydantic import BaseModel


# ============ ENUMS & MODELS ============

class Position(str, Enum):
    GKP = "GKP"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


class MinutesOverride(BaseModel):
    player_id: int
    expected_minutes: float
    manager_id: Optional[int] = None  # Optional for backwards compatibility


# ============ RESPONSE SCHEMAS ============
# These provide contract stability between frontend and backend

class PlayerRanking(BaseModel):
    """Schema for a player in rankings response."""
    id: int
    name: str
    team: str
    team_id: int
    position: str
    price: float
    ownership: float
    form: float
    total_points: int
    minutes: int
    xpts: float
    xpts_per_90: float
    xpts_per_cost: float
    exp_mins: float
    mins_reason: str
    fixture_ticker: str
    avg_fdr: float

    class Config:
        extra = "allow"  # Allow additional fields


class SquadPlayer(BaseModel):
    """Schema for a player in squad response."""
    id: int
    name: str
    team: str
    team_id: int
    position: str
    price: float
    selling_price: float
    is_captain: bool
    is_vice_captain: bool
    multiplier: int
    xpts_single_gw: float
    exp_mins: float
    fixture_ticker: str

    class Config:
        extra = "allow"


# =============================================================================
# MODEL DATA CLASSES
# =============================================================================

@dataclass
class HomeAwaySplit:
    """Player's home vs away performance statistics."""
    home_xG90: float = 0.0
    away_xG90: float = 0.0
    home_xA90: float = 0.0
    away_xA90: float = 0.0
    home_pts_per_90: float = 0.0
    away_pts_per_90: float = 0.0
    home_games: int = 0
    away_games: int = 0
    has_sufficient_data: bool = False

    @property
    def home_xGI90(self) -> float:
        return self.home_xG90 + self.home_xA90

    @property
    def away_xGI90(self) -> float:
        return self.away_xG90 + self.away_xA90


@dataclass
class CSProbability:
    """Clean sheet probability for a fixture."""
    cs_prob: float
    expected_goals_against: float
    data_source: str
    opponent_attack_strength: float = 1.0


@dataclass
class AttackingEstimate:
    """Attacking returns estimate for a fixture."""
    xG90: float
    xA90: float
    attack_multiplier: float
    fixture_xG90: float
    fixture_xA90: float
    data_source: str


@dataclass
class BonusEstimate:
    """Bonus points estimate."""
    expected_bonus: float
    estimated_bps: float
    bonus_per_90_historical: float
    teammate_dilution_applied: bool = False


@dataclass
class VarianceEstimate:
    """Ceiling and floor estimates."""
    ceiling: float
    floor: float
    std_dev: float
    data_source: str


@dataclass
class FixtureXpts:
    """xPts breakdown for a single fixture."""
    fixture_gw: int
    opponent_id: int
    opponent_name: str
    is_home: bool
    difficulty: int

    appearance_pts: float = 0.0
    goal_pts: float = 0.0
    assist_pts: float = 0.0
    cs_pts: float = 0.0
    bonus_pts: float = 0.0
    save_pts: float = 0.0
    defcon_pts: float = 0.0

    yellow_deduction: float = 0.0
    og_deduction: float = 0.0
    gc_deduction: float = 0.0

    cs_prob: float = 0.0
    expected_goals_against: float = 0.0
    attack_multiplier: float = 1.0

    @property
    def total_xpts(self) -> float:
        return (
            self.appearance_pts + self.goal_pts + self.assist_pts +
            self.cs_pts + self.bonus_pts + self.save_pts + self.defcon_pts -
            self.yellow_deduction - self.og_deduction - self.gc_deduction
        )


@dataclass
class PlayerFormResult:
    """Result of player form calculation."""
    xg_form: float = 1.0       # Recent xG90 / Season xG90
    xa_form: float = 1.0       # Recent xA90 / Season xA90
    pts_form: float = 1.0      # Recent PPG / Season PPG
    combined_form: float = 1.0 # Blended form factor for xGI
    confidence: float = 0.0    # Weight given to form (0-0.5)
    recent_xg90: float = 0.0
    recent_xa90: float = 0.0
    recent_ppg: float = 0.0
    season_ppg: float = 0.0
    games_used: int = 0


@dataclass
class TeamFormResult:
    """Result of team form calculation."""
    attack_form: float = 1.0   # Team's attacking form multiplier
    defence_form: float = 1.0  # Team's defensive form multiplier
    source: str = "default"
    attack_delta: float = 0.0  # Raw xG delta from Analytic FPL
    defence_delta: float = 0.0 # Raw xGA delta from Analytic FPL


@dataclass
class SquadHealthIssue:
    """Represents a health issue with a squad player."""
    player_id: int
    player_name: str
    team: str
    position: str
    issue_type: str  # injury, rotation, bgw_exposure, form_concern
    severity: str  # critical, warning, minor
    description: str
    affected_gws: List[int] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class FixtureSwing:
    """Represents a team's fixture swing trajectory."""
    team_id: int
    team_name: str
    team_short: str
    current_fdr: float  # Avg FDR for GW [now, now+3]
    upcoming_fdr: float  # Avg FDR for GW [now+3, now+6]
    swing: float  # negative = improving, positive = worsening
    dgw_in_horizon: List[int] = field(default_factory=list)
    bgw_in_horizon: List[int] = field(default_factory=list)
    rating: str = "NEUTRAL"  # IMPROVING, NEUTRAL, WORSENING


@dataclass
class TransferRecommendation:
    """A single transfer recommendation."""
    out_player: Dict
    in_player: Dict
    gw: int
    xpts_gain: float
    reason: str
    is_essential: bool  # Injury replacement, BGW fix
    is_hit: bool
    priority: int  # 1 = highest


@dataclass
class ChipRecommendation:
    """Chip usage recommendation."""
    chip: str  # wildcard, freehit, bboost, 3xc
    recommended_gw: int
    confidence: str  # high, medium, low
    reason: str
    expected_value: float  # xPts gained from chip


@dataclass
class ChipPlacement:
    """A chip actually placed/activated in a strategy plan."""
    chip: str           # wildcard, freehit, bboost, 3xc
    gw: int
    marginal_xpts: float
    reason: str = ""


@dataclass
class StrategyPlan:
    """A complete transfer strategy plan."""
    name: str  # safe, balanced, risky
    description: str
    gw_actions: Dict[int, Dict]  # gw -> {action, transfers, chip, reasoning}
    total_xpts: float
    hit_cost: int
    transfers_made: int
    chip_recommendations: List[ChipRecommendation]
    risk_score: float  # 0-10 (10 = riskiest)
    headline: str
    chip_placements: List[ChipPlacement] = field(default_factory=list)


class TransferPlannerConfig:
    """Configuration for different strategy types."""

    STRATEGIES = {
        "safe": {
            "min_xpts_gain_for_transfer": 1.5,  # Only transfer if gain > 1.5 xPts
            "min_xpts_gain_for_hit": 15.0,  # Very high - hits almost never worth it
            "max_hits_per_horizon": 0,  # No hits for safe
            "prefer_nailed": True,
            "ownership_preference": "template",  # High ownership = safe
            "ceiling_weight": 0.2,  # Low ceiling weight
            "floor_weight": 0.4,  # High floor weight
            "risk_score": 2,
            "description": "Conservative: No hits, only use free transfers for clear upgrades",
            "min_chip_marginal_xpts": 4.0,  # Only activate chips with high value
            "beam_width": 12,              # Wider beam — safe has fewer viable candidates
            "beam_candidates": 4,          # Top-N single transfers per path per GW
            "beam_max_pairs": 2,           # Top-M transfer pairs (if FT >= 2)
        },
        "balanced": {
            "min_xpts_gain_for_transfer": 1.0,
            "min_xpts_gain_for_hit": 12.0,  # High threshold - hit must be very valuable
            "max_hits_per_horizon": 1,  # At most 1 hit across entire horizon
            "prefer_nailed": False,
            "ownership_preference": "mixed",
            "ceiling_weight": 0.35,
            "floor_weight": 0.25,
            "risk_score": 5,
            "description": "Balanced: Free transfers for upgrades, max 1 hit only if exceptional value",
            "min_chip_marginal_xpts": 2.5,
            "beam_width": 10,
            "beam_candidates": 5,
            "beam_max_pairs": 3,
        },
        "risky": {
            "min_xpts_gain_for_transfer": 0.6,
            "min_xpts_gain_for_hit": 8.0,  # Still need 8+ xPts to justify -4 hit
            "max_hits_per_horizon": 2,  # Max 2 hits
            "prefer_nailed": False,
            "ownership_preference": "differential",  # Low ownership = upside
            "ceiling_weight": 0.5,  # High ceiling weight
            "floor_weight": 0.1,  # Low floor weight
            "risk_score": 8,
            "description": "Aggressive: Chase differentials, willing to take hits for high upside",
            "min_chip_marginal_xpts": 1.0,  # Use chips readily
            "beam_width": 8,               # Narrower beam but deeper candidates
            "beam_candidates": 6,
            "beam_max_pairs": 4,
        }
    }

    # Roll value constants
    ROLL_VALUE = 0.5  # Slightly higher - rolling is valuable
    MAX_FT = 5  # 2024/25 rules


class BookedTransfer(BaseModel):
    """Pydantic model for booked transfer input."""
    out_id: int
    in_id: int
    gw: int
    reason: Optional[str] = ""


class ChipOverride(BaseModel):
    """User override for chip placement: lock to a GW or block entirely."""
    chip: str          # "wildcard", "freehit", "bboost", "3xc"
    action: str        # "lock" (force to gw) or "block" (never use)
    gw: Optional[int] = None  # Required for "lock", ignored for "block"


class TransferPlannerRequest(BaseModel):
    """Request body for transfer planner."""
    booked_transfers: Optional[List[BookedTransfer]] = []
    chip_overrides: Optional[List[ChipOverride]] = []
