"""
FPL Assistant - Backtest Validation Module

Core logic for comparing xPts predictions against actual historical results.
Callable from API endpoints, tests, or CLI.

CAVEAT: The FPL API only provides current-season cumulative stats. When
backtesting GW N, we use the CURRENT season stats (through latest GW),
not stats-as-of-GW-(N-1). This introduces look-ahead bias. Results should be
interpreted as "how well does the model's STRUCTURE capture reality?" rather
than "how well would the model have performed live?"
"""

import asyncio
import math
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, List, Dict

from fpl_assistant.constants import POSITION_MAP, POSITION_ID_MAP
from fpl_assistant.models import (
    BacktestPrediction, SegmentStats, ComponentAccuracy, BacktestResult,
)
from fpl_assistant.services import (
    calculate_expected_points, get_player_upcoming_fixtures,
)

logger = logging.getLogger("fpl_assistant")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_correlation(predicted: List[float], actual: List[float]) -> Optional[float]:
    """Pearson correlation coefficient. Returns None if < 3 data points or zero variance."""
    n = len(predicted)
    if n < 3:
        return None

    mean_p = sum(predicted) / n
    mean_a = sum(actual) / n

    num = sum((p - mean_p) * (a - mean_a) for p, a in zip(predicted, actual))
    denom_p = math.sqrt(sum((p - mean_p) ** 2 for p in predicted))
    denom_a = math.sqrt(sum((a - mean_a) ** 2 for a in actual))

    if denom_p == 0 or denom_a == 0:
        return None
    return num / (denom_p * denom_a)


def _compute_segment_stats(label: str, predictions: List[BacktestPrediction]) -> SegmentStats:
    """Compute MAE, RMSE, correlation, bias for a list of predictions."""
    n = len(predictions)
    if n == 0:
        return SegmentStats(label=label, count=0, mae=0, rmse=0, avg_error=0, correlation=None)

    errors = [p.error for p in predictions]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
    avg_error = sum(errors) / n

    predicted = [p.predicted_xpts for p in predictions]
    actual = [float(p.actual_points) for p in predictions]
    correlation = _compute_correlation(predicted, actual)

    return SegmentStats(
        label=label,
        count=n,
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        avg_error=round(avg_error, 2),
        correlation=round(correlation, 3) if correlation is not None else None,
    )


def _compute_segmented_breakdown(
    predictions: List[BacktestPrediction],
    key_fn,
) -> List[SegmentStats]:
    """Group predictions by a key function and compute stats for each group."""
    groups = defaultdict(list)
    for p in predictions:
        groups[key_fn(p)].append(p)

    segments = []
    for label in sorted(groups.keys()):
        segments.append(_compute_segment_stats(label, groups[label]))
    return segments


def _compute_component_accuracy(
    predictions: List[BacktestPrediction],
) -> List[ComponentAccuracy]:
    """
    Compare predicted component rates vs actual outcomes.

    - Clean sheets: predicted cs_prob vs actual CS rate
    - Bonus: average predicted bonus vs average actual bonus
    - Goals: approximate — we don't store predicted goal count per GW,
      but we can compare rates
    """
    if not predictions:
        return []

    components = []
    n = len(predictions)

    # Clean sheet calibration (DEF/GKP only — positions where CS matters most)
    cs_preds = [p for p in predictions if p.position in ("GKP", "DEF")]
    if cs_preds:
        # We don't store predicted CS prob per prediction (it's aggregated in xpts),
        # so use actual CS rate across all DEF/GKP predictions as a proxy metric
        actual_cs_rate = sum(1 for p in cs_preds if p.actual_cs > 0) / len(cs_preds)
        # Rough predicted CS rate: assume ~35% base for top-half defence
        # This is a placeholder — actual predicted_cs_prob would need to be stored
        # For now, compute from prediction vs actual points differential
        components.append(ComponentAccuracy(
            component="clean_sheets",
            predicted_rate=0,  # Will be populated if we add cs_prob to predictions
            actual_rate=round(actual_cs_rate, 3),
            calibration_error=0,
            count=len(cs_preds),
        ))

    # Bonus calibration
    avg_predicted_bonus = 0  # placeholder — expected_bonus not stored per prediction
    avg_actual_bonus = sum(p.actual_bonus for p in predictions) / n
    components.append(ComponentAccuracy(
        component="bonus",
        predicted_rate=round(avg_predicted_bonus, 3),
        actual_rate=round(avg_actual_bonus, 3),
        calibration_error=round(abs(avg_predicted_bonus - avg_actual_bonus), 3),
        count=n,
    ))

    # Goals rate (per appearance)
    avg_actual_goals = sum(p.actual_goals for p in predictions) / n
    components.append(ComponentAccuracy(
        component="goals",
        predicted_rate=0,
        actual_rate=round(avg_actual_goals, 3),
        calibration_error=0,
        count=n,
    ))

    # Assists rate (per appearance)
    avg_actual_assists = sum(p.actual_assists for p in predictions) / n
    components.append(ComponentAccuracy(
        component="assists",
        predicted_rate=0,
        actual_rate=round(avg_actual_assists, 3),
        calibration_error=0,
        count=n,
    ))

    return components


# =============================================================================
# CORE BACKTEST FUNCTION (SYNC)
# =============================================================================

def run_backtest_sync(
    elements: List[Dict],
    fixtures: List[Dict],
    teams_dict: Dict[int, Dict],
    events: List[Dict],
    player_histories: Dict[int, Dict],
    gw_start: int = 1,
    gw_end: int = 10,
    position_filter: Optional[str] = None,
    min_season_minutes: int = 200,
    min_gw_minutes: int = 45,
    player_ids: Optional[List[int]] = None,
) -> BacktestResult:
    """
    Run backtest comparing predicted xPts vs actual points.

    This function is synchronous and takes pre-fetched data as arguments,
    making it fully testable without async/HTTP mocking.
    """
    # Filter elements
    filtered = elements
    if position_filter and position_filter in POSITION_ID_MAP:
        pos_id = POSITION_ID_MAP[position_filter]
        filtered = [e for e in filtered if e["element_type"] == pos_id]
    if min_season_minutes > 0:
        filtered = [e for e in filtered if e.get("minutes", 0) >= min_season_minutes]
    if player_ids:
        id_set = set(player_ids)
        filtered = [e for e in filtered if e["id"] in id_set]

    # Determine finished GWs
    finished_gws = set()
    for ev in events:
        gw = ev.get("id")
        if gw and gw_start <= gw <= gw_end and ev.get("finished"):
            finished_gws.add(gw)

    if not finished_gws:
        return BacktestResult(
            gw_start=gw_start, gw_end=gw_end, total_predictions=0,
            overall_mae=0, overall_rmse=0, overall_correlation=None, overall_bias=0,
            by_position=[], by_price_tier=[], by_fdr=[], by_gameweek=[],
            component_accuracy=[], predictions=[],
            look_ahead_bias=True, timestamp=datetime.now(timezone.utc).isoformat(),
        )

    predictions = []

    for gw in sorted(finished_gws):
        for player in filtered:
            pid = player["id"]
            history = player_histories.get(pid)
            if not history:
                continue

            # Find this GW's history entry
            gw_data = None
            for h in history.get("history", []):
                if h.get("round") == gw:
                    gw_data = h
                    break
            if not gw_data:
                continue

            actual_minutes = gw_data.get("minutes", 0)
            if actual_minutes < min_gw_minutes:
                continue

            # Get fixture info for prediction
            position_id = player["element_type"]
            upcoming = get_player_upcoming_fixtures(
                player["team"], fixtures, gw, gw, teams_dict, position_id
            )
            if not upcoming:
                continue

            # Calculate predicted xPts
            try:
                stats = calculate_expected_points(
                    player, position_id, gw, upcoming, teams_dict, fixtures, events
                )
            except Exception as e:
                logger.debug(f"Backtest xPts calc failed for player {pid} GW{gw}: {e}")
                continue

            predicted_xpts = stats.get("xpts", 0)
            predicted_minutes = stats.get("expected_minutes", 0)

            # Extract fixture context from the first upcoming fixture
            fix = upcoming[0]
            opponent_id = fix.get("opponent_id", 0)
            opponent_name = fix.get("opponent", "???")
            is_home = fix.get("is_home", True)
            fdr = fix.get("difficulty", 5)

            predictions.append(BacktestPrediction(
                player_id=pid,
                player_name=player.get("web_name", "Unknown"),
                team_short=teams_dict.get(player["team"], {}).get("short_name", "???"),
                position=POSITION_MAP.get(position_id, "MID"),
                position_id=position_id,
                price=player.get("now_cost", 50) / 10,
                gameweek=gw,
                predicted_xpts=round(predicted_xpts, 2),
                predicted_minutes=round(predicted_minutes, 1),
                actual_points=gw_data.get("total_points", 0),
                actual_minutes=actual_minutes,
                actual_goals=gw_data.get("goals_scored", 0),
                actual_assists=gw_data.get("assists", 0),
                actual_cs=gw_data.get("clean_sheets", 0),
                actual_bonus=gw_data.get("bonus", 0),
                opponent_id=opponent_id,
                opponent_name=opponent_name,
                is_home=is_home,
                fdr=fdr,
            ))

    # Aggregate results
    if not predictions:
        return BacktestResult(
            gw_start=gw_start, gw_end=gw_end, total_predictions=0,
            overall_mae=0, overall_rmse=0, overall_correlation=None, overall_bias=0,
            by_position=[], by_price_tier=[], by_fdr=[], by_gameweek=[],
            component_accuracy=[], predictions=[],
            look_ahead_bias=True, timestamp=datetime.now(timezone.utc).isoformat(),
        )

    overall = _compute_segment_stats("overall", predictions)

    by_position = _compute_segmented_breakdown(predictions, lambda p: p.position)
    by_price_tier = _compute_segmented_breakdown(predictions, lambda p: p.price_tier)
    by_fdr = _compute_segmented_breakdown(predictions, lambda p: p.fdr_bucket)
    by_gameweek = _compute_segmented_breakdown(predictions, lambda p: f"GW{p.gameweek}")
    component_accuracy = _compute_component_accuracy(predictions)

    return BacktestResult(
        gw_start=gw_start,
        gw_end=gw_end,
        total_predictions=len(predictions),
        overall_mae=overall.mae,
        overall_rmse=overall.rmse,
        overall_correlation=overall.correlation,
        overall_bias=overall.avg_error,
        by_position=by_position,
        by_price_tier=by_price_tier,
        by_fdr=by_fdr,
        by_gameweek=by_gameweek,
        component_accuracy=component_accuracy,
        predictions=predictions,
        look_ahead_bias=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# ASYNC WRAPPER
# =============================================================================

async def run_backtest_async(
    gw_start: int = 1,
    gw_end: int = 10,
    position_filter: Optional[str] = None,
    min_season_minutes: int = 200,
    min_gw_minutes: int = 45,
    player_ids: Optional[List[int]] = None,
    batch_size: int = 50,
) -> BacktestResult:
    """
    Async wrapper that fetches required data then runs the sync backtest.
    Fetches player histories in batches to avoid overwhelming the FPL API.
    """
    from fpl_assistant.services import (
        fetch_fpl_data, fetch_fixtures, refresh_fdr_data,
        fetch_player_history,
    )

    await refresh_fdr_data()
    data = await fetch_fpl_data()
    fixtures = await fetch_fixtures()

    elements = data["elements"]
    teams_dict = {t["id"]: t for t in data["teams"]}
    events = data["events"]

    # Pre-filter elements to minimize history fetches
    filtered = elements
    if position_filter and position_filter in POSITION_ID_MAP:
        pos_id = POSITION_ID_MAP[position_filter]
        filtered = [e for e in filtered if e["element_type"] == pos_id]
    if min_season_minutes > 0:
        filtered = [e for e in filtered if e.get("minutes", 0) >= min_season_minutes]
    if player_ids:
        id_set = set(player_ids)
        filtered = [e for e in filtered if e["id"] in id_set]

    # Batch-fetch player histories
    player_histories = {}
    for i in range(0, len(filtered), batch_size):
        batch = filtered[i:i + batch_size]
        histories = await asyncio.gather(*[
            fetch_player_history(p["id"]) for p in batch
        ])
        for player, history in zip(batch, histories):
            player_histories[player["id"]] = history

    return run_backtest_sync(
        elements=elements,
        fixtures=fixtures,
        teams_dict=teams_dict,
        events=events,
        player_histories=player_histories,
        gw_start=gw_start,
        gw_end=gw_end,
        position_filter=position_filter,
        min_season_minutes=min_season_minutes,
        min_gw_minutes=min_gw_minutes,
        player_ids=player_ids,
    )


# =============================================================================
# SERIALIZATION
# =============================================================================

def _segment_to_dict(s: SegmentStats) -> Dict:
    return {
        "label": s.label,
        "count": s.count,
        "mae": s.mae,
        "rmse": s.rmse,
        "avg_error": s.avg_error,
        "correlation": s.correlation,
    }


def _component_to_dict(c: ComponentAccuracy) -> Dict:
    return {
        "component": c.component,
        "predicted_rate": c.predicted_rate,
        "actual_rate": c.actual_rate,
        "calibration_error": c.calibration_error,
        "count": c.count,
    }


def _prediction_to_dict(p: BacktestPrediction) -> Dict:
    return {
        "id": p.player_id,
        "name": p.player_name,
        "team": p.team_short,
        "position": p.position,
        "price": p.price,
        "gw": p.gameweek,
        "predicted": p.predicted_xpts,
        "actual": p.actual_points,
        "error": round(p.error, 2),
        "abs_error": round(p.abs_error, 2),
        "opponent": p.opponent_name,
        "is_home": p.is_home,
        "fdr": p.fdr,
        "minutes": p.actual_minutes,
    }


def backtest_result_to_dict(result: BacktestResult) -> Dict:
    """Convert BacktestResult to a JSON-serializable dict for API response."""
    return {
        "gw_range": f"GW{result.gw_start}-GW{result.gw_end}",
        "total_predictions": result.total_predictions,
        "look_ahead_bias": result.look_ahead_bias,
        "timestamp": result.timestamp,
        "overall": {
            "mae": result.overall_mae,
            "rmse": result.overall_rmse,
            "correlation": result.overall_correlation,
            "bias": result.overall_bias,
        },
        "by_position": [_segment_to_dict(s) for s in result.by_position],
        "by_price_tier": [_segment_to_dict(s) for s in result.by_price_tier],
        "by_fdr": [_segment_to_dict(s) for s in result.by_fdr],
        "by_gameweek": [_segment_to_dict(s) for s in result.by_gameweek],
        "component_accuracy": [_component_to_dict(c) for c in result.component_accuracy],
        "worst_predictions": [_prediction_to_dict(p) for p in
                              sorted(result.predictions, key=lambda x: -x.abs_error)[:15]],
        "best_predictions": [_prediction_to_dict(p) for p in
                             sorted(result.predictions, key=lambda x: x.abs_error)[:15]],
    }
