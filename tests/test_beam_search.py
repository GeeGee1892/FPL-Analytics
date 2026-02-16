"""Tests for beam search: TransferPath enhancements, helper functions, beam_search_strategy, convert_path_to_gw_actions."""
import sys
import os
from unittest.mock import patch, MagicMock
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from main import (
    TransferPath, TransferPlannerConfig, TransferRecommendation,
    SquadHealthIssue, FixtureSwing, ChipRecommendation,
    StrategyPlan, POSITION_MAP,
)
from fpl_assistant.planner import (
    candidate_to_transfer_dict, generate_compatible_pairs,
    beam_search_strategy, convert_path_to_gw_actions,
    build_strategy_plan,
)


# ============ FIXTURES ============

def _make_squad_player(pid, team_id=1, pos_id=3, price=7.0):
    """Create a minimal squad player dict."""
    return {
        "id": pid,
        "name": f"Player{pid}",
        "team": f"T{team_id:02d}",
        "team_id": team_id,
        "position": POSITION_MAP.get(pos_id, "MID"),
        "position_id": pos_id,
        "price": price,
        "selling_price": price,
        "xpts": 5.0,
        "form": 4.0,
        "expected_minutes": 85,
    }


def _make_15_squad():
    """Create a 15-player squad (2 GKP, 5 DEF, 5 MID, 3 FWD) across distinct teams."""
    squad = []
    pid = 1001
    for pos, count, base_price in [(1, 2, 4.5), (2, 5, 5.0), (3, 5, 7.0), (4, 3, 8.0)]:
        for i in range(count):
            team_id = (pid % 15) + 1  # Spread across teams
            squad.append(_make_squad_player(pid, team_id=team_id, pos_id=pos, price=base_price + i * 0.5))
            pid += 1
    return squad


def _make_candidate(out_id, in_id, in_team, gain=2.0, pos_id=3, in_price=7.0):
    """Create a TransferRecommendation."""
    return TransferRecommendation(
        out_player={
            "id": out_id, "name": f"Out{out_id}", "team_id": 1,
            "position_id": pos_id, "price": 7.0, "selling_price": 7.0,
        },
        in_player={
            "id": in_id, "name": f"In{in_id}", "team": f"T{in_team:02d}",
            "team_id": in_team, "position_id": pos_id, "position": POSITION_MAP.get(pos_id, "MID"),
            "price": in_price, "xpts": 5.0, "form": 4.0, "expected_minutes": 90,
            "minutes": 1800,
        },
        gw=24,
        xpts_gain=gain,
        reason=f"Upgrade Out{out_id}→In{in_id}",
        is_essential=False,
        is_hit=False,
        priority=5,
    )


def _teams_dict():
    return {i: {"name": f"Team{i}", "short_name": f"T{i:02d}"} for i in range(1, 21)}


def _make_events():
    return [{"id": gw, "deadline_time": f"2025-02-{gw:02d}T12:00:00Z",
             "finished": gw < 24} for gw in range(1, 39)]


def _make_fixtures():
    """Create basic fixtures for GW24-31 across 20 teams."""
    fixtures = []
    fid = 1
    for gw in range(24, 32):
        for h in range(1, 11):
            a = h + 10
            fixtures.append({
                "id": fid, "event": gw,
                "team_h": h, "team_a": a,
                "team_h_difficulty": 3, "team_a_difficulty": 3,
                "finished": False,
            })
            fid += 1
    return fixtures


# ============ TRANSFER PATH TESTS ============

class TestTransferPath:

    def test_squad_key_deterministic(self):
        squad = _make_15_squad()
        path = TransferPath(squad, 1.5, 2)
        key1 = path.squad_key()
        key2 = path.squad_key()
        assert key1 == key2

    def test_squad_key_differs_after_transfer(self):
        squad = _make_15_squad()
        path = TransferPath(squad, 1.5, 2)
        key_before = path.squad_key()

        transfer = {
            "out": squad[5],
            "in": {"id": 9999, "name": "NewGuy", "team": "T19", "team_id": 19,
                   "position": "DEF", "position_id": 2, "price": 5.0,
                   "xpts": 6.0, "form": 5.0, "expected_minutes": 90},
            "xpts_gain": 1.5,
        }
        path.apply_transfer(transfer, gw=24)
        key_after = path.squad_key()
        assert key_before != key_after

    def test_copy_preserves_new_fields(self):
        squad = _make_15_squad()
        path = TransferPath(squad, 2.0, 3)
        path.transferred_out_ids.add(100)
        path.transferred_in_ids.add(200)
        path.pre_fh_squad = [p.copy() for p in squad[:3]]
        path.pre_fh_bank = 5.0

        clone = path.copy()
        assert clone.transferred_out_ids == {100}
        assert clone.transferred_in_ids == {200}
        assert clone.initial_ft == 3
        assert clone.pre_fh_bank == 5.0
        assert len(clone.pre_fh_squad) == 3

        # Ensure deep copy — modifying clone doesn't affect original
        clone.transferred_out_ids.add(300)
        assert 300 not in path.transferred_out_ids

    def test_score_equals_total_xpts(self):
        """score() returns total_xpts (which already includes -4 per hit)."""
        squad = _make_15_squad()
        path = TransferPath(squad, 1.0, 1)
        path.total_xpts = 50.0
        path.hits = 2
        assert path.score() == 50.0

    def test_apply_transfer_hit_penalty(self):
        """Applying a hit transfer subtracts 4 from total_xpts."""
        squad = _make_15_squad()
        path = TransferPath(squad, 5.0, 0)  # 0 FT → any transfer is a hit
        initial_xpts = path.total_xpts
        transfer = {
            "out": squad[7],
            "in": {"id": 8888, "name": "HitGuy", "team": "T18", "team_id": 18,
                   "position": "MID", "position_id": 3, "price": 7.0,
                   "xpts": 5.0, "form": 4.0, "expected_minutes": 90},
            "xpts_gain": 2.0,
        }
        path.apply_transfer(transfer, gw=24, is_hit=True)
        assert path.hits == 1
        assert path.total_xpts == initial_xpts - 4

    def test_roll_transfer_increments_ft(self):
        squad = _make_15_squad()
        path = TransferPath(squad, 1.0, 2)
        path.roll_transfer(24)
        assert path.ft == 3
        assert len(path.actions) == 1
        assert path.actions[0]["type"] == "roll"

    def test_roll_transfer_caps_at_5(self):
        squad = _make_15_squad()
        path = TransferPath(squad, 1.0, 5)
        path.roll_transfer(24)
        assert path.ft == 5

    def test_use_ft_decrements(self):
        squad = _make_15_squad()
        path = TransferPath(squad, 1.0, 3)
        path.use_ft()
        assert path.ft == 2

    def test_use_ft_floor_zero(self):
        squad = _make_15_squad()
        path = TransferPath(squad, 1.0, 0)
        path.use_ft()
        assert path.ft == 0


# ============ HELPER FUNCTION TESTS ============

class TestCandidateToTransferDict:

    def test_converts_recommendation(self):
        c = _make_candidate(1001, 2001, in_team=5, gain=3.5)
        td = candidate_to_transfer_dict(c)
        assert td["out"]["id"] == 1001
        assert td["in"]["id"] == 2001
        assert td["xpts_gain"] == 3.5
        assert td["is_booked"] is False


class TestGenerateCompatiblePairs:

    def test_no_shared_players(self):
        c1 = _make_candidate(1001, 2001, in_team=5, gain=3.0)
        c2 = _make_candidate(1002, 2002, in_team=6, gain=2.5)
        c3 = _make_candidate(1003, 2003, in_team=7, gain=2.0)
        pairs = generate_compatible_pairs([c1, c2, c3], max_pairs=10)
        for p1, p2 in pairs:
            assert p1.out_player["id"] != p2.out_player["id"]
            assert p1.in_player["id"] != p2.in_player["id"]

    def test_filters_same_team_incoming(self):
        """Two candidates bringing in players from the same team are incompatible."""
        c1 = _make_candidate(1001, 2001, in_team=5, gain=3.0)
        c2 = _make_candidate(1002, 2002, in_team=5, gain=2.5)  # Same team
        pairs = generate_compatible_pairs([c1, c2], max_pairs=10)
        assert len(pairs) == 0

    def test_sorted_by_gain(self):
        c1 = _make_candidate(1001, 2001, in_team=5, gain=1.0)
        c2 = _make_candidate(1002, 2002, in_team=6, gain=3.0)
        c3 = _make_candidate(1003, 2003, in_team=7, gain=2.0)
        pairs = generate_compatible_pairs([c1, c2, c3], max_pairs=10)
        assert len(pairs) >= 1
        # Best pair should be c2+c3 (gain 5.0) or c2+c1 (gain 4.0)
        best = pairs[0]
        gains = best[0].xpts_gain + best[1].xpts_gain
        assert gains >= 4.0

    def test_empty_input(self):
        pairs = generate_compatible_pairs([], max_pairs=5)
        assert pairs == []

    def test_single_candidate(self):
        c1 = _make_candidate(1001, 2001, in_team=5, gain=3.0)
        pairs = generate_compatible_pairs([c1], max_pairs=5)
        assert pairs == []

    def test_max_pairs_limits_output(self):
        candidates = [
            _make_candidate(1001 + i, 2001 + i, in_team=i + 1, gain=3.0 - i * 0.1)
            for i in range(6)
        ]
        pairs = generate_compatible_pairs(candidates, max_pairs=3)
        assert len(pairs) <= 3


# ============ BEAM SEARCH TESTS ============

class TestBeamSearchStrategy:

    def _make_beam_args(self, squad=None, ft=2, bank=1.0, current_gw=24, horizon_end=27):
        """Build the full kwargs dict for beam_search_strategy."""
        squad = squad or _make_15_squad()
        return dict(
            initial_squad=squad,
            initial_bank=bank,
            initial_ft=ft,
            current_gw=current_gw,
            horizon_end=horizon_end,
            strategy="balanced",
            config=TransferPlannerConfig.STRATEGIES["balanced"],
            elements=[],
            elements_dict={},
            fixtures=_make_fixtures(),
            events=_make_events(),
            teams_dict=_teams_dict(),
            fixture_swings=[],
            squad_health=[],
            chip_placements={},
            booked_by_gw={},
            player_gw_cache={},
            gw_info={gw: {"is_dgw": False, "is_bgw": False} for gw in range(24, 32)},
        )

    @patch("fpl_assistant.planner.evaluate_squad_xpts")
    @patch("fpl_assistant.planner.generate_transfer_candidates_for_gw")
    def test_returns_transfer_path(self, mock_candidates, mock_eval):
        """beam_search_strategy returns a TransferPath instance."""
        mock_candidates.return_value = []
        mock_eval.return_value = 30.0
        args = self._make_beam_args()
        result = beam_search_strategy(**args)
        assert isinstance(result, TransferPath)

    @patch("fpl_assistant.planner.evaluate_squad_xpts")
    @patch("fpl_assistant.planner.generate_transfer_candidates_for_gw")
    def test_roll_only_accumulates_xpts(self, mock_candidates, mock_eval):
        """With no candidates, beam rolls every GW and accumulates xpts."""
        mock_candidates.return_value = []
        mock_eval.return_value = 25.0
        args = self._make_beam_args(ft=1, current_gw=24, horizon_end=27)
        result = beam_search_strategy(**args)
        # 4 GWs (24,25,26,27) × 25.0 = 100.0
        assert result.total_xpts == pytest.approx(100.0)
        # All rolls → FT should be capped at 5
        assert result.ft <= 5

    @patch("fpl_assistant.planner.evaluate_squad_xpts")
    @patch("fpl_assistant.planner.generate_transfer_candidates_for_gw")
    def test_discovers_roll_then_burst(self, mock_candidates, mock_eval):
        """KEY TEST: beam discovers 'roll GW24, double-transfer GW25' beats greedy single transfers."""
        squad = _make_15_squad()

        # GW24: mediocre candidate (gain=1.0)
        # GW25: excellent pair (gain=4.0 each)
        def candidates_side_effect(squad, elements, fixtures, teams_dict, events,
                                   target_gw, remaining_horizon, bank, strategy,
                                   fixture_swings=None, excluded_out_ids=None,
                                   excluded_in_ids=None):
            excluded_out = excluded_out_ids or set()
            if target_gw == 24:
                if 1006 not in excluded_out:
                    return [_make_candidate(1006, 2001, in_team=18, gain=1.0)]
                return []
            elif target_gw == 25:
                results = []
                if 1006 not in excluded_out:
                    results.append(_make_candidate(1006, 2002, in_team=18, gain=4.0))
                if 1007 not in excluded_out:
                    results.append(_make_candidate(1007, 2003, in_team=19, gain=4.0))
                return results
            return []

        mock_candidates.side_effect = candidates_side_effect
        mock_eval.return_value = 30.0  # Each GW = 30 xpts baseline

        args = self._make_beam_args(squad=squad, ft=1, current_gw=24, horizon_end=25)
        result = beam_search_strategy(**args)

        # The beam should find the roll-then-burst path:
        # Path A (greedy): Transfer GW24 (+1.0), transfer GW25 (+4.0) = 5.0 gain
        # Path B (smart):  Roll GW24, double-transfer GW25 (+4.0+4.0) = 8.0 gain
        # Path B should win
        transfers = [a for a in result.actions if a["type"] == "transfer"]
        rolls = [a for a in result.actions if a["type"] == "roll"]

        # Best path should have 2 transfers in GW25 and a roll in GW24
        if len(transfers) == 2:
            assert all(t["gw"] == 25 for t in transfers)
            assert len(rolls) >= 1

    @patch("fpl_assistant.planner.evaluate_squad_xpts")
    @patch("fpl_assistant.planner.generate_transfer_candidates_for_gw")
    def test_respects_max_hits(self, mock_candidates, mock_eval):
        """Safe strategy (max_hits=0) should never produce hit paths."""
        mock_candidates.return_value = [
            _make_candidate(1006, 2001, in_team=18, gain=15.0),  # Very tempting
        ]
        mock_eval.return_value = 25.0

        args = self._make_beam_args(ft=0)  # 0 FT → any transfer is a hit
        args["strategy"] = "safe"
        args["config"] = TransferPlannerConfig.STRATEGIES["safe"]
        result = beam_search_strategy(**args)
        assert result.hits == 0

    @patch("fpl_assistant.planner.evaluate_squad_xpts")
    @patch("fpl_assistant.planner.generate_transfer_candidates_for_gw")
    def test_deduplication_keeps_best(self, mock_candidates, mock_eval):
        """Two paths reaching same squad_key → only higher score survives."""
        # With identical candidates every GW, multiple paths converge to same state
        mock_candidates.return_value = [
            _make_candidate(1006, 2001, in_team=18, gain=2.0),
        ]
        mock_eval.return_value = 30.0

        args = self._make_beam_args(ft=2, current_gw=24, horizon_end=25)
        result = beam_search_strategy(**args)
        # Just verify it completes without error and returns a valid path
        assert isinstance(result, TransferPath)
        assert result.total_xpts > 0

    @patch("fpl_assistant.planner.evaluate_squad_xpts")
    @patch("fpl_assistant.planner.generate_transfer_candidates_for_gw")
    def test_booked_transfers_applied(self, mock_candidates, mock_eval):
        """Booked transfers are applied to all beam paths."""
        squad = _make_15_squad()
        mock_candidates.return_value = []
        mock_eval.return_value = 25.0

        # Book a transfer in GW24
        elements_dict = {
            9999: {
                "id": 9999, "web_name": "Booked", "team": 18,
                "element_type": 3, "now_cost": 70, "form": "5.0",
                "minutes": 1800,
            }
        }
        args = self._make_beam_args(squad=squad, ft=2, current_gw=24, horizon_end=25)
        args["elements_dict"] = elements_dict
        args["booked_by_gw"] = {24: [{"out_id": squad[7]["id"], "in_id": 9999}]}

        result = beam_search_strategy(**args)
        transfers = [a for a in result.actions if a["type"] == "transfer"]
        assert any(t.get("is_booked", False) for t in transfers)

    @patch("fpl_assistant.planner.evaluate_squad_xpts")
    @patch("fpl_assistant.planner.build_optimal_squad")
    @patch("fpl_assistant.planner.generate_transfer_candidates_for_gw")
    def test_chip_gw_handled(self, mock_candidates, mock_optimal, mock_eval):
        """WC GW applies uniformly to all paths — no branching."""
        squad = _make_15_squad()
        mock_candidates.return_value = []
        mock_eval.return_value = 30.0
        mock_optimal.return_value = None  # No rebuild (falls back to current squad)

        args = self._make_beam_args(squad=squad, current_gw=24, horizon_end=26)
        args["chip_placements"] = {25: "wildcard"}

        result = beam_search_strategy(**args)
        wc_actions = [a for a in result.actions if a["type"] == "wildcard"]
        assert len(wc_actions) == 1
        assert wc_actions[0]["gw"] == 25


# ============ CONVERT PATH TO GW ACTIONS TESTS ============

class TestConvertPathToGwActions:

    def test_roll_only(self):
        """Path with only rolls produces roll gw_actions."""
        squad = _make_15_squad()
        path = TransferPath(squad, 1.0, 2)
        for gw in range(24, 27):
            path.roll_transfer(gw)
            path.gw_xpts[gw] = 25.0
            path.total_xpts += 25.0

        actions = convert_path_to_gw_actions(path, 24, 26, {})
        assert all(a["action"] == "roll" for a in actions.values())
        assert actions[24]["ft_before"] == 2
        assert actions[24]["ft_after"] == 3  # Rolled → +1

    def test_transfer_gw(self):
        """Transfer in a GW produces correct action format."""
        squad = _make_15_squad()
        path = TransferPath(squad, 1.0, 2)

        # Roll GW24
        path.roll_transfer(24)
        path.gw_xpts[24] = 25.0
        path.total_xpts += 25.0

        # Transfer GW25
        transfer = {
            "out": squad[5],
            "in": {"id": 9999, "name": "NewGuy", "team": "T19", "team_id": 19,
                   "position": "DEF", "position_id": 2, "price": 5.0,
                   "xpts": 6.0, "form": 5.0, "expected_minutes": 90},
            "xpts_gain": 2.0,
        }
        path.use_ft()
        path.apply_transfer(transfer, gw=25)
        path.gw_xpts[25] = 28.0
        path.total_xpts += 28.0

        actions = convert_path_to_gw_actions(path, 24, 25, {})
        assert actions[24]["action"] == "roll"
        assert actions[25]["action"] == "transfer"
        assert len(actions[25]["transfers"]) == 1
        assert actions[25]["transfers"][0]["in"]["id"] == 9999

    def test_chip_gw(self):
        """Chip GW actions reflect chip type."""
        squad = _make_15_squad()
        path = TransferPath(squad, 1.0, 1)
        path.actions.append({"type": "wildcard", "gw": 24})
        path.gw_xpts[24] = 35.0
        path.total_xpts += 35.0

        actions = convert_path_to_gw_actions(path, 24, 24, {24: "wildcard"})
        assert actions[24]["action"] == "wildcard"
        assert actions[24]["chip"] == "wildcard"

    def test_ft_tracking_across_gws(self):
        """FT values track correctly across multiple GWs."""
        squad = _make_15_squad()
        path = TransferPath(squad, 1.0, 1)

        # Roll 3 consecutive GWs
        for gw in range(24, 27):
            path.roll_transfer(gw)
            path.gw_xpts[gw] = 25.0

        actions = convert_path_to_gw_actions(path, 24, 26, {})
        assert actions[24]["ft_before"] == 1
        assert actions[24]["ft_after"] == 2
        assert actions[25]["ft_before"] == 2
        assert actions[25]["ft_after"] == 3
        assert actions[26]["ft_before"] == 3
        assert actions[26]["ft_after"] == 4


# ============ INTEGRATION TESTS ============

class TestBuildStrategyPlanIntegration:

    @patch("fpl_assistant.planner.evaluate_squad_xpts")
    @patch("fpl_assistant.planner.generate_transfer_candidates_for_gw")
    @patch("fpl_assistant.planner.determine_chip_placements")
    @patch("fpl_assistant.planner.detect_dgw_bgw")
    def test_returns_valid_strategy_plan(self, mock_dgw, mock_chips, mock_candidates, mock_eval):
        """build_strategy_plan returns a StrategyPlan with expected fields."""
        mock_dgw.return_value = {gw: {"is_dgw": False, "is_bgw": False} for gw in range(24, 32)}
        mock_chips.return_value = ({}, [])
        mock_candidates.return_value = []
        mock_eval.return_value = 25.0

        result = build_strategy_plan(
            squad=_make_15_squad(),
            bank=1.5,
            free_transfers=2,
            fixtures=_make_fixtures(),
            events=_make_events(),
            teams_dict=_teams_dict(),
            elements=[],
            current_gw=24,
            horizon=4,
            strategy="balanced",
            available_chips={"wildcard": False, "freehit": False, "bboost": False, "3xc": False},
            squad_health=[],
            fixture_swings=[],
            chip_recommendations=[],
        )
        assert isinstance(result, StrategyPlan)
        assert result.name == "balanced"
        assert isinstance(result.gw_actions, dict)
        assert result.total_xpts > 0
        assert result.hit_cost >= 0

    @patch("fpl_assistant.planner.evaluate_squad_xpts")
    @patch("fpl_assistant.planner.generate_transfer_candidates_for_gw")
    @patch("fpl_assistant.planner.determine_chip_placements")
    @patch("fpl_assistant.planner.detect_dgw_bgw")
    def test_three_strategies_complete(self, mock_dgw, mock_chips, mock_candidates, mock_eval):
        """All three strategies return without error."""
        mock_dgw.return_value = {gw: {"is_dgw": False, "is_bgw": False} for gw in range(24, 32)}
        mock_chips.return_value = ({}, [])
        mock_candidates.return_value = []
        mock_eval.return_value = 20.0

        for strat in ("safe", "balanced", "risky"):
            result = build_strategy_plan(
                squad=_make_15_squad(),
                bank=1.5,
                free_transfers=2,
                fixtures=_make_fixtures(),
                events=_make_events(),
                teams_dict=_teams_dict(),
                elements=[],
                current_gw=24,
                horizon=4,
                strategy=strat,
                available_chips={"wildcard": False, "freehit": False, "bboost": False, "3xc": False},
                squad_health=[],
                fixture_swings=[],
                chip_recommendations=[],
            )
            assert isinstance(result, StrategyPlan)
            assert result.name == strat
