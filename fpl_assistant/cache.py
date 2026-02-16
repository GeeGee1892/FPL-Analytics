import json
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple, List

logger = logging.getLogger("fpl_assistant")

PLAYER_HISTORY_CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "player_history_cache.json")


class DataCache:
    def __init__(self):
        self.bootstrap_data: Optional[Dict] = None
        self.fixtures_data: Optional[List] = None
        self.element_summary_cache: Dict[int, Dict] = {}
        self.last_update: Optional[datetime] = None
        self.fixtures_last_update: Optional[datetime] = None
        self.cache_duration = 300
        # Minutes overrides scoped by (manager_id, player_id) - None manager_id for global
        self.minutes_overrides: Dict[Tuple[Optional[int], int], float] = {}
        # FDR cache
        self.fdr_data: Dict[int, Dict] = {}  # team_id -> {attack_fdr, defence_fdr, form_xg, form_xga, ...}
        self.fdr_last_update: Optional[datetime] = None
        # Predicted minutes from external sources (e.g., FPL Review)
        self.predicted_minutes: Dict[int, float] = {}  # player_id -> predicted_minutes
        self.predicted_minutes_last_update: Optional[datetime] = None
        # Manual team strength data (from Analytic FPL)
        self.manual_team_strength: Dict[str, Dict] = {}  # team_name -> {adjxg_for, adjxg_ag}
        self.manual_team_strength_last_update: Optional[datetime] = None
        # Player history cache (reduces API calls significantly)
        self.player_histories: Dict[int, Dict] = {}  # player_id -> history data
        self.player_histories_last_update: Dict[int, datetime] = {}  # player_id -> last update time
        self.player_history_cache_duration = 1800  # 30 minutes
        # Fixture-specific xG projections (from Analytic FPL or similar)
        # Key: (home_team_id, away_team_id, gw) -> {home_xg, away_xg}
        self.fixture_xg: Dict[Tuple[int, int, int], Dict] = {}
        self.fixture_xg_last_update: Optional[datetime] = None

    def get_minutes_override(self, player_id: int, manager_id: Optional[int] = None) -> Optional[float]:
        """Get minutes override, checking manager-specific first, then global, then predicted."""
        # Check manager-specific override first
        if manager_id is not None and (manager_id, player_id) in self.minutes_overrides:
            return self.minutes_overrides[(manager_id, player_id)]
        # Fall back to global override
        if (None, player_id) in self.minutes_overrides:
            return self.minutes_overrides[(None, player_id)]
        # Fall back to predicted minutes (from FPL Review etc.)
        if player_id in self.predicted_minutes:
            return self.predicted_minutes[player_id]
        return None

    def set_minutes_override(self, player_id: int, minutes: float, manager_id: Optional[int] = None):
        """Set minutes override, scoped by manager_id."""
        self.minutes_overrides[(manager_id, player_id)] = minutes

    def delete_minutes_override(self, player_id: int, manager_id: Optional[int] = None) -> bool:
        """Delete minutes override. Returns True if deleted."""
        key = (manager_id, player_id)
        if key in self.minutes_overrides:
            del self.minutes_overrides[key]
            return True
        return False

    def predicted_minutes_is_stale(self) -> bool:
        return self.predicted_minutes_last_update is None or (datetime.now() - self.predicted_minutes_last_update).total_seconds() > 3600  # 1 hour

    def is_stale(self) -> bool:
        return self.last_update is None or (datetime.now() - self.last_update).total_seconds() > self.cache_duration

    def fixtures_is_stale(self) -> bool:
        return self.fixtures_last_update is None or (datetime.now() - self.fixtures_last_update).total_seconds() > self.cache_duration

    def fdr_is_stale(self) -> bool:
        return self.fdr_last_update is None or (datetime.now() - self.fdr_last_update).total_seconds() > 21600  # 6 hours

    def get_player_history(self, player_id: int) -> Optional[Dict]:
        """Get cached player history if not stale."""
        if player_id not in self.player_histories:
            return None
        last_update = self.player_histories_last_update.get(player_id)
        if last_update is None:
            return None
        if (datetime.now() - last_update).total_seconds() > self.player_history_cache_duration:
            return None  # Stale
        return self.player_histories[player_id]

    def set_player_history(self, player_id: int, history: Dict):
        """Cache player history."""
        self.player_histories[player_id] = history
        self.player_histories_last_update[player_id] = datetime.now()

    def clear_stale_player_histories(self):
        """Remove stale player histories to prevent memory bloat."""
        now = datetime.now()
        stale_ids = [
            pid for pid, last_update in self.player_histories_last_update.items()
            if (now - last_update).total_seconds() > self.player_history_cache_duration * 2
        ]
        for pid in stale_ids:
            self.player_histories.pop(pid, None)
            self.player_histories_last_update.pop(pid, None)

    # v5.3: Disk persistence for player history cache
    def save_player_histories_to_disk(self):
        """Persist player history cache to disk for faster startup."""
        try:
            os.makedirs(os.path.dirname(PLAYER_HISTORY_CACHE_FILE), exist_ok=True)
            data = {
                "histories": {str(k): v for k, v in self.player_histories.items()},
                "timestamps": {str(k): v.isoformat() for k, v in self.player_histories_last_update.items()},
                "saved_at": datetime.now().isoformat(),
            }
            with open(PLAYER_HISTORY_CACHE_FILE, "w") as f:
                json.dump(data, f)
            logger.info(f"Saved {len(self.player_histories)} player histories to disk")
        except Exception as e:
            logger.warning(f"Failed to save player history cache: {e}")

    def load_player_histories_from_disk(self):
        """Load player history cache from disk if available and fresh."""
        try:
            if not os.path.exists(PLAYER_HISTORY_CACHE_FILE):
                return False
            with open(PLAYER_HISTORY_CACHE_FILE, "r") as f:
                data = json.load(f)

            saved_at = datetime.fromisoformat(data.get("saved_at", "2000-01-01"))
            # Only use if saved within last 6 hours
            if (datetime.now() - saved_at).total_seconds() > 21600:
                logger.info("Disk cache too old, ignoring")
                return False

            histories = data.get("histories", {})
            timestamps = data.get("timestamps", {})

            for pid_str, hist in histories.items():
                pid = int(pid_str)
                self.player_histories[pid] = hist
                ts_str = timestamps.get(pid_str)
                if ts_str:
                    self.player_histories_last_update[pid] = datetime.fromisoformat(ts_str)

            logger.info(f"Loaded {len(histories)} player histories from disk cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to load player history cache from disk: {e}")
            return False


cache = DataCache()
