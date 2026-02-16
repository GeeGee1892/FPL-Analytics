"""
FPL Assistant Backend — entry point and backward-compat re-exports.

All code has been split into fpl_assistant/ modules:
- config.py:      MODEL_CONFIG + 8 dataclass configs
- constants.py:   All constants, team mappings, utility functions
- models.py:      Pydantic models, enums, dataclasses
- cache.py:       DataCache singleton
- calculators.py: FDR mapping, matchup calcs, calculator classes
- services.py:    HTTP client, API fetchers, FDR, form, xPts
- planner.py:     Transfer planner, squad evaluation, strategy
- endpoints.py:   FastAPI app + all 35 API endpoints

Tests import from `main` — star-imports re-export everything for backward compat.
"""

# Re-export everything so `from main import X` still works
from fpl_assistant.config import *       # noqa: F401,F403
from fpl_assistant.constants import *    # noqa: F401,F403
from fpl_assistant.constants import _generate_fixture_weights  # noqa: F401  # tests use this
from fpl_assistant.models import *       # noqa: F401,F403
from fpl_assistant.cache import *        # noqa: F401,F403
from fpl_assistant.calculators import *  # noqa: F401,F403
from fpl_assistant.services import *     # noqa: F401,F403
from fpl_assistant.planner import *      # noqa: F401,F403
from fpl_assistant.endpoints import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
