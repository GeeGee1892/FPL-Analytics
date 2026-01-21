# FPL Analytics Pro Dashboard

A comprehensive Fantasy Premier League analytics application with player rankings, fixture analysis, differentials finder, price predictions, and team optimization.

## Features

### 1. Player Rankings by Position (GKP/DEF/MID/FWD)
- Custom gameweek range selector (max 8 GWs)
- **Expected points calculation that HEAVILY weights expected minutes**
- Players with low mins/GW are penalized significantly
- Shows: xPts, Form, Total Points, Mins/GW, Ownership %, xG, xA
- Color-coded fixture difficulty badges with GW numbers

### 2. Fixture Difficulty Ratings
- Team-by-team fixture analysis
- Average difficulty scores
- Easy/Medium/Hard ratings
- Visual fixture calendar

### 3. Differential Finder
- Low ownership + high form players
- Adjustable ownership threshold
- Differential score calculation

### 4. Price Change Predictions
- Rising/falling based on transfer activity
- Likelihood percentage bars

### 5. Team Optimizer
- Multiple formations (3-4-3, 4-4-2, etc.)
- Budget constraint (default £100m)
- Max 3 players per team
- Uses PuLP linear programming for optimization

## Critical Algorithm: Minutes Factor

The key differentiator of this app is the **heavy emphasis on expected minutes**. Players who don't play regularly rank LOW, regardless of their underlying stats.

```python
Minutes Factor Calculation:
- < 200 total mins: 0.1 (basically excluded)
- < 30 mins/GW: 0.15
- < 45 mins/GW: 0.3
- < 60 mins/GW: 0.5
- < 75 mins/GW: 0.7
- < 85 mins/GW: 0.85
- >= 85 mins/GW: 1.0 (full credit)
```

## Installation & Running

### Prerequisites
- Python 3.9+
- pip

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Open browser at http://localhost:8000
```

### Using the Run Script
```bash
chmod +x run.sh
./run.sh
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/rankings/{position}` | GET | Get player rankings by position |
| `/api/fixture-ratings` | GET | Get fixture difficulty ratings |
| `/api/differentials` | GET | Find differential players |
| `/api/price-changes` | GET | Get price change predictions |
| `/api/optimize-team` | POST | Optimize team selection |
| `/api/transfer-recommendations` | GET | Get transfer recommendations |

### Example API Calls

```bash
# Get MID rankings for GW 21-28
curl "http://localhost:8000/api/rankings/MID?gw_start=21&gw_end=28&min_minutes=400"

# Get fixture ratings for next 8 GWs
curl "http://localhost:8000/api/fixture-ratings?horizon=8"

# Find differentials under 10% ownership
curl "http://localhost:8000/api/differentials?max_ownership=10&min_form=4"

# Optimize team
curl -X POST "http://localhost:8000/api/optimize-team" \
  -H "Content-Type: application/json" \
  -d '{"budget": 100, "formation": "3-4-3", "gw_start": 21, "gw_end": 28}'
```

## Tech Stack

- **Backend**: Python FastAPI
- **Frontend**: HTML/CSS/JS (single file, dark theme)
- **Data Source**: Official FPL API (fantasy.premierleague.com/api)
- **Optimization**: PuLP (linear programming)

## Data Sources

The app pulls data from multiple sources for comprehensive analytics:

### 1. FPL API (Primary)
- **Bootstrap Static**: `/api/bootstrap-static/` - Players, teams, events, base xG/xA
- **Fixtures**: `/api/fixtures/` - All fixtures with FDR
- **Element Summary**: `/api/element-summary/{id}/` - Individual player history

### 2. Understat (Enhanced xG)
- Non-penalty xG (npxG) - more predictive than raw xG
- xGChain - involvement in goal-scoring chains
- xGBuildup - involvement in buildup play
- Falls back gracefully if unavailable

### 3. FBRef (Fallback)
- Used if Understat data unavailable
- npxG, xG, xA stats from StatsBomb/Opta

**Note**: Understat and FBRef use web scraping which may occasionally fail due to rate limiting or site changes. The app gracefully falls back to FPL API data (which uses Opta xG) if external sources are unavailable.

## File Structure

```
fpl_app/
├── backend/
│   └── main.py          # FastAPI backend
├── frontend/
│   └── index.html       # Single-file frontend
├── requirements.txt     # Python dependencies
├── run.sh              # Run script
└── README.md           # This file
```

## Default Filters

- **Minimum Minutes**: 400 (filters out bench warmers)
- **Max GW Range**: 8 gameweeks
- **Results Limit**: 50 players per position

## License

MIT License - Free to use and modify.
