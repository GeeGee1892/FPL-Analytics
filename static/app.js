const API_BASE = '';
let currentGw = 1;
let currentPosition = 'GKP';
let savedTeamId = localStorage.getItem('fpl_team_id');
let myTeamData = null;
let rankingsData = [];
let currentSort = { field: 'xpts', order: 'desc' };
let fixturesLoaded = false;
let rankingsLoaded = false;
let bootstrapLoaded = false;

document.addEventListener('DOMContentLoaded', async () => {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', async () => {
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.section).classList.add('active');
            
            // Load fixtures on first visit to fixtures tab
            if (tab.dataset.section === 'fixtures' && !fixturesLoaded) {
                loadFixtureRatings();
                fixturesLoaded = true;
            }
            
            // Load rankings on first visit to rankings tab (after bootstrap loaded)
            if (tab.dataset.section === 'rankings' && !rankingsLoaded && bootstrapLoaded) {
                loadRankings();
                rankingsLoaded = true;
            }
            
            // Pre-fill planner team ID from saved or current loaded team
            if (tab.dataset.section === 'planner') {
                const plannerInput = document.getElementById('plannerTeamIdInput');
                if (!plannerInput.value && savedTeamId) {
                    plannerInput.value = savedTeamId;
                }
            }
        });
    });

    document.querySelectorAll('.position-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.position-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentPosition = tab.dataset.position;
            updatePriceOptionsForPosition(currentPosition);
            loadRankings();
        });
    });

    document.querySelectorAll('.rec-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.rec-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.rec-panel').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.panel + 'Panel').classList.add('active');
        });
    });

    // Sortable headers
    document.querySelectorAll('#rankingsTable th[data-sort]').forEach(th => {
        th.addEventListener('click', () => sortTable(th.dataset.sort));
    });

    await loadBootstrap();
    checkAdminAccess(); // Check admin access on load
    if (savedTeamId) {
        document.getElementById('teamIdInput').value = savedTeamId;
        loadMyTeam();
    }
});

function sortTable(field) {
    if (currentSort.field === field) {
        currentSort.order = currentSort.order === 'desc' ? 'asc' : 'desc';
    } else {
        currentSort.field = field;
        currentSort.order = 'desc';
    }

    const tableEl = document.getElementById('rankingsTable');
    tableEl.querySelectorAll('th').forEach(th => {
        th.classList.remove('sorted');
        if (th.dataset.sort === field) {
            th.classList.add('sorted');
            th.querySelector('.sort-icon').textContent = currentSort.order === 'desc' ? '▼' : '▲';
        }
    });

    loadRankings();
}

async function loadBootstrap() {
    try {
        const response = await fetch(`${API_BASE}/api/bootstrap`);
        const data = await response.json();
        const currentEvent = data.events.find(e => e.is_current);
        const nextEvent = data.events.find(e => e.is_next);
        
        if (currentEvent) {
            currentGw = currentEvent.id;
            document.getElementById('currentGwDisplay').textContent = `GW ${currentGw}`;
        }
        
        // For rankings, use NEXT GW since current deadline has passed
        const rankingsGw = nextEvent ? nextEvent.id : (currentGw + 1);
        document.getElementById('gwStart').value = Math.min(rankingsGw, 38);
        document.getElementById('gwEnd').value = Math.min(rankingsGw + 5, 38);
        
        // Populate team filter dropdown
        populateTeamFilter(data.teams);

        // Populate planner GW dropdowns dynamically
        populatePlannerGwSelects(rankingsGw);

        // Store players for search
        allPlayers = data.elements || [];

        bootstrapLoaded = true;
    } catch (error) {
        console.error('Failed to load bootstrap:', error);
        // Still populate team filter with fallback
        populateTeamFilter(null);
        // Still mark as loaded so rankings can proceed with defaults
        bootstrapLoaded = true;
    }
}

// Store all players for search functionality
let allPlayers = [];

// Fallback team list for when API is unavailable
const FALLBACK_TEAMS = [
    { id: 1, name: 'Arsenal', short_name: 'ARS' },
    { id: 2, name: 'Aston Villa', short_name: 'AVL' },
    { id: 3, name: 'Bournemouth', short_name: 'BOU' },
    { id: 4, name: 'Brentford', short_name: 'BRE' },
    { id: 5, name: 'Brighton', short_name: 'BHA' },
    { id: 6, name: 'Burnley', short_name: 'BUR' },
    { id: 7, name: 'Chelsea', short_name: 'CHE' },
    { id: 8, name: 'Crystal Palace', short_name: 'CRY' },
    { id: 9, name: 'Everton', short_name: 'EVE' },
    { id: 10, name: 'Fulham', short_name: 'FUL' },
    { id: 11, name: 'Leeds', short_name: 'LEE' },
    { id: 12, name: 'Liverpool', short_name: 'LIV' },
    { id: 13, name: 'Man City', short_name: 'MCI' },
    { id: 14, name: 'Man Utd', short_name: 'MUN' },
    { id: 15, name: 'Newcastle', short_name: 'NEW' },
    { id: 16, name: 'Nott\'m Forest', short_name: 'NFO' },
    { id: 17, name: 'Sunderland', short_name: 'SUN' },
    { id: 18, name: 'Spurs', short_name: 'TOT' },
    { id: 19, name: 'West Ham', short_name: 'WHU' },
    { id: 20, name: 'Wolves', short_name: 'WOL' },
];

// Populate team filter dropdown
function populateTeamFilter(teams) {
    const teamSelect = document.getElementById('teamFilter');
    if (!teamSelect) return;
    
    // Use API teams or fallback
    let teamList = [];
    if (teams && typeof teams === 'object') {
        teamList = Object.values(teams);
    }
    if (teamList.length === 0) {
        teamList = FALLBACK_TEAMS;
    }
    
    // Sort teams alphabetically by name
    const sortedTeams = teamList.sort((a, b) => 
        (a.name || a.short_name || '').localeCompare(b.name || b.short_name || '')
    );
    
    // Clear existing options except first "All Teams"
    teamSelect.innerHTML = '<option value="">All Teams</option>';
    
    sortedTeams.forEach(team => {
        const option = document.createElement('option');
        option.value = team.short_name || team.name;
        option.textContent = team.name || team.short_name;
        teamSelect.appendChild(option);
    });
}

function populatePlannerGwSelects(startGw) {
    const start = Math.max(1, startGw);
    const selectIds = [
        'preBookedGw',
        'chipGw_wildcard',
        'chipGw_freehit',
        'chipGw_bboost',
        'chipGw_3xc',
    ];
    for (const id of selectIds) {
        const sel = document.getElementById(id);
        if (!sel) continue;
        sel.innerHTML = '';
        for (let gw = start; gw <= 38; gw++) {
            const opt = document.createElement('option');
            opt.value = gw;
            opt.textContent = `GW${gw}`;
            sel.appendChild(opt);
        }
    }
}

// Player search functionality
function initPlayerSearch() {
    const searchInput = document.getElementById('playerSearch');
    const searchContainer = document.getElementById('playerSearchContainer');
    const searchDropdown = document.getElementById('searchDropdown');
    const searchClear = document.getElementById('searchClear');
    
    let highlightedIndex = -1;
    
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        
        // Update clear button visibility
        if (query.length > 0) {
            searchContainer.classList.add('has-value');
        } else {
            searchContainer.classList.remove('has-value');
        }
        
        // Require at least 3 characters
        if (query.length < 3) {
            searchDropdown.classList.remove('active');
            return;
        }
        
        // Filter players
        const filtered = allPlayers
            .filter(p => p.web_name.toLowerCase().includes(query.toLowerCase()))
            .slice(0, 10);  // Limit to 10 results
        
        if (filtered.length === 0) {
            searchDropdown.innerHTML = '<div class="search-no-results">No players found</div>';
        } else {
            searchDropdown.innerHTML = filtered.map((p, i) => `
                <div class="search-item ${i === highlightedIndex ? 'highlighted' : ''}" 
                     data-player-id="${p.id}" 
                     data-player-name="${p.web_name}"
                     data-position="${p.element_type}">
                    <div>
                        <span class="search-item-name">${p.web_name}</span>
                        <span class="search-item-team">${getTeamShortName(p.team)}</span>
                    </div>
                    <span class="search-item-price">£${(p.now_cost / 10).toFixed(1)}</span>
                </div>
            `).join('');
            
            // Add click handlers
            searchDropdown.querySelectorAll('.search-item').forEach(item => {
                item.addEventListener('click', () => selectPlayer(item));
            });
        }
        
        searchDropdown.classList.add('active');
        highlightedIndex = -1;
    });
    
    // Keyboard navigation
    searchInput.addEventListener('keydown', (e) => {
        const items = searchDropdown.querySelectorAll('.search-item');
        
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            highlightedIndex = Math.min(highlightedIndex + 1, items.length - 1);
            updateHighlight(items);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            highlightedIndex = Math.max(highlightedIndex - 1, 0);
            updateHighlight(items);
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (highlightedIndex >= 0 && items[highlightedIndex]) {
                selectPlayer(items[highlightedIndex]);
            } else {
                // Just filter the table with current search
                loadRankings();
                searchDropdown.classList.remove('active');
            }
        } else if (e.key === 'Escape') {
            searchDropdown.classList.remove('active');
        }
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchContainer.contains(e.target)) {
            searchDropdown.classList.remove('active');
        }
    });
    
    function updateHighlight(items) {
        items.forEach((item, i) => {
            item.classList.toggle('highlighted', i === highlightedIndex);
        });
    }
    
    function selectPlayer(item) {
        const name = item.dataset.playerName;
        const position = item.dataset.position;
        
        // Set the search input value
        searchInput.value = name;
        searchContainer.classList.add('has-value');
        
        // Set position tab based on player
        const posMap = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'};
        const posName = posMap[position] || currentPosition;
        
        // Update position tab
        document.querySelectorAll('.position-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.position === posName);
        });
        currentPosition = posName;
        
        // Update price options for new position
        updatePriceOptionsForPosition(posName);
        
        // Close dropdown and load rankings
        searchDropdown.classList.remove('active');
        loadRankings();
    }
}

// Get team short name by ID
function getTeamShortName(teamId) {
    const teamMap = {
        1: 'ARS', 2: 'AVL', 3: 'BOU', 4: 'BRE', 5: 'BHA', 6: 'BUR', 7: 'CHE', 8: 'CRY',
        9: 'EVE', 10: 'FUL', 11: 'LEE', 12: 'LIV', 13: 'MCI', 14: 'MUN', 15: 'NEW',
        16: 'NFO', 17: 'SUN', 18: 'TOT', 19: 'WHU', 20: 'WOL'
    };
    return teamMap[teamId] || '???';
}

// Update price dropdown options based on position
function updatePriceOptionsForPosition(position) {
    const priceSelect = document.getElementById('maxPriceFilter');
    const currentValue = priceSelect.value;
    
    // Position-specific price brackets (descending order)
    const priceOptions = {
        'GKP': [
            { value: '7.0', text: 'All Prices' },
            { value: '6.0', text: 'Under £6.0m' },
            { value: '5.5', text: 'Under £5.5m' },
            { value: '5.0', text: 'Under £5.0m' },
            { value: '4.5', text: 'Under £4.5m' },
        ],
        'DEF': [
            { value: '8.0', text: 'All Prices' },
            { value: '7.0', text: 'Under £7.0m' },
            { value: '6.5', text: 'Under £6.5m' },
            { value: '6.0', text: 'Under £6.0m' },
            { value: '5.5', text: 'Under £5.5m' },
            { value: '5.0', text: 'Under £5.0m' },
            { value: '4.5', text: 'Under £4.5m' },
        ],
        'MID': [
            { value: '15.0', text: 'All Prices' },
            { value: '10.0', text: 'Under £10.0m' },
            { value: '8.0', text: 'Under £8.0m' },
            { value: '7.5', text: 'Under £7.5m' },
            { value: '7.0', text: 'Under £7.0m' },
            { value: '6.5', text: 'Under £6.5m' },
            { value: '6.0', text: 'Under £6.0m' },
            { value: '5.5', text: 'Under £5.5m' },
            { value: '5.0', text: 'Under £5.0m' },
        ],
        'FWD': [
            { value: '16.0', text: 'All Prices' },
            { value: '10.0', text: 'Under £10.0m' },
            { value: '9.0', text: 'Under £9.0m' },
            { value: '8.0', text: 'Under £8.0m' },
            { value: '7.5', text: 'Under £7.5m' },
            { value: '7.0', text: 'Under £7.0m' },
            { value: '6.5', text: 'Under £6.5m' },
            { value: '6.0', text: 'Under £6.0m' },
            { value: '5.5', text: 'Under £5.5m' },
        ]
    };
    
    const options = priceOptions[position] || priceOptions['MID'];
    
    // Clear and repopulate
    priceSelect.innerHTML = '';
    options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt.value;
        option.textContent = opt.text;
        priceSelect.appendChild(option);
    });
    
    // Try to keep current value, otherwise default to "All Prices"
    const hasCurrentValue = options.some(opt => opt.value === currentValue);
    priceSelect.value = hasCurrentValue ? currentValue : options[0].value;
}

// Clear player search
function clearPlayerSearch() {
    const searchInput = document.getElementById('playerSearch');
    const searchContainer = document.getElementById('playerSearchContainer');
    const searchDropdown = document.getElementById('searchDropdown');
    
    searchInput.value = '';
    searchContainer.classList.remove('has-value');
    searchDropdown.classList.remove('active');
    loadRankings();
}

// Initialize search on page load
document.addEventListener('DOMContentLoaded', () => {
    initPlayerSearch();
    updatePriceOptionsForPosition(currentPosition);
    // Populate team filter with fallback (will be overwritten by API if available)
    populateTeamFilter(null);
});

async function loadMyTeam() {
    const teamId = document.getElementById('teamIdInput').value;
    if (!teamId) return alert('Please enter your FPL Team ID');
    localStorage.setItem('fpl_team_id', teamId);
    savedTeamId = teamId;
    
    // Show content area with loading state
    document.getElementById('myTeamContent').style.display = 'block';
    document.getElementById('teamBanner').style.opacity = '0.5';

    try {
        const teamResponse = await fetch(`${API_BASE}/api/my-team/${teamId}`);
        if (!teamResponse.ok) throw new Error('Team not found');
        myTeamData = await teamResponse.json();
        
        // Hide full input
        document.getElementById('teamInputSection').style.display = 'none';
        document.getElementById('teamBanner').style.opacity = '1';
        
        // Update compact banner
        const m = myTeamData.manager;
        document.getElementById('bannerTeamName').textContent = m.team_name || 'My Team';
        document.getElementById('bannerManagerName').textContent = m.name || 'Manager';
        document.getElementById('bannerTotalPts').textContent = m.total_points?.toLocaleString() || '-';
        document.getElementById('bannerOR').textContent = formatRank(m.overall_rank);
        
        renderOptimizedSquad(myTeamData);
        
        // Check admin access for admin panel
        checkAdminAccess();
        
        // Load stats in background
        loadManagerStats(teamId);
    } catch (error) {
        document.getElementById('teamBanner').innerHTML = `<div class="empty-state"><div class="empty-state-icon">❌</div><p>${error.message}</p></div>`;
    }
}

function showTeamInput() {
    document.getElementById('teamInputSection').style.display = 'block';
    document.getElementById('teamIdInput').focus();
}

function formatRank(rank) {
    if (!rank) return '-';
    if (rank >= 1000000) return (rank / 1000000).toFixed(1) + 'M';
    if (rank >= 1000) return (rank / 1000).toFixed(0) + 'K';
    return rank.toLocaleString();
}

// Check prediction status and update UI
async function checkPredictionStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/predicted-minutes/status`);
        const status = await response.json();
        
        const statusEl = document.getElementById('predictionStatus');
        if (status.cached_count > 0) {
            const isStale = status.is_stale;
            statusEl.textContent = `${status.cached_count} predictions${isStale ? ' (stale)' : ''}`;
            statusEl.className = `prediction-status ${isStale ? 'stale' : 'synced'}`;
        } else {
            statusEl.textContent = 'No predictions';
            statusEl.className = 'prediction-status';
        }
    } catch (e) {
        console.error('Failed to check prediction status:', e);
    }
}

// Sync predicted lineups from FPL Review
async function syncPredictedLineups() {
    const statusEl = document.getElementById('predictionStatus');
    statusEl.textContent = 'Syncing...';
    statusEl.className = 'prediction-status';
    
    try {
        // Try FPL Review first
        let response = await fetch(`${API_BASE}/api/predicted-minutes/fetch?source=fplreview`, {
            method: 'POST'
        });
        let result = await response.json();
        
        // If FPL Review failed, try Rotowire
        if (result.imported === 0) {
            response = await fetch(`${API_BASE}/api/predicted-minutes/fetch?source=rotowire`, {
                method: 'POST'
            });
            result = await response.json();
        }
        
        if (result.imported > 0) {
            statusEl.textContent = `${result.imported} synced`;
            statusEl.className = 'prediction-status synced';
            
            // Reload team to reflect new predictions
            if (savedTeamId) {
                loadMyTeam();
            }
            loadRankings();
        } else {
            statusEl.textContent = 'Sync failed';
            statusEl.className = 'prediction-status stale';
            
            // Show manual import option
            if (confirm('Auto-sync failed. Would you like to manually import predictions?\n\nYou can paste predicted minutes as JSON:\n{player_id: minutes, ...}')) {
                const json = prompt('Paste JSON predictions:');
                if (json) {
                    try {
                        const predictions = JSON.parse(json);
                        const importResponse = await fetch(`${API_BASE}/api/predicted-minutes/bulk`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ predictions, source: 'manual' })
                        });
                        const importResult = await importResponse.json();
                        statusEl.textContent = `${importResult.imported} imported`;
                        statusEl.className = 'prediction-status synced';
                        
                        if (savedTeamId) loadMyTeam();
                        loadRankings();
                    } catch (e) {
                        alert('Invalid JSON format');
                    }
                }
            }
        }
    } catch (e) {
        console.error('Failed to sync predictions:', e);
        statusEl.textContent = 'Error';
        statusEl.className = 'prediction-status stale';
    }
}

/**
 * Select optimal starting XI from 15-player squad
 * Respects FPL formation constraints:
 * - Exactly 1 GKP
 * - 3-5 DEF
 * - 2-5 MID
 * - 1-3 FWD
 * - Total = 11 players
 */
function selectOptimalXI(squad) {
    // Separate by position and sort by xPts descending
    const byPos = {
        GKP: squad.filter(p => p.position === 'GKP').sort((a, b) => b.xpts_single_gw - a.xpts_single_gw),
        DEF: squad.filter(p => p.position === 'DEF').sort((a, b) => b.xpts_single_gw - a.xpts_single_gw),
        MID: squad.filter(p => p.position === 'MID').sort((a, b) => b.xpts_single_gw - a.xpts_single_gw),
        FWD: squad.filter(p => p.position === 'FWD').sort((a, b) => b.xpts_single_gw - a.xpts_single_gw),
    };
    
    // Formation constraints
    const minMax = {
        GKP: { min: 1, max: 1 },
        DEF: { min: 3, max: 5 },
        MID: { min: 2, max: 5 },
        FWD: { min: 1, max: 3 },
    };
    
    // Start with minimum requirements
    const starting = [];
    const selected = { GKP: [], DEF: [], MID: [], FWD: [] };
    
    for (const pos of ['GKP', 'DEF', 'MID', 'FWD']) {
        const needed = minMax[pos].min;
        selected[pos] = byPos[pos].slice(0, needed);
        starting.push(...selected[pos]);
    }
    
    // Fill remaining slots (11 - 7 = 4 more players)
    let slotsRemaining = 11 - starting.length;
    
    // Create pool of remaining players with their positions
    const remainingPool = [];
    for (const pos of ['DEF', 'MID', 'FWD']) {
        const alreadySelected = selected[pos].length;
        const maxMore = minMax[pos].max - alreadySelected;
        const remaining = byPos[pos].slice(alreadySelected, alreadySelected + maxMore);
        remaining.forEach(p => remainingPool.push({ player: p, pos }));
    }
    
    // Sort remaining pool by xPts and fill slots
    remainingPool.sort((a, b) => b.player.xpts_single_gw - a.player.xpts_single_gw);
    
    const posCount = { DEF: selected.DEF.length, MID: selected.MID.length, FWD: selected.FWD.length };
    
    for (const item of remainingPool) {
        if (slotsRemaining === 0) break;
        if (posCount[item.pos] < minMax[item.pos].max) {
            starting.push(item.player);
            posCount[item.pos]++;
            slotsRemaining--;
        }
    }
    
    // Create bench from remaining players
    const startingIds = new Set(starting.map(p => p.id));
    const bench = squad.filter(p => !startingIds.has(p.id));
    
    return { starting, bench };
}

function renderOptimizedSquad(data) {
    const squad = data.squad;
    const nextGw = data.next_gw;
    
    // Select optimal starting XI based on xPts while respecting formation constraints
    const { starting, bench } = selectOptimalXI(squad);
    
    // Sort all players by xPts for captain selection (from starting XI only)
    const sortedByXpts = [...starting].sort((a, b) => b.xpts_single_gw - a.xpts_single_gw);
    
    // Determine optimal captain (highest xPts) and vice (second highest)
    const captain = sortedByXpts[0];
    const viceCaptain = sortedByXpts[1];
    
    // Sort bench: GK first (required), then outfield by xPts
    const benchGK = bench.filter(p => p.position === 'GKP');
    const benchOutfield = bench.filter(p => p.position !== 'GKP').sort((a, b) => b.xpts_single_gw - a.xpts_single_gw);
    const optimalBench = [...benchGK, ...benchOutfield];
    
    // Group starting XI by position
    const gkp = starting.filter(p => p.position === 'GKP');
    const def = starting.filter(p => p.position === 'DEF');
    const mid = starting.filter(p => p.position === 'MID');
    const fwd = starting.filter(p => p.position === 'FWD');

    // Render player on pitch with optimal C/VC badges
    // Click to override expected minutes
    const renderPlayer = p => {
        const isCaptain = p.id === captain.id;
        const isVice = p.id === viceCaptain.id;
        const expMins = p.expected_minutes || 0;
        const expMinsWarning = expMins < 60 ? 'low-mins' : '';
        const minsReason = p.minutes_reason || 'season_average';
        const isPredicted = minsReason === 'predicted';
        const isOverride = minsReason === 'user_override';
        const minsClass = isOverride ? 'override' : (isPredicted ? 'predicted' : '');
        const minsLabel = isOverride ? '✓' : (isPredicted ? '⚡' : '');
        
        // Get next fixture info
        const nextFix = p.fixtures && p.fixtures.length > 0 ? p.fixtures[0] : null;
        const oppDisplay = nextFix ? `${nextFix.is_home ? '' : '@'}${nextFix.opponent}` : '';
        const fdrClass = nextFix ? `fdr-${nextFix.fdr || nextFix.difficulty || 5}` : '';
        
        return `
            <div class="pitch-player ${isCaptain ? 'captain' : ''} ${isVice ? 'vice' : ''} ${expMinsWarning}"
                 onclick="promptMinutesOverride(${p.id}, '${p.name}', ${expMins})"
                 title="${minsReason.replace('_', ' ')}: ${expMins.toFixed(0)} mins (click to override)">
                <div class="name">${p.name}</div>
                <div class="team">${p.team}</div>
                ${nextFix ? `<div class="fixture ${fdrClass}">${oppDisplay}</div>` : ''}
                <div class="xpts">${p.xpts_single_gw.toFixed(1)}</div>
                <div class="exp-mins-tag ${minsClass}">${expMins.toFixed(0)}m${minsLabel}</div>
                ${isCaptain ? '<div class="badge">C</div>' : ''}
                ${isVice ? '<div class="badge vc">V</div>' : ''}
            </div>
        `;
    };

    // Render pitch with bench at bottom
    document.getElementById('squadPitch').innerHTML = `
        <div class="pitch-row">${gkp.map(renderPlayer).join('')}</div>
        <div class="pitch-row">${def.map(renderPlayer).join('')}</div>
        <div class="pitch-row">${mid.map(renderPlayer).join('')}</div>
        <div class="pitch-row">${fwd.map(renderPlayer).join('')}</div>
        <div class="bench-row">
            <div class="label">Bench</div>
            <div class="pitch-row">${optimalBench.map(renderPlayer).join('')}</div>
        </div>
        <div class="fdr-legend">
            <div class="fdr-legend-item"><div class="fdr-legend-dot easy"></div> Easy (FDR 1-3)</div>
            <div class="fdr-legend-item"><div class="fdr-legend-dot medium"></div> Medium (FDR 4-5)</div>
            <div class="fdr-legend-item"><div class="fdr-legend-dot hard"></div> Hard (FDR 6-10)</div>
        </div>
    `;
    
    // Update header
    document.getElementById('optimizerGwBadge').textContent = `GW ${nextGw}`;
    
    // Calculate total xPts (starting XI + captain bonus)
    const startingXpts = starting.reduce((sum, p) => sum + p.xpts_single_gw, 0);
    const totalXpts = startingXpts + captain.xpts_single_gw; // Captain counted twice
    document.getElementById('optimizerTotalXpts').textContent = totalXpts.toFixed(1);
    
    // Update team value header
    document.getElementById('squadValueDisplay').textContent = `£${data.total_value.toFixed(1)}m`;
    document.getElementById('bankDisplay').textContent = `£${data.bank.toFixed(1)}m`;
    document.getElementById('ftDisplay').textContent = data.free_transfers;
}

// ============ STATS TAB ============
let orChart = null;
let transferPlChart = null;

async function loadManagerStats(teamId) {
    const loadingEl = document.getElementById('statsLoading');
    const contentEl = document.getElementById('statsContent');

    loadingEl.style.display = 'flex';
    loadingEl.innerHTML = `
        <div class="loading-spinner"></div>
        <span class="stats-loading-text">Loading season stats...</span>
        <span class="stats-loading-sub">Analyzing transfers, bench points, and chip usage</span>
    `;
    contentEl.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/api/my-team/${teamId}/stats`);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Failed to load stats (${response.status})`);
        }
        const stats = await response.json();
        renderManagerStats(stats);
    } catch (error) {
        console.error('Failed to load stats:', error);
        loadingEl.innerHTML = `
            <div style="color: var(--accent-red); font-size: 1.5rem;">❌</div>
            <span style="color: var(--accent-red);">Failed to load stats</span>
            <span class="stats-loading-sub">${error.message}</span>
            <button class="btn btn-secondary" onclick="loadManagerStats('${teamId}')" style="margin-top: 0.5rem;">🔄 Retry</button>
        `;
    }
}

const formatRankChange = (change) => {
    const abs = Math.abs(change);
    if (abs >= 1000000) return (abs/1000000).toFixed(1) + 'M';
    if (abs >= 1000) return (abs/1000).toFixed(0) + 'K';
    return abs.toString();
};

function renderManagerStats(stats) {
    // Best GW
    document.getElementById('bestGwPoints').textContent = `${stats.best_gw?.points || '-'} pts`;
    document.getElementById('bestGwRank').textContent = `GW ${stats.best_gw?.gw || '-'} · Rank ${stats.best_gw?.rank?.toLocaleString() || '-'}`;

    // Best Transfer
    if (stats.best_transfer) {
        const bt = stats.best_transfer;
        document.getElementById('bestTransferPts').textContent = `+${bt.pts_diff} pts`;
        document.getElementById('bestTransferPts').className = `stat-summary-value ${bt.pts_diff >= 0 ? 'positive' : 'negative'}`;
        document.getElementById('bestTransferDetail').textContent = `${bt.out.name} → ${bt.in.name} (GW${bt.gw})`;
    } else {
        document.getElementById('bestTransferPts').textContent = '-';
        document.getElementById('bestTransferDetail').textContent = 'No transfers yet';
    }

    // Worst Transfer
    if (stats.worst_transfer) {
        const wt = stats.worst_transfer;
        document.getElementById('worstTransferPts').textContent = `${wt.pts_diff} pts`;
        document.getElementById('worstTransferPts').className = `stat-summary-value ${wt.pts_diff >= 0 ? 'positive' : 'negative'}`;
        document.getElementById('worstTransferDetail').textContent = `${wt.out.name} → ${wt.in.name} (GW${wt.gw})`;
    } else {
        document.getElementById('worstTransferPts').textContent = '-';
        document.getElementById('worstTransferDetail').textContent = 'No transfers yet';
    }

    // Best Differential
    if (stats.best_differential) {
        const diff = stats.best_differential;
        document.getElementById('bestDiffPts').textContent = `${diff.pts} pts`;
        document.getElementById('bestDiffDetail').textContent = `${diff.player} (${diff.ownership}%) GW${diff.gw}`;
    } else {
        document.getElementById('bestDiffPts').textContent = '-';
        document.getElementById('bestDiffDetail').textContent = 'No differentials (<10%)';
    }

    // Biggest Rank Gain
    if (stats.biggest_green_arrow && stats.biggest_green_arrow.change > 0) {
        const g = stats.biggest_green_arrow;
        const pctGain = ((g.change / g.from) * 100).toFixed(1);
        document.getElementById('biggestGreenValue').textContent = `↑ ${formatRankChange(g.change)}`;
        document.getElementById('biggestGreenDetail').textContent = `GW${g.gw} · ${formatRankChange(g.from)} → ${formatRankChange(g.to)} (${pctGain}%)`;
    } else {
        document.getElementById('biggestGreenValue').textContent = '-';
        document.getElementById('biggestGreenDetail').textContent = 'No rank gains';
    }

    // Biggest Rank Drop
    if (stats.biggest_red_arrow && stats.biggest_red_arrow.change < 0) {
        const r = stats.biggest_red_arrow;
        const pctDrop = ((Math.abs(r.change) / r.from) * 100).toFixed(1);
        document.getElementById('biggestRedValue').textContent = `↓ ${formatRankChange(r.change)}`;
        document.getElementById('biggestRedDetail').textContent = `GW${r.gw} · ${formatRankChange(r.from)} → ${formatRankChange(r.to)} (${pctDrop}%)`;
    } else {
        document.getElementById('biggestRedValue').textContent = '-';
        document.getElementById('biggestRedDetail').textContent = 'No rank drops';
    }

    // Total Hits
    document.getElementById('totalHitsValue').textContent = stats.total_hits || 0;
    document.getElementById('totalHitsCost').textContent = stats.total_hit_cost > 0 ? `-${stats.total_hit_cost} pts` : '0 pts';

    // Average GW Points
    document.getElementById('avgGwPoints').textContent = stats.avg_gw_points || '-';
    document.getElementById('totalPointsSub').textContent = `${stats.total_points || 0} total`;

    // Captain Performance
    renderCaptainPanel(stats.captain_performance);

    // Chip Performance (collapsible)
    renderChipSection(stats);

    // OR Chart
    renderORChart(stats.or_progression);

    // Transfer P/L Chart
    renderTransferPlChart(stats.transfer_pl);

    document.getElementById('statsLoading').style.display = 'none';
    document.getElementById('statsContent').style.display = 'block';
}

// --- Captain Performance ---
function renderCaptainPanel(cap) {
    if (!cap || !cap.total_gws) {
        document.getElementById('captainTotalPts').textContent = '-';
        document.getElementById('captainHitRate').textContent = '-';
        return;
    }

    document.getElementById('captainTotalPts').textContent = cap.total_pts;
    document.getElementById('captainHitRate').textContent = `${cap.hit_rate}%`;

    if (cap.best) {
        document.getElementById('bestCaptainValue').textContent = `${cap.best.base_pts} pts`;
        document.getElementById('bestCaptainDetail').textContent = `${cap.best.player} GW${cap.best.gw}`;
    }
    if (cap.worst) {
        document.getElementById('worstCaptainValue').textContent = `${cap.worst.base_pts} pts`;
        document.getElementById('worstCaptainDetail').textContent = `${cap.worst.player} GW${cap.worst.gw}`;
    }

    // Build expandable picks list
    const listEl = document.getElementById('captainPicksList');
    if (!listEl || !cap.picks) return;

    listEl.innerHTML = cap.picks.map(p => {
        const ptsClass = p.base_pts >= 6 ? 'positive' : p.base_pts <= 2 ? 'negative' : '';
        const tcBadge = p.multiplier === 3 ? ' <span class="chip-badge tc" style="font-size:0.6rem;padding:0.1rem 0.3rem;">TC</span>' : '';
        return `<div class="captain-pick-row">
            <span class="captain-pick-gw">GW${p.gw}</span>
            <span class="captain-pick-name">${p.player}${tcBadge}</span>
            <span class="captain-pick-pts ${ptsClass}">${p.base_pts} pts</span>
        </div>`;
    }).join('');
}

function toggleCaptainPanel() {
    const panel = document.getElementById('captainPicksPanel');
    const arrow = document.getElementById('captainArrow');
    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        arrow.textContent = '▲';
    } else {
        panel.style.display = 'none';
        arrow.textContent = '▼';
    }
}

// --- Chip Performance (collapsible) ---
function renderChipSection(stats) {
    const badgesEl = document.getElementById('chipSummaryBadges');
    const detailsEl = document.getElementById('chipDetailsList');
    if (!badgesEl || !detailsEl) return;

    const chipConfigs = [
        { key: 'wc1', label: 'WC1', analysis: stats.wc1_analysis, color: '#f59e0b', type: 'wc' },
        { key: 'wc2', label: 'WC2', analysis: stats.wc2_analysis, color: '#f59e0b', type: 'wc' },
        { key: 'fh1', label: 'FH1', analysis: stats.fh1_analysis, color: '#8b5cf6', type: 'fh' },
        { key: 'fh2', label: 'FH2', analysis: stats.fh2_analysis, color: '#8b5cf6', type: 'fh' },
        { key: 'tc1', label: 'TC1', analysis: stats.tc1_analysis, color: '#ec4899', type: 'tc' },
        { key: 'tc2', label: 'TC2', analysis: stats.tc2_analysis, color: '#ec4899', type: 'tc' },
        { key: 'bb1', label: 'BB1', analysis: stats.bb1_analysis, color: '#3b82f6', type: 'bb' },
        { key: 'bb2', label: 'BB2', analysis: stats.bb2_analysis, color: '#3b82f6', type: 'bb' },
    ];

    // Also check chips_used for chips without analysis
    const chipsUsed = stats.chips_used || [];
    const chipNameMap = { wildcard: 'wc', freehit: 'fh', '3xc': 'tc', bboost: 'bb' };

    // Build summary badges (inline preview) and detail cards
    let badgesHtml = '';
    let detailsHtml = '';
    let usedCount = 0;

    for (const cfg of chipConfigs) {
        const a = cfg.analysis;
        // Check if chip was used (even without analysis)
        const chipApiName = { wc: 'wildcard', fh: 'freehit', tc: '3xc', bb: 'bboost' }[cfg.type];
        const isSecondHalf = cfg.key.endsWith('2');
        const chipUse = chipsUsed.find(c => c.name === chipApiName && (isSecondHalf ? c.event >= 20 : c.event < 20));

        if (a) {
            usedCount++;
            let valueText = '';
            let valueClass = '';
            let detail = '';

            if (cfg.type === 'wc' || cfg.type === 'fh') {
                const sign = a.net_pts >= 0 ? '+' : '';
                valueText = `${sign}${a.net_pts}`;
                valueClass = a.net_pts >= 0 ? 'positive' : 'negative';
                detail = cfg.type === 'wc'
                    ? `${a.players_in} players in, ${a.players_out} out · ${a.gw_range}`
                    : `FH: ${a.fh_pts} pts vs prev team: ${a.prev_team_pts} pts`;
            } else if (cfg.type === 'tc') {
                valueText = `+${a.extra_pts}`;
                valueClass = 'positive';
                detail = `${a.player} scored ${a.captain_pts} base pts`;
            } else if (cfg.type === 'bb') {
                valueText = `+${a.bench_pts}`;
                valueClass = 'positive';
                detail = `Bench scored ${a.bench_pts} pts`;
            }

            badgesHtml += `<span class="chip-summary-badge" style="border-color: ${cfg.color}; color: ${cfg.color};">${cfg.label} GW${a.gw} <strong class="${valueClass}">${valueText}</strong></span>`;
            detailsHtml += `<div class="chip-detail-card" style="border-left: 3px solid ${cfg.color};">
                <div class="chip-detail-header"><span class="chip-badge" style="background: ${cfg.color};">${cfg.label}</span><span>GW ${a.gw}</span></div>
                <div class="chip-detail-value ${valueClass}">${valueText} pts</div>
                <div class="chip-detail-sub">${detail}</div>
            </div>`;
        } else if (chipUse) {
            usedCount++;
            badgesHtml += `<span class="chip-summary-badge" style="border-color: ${cfg.color}; color: ${cfg.color};">${cfg.label} GW${chipUse.event}</span>`;
            detailsHtml += `<div class="chip-detail-card" style="border-left: 3px solid ${cfg.color};">
                <div class="chip-detail-header"><span class="chip-badge" style="background: ${cfg.color};">${cfg.label}</span><span>GW ${chipUse.event}</span></div>
                <div class="chip-detail-value">Used</div>
            </div>`;
        }
    }

    if (usedCount === 0) {
        badgesHtml = '<span style="color: var(--text-muted); font-size: 0.75rem;">No chips used yet</span>';
        detailsHtml = '<div style="color: var(--text-muted); padding: 1rem; text-align: center;">No chips have been played this season</div>';
    }

    badgesEl.innerHTML = badgesHtml;
    detailsEl.innerHTML = detailsHtml;
}

function toggleChipPanel() {
    const panel = document.getElementById('chipDetailsPanel');
    const arrow = document.getElementById('chipArrow');
    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        arrow.textContent = '▲';
    } else {
        panel.style.display = 'none';
        arrow.textContent = '▼';
    }
}

// --- Transfer P/L Chart ---
function renderTransferPlChart(transferPl) {
    const ctx = document.getElementById('transferPlChart');
    const summaryEl = document.getElementById('transferPlSummary');
    if (!ctx) return;

    if (transferPlChart) transferPlChart.destroy();

    if (!transferPl || transferPl.length === 0) {
        if (summaryEl) summaryEl.innerHTML = '<span style="color: var(--text-muted);">No transfers made yet</span>';
        return;
    }

    // Data is already grouped by GW from backend
    const finalPl = transferPl[transferPl.length - 1].cumulative;
    const totalGws = transferPl.length;
    const totalTransfers = transferPl.reduce((sum, t) => sum + t.details.length, 0);
    const profitableGws = transferPl.filter(t => t.pts_diff > 0).length;

    if (summaryEl) {
        const plClass = finalPl >= 0 ? 'positive' : 'negative';
        const plSign = finalPl >= 0 ? '+' : '';
        summaryEl.innerHTML = `
            <span class="transfer-pl-stat"><strong class="${plClass}">${plSign}${finalPl}</strong> net pts</span>
            <span class="transfer-pl-stat">${totalTransfers} transfers across ${totalGws} GWs</span>
            <span class="transfer-pl-stat">${profitableGws}/${totalGws} GWs profitable</span>
        `;
    }

    const labels = transferPl.map(t => `GW${t.gw}`);
    const cumulativeData = transferPl.map(t => t.cumulative);
    const perGwData = transferPl.map(t => t.pts_diff);
    const barColors = perGwData.map(v => v >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)');

    transferPlChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                type: 'line',
                label: 'Cumulative P/L',
                data: cumulativeData,
                borderColor: '#00d4aa',
                backgroundColor: 'rgba(0, 212, 170, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 4,
                pointBackgroundColor: '#00d4aa',
                yAxisID: 'y1',
                order: 0,
            }, {
                type: 'bar',
                label: 'GW Transfer P/L',
                data: perGwData,
                backgroundColor: barColors,
                borderRadius: 4,
                yAxisID: 'y',
                order: 1,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { color: '#9090a0' } },
                tooltip: {
                    callbacks: {
                        afterBody: function(context) {
                            const idx = context[0].dataIndex;
                            const t = transferPl[idx];
                            return t.details || [];
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#606070', maxRotation: 45 },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    position: 'left',
                    ticks: { color: '#9090a0' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    title: { display: true, text: 'GW P/L', color: '#9090a0' }
                },
                y1: {
                    position: 'right',
                    ticks: { color: '#00d4aa' },
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Cumulative', color: '#00d4aa' }
                }
            }
        }
    });
}

function renderORChart(progression) {
    const ctx = document.getElementById('orChart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (orChart) {
        orChart.destroy();
    }
    
    const labels = progression.map(d => `GW${d.gw}`);
    const orData = progression.map(d => d.or);
    const pointsData = progression.map(d => d.points);
    
    // Create point styles - chips and hits get special markers
    const chipColors = {
        'wildcard': '#f59e0b',
        '3xc': '#ec4899',
        'bboost': '#3b82f6',
        'freehit': '#8b5cf6',
    };
    
    const pointBackgrounds = progression.map(d => {
        if (d.chip) return chipColors[d.chip] || '#00d4aa';
        if (d.hit_cost > 0) return '#ef4444';  // Red for hits
        return '#00d4aa';
    });
    
    const pointRadii = progression.map(d => {
        if (d.chip) return 8;  // Bigger for chips
        if (d.hit_cost > 0) return 6;  // Medium for hits
        return 3;  // Default
    });
    
    const pointStyles = progression.map(d => {
        if (d.chip) return 'star';
        if (d.hit_cost > 0) return 'triangle';
        return 'circle';
    });
    
    // Auto-scale Y-axis based on actual rank range
    const validOr = orData.filter(v => v > 0);
    const minRank = Math.min(...validOr);
    const maxRank = Math.max(...validOr);

    // Build nice tick values that span the actual range
    const allTicks = [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2000000, 4000000, 8000000, 12000000];
    // Pad range by ~20% on each side (in log space)
    const logMin = Math.log10(minRank) - 0.2;
    const logMax = Math.log10(maxRank) + 0.2;
    const scaledMin = Math.max(100, Math.pow(10, logMin));
    const scaledMax = Math.min(12000000, Math.pow(10, logMax));
    const filteredTicks = allTicks.filter(t => t >= scaledMin * 0.8 && t <= scaledMax * 1.2);
    // Ensure we always have the boundary ticks
    if (filteredTicks.length === 0 || filteredTicks[0] > scaledMin) filteredTicks.unshift(Math.round(scaledMin));
    if (filteredTicks[filteredTicks.length - 1] < scaledMax) filteredTicks.push(Math.round(scaledMax));

    orChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Overall Rank',
                data: orData,
                borderColor: '#00d4aa',
                backgroundColor: 'rgba(0, 212, 170, 0.1)',
                fill: true,
                tension: 0.3,
                yAxisID: 'y',
                pointBackgroundColor: pointBackgrounds,
                pointRadius: pointRadii,
                pointStyle: pointStyles,
                pointBorderColor: pointBackgrounds,
                pointBorderWidth: 2,
            }, {
                label: 'GW Points',
                data: pointsData,
                borderColor: '#8b5cf6',
                backgroundColor: 'transparent',
                borderDash: [5, 5],
                tension: 0.3,
                yAxisID: 'y1',
                pointRadius: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { color: '#9090a0' } },
                tooltip: {
                    callbacks: {
                        afterBody: function(context) {
                            const idx = context[0].dataIndex;
                            const d = progression[idx];
                            let lines = [];
                            if (d.rank_change) {
                                const arrow = d.rank_change > 0 ? '🟢 ↑' : '🔴 ↓';
                                lines.push(`${arrow} ${formatRankChange(d.rank_change)} places`);
                            }
                            if (d.chip) {
                                const chipNames = { wildcard: 'Wildcard', '3xc': 'Triple Captain', bboost: 'Bench Boost', freehit: 'Free Hit' };
                                lines.push(`⭐ ${chipNames[d.chip] || d.chip}`);
                            }
                            if (d.hit_cost > 0) lines.push(`🔻 -${d.hit_cost} pts (hit)`);
                            if (d.bench_pts > 5) lines.push(`🪑 ${d.bench_pts} pts on bench`);
                            return lines;
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#606070' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    type: 'logarithmic',
                    display: true,
                    position: 'left',
                    reverse: true,
                    min: scaledMin,
                    max: scaledMax,
                    afterBuildTicks: function(axis) {
                        axis.ticks = filteredTicks.map(v => ({ value: v }));
                    },
                    ticks: {
                        color: '#00d4aa',
                        callback: function(value) {
                            if (value >= 1000000) return (value/1000000).toFixed(value >= 10000000 ? 0 : 1) + 'M';
                            if (value >= 1000) return (value/1000).toFixed(0) + 'K';
                            return value;
                        }
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    title: { display: true, text: 'Overall Rank', color: '#00d4aa' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    ticks: { color: '#8b5cf6' },
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'GW Points', color: '#8b5cf6' }
                }
            }
        }
    });
}

async function loadRankings() {
    const gwStart = document.getElementById('gwStart').value;
    const gwEnd = document.getElementById('gwEnd').value;
    const minMinutes = document.getElementById('minMinutes').value;
    const maxPrice = document.getElementById('maxPriceFilter').value;
    const maxOwnership = document.getElementById('maxOwnership').value;
    const teamFilter = document.getElementById('teamFilter').value;
    const searchQuery = document.getElementById('playerSearch').value.trim();

    document.getElementById('rankingsBody').innerHTML = `<tr><td colspan="13" class="loading"><div class="loading-spinner"></div><span style="margin-left: 0.5rem">Loading ${currentPosition} rankings...</span></td></tr>`;
    
    // Update table class for position-specific columns
    const table = document.getElementById('rankingsTable');
    table.className = `data-table pos-${currentPosition.toLowerCase()}`;

    try {
        // Build API URL with filters
        let url = `${API_BASE}/api/rankings/${currentPosition}?gw_start=${gwStart}&gw_end=${gwEnd}&min_minutes=${minMinutes}&min_price=3.5&max_price=${maxPrice}&max_ownership=${maxOwnership}&sort_by=${currentSort.field}&sort_order=${currentSort.order}`;
        
        // Add team filter if selected
        if (teamFilter) {
            url += `&team=${encodeURIComponent(teamFilter)}`;
        }
        
        const response = await fetch(url);
        const data = await response.json();
        let players = data.players;
        
        // Apply client-side search filter if query exists
        if (searchQuery && searchQuery.length >= 3) {
            const query = searchQuery.toLowerCase();
            players = players.filter(p => 
                p.name.toLowerCase().includes(query) || 
                p.team.toLowerCase().includes(query)
            );
        }
        
        rankingsData = players;
        document.getElementById('rankingsTitle').textContent = `${currentPosition} Rankings (${data.gw_range})`;
        document.getElementById('rankingsCount').textContent = `${players.length} players`;

        const html = players.map((p, i) => {
            // DEFCON column (shown for DEF/MID)
            const defconClass = p.defcon_per_90 >= 10 ? 'defcon-high' : (p.defcon_per_90 >= 7 ? 'defcon-medium' : 'defcon-low');
            const defconHtml = `<td class="pos-col def-col mid-col"><span class="defcon-badge ${defconClass}">${p.defcon_per_90 !== null ? p.defcon_per_90.toFixed(1) : '-'}</span></td>`;
            
            // Saves column (shown for GKP)
            const savesHtml = `<td class="pos-col gkp-col stat-value">${p.saves_per_90 !== null ? p.saves_per_90.toFixed(1) : '-'}</td>`;
            
            return `
            <tr>
                <td>${i + 1}</td>
                <td><div class="player-info"><span class="player-name">${p.name}</span><span class="player-team">${p.team}</span></div></td>
                <td class="stat-value">£${p.price.toFixed(1)}</td>
                <td class="stat-value stat-highlight" title="Range: ${(p.xpts_floor || p.xpts * 0.7).toFixed(1)} - ${(p.xpts_ceiling || p.xpts * 1.2).toFixed(1)}">
                    ${p.xpts.toFixed(1)}
                    <span class="xpts-range">${(p.xpts_floor || p.xpts * 0.7).toFixed(1)}-${(p.xpts_ceiling || p.xpts * 1.2).toFixed(1)}</span>
                </td>
                <td>
                    <div class="exp-mins-cell ${p.minutes_reason === 'user_override' ? 'has-override' : ''} ${p.minutes_reason === 'predicted' ? 'has-prediction' : ''}">
                        <input type="number" class="exp-mins-input ${p.minutes_reason === 'user_override' ? 'override-active' : ''}" 
                            data-player-id="${p.id}" 
                            value="${p.expected_minutes.toFixed(0)}" 
                            min="0" max="90" step="1"
                            onchange="setMinutesOverride(${p.id}, this.value)"
                            title="Edit to override expected minutes (${((p.prob_60_plus || 0.8) * 100).toFixed(0)}% chance of 60+ mins)">
                        <span class="exp-mins-reason ${p.minutes_reason}">${p.minutes_reason === 'user_override' ? '✓ override' : (p.minutes_reason === 'predicted' ? '⚡ predicted' : p.minutes_reason.replace('_', ' '))}</span>
                    </div>
                </td>
                <td class="stat-value">${p.xgi_per_90.toFixed(2)}</td>
                <td class="stat-value">${p.bonus_per_90.toFixed(2)}</td>
                ${defconHtml}
                ${savesHtml}
                <td class="stat-value">${(p.form || 0).toFixed(1)}</td>
                <td class="stat-value">${p.total_points}</td>
                <td class="stat-value">
                    <span class="ownership-cell">
                        ${p.ownership.toFixed(1)}%
                        <span class="ownership-tier ${p.ownership_tier || ''}" title="${p.ownership_tier_desc || ''}">${
                            p.ownership_tier === 'template' ? '📌' : 
                            p.ownership_tier === 'popular' ? '' : 
                            p.ownership_tier === 'differential' ? '💎' : 
                            p.ownership_tier === 'punt' ? '🎲' : ''
                        }</span>
                    </span>
                </td>
                <td>${renderFixtures(p.fixtures)}</td>
            </tr>
        `}).join('');
        document.getElementById('rankingsBody').innerHTML = html || '<tr><td colspan="13" class="empty-state">No players found</td></tr>';
    } catch (error) {
        console.error('Failed to load rankings:', error);
    }
}

// Unified function to refresh all views after minutes override
function refreshAfterOverride() {
    loadRankings();
    if (savedTeamId && myTeamData) {
        loadMyTeam();
    }
}

// Minutes override function (from Rankings table)
async function setMinutesOverride(playerId, minutes) {
    const input = document.querySelector(`input[data-player-id="${playerId}"]`);
    try {
        const response = await fetch(`${API_BASE}/api/minutes-override`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ player_id: playerId, expected_minutes: parseFloat(minutes) })
        });
        if (response.ok) {
            input.classList.add('override-active');
            // Refresh both views
            refreshAfterOverride();
        }
    } catch (error) {
        console.error('Failed to set override:', error);
    }
}

// Quick override from My Team pitch
async function promptMinutesOverride(playerId, playerName, currentMins) {
    const newMins = prompt(
        `Override expected minutes for ${playerName}\n\nCurrent: ${currentMins.toFixed(0)} mins\n\nEnter new value (0-90):`,
        Math.round(currentMins)
    );
    
    if (newMins === null) return; // Cancelled
    
    const mins = parseInt(newMins);
    if (isNaN(mins) || mins < 0 || mins > 90) {
        alert('Please enter a valid number between 0 and 90');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/minutes-override`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ player_id: playerId, expected_minutes: mins })
        });
        
        if (response.ok) {
            // Refresh both views
            refreshAfterOverride();
        } else {
            alert('Failed to save override');
        }
    } catch (error) {
        console.error('Failed to set override:', error);
        alert('Failed to save override');
    }
}

// ==================== ADMIN PANEL: Team Strength ====================
const ADMIN_MANAGER_ID = 616495;

function checkAdminAccess() {
    // Show admin panel if current team ID matches admin
    const teamIdInput = document.getElementById('teamIdInput');
    const teamId = parseInt(teamIdInput?.value || localStorage.getItem('fpl_team_id') || '0');
    const adminPanel = document.getElementById('adminPanel');
    if (adminPanel) {
        adminPanel.classList.toggle('visible', teamId === ADMIN_MANAGER_ID);
    }
}

async function submitTeamStrength() {
    const input = document.getElementById('teamStrengthInput');
    const status = document.getElementById('teamStrengthStatus');
    const rawText = input.value.trim();
    
    if (!rawText) {
        status.textContent = 'Please paste team data';
        status.className = 'admin-status error';
        return;
    }
    
    // Parse the input - supports multiple formats
    const teams = [];
    const lines = rawText.split('\n').filter(line => line.trim());
    
    for (const line of lines) {
        // Try CSV format: Team,xGFor,xGAg
        const csvMatch = line.match(/^([^,]+),\s*([\d.]+),\s*([\d.]+)/);
        if (csvMatch) {
            teams.push({
                team: csvMatch[1].trim(),
                adjxg_for: parseFloat(csvMatch[2]),
                adjxg_ag: parseFloat(csvMatch[3])
            });
            continue;
        }
        
        // Try table format: Rank Team Played Points xPoints xGFor xGAg xGDiff
        // or just: Team ... xGFor xGAg ...
        const parts = line.split(/\s{2,}|\t/).map(p => p.trim()).filter(p => p);
        if (parts.length >= 3) {
            // Find the team name (first non-numeric that's not a rank)
            let teamName = null;
            let numbers = [];
            
            for (const part of parts) {
                if (!teamName && isNaN(parseFloat(part))) {
                    teamName = part;
                } else if (teamName && !isNaN(parseFloat(part))) {
                    numbers.push(parseFloat(part));
                }
            }
            
            // We need at least team name and some numbers
            // Format from screenshot: Played, Points, xPoints, AdjxG For, AdjxG Ag, AdjxG Diff
            // Positions: [0]=Played, [1]=Points, [2]=xPoints, [3]=AdjxG For, [4]=AdjxG Ag, [5]=AdjxG Diff
            if (teamName && numbers.length >= 5) {
                teams.push({
                    team: teamName,
                    adjxg_for: numbers[3],  // AdjxG For
                    adjxg_ag: numbers[4]    // AdjxG Ag
                });
            }
        }
    }
    
    if (teams.length === 0) {
        status.textContent = 'Could not parse any teams. Use format: Team,xGFor,xGAg';
        status.className = 'admin-status error';
        return;
    }
    
    status.textContent = `Parsed ${teams.length} teams, submitting...`;
    status.className = 'admin-status';
    
    try {
        const response = await fetch(`${API_BASE}/api/team-strength`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                teams: teams,
                manager_id: ADMIN_MANAGER_ID
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            status.textContent = `✓ Updated ${result.count} teams: ${result.updated_teams.join(', ')}`;
            status.className = 'admin-status success';
            input.value = '';
        } else {
            status.textContent = `Error: ${result.detail || 'Unknown error'}`;
            status.className = 'admin-status error';
        }
    } catch (error) {
        status.textContent = `Error: ${error.message}`;
        status.className = 'admin-status error';
    }
}

async function viewTeamStrength() {
    const status = document.getElementById('teamStrengthStatus');
    const input = document.getElementById('teamStrengthInput');
    
    try {
        const response = await fetch(`${API_BASE}/api/team-strength`);
        const data = await response.json();
        
        // Format the data for display
        let output = '# Current Team Strength Data\n';
        output += '# Team, Manual xG For, Manual xG Ag, Form xG, Form xGA, Blended xG, Blended xGA, CS%\n\n';
        
        const teams = Object.values(data.teams).sort((a, b) => 
            (b.blended_xg || 0) - (a.blended_xg || 0)
        );
        
        for (const team of teams) {
            const manual = team.manual_adjxg_for ? 
                `${team.manual_adjxg_for.toFixed(2)}, ${team.manual_adjxg_ag.toFixed(2)}` : 
                '-, -';
            output += `${team.team_name}, ${manual}, ${team.form_xg || '-'}, ${team.form_xga || '-'}, ${team.blended_xg || '-'}, ${team.blended_xga || '-'}, ${((team.cs_probability || 0) * 100).toFixed(1)}%\n`;
        }
        
        output += `\n# Last manual update: ${data.manual_last_update || 'Never'}`;
        output += `\n# Last FDR update: ${data.fdr_last_update || 'Never'}`;
        
        input.value = output;
        status.textContent = `Loaded ${teams.length} teams`;
        status.className = 'admin-status success';
    } catch (error) {
        status.textContent = `Error: ${error.message}`;
        status.className = 'admin-status error';
    }
}

async function clearTeamStrength() {
    if (!confirm('Clear all manual team strength data? FDR will revert to Understat-only calculations.')) {
        return;
    }
    
    const status = document.getElementById('teamStrengthStatus');
    
    try {
        const response = await fetch(`${API_BASE}/api/team-strength?manager_id=${ADMIN_MANAGER_ID}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            status.textContent = '✓ Team strength data cleared';
            status.className = 'admin-status success';
        } else {
            const result = await response.json();
            status.textContent = `Error: ${result.detail || 'Unknown error'}`;
            status.className = 'admin-status error';
        }
    } catch (error) {
        status.textContent = `Error: ${error.message}`;
        status.className = 'admin-status error';
    }
}

async function loadFixtureRatings() {
    document.getElementById('fdrGrid').innerHTML = '<div class="loading" style="grid-column: 1 / -1; padding: 2rem;"><div class="loading-spinner"></div></div>';
    try {
        const response = await fetch(`${API_BASE}/api/fdr-grid`);
        const data = await response.json();
        const currentGW = data.current_gw || 22;
        const endGW = 38;
        const numGWs = endGW - currentGW + 1;
        
        // Update grid template columns dynamically
        const gridEl = document.getElementById('fdrGrid');
        gridEl.style.gridTemplateColumns = `100px repeat(${numGWs}, minmax(45px, 1fr))`;
        
        // Build header row - only show current GW onwards
        let html = '<div class="fdr-grid-header">';
        html += '<div class="team-col">Team</div>';
        for (let gw = currentGW; gw <= endGW; gw++) {
            const isCurrent = gw === currentGW;
            html += `<div class="gw-col ${isCurrent ? 'current' : ''}">${gw}</div>`;
        }
        html += '</div>';
        
        // Build team rows - only show current GW onwards
        for (const team of data.teams) {
            html += '<div class="fdr-grid-row">';
            html += `<div class="team-col">${team.team_short}</div>`;
            
            for (let gw = currentGW; gw <= endGW; gw++) {
                const fixtures = team.fixtures[gw] || [];
                
                if (fixtures.length === 0) {
                    html += '<div><div class="fdr-cell blank">-</div></div>';
                } else if (fixtures.length === 1) {
                    const f = fixtures[0];
                    html += `<div><div class="fdr-cell fdr-${f.fdr}">
                        <span class="opp">${f.is_home ? '' : '@'}${f.opponent}</span>
                    </div></div>`;
                } else {
                    // DGW - show both fixtures stacked
                    html += `<div><div class="fdr-cell fdr-${fixtures[0].fdr} dgw" style="font-size:0.55rem; padding: 0.1rem;">
                        ${fixtures.map(f => `<span>${f.is_home ? '' : '@'}${f.opponent}</span>`).join('')}
                    </div></div>`;
                }
            }
            html += '</div>';
        }
        
        document.getElementById('fdrGrid').innerHTML = html;
    } catch (error) {
        console.error('Failed to load FDR grid:', error);
        document.getElementById('fdrGrid').innerHTML = '<div style="grid-column: 1 / -1; padding: 2rem; text-align: center; color: var(--text-muted);">Failed to load fixtures</div>';
    }
}

function renderFixtures(fixtures) {
    if (!fixtures?.length) return '-';
    return `<div class="fixtures-row">${fixtures.slice(0, 5).map(f => {
        const fdr = f.fdr || f.difficulty || 5;
        return `<div class="fixture-badge fdr-${fdr}"><span class="gw">${f.gameweek}</span><span class="opp">${f.is_home ? '' : '@'}${f.opponent}</span></div>`;
    }).join('')}</div>`;
}

function formatNumber(num) {
    if (Math.abs(num) >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (Math.abs(num) >= 1000) return (num / 1000).toFixed(0) + 'K';
    return num.toString();
}

// ===== PLANNER FUNCTIONALITY =====
let plannerData = null;
let selectedStrategy = 'balanced';
let bookedTransfers = [];
let preBookedTransfers = [];  // Before team load (by name)
let plannerLoaded = false;
let allPlayersBootstrap = [];
let activeAutocompleteInput = null;

// Load bootstrap data for autocomplete (called once when planner tab is first accessed)
async function ensureBootstrapLoaded() {
    if (allPlayersBootstrap.length === 0) {
        try {
            const bootstrapRes = await fetch(`${API_BASE}/api/bootstrap`);
            const bootstrapData = await bootstrapRes.json();
            allPlayersBootstrap = bootstrapData.elements || [];
        } catch (err) {
            console.error('Failed to load bootstrap for autocomplete:', err);
        }
    }
}

// Autocomplete functionality with debounce
let autocompleteTimer = null;
function showAutocomplete(input, dropdownId) {
    clearTimeout(autocompleteTimer);
    autocompleteTimer = setTimeout(() => showAutocompleteImmediate(input, dropdownId), 250);
}
function showAutocompleteImmediate(input, dropdownId) {
    const query = input.value.trim().toLowerCase();
    const dropdown = document.getElementById(dropdownId);
    
    if (query.length < 3) {
        dropdown.style.display = 'none';
        return;
    }

    // Ensure bootstrap is loaded
    if (allPlayersBootstrap.length === 0) {
        ensureBootstrapLoaded().then(() => showAutocompleteImmediate(input, dropdownId));
        return;
    }

    // Find matching players
    const matches = allPlayersBootstrap
        .filter(p => p.web_name.toLowerCase().includes(query) || 
                    (p.first_name + ' ' + p.second_name).toLowerCase().includes(query))
        .slice(0, 10);

    if (matches.length === 0) {
        dropdown.style.display = 'none';
        return;
    }

    // Get team names
    const posMap = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'};

    dropdown.innerHTML = matches.map(p => `
        <div class="autocomplete-item" onclick="selectAutocomplete('${input.id}', '${dropdownId}', '${p.web_name}', ${p.id})">
            <div class="player-info">
                <span class="player-name">${p.web_name}</span>
                <span class="player-meta">${posMap[p.element_type] || 'MID'} • ${p.team_short || ''}</span>
            </div>
            <span class="player-price">£${(p.now_cost / 10).toFixed(1)}m</span>
        </div>
    `).join('');

    dropdown.style.display = 'block';
    activeAutocompleteInput = input;
}

function selectAutocomplete(inputId, dropdownId, name, playerId) {
    const input = document.getElementById(inputId);
    const dropdown = document.getElementById(dropdownId);
    
    input.value = name;
    input.dataset.playerId = playerId;
    dropdown.style.display = 'none';
}

// Close autocomplete when clicking outside
document.addEventListener('click', function(e) {
    if (!e.target.closest('.autocomplete-wrapper')) {
        document.querySelectorAll('.autocomplete-dropdown').forEach(d => d.style.display = 'none');
    }
});

// Pre-booked transfer (before loading, by name + stored player ID)
function addPreBookedTransfer() {
    const outInput = document.getElementById('preBookedOut');
    const inInput = document.getElementById('preBookedIn');
    const outName = outInput.value.trim();
    const inName = inInput.value.trim();
    const outId = outInput.dataset.playerId ? parseInt(outInput.dataset.playerId) : null;
    const inId = inInput.dataset.playerId ? parseInt(inInput.dataset.playerId) : null;
    const gw = parseInt(document.getElementById('preBookedGw').value);

    if (!outName || !inName) {
        alert('Please enter both player names');
        return;
    }

    preBookedTransfers.push({
        out_name: outName,
        in_name: inName,
        out_id: outId,
        in_id: inId,
        gw: gw
    });

    outInput.value = '';
    inInput.value = '';

    renderPreBookedList();
}

function removePreBookedTransfer(idx) {
    preBookedTransfers.splice(idx, 1);
    renderPreBookedList();
}

function renderPreBookedList() {
    const container = document.getElementById('preBookedList');
    if (preBookedTransfers.length === 0) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = preBookedTransfers.map((bt, idx) => `
        <div class="booked-item">
            <div class="booked-players">
                <span style="color: var(--text-muted);">GW${bt.gw}:</span>
                <span class="booked-out">${bt.out_name}</span>
                →
                <span class="booked-in">${bt.in_name}</span>
            </div>
            <button class="remove-booked" onclick="removePreBookedTransfer(${idx})">×</button>
        </div>
    `).join('');
}

function toggleChipGw(chip) {
    const action = document.getElementById(`chipOverride_${chip}`).value;
    const gwSelect = document.getElementById(`chipGw_${chip}`);
    gwSelect.style.display = action === 'lock' ? 'inline-block' : 'none';
}

function getChipOverrides() {
    const chips = ['wildcard', 'freehit', 'bboost', '3xc'];
    const overrides = [];
    for (const chip of chips) {
        const actionEl = document.getElementById(`chipOverride_${chip}`);
        if (!actionEl) continue;
        const action = actionEl.value;
        if (action === 'auto') continue;
        const override = { chip, action };
        if (action === 'lock') {
            const gwEl = document.getElementById(`chipGw_${chip}`);
            override.gw = parseInt(gwEl.value);
        }
        overrides.push(override);
    }
    return overrides;
}

async function loadPlanner() {
    const teamIdInput = document.getElementById('plannerTeamIdInput');
    const horizonSelect = document.getElementById('plannerHorizonSelect');
    const managerId = teamIdInput.value.trim() || savedTeamId;
    const horizon = horizonSelect.value;

    if (!managerId) {
        alert('Please enter a Manager ID');
        return;
    }

    // Show loading
    document.getElementById('plannerInputSection').style.display = 'none';
    document.getElementById('plannerLoading').style.display = 'block';
    document.getElementById('plannerContent').style.display = 'none';

    try {
        // Load bootstrap for player names if not loaded
        if (allPlayersBootstrap.length === 0) {
            const bootstrapRes = await fetch(`${API_BASE}/api/bootstrap`);
            const bootstrapData = await bootstrapRes.json();
            const teams = bootstrapData.teams || [];
            const teamMap = {};
            teams.forEach(t => teamMap[t.id] = t.short_name);
            
            allPlayersBootstrap = (bootstrapData.elements || []).map(p => ({
                ...p,
                team_short: teamMap[p.team] || ''
            }));
        }

        // Convert pre-booked transfers to booked transfers (by ID)
        // Prefer stored player IDs from autocomplete; fall back to exact name match
        bookedTransfers = [];
        for (const pbt of preBookedTransfers) {
            const outId = pbt.out_id || (allPlayersBootstrap.find(p =>
                p.web_name.toLowerCase() === pbt.out_name.toLowerCase()
            ) || {}).id;
            const inId = pbt.in_id || (allPlayersBootstrap.find(p =>
                p.web_name.toLowerCase() === pbt.in_name.toLowerCase()
            ) || {}).id;

            if (outId && inId) {
                bookedTransfers.push({
                    out_id: outId,
                    in_id: inId,
                    gw: pbt.gw,
                    reason: ''
                });
            }
        }

        // Collect chip overrides
        const chipOverrides = getChipOverrides();

        // Load planner data
        const res = await fetch(`${API_BASE}/api/transfer-planner/${managerId}?horizon=${horizon}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                booked_transfers: bookedTransfers,
                chip_overrides: chipOverrides,
            })
        });

        if (!res.ok) {
            throw new Error('Failed to load planner');
        }

        plannerData = await res.json();
        plannerLoaded = true;
        renderPlanner();

    } catch (err) {
        console.error('Planner error:', err);
        alert('Failed to load planner: ' + err.message);
        document.getElementById('plannerContent').style.display = 'none';
        plannerData = null;
        plannerLoaded = false;
        document.getElementById('plannerInputSection').style.display = 'block';
    } finally {
        document.getElementById('plannerLoading').style.display = 'none';
    }
}

function renderPlanner() {
    if (!plannerData) return;

    document.getElementById('plannerContent').style.display = 'block';

    // Top bar info
    document.getElementById('plannerManagerName').textContent = plannerData.manager.name;
    document.getElementById('plannerBank').textContent = `£${plannerData.bank.toFixed(1)}m`;
    document.getElementById('plannerFT').textContent = plannerData.free_transfers;
    document.getElementById('plannerHorizon').textContent = plannerData.horizon_range;

    // Update GW selector for booked transfers
    const gwSelect = document.getElementById('bookedGw');
    gwSelect.innerHTML = '';
    for (let gw = plannerData.planning_gw; gw <= plannerData.planning_gw + plannerData.horizon; gw++) {
        gwSelect.innerHTML += `<option value="${gw}">GW${gw}</option>`;
    }

    // Chips - render as styled badges with planned GW info
    const chipsDiv = document.getElementById('plannerChips');
    const selectedPlacements = plannerData.strategies[selectedStrategy]?.chip_placements || [];
    const chipPlacementMap = {};
    selectedPlacements.forEach(cp => { chipPlacementMap[cp.chip_id || cp.chip] = cp.gw; });
    chipsDiv.innerHTML = Object.entries(plannerData.available_chips).map(([chip, available]) => {
        const plannedGw = chipPlacementMap[chip];
        const plannedText = plannedGw ? ` (GW${plannedGw})` : '';
        const cls = available ? (plannedGw ? 'planned' : 'available') : 'used';
        return `<span class="planner-chip ${cls}">${chip}${plannedText}</span>`;
    }).join('');

    // Squad health
    const health = plannerData.squad_health;
    const scoreEl = document.getElementById('healthScore');
    scoreEl.textContent = health.health_score;
    scoreEl.className = 'health-score ' + (
        health.health_score >= 80 ? 'good' :
        health.health_score >= 50 ? 'warning' : 'critical'
    );

    document.getElementById('criticalCount').textContent = health.critical_count;
    document.getElementById('warningCount').textContent = health.warning_count;
    document.getElementById('minorCount').textContent = health.minor_count;

    // Issue list
    const issueList = document.getElementById('issueList');
    if (health.issues.length === 0) {
        issueList.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 1rem;">✓ No issues detected</div>';
    } else {
        issueList.innerHTML = health.issues.map(issue => `
            <div class="issue-item ${issue.severity}">
                <div class="issue-player">${issue.player_name} (${issue.team})</div>
                <div class="issue-desc">${issue.description}</div>
                <div class="issue-rec">${issue.recommendation}</div>
            </div>
        `).join('');
    }

    // Chip recommendations
    const chipRecs = document.getElementById('chipRecommendations');
    const allChipRecs = [];
    Object.values(plannerData.strategies).forEach(s => {
        s.chip_recommendations.forEach(c => {
            if (!allChipRecs.find(r => r.chip === c.chip && r.recommended_gw === c.recommended_gw)) {
                allChipRecs.push(c);
            }
        });
    });

    if (allChipRecs.length === 0) {
        chipRecs.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem;">No chip recommendations in horizon</div>';
    } else {
        chipRecs.innerHTML = allChipRecs.slice(0, 3).map(rec => `
            <div class="chip-rec ${rec.confidence}">
                <div class="chip-rec-header">
                    <span class="chip-rec-chip">${rec.chip}</span>
                    <span class="chip-rec-gw">GW${rec.recommended_gw}</span>
                </div>
                <div class="chip-rec-reason">${rec.reason}</div>
            </div>
        `).join('');
    }

    // Squad mini
    const squadMini = document.getElementById('squadMini');
    const issuePlayerIds = new Set(health.issues.filter(i => i.severity === 'critical').map(i => i.player_id));

    squadMini.innerHTML = plannerData.squad.map(p => {
        const isIssue = issuePlayerIds.has(p.id);
        const isBench = p.squad_position > 11;
        return `
            <div class="squad-mini-player ${isBench ? 'bench' : 'starting'} ${isIssue ? 'issue' : ''}">
                <span class="player-pos">${p.position}</span>
                <span class="player-name">${p.name}</span>
                <span class="player-xpts">${p.xpts.toFixed(1)}</span>
            </div>
        `;
    }).join('');

    // Booked transfers (resolved from API response)
    renderBookedTransfers();

    // Strategies
    renderStrategies();

    // Timeline
    renderTimeline();

    // Heatmap
    renderHeatmap();
}

function renderStrategies() {
    const container = document.getElementById('strategyContainer');
    container.innerHTML = Object.entries(plannerData.strategies).map(([key, strategy]) => {
        const riskBars = Array(10).fill(0).map((_, i) => {
            const filled = i < strategy.risk_score;
            const level = i >= 7 ? 'max' : i >= 4 ? 'high' : '';
            return `<div class="risk-bar ${filled ? 'filled' : ''} ${level}"></div>`;
        }).join('');

        return `
            <div class="strategy-card ${key === selectedStrategy ? 'selected' : ''}" 
                 onclick="selectStrategy('${key}')">
                <div class="strategy-header">
                    <span class="strategy-name ${key}">${strategy.name}</span>
                    <div class="risk-meter">${riskBars}</div>
                </div>
                <div class="strategy-headline">${strategy.headline}</div>
                <div class="strategy-stats">
                    <div>
                        <div class="strategy-stat-value">${strategy.total_xpts.toFixed(1)}</div>
                        <div class="strategy-stat-label">xPts</div>
                    </div>
                    <div>
                        <div class="strategy-stat-value">${strategy.transfers_made}</div>
                        <div class="strategy-stat-label">Transfers</div>
                    </div>
                    <div>
                        <div class="strategy-stat-value">${strategy.hit_cost > 0 ? '-' + strategy.hit_cost : '0'}</div>
                        <div class="strategy-stat-label">Hits</div>
                    </div>
                </div>
                ${(strategy.chip_placements && strategy.chip_placements.length > 0) ? `
                    <div class="strategy-chip-tags">
                        ${strategy.chip_placements.map(cp =>
                            `<span class="strategy-chip-tag">${cp.chip} GW${cp.gw}</span>`
                        ).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

function selectStrategy(strategy) {
    selectedStrategy = strategy;
    // Update chip badges to show placements for selected strategy
    const chipsDiv = document.getElementById('plannerChips');
    if (chipsDiv && plannerData) {
        const selectedPlacements = plannerData.strategies[selectedStrategy]?.chip_placements || [];
        const chipPlacementMap = {};
        selectedPlacements.forEach(cp => { chipPlacementMap[cp.chip_id || cp.chip] = cp.gw; });
        chipsDiv.innerHTML = Object.entries(plannerData.available_chips).map(([chip, available]) => {
            const plannedGw = chipPlacementMap[chip];
            const plannedText = plannedGw ? ` (GW${plannedGw})` : '';
            const cls = available ? (plannedGw ? 'planned' : 'available') : 'used';
            return `<span class="planner-chip ${cls}">${chip}${plannedText}</span>`;
        }).join('');
    }
    renderStrategies();
    renderTimeline();
}

function renderTimeline() {
    const strategy = plannerData.strategies[selectedStrategy];
    document.getElementById('selectedStrategyName').textContent = strategy.name;

    const timeline = document.getElementById('timeline');
    const gws = Object.values(strategy.gw_actions).sort((a, b) => a.gw - b.gw);

    // Build chip placement lookup from strategy
    const chipMap = {};
    const chipLabels = {wildcard:'WILDCARD', freehit:'FREE HIT', bboost:'BENCH BOOST', '3xc':'TRIPLE CAPTAIN'};
    (strategy.chip_placements || []).forEach(cp => { chipMap[cp.gw] = cp; });

    // GW jump navigation
    let navHtml = '<div class="timeline-gw-nav">';
    gws.forEach(gw => {
        const hasTransfer = gw.action !== 'roll' && gw.transfers && gw.transfers.length > 0;
        const hasHit = hasTransfer && gw.transfers.some(t => t.is_hit);
        const hasChip = !!chipMap[gw.gw];
        const cls = hasChip ? 'has-chip' : (hasHit ? 'has-hit' : (hasTransfer ? 'has-action' : ''));
        navHtml += `<button class="timeline-gw-jump ${cls}" onclick="document.getElementById('tl-gw-${gw.gw}').scrollIntoView({behavior:'smooth',block:'center'})">GW${gw.gw}</button>`;
    });
    navHtml += '</div>';

    let html = navHtml + `
        <div class="timeline-header">
            <div>GW</div>
            <div>Action</div>
            <div>xPts</div>
            <div>FT</div>
        </div>
    `;

    gws.forEach(gw => {
        let actionHtml = '';
        const chipPlacement = chipMap[gw.gw];
        const chipId = gw.chip || chipPlacement?.chip_id;

        // Chip badge if active this GW
        if (chipId) {
            const chipCls = chipId === '3xc' ? 'tc' : chipId;
            actionHtml += `<span class="chip-active-badge ${chipCls}">${chipLabels[chipId] || chipId.toUpperCase()}</span><br>`;
        }

        if (gw.action === 'wildcard') {
            const wcSquad = gw.wildcard_squad;
            if (wcSquad && wcSquad.length > 0) {
                actionHtml += `<button class="wc-squad-toggle" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'; this.textContent = this.nextElementSibling.style.display === 'none' ? 'View WC Squad ▼' : 'Hide WC Squad ▲'">View WC Squad ▼</button>`;
                actionHtml += `<div class="wc-squad-panel" style="display:none;">`;
                const posOrder = {'GKP':1,'DEF':2,'MID':3,'FWD':4};
                const sorted = [...wcSquad].sort((a,b) => (posOrder[a.position]||9) - (posOrder[b.position]||9));
                sorted.forEach(p => {
                    actionHtml += `<div class="wc-squad-player"><span class="wc-pos">${p.position}</span> <span class="wc-name">${p.name}</span> <span class="wc-team">${p.team}</span> <span class="wc-price">£${p.price.toFixed(1)}m</span></div>`;
                });
                actionHtml += `</div>`;
            } else {
                actionHtml += '<span class="action-badge roll">Squad rebuilt</span>';
            }
        } else if (gw.action === 'freehit') {
            const fhSquad = gw.freehit_squad;
            if (fhSquad && fhSquad.length > 0) {
                actionHtml += `<button class="wc-squad-toggle" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'; this.textContent = this.nextElementSibling.style.display === 'none' ? 'View FH Squad ▼' : 'Hide FH Squad ▲'">View FH Squad ▼</button>`;
                actionHtml += `<div class="wc-squad-panel" style="display:none;">`;
                const posOrder = {'GKP':1,'DEF':2,'MID':3,'FWD':4};
                const sorted = [...fhSquad].sort((a,b) => (posOrder[a.position]||9) - (posOrder[b.position]||9));
                sorted.forEach(p => {
                    actionHtml += `<div class="wc-squad-player"><span class="wc-pos">${p.position}</span> <span class="wc-name">${p.name}</span> <span class="wc-team">${p.team}</span> <span class="wc-price">£${p.price.toFixed(1)}m</span></div>`;
                });
                actionHtml += `</div>`;
            } else {
                actionHtml += '<span class="action-badge roll">Temporary squad</span>';
            }
        } else if (gw.action === 'roll') {
            actionHtml += `<span class="action-badge roll">🔄 Roll FT</span>`;
        } else if (gw.transfers && gw.transfers.length > 0) {
            actionHtml += gw.transfers.map(t => {
                const badgeClass = t.is_hit ? 'hit' : t.is_booked ? 'booked' : 'transfer';
                const badgeText = t.is_hit ? '⚠️ HIT' : t.is_booked ? '📌 BOOKED' : '✅ FREE';
                return `
                    <div class="transfer-detail">
                        <span class="action-badge ${badgeClass}">${badgeText}</span>
                        <span class="transfer-out">${t.out?.name || '?'}</span>
                        →
                        <span class="transfer-in">${t.in?.name || '?'}</span>
                        <span class="xpts-gain">+${(t.xpts_gain || 0).toFixed(1)}</span>
                    </div>
                `;
            }).join('');
        } else {
            actionHtml += `<span class="action-badge roll">— Hold</span>`;
        }

        html += `
            <div class="timeline-row" id="tl-gw-${gw.gw}">
                <div class="gw-badge">GW${gw.gw}</div>
                <div>${actionHtml}</div>
                <div class="timeline-xpts">${gw.xpts?.toFixed(1) || '-'}</div>
                <div class="ft-badge">${gw.ft_before}→${gw.ft_after}</div>
            </div>
        `;
    });

    timeline.innerHTML = html;
}

function renderHeatmap() {
    const heatmap = plannerData.fixture_heatmap;
    if (!heatmap || heatmap.length === 0) return;

    // Get all GWs
    const gws = new Set();
    heatmap.forEach(team => {
        team.fixtures.forEach(f => gws.add(f.gw));
    });
    const sortedGws = Array.from(gws).sort((a, b) => a - b);

    let html = `
        <thead>
            <tr>
                <th>Team (Avg FDR)</th>
                ${sortedGws.map(gw => `<th>GW${gw}</th>`).join('')}
            </tr>
        </thead>
        <tbody>
    `;

    heatmap.forEach(team => {
        // Calculate average FDR for display
        const fdrValues = team.fixtures.map(f => f.fdr);
        const avgFdr = fdrValues.length > 0 
            ? (fdrValues.reduce((a, b) => a + b, 0) / fdrValues.length).toFixed(1) 
            : '-';
        
        const swingBadge = team.rating === 'IMPROVING' 
            ? `<span class="team-swing improving">↑</span>`
            : team.rating === 'WORSENING'
            ? `<span class="team-swing worsening">↓</span>`
            : '';

        html += `<tr><td>${team.team_short} <span style="color: var(--text-muted); font-size: 0.65rem;">(${avgFdr})</span>${swingBadge}</td>`;

        sortedGws.forEach(gw => {
            const fixture = team.fixtures.find(f => f.gw === gw);
            if (fixture) {
                const venue = fixture.is_home ? '' : '@';
                html += `
                    <td>
                        <div class="fixture-cell hm-fdr-${fixture.fdr} ${fixture.is_home ? '' : 'away'}">
                            ${venue}${fixture.opponent}
                        </div>
                    </td>
                `;
            } else {
                // Check for DGW or BGW
                const isBgw = team.bgw_gws?.includes(gw);
                html += `<td>${isBgw ? '<span style="color: var(--text-muted);">BGW</span>' : '-'}</td>`;
            }
        });

        html += '</tr>';
    });

    html += '</tbody>';
    document.getElementById('heatmapTable').innerHTML = html;
}

function renderBookedTransfers() {
    const container = document.getElementById('bookedTransfers');

    if (bookedTransfers.length === 0 && preBookedTransfers.length === 0) {
        container.innerHTML = '<div style="color: var(--text-muted); font-size: 0.75rem;">No booked transfers. Use the form above to plan ahead.</div>';
        return;
    }

    // Show resolved booked transfers
    container.innerHTML = bookedTransfers.map((bt, idx) => {
        const outPlayer = allPlayersBootstrap.find(p => p.id === bt.out_id);
        const inPlayer = allPlayersBootstrap.find(p => p.id === bt.in_id);

        return `
            <div class="booked-item">
                <div class="booked-players">
                    <span style="color: var(--text-muted);">GW${bt.gw}:</span>
                    <span class="booked-out">${outPlayer?.web_name || bt.out_id}</span>
                    →
                    <span class="booked-in">${inPlayer?.web_name || bt.in_id}</span>
                </div>
                <button class="remove-booked" onclick="removeBookedTransfer(${idx})">×</button>
            </div>
        `;
    }).join('');
}

function addBookedTransfer() {
    const outName = document.getElementById('bookedOut').value.trim();
    const inName = document.getElementById('bookedIn').value.trim();
    const gw = parseInt(document.getElementById('bookedGw').value);

    if (!outName || !inName) {
        alert('Please enter both player names');
        return;
    }

    // Find players by name
    const outPlayer = allPlayersBootstrap.find(p => 
        p.web_name.toLowerCase().includes(outName.toLowerCase())
    );
    const inPlayer = allPlayersBootstrap.find(p => 
        p.web_name.toLowerCase().includes(inName.toLowerCase())
    );

    if (!outPlayer) {
        alert(`Could not find player: ${outName}`);
        return;
    }
    if (!inPlayer) {
        alert(`Could not find player: ${inName}`);
        return;
    }

    bookedTransfers.push({
        out_id: outPlayer.id,
        in_id: inPlayer.id,
        gw: gw,
        reason: ''
    });

    // Also add to preBooked for persistence
    preBookedTransfers.push({
        out_name: outPlayer.web_name,
        in_name: inPlayer.web_name,
        gw: gw
    });

    // Clear inputs
    document.getElementById('bookedOut').value = '';
    document.getElementById('bookedIn').value = '';

    // Reload planner with new booked transfers
    loadPlanner();
}

function removeBookedTransfer(idx) {
    bookedTransfers.splice(idx, 1);
    if (preBookedTransfers[idx]) {
        preBookedTransfers.splice(idx, 1);
    }
    loadPlanner();
}

// Back button to return to input
function showPlannerInput() {
    document.getElementById('plannerContent').style.display = 'none';
    document.getElementById('plannerInputSection').style.display = 'block';
    renderPreBookedList();
}
