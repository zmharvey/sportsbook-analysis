// Sports Betting Dashboard - Main JavaScript

// Dark mode toggle
document.addEventListener('DOMContentLoaded', function() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const html = document.documentElement;

    // Load saved preference
    const savedTheme = localStorage.getItem('theme') || 'dark';
    html.setAttribute('data-bs-theme', savedTheme);
    darkModeToggle.checked = savedTheme === 'dark';

    // Toggle handler
    darkModeToggle.addEventListener('change', function() {
        const theme = this.checked ? 'dark' : 'light';
        html.setAttribute('data-bs-theme', theme);
        localStorage.setItem('theme', theme);
    });

    // Update collector status
    updateCollectorStatus();
    setInterval(updateCollectorStatus, 30000);

    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));
});

// Update collector status badge
async function updateCollectorStatus() {
    const statusBadge = document.getElementById('collector-status');
    try {
        const response = await fetch('/api/collector-status');
        const data = await response.json();

        if (data.running) {
            statusBadge.className = 'badge bg-success me-3';
            statusBadge.innerHTML = '<i class="bi bi-circle-fill"></i> Running';
        } else if (data.last_poll) {
            const lastPoll = new Date(data.last_poll);
            const minutesAgo = Math.floor((Date.now() - lastPoll) / 60000);
            if (minutesAgo > 30) {
                statusBadge.className = 'badge bg-warning text-dark me-3';
                statusBadge.innerHTML = `<i class="bi bi-exclamation-triangle-fill"></i> Stale (${minutesAgo}m)`;
            } else {
                statusBadge.className = 'badge bg-info me-3';
                statusBadge.innerHTML = `<i class="bi bi-clock"></i> ${minutesAgo}m ago`;
            }
        } else {
            statusBadge.className = 'badge bg-secondary me-3';
            statusBadge.innerHTML = '<i class="bi bi-circle"></i> No data';
        }
    } catch (e) {
        statusBadge.className = 'badge bg-danger me-3';
        statusBadge.innerHTML = '<i class="bi bi-x-circle-fill"></i> Error';
    }
}

// Format American odds for display
function formatOdds(odds) {
    if (odds === null || odds === undefined) return '--';
    const num = parseFloat(odds);
    if (num > 0) return `+${num}`;
    return num.toString();
}

// Get odds class for styling
function getOddsClass(odds) {
    const num = parseFloat(odds);
    return num > 0 ? 'positive-odds' : 'negative-odds';
}

// Format EV percentage
function formatEV(ev) {
    if (ev === null || ev === undefined) return '--';
    const num = parseFloat(ev);
    const formatted = num.toFixed(2);
    return num > 0 ? `+${formatted}%` : `${formatted}%`;
}

// Get EV class for styling
function getEVClass(ev) {
    const num = parseFloat(ev);
    if (num >= 5) return 'ev-high';
    if (num >= 1) return 'ev-positive';
    return '';
}

// Format date/time
function formatDateTime(dateStr) {
    if (!dateStr) return '--';
    const date = new Date(dateStr);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    });
}

// Format time ago
function formatTimeAgo(dateStr) {
    if (!dateStr) return '--';
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
}

// Sport name mapping
const SPORT_NAMES = {
    'americanfootball_nfl': 'NFL',
    'americanfootball_ncaaf': 'NCAAF',
    'basketball_nba': 'NBA',
    'basketball_ncaab': 'NCAAB',
    'icehockey_nhl': 'NHL',
    'baseball_mlb': 'MLB',
    'soccer': 'Soccer',
    'soccer_epl': 'EPL',
    'soccer_usa_mls': 'MLS',
    'mma_mixed_martial_arts': 'MMA',
    'tennis_atp': 'ATP',
    'golf_pga': 'PGA'
};

function getSportName(sportKey) {
    return SPORT_NAMES[sportKey] || sportKey;
}

// Bookmaker name formatting
function formatBookmaker(bookmaker) {
    const names = {
        'draftkings': 'DraftKings',
        'fanduel': 'FanDuel',
        'betmgm': 'BetMGM',
        'caesars': 'Caesars',
        'williamhill_us': 'Caesars',
        'pointsbet_us': 'PointsBet',
        'pinnacle': 'Pinnacle',
        'bovada': 'Bovada',
        'betonlineag': 'BetOnline',
        'fanatics': 'Fanatics',
        // Sharp books
        'betfair': 'Betfair',
        'lowvig': 'LowVig',
        'betanyports': 'BetAnySports',
        'circa': 'Circa',
        // DFS (Daily Fantasy Sports) Books
        'prizepicks': 'PrizePicks',
        'underdog': 'Underdog',
        'betr_us_dfs': 'Betr'
    };
    return names[bookmaker] || bookmaker;
}

// Market name formatting
function formatMarket(market) {
    const names = {
        'h2h': 'Moneyline',
        'spreads': 'Spread',
        'totals': 'Total'
    };
    return names[market] || market;
}

// Show loading state
function showLoading(container) {
    container.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2 text-muted">Loading data...</p>
        </div>
    `;
}

// Show error state
function showError(container, message) {
    container.innerHTML = `
        <div class="text-center py-5">
            <i class="bi bi-exclamation-triangle text-warning" style="font-size: 3rem;"></i>
            <p class="mt-2 text-muted">${message}</p>
            <button class="btn btn-outline-primary btn-sm" onclick="location.reload()">
                <i class="bi bi-arrow-clockwise"></i> Retry
            </button>
        </div>
    `;
}

// Show empty state
function showEmpty(container, message) {
    container.innerHTML = `
        <div class="text-center py-5">
            <i class="bi bi-inbox text-muted" style="font-size: 3rem;"></i>
            <p class="mt-2 text-muted">${message}</p>
        </div>
    `;
}

// Update last updated timestamp
function updateLastUpdated() {
    const el = document.getElementById('last-updated');
    if (el) {
        el.textContent = 'Last updated: ' + new Date().toLocaleTimeString();
    }
}

// Debounce function for search
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export functions for use in other scripts
window.dashboardUtils = {
    formatOdds,
    getOddsClass,
    formatEV,
    getEVClass,
    formatDateTime,
    formatTimeAgo,
    getSportName,
    formatBookmaker,
    formatMarket,
    showLoading,
    showError,
    showEmpty,
    updateLastUpdated,
    debounce
};
