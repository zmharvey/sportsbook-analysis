"""
Player Props Module.
Handles player prop bet processing, organization, and +EV detection.

Player props differ from game lines in several ways:
- They are player-specific rather than team-specific
- They have a description field containing the player name
- Over/Under outcomes are paired for the same player/stat
- Different sports have different prop types
- Props tend to have wider markets (more vig)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime, timezone, timedelta

from arbitrage import american_to_decimal, decimal_to_american, is_game_upcoming
from ev_detector import (
    calculate_no_vig_probability,
    calculate_ev,
    calculate_kelly_units,
    calculate_sharp_width_cents,
    get_kelly_fraction_for_width,
    get_hours_until_game,
    DevigMethod,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PlayerPropOutcome:
    """Represents a single player prop outcome (e.g., LeBron James Over 25.5 Points)."""
    player_name: str
    prop_type: str  # e.g., "player_points", "player_rebounds"
    outcome: str  # "Over" or "Under"
    line: float  # e.g., 25.5
    bookmaker: str
    odds_american: int
    odds_decimal: float

    def __str__(self) -> str:
        return f"{self.player_name} {self.outcome} {self.line} ({self.prop_type}) @ {self.bookmaker} {self.odds_american:+d}"


@dataclass
class PlayerPropMarket:
    """
    Represents a complete player prop market (Over/Under pair for one player/stat/line).

    A market groups all bookmaker offerings for the same player, prop type, and line.
    This enables proper devigging and +EV detection.
    """
    player_name: str
    prop_type: str  # e.g., "player_points"
    line: float  # e.g., 25.5
    outcomes: list[PlayerPropOutcome] = field(default_factory=list)

    def get_over_outcomes(self) -> list[PlayerPropOutcome]:
        """Get all Over outcomes from different bookmakers."""
        return [o for o in self.outcomes if o.outcome == "Over"]

    def get_under_outcomes(self) -> list[PlayerPropOutcome]:
        """Get all Under outcomes from different bookmakers."""
        return [o for o in self.outcomes if o.outcome == "Under"]

    def get_best_over(self) -> Optional[PlayerPropOutcome]:
        """Get the best Over odds across all bookmakers."""
        overs = self.get_over_outcomes()
        return max(overs, key=lambda x: x.odds_decimal) if overs else None

    def get_best_under(self) -> Optional[PlayerPropOutcome]:
        """Get the best Under odds across all bookmakers."""
        unders = self.get_under_outcomes()
        return max(unders, key=lambda x: x.odds_decimal) if unders else None

    @property
    def market_key(self) -> str:
        """Unique key for this market (player + prop type + line)."""
        return f"{self.player_name}|{self.prop_type}|{self.line}"

    def __str__(self) -> str:
        over = self.get_best_over()
        under = self.get_best_under()
        over_str = f"O {over.odds_american:+d}" if over else "N/A"
        under_str = f"U {under.odds_american:+d}" if under else "N/A"
        return f"{self.player_name} {self.prop_type} {self.line}: {over_str} / {under_str}"


@dataclass
class PropEVOpportunity:
    """Represents a +EV opportunity on a player prop."""
    event_id: str
    sport: str
    home_team: str
    away_team: str
    commence_time: str

    # Player and prop info
    player_name: str
    prop_type: str
    line: float
    outcome: str  # "Over" or "Under"

    # Sharp book reference
    sharp_book: str
    sharp_odds_american: int
    sharp_odds_decimal: float
    sharp_implied_prob: float  # Fair probability as percentage

    # Soft book opportunity
    soft_book: str
    soft_odds_american: int
    soft_odds_decimal: float
    soft_implied_prob: float

    # EV and sizing
    ev_percent: float
    edge_percent: float
    sharp_width_cents: Optional[int] = None
    kelly_fraction: float = 0.10
    units: float = 0.0

    def __str__(self) -> str:
        return (
            f"+EV Prop: {self.player_name} {self.outcome} {self.line} {self.prop_type} "
            f"@ {self.soft_book} ({self.soft_odds_american:+d}) | "
            f"EV: {self.ev_percent:.2f}% | {self.units:.2f}u"
        )


# =============================================================================
# PROP TYPE UTILITIES
# =============================================================================

# Human-readable names for prop types
PROP_TYPE_NAMES = {
    # Basketball
    "player_points": "Points",
    "player_rebounds": "Rebounds",
    "player_assists": "Assists",
    "player_threes": "3-Pointers",
    "player_points_rebounds_assists": "Pts+Reb+Ast",
    "player_points_rebounds": "Pts+Reb",
    "player_points_assists": "Pts+Ast",
    "player_rebounds_assists": "Reb+Ast",
    "player_blocks": "Blocks",
    "player_steals": "Steals",
    "player_blocks_steals": "Blks+Stls",
    "player_turnovers": "Turnovers",
    "player_double_double": "Double-Double",
    "player_triple_double": "Triple-Double",

    # Football
    "player_pass_yds": "Pass Yards",
    "player_pass_tds": "Pass TDs",
    "player_pass_completions": "Completions",
    "player_pass_attempts": "Pass Attempts",
    "player_pass_interceptions": "Interceptions",
    "player_rush_yds": "Rush Yards",
    "player_rush_attempts": "Rush Attempts",
    "player_receptions": "Receptions",
    "player_reception_yds": "Rec Yards",
    "player_anytime_td": "Anytime TD",
    "player_1st_td": "First TD",
    "player_kicking_points": "Kicking Points",

    # Hockey
    "player_goals": "Goals",
    "player_shots_on_goal": "Shots on Goal",
    "player_total_saves": "Saves",
    "player_goal_scorer_anytime": "Anytime Goal",

    # Baseball (batter)
    "batter_home_runs": "Home Runs",
    "batter_hits": "Hits",
    "batter_rbis": "RBIs",
    "batter_runs_scored": "Runs",
    "batter_total_bases": "Total Bases",
    "batter_strikeouts": "Strikeouts",
    "batter_walks": "Walks",
    "batter_stolen_bases": "Stolen Bases",

    # Baseball (pitcher)
    "pitcher_strikeouts": "Strikeouts",
    "pitcher_hits_allowed": "Hits Allowed",
    "pitcher_earned_runs": "Earned Runs",
    "pitcher_outs": "Outs Recorded",
}


def get_prop_display_name(prop_type: str) -> str:
    """Get human-readable name for a prop type."""
    return PROP_TYPE_NAMES.get(prop_type, prop_type.replace("_", " ").title())


def is_player_prop_market(market_key: str) -> bool:
    """Check if a market key is a player prop market."""
    return (
        market_key.startswith("player_") or
        market_key.startswith("batter_") or
        market_key.startswith("pitcher_")
    )


# =============================================================================
# PROP PARSING AND ORGANIZATION
# =============================================================================

def extract_player_name_from_outcome(outcome: dict) -> str:
    """
    Extract player name from an outcome dict.

    The API returns player props with the player name in the 'description' field.
    Example: {"name": "Over", "description": "LeBron James", "price": -110, "point": 25.5}
    """
    return outcome.get("description", "Unknown Player")


def parse_prop_outcomes_from_event(
    event: dict,
    prop_markets: list[str],
    include_bookmakers: list[str],
    exclude_bookmakers: list[str],
) -> list[PlayerPropOutcome]:
    """
    Parse all player prop outcomes from an event.

    Args:
        event: Event data from the API (with bookmakers and markets)
        prop_markets: List of prop market keys to include
        include_bookmakers: Only include these bookmakers (empty = all)
        exclude_bookmakers: Exclude these bookmakers

    Returns:
        List of PlayerPropOutcome objects
    """
    outcomes = []

    for bookmaker in event.get("bookmakers", []):
        book_key = bookmaker["key"]

        # Filter bookmakers
        if include_bookmakers and book_key not in include_bookmakers:
            continue
        if book_key in exclude_bookmakers:
            continue

        for market in bookmaker.get("markets", []):
            market_key = market["key"]

            # Only process player prop markets
            if market_key not in prop_markets:
                continue

            for outcome in market.get("outcomes", []):
                # Extract player name and line
                player_name = extract_player_name_from_outcome(outcome)
                line = outcome.get("point")
                outcome_name = outcome["name"]  # "Over" or "Under"
                price = outcome["price"]

                # Skip if missing critical data
                if not player_name or line is None:
                    continue

                # Convert odds
                if isinstance(price, int) or (isinstance(price, float) and (price > 50 or price < -50)):
                    odds_american = int(price)
                    odds_decimal = american_to_decimal(odds_american)
                else:
                    odds_decimal = float(price)
                    odds_american = decimal_to_american(odds_decimal)

                outcomes.append(PlayerPropOutcome(
                    player_name=player_name,
                    prop_type=market_key,
                    outcome=outcome_name,
                    line=line,
                    bookmaker=book_key,
                    odds_american=odds_american,
                    odds_decimal=odds_decimal,
                ))

    return outcomes


def group_outcomes_into_markets(outcomes: list[PlayerPropOutcome]) -> dict[str, PlayerPropMarket]:
    """
    Group individual prop outcomes into complete markets.

    Markets are grouped by: player_name + prop_type + line
    This allows proper Over/Under pairing and cross-book comparison.

    Returns:
        Dictionary mapping market_key to PlayerPropMarket
    """
    markets: dict[str, PlayerPropMarket] = {}

    for outcome in outcomes:
        market_key = f"{outcome.player_name}|{outcome.prop_type}|{outcome.line}"

        if market_key not in markets:
            markets[market_key] = PlayerPropMarket(
                player_name=outcome.player_name,
                prop_type=outcome.prop_type,
                line=outcome.line,
            )

        markets[market_key].outcomes.append(outcome)

    return markets


def organize_props_by_player(markets: dict[str, PlayerPropMarket]) -> dict[str, list[PlayerPropMarket]]:
    """
    Organize markets by player name for cleaner display.

    Returns:
        Dictionary mapping player_name to list of their prop markets
    """
    by_player: dict[str, list[PlayerPropMarket]] = {}

    for market in markets.values():
        if market.player_name not in by_player:
            by_player[market.player_name] = []
        by_player[market.player_name].append(market)

    # Sort each player's markets by prop type, then by line
    for player_markets in by_player.values():
        player_markets.sort(key=lambda m: (m.prop_type, m.line))

    return by_player


def organize_props_by_line(markets: dict[str, PlayerPropMarket]) -> dict[float, list[PlayerPropMarket]]:
    """
    Organize markets by line value.

    This groups all props with the same numerical line together,
    which is useful for finding related props across players.

    Returns:
        Dictionary mapping line value to list of markets with that line
    """
    by_line: dict[float, list[PlayerPropMarket]] = {}

    for market in markets.values():
        if market.line not in by_line:
            by_line[market.line] = []
        by_line[market.line].append(market)

    # Sort each line's markets by player name
    for line_markets in by_line.values():
        line_markets.sort(key=lambda m: m.player_name)

    return by_line


# =============================================================================
# +EV DETECTION FOR PROPS
# =============================================================================

def get_sharp_prop_outcomes(
    market: PlayerPropMarket,
    sharp_book: str,
) -> tuple[Optional[PlayerPropOutcome], Optional[PlayerPropOutcome]]:
    """
    Get the sharp book's Over and Under for a prop market.

    Returns:
        Tuple of (sharp_over, sharp_under), either may be None
    """
    sharp_over = None
    sharp_under = None

    for outcome in market.outcomes:
        if outcome.bookmaker == sharp_book:
            if outcome.outcome == "Over":
                sharp_over = outcome
            elif outcome.outcome == "Under":
                sharp_under = outcome

    return sharp_over, sharp_under


def calculate_weighted_sharp_prop_price(
    market: PlayerPropMarket,
    primary_sharp: str,
    additional_sharp_books: list[str],
    pinnacle_weight: float = 0.60,
    other_sharps_weight: float = 0.40,
    devig_method: DevigMethod = "power",
) -> Optional[dict]:
    """
    Calculate weighted sharp price for a prop market (Over/Under pair).
    
    Args:
        market: PlayerPropMarket with all outcomes
        primary_sharp: Primary sharp book (typically "pinnacle")
        additional_sharp_books: List of additional sharp books
        pinnacle_weight: Weight for primary sharp (default 0.60)
        other_sharps_weight: Combined weight for other sharps (default 0.40)
        devig_method: Method to remove vig
    
    Returns:
        Dictionary with weighted fair probabilities for Over/Under, or None
    """
    # Get primary sharp Over/Under
    primary_over, primary_under = get_sharp_prop_outcomes(market, primary_sharp)
    if not primary_over or not primary_under:
        return None
    
    # Calculate fair probabilities from primary sharp
    primary_decimals = [primary_over.odds_decimal, primary_under.odds_decimal]
    primary_fair_probs = calculate_no_vig_probability(primary_decimals, method=devig_method)
    primary_fair_over, primary_fair_under = primary_fair_probs[0], primary_fair_probs[1]
    
    # Collect fair probabilities from additional sharp books
    other_sharp_overs = []
    other_sharp_unders = []
    available_sharp_books = []
    
    for alt_sharp in additional_sharp_books:
        alt_over, alt_under = get_sharp_prop_outcomes(market, alt_sharp)
        if not alt_over or not alt_under:
            continue
        
        # Calculate fair probabilities from alternative sharp
        alt_decimals = [alt_over.odds_decimal, alt_under.odds_decimal]
        alt_fair_probs = calculate_no_vig_probability(alt_decimals, method=devig_method)
        other_sharp_overs.append(alt_fair_probs[0])
        other_sharp_unders.append(alt_fair_probs[1])
        available_sharp_books.append(alt_sharp)
    
    # Calculate weighted fair probabilities
    if other_sharp_overs and other_sharp_unders:
        avg_other_fair_over = sum(other_sharp_overs) / len(other_sharp_overs)
        avg_other_fair_under = sum(other_sharp_unders) / len(other_sharp_unders)
        weighted_fair_over = (pinnacle_weight * primary_fair_over) + (other_sharps_weight * avg_other_fair_over)
        weighted_fair_under = (pinnacle_weight * primary_fair_under) + (other_sharps_weight * avg_other_fair_under)
    else:
        # Only primary sharp available
        weighted_fair_over = primary_fair_over
        weighted_fair_under = primary_fair_under
    
    return {
        "weighted_fair_over": weighted_fair_over,
        "weighted_fair_under": weighted_fair_under,
        "primary_fair_over": primary_fair_over,
        "primary_fair_under": primary_fair_under,
        "primary_over_odds_american": primary_over.odds_american,
        "primary_over_odds_decimal": primary_over.odds_decimal,
        "primary_under_odds_american": primary_under.odds_american,
        "primary_under_odds_decimal": primary_under.odds_decimal,
        "available_sharp_books": [primary_sharp] + available_sharp_books,
        "num_sharp_books": 1 + len(available_sharp_books),
    }


def detect_prop_ev_opportunities(
    event: dict,
    prop_markets: list[str],
    sharp_book: str,
    soft_books: list[str],
    min_ev_percent: float = 3.0,
    max_ev_percent: float = 25.0,
    devig_method: DevigMethod = "power",
    max_width_cents: Optional[int] = None,
    kelly_by_width: Optional[list[tuple[int, float]]] = None,
    default_kelly: float = 0.10,
    additional_sharp_books: Optional[list[str]] = None,
    use_weighted_consensus: bool = False,
    pinnacle_weight: float = 0.60,
    other_sharps_weight: float = 0.40,
    weighted_sharp_min_ev: float = 3.0,
    min_hours_before_game: Optional[float] = None,
    max_hours_before_game: Optional[float] = None,
) -> list[PropEVOpportunity]:
    """
    Detect +EV opportunities in player props for an event.

    This works similarly to game line EV detection, but is prop-specific:
    1. Parse all prop outcomes from the event
    2. Group outcomes into markets (player + prop_type + line)
    3. For each market with sharp book data, calculate fair probabilities
    4. Compare soft book odds against fair probabilities
    5. Return opportunities meeting EV thresholds

    Args:
        event: Event data with bookmakers and prop markets
        prop_markets: List of prop market keys to check
        sharp_book: Sharp bookmaker for reference (e.g., "pinnacle")
        soft_books: List of soft bookmakers to check for +EV
        min_ev_percent: Minimum EV to report
        max_ev_percent: Maximum EV (filter likely errors)
        devig_method: Method to remove vig from sharp lines
        max_width_cents: Maximum sharp market width to accept
        kelly_by_width: List of (max_width, kelly_fraction) for sizing
        default_kelly: Default Kelly fraction
        additional_sharp_books: List of additional sharp books for weighted consensus
        use_weighted_consensus: If True, use weighted consensus from multiple sharp books
        pinnacle_weight: Weight for primary sharp book (default 0.60)
        other_sharps_weight: Combined weight for other sharp books (default 0.40)
        weighted_sharp_min_ev: Minimum EV% required for weighted sharp price (default 3.0)
        min_hours_before_game: Minimum hours before game start (filter out games starting too soon)
        max_hours_before_game: Maximum hours before game start (filter out games starting > X hours)

    Returns:
        List of PropEVOpportunity objects
    """
    opportunities = []

    # Event info
    event_id = event.get("id", "")
    sport = event.get("sport_key", "")
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    commence_time = event.get("commence_time", "")

    # Skip games that have already started
    if not is_game_upcoming(commence_time):
        logger.debug(f"Skipping {event_id}: game has already started")
        return []
    
    # Time-window filter (Golden Hour: 1-4 hours before game start)
    if min_hours_before_game is not None or max_hours_before_game is not None:
        hours_until = get_hours_until_game(commence_time)
        if hours_until is None:
            logger.debug(f"Skipping {event_id}: could not parse commence_time")
            return []
        
        if min_hours_before_game is not None and hours_until < min_hours_before_game:
            logger.debug(f"Skipping {event_id}: game starts in {hours_until:.1f}h (too soon, min: {min_hours_before_game}h)")
            return []
        
        if max_hours_before_game is not None and hours_until > max_hours_before_game:
            logger.debug(f"Skipping {event_id}: game starts in {hours_until:.1f}h (too far, max: {max_hours_before_game}h)")
            return []

    # Parse all prop outcomes
    all_outcomes = parse_prop_outcomes_from_event(
        event=event,
        prop_markets=prop_markets,
        include_bookmakers=soft_books + [sharp_book],
        exclude_bookmakers=[],
    )

    if not all_outcomes:
        return []

    # Group into markets
    markets = group_outcomes_into_markets(all_outcomes)

    # Process each market
    for market_key, market in markets.items():
        # Get sharp book's Over/Under
        sharp_over, sharp_under = get_sharp_prop_outcomes(market, sharp_book)

        # Need both Over and Under from sharp book to devig
        if not sharp_over or not sharp_under:
            logger.debug(f"Skipping {market_key}: missing sharp book data")
            continue

        # Calculate market width
        sharp_outcomes_for_width = [
            {"decimal": sharp_over.odds_decimal},
            {"decimal": sharp_under.odds_decimal},
        ]
        width_cents = calculate_sharp_width_cents(sharp_outcomes_for_width)

        # Skip if market is too wide
        if max_width_cents is not None and width_cents is not None:
            if width_cents > max_width_cents:
                logger.debug(f"Skipping {market_key}: width {width_cents} > max {max_width_cents}")
                continue

        # Determine Kelly fraction based on width
        if kelly_by_width and width_cents is not None:
            kelly_fraction = get_kelly_fraction_for_width(width_cents, kelly_by_width, default_kelly)
        else:
            kelly_fraction = default_kelly

        # Calculate fair probabilities - use weighted consensus if enabled
        if use_weighted_consensus and additional_sharp_books:
            weighted_data = calculate_weighted_sharp_prop_price(
                market=market,
                primary_sharp=sharp_book,
                additional_sharp_books=additional_sharp_books,
                pinnacle_weight=pinnacle_weight,
                other_sharps_weight=other_sharps_weight,
                devig_method=devig_method,
            )
            
            if weighted_data:
                fair_prob_over = weighted_data["weighted_fair_over"]
                fair_prob_under = weighted_data["weighted_fair_under"]
                sharp_over_odds_american = weighted_data["primary_over_odds_american"]
                sharp_over_odds_decimal = weighted_data["primary_over_odds_decimal"]
                sharp_under_odds_american = weighted_data["primary_under_odds_american"]
                sharp_under_odds_decimal = weighted_data["primary_under_odds_decimal"]
            else:
                # Fallback to primary sharp only
                sharp_decimals = [sharp_over.odds_decimal, sharp_under.odds_decimal]
                fair_probs = calculate_no_vig_probability(sharp_decimals, method=devig_method)
                fair_prob_over, fair_prob_under = fair_probs[0], fair_probs[1]
                sharp_over_odds_american = sharp_over.odds_american
                sharp_over_odds_decimal = sharp_over.odds_decimal
                sharp_under_odds_american = sharp_under.odds_american
                sharp_under_odds_decimal = sharp_under.odds_decimal
        else:
            # Use primary sharp book only (original logic)
            sharp_decimals = [sharp_over.odds_decimal, sharp_under.odds_decimal]
            fair_probs = calculate_no_vig_probability(sharp_decimals, method=devig_method)
            fair_prob_over, fair_prob_under = fair_probs[0], fair_probs[1]
            sharp_over_odds_american = sharp_over.odds_american
            sharp_over_odds_decimal = sharp_over.odds_decimal
            sharp_under_odds_american = sharp_under.odds_american
            sharp_under_odds_decimal = sharp_under.odds_decimal

        # Check each soft book outcome
        for outcome in market.outcomes:
            if outcome.bookmaker == sharp_book:
                continue
            if outcome.bookmaker not in soft_books:
                continue

            # Get fair probability for this outcome type
            if outcome.outcome == "Over":
                fair_prob = fair_prob_over
                sharp_outcome_american = sharp_over_odds_american
                sharp_outcome_decimal = sharp_over_odds_decimal
            else:
                fair_prob = fair_prob_under
                sharp_outcome_american = sharp_under_odds_american
                sharp_outcome_decimal = sharp_under_odds_decimal

            # Calculate EV
            ev = calculate_ev(fair_prob, outcome.odds_decimal)

            # For weighted consensus, require minimum EV threshold
            if use_weighted_consensus and ev < weighted_sharp_min_ev:
                logger.debug(
                    f"Skipping {market_key}/{outcome.outcome} @ {outcome.bookmaker}: "
                    f"EV {ev:.2f}% < weighted sharp min {weighted_sharp_min_ev}%"
                )
                continue

            # Calculate edge
            soft_implied = 1 / outcome.odds_decimal
            edge = (fair_prob - soft_implied) * 100

            # Calculate Kelly units
            units = calculate_kelly_units(fair_prob, outcome.odds_decimal, kelly_fraction)

            if min_ev_percent <= ev <= max_ev_percent:
                opp = PropEVOpportunity(
                    event_id=event_id,
                    sport=sport,
                    home_team=home_team,
                    away_team=away_team,
                    commence_time=commence_time,
                    player_name=outcome.player_name,
                    prop_type=outcome.prop_type,
                    line=outcome.line,
                    outcome=outcome.outcome,
                    sharp_book=sharp_book,
                    sharp_odds_american=sharp_outcome_american,
                    sharp_odds_decimal=sharp_outcome_decimal,
                    sharp_implied_prob=fair_prob * 100,
                    soft_book=outcome.bookmaker,
                    soft_odds_american=outcome.odds_american,
                    soft_odds_decimal=outcome.odds_decimal,
                    soft_implied_prob=soft_implied * 100,
                    ev_percent=ev,
                    edge_percent=edge,
                    sharp_width_cents=width_cents,
                    kelly_fraction=kelly_fraction,
                    units=units,
                )
                opportunities.append(opp)
                logger.debug(f"Found prop +EV: {opp}")

    # Sort by EV percentage (highest first)
    opportunities.sort(key=lambda x: x.ev_percent, reverse=True)

    return opportunities


# =============================================================================
# ARBITRAGE DETECTION FOR PROPS
# =============================================================================

def find_prop_arbitrage(
    market: PlayerPropMarket,
    total_stake: float = 100.0,
    min_profit: float = 2.0,
    max_profit: float = 20.0,
) -> Optional[tuple[float, PlayerPropOutcome, PlayerPropOutcome]]:
    """
    Find arbitrage opportunity in a single prop market.

    For props, arbitrage requires finding the best Over at one book
    and best Under at another book where implied probs sum < 100%.

    Args:
        market: PlayerPropMarket with all bookmaker outcomes
        total_stake: Total amount to stake
        min_profit: Minimum profit percentage
        max_profit: Maximum profit percentage (filter errors)

    Returns:
        Tuple of (profit_percent, best_over, best_under) or None if no arb
    """
    best_over = market.get_best_over()
    best_under = market.get_best_under()

    if not best_over or not best_under:
        return None

    # Calculate implied probabilities
    prob_over = 1 / best_over.odds_decimal
    prob_under = 1 / best_under.odds_decimal
    total_prob = prob_over + prob_under

    # Arbitrage exists if total implied probability < 1
    if total_prob >= 1:
        return None

    # Calculate profit percentage
    profit_percent = ((1 / total_prob) - 1) * 100

    if min_profit <= profit_percent <= max_profit:
        return (profit_percent, best_over, best_under)

    return None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def filter_events_for_props(
    events: list[dict],
    hours_before_game: int = 24,
) -> list[dict]:
    """
    Filter events to only those suitable for prop fetching.

    Props are typically only available close to game time,
    so we filter to games starting within the specified hours.

    Args:
        events: List of event dicts
        hours_before_game: Only include games starting within this many hours

    Returns:
        Filtered list of events
    """
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_before_game)

    filtered = []
    for event in events:
        commence_time = event.get("commence_time", "")
        try:
            if commence_time.endswith("Z"):
                event_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            else:
                event_time = datetime.fromisoformat(commence_time)

            # Include if game is in the future but within our window
            if now < event_time <= cutoff:
                filtered.append(event)
        except (ValueError, AttributeError):
            continue

    return filtered


def get_prop_markets_for_sport(sport: str, prop_markets_config: dict) -> list[str]:
    """
    Get the list of prop market keys for a specific sport.

    Args:
        sport: Sport key (e.g., "basketball_nba")
        prop_markets_config: Dictionary mapping sports to their prop markets

    Returns:
        List of prop market keys for the sport, or empty list if not configured
    """
    return prop_markets_config.get(sport, [])


def format_prop_opportunity(opp: PropEVOpportunity) -> str:
    """Format a prop EV opportunity for console display."""
    prop_name = get_prop_display_name(opp.prop_type)
    # Convert fair probability back to fair odds
    fair_decimal = 1 / (opp.sharp_implied_prob / 100)
    fair_american = decimal_to_american(fair_decimal)

    lines = [
        f"\n{'=' * 60}",
        f"+EV PROP: {opp.ev_percent:.2f}% EV | {opp.units:.2f} units",
        f"{'=' * 60}",
        f"  {opp.away_team} @ {opp.home_team}",
        f"  Player: {opp.player_name}",
        f"  Prop: {prop_name} {opp.outcome} {opp.line}",
        f"  Book: {opp.soft_book.upper()} @ {opp.soft_odds_american:+d}",
        f"  Fair Odds: {fair_american:+d} ({opp.sharp_implied_prob:.1f}%)",
    ]

    if opp.sharp_width_cents is not None:
        lines.append(f"  Width: {opp.sharp_width_cents} cents | Kelly: {opp.kelly_fraction:.0%}")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)
