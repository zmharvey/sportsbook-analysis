"""
Arbitrage detection algorithms for sports betting.
Supports moneyline (h2h), spreads, and totals markets.
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class BetLeg:
    """Represents one leg of an arbitrage bet."""
    bookmaker: str
    outcome: str
    odds_american: int
    odds_decimal: float
    point: Optional[float]  # For spreads/totals
    stake: float
    payout: float


@dataclass
class ArbitrageOpportunity:
    """Represents a complete arbitrage opportunity."""
    sport: str
    event_id: str
    home_team: str
    away_team: str
    commence_time: str
    market: str
    profit_percent: float
    total_stake: float
    guaranteed_profit: float
    legs: list[BetLeg]

    def __str__(self) -> str:
        return (
            f"{self.home_team} vs {self.away_team} | "
            f"{self.market.upper()} | "
            f"Profit: {self.profit_percent:.2f}%"
        )


@dataclass
class DebugInfo:
    """Debug information for a market analysis."""
    sport: str
    event_id: str
    home_team: str
    away_team: str
    commence_time: str
    market: str
    point: Optional[float]
    total_implied_prob: float  # As percentage (e.g., 101.5 means 101.5%)
    margin_from_arb: float  # How far from arbitrage (negative = arb exists)
    best_odds: list[tuple[str, str, int, float]]  # (bookmaker, outcome, american, decimal)
    num_bookmakers: int


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))


def calculate_implied_probability(decimal_odds: float) -> float:
    """Calculate implied probability from decimal odds."""
    return 1 / decimal_odds


def find_two_way_arbitrage(
    outcomes: list[dict],
    total_stake: float = 100.0,
) -> Optional[tuple[float, list[tuple[str, str, float, float, float, Optional[float]]]]]:
    """
    Find arbitrage opportunity in a two-way market.

    Args:
        outcomes: List of dicts with bookmaker, outcome name, decimal odds, and optional point
        total_stake: Total amount to stake across all bets

    Returns:
        Tuple of (profit_percent, list of (bookmaker, outcome, decimal_odds, stake, payout, point))
        or None if no arbitrage exists
    """
    if len(outcomes) < 2:
        return None

    # Group outcomes by their name (and point for spreads/totals)
    outcome_groups = {}
    for o in outcomes:
        # Create a key that includes the point for spreads/totals
        key = (o["outcome"], o.get("point"))
        if key not in outcome_groups:
            outcome_groups[key] = []
        outcome_groups[key].append(o)

    # For two-way markets, we need exactly 2 distinct outcomes
    if len(outcome_groups) != 2:
        return None

    # Find best odds for each outcome
    best_odds = {}
    for key, options in outcome_groups.items():
        best = max(options, key=lambda x: x["decimal_odds"])
        best_odds[key] = best

    keys = list(best_odds.keys())
    odds1 = best_odds[keys[0]]
    odds2 = best_odds[keys[1]]

    # Calculate implied probabilities
    prob1 = calculate_implied_probability(odds1["decimal_odds"])
    prob2 = calculate_implied_probability(odds2["decimal_odds"])
    total_prob = prob1 + prob2

    # Arbitrage exists if total implied probability < 1
    if total_prob >= 1:
        return None

    # Calculate profit percentage
    profit_percent = ((1 / total_prob) - 1) * 100

    # Calculate optimal stakes
    stake1 = (total_stake * prob1) / total_prob
    stake2 = (total_stake * prob2) / total_prob

    payout1 = stake1 * odds1["decimal_odds"]
    payout2 = stake2 * odds2["decimal_odds"]

    result = [
        (
            odds1["bookmaker"],
            odds1["outcome"],
            odds1["decimal_odds"],
            stake1,
            payout1,
            odds1.get("point"),
        ),
        (
            odds2["bookmaker"],
            odds2["outcome"],
            odds2["decimal_odds"],
            stake2,
            payout2,
            odds2.get("point"),
        ),
    ]

    return (profit_percent, result)


def find_three_way_arbitrage(
    outcomes: list[dict],
    total_stake: float = 100.0,
) -> Optional[tuple[float, list[tuple[str, str, float, float, float, Optional[float]]]]]:
    """
    Find arbitrage opportunity in a three-way market (e.g., soccer with draw).

    Args:
        outcomes: List of dicts with bookmaker, outcome name, and decimal odds
        total_stake: Total amount to stake across all bets

    Returns:
        Tuple of (profit_percent, list of (bookmaker, outcome, decimal_odds, stake, payout, point))
        or None if no arbitrage exists
    """
    if len(outcomes) < 3:
        return None

    # Group outcomes by name
    outcome_groups = {}
    for o in outcomes:
        name = o["outcome"]
        if name not in outcome_groups:
            outcome_groups[name] = []
        outcome_groups[name].append(o)

    # Need exactly 3 outcomes (home, away, draw)
    if len(outcome_groups) != 3:
        return None

    # Find best odds for each outcome
    best_odds = {}
    for name, options in outcome_groups.items():
        best = max(options, key=lambda x: x["decimal_odds"])
        best_odds[name] = best

    # Calculate implied probabilities
    probs = {
        name: calculate_implied_probability(o["decimal_odds"])
        for name, o in best_odds.items()
    }
    total_prob = sum(probs.values())

    # Arbitrage exists if total implied probability < 1
    if total_prob >= 1:
        return None

    profit_percent = ((1 / total_prob) - 1) * 100

    result = []
    for name, odds_info in best_odds.items():
        stake = (total_stake * probs[name]) / total_prob
        payout = stake * odds_info["decimal_odds"]
        result.append((
            odds_info["bookmaker"],
            name,
            odds_info["decimal_odds"],
            stake,
            payout,
            odds_info.get("point"),
        ))

    return (profit_percent, result)


def extract_outcomes_from_event(
    event: dict,
    market_key: str,
    include_bookmakers: list[str],
    exclude_bookmakers: list[str],
    odds_format: str = "american",
) -> list[dict]:
    """
    Extract all outcomes for a specific market from an event.

    Args:
        event: Event data from the API
        market_key: Market to extract (h2h, spreads, totals)
        include_bookmakers: Only include these bookmakers (empty = all)
        exclude_bookmakers: Exclude these bookmakers
        odds_format: Format of odds in the data

    Returns:
        List of outcome dictionaries
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
            if market["key"] != market_key:
                continue

            for outcome in market.get("outcomes", []):
                price = outcome["price"]

                # Convert to decimal if needed
                if odds_format == "american":
                    decimal_odds = american_to_decimal(int(price))
                    american_odds = int(price)
                else:
                    decimal_odds = float(price)
                    american_odds = decimal_to_american(decimal_odds)

                outcomes.append({
                    "bookmaker": book_key,
                    "bookmaker_title": bookmaker["title"],
                    "outcome": outcome["name"],
                    "decimal_odds": decimal_odds,
                    "american_odds": american_odds,
                    "point": outcome.get("point"),
                })

    return outcomes


def analyze_market_debug(
    outcomes: list[dict],
    event: dict,
    sport: str,
    market_key: str,
    point: Optional[float] = None,
) -> Optional[DebugInfo]:
    """
    Analyze a market and return debug information.

    Returns DebugInfo with implied probabilities and best odds for each outcome.
    """
    if len(outcomes) < 2:
        return None

    # Group outcomes by name (and point for spreads/totals)
    outcome_groups = {}
    for o in outcomes:
        key = o["outcome"]
        if key not in outcome_groups:
            outcome_groups[key] = []
        outcome_groups[key].append(o)

    if len(outcome_groups) < 2:
        return None

    # Find best odds for each outcome
    best_odds = []
    total_implied_prob = 0.0

    for outcome_name, options in outcome_groups.items():
        best = max(options, key=lambda x: x["decimal_odds"])
        prob = calculate_implied_probability(best["decimal_odds"])
        total_implied_prob += prob
        best_odds.append((
            best["bookmaker"],
            outcome_name,
            best["american_odds"],
            best["decimal_odds"],
        ))

    # Count unique bookmakers
    unique_books = set(o["bookmaker"] for o in outcomes)

    return DebugInfo(
        sport=sport,
        event_id=event["id"],
        home_team=event["home_team"],
        away_team=event["away_team"],
        commence_time=event["commence_time"],
        market=market_key,
        point=point,
        total_implied_prob=total_implied_prob * 100,  # Convert to percentage
        margin_from_arb=(total_implied_prob - 1) * 100,  # Positive = no arb, negative = arb
        best_odds=best_odds,
        num_bookmakers=len(unique_books),
    )


def is_game_upcoming(commence_time: str) -> bool:
    """Check if a game hasn't started yet based on commence_time."""
    from datetime import datetime, timezone

    try:
        # Parse ISO format time
        if commence_time.endswith("Z"):
            event_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        else:
            event_time = datetime.fromisoformat(commence_time)

        now = datetime.now(timezone.utc)
        return event_time > now
    except (ValueError, AttributeError):
        # If we can't parse, include it to be safe
        return True


def find_arbitrage_opportunities(
    events: list[dict],
    sport: str,
    markets: list[str],
    include_bookmakers: list[str],
    exclude_bookmakers: list[str],
    min_profit: float,
    max_profit: float,
    total_stake: float,
    odds_format: str = "american",
    debug_mode: bool = False,
    near_miss_threshold: float = 2.0,
    upcoming_only: bool = True,
) -> tuple[list[ArbitrageOpportunity], list[DebugInfo]]:
    """
    Find all arbitrage opportunities across events.

    Args:
        events: List of events from the API
        sport: Sport key
        markets: List of markets to scan
        include_bookmakers: Only include these bookmakers
        exclude_bookmakers: Exclude these bookmakers
        min_profit: Minimum profit percentage
        max_profit: Maximum profit percentage
        total_stake: Total stake amount
        odds_format: Odds format from API
        debug_mode: If True, collect debug info for near-misses
        near_miss_threshold: Show markets within this % of being an arb
        upcoming_only: If True, only include games that haven't started

    Returns:
        Tuple of (list of ArbitrageOpportunity, list of DebugInfo)
    """
    opportunities = []
    debug_infos = []

    for event in events:
        # Skip live/in-progress games if upcoming_only is enabled
        if upcoming_only and not is_game_upcoming(event.get("commence_time", "")):
            continue
        for market_key in markets:
            outcomes = extract_outcomes_from_event(
                event,
                market_key,
                include_bookmakers,
                exclude_bookmakers,
                odds_format,
            )

            if not outcomes:
                continue

            # For spreads and totals, group by point value and find arb for each
            if market_key in ["spreads", "totals"]:
                # Group by point value
                point_groups = {}
                for o in outcomes:
                    point = o.get("point")
                    if point not in point_groups:
                        point_groups[point] = []
                    point_groups[point].append(o)

                # Check each point value group
                for point, group_outcomes in point_groups.items():
                    # Collect debug info if enabled
                    if debug_mode:
                        debug_info = analyze_market_debug(
                            group_outcomes, event, sport, market_key, point
                        )
                        if debug_info and debug_info.margin_from_arb <= near_miss_threshold:
                            debug_infos.append(debug_info)

                    result = find_two_way_arbitrage(group_outcomes, total_stake)
                    if result:
                        profit_percent, legs_data = result

                        if min_profit <= profit_percent <= max_profit:
                            legs = [
                                BetLeg(
                                    bookmaker=leg[0],
                                    outcome=leg[1],
                                    odds_american=decimal_to_american(leg[2]),
                                    odds_decimal=leg[2],
                                    point=leg[5],
                                    stake=leg[3],
                                    payout=leg[4],
                                )
                                for leg in legs_data
                            ]

                            opportunities.append(ArbitrageOpportunity(
                                sport=sport,
                                event_id=event["id"],
                                home_team=event["home_team"],
                                away_team=event["away_team"],
                                commence_time=event["commence_time"],
                                market=market_key,
                                profit_percent=profit_percent,
                                total_stake=total_stake,
                                guaranteed_profit=profit_percent * total_stake / 100,
                                legs=legs,
                            ))
            else:
                # H2H market - check for 2-way or 3-way
                unique_outcomes = set(o["outcome"] for o in outcomes)

                # Collect debug info if enabled
                if debug_mode:
                    debug_info = analyze_market_debug(
                        outcomes, event, sport, market_key
                    )
                    if debug_info and debug_info.margin_from_arb <= near_miss_threshold:
                        debug_infos.append(debug_info)

                if len(unique_outcomes) == 2:
                    result = find_two_way_arbitrage(outcomes, total_stake)
                elif len(unique_outcomes) == 3:
                    result = find_three_way_arbitrage(outcomes, total_stake)
                else:
                    continue

                if result:
                    profit_percent, legs_data = result

                    if min_profit <= profit_percent <= max_profit:
                        legs = [
                            BetLeg(
                                bookmaker=leg[0],
                                outcome=leg[1],
                                odds_american=decimal_to_american(leg[2]),
                                odds_decimal=leg[2],
                                point=leg[5],
                                stake=leg[3],
                                payout=leg[4],
                            )
                            for leg in legs_data
                        ]

                        opportunities.append(ArbitrageOpportunity(
                            sport=sport,
                            event_id=event["id"],
                            home_team=event["home_team"],
                            away_team=event["away_team"],
                            commence_time=event["commence_time"],
                            market=market_key,
                            profit_percent=profit_percent,
                            total_stake=total_stake,
                            guaranteed_profit=profit_percent * total_stake / 100,
                            legs=legs,
                        ))

    return opportunities, debug_infos
