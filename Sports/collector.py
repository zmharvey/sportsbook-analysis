"""
Data collector service.
Polls The Odds API at regular intervals and stores data to SQLite.
Runs as a separate background process.
"""

import logging
import time
import sys
from datetime import datetime, timezone
from typing import Optional

import config
from odds_api import OddsAPIClient
from database import (
    init_database,
    upsert_event,
    bulk_insert_odds,
    bulk_insert_prop_snapshots,
    cleanup_old_data,
    cleanup_old_prop_data,
    get_database_stats,
    process_closing_lines,
    process_prop_closing_lines,
)
from arbitrage import american_to_decimal, decimal_to_american
from ev_detector import detect_ev_opportunities, EVOpportunity
from alerts import (
    send_discord_alert,
    format_arb_alert,
    send_ev_alert_and_log_play,
    send_arb_alert_and_log_play,
    send_prop_ev_alert_and_log_play,
)
from arbitrage import find_arbitrage_opportunities
from player_props import (
    PropEVOpportunity,
    detect_prop_ev_opportunities,
    filter_events_for_props,
    get_prop_markets_for_sport,
    get_prop_display_name,
    format_prop_opportunity,
    parse_prop_outcomes_from_event,
)

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the collector."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def process_api_response(events: list[dict], sport: str, timestamp: str) -> list[dict]:
    """
    Process API response and prepare data for database insertion.

    Returns list of odds snapshot dicts ready for bulk insert.
    """
    snapshots = []

    for event in events:
        # Upsert event
        upsert_event(
            event_id=event["id"],
            sport=sport,
            home_team=event["home_team"],
            away_team=event["away_team"],
            commence_time=event["commence_time"],
        )

        # Process each bookmaker's odds
        for bookmaker in event.get("bookmakers", []):
            book_key = bookmaker["key"]

            for market in bookmaker.get("markets", []):
                market_key = market["key"]

                for outcome in market.get("outcomes", []):
                    price = outcome["price"]
                    point = outcome.get("point")

                    # Handle American vs Decimal format
                    if config.ODDS_FORMAT == "american":
                        price_american = int(price)
                        price_decimal = american_to_decimal(price_american)
                    else:
                        price_decimal = float(price)
                        price_american = decimal_to_american(price_decimal)

                    snapshots.append({
                        "event_id": event["id"],
                        "bookmaker": book_key,
                        "market": market_key,
                        "outcome": outcome["name"],
                        "price_american": price_american,
                        "price_decimal": price_decimal,
                        "point": point,
                        "timestamp": timestamp,
                    })

    return snapshots


def run_collection_cycle(
    client: OddsAPIClient,
    sports: list[str],
    us_regions: list[str],
    us_bookmakers: list[str],
    markets: list[str],
    sharp_book: str = "pinnacle",
    additional_sharp_books: Optional[list[str]] = None,
    dfs_regions: Optional[list[str]] = None,
    dfs_bookmakers: Optional[list[str]] = None,
) -> tuple[int, list[dict]]:
    """
    Run a single collection cycle.

    Returns tuple of (total_snapshots_collected, all_events_raw)
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    total_snapshots = 0
    all_events = []
    
    # Combine all sharp books for fetching
    all_sharp_books = [sharp_book] + (additional_sharp_books or [])

    for sport in sports:
        try:
            # Fetch US bookmakers
            logger.info(f"Fetching {sport} from US regions...")
            us_events = client.get_odds(
                sport=sport,
                regions=us_regions,
                markets=markets,
                odds_format=config.ODDS_FORMAT,
                bookmakers=us_bookmakers if us_bookmakers else None,
            )

            if us_events:
                snapshots = process_api_response(us_events, sport, timestamp)
                bulk_insert_odds(snapshots)
                total_snapshots += len(snapshots)
                all_events.extend(us_events)
                logger.info(f"  US: {len(us_events)} events, {len(snapshots)} snapshots")

            # Fetch DFS (Daily Fantasy Sports) bookmakers
            if dfs_regions and dfs_bookmakers:
                logger.info(f"Fetching {sport} from DFS region...")
                dfs_events = client.get_odds(
                    sport=sport,
                    regions=dfs_regions,
                    markets=markets,
                    odds_format=config.ODDS_FORMAT,
                    bookmakers=dfs_bookmakers,
                )

                if dfs_events:
                    snapshots = process_api_response(dfs_events, sport, timestamp)
                    bulk_insert_odds(snapshots)
                    total_snapshots += len(snapshots)
                    # Merge DFS odds into existing events
                    for dfs_event in dfs_events:
                        for existing_event in all_events:
                            if existing_event["id"] == dfs_event["id"]:
                                existing_event["bookmakers"].extend(dfs_event.get("bookmakers", []))
                                break
                        else:
                            all_events.append(dfs_event)
                    logger.info(f"  DFS: {len(dfs_events)} events, {len(snapshots)} snapshots")

            # Fetch sharp books (Pinnacle + additional sharps like BetOnline, lowvig, betanyports, betfair)
            # Fetch from multiple regions to ensure we get all sharp books:
            # - Pinnacle: "eu"
            # - BetOnline: "us2"
            # - lowvig, betanyports: "us", "us2", or "eu"
            # - betfair: "uk" or "eu"
            sharp_books_str = ", ".join(all_sharp_books)
            logger.info(f"Fetching {sport} sharp books ({sharp_books_str})...")
            sharp_events = client.get_odds(
                sport=sport,
                regions=["us", "uk", "eu", "us2"],  # Multiple regions to capture all sharp books
                markets=markets,
                odds_format=config.ODDS_FORMAT,
                bookmakers=all_sharp_books,
            )

            if sharp_events:
                snapshots = process_api_response(sharp_events, sport, timestamp)
                bulk_insert_odds(snapshots)
                total_snapshots += len(snapshots)
                # Merge sharp odds into existing events
                for sharp_event in sharp_events:
                    # Find matching US event and add sharp odds
                    for us_event in all_events:
                        if us_event["id"] == sharp_event["id"]:
                            us_event["bookmakers"].extend(sharp_event.get("bookmakers", []))
                            break
                    else:
                        # Event only has sharp odds
                        all_events.append(sharp_event)
                logger.info(f"  Sharp: {len(sharp_events)} events, {len(snapshots)} snapshots")

        except Exception as e:
            logger.error(f"Error fetching {sport}: {e}")

    return total_snapshots, all_events


def format_market_name(market: str) -> str:
    """Format market key for display."""
    # Check if it's a player prop market
    if market.startswith("player_") or market.startswith("batter_") or market.startswith("pitcher_"):
        return get_prop_display_name(market)
    return {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}.get(market, market.upper())


def get_current_poll_interval() -> int:
    """
    Get the appropriate poll interval based on current time.
    
    Peak hours (2pm-10pm): poll every 5 minutes
    Off-peak: poll every 30 minutes
    """
    current_hour = datetime.now().hour
    
    peak_start = getattr(config, "PEAK_HOURS_START", 14)
    peak_end = getattr(config, "PEAK_HOURS_END", 22)
    peak_interval = getattr(config, "POLL_INTERVAL_PEAK", 300)
    offpeak_interval = getattr(config, "POLL_INTERVAL_OFFPEAK", 1800)
    
    is_peak = peak_start <= current_hour < peak_end
    return peak_interval if is_peak else offpeak_interval


def print_arb_opportunity(opp) -> None:
    """Print an arbitrage opportunity to console."""
    print("\n" + "=" * 60)
    print(f"ARBITRAGE FOUND - {opp.profit_percent:.2f}% PROFIT")
    print("=" * 60)
    print(f"  {opp.away_team} @ {opp.home_team}")
    print(f"  Market: {format_market_name(opp.market)}")
    print(f"  Profit: ${opp.profit_percent * config.DEFAULT_STAKE / 100:.2f} on ${config.DEFAULT_STAKE:.2f}")
    print("\n  Legs:")
    for i, leg in enumerate(opp.legs, 1):
        point_str = f" ({leg.point:+.1f})" if leg.point is not None else ""
        print(f"    {i}. {leg.outcome}{point_str} @ {leg.bookmaker.upper()}: {leg.odds_american:+d}")
    print("=" * 60)


def print_ev_opportunity(opp) -> None:
    """Print a +EV opportunity to console."""
    point_str = f" ({opp.point:+.1f})" if opp.point is not None else ""
    # Convert fair probability back to fair odds
    fair_decimal = 1 / (opp.sharp_implied_prob / 100)
    fair_american = decimal_to_american(fair_decimal)
    print("\n" + "-" * 60)
    print(f"+EV OPPORTUNITY - {opp.ev_percent:.2f}% EV | {opp.units:.2f} units")
    print("-" * 60)
    print(f"  {opp.away_team} @ {opp.home_team}")
    print(f"  Market: {format_market_name(opp.market)}")
    print(f"  Bet: {opp.outcome}{point_str} @ {opp.soft_book.upper()}")
    print(f"  Odds: {opp.soft_odds_american:+d}")
    print(f"  Fair Odds: {fair_american:+d} ({opp.sharp_implied_prob:.1f}%)")
    if opp.sharp_width_cents is not None:
        print(f"  Width: {opp.sharp_width_cents} cents | Kelly: {opp.kelly_fraction:.0%}")
    print("-" * 60)


def print_prop_ev_opportunity(opp: PropEVOpportunity) -> None:
    """Print a player prop +EV opportunity to console."""
    prop_name = get_prop_display_name(opp.prop_type)
    # Convert fair probability back to fair odds
    fair_decimal = 1 / (opp.sharp_implied_prob / 100)
    fair_american = decimal_to_american(fair_decimal)
    print("\n" + "=" * 60)
    print(f"+EV PROP - {opp.ev_percent:.2f}% EV | {opp.units:.2f} units")
    print("=" * 60)
    print(f"  {opp.away_team} @ {opp.home_team}")
    print(f"  Player: {opp.player_name}")
    print(f"  Prop: {prop_name} {opp.outcome} {opp.line}")
    print(f"  Book: {opp.soft_book.upper()} @ {opp.soft_odds_american:+d}")
    print(f"  Fair Odds: {fair_american:+d} ({opp.sharp_implied_prob:.1f}%)")
    if opp.sharp_width_cents is not None:
        print(f"  Width: {opp.sharp_width_cents} cents | Kelly: {opp.kelly_fraction:.0%}")
    print("=" * 60)


def check_and_alert(
    all_events: list[dict],
    sports: list[str],
    markets: list[str],
    us_bookmakers: list[str],
    sharp_book: str,
) -> tuple[int, int]:
    """Check for arb and +EV opportunities, print to console and send alerts.

    Returns tuple of (arb_count, ev_count).
    """
    total_arbs = 0
    total_evs = 0

    # Check for arbitrage
    for sport in sports:
        sport_events = [e for e in all_events if e.get("sport_key") == sport]

        arb_opps, _ = find_arbitrage_opportunities(
            events=sport_events,
            sport=sport,
            markets=markets,
            include_bookmakers=us_bookmakers + [sharp_book],
            exclude_bookmakers=[],
            min_profit=config.MIN_PROFIT_PERCENT,
            max_profit=config.MAX_PROFIT_PERCENT,
            total_stake=config.DEFAULT_STAKE,
            odds_format=config.ODDS_FORMAT,
            debug_mode=False,
            upcoming_only=config.UPCOMING_ONLY,
        )

        for opp in arb_opps:
            total_arbs += 1
            print_arb_opportunity(opp)

            if config.DISCORD_WEBHOOK_URL and config.ALERT_ON_ARB:
                send_arb_alert_and_log_play(opp)

    # Check for +EV opportunities - exclude sharp books from soft books list
    additional_sharps = config.SHARP_BOOKMAKERS if hasattr(config, "SHARP_BOOKMAKERS") else []
    all_sharp_books = [sharp_book] + additional_sharps
    soft_books_only = [book for book in us_bookmakers if book not in all_sharp_books]

    logger.info(f"Sharp books: {all_sharp_books} | Soft books: {soft_books_only}")

    # Get additional sharp books for weighted consensus
    additional_sharp_books_for_consensus = getattr(config, "ADDITIONAL_SHARP_BOOKS", [])
    all_consensus_sharps = additional_sharps + additional_sharp_books_for_consensus
    
    ev_opps = detect_ev_opportunities(
        events=all_events,
        sharp_book=sharp_book,
        soft_books=soft_books_only,
        markets=markets,
        min_ev_percent=config.MIN_EV_PERCENT,
        max_ev_percent=config.MAX_EV_PERCENT,
        devig_method=config.DEVIG_METHOD,
        max_width_cents=config.MAX_SHARP_WIDTH_CENTS,
        kelly_by_width=config.KELLY_BY_WIDTH,
        default_kelly=config.DEFAULT_KELLY_FRACTION,
        additional_sharp_books=all_consensus_sharps,
        require_sharp_consensus=config.REQUIRE_SHARP_CONSENSUS,
        max_sharp_disagreement=config.MAX_SHARP_DISAGREEMENT_PERCENT,
        enable_market_consensus=config.ENABLE_MARKET_CONSENSUS_CHECK,
        max_market_deviation=config.MAX_MARKET_DEVIATION_PERCENT,
        min_books_for_consensus=config.MIN_BOOKS_FOR_CONSENSUS,
        use_weighted_consensus=True,  # Enable weighted consensus model
        pinnacle_weight=getattr(config, "PINNACLE_WEIGHT", 0.60),
        other_sharps_weight=getattr(config, "OTHER_SHARPS_WEIGHT", 0.40),
        weighted_sharp_min_ev=getattr(config, "WEIGHTED_SHARP_MIN_EV", 3.0),
        min_hours_before_game=getattr(config, "MIN_HOURS_BEFORE_GAME", 1),
        max_hours_before_game=getattr(config, "MAX_HOURS_BEFORE_GAME", 4),
    )

    for ev_opp in ev_opps:
        total_evs += 1
        print_ev_opportunity(ev_opp)

        if config.DISCORD_WEBHOOK_URL and config.ALERT_ON_EV:
            send_ev_alert_and_log_play(ev_opp)

    return total_arbs, total_evs


# =============================================================================
# PLAYER PROPS COLLECTION
# =============================================================================

def adjust_underdog_odds(over_api_odds: int, under_api_odds: int) -> tuple[int, int]:
    """
    Adjust Underdog Fantasy odds to reflect true single-pick value.
    
    Underdog's 3-leg (6x) and 5-leg (20x) parlays imply -122 per leg.
    The API may return different odds (like -137), so we adjust.
    
    For standard lines (Over == Under in API): Both become -122
    For non-standard lines (different payouts): Adjust relative to -122
    
    Args:
        over_api_odds: API's American odds for Over
        under_api_odds: API's American odds for Under
        
    Returns:
        Tuple of (adjusted_over_american, adjusted_under_american)
    """
    standard_american = getattr(config, "UNDERDOG_STANDARD_ODDS", -122)
    standard_decimal = american_to_decimal(standard_american)  # 1.82
    standard_profit = standard_decimal - 1  # 0.82
    
    over_dec = american_to_decimal(over_api_odds)
    under_dec = american_to_decimal(under_api_odds)
    
    # Check if it's a standard line (both sides have same/similar API odds)
    # Allow small tolerance for rounding differences
    if abs(over_dec - under_dec) < 0.03:
        # Standard line - both sides should be -122
        return standard_american, standard_american
    
    # Non-standard line - calculate multipliers relative to API's implied standard
    # The geometric mean of Over/Under gives us the API's "standard" for this line
    api_standard_dec = (over_dec * under_dec) ** 0.5
    api_standard_profit = api_standard_dec - 1
    
    if api_standard_profit <= 0:
        # Fallback if something's wrong
        return standard_american, standard_american
    
    # Calculate profit multipliers (0.9x, 1.1x, etc.)
    over_multiplier = (over_dec - 1) / api_standard_profit
    under_multiplier = (under_dec - 1) / api_standard_profit
    
    # Apply multipliers to our standard -122 odds
    new_over_profit = standard_profit * over_multiplier
    new_under_profit = standard_profit * under_multiplier
    
    # Convert back to American odds
    new_over_dec = 1 + new_over_profit
    new_under_dec = 1 + new_under_profit
    
    # Ensure valid decimal odds (minimum 1.01)
    new_over_dec = max(1.01, new_over_dec)
    new_under_dec = max(1.01, new_under_dec)
    
    return decimal_to_american(new_over_dec), decimal_to_american(new_under_dec)


def process_prop_api_response(
    event: dict,
    prop_markets: list[str],
    timestamp: str,
) -> list[dict]:
    """
    Process player prop API response and prepare data for database insertion.

    Returns list of prop snapshot dicts ready for bulk insert.
    """
    snapshots = []
    
    # First pass: collect all Underdog outcomes grouped by player/line for adjustment
    underdog_outcomes = {}  # key: (player, prop_type, line) -> {"Over": odds, "Under": odds}

    for bookmaker in event.get("bookmakers", []):
        book_key = bookmaker["key"]
        
        # Collect Underdog outcomes for adjustment
        if book_key == "underdog":
            for market in bookmaker.get("markets", []):
                market_key = market["key"]
                if market_key not in prop_markets:
                    continue
                    
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description", "Unknown")
                    line = outcome.get("point")
                    outcome_name = outcome["name"]  # "Over" or "Under"
                    price = outcome["price"]
                    
                    if not player_name or line is None:
                        continue
                    
                    key = (player_name, market_key, line)
                    if key not in underdog_outcomes:
                        underdog_outcomes[key] = {}
                    
                    # Store API odds
                    if config.ODDS_FORMAT == "american":
                        underdog_outcomes[key][outcome_name] = int(price)
                    else:
                        underdog_outcomes[key][outcome_name] = decimal_to_american(float(price))
    
    # Calculate adjusted Underdog odds
    underdog_adjusted = {}  # key: (player, prop_type, line, outcome) -> adjusted_american
    for key, outcomes in underdog_outcomes.items():
        over_odds = outcomes.get("Over")
        under_odds = outcomes.get("Under")
        
        if over_odds is not None and under_odds is not None:
            # We have both sides - adjust them
            adj_over, adj_under = adjust_underdog_odds(over_odds, under_odds)
            underdog_adjusted[(key[0], key[1], key[2], "Over")] = adj_over
            underdog_adjusted[(key[0], key[1], key[2], "Under")] = adj_under
        elif over_odds is not None:
            # Only Over available - use standard odds
            underdog_adjusted[(key[0], key[1], key[2], "Over")] = getattr(config, "UNDERDOG_STANDARD_ODDS", -122)
        elif under_odds is not None:
            # Only Under available - use standard odds
            underdog_adjusted[(key[0], key[1], key[2], "Under")] = getattr(config, "UNDERDOG_STANDARD_ODDS", -122)

    # Second pass: create snapshots with adjusted odds for Underdog
    for bookmaker in event.get("bookmakers", []):
        book_key = bookmaker["key"]

        for market in bookmaker.get("markets", []):
            market_key = market["key"]

            # Only process player prop markets
            if market_key not in prop_markets:
                continue

            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description", "Unknown")
                line = outcome.get("point")
                outcome_name = outcome["name"]
                price = outcome["price"]

                # Skip if missing critical data
                if not player_name or line is None:
                    continue

                # Handle American vs Decimal format
                if config.ODDS_FORMAT == "american":
                    price_american = int(price)
                    price_decimal = american_to_decimal(price_american)
                else:
                    price_decimal = float(price)
                    price_american = decimal_to_american(price_decimal)

                # Apply Underdog odds adjustment
                if book_key == "underdog":
                    adj_key = (player_name, market_key, line, outcome_name)
                    if adj_key in underdog_adjusted:
                        price_american = underdog_adjusted[adj_key]
                        price_decimal = american_to_decimal(price_american)

                snapshots.append({
                    "event_id": event["id"],
                    "bookmaker": book_key,
                    "prop_type": market_key,
                    "player_name": player_name,
                    "outcome": outcome_name,
                    "line": line,
                    "price_american": price_american,
                    "price_decimal": price_decimal,
                    "timestamp": timestamp,
                })

    return snapshots


def run_props_collection_cycle(
    client: OddsAPIClient,
    sports: list[str],
    us_bookmakers: list[str],
    prop_sharp_book: str,
    prop_markets_config: dict,
    max_events_per_cycle: int = 20,
    hours_before_game: int = 24,
    additional_sharp_books: Optional[list[str]] = None,
    dfs_regions: Optional[list[str]] = None,
    dfs_bookmakers: Optional[list[str]] = None,
) -> tuple[int, list[dict]]:
    """
    Run a player props collection cycle.

    Fetches props for upcoming events and returns collected data.

    Args:
        client: OddsAPIClient instance
        sports: List of sports to check
        us_bookmakers: US bookmakers to fetch
        prop_sharp_book: Sharp book for props (e.g., "pinnacle")
        prop_markets_config: Dict mapping sports to their prop markets
        max_events_per_cycle: Maximum events to fetch props for (0 = unlimited)
        hours_before_game: Only fetch props for games starting within this many hours
        additional_sharp_books: Additional sharp books to fetch (e.g., ["betonlineag"])
        dfs_regions: DFS regions to fetch (e.g., ["us_dfs"])
        dfs_bookmakers: DFS bookmakers to fetch (e.g., ["prizepicks", "underdog"])

    Returns:
        Tuple of (total_snapshots, events_with_props)
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    total_snapshots = 0
    events_with_props = []
    all_prop_sharps = [prop_sharp_book] + (additional_sharp_books or [])
    events_processed = 0

    for sport in sports:
        # Get prop markets for this sport
        prop_markets = get_prop_markets_for_sport(sport, prop_markets_config)
        if not prop_markets:
            logger.debug(f"No prop markets configured for {sport}")
            continue

        try:
            # First, get list of events for this sport
            logger.info(f"Fetching {sport} events for props...")
            events = client.get_events(sport)

            if not events:
                continue

            # Filter to games starting within our window
            events = filter_events_for_props(events, hours_before_game)
            logger.info(f"  Found {len(events)} events within {hours_before_game}h window")

            # Fetch props for each event (up to max)
            for event in events:
                if max_events_per_cycle > 0 and events_processed >= max_events_per_cycle:
                    logger.info(f"  Reached max events per cycle ({max_events_per_cycle})")
                    break

                event_id = event["id"]
                home_team = event.get("home_team", "Unknown")
                away_team = event.get("away_team", "Unknown")

                # Upsert event to database
                upsert_event(
                    event_id=event_id,
                    sport=sport,
                    home_team=home_team,
                    away_team=away_team,
                    commence_time=event.get("commence_time", ""),
                )

                # Fetch props from US bookmakers
                logger.debug(f"  Fetching props for {away_team} @ {home_team}...")
                us_props = client.get_player_props(
                    sport=sport,
                    event_id=event_id,
                    regions=["us"],
                    markets=prop_markets,
                    odds_format=config.ODDS_FORMAT,
                    bookmakers=us_bookmakers,
                )

                if us_props:
                    # Add sport_key for consistency
                    us_props["sport_key"] = sport

                    snapshots = process_prop_api_response(us_props, prop_markets, timestamp)
                    if snapshots:
                        bulk_insert_prop_snapshots(snapshots)
                        total_snapshots += len(snapshots)

                # Fetch props from DFS bookmakers (PrizePicks, Underdog, etc.)
                dfs_props = None
                if dfs_regions and dfs_bookmakers:
                    dfs_props = client.get_player_props(
                        sport=sport,
                        event_id=event_id,
                        regions=dfs_regions,
                        markets=prop_markets,
                        odds_format=config.ODDS_FORMAT,
                        bookmakers=dfs_bookmakers,
                    )

                    if dfs_props:
                        dfs_props["sport_key"] = sport
                        snapshots = process_prop_api_response(dfs_props, prop_markets, timestamp)
                        if snapshots:
                            bulk_insert_prop_snapshots(snapshots)
                            total_snapshots += len(snapshots)

                        # Merge DFS props into US props
                        if us_props:
                            us_props["bookmakers"].extend(dfs_props.get("bookmakers", []))
                        else:
                            us_props = dfs_props

                # Fetch props from sharp books (multiple regions to capture all sharps)
                sharp_props = client.get_player_props(
                    sport=sport,
                    event_id=event_id,
                    regions=["us", "uk", "eu", "us2"],  # Multiple regions for lowvig, betanyports, betfair, etc.
                    markets=prop_markets,
                    odds_format=config.ODDS_FORMAT,
                    bookmakers=all_prop_sharps,
                )

                if sharp_props:
                    sharp_props["sport_key"] = sport
                    snapshots = process_prop_api_response(sharp_props, prop_markets, timestamp)
                    if snapshots:
                        bulk_insert_prop_snapshots(snapshots)
                        total_snapshots += len(snapshots)

                    # Merge props data for EV detection
                    if us_props:
                        us_props["bookmakers"].extend(sharp_props.get("bookmakers", []))
                        events_with_props.append(us_props)
                    else:
                        events_with_props.append(sharp_props)
                elif us_props:
                    events_with_props.append(us_props)

                events_processed += 1

            logger.info(f"  Processed {events_processed} events, {total_snapshots} prop snapshots")

        except Exception as e:
            logger.error(f"Error fetching props for {sport}: {e}")

    return total_snapshots, events_with_props


def check_and_alert_props(
    events_with_props: list[dict],
    us_bookmakers: list[str],
    prop_sharp_book: str,
    prop_markets_config: dict,
) -> int:
    """
    Check for +EV opportunities in player props and send alerts.

    Returns the count of +EV opportunities found.
    """
    total_prop_evs = 0

    # Get soft books (exclude sharp book)
    soft_books = [book for book in us_bookmakers if book != prop_sharp_book]

    for event in events_with_props:
        sport = event.get("sport_key", "")
        prop_markets = get_prop_markets_for_sport(sport, prop_markets_config)

        if not prop_markets:
            continue

        # Detect +EV opportunities
        additional_sharps = getattr(config, "SHARP_BOOKMAKERS", [])
        additional_sharp_books_for_consensus = getattr(config, "ADDITIONAL_SHARP_BOOKS", [])
        all_consensus_sharps = additional_sharps + additional_sharp_books_for_consensus
        
        prop_opps = detect_prop_ev_opportunities(
            event=event,
            prop_markets=prop_markets,
            sharp_book=prop_sharp_book,
            soft_books=soft_books,
            min_ev_percent=config.PROP_MIN_EV_PERCENT,
            max_ev_percent=config.PROP_MAX_EV_PERCENT,
            devig_method=config.DEVIG_METHOD,
            max_width_cents=config.PROP_MAX_SHARP_WIDTH_CENTS,
            kelly_by_width=config.PROP_KELLY_BY_WIDTH,
            default_kelly=config.PROP_DEFAULT_KELLY_FRACTION,
            additional_sharp_books=all_consensus_sharps,
            use_weighted_consensus=True,  # Enable weighted consensus model
            pinnacle_weight=getattr(config, "PINNACLE_WEIGHT", 0.60),
            other_sharps_weight=getattr(config, "OTHER_SHARPS_WEIGHT", 0.40),
            weighted_sharp_min_ev=getattr(config, "WEIGHTED_SHARP_MIN_EV", 3.0),
            min_hours_before_game=getattr(config, "MIN_HOURS_BEFORE_GAME", 1),
            max_hours_before_game=getattr(config, "MAX_HOURS_BEFORE_GAME", 4),
        )

        for prop_opp in prop_opps:
            total_prop_evs += 1
            print_prop_ev_opportunity(prop_opp)

            if config.DISCORD_WEBHOOK_URL and getattr(config, "ALERT_ON_PROP_EV", True):
                send_prop_ev_alert_and_log_play(prop_opp)

    return total_prop_evs


def main():
    """Main collector loop."""
    setup_logging(config.LOG_LEVEL)

    # Initialize database
    init_database()
    logger.info("Database initialized")

    # Initialize API client
    client = OddsAPIClient(config.API_KEY, config.BASE_URL)

    # Configuration
    sports = config.SPORTS
    us_regions = ["us"]  # Dropped us2 (fliff)
    us_bookmakers = [b for b in config.INCLUDE_BOOKMAKERS if b != "fliff"]
    markets = config.MARKETS
    sharp_book = config.SHARP_BOOKMAKER
    additional_sharps = getattr(config, "SHARP_BOOKMAKERS", [])
    # Additional sharp books for weighted consensus (lowvig, betanyports, betfair)
    additional_consensus_sharps = getattr(config, "ADDITIONAL_SHARP_BOOKS", [])
    # Combine all sharp books for fetching
    all_sharp_books_for_fetch = additional_sharps + additional_consensus_sharps
    retention_days = config.DATA_RETENTION_DAYS

    # DFS (Daily Fantasy Sports) configuration
    dfs_regions = getattr(config, "DFS_REGIONS", [])
    dfs_bookmakers = getattr(config, "DFS_BOOKMAKERS", [])

    # Player props configuration
    enable_props = getattr(config, "ENABLE_PLAYER_PROPS", False)
    prop_markets_config = getattr(config, "PLAYER_PROP_MARKETS", {})
    prop_sharp_book = getattr(config, "PROP_SHARP_BOOKMAKER", sharp_book)
    max_prop_events = getattr(config, "MAX_PROP_EVENTS_PER_CYCLE", 20)
    prop_hours_before = getattr(config, "PROP_HOURS_BEFORE_GAME", 24)

    # Get polling config (time-based)
    peak_interval = getattr(config, "POLL_INTERVAL_PEAK", 300)
    offpeak_interval = getattr(config, "POLL_INTERVAL_OFFPEAK", 1800)
    peak_start = getattr(config, "PEAK_HOURS_START", 14)
    peak_end = getattr(config, "PEAK_HOURS_END", 22)

    logger.info("=" * 60)
    logger.info("ODDS COLLECTOR STARTED")
    logger.info("=" * 60)
    logger.info(f"  Sports: {', '.join(sports)}")
    logger.info(f"  US Bookmakers: {', '.join(us_bookmakers)}")
    if dfs_bookmakers:
        logger.info(f"  DFS Bookmakers: {', '.join(dfs_bookmakers)}")
    all_sharps = [sharp_book] + all_sharp_books_for_fetch
    logger.info(f"  Sharp Books: {', '.join(all_sharps)}")
    logger.info(f"  Market Consensus Model: Weighted (60% Pinnacle, 40% other sharps)")
    logger.info(f"  Market Width Limits: {config.MAX_SHARP_WIDTH_CENTS} cents (main lines), {config.PROP_MAX_SHARP_WIDTH_CENTS} cents (props)")
    logger.info(f"  Time Window: {getattr(config, 'MIN_HOURS_BEFORE_GAME', 1)}-{getattr(config, 'MAX_HOURS_BEFORE_GAME', 4)} hours before game")
    logger.info(f"  Markets: {', '.join(markets)}")
    logger.info(f"  Poll Interval: {peak_interval//60}min peak ({peak_start}:00-{peak_end}:00), {offpeak_interval//60}min off-peak")
    logger.info(f"  Data Retention: {retention_days} days")
    if enable_props:
        prop_sports = [s for s in sports if s in prop_markets_config]
        all_prop_sharps = [prop_sharp_book] + additional_sharps
        logger.info(f"  Player Props: ENABLED for {', '.join(prop_sports)}")
        logger.info(f"  Props Sharp Books: {', '.join(all_prop_sharps)}")
        logger.info(f"  Max Props Events/Cycle: {max_prop_events if max_prop_events > 0 else 'unlimited'}")
    else:
        logger.info(f"  Player Props: DISABLED")
    logger.info("=" * 60)

    cycle_count = 0
    last_cleanup = datetime.now()

    try:
        while True:
            cycle_count += 1
            cycle_start = datetime.now()
            logger.info(f"\n[Cycle {cycle_count}] Starting at {cycle_start.strftime('%H:%M:%S')}")

            # Reset API cost tracking
            client.reset_scan_cost()

            # Run collection for game lines
            total_snapshots, all_events = run_collection_cycle(
                client=client,
                sports=sports,
                us_regions=us_regions,
                us_bookmakers=us_bookmakers,
                markets=markets,
                sharp_book=sharp_book,
                additional_sharp_books=all_sharp_books_for_fetch,
                dfs_regions=dfs_regions,
                dfs_bookmakers=dfs_bookmakers,
            )

            # Check for opportunities (always runs, prints to console)
            # Combine US and DFS bookmakers for soft books
            all_soft_bookmakers = us_bookmakers + dfs_bookmakers
            arb_count, ev_count = check_and_alert(
                all_events=all_events,
                sports=sports,
                markets=markets,
                us_bookmakers=all_soft_bookmakers,
                sharp_book=sharp_book,
            )

            # Log game lines stats
            if client.usage:
                logger.info(f"  API: {client.scan_cost} credits used, {client.usage.requests_remaining} remaining")
            logger.info(f"  Stored {total_snapshots} odds snapshots")
            logger.info(f"  Found: {arb_count} arbs, {ev_count} +EV opportunities")

            # =================================================================
            # PLAYER PROPS COLLECTION
            # =================================================================
            prop_ev_count = 0
            prop_snapshots = 0

            if enable_props:
                logger.info("\n  --- Player Props Collection ---")

                prop_snapshots, events_with_props = run_props_collection_cycle(
                    client=client,
                    sports=sports,
                    us_bookmakers=us_bookmakers,
                    prop_sharp_book=prop_sharp_book,
                    prop_markets_config=prop_markets_config,
                    max_events_per_cycle=max_prop_events,
                    hours_before_game=prop_hours_before,
                    additional_sharp_books=all_sharp_books_for_fetch,
                    dfs_regions=dfs_regions,
                    dfs_bookmakers=dfs_bookmakers,
                )

                if events_with_props:
                    prop_ev_count = check_and_alert_props(
                        events_with_props=events_with_props,
                        us_bookmakers=all_soft_bookmakers,
                        prop_sharp_book=prop_sharp_book,
                        prop_markets_config=prop_markets_config,
                    )

                logger.info(f"  Props: {prop_snapshots} snapshots, {prop_ev_count} +EV opportunities")
                if client.usage:
                    logger.info(f"  API after props: {client.scan_cost} credits used, {client.usage.requests_remaining} remaining")

            # Process closing lines for any games that have started
            closing_updated = process_closing_lines()
            prop_closing_updated = process_prop_closing_lines() if enable_props else 0
            total_closing = closing_updated + prop_closing_updated
            if total_closing > 0:
                logger.info(f"  Closing lines: captured {closing_updated} game lines, {prop_closing_updated} props")

            # Periodic cleanup (once per hour)
            if (datetime.now() - last_cleanup).total_seconds() > 3600:
                deleted = cleanup_old_data(retention_days)
                prop_deleted = cleanup_old_prop_data(retention_days) if enable_props else 0
                logger.info(f"  Cleanup: removed {deleted} old records, {prop_deleted} old prop records")
                last_cleanup = datetime.now()

                # Log database stats
                stats = get_database_stats()
                logger.info(f"  DB Stats: {stats['events']} events, {stats['snapshots']} snapshots, {stats['db_size_mb']} MB")

            # Calculate sleep time using time-based interval
            poll_interval = get_current_poll_interval()
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            sleep_time = max(0, poll_interval - cycle_duration)

            # Summary
            total_opps = arb_count + ev_count + prop_ev_count
            logger.info(f"\n  [Cycle {cycle_count} Summary] {total_opps} total opportunities ({arb_count} arbs, {ev_count} game EV, {prop_ev_count} prop EV)")

            if sleep_time > 0:
                current_hour = datetime.now().hour
                mode = "peak" if peak_start <= current_hour < peak_end else "off-peak"
                logger.info(f"  Next poll: {sleep_time//60:.0f}m {sleep_time%60:.0f}s ({mode})")
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\nCollector stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
