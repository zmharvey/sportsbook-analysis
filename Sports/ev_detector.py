"""
+EV (Positive Expected Value) Detection Module.
Compares soft book odds against sharp book consensus to find +EV opportunities.

Supports multiple devigging methods:
- multiplicative: Proportionally removes vig (original method, favors longshots)
- additive: Subtracts equal vig from each outcome
- power: Uses power method to solve for fair probabilities (most accurate)
- shin: Models vig as coming from informed bettors
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime, timezone, timedelta
from arbitrage import american_to_decimal, decimal_to_american, is_game_upcoming

logger = logging.getLogger(__name__)

# Type alias for devig methods
DevigMethod = Literal["multiplicative", "additive", "power", "shin"]


@dataclass
class EVOpportunity:
    """Represents a +EV betting opportunity."""
    event_id: str
    sport: str
    home_team: str
    away_team: str
    commence_time: str
    market: str
    outcome: str
    point: Optional[float]

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

    # EV calculation
    ev_percent: float
    edge_percent: float  # Difference in implied probability

    # Kelly and width (set after detection)
    sharp_width_cents: Optional[int] = None
    kelly_fraction: float = 0.25
    units: float = 0.0

    def __str__(self) -> str:
        point_str = f" ({self.point:+.1f})" if self.point is not None else ""
        return (
            f"+EV: {self.outcome}{point_str} @ {self.soft_book} "
            f"({self.soft_odds_american:+d}) | "
            f"EV: {self.ev_percent:.2f}% | "
            f"{self.units:.2f} units"
        )


def devig_multiplicative(odds_list: list[float]) -> list[float]:
    """
    Multiplicative devig - proportionally removes vig.

    This is the simplest method but tends to overcorrect longshots,
    making them appear more +EV than they actually are.

    Formula: fair_prob = implied_prob / total_implied_prob
    """
    implied_probs = [1 / odds for odds in odds_list]
    total = sum(implied_probs)
    return [prob / total for prob in implied_probs]


def devig_additive(odds_list: list[float]) -> list[float]:
    """
    Additive devig - subtracts equal vig from each outcome.

    Better for balanced markets but can produce negative probabilities
    on heavy favorites (which we clamp to a minimum).

    Formula: fair_prob = implied_prob - (overround / num_outcomes)
    """
    implied_probs = [1 / odds for odds in odds_list]
    total = sum(implied_probs)
    overround = total - 1.0
    adjustment = overround / len(odds_list)

    # Subtract equal vig and ensure no negative probabilities
    fair_probs = [max(0.001, prob - adjustment) for prob in implied_probs]

    # Renormalize to sum to 1.0
    fair_total = sum(fair_probs)
    return [p / fair_total for p in fair_probs]


def devig_power(odds_list: list[float], tolerance: float = 1e-8, max_iterations: int = 100) -> list[float]:
    """
    Power method devig - solves for the exponent that makes probabilities sum to 1.

    This is considered the most accurate method for sports betting as it
    applies more vig adjustment to longshots (where books typically add more vig).

    Solves: sum(implied_prob_i ^ k) = 1 for k
    Then: fair_prob_i = implied_prob_i ^ k
    """
    implied_probs = [1 / odds for odds in odds_list]

    # Binary search for the power k that makes probs sum to 1
    k_low, k_high = 0.5, 2.0

    for _ in range(max_iterations):
        k = (k_low + k_high) / 2
        powered_probs = [p ** k for p in implied_probs]
        total = sum(powered_probs)

        if abs(total - 1.0) < tolerance:
            break
        elif total > 1.0:
            k_low = k
        else:
            k_high = k

    # Calculate fair probabilities with found k
    fair_probs = [p ** k for p in implied_probs]

    # Normalize to ensure exactly 1.0
    total = sum(fair_probs)
    return [p / total for p in fair_probs]


def devig_shin(odds_list: list[float], tolerance: float = 1e-8, max_iterations: int = 100) -> list[float]:
    """
    Shin method devig - models vig as coming from informed bettors.

    Based on Shin (1991, 1993) papers on market microstructure.
    Assumes some proportion z of bettors are "informed" and always bet correctly.

    This method is theoretically grounded and handles longshots well.
    """
    implied_probs = [1 / odds for odds in odds_list]
    n = len(implied_probs)
    total = sum(implied_probs)

    # Binary search for z (proportion of informed traders)
    z_low, z_high = 0.0, 0.5

    for _ in range(max_iterations):
        z = (z_low + z_high) / 2

        # Calculate fair probabilities given z
        fair_probs = []
        for imp_prob in implied_probs:
            # Shin formula: p = (sqrt(z^2 + 4*(1-z)*imp_prob^2/total) - z) / (2*(1-z))
            discriminant = z ** 2 + 4 * (1 - z) * (imp_prob ** 2) / total
            if discriminant < 0:
                fair_probs.append(imp_prob / total)  # Fallback to multiplicative
            else:
                fair_p = (math.sqrt(discriminant) - z) / (2 * (1 - z)) if z < 1 else imp_prob / total
                fair_probs.append(max(0.001, fair_p))

        prob_sum = sum(fair_probs)

        if abs(prob_sum - 1.0) < tolerance:
            break
        elif prob_sum > 1.0:
            z_low = z
        else:
            z_high = z

    # Normalize to ensure exactly 1.0
    total = sum(fair_probs)
    return [p / total for p in fair_probs]


def calculate_no_vig_probability(
    odds_list: list[float],
    method: DevigMethod = "power"
) -> list[float]:
    """
    Calculate no-vig (fair) probabilities from a set of odds.

    Args:
        odds_list: List of decimal odds for all outcomes in a market
        method: Devigging method to use:
            - "multiplicative": Simple proportional (favors longshots)
            - "additive": Equal vig subtraction
            - "power": Power method (recommended, most accurate)
            - "shin": Shin model (accounts for informed bettors)

    Returns:
        List of fair probabilities (sum to 1.0)
    """
    if method == "multiplicative":
        return devig_multiplicative(odds_list)
    elif method == "additive":
        return devig_additive(odds_list)
    elif method == "power":
        return devig_power(odds_list)
    elif method == "shin":
        return devig_shin(odds_list)
    else:
        logger.warning(f"Unknown devig method '{method}', using power")
        return devig_power(odds_list)


def calculate_ev(fair_prob: float, decimal_odds: float) -> float:
    """
    Calculate Expected Value percentage.

    EV = (Fair Probability × Decimal Odds) - 1

    Args:
        fair_prob: The true/fair probability of the outcome
        decimal_odds: The decimal odds being offered

    Returns:
        EV as a percentage (e.g., 3.5 means +3.5% EV)
    """
    return (fair_prob * decimal_odds - 1) * 100


def calculate_sharp_width_cents(sharp_outcomes: list[dict]) -> Optional[int]:
    """
    Calculate the width of a sharp book's market in cents.

    Width = sum of absolute American odds distances from -100
    For a 2-way market: if odds are +150/-200, width = 50 + 100 = 150? No.
    Actually width is typically: abs(fav_odds) - abs(dog_odds) when both negative,
    or for mixed: the "juice" spread.

    Simpler definition: For 2-way, width = |american1| + |american2| - 200
    For -110/-110: 110 + 110 - 200 = 20 cents
    For +150/-200: 150 + 200 - 200 = 150 cents? That seems too high.

    Let's use: total implied prob - 100 in basis points, roughly.
    Or simpler: just the difference in implied probs * 100.

    Actually, the standard way: for a 2-way market,
    width = (1/decimal1 + 1/decimal2 - 1) * 100 in "cents" terms.

    Let's just use overround as a proxy for width.
    """
    if len(sharp_outcomes) < 2:
        return None

    # Calculate total implied probability (overround)
    total_implied = sum(1 / o["decimal"] for o in sharp_outcomes)
    # Convert to cents (overround of 1.04 = 4 cents)
    width_cents = int((total_implied - 1) * 100)
    return max(0, width_cents)


def calculate_kelly_units(
    fair_prob: float,
    decimal_odds: float,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Calculate Kelly criterion bet size in units.

    Full Kelly = (p * (odds - 1) - (1 - p)) / (odds - 1)
              = (p * odds - 1) / (odds - 1)

    This gives the fraction of bankroll to bet. We convert to units where
    1 unit = 1% of bankroll, so we multiply by 100.

    Args:
        fair_prob: True probability of winning (0 to 1)
        decimal_odds: Decimal odds being offered
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)

    Returns:
        Bet size in units (e.g., 0.5 = half a unit, where 1 unit = 1% bankroll)
    """
    if decimal_odds <= 1:
        return 0.0

    # Full Kelly formula gives fraction of bankroll
    edge = fair_prob * decimal_odds - 1
    if edge <= 0:
        return 0.0

    full_kelly = edge / (decimal_odds - 1)

    # Apply Kelly fraction
    kelly_bankroll_fraction = full_kelly * kelly_fraction

    # Convert to units (1 unit = 1% of bankroll)
    # So kelly_fraction of 0.01 (1%) = 1 unit
    units = kelly_bankroll_fraction * 100

    # Cap at reasonable maximum (e.g., 5 units)
    return min(units, 5.0)


def get_kelly_fraction_for_width(
    width_cents: int,
    kelly_by_width: list[tuple[int, float]],
    default_kelly: float = 0.25,
) -> float:
    """
    Get the Kelly fraction based on market width.

    Args:
        width_cents: Width of the sharp market in cents
        kelly_by_width: List of (max_width, kelly_fraction) tuples
        default_kelly: Default if no rule matches

    Returns:
        Kelly fraction to use
    """
    for max_width, kelly in kelly_by_width:
        if width_cents <= max_width:
            return kelly
    return default_kelly


def get_sharp_odds_for_outcome(
    event: dict,
    sharp_book: str,
    market_key: str,
    outcome_name: str,
    point: Optional[float] = None,
) -> Optional[tuple[int, float]]:
    """
    Get the sharp book's odds for a specific outcome.

    Returns:
        Tuple of (american_odds, decimal_odds) or None if not found
    """
    for bookmaker in event.get("bookmakers", []):
        if bookmaker["key"] != sharp_book:
            continue

        for market in bookmaker.get("markets", []):
            if market["key"] != market_key:
                continue

            for outcome in market.get("outcomes", []):
                outcome_point = outcome.get("point")

                # Match outcome name and point
                if outcome["name"] == outcome_name:
                    if point is None and outcome_point is None:
                        return _extract_odds(outcome)
                    elif point is not None and outcome_point is not None:
                        if abs(point - outcome_point) < 0.01:
                            return _extract_odds(outcome)

    return None


def _extract_odds(outcome: dict) -> tuple[int, float]:
    """Extract odds from outcome dict, handling both formats."""
    price = outcome["price"]

    # Determine format based on value
    if isinstance(price, int) or (isinstance(price, float) and (price > 50 or price < -50)):
        # American format
        american = int(price)
        decimal = american_to_decimal(american)
    else:
        # Decimal format
        decimal = float(price)
        american = decimal_to_american(decimal)

    return american, decimal


def get_all_sharp_outcomes(
    event: dict,
    sharp_book: str,
    market_key: str,
) -> list[dict]:
    """Get all outcomes from the sharp book for a specific market."""
    outcomes = []

    for bookmaker in event.get("bookmakers", []):
        if bookmaker["key"] != sharp_book:
            continue

        for market in bookmaker.get("markets", []):
            if market["key"] != market_key:
                continue

            for outcome in market.get("outcomes", []):
                american, decimal = _extract_odds(outcome)
                outcomes.append({
                    "name": outcome["name"],
                    "point": outcome.get("point"),
                    "american": american,
                    "decimal": decimal,
                })

    return outcomes


def get_hours_until_game(commence_time: str) -> Optional[float]:
    """
    Calculate hours until game start.
    
    Returns:
        Hours until game start, or None if parsing fails
    """
    try:
        if commence_time.endswith("Z"):
            event_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        else:
            event_time = datetime.fromisoformat(commence_time)
        
        now = datetime.now(timezone.utc)
        time_diff = event_time - now
        return time_diff.total_seconds() / 3600
    except (ValueError, AttributeError):
        return None


def calculate_weighted_sharp_price(
    event: dict,
    market_key: str,
    outcome_name: str,
    outcome_point: Optional[float],
    primary_sharp: str,
    additional_sharp_books: list[str],
    pinnacle_weight: float = 0.60,
    other_sharps_weight: float = 0.40,
    devig_method: DevigMethod = "power",
) -> Optional[dict]:
    """
    Calculate weighted sharp price from multiple sharp books.
    
    Weighted Sharp Price = (Pinnacle weight * Pinnacle fair prob) + 
                          (Other sharps weight * Average of other sharps fair prob)
    
    Args:
        event: Event dictionary from API
        market_key: Market to check (e.g., "h2h", "spreads", "totals")
        outcome_name: Outcome name to find consensus for
        outcome_point: Point/spread value (None for moneyline)
        primary_sharp: Primary sharp book (typically "pinnacle")
        additional_sharp_books: List of additional sharp books to include
        pinnacle_weight: Weight for primary sharp book (default 0.60)
        other_sharps_weight: Combined weight for other sharp books (default 0.40)
        devig_method: Method to remove vig from sharp lines
    
    Returns:
        Dictionary with weighted fair probability and metadata, or None if insufficient data
    """
    # Get primary sharp book odds
    primary_odds = get_sharp_odds_for_outcome(event, primary_sharp, market_key, outcome_name, outcome_point)
    if not primary_odds:
        return None
    
    # Get all outcomes from primary sharp for devigging
    primary_outcomes = get_all_sharp_outcomes(event, primary_sharp, market_key)
    if len(primary_outcomes) < 2:
        return None
    
    # Calculate fair probabilities from primary sharp
    primary_decimals = [o["decimal"] for o in primary_outcomes]
    primary_fair_probs = calculate_no_vig_probability(primary_decimals, method=devig_method)
    
    # Find the fair probability for this specific outcome
    primary_fair_prob = None
    for i, outcome in enumerate(primary_outcomes):
        outcome_pt = outcome.get("point")
        points_match = (
            (outcome_point is None and outcome_pt is None) or
            (outcome_point is not None and outcome_pt is not None
             and abs(outcome_point - outcome_pt) < 0.01)
        )
        if outcome["name"] == outcome_name and points_match:
            primary_fair_prob = primary_fair_probs[i]
            break
    
    if primary_fair_prob is None:
        return None
    
    # Collect fair probabilities from additional sharp books
    other_sharp_fair_probs = []
    available_sharp_books = []
    
    for alt_sharp in additional_sharp_books:
        alt_outcomes = get_all_sharp_outcomes(event, alt_sharp, market_key)
        if not alt_outcomes or len(alt_outcomes) != len(primary_outcomes):
            continue
        
        # Calculate fair probabilities from alternative sharp
        alt_decimals = [o["decimal"] for o in alt_outcomes]
        alt_fair_probs = calculate_no_vig_probability(alt_decimals, method=devig_method)
        
        # Find matching outcome
        for i, alt_outcome in enumerate(alt_outcomes):
            alt_pt = alt_outcome.get("point")
            points_match = (
                (outcome_point is None and alt_pt is None) or
                (outcome_point is not None and alt_pt is not None
                 and abs(outcome_point - alt_pt) < 0.01)
            )
            if alt_outcome["name"] == outcome_name and points_match:
                other_sharp_fair_probs.append(alt_fair_probs[i])
                available_sharp_books.append(alt_sharp)
                break
    
    # Calculate weighted fair probability
    if other_sharp_fair_probs:
        # Average of other sharps
        avg_other_fair_prob = sum(other_sharp_fair_probs) / len(other_sharp_fair_probs)
        # Weighted combination
        weighted_fair_prob = (pinnacle_weight * primary_fair_prob) + (other_sharps_weight * avg_other_fair_prob)
    else:
        # Only primary sharp available - use it directly
        weighted_fair_prob = primary_fair_prob
    
    return {
        "weighted_fair_prob": weighted_fair_prob,
        "primary_fair_prob": primary_fair_prob,
        "primary_odds_american": primary_odds[0],
        "primary_odds_decimal": primary_odds[1],
        "available_sharp_books": [primary_sharp] + available_sharp_books,
        "num_sharp_books": 1 + len(available_sharp_books),
    }


def calculate_market_consensus(
    event: dict,
    market_key: str,
    outcome_name: str,
    outcome_point: Optional[float],
    exclude_books: Optional[list[str]] = None,
) -> tuple[Optional[float], int]:
    """
    Calculate market-wide consensus implied probability for an outcome.

    Args:
        event: Event dictionary from API
        market_key: Market to check (e.g., "h2h", "spreads", "totals")
        outcome_name: Outcome name to find consensus for
        outcome_point: Point/spread value (None for moneyline)
        exclude_books: Books to exclude from consensus

    Returns:
        Tuple of (average implied probability as %, number of books)
    """
    exclude_books = exclude_books or []
    implied_probs = []

    for bookmaker in event.get("bookmakers", []):
        if bookmaker["key"] in exclude_books:
            continue

        for market in bookmaker.get("markets", []):
            if market["key"] != market_key:
                continue

            for outcome in market.get("outcomes", []):
                if outcome["name"] != outcome_name:
                    continue

                outcome_pt = outcome.get("point")

                # Check point match
                points_match = (
                    (outcome_point is None and outcome_pt is None) or
                    (outcome_point is not None and outcome_pt is not None
                     and abs(outcome_point - outcome_pt) < 0.01)
                )

                if points_match:
                    _, decimal = _extract_odds(outcome)
                    implied_probs.append((1 / decimal) * 100)

    return (sum(implied_probs) / len(implied_probs), len(implied_probs)) if implied_probs else (None, 0)


def detect_ev_opportunities(
    events: list[dict],
    sharp_book: str,
    soft_books: list[str],
    markets: list[str],
    min_ev_percent: float = 1.0,
    max_ev_percent: float = 20.0,
    devig_method: DevigMethod = "power",
    max_width_cents: Optional[int] = None,
    kelly_by_width: Optional[list[tuple[int, float]]] = None,
    default_kelly: float = 0.25,
    additional_sharp_books: Optional[list[str]] = None,
    require_sharp_consensus: bool = False,
    max_sharp_disagreement: float = 2.0,
    enable_market_consensus: bool = False,
    max_market_deviation: float = 5.0,
    min_books_for_consensus: int = 3,
    use_weighted_consensus: bool = False,
    pinnacle_weight: float = 0.60,
    other_sharps_weight: float = 0.40,
    weighted_sharp_min_ev: float = 3.0,
    min_hours_before_game: Optional[float] = None,
    max_hours_before_game: Optional[float] = None,
) -> list[EVOpportunity]:
    """
    Detect +EV opportunities by comparing soft books against sharp book(s).

    Args:
        events: List of event dictionaries from the API
        sharp_book: The primary sharp bookmaker to use as reference (e.g., "pinnacle")
        soft_books: List of soft bookmakers to check for +EV
        markets: List of markets to check (e.g., ["h2h", "spreads", "totals"])
        min_ev_percent: Minimum EV percentage to report
        max_ev_percent: Maximum EV percentage (filter out likely errors)
        devig_method: Method to remove vig from sharp lines
        max_width_cents: Maximum sharp market width to accept (None = no limit)
        kelly_by_width: List of (max_width, kelly_fraction) for sizing
        default_kelly: Default Kelly fraction if no width rules match
        additional_sharp_books: List of additional sharp books for consensus validation
        require_sharp_consensus: If True, filter out plays where sharp books disagree
        max_sharp_disagreement: Maximum disagreement in implied prob % between sharp books
        enable_market_consensus: If True, validate against market-wide consensus
        max_market_deviation: Max deviation from market average implied prob (%)
        min_books_for_consensus: Minimum books needed to calculate market consensus
        use_weighted_consensus: If True, use weighted consensus from multiple sharp books
        pinnacle_weight: Weight for primary sharp book in weighted consensus (default 0.60)
        other_sharps_weight: Combined weight for other sharp books (default 0.40)
        weighted_sharp_min_ev: Minimum EV% required for weighted sharp price (default 3.0)
        min_hours_before_game: Minimum hours before game start (filter out games starting too soon)
        max_hours_before_game: Maximum hours before game start (filter out games starting > X hours)

    Returns:
        List of EVOpportunity objects
    """
    opportunities = []

    for event in events:
        event_id = event.get("id", "")
        sport = event.get("sport_key", "")
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        commence_time = event.get("commence_time", "")

        # Skip games that have already started
        if not is_game_upcoming(commence_time):
            logger.debug(f"Skipping {event_id}: game has already started")
            continue
        
        # Time-window filter (Golden Hour: 1-4 hours before game start)
        if min_hours_before_game is not None or max_hours_before_game is not None:
            hours_until = get_hours_until_game(commence_time)
            if hours_until is None:
                logger.debug(f"Skipping {event_id}: could not parse commence_time")
                continue
            
            if min_hours_before_game is not None and hours_until < min_hours_before_game:
                logger.debug(f"Skipping {event_id}: game starts in {hours_until:.1f}h (too soon, min: {min_hours_before_game}h)")
                continue
            
            if max_hours_before_game is not None and hours_until > max_hours_before_game:
                logger.debug(f"Skipping {event_id}: game starts in {hours_until:.1f}h (too far, max: {max_hours_before_game}h)")
                continue

        for market_key in markets:
            # Get all sharp book outcomes for this market
            sharp_outcomes = get_all_sharp_outcomes(event, sharp_book, market_key)

            if not sharp_outcomes:
                continue

            # Calculate no-vig probabilities from sharp book
            if len(sharp_outcomes) < 2:
                continue

            # Check consensus with additional sharp books if enabled
            if require_sharp_consensus and additional_sharp_books:
                logger.debug(f"Checking sharp consensus for {event_id}/{market_key} with books: {additional_sharp_books}")

                # Get fair probabilities from primary sharp book
                primary_decimals = [o["decimal"] for o in sharp_outcomes]
                primary_fair_probs = calculate_no_vig_probability(primary_decimals, method=devig_method)

                # Check each additional sharp book
                consensus_valid = True
                for alt_sharp_book in additional_sharp_books:
                    alt_outcomes = get_all_sharp_outcomes(event, alt_sharp_book, market_key)

                    if not alt_outcomes or len(alt_outcomes) != len(sharp_outcomes):
                        # Alternative sharp book doesn't have this market - skip consensus check for this book
                        logger.debug(f"Sharp book {alt_sharp_book} missing market {market_key} for {event_id}, skipping consensus check with this book")
                        continue

                    # Calculate fair probabilities from alternative sharp book
                    alt_decimals = [o["decimal"] for o in alt_outcomes]
                    alt_fair_probs = calculate_no_vig_probability(alt_decimals, method=devig_method)

                    # Check disagreement for each outcome
                    for i, sharp_outcome in enumerate(sharp_outcomes):
                        # Match outcomes by name and point
                        outcome_name = sharp_outcome["name"]
                        outcome_point = sharp_outcome.get("point")

                        # Find matching outcome in alternative book
                        alt_idx = None
                        for j, alt_outcome in enumerate(alt_outcomes):
                            if alt_outcome["name"] == outcome_name:
                                if outcome_point is None and alt_outcome.get("point") is None:
                                    alt_idx = j
                                    break
                                elif outcome_point is not None and alt_outcome.get("point") is not None:
                                    if abs(outcome_point - alt_outcome.get("point")) < 0.01:
                                        alt_idx = j
                                        break

                        if alt_idx is None:
                            continue

                        # Compare implied probabilities (convert to %)
                        primary_prob_pct = primary_fair_probs[i] * 100
                        alt_prob_pct = alt_fair_probs[alt_idx] * 100
                        disagreement = abs(primary_prob_pct - alt_prob_pct)

                        if disagreement > max_sharp_disagreement:
                            logger.debug(
                                f"Sharp consensus failed for {event_id}/{market_key}/{outcome_name}: "
                                f"{sharp_book}={primary_prob_pct:.1f}% vs {alt_sharp_book}={alt_prob_pct:.1f}% "
                                f"(disagreement: {disagreement:.1f}%)"
                            )
                            consensus_valid = False
                            break

                    if not consensus_valid:
                        break

                # Skip this market if consensus failed
                if not consensus_valid:
                    continue

            # Calculate market width
            width_cents = calculate_sharp_width_cents(sharp_outcomes)

            # Skip if market is too wide
            if max_width_cents is not None and width_cents is not None:
                if width_cents > max_width_cents:
                    logger.debug(f"Skipping {event_id}/{market_key}: width {width_cents} > max {max_width_cents}")
                    continue

            # Determine Kelly fraction based on width
            if kelly_by_width and width_cents is not None:
                kelly_fraction = get_kelly_fraction_for_width(width_cents, kelly_by_width, default_kelly)
            else:
                kelly_fraction = default_kelly

            # Calculate fair probabilities - use weighted consensus if enabled
            if use_weighted_consensus and additional_sharp_books:
                # Build weighted consensus lookup for each outcome
                fair_prob_lookup = {}
                for outcome in sharp_outcomes:
                    outcome_name = outcome["name"]
                    outcome_point = outcome.get("point")
                    key = (outcome_name, outcome_point)
                    
                    # Calculate weighted sharp price for this outcome
                    weighted_data = calculate_weighted_sharp_price(
                        event=event,
                        market_key=market_key,
                        outcome_name=outcome_name,
                        outcome_point=outcome_point,
                        primary_sharp=sharp_book,
                        additional_sharp_books=additional_sharp_books,
                        pinnacle_weight=pinnacle_weight,
                        other_sharps_weight=other_sharps_weight,
                        devig_method=devig_method,
                    )
                    
                    if weighted_data:
                        fair_prob_lookup[key] = {
                            "fair_prob": weighted_data["weighted_fair_prob"],
                            "american": weighted_data["primary_odds_american"],
                            "decimal": weighted_data["primary_odds_decimal"],
                            "num_sharp_books": weighted_data["num_sharp_books"],
                        }
                    else:
                        # Fallback to primary sharp only if weighted calculation fails
                        sharp_decimals = [o["decimal"] for o in sharp_outcomes]
                        fair_probs = calculate_no_vig_probability(sharp_decimals, method=devig_method)
                        for i, o in enumerate(sharp_outcomes):
                            if o["name"] == outcome_name and (
                                (outcome_point is None and o.get("point") is None) or
                                (outcome_point is not None and o.get("point") is not None
                                 and abs(outcome_point - o.get("point")) < 0.01)
                            ):
                                fair_prob_lookup[key] = {
                                    "fair_prob": fair_probs[i],
                                    "american": o["american"],
                                    "decimal": o["decimal"],
                                    "num_sharp_books": 1,
                                }
                                break
            else:
                # Use primary sharp book only (original logic)
                sharp_decimals = [o["decimal"] for o in sharp_outcomes]
                fair_probs = calculate_no_vig_probability(sharp_decimals, method=devig_method)

                # Create lookup for fair probabilities
                fair_prob_lookup = {}
                for i, outcome in enumerate(sharp_outcomes):
                    key = (outcome["name"], outcome.get("point"))
                    fair_prob_lookup[key] = {
                        "fair_prob": fair_probs[i],
                        "american": outcome["american"],
                        "decimal": outcome["decimal"],
                    }

            # Check each soft book
            for bookmaker in event.get("bookmakers", []):
                book_key = bookmaker["key"]

                if book_key not in soft_books or book_key == sharp_book:
                    continue

                for market in bookmaker.get("markets", []):
                    if market["key"] != market_key:
                        continue

                    for outcome in market.get("outcomes", []):
                        outcome_name = outcome["name"]
                        outcome_point = outcome.get("point")
                        key = (outcome_name, outcome_point)

                        if key not in fair_prob_lookup:
                            continue

                        fair_data = fair_prob_lookup[key]
                        fair_prob = fair_data["fair_prob"]

                        # Get soft book odds
                        soft_american, soft_decimal = _extract_odds(outcome)
                        soft_implied = 1 / soft_decimal

                        # Calculate EV
                        ev = calculate_ev(fair_prob, soft_decimal)

                        # For weighted consensus, require minimum EV threshold
                        if use_weighted_consensus and ev < weighted_sharp_min_ev:
                            logger.debug(
                                f"Skipping {event_id}/{market_key}/{outcome_name} @ {book_key}: "
                                f"EV {ev:.2f}% < weighted sharp min {weighted_sharp_min_ev}%"
                            )
                            continue

                        # Calculate edge (difference in implied probability)
                        edge = (fair_prob - soft_implied) * 100

                        # Calculate Kelly units
                        units = calculate_kelly_units(fair_prob, soft_decimal, kelly_fraction)

                        if min_ev_percent <= ev <= max_ev_percent:
                            # Market-wide consensus validation
                            if enable_market_consensus:
                                market_avg_prob, num_books = calculate_market_consensus(
                                    event, market_key, outcome_name, outcome_point, [book_key]
                                )

                                if num_books >= min_books_for_consensus:
                                    soft_prob = soft_implied * 100
                                    deviation = abs(soft_prob - market_avg_prob)

                                    if deviation > max_market_deviation:
                                        logger.debug(
                                            f"Market consensus failed for {event_id}/{market_key}/{outcome_name} @ {book_key}: "
                                            f"soft={soft_prob:.1f}% vs market={market_avg_prob:.1f}% "
                                            f"(deviation={deviation:.1f}%, {num_books} books)"
                                        )
                                        continue  # Skip this opportunity

                                    logger.debug(
                                        f"Market consensus OK: {event_id}/{market_key}/{outcome_name} @ {book_key} "
                                        f"(deviation={deviation:.1f}%)"
                                    )
                                else:
                                    logger.debug(
                                        f"Skipping consensus check for {event_id}/{market_key}/{outcome_name}: "
                                        f"only {num_books}/{min_books_for_consensus} books"
                                    )

                            opp = EVOpportunity(
                                event_id=event_id,
                                sport=sport,
                                home_team=home_team,
                                away_team=away_team,
                                commence_time=commence_time,
                                market=market_key,
                                outcome=outcome_name,
                                point=outcome_point,
                                sharp_book=sharp_book,
                                sharp_odds_american=fair_data["american"],
                                sharp_odds_decimal=fair_data["decimal"],
                                sharp_implied_prob=fair_prob * 100,
                                soft_book=book_key,
                                soft_odds_american=soft_american,
                                soft_odds_decimal=soft_decimal,
                                soft_implied_prob=soft_implied * 100,
                                ev_percent=ev,
                                edge_percent=edge,
                                sharp_width_cents=width_cents,
                                kelly_fraction=kelly_fraction,
                                units=units,
                            )
                            opportunities.append(opp)
                            logger.debug(f"Found +EV: {opp}")

    # Sort by EV percentage (highest first)
    opportunities.sort(key=lambda x: x.ev_percent, reverse=True)

    return opportunities


def format_ev_opportunity(opp: EVOpportunity, odds_format: str = "american") -> str:
    """Format an EV opportunity for display."""
    point_str = f" ({opp.point:+.1f})" if opp.point is not None else ""

    if odds_format == "american":
        soft_odds = f"{opp.soft_odds_american:+d}"
        sharp_odds = f"{opp.sharp_odds_american:+d}"
    else:
        soft_odds = f"{opp.soft_odds_decimal:.2f}"
        sharp_odds = f"{opp.sharp_odds_decimal:.2f}"

    return f"""
+EV OPPORTUNITY
{'=' * 50}
{opp.away_team} @ {opp.home_team}
Market: {opp.market.upper()}{point_str}

Bet: {opp.outcome} @ {opp.soft_book.upper()}
Odds: {soft_odds}

Reference: {opp.sharp_book.upper()} @ {sharp_odds}
Fair Prob: {opp.sharp_implied_prob:.1f}%

EV: +{opp.ev_percent:.2f}%
Edge: +{opp.edge_percent:.2f}%
{'=' * 50}
"""
