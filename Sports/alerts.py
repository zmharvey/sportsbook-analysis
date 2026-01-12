"""
Alert system for arbitrage and +EV opportunities.
Supports Discord webhooks with duplicate prevention.
"""

import logging
import requests
from typing import Optional
from datetime import datetime, timezone, timedelta

import config
from database import check_alert_sent, log_alert, log_play, log_arb_play, log_prop_play
from arbitrage import ArbitrageOpportunity, decimal_to_american
from ev_detector import EVOpportunity
from player_props import PropEVOpportunity, get_prop_display_name

logger = logging.getLogger(__name__)

# CST timezone (UTC-6)
CST = timezone(timedelta(hours=-6))


def get_cst_timestamp() -> str:
    """Get current timestamp in CST format."""
    return datetime.now(CST).strftime("%Y-%m-%d %I:%M %p CST")


def format_game_time(commence_time: str) -> str:
    """Format game commence time to CST format."""
    try:
        # Parse ISO format time
        if commence_time.endswith("Z"):
            game_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        else:
            game_time = datetime.fromisoformat(commence_time)

        # Convert to CST
        game_time_cst = game_time.astimezone(CST)
        return game_time_cst.strftime("%I:%M %p CST")
    except (ValueError, AttributeError):
        return "Unknown"


def send_discord_alert(
    message: str,
    alert_type: str,
    event_id: str,
    market: str,
    outcome: Optional[str] = None,
    ev_percent: Optional[float] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """
    Send an alert to Discord webhook.

    Args:
        message: The formatted message to send
        alert_type: Type of alert ("arb", "ev", or "prop_ev")
        event_id: Event ID for deduplication
        market: Market for deduplication
        outcome: Outcome name for deduplication (e.g., team name or Over/Under)
        ev_percent: EV percentage (logged for reference, not used for re-alerting)
        webhook_url: Discord webhook URL (defaults to config)

    Returns:
        True if sent successfully, False otherwise
    
    Note:
        Duplicate prevention is strict - once an alert is sent for a specific
        event/market/outcome combination, no further alerts will be sent for it.
    """
    url = webhook_url or config.DISCORD_WEBHOOK_URL

    if not url:
        return False

    outcome_key = outcome or "all"

    # Check if should skip (no duplicates ever)
    if check_alert_sent(alert_type, event_id, market, outcome_key, ev_percent):
        logger.debug(f"Alert already sent for {event_id}/{market}/{outcome} - skipping duplicate")
        return False

    try:
        payload = {
            "content": message,
            "username": "Arb Bot",
        }

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        # Log the alert with EV
        log_alert(alert_type, event_id, market, outcome_key, ev_percent)
        logger.info(f"Discord alert sent: {alert_type} for {event_id} (EV: {ev_percent}%)")
        return True

    except requests.RequestException as e:
        logger.error(f"Failed to send Discord alert: {e}")
        return False


def format_arb_alert(opp: ArbitrageOpportunity) -> str:
    """Format an arbitrage opportunity for Discord alert."""
    timestamp = get_cst_timestamp()

    # Format point/spread if applicable
    point_str = ""
    if opp.legs[0].point is not None:
        point_str = f" ({opp.legs[0].point:+.1f})"

    # Format market name
    market_name = {
        "h2h": "Moneyline",
        "spreads": "Spread",
        "totals": "Total",
    }.get(opp.market, opp.market.upper())

    lines = [
        f"**ARBITRAGE OPPORTUNITY**",
        f"```",
        f"Game: {opp.away_team} @ {opp.home_team}",
        f"Market: {market_name}{point_str}",
        f"Profit: {opp.profit_percent:.2f}%",
        f"",
    ]

    for i, leg in enumerate(opp.legs, 1):
        lines.append(f"Leg {i}: {leg.outcome} @ {leg.bookmaker.upper()}")
        lines.append(f"        Odds: {leg.odds_american:+d} | Stake: ${leg.stake:.2f}")

    lines.extend([
        f"",
        f"Total Stake: ${opp.total_stake:.2f}",
        f"Guaranteed Return: ${opp.total_stake + opp.guaranteed_profit:.2f}",
        f"Profit: ${opp.guaranteed_profit:.2f}",
        f"```",
        f"*{timestamp}*",
    ])

    return "\n".join(lines)


def format_ev_alert(opp: EVOpportunity) -> str:
    """Format a +EV opportunity for Discord alert."""
    timestamp = get_cst_timestamp()
    game_time = format_game_time(opp.commence_time)

    # Format point if applicable
    point_str = ""
    if opp.point is not None:
        point_str = f" ({opp.point:+.1f})"

    # Format market name
    market_name = {
        "h2h": "Moneyline",
        "spreads": "Spread",
        "totals": "Total",
    }.get(opp.market, opp.market.upper())

    # Convert fair probability to fair odds
    fair_decimal = 1 / (opp.sharp_implied_prob / 100)
    fair_american = decimal_to_american(fair_decimal)

    lines = [
        f"**+EV OPPORTUNITY**",
        f"```",
        f"Game: {opp.away_team} @ {opp.home_team}",
        f"Starts: {game_time}",
        f"Market: {market_name}",
        f"",
        f"Bet: {opp.outcome}{point_str}",
        f"Book: {opp.soft_book.upper()}",
        f"Odds: {opp.soft_odds_american:+d}",
        f"",
        f"Fair Odds: {fair_american:+d} ({opp.sharp_implied_prob:.1f}%)",
        f"EV: +{opp.ev_percent:.2f}%",
        f"Units: {opp.units:.2f}",
        f"```",
        f"*{timestamp}*",
    ]

    return "\n".join(lines)


def send_startup_alert(webhook_url: Optional[str] = None) -> bool:
    """Send a startup notification to Discord."""
    url = webhook_url or getattr(config, "DISCORD_WEBHOOK_URL", None)

    if not url:
        return False

    timestamp = get_cst_timestamp()
    message = f"**Arb Bot Started**\nMonitoring for opportunities...\n*{timestamp}*"

    try:
        payload = {"content": message, "username": "Arb Bot"}
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def send_summary_alert(
    arb_count: int,
    ev_count: int,
    cycle_count: int,
    webhook_url: Optional[str] = None,
) -> bool:
    """Send a periodic summary alert."""
    url = webhook_url or getattr(config, "DISCORD_WEBHOOK_URL", None)

    if not url:
        return False

    timestamp = get_cst_timestamp()
    message = (
        f"**Hourly Summary**\n"
        f"```\n"
        f"Cycles: {cycle_count}\n"
        f"Arbs Found: {arb_count}\n"
        f"+EV Found: {ev_count}\n"
        f"```\n"
        f"*{timestamp}*"
    )

    try:
        payload = {"content": message, "username": "Arb Bot"}
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def send_arb_alert_and_log_play(opp: ArbitrageOpportunity, webhook_url: Optional[str] = None) -> bool:
    """
    Send an arb alert to Discord and log the play for tracking.

    Returns True if the alert was sent (not a duplicate).
    """
    alert_msg = format_arb_alert(opp)
    sent = send_discord_alert(
        message=alert_msg,
        alert_type="arb",
        event_id=opp.event_id,
        market=opp.market,
        outcome="arb",  # Use "arb" as outcome key for deduplication
        ev_percent=opp.profit_percent,  # Use profit_percent for comparison
        webhook_url=webhook_url,
    )

    if sent:
        # Log the arb play with leg details
        sent_at = datetime.now(timezone.utc).isoformat()
        legs = [
            {
                "bookmaker": leg.bookmaker,
                "outcome": leg.outcome,
                "odds_american": leg.odds_american,
                "odds_decimal": leg.odds_decimal,
                "point": leg.point,
                "stake": leg.stake,
                "payout": leg.payout,
            }
            for leg in opp.legs
        ]
        log_arb_play(
            event_id=opp.event_id,
            sport=opp.sport,
            home_team=opp.home_team,
            away_team=opp.away_team,
            commence_time=opp.commence_time,
            market=opp.market,
            profit_percent=opp.profit_percent,
            total_stake=opp.total_stake,
            guaranteed_profit=opp.guaranteed_profit,
            legs=legs,
            sent_at=sent_at,
        )

    return sent


def send_ev_alert_and_log_play(opp: EVOpportunity, webhook_url: Optional[str] = None) -> bool:
    """
    Send an EV alert to Discord and log the play for CLV tracking.

    Returns True if the alert was sent (not a duplicate).
    """
    alert_msg = format_ev_alert(opp)
    sent = send_discord_alert(
        message=alert_msg,
        alert_type="ev",
        event_id=opp.event_id,
        market=opp.market,
        outcome=opp.outcome,
        ev_percent=opp.ev_percent,
        webhook_url=webhook_url,
    )

    if sent:
        # Log the play for CLV tracking
        sent_at = datetime.now(timezone.utc).isoformat()
        log_play(
            event_id=opp.event_id,
            sport=opp.sport,
            home_team=opp.home_team,
            away_team=opp.away_team,
            commence_time=opp.commence_time,
            market=opp.market,
            outcome=opp.outcome,
            point=opp.point,
            soft_book=opp.soft_book,
            sent_odds_american=opp.soft_odds_american,
            sent_odds_decimal=opp.soft_odds_decimal,
            sent_ev_percent=opp.ev_percent,
            sent_edge_percent=opp.edge_percent,
            fair_prob=opp.sharp_implied_prob / 100,  # Convert back to 0-1
            kelly_fraction=opp.kelly_fraction,
            units=opp.units,
            sharp_width_cents=opp.sharp_width_cents,
            sent_at=sent_at,
        )

    return sent


# =============================================================================
# PLAYER PROPS ALERTS
# =============================================================================

def format_prop_ev_alert(opp: PropEVOpportunity) -> str:
    """Format a player prop +EV opportunity for Discord alert."""
    timestamp = get_cst_timestamp()
    game_time = format_game_time(opp.commence_time)
    prop_name = get_prop_display_name(opp.prop_type)

    # Convert fair probability to fair odds
    fair_decimal = 1 / (opp.sharp_implied_prob / 100)
    fair_american = decimal_to_american(fair_decimal)

    lines = [
        f"**+EV PLAYER PROP**",
        f"```",
        f"Game: {opp.away_team} @ {opp.home_team}",
        f"Starts: {game_time}",
        f"",
        f"Player: {opp.player_name}",
        f"Prop: {prop_name}",
        f"Bet: {opp.outcome} {opp.line}",
        f"Book: {opp.soft_book.upper()}",
        f"Odds: {opp.soft_odds_american:+d}",
        f"",
        f"Fair Odds: {fair_american:+d} ({opp.sharp_implied_prob:.1f}%)",
        f"EV: +{opp.ev_percent:.2f}%",
        f"Units: {opp.units:.2f}",
        f"```",
        f"*{timestamp}*",
    ]

    return "\n".join(lines)


def send_prop_ev_alert_and_log_play(opp: PropEVOpportunity, webhook_url: Optional[str] = None) -> bool:
    """
    Send a player prop EV alert to Discord and log the play for CLV tracking.

    Returns True if the alert was sent (not a duplicate).
    """
    alert_msg = format_prop_ev_alert(opp)

    # Create a unique outcome key for deduplication
    # Include player name, prop type, and line to distinguish different props
    outcome_key = f"{opp.player_name}|{opp.prop_type}|{opp.line}|{opp.outcome}"

    sent = send_discord_alert(
        message=alert_msg,
        alert_type="prop_ev",
        event_id=opp.event_id,
        market=opp.prop_type,
        outcome=outcome_key,
        ev_percent=opp.ev_percent,
        webhook_url=webhook_url,
    )

    if sent:
        # Log the prop play for CLV tracking
        sent_at = datetime.now(timezone.utc).isoformat()
        log_prop_play(
            event_id=opp.event_id,
            sport=opp.sport,
            home_team=opp.home_team,
            away_team=opp.away_team,
            commence_time=opp.commence_time,
            player_name=opp.player_name,
            prop_type=opp.prop_type,
            line=opp.line,
            outcome=opp.outcome,
            soft_book=opp.soft_book,
            sent_odds_american=opp.soft_odds_american,
            sent_odds_decimal=opp.soft_odds_decimal,
            sent_ev_percent=opp.ev_percent,
            sent_edge_percent=opp.edge_percent,
            fair_prob=opp.sharp_implied_prob / 100,  # Convert back to 0-1
            kelly_fraction=opp.kelly_fraction,
            units=opp.units,
            sharp_width_cents=opp.sharp_width_cents,
            sent_at=sent_at,
        )

    return sent
