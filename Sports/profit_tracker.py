"""
Profit tracking and visualization for +EV plays.
Generates graphs showing cumulative profit in units over time.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from database import get_all_plays, get_plays_profit_summary, init_database

logger = logging.getLogger(__name__)

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Install with: pip install matplotlib")


def generate_profit_graph(
    output_path: Optional[str] = None,
    days_back: Optional[int] = None,
    show_plot: bool = False,
) -> Optional[str]:
    """
    Generate a graph of cumulative profit in units over time.

    Args:
        output_path: Path to save the graph image (default: profit_graph.png)
        days_back: Only include plays from the last N days
        show_plot: Whether to display the plot interactively

    Returns:
        Path to the saved image, or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib is required for graph generation.")
        print("Install with: pip install matplotlib")
        return None

    plays = get_all_plays(days_back)

    if not plays:
        print("No plays found in database.")
        return None

    # Sort by sent_at timestamp
    plays.sort(key=lambda x: x["sent_at"])

    # Calculate cumulative profit
    dates = []
    cumulative_profits = []
    cumulative = 0.0

    for play in plays:
        sent_at = play["sent_at"]
        profit = play.get("profit_units") or 0.0

        # Parse the timestamp
        try:
            if sent_at.endswith("Z"):
                dt = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(sent_at)
        except (ValueError, AttributeError):
            continue

        cumulative += profit
        dates.append(dt)
        cumulative_profits.append(cumulative)

    if not dates:
        print("No valid play data to graph.")
        return None

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot cumulative profit
    ax.plot(dates, cumulative_profits, 'b-', linewidth=2, label='Cumulative Profit')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Fill areas
    ax.fill_between(dates, cumulative_profits, 0,
                    where=[p >= 0 for p in cumulative_profits],
                    color='green', alpha=0.2, label='Profit')
    ax.fill_between(dates, cumulative_profits, 0,
                    where=[p < 0 for p in cumulative_profits],
                    color='red', alpha=0.2, label='Loss')

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Profit (Units)', fontsize=12)
    ax.set_title('Cumulative Profit from +EV Plays', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    # Tight layout
    plt.tight_layout()

    # Save the figure
    if output_path is None:
        output_path = str(Path(__file__).parent / "profit_graph.png")

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graph saved to: {output_path}")

    if show_plot:
        plt.show()

    plt.close()
    return output_path


def print_profit_summary() -> None:
    """Print a summary of profit/loss from all plays."""
    summary = get_plays_profit_summary()

    print("\n" + "=" * 50)
    print("PROFIT SUMMARY")
    print("=" * 50)
    print(f"Total Units Wagered: {summary['total_units_wagered']:.2f}")
    print(f"Total Profit (Units): {summary['total_profit_units']:+.2f}")
    print(f"ROI: {summary['roi_percent']:+.2f}%")
    print(f"Pending Plays: {summary['pending_plays']}")
    print()

    results = summary.get("results", {})
    if results:
        print("Results Breakdown:")
        for result, data in results.items():
            print(f"  {result}: {data['count']} plays, {data['profit']:+.2f} units")
    print("=" * 50)


def print_recent_plays(limit: int = 20) -> None:
    """Print recent plays with their status."""
    plays = get_all_plays()[:limit]

    if not plays:
        print("No plays in database.")
        return

    print("\n" + "=" * 100)
    print(f"RECENT PLAYS (last {limit})")
    print("=" * 100)

    for play in plays:
        sent_at = play["sent_at"][:16].replace("T", " ")
        outcome = play["outcome"]
        soft_book = play["soft_book"].upper()
        sent_odds = play["sent_odds_american"]
        closing_odds = play.get("closing_odds_american")
        ev = play["sent_ev_percent"]
        units = play["units"]
        result = play.get("result") or "pending"
        profit = play.get("profit_units")

        # Format closing line info
        if closing_odds is not None:
            close_str = f"Close: {closing_odds:+d}"
        else:
            close_str = "Close: --"

        profit_str = f"{profit:+.2f}u" if profit is not None else ""

        print(f"{sent_at} | {outcome[:18]:18} @ {soft_book:8} | "
              f"{sent_odds:+4d} -> {close_str:12} | EV: {ev:+.1f}% | {units:.2f}u | {result:8} {profit_str}")

    print("=" * 100)


def print_clv_summary() -> None:
    """Print CLV (Closing Line Value) analysis."""
    plays = get_all_plays()

    if not plays:
        print("No plays in database.")
        return

    # Filter to plays with closing line data
    plays_with_closing = [p for p in plays if p.get("closing_odds_american") is not None]

    if not plays_with_closing:
        print("\nNo plays with closing line data yet.")
        print("Closing lines are captured when games start.")
        return

    print("\n" + "=" * 60)
    print("CLOSING LINE VALUE (CLV) ANALYSIS")
    print("=" * 60)

    total_plays = len(plays_with_closing)
    beat_close = 0
    total_clv_cents = 0

    for play in plays_with_closing:
        sent = play["sent_odds_american"]
        close = play["closing_odds_american"]

        # Calculate if we beat the closing line
        # For positive odds: higher is better
        # For negative odds: less negative (closer to 0) is better
        if sent > 0 and close > 0:
            if sent > close:
                beat_close += 1
            clv = sent - close
        elif sent < 0 and close < 0:
            if sent > close:  # e.g., -110 is better than -120
                beat_close += 1
            clv = sent - close  # Will be positive if we beat
        else:
            # Mixed signs - compare implied probabilities
            sent_implied = 100 / (sent + 100) if sent > 0 else abs(sent) / (abs(sent) + 100)
            close_implied = 100 / (close + 100) if close > 0 else abs(close) / (abs(close) + 100)
            if sent_implied < close_implied:
                beat_close += 1
            clv = int((close_implied - sent_implied) * 100)

        total_clv_cents += clv

    beat_pct = (beat_close / total_plays) * 100 if total_plays > 0 else 0
    avg_clv = total_clv_cents / total_plays if total_plays > 0 else 0

    print(f"Plays with closing line: {total_plays}")
    print(f"Beat closing line: {beat_close} ({beat_pct:.1f}%)")
    print(f"Average CLV: {avg_clv:+.1f} cents")
    print()
    print("Note: Positive CLV means you got better odds than the market closing price.")
    print("Consistently beating the closing line indicates +EV betting skill.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profit Tracking for +EV Plays")
    parser.add_argument("--graph", action="store_true", help="Generate profit graph")
    parser.add_argument("--summary", action="store_true", help="Print profit summary")
    parser.add_argument("--clv", action="store_true", help="Print CLV analysis")
    parser.add_argument("--plays", type=int, default=0, help="Print recent N plays")
    parser.add_argument("--days", type=int, help="Only include last N days")
    parser.add_argument("--output", type=str, help="Output path for graph")
    parser.add_argument("--show", action="store_true", help="Show graph interactively")

    args = parser.parse_args()

    # Initialize database
    init_database()

    # Default to summary if no args specified
    if args.summary or (not args.graph and not args.clv and args.plays == 0):
        print_profit_summary()

    if args.clv:
        print_clv_summary()

    if args.plays > 0:
        print_recent_plays(args.plays)

    if args.graph:
        generate_profit_graph(
            output_path=args.output,
            days_back=args.days,
            show_plot=args.show,
        )
