"""
Analysis: Compare profitability of plays alerted >4 hours before game start
vs those alerted within 4 hours of game start.
"""
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "odds_history.db"

def analyze_alert_timing():
    """Analyze profitability by alert timing relative to game start."""
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=" * 80)
    print("ALERT TIMING ANALYSIS")
    print("Comparing plays alerted >4 hours before game vs within 4 hours")
    print("=" * 80)
    
    # Get all completed EV plays
    cursor.execute("""
        SELECT 
            id,
            event_id,
            sport,
            home_team,
            away_team,
            market,
            outcome,
            sent_at,
            commence_time,
            sent_odds_american,
            sent_odds_decimal,
            sent_ev_percent,
            units,
            result,
            profit_units,
            soft_book
        FROM plays
        WHERE result IS NOT NULL
        ORDER BY sent_at
    """)
    
    ev_plays = [dict(row) for row in cursor.fetchall()]
    
    # Get all completed prop plays
    cursor.execute("""
        SELECT 
            id,
            event_id,
            sport,
            home_team,
            away_team,
            player_name,
            prop_type,
            line,
            outcome,
            sent_at,
            commence_time,
            sent_odds_american,
            sent_odds_decimal,
            sent_ev_percent,
            units,
            result,
            profit_units,
            soft_book
        FROM prop_plays
        WHERE result IS NOT NULL
        ORDER BY sent_at
    """)
    
    prop_plays = [dict(row) for row in cursor.fetchall()]
    
    all_plays = []
    for p in ev_plays:
        p['type'] = 'game_line'
        all_plays.append(p)
    for p in prop_plays:
        p['type'] = 'prop'
        all_plays.append(p)
    
    print(f"\nTotal completed plays: {len(all_plays)}")
    print(f"  - Game Lines: {len(ev_plays)}")
    print(f"  - Props: {len(prop_plays)}")
    
    if len(all_plays) == 0:
        print("\n[!] No completed plays found!")
        return
    
    # Analyze by timing
    early_plays = []  # >4 hours before game start
    late_plays = []   # <=4 hours before game start
    
    for play in all_plays:
        sent_at_str = play['sent_at']
        commence_time_str = play['commence_time']
        
        try:
            sent_at = datetime.fromisoformat(sent_at_str.replace('Z', '+00:00'))
            if sent_at.tzinfo is None:
                sent_at = sent_at.replace(tzinfo=timezone.utc)
            
            commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
            if commence_time.tzinfo is None:
                commence_time = commence_time.replace(tzinfo=timezone.utc)
            
            time_diff = commence_time - sent_at
            hours_before = time_diff.total_seconds() / 3600
            
            play['hours_before_game'] = hours_before
            
            if hours_before > 4:
                early_plays.append(play)
            else:
                late_plays.append(play)
        except Exception as e:
            print(f"Warning: Could not parse dates for play {play.get('id')}: {e}")
            continue
    
    print(f"\n" + "=" * 80)
    print("PLAYS BY TIMING")
    print("=" * 80)
    print(f"\nEarly Alerts (>4 hours before game): {len(early_plays)}")
    print(f"Late Alerts (<=4 hours before game): {len(late_plays)}")
    
    # Analyze early plays
    if early_plays:
        early_units = sum(p['units'] for p in early_plays)
        early_profit = sum(p.get('profit_units') or 0 for p in early_plays)
        early_roi = (early_profit / early_units * 100) if early_units > 0 else 0
        
        early_wins = len([p for p in early_plays if p['result'] == 'win'])
        early_losses = len([p for p in early_plays if p['result'] == 'loss'])
        early_pushes = len([p for p in early_plays if p['result'] == 'push'])
        early_total = early_wins + early_losses + early_pushes
        early_wr = (early_wins / early_total * 100) if early_total > 0 else 0
        
        early_avg_ev = sum(p.get('sent_ev_percent', 0) for p in early_plays) / len(early_plays) if early_plays else 0
        early_avg_hours = sum(p['hours_before_game'] for p in early_plays) / len(early_plays) if early_plays else 0
        
        print(f"\n" + "=" * 80)
        print("EARLY ALERTS (>4 hours before game start)")
        print("=" * 80)
        print(f"Count: {len(early_plays)}")
        print(f"Average Hours Before Game: {early_avg_hours:.1f}")
        print(f"Total Units: {early_units:.2f}")
        print(f"Total Profit: {early_profit:.2f} units")
        print(f"ROI: {early_roi:.2f}%")
        print(f"\nResults:")
        print(f"  Wins: {early_wins} ({early_wr:.1f}%)")
        print(f"  Losses: {early_losses} ({early_losses/early_total*100:.1f}%)")
        print(f"  Pushes: {early_pushes} ({early_pushes/early_total*100:.1f}%)")
        print(f"\nAverage EV: {early_avg_ev:.2f}%")
    
    # Analyze late plays
    if late_plays:
        late_units = sum(p['units'] for p in late_plays)
        late_profit = sum(p.get('profit_units') or 0 for p in late_plays)
        late_roi = (late_profit / late_units * 100) if late_units > 0 else 0
        
        late_wins = len([p for p in late_plays if p['result'] == 'win'])
        late_losses = len([p for p in late_plays if p['result'] == 'loss'])
        late_pushes = len([p for p in late_plays if p['result'] == 'push'])
        late_total = late_wins + late_losses + late_pushes
        late_wr = (late_wins / late_total * 100) if late_total > 0 else 0
        
        late_avg_ev = sum(p.get('sent_ev_percent', 0) for p in late_plays) / len(late_plays) if late_plays else 0
        late_avg_hours = sum(p['hours_before_game'] for p in late_plays) / len(late_plays) if late_plays else 0
        
        print(f"\n" + "=" * 80)
        print("LATE ALERTS (<=4 hours before game start)")
        print("=" * 80)
        print(f"Count: {len(late_plays)}")
        print(f"Average Hours Before Game: {late_avg_hours:.1f}")
        print(f"Total Units: {late_units:.2f}")
        print(f"Total Profit: {late_profit:.2f} units")
        print(f"ROI: {late_roi:.2f}%")
        print(f"\nResults:")
        print(f"  Wins: {late_wins} ({late_wr:.1f}%)")
        print(f"  Losses: {late_losses} ({late_losses/late_total*100:.1f}%)")
        print(f"  Pushes: {late_pushes} ({late_pushes/late_total*100:.1f}%)")
        print(f"\nAverage EV: {late_avg_ev:.2f}%")
    
    # Comparison
    if early_plays and late_plays:
        print(f"\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"\nROI Difference: {early_roi - late_roi:+.2f}%")
        print(f"  Early: {early_roi:.2f}%")
        print(f"  Late:  {late_roi:.2f}%")
        
        print(f"\nWin Rate Difference: {early_wr - late_wr:+.1f}%")
        print(f"  Early: {early_wr:.1f}%")
        print(f"  Late:  {late_wr:.1f}%")
        
        print(f"\nAverage EV Difference: {early_avg_ev - late_avg_ev:+.2f}%")
        print(f"  Early: {early_avg_ev:.2f}%")
        print(f"  Late:  {late_avg_ev:.2f}%")
        
        profit_diff = early_profit - late_profit
        print(f"\nProfit Difference: {profit_diff:+.2f} units")
        print(f"  Early: {early_profit:+.2f} units")
        print(f"  Late:  {late_profit:+.2f} units")
    
    # Breakdown by type
    print(f"\n" + "=" * 80)
    print("BREAKDOWN BY PLAY TYPE")
    print("=" * 80)
    
    early_game_lines = [p for p in early_plays if p['type'] == 'game_line']
    early_props = [p for p in early_plays if p['type'] == 'prop']
    late_game_lines = [p for p in late_plays if p['type'] == 'game_line']
    late_props = [p for p in late_plays if p['type'] == 'prop']
    
    for label, plays in [
        ("Early Game Lines", early_game_lines),
        ("Early Props", early_props),
        ("Late Game Lines", late_game_lines),
        ("Late Props", late_props)
    ]:
        if not plays:
            continue
        
        units = sum(p['units'] for p in plays)
        profit = sum(p.get('profit_units') or 0 for p in plays)
        roi = (profit / units * 100) if units > 0 else 0
        wins = len([p for p in plays if p['result'] == 'win'])
        total = len([p for p in plays if p['result'] in ['win', 'loss', 'push']])
        wr = (wins / total * 100) if total > 0 else 0
        
        print(f"\n{label}: {len(plays)} plays")
        print(f"  Units: {units:.2f} | Profit: {profit:+.2f} units | ROI: {roi:.2f}%")
        print(f"  Win Rate: {wr:.1f}% ({wins}W / {total}T)")
    
    # Detailed examples
    print(f"\n" + "=" * 80)
    print("EARLY ALERT EXAMPLES (Best & Worst)")
    print("=" * 80)
    
    if early_plays:
        early_sorted = sorted(early_plays, key=lambda x: x.get('profit_units') or 0, reverse=True)
        print("\nTop 5 Early Alerts (by profit):")
        for i, p in enumerate(early_sorted[:5], 1):
            if p['type'] == 'game_line':
                desc = f"{p.get('market', 'N/A')} {p.get('outcome', 'N/A')}"
            else:
                desc = f"{p.get('player_name', '?')} {p.get('prop_type', 'N/A')} {p.get('outcome', 'N/A')} {p.get('line', 0)}"
            
            hours = p['hours_before_game']
            result = p['result']
            profit = p.get('profit_units', 0) or 0
            game = f"{p.get('away_team', '?')} @ {p.get('home_team', '?')}"
            
            print(f"  {i}. {game} | {desc}")
            print(f"     {hours:.1f}h before | {result.upper()} | {profit:+.2f}u")
        
        print("\nWorst 5 Early Alerts (by loss):")
        early_worst = sorted(early_plays, key=lambda x: x.get('profit_units') or 0)
        for i, p in enumerate(early_worst[:5], 1):
            if p['type'] == 'game_line':
                desc = f"{p.get('market', 'N/A')} {p.get('outcome', 'N/A')}"
            else:
                desc = f"{p.get('player_name', '?')} {p.get('prop_type', 'N/A')} {p.get('outcome', 'N/A')} {p.get('line', 0)}"
            
            hours = p['hours_before_game']
            result = p['result']
            profit = p.get('profit_units', 0) or 0
            game = f"{p.get('away_team', '?')} @ {p.get('home_team', '?')}"
            
            print(f"  {i}. {game} | {desc}")
            print(f"     {hours:.1f}h before | {result.upper()} | {profit:+.2f}u")
    
    print(f"\n" + "=" * 80)
    print("LATE ALERT EXAMPLES (Best & Worst)")
    print("=" * 80)
    
    if late_plays:
        late_sorted = sorted(late_plays, key=lambda x: x.get('profit_units') or 0, reverse=True)
        print("\nTop 5 Late Alerts (by profit):")
        for i, p in enumerate(late_sorted[:5], 1):
            if p['type'] == 'game_line':
                desc = f"{p.get('market', 'N/A')} {p.get('outcome', 'N/A')}"
            else:
                desc = f"{p.get('player_name', '?')} {p.get('prop_type', 'N/A')} {p.get('outcome', 'N/A')} {p.get('line', 0)}"
            
            hours = p['hours_before_game']
            result = p['result']
            profit = p.get('profit_units', 0) or 0
            game = f"{p.get('away_team', '?')} @ {p.get('home_team', '?')}"
            
            print(f"  {i}. {game} | {desc}")
            print(f"     {hours:.1f}h before | {result.upper()} | {profit:+.2f}u")
        
        print("\nWorst 5 Late Alerts (by loss):")
        late_worst = sorted(late_plays, key=lambda x: x.get('profit_units') or 0)
        for i, p in enumerate(late_worst[:5], 1):
            if p['type'] == 'game_line':
                desc = f"{p.get('market', 'N/A')} {p.get('outcome', 'N/A')}"
            else:
                desc = f"{p.get('player_name', '?')} {p.get('prop_type', 'N/A')} {p.get('outcome', 'N/A')} {p.get('line', 0)}"
            
            hours = p['hours_before_game']
            result = p['result']
            profit = p.get('profit_units', 0) or 0
            game = f"{p.get('away_team', '?')} @ {p.get('home_team', '?')}"
            
            print(f"  {i}. {game} | {desc}")
            print(f"     {hours:.1f}h before | {result.upper()} | {profit:+.2f}u")
    
    # Time distribution analysis
    print(f"\n" + "=" * 80)
    print("TIME DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    time_buckets = {
        "0-1h": [],
        "1-2h": [],
        "2-4h": [],
        "4-8h": [],
        "8-12h": [],
        "12-24h": [],
        "24h+": []
    }
    
    for play in all_plays:
        hours = play.get('hours_before_game', 0)
        if hours <= 1:
            time_buckets["0-1h"].append(play)
        elif hours <= 2:
            time_buckets["1-2h"].append(play)
        elif hours <= 4:
            time_buckets["2-4h"].append(play)
        elif hours <= 8:
            time_buckets["4-8h"].append(play)
        elif hours <= 12:
            time_buckets["8-12h"].append(play)
        elif hours <= 24:
            time_buckets["12-24h"].append(play)
        else:
            time_buckets["24h+"].append(play)
    
    for bucket, plays in time_buckets.items():
        if not plays:
            continue
        
        units = sum(p['units'] for p in plays)
        profit = sum(p.get('profit_units') or 0 for p in plays)
        roi = (profit / units * 100) if units > 0 else 0
        wins = len([p for p in plays if p['result'] == 'win'])
        total = len([p for p in plays if p['result'] in ['win', 'loss', 'push']])
        wr = (wins / total * 100) if total > 0 else 0
        
        print(f"\n{bucket} before game: {len(plays)} plays")
        print(f"  Units: {units:.2f} | Profit: {profit:+.2f} units | ROI: {roi:+.2f}% | WR: {wr:.1f}%")
    
    conn.close()
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    analyze_alert_timing()
