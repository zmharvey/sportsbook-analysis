"""
SQLite database for storing odds history.
Supports time-series analysis and +EV detection.
"""

import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

# CST/CDT timezone (Central Time - handles DST automatically)
try:
    from zoneinfo import ZoneInfo
    CST_TZ = ZoneInfo("America/Chicago")  # Automatically handles CST/CDT
except ImportError:
    # Fallback for Python < 3.9 - use UTC-6 (no DST)
    CST_TZ = timezone(timedelta(hours=-6))

def get_cst_now():
    """Get current time in Central timezone (CST/CDT)."""
    return datetime.now(CST_TZ)

def get_cst_today_start():
    """Get start of today (midnight) in Central timezone, returned as UTC ISO string for database queries."""
    cst_now = get_cst_now()
    cst_midnight = cst_now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Convert to UTC for database queries (database stores UTC)
    utc_midnight = cst_midnight.astimezone(timezone.utc)
    return utc_midnight.isoformat()

def get_cst_week_start():
    """Get start of 7 days ago in Central timezone, returned as UTC ISO string for database queries."""
    cst_now = get_cst_now()
    cst_week_ago = cst_now - timedelta(days=7)
    # Convert to UTC for database queries
    utc_week_ago = cst_week_ago.astimezone(timezone.utc)
    return utc_week_ago.isoformat()

def cst_date_from_iso(iso_string):
    """Convert UTC ISO string from database to Central timezone datetime."""
    if not iso_string:
        return None
    dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(CST_TZ)

logger = logging.getLogger(__name__)

# Database file location
DB_PATH = Path(__file__).parent / "odds_history.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)  # 30 second timeout for concurrent access
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database() -> None:
    """Initialize the database schema."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Events table - stores unique games/matches
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                sport TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                commence_time TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Odds snapshots - stores odds at each poll interval
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                bookmaker TEXT NOT NULL,
                market TEXT NOT NULL,
                outcome TEXT NOT NULL,
                price_american INTEGER NOT NULL,
                price_decimal REAL NOT NULL,
                point REAL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (event_id) REFERENCES events(id)
            )
        """)

        # Create indexes for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_event_time
            ON odds_snapshots(event_id, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp
            ON odds_snapshots(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_bookmaker
            ON odds_snapshots(bookmaker, market, outcome)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_sport
            ON events(sport)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_commence
            ON events(commence_time)
        """)

        # Alerts log - track sent alerts with EV for re-alerting on higher EV
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                event_id TEXT NOT NULL,
                market TEXT NOT NULL,
                outcome TEXT NOT NULL,
                ev_percent REAL,
                sent_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_lookup
            ON alerts_log(alert_type, event_id, market, outcome)
        """)

        # Plays tracking - stores all sent plays for CLV analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plays (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                sport TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                commence_time TEXT NOT NULL,
                market TEXT NOT NULL,
                outcome TEXT NOT NULL,
                point REAL,
                soft_book TEXT NOT NULL,
                sent_odds_american INTEGER NOT NULL,
                sent_odds_decimal REAL NOT NULL,
                sent_ev_percent REAL NOT NULL,
                sent_edge_percent REAL NOT NULL,
                fair_prob REAL NOT NULL,
                kelly_fraction REAL NOT NULL,
                units REAL NOT NULL,
                sharp_width_cents INTEGER,
                sent_at TEXT NOT NULL,
                closing_odds_american INTEGER,
                closing_odds_decimal REAL,
                closing_ev_percent REAL,
                result TEXT,
                profit_units REAL,
                FOREIGN KEY (event_id) REFERENCES events(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_plays_event
            ON plays(event_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_plays_sent_at
            ON plays(sent_at)
        """)

        # Arb plays tracking - stores arbitrage opportunities with leg details
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arb_plays (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                sport TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                commence_time TEXT NOT NULL,
                market TEXT NOT NULL,
                profit_percent REAL NOT NULL,
                total_stake REAL NOT NULL,
                guaranteed_profit REAL NOT NULL,
                legs_json TEXT NOT NULL,
                sent_at TEXT NOT NULL,
                FOREIGN KEY (event_id) REFERENCES events(id)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_arb_plays_event
            ON arb_plays(event_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_arb_plays_sent_at
            ON arb_plays(sent_at)
        """)

        # =================================================================
        # PLAYER PROPS TABLES
        # =================================================================

        # Prop snapshots - stores player prop odds at each poll interval
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prop_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                bookmaker TEXT NOT NULL,
                prop_type TEXT NOT NULL,
                player_name TEXT NOT NULL,
                outcome TEXT NOT NULL,
                line REAL NOT NULL,
                price_american INTEGER NOT NULL,
                price_decimal REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (event_id) REFERENCES events(id)
            )
        """)

        # Indexes for prop snapshots
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prop_snapshots_event_time
            ON prop_snapshots(event_id, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prop_snapshots_player
            ON prop_snapshots(player_name, prop_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prop_snapshots_lookup
            ON prop_snapshots(event_id, prop_type, player_name, line, outcome)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prop_snapshots_timestamp
            ON prop_snapshots(timestamp)
        """)

        # Prop plays - stores +EV prop plays for CLV tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prop_plays (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                sport TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                commence_time TEXT NOT NULL,
                player_name TEXT NOT NULL,
                prop_type TEXT NOT NULL,
                line REAL NOT NULL,
                outcome TEXT NOT NULL,
                soft_book TEXT NOT NULL,
                sent_odds_american INTEGER NOT NULL,
                sent_odds_decimal REAL NOT NULL,
                sent_ev_percent REAL NOT NULL,
                sent_edge_percent REAL NOT NULL,
                fair_prob REAL NOT NULL,
                kelly_fraction REAL NOT NULL,
                units REAL NOT NULL,
                sharp_width_cents INTEGER,
                sent_at TEXT NOT NULL,
                closing_odds_american INTEGER,
                closing_odds_decimal REAL,
                closing_ev_percent REAL,
                closing_line REAL,
                result TEXT,
                profit_units REAL,
                FOREIGN KEY (event_id) REFERENCES events(id)
            )
        """)
        
        # Migration: Add closing_line column if it doesn't exist (for existing databases)
        cursor.execute("PRAGMA table_info(prop_plays)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'closing_line' not in columns:
            cursor.execute("ALTER TABLE prop_plays ADD COLUMN closing_line REAL")
            logger.info("Added closing_line column to prop_plays table")
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prop_plays_event
            ON prop_plays(event_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prop_plays_sent_at
            ON prop_plays(sent_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prop_plays_player
            ON prop_plays(player_name, prop_type)
        """)

        logger.info(f"Database initialized at {DB_PATH}")


def upsert_event(
    event_id: str,
    sport: str,
    home_team: str,
    away_team: str,
    commence_time: str,
) -> None:
    """Insert or update an event."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO events (id, sport, home_team, away_team, commence_time)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                sport = excluded.sport,
                home_team = excluded.home_team,
                away_team = excluded.away_team,
                commence_time = excluded.commence_time
        """, (event_id, sport, home_team, away_team, commence_time))


def insert_odds_snapshot(
    event_id: str,
    bookmaker: str,
    market: str,
    outcome: str,
    price_american: int,
    price_decimal: float,
    point: Optional[float],
    timestamp: str,
) -> None:
    """Insert a single odds snapshot."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO odds_snapshots
            (event_id, bookmaker, market, outcome, price_american, price_decimal, point, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (event_id, bookmaker, market, outcome, price_american, price_decimal, point, timestamp))


def bulk_insert_odds(snapshots: list[dict]) -> None:
    """Bulk insert odds snapshots for efficiency."""
    if not snapshots:
        return

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO odds_snapshots
            (event_id, bookmaker, market, outcome, price_american, price_decimal, point, timestamp)
            VALUES (:event_id, :bookmaker, :market, :outcome, :price_american, :price_decimal, :point, :timestamp)
        """, snapshots)
        logger.debug(f"Inserted {len(snapshots)} odds snapshots")


def bulk_insert_prop_snapshots(snapshots: list[dict]) -> None:
    """Bulk insert player prop snapshots for efficiency."""
    if not snapshots:
        return

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO prop_snapshots
            (event_id, bookmaker, prop_type, player_name, outcome, line, price_american, price_decimal, timestamp)
            VALUES (:event_id, :bookmaker, :prop_type, :player_name, :outcome, :line, :price_american, :price_decimal, :timestamp)
        """, snapshots)
        logger.debug(f"Inserted {len(snapshots)} prop snapshots")


def get_latest_odds_for_event(event_id: str, market: str) -> list[dict]:
    """Get the most recent odds for an event/market from each bookmaker."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                s.bookmaker,
                s.outcome,
                s.price_american,
                s.price_decimal,
                s.point,
                s.timestamp
            FROM odds_snapshots s
            INNER JOIN (
                SELECT bookmaker, outcome, point, MAX(timestamp) as max_ts
                FROM odds_snapshots
                WHERE event_id = ? AND market = ?
                GROUP BY bookmaker, outcome, point
            ) latest ON s.bookmaker = latest.bookmaker
                AND s.outcome = latest.outcome
                AND (s.point = latest.point OR (s.point IS NULL AND latest.point IS NULL))
                AND s.timestamp = latest.max_ts
            WHERE s.event_id = ? AND s.market = ?
        """, (event_id, market, event_id, market))

        return [dict(row) for row in cursor.fetchall()]


def get_odds_history(
    event_id: str,
    market: str,
    bookmaker: Optional[str] = None,
    outcome: Optional[str] = None,
    point: Optional[float] = None,
    hours_back: int = 24,
) -> list[dict]:
    """Get historical odds for charting/analysis."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()

    with get_db() as conn:
        cursor = conn.cursor()

        query = """
            SELECT bookmaker, outcome, price_american, price_decimal, point, timestamp
            FROM odds_snapshots
            WHERE event_id = ? AND market = ? AND timestamp > ?
        """
        params = [event_id, market, cutoff]

        if bookmaker:
            query += " AND bookmaker = ?"
            params.append(bookmaker)
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome)
        if point is not None:
            query += " AND point = ?"
            params.append(point)

        query += " ORDER BY timestamp ASC"

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_upcoming_events(sport: Optional[str] = None) -> list[dict]:
    """Get events that haven't started yet."""
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cursor = conn.cursor()

        if sport:
            cursor.execute("""
                SELECT * FROM events
                WHERE commence_time > ? AND sport = ?
                ORDER BY commence_time ASC
            """, (now, sport))
        else:
            cursor.execute("""
                SELECT * FROM events
                WHERE commence_time > ?
                ORDER BY commence_time ASC
            """, (now,))

        return [dict(row) for row in cursor.fetchall()]


def get_event_by_id(event_id: str) -> Optional[dict]:
    """Get a single event by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def cleanup_old_data(days: int = 7) -> int:
    """Remove data older than specified days. Returns count of deleted rows."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    with get_db() as conn:
        cursor = conn.cursor()

        # Delete old snapshots
        cursor.execute("DELETE FROM odds_snapshots WHERE timestamp < ?", (cutoff,))
        deleted_snapshots = cursor.rowcount

        # Delete old alerts
        cursor.execute("DELETE FROM alerts_log WHERE sent_at < ?", (cutoff,))

        # Delete events with no remaining snapshots and past commence time
        cursor.execute("""
            DELETE FROM events
            WHERE id NOT IN (SELECT DISTINCT event_id FROM odds_snapshots)
            AND commence_time < ?
        """, (cutoff,))
        deleted_events = cursor.rowcount

        logger.info(f"Cleanup: removed {deleted_snapshots} snapshots, {deleted_events} events")
        return deleted_snapshots


def get_max_ev_sent(alert_type: str, event_id: str, market: str, outcome: str) -> Optional[float]:
    """Get the maximum EV% already sent for this opportunity."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(ev_percent) FROM alerts_log
            WHERE alert_type = ? AND event_id = ? AND market = ? AND outcome = ?
        """, (alert_type, event_id, market, outcome))
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else None


def check_alert_sent(alert_type: str, event_id: str, market: str, outcome: str, ev_percent: Optional[float] = None) -> bool:
    """
    Check if an alert should be skipped (already sent).

    Returns True if an alert was already sent for this exact play.
    No duplicates are ever sent - once alerted, always skipped.
    """
    max_ev = get_max_ev_sent(alert_type, event_id, market, outcome)

    if max_ev is None:
        return False  # Never sent, should send

    # Already sent - skip (no duplicates ever)
    return True


def log_alert(alert_type: str, event_id: str, market: str, outcome: str, ev_percent: Optional[float] = None) -> None:
    """Log that an alert was sent."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts_log (alert_type, event_id, market, outcome, ev_percent)
            VALUES (?, ?, ?, ?, ?)
        """, (alert_type, event_id, market, outcome, ev_percent))


def log_play(
    event_id: str,
    sport: str,
    home_team: str,
    away_team: str,
    commence_time: str,
    market: str,
    outcome: str,
    point: Optional[float],
    soft_book: str,
    sent_odds_american: int,
    sent_odds_decimal: float,
    sent_ev_percent: float,
    sent_edge_percent: float,
    fair_prob: float,
    kelly_fraction: float,
    units: float,
    sharp_width_cents: Optional[int],
    sent_at: str,
) -> int:
    """Log a play for CLV tracking. Returns the play ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO plays (
                event_id, sport, home_team, away_team, commence_time,
                market, outcome, point, soft_book,
                sent_odds_american, sent_odds_decimal,
                sent_ev_percent, sent_edge_percent, fair_prob,
                kelly_fraction, units, sharp_width_cents, sent_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id, sport, home_team, away_team, commence_time,
            market, outcome, point, soft_book,
            sent_odds_american, sent_odds_decimal,
            sent_ev_percent, sent_edge_percent, fair_prob,
            kelly_fraction, units, sharp_width_cents, sent_at
        ))
        return cursor.lastrowid


def log_arb_play(
    event_id: str,
    sport: str,
    home_team: str,
    away_team: str,
    commence_time: str,
    market: str,
    profit_percent: float,
    total_stake: float,
    guaranteed_profit: float,
    legs: list[dict],
    sent_at: str,
) -> int:
    """Log an arbitrage play with leg details. Returns the arb play ID."""
    import json
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO arb_plays (
                event_id, sport, home_team, away_team, commence_time,
                market, profit_percent, total_stake, guaranteed_profit,
                legs_json, sent_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id, sport, home_team, away_team, commence_time,
            market, profit_percent, total_stake, guaranteed_profit,
            json.dumps(legs), sent_at
        ))
        return cursor.lastrowid


def log_prop_play(
    event_id: str,
    sport: str,
    home_team: str,
    away_team: str,
    commence_time: str,
    player_name: str,
    prop_type: str,
    line: float,
    outcome: str,
    soft_book: str,
    sent_odds_american: int,
    sent_odds_decimal: float,
    sent_ev_percent: float,
    sent_edge_percent: float,
    fair_prob: float,
    kelly_fraction: float,
    units: float,
    sharp_width_cents: Optional[int],
    sent_at: str,
) -> int:
    """Log a player prop play for CLV tracking. Returns the play ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prop_plays (
                event_id, sport, home_team, away_team, commence_time,
                player_name, prop_type, line, outcome, soft_book,
                sent_odds_american, sent_odds_decimal,
                sent_ev_percent, sent_edge_percent, fair_prob,
                kelly_fraction, units, sharp_width_cents, sent_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id, sport, home_team, away_team, commence_time,
            player_name, prop_type, line, outcome, soft_book,
            sent_odds_american, sent_odds_decimal,
            sent_ev_percent, sent_edge_percent, fair_prob,
            kelly_fraction, units, sharp_width_cents, sent_at
        ))
        return cursor.lastrowid


def get_prop_plays_needing_closing_line() -> list[dict]:
    """Get prop plays that need closing line data (game has started)."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM prop_plays
            WHERE closing_odds_american IS NULL
            AND commence_time < ?
        """, (now,))
        return [dict(row) for row in cursor.fetchall()]


def get_prop_odds_history(
    event_id: str,
    player_name: str,
    prop_type: str,
    line: Optional[float] = None,
    hours_back: int = 48,
) -> list[dict]:
    """Get historical prop odds for charting/analysis."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()

    with get_db() as conn:
        cursor = conn.cursor()

        query = """
            SELECT bookmaker, player_name, prop_type, outcome, line, 
                   price_american, price_decimal, timestamp
            FROM prop_snapshots
            WHERE event_id = ? AND player_name = ? AND prop_type = ? AND timestamp > ?
        """
        params = [event_id, player_name, prop_type, cutoff]

        if line is not None:
            query += " AND line = ?"
            params.append(line)

        query += " ORDER BY timestamp ASC"

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_closing_line_for_prop_play(play: dict) -> Optional[dict]:
    """
    Get the closing line (last odds before game start) for a prop play.
    
    First tries exact line match, then falls back to closest available line.

    Returns dict with closing odds or None if not found.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # First try exact line match at the same book
        cursor.execute("""
            SELECT price_american, price_decimal, line, timestamp
            FROM prop_snapshots
            WHERE event_id = ?
            AND bookmaker = ?
            AND prop_type = ?
            AND player_name = ?
            AND line = ?
            AND outcome = ?
            AND timestamp < ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (
            play["event_id"],
            play["soft_book"],
            play["prop_type"],
            play["player_name"],
            play["line"],
            play["outcome"],
            play["commence_time"],
        ))

        row = cursor.fetchone()
        
        # If no exact match, try closest line at the same book
        if not row:
            cursor.execute("""
                SELECT price_american, price_decimal, line, timestamp
                FROM prop_snapshots
                WHERE event_id = ?
                AND bookmaker = ?
                AND prop_type = ?
                AND player_name = ?
                AND outcome = ?
                AND timestamp < ?
                ORDER BY ABS(line - ?) ASC, timestamp DESC
                LIMIT 1
            """, (
                play["event_id"],
                play["soft_book"],
                play["prop_type"],
                play["player_name"],
                play["outcome"],
                play["commence_time"],
                play["line"],
            ))
            row = cursor.fetchone()
        
        # If still no match, try any bookmaker at closest line
        if not row:
            cursor.execute("""
                SELECT price_american, price_decimal, line, timestamp
                FROM prop_snapshots
                WHERE event_id = ?
                AND prop_type = ?
                AND player_name = ?
                AND outcome = ?
                AND timestamp < ?
                ORDER BY ABS(line - ?) ASC, timestamp DESC
                LIMIT 1
            """, (
                play["event_id"],
                play["prop_type"],
                play["player_name"],
                play["outcome"],
                play["commence_time"],
                play["line"],
            ))
            row = cursor.fetchone()
        
        if row:
            return {
                "closing_odds_american": row["price_american"],
                "closing_odds_decimal": row["price_decimal"],
                "closing_line": row["line"],
                "timestamp": row["timestamp"],
            }
        return None


def update_prop_play_closing_line(
    play_id: int,
    closing_odds_american: int,
    closing_odds_decimal: float,
    closing_ev_percent: float,
    closing_line: Optional[float] = None,
) -> None:
    """Update a prop play with closing line data."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE prop_plays SET
                closing_odds_american = ?,
                closing_odds_decimal = ?,
                closing_ev_percent = ?,
                closing_line = ?
            WHERE id = ?
        """, (closing_odds_american, closing_odds_decimal, closing_ev_percent, closing_line, play_id))


def update_prop_play_result(play_id: int, result: str, profit_units: float) -> None:
    """Update a prop play with the result (win/loss/push)."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE prop_plays SET result = ?, profit_units = ?
            WHERE id = ?
        """, (result, profit_units, play_id))


def process_prop_closing_lines() -> int:
    """
    Process all prop plays needing closing lines.

    Returns the number of plays updated.
    """
    plays = get_prop_plays_needing_closing_line()
    updated = 0

    for play in plays:
        closing = get_closing_line_for_prop_play(play)

        if closing:
            # Calculate closing EV using the stored fair_prob
            fair_prob = play["fair_prob"]
            closing_decimal = closing["closing_odds_decimal"]
            closing_ev = (fair_prob * closing_decimal - 1) * 100
            
            # Get the closing line (may differ from original if line moved)
            closing_line = closing.get("closing_line")

            update_prop_play_closing_line(
                play_id=play["id"],
                closing_odds_american=closing["closing_odds_american"],
                closing_odds_decimal=closing["closing_odds_decimal"],
                closing_ev_percent=closing_ev,
                closing_line=closing_line,
            )
            updated += 1
            
            # Log with line info if it differs
            line_info = ""
            if closing_line is not None and closing_line != play["line"]:
                line_info = f" (line moved: {play['line']} → {closing_line})"
            logger.info(
                f"Prop closing line captured for play {play['id']}: "
                f"{play['player_name']} {play['outcome']} {play['line']} @ {play['soft_book']} - "
                f"Sent: {play['sent_odds_american']:+d}, Close: {closing['closing_odds_american']:+d}{line_info}"
            )

    return updated


def get_all_prop_plays(days_back: Optional[int] = None) -> list[dict]:
    """Get all prop plays, optionally filtered by days back."""
    with get_db() as conn:
        cursor = conn.cursor()
        if days_back:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
            cursor.execute("SELECT * FROM prop_plays WHERE sent_at > ? ORDER BY sent_at DESC", (cutoff,))
        else:
            cursor.execute("SELECT * FROM prop_plays ORDER BY sent_at DESC")
        return [dict(row) for row in cursor.fetchall()]


def get_prop_plays_profit_summary() -> dict:
    """Get summary of prop play profit in units."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Total units wagered
        cursor.execute("SELECT SUM(units) FROM prop_plays")
        total_units = cursor.fetchone()[0] or 0

        # Total profit (from completed plays)
        cursor.execute("SELECT SUM(profit_units) FROM prop_plays WHERE result IS NOT NULL")
        total_profit = cursor.fetchone()[0] or 0

        # Count by result
        cursor.execute("""
            SELECT result, COUNT(*), SUM(profit_units)
            FROM prop_plays WHERE result IS NOT NULL
            GROUP BY result
        """)
        results = {row[0]: {"count": row[1], "profit": row[2]} for row in cursor.fetchall()}

        # Pending plays
        cursor.execute("SELECT COUNT(*) FROM prop_plays WHERE result IS NULL")
        pending = cursor.fetchone()[0]

        return {
            "total_units_wagered": total_units,
            "total_profit_units": total_profit,
            "results": results,
            "pending_plays": pending,
            "roi_percent": (total_profit / total_units * 100) if total_units > 0 else 0,
        }


def cleanup_old_prop_data(days: int = 7) -> int:
    """Remove prop data older than specified days. Returns count of deleted rows."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    with get_db() as conn:
        cursor = conn.cursor()

        # Delete old prop snapshots
        cursor.execute("DELETE FROM prop_snapshots WHERE timestamp < ?", (cutoff,))
        deleted = cursor.rowcount

        logger.info(f"Prop cleanup: removed {deleted} old prop snapshots")
        return deleted


def update_play_closing_line(
    play_id: int,
    closing_odds_american: int,
    closing_odds_decimal: float,
    closing_ev_percent: float,
) -> None:
    """Update a play with closing line data."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE plays SET
                closing_odds_american = ?,
                closing_odds_decimal = ?,
                closing_ev_percent = ?
            WHERE id = ?
        """, (closing_odds_american, closing_odds_decimal, closing_ev_percent, play_id))


def update_play_result(play_id: int, result: str, profit_units: float) -> None:
    """Update a play with the result (win/loss/push)."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE plays SET result = ?, profit_units = ?
            WHERE id = ?
        """, (result, profit_units, play_id))


def get_plays_needing_closing_line() -> list[dict]:
    """Get plays that need closing line data (game has started)."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM plays
            WHERE closing_odds_american IS NULL
            AND commence_time < ?
        """, (now,))
        return [dict(row) for row in cursor.fetchall()]


def get_closing_line_for_play(play: dict) -> Optional[dict]:
    """
    Get the closing line (last odds before game start) for a play.

    Returns dict with closing odds or None if not found.
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # Build query to find the last odds snapshot before commence_time
        # for this specific outcome at the soft book
        if play["point"] is not None:
            cursor.execute("""
                SELECT price_american, price_decimal, timestamp
                FROM odds_snapshots
                WHERE event_id = ?
                AND bookmaker = ?
                AND market = ?
                AND outcome = ?
                AND point = ?
                AND timestamp < ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (
                play["event_id"],
                play["soft_book"],
                play["market"],
                play["outcome"],
                play["point"],
                play["commence_time"],
            ))
        else:
            cursor.execute("""
                SELECT price_american, price_decimal, timestamp
                FROM odds_snapshots
                WHERE event_id = ?
                AND bookmaker = ?
                AND market = ?
                AND outcome = ?
                AND point IS NULL
                AND timestamp < ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (
                play["event_id"],
                play["soft_book"],
                play["market"],
                play["outcome"],
                play["commence_time"],
            ))

        row = cursor.fetchone()
        if row:
            return {
                "closing_odds_american": row["price_american"],
                "closing_odds_decimal": row["price_decimal"],
                "timestamp": row["timestamp"],
            }
        return None


def process_closing_lines() -> int:
    """
    Process all plays needing closing lines.

    Finds plays where the game has started but closing line hasn't been captured,
    looks up the last odds before game start, and updates the play record.

    Returns the number of plays updated.
    """
    from arbitrage import american_to_decimal

    plays = get_plays_needing_closing_line()
    updated = 0

    for play in plays:
        closing = get_closing_line_for_play(play)

        if closing:
            # Calculate closing EV using the stored fair_prob
            fair_prob = play["fair_prob"]  # Already in 0-1 format
            closing_decimal = closing["closing_odds_decimal"]
            closing_ev = (fair_prob * closing_decimal - 1) * 100

            update_play_closing_line(
                play_id=play["id"],
                closing_odds_american=closing["closing_odds_american"],
                closing_odds_decimal=closing["closing_odds_decimal"],
                closing_ev_percent=closing_ev,
            )
            updated += 1
            logger.info(
                f"Closing line captured for play {play['id']}: "
                f"{play['outcome']} @ {play['soft_book']} - "
                f"Sent: {play['sent_odds_american']:+d}, Close: {closing['closing_odds_american']:+d}"
            )

    return updated


def get_all_plays(days_back: Optional[int] = None) -> list[dict]:
    """Get all plays, optionally filtered by days back."""
    with get_db() as conn:
        cursor = conn.cursor()
        if days_back:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
            cursor.execute("SELECT * FROM plays WHERE sent_at > ? ORDER BY sent_at DESC", (cutoff,))
        else:
            cursor.execute("SELECT * FROM plays ORDER BY sent_at DESC")
        return [dict(row) for row in cursor.fetchall()]


def get_plays_profit_summary() -> dict:
    """Get summary of profit in units."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Total units wagered
        cursor.execute("SELECT SUM(units) FROM plays")
        total_units = cursor.fetchone()[0] or 0

        # Total profit (from completed plays)
        cursor.execute("SELECT SUM(profit_units) FROM plays WHERE result IS NOT NULL")
        total_profit = cursor.fetchone()[0] or 0

        # Count by result
        cursor.execute("""
            SELECT result, COUNT(*), SUM(profit_units)
            FROM plays WHERE result IS NOT NULL
            GROUP BY result
        """)
        results = {row[0]: {"count": row[1], "profit": row[2]} for row in cursor.fetchall()}

        # Pending plays
        cursor.execute("SELECT COUNT(*) FROM plays WHERE result IS NULL")
        pending = cursor.fetchone()[0]

        return {
            "total_units_wagered": total_units,
            "total_profit_units": total_profit,
            "results": results,
            "pending_plays": pending,
            "roi_percent": (total_profit / total_units * 100) if total_units > 0 else 0,
        }


def get_database_stats() -> dict:
    """Get statistics about the database."""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM events")
        event_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM odds_snapshots")
        snapshot_count = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM odds_snapshots")
        row = cursor.fetchone()
        oldest, newest = row[0], row[1]

        # Get database file size
        db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0

        return {
            "events": event_count,
            "snapshots": snapshot_count,
            "oldest_data": oldest,
            "newest_data": newest,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
        }


class Database:
    """Database wrapper class for dashboard use."""

    def __init__(self):
        init_database()

    def get_play_stats(self) -> dict:
        """Get comprehensive stats for dashboard home."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Use CST-based dates
            today_start = get_cst_today_start()
            week_start = get_cst_week_start()

            # Today's plays
            cursor.execute(
                "SELECT COUNT(*) FROM plays WHERE sent_at >= ?",
                (today_start,)
            )
            plays_today = cursor.fetchone()[0]

            # This week's plays
            cursor.execute(
                "SELECT COUNT(*) FROM plays WHERE sent_at >= ?",
                (week_start,)
            )
            plays_week = cursor.fetchone()[0]

            # Total plays
            cursor.execute("SELECT COUNT(*) FROM plays")
            total_plays = cursor.fetchone()[0]

            # Win/Loss/Push counts
            cursor.execute("""
                SELECT result, COUNT(*), COALESCE(SUM(profit_units), 0)
                FROM plays WHERE result IS NOT NULL
                GROUP BY result
            """)
            results_raw = cursor.fetchall()
            results = {row[0]: {"count": row[1], "profit": row[2]} for row in results_raw}

            # Pending plays
            cursor.execute("SELECT COUNT(*) FROM plays WHERE result IS NULL")
            pending = cursor.fetchone()[0]

            # Total units wagered and profit
            cursor.execute("SELECT COALESCE(SUM(units), 0) FROM plays")
            total_units = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COALESCE(SUM(profit_units), 0) FROM plays WHERE result IS NOT NULL"
            )
            total_profit = cursor.fetchone()[0]

            # CLV stats (plays that beat closing line)
            cursor.execute("""
                SELECT COUNT(*) FROM plays
                WHERE closing_odds_american IS NOT NULL
                AND sent_odds_american > closing_odds_american
            """)
            beat_close = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM plays WHERE closing_odds_american IS NOT NULL"
            )
            with_close = cursor.fetchone()[0]

            # Average EV
            cursor.execute("SELECT AVG(sent_ev_percent) FROM plays")
            avg_ev = cursor.fetchone()[0] or 0

            roi = (total_profit / total_units * 100) if total_units > 0 else 0
            clv_rate = (beat_close / with_close * 100) if with_close > 0 else 0

            wins = results.get("win", {}).get("count", 0)
            losses = results.get("loss", {}).get("count", 0)
            completed = wins + losses
            win_rate = (wins / completed * 100) if completed > 0 else 0

            return {
                "plays_today": plays_today,
                "plays_week": plays_week,
                "total_plays": total_plays,
                "pending_plays": pending,
                "wins": wins,
                "losses": losses,
                "pushes": results.get("push", {}).get("count", 0),
                "win_rate": round(win_rate, 1),
                "total_units": round(total_units, 2),
                "total_profit": round(total_profit, 2),
                "roi": round(roi, 2),
                "clv_rate": round(clv_rate, 1),
                "avg_ev": round(avg_ev, 2),
            }

    def get_recent_plays(self, limit: int = 100) -> list[dict]:
        """Get recent plays for EV display."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.*, e.home_team, e.away_team, e.commence_time as game_time
                FROM plays p
                LEFT JOIN events e ON p.event_id = e.id
                ORDER BY p.sent_at DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_all_plays(self, search: str = "", page: int = 1, per_page: int = 50) -> dict:
        """Get all plays with search and pagination."""
        with get_db() as conn:
            cursor = conn.cursor()
            offset = (page - 1) * per_page

            # Build search query
            if search:
                search_pattern = f"%{search}%"
                cursor.execute("""
                    SELECT COUNT(*) FROM plays
                    WHERE home_team LIKE ? OR away_team LIKE ?
                    OR outcome LIKE ? OR soft_book LIKE ? OR sport LIKE ?
                """, (search_pattern,) * 5)
            else:
                cursor.execute("SELECT COUNT(*) FROM plays")

            total = cursor.fetchone()[0]

            if search:
                cursor.execute("""
                    SELECT * FROM plays
                    WHERE home_team LIKE ? OR away_team LIKE ?
                    OR outcome LIKE ? OR soft_book LIKE ? OR sport LIKE ?
                    ORDER BY sent_at DESC
                    LIMIT ? OFFSET ?
                """, (search_pattern,) * 5 + (per_page, offset))
            else:
                cursor.execute("""
                    SELECT * FROM plays
                    ORDER BY sent_at DESC
                    LIMIT ? OFFSET ?
                """, (per_page, offset))

            plays = [dict(row) for row in cursor.fetchall()]

            return {
                "plays": plays,
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page,
            }

    def get_recent_arb_opportunities(self) -> list[dict]:
        """Get recent arbitrage plays from the database."""
        import json
        with get_db() as conn:
            cursor = conn.cursor()
            # Get arb plays from last 24 hours
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            cursor.execute("""
                SELECT * FROM arb_plays
                WHERE sent_at >= ?
                ORDER BY sent_at DESC
            """, (cutoff,))
            rows = cursor.fetchall()

            result = []
            for row in rows:
                arb = dict(row)
                # Parse legs JSON
                arb['legs'] = json.loads(arb['legs_json'])
                del arb['legs_json']
                result.append(arb)

            return result

    def get_prop_odds_history(self, event_id: str, player_name: str, prop_type: str, line: float = None) -> dict:
        """Get prop odds history for graphing."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Get event info
            cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
            event_row = cursor.fetchone()
            event = dict(event_row) if event_row else {}

            # Get all prop odds for this player/prop combination
            query = """
                SELECT bookmaker, player_name, prop_type, outcome, line,
                       price_american, price_decimal, timestamp
                FROM prop_snapshots
                WHERE event_id = ? AND player_name = ? AND prop_type = ?
            """
            params = [event_id, player_name, prop_type]

            if line is not None:
                query += " AND line = ?"
                params.append(line)

            query += " ORDER BY timestamp ASC"

            cursor.execute(query, params)
            snapshots = [dict(row) for row in cursor.fetchall()]

            # Get prop play info if exists
            play_query = """
                SELECT sent_at, sent_odds_american, closing_odds_american, outcome, line
                FROM prop_plays
                WHERE event_id = ? AND player_name = ? AND prop_type = ?
            """
            play_params = [event_id, player_name, prop_type]
            if line is not None:
                play_query += " AND line = ?"
                play_params.append(line)

            cursor.execute(play_query, play_params)
            plays = [dict(row) for row in cursor.fetchall()]

            # Group by bookmaker for charting
            by_bookmaker = {}
            available_lines = set()
            for snap in snapshots:
                book = snap["bookmaker"]
                if book not in by_bookmaker:
                    by_bookmaker[book] = []
                by_bookmaker[book].append({
                    "timestamp": snap["timestamp"],
                    "odds": snap["price_american"],
                    "outcome": snap["outcome"],
                    "line": snap["line"],
                })
                available_lines.add(snap["line"])

            return {
                "event": event,
                "player_name": player_name,
                "prop_type": prop_type,
                "requested_line": line,
                "available_lines": sorted(list(available_lines)),
                "by_bookmaker": by_bookmaker,
                "plays": plays,
                "total_snapshots": len(snapshots),
            }

    def get_odds_history(self, event_id: str, market: str, outcome: str = "") -> dict:
        """Get odds history for graphing."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Get event info
            cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
            event_row = cursor.fetchone()
            event = dict(event_row) if event_row else {}

            # Get all odds for this event/market
            query = """
                SELECT bookmaker, outcome, price_american, price_decimal, point, timestamp
                FROM odds_snapshots
                WHERE event_id = ? AND market = ?
            """
            params = [event_id, market]

            if outcome:
                query += " AND outcome = ?"
                params.append(outcome)

            query += " ORDER BY timestamp ASC"

            cursor.execute(query, params)
            snapshots = [dict(row) for row in cursor.fetchall()]

            # Get play info if exists
            cursor.execute("""
                SELECT sent_at, sent_odds_american, closing_odds_american
                FROM plays
                WHERE event_id = ? AND market = ?
            """, (event_id, market))
            plays = [dict(row) for row in cursor.fetchall()]

            # Group by bookmaker for charting
            by_bookmaker = {}
            for snap in snapshots:
                book = snap["bookmaker"]
                if book not in by_bookmaker:
                    by_bookmaker[book] = []
                by_bookmaker[book].append({
                    "timestamp": snap["timestamp"],
                    "odds": snap["price_american"],
                    "outcome": snap["outcome"],
                    "point": snap["point"],
                })

            return {
                "event": event,
                "market": market,
                "outcome_filter": outcome,
                "by_bookmaker": by_bookmaker,
                "plays": plays,
                "total_snapshots": len(snapshots),
            }

    def get_events(self, sport: str = "", search: str = "", limit: int = 100) -> list[dict]:
        """Get events for browsing."""
        with get_db() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM events WHERE 1=1"
            params = []

            if sport:
                query += " AND sport = ?"
                params.append(sport)

            if search:
                query += " AND (home_team LIKE ? OR away_team LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])

            query += " ORDER BY commence_time DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_collector_status(self) -> dict:
        """Get collector status based on recent activity."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Check most recent odds snapshot
            cursor.execute(
                "SELECT MAX(timestamp) FROM odds_snapshots"
            )
            last_poll = cursor.fetchone()[0]

            # Check if collector appears to be running
            running = False
            if last_poll:
                last_dt = datetime.fromisoformat(last_poll.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                minutes_ago = (now - last_dt).total_seconds() / 60
                running = minutes_ago < 15  # Consider running if data < 15 min old

            # Get snapshot count today (using CST)
            today_start = get_cst_today_start()
            cursor.execute(
                "SELECT COUNT(*) FROM odds_snapshots WHERE timestamp >= ?",
                (today_start,)
            )
            snapshots_today = cursor.fetchone()[0]

            return {
                "running": running,
                "last_poll": last_poll,
                "snapshots_today": snapshots_today,
            }

    def get_upcoming_events_with_odds(self, sport: str = "", search: str = "") -> list[dict]:
        """Get all upcoming events with their latest odds for browsing."""
        now = datetime.now(timezone.utc).isoformat()

        with get_db() as conn:
            cursor = conn.cursor()

            # Get upcoming events
            query = """
                SELECT e.*,
                    (SELECT COUNT(DISTINCT bookmaker) FROM odds_snapshots WHERE event_id = e.id) as bookmaker_count,
                    (SELECT COUNT(DISTINCT market) FROM odds_snapshots WHERE event_id = e.id) as market_count
                FROM events e
                WHERE e.commence_time > ?
            """
            params = [now]

            if sport:
                query += " AND e.sport = ?"
                params.append(sport)

            if search:
                query += " AND (e.home_team LIKE ? OR e.away_team LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])

            query += " ORDER BY e.commence_time ASC LIMIT 200"

            cursor.execute(query, params)
            events = [dict(row) for row in cursor.fetchall()]

            # For each event, get available markets
            for event in events:
                cursor.execute("""
                    SELECT DISTINCT market FROM odds_snapshots
                    WHERE event_id = ?
                """, (event['id'],))
                event['markets'] = [row[0] for row in cursor.fetchall()]

            return events

    def get_play_by_id(self, play_id: int) -> Optional[dict]:
        """Get a single play by ID."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM plays WHERE id = ?", (play_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_play_result_and_profit(self, play_id: int, result: str) -> float:
        """Update play result and calculate profit/loss."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Get the play details
            cursor.execute("SELECT * FROM plays WHERE id = ?", (play_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Play {play_id} not found")

            play = dict(row)
            units = play['units']
            decimal_odds = play['sent_odds_decimal']

            # Calculate profit based on result
            if result == 'win':
                profit = units * (decimal_odds - 1)  # Win returns (odds - 1) * stake
            elif result == 'loss':
                profit = -units  # Lose the stake
            elif result == 'push':
                profit = 0.0  # No profit or loss
            else:
                profit = None

            # Update the play
            cursor.execute("""
                UPDATE plays SET result = ?, profit_units = ?
                WHERE id = ?
            """, (result, profit, play_id))

            return profit

    def get_profit_history(self) -> dict:
        """Get cumulative profit over time for graphing."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Get all plays with results, ordered by sent_at
            cursor.execute("""
                SELECT sent_at, profit_units
                FROM plays
                WHERE result IS NOT NULL AND profit_units IS NOT NULL
                ORDER BY sent_at ASC
            """)

            rows = cursor.fetchall()

            if not rows:
                return {"points": []}

            # Calculate cumulative profit
            points = []
            cumulative = 0.0

            for row in rows:
                cumulative += row["profit_units"]
                points.append({
                    "date": row["sent_at"],
                    "profit": row["profit_units"],
                    "cumulative_profit": round(cumulative, 2)
                })

            return {"points": points}

    def get_recent_prop_plays(self, limit: int = 200) -> list[dict]:
        """Get recent player prop plays for display."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.*, e.home_team, e.away_team, e.commence_time as game_time
                FROM prop_plays p
                LEFT JOIN events e ON p.event_id = e.id
                ORDER BY p.sent_at DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_all_available_props(
        self, 
        sport: str = None, 
        player_search: str = None,
        prop_type: str = None,
        bookmaker: str = None
    ) -> list[dict]:
        """
        Get all available player props from latest snapshots.
        Groups by (event_id, player_name, prop_type) to show one entry per player-prop combination.
        Only shows props for upcoming games.
        """
        now = datetime.now(timezone.utc).isoformat()
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Step 1: Get latest snapshot for each unique prop (event_id, player_name, prop_type, line, outcome, bookmaker)
            # Step 2: Group by (event_id, player_name, prop_type) and pick one entry per group (most recent)
            query = """
                WITH latest_snapshots AS (
                    SELECT 
                        ps.event_id,
                        ps.bookmaker,
                        ps.prop_type,
                        ps.player_name,
                        ps.outcome,
                        ps.line,
                        ps.price_american,
                        ps.price_decimal,
                        ps.timestamp,
                        e.sport,
                        e.home_team,
                        e.away_team,
                        e.commence_time,
                        ROW_NUMBER() OVER (
                            PARTITION BY ps.event_id, ps.bookmaker, ps.prop_type, 
                                       ps.player_name, ps.outcome, ps.line
                            ORDER BY ps.timestamp DESC
                        ) as snapshot_rn
                    FROM prop_snapshots ps
                    INNER JOIN events e ON ps.event_id = e.id
                    WHERE e.commence_time > ?
            """
            params = [now]
            
            if sport:
                query += " AND e.sport = ?"
                params.append(sport)
            
            if player_search:
                query += " AND LOWER(ps.player_name) LIKE ?"
                params.append(f"%{player_search.lower()}%")
            
            if prop_type:
                query += " AND ps.prop_type = ?"
                params.append(prop_type)
            
            if bookmaker:
                query += " AND ps.bookmaker = ?"
                params.append(bookmaker)
            
            query += """
                ),
                unique_props AS (
                    SELECT 
                        event_id,
                        bookmaker,
                        prop_type,
                        player_name,
                        outcome,
                        line,
                        price_american,
                        price_decimal,
                        timestamp,
                        sport,
                        home_team,
                        away_team,
                        commence_time,
                        ROW_NUMBER() OVER (
                            PARTITION BY event_id, player_name, prop_type
                            ORDER BY timestamp DESC
                        ) as prop_rn
                    FROM latest_snapshots
                    WHERE snapshot_rn = 1
                )
                SELECT 
                    event_id,
                    bookmaker,
                    prop_type,
                    player_name,
                    outcome,
                    line,
                    price_american,
                    price_decimal,
                    timestamp,
                    sport,
                    home_team,
                    away_team,
                    commence_time
                FROM unique_props
                WHERE prop_rn = 1
                ORDER BY commence_time ASC, player_name ASC, prop_type ASC
            """
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_underdog_props_with_ev(self, sharp_book: str = "pinnacle", devig_method: str = "power") -> list[dict]:
        """
        Get all Underdog props with fair probabilities and EV calculated from sharp book odds.
        Returns props sorted by fair probability (ascending - most likely first).
        """
        from ev_detector import calculate_no_vig_probability, calculate_ev
        
        now = datetime.now(timezone.utc).isoformat()
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get all Underdog props from latest snapshots
            query = """
                WITH latest_underdog AS (
                    SELECT 
                        ps.event_id,
                        ps.prop_type,
                        ps.player_name,
                        ps.outcome,
                        ps.line,
                        ps.price_american,
                        ps.price_decimal,
                        ps.timestamp,
                        e.sport,
                        e.home_team,
                        e.away_team,
                        e.commence_time,
                        ROW_NUMBER() OVER (
                            PARTITION BY ps.event_id, ps.prop_type, ps.player_name, ps.outcome, ps.line
                            ORDER BY ps.timestamp DESC
                        ) as rn
                    FROM prop_snapshots ps
                    INNER JOIN events e ON ps.event_id = e.id
                    WHERE ps.bookmaker = 'underdog'
                        AND e.commence_time > ?
                )
                SELECT 
                    event_id,
                    prop_type,
                    player_name,
                    outcome,
                    line,
                    price_american,
                    price_decimal,
                    timestamp,
                    sport,
                    home_team,
                    away_team,
                    commence_time
                FROM latest_underdog
                WHERE rn = 1
            """
            
            cursor.execute(query, (now,))
            underdog_props = [dict(row) for row in cursor.fetchall()]
            
            results = []
            
            # For each Underdog prop, find matching sharp book odds and calculate EV
            for prop in underdog_props:
                # Find matching Over/Under pair from sharp book
                cursor.execute("""
                    SELECT 
                        outcome,
                        price_american,
                        price_decimal,
                        timestamp
                    FROM prop_snapshots
                    WHERE event_id = ?
                        AND bookmaker = ?
                        AND prop_type = ?
                        AND player_name = ?
                        AND line = ?
                        AND outcome IN ('Over', 'Under')
                    ORDER BY timestamp DESC
                """, (prop['event_id'], sharp_book, prop['prop_type'], 
                      prop['player_name'], prop['line']))
                
                sharp_outcomes = cursor.fetchall()
                
                # Need both Over and Under to calculate fair probability
                sharp_over = None
                sharp_under = None
                for row in sharp_outcomes:
                    outcome_dict = dict(row)
                    if outcome_dict['outcome'] == 'Over':
                        sharp_over = outcome_dict
                    elif outcome_dict['outcome'] == 'Under':
                        sharp_under = outcome_dict
                
                if sharp_over and sharp_under:
                    # Calculate fair probability from sharp book odds
                    sharp_decimals = [sharp_over['price_decimal'], sharp_under['price_decimal']]
                    fair_probs = calculate_no_vig_probability(sharp_decimals, method=devig_method)
                    
                    # Get fair probability for this outcome
                    if prop['outcome'] == 'Over':
                        fair_prob = fair_probs[0]
                        sharp_odds_american = sharp_over['price_american']
                        sharp_odds_decimal = sharp_over['price_decimal']
                    else:  # Under
                        fair_prob = fair_probs[1]
                        sharp_odds_american = sharp_under['price_american']
                        sharp_odds_decimal = sharp_under['price_decimal']
                    
                    # Calculate EV for Underdog odds
                    ev = calculate_ev(fair_prob, prop['price_decimal'])
                    
                    # Only include props with positive EV
                    if ev > 0:
                        results.append({
                            'event_id': prop['event_id'],
                            'sport': prop['sport'],
                            'home_team': prop['home_team'],
                            'away_team': prop['away_team'],
                            'commence_time': prop['commence_time'],
                            'prop_type': prop['prop_type'],
                            'player_name': prop['player_name'],
                            'outcome': prop['outcome'],
                            'line': prop['line'],
                            'underdog_odds_american': prop['price_american'],
                            'underdog_odds_decimal': prop['price_decimal'],
                            'sharp_odds_american': sharp_odds_american,
                            'sharp_odds_decimal': sharp_odds_decimal,
                            'fair_prob': fair_prob,
                            'ev': ev,
                        })
            
            # Sort by fair probability (ascending - most likely first)
            results.sort(key=lambda x: x['fair_prob'])
            
            return results

    def get_prop_play_stats(self) -> dict:
        """Get comprehensive stats for player props."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Use CST-based dates
            today_start = get_cst_today_start()

            # Props today
            cursor.execute(
                "SELECT COUNT(*) FROM prop_plays WHERE sent_at >= ?",
                (today_start,)
            )
            props_today = cursor.fetchone()[0]

            # Pending props (no result yet)
            cursor.execute("SELECT COUNT(*) FROM prop_plays WHERE result IS NULL")
            pending_props = cursor.fetchone()[0]

            # Total profit from props
            cursor.execute(
                "SELECT COALESCE(SUM(profit_units), 0) FROM prop_plays WHERE result IS NOT NULL"
            )
            total_profit = cursor.fetchone()[0]

            # Average EV
            cursor.execute("SELECT AVG(sent_ev_percent) FROM prop_plays")
            avg_ev = cursor.fetchone()[0] or 0

            # Win/Loss/Push stats
            cursor.execute("""
                SELECT result, COUNT(*), COALESCE(SUM(profit_units), 0)
                FROM prop_plays WHERE result IS NOT NULL
                GROUP BY result
            """)
            results_raw = cursor.fetchall()
            results = {row[0]: {"count": row[1], "profit": row[2]} for row in results_raw}

            wins = results.get("win", {}).get("count", 0)
            losses = results.get("loss", {}).get("count", 0)
            completed = wins + losses
            win_rate = (wins / completed * 100) if completed > 0 else 0

            # Total units wagered
            cursor.execute("SELECT COALESCE(SUM(units), 0) FROM prop_plays")
            total_units = cursor.fetchone()[0]

            roi = (total_profit / total_units * 100) if total_units > 0 else 0

            return {
                "props_today": props_today,
                "pending_props": pending_props,
                "total_profit": round(total_profit, 2),
                "avg_ev": round(avg_ev, 2),
                "wins": wins,
                "losses": losses,
                "pushes": results.get("push", {}).get("count", 0),
                "win_rate": round(win_rate, 1),
                "total_units": round(total_units, 2),
                "roi": round(roi, 2),
            }

    def update_prop_play_result_and_profit(self, play_id: int, result: str) -> float:
        """Update prop play result and calculate profit/loss."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Get the play details
            cursor.execute("SELECT * FROM prop_plays WHERE id = ?", (play_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Prop play {play_id} not found")

            play = dict(row)
            units = play['units']
            decimal_odds = play['sent_odds_decimal']

            # Calculate profit based on result
            if result == 'win':
                profit = units * (decimal_odds - 1)
            elif result == 'loss':
                profit = -units
            elif result == 'push':
                profit = 0.0
            else:
                profit = None

            # Update the play
            cursor.execute("""
                UPDATE prop_plays SET result = ?, profit_units = ?
                WHERE id = ?
            """, (result, profit, play_id))

            return profit

    def get_prop_profit_history(self) -> dict:
        """Get cumulative profit over time for props graphing."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT sent_at, profit_units
                FROM prop_plays
                WHERE result IS NOT NULL AND profit_units IS NOT NULL
                ORDER BY sent_at ASC
            """)

            rows = cursor.fetchall()

            if not rows:
                return {"points": []}

            points = []
            cumulative = 0.0

            for row in rows:
                cumulative += row["profit_units"]
                points.append({
                    "date": row["sent_at"],
                    "profit": row["profit_units"],
                    "cumulative_profit": round(cumulative, 2)
                })

            return {"points": points}

    def get_current_value_stats(self) -> dict:
        """
        Calculate current value by comparing sent odds to current (latest) odds.
        
        Current Value shows if you got better odds than what's currently available.
        Positive = you got better odds than current market (good value captured)
        Negative = odds improved after your bet (could have waited)
        
        Returns unit-weighted average to properly reflect value across different bet sizes.
        """
        with get_db() as conn:
            cursor = conn.cursor()
            
            now = datetime.now(timezone.utc).isoformat()
            
            # Get all EV plays without results assigned - only include those without results
            cursor.execute("""
                SELECT id, event_id, market, outcome, point, soft_book,
                       sent_odds_american, sent_odds_decimal, units, sent_ev_percent
                FROM plays
                WHERE result IS NULL AND commence_time > ?
            """, (now,))
            ev_plays = [dict(row) for row in cursor.fetchall()]
            
            # Get all prop plays without results assigned - only include those without results
            cursor.execute("""
                SELECT id, event_id, prop_type, player_name, outcome, line, soft_book,
                       sent_odds_american, sent_odds_decimal, units, sent_ev_percent
                FROM prop_plays
                WHERE result IS NULL AND commence_time > ?
            """, (now,))
            prop_plays = [dict(row) for row in cursor.fetchall()]
            
            current_values = []
            plays_with_value = 0
            
            # Process EV plays - only those without results assigned
            for play in ev_plays:
                units = play.get('units', 1.0) or 1.0
                value_pct = None
                current_odds = None
                
                # Get latest odds for this market/outcome from the same book
                if play['point'] is not None:
                    cursor.execute("""
                        SELECT price_american, price_decimal, timestamp
                        FROM odds_snapshots
                        WHERE event_id = ? AND bookmaker = ? AND market = ? 
                              AND outcome = ? AND point = ?
                        ORDER BY timestamp DESC LIMIT 1
                    """, (play['event_id'], play['soft_book'], play['market'],
                          play['outcome'], play['point']))
                else:
                    cursor.execute("""
                        SELECT price_american, price_decimal, timestamp
                        FROM odds_snapshots
                        WHERE event_id = ? AND bookmaker = ? AND market = ? 
                              AND outcome = ? AND point IS NULL
                        ORDER BY timestamp DESC LIMIT 1
                    """, (play['event_id'], play['soft_book'], play['market'],
                          play['outcome']))
                
                current = cursor.fetchone()
                if current:
                    # Calculate value: positive means you got better odds
                    # Using decimal odds: (your_decimal - current_decimal) / current_decimal * 100
                    sent_dec = play['sent_odds_decimal']
                    curr_dec = current['price_decimal']
                    value_pct = ((sent_dec - curr_dec) / curr_dec) * 100
                    current_odds = current['price_american']
                    plays_with_value += 1
                else:
                    # No current odds available - use sent EV as value (it was +EV when sent)
                    if play.get('sent_ev_percent') is not None:
                        value_pct = play['sent_ev_percent']
                    else:
                        value_pct = 0
                
                current_values.append({
                    'value_pct': value_pct,
                    'units': units,
                    'sent_odds': play['sent_odds_american'],
                    'current_odds': current_odds,
                    'type': 'ev'
                })
            
            # Process prop plays - only those without results assigned
            for play in prop_plays:
                units = play.get('units', 1.0) or 1.0
                value_pct = None
                current_odds = None
                alternate_line = None
                
                # First try exact line match
                cursor.execute("""
                    SELECT price_american, price_decimal, line, timestamp
                    FROM prop_snapshots
                    WHERE event_id = ? AND bookmaker = ? AND prop_type = ?
                          AND player_name = ? AND outcome = ? AND line = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (play['event_id'], play['soft_book'], play['prop_type'],
                      play['player_name'], play['outcome'], play['line']))
                
                current = cursor.fetchone()
                
                # If no exact match, find closest available line
                if not current:
                    cursor.execute("""
                        SELECT price_american, price_decimal, line, timestamp
                        FROM prop_snapshots
                        WHERE event_id = ? AND prop_type = ?
                              AND player_name = ? AND outcome = ?
                        ORDER BY ABS(line - ?) ASC, timestamp DESC
                        LIMIT 1
                    """, (play['event_id'], play['prop_type'],
                          play['player_name'], play['outcome'], play['line']))
                    current = cursor.fetchone()
                    if current:
                        alternate_line = current['line']
                
                if current:
                    sent_dec = play['sent_odds_decimal']
                    curr_dec = current['price_decimal']
                    
                    # If using alternate line, estimate fair odds at original line
                    if alternate_line is not None and alternate_line != play['line']:
                        estimated_fair = self._estimate_odds_at_original_line(
                            current_odds_decimal=curr_dec,
                            current_line=alternate_line,
                            original_line=play['line'],
                            outcome=play['outcome'],
                            prop_type=play['prop_type']
                        )
                        value_pct = ((sent_dec - estimated_fair) / estimated_fair) * 100
                    else:
                        value_pct = ((sent_dec - curr_dec) / curr_dec) * 100
                    
                    current_odds = current['price_american']
                    plays_with_value += 1
                else:
                    # No current odds available - use sent EV as value (it was +EV when sent)
                    if play.get('sent_ev_percent') is not None:
                        value_pct = play['sent_ev_percent']
                    else:
                        value_pct = 0
                
                current_values.append({
                    'value_pct': value_pct,
                    'units': units,
                    'sent_odds': play['sent_odds_american'],
                    'current_odds': current_odds,
                    'type': 'prop',
                    'line_adjusted': alternate_line is not None
                })
            
            # Calculate unit-weighted average - only plays without results assigned
            if current_values:
                total_units = sum(v['units'] for v in current_values)
                weighted_sum = sum(v['value_pct'] * v['units'] for v in current_values)
                avg_current_value = weighted_sum / total_units if total_units > 0 else 0
                
                positive_value_count = sum(1 for v in current_values if v['value_pct'] > 0)
                positive_value_rate = (positive_value_count / len(current_values)) * 100
            else:
                avg_current_value = 0
                positive_value_rate = 0
                total_units = 0
            
            return {
                'avg_current_value': round(avg_current_value, 2),
                'positive_value_rate': round(positive_value_rate, 1),
                'plays_measured': plays_with_value,  # Plays with current odds data
                'total_pending': len(ev_plays) + len(prop_plays),  # ALL upcoming plays
                'total_units': round(total_units, 2),
            }

    def get_clv_stats(self) -> dict:
        """
        Calculate Closing Line Value stats for games that have started.
        
        CLV shows if you got better odds than the devigged fair closing line from Pinnacle.
        Positive CLV = you beat the closing line (good long-term indicator)
        Uses devigged Pinnacle odds, with line-adjusted estimates when needed.
        Returns unit-weighted average to properly reflect value across different bet sizes.
        """
        with get_db() as conn:
            cursor = conn.cursor()
            
            now = datetime.now(timezone.utc).isoformat()
            
            # Get ALL post-game EV plays (not just those with stored closing odds)
            cursor.execute("""
                SELECT id, event_id, market, outcome, point, soft_book,
                       sent_odds_decimal, units, 'ev' as play_type
                FROM plays
                WHERE commence_time <= ?
            """, (now,))
            ev_plays = [dict(row) for row in cursor.fetchall()]
            
            # Get ALL post-game prop plays
            cursor.execute("""
                SELECT id, event_id, prop_type, player_name, outcome, line,
                       sent_odds_decimal, units, 'prop' as play_type
                FROM prop_plays
                WHERE commence_time <= ?
            """, (now,))
            prop_plays = [dict(row) for row in cursor.fetchall()]
            
            clv_values = []
            
            # Process all post-game plays using devigged Pinnacle closing odds
            for play in ev_plays + prop_plays:
                sent_dec = play['sent_odds_decimal']
                units = play.get('units', 1.0) or 1.0
                
                # Try to get devigged fair odds from Pinnacle first
                devigged = self._get_devigged_closing_odds(cursor, play)
                
                if devigged:
                    fair_dec = devigged['fair_decimal']
                    
                    # Check if using alternate line (props only)
                    if devigged.get('alternate_line') is not None and play['play_type'] == 'prop':
                        estimated_fair = self._estimate_odds_at_original_line(
                            current_odds_decimal=fair_dec,
                            current_line=devigged['alternate_line'],
                            original_line=play['line'],
                            outcome=play.get('outcome', 'Over'),
                            prop_type=play.get('prop_type')
                        )
                        clv_pct = ((sent_dec - estimated_fair) / estimated_fair) * 100
                    else:
                        clv_pct = ((sent_dec - fair_dec) / fair_dec) * 100
                    
                    clv_values.append({'clv_pct': clv_pct, 'units': units})
                else:
                    # Fall back to raw odds if devigged not available
                    closing = self._get_last_odds_for_play(cursor, play)
                    
                    if closing:
                        closing_dec = closing['price_decimal']
                        
                        if closing.get('alternate_line') is not None and play['play_type'] == 'prop':
                            estimated_closing = self._estimate_odds_at_original_line(
                                current_odds_decimal=closing_dec,
                                current_line=closing['alternate_line'],
                                original_line=play['line'],
                                outcome=play.get('outcome', 'Over'),
                                prop_type=play.get('prop_type')
                            )
                            clv_pct = ((sent_dec - estimated_closing) / estimated_closing) * 100
                        else:
                            clv_pct = ((sent_dec - closing_dec) / closing_dec) * 100
                        
                        clv_values.append({'clv_pct': clv_pct, 'units': units})
            
            # Calculate unit-weighted average
            if clv_values:
                total_units = sum(v['units'] for v in clv_values)
                weighted_sum = sum(v['clv_pct'] * v['units'] for v in clv_values)
                avg_clv = weighted_sum / total_units if total_units > 0 else 0
                
                positive_clv_count = sum(1 for v in clv_values if v['clv_pct'] > 0)
                positive_clv_rate = (positive_clv_count / len(clv_values)) * 100
            else:
                avg_clv = 0
                positive_clv_rate = 0
            
            return {
                'avg_clv': round(avg_clv, 2),
                'positive_clv_rate': round(positive_clv_rate, 1),
                'plays_with_clv': len(clv_values),
            }

    def get_clv_breakdown(self) -> dict:
        """
        Get CLV stats broken down by different dimensions:
        - By bookmaker (FanDuel, DraftKings, etc.)
        - By sport (NBA, NFL, etc.)
        - By play type (props vs EV plays)
        - By prop type (for props only)
        """
        import time
        
        # Retry logic for database locked errors
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                with get_db() as conn:
                    cursor = conn.cursor()
                    
                    now = datetime.now(timezone.utc).isoformat()
                    
                    # Get all post-game plays with their metadata
                    cursor.execute("""
                        SELECT 
                            p.id, p.event_id, p.market, p.outcome, p.point, p.soft_book,
                            p.sent_odds_decimal, p.units, 'ev' as play_type,
                            e.sport
                        FROM plays p
                        INNER JOIN events e ON p.event_id = e.id
                        WHERE p.commence_time <= ?
                    """, (now,))
                    ev_plays = [dict(row) for row in cursor.fetchall()]
                    
                    cursor.execute("""
                        SELECT 
                            pp.id, pp.event_id, pp.prop_type, pp.player_name, pp.outcome, pp.line,
                            pp.sent_odds_decimal, pp.units, pp.soft_book, 'prop' as play_type,
                            e.sport
                        FROM prop_plays pp
                        INNER JOIN events e ON pp.event_id = e.id
                        WHERE pp.commence_time <= ?
                    """, (now,))
                    prop_plays = [dict(row) for row in cursor.fetchall()]
                    
                    # Calculate CLV for each play
                    all_plays_with_clv = []
                    
                    for play in ev_plays + prop_plays:
                        sent_dec = play['sent_odds_decimal']
                        units = play.get('units', 1.0) or 1.0
                        
                        # Try to get devigged fair odds from Pinnacle first
                        devigged = self._get_devigged_closing_odds(cursor, play)
                        
                        clv_pct = None
                        if devigged:
                            fair_dec = devigged['fair_decimal']
                            
                            # Check if using alternate line (props only)
                            if devigged.get('alternate_line') is not None and play['play_type'] == 'prop':
                                estimated_fair = self._estimate_odds_at_original_line(
                                    current_odds_decimal=fair_dec,
                                    current_line=devigged['alternate_line'],
                                    original_line=play['line'],
                                    outcome=play.get('outcome', 'Over'),
                                    prop_type=play.get('prop_type')
                                )
                                clv_pct = ((sent_dec - estimated_fair) / estimated_fair) * 100
                            else:
                                clv_pct = ((sent_dec - fair_dec) / fair_dec) * 100
                        else:
                            # Fall back to raw odds if devigged not available
                            closing = self._get_last_odds_for_play(cursor, play)
                            
                            if closing:
                                closing_dec = closing['price_decimal']
                                
                                if closing.get('alternate_line') is not None and play['play_type'] == 'prop':
                                    estimated_closing = self._estimate_odds_at_original_line(
                                        current_odds_decimal=closing_dec,
                                        current_line=closing['alternate_line'],
                                        original_line=play['line'],
                                        outcome=play.get('outcome', 'Over'),
                                        prop_type=play.get('prop_type')
                                    )
                                    clv_pct = ((sent_dec - estimated_closing) / estimated_closing) * 100
                                else:
                                    clv_pct = ((sent_dec - closing_dec) / closing_dec) * 100
                        
                        if clv_pct is not None:
                            play['clv_pct'] = clv_pct
                            play['units'] = units
                            all_plays_with_clv.append(play)
                    
                    # Helper function to calculate stats for a group
                    def calc_group_stats(plays):
                        if not plays:
                            return {
                                'avg_clv': 0,
                                'positive_clv_rate': 0,
                                'plays_count': 0,
                                'total_units': 0,
                            }
                        
                        total_units = sum(p['units'] for p in plays)
                        weighted_sum = sum(p['clv_pct'] * p['units'] for p in plays)
                        avg_clv = weighted_sum / total_units if total_units > 0 else 0
                        
                        positive_count = sum(1 for p in plays if p['clv_pct'] > 0)
                        positive_rate = (positive_count / len(plays)) * 100 if plays else 0
                        
                        return {
                            'avg_clv': round(avg_clv, 2),
                            'positive_clv_rate': round(positive_rate, 1),
                            'plays_count': len(plays),
                            'total_units': round(total_units, 2),
                        }
                    
                    # Breakdown by bookmaker
                    by_bookmaker = {}
                    for play in all_plays_with_clv:
                        book = play.get('soft_book', 'unknown')
                        if book not in by_bookmaker:
                            by_bookmaker[book] = []
                        by_bookmaker[book].append(play)
                    
                    bookmaker_stats = {
                        book: calc_group_stats(plays)
                        for book, plays in by_bookmaker.items()
                    }
                    
                    # Breakdown by sport
                    by_sport = {}
                    for play in all_plays_with_clv:
                        sport = play.get('sport', 'unknown')
                        if sport not in by_sport:
                            by_sport[sport] = []
                        by_sport[sport].append(play)
                    
                    sport_stats = {
                        sport: calc_group_stats(plays)
                        for sport, plays in by_sport.items()
                    }
                    
                    # Breakdown by play type
                    by_play_type = {}
                    for play in all_plays_with_clv:
                        play_type = play.get('play_type', 'unknown')
                        if play_type not in by_play_type:
                            by_play_type[play_type] = []
                        by_play_type[play_type].append(play)
                    
                    play_type_stats = {
                        play_type: calc_group_stats(plays)
                        for play_type, plays in by_play_type.items()
                    }
                    
                    # Breakdown by prop type (for props only)
                    by_prop_type = {}
                    for play in all_plays_with_clv:
                        if play.get('play_type') == 'prop':
                            prop_type = play.get('prop_type', 'unknown')
                            if prop_type not in by_prop_type:
                                by_prop_type[prop_type] = []
                            by_prop_type[prop_type].append(play)
                    
                    prop_type_stats = {
                        prop_type: calc_group_stats(plays)
                        for prop_type, plays in by_prop_type.items()
                    }
                    
                    # Combined sport + play type (e.g., "NBA Props", "NFL EV")
                    by_sport_play_type = {}
                    for play in all_plays_with_clv:
                        sport = play.get('sport', 'unknown')
                        play_type = play.get('play_type', 'unknown')
                        key = f"{sport}_{play_type}"
                        if key not in by_sport_play_type:
                            by_sport_play_type[key] = []
                        by_sport_play_type[key].append(play)
                    
                    sport_play_type_stats = {
                        key: calc_group_stats(plays)
                        for key, plays in by_sport_play_type.items()
                    }
                    
                    return {
                        'by_bookmaker': bookmaker_stats,
                        'by_sport': sport_stats,
                        'by_play_type': play_type_stats,
                        'by_prop_type': prop_type_stats,
                        'by_sport_play_type': sport_play_type_stats,
                        'overall': calc_group_stats(all_plays_with_clv),
                    }
                    
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise
            
            # Helper function to calculate stats for a group
            def calc_group_stats(plays):
                if not plays:
                    return {
                        'avg_clv': 0,
                        'positive_clv_rate': 0,
                        'plays_count': 0,
                        'total_units': 0,
                    }
                
                total_units = sum(p['units'] for p in plays)
                weighted_sum = sum(p['clv_pct'] * p['units'] for p in plays)
                avg_clv = weighted_sum / total_units if total_units > 0 else 0
                
                positive_count = sum(1 for p in plays if p['clv_pct'] > 0)
                positive_rate = (positive_count / len(plays)) * 100 if plays else 0
                
                return {
                    'avg_clv': round(avg_clv, 2),
                    'positive_clv_rate': round(positive_rate, 1),
                    'plays_count': len(plays),
                    'total_units': round(total_units, 2),
                }
            
            # Breakdown by bookmaker
            by_bookmaker = {}
            for play in all_plays_with_clv:
                book = play.get('soft_book', 'unknown')
                if book not in by_bookmaker:
                    by_bookmaker[book] = []
                by_bookmaker[book].append(play)
            
            bookmaker_stats = {
                book: calc_group_stats(plays)
                for book, plays in by_bookmaker.items()
            }
            
            # Breakdown by sport
            by_sport = {}
            for play in all_plays_with_clv:
                sport = play.get('sport', 'unknown')
                if sport not in by_sport:
                    by_sport[sport] = []
                by_sport[sport].append(play)
            
            sport_stats = {
                sport: calc_group_stats(plays)
                for sport, plays in by_sport.items()
            }
            
            # Breakdown by play type
            by_play_type = {}
            for play in all_plays_with_clv:
                play_type = play.get('play_type', 'unknown')
                if play_type not in by_play_type:
                    by_play_type[play_type] = []
                by_play_type[play_type].append(play)
            
            play_type_stats = {
                play_type: calc_group_stats(plays)
                for play_type, plays in by_play_type.items()
            }
            
            # Breakdown by prop type (for props only)
            by_prop_type = {}
            for play in all_plays_with_clv:
                if play.get('play_type') == 'prop':
                    prop_type = play.get('prop_type', 'unknown')
                    if prop_type not in by_prop_type:
                        by_prop_type[prop_type] = []
                    by_prop_type[prop_type].append(play)
            
            prop_type_stats = {
                prop_type: calc_group_stats(plays)
                for prop_type, plays in by_prop_type.items()
            }
            
            # Combined sport + play type (e.g., "NBA Props", "NFL EV")
            by_sport_play_type = {}
            for play in all_plays_with_clv:
                sport = play.get('sport', 'unknown')
                play_type = play.get('play_type', 'unknown')
                key = f"{sport}_{play_type}"
                if key not in by_sport_play_type:
                    by_sport_play_type[key] = []
                by_sport_play_type[key].append(play)
            
            sport_play_type_stats = {
                key: calc_group_stats(plays)
                for key, plays in by_sport_play_type.items()
            }
            
            return {
                'by_bookmaker': bookmaker_stats,
                'by_sport': sport_stats,
                'by_play_type': play_type_stats,
                'by_prop_type': prop_type_stats,
                'by_sport_play_type': sport_play_type_stats,
                'overall': calc_group_stats(all_plays_with_clv),
            }

    def get_combined_stats(self) -> dict:
        """Get combined stats for all play types (EV, props, arbs)."""
        with get_db() as conn:
            cursor = conn.cursor()

            # Use CST-based dates
            today_start = get_cst_today_start()
            week_start = get_cst_week_start()

            # Plays today (EV + props)
            cursor.execute("SELECT COUNT(*) FROM plays WHERE sent_at >= ?", (today_start,))
            ev_today = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM prop_plays WHERE sent_at >= ?", (today_start,))
            props_today = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM arb_plays WHERE sent_at >= ?", (today_start,))
            arbs_today = cursor.fetchone()[0]

            # Plays this week
            cursor.execute("SELECT COUNT(*) FROM plays WHERE sent_at >= ?", (week_start,))
            ev_week = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM prop_plays WHERE sent_at >= ?", (week_start,))
            props_week = cursor.fetchone()[0]

            # Pending plays
            cursor.execute("SELECT COUNT(*) FROM plays WHERE result IS NULL")
            ev_pending = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM prop_plays WHERE result IS NULL")
            props_pending = cursor.fetchone()[0]

            # Win/Loss/Push from EV plays
            cursor.execute("""
                SELECT result, COUNT(*), COALESCE(SUM(profit_units), 0)
                FROM plays WHERE result IS NOT NULL
                GROUP BY result
            """)
            ev_results = {row[0]: {"count": row[1], "profit": row[2]} for row in cursor.fetchall()}

            # Win/Loss/Push from prop plays
            cursor.execute("""
                SELECT result, COUNT(*), COALESCE(SUM(profit_units), 0)
                FROM prop_plays WHERE result IS NOT NULL
                GROUP BY result
            """)
            prop_results = {row[0]: {"count": row[1], "profit": row[2]} for row in cursor.fetchall()}

            # Combine results
            wins = ev_results.get("win", {}).get("count", 0) + prop_results.get("win", {}).get("count", 0)
            losses = ev_results.get("loss", {}).get("count", 0) + prop_results.get("loss", {}).get("count", 0)
            pushes = ev_results.get("push", {}).get("count", 0) + prop_results.get("push", {}).get("count", 0)

            # Total profit
            ev_profit = sum(r.get("profit", 0) for r in ev_results.values())
            prop_profit = sum(r.get("profit", 0) for r in prop_results.values())
            total_profit = ev_profit + prop_profit

            # Total units wagered
            cursor.execute("SELECT COALESCE(SUM(units), 0) FROM plays")
            ev_units = cursor.fetchone()[0]
            cursor.execute("SELECT COALESCE(SUM(units), 0) FROM prop_plays")
            prop_units = cursor.fetchone()[0]
            total_units = ev_units + prop_units

            # CLV stats (from EV plays only since props don't have same CLV tracking)
            cursor.execute("""
                SELECT COUNT(*) FROM plays
                WHERE closing_odds_american IS NOT NULL
                AND sent_odds_american > closing_odds_american
            """)
            beat_close = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM plays WHERE closing_odds_american IS NOT NULL")
            with_close = cursor.fetchone()[0]

            # Average EV (combined)
            cursor.execute("SELECT AVG(sent_ev_percent) FROM plays")
            ev_avg = cursor.fetchone()[0] or 0
            cursor.execute("SELECT AVG(sent_ev_percent) FROM prop_plays")
            prop_avg = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM plays")
            ev_total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM prop_plays")
            prop_total = cursor.fetchone()[0]

            # Weighted average EV
            if ev_total + prop_total > 0:
                avg_ev = (ev_avg * ev_total + prop_avg * prop_total) / (ev_total + prop_total)
            else:
                avg_ev = 0

            completed = wins + losses
            win_rate = (wins / completed * 100) if completed > 0 else 0
            roi = (total_profit / total_units * 100) if total_units > 0 else 0
            clv_rate = (beat_close / with_close * 100) if with_close > 0 else 0

            return {
                "plays_today": ev_today + props_today,
                "arbs_today": arbs_today,
                "plays_week": ev_week + props_week,
                "pending_plays": ev_pending + props_pending,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "win_rate": round(win_rate, 1),
                "total_units": round(total_units, 2),
                "total_profit": round(total_profit, 2),
                "roi": round(roi, 2),
                "clv_rate": round(clv_rate, 1),
                "avg_ev": round(avg_ev, 2),
                "total_plays": ev_total + prop_total,
            }

    def get_all_plays_combined(self, search: str = "", page: int = 1, per_page: int = 50) -> dict:
        """Get all plays (EV + props) combined with search and pagination, including current odds."""
        with get_db() as conn:
            cursor = conn.cursor()
            offset = (page - 1) * per_page

            # Build search conditions
            search_cond = ""
            search_params = []
            if search:
                search_pattern = f"%{search}%"
                search_cond = """
                    AND (home_team LIKE ? OR away_team LIKE ?
                    OR outcome LIKE ? OR soft_book LIKE ? OR sport LIKE ?)
                """
                search_params = [search_pattern] * 5

            # Count total from both tables
            cursor.execute(f"SELECT COUNT(*) FROM plays WHERE 1=1 {search_cond}", search_params)
            ev_count = cursor.fetchone()[0]
            cursor.execute(f"SELECT COUNT(*) FROM prop_plays WHERE 1=1 {search_cond}", search_params)
            prop_count = cursor.fetchone()[0]
            total = ev_count + prop_count

            # Union query to get both types
            query = f"""
                SELECT 
                    id, 'ev' as play_type, event_id, sport, home_team, away_team, 
                    commence_time, market, outcome, point, soft_book,
                    sent_odds_american, sent_odds_decimal, sent_ev_percent, 
                    sent_edge_percent, fair_prob, kelly_fraction, units,
                    sharp_width_cents, sent_at, closing_odds_american,
                    closing_odds_decimal, closing_ev_percent, result, profit_units,
                    NULL as player_name, NULL as prop_type, NULL as line
                FROM plays WHERE 1=1 {search_cond}
                UNION ALL
                SELECT 
                    id, 'prop' as play_type, event_id, sport, home_team, away_team,
                    commence_time, prop_type as market, outcome, line as point, soft_book,
                    sent_odds_american, sent_odds_decimal, sent_ev_percent,
                    sent_edge_percent, fair_prob, kelly_fraction, units,
                    sharp_width_cents, sent_at, closing_odds_american,
                    closing_odds_decimal, closing_ev_percent, result, profit_units,
                    player_name, prop_type, line
                FROM prop_plays WHERE 1=1 {search_cond}
                ORDER BY sent_at DESC
                LIMIT ? OFFSET ?
            """

            cursor.execute(query, search_params + search_params + [per_page, offset])
            plays = [dict(row) for row in cursor.fetchall()]

            now = datetime.now(timezone.utc)

            # For each play, determine if game has started and fetch appropriate data
            for play in plays:
                # Parse commence time
                commence_str = play.get('commence_time', '')
                try:
                    if commence_str.endswith('Z'):
                        commence_time = datetime.fromisoformat(commence_str.replace('Z', '+00:00'))
                    else:
                        commence_time = datetime.fromisoformat(commence_str)
                except (ValueError, AttributeError):
                    commence_time = None

                game_started = commence_time is not None and now >= commence_time
                play['game_started'] = game_started

                if not game_started:
                    # Game hasn't started - show current odds and current value
                    current = self._get_current_odds_for_play(cursor, play)
                    if current:
                        play['current_odds_american'] = current['price_american']
                        play['current_odds_decimal'] = current['price_decimal']
                        play['current_best_book'] = current.get('bookmaker')
                        play['alternate_line'] = current.get('alternate_line')  # Set if line moved
                        
                        sent_dec = play['sent_odds_decimal']
                        curr_dec = current['price_decimal']
                        
                        # If line moved, estimate true value by adjusting for line difference
                        if current.get('alternate_line') is not None and play['play_type'] == 'prop':
                            estimated_fair = self._estimate_odds_at_original_line(
                                current_odds_decimal=curr_dec,
                                current_line=current['alternate_line'],
                                original_line=play['line'],
                                outcome=play['outcome'],
                                prop_type=play.get('prop_type')
                            )
                            play['estimated_fair_decimal'] = estimated_fair
                            # Value = how much better our odds are vs estimated fair
                            play['current_value'] = round(((sent_dec - estimated_fair) / estimated_fair) * 100, 2)
                            play['value_is_estimated'] = True
                        else:
                            # Exact line match - simple comparison
                            play['current_value'] = round(((sent_dec - curr_dec) / curr_dec) * 100, 2)
                    else:
                        play['current_odds_american'] = None
                        play['current_odds_decimal'] = None
                        play['current_best_book'] = None
                        play['alternate_line'] = None
                        # Use sent_ev_percent as fallback value when no current odds available
                        play['current_value'] = play.get('sent_ev_percent')
                        play['value_is_sent_ev'] = True  # Flag to indicate this is sent EV, not current value
                    # Clear CLV data for pre-game plays
                    play['clv_percent'] = None
                else:
                    # Game has started - get devigged closing odds from Pinnacle
                    play['current_value'] = None
                    
                    # Try to get devigged fair odds from sharp book (Pinnacle)
                    devigged = self._get_devigged_closing_odds(cursor, play)
                    
                    if devigged:
                        # Display the devigged fair odds as closing line
                        play['current_odds_american'] = devigged['fair_american']
                        play['current_odds_decimal'] = devigged['fair_decimal']
                        play['current_best_book'] = devigged.get('bookmaker', 'pinnacle')
                        play['alternate_line'] = devigged.get('alternate_line')
                        play['closing_is_devigged'] = True
                        
                        # Calculate CLV from devigged fair odds
                        sent_dec = play['sent_odds_decimal']
                        fair_dec = devigged['fair_decimal']
                        
                        # Check if using alternate line (props only)
                        if devigged.get('alternate_line') is not None and play['play_type'] == 'prop':
                            # Estimate fair odds at original line
                            estimated_fair = self._estimate_odds_at_original_line(
                                current_odds_decimal=fair_dec,
                                current_line=devigged['alternate_line'],
                                original_line=play['line'],
                                outcome=play.get('outcome', 'Over'),
                                prop_type=play.get('prop_type')
                            )
                            play['clv_percent'] = round(((sent_dec - estimated_fair) / estimated_fair) * 100, 2)
                            play['clv_is_estimated'] = True
                            play['closing_line_used'] = devigged['alternate_line']
                        else:
                            # Exact line match
                            play['clv_percent'] = round(((sent_dec - fair_dec) / fair_dec) * 100, 2)
                    else:
                        # Fall back to raw odds if devigged not available
                        closing = self._get_last_odds_for_play(cursor, play)
                        
                        if closing:
                            play['current_odds_american'] = closing['price_american']
                            play['current_odds_decimal'] = closing['price_decimal']
                            play['current_best_book'] = closing.get('bookmaker')
                            play['alternate_line'] = closing.get('alternate_line')
                            
                            sent_dec = play['sent_odds_decimal']
                            closing_dec = closing['price_decimal']
                            
                            if closing.get('alternate_line') is not None and play['play_type'] == 'prop':
                                estimated_closing = self._estimate_odds_at_original_line(
                                    current_odds_decimal=closing_dec,
                                    current_line=closing['alternate_line'],
                                    original_line=play['line'],
                                    outcome=play.get('outcome', 'Over'),
                                    prop_type=play.get('prop_type')
                                )
                                play['clv_percent'] = round(((sent_dec - estimated_closing) / estimated_closing) * 100, 2)
                                play['clv_is_estimated'] = True
                                play['closing_line_used'] = closing['alternate_line']
                            else:
                                play['clv_percent'] = round(((sent_dec - closing_dec) / closing_dec) * 100, 2)
                        else:
                            # No closing odds available
                            play['current_odds_american'] = None
                            play['current_odds_decimal'] = None
                            play['current_best_book'] = None
                            play['alternate_line'] = None
                            play['clv_percent'] = None

            return {
                "plays": plays,
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page if total > 0 else 1,
            }

    def get_live_plays_combined(self, search: str = "", type_filter: str = "", sport_filter: str = "") -> list[dict]:
        """Get all plays from today with search/filter, including current odds."""
        with get_db() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            
            # Build search conditions for EV plays - filter by today's date
            ev_search_cond = "AND DATE(sent_at) = DATE('now', 'localtime')"
            ev_params = []
            
            # Build search conditions for Prop plays - filter by today's date
            prop_search_cond = "AND DATE(sent_at) = DATE('now', 'localtime')"
            prop_params = []
            
            if search:
                search_pattern = f"%{search}%"
                ev_search_cond += """
                    AND (home_team LIKE ? OR away_team LIKE ?
                    OR outcome LIKE ? OR soft_book LIKE ? OR sport LIKE ?)
                """
                ev_params.extend([search_pattern] * 5)
                
                prop_search_cond += """
                    AND (home_team LIKE ? OR away_team LIKE ?
                    OR outcome LIKE ? OR soft_book LIKE ? OR sport LIKE ?
                    OR player_name LIKE ?)
                """
                prop_params.extend([search_pattern] * 6)
            
            if sport_filter:
                ev_search_cond += " AND sport = ?"
                ev_params.append(sport_filter)
                prop_search_cond += " AND sport = ?"
                prop_params.append(sport_filter)

            # Build queries separately for cleaner parameter handling
            ev_query = f"""
                SELECT 
                    id, 'ev' as play_type, event_id, sport, home_team, away_team, 
                    commence_time, market, outcome, point, soft_book,
                    sent_odds_american, sent_odds_decimal, sent_ev_percent, 
                    sent_edge_percent, fair_prob, kelly_fraction, units,
                    sharp_width_cents, sent_at, closing_odds_american,
                    closing_odds_decimal, closing_ev_percent, result, profit_units,
                    NULL as player_name, NULL as prop_type, NULL as line
                FROM plays WHERE 1=1 {ev_search_cond}
            """
            
            prop_query = f"""
                SELECT 
                    id, 'prop' as play_type, event_id, sport, home_team, away_team,
                    commence_time, prop_type as market, outcome, line as point, soft_book,
                    sent_odds_american, sent_odds_decimal, sent_ev_percent,
                    sent_edge_percent, fair_prob, kelly_fraction, units,
                    sharp_width_cents, sent_at, closing_odds_american,
                    closing_odds_decimal, closing_ev_percent, result, profit_units,
                    player_name, prop_type, line
                FROM prop_plays WHERE 1=1 {prop_search_cond}
            """
            
            plays = []
            
            # Execute queries based on type filter
            if type_filter == 'ev':
                cursor.execute(ev_query, ev_params)
                plays = [dict(row) for row in cursor.fetchall()]
            elif type_filter == 'prop':
                cursor.execute(prop_query, prop_params)
                plays = [dict(row) for row in cursor.fetchall()]
            else:
                # Both types - execute separately and combine
                cursor.execute(ev_query, ev_params)
                ev_plays = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute(prop_query, prop_params)
                prop_plays = [dict(row) for row in cursor.fetchall()]
                
                plays = ev_plays + prop_plays

            # For each play, fetch current odds and value
            now_iso = now.isoformat()
            for play in plays:
                commence_time = play.get('commence_time', '')
                play['game_started'] = commence_time <= now_iso
                
                # Only get current odds and value for plays where game hasn't started yet
                if not play['game_started']:
                    # Get current odds and calculate value
                    current = self._get_current_odds_for_play(cursor, play)
                    if current:
                        play['current_odds_american'] = current['price_american']
                        play['current_odds_decimal'] = current['price_decimal']
                        play['current_best_book'] = current.get('bookmaker')
                        play['alternate_line'] = current.get('alternate_line')
                        
                        sent_dec = play['sent_odds_decimal']
                        curr_dec = current['price_decimal']
                        
                        # If line moved, estimate true value
                        if current.get('alternate_line') is not None and play['play_type'] == 'prop':
                            estimated_fair = self._estimate_odds_at_original_line(
                                current_odds_decimal=curr_dec,
                                current_line=current['alternate_line'],
                                original_line=play['line'],
                                outcome=play['outcome'],
                                prop_type=play.get('prop_type')
                            )
                            play['current_value'] = round(((sent_dec - estimated_fair) / estimated_fair) * 100, 2)
                            play['value_is_estimated'] = True
                        else:
                            play['current_value'] = round(((sent_dec - curr_dec) / curr_dec) * 100, 2)
                    else:
                        play['current_odds_american'] = None
                        play['current_odds_decimal'] = None
                        play['current_best_book'] = None
                        play['alternate_line'] = None
                        # Use sent_ev_percent as fallback value when no current odds available
                        play['current_value'] = play.get('sent_ev_percent')
                        play['value_is_sent_ev'] = True
                else:
                    # Game has started - no current odds/value
                    play['current_odds_american'] = None
                    play['current_odds_decimal'] = None
                    play['current_best_book'] = None
                    play['alternate_line'] = None
                    play['current_value'] = None
                
                # Clear CLV data for pre-game plays
                play['clv_percent'] = None
            
            # Sort by commence_time ASC, then sent_at DESC
            plays.sort(key=lambda p: (
                p.get('commence_time', ''),
                datetime.fromisoformat(p.get('sent_at', '').replace('Z', '+00:00')) if p.get('sent_at') else datetime.min
            ), reverse=False)

            return plays

    def _get_current_odds_for_play(self, cursor, play: dict, max_age_minutes: int = 30) -> Optional[dict]:
        """Get the best current odds across all books for a play (only recent snapshots).
        
        For props, if exact line not found, finds the closest available line.
        Returns dict with price_american, price_decimal, bookmaker, and optionally
        'alternate_line' if using a different line than the original bet.
        """
        # Only consider snapshots from the last X minutes
        from datetime import datetime, timezone, timedelta
        cutoff_time = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
        
        if play['play_type'] == 'prop':
            # First try exact line match
            cursor.execute("""
                SELECT bookmaker, price_american, price_decimal, line, MAX(timestamp) as latest
                FROM prop_snapshots
                WHERE event_id = ? AND prop_type = ?
                      AND player_name = ? AND outcome = ? AND line = ?
                      AND timestamp >= ?
                GROUP BY bookmaker
            """, (play['event_id'], play['prop_type'],
                  play['player_name'], play['outcome'], play['line'], cutoff_time))
            
            rows = cursor.fetchall()
            
            # If no exact match, find closest available line for this player/prop
            if not rows:
                cursor.execute("""
                    SELECT bookmaker, price_american, price_decimal, line, MAX(timestamp) as latest
                    FROM prop_snapshots
                    WHERE event_id = ? AND prop_type = ?
                          AND player_name = ? AND outcome = ?
                          AND timestamp >= ?
                    GROUP BY bookmaker, line
                    ORDER BY ABS(line - ?) ASC
                """, (play['event_id'], play['prop_type'],
                      play['player_name'], play['outcome'], cutoff_time, play['line']))
                
                rows = cursor.fetchall()
                
                if rows:
                    # Find the best odds at the closest line
                    closest_line = dict(rows[0])['line']
                    # Filter to only rows at the closest line
                    rows = [r for r in rows if dict(r)['line'] == closest_line]
        else:
            # EV play - get latest from each bookmaker (only recent ones)
            if play['point'] is not None:
                cursor.execute("""
                    SELECT bookmaker, price_american, price_decimal, point as line, MAX(timestamp) as latest
                    FROM odds_snapshots
                    WHERE event_id = ? AND market = ?
                          AND outcome = ? AND point = ?
                          AND timestamp >= ?
                    GROUP BY bookmaker
                """, (play['event_id'], play['market'],
                      play['outcome'], play['point'], cutoff_time))
            else:
                cursor.execute("""
                    SELECT bookmaker, price_american, price_decimal, point as line, MAX(timestamp) as latest
                    FROM odds_snapshots
                    WHERE event_id = ? AND market = ?
                          AND outcome = ? AND point IS NULL
                          AND timestamp >= ?
                    GROUP BY bookmaker
                """, (play['event_id'], play['market'],
                      play['outcome'], cutoff_time))
            
            rows = cursor.fetchall()

        if not rows:
            return None
        
        # Find the best odds (highest decimal = best payout)
        best = None
        for row in rows:
            row_dict = dict(row)
            if best is None or row_dict['price_decimal'] > best['price_decimal']:
                best = row_dict
        
        if not best:
            return None
            
        result = {
            'price_american': best['price_american'],
            'price_decimal': best['price_decimal'],
            'bookmaker': best['bookmaker']
        }
        
        # Check if we're using an alternate line
        original_line = play.get('line') if play['play_type'] == 'prop' else play.get('point')
        current_line = best.get('line')
        if current_line is not None and original_line is not None and current_line != original_line:
            result['alternate_line'] = current_line
            
        return result

    def _get_last_odds_for_play(self, cursor, play: dict) -> Optional[dict]:
        """Get the most recent odds snapshot for a play (no time cutoff).
        
        Used for closing line - fetches the last available odds regardless of when captured.
        For props, if exact line not found, finds the closest available line.
        Returns dict with price_american, price_decimal, bookmaker, and optionally
        'alternate_line' if using a different line than the original bet.
        """
        if play['play_type'] == 'prop':
            # First try exact line match - get the most recent snapshot
            cursor.execute("""
                SELECT bookmaker, price_american, price_decimal, line
                FROM prop_snapshots
                WHERE event_id = ? AND prop_type = ?
                      AND player_name = ? AND outcome = ? AND line = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (play['event_id'], play['prop_type'],
                  play['player_name'], play['outcome'], play['line']))
            
            row = cursor.fetchone()
            
            # If no exact match, find closest available line
            if not row:
                cursor.execute("""
                    SELECT bookmaker, price_american, price_decimal, line
                    FROM prop_snapshots
                    WHERE event_id = ? AND prop_type = ?
                          AND player_name = ? AND outcome = ?
                    ORDER BY ABS(line - ?) ASC, timestamp DESC
                    LIMIT 1
                """, (play['event_id'], play['prop_type'],
                      play['player_name'], play['outcome'], play['line']))
                row = cursor.fetchone()
        else:
            # EV play - get the most recent snapshot
            if play['point'] is not None:
                cursor.execute("""
                    SELECT bookmaker, price_american, price_decimal, point as line
                    FROM odds_snapshots
                    WHERE event_id = ? AND market = ?
                          AND outcome = ? AND point = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (play['event_id'], play['market'],
                      play['outcome'], play['point']))
            else:
                cursor.execute("""
                    SELECT bookmaker, price_american, price_decimal, point as line
                    FROM odds_snapshots
                    WHERE event_id = ? AND market = ?
                          AND outcome = ? AND point IS NULL
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (play['event_id'], play['market'],
                      play['outcome']))
            
            row = cursor.fetchone()

        if not row:
            return None
        
        row_dict = dict(row)
        result = {
            'price_american': row_dict['price_american'],
            'price_decimal': row_dict['price_decimal'],
            'bookmaker': row_dict['bookmaker']
        }
        
        # Check if we're using an alternate line
        original_line = play.get('line') if play['play_type'] == 'prop' else play.get('point')
        current_line = row_dict.get('line')
        if current_line is not None and original_line is not None and current_line != original_line:
            result['alternate_line'] = current_line
            
        return result

    def _get_devigged_closing_odds(self, cursor, play: dict, sharp_book: str = "pinnacle") -> Optional[dict]:
        """Get devigged (no-vig) closing odds from sharp book.
        
        Fetches both sides of the bet from the sharp book, devigs them,
        and returns fair odds for the side we bet on.
        
        Args:
            cursor: Database cursor
            play: Play dict with event_id, outcome, line/point, etc.
            sharp_book: Sharp bookmaker to use for fair odds (default: pinnacle)
            
        Returns:
            Dict with fair_decimal, fair_american, raw_decimal, bookmaker, 
            and optionally 'alternate_line' if using different line.
            Returns None if can't get both sides from sharp book.
        """
        from ev_detector import calculate_no_vig_probability
        from arbitrage import decimal_to_american
        import config
        
        devig_method = getattr(config, 'DEVIG_METHOD', 'power')
        
        if play['play_type'] == 'prop':
            original_line = play.get('line')
            outcome = play.get('outcome', 'Over')
            opposite = 'Under' if outcome == 'Over' else 'Over'
            
            # First try exact line match from sharp book - get both sides
            cursor.execute("""
                SELECT outcome, price_decimal, line
                FROM prop_snapshots
                WHERE event_id = ? AND prop_type = ?
                      AND player_name = ? AND line = ?
                      AND bookmaker = ?
                      AND outcome IN ('Over', 'Under')
                ORDER BY timestamp DESC
            """, (play['event_id'], play['prop_type'],
                  play['player_name'], original_line, sharp_book))
            
            rows = cursor.fetchall()
            
            # Group by outcome to get latest for each
            outcomes = {}
            for row in rows:
                row_dict = dict(row)
                if row_dict['outcome'] not in outcomes:
                    outcomes[row_dict['outcome']] = row_dict
            
            used_line = original_line
            
            # If we don't have both sides at exact line, try closest line
            if len(outcomes) < 2:
                cursor.execute("""
                    SELECT outcome, price_decimal, line
                    FROM prop_snapshots
                    WHERE event_id = ? AND prop_type = ?
                          AND player_name = ? AND bookmaker = ?
                          AND outcome IN ('Over', 'Under')
                    ORDER BY ABS(line - ?) ASC, timestamp DESC
                """, (play['event_id'], play['prop_type'],
                      play['player_name'], sharp_book, original_line))
                
                rows = cursor.fetchall()
                outcomes = {}
                for row in rows:
                    row_dict = dict(row)
                    # Only include outcomes from the same line
                    if not outcomes:
                        used_line = row_dict['line']
                    if row_dict['line'] == used_line and row_dict['outcome'] not in outcomes:
                        outcomes[row_dict['outcome']] = row_dict
            
            # Need both sides to devig
            if outcome not in outcomes or opposite not in outcomes:
                return None
            
            our_side = outcomes[outcome]
            opp_side = outcomes[opposite]
            
            # Devig to get fair probabilities
            decimals = [our_side['price_decimal'], opp_side['price_decimal']]
            fair_probs = calculate_no_vig_probability(decimals, method=devig_method)
            
            # fair_probs[0] is for our side
            fair_prob = fair_probs[0]
            fair_decimal = 1 / fair_prob if fair_prob > 0 else 1.01
            fair_american = decimal_to_american(fair_decimal)
            
            result = {
                'fair_decimal': fair_decimal,
                'fair_american': fair_american,
                'raw_decimal': our_side['price_decimal'],
                'bookmaker': sharp_book,
            }
            
            if used_line != original_line:
                result['alternate_line'] = used_line
                
            return result
            
        else:
            # EV play (game lines) - get both sides of the market
            outcome = play.get('outcome')
            market = play.get('market')
            point = play.get('point')
            
            if point is not None:
                cursor.execute("""
                    SELECT outcome, price_decimal, point
                    FROM odds_snapshots
                    WHERE event_id = ? AND market = ? AND point = ?
                          AND bookmaker = ?
                    ORDER BY timestamp DESC
                """, (play['event_id'], market, point, sharp_book))
            else:
                cursor.execute("""
                    SELECT outcome, price_decimal, point
                    FROM odds_snapshots
                    WHERE event_id = ? AND market = ? AND point IS NULL
                          AND bookmaker = ?
                    ORDER BY timestamp DESC
                """, (play['event_id'], market, sharp_book))
            
            rows = cursor.fetchall()
            
            # Group by outcome
            outcomes = {}
            for row in rows:
                row_dict = dict(row)
                if row_dict['outcome'] not in outcomes:
                    outcomes[row_dict['outcome']] = row_dict
            
            # Need at least 2 outcomes to devig
            if len(outcomes) < 2 or outcome not in outcomes:
                return None
            
            # Get all decimal odds for devigging
            decimals = [outcomes[o]['price_decimal'] for o in outcomes]
            fair_probs = calculate_no_vig_probability(decimals, method=devig_method)
            
            # Find the fair prob for our outcome
            outcome_keys = list(outcomes.keys())
            our_idx = outcome_keys.index(outcome)
            fair_prob = fair_probs[our_idx]
            fair_decimal = 1 / fair_prob if fair_prob > 0 else 1.01
            fair_american = decimal_to_american(fair_decimal)
            
            return {
                'fair_decimal': fair_decimal,
                'fair_american': fair_american,
                'raw_decimal': outcomes[outcome]['price_decimal'],
                'bookmaker': sharp_book,
            }

    def _get_cents_per_half_point(self, prop_type: str) -> int:
        """
        Get the appropriate cents-per-half-point adjustment for a prop type.
        
        Low-volume stats (assists, rebounds, 3s) have higher cents per half-point
        because each 0.5 represents a larger portion of expected output.
        
        High-volume stats (points) have lower cents per half-point.
        """
        # Low-volume props: ~40-50 cents per half-point
        low_volume_props = {
            'player_assists': 45,
            'player_rebounds': 40,
            'player_threes': 50,  # 3-pointers are very low volume
            'player_blocks': 50,
            'player_steals': 50,
            'player_turnovers': 45,
            'batter_hits': 45,
            'batter_rbis': 50,
            'batter_runs': 50,
            'batter_home_runs': 60,
            'pitcher_hits_allowed': 35,
            'pitcher_walks': 50,
            'player_receptions': 40,
            'player_rush_attempts': 35,
        }
        
        # Medium-volume props: ~30-35 cents per half-point
        medium_volume_props = {
            'player_points_rebounds_assists': 30,
            'player_points_rebounds': 30,
            'player_points_assists': 30,
            'player_rebounds_assists': 35,
            'pitcher_strikeouts': 35,
            'pitcher_outs': 30,
        }
        
        # High-volume props: ~20-25 cents per half-point
        high_volume_props = {
            'player_points': 25,
            'player_pass_yds': 20,
            'player_rush_yds': 25,
            'player_reception_yds': 25,
        }
        
        if prop_type in low_volume_props:
            return low_volume_props[prop_type]
        elif prop_type in medium_volume_props:
            return medium_volume_props[prop_type]
        elif prop_type in high_volume_props:
            return high_volume_props[prop_type]
        else:
            # Default to medium-high for unknown props
            return 35

    def _estimate_odds_at_original_line(
        self,
        current_odds_decimal: float,
        current_line: float,
        original_line: float,
        outcome: str,
        prop_type: str = None
    ) -> float:
        """
        Estimate what the odds would be at the original line based on current alternate line odds.
        
        Uses prop-type-specific adjustments - low-volume stats like assists have higher
        cents-per-half-point than high-volume stats like points.
        
        Args:
            current_odds_decimal: Current best odds at the alternate line
            current_line: The alternate line we found odds for
            original_line: The line we originally bet on
            outcome: "Over" or "Under"
            prop_type: The prop type (e.g., 'player_assists') for specific adjustments
            
        Returns:
            Estimated decimal odds at the original line
        """
        from arbitrage import american_to_decimal, decimal_to_american
        
        # Get prop-type-specific cents per half-point
        cents_per_half_point = self._get_cents_per_half_point(prop_type) if prop_type else 35
        
        # Convert current odds to American for easier cents math
        current_american = decimal_to_american(current_odds_decimal)
        
        # Calculate line difference (in half-points)
        line_diff = original_line - current_line  # positive = original is higher
        half_points = line_diff / 0.5
        
        # Calculate cents adjustment
        # line_diff is negative when original is lower (e.g., 4.5 - 5.5 = -1)
        # For Over: lower original line = easier to hit = SHORTER odds (more negative)
        # For Under: lower original line = harder to hit = LONGER odds (more positive)
        cents_adjustment = half_points * cents_per_half_point
        
        if outcome.lower() == "over":
            # For Over: add cents_adjustment (which is negative when line went up)
            # Example: 4.5 vs 5.5, Over 4.5 easier, -107 + (-90) = -197 (shorter)
            adjusted_american = current_american + cents_adjustment
        else:  # Under
            # For Under: subtract cents_adjustment
            # Example: 4.5 vs 5.5, Under 4.5 harder, -107 - (-90) = -17 → longer odds
            adjusted_american = current_american - cents_adjustment
        
        # Handle the American odds sign flip zone (-100 to +100 doesn't exist)
        if -100 < adjusted_american < 100:
            if adjusted_american >= 0:
                adjusted_american = 100
            else:
                adjusted_american = -100
        
        # Convert back to decimal, ensuring minimum valid odds
        estimated_decimal = american_to_decimal(int(adjusted_american))
        return max(1.01, estimated_decimal)

    def get_play_by_id_combined(self, play_id: int, play_type: str) -> Optional[dict]:
        """Get a single play by ID and type."""
        with get_db() as conn:
            cursor = conn.cursor()
            if play_type == 'ev':
                cursor.execute("SELECT *, 'ev' as play_type FROM plays WHERE id = ?", (play_id,))
            else:
                cursor.execute("SELECT *, 'prop' as play_type FROM prop_plays WHERE id = ?", (play_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_play_result_combined(self, play_id: int, play_type: str, result: str) -> float:
        """Update play result for either EV or prop play."""
        if play_type == 'prop':
            return self.update_prop_play_result_and_profit(play_id, result)
        else:
            return self.update_play_result_and_profit(play_id, result)

    def get_combined_profit_history(self) -> dict:
        """Get cumulative profit over time for all plays combined.
        
        Groups plays by CST date to show daily progression.
        Returns dates as ISO strings in CST timezone for Chart.js.
        """
        with get_db() as conn:
            cursor = conn.cursor()

            # Get all completed plays from both tables
            cursor.execute("""
                SELECT sent_at, profit_units, 'ev' as type FROM plays
                WHERE result IS NOT NULL AND profit_units IS NOT NULL
                UNION ALL
                SELECT sent_at, profit_units, 'prop' as type FROM prop_plays
                WHERE result IS NOT NULL AND profit_units IS NOT NULL
                ORDER BY sent_at ASC
            """)

            rows = cursor.fetchall()

            if not rows:
                return {"points": []}

            # Group by CST date and aggregate daily profits
            daily_profits = {}
            
            for row in rows:
                # Convert UTC timestamp to Central timezone date
                utc_dt = datetime.fromisoformat(row["sent_at"].replace('Z', '+00:00'))
                if utc_dt.tzinfo is None:
                    utc_dt = utc_dt.replace(tzinfo=timezone.utc)
                cst_dt = utc_dt.astimezone(CST_TZ)
                # Get date only (YYYY-MM-DD)
                cst_date_key = cst_dt.date()
                
                if cst_date_key not in daily_profits:
                    daily_profits[cst_date_key] = 0.0
                
                daily_profits[cst_date_key] += row["profit_units"]

            # Convert to sorted list of points with cumulative profit
            sorted_dates = sorted(daily_profits.keys())
            points = []
            cumulative = 0.0

            for date_key in sorted_dates:
                daily_profit = daily_profits[date_key]
                cumulative += daily_profit
                
                # Create datetime at midnight Central time for this date
                cst_midnight = datetime.combine(date_key, datetime.min.time()).replace(tzinfo=CST_TZ)
                # Return as ISO string (Chart.js can parse this)
                date_iso = cst_midnight.isoformat()
                
                points.append({
                    "date": date_iso,
                    "profit": round(daily_profit, 2),
                    "cumulative_profit": round(cumulative, 2)
                })

            return {"points": points}


if __name__ == "__main__":
    # Initialize database when run directly
    logging.basicConfig(level=logging.INFO)
    init_database()
    print("Database initialized successfully!")
    stats = get_database_stats()
    print(f"Stats: {stats}")
