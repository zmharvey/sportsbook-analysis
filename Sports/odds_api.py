"""
Client for The Odds API.
Handles all API requests and data fetching.
"""

import requests
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class APIUsage:
    """Tracks API quota usage."""
    requests_remaining: int
    requests_used: int
    last_request_cost: int


class OddsAPIClient:
    """Client for interacting with The Odds API."""

    def __init__(self, api_key: str, base_url: str = "https://api.the-odds-api.com/v4"):
        self.api_key = api_key
        self.base_url = base_url
        self.usage: Optional[APIUsage] = None
        self.scan_cost: int = 0  # Track cost for current scan

    def reset_scan_cost(self) -> None:
        """Reset the scan cost counter (call at start of each scan)."""
        self.scan_cost = 0

    def _update_usage(self, response: requests.Response) -> None:
        """Update API usage stats from response headers."""
        try:
            last_cost = int(response.headers.get("x-requests-last", 0))
            self.scan_cost += last_cost
            self.usage = APIUsage(
                requests_remaining=int(response.headers.get("x-requests-remaining", 0)),
                requests_used=int(response.headers.get("x-requests-used", 0)),
                last_request_cost=last_cost,
            )
        except (ValueError, TypeError):
            pass

    def get_sports(self, all_sports: bool = False) -> list[dict]:
        """
        Get list of available sports.

        Args:
            all_sports: If True, include out-of-season sports.

        Returns:
            List of sport dictionaries with keys: key, group, title, description, active, has_outrights
        """
        params = {"apiKey": self.api_key}
        if all_sports:
            params["all"] = "true"

        response = requests.get(f"{self.base_url}/sports", params=params)
        self._update_usage(response)
        response.raise_for_status()

        return response.json()

    def get_odds(
        self,
        sport: str,
        regions: list[str],
        markets: list[str],
        odds_format: str = "american",
        bookmakers: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Get odds for a specific sport.

        Args:
            sport: Sport key (e.g., "americanfootball_nfl")
            regions: List of regions (e.g., ["us", "uk"])
            markets: List of markets (e.g., ["h2h", "spreads", "totals"])
            odds_format: "american" or "decimal"
            bookmakers: Optional list of specific bookmakers to include

        Returns:
            List of event dictionaries with odds data
        """
        params = {
            "apiKey": self.api_key,
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        url = f"{self.base_url}/sports/{sport}/odds"
        logger.debug(f"Fetching odds from: {url}")

        response = requests.get(url, params=params)
        self._update_usage(response)

        if response.status_code == 404:
            logger.warning(f"No events found for sport: {sport}")
            return []

        response.raise_for_status()
        return response.json()

    def get_event_odds(
        self,
        sport: str,
        event_id: str,
        regions: list[str],
        markets: list[str],
        odds_format: str = "american",
        bookmakers: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """
        Get odds for a specific event.

        Args:
            sport: Sport key
            event_id: Event ID
            regions: List of regions
            markets: List of markets
            odds_format: "american" or "decimal"
            bookmakers: Optional list of specific bookmakers to include

        Returns:
            Event dictionary with odds data, or None if not found
        """
        params = {
            "apiKey": self.api_key,
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        url = f"{self.base_url}/sports/{sport}/events/{event_id}/odds"
        response = requests.get(url, params=params)
        self._update_usage(response)

        if response.status_code == 404:
            return None

        response.raise_for_status()
        return response.json()

    def get_events(
        self,
        sport: str,
    ) -> list[dict]:
        """
        Get list of events for a sport (without odds).
        This is useful for getting event IDs before fetching player props.

        Args:
            sport: Sport key (e.g., "basketball_nba")

        Returns:
            List of event dictionaries with id, sport_key, sport_title,
            commence_time, home_team, away_team
        """
        params = {"apiKey": self.api_key}
        url = f"{self.base_url}/sports/{sport}/events"

        logger.debug(f"Fetching events from: {url}")
        response = requests.get(url, params=params)
        self._update_usage(response)

        if response.status_code == 404:
            logger.warning(f"No events found for sport: {sport}")
            return []

        response.raise_for_status()
        return response.json()

    def get_player_props(
        self,
        sport: str,
        event_id: str,
        regions: list[str],
        markets: list[str],
        odds_format: str = "american",
        bookmakers: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """
        Get player prop odds for a specific event.

        Uses the event odds endpoint with player prop market keys.
        Player props are fetched per-event, which costs more API credits
        but provides detailed prop lines.

        Args:
            sport: Sport key (e.g., "basketball_nba")
            event_id: Event ID from get_events()
            regions: List of regions (e.g., ["us", "eu"])
            markets: List of player prop markets (e.g., ["player_points", "player_rebounds"])
            odds_format: "american" or "decimal"
            bookmakers: Optional list of specific bookmakers to include

        Returns:
            Event dictionary with player prop odds data, or None if not found
        """
        params = {
            "apiKey": self.api_key,
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        url = f"{self.base_url}/sports/{sport}/events/{event_id}/odds"
        logger.debug(f"Fetching player props from: {url} with markets: {markets}")

        response = requests.get(url, params=params)
        self._update_usage(response)

        if response.status_code == 404:
            logger.debug(f"No player props found for event: {event_id}")
            return None

        response.raise_for_status()
        return response.json()

    def get_quota_usage(self) -> Optional[APIUsage]:
        """Get the current API quota usage stats."""
        return self.usage

    def print_usage(self) -> None:
        """Print current API usage to console."""
        if self.usage:
            print(f"\n📊 API Usage:")
            print(f"   Requests remaining: {self.usage.requests_remaining}")
            print(f"   Requests used (total): {self.usage.requests_used}")
            print(f"   This scan cost: {self.scan_cost}")
