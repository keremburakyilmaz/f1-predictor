"""OpenF1 API client with local JSON caching."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OPENF1_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "openf1"
OPENF1_BASE_URL = "https://api.openf1.org/v1"


class OpenF1Client:
    """Small cached wrapper around the OpenF1 REST API."""

    def __init__(
        self,
        base_url: str = OPENF1_BASE_URL,
        cache_dir: Path = OPENF1_CACHE_DIR,
        timeout_seconds: int = 30,
    ) -> None:
        """Create an OpenF1 client.

        Args:
            base_url: OpenF1 API base URL.
            cache_dir: Directory for cached JSON responses.
            timeout_seconds: HTTP request timeout.

        Returns:
            None.
        """
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        normalized = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        safe_endpoint = endpoint.strip("/").replace("/", "_")
        return self.cache_dir / safe_endpoint / f"{digest}.json"

    def get(
        self, endpoint: str, params: dict[str, Any] | None = None, force: bool = False
    ) -> list[dict[str, Any]]:
        """Fetch an OpenF1 endpoint with disk caching.

        Args:
            endpoint: Endpoint path, for example `sessions`.
            params: Query parameters.
            force: When True, bypass the JSON cache.

        Returns:
            Parsed JSON list from OpenF1.
        """
        query_params = {key: value for key, value in (params or {}).items() if value is not None}
        cache_path = self._cache_path(endpoint, query_params)
        if cache_path.exists() and not force:
            LOGGER.info("Loading cached OpenF1 response from %s", cache_path)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        url = f"{self.base_url}/{endpoint.strip('/')}"
        LOGGER.info("Fetching OpenF1 endpoint %s with params %s", endpoint, query_params)
        response = self.session.get(url, params=query_params, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def lookup_session_key(
        self, year: int, round_num: int, session_name: str = "Race", force: bool = False
    ) -> int:
        """Lookup an OpenF1 session key by chronological round number.

        Args:
            year: F1 season year.
            round_num: Race round number.
            session_name: OpenF1 session name, usually `Race`.
            force: When True, bypass the JSON cache.

        Returns:
            OpenF1 session key.
        """
        sessions = self.get(
            "sessions",
            params={"year": year, "session_name": session_name},
            force=force,
        )
        ordered = sorted(sessions, key=lambda row: row.get("date_start", ""))
        try:
            return int(ordered[round_num - 1]["session_key"])
        except IndexError as exc:
            raise ValueError(
                f"Could not find OpenF1 session for {year} round {round_num}"
            ) from exc

    def get_driver_info(
        self, session_key: int, force: bool = False
    ) -> list[dict[str, Any]]:
        """Return OpenF1 driver metadata for a session.

        Args:
            session_key: OpenF1 session key.
            force: When True, bypass the JSON cache.

        Returns:
            List of driver metadata records.
        """
        return self.get("drivers", params={"session_key": session_key}, force=force)

    def get_pit_stops(
        self, year: int, round_num: int, force: bool = False
    ) -> list[dict[str, Any]]:
        """Return pit stop records for a race round.

        Args:
            year: F1 season year.
            round_num: Race round number.
            force: When True, bypass the JSON cache.

        Returns:
            List of pit stop records.
        """
        session_key = self.lookup_session_key(year, round_num, force=force)
        return self.get("pit", params={"session_key": session_key}, force=force)

    def get_stints(
        self, year: int, round_num: int, force: bool = False
    ) -> list[dict[str, Any]]:
        """Return stint records for a race round.

        Args:
            year: F1 season year.
            round_num: Race round number.
            force: When True, bypass the JSON cache.

        Returns:
            List of stint records.
        """
        session_key = self.lookup_session_key(year, round_num, force=force)
        return self.get("stints", params={"session_key": session_key}, force=force)
