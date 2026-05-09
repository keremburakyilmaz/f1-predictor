"""FastF1-backed schedule helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import fastf1
import pandas as pd


def _race_datetime(event: pd.Series) -> pd.Timestamp:
    """Return the best available race timestamp for a FastF1 schedule row."""
    for column in ("Session5DateUtc", "Session5Date", "EventDate"):
        if column in event and pd.notna(event[column]):
            timestamp = pd.Timestamp(event[column])
            if timestamp.tzinfo is None:
                return timestamp.tz_localize(UTC)
            return timestamp.tz_convert(UTC)
    raise ValueError(f"Schedule row has no race date: {event.to_dict()}")


def _clean_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Keep race rounds and attach a normalized race timestamp."""
    if schedule.empty:
        return schedule.copy()

    cleaned = schedule.copy()
    if "RoundNumber" not in cleaned.columns:
        raise ValueError("FastF1 schedule is missing RoundNumber")

    cleaned = cleaned[cleaned["RoundNumber"].fillna(0).astype(int) > 0].copy()
    cleaned["race_date"] = cleaned.apply(_race_datetime, axis=1)
    return cleaned.sort_values(["race_date", "RoundNumber"]).reset_index(drop=True)


def get_completed_rounds(year: int, now: datetime | None = None) -> list[int]:
    """Return completed round numbers for a season.

    Args:
        year: F1 season year.
        now: Optional comparison timestamp, primarily for tests.

    Returns:
        Completed round numbers sorted by race date.
    """
    comparison_time = now or datetime.now(UTC)
    if comparison_time.tzinfo is None:
        comparison_time = comparison_time.replace(tzinfo=UTC)

    schedule = _clean_schedule(fastf1.get_event_schedule(year))
    completed = schedule[schedule["race_date"] < pd.Timestamp(comparison_time)]
    return completed["RoundNumber"].astype(int).tolist()


def get_next_round(year: int, now: datetime | None = None) -> dict[str, Any] | None:
    """Return metadata for the next upcoming race in a season.

    Args:
        year: F1 season year.
        now: Optional comparison timestamp, primarily for tests.

    Returns:
        A metadata dictionary for the next race, or None when the season is over.
    """
    comparison_time = now or datetime.now(UTC)
    if comparison_time.tzinfo is None:
        comparison_time = comparison_time.replace(tzinfo=UTC)

    schedule = _clean_schedule(fastf1.get_event_schedule(year))
    upcoming = schedule[schedule["race_date"] >= pd.Timestamp(comparison_time)]
    if upcoming.empty:
        return None

    event = upcoming.iloc[0]
    race_date = pd.Timestamp(event["race_date"])
    return {
        "year": year,
        "round": int(event["RoundNumber"]),
        "event_name": str(event.get("EventName", "")),
        "official_event_name": str(event.get("OfficialEventName", "")),
        "location": str(event.get("Location", "")),
        "country": str(event.get("Country", "")),
        "race_date": race_date.isoformat(),
        "days_until_race": max(
            0, int((race_date - pd.Timestamp(comparison_time)).total_seconds() // 86400)
        ),
    }


def get_season_total_rounds(year: int) -> int:
    """Return the number of race rounds in a season.

    Args:
        year: F1 season year.

    Returns:
        Count of race rounds in the FastF1 event schedule.
    """
    schedule = _clean_schedule(fastf1.get_event_schedule(year))
    return int(schedule["RoundNumber"].nunique())
