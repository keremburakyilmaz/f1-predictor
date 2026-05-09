"""FastF1 race and qualifying ingestion."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import fastf1
import pandas as pd

from f1predictor.data.schedule import get_completed_rounds

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FASTF1_CACHE_DIR = RAW_DIR / "fastf1_cache"
FASTF1_PARQUET_DIR = RAW_DIR / "fastf1"
MAX_DNF_POSITION = 22


def configure_fastf1_cache(cache_dir: Path = FASTF1_CACHE_DIR) -> None:
    """Enable the FastF1 disk cache.

    Args:
        cache_dir: Directory used by FastF1 for HTTP/session cache files.

    Returns:
        None.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def _race_cache_path(year: int, round_num: int) -> Path:
    return FASTF1_PARQUET_DIR / f"{year}_R{round_num:02d}_race.parquet"


def _qualifying_cache_path(year: int, round_num: int) -> Path:
    return FASTF1_PARQUET_DIR / f"{year}_R{round_num:02d}_qualifying.parquet"


def _session_race_id(year: int, round_num: int) -> str:
    return f"{year}_R{round_num:02d}"


def _safe_numeric(value: object, default: float | None = None) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return float(numeric)


def _finish_position(row: pd.Series) -> int:
    classified = str(row.get("ClassifiedPosition", "")).strip()
    status = str(row.get("Status", "")).strip().lower()

    if classified.isdigit():
        position = int(classified)
        if "disqualified" in status or "not started" in status or status == "withdrawn":
            return MAX_DNF_POSITION
        return position

    position = _safe_numeric(row.get("Position"))
    if position is None:
        return MAX_DNF_POSITION
    return min(int(position), MAX_DNF_POSITION)


def _weather_aggregates(session: fastf1.core.Session) -> dict[str, float | bool | None]:
    weather = getattr(session, "weather_data", None)
    if weather is None or weather.empty:
        return {
            "track_temp_mean": None,
            "air_temp_mean": None,
            "wind_speed_mean": None,
            "is_wet_race": False,
        }

    track_col = "TrackTemp" if "TrackTemp" in weather.columns else "TrackTemperature"
    air_col = "AirTemp" if "AirTemp" in weather.columns else "AirTemperature"

    rainfall = weather["Rainfall"] if "Rainfall" in weather.columns else pd.Series(dtype=bool)
    return {
        "track_temp_mean": _safe_numeric(weather.get(track_col, pd.Series()).mean()),
        "air_temp_mean": _safe_numeric(weather.get(air_col, pd.Series()).mean()),
        "wind_speed_mean": _safe_numeric(weather.get("WindSpeed", pd.Series()).mean()),
        "is_wet_race": bool(rainfall.fillna(False).astype(bool).any()),
    }


def _fastest_lap_drivers(session: fastf1.core.Session) -> set[str]:
    laps = getattr(session, "laps", None)
    if laps is None or laps.empty or "LapTime" not in laps.columns or "Driver" not in laps.columns:
        return set()

    timed_laps = laps.dropna(subset=["LapTime"])
    if timed_laps.empty:
        return set()

    fastest_lap = timed_laps.loc[timed_laps["LapTime"].idxmin()]
    return {str(fastest_lap["Driver"])}


def fetch_race(year: int, round_num: int, force: bool = False) -> pd.DataFrame:
    """Fetch and cache race result rows for one round.

    Args:
        year: F1 season year.
        round_num: Race round number.
        force: When True, refetch even if the parquet cache exists.

    Returns:
        One row per driver with race result and weather columns.
    """
    configure_fastf1_cache()
    cache_path = _race_cache_path(year, round_num)
    if cache_path.exists() and not force:
        LOGGER.info("Loading cached race data from %s", cache_path)
        return pd.read_parquet(cache_path)

    LOGGER.info("Fetching race data for %s round %s", year, round_num)
    session = fastf1.get_session(year, round_num, "R")
    session.load(weather=True, laps=True, messages=True)

    results = session.results.copy()
    if results.empty:
        raise ValueError(f"No race results returned for {year} round {round_num}")

    event = session.event
    weather = _weather_aggregates(session)
    fastest_lap_drivers = _fastest_lap_drivers(session)
    race_id = _session_race_id(year, round_num)

    rows: list[dict[str, object]] = []
    for _, result in results.iterrows():
        driver = str(result.get("Abbreviation", "")).strip()
        rows.append(
            {
                "race_id": race_id,
                "year": year,
                "round": round_num,
                "circuit": str(event.get("Location", "")),
                "country": str(event.get("Country", "")),
                "race_date": pd.Timestamp(event.get("EventDate")).date().isoformat(),
                "driver": driver,
                "team": str(result.get("TeamName", "")).strip(),
                "grid_pos": int(_safe_numeric(result.get("GridPosition"), MAX_DNF_POSITION)),
                "finish_pos": _finish_position(result),
                "finish_status": str(result.get("Status", "")).strip(),
                "points": float(_safe_numeric(result.get("Points"), 0.0)),
                "fastest_lap": driver in fastest_lap_drivers,
                **weather,
            }
        )

    df = pd.DataFrame(rows).sort_values(["finish_pos", "driver"]).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def _best_qualifying_time(row: pd.Series) -> pd.Timedelta | pd.NaT:
    times = [row.get(column) for column in ("Q1", "Q2", "Q3")]
    valid_times = [pd.Timedelta(value) for value in times if pd.notna(value)]
    if not valid_times:
        return pd.NaT
    return min(valid_times)


def fetch_qualifying(year: int, round_num: int, force: bool = False) -> pd.DataFrame:
    """Fetch and cache qualifying rows for one round.

    Args:
        year: F1 season year.
        round_num: Race round number.
        force: When True, refetch even if the parquet cache exists.

    Returns:
        One row per driver with qualifying position and gap to pole in ms.
    """
    configure_fastf1_cache()
    cache_path = _qualifying_cache_path(year, round_num)
    if cache_path.exists() and not force:
        LOGGER.info("Loading cached qualifying data from %s", cache_path)
        return pd.read_parquet(cache_path)

    LOGGER.info("Fetching qualifying data for %s round %s", year, round_num)
    session = fastf1.get_session(year, round_num, "Q")
    session.load(weather=False, laps=False, messages=False)

    results = session.results.copy()
    if results.empty:
        raise ValueError(f"No qualifying results returned for {year} round {round_num}")

    results["best_time"] = results.apply(_best_qualifying_time, axis=1)
    pole_time = results["best_time"].dropna().min()

    rows: list[dict[str, object]] = []
    for _, result in results.iterrows():
        best_time = result.get("best_time")
        gap_ms = None
        if pd.notna(best_time) and pd.notna(pole_time):
            gap_ms = float((best_time - pole_time).total_seconds() * 1000)

        rows.append(
            {
                "race_id": _session_race_id(year, round_num),
                "year": year,
                "round": round_num,
                "driver": str(result.get("Abbreviation", "")).strip(),
                "quali_pos": int(_safe_numeric(result.get("Position"), MAX_DNF_POSITION)),
                "quali_gap_to_pole_ms": gap_ms,
            }
        )

    df = pd.DataFrame(rows).sort_values(["quali_pos", "driver"]).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def _fetch_round(year: int, round_num: int, force: bool) -> pd.DataFrame:
    race = fetch_race(year, round_num, force=force)
    fetch_qualifying(year, round_num, force=force)
    return race


def fetch_season(year: int, force: bool = False, max_workers: int = 4) -> pd.DataFrame:
    """Fetch completed race rounds for a season.

    Args:
        year: F1 season year.
        force: When True, refetch even if caches exist.
        max_workers: Maximum concurrent FastF1 round fetches.

    Returns:
        Race result dataframe for all completed rounds in the season.
    """
    rounds = get_completed_rounds(year)
    if not rounds:
        LOGGER.warning("No completed rounds found for %s", year)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_round, year, round_num, force): round_num
            for round_num in rounds
        }
        for future in as_completed(futures):
            round_num = futures[future]
            try:
                frames.append(future.result())
            except Exception:
                LOGGER.exception("Failed to fetch %s round %s", year, round_num)
                raise

    return pd.concat(frames, ignore_index=True).sort_values(["year", "round", "finish_pos"])


def fetch_all(years: Iterable[int], force: bool = False) -> pd.DataFrame:
    """Fetch completed rounds for multiple seasons.

    Args:
        years: Iterable of F1 season years.
        force: When True, refetch even if caches exist.

    Returns:
        Concatenated race result dataframe for all fetched seasons.
    """
    frames = [fetch_season(year, force=force) for year in years]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["year", "round", "finish_pos"])
