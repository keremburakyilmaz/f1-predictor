"""Command line entry point for Phase 1 data ingestion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from f1predictor.data.fastf1_loader import fetch_all

LOGGER = logging.getLogger(__name__)


def _load_years(params_path: Path) -> list[int]:
    params: dict[str, Any] = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    years = params.get("data", {}).get("years")
    if not years:
        raise ValueError(f"No data.years configured in {params_path}")
    return [int(year) for year in years]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        None.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Fetch completed F1 race data.")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the Phase 1 fetch pipeline.

    Args:
        None.

    Returns:
        None.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    years = _load_years(args.params)
    LOGGER.info("Fetching years: %s", years)

    df = fetch_all(years, force=args.force)
    if df.empty:
        LOGGER.warning("Fetch completed but no race rows were returned")
        return

    summary_path = Path("data/raw/fastf1/all_races.parquet")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(summary_path, index=False)

    round_counts = (
        df.groupby("year", as_index=False)["round"]
        .nunique()
        .rename(columns={"round": "completed_rounds"})
    )
    LOGGER.info("Fetched race rows: %s", len(df))
    LOGGER.info("Completed rounds by year:\n%s", round_counts.to_string(index=False))


if __name__ == "__main__":
    main()
