# F1 Predictor

Production-oriented F1 race finish-position prediction system.

This repository is being rebuilt from the Phase 1 implementation plan:

- Python package scaffold under `src/f1predictor`
- FastF1 race and qualifying ingestion
- OpenF1 API client with local JSON caching
- Schedule utilities for completed and upcoming rounds
- DVC/params/Makefile plumbing for the data pipeline

Phase 1 deliberately stops at the data layer. Feature engineering, model training,
pipeline automation, API serving, and the website are later phases.

## Quick Start

```powershell
python -m pip install -e ".[core,dev]"
make fetch
make test
```

`make fetch` downloads completed seasons only. For an in-progress season, the
schedule utility asks FastF1 which rounds have race dates in the past and fetches
only those rounds.
