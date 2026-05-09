.PHONY: install fetch features train serve test lint format

install:
	python -m pip install -e ".[core,ml,tracking,pipeline,api,dev]"

fetch:
	python -m f1predictor.data.fetch --params params.yaml

features:
	python -m f1predictor.features.build --params params.yaml

train:
	python -m f1predictor.models.train --params params.yaml

serve:
	uvicorn f1predictor.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check src tests

format:
	black src tests
