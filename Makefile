# Convenience targets. Everything here is a thin wrapper around the qmr CLI.

PYTHON ?= python
PORT   ?= 8501

.PHONY: help install install-dev samples datasets run compare console lint format clean

help:
	@echo "install      install the package and its dependencies"
	@echo "install-dev  install with the development extras"
	@echo "samples      rebuild the trimmed sample datasets from data/raw"
	@echo "datasets     list the discovered market history"
	@echo "run          run one study with the default configuration"
	@echo "compare      sweep learners against regime methods"
	@echo "console      launch the research console (PORT=$(PORT))"
	@echo "lint         ruff check"
	@echo "format       black + ruff --fix"
	@echo "clean        remove caches and build artefacts"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

samples:
	$(PYTHON) scripts/prepare_sample_data.py --bars 15000

datasets:
	$(PYTHON) -m qmr.cli datasets

run:
	$(PYTHON) -m qmr.cli run

compare:
	$(PYTHON) -m qmr.cli compare \
		--models logistic,random_forest,xgboost,lightgbm \
		--regimes none,kmeans \
		--output reports/model_comparison.csv

console:
	$(PYTHON) -m streamlit run app/main.py --server.port $(PORT)

lint:
	ruff check src app scripts

format:
	black src app scripts
	ruff check --fix src app scripts

clean:
	rm -rf build dist *.egg-info .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
