# uv manages Python (see .python-version), .venv, and the locked dependencies.
UV ?= uv
DATA_DIR ?= data
RUN := $(UV) run --locked

.PHONY: help install setup samples setup-samples setup-real download run \
        dev lint format format-check setup-llm setup-embeddings setup-all check test smoke validate render clean-data clean-caches clean

help:
	@echo "make setup          Sync locked dependencies and generate sample data"
	@echo "make run            Launch Jupyter in the uv-managed environment"
	@echo "make setup-llm      Add MLX-LM / LoRA dependencies"
	@echo "make setup-embeddings Add Sentence Transformers / PyTorch"
	@echo "make setup-all      Install both optional groups"
	@echo "make format-check   Check reusable Python formatting"
	@echo "make setup-samples  Generate sample data without installing dependencies"
	@echo "make setup-real     Download SNIPS, IMDB, Banking77, and WikiText"
	@echo "make check          Run lint, notebook validation, and regression tests"
	@echo "make smoke          Execute offline notebooks with reduced training"
	@echo "make validate       Execute offline notebooks at full budget, enforcing sanity checks"
	@echo "make render         Like validate, saving executed notebooks in rendered/"

install dev:
	$(UV) sync --locked

# The sample generator uses only the standard library. --no-project avoids
# downloading the ML stack and does not modify the project's environment.
samples setup-samples:
	$(UV) run --no-project --python "$(shell cat .python-version)" scripts/download_datasets.py --samples --data-dir "$(DATA_DIR)"

setup: install samples
	@echo "Setup complete. Run 'make run', then open 00_Overview.ipynb."

run:
	$(RUN) jupyter notebook --notebook-dir=notebooks

setup-real:
	$(RUN) python scripts/download_datasets.py --all --data-dir "$(DATA_DIR)"

download: setup-real

setup-llm:
	$(UV) sync --locked --group llm

setup-embeddings:
	$(UV) sync --locked --group embeddings

setup-all:
	$(UV) sync --locked --all-groups

format:
	$(RUN) ruff format notebooks scripts tests

format-check:
	$(RUN) ruff format --check notebooks scripts tests

lint:
	$(RUN) ruff check notebooks scripts tests

check: lint format-check
	$(RUN) python scripts/check_notebooks.py
	$(RUN) python -m pytest -q

test:
	$(RUN) python -m pytest -q

smoke:
	$(RUN) python scripts/check_notebooks.py --execute --quick

validate:
	$(RUN) python scripts/check_notebooks.py --execute

render:
	$(RUN) python scripts/check_notebooks.py --execute --output-dir rendered

clean-data:
	rm -rf -- "$(DATA_DIR)/imdb" "$(DATA_DIR)/snips" "$(DATA_DIR)/banking77" "$(DATA_DIR)/wikitext"

clean-caches:
	find notebooks scripts tests -type d -name __pycache__ -exec rm -rf {} +
	rm -rf -- .pytest_cache .ruff_cache

clean: clean-data clean-caches
	rm -rf -- .venv
