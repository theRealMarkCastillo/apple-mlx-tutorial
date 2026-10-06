# Python 3.12+ on Apple Silicon. Override PY or VENV if needed.
PY ?= python3
VENV ?= .venv
PYBIN := $(VENV)/bin/python
JUPYTER := $(VENV)/bin/jupyter
DATA_DIR ?= data

.PHONY: help venv install setup samples setup-samples setup-real download run \
        dev check test smoke clean-data clean-caches clean

help:
	@echo "make setup          Install pinned dependencies and generate sample data"
	@echo "make run            Launch Jupyter (no reinstall when dependencies are unchanged)"
	@echo "make setup-samples  Generate offline sample data only"
	@echo "make setup-real     Download SNIPS, IMDB, Banking77, and WikiText"
	@echo "make check          Run lint, notebook validation, and regression tests"
	@echo "make smoke          Execute offline notebooks with reduced training"

$(PYBIN):
	$(PY) -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ required"'
	$(PY) -m venv $(VENV)

venv: $(PYBIN)

$(VENV)/.deps.stamp: requirements.txt $(PYBIN)
	$(PYBIN) -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ required"'
	$(PYBIN) -m ensurepip --upgrade
	$(PYBIN) -m pip install -r requirements.txt
	@touch $@

install: $(VENV)/.deps.stamp

samples setup-samples: venv
	$(PYBIN) scripts/download_datasets.py --samples --data-dir "$(DATA_DIR)"

setup: install samples
	@echo "Setup complete. Run 'make run', then open 00_Overview.ipynb."

run: install
	$(JUPYTER) notebook --notebook-dir=notebooks

setup-real: install
	$(PYBIN) scripts/download_datasets.py --all --data-dir "$(DATA_DIR)"

download: setup-real

$(VENV)/.dev.stamp: requirements-dev.txt $(VENV)/.deps.stamp
	$(PYBIN) -m pip install -r requirements-dev.txt
	@touch $@

dev: $(VENV)/.dev.stamp

check: dev
	$(VENV)/bin/ruff check notebooks scripts tests --select F821,E9
	$(PYBIN) scripts/check_notebooks.py
	$(PYBIN) -m pytest -q

test: dev
	$(PYBIN) -m pytest -q

smoke: dev
	$(PYBIN) scripts/check_notebooks.py --execute --quick

clean-data:
	rm -rf $(DATA_DIR)/imdb $(DATA_DIR)/snips $(DATA_DIR)/banking77 $(DATA_DIR)/wikitext

clean-caches:
	find notebooks scripts tests -type d -name __pycache__ -exec rm -rf {} +

clean: clean-data clean-caches
	rm -rf $(VENV)
