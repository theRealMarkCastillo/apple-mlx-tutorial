# MLX NLP Tutorial — single-command setup.
#
# Typical first run:
#   make setup      # create a venv, install deps, generate sample data
#   make run        # launch Jupyter inside the venv
#
# Common subsets:
#   make setup-samples      # just the toy data
#   make setup-real         # SNIPS + IMDB + Banking77 + WikiText
#   make download              # alias for setup-real
#
# Maintenance:
#   make clean-data    # drop downloaded datasets (samples preserved)
#   make clean-caches  # drop __pycache__ directories
#   make venv          # create the .venv without installing anything

PY     ?= python3
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PYBIN  := $(VENV)/bin/python
JUPYTER := $(VENV)/bin/jupyter

DATA_DIR := data

# ---------------------------------------------------------------------------
# Top-level targets
# ---------------------------------------------------------------------------

.PHONY: help setup run install samples setup-samples setup-real download \
        clean-data clean-caches venv clean

help:
	@echo "Targets:"
	@echo "  make setup        Bootstraps .venv, installs deps, generates sample data"
	@echo "  make run          Launches Jupyter in the notebooks/ folder"
	@echo "  make setup-real   Downloads SNIPS, IMDB, Banking77, and WikiText"
	@echo "  make clean-data   Removes downloaded real datasets (keeps samples)"
	@echo "  make clean-caches Removes __pycache__ directories"
	@echo "  make clean        Removes .venv and all generated artefacts"

venv:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt

samples: install
	$(PYBIN) scripts/download_datasets.py --samples

setup: install samples
	@echo
	@echo "Setup complete. Run 'make run' to launch Jupyter."
	@echo "Start with notebooks/00_Overview.ipynb."

run: install
	cd notebooks && $(JUPYTER) notebook

setup-real: install
	$(PYBIN) scripts/download_datasets.py --all
	@echo "Real datasets downloaded under $(DATA_DIR)/"

download: setup-real

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean-data:
	rm -rf $(DATA_DIR)/imdb $(DATA_DIR)/snips $(DATA_DIR)/banking77 $(DATA_DIR)/wikitext
	@echo "Real datasets removed (sample data and LoRA chat files preserved)."

clean-caches:
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "All __pycache__ directories removed."

clean: clean-data clean-caches
	rm -rf $(VENV)
	@echo "Removed .venv and generated artefacts."
