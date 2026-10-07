# MLX NLP Tutorial

Hands-on NLP with **Apple's MLX framework** on Apple Silicon, from a first
LSTM classifier to a GPT, LoRA fine-tuning, and retrieval. The emphasis is not
only *how to build* each model but **how to tell whether it works**: baselines,
leakage checks, confidence intervals, challenge sets, and ablations.

> **Versions (October 2026):** MLX 0.32.3, MLX-LM 0.32.0, Transformers 5.19.0,
> Sentence Transformers 6.1.0. Python 3.12+, Apple Silicon, macOS 14+.
> `uv` defaults to Python 3.13 via `.python-version`. Direct dependencies live
> in `pyproject.toml`; `uv.lock` pins the full dependency graph.

## What you will learn

| Phase | Notebooks | Topics |
|---|---|---|
| Framework | 00, 00b | MLX basics: lazy evaluation, unified memory, `grad`, `compile`, benchmarking |
| Classic models, evaluated honestly | 01–04 | LSTM classifiers, baselines, train/validation leakage, perplexity, a multi-model chatbot |
| Transformers | 05–07 | Fair LSTM-vs-Transformer comparison, attention from scratch, a GPT, attention maps, ablations |
| LLM applications | 06b, 08–10 | Prompt/attention behavior, LoRA fine-tuning, RAG, embeddings |

See **[notebooks/README.md](notebooks/README.md)** for the full curriculum, time
estimates, learning paths, and what each notebook measures.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first
(on macOS with Homebrew: `brew install uv`). No separate Python installation
or environment activation is needed.

```bash
git clone https://github.com/theRealMarkCastillo/apple-mlx-tutorial.git
cd apple-mlx-tutorial
make setup      # uv syncs .venv from uv.lock and writes sample data
make run        # starts Jupyter; open notebooks/00_Overview.ipynb
```

Or use `uv` directly:

```bash
uv sync --locked
uv run --locked python scripts/download_datasets.py --samples
uv run --locked jupyter notebook --notebook-dir=notebooks
```

`uv` installs the selected Python if needed and manages `.venv` for you. If
you already used this repo's old setup, `make setup` syncs the existing
environment to the lockfile. The requirements files and dependency stamps
have been replaced by `pyproject.toml` and `uv.lock`.

Run each notebook top to bottom in a fresh kernel. The default lessons use
small bundled datasets and finish in seconds to minutes. Notebooks 08 and 10
download pretrained models (about 1.7 GB and 90 MB) on first use.

### Make targets

| Command | What it does |
|---|---|
| `make setup` | Sync locked dependencies (including dev tools) + sample data |
| `make run` | Launch Jupyter (no reinstall when dependencies are unchanged) |
| `make setup-samples` | Generate sample data without installing the ML dependencies |
| `make setup-real` | Download SNIPS, IMDB, Banking77 and WikiText-2 (optional "real data" cells) |
| `make lint` | Check Python and notebook code with Ruff |
| `make check` | Lint, validate notebook structure, run unit tests |
| `make smoke` | Execute the offline notebooks with 1-epoch training (fast crash check) |
| `make validate` | Execute the offline notebooks at **full budget**; fails if a notebook's `sanity_check` claim stops holding |
| `make render` | Like `validate`, saving executed notebooks (with plots) to `rendered/` |
| `make clean` | Remove `.venv`, downloaded datasets, and caches (keeps samples) |

Execution uses temporary copies, so source notebooks stay free of outputs.
Notebooks 08 and 10 run only with
`uv run --locked python scripts/check_notebooks.py --execute --include-manual --notebook 08`.

## Using the notebooks well

* **Predict before you run.** Cells marked 🤔 ask for a guess first.
* **Try the exercises** before opening the *What to look for* answers.
* **Change one thing at a time**, then rerun the evaluation cells. Many
  lessons are about *why* a number moved.
* Sample data is tiny and partly templated *on purpose*, so you can see
  evaluation pitfalls that real data would hide.

## Project structure

```
apple-mlx-tutorial/
├── notebooks/
│   ├── mlx_nlp_utils.py         # shared models, trainer, tokenizer, evaluation helpers
│   ├── README.md                # curriculum guide
│   └── 00_Overview … 10_Embeddings_Deep_Dive (13 notebooks incl. 00b, 06b)
├── data/                        # sample data (committed); real datasets (opt-in downloads)
│   ├── intent_samples/  sentiment_samples/  text_gen_samples/
│   ├── rag_samples/             # knowledge base + 40 labeled retrieval queries
│   └── train.jsonl, valid.jsonl # chat data for LoRA (notebook 08)
├── scripts/
│   ├── generate_synthetic_data.py   # single source of the sample data
│   ├── download_datasets.py         # samples and real-dataset downloads
│   └── check_notebooks.py           # validate / execute notebooks
├── tests/                       # unit tests for the shared code and data scripts
├── Makefile  pyproject.toml  uv.lock  .python-version
├── .github/workflows/check.yml  # lint, notebook structure, and data tests
└── PRODUCTION_README.md         # real-data adaptation and deployment notes
```

## Shared code

`notebooks/mlx_nlp_utils.py` is meant to be read:

* **Models:** `IntentLSTM`, `SentimentLSTM` (configurable dropout), `TextLSTM`.
* **Training:** `train_model` (shuffled mini-batches, gradient clipping, a
  compiled update that captures model, optimizer and RNG state, metrics with
  dropout off) and `make_train_step`.
* **Text:** `tokenize` (one tokenizer for training and inference),
  `create_vocabulary`, `pad_sequences`, `texts_to_sequences`.
* **Evaluation:** `group_train_val_split`, `find_near_duplicates`,
  `majority_baseline_accuracy`, `bootstrap_ci`, `clean_holdout_slice`,
  `sanity_check`.

The Transformer classifier (05) and GPT (07) are defined inside their
notebooks so the lessons can build them step by step.

## Development

```bash
make check      # ruff (undefined names, syntax), notebook structure, pytest
make validate   # full-budget notebook run (several minutes)
```

Notebook source files are `.ipynb`; they are committed without outputs.
Optional experiments (real datasets, the LoRA rank sweep, the long-context
needle test) are off by default or skip themselves when data is missing.

Ruff and pytest settings are centralized in `pyproject.toml`. GitHub Actions
checks the lockfile, lint, notebook structure, and data tests on an Apple
Silicon macOS runner. Run `make check` and `make smoke` locally for the MLX
model tests and Metal notebook execution; those are outside the hosted CI job.

Make commands use `--locked` so a stale lockfile fails instead of silently
changing dependencies. To deliberately change a dependency, use
`uv add 'package==VERSION'` (or `uv add --dev 'tool==VERSION'`). To refresh
transitive dependencies within the existing direct pins, use `uv lock --upgrade`.
Commit both `pyproject.toml` and `uv.lock`, then run `make check` and `make smoke`.
Use `make validate` when changing model or training dependencies.

For a tool that requires pip-style input, export it instead of maintaining a
second dependency list:

```bash
uv export --locked --no-dev --format requirements-txt --output-file /tmp/mlx-requirements.txt
```

`make setup-samples` needs no third-party packages; its first run may still
download Python. Use `DATA_DIR=/path/to/data make setup-samples` to choose its
output directory. For custom Python/environment locations, use uv's
`UV_PYTHON` and `UV_PROJECT_ENVIRONMENT` settings with the project commands.

## Further reading

* MLX documentation: <https://ml-explore.github.io/mlx/>
* *Attention Is All You Need* (Vaswani et al., 2017)
* *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList* (Ribeiro et al., 2020), behind notebook 02's challenge set
* *Lost in the Middle* (Liu et al., 2023), context for notebook 06b
* [MLX-LM LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

## License

For educational and demonstration purposes.
