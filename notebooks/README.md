# MLX NLP notebooks: curriculum guide

Python 3.12+, Apple Silicon, macOS 14+. Install `uv` (see the root README).
From the repository root run `make setup` then `make run`, and open
`00_Overview.ipynb`. uv manages Python 3.13 and the locked environment.

## How these notebooks teach

Each notebook follows the same shape: **objectives → concept → build →
measure → break it → exercises → summary**. Three habits recur on purpose:

* **🤔 Predict, then run.** Write down a guess before running the cell. Short
  answers to exercises are in collapsible *What to look for* blocks, but try first.
* **Measure against a baseline.** A model's score means little until you know
  what a trivial model scores (majority class, TF-IDF, n-gram counts, the base LLM).
* **Check the evaluation itself.** Near-duplicate leakage, tiny validation sets,
  and a single random seed are the usual reasons a number is too good.

Cells that call `sanity_check(...)` assert a claim the text makes (for example
"the LSTM beats the majority baseline"). `make validate` runs every offline
notebook at full budget and fails if one of those claims stops being true.

## The notebooks

| # | Notebook | You learn | Time |
|---|---|---|---|
| 00 | `00_Overview` | Three tiny models end to end (demo, no validation) | 15 min |
| 00b | `00b_MLX_Fundamentals` | Lazy evaluation, unified memory, `grad`, `compile`, benchmarking correctly | 30 min |
| 01 | `01_Intent_Classification` | Tokenizing, LSTM classifier, baselines, **leak detection**, grouped splits, confidence intervals | 60 min |
| 02 | `02_Sentiment_Analysis` | Dropout, overfitting curves, per-class metrics, calibration, **challenge sets** | 75 min |
| 03 | `03_Text_Generation` | Language modeling, perplexity, n-gram baselines, leaky vs clean splits, temperature, memorization | 90 min |
| 04 | `04_Complete_Pipeline` | Router + analyst + generator, thresholds, latency measurement, saving configs with weights | 60 min |
| 05 | `05_Transformer_Classifier` | Positional encoding, masked pooling, a fair LSTM-vs-Transformer comparison | 45 min |
| 06 | `06_Attention_Mechanism` | Implement attention, why √d, *learned* attention, causal masks | 45 min |
| 06b | `06b_Prompt_Engineering` | Softmax dilution, similarity, no inherent position bias, needle-in-a-haystack on a real LLM | 45 min |
| 07 | `07_Build_NanoGPT` | A GPT from scratch, early stopping, attention-head maps, positional ablation | 90 min |
| 07b | `07b_Modern_Decoder` | Train-only BPE, RoPE, RMSNorm, SwiGLU, GQA, cached/full equivalence | 60 min |
| 08 | `08_Fine_Tuning_with_LoRA` | LoRA/QLoRA, strict output scoring, development selection, frozen final evaluation | 60 min + training |
| 09 | `09_RAG_from_Scratch` | Embedding, retrieval, recall@k and MRR, IDF, refusal thresholds | 60 min |
| 10 | `10_Embeddings_Deep_Dive` | Learned embeddings vs word counts, model comparison, embeddings as features | 60 min |
| 11 | `11_Local_LLM_Inference` | Weight/KV quantization, prefill vs decode, prefix reuse, latency and memory | 45 min |
| 12 | `12_Hybrid_Retrieval` | BM25, dense retrieval, RRF, reranking, chunk relevance, citation checks | 60 min |

Notebooks **08**, **10**, and **11** download pretrained models and are not part of
`make smoke`/`make validate`; run them yourself, or use
`uv run --locked --group llm python scripts/check_notebooks.py --execute --include-manual --notebook 08 --download-budget-gb 2`.
See the root README for optional groups and explicit download budgets.
Optional cells (real datasets, the LoRA rank sweep, the needle-in-a-haystack
test) are switched off or skip themselves when their data is missing.

## Learning paths

* **Quick tour (2 h):** 00 → 01 → 03 → 06 → 07.
* **Solid foundations (6 h):** 00b → 01 → 02 → 03 → 05 → 06 → 07.
* **LLM applications (7 h):** 00b → 06 → 06b → 08 → 09 → 10 → 11 → 12.
* **Everything, in order (15 h):** 00 → 00b → 01 → … → 12 (including 07b).

## Data

`make setup` creates small synthetic datasets in `data/`. They are
intentionally tiny and partly templated; the notebooks teach what that means
for evaluation (for example, 60 base phrases decorated into 160 intent examples).
`make setup-real` downloads SNIPS, IMDB, Banking77 and WikiText-2 for the
optional "real data" cells; see `PRODUCTION_README.md` for formats.

| File | Used by |
|---|---|
| `intent_samples/data.json` | 00, 01, 04, 05, 10 |
| `sentiment_samples/data.json` | 00, 02, 04, 05 |
| `text_gen_samples/corpus.txt` | 00, 03, 04, 07, 07b |
| `rag_samples/knowledge_base.json`, `eval_queries.json` | 06b, 09, 10, 12 |
| `train.jsonl`, `valid.jsonl` | 08 |

## Shared code

`notebooks/mlx_nlp_utils.py` holds what the notebooks reuse: model classes,
the compiled trainer (`train_model`, `make_train_step`), the tokenizer, and
the evaluation toolkit (`group_train_val_split`, `find_near_duplicates`,
`majority_baseline_accuracy`, `bootstrap_ci`, `sanity_check`). Notebooks
05 and 07 define their Transformer and GPT classes inline so you can read them.

## Troubleshooting

* **Imports fail:** run notebooks through `make run` so they use uv's `.venv`; `mlx_nlp_utils`
  is imported from the `notebooks/` directory.
* **A section says data is missing:** run `make setup-samples` (sample data) or
  `make setup-real` (real datasets).
* **Out of memory in 08:** lower `batch_size`/`max_seq_length` in `train_adapter`
  or switch `MODEL_NAME` to `mlx-community/Qwen2.5-0.5B-Instruct-4bit`.
* **Different numbers than the text:** most notebooks fix seeds, but timings and
  anything involving downloaded models vary by machine. Where a number matters,
  the text says what to compare rather than quoting a value.
* **Stale weights:** notebook 04 saves to `notebooks/saved_models/` and 08 to
  `notebooks/adapters/`. Delete them after changing model sizes.

## Evaluation and provenance

07b compares architectures at equal token budgets, with one train-only tokenizer.
08 separates label accuracy from strict format compliance and freezes adapter
choices on development data before final testing. 10 calibrates refusal on
document-disjoint questions and tests on harder unseen negatives. 11 separates
prefix-build cost from cache-hit latency. 12 preserves source relevance when
changing chunk size; citation-ID validity alone is not factual entailment.

All pretrained loads use `config/models.json`. CLI runs save records under
`results/<run-id>/`; inspect them alongside executed notebooks.
