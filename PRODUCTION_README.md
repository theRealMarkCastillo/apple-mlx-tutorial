# From tutorial models to a local application

These notebooks teach MLX training and inference. They are not a production
service or a verified benchmark suite. Notebook 04 combines three models trained
on synthetic samples; notebook 09 evaluates word-overlap retrieval on 40
labeled queries (and optionally generates answers with an LLM). The sample data
is tiny and partly templated, so the numbers the notebooks print are for
learning evaluation technique, not performance claims. Timings depend on your hardware.

## Environment and checks

Use Python 3.12+ on Apple Silicon with macOS 14 or later. Install `uv`, then run
`make setup`, `make check`, and `make smoke`. The default Python is 3.13 (see
`.python-version`); `pyproject.toml` declares dependencies and `uv.lock` records
their full resolution. Keep the lockfile, installed dependency inventory, hardware,
random seeds, and data revision alongside any results you publish.

The repository targets Metal on macOS. Upstream MLX also provides Linux CPU and
CUDA backends; those environments are outside this tutorial's validation scope.
See the [MLX installation guide](https://ml-explore.github.io/mlx/build/html/install.html).
A Linux Docker container on a Mac does not provide the native Metal environment
used by these notebooks.

## Train on real data

```bash
make setup-real
# Or choose a dataset:
uv run --locked python scripts/download_datasets.py --imdb --max-samples 5000
```

The downloader writes:

| Directory | Data | Labels / splits |
|---|---|---|
| `data/snips` | Original SNIPS custom-intent benchmark JSON | Seven intent strings; original training/validation files, saved as train/test |
| `data/banking77` | Publisher's Banking77 CSV files | 77 intent strings; original train/test |
| `data/imdb` | `stanfordnlp/imdb` | 0 = negative, 1 = positive; train/test |
| `data/wikitext` | `Salesforce/wikitext`, `wikitext-2-v1` | Separate train, validation, and test text files |

SNIPS is read from its publisher's JSON and Banking77 through the generic CSV
builder. This avoids the dataset scripts removed in recent Hugging Face
`datasets` versions. Failed real downloads raise an error; synthetic data is
never silently substituted. `make setup-samples` generates offline examples and
respects `DATA_DIR` without importing `datasets`.

To adapt a classifier notebook:

1. Split training data into training and validation partitions before fitting
   the vocabulary. Group duplicate texts together. Keep the official test split
   for final evaluation, not model selection.
2. Fit the word vocabulary and sequence length using training data only. Reuse
   that mapping for validation, test, and inference; unseen words map to `<UNK>`.
3. Build the label mapping from the dataset and change `output_size`. The three
   toy sentiment labels differ from IMDB's two classes; SNIPS and Banking77 also
   differ from the toy greeting/question/command task.
4. Pass both validation arrays to `train_model`. Use `batch_size` to control
   memory. Inspect per-class precision/recall and class balance, not just accuracy.
5. Save weights, vocabulary, label order, preprocessing settings, architecture,
   and dataset identity together. Weights alone cannot reconstruct a model.

For character generation, split documents before creating overlapping windows.
The bundled corpus no longer repeats the same passage five times, but it is
still a tiny educational corpus. Its validation loss is not a language-model
quality benchmark. Prefer WikiText's separate splits for meaningful experiments.

## Current training patterns

`notebooks/mlx_nlp_utils.py` provides a compiled update which captures model,
optimizer, and random state. Including random state ensures dropout samples
advance across compiled calls. Each batch evaluates loss and updated state to
avoid an ever-growing lazy graph. Gradient clipping is enabled by default;
`max_grad_norm=None` disables it. `compile_step=False` is useful when debugging.

Both classifier readout and Transformer pooling ignore padding. The Transformer
also masks padded keys; an empty input has a safe attention key and zero pooled
representation. Notebook 07 uses fused causal attention with residual dropout.
Notebook 06 exposes the full attention matrix for visualization, which has
quadratic memory cost and is not the efficient path for long contexts.

Notebook 08 uses the active kernel's Python for `mlx_lm.lora`, checks subprocess
failure, and only loads adapters when the weights file exists. The example uses
a quantized base model, gradient checkpointing, prompt-loss masking, and a
256-token sequence limit. These reduce memory demands but do not guarantee a
particular device will fit every model/batch. The generated training targets
are intent labels, and validation prompts are excluded from training.

For pretrained generation, use `tokenizer.apply_chat_template` and
`mlx_lm.sample_utils.make_sampler`. Notebook 09 returns messages offline or
applies the tokenizer template when provided. It does not hard-code another
model's special tokens. For repeated conversations, MLX-LM supports prompt/KV
caches; use its documented cache APIs and keep cache identity tied to the model
and exact prompt prefix. See [MLX-LM](https://github.com/ml-explore/mlx-lm) and
[LoRA guidance](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

## Save and evaluate

In a notebook, after training and defining validation arrays:

```python
from mlx_nlp_utils import save_model, load_model, evaluate_model

save_model(model, "saved_models/classifier.safetensors")
# Construct a fresh instance with the same architecture before loading.
load_model(restored_model, "saved_models/classifier.safetensors")
accuracy, expected, predicted = evaluate_model(restored_model, X_val, y_val)
```

Notebook 04 also writes vocabulary/configuration JSON next to its weights.
JSON object keys are strings: convert a saved `idx_to_char` mapping back to
integer keys before using it for generation.

## Measure before serving

Run inference warmups, set `model.eval()`, and include `mx.eval(output)` before
stopping the timer; otherwise you may measure graph construction. Report batch
size, token lengths, precision, and hardware. Measure first-token latency and
subsequent-token throughput separately for LLMs. The reduced smoke tests check
execution, not training convergence or performance.

A local Python service can load a model once and accept requests from a Mac or
mobile client. Serialize model/cache mutations and bound input length and
concurrency. Evaluate domain-specific failures and confidence calibration before
using predictions to take actions. The toy chatbot only emits text; its command
responses do not execute commands.

For native applications, investigate MLX Swift or reimplement/export a supported
architecture through PyTorch and Core ML tooling. MLX `.npz`/`.safetensors` files
are not themselves Core ML packages. Deployment conversion is outside this repo.

## Validation scope

`make check` validates every notebook, checks undefined names/syntax, and runs
regressions for masking, padding, compiled random state, partial batches,
checkpoint round-trips, generation, dataset handling, and the evaluation
helpers (tokenizer, grouped splits, near-duplicate detection, bootstrap
intervals). `make smoke` executes the offline notebooks (00–07b, 09, and 12) with 1-epoch training in a temporary copy, to catch crashes quickly.
`make validate` runs the same notebooks at full training budgets and also
enforces each notebook's `sanity_check` claims (for example, that a model beats
the majority baseline), so a lesson whose narrative stops matching its results
fails the build. Notebooks 08, 10, and 11 need optional dependency groups and explicit model
download budgets; they execute with `--include-manual`. Notebook 12's dense
retrieval and reranking require the `embeddings` group and `--advanced`.
See the root README for local validation commands and pretrained lesson setup.

## Reproducible model and evaluation artifacts

Pin a model's full Hub commit in `config/models.json`, including its tokenizer
files. Record the corpus revision/hash, split policy, adapter configuration and
weight hashes, prompt format, decoding settings, hardware, and package lock.
The local notebook runner and `experiment_utils.write_record` capture these in ignored JSON
artifacts under `results/`; archive them alongside reported measurements.

Choose refusal thresholds on calibration data and compare hyperparameters on
development data. Freeze them before running the final test. Expand the tiny
synthetic splits before making deployment or model-quality claims. For RAG,
measure candidate recall separately from reranking, and assess citation support
separately from whether a citation names a real retrieved document.

Use notebook 11 to measure prefill, first-token latency, generation throughput,
and MLX peak memory under weight/KV quantization. Cache reuse requires the same
model/adapter and exact token prefix; it must not accidentally include a previous
user's answer. Compare task quality under each quantization setting.
