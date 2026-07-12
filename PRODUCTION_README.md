# Local Deployment & Real-Data Ranges

Notes on training the LSTM classifiers from Notebooks 01–04 against real datasets, and the realistic options for serving or shipping them.

## 🚀 Quick Start

### 1. Install Additional Dependencies

```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Install datasets library
pip install datasets
```

### 2. Run Production Example

```bash
# Start Jupyter notebooks
cd notebooks
jupyter notebook

# Open 04_Complete_Pipeline.ipynb
```

This notebook demonstrates the full production pipeline:
- Download 5,000 IMDB movie reviews
- Clean and preprocess the data
- Build vocabulary
- Train LSTM sentiment classifier
- Evaluate with comprehensive metrics
- Save versioned model
- Show demo predictions

**Expected runtime**: ~10-15 minutes on M1/M2/M3 Mac

### 3. Advanced Production: RAG & Fine-Tuning

For modern LLM workflows, check out:
- **08_Fine_Tuning_with_LoRA.ipynb**: Fine-tune Llama-3.2 on your own data
- **09_RAG_from_Scratch.ipynb**: Build a production-ready RAG system with vector search

## 📊 What Datasets Are Available?

### Sentiment Analysis
- **IMDB Reviews**: 50K movie reviews (positive/negative)
- **Amazon Reviews**: Millions of product reviews (1-5 stars)
- **Yelp Reviews**: 560K business reviews (1-5 stars)
- **Twitter Sentiment140**: 1.6M tweets

### Intent Classification
- **ATIS**: 5,871 flight booking queries (26 intents)
- **SNIPS**: 16K+ queries (6 intents: weather, music, etc.)
- **Banking77**: 13K banking queries (77 fine-grained intents)

### Text Generation
- **WikiText**: 100M+ tokens from Wikipedia
- **OpenWebText**: 38GB web text
- **DailyDialog**: 13K conversations
- **PersonaChat**: Conversations with personalities

## 📚 Documentation

All documentation is in the repository root and notebooks directories:

1. **Datasets & Preprocessing**
   - See `notebooks/` for data loading in each notebook
   - `scripts/download_datasets.py` for downloading real datasets
   - `scripts/generate_synthetic_data.py` for generating synthetic data

2. **Production Deployment**
   - Model saving/loading: `notebooks/mlx_nlp_utils.py` (`save_model`, `load_model`)
   - REST API pattern: See example in `04_Complete_Pipeline.ipynb`
   - Performance benchmarks: See training times below

## 🔧 Customizing the Production Example

### Change Model Architecture

Notebook 04 builds each model from the shared classes in `mlx_nlp_utils.py`. To change hyperparameters, edit the cell that constructs `IntentLSTM`, `SentimentLSTM`, or `TextLSTM`, e.g.:

```python
sentiment_model = SentimentLSTM(
    vocab_size=len(word_to_idx),
    embedding_dim=128,   # was 64
    hidden_size=256,     # was 128
    output_size=3,
)
```

### Try a Different Dataset

The notebook loads JSON files from `data/intent_samples/`, `data/sentiment_samples/`, and `data/text_gen_samples/`. To use a different source:

1. Write a small Python script that drops the JSON your notebook expects into those directories.
2. Or modify the notebook's `load_sample_*_data()` calls to point at a different path.

A more advanced path is to plug in a real pretrained embedding model (Notebook 10 covers this), which typically pushes accuracy into the ~94% range on IMDB — much better than the small custom LSTM we build here.

## 📈 Expected Results

> **Numbers below are honest ranges, not guarantees.** IMDB accuracy with an LSTM
> + a small custom embedding lands roughly in the **80–92% range**, depending on
> preprocessing, vocab size, hidden size, and how long you train. Treat the
> table as a sanity check, not a target to hit exactly on the first run.

What you should see when you train `SentimentLSTM` (embedding=64, hidden=128,
dropout=0.3) on IMDB:

| Samples | Epochs | Typical Val Accuracy | M1 Training Time |
|---------|--------|----------------------|------------------|
| 1,000   | 5      | ~70–80%              | ~2 min           |
| 5,000   | 5      | ~80–86%              | ~10 min          |
| 10,000  | 10     | ~85–90%              | ~30 min          |
| 25,000  | 10     | ~88–92%              | ~90 min          |

If your numbers are below the lower end of the band, the usual suspects are:
- truncation too aggressive (keep max_len ≥ 200 words for IMDB),
- vocabulary capped too low (let it grow to 30K–100K for IMDB),
- learning rate too high (try 5e-4 with Adam, or add a learning-rate schedule).

A transformer (notebook 05) lands in the same range with the same IMDB data,
but converges in fewer epochs and generalises a bit further. The numbers do
not move dramatically because both architectures are limited by the small
custom word embeddings — a pretrained encoder would close the gap to ~94%.

## 🏭 Production Deployment

### 1. Train Full Model

Run `04_Complete_Pipeline.ipynb` with `max_samples=25000` (or the real-data flag in the notebook's data-loading cell) to train on the full dataset. The notebook saves weights to `saved_models/` next to its own directory.

### 2. Load and Serve Model

Use `mlx_nlp_utils.py` to load the saved weights back:

```python
from mlx_nlp_utils import SentimentLSTM, predict_sentiment, load_model

# Rebuild the architecture using whatever hyperparams you trained with.
model = SentimentLSTM(
    vocab_size=len(word_to_idx),
    embedding_dim=64,       # match the training-time hyperparams
    hidden_size=128,
    output_size=3,
)
load_model(model, "saved_models/sentiment_model.npz")

# Predict
text = "This movie is amazing!"
label, conf = predict_sentiment(model, text, word_to_idx, sentiment_names, max_len)
print((label, conf))   # ('positive', 0.95)
```

### 3. Deploy as REST API

```python
# Example using FastAPI
from fastapi import FastAPI
from notebooks.mlx_nlp_utils import SentimentLSTM, load_model

app = FastAPI()

@app.post("/predict")
async def predict(text: str):
    return model.predict(text)
```

### 4. Local Serving Options

> **MLX only runs on Apple Silicon** — there is no CUDA or x86 path. So
> Docker-on-Linux or cloud GPU deployments are not directly supported. For
> on-device use on a Mac, two practical patterns work today:

**Option A: Serve locally from a Mac (recommended for learning and small apps).**
Run a local FastAPI process that loads your MLX model, and call it from a
SwiftUI / iPadOS / other client app over HTTP or WebSocket:

```python
# server.py
from fastapi import FastAPI
from mlx_nlp_utils import SentimentLSTM, predict_sentiment, load_model

app = FastAPI()
model = SentimentLSTM(vocab_size, embedding_dim=64, hidden_size=128, output_size=3)
load_model(model, "saved_models/sentiment_model.npz")

@app.post("/predict")
async def predict(text: str) -> dict:
    label, conf = predict_sentiment(model, text, word_to_idx, names, max_len)
    return {"label": label, "confidence": conf}
```

`uvicorn server:app` and you've got a tiny MLX inference endpoint on M-series.
Latency on M1 is single-digit ms for an LSTM classifier.

**Option B: Convert to Core ML — but expect to re-write the model.**
MLX itself has no Core ML exporter, and `coremltools` does not consume MLX
modules. The honest paths are:

1. Re-implement the model in **PyTorch** (the architectures in notebooks 01–05
   drop in directly), export ONNX, convert with `coremltools`.
2. For `mlx_lm` checkpoints, convert the original **Hugging Face** model to
   Core ML — not the MLX-tuned adapter, but the underlying base architecture.
3. Stay in MLX and serve over Option A — fine for anything that runs on a
   Mac or iPad.

**Option C: Don't bother for an LSTM.** The LSTM classifiers in this tutorial
are < 100K parameters; serving them in-process from a Swift app via a tiny
PythonKit bridge or a Core ML reimplementation is faster than any network call.
The interesting MLX production stories are the **LLM notebooks (08 / 09 / 10)**,
where the unified-memory advantage over a CPU-on-Linux deployment is real.

## 🧪 Running Experiments

### Hyperparameter Search

You can use Python loops in the notebook to run grid searches:

```python
# In a notebook cell:
learning_rates = [0.0001, 0.001, 0.01]
results = []

for lr in learning_rates:
    print(f"\n=== Training with LR={lr} ===")
    config['learning_rate'] = lr
    # ... run training function ...
    # ... append metrics to results ...
```

### Compare Model Weights

Compare saved model files by accuracy on a validation set:

```python
from notebooks.mlx_nlp_utils import load_model, evaluate_model

# Load and evaluate different checkpoints
for path in sorted(Path('checkpoints').glob('*.safetensors')):
    model = SentimentLSTM(vocab_size, embedding_dim, hidden_size, output_size)
    load_model(model, path)
    acc, _, _ = evaluate_model(model, X_val, y_val)
    print(f"{path.stem}: {acc:.2%}")
```

## 📊 Performance Benchmarks

These are the same numbers as the *Expected Results* table above, plus
inference latency. They were measured on an M1 Pro with MLX 0.32 and are
within ~2× of what you should see on M1 / M2 / M3 / M4 machines. They are
**rough** — the point of this table is to set expectations, not to predict
your exact numbers.

### IMDB Sentiment Analysis

| Samples | Epochs | Batch Size | Accuracy | Training Time (M1) |
|---------|--------|------------|----------|-------------------|
| 1,000   | 5      | 32         | ~70–80%  | ~2 min            |
| 5,000   | 5      | 32         | ~80–86%  | ~10 min           |
| 10,000  | 10     | 64         | ~85–90%  | ~30 min           |
| 25,000  | 10     | 64         | ~88–92%  | ~90 min           |

### Inference Speed (M1 Mac, SentimentLSTM)

| Batch Size | Latency (ms) | Throughput (samples/sec) |
|------------|--------------|-------------------------|
| 1          | ~5–10 ms     | 100–200                 |
| 32         | ~50–80 ms    | 400–700                 |
| 64         | ~90–150 ms   | 400–700                 |

Latency for transformer-based classifiers (notebook 05) is comparable on M1
since the model is tiny; throughput for the LLM notebooks is dominated by
tokenisation and not in this table — see `mlx_lm` benchmarks for those.

## 🎯 Next Steps

1. **Experiment with datasets**: Try different domains (products, tweets, news)
2. **Tune hyperparameters**: Learning rate, model size, dropout
3. **Data augmentation**: Use techniques from the preprocessing guide
4. **Deploy**: Set up REST API with monitoring
5. **Scale**: Use larger datasets and longer training

## 📖 Learn More

- **[notebooks/README.md](notebooks/README.md)** — Complete notebook guide with learning paths
- **[README.md](README.md)** — Project overview, setup, and learning paths
- **MLX Docs**: https://ml-explore.github.io/mlx/

## 🆘 Troubleshooting

**Out of memory during training?**
- Reduce batch_size in notebook config: `config['batch_size'] = 16`
- Reduce model size: `config['hidden_dim'] = 128`

**Training too slow?**
- Increase batch_size: `config['batch_size'] = 64`
- Use fewer samples for initial experiments

**Poor accuracy?**
- Train longer: `config['epochs'] = 10`
- Use more training data: `max_samples=25000`
- Increase model capacity: `config['hidden_dim'] = 512`

**Dataset download failing?**
- Check internet connection
- Try: `export HF_DATASETS_OFFLINE=0`
- Clear cache: `rm -rf ~/.cache/huggingface/datasets`

---

**Ready to build production NLP systems!** 🚀
