> Updated October 2026. Use Python 3.12+, Apple Silicon macOS 14+, and the
> pinned environment from `make setup`. Run `make check` and `make smoke` from
> the repository root. Notebook 08 downloads and fine-tunes a pretrained model;
> notebook 10 downloads PyTorch embedding models. Both require separate manual runs.

# MLX NLP Jupyter Notebooks

Interactive tutorials for learning NLP with MLX on Apple Silicon.

## 📚 Notebooks

### 0. Overview (`00_Overview.ipynb`)
Quick introduction and demo of all three models
- **Time**: 15-20 minutes
- **Level**: Beginner
- **Content**: Quick demos, model comparison

### 1. Intent Classification (`01_Intent_Classification.ipynb`)
Learn to classify user commands into intents
- **Time**: 45-60 minutes
- **Level**: Beginner
- **Visualizations**:
  - Intent distribution (bar/pie charts)
  - Sequence length histogram
  - Model architecture diagram
  - Training curves (loss & accuracy)
  - Prediction confidence bars
  - Confusion matrix heatmap

### 2. Sentiment Analysis (`02_Sentiment_Analysis.ipynb`)
Classify text as negative, neutral, or positive
- **Time**: 60-75 minutes
- **Level**: Intermediate
- **Visualizations**:
  - Sentiment distribution
  - Word clouds for all three sentiment classes
  - Training progress
  - ROC curve & AUC
  - Prediction probabilities
  - Misclassification analysis

### 3. Text Generation (`03_Text_Generation.ipynb`)
Generate text and autocomplete suggestions
- **Time**: 75-90 minutes
- **Level**: Advanced
- **Visualizations**:
  - Vocabulary growth
  - Perplexity curves
  - Generation samples
  - Temperature comparison
  - N-gram distribution

### 4. Complete Pipeline (`04_Complete_Pipeline.ipynb`)
End-to-end chatbot combining all techniques
- **Time**: 90-120 minutes
- **Level**: Advanced
- **Content**:
  - Data preprocessing pipeline
  - Multi-model training
  - Intent routing, sentiment adjustment, and generation fallback
  - Deployment workflow

### 5. Transformer Classifier (`05_Transformer_Classifier.ipynb`)
Bridge from LSTMs to Transformers — build a Transformer-based classifier
- **Time**: 45-60 minutes
- **Level**: Intermediate
- **Content**:
  - Positional encoding from scratch
  - Multi-Head Self-Attention in MLX
  - LSTM vs Transformer comparison
  - Padding masks, fixed positions, and training-curve comparison

### 6. Attention Mechanism (`06_Attention_Mechanism.ipynb`)
Understand the math behind Transformers
- **Time**: 45-60 minutes
- **Level**: Advanced
- **Visualizations**:
  - Attention heatmaps
  - Cross-attention patterns

### 6b. Prompt Engineering (`06b_Prompt_Engineering.ipynb`)
Translates attention theory into prompt-design rules
- **Time**: 30-45 minutes
- **Level**: Advanced
- **Content**:
  - "Lost in the Middle" — positional bias in attention
  - Context-length dilution
  - Semantic-similarity effects (why RAG works)
  - Prompt-engineering cheat sheet

### 7. Build NanoGPT (`07_Build_NanoGPT.ipynb`)
Build a GPT model from scratch and train it on a small character-level corpus
- **Time**: 90-120 minutes
- **Level**: Expert
- **Content**:
  - Fused causal multi-head attention
  - Transformer blocks and compiled updates with random state
  - Training on a 4-line Hamlet excerpt (the demo corpus; swap in your own data when you scale up)

### 8. Fine-Tuning with LoRA (`08_Fine_Tuning_with_LoRA.ipynb`)
Fine-tune Llama-3.2 on your own data
- **Time**: 60-90 minutes
- **Level**: Expert
- **Content**:
  - LoRA (Low-Rank Adaptation)
  - 4-bit Quantization
  - Label-only chat targets and disjoint validation prompts
  - Gradient checkpointing and prompt-loss masking

### 9. RAG from Scratch (`09_RAG_from_Scratch.ipynb`)
Walk through a Retrieval Augmented Generation system end-to-end (toy BoW embeddings; swap in real embeddings per Notebook 10)
- **Time**: 60-90 minutes
- **Level**: Expert
- **Content**:
  - Vector Search using cosine similarity
  - System Design (Scaling to 100M docs)
  - RAG vs Fine-Tuning trade-offs

### 10. Embeddings Deep Dive (`10_Embeddings_Deep_Dive.ipynb`)
Advanced optimization for RAG systems
- **Time**: 60-90 minutes
- **Level**: Expert
- **Content**:
  - Visualizing Embedding Space (t-SNE)
  - Benchmarking Models (Speed vs Quality)
  - Domain-specific retrieval evaluation (no embedding fine-tuning)
  - Query/document encoding and top-1 retrieval accuracy

## 🚀 Quick Start

The repository now ships with a `Makefile` at the project root that handles venv setup and data generation. Run `make setup` once, then `make run` from the project root. This uses Jupyter from `.venv` without requiring shell activation.

If you prefer to drive the steps by hand:

```bash
# From the project root:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # all notebook deps in one go
python scripts/download_datasets.py --samples # create the 160/150 sample datasets
cd notebooks && jupyter notebook              # launch Jupyter

# Open 00_Overview.ipynb to start
```

## 📊 What You'll Learn

### Core Concepts
- Word embeddings and tokenization
- LSTM networks for sequence modeling
- Transformer architectures (Attention, GPT)
- Fine-tuning LLMs with LoRA
- RAG (Retrieval Augmented Generation)
- Training loops and optimization
- Model evaluation and metrics
- Production deployment

### Practical Skills
- Building NLP models with MLX
- Training on real datasets (IMDB, SNIPS, WikiText)
- Creating visualizations with matplotlib/seaborn
- Debugging and improving model accuracy
- Deploying models for production use

### Datasets
- **Sample Data**: Small educational datasets
  - 160 intent examples
  - 150 sentiment reviews
  - Small text corpus
  - 800 chat-format messages for LoRA demo
- **Real Datasets**: Optional downloads; training time depends on model and hardware
  - SNIPS: 13,784 training / 700 validation queries, 7 intents
  - IMDB: 50K movie reviews
  - WikiText-2: ~2M training tokens
  - NanoGPT: A short hardcoded Shakespeare excerpt

## 🎯 Learning Paths

### Path 1: Quick Learner (2 hours)
1. Overview notebook (20 min)
2. Intent Classification (45 min)
3. Run examples with sample data
4. Experiment with parameters

### Path 2: Deep Dive (5 hours)
1. All 4 notebooks in order
2. Complete all exercises
3. Train on real datasets
4. Compare sample vs production

### Path 3: Project Builder (8+ hours)
1. Complete notebooks 01-05 (LSTM + Transformer bridge)
2. Build custom chatbot
3. Deploy to production
4. Add new features

### Path 4: LLM Specialist (10+ hours)
1. Transformer Classifier (Notebook 05) — bridge
2. Attention Mechanism (Notebook 06)
3. Build NanoGPT (Notebook 07)
4. Fine-Tuning with LoRA (Notebook 08)
5. RAG from Scratch (Notebook 09)
6. Embeddings Deep Dive (Notebook 10)

## Validation and interpreting results

The October 2026 refresh was checked on Apple Silicon with Python 3.13:
18 regression tests passed, all 12 notebooks passed structure/syntax checks,
and the 10 offline notebooks executed in fresh kernels with reduced training.
SNIPS and Banking77 downloads were also exercised. These are execution checks,
not accuracy, convergence, or performance benchmarks.

```bash
# From the repository root:
make check
make smoke
# Execute a single offline lesson with reduced training:
.venv/bin/python scripts/check_notebooks.py --execute --quick --notebook 05
```

Notebooks 08 and 10 are validated statically but excluded from the execution
runner because they download pretrained models. Their full LoRA training and
embedding benchmarks have not been validated in this refresh. Run them manually
before relying on their results. Published performance claims need a recorded
hardware configuration, dataset/split, training budget, and measured results.
The toy validation sets are too small to establish deployment quality.

## 🔧 Tips for Success

1. **Run cells in order** - Each cell depends on previous ones
2. **Read explanations** - Markdown cells contain key concepts
3. **Experiment** - Change parameters and observe results
4. **Visualize** - Graphs help understand model behavior
5. **Start small** - Use sample data first, then scale up
6. **Ask questions** - Add markdown cells with your notes
7. **Save often** - Jupyter can crash, save your work

## 🎨 Visualization Gallery

### Training Curves
- Loss over time
- Accuracy progression
- Learning rate effects

### Model Analysis
- Confusion matrices
- ROC curves
- Attention heatmaps

### Data Exploration
- Distribution plots
- Word clouds
- Embedding visualizations

### Performance Metrics
- Confidence bars
- Error analysis
- Comparison charts

## 📚 Additional Resources

### Documentation
- **Source code**: `notebooks/mlx_nlp_utils.py` (shared utilities)
- `../PRODUCTION_README.md` - Real-data adaptation, measurement, and local-serving scope
- `../README.md` - Project overview

### Code Examples
- `mlx_nlp_utils.py` - Consolidated model implementations
- `04_Complete_Pipeline.ipynb` - Complete toy pipeline
- `01_Intent_Classification.ipynb` - Training examples

### Datasets
- `../data/intent_samples/` - Sample intent data
- `../data/sentiment_samples/` - Sample reviews
- `../data/text_gen_samples/` - Sample corpus
- `../data/train.jsonl` / `../data/valid.jsonl` - Chat data for LoRA fine-tuning (mlx_lm's default filenames)

## 🐛 Troubleshooting

### Jupyter or imports fail

Use the project environment instead of upgrading individual packages outside
its pins. From the repository root:

```bash
make install
.venv/bin/python -m pip check
make run
```

If you manually changed packages after setup, `make install` may find its stamp
up to date. Restore the pinned versions explicitly, then restart the kernel:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Run notebooks from the `notebooks/` folder, as configured by `make run`, so the
shared helper and `../data` paths resolve. In a notebook, `import sys;
print(sys.executable)` should point to this repository's `.venv`.

### Plots don't show

Restart the project kernel and rerun the imports. For static Matplotlib plots,
add `%matplotlib inline` in an interactive notebook. Plotly and widget displays
need a working Jupyter frontend; the smoke runner uses noninteractive renderers.

### Out of memory or unexpected training behavior

Reduce the `batch_size` argument to `train_model`; for notebook 08 reduce the
LoRA `--batch-size` and `--max-seq-length` settings. The LoRA lesson already uses
gradient checkpointing. Pass `compile_step=False` to the shared trainer when
debugging and rerun from a fresh model. Keep both validation arrays together.

### Existing checkpoints after this update

Retrain the small teaching models when comparing results with the refreshed
notebooks: vocabulary splits and padding behavior changed. Older notebook 05
checkpoints may include a positional-encoding parameter that is now fixed and
excluded from trainable weights. Earlier LoRA data trained conversational
responses; regenerate samples and retrain adapters for the label-only task.

## 🎓 Interactive Features

Across the series, the notebooks provide:
- ✅ Step-by-step explanations
- ✅ Runnable code cells
- ✅ Visual outputs
- ✅ Interactive testing
- ✅ Practice exercises
- ✅ Comparison charts
- ✅ Real-time metrics

## 🎓 Learning Objectives

By completing these notebooks, you will:
- Understand NLP fundamentals
- Build LSTM models with MLX
- Train on real-world datasets
- Visualize model performance
- Debug and improve accuracy
- Identify the extra evaluation and serving work needed for deployment
- Build complete chatbots

## 🌟 What Makes These Special

1. **Apple Silicon Optimized** - Uses MLX for M1/M2/M3
2. **Optional real datasets** - Downloads and adaptation guidance alongside toy examples
3. **Rich Visualizations** - 20+ different plots
4. **Deployment discussion** - Educational patterns requiring application-specific validation
5. **Interactive** - Modify and experiment
6. **Comprehensive** - Covering theory, code, visualisations, and exercises in one place

## 🚀 Ready to Start?

Open `00_Overview.ipynb` and let's begin your NLP journey!

Happy learning! 🎓
