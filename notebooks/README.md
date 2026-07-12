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
Detect emotions in text (positive/negative)
- **Time**: 60-75 minutes
- **Level**: Intermediate
- **Visualizations**:
  - Sentiment distribution
  - Word clouds (positive/negative)
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
  - Ensemble predictions
  - Deployment workflow

### 5. Transformer Classifier (`05_Transformer_Classifier.ipynb`)
Bridge from LSTMs to Transformers — build a Transformer-based classifier
- **Time**: 45-60 minutes
- **Level**: Intermediate
- **Content**:
  - Positional encoding from scratch
  - Multi-Head Self-Attention in MLX
  - LSTM vs Transformer comparison
  - Attention weight visualization

### 6. Attention Mechanism (`06_Attention_Mechanism.ipynb`)
Understand the math behind Transformers
- **Time**: 45-60 minutes
- **Level**: Advanced
- **Visualizations**:
  - Attention heatmaps
  - Cross-attention patterns

### 7. Build NanoGPT (`07_Build_NanoGPT.ipynb`)
Build a GPT model from scratch
- **Time**: 90-120 minutes
- **Level**: Expert
- **Content**:
  - Multi-Head Attention
  - Transformer Blocks
  - Training on Shakespeare

### 8. Fine-Tuning with LoRA (`08_Fine_Tuning_with_LoRA.ipynb`)
Fine-tune Llama-3.2 on your own data
- **Time**: 60-90 minutes
- **Level**: Expert
- **Content**:
  - LoRA (Low-Rank Adaptation)
  - 4-bit Quantization
  - Custom Dataset Preparation

### 9. RAG from Scratch (`09_RAG_from_Scratch.ipynb`)
Build a Retrieval Augmented Generation system
- **Time**: 60-90 minutes
- **Level**: Expert
- **Content**:
  - Vector Search & Embeddings
  - System Design (Scaling to 100M docs)
  - RAG vs Fine-Tuning

### 10. Embeddings Deep Dive (`10_Embeddings_Deep_Dive.ipynb`)
Advanced optimization for RAG systems
- **Time**: 60-90 minutes
- **Level**: Expert
- **Content**:
  - Visualizing Embedding Space (t-SNE)
  - Benchmarking Models (Speed vs Quality)
  - Domain-Specific Fine-Tuning
  - Trace Retrieval Optimization

## 🚀 Quick Start

```bash
# 1. Activate virtual environment
source ../.venv/bin/activate

# 2. Install Jupyter and visualization libraries
pip install jupyter matplotlib seaborn plotly scikit-learn wordcloud mlx-lm

# 3. Download sample datasets
python ../scripts/download_datasets.py --samples

# 4. Launch Jupyter
jupyter notebook

# 5. Open 00_Overview.ipynb to start
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
- **Sample Data**: Quick testing (< 1 second)
  - 9 intent examples
  - 8 sentiment reviews
  - Small text corpus
- **Real Datasets**: Production training (10-30 seconds)
  - SNIPS: 16K+ voice queries
  - IMDB: 50K movie reviews
  - WikiText: 100M+ tokens
  - Shakespeare: For training NanoGPT

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

## 📈 Expected Results

### Sample Data Performance
| Model | Training Time | Accuracy | Notes |
|-------|--------------|----------|-------|
| Intent | < 1 min | ~80% | Limited by small dataset |
| Sentiment | < 1 min | ~90% | May overfit |
| Generation | < 2 min | Basic | Simple patterns |

### Real Data Performance
| Model | Dataset | Training Time | Accuracy | Notes |
|-------|---------|--------------|----------|-------|
| Intent | SNIPS | 2-5 min | 90-95% | Production-ready |
| Sentiment | IMDB | 5-10 min | 88-92% | Robust |
| Generation | WikiText | 10-20 min | Coherent | Multi-sentence |

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
- **Source code**: `notebooks/mlx_nlp_utils.py` (460+ lines of shared utilities)
- `../TRAINING_GUIDE.md` - Complete training instructions
- `../README.md` - Project overview

### Code Examples
- `mlx_nlp_utils.py` - Consolidated model implementations
- `04_Complete_Pipeline.ipynb` - Full production pipeline
- `01_Intent_Classification.ipynb` - Training examples

### Datasets
- `../data/intent_samples/` - Sample intent data
- `../data/sentiment_samples/` - Sample reviews
- `../data/text_gen_samples/` - Sample corpus

## 🐛 Troubleshooting

### Jupyter won't start
```bash
pip install --upgrade jupyter
jupyter notebook --no-browser
```

### Plots don't show
```bash
pip install matplotlib seaborn
# Add to first cell: %matplotlib inline
```

### Can't find modules
```python
import sys
sys.path.append('..')
```

### MLX errors
```bash
# Reinstall MLX
pip install --upgrade mlx
```

### Out of memory
```python
# Reduce batch size or use subset of data
X_subset = X[:100]
y_subset = y[:100]
```

## 🎓 Interactive Features

Each notebook includes:
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
- Deploy production models
- Build complete chatbots

## 🌟 What Makes These Special

1. **Apple Silicon Optimized** - Uses MLX for M1/M2/M3
2. **Real Datasets** - Not just toy examples
3. **Rich Visualizations** - 20+ different plots
4. **Production Focus** - Deploy-ready code
5. **Interactive** - Modify and experiment
6. **Comprehensive** - 500+ lines per notebook

## 🚀 Ready to Start?

Open `00_Overview.ipynb` and let's begin your NLP journey!

Happy learning! 🎓
