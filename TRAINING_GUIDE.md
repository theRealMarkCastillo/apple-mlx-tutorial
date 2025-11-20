# Training Scripts with Real Data

All training is now done through interactive Jupyter notebooks with rich visualizations and explanations.

## 🚀 Quick Start

### Step 1: Download Data

```bash
# Python script
python scripts/download_datasets.py --samples

# For all datasets
python scripts/download_datasets.py --all
```

### Step 2: Launch Notebooks

```bash
cd notebooks
jupyter notebook
```

### Step 3: Train Models

Open the appropriate notebook:
- `01_Intent_Classification.ipynb` - For intent classification
- `02_Sentiment_Analysis.ipynb` - For sentiment analysis
- `03_Text_Generation.ipynb` - For text generation
- `04_Complete_Pipeline.ipynb` - For complete chatbot
- `06_Build_NanoGPT.ipynb` - For training GPT from scratch
- `07_Fine_Tuning_with_LoRA.ipynb` - For fine-tuning LLMs

Each notebook includes:
- Dataset loading and exploration
- Data preprocessing and cleaning
- Model training with progress tracking
- Evaluation with metrics and visualizations
- Interactive predictions

## 📊 Available Datasets

### Download Commands

```bash
# Download specific datasets
python scripts/download_datasets.py --imdb          # 50K movie reviews
python scripts/download_datasets.py --snips         # 16K+ intent queries
python scripts/download_datasets.py --banking77     # 13K banking intents
python scripts/download_datasets.py --wikitext      # 100M tokens

# Download by category
python scripts/download_datasets.py --sentiment     # All sentiment datasets
python scripts/download_datasets.py --intent        # All intent datasets
python scripts/download_datasets.py --generation    # All generation datasets

# Download everything
python scripts/download_datasets.py --all
```

### Dataset Sizes

| Dataset | Size | Download Time | Use Case |
|---------|------|---------------|----------|
| Samples | ~30 examples | <1 sec | Quick testing |
| SNIPS | 16K queries | ~10 sec | Intent classification |
| IMDB | 50K reviews | ~30 sec | Sentiment analysis |
| Banking77 | 13K queries | ~15 sec | Fine-grained intents |
| WikiText-2 | 2M tokens | ~20 sec | Text generation |
| WikiText-103 | 100M tokens | ~2 min | Large-scale generation |

## 🎯 Usage Examples

### Example 1: Train on Sample Data

```bash
# 1. Download samples
python scripts/download_datasets.py --samples

# 2. Start notebooks
cd notebooks
jupyter notebook

# 3. Open 01_Intent_Classification.ipynb
# 4. Run cells to load data and train
```

**Expected output in notebook:**
```
Loaded 9 examples
Intents: ['command', 'greeting', 'question']
Training...
Epoch      Loss         Accuracy
-----------------------------------
1          1.0987       0.33
5          0.5432       0.67
10         0.2345       0.89
[Training curves visualization displayed]
[Confusion matrix heatmap displayed]
```

### Example 2: Train on SNIPS Data

```bash
# 1. Download SNIPS dataset
python scripts/download_datasets.py --snips

# 2. Start notebooks and open 01_Intent_Classification.ipynb
cd notebooks
jupyter notebook

# 3. Follow notebook instructions to:
#    - Load SNIPS data
#    - Train with custom parameters
#    - Visualize results
#    - Test predictions
```

### Example 3: IMDB Sentiment Analysis

```bash
# 1. Download IMDB data
python scripts/download_datasets.py --imdb

# 2. Start notebooks and open 02_Sentiment_Analysis.ipynb
cd notebooks
jupyter notebook

# 3. Train on IMDB reviews with visualizations
```

## 📁 Data Directory Structure

After downloading, your `data/` directory will look like:

```
data/
├── intent_samples/
│   └── data.json              # 9 intent examples
├── sentiment_samples/
│   └── data.json              # 8 sentiment examples
├── text_gen_samples/
│   └── corpus.txt             # Small text corpus
├── snips/
│   ├── train.json             # SNIPS training data
│   └── test.json              # SNIPS test data
├── imdb/
│   ├── train.json             # 25K IMDB train reviews
│   └── test.json              # 25K IMDB test reviews
├── banking77/
│   ├── train.json             # Banking intents train
│   └── test.json              # Banking intents test
└── wikitext/
    ├── train.txt              # WikiText training
    ├── validation.txt         # WikiText validation
    └── test.txt               # WikiText test
```

## 🔧 Training Configuration

All training parameters are configured in the notebook cells:

### Model Parameters

```python
# In notebook cells, modify:
config = {
    'vocab_size': 5000,       # Max vocabulary size
    'embedding_dim': 128,     # Embedding dimension
    'hidden_dim': 256,        # LSTM hidden size
    'dropout': 0.3,           # Dropout rate
    'epochs': 50,             # Training epochs
    'batch_size': 32,         # Batch size
    'learning_rate': 0.001    # Learning rate
}
```

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 1000-5000 | Maximum vocabulary size |
| `embedding_dim` | 32-128 | Embedding dimension |
| `hidden_dim` | 64-256 | LSTM hidden size |
| `dropout` | 0.3 | Dropout rate (0-0.5) |
| `epochs` | 30-100 | Number of training epochs |
| `batch_size` | 16-64 | Mini-batch size |
| `learning_rate` | 0.001-0.01 | Optimizer learning rate |

## 📊 Expected Results

### Intent Classification (Sample Data)

```
Dataset: 9 examples, 3 intents
Training: 30 epochs, ~5 seconds
Final Accuracy: 90-100%

Test examples:
  'hello how are you' → greeting
  'what time is it' → question
  'please turn on the lights' → command

[Rich visualizations in notebook]
```

### Intent Classification (SNIPS Data)

```
Dataset: 16K+ examples, 7 intents
Training: 50 epochs, ~2 minutes
Final Accuracy: 85-95%

Intents: PlayMusic, GetWeather, BookRestaurant,
         SearchCreativeWork, AddToPlaylist, RateBook

[Training curves, confusion matrix, and more]
```

### Sentiment Analysis (IMDB Data)

```
Dataset: 50K examples, 2 sentiments
Training: 20 epochs, ~30 minutes
Final Accuracy: 88-92%

[Word clouds, ROC curves, confidence plots]
```

## 🎓 Learning Path

### Beginner: Sample Data

1. Download samples: `python scripts/download_datasets.py --samples`
2. Start notebooks: `cd notebooks && jupyter notebook`
3. Open `00_Overview.ipynb` for quick intro
4. Work through `01_Intent_Classification.ipynb`
5. Understand: load → train → predict → visualize

### Intermediate: Real Datasets

1. Download SNIPS: `python scripts/download_datasets.py --snips`
2. Open `01_Intent_Classification.ipynb`
3. Load SNIPS data in notebook
4. Experiment with parameters (epochs, batch size, learning rate)
5. Analyze training curves and accuracy

### Advanced: Production Scale

1. Download all datasets: `python scripts/download_datasets.py --all`
2. Work through all notebooks (01-04)
3. Train on full IMDB dataset (50K samples)
4. Study the complete pipeline in `04_Complete_Pipeline.ipynb`
5. Understand data cleaning, model versioning, evaluation

## 🐛 Troubleshooting

### "datasets library not installed"

```bash
pip install datasets
```

### "Data file not found"

Run the download script first:
```bash
python scripts/download_datasets.py --samples
```

### "Out of memory"

In notebook cells, reduce batch size:
```python
config['batch_size'] = 8
```

### "Training too slow"

Reduce dataset size in notebook or use sample data

### "Poor accuracy"

Try in notebook cells:
- More epochs: `config['epochs'] = 100`
- Larger model: `config['hidden_dim'] = 512`
- More data: Download full datasets
- Check visualizations for insights

## 📚 Documentation

- **[notebooks/README.md](notebooks/README.md)** - Complete notebook guide with learning paths
- **[README.md](README.md)** - Project overview and setup
- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference guide
- **[PRODUCTION_README.md](PRODUCTION_README.md)** - Deployment information

## 🔄 Workflow

```
1. Download Data
   ↓
   python scripts/download_datasets.py --samples

2. Launch Notebooks
   ↓
   cd notebooks && jupyter notebook

3. Train Models
   ↓
   Open and run notebook cells

4. Evaluate & Iterate
   ↓
   Adjust parameters, visualize results

5. Deploy
   ↓
   Export models for production use
```

## ⚡ Quick Reference

```bash
# Get started (30 seconds)
python scripts/download_datasets.py --samples
cd notebooks && jupyter notebook

# Train intent classifier
# Open 01_Intent_Classification.ipynb in Jupyter

# Train sentiment analyzer  
# Open 02_Sentiment_Analysis.ipynb in Jupyter

# Complete pipeline
# Open 04_Complete_Pipeline.ipynb in Jupyter
```

---

**Ready to train with real data!** 🚀
