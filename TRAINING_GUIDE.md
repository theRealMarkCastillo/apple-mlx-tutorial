# Training Scripts with Real Data

This directory contains scripts to download datasets and train models on real data.

## 🚀 Quick Start

### Step 1: Download Sample Data

```bash
# Python script
python scripts/download_datasets.py --samples

# Or shell script
./scripts/download_datasets.sh --samples
```

This creates small sample datasets for quick testing (~30 examples total).

### Step 2: Run Examples

```bash
python examples_with_real_data.py
```

Choose from the menu to run different examples with real data.

### Step 3: Train Models

```bash
# Train intent classifier
python train_intent_classifier.py

# Train on specific data
python train_intent_classifier.py --data data/snips --epochs 50
```

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

# 2. Train intent classifier
python train_intent_classifier.py

# 3. See results
# Model saved to: trained_models/intent_classifier.npz
```

**Expected output:**
```
Loaded 9 examples
Intents: ['command', 'greeting', 'question']
Training...
Epoch      Loss         Accuracy
-----------------------------------
1          1.0987       0.33
5          0.5432       0.67
10         0.2345       0.89
...
✓ Training complete!
```

### Example 2: Train on SNIPS Data

```bash
# 1. Download SNIPS dataset
python scripts/download_datasets.py --snips

# 2. Train with custom parameters
python train_intent_classifier.py \
    --data data/snips \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001

# 3. Test the model
# See predictions on test examples
```

**Dataset info:**
- ~30 examples total (6 intents)
- Intents: PlayMusic, GetWeather, BookRestaurant, etc.
- Training time: ~2 minutes on M1 Mac

### Example 3: Run All Examples

```bash
# 1. Download all sample data
python scripts/download_datasets.py --samples

# 2. Run examples script
python examples_with_real_data.py

# 3. Choose option 7 (Run all examples)
# This will:
#   - Train intent classifier
#   - Train sentiment analyzer
#   - Train text generator
#   - Show predictions
```

### Example 4: IMDB Sentiment Analysis

```bash
# 1. Download IMDB data (may take 30 seconds)
python scripts/download_datasets.py --imdb

# 2. Run IMDB example
python examples_with_real_data.py
# Choose option 4

# This trains on 1,000 IMDB reviews
# Full 25K training takes ~1-2 hours
```

### Example 5: Production Training

```bash
# Use the full production pipeline
python production_example.py

# This:
# - Downloads IMDB automatically
# - Cleans and preprocesses
# - Trains production model
# - Evaluates comprehensively
# - Saves versioned model
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

## 🔧 Training Script Options

### Intent Classifier

```bash
python train_intent_classifier.py \
    --data data/snips \              # Data path
    --epochs 50 \                    # Number of epochs
    --batch-size 32 \                # Batch size
    --learning-rate 0.001 \          # Learning rate
    --vocab-size 5000 \              # Max vocabulary size
    --max-len 20                     # Max sequence length
```

### Common Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | `data/intent_samples/data.json` | Path to training data |
| `--epochs` | 30 | Number of training epochs |
| `--batch-size` | 16 | Mini-batch size |
| `--learning-rate` | 0.01 | Optimizer learning rate |
| `--vocab-size` | 1000 | Maximum vocabulary size |
| `--max-len` | 20 | Maximum sequence length |

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
```

### Intent Classification (SNIPS Data)

```
Dataset: 30 examples, 6 intents
Training: 50 epochs, ~2 minutes
Final Accuracy: 70-85%

Intents: PlayMusic, GetWeather, BookRestaurant,
         SearchCreativeWork, AddToPlaylist, RateBook
```

### Sentiment Analysis (Sample Data)

```
Dataset: 8 examples, 3 sentiments
Training: 50 epochs, ~5 seconds
Final Accuracy: 85-100%

Sentiments: positive, negative, neutral
```

### Sentiment Analysis (IMDB Subset)

```
Dataset: 1,000 examples, 2 sentiments
Training: 20 epochs, ~5 minutes
Final Accuracy: 75-85%

Full 25K dataset: ~90% accuracy, ~1-2 hours training
```

## 🎓 Learning Path

### Beginner: Sample Data

1. Download samples: `python scripts/download_datasets.py --samples`
2. Run examples: `python examples_with_real_data.py`
3. Try option 1, 2, 3, and 6
4. Understand the flow: load → train → predict

### Intermediate: Real Datasets

1. Download SNIPS: `python scripts/download_datasets.py --snips`
2. Train intent: `python train_intent_classifier.py --data data/snips`
3. Experiment with parameters (epochs, batch size, learning rate)
4. Understand training curves and accuracy

### Advanced: Production Scale

1. Download IMDB: `python scripts/download_datasets.py --imdb`
2. Run production: `python production_example.py`
3. Study the full pipeline:
   - Data cleaning
   - Quality filtering
   - Vocabulary building
   - Train/val/test splits
   - Model versioning
   - Comprehensive evaluation

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

Reduce batch size:
```bash
python train_intent_classifier.py --batch-size 8
```

### "Training too slow"

Reduce dataset size or use sample data:
```bash
# Use first 1000 examples
python scripts/download_datasets.py --imdb --max-samples 1000
```

### "Poor accuracy"

Try:
- More epochs: `--epochs 100`
- Larger model: Edit script to increase `hidden_dim`
- More data: Download full datasets
- Better preprocessing: See `docs/DATASETS_AND_PREPROCESSING.md`

## 📚 Documentation

- **[DATASETS_AND_PREPROCESSING.md](../docs/DATASETS_AND_PREPROCESSING.md)** - Complete data guide
- **[PRODUCTION_BEST_PRACTICES.md](../docs/PRODUCTION_BEST_PRACTICES.md)** - Deployment guide
- **[PRODUCTION_README.md](../PRODUCTION_README.md)** - Production setup

## 🔄 Workflow

```
1. Download Data
   ↓
   python scripts/download_datasets.py --samples

2. Explore Examples
   ↓
   python examples_with_real_data.py

3. Train Models
   ↓
   python train_intent_classifier.py

4. Evaluate & Iterate
   ↓
   Adjust parameters, try different data

5. Production Deploy
   ↓
   python production_example.py
   Deploy with REST API
```

## ⚡ Quick Reference

```bash
# Most common commands

# Get started (30 seconds)
python scripts/download_datasets.py --samples
python examples_with_real_data.py

# Train intent classifier (2 minutes)
python scripts/download_datasets.py --snips
python train_intent_classifier.py --data data/snips

# Production sentiment (30 minutes)
python scripts/download_datasets.py --imdb --max-samples 5000
python production_example.py

# Full pipeline (2 hours)
python scripts/download_datasets.py --all
python production_example.py  # Uses full IMDB dataset
```

---

**Ready to train with real data!** 🚀
