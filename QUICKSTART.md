# MLX NLP Chatbot - Quick Reference

## 🚀 Getting Started (30 seconds)

```bash
# Activate environment
source .venv/bin/activate

# Start Jupyter notebooks
cd notebooks
jupyter notebook

# Open 00_Overview.ipynb to get started!
```

## 📚 What's Inside

All functionality is now in interactive Jupyter notebooks:

### 1. **Intent Classification** 
Learn to classify user input into: greeting, question, or command
- **Notebook**: `01_Intent_Classification.ipynb`
- **Sample data**: 9 examples for quick testing
- **Real datasets**: SNIPS (16K+), Banking77 (13K)
- LSTM-based architecture with full explanations

### 2. **Sentiment Analysis**
Detect sentiment: positive, negative, or neutral
- **Notebook**: `02_Sentiment_Analysis.ipynb`
- **Sample data**: 8 reviews for quick testing
- **Real dataset**: IMDB movie reviews (50K)
- Returns probability distributions with visualizations

### 3. **Text Generation**
Generate text and suggest completions
- **Notebook**: `03_Text_Generation.ipynb`
- **Sample data**: Small corpus for testing
- **Real dataset**: WikiText-2 (36K articles)
- Temperature-controlled sampling with examples

## 🎯 Common Tasks

### Download Datasets
```bash
# Sample data (< 1 second)
python scripts/download_datasets.py --samples

# All real datasets (10-30 seconds each)
python scripts/download_datasets.py --all

# Specific datasets
python scripts/download_datasets.py --imdb     # 50K reviews
python scripts/download_datasets.py --snips    # 16K+ intents
python scripts/download_datasets.py --wikitext # 36K articles
```

### Train with Real Data
All training is done in Jupyter notebooks:

```bash
# Start notebooks
cd notebooks
jupyter notebook

# Open the notebook for your task:
# - 01_Intent_Classification.ipynb
# - 02_Sentiment_Analysis.ipynb  
# - 03_Text_Generation.ipynb
```

Each notebook includes:
- Data loading and preprocessing
- Model training with visualizations
- Evaluation and testing
- Interactive predictions

## 🛠️ Customization

### Add More Training Data
Add data directly in the notebooks by modifying the data loading cells.

### Adjust Model Parameters
In the notebook cells, modify parameters like:
```python
# Larger model for better accuracy
model = IntentClassifier(
    vocab_size=len(vocab),
    embedding_dim=64,      # Increase from 32
    hidden_dim=128,        # Increase from 64
    num_classes=len(intents)
)
```

### Training Options
Adjust training in the notebook cells:
```python
# Train longer or with different learning rate
model = train_model(
    model, X, y, 
    epochs=100,           # More epochs
    learning_rate=0.001   # Lower learning rate
)
```

## 📊 Performance Tips

### For Better Accuracy
1. Add more training data (50-100+ examples per class)
2. Increase model size (embedding_dim, hidden_dim)
3. Train for more epochs (100-200)
4. Lower learning rate (0.001-0.005)

### For Faster Training
1. Reduce model size
2. Fewer epochs (20-50)
3. Higher learning rate (0.01-0.1)
4. Smaller vocabulary

### For Better Text Generation
1. More training text
2. Longer sequences (seq_length)
3. Larger hidden size
4. Temperature tuning (0.5-1.0)

## 🔍 Troubleshooting

### "MLX not found"
```bash
pip install mlx numpy scikit-learn
```

### Models not learning
- Add more training data
- Increase epochs
- Check data quality
- Try different learning rate

### Low accuracy
- Need more diverse training examples
- Increase model size
- Train longer
- Check for data imbalance

### Text generation repetitive
- Adjust temperature (try 0.8-1.2)
- More diverse training corpus
- Larger model
- Longer sequences

## 📝 File Structure

```
Main Resources:
- notebooks/              → All interactive notebooks
- notebooks/mlx_nlp_utils.py → Model implementations
- scripts/                → Dataset downloaders
- data/                   → Downloaded datasets

Documentation:
- README.md              → Full documentation
- QUICKSTART.md          → This file
- TRAINING_GUIDE.md      → Training workflows
- PRODUCTION_README.md   → Deployment guide
```

## 🎓 Learning Path

1. **Start here:** `cd notebooks && jupyter notebook`
2. **Open:** `00_Overview.ipynb` for quick intro
3. **Learn:** Work through notebooks 01-04 in order
4. **Customize:** Modify notebooks for your needs
5. **Build:** Use `notebooks/mlx_nlp_utils.py` as reference

## 💡 Use Cases

### Chatbot Features
- Command routing (intent classification)
- Mood detection (sentiment analysis)
- Smart replies (text generation)
- Autocomplete (text generation)

### Production Ideas
- Customer support bot
- FAQ automation
- Social media monitoring
- Email classification
- Content moderation

## 🚀 Next Steps

1. **Learn with notebooks** - Interactive tutorials with visualizations
2. **Download datasets** - Try with real data (IMDB, SNIPS, WikiText)
3. **Fine-tune models** - Adjust hyperparameters in notebooks
4. **Build your project** - Use notebook code as reference
5. **Deploy** - Convert to Core ML or REST API

## 📖 Resources

- **Notebooks README**: `notebooks/README.md` - Complete learning guide
- **Training Guide**: `TRAINING_GUIDE.md` - Training workflows
- **Main README**: `README.md` - Full documentation
- **MLX Docs**: https://ml-explore.github.io/mlx/

---

**Quick Start:** `cd notebooks && jupyter notebook` (open `00_Overview.ipynb`)

Happy coding! 🎉
