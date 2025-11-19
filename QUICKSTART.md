# MLX NLP Chatbot - Quick Reference

## 🚀 Getting Started (30 seconds)

```bash
# Activate environment
source .venv/bin/activate

# Download sample datasets (< 1 second)
python scripts/download_datasets.py --samples

# Run interactive examples with real data
python examples_with_real_data.py

# Or quick demo with built-in data
python quick_demo.py

# Full menu
python main.py
```

## 📚 What's Inside

### 1. **Intent Classification** (`intent_classifier.py`)
Classifies user input into: greeting, question, or command
- **Sample data**: 9 examples for quick testing
- **Real datasets**: SNIPS (16K+), Banking77 (13K)
- LSTM-based architecture
- Interactive mode included

### 2. **Sentiment Analysis** (`sentiment_analysis.py`)
Detects sentiment: positive or negative
- **Sample data**: 8 reviews for quick testing
- **Real dataset**: IMDB movie reviews (50K)
- Returns probability distributions
- Great for customer feedback

### 3. **Text Generation** (`text_generator.py`)
Generates text and suggests completions
- **Sample data**: Small corpus for testing
- **Real dataset**: WikiText-2 (36K articles)
- Temperature-controlled sampling
- Autocomplete functionality

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

# Shell script
./scripts/download_datasets.sh
```

### Train with Real Data
```bash
# Interactive examples (6 scenarios)
python examples_with_real_data.py

# Standalone training
python train_intent_classifier.py --data data/snips --epochs 40

# Production pipeline
python production_example.py
```

### Run a Quick Demo
```bash
python quick_demo.py
```

### Run Specific Model
```bash
python main.py intent      # Intent classification
python main.py sentiment   # Sentiment analysis
python main.py text        # Text generation
```

### Use in Your Code
```python
# See examples.py for full examples
from intent_classifier import predict_intent

# Train your model...
intent, confidence = predict_intent(model, "Hello", vocab, names, max_len)
```

## 🛠️ Customization

### Add More Training Data
Edit the training data in each file:
- `intent_classifier.py` - Add to `sentences` and `intent_labels`
- `sentiment_analysis.py` - Add to `sentences` and `sentiment_labels`
- `text_generator.py` - Add to `texts` list

### Adjust Model Parameters
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
Main Scripts:
- main.py              → Run everything
- quick_demo.py        → Fast test
- examples.py          → Code examples

Models:
- intent_classifier.py
- sentiment_analysis.py
- text_generator.py

Utilities:
- run.sh              → Quick start
- requirements.txt    → Dependencies
- README.md           → Full docs
```

## 🎓 Learning Path

1. **Start here:** `python quick_demo.py`
2. **Try each model:** `python main.py`
3. **Learn the code:** Read `examples.py`
4. **Customize:** Edit training data
5. **Build:** Use models in your own project

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

1. **Expand training data** - Add domain-specific examples
2. **Fine-tune models** - Adjust hyperparameters
3. **Combine models** - Build complete chatbot
4. **Deploy to iOS** - Convert to Core ML
5. **Add features** - NER, Q&A, translation

## 📖 Resources

- MLX Documentation: https://ml-explore.github.io/mlx/
- Apple Silicon ML: https://developer.apple.com/machine-learning/
- This project: README.md for full documentation

---

**Quick Start:** `python quick_demo.py`  
**Full Experience:** `python main.py`  
**Learn More:** `python examples.py`

Happy coding! 🎉
