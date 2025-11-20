# MLX NLP Chatbot Demo

A comprehensive demonstration of Natural Language Processing (NLP) capabilities for chatbots using **Apple's MLX framework**, optimized for Apple Silicon (M1/M2/M3) Neural Processing Units (NPU).

## 🎯 Project Overview

This project showcases three practical NLP use cases for chatbots, trained on **real-world datasets** with production-ready pipelines:

1. **Intent Classification** - Classify user commands into categories (trained on SNIPS & Banking77 datasets)
2. **Sentiment Analysis** - Detect emotions in messages (trained on IMDB movie reviews)
3. **Text Generation** - Generate responses and provide autocomplete suggestions (trained on WikiText corpus)

All models are built using MLX, leveraging the power of Apple Silicon for efficient on-device text processing. The project includes **50,000+ real training examples** with automated dataset downloading and preprocessing.

## 🚀 Features

### 1. Intent Classification (`intent_classifier.py`)
- Classifies user input into predefined intents
- Uses LSTM-based architecture
- **Real datasets**: SNIPS (16K+ queries), Banking77 (13K banking intents)
- **Sample data**: 9 examples for quick testing
- Interactive prediction mode with confidence scores

**Example Usage:**
```
Input: "Hello there"
→ Intent: greeting (confidence: 95%)

Input: "What's the weather like"
→ Intent: question (confidence: 92%)

Input: "Turn off the lights"
→ Intent: command (confidence: 88%)
```

### 2. Sentiment Analysis (`sentiment_analysis.py`)
- Detects sentiment in user messages
- Binary classification: positive, negative
- **Real dataset**: IMDB movie reviews (50K labeled reviews)
- **Sample data**: 8 reviews for quick testing
- LSTM with dropout for better generalization
- Returns confidence scores with probabilities

**Example Usage:**
```
Input: "I love this product"
→ Sentiment: positive
→ Probabilities: negative=2% neutral=5% positive=93%

Input: "This is terrible"
→ Sentiment: negative
→ Probabilities: negative=90% neutral=7% positive=3%
```

### 3. Text Generation (`text_generator.py`)
- Generates text continuations from seed text
- Provides autocomplete suggestions
- **Real dataset**: WikiText-2 (36K+ Wikipedia articles, 100M tokens)
- **Sample data**: Small corpus for quick testing
- Temperature-controlled sampling
- LSTM-based sequence-to-sequence generation

**Example Usage:**
```
Seed: "hello how are"
→ Generated: "hello how are you today"

Text: "thank you very"
→ Suggestions: much (45%), well (30%), nice (15%)
```

## 📦 Installation

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8 or higher

### Setup

1. **Clone the repository:**
```bash

```

2. **Activate virtual environment:**
```bash
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
# Installs: mlx, numpy, scikit-learn, datasets, jupyter, matplotlib, seaborn, plotly, wordcloud
```

## 🚀 Quick Start

### Interactive Jupyter Notebooks (Recommended)

```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Navigate to notebooks folder
cd notebooks

# 3. Start Jupyter
jupyter notebook
```

Open any notebook:
- **`00_Overview.ipynb`** - Quick 15-minute intro to all three techniques
- **`01_Intent_Classification.ipynb`** - 60-minute complete tutorial with theory
- **`02_Sentiment_Analysis.ipynb`** - 75-minute complete tutorial with theory
- **`03_Text_Generation.ipynb`** - 90-minute complete tutorial with theory
- **`04_Complete_Pipeline.ipynb`** - 120-minute full integration

**Each notebook is completely self-contained** with:
- ✅ Complete theory and explanations
- ✅ Working code examples
- ✅ Beautiful visualizations
- ✅ Hands-on exercises
- ✅ No external file dependencies

### Training with Real Datasets

**Download datasets:**
```bash
# Download all real datasets (IMDB 50K, SNIPS 16K+, Banking77 13K, WikiText 36K)
python scripts/download_datasets.py --all

# Or download specific datasets
python scripts/download_datasets.py --imdb     # Sentiment analysis
python scripts/download_datasets.py --snips    # Intent classification
python scripts/download_datasets.py --wikitext # Text generation
```

**Train models in notebooks:**
All training is now done through the Jupyter notebooks. Each notebook includes:
- Dataset loading and preprocessing
- Model training with progress visualization
- Evaluation and testing
- Interactive predictions

**Available Datasets:**
- **IMDB**: 50K movie reviews (sentiment)
- **SNIPS**: 16K+ voice assistant queries (7 intents)
- **Banking77**: 13K banking queries (77 intents)
- **WikiText-2**: 36K articles, 100M tokens (generation)

**See [notebooks/README.md](notebooks/README.md) for complete learning guide**

## 📁 Project Structure

```
apple-mlx-tutorial/
├── .venv/                           # Virtual environment
├── notebooks/                       # 📓 Interactive Jupyter notebooks
│   ├── mlx_nlp_utils.py             # Consolidated model code (300+ lines)
│   ├── README.md                    # Notebooks guide
│   ├── 00_Overview.ipynb            # Quick intro & demos (15 min)
│   ├── 01_Intent_Classification.ipynb    # Complete tutorial (60 min)
│   ├── 02_Sentiment_Analysis.ipynb       # Complete tutorial (75 min)
│   ├── 03_Text_Generation.ipynb          # Complete tutorial (90 min)
│   ├── 04_Complete_Pipeline.ipynb        # Full integration (120 min)
│   ├── 05_Attention_Mechanism.ipynb      # Advanced: Attention theory
│   ├── 06_Build_NanoGPT.ipynb            # Advanced: Build Transformer
│   ├── 07_Fine_Tuning_with_LoRA.ipynb    # Pro: Fine-tune LLMs
│   └── 08_RAG_from_Scratch.ipynb         # Architect: RAG System Design
├── data/                            # Datasets directory
│   ├── intent_samples/              # Sample intent data (9 examples)
│   ├── sentiment_samples/           # Sample sentiment data (8 reviews)
│   ├── text_gen_samples/            # Sample text corpus
│   ├── imdb/                        # IMDB movie reviews (50K)
│   ├── snips/                       # SNIPS intents (16K+)
│   ├── banking77/                   # Banking77 intents (13K)
│   └── wikitext/                    # WikiText corpus (36K articles)
├── scripts/                         # Utility scripts
│   ├── download_datasets.py         # Dataset downloader (Python)
│   └── download_datasets.sh         # Dataset downloader (Shell)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── QUICKSTART.md                    # Quick reference guide
├── TRAINING_GUIDE.md                # Training documentation
└── PRODUCTION_README.md             # Production deployment guide
```

## 🔧 Technical Details

### Model Architectures

All models are implemented in the notebooks with full explanations. You can find the complete code in `notebooks/mlx_nlp_utils.py`.

**Intent Classifier:**
- Embedding dimension: 32
- LSTM hidden size: 64
- 3 output classes
- Training: 50 epochs with SGD

**Sentiment Analyzer:**
- Embedding dimension: 64
- LSTM hidden size: 128
- Dropout: 0.3
- 3 output classes
- Training: 100 epochs with Adam

**Text Generator:**
- Embedding dimension: 128
- LSTM hidden size: 256
- Sequence length: 5 words
- Training: 200 epochs with Adam

### MLX Framework Benefits

- **Native Apple Silicon support** - Runs on M1/M2/M3 NPU
- **Efficient memory usage** - Optimized for unified memory architecture
- **Low latency** - On-device inference without cloud dependency
- **Familiar API** - NumPy-like interface with PyTorch-style modules

## 🎯 Use Cases

### For Chatbots
1. **Intent Classification** - Route user queries to appropriate handlers
2. **Sentiment Analysis** - Adjust tone of responses based on user emotion
3. **Text Generation** - Provide smart autocomplete and response suggestions

### For Production
- Combine with Core ML for deployment in iOS/macOS apps
- Fine-tune models on domain-specific data
- Extend with more intents, sentiments, or training data

## 🔄 Next Steps

### Expand Functionality
- Add Named Entity Recognition (NER) for extracting entities
- Implement question-answering with FAQ matching
- Add multilingual support with translation models
- Fine-tune on domain-specific datasets (customer support, medical, legal)

### Improve Models
- Experiment with Transformer architectures (attention mechanisms)
- Add beam search for better text generation
- Implement model ensembling for better accuracy
- Try different hyperparameters in the notebooks

### Production Deployment
- Convert models to Core ML format
- Build iOS/macOS app interface
- Add model versioning and A/B testing
- Deploy REST API with FastAPI/Flask

## 📝 Example Training Output

```
Training in Jupyter Notebook:

Epoch 10/50 - Loss: 0.8234 - Accuracy: 0.6667
Epoch 20/50 - Loss: 0.4521 - Accuracy: 0.8333
Epoch 30/50 - Loss: 0.2341 - Accuracy: 0.9333
Epoch 40/50 - Loss: 0.1234 - Accuracy: 0.9667
Epoch 50/50 - Loss: 0.0823 - Accuracy: 1.0000

[Training curves visualization displayed]
[Confusion matrix heatmap displayed]
```

## 🤝 Contributing

Feel free to extend this project with:
- Additional NLP tasks (NER, question-answering, etc.)
- Larger/better training datasets
- Different model architectures (Transformers, CNNs)
- Performance benchmarks
- iOS/macOS app integration

## 📖 Educational Resources

**🎓 Interactive Learning with Jupyter Notebooks**

### Learning Notebooks (Recommended)
**Visual, hands-on learning with 20+ types of visualizations:**
- **[notebooks/00_Overview.ipynb](notebooks/00_Overview.ipynb)** - Quick intro with demos of all 3 techniques (15 min)
- **[notebooks/01_Intent_Classification.ipynb](notebooks/01_Intent_Classification.ipynb)** - Full tutorial with training curves, confusion matrices (60 min)
- **[notebooks/02_Sentiment_Analysis.ipynb](notebooks/02_Sentiment_Analysis.ipynb)** - Word clouds, ROC curves (75 min)
- **[notebooks/03_Text_Generation.ipynb](notebooks/03_Text_Generation.ipynb)** - Perplexity, temperature comparison (90 min)
- **[notebooks/04_Complete_Pipeline.ipynb](notebooks/04_Complete_Pipeline.ipynb)** - End-to-end chatbot (120 min)
- **[notebooks/05_Attention_Mechanism.ipynb](notebooks/05_Attention_Mechanism.ipynb)** - The "brain" of Transformers (45 min)
- **[notebooks/06_Build_NanoGPT.ipynb](notebooks/06_Build_NanoGPT.ipynb)** - Build a GPT model from scratch (90 min)
- **[notebooks/07_Fine_Tuning_with_LoRA.ipynb](notebooks/07_Fine_Tuning_with_LoRA.ipynb)** - Fine-tune LLMs on Apple Silicon (60 min)
- **[notebooks/08_RAG_from_Scratch.ipynb](notebooks/08_RAG_from_Scratch.ipynb)** - Architect-level RAG system design (60 min)

**📘 See [notebooks/README.md](notebooks/README.md)** for learning paths, installation, and expected results.

### Documentation Guides

Quick reference for specific topics:
- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference guide
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Training workflows and benchmarks  
- **[PRODUCTION_README.md](PRODUCTION_README.md)** - Production deployment guide

### Learning Path Recommendations

**🔰 Total Beginner? (6 hours)**
1. Start with **notebooks/00_Overview.ipynb** (15 min) - Visual introduction
2. Work through **notebooks/01_Intent_Classification.ipynb** (60 min) - Complete tutorial
3. Continue with **notebooks/02_Sentiment_Analysis.ipynb** (75 min) - Build on classification
4. Explore **notebooks/03_Text_Generation.ipynb** (90 min) - Most advanced technique
5. Finish with **notebooks/04_Complete_Pipeline.ipynb** (120 min) - Full integration
6. **Advanced:** Dive into **notebooks/05_Attention_Mechanism.ipynb** and **06_Build_NanoGPT.ipynb** to understand Transformers.
7. **Pro:** Learn to fine-tune LLMs with **notebooks/07_Fine_Tuning_with_LoRA.ipynb**.
8. **Architect:** Master System Design with **notebooks/08_RAG_from_Scratch.ipynb**.

**Each notebook is 100% self-contained** - no jumping between files!

**🚀 Advanced Developer?**
1. Jump directly to notebooks based on your interest
2. Review **notebooks/mlx_nlp_utils.py** for model implementations
3. Download real datasets with `python scripts/download_datasets.py --all`
4. Build your own project using the notebook code as reference

### Quick Start Guide

**For Learning:**
```bash
cd notebooks && jupyter notebook
# Open 00_Overview.ipynb
```

**For Production Training:**
```bash
python scripts/download_datasets.py --all  # Download datasets
cd notebooks && jupyter notebook           # Train in notebooks
```

**Total Learning Content:**
- 5 complete notebooks with embedded theory
- 28+ visualization types
- 50+ diagrams and examples
- Beginner through advanced exercises
- 6 hours of guided learning

## 📄 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgments

- Apple MLX team for the excellent framework
- Inspired by practical chatbot NLP applications
- Built with ❤️ for Apple Silicon

## 📚 Further Reading

- **MLX Documentation**: https://ml-explore.github.io/mlx/
- **LSTM Paper**: "Long Short-Term Memory" by Hochreiter & Schmidhuber
- **Word Embeddings**: "Efficient Estimation of Word Representations" (Word2Vec)
- **Attention**: "Neural Machine Translation by Jointly Learning to Align and Translate"

---

**Ready to explore NLP on Apple Silicon?** 

**Start learning:** `cd notebooks && jupyter notebook` (open `00_Overview.ipynb`)

🚀 All code in easy-to-use notebooks - theory, code, and visualizations in one place!
