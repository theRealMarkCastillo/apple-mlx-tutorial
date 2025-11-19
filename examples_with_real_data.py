"""
Examples Using Real Datasets
=============================

This script demonstrates how to use real datasets with our NLP models.

Prerequisites:
    1. Download datasets: python scripts/download_datasets.py --samples
    2. Run this script: python examples_with_real_data.py
"""

import json
import mlx.core as mx
from pathlib import Path

# Import our models and training functions
from intent_classifier import IntentClassifier, train_model as train_intent, predict_intent
from sentiment_analysis import SentimentAnalyzer, train_sentiment_model, predict_sentiment
from text_generator import TextGenerator, train_text_generator, generate_text


def load_json_dataset(filepath):
    """Load dataset from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['texts'], data['labels']


def load_text_corpus(filepath):
    """Load text corpus for generation"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


# ============================================================================
# Example 1: Intent Classification with Real Data
# ============================================================================

def example_intent_with_real_data():
    """Train intent classifier on real SNIPS-like data"""
    print("=" * 60)
    print("EXAMPLE 1: Intent Classification with Real Data")
    print("=" * 60)
    
    # Check if data exists
    data_file = Path("data/intent_samples/data.json")
    if not data_file.exists():
        print("\n⚠️  Data not found. Please run:")
        print("    python scripts/download_datasets.py --samples")
        return
    
    # Load data
    print("\nLoading intent data...")
    texts, labels = load_json_dataset(data_file)
    
    print(f"Loaded {len(texts)} examples")
    print(f"Intents: {set(labels)}")
    
    # Show samples
    print("\nSample data:")
    for text, label in list(zip(texts, labels))[:3]:
        print(f"  '{text}' → {label}")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    # Map intents
    unique_intents = sorted(set(labels))
    intent2idx = {intent: i for i, intent in enumerate(unique_intents)}
    
    # Prepare training data
    max_len = max(len(t.split()) for t in texts)
    X = []
    for text in texts:
        tokens = [vocab.get(word.lower(), vocab["<UNK>"]) for word in text.split()]
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
        X.append(tokens)
    
    X = mx.array(X, dtype=mx.int32)
    y = mx.array([intent2idx[label] for label in labels], dtype=mx.int32)
    
    # Initialize and train model
    print("\nTraining intent classifier...")
    model = IntentClassifier(vocab_size=len(vocab), embedding_dim=32, hidden_dim=64, num_classes=len(unique_intents))
    model = train_intent(model, X, y, epochs=30)
    
    # Test predictions
    print("\n" + "-" * 60)
    print("Testing on new examples:")
    print("-" * 60)
    
    test_examples = [
        "hey there how are you doing",
        "what is the weather forecast",
        "please open the window",
        "turn off the lights please",
    ]
    
    for text in test_examples:
        intent, confidence = predict_intent(model, text, vocab, unique_intents, max_len)
        print(f"'{text}'")
        print(f"  → Predicted intent: {intent} (confidence: {confidence:.2f})\n")
    
    print("✓ Intent classification complete!")


# ============================================================================
# Example 2: Sentiment Analysis with Real Data
# ============================================================================

def example_sentiment_with_real_data():
    """Train sentiment analyzer on real review data"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Sentiment Analysis with Real Data")
    print("=" * 60)
    
    # Check if data exists
    data_file = Path("data/sentiment_samples/data.json")
    if not data_file.exists():
        print("\n⚠️  Data not found. Please run:")
        print("    python scripts/download_datasets.py --samples")
        return
    
    # Load data
    print("\nLoading sentiment data...")
    texts, labels = load_json_dataset(data_file)
    
    # Convert string labels to binary
    label_map = {"positive": 1, "negative": 0, "neutral": 0}  # Treat neutral as negative for binary
    labels = [label_map.get(l, 0) for l in labels]
    
    print(f"Loaded {len(texts)} reviews")
    print(f"Sentiments: positive={sum(labels)}, negative={len(labels)-sum(labels)}")
    
    # Show samples
    print("\nSample reviews:")
    for text, label in list(zip(texts, labels))[:3]:
        sentiment = "positive" if label == 1 else "negative"
        print(f"  '{text[:60]}...' → {sentiment}")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    # Prepare training data
    max_len = min(50, max(len(t.split()) for t in texts))  # Cap at 50 for speed
    X = []
    for text in texts:
        tokens = [vocab.get(word.lower(), vocab["<UNK>"]) for word in text.split()]
        tokens = tokens[:max_len]  # Truncate
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
        X.append(tokens)
    
    X = mx.array(X, dtype=mx.int32)
    y = mx.array(labels, dtype=mx.int32)
    
    # Initialize and train model
    print("\nTraining sentiment analyzer...")
    model = SentimentAnalyzer(vocab_size=len(vocab), embedding_dim=32, hidden_dim=64, num_classes=2)
    model = train_sentiment_model(model, X, y, epochs=30)
    
    # Test predictions
    print("\n" + "-" * 60)
    print("Testing on new reviews:")
    print("-" * 60)
    
    test_reviews = [
        "This product is amazing! I love it.",
        "Terrible quality, waste of money.",
        "Pretty good, does what it says.",
        "Disappointed, expected better.",
    ]
    
    sentiment_names = ["negative", "positive"]
    for review in test_reviews:
        sentiment, probs = predict_sentiment(model, review, vocab, sentiment_names, max_len)
        # Extract confidence as float
        sentiment_idx = sentiment_names.index(sentiment)
        confidence = probs[sentiment_idx].item()
        print(f"'{review}'")
        print(f"  → Sentiment: {sentiment} (confidence: {confidence:.2f})\n")
    
    print("✓ Sentiment analysis complete!")


# ============================================================================
# Example 3: Text Generation with Real Data
# ============================================================================

def example_generation_with_real_data():
    """Train text generator on real corpus"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Text Generation with Real Data")
    print("=" * 60)
    
    # Check if data exists
    data_file = Path("data/text_gen_samples/corpus.txt")
    if not data_file.exists():
        print("\n⚠️  Data not found. Please run:")
        print("    python scripts/download_datasets.py --samples")
        return
    
    # Load corpus
    print("\nLoading text corpus...")
    corpus = load_text_corpus(data_file)
    
    print(f"Loaded corpus with {len(corpus)} characters")
    print(f"Words: {len(corpus.split())}")
    
    # Show samples
    print("\nSample text:")
    print(corpus[:200], "...")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word in corpus.split():
        if word not in vocab:
            vocab[word] = len(vocab)
    
    # Prepare training data (create sequences)
    print("Preparing sequences...")
    words = corpus.split()
    seq_len = 10
    X, y = [], []
    for i in range(len(words) - seq_len):
        X.append([vocab.get(w, vocab["<UNK>"]) for w in words[i:i+seq_len]])
        y.append(vocab.get(words[i+seq_len], vocab["<UNK>"]))
    
    X = mx.array(X, dtype=mx.int32)
    y = mx.array(y, dtype=mx.int32)
    
    # Initialize and train model
    print("\nTraining text generator...")
    model = TextGenerator(vocab_size=len(vocab), embedding_dim=32, hidden_dim=64)
    model = train_text_generator(model, X, y, epochs=50)
    
    # Test generation
    print("\n" + "-" * 60)
    print("Generating text:")
    print("-" * 60)
    
    prompts = [
        "The weather",
        "I love",
        "Technology is",
    ]
    
    idx2word = {v: k for k, v in vocab.items()}
    for prompt in prompts:
        generated = generate_text(model, prompt, vocab, idx2word, seq_length=seq_len, max_words=10)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: {generated}\n")
    
    print("✓ Text generation complete!")


# ============================================================================
# Example 4: IMDB Sentiment (Full Dataset)
# ============================================================================

def example_imdb_sentiment():
    """Train on full IMDB dataset (25K reviews)"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: IMDB Sentiment Analysis (25K Reviews)")
    print("=" * 60)
    
    # Check if data exists
    train_file = Path("data/imdb/train.json")
    if not train_file.exists():
        print("\n⚠️  IMDB data not found. Please run:")
        print("    python scripts/download_datasets.py --imdb")
        return
    
    print("\nLoading IMDB dataset...")
    print("(Using subset for demo - 1000 samples)")
    
    texts, labels = load_json_dataset(train_file)
    
    # Use subset for demo
    subset_size = 1000
    texts = texts[:subset_size]
    labels = labels[:subset_size]
    
    print(f"Training on {len(texts)} reviews")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    # Prepare data
    max_len = 100  # Limit for speed
    X = []
    for text in texts:
        tokens = [vocab.get(word.lower(), vocab["<UNK>"]) for word in text.split()]
        tokens = tokens[:max_len]
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
        X.append(tokens)
    
    X = mx.array(X, dtype=mx.int32)
    y = mx.array(labels, dtype=mx.int32)
    
    # Train model
    print("\nTraining on IMDB reviews...")
    model = SentimentAnalyzer(vocab_size=len(vocab), embedding_dim=64, hidden_dim=128, num_classes=2)
    model = train_sentiment_model(model, X, y, epochs=20)
    
    # Test
    test_reviews = [
        "This movie was absolutely fantastic! One of the best I've seen.",
        "Waste of time. Terrible acting and boring plot.",
        "Not bad, but could have been better. Decent entertainment.",
    ]
    
    sentiment_names = ["negative", "positive"]
    print("\nTesting:")
    for review in test_reviews:
        sentiment, probs = predict_sentiment(model, review, vocab, sentiment_names, max_len)
        sentiment_idx = sentiment_names.index(sentiment)
        confidence = probs[sentiment_idx].item()
        print(f"'{review[:60]}...'")
        print(f"  → {sentiment} ({confidence:.2f})\n")
    
    print("✓ IMDB analysis complete!")


# ============================================================================
# Example 5: SNIPS Intents (Full Dataset)
# ============================================================================

def example_snips_intents():
    """Train on full SNIPS dataset"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: SNIPS Intent Classification")
    print("=" * 60)
    
    # Check if data exists
    train_file = Path("data/snips/train.json")
    if not train_file.exists():
        print("\n⚠️  SNIPS data not found. Please run:")
        print("    python scripts/download_datasets.py --snips")
        return
    
    print("\nLoading SNIPS dataset...")
    texts, labels = load_json_dataset(train_file)
    
    print(f"Training on {len(texts)} examples")
    print(f"Unique intents: {len(set(labels))}")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    # Map intents
    unique_intents = sorted(set(labels))
    intent2idx = {intent: i for i, intent in enumerate(unique_intents)}
    
    # Prepare data
    max_len = max(len(t.split()) for t in texts)
    X = []
    for text in texts:
        tokens = [vocab.get(word.lower(), vocab["<UNK>"]) for word in text.split()]
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
        X.append(tokens)
    
    X = mx.array(X, dtype=mx.int32)
    y = mx.array([intent2idx[label] for label in labels], dtype=mx.int32)
    
    # Train
    print("\nTraining intent classifier...")
    model = IntentClassifier(vocab_size=len(vocab), embedding_dim=64, hidden_dim=128, num_classes=len(unique_intents))
    model = train_intent(model, X, y, epochs=40)
    
    # Test
    test_queries = [
        "what's the weather like tomorrow",
        "play some music please",
        "set an alarm for 7 am",
    ]
    
    print("\nTesting:")
    for query in test_queries:
        intent, confidence = predict_intent(model, query, vocab, unique_intents, max_len)
        print(f"'{query}'")
        print(f"  → {intent} ({confidence:.2f})\n")
    
    print("✓ SNIPS classification complete!")


# ============================================================================
# Example 6: Complete Pipeline Demo
# ============================================================================

def example_complete_pipeline():
    """Demonstrate complete train/val/test workflow"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Complete ML Pipeline")
    print("=" * 60)
    
    # Check if data exists
    data_file = Path("data/sentiment_samples/data.json")
    if not data_file.exists():
        print("\n⚠️  Data not found. Please run:")
        print("    python scripts/download_datasets.py --samples")
        return
    
    print("\nLoading data...")
    texts, labels = load_json_dataset(data_file)
    
    # Convert string labels to binary
    label_map = {"positive": 1, "negative": 0, "neutral": 0}
    labels = [label_map.get(l, 0) for l in labels]
    
    # Split into train/val/test
    n = len(texts)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    train_texts, train_labels = texts[:train_size], labels[:train_size]
    val_texts, val_labels = texts[train_size:train_size+val_size], labels[train_size:train_size+val_size]
    test_texts, test_labels = texts[train_size+val_size:], labels[train_size+val_size:]
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Build vocabulary (only from training data)
    print("\nBuilding vocabulary from training data...")
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    # Prepare all datasets
    max_len = 50
    
    def prepare_dataset(texts, labels):
        X = []
        for text in texts:
            tokens = [vocab.get(word.lower(), vocab["<UNK>"]) for word in text.split()]
            tokens = tokens[:max_len]
            tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
            X.append(tokens)
        return mx.array(X, dtype=mx.int32), mx.array(labels, dtype=mx.int32)
    
    X_train, y_train = prepare_dataset(train_texts, train_labels)
    X_val, y_val = prepare_dataset(val_texts, val_labels)
    X_test, y_test = prepare_dataset(test_texts, test_labels)
    
    # Train model
    print("\nTraining...")
    model = SentimentAnalyzer(vocab_size=len(vocab), embedding_dim=32, hidden_dim=64, num_classes=2)
    model = train_sentiment_model(model, X_train, y_train, epochs=20)
    
    # Evaluate on validation set
    print("\nValidation accuracy:")
    val_logits = model(X_val)
    val_preds = mx.argmax(val_logits, axis=1)
    val_acc = mx.mean(val_preds == y_val).item()
    print(f"  Accuracy: {val_acc:.4f}")
    
    # Evaluate on test set
    print("\nTest accuracy:")
    test_logits = model(X_test)
    test_preds = mx.argmax(test_logits, axis=1)
    test_acc = mx.mean(test_preds == y_test).item()
    print(f"  Accuracy: {test_acc:.4f}")
    
    print("\n✓ Pipeline complete!")
    print(f"Final model performance: {test_acc:.1%} test accuracy")


# ============================================================================
# Main Menu
# ============================================================================

def main():
    """Interactive menu for examples"""
    print("\n" + "=" * 60)
    print("EXAMPLES WITH REAL DATASETS")
    print("=" * 60)
    
    while True:
        print("\nAvailable examples:")
        print("  1. Intent Classification (sample data)")
        print("  2. Sentiment Analysis (sample data)")
        print("  3. Text Generation (sample data)")
        print("  4. IMDB Sentiment (25K reviews)")
        print("  5. SNIPS Intents (16K+ queries)")
        print("  6. Complete Pipeline Demo")
        print("  7. Run all sample examples (1-3, 6)")
        print("  0. Exit")
        
        choice = input("\nEnter choice (0-7): ").strip()
        
        if choice == "0":
            print("\nGoodbye!")
            break
        elif choice == "1":
            example_intent_with_real_data()
        elif choice == "2":
            example_sentiment_with_real_data()
        elif choice == "3":
            example_generation_with_real_data()
        elif choice == "4":
            example_imdb_sentiment()
        elif choice == "5":
            example_snips_intents()
        elif choice == "6":
            example_complete_pipeline()
        elif choice == "7":
            print("\nRunning all sample examples...\n")
            example_intent_with_real_data()
            example_sentiment_with_real_data()
            example_generation_with_real_data()
            example_complete_pipeline()
            print("\n" + "=" * 60)
            print("✓ ALL EXAMPLES COMPLETE!")
            print("=" * 60)
        else:
            print("\n❌ Invalid choice. Please enter 0-7.")


if __name__ == "__main__":
    main()
