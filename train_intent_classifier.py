"""
Train Intent Classifier with Real Data
======================================

Train on SNIPS or sample intent data.

Usage:
    # Train on sample data
    python train_intent_classifier.py
    
    # Train on SNIPS data
    python train_intent_classifier.py --data data/snips
    
    # Custom epochs and batch size
    python train_intent_classifier.py --epochs 50 --batch-size 32
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# Simple Intent Classifier (compatible with existing code)
class SimpleIntentClassifier(nn.Module):
    """Lightweight intent classifier"""
    
    def __init__(self, vocab_size, num_classes, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        self.word2idx = {}
        self.idx2word = {}
        self.label2idx = {}
        self.idx2label = {}
    
    def __call__(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch, seq, emb)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]  # Last timestep
        logits = self.fc(last_hidden)
        return logits


def load_data(data_path):
    """Load intent data from JSON"""
    data_file = Path(data_path)
    
    if data_file.is_dir():
        # Directory provided, look for train.json
        data_file = data_file / 'train.json'
    
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        print("\nPlease download data first:")
        print("  python scripts/download_datasets.py --samples")
        sys.exit(1)
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['texts'], data['labels']


def build_vocab(texts, max_vocab=5000):
    """Build vocabulary from texts"""
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    
    # Count words
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # Add most common words
    for word, _ in word_counts.most_common(max_vocab - 2):
        idx = len(word2idx)
        word2idx[word] = idx
        idx2word[idx] = word
    
    return word2idx, idx2word


def build_label_mapping(labels):
    """Create label to index mapping"""
    unique_labels = sorted(set(labels))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label


def tokenize_text(text, word2idx, max_len=20):
    """Convert text to token IDs"""
    words = text.lower().split()
    tokens = [word2idx.get(word, 1) for word in words]  # 1 = <UNK>
    
    # Pad or truncate
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    
    return tokens


def prepare_dataset(texts, labels, word2idx, label2idx, max_len=20):
    """Prepare full dataset"""
    X = []
    y = []
    
    for text, label in zip(texts, labels):
        tokens = tokenize_text(text, word2idx, max_len)
        label_idx = label2idx[label]
        
        X.append(tokens)
        y.append(label_idx)
    
    return mx.array(X, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def train_model(model, X_train, y_train, epochs=30, batch_size=16, learning_rate=0.01):
    """Train the model"""
    optimizer = optim.SGD(learning_rate=learning_rate)
    
    def loss_fn(model, X, y):
        logits = model(X)
        return mx.mean(nn.losses.cross_entropy(logits, y))
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    print("\nTraining...")
    print(f"{'Epoch':<10} {'Loss':<12} {'Accuracy':<10}")
    print("-" * 35)
    
    n_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        # Shuffle data
        indices = mx.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        total_loss = 0
        correct = 0
        n_batches = 0
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # Compute loss and gradients
            loss, grads = loss_and_grad_fn(model, batch_X, batch_y)
            
            # Update weights
            optimizer.update(model, grads)
            
            # Evaluate
            mx.eval(model.parameters(), optimizer.state)
            
            total_loss += loss.item()
            n_batches += 1
            
            # Compute accuracy
            logits = model(batch_X)
            predictions = mx.argmax(logits, axis=-1)
            correct += mx.sum(predictions == batch_y).item()
        
        avg_loss = total_loss / n_batches
        accuracy = correct / n_samples
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:<10} {avg_loss:<12.4f} {accuracy:<10.2%}")
    
    print("\n✓ Training complete!")


def test_model(model, word2idx, idx2label, max_len=20):
    """Test the trained model"""
    print("\n" + "=" * 60)
    print("TESTING MODEL")
    print("=" * 60)
    
    test_examples = [
        "hello how are you",
        "what time is it",
        "please turn on the lights",
        "hey there",
        "when is the meeting",
        "open the door please",
    ]
    
    for text in test_examples:
        tokens = tokenize_text(text, word2idx, max_len)
        X = mx.array([tokens], dtype=mx.int32)
        
        logits = model(X)
        pred_idx = mx.argmax(logits, axis=-1).item()
        intent = idx2label[pred_idx]
        
        print(f"'{text}'")
        print(f"  → Intent: {intent}\n")


def main():
    parser = argparse.ArgumentParser(description="Train intent classifier")
    parser.add_argument('--data', default='data/intent_samples/data.json',
                       help='Path to training data')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--vocab-size', type=int, default=1000,
                       help='Maximum vocabulary size')
    parser.add_argument('--max-len', type=int, default=20,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("INTENT CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Load data
    print("\nLoading data...")
    texts, labels = load_data(args.data)
    print(f"Loaded {len(texts)} examples")
    print(f"Intents: {sorted(set(labels))}")
    
    # Show distribution
    label_counts = Counter(labels)
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count} examples")
    
    # Build vocab and label mapping
    print("\nBuilding vocabulary...")
    word2idx, idx2word = build_vocab(texts, max_vocab=args.vocab_size)
    print(f"Vocabulary size: {len(word2idx)}")
    
    label2idx, idx2label = build_label_mapping(labels)
    print(f"Number of intents: {len(label2idx)}")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    X_train, y_train = prepare_dataset(texts, labels, word2idx, label2idx, args.max_len)
    print(f"X shape: {X_train.shape}")
    print(f"y shape: {y_train.shape}")
    
    # Create model
    print("\nCreating model...")
    model = SimpleIntentClassifier(
        vocab_size=len(word2idx),
        num_classes=len(label2idx),
        embedding_dim=64,
        hidden_dim=128
    )
    model.word2idx = word2idx
    model.idx2word = idx2word
    model.label2idx = label2idx
    model.idx2label = idx2label
    
    # Train
    train_model(
        model, X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Test
    test_model(model, word2idx, idx2label, args.max_len)
    
    # Save model
    print("\n" + "=" * 60)
    print("Saving model...")
    output_dir = Path("trained_models")
    output_dir.mkdir(exist_ok=True)
    
    model_file = output_dir / "intent_classifier.npz"
    model.save_weights(str(model_file))
    
    # Save vocab and labels
    with open(output_dir / "intent_vocab.json", 'w') as f:
        json.dump({
            'word2idx': word2idx,
            'label2idx': label2idx,
            'idx2label': {str(k): v for k, v in idx2label.items()}
        }, f, indent=2)
    
    print(f"✓ Model saved to: {model_file}")
    print(f"✓ Vocab saved to: {output_dir / 'intent_vocab.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
