"""
Production-Ready Sentiment Analysis Example
============================================

This example demonstrates:
1. Loading real-world IMDB dataset
2. Data cleaning and preprocessing
3. Train/Val/Test splits
4. Model training with MLX
5. Comprehensive evaluation
6. Model versioning
7. Deployment-ready prediction API

Requirements:
    pip install datasets mlx numpy scikit-learn
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================================
# 1. DATA LOADING & CLEANING
# ============================================================================

class TextCleaner:
    """Clean and normalize text"""
    
    def clean(self, text):
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        return text.strip()

def load_and_clean_imdb(max_samples=25000):
    """Load IMDB dataset and clean it"""
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    
    cleaner = TextCleaner()
    
    # Process training data
    train_texts = []
    train_labels = []
    
    for i, example in enumerate(dataset['train']):
        if i >= max_samples:
            break
        
        text = cleaner.clean(example['text'])
        
        # Quality filter
        if len(text) < 20 or len(text.split()) < 5:
            continue
        
        train_texts.append(text)
        train_labels.append(example['label'])
    
    # Process test data
    test_texts = []
    test_labels = []
    
    for i, example in enumerate(dataset['test']):
        if i >= max_samples // 5:  # 20% of training size
            break
        
        text = cleaner.clean(example['text'])
        
        if len(text) < 20 or len(text.split()) < 5:
            continue
        
        test_texts.append(text)
        test_labels.append(example['label'])
    
    print(f"Loaded {len(train_texts)} training examples")
    print(f"Loaded {len(test_texts)} test examples")
    
    return train_texts, train_labels, test_texts, test_labels

# ============================================================================
# 2. TOKENIZATION & PREPROCESSING
# ============================================================================

class Tokenizer:
    """Production-ready tokenizer"""
    
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_counts = Counter()
        
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Take most common words
        most_common = word_counts.most_common(self.vocab_size - 2)
        
        for word, count in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx)}")
    
    def tokenize(self, text):
        """Convert text to token IDs"""
        words = text.split()
        return [self.word2idx.get(word, 1) for word in words]  # 1 = <UNK>
    
    def tokenize_batch(self, texts):
        """Tokenize multiple texts"""
        return [self.tokenize(text) for text in texts]
    
    def save(self, filepath):
        """Save vocabulary"""
        with open(filepath, 'w') as f:
            json.dump(self.word2idx, f)
    
    def load(self, filepath):
        """Load vocabulary"""
        with open(filepath, 'r') as f:
            self.word2idx = json.load(f)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}

def pad_sequences(sequences, max_length=256, value=0):
    """Pad sequences to same length"""
    padded = []
    
    for seq in sequences:
        if len(seq) > max_length:
            padded_seq = seq[:max_length]
        else:
            padded_seq = seq + [value] * (max_length - len(seq))
        
        padded.append(padded_seq)
    
    return padded

# ============================================================================
# 3. MODEL DEFINITION
# ============================================================================

class SentimentClassifier(nn.Module):
    """Production LSTM sentiment classifier"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_classes=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def __call__(self, x):
        # x shape: (batch_size, seq_length)
        
        # Embed
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_length, hidden_dim)
        
        # Take last timestep
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Dropout
        dropped = self.dropout(last_hidden)
        
        # Classification
        logits = self.fc(dropped)  # (batch_size, num_classes)
        
        return logits

# ============================================================================
# 4. TRAINING
# ============================================================================

def train_model(model, train_X, train_y, val_X, val_y, 
                epochs=10, batch_size=32, learning_rate=0.001):
    """Train model with validation"""
    
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    def loss_fn(model, X, y):
        logits = model(X)
        return mx.mean(nn.losses.cross_entropy(logits, y))
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    print("\nTraining...")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<10}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training
        total_train_loss = 0
        n_train_batches = 0
        
        indices = list(range(len(train_X)))
        import random
        random.shuffle(indices)
        
        for i in range(0, len(train_X), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = mx.array([train_X[j] for j in batch_indices])
            batch_y = mx.array([train_y[j] for j in batch_indices])
            
            # Forward and backward
            loss, grads = loss_and_grad_fn(model, batch_X, batch_y)
            
            # Update
            optimizer.update(model, grads)
            
            # Evaluate
            mx.eval(model.parameters(), optimizer.state)
            
            total_train_loss += loss.item()
            n_train_batches += 1
        
        avg_train_loss = total_train_loss / n_train_batches
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_X, val_y, batch_size)
        
        print(f"{epoch+1:<8} {avg_train_loss:<12.4f} {val_loss:<12.4f} {val_acc:<10.2%}")
    
    print("\n✓ Training complete!")

def evaluate_model(model, X, y, batch_size=32):
    """Evaluate model on dataset"""
    total_loss = 0
    correct = 0
    total = 0
    
    for i in range(0, len(X), batch_size):
        batch_X = mx.array(X[i:i+batch_size])
        batch_y = mx.array(y[i:i+batch_size])
        
        # Forward pass
        logits = model(batch_X)
        
        # Loss
        loss = mx.mean(nn.losses.cross_entropy(logits, batch_y))
        total_loss += loss.item()
        
        # Accuracy
        predictions = mx.argmax(logits, axis=-1)
        correct += mx.sum(predictions == batch_y).item()
        total += len(batch_y)
    
    avg_loss = total_loss / ((len(X) + batch_size - 1) // batch_size)
    accuracy = correct / total
    
    return avg_loss, accuracy

# ============================================================================
# 5. COMPREHENSIVE EVALUATION
# ============================================================================

def comprehensive_evaluation(model, test_X, test_y, class_names, batch_size=32):
    """Comprehensive model evaluation"""
    
    # Get all predictions
    all_predictions = []
    
    for i in range(0, len(test_X), batch_size):
        batch_X = mx.array(test_X[i:i+batch_size])
        logits = model(batch_X)
        predictions = mx.argmax(logits, axis=-1)
        all_predictions.extend(predictions.tolist())
    
    # Compute confusion matrix
    n_classes = len(class_names)
    confusion_matrix = [[0] * n_classes for _ in range(n_classes)]
    
    for true_label, pred_label in zip(test_y, all_predictions):
        confusion_matrix[true_label][pred_label] += 1
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    # Overall accuracy
    correct = sum(confusion_matrix[i][i] for i in range(n_classes))
    total = len(test_y)
    accuracy = correct / total
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Per-class metrics
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        tp = confusion_matrix[i][i]
        fp = sum(confusion_matrix[j][i] for j in range(n_classes)) - tp
        fn = sum(confusion_matrix[i][j] for j in range(n_classes)) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = sum(confusion_matrix[i][j] for j in range(n_classes))
        
        print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("        ", "  ".join(f"{cn:>8}" for cn in class_names))
    for i, class_name in enumerate(class_names):
        row = "  ".join(f"{confusion_matrix[i][j]:>8}" for j in range(n_classes))
        print(f"{class_name:<8} {row}")
    
    print("="*60)

# ============================================================================
# 6. MODEL VERSIONING
# ============================================================================

class ModelVersioning:
    """Save and load model versions"""
    
    def __init__(self, base_dir='production_models'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def save_model(self, model, tokenizer, metrics, config):
        """Save model with metadata"""
        from datetime import datetime
        
        version_id = datetime.now().strftime('v_%Y%m%d_%H%M%S')
        version_dir = self.base_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Save model weights
        model.save_weights(str(version_dir / 'model.npz'))
        
        # Save tokenizer
        tokenizer.save(str(version_dir / 'vocab.json'))
        
        # Save metadata
        metadata = {
            'version_id': version_id,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics,
            'config': config,
        }
        
        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Model saved: {version_id}")
        print(f"  Location: {version_dir}")
        
        return version_id
    
    def load_model(self, version_id):
        """Load model version"""
        version_dir = self.base_dir / version_id
        
        # Load metadata
        with open(version_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Loaded model: {version_id}")
        print(f"  Accuracy: {metadata['metrics']['accuracy']:.2%}")
        
        return str(version_dir / 'model.npz'), str(version_dir / 'vocab.json'), metadata

# ============================================================================
# 7. PREDICTION API
# ============================================================================

class SentimentPredictor:
    """Production-ready prediction API"""
    
    def __init__(self, model, tokenizer, max_length=256):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cleaner = TextCleaner()
        self.class_names = ['Negative', 'Positive']
    
    def predict(self, text):
        """Predict sentiment for single text"""
        # Clean
        text = self.cleaner.clean(text)
        
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Pad
        tokens = tokens[:self.max_length]
        tokens += [0] * (self.max_length - len(tokens))
        
        # Predict
        X = mx.array([tokens])
        logits = self.model(X)
        
        # Get probabilities
        probs = mx.softmax(logits, axis=-1)
        prediction = mx.argmax(logits, axis=-1)
        
        return {
            'sentiment': self.class_names[int(prediction[0])],
            'confidence': float(probs[0, prediction[0]]),
            'probabilities': {
                name: float(prob) 
                for name, prob in zip(self.class_names, probs[0])
            }
        }
    
    def predict_batch(self, texts, batch_size=32):
        """Predict sentiment for multiple texts"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Process batch
            batch_tokens = []
            for text in batch_texts:
                text = self.cleaner.clean(text)
                tokens = self.tokenizer.tokenize(text)
                tokens = tokens[:self.max_length]
                tokens += [0] * (self.max_length - len(tokens))
                batch_tokens.append(tokens)
            
            # Predict
            X = mx.array(batch_tokens)
            logits = self.model(X)
            probs = mx.softmax(logits, axis=-1)
            predictions = mx.argmax(logits, axis=-1)
            
            # Convert to results
            for j in range(len(batch_texts)):
                results.append({
                    'sentiment': self.class_names[int(predictions[j])],
                    'confidence': float(probs[j, predictions[j]]),
                })
        
        return results

# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def main():
    """Run complete production pipeline"""
    
    print("="*60)
    print("PRODUCTION SENTIMENT ANALYSIS PIPELINE")
    print("="*60)
    
    # Configuration
    config = {
        'vocab_size': 5000,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'dropout': 0.3,
        'max_length': 256,
        'batch_size': 32,
        'epochs': 5,
        'learning_rate': 0.001,
    }
    
    # 1. Load and clean data
    print("\n[1/7] Loading and cleaning data...")
    train_texts, train_labels, test_texts, test_labels = load_and_clean_imdb(max_samples=5000)
    
    # 2. Build vocabulary
    print("\n[2/7] Building vocabulary...")
    tokenizer = Tokenizer(vocab_size=config['vocab_size'])
    tokenizer.build_vocab(train_texts)
    
    # 3. Tokenize and pad
    print("\n[3/7] Tokenizing and padding...")
    train_X = tokenizer.tokenize_batch(train_texts)
    train_X = pad_sequences(train_X, max_length=config['max_length'])
    
    test_X = tokenizer.tokenize_batch(test_texts)
    test_X = pad_sequences(test_X, max_length=config['max_length'])
    
    # Create validation split (80/20 split of training data)
    split_idx = int(len(train_X) * 0.8)
    val_X = train_X[split_idx:]
    val_y = train_labels[split_idx:]
    train_X = train_X[:split_idx]
    train_y = train_labels[:split_idx]
    
    print(f"Train: {len(train_X)} examples")
    print(f"Val:   {len(val_X)} examples")
    print(f"Test:  {len(test_X)} examples")
    
    # 4. Create model
    print("\n[4/7] Creating model...")
    model = SentimentClassifier(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout']
    )
    
    # 5. Train model
    print("\n[5/7] Training model...")
    train_model(
        model, train_X, train_y, val_X, val_y,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )
    
    # 6. Evaluate
    print("\n[6/7] Evaluating model...")
    test_loss, test_accuracy = evaluate_model(model, test_X, test_labels, config['batch_size'])
    
    comprehensive_evaluation(
        model, test_X, test_labels,
        class_names=['Negative', 'Positive'],
        batch_size=config['batch_size']
    )
    
    # 7. Save model
    print("\n[7/7] Saving model...")
    versioning = ModelVersioning()
    version_id = versioning.save_model(
        model, tokenizer,
        metrics={'accuracy': test_accuracy, 'loss': test_loss},
        config=config
    )
    
    # Demo predictions
    print("\n" + "="*60)
    print("DEMO PREDICTIONS")
    print("="*60)
    
    predictor = SentimentPredictor(model, tokenizer, max_length=config['max_length'])
    
    demo_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible waste of time. Do not watch this movie.",
        "It was okay, nothing special but not terrible either.",
        "Best film I've seen this year! Highly recommended!",
        "Boring and predictable. Very disappointed.",
    ]
    
    for text in demo_texts:
        result = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nModel saved as: {version_id}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"\nYou can now:")
    print("  1. Load this model for inference")
    print("  2. Deploy as REST API")
    print("  3. Convert to Core ML for iOS")

if __name__ == "__main__":
    main()
