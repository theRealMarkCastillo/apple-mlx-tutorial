"""
MLX NLP Utilities - Complete Implementation
All model classes and utility functions in one file for notebooks.
This consolidates intent_classifier.py, sentiment_analysis.py, and text_generator.py
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

def set_device(device_type='gpu'):
    """
    Set the default device for MLX operations.
    
    Args:
        device_type (str): 'gpu' or 'cpu'. Defaults to 'gpu' if available.
    """
    if device_type == 'gpu':
        mx.set_default_device(mx.Device(mx.gpu))
    else:
        mx.set_default_device(mx.Device(mx.cpu))

def print_device_info():
    """Print current MLX device information and hardware acceleration status."""
    device = mx.default_device()
    print(f"\n🖥️  Hardware Acceleration Check:")
    print(f"   Device: {device}")
    
    if device == mx.Device(mx.gpu):
        print("   ✅ Using Apple Silicon GPU (Metal)")
        print("   ℹ️  MLX automatically optimizes for the GPU's Unified Memory.")
        print("   ℹ️  Note: While Apple Silicon has an NPU (Neural Engine), MLX primarily")
        print("       uses the powerful GPU for general-purpose training tasks like LSTMs.")
    else:
        print("   ⚠️  Using CPU (Slower)")
        print("   ℹ️  Consider switching to GPU if on Apple Silicon.")


# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

class IntentLSTM(nn.Module):
    """LSTM-based intent classifier"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def __call__(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        logits = self.linear(last_output)
        return logits


def create_vocabulary(texts: List[str]) -> Tuple[set, dict]:
    """Create vocabulary from texts"""
    vocab = set()
    for text in texts:
        for word in text.lower().split():
            vocab.add(word)
    word_to_idx = {word: i+2 for i, word in enumerate(sorted(vocab))}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    return set(word_to_idx.keys()), word_to_idx


def preprocess_text(text: str) -> List[str]:
    """Preprocess text for classification"""
    # Simple punctuation removal
    text = text.lower()
    for char in '.,!?;:':
        text = text.replace(char, ' ')
    return text.split()


def texts_to_sequences(texts: List[str], word_to_idx: dict) -> List[List[int]]:
    """Convert texts to sequences of indices"""
    sequences = []
    for text in texts:
        seq = [word_to_idx.get(word, word_to_idx['<UNK>']) 
               for word in text.lower().split()]
        sequences.append(seq)
    return sequences


def pad_sequences(sequences: List[List[int]], max_len: int) -> np.ndarray:
    """Pad sequences to same length"""
    padded = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return padded


def train_model(model: nn.Module, X: mx.array, y: mx.array, epochs: int = 50, learning_rate: float = 0.01) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Generic training loop for MLX models.
    
    Args:
        model: The MLX model to train
        X: Input features (mx.array)
        y: Target labels (mx.array)
        epochs: Number of training iterations
        learning_rate: Step size for the optimizer
        
    Returns:
        model: The trained model
        history: Dictionary with 'loss' and 'accuracy' lists
    """
    # Use Adam optimizer for better convergence
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    def loss_fn(model, X, y):
        logits = model(mx.array(X))
        # Handle different output shapes
        # If logits is 3D (batch, seq, vocab) and targets is 1D (batch), take last timestep
        if len(logits.shape) == 3 and len(y.shape) == 1:
            logits = logits[:, -1, :]
        return mx.mean(nn.losses.cross_entropy(logits, mx.array(y)))
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        
        # Force evaluation (Lazy Evaluation pattern)
        # MLX is lazy - it builds a computation graph but doesn't run it until needed.
        # mx.eval() forces the computation to happen now, ensuring the model parameters
        # and optimizer state are actually updated in memory.
        mx.eval(model.parameters(), optimizer.state)
        
        # Calculate accuracy
        logits = model(mx.array(X))
        if len(logits.shape) == 3 and len(y.shape) == 1:
            logits = logits[:, -1, :]
        predictions = mx.argmax(logits, axis=1)
        accuracy = mx.mean(predictions == mx.array(y))
        
        history['loss'].append(float(loss))
        history['accuracy'].append(float(accuracy))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
    
    return model, history


def predict_intent(model, text: str, word_to_idx: dict, intent_names: List[str], max_len: int) -> Tuple[str, float]:
    """Predict intent for a single text"""
    tokens = [word_to_idx.get(word.lower(), word_to_idx['<UNK>']) 
              for word in text.split()]
    tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
    
    X = mx.array([tokens])
    logits = model(X)
    probs = mx.softmax(logits, axis=-1)[0]
    pred_idx = int(mx.argmax(probs))
    confidence = float(probs[pred_idx])
    
    return intent_names[pred_idx], confidence


# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

class SentimentLSTM(nn.Module):
    """LSTM-based sentiment analyzer with dropout"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def __call__(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        logits = self.linear(dropped)
        return logits


def predict_sentiment(model, text: str, word_to_idx: dict, sentiment_names: List[str], max_len: int) -> Tuple[str, float]:
    """Predict sentiment for a single text"""
    tokens = [word_to_idx.get(word.lower(), word_to_idx['<UNK>']) 
              for word in text.split()]
    tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
    
    X = mx.array([tokens])
    logits = model(X)
    probs = mx.softmax(logits, axis=-1)[0]
    pred_idx = int(mx.argmax(probs))
    confidence = float(probs[pred_idx])
    
    return sentiment_names[pred_idx], confidence


# ============================================================================
# TEXT GENERATION
# ============================================================================

class TextLSTM(nn.Module):
    """LSTM-based text generator"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def __call__(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.linear(lstm_out)
        return logits


def create_char_vocab(text: str) -> Tuple[set, dict, dict]:
    """Create character vocabulary"""
    vocab = sorted(set(text))
    char_to_idx = {char: i for i, char in enumerate(vocab)}
    idx_to_char = {i: char for i, char in enumerate(vocab)}
    return set(vocab), char_to_idx, idx_to_char


def text_to_sequences(text: str, char_to_idx: dict, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert text to training sequences"""
    X, y = [], []
    for i in range(len(text) - seq_length):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        X.append([char_to_idx[char] for char in seq_in])
        y.append(char_to_idx[seq_out])
    
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


def generate_text(model, seed: str, char_to_idx: dict, idx_to_char: dict, 
                 length: int = 100, temperature: float = 1.0) -> str:
    """Generate text from seed"""
    generated = seed
    current_seq = [char_to_idx.get(c, 0) for c in seed[-5:]]
    
    for _ in range(length):
        X = mx.array([current_seq])
        logits = model(X)
        
        # Apply temperature
        logits = logits[0, -1, :] / temperature
        probs = mx.softmax(logits)
        
        # Sample
        next_idx = int(mx.random.categorical(mx.log(probs), num_samples=1)[0])
        next_char = idx_to_char[next_idx]
        
        generated += next_char
        current_seq = current_seq[1:] + [next_idx]
    
    return generated


# ============================================================================
# SAMPLE DATA LOADERS
# ============================================================================

def _find_data_file(filename):
    """Helper to find data file in common locations"""
    candidates = [
        Path(filename),
        Path("..") / filename,
        Path("data") / filename,
        Path("../data") / filename,
        Path("notebooks/data") / filename
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

def load_sample_intent_data():
    """Load sample intent classification data"""
    data_path = _find_data_file("intent_samples/data.json")
    
    if data_path:
        print(f"Loading intent data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = data['texts']
        labels = data['labels']
    else:
        print("Using hardcoded intent data (synthetic data not found)")
        texts = [
            "Hello", "Hi there", "Good morning",
            "What's the weather", "Tell me the time", "How are you",
            "Turn on the light", "Set a timer", "Play music"
        ]
        labels = ['greeting', 'greeting', 'greeting', 'question', 'question', 'question', 'command', 'command', 'command']
    
    # Build vocabulary
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    # Build intent mapping
    intent2idx = {'greeting': 0, 'question': 1, 'command': 2}
    
    return texts, labels, vocab, intent2idx


def load_sample_sentiment_data():
    """Load sample sentiment analysis data"""
    data_path = _find_data_file("sentiment_samples/data.json")
    
    if data_path:
        print(f"Loading sentiment data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = data['texts']
        labels = data['labels']
    else:
        print("Using hardcoded sentiment data (synthetic data not found)")
        texts = [
            "I love this", "This is amazing", "Fantastic",
            "I hate this", "This is terrible", "Awful",
            "It's okay", "Not bad", "Average"
        ]
        labels = ['positive', 'positive', 'positive', 'negative', 'negative', 'negative', 'neutral', 'neutral', 'neutral']
    
    # Build vocabulary
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    # Build sentiment mapping
    sentiment2idx = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    return texts, labels, vocab, sentiment2idx


def load_sample_corpus():
    """Load sample text generation corpus"""
    data_path = _find_data_file("text_gen_samples/corpus.txt")
    
    if data_path:
        print(f"Loading corpus from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            corpus = f.read()
    else:
        print("Using hardcoded corpus (synthetic data not found)")
        corpus = """hello how are you today what is your name thank you very much"""
    
    # Build character vocabulary
    vocab = {c: i for i, c in enumerate(sorted(set(corpus)))}
    idx2char = {i: c for c, i in vocab.items()}
    
    return corpus, vocab, idx2char

def load_rag_knowledge_base():
    """Load sample RAG knowledge base"""
    data_path = _find_data_file("rag_samples/knowledge_base.json")
    
    if data_path:
        print(f"Loading knowledge base from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
    else:
        print("Using hardcoded knowledge base (synthetic data not found)")
        documents = [
            "MLX is an array framework for machine learning on Apple Silicon.",
            "The Unified Memory architecture allows CPU and GPU to share memory.",
            "LSTMs are recurrent neural networks capable of learning long-term dependencies."
        ]
    
    return documents


# ============================================================================
# MODEL PERSISTENCE & EVALUATION
# ============================================================================

def save_model(model: nn.Module, path: str):
    """
    Save model weights to a file.
    
    Args:
        model: The MLX model to save
        path: Path to save the weights (e.g., 'model.npz' or 'model.safetensors')
    """
    path = str(path)
    print(f"Saving model weights to {path}...")
    model.save_weights(path)
    print("✅ Model saved successfully.")


def load_model(model: nn.Module, path: str):
    """
    Load model weights from a file.
    
    Args:
        model: The MLX model instance (must be initialized with same architecture)
        path: Path to the weights file
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")
        
    print(f"Loading model weights from {path}...")
    model.load_weights(path)
    print("✅ Model loaded successfully.")


def evaluate_model(model: nn.Module, X: mx.array, y: mx.array) -> Tuple[float, List[int], List[int]]:
    """
    Evaluate model and return accuracy + predictions for confusion matrix.
    
    Args:
        model: Trained MLX model
        X: Input features
        y: True labels
        
    Returns:
        accuracy: Float
        y_true: List of true labels
        y_pred: List of predicted labels
    """
    # Ensure evaluation
    mx.eval(X, y)
    
    logits = model(X)
    if len(logits.shape) == 3 and len(y.shape) == 1:
        logits = logits[:, -1, :]
        
    predictions = mx.argmax(logits, axis=1)
    accuracy = mx.mean(predictions == y).item()
    
    return accuracy, y.tolist(), predictions.tolist()

