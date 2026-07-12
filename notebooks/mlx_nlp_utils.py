"""
MLX NLP Utilities - Complete Implementation
All model classes and utility functions in one file for notebooks.
This consolidates intent_classifier.py, sentiment_analysis.py, and text_generator.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


_PAD_IDX = 0
_UNK_IDX = 1
_CHAR_PAD = "<PAD>"
_CHAR_UNK = "<UNK>"


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

def has_gpu() -> bool:
    """Return True if an Apple Silicon GPU is available via MLX."""
    try:
        return bool(mx.metal.is_available())
    except AttributeError:
        # Older MLX versions exposed mx.gpu / mx.cpu device handles.
        try:
            mx.set_default_device(mx.gpu)
            mx.eval(mx.array([0]))
            return True
        except Exception:
            return False
    except Exception:
        return False


def set_device(device_type: str = "gpu") -> None:
    """
    Set the default MLX device.

    Args:
        device_type: 'gpu' or 'cpu'. Defaults to 'gpu' if available, else 'cpu'.
    """
    device_type = (device_type or "gpu").lower()
    if device_type == "gpu" and has_gpu():
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)


def print_device_info() -> None:
    """Print current MLX device information and hardware acceleration status."""
    device = mx.default_device()
    print("\n🖥️  Hardware Acceleration Check:")
    print(f"   Device: {device}")

    on_gpu = False
    try:
        on_gpu = bool(mx.metal.is_available()) and (
            "gpu" in str(device).lower() or "metal" in str(device).lower()
        )
    except AttributeError:
        on_gpu = "gpu" in str(device).lower()

    if on_gpu:
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
    """LSTM-based intent classifier."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, output_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        logits = self.linear(last_output)
        return logits


def create_vocabulary(texts: list[str]) -> dict[str, int]:
    """Build a word vocabulary with reserved <PAD>=0 and <UNK>=1."""
    vocab = {"<PAD>": _PAD_IDX, "<UNK>": _UNK_IDX}
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def preprocess_text(text: str) -> list[str]:
    """Lowercase text and strip common punctuation, returning whitespace tokens."""
    text = text.lower()
    for char in ".,!?;:":
        text = text.replace(char, " ")
    return text.split()


def texts_to_sequences(texts: list[str], word_to_idx: dict) -> list[list[int]]:
    """Convert texts to lists of vocabulary indices, mapping unknowns to <UNK>."""
    sequences = []
    for text in texts:
        seq = [
            word_to_idx[word] if word in word_to_idx else word_to_idx["<UNK>"]
            for word in text.lower().split()
        ]
        sequences.append(seq)
    return sequences


def pad_sequences(sequences: list[list[int]], max_len: int) -> np.ndarray:
    """Pad / truncate sequences to ``max_len``, filling with the <PAD> index."""
    padded = np.full((len(sequences), max_len), _PAD_IDX, dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        if length > 0:
            padded[i, :length] = seq[:length]
    return padded


def train_model(
    model: nn.Module,
    X: mx.array,
    y: mx.array,
    epochs: int = 50,
    learning_rate: float = 0.01,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """
    Generic training loop for MLX models.

    Uses Adam and the lazy-evaluation pattern; ``mx.eval`` is called every step to
    force the parameter and optimizer-state updates to materialize.

    Returns:
        The trained model and a history dict with 'loss' and 'accuracy' lists.
    """
    if hasattr(model, "train"):
        model.train()

    optimizer = optim.Adam(learning_rate=learning_rate)

    def loss_fn(model, X, y):
        logits = model(X)
        # ``TextLSTM`` emits (batch, seq, vocab); collapse to last timestep for
        # classification losses when targets are 1-D class indices.
        if len(logits.shape) == 3 and len(y.shape) == 1:
            logits = logits[:, -1, :]
        return mx.mean(nn.losses.cross_entropy(logits, y))

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)

        # MLX is lazy: build the graph, but only materialize when needed.
        # ``mx.eval`` forces the parameter and optimizer-state updates to land
        # in memory before we read metrics below.
        mx.eval(model.parameters(), optimizer.state)

        logits = model(X)
        if len(logits.shape) == 3 and len(y.shape) == 1:
            logits = logits[:, -1, :]
        predictions = mx.argmax(logits, axis=-1)
        accuracy = mx.mean(predictions == y)

        history["loss"].append(float(loss))
        history["accuracy"].append(float(accuracy))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

    if hasattr(model, "eval"):
        model.eval()

    return model, history


def predict_intent(
    model,
    text: str,
    word_to_idx: dict,
    intent_names: list[str],
    max_len: int,
) -> tuple[str, float]:
    """Predict the intent for a single text."""
    if hasattr(model, "eval"):
        model.eval()
    tokens = [
        word_to_idx[word] if word in word_to_idx else word_to_idx["<UNK>"]
        for word in text.lower().split()
    ]
    tokens = tokens[:max_len] + [_PAD_IDX] * (max_len - len(tokens))

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
    """LSTM-based sentiment analyzer with dropout. Dropout is disabled in eval mode."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, output_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, output_size)

    def __call__(self, x: mx.array) -> mx.array:
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        # ``model.training`` flips via ``model.train()`` / ``model.eval()``,
        # so dropout is skipped at inference time.
        if self.training:
            last_output = self.dropout(last_output)
        logits = self.linear(last_output)
        return logits


def predict_sentiment(
    model,
    text: str,
    word_to_idx: dict,
    sentiment_names: list[str],
    max_len: int,
) -> tuple[str, float]:
    """Predict the sentiment for a single text."""
    if hasattr(model, "eval"):
        model.eval()
    tokens = [
        word_to_idx[word] if word in word_to_idx else word_to_idx["<UNK>"]
        for word in text.lower().split()
    ]
    tokens = tokens[:max_len] + [_PAD_IDX] * (max_len - len(tokens))

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
    """LSTM-based text generator."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.linear(lstm_out)
        return logits


def create_char_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build a character vocabulary that reserves ``<PAD>=0`` and ``<UNK>=1``.

    Returns ``(char_to_idx, idx_to_char)``. The first two indices are reserved
    so that OOV characters never collide with valid ones.
    """
    unique_chars = sorted(set(text))
    char_to_idx = {_CHAR_PAD: _PAD_IDX, _CHAR_UNK: _UNK_IDX}
    for c in unique_chars:
        if c not in char_to_idx:
            char_to_idx[c] = len(char_to_idx)
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def text_to_sequences(
    text: str,
    char_to_idx: dict,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert text to ``(X, y)`` training pairs of length ``seq_length``."""
    X: list[list[int]] = []
    y: list[int] = []
    for i in range(len(text) - seq_length):
        seq_in = text[i : i + seq_length]
        seq_out = text[i + seq_length]
        X.append([char_to_idx.get(c, _UNK_IDX) for c in seq_in])
        y.append(char_to_idx.get(seq_out, _UNK_IDX))

    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


def generate_text(
    model,
    seed: str,
    char_to_idx: dict,
    idx_to_char: dict,
    length: int = 100,
    temperature: float = 1.0,
) -> str:
    """Sample ``length`` characters of text continuation from ``seed``."""
    if hasattr(model, "eval"):
        model.eval()
    if not char_to_idx:
        raise ValueError("char_to_idx is empty — call create_char_vocab() first.")

    # Use the most recent ``seq_length`` chars when possible; fall back to
    # padding when the seed is shorter than the expected window.
    seq_length = max(1, len(char_to_idx) - 2)  # heuristic; the caller should pass an explicit seq_length
    # In practice we only need the last 5 chars for the LSTM used in this tutorial.
    window = 5
    generated = seed
    current_seq = [char_to_idx.get(c, _UNK_IDX) for c in seed[-window:]]
    current_seq = ([_PAD_IDX] * (window - len(current_seq))) + current_seq

    for _ in range(length):
        X = mx.array([current_seq])
        logits = model(X)

        # Last timestep, with optional temperature scaling.
        logits = logits[0, -1, :] / max(temperature, 1e-8)
        probs = mx.softmax(logits)

        next_idx = int(mx.random.categorical(mx.log(probs), num_samples=1)[0])
        next_char = idx_to_char.get(next_idx, _CHAR_UNK)

        generated += next_char
        current_seq = current_seq[1:] + [next_idx]
        # Materialize the per-step computation graph; cheap for one step.
        mx.eval(current_seq)

    return generated


# ============================================================================
# SAMPLE DATA LOADERS
# ============================================================================

def _find_data_file(filename: str) -> Path | None:
    """Find ``filename`` relative to the repo root, the cwd, or common subdirs."""
    candidates: list[Path] = []
    # Canonical: repo-root/data/...
    try:
        repo_root = Path(__file__).resolve().parent.parent
        candidates.append(repo_root / "data" / filename)
    except (NameError, IndexError):
        pass
    # Fallbacks for running from various cwd locations
    candidates += [
        Path(filename),
        Path("data") / filename,
        Path("../data") / filename,
        Path("notebooks/data") / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_sample_intent_data() -> tuple[list[str], list[str], dict[str, int], dict[str, int]]:
    """Load sample intent classification data from ``data/intent_samples/data.json``."""
    data_path = _find_data_file("intent_samples/data.json")

    if data_path:
        print(f"Loading intent data from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = data["texts"]
        labels = data["labels"]
    else:
        print("Using hardcoded intent data (synthetic data not found)")
        texts = [
            "Hello", "Hi there", "Good morning",
            "What's the weather", "Tell me the time", "How are you",
            "Turn on the light", "Set a timer", "Play music",
        ]
        labels = ["greeting", "greeting", "greeting",
                  "question", "question", "question",
                  "command", "command", "command"]

    vocab = create_vocabulary(texts)
    intent2idx = {"greeting": 0, "question": 1, "command": 2}
    return texts, labels, vocab, intent2idx


def load_sample_sentiment_data() -> tuple[list[str], list[str], dict[str, int], dict[str, int]]:
    """Load sample sentiment data from ``data/sentiment_samples/data.json``."""
    data_path = _find_data_file("sentiment_samples/data.json")

    if data_path:
        print(f"Loading sentiment data from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = data["texts"]
        labels = data["labels"]
    else:
        print("Using hardcoded sentiment data (synthetic data not found)")
        texts = [
            "I love this", "This is amazing", "Fantastic",
            "I hate this", "This is terrible", "Awful",
            "It's okay", "Not bad", "Average",
        ]
        labels = ["positive", "positive", "positive",
                  "negative", "negative", "negative",
                  "neutral", "neutral", "neutral"]

    vocab = create_vocabulary(texts)
    sentiment2idx = {"negative": 0, "neutral": 1, "positive": 2}
    return texts, labels, vocab, sentiment2idx


def load_sample_corpus() -> tuple[str, dict[str, int], dict[int, str]]:
    """Load the sample text-generation corpus."""
    data_path = _find_data_file("text_gen_samples/corpus.txt")

    if data_path:
        print(f"Loading corpus from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            corpus = f.read()
    else:
        print("Using hardcoded corpus (synthetic data not found)")
        corpus = "hello how are you today what is your name thank you very much"

    char_to_idx, idx_to_char = create_char_vocab(corpus)
    return corpus, char_to_idx, idx_to_char


def load_rag_knowledge_base() -> list[str]:
    """Load the sample RAG knowledge base."""
    data_path = _find_data_file("rag_samples/knowledge_base.json")

    if data_path:
        print(f"Loading knowledge base from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
    else:
        print("Using hardcoded knowledge base (synthetic data not found)")
        documents = [
            "MLX is an array framework for machine learning on Apple Silicon.",
            "The Unified Memory architecture allows CPU and GPU to share memory.",
            "LSTMs are recurrent neural networks capable of learning long-term dependencies.",
        ]

    return documents


# ============================================================================
# MODEL PERSISTENCE & EVALUATION
# ============================================================================

def save_model(model: nn.Module, path: str) -> None:
    """Save model weights to ``path`` (``model.save_weights`` handles the format)."""
    path = str(path)
    print(f"Saving model weights to {path}...")
    model.save_weights(path)
    print("✅ Model saved successfully.")


def load_model(model: nn.Module, path: str) -> nn.Module:
    """Load weights into ``model`` from ``path``. The architecture must match."""
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading model weights from {path}...")
    model.load_weights(path)
    print("✅ Model loaded successfully.")
    return model


def evaluate_model(
    model: nn.Module,
    X: mx.array,
    y: mx.array,
) -> tuple[float, list[int], list[int]]:
    """
    Evaluate ``model`` on ``(X, y)`` and return ``(accuracy, y_true, y_pred)``.

    The ``y_true`` / ``y_pred`` lists are convenient for downstream
    confusion-matrix plotting. The 3-D logits branch handles ``TextLSTM``
    by collapsing to the final timestep before argmax.
    """
    if hasattr(model, "eval"):
        model.eval()
    mx.eval(X, y)

    logits = model(X)
    if len(logits.shape) == 3 and len(y.shape) == 1:
        logits = logits[:, -1, :]

    predictions = mx.argmax(logits, axis=-1)
    accuracy = float(mx.mean(predictions == y))

    return accuracy, y.tolist(), predictions.tolist()