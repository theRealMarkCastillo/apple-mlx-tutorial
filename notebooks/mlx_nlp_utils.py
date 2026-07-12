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


def create_vocabulary(texts: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """Build a word vocabulary with reserved ``<PAD>=0`` and ``<UNK>=1``.

    Returns ``(vocab, word_to_idx)`` — both dicts share the same entries but are
    returned separately to mirror the symmetric vocabulary / index mapping used
    by ``texts_to_sequences`` and the pipeline notebooks. ``vocab`` documents
    what was learned; ``word_to_idx`` is the lookup used at train/inference.
    """
    vocab = {"<PAD>": _PAD_IDX, "<UNK>": _UNK_IDX}
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = len(vocab)
    # vocab and word_to_idx map the same keys to the same indices, so a
    # shallow dict copy is the right semantics here.
    return vocab, dict(vocab)


def preprocess_text(text: str) -> list[str]:
    """Lowercase text and strip common punctuation, returning whitespace tokens."""
    text = text.lower()
    for char in ".,!?;:":
        text = text.replace(char, " ")
    return text.split()


def train_val_split(
    items: list,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list, list]:
    """Shuffle ``items`` deterministically and split into (train, val).

    A fixed seed makes notebook output reproducible. ``val_fraction`` is the
    proportion held out for validation; the rest goes to training.

    >>> train, val = train_val_split([0, 1, 2, 3, 4], val_fraction=0.4, seed=0)
    >>> sorted(train), sorted(val)
    ([0, 2, 3], [1, 4])
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(items))
    rng.shuffle(indices)
    cut = int(round(len(items) * (1.0 - val_fraction)))
    train_idx = indices[:cut].tolist()
    val_idx = indices[cut:].tolist()
    return [items[i] for i in train_idx], [items[i] for i in val_idx]


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


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Reference implementation of scaled dot-product attention.

    Computes ``softmax(QK^T / sqrt(d_k)) @ V``. The optional ``mask`` is an
    *additive* mask (0 for unmasked positions, a very large negative for
    masked positions); ``None`` means "no masking". This is the canonical
    reference used by notebooks 06 and 06b — both import it from here so
    each notebook does not have to redefine it.

    Args:
        query: (..., seq_len_q, d_k)
        key:   (..., seq_len_k, d_k)
        value: (..., seq_len_v, d_v)  ``seq_len_v == seq_len_k``
        mask:  (..., seq_len_q, seq_len_k) additive mask, optional

    Returns:
        (output, attention_weights)
    """
    d_k = query.shape[-1]
    scores = mx.matmul(query, mx.transpose(key, (0, 2, 1)))
    scores = scores / np.sqrt(d_k)
    if mask is not None:
        scores = scores + (mask * -1e9)
    attn_weights = mx.softmax(scores, axis=-1)
    output = mx.matmul(attn_weights, value)
    return output, attn_weights


def train_model(
    model: nn.Module,
    X: mx.array,
    y: mx.array,
    epochs: int = 50,
    learning_rate: float = 0.01,
    X_val: mx.array | None = None,
    y_val: mx.array | None = None,
) -> tuple[nn.Module, dict[str, list]]:
    """
    Generic training loop for MLX models.

    Uses Adam and the lazy-evaluation pattern; ``mx.eval`` is called every step
    to force the parameter and optimizer-state updates to materialize. The
    forward pass automatically adapts: ``TextLSTM`` emits
    ``(batch, seq, vocab)`` and that branch is collapsed to ``(batch, vocab)``
    on the last timestep when targets are 1-D class indices.

    Args:
        X, y: training arrays. ``y`` may be class indices (``int32`` of shape
            ``(N,)``) or, for the text-generation branch, float logits targets
            (this loop always uses cross-entropy, so pass class indices).
        X_val, y_val: optional held-out arrays. When both are provided,
            ``val_loss`` and ``val_accuracy`` are appended to ``history`` each
            epoch so notebooks can plot a real generalization curve.

    Returns:
        The trained model and a history dict containing ``loss`` / ``accuracy``
        lists, plus ``val_loss`` / ``val_accuracy`` when validation data is
        supplied.
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

    history: dict[str, list] = {"loss": [], "accuracy": []}
    has_val = X_val is not None and y_val is not None
    if has_val:
        history["val_loss"] = []
        history["val_accuracy"] = []
        # Materialise the held-out arrays up front so per-step ``mx.eval`` only
        # touches the small validation graph.
        mx.eval(X_val, y_val)

    def _forward_metrics(X_in, y_in):
        out = model(X_in)
        if len(out.shape) == 3 and len(y_in.shape) == 1:
            out = out[:, -1, :]
        preds = mx.argmax(out, axis=-1)
        acc = mx.mean(preds == y_in)
        loss = mx.mean(nn.losses.cross_entropy(out, y_in))
        return float(loss), float(acc)

    for epoch in range(epochs):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)

        # MLX is lazy: build the graph, but only materialize when needed.
        # ``mx.eval`` forces the parameter and optimizer-state updates to land
        # in memory before we read metrics below.
        mx.eval(model.parameters(), optimizer.state)

        train_loss, train_acc = _forward_metrics(X, y)
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)

        val_loss = 0.0
        val_acc = 0.0
        if has_val:
            # Evaluate at inference (dropout off, etc.); restore training
            # mode afterwards so the next optimiser step still updates.
            if hasattr(model, "eval"):
                model.eval()
            val_loss, val_acc = _forward_metrics(X_val, y_val)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            if hasattr(model, "train"):
                model.train()

        if (epoch + 1) % 10 == 0:
            msg = (
                f"Epoch {epoch + 1:3d}/{epochs} - "
                f"Loss: {train_loss:.4f} - Accuracy: {train_acc:.4f}"
            )
            if has_val:
                msg += f" - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}"
            print(msg)

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
    """LSTM-based sentiment analyzer with dropout.

    ``nn.Dropout`` already checks ``self.training`` internally, so passing the
    LSTM output through it is enough to disable dropout at inference time
    (which ``predict_*`` helpers trigger via ``model.eval()``).
    """

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
    window: int = 5,
) -> str:
    """Sample ``length`` characters of text continuation from ``seed``.

    The model is unrolled one character at a time, each step conditioned on the
    last ``window`` characters seen so far. If the model was trained with a
    different context length, pass the matching ``window`` here.

    Args:
        window: number of previous characters fed to the model on each step.
                Must match the sequence length used by ``text_to_sequences``
                when the model was trained. Defaults to 5 (the value used in
                notebooks 00 / 03).
    """
    if hasattr(model, "eval"):
        model.eval()
    if not char_to_idx:
        raise ValueError("char_to_idx is empty — call create_char_vocab() first.")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

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
    """Load sample intent classification data from ``data/intent_samples/data.json``.

    Returns ``(texts, labels, vocab, intent2idx)``. ``vocab`` maps token -> id
    and ``intent2idx`` maps label string -> class index.
    """
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

    vocab, _word_to_idx = create_vocabulary(texts)
    intent2idx = {"greeting": 0, "question": 1, "command": 2}
    return texts, labels, vocab, intent2idx


def load_sample_sentiment_data() -> tuple[list[str], list[str], dict[str, int], dict[str, int]]:
    """Load sample sentiment data from ``data/sentiment_samples/data.json``.

    Returns ``(texts, labels, vocab, sentiment2idx)``.
    """
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

    vocab, _word_to_idx = create_vocabulary(texts)
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