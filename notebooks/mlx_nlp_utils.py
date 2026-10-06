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
    return mx.metal.is_available()


def set_device(device_type: str = "gpu") -> None:
    """
    Set the default MLX device.

    Args:
        device_type: 'gpu' or 'cpu'. Defaults to 'gpu' if available, else 'cpu'.
    """
    device_type = (device_type or "gpu").lower()
    if device_type not in {"gpu", "cpu"}:
        raise ValueError("device_type must be gpu or cpu")
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
        last_output = last_non_padding(lstm_out, x)
        logits = self.linear(last_output)
        return logits


def last_non_padding(hidden: mx.array, tokens: mx.array) -> mx.array:
    """Select the last real timestep of right-padded token sequences.

    Empty strings have an all-zero representation instead of a PAD embedding.
    """
    lengths = mx.sum(tokens != _PAD_IDX, axis=1)
    indices = mx.maximum(lengths - 1, 0).astype(mx.int32)
    selected = hidden[mx.arange(tokens.shape[0]), indices]
    return mx.where(lengths[:, None] > 0, selected, mx.zeros_like(selected))


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
    if len(items) < 2:
        raise ValueError("At least two items are required for a train/validation split")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(items))
    rng.shuffle(indices)
    cut = min(len(items) - 1, max(1, round(len(items) * (1.0 - val_fraction))))
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
    if max_len < 1:
        raise ValueError("max_len must be positive")
    padded = np.full((len(sequences), max_len), _PAD_IDX, dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        if length > 0:
            padded[i, :length] = seq[:length]
    return padded


def scaled_dot_product_attention(query, key, value, mask=None):
    """Reference attention returning (output, weights) for notebook heatmaps.

    Inputs have shape (..., sequence, features). As in MLX fast attention,
    boolean masks use True for *allowed* positions; floating masks are added
    to scores (zero for visible, -inf for blocked). Use mx.fast SDPA when
    attention weights are not needed, to avoid materializing the score matrix.
    """
    scores = (query @ mx.swapaxes(key, -1, -2)) * query.shape[-1] ** -0.5
    if mask is not None:
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, -float("inf"))
        else:
            scores = scores + mask
    attn_weights = mx.softmax(scores, axis=-1, precise=True)
    return attn_weights @ value, attn_weights


def make_train_step(model, optimizer, loss_fn, *, compile_step=True, max_grad_norm=1.0):
    """Capture model, optimizer AND random state in a compiled update.

    Call with model.train() enabled. Evaluate the returned loss and state after
    each update to bound the lazy graph. max_grad_norm=None disables clipping.
    """
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive or None")
    optimizer.init(model.trainable_parameters())
    state = [model.state, optimizer.state, mx.random.state]
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(X, y):
        loss, grads = loss_and_grad(model, X, y)
        if max_grad_norm is not None:
            grads, _ = optim.clip_grad_norm(grads, max_grad_norm)
        optimizer.update(model, grads)
        return loss

    return (mx.compile(step, inputs=state, outputs=state) if compile_step else step), state


def train_model(
    model: nn.Module,
    X: mx.array,
    y: mx.array,
    epochs: int = 50,
    learning_rate: float = 0.01,
    X_val: mx.array | None = None,
    y_val: mx.array | None = None,
    *,
    batch_size: int = 32,
    seed: int = 0,
    compile_step: bool = True,
    max_grad_norm: float | None = 1.0,
) -> tuple[nn.Module, dict[str, list]]:
    """Mini-batch Adam training with compiled updates and held-out metrics.

    The final partial batch is included. Both training and validation metrics
    are evaluated with dropout disabled and weighted by target count. For
    next-character classification, 3-D logits use their final timestep when
    targets are 1-D; sequence targets retain all timesteps. Supply both
    validation arrays or neither. Returns (model in eval mode, history).
    """
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if len(X) == 0 or len(X) != len(y):
        raise ValueError("Training arrays must be nonempty and have equal lengths")
    if (X_val is None) != (y_val is None):
        raise ValueError("Supply both X_val and y_val")
    has_val = X_val is not None
    if has_val and (len(X_val) == 0 or len(X_val) != len(y_val)):
        raise ValueError("Validation arrays must be nonempty and have equal lengths")

    def logits_for(X_in, y_in):
        logits = model(X_in)
        return logits[:, -1, :] if logits.ndim == 3 and y_in.ndim == 1 else logits

    def loss_fn(model, X_batch, y_batch):
        return nn.losses.cross_entropy(logits_for(X_batch, y_batch), y_batch, reduction="mean")

    model.train()
    optimizer = optim.Adam(learning_rate=learning_rate)
    step, state = make_train_step(model, optimizer, loss_fn,
                                  compile_step=compile_step, max_grad_norm=max_grad_norm)
    mx.eval(state, X, y)
    rng = np.random.default_rng(seed)
    history = {"loss": [], "accuracy": []}
    if has_val:
        history.update(val_loss=[], val_accuracy=[])

    def metrics(X_in, y_in):
        loss_sum, correct, count = 0.0, 0, 0
        for start in range(0, len(X_in), batch_size):
            targets = y_in[start:start + batch_size]
            logits = logits_for(X_in[start:start + batch_size], targets)
            loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
            hits = mx.sum(mx.argmax(logits, axis=-1) == targets)
            mx.eval(loss, hits)
            loss_sum += loss.item()
            correct += hits.item()
            count += targets.size
        return loss_sum / count, correct / count

    for epoch in range(epochs):
        model.train()
        indices = mx.array(rng.permutation(len(X)), dtype=mx.int32)
        for start in range(0, len(X), batch_size):
            batch = indices[start:start + batch_size]
            loss = step(X[batch], y[batch])
            mx.eval(loss, state)
        model.eval()
        train_loss, train_acc = metrics(X, y)
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        if has_val:
            val_loss, val_acc = metrics(X_val, y_val)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
        if (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch + 1:3d}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {train_acc:.4f}"
            if has_val:
                msg += f" - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}"
            print(msg)
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
        last_output = last_non_padding(lstm_out, x)
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
    if seq_length < 1 or len(text) <= seq_length:
        raise ValueError("text must be longer than a positive seq_length")
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

    if temperature < 0 or length < 0:
        raise ValueError("temperature and length must be nonnegative")

    generated = seed
    current_seq = [char_to_idx.get(c, _UNK_IDX) for c in seed[-window:]]
    current_seq = ([_PAD_IDX] * (window - len(current_seq))) + current_seq

    for _ in range(length):
        X = mx.array([current_seq])
        logits = model(X)

        # Sample logits directly; special vocabulary entries are never text.
        logits = logits[0, -1, :]
        logits = mx.where(mx.arange(logits.size) < 2, -float("inf"), logits)
        next_token = mx.argmax(logits) if temperature == 0 else mx.random.categorical(logits / temperature)
        next_idx = next_token.item()  # Materializes this step before extending context.
        generated += idx_to_char[next_idx]
        current_seq = current_seq[1:] + [next_idx]

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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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