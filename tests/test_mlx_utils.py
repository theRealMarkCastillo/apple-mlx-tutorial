import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from notebooks.mlx_nlp_utils import (
    IntentLSTM, SentimentLSTM, TextLSTM, create_char_vocab, generate_text,
    make_train_step, scaled_dot_product_attention, train_model, save_model,
    load_model, train_val_split,
)


@pytest.mark.parametrize('cls', [IntentLSTM, SentimentLSTM])
def test_padding_does_not_change_classifier(cls):
    model = cls(12, 8, 12, 3)
    model.eval()
    short = model(mx.array([[2, 3]]))
    padded = model(mx.array([[2, 3, 0, 0, 0]]))
    np.testing.assert_allclose(np.array(short), np.array(padded), atol=1e-6)
    assert bool(mx.all(mx.isfinite(model(mx.zeros((2, 4), dtype=mx.int32)))))


@pytest.mark.parametrize('heads', [False, True])
@pytest.mark.parametrize('boolean', [False, True])
def test_reference_mask_matches_fused_attention(heads, boolean):
    mx.random.seed(3)
    shape = (2, 3, 4, 8) if heads else (2, 4, 8)
    q, k, v = [mx.random.normal(shape) for _ in range(3)]
    allowed = mx.tril(mx.ones((4, 4), dtype=mx.bool_))
    mask = allowed if boolean else mx.where(allowed, 0.0, -float('inf'))
    output, weights = scaled_dot_product_attention(q, k, v, mask)
    args = (q, k, v) if heads else (q[:, None], k[:, None], v[:, None])
    fast = mx.fast.scaled_dot_product_attention(*args, scale=8 ** -0.5, mask=mask)
    if not heads:
        fast = fast[:, 0]
    np.testing.assert_allclose(np.array(output), np.array(fast), atol=1e-5)
    np.testing.assert_allclose(np.array(weights.sum(-1)), 1, atol=1e-6)
    assert bool(mx.all(mx.where(allowed, 0, weights) == 0))


def test_compiled_dropout_updates_match_eager_and_advance_rng():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.4)
            self.linear = nn.Linear(8, 2)
        def __call__(self, x):
            return self.linear(self.dropout(x))
    def run(compiled):
        mx.random.seed(7)
        model = Model()
        model.train()
        optimizer = optim.Adam(0.01)
        step, state = make_train_step(model, optimizer,
            lambda m, x, y: nn.losses.cross_entropy(m(x), y, reduction='mean'),
            compile_step=compiled)
        mx.eval(state)
        keys = []
        for _ in range(3):
            loss = step(mx.ones((5, 8)), mx.array([0, 1, 0, 1, 0]))
            mx.eval(loss, state)
            keys.append(np.array(mx.random.state[0]))
        return [np.array(v) for _, v in tree_flatten(model.parameters())], keys
    eager, _ = run(False)
    compiled, keys = run(True)
    for a, b in zip(eager, compiled):
        np.testing.assert_allclose(a, b, atol=1e-5)
    assert not np.array_equal(keys[0], keys[1])
    assert not np.array_equal(keys[1], keys[2])


def test_partial_batch_is_trained_and_metrics_use_eval_mode():
    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self._batches = []
        def __call__(self, x):
            self._batches.append((self.training, len(x)))
            return self.linear(x)
    model = Recorder()
    X, y = mx.ones((5, 2)), mx.array([0, 1, 0, 1, 0])
    _, history = train_model(model, X, y, epochs=1, batch_size=2,
                             compile_step=False, X_val=X, y_val=y)
    assert [size for training, size in model._batches if training] == [2, 2, 1]
    assert [size for training, size in model._batches if not training] == [2, 2, 1] * 2
    assert not model.training
    assert len(history['val_loss']) == 1
    with pytest.raises(ValueError, match='both'):
        train_model(model, X, y, X_val=X)


def test_training_reduces_loss_and_weights_round_trip(tmp_path):
    mx.random.seed(2)
    model = IntentLSTM(8, 4, 8, 2)
    X, y = mx.array([[2, 0], [3, 0], [2, 0], [3, 0]]), mx.array([0, 1, 0, 1])
    _, history = train_model(model, X, y, epochs=8, learning_rate=0.05, batch_size=3)
    assert history['loss'][-1] < history['loss'][0]
    path = tmp_path / 'model.safetensors'
    save_model(model, str(path))
    restored = load_model(IntentLSTM(8, 4, 8, 2), str(path))
    np.testing.assert_allclose(np.array(model(X)), np.array(restored(X)), atol=1e-6)


def test_generation_suppresses_special_tokens_and_supports_greedy():
    vocab, inverse = create_char_vocab('ab')
    class Fixed(nn.Module):
        def __call__(self, x):
            return mx.broadcast_to(mx.array([1000., 999., 1., 2.]), (*x.shape, 4))
    assert generate_text(Fixed(), 'a', vocab, inverse, length=5, temperature=0) == 'abbbbb'
    with pytest.raises(ValueError):
        generate_text(Fixed(), 'a', vocab, inverse, temperature=-1)


def test_sequence_targets_train():
    model = TextLSTM(8, 4, 6)
    _, history = train_model(model, mx.array([[2, 3], [3, 4]]),
                            mx.array([[3, 4], [4, 5]]), epochs=1)
    assert np.isfinite(history['loss'][0])


def test_split_never_silently_returns_empty_partitions():
    assert all(train_val_split([1, 2], val_fraction=0.01))
    with pytest.raises(ValueError):
        train_val_split([1])


def notebook_definitions(filename, names, namespace):
    """Load the actual teaching definitions without training/plotting cells."""
    import ast
    import json
    from pathlib import Path
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'notebooks' / filename).read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace


def test_transformer_padding_and_fixed_positions():
    ns = notebook_definitions('05_Transformer_Classifier.ipynb',
                              {'sinusoidal_positional_encoding', 'TransformerClassifier'},
                              {'mx': mx, 'nn': nn})
    model = ns['TransformerClassifier'](12, d_model=8, num_heads=2, num_layers=1)
    model.eval()
    a = model(mx.array([[2, 3]]))
    b = model(mx.array([[2, 3, 0, 0]]))
    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-5)
    assert bool(mx.all(mx.isfinite(model(mx.zeros((1, 4), dtype=mx.int32)))))
    assert not any('pos_encoding' in key for key, _ in tree_flatten(model.trainable_parameters()))
    assert ns['sinusoidal_positional_encoding'](5, 7).shape == (5, 7)


def test_nanogpt_cannot_attend_to_future_and_batches_are_shifted():
    ns = notebook_definitions('07_Build_NanoGPT.ipynb',
        {'MultiHeadAttention', 'FeedForward', 'Block', 'GPT', 'get_batch'}, {'mx': mx, 'nn': nn})
    model = ns['GPT'](8, n_layer=1, n_head=2, n_embd=8, block_size=4)
    model.eval()
    a = model(mx.array([[1, 2, 3, 4]]))
    b = model(mx.array([[1, 2, 6, 7]]))
    np.testing.assert_allclose(np.array(a[:, :2]), np.array(b[:, :2]), atol=1e-5)
    x, y = ns['get_batch'](mx.arange(40), block_size=5, batch_size=3)
    np.testing.assert_array_equal(np.array(x + 1), np.array(y))
