import json
from pathlib import Path

import nbformat
import numpy as np
import pytest

from notebooks.experiment_utils import (
    choose_threshold,
    refusal_metrics,
    score_label,
    rrf,
    retrieval_metrics,
    citation_metrics,
    write_record,
    model_path,
)
from scripts.check_notebooks import select_notebooks, validate_source


def test_label_correctness_does_not_hide_format_failure():
    labels = ["greeting", "question", "command"]
    assert score_label("greeting because hello", "greeting", labels) == {
        "correct": True,
        "format_valid": False,
        "strict_correct": False,
    }
    assert score_label(" Greeting\n", "greeting", labels)["strict_correct"]
    assert not score_label("greetings", "greeting", labels)["correct"]


def test_calibration_freezes_threshold_before_unseen_negatives():
    threshold = choose_threshold([0.8, 0.9], [0.1, 0.2])
    metrics = refusal_metrics([0.85], [0.95], threshold)
    assert metrics == {"covered_accept_rate": 1.0, "uncovered_accept_rate": 1.0}
    with pytest.raises(ValueError):
        choose_threshold([], [0.2])


def test_rank_fusion_and_multi_source_relevance():
    fused = rrf([[0, 1, 2], [1, 0, 2]])
    assert set(fused[:2]) == {0, 1}
    metrics = retrieval_metrics([[2, 0, 1]], [{0, 2}], k=2)
    assert metrics["recall@2"] == metrics["MRR"] == metrics["nDCG@2"] == 1
    with pytest.raises(ValueError):
        rrf([[0, 0, 2]])
    assert not citation_metrics([999], [1, 2], [2])["valid_ids"]
    assert not citation_metrics([1], [1, 2], [2])["gold_source_cited"]


def test_notebook_hygiene_and_missing_selection():
    path = Path("example.ipynb")
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 1")])
    validate_source(notebook, path)
    notebook.cells[0].execution_count = 1
    with pytest.raises(ValueError, match="clear outputs"):
        validate_source(notebook, path)
    notebook.cells[0].execution_count = None
    notebook.cells[0].source = "secret = '" + "hf_" + "x" * 35 + "'"
    with pytest.raises(ValueError, match="credential"):
        validate_source(notebook, path)
    with pytest.raises(ValueError, match="No notebooks"):
        select_notebooks([path], "does-not-exist")


def test_record_captures_data_and_model_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_TUTORIAL_RESULTS_DIR", str(tmp_path))
    data = tmp_path / "input.txt"
    data.write_text("example")
    path = write_record(
        "test",
        settings={"temperature": 0},
        metrics={"accuracy": 0.5},
        models=["minilm"],
        data_files=[data],
    )
    record = json.loads(path.read_text())
    assert len(record["data_sha256"][str(data)]) == 64
    assert len(record["models"]["minilm"]["revision"]) == 40
    assert record["settings"]["temperature"] == 0


def test_download_budget_blocks_before_download(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["dry_run"] is True
        return [SimpleNamespace(file_size=1024, will_download=True)]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setenv("MLX_TUTORIAL_DOWNLOAD_BUDGET_GB", "0")
    monkeypatch.setenv("MLX_TUTORIAL_DOWNLOAD_LEDGER", str(tmp_path / "ledger.json"))
    with pytest.raises(RuntimeError, match="budget"):
        model_path("minilm")
    assert len(calls) == 1


def test_decoder_cache_matches_full_forward():
    import mlx.core as mx
    from notebooks.modern_decoder import Decoder, cache_bytes

    for rope in [False, True]:
        mx.random.seed(0)
        model = Decoder(32, dims=32, heads=4, kv_heads=2, rope=rope)
        tokens = mx.array([[1, 2, 3, 4, 5]])
        full = model(tokens)
        first, cache = model(tokens[:, :2], return_cache=True)
        rest, cache = model(tokens[:, 2:], cache=cache, return_cache=True)
        np.testing.assert_allclose(
            np.array(full), np.array(mx.concatenate([first, rest], axis=1)), atol=2e-5
        )
        assert cache_bytes(cache) == 2 * 2 * 1 * 5 * 2 * 8 * 4
        # Future tokens cannot change earlier logits.
        changed = model(mx.array([[1, 2, 3, 9, 9]]))
        np.testing.assert_allclose(
            np.array(full[:, :3]), np.array(changed[:, :3]), atol=1e-6
        )


def test_perfect_small_sample_does_not_imply_certain_accuracy():
    from notebooks.experiment_utils import accuracy_interval

    accuracy, low, high = accuracy_interval([True] * 12)
    assert accuracy == 1 and 0.7 < low < 0.8 and high == 1
    accuracy, low, high = accuracy_interval([False] * 12)
    assert accuracy == 0 and low == 0 and high > 0.2


def test_download_budget_is_shared_and_cached_files_are_free(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub

    downloaded = []
    cached = False

    def download(**kwargs):
        if kwargs.get("dry_run"):
            return [SimpleNamespace(file_size=800, will_download=not cached)]
        downloaded.append(kwargs["repo_id"])
        return "/cached/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setenv("MLX_TUTORIAL_DOWNLOAD_BUDGET_GB", str(1000 / 1024**3))
    monkeypatch.setenv("MLX_TUTORIAL_DOWNLOAD_LEDGER", str(tmp_path / "ledger.json"))
    assert model_path("minilm") == "/cached/snapshot"
    with pytest.raises(RuntimeError, match="budget"):
        model_path("qwen_embedding")
    cached = True
    monkeypatch.setenv("MLX_TUTORIAL_DOWNLOAD_BUDGET_GB", "0")
    assert model_path("minilm") == "/cached/snapshot"
    assert len(downloaded) == 2
