"""Small, inspectable helpers for model provenance and honest evaluation."""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_RESERVED_BYTES = 0


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_spec(key):
    spec = json.loads((ROOT / "config/models.json").read_text())[key]
    if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
        raise ValueError("Models must be pinned to a full commit revision")
    return spec


def model_path(key):
    """Resolve a pinned snapshot, reserving uncached bytes from a run budget.

    Budget is GiB of model files, not HTTP overhead. A runner supplies a shared
    ledger across notebook kernels. A failed download keeps its reservation.
    Runs are sequential; this ledger is not a concurrent download coordinator.
    """
    from huggingface_hub import snapshot_download

    spec = model_spec(key)
    options = dict(
        **spec,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        ignore_patterns=["onnx/*", "openvino/*", "*.onnx*", "*.h5"],
    )
    plan = snapshot_download(**options, dry_run=True)
    needed = sum(item.file_size for item in plan if item.will_download)
    budget = float(os.environ.get("MLX_TUTORIAL_DOWNLOAD_BUDGET_GB", "0"))
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("Download budget must be finite and nonnegative")
    global _RESERVED_BYTES
    ledger_name = os.environ.get("MLX_TUTORIAL_DOWNLOAD_LEDGER")
    ledger = Path(ledger_name) if ledger_name else None
    used = (
        (json.loads(ledger.read_text())["reserved_bytes"] if ledger.exists() else 0)
        if ledger is not None
        else _RESERVED_BYTES
    )
    if needed and used + needed > budget * 1024**3:
        raise RuntimeError(
            f"{key} needs {needed / 1024**3:.2f} GiB; remaining model download budget "
            f"is {max(0, budget - used / 1024**3):.2f} GiB. Set "
            "MLX_TUTORIAL_DOWNLOAD_BUDGET_GB or the runner's --download-budget-gb."
        )
    if needed:
        _RESERVED_BYTES = used + needed
        if ledger is not None:
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text(json.dumps({"reserved_bytes": used + needed}))
    return snapshot_download(**options)


def write_record(name, *, settings, metrics, models=(), data_files=(), seed=0):
    """Persist a JSON record beside rendered output, without serializing secrets."""
    output = Path(os.environ.get("MLX_TUTORIAL_RESULTS_DIR", ROOT / "results"))
    output.mkdir(parents=True, exist_ok=True)
    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        for label, key in [
            ("chip", "machdep.cpu.brand_string"),
            ("memory_bytes", "hw.memsize"),
        ]:
            result = subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True
            )
            hardware[label] = result.stdout.strip()
    packages = {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
    }
    record = {
        "schema_version": 1,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "hardware": hardware,
        "packages": packages,
        "models": {key: model_spec(key) for key in models},
        "data_sha256": {str(p): sha256_file(p) for p in data_files},
        "settings": settings,
        "metrics": metrics,
    }
    if (ROOT / "uv.lock").exists():
        record["uv_lock_sha256"] = sha256_file(ROOT / "uv.lock")
    path = output / f"{name}.json"
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return path


def score_label(reply, gold, labels):
    """Separate recoverable first-label accuracy from exact output compliance."""
    normalized = reply.strip().lower()
    first = re.split(r"[\s.:,!?]+", normalized)[0]
    return {
        "correct": first == gold,
        "format_valid": normalized in labels,
        "strict_correct": normalized == gold,
    }


def accuracy_interval(correct, confidence=0.95):
    """Wilson interval for binary accuracy; remains informative at 0/n and n/n."""
    values = np.asarray(correct, dtype=float)
    if not values.size or not np.isin(values, [0, 1]).all() or not 0 < confidence < 1:
        raise ValueError("Need binary outcomes and confidence between zero and one")
    n, p = values.size, float(values.mean())
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return p, max(0.0, center - radius), min(1.0, center + radius)


def choose_threshold(positive, negative):
    """Maximize balanced accuracy on calibration data only; ties favor refusal."""
    positive, negative = np.asarray(positive), np.asarray(negative)
    if not positive.size or not negative.size:
        raise ValueError("Both calibration classes must be nonempty")
    values = np.concatenate([positive, negative])
    if not np.isfinite(values).all():
        raise ValueError("Scores must be finite")
    thresholds = np.r_[
        np.nextafter(values.min(), -np.inf),
        np.unique(values),
        np.nextafter(values.max(), np.inf),
    ]
    return float(
        max(
            thresholds,
            key=lambda t: (np.mean(positive >= t) + np.mean(negative < t), t),
        )
    )


def refusal_metrics(positive, negative, threshold):
    return {
        "covered_accept_rate": float(np.mean(np.asarray(positive) >= threshold)),
        "uncovered_accept_rate": float(np.mean(np.asarray(negative) >= threshold)),
    }


def rrf(rankings, k=60):
    """Fuse complete ranked document-ID lists without mixing score scales."""
    rankings = np.asarray(rankings)
    if rankings.ndim != 2 or k < 0:
        raise ValueError("Expected ranked lists and nonnegative k")
    n = rankings.shape[1]
    scores = np.zeros(n)
    for ranking in rankings:
        if sorted(ranking.tolist()) != list(range(n)):
            raise ValueError("Each ranking must be a permutation of document IDs")
        scores[ranking] += 1 / (k + np.arange(1, n + 1))
    return np.argsort(-scores, kind="stable")


def retrieval_metrics(rankings, relevant, k=3):
    """Support multiple relevant chunks per query, including chunk-size changes."""
    recalls, reciprocal, ndcg = [], [], []
    for ranking, gold in zip(rankings, relevant, strict=True):
        gold = set(gold)
        if not gold:
            raise ValueError("Each query needs at least one relevant document")
        hits = [int(doc in gold) for doc in ranking]
        recalls.append(sum(hits[:k]) / len(gold))
        reciprocal.append(1 / (hits.index(1) + 1) if 1 in hits else 0)
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits[:k]))
        ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(gold))))
        ndcg.append(dcg / ideal)
    return {
        f"recall@{k}": float(np.mean(recalls)),
        "MRR": float(np.mean(reciprocal)),
        f"nDCG@{k}": float(np.mean(ndcg)),
    }


def citation_metrics(cited_ids, retrieved_ids, relevant_ids):
    """Structural validity and gold-source support; not semantic entailment."""
    citations = set(cited_ids)
    return {
        "valid_ids": bool(citations <= set(retrieved_ids)),
        "gold_source_cited": bool(citations & set(relevant_ids)),
        "has_citation": bool(citations),
    }
