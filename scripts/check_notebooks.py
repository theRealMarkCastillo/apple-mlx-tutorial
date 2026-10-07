#!/usr/bin/env python3
"""Validate clean notebook sources and optionally execute lessons in isolation.

Quick runs reduce teaching budgets and relax narrative sanity checks. Full runs
keep those budgets. Downloading lessons require --include-manual; optional
comparisons require --advanced. Model downloads have an explicit shared budget.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import platform
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
TRAINERS = {"train_model", "mlx_train_model", "train_transformer", "train_gpt"}
QUICK_BUDGETS = {"train_steps": 2, "rank_sweep_iters": 2}
MANUAL = ("08_", "10_", "11_")
ADVANCED = {"RUN_DENSE", "COMPARE_QWEN"}
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]


def validate_source(notebook, path):
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if any(pattern.search(cell.source) for pattern in SECRET_PATTERNS):
            # Never print the matching value.
            raise ValueError(f"{path.name} cell {index}: possible credential in source")
        if cell.cell_type == "code":
            if cell.outputs or cell.execution_count is not None:
                raise ValueError(
                    f"{path.name} cell {index}: clear outputs and execution counts"
                )
            ast.parse(cell.source, filename=str(path))


def select_notebooks(paths, prefix=None):
    selected = [p for p in sorted(paths) if prefix is None or p.stem.startswith(prefix)]
    if not selected:
        raise ValueError(f"No notebooks match {prefix!r}")
    return selected


class QuickTraining(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if getattr(node.func, "id", "") in TRAINERS:
            for keyword in node.keywords:
                if keyword.arg in {"epochs", "steps"}:
                    keyword.value = ast.Constant(1 if keyword.arg == "epochs" else 2)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        for target in node.targets:
            if getattr(target, "id", "") in QUICK_BUDGETS:
                node.value = ast.Constant(QUICK_BUDGETS[target.id])
        return node


class AdvancedExperiments(ast.NodeTransformer):
    def visit_Assign(self, node):
        self.generic_visit(node)
        if any(getattr(target, "id", "") in ADVANCED for target in node.targets):
            node.value = ast.Constant(True)
        return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--notebook", help="Notebook stem prefix, e.g. 07b")
    parser.add_argument("--include-manual", action="store_true")
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Enable dense retrieval and Qwen embedding comparisons",
    )
    parser.add_argument(
        "--download-budget-gb",
        type=float,
        default=0,
        help="Maximum new model files in GiB (default: cache only)",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not 0 <= args.download_budget_gb < float("inf"):
        parser.error("Download budget must be finite and nonnegative")
    try:
        paths = select_notebooks((ROOT / "notebooks").glob("*.ipynb"), args.notebook)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.execute
        and args.notebook
        and not args.include_manual
        and all(p.name.startswith(MANUAL) for p in paths)
    ):
        parser.error(
            "Selected notebook needs --include-manual and optional dependencies"
        )
    for path in paths:
        validate_source(nbformat.read(path, as_version=4), path)
        print(f"Validated {path.name}", flush=True)
    if not args.execute:
        return
    if args.quick:
        os.environ["MLX_TUTORIAL_QUICK"] = "1"
    else:
        os.environ.pop("MLX_TUTORIAL_QUICK", None)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result_dir = args.results_dir.resolve() / run_id
    result_dir.mkdir(parents=True)
    os.environ["MLX_TUTORIAL_RESULTS_DIR"] = str(result_dir)
    os.environ["MLX_TUTORIAL_DOWNLOAD_LEDGER"] = str(
        result_dir / "download-budget.json"
    )
    os.environ["MLX_TUTORIAL_DOWNLOAD_BUDGET_GB"] = str(args.download_budget_gb)
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    manifest = {
        "schema_version": 1,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "models": json.loads((ROOT / "config/models.json").read_text()),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "uv.lock",
                *sorted((ROOT / "notebooks").glob("*.py")),
                *sorted((ROOT / "data").glob("*samples/*")),
            ]
            if p.is_file()
        },
        "seed": 0,
        "git_commit": git.stdout.strip(),
        "dirty": bool(dirty.stdout),
        "quick": args.quick,
        "advanced": args.advanced,
        "download_budget_gb": args.download_budget_gb,
        "notebooks": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="mlx-notebooks-") as tmp:
            work = Path(tmp)
            shutil.copytree(
                ROOT / "data",
                work / "data",
                ignore=shutil.ignore_patterns("imdb", "snips", "banking77", "wikitext"),
            )
            shutil.copytree(ROOT / "config", work / "config")
            shutil.copy(ROOT / "uv.lock", work / "uv.lock")
            (work / "notebooks").mkdir()
            for helper in (ROOT / "notebooks").glob("*.py"):
                shutil.copy(helper, work / "notebooks")
            for path in paths:
                entry = {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "status": "skipped",
                }
                manifest["notebooks"].append(entry)
                if path.name.startswith(MANUAL) and not args.include_manual:
                    continue
                notebook = nbformat.read(path, as_version=4)
                for cell in notebook.cells:
                    if cell.cell_type != "code" or not (args.quick or args.advanced):
                        continue
                    tree = ast.parse(cell.source)
                    if args.quick:
                        tree = QuickTraining().visit(tree)
                    if args.advanced:
                        tree = AdvancedExperiments().visit(tree)
                    cell.source = ast.unparse(ast.fix_missing_locations(tree))
                setup = nbformat.v4.new_code_cell(
                    "import plotly.io as pio\npio.renderers.default = 'json'\nimport mlx.core as mx\nmx.random.seed(0)"
                )
                notebook.cells.insert(0, setup)
                entry["status"] = "failed"
                try:
                    NotebookClient(
                        notebook,
                        timeout=3600,
                        kernel_name="python3",
                        resources={"metadata": {"path": str(work / "notebooks")}},
                    ).execute()
                    entry["status"] = "passed"
                finally:
                    if args.output_dir:
                        notebook.cells.remove(setup)
                        args.output_dir.mkdir(parents=True, exist_ok=True)
                        nbformat.write(notebook, args.output_dir / path.name)
                print(
                    f"Executed {path.name}{' (quick)' if args.quick else ''}",
                    flush=True,
                )
    finally:
        (result_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Run records: {result_dir}", flush=True)


if __name__ == "__main__":
    main()
