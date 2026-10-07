#!/usr/bin/env python3
"""Validate notebooks; optionally execute offline lessons.

``--execute --quick`` reduces training epochs/steps and sets
``MLX_TUTORIAL_QUICK`` so ``sanity_check`` cells warn instead of failing.
``--execute`` alone uses the full training budgets and enforces those checks.
Executed copies, plots, and weights go into a temporary workspace; source
notebooks stay unexecuted unless ``--output-dir`` asks for executed copies.
Notebooks 08 and 10 download pretrained models; they run only with --include-manual.
"""
import argparse
import ast
import os
import shutil
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
TRAINERS = {"train_model", "mlx_train_model", "train_transformer", "train_gpt"}
# Assignments to these names are teaching budgets (training steps, etc.).
QUICK_BUDGETS = {"train_steps": 2, "rank_sweep_iters": 2}
MANUAL = ("08_", "10_")


class QuickTraining(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        name = getattr(node.func, "id", "")
        if name in TRAINERS:
            for keyword in node.keywords:
                if keyword.arg == "epochs":
                    keyword.value = ast.Constant(1)
                if keyword.arg == "steps":
                    keyword.value = ast.Constant(2)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in QUICK_BUDGETS:
                node.value = ast.Constant(QUICK_BUDGETS[target.id])
        return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--notebook", help="Run one notebook by stem prefix, e.g. 03")
    parser.add_argument("--include-manual", action="store_true",
                        help="Also execute 08 and 10 (downloads pretrained models)")
    parser.add_argument("--output-dir", type=Path,
                        help="Save executed notebooks here (e.g. rendered/)")
    args = parser.parse_args()
    if args.quick:
        os.environ["MLX_TUTORIAL_QUICK"] = "1"
    else:
        os.environ.pop("MLX_TUTORIAL_QUICK", None)
    with tempfile.TemporaryDirectory(prefix="mlx-notebooks-") as tmp:
        work = Path(tmp)
        shutil.copytree(ROOT / "data", work / "data", ignore=shutil.ignore_patterns(
            "imdb", "snips", "banking77", "wikitext"))
        (work / "notebooks").mkdir()
        shutil.copy(ROOT / "notebooks/mlx_nlp_utils.py", work / "notebooks")
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            if args.notebook and not path.stem.startswith(args.notebook):
                continue
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    ast.parse(cell.source, filename=str(path))
            print(f"Validated {path.name}", flush=True)
            if not args.execute or (path.name.startswith(MANUAL) and not args.include_manual):
                continue
            if args.quick:
                for cell in notebook.cells:
                    if cell.cell_type != "code":
                        continue
                    tree = QuickTraining().visit(ast.parse(cell.source))
                    cell.source = ast.unparse(ast.fix_missing_locations(tree))
            setup = nbformat.v4.new_code_cell(
                "import plotly.io as pio\npio.renderers.default = 'json'\n"
                "import mlx.core as mx\nmx.random.seed(0)")
            notebook.cells.insert(0, setup)
            NotebookClient(notebook, timeout=3600, kernel_name="python3", resources={
                "metadata": {"path": str(work / "notebooks")},
            }).execute()
            print(f"Executed {path.name}{' (quick)' if args.quick else ''}", flush=True)
            if args.output_dir:
                notebook.cells.remove(setup)
                args.output_dir.mkdir(parents=True, exist_ok=True)
                nbformat.write(notebook, args.output_dir / path.name)


if __name__ == "__main__":
    main()
