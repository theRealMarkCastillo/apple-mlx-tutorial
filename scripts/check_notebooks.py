#!/usr/bin/env python3
"""Validate notebooks; optionally execute offline lessons with --execute --quick.

Quick mode reduces training epochs/steps only. Executed copies and all plots and
weights go into a temporary workspace; source notebooks stay unexecuted.
Notebooks 08 and 10 require model downloads and are intentionally opt-in/manual.
"""
import argparse
import ast
import shutil
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent


class QuickTraining(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        name = getattr(node.func, "id", "")
        if name in {"train_model", "mlx_train_model", "train_transformer"}:
            for keyword in node.keywords:
                if keyword.arg == "epochs":
                    keyword.value = ast.Constant(1)
        return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--notebook", help="Run one notebook by stem prefix, e.g. 03")
    args = parser.parse_args()
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
            if not args.execute or path.name.startswith(("08_", "10_")):
                continue
            if args.quick:
                for cell in notebook.cells:
                    if cell.cell_type != "code":
                        continue
                    tree = QuickTraining().visit(ast.parse(cell.source))
                    cell.source = ast.unparse(ast.fix_missing_locations(tree))
                    if path.name.startswith("07_") and "for step in range(100)" in cell.source:
                        cell.source = cell.source.replace("for step in range(100)", "for step in range(2)")
            notebook.cells.insert(0, nbformat.v4.new_code_cell(
                "import os\nos.environ['MPLBACKEND'] = 'Agg'\n"
                "import plotly.io as pio\npio.renderers.default = 'json'\n"
                "import mlx.core as mx\nmx.random.seed(0)"))
            NotebookClient(notebook, timeout=600, kernel_name="python3", resources={
                "metadata": {"path": str(work / "notebooks")},
            }).execute()
            print(f"Executed {path.name}{' (quick)' if args.quick else ''}", flush=True)


if __name__ == "__main__":
    main()
