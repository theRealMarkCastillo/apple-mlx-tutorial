#!/usr/bin/env bash
# Dataset Downloader (Shell Wrapper)
# A convenient wrapper around download_datasets.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/download_datasets.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: download_datasets.py not found at $PYTHON_SCRIPT" >&2
    exit 1
fi

if ! python3 -c "import datasets" &>/dev/null; then
    echo "WARNING: 'datasets' library not installed." >&2
    echo "Install it with: pip install datasets" >&2
fi

echo "📦 Running dataset downloader..."
python3 "$PYTHON_SCRIPT" "$@"
