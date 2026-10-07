#!/usr/bin/env bash
# Run the downloader in the repository's uv-managed environment from any cwd.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required. See https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

exec uv run --locked --project "$PROJECT_DIR" python "$SCRIPT_DIR/download_datasets.py" "$@"
