#!/bin/bash
# Dataset Download Script (Shell version)
# Usage: ./scripts/download_datasets.sh [--all|--samples|--imdb|--snips]

set -e

DATA_DIR="data"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  DATASET DOWNLOADER (Shell)"
echo "=========================================="

# Check if Python script exists
if [ ! -f "scripts/download_datasets.py" ]; then
    echo -e "${RED}ERROR: Python script not found${NC}"
    echo "Please run this from the project root directory"
    exit 1
fi

# Check if datasets library is installed
python3 -c "import datasets" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing datasets library...${NC}"
    pip install datasets
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}WARNING: Virtual environment not activated${NC}"
    echo "Consider running: source .venv/bin/activate"
    echo ""
fi

# Parse arguments or use defaults
if [ $# -eq 0 ]; then
    echo "No arguments provided. Showing help..."
    echo ""
    python3 scripts/download_datasets.py --help
    exit 0
fi

# Run Python downloader with arguments
echo -e "${GREEN}Starting download...${NC}"
echo ""

python3 scripts/download_datasets.py "$@"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Download complete!${NC}"
    echo ""
    echo "Datasets saved to: ${DATA_DIR}/"
    
    # List downloaded datasets
    if [ -d "$DATA_DIR" ]; then
        echo ""
        echo "Downloaded datasets:"
        for dir in "$DATA_DIR"/*; do
            if [ -d "$dir" ]; then
                count=$(find "$dir" -type f | wc -l | tr -d ' ')
                echo "  • $(basename "$dir")/ (${count} files)"
            fi
        done
    fi
    
    echo ""
    echo "Next steps:"
    echo "  1. Start notebooks: cd notebooks && jupyter notebook"
    echo "  2. Open 00_Overview.ipynb to get started"
    echo "  3. See notebooks/README.md for learning paths"
else
    echo -e "${RED}✗ Download failed${NC}"
    exit 1
fi
