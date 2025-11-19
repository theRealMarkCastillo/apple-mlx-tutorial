#!/bin/bash

# MLX NLP Chatbot Demo - Quick Start Script

echo "=========================================="
echo "  MLX NLP Chatbot Demo"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please create one with: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import mlx.core; import mlx.nn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -q mlx numpy scikit-learn
fi

echo ""
echo "=========================================="
echo "  Ready to run!"
echo "=========================================="
echo ""
echo "Available commands:"
echo "  python main.py              - Interactive menu"
echo "  python main.py intent       - Intent classification"
echo "  python main.py sentiment    - Sentiment analysis"
echo "  python main.py text         - Text generation"
echo "  python main.py all          - Run all demos"
echo ""
echo "Or run individual scripts:"
echo "  python intent_classifier.py"
echo "  python sentiment_analysis.py"
echo "  python text_generator.py"
echo ""

# Ask user what to run
read -p "Run main menu now? (y/n): " choice
if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    python main.py
fi
