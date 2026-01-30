#!/bin/bash
# Script to install dependencies and run Python baseline benchmarks

set -e

echo "Installing required Python packages..."
pip3 install --user pandas-ta memory-profiler

echo ""
echo "Attempting to install TA-Lib (optional)..."
if command -v brew &> /dev/null; then
    echo "Homebrew detected, installing TA-Lib via brew..."
    brew install ta-lib || echo "Warning: TA-Lib installation failed, will use pandas_ta fallback"
    pip3 install --user TA-Lib || echo "Warning: TA-Lib Python bindings installation failed"
else
    echo "Homebrew not found, skipping TA-Lib installation"
    echo "TA-Lib is optional. The script will use pandas_ta for MACD if TA-Lib is not available."
fi

echo ""
echo "Running Python baseline benchmarks..."
python3 benchmarks/python_baseline.py

echo ""
echo "Benchmark complete! Results saved to benchmarks/results/baseline.csv"
