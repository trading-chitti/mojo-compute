# Benchmark Suite

This directory contains benchmark scripts for comparing Python baseline performance with Mojo implementations.

## Python Baseline Benchmarks

The `python_baseline.py` script benchmarks popular Python libraries for technical indicator calculations.

### Prerequisites

Install the required dependencies:

```bash
pip install pandas-ta memory-profiler
```

Note: TA-Lib installation may require additional system libraries. If TA-Lib is not available, the script will automatically fall back to using pandas_ta for MACD calculations.

#### Installing TA-Lib (Optional)

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Ubuntu/Debian:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

### Running the Benchmarks

From the repository root:

```bash
python3 benchmarks/python_baseline.py
```

This will:
1. Generate 10,000 data points (simulating ~40 years of daily trading data)
2. Benchmark the following indicators:
   - **SMA** (Simple Moving Average) - periods: 5, 10, 20, 50, 200
   - **RSI** (Relative Strength Index) - period: 14
   - **MACD** (Moving Average Convergence Divergence) - 12, 26, 9
   - **Bollinger Bands** - 20-day, 2 standard deviations
3. Run each benchmark 100 times and compute average and standard deviation
4. Measure memory usage for each indicator
5. Save results to `benchmarks/results/baseline.csv`

### Results Format

The output CSV file contains:
- `indicator`: Name of the technical indicator
- `period`: Period parameter(s) used
- `avg_time_ms`: Average execution time in milliseconds
- `std_time_ms`: Standard deviation of execution time
- `memory_mb`: Peak memory usage in megabytes

### Libraries Used

- **NumPy** - For SMA and Bollinger Bands calculations
- **pandas_ta** - For RSI calculations
- **TA-Lib** - For MACD calculations (with pandas_ta fallback)
- **memory_profiler** - For memory usage tracking

## Mojo Benchmarks

Mojo benchmark implementations will be added in future iterations to compare performance against the Python baseline.
