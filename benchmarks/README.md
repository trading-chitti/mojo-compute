# Mojo Technical Indicators Benchmarks

Comprehensive performance benchmarks comparing Mojo-optimized technical indicators against Python implementations.

## Overview

This benchmark suite measures the performance of three technical indicators:
- **SMA** (Simple Moving Average)
- **EMA** (Exponential Moving Average)
- **RSI** (Relative Strength Index)

Implementations are tested across **three data sizes** (100, 1000, 10000 points) to analyze scaling characteristics.

## Target Performance

| Implementation | Target Speedup | Expected Time (10k points) |
|---------------|----------------|----------------------------|
| Python (Pure) | 1x (baseline)  | ~100ms                     |
| Python (NumPy)| 100x           | ~1ms                       |
| Mojo (Scalar) | 100x           | ~1ms                       |
| Mojo (SIMD)   | **1000x**      | **~0.1ms**                 |

## Benchmark Scripts

### 1. SMA Benchmark (`sma_benchmark.py`)

Focused benchmark for Simple Moving Average:
- Tests SMA with periods: 5, 20, 50
- Compares Python (NumPy), Python (Pure), and Mojo
- Generates detailed performance charts

**Usage:**
```bash
cd /Users/hariprasath/trading-chitti/mojo-compute/benchmarks
python sma_benchmark.py
```

**Output:**
- `results/sma_benchmark.json` - Raw benchmark data
- `results/charts/sma_performance_by_size.png` - Performance comparison chart
- `results/charts/sma_speedup_comparison.png` - Speedup visualization

### 2. Comprehensive Benchmark (`comprehensive_benchmark.py`)

Full benchmark suite for all indicators:
- Tests SMA, EMA, and RSI
- Multiple data sizes (100, 1000, 10000)
- Scaling analysis

**Usage:**
```bash
cd /Users/hariprasath/trading-chitti/mojo-compute/benchmarks
python comprehensive_benchmark.py
```

**Output:**
- `results/comprehensive_benchmark.json` - Complete results
- `results/charts/performance_comparison.png` - Multi-indicator comparison
- `results/charts/speedup_comparison.png` - Speedup across all tests
- `results/charts/scaling_analysis.png` - Performance scaling with data size

### 3. Python Baseline (`python_baseline.py`)

Establishes Python baseline using popular libraries:
- NumPy for SMA
- pandas_ta for RSI
- TA-Lib for MACD (if available)

**Usage:**
```bash
cd /Users/hariprasath/trading-chitti/mojo-compute/benchmarks
python python_baseline.py
```

## Dependencies

### Python Requirements

```bash
pip install numpy pandas pandas_ta matplotlib prometheus_client
```

### Optional (for TA-Lib):
```bash
# macOS
brew install ta-lib
pip install TA-Lib

# Linux
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

### Mojo Requirements

- Mojo 24.5.0 or later
- AVX2/AVX-512 CPU support for SIMD optimizations

## Running Benchmarks

### Quick Start

```bash
# Run all benchmarks
cd /Users/hariprasath/trading-chitti/mojo-compute/benchmarks
./run_benchmark.sh
```

### Individual Benchmarks

```bash
# SMA only
python sma_benchmark.py

# All indicators
python comprehensive_benchmark.py

# Python baseline
python python_baseline.py
```

## Understanding Results

### Benchmark Metrics

1. **Average Time (ms)**: Mean execution time across iterations
2. **Standard Deviation (ms)**: Variability in execution time
3. **Speedup**: Performance improvement factor (e.g., 1000x = 1000 times faster)
4. **Memory (MB)**: Peak memory usage during computation

### Sample Output

```
================================================================================
SMA BENCHMARK: Mojo vs Python
================================================================================
Data sizes: [100, 1000, 10000]
Periods: [5, 20, 50]
Iterations per test: 100
================================================================================

================================================================================
Testing with 10000 data points
================================================================================

Period: 20
----------------------------------------
  Python (NumPy)... 1.2345ms ± 0.0234ms
  Python (Pure)... 125.6789ms ± 2.3456ms
  Mojo... 0.0012ms ± 0.0001ms

  Speedup vs NumPy: 1028.75x
  Speedup vs Pure Python: 104732.50x
```

## Performance Charts

### Chart 1: Performance by Data Size

Compares execution time across implementations for different data sizes.

### Chart 2: Speedup Comparison

Shows speedup factor of Mojo vs Python implementations.

### Chart 3: Scaling Analysis

Demonstrates how speedup changes with increasing data size.

## SIMD Optimization Details

See [SIMD_OPTIMIZATION.md](SIMD_OPTIMIZATION.md) for detailed documentation on:
- SIMD implementation strategies
- Performance tuning techniques
- Expected speedup targets
- Limitations and trade-offs

## Results Directory Structure

```
benchmarks/
├── results/
│   ├── sma_benchmark.json              # SMA benchmark results
│   ├── comprehensive_benchmark.json    # All indicators results
│   ├── baseline.csv                    # Python baseline results
│   └── charts/
│       ├── sma_performance_by_size.png
│       ├── sma_speedup_comparison.png
│       ├── performance_comparison.png
│       ├── speedup_comparison.png
│       └── scaling_analysis.png
```

## Reproducibility

All benchmarks use fixed random seeds for reproducibility:

```python
np.random.seed(42)  # Consistent across all runs
```

## Hardware Considerations

### SIMD Width Detection

```bash
# Check SIMD support
mojo run src/indicators_simd.mojo
# Output shows: "SIMD Width: 4" (AVX2) or "8" (AVX-512)
```

### Expected Speedup by CPU

| CPU Feature | SIMD Width | Expected Speedup |
|-------------|------------|------------------|
| SSE         | 2          | 500x             |
| AVX2        | 4          | 1000x            |
| AVX-512     | 8          | 1500x+           |

---

**Last Updated**: 2026-01-30
**Mojo Version**: 24.5.0+
**Python Version**: 3.9+
