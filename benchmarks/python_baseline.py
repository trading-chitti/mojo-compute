#!/usr/bin/env python3
"""
Python Baseline Benchmarks for Technical Indicators

This script benchmarks popular Python libraries for technical indicator calculations
to establish a baseline for Mojo performance comparison.

Libraries used:
- NumPy for SMA and Bollinger Bands
- pandas_ta for RSI
- TA-Lib for MACD (fallback to pandas_ta if unavailable)
"""

import time
import csv
import sys
from pathlib import Path
from typing import Callable, Tuple, List
import tracemalloc

import numpy as np
import pandas as pd

# Import technical analysis libraries
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("Warning: pandas_ta not available")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available, will use pandas_ta for MACD")


def generate_price_data(n_points: int = 10000, seed: int = 42) -> np.ndarray:
    """
    Generate realistic price data for benchmarking.

    Args:
        n_points: Number of data points to generate (default: 10000 for ~40 years daily)
        seed: Random seed for reproducibility

    Returns:
        numpy array of price data
    """
    np.random.seed(seed)

    # Generate random walk with drift (realistic price movement)
    returns = np.random.normal(0.0005, 0.02, n_points)  # ~0.05% daily drift, 2% volatility
    prices = 100 * np.exp(np.cumsum(returns))  # Start at $100

    return prices


def numpy_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average using NumPy."""
    sma = np.convolve(prices, np.ones(period) / period, mode='valid')
    # Pad with NaN to match input length
    result = np.empty(len(prices))
    result[:period-1] = np.nan
    result[period-1:] = sma
    return result


def numpy_bollinger_bands(prices: np.ndarray, period: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands using NumPy."""
    # Calculate SMA for middle band
    middle = numpy_sma(prices, period)

    # Calculate standard deviation for each window
    std = np.empty(len(prices))
    std[:period-1] = np.nan

    for i in range(period-1, len(prices)):
        std[i] = np.std(prices[i-period+1:i+1])

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    return upper, middle, lower


def pandas_ta_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using pandas_ta."""
    if not PANDAS_TA_AVAILABLE:
        raise ImportError("pandas_ta is not available")

    # Convert to pandas Series
    series = pd.Series(prices)
    rsi = ta.rsi(series, length=period)
    return rsi.values


def talib_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate MACD using TA-Lib."""
    if not TALIB_AVAILABLE:
        raise ImportError("TA-Lib is not available")

    macd, signal_line, histogram = talib.MACD(prices, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    return macd, signal_line, histogram


def pandas_ta_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate MACD using pandas_ta (fallback)."""
    if not PANDAS_TA_AVAILABLE:
        raise ImportError("pandas_ta is not available")

    series = pd.Series(prices)
    macd_result = ta.macd(series, fast=fast, slow=slow, signal=signal)

    # pandas_ta returns a DataFrame with MACD_fast_slow_signal, MACDh_fast_slow_signal, MACDs_fast_slow_signal
    macd = macd_result[f'MACD_{fast}_{slow}_{signal}'].values
    signal_line = macd_result[f'MACDs_{fast}_{slow}_{signal}'].values
    histogram = macd_result[f'MACDh_{fast}_{slow}_{signal}'].values

    return macd, signal_line, histogram


def benchmark_function(func: Callable, *args, n_iterations: int = 100) -> Tuple[float, float, float]:
    """
    Benchmark a function by running it multiple times and measuring performance.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to the function
        n_iterations: Number of times to run the function

    Returns:
        Tuple of (avg_time_ms, std_time_ms, memory_mb)
    """
    times = []

    # Warmup run
    _ = func(*args)

    # Memory measurement
    tracemalloc.start()
    _ = func(*args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak / 1024 / 1024  # Convert to MB

    # Time measurements
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    avg_time = np.mean(times)
    std_time = np.std(times)

    return avg_time, std_time, memory_mb


def run_benchmarks(prices: np.ndarray, n_iterations: int = 100) -> List[dict]:
    """
    Run all benchmarks and return results.

    Args:
        prices: Price data to use for benchmarks
        n_iterations: Number of iterations for each benchmark

    Returns:
        List of result dictionaries
    """
    results = []

    print(f"Running benchmarks with {len(prices)} data points, {n_iterations} iterations each...\n")

    # SMA Benchmarks
    sma_periods = [5, 10, 20, 50, 200]
    for period in sma_periods:
        print(f"Benchmarking SMA (period={period})...")
        avg_time, std_time, memory_mb = benchmark_function(numpy_sma, prices, period, n_iterations=n_iterations)
        results.append({
            'indicator': 'SMA',
            'period': period,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'memory_mb': memory_mb
        })
        print(f"  Avg: {avg_time:.4f}ms, Std: {std_time:.4f}ms, Memory: {memory_mb:.4f}MB\n")

    # RSI Benchmark
    if PANDAS_TA_AVAILABLE:
        print("Benchmarking RSI (period=14)...")
        avg_time, std_time, memory_mb = benchmark_function(pandas_ta_rsi, prices, 14, n_iterations=n_iterations)
        results.append({
            'indicator': 'RSI',
            'period': 14,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'memory_mb': memory_mb
        })
        print(f"  Avg: {avg_time:.4f}ms, Std: {std_time:.4f}ms, Memory: {memory_mb:.4f}MB\n")
    else:
        print("Skipping RSI (pandas_ta not available)\n")

    # MACD Benchmark
    macd_func = talib_macd if TALIB_AVAILABLE else pandas_ta_macd
    macd_lib = "TA-Lib" if TALIB_AVAILABLE else "pandas_ta"

    if TALIB_AVAILABLE or PANDAS_TA_AVAILABLE:
        print(f"Benchmarking MACD using {macd_lib} (12, 26, 9)...")
        avg_time, std_time, memory_mb = benchmark_function(macd_func, prices, 12, 26, 9, n_iterations=n_iterations)
        results.append({
            'indicator': f'MACD_{macd_lib}',
            'period': '12,26,9',
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'memory_mb': memory_mb
        })
        print(f"  Avg: {avg_time:.4f}ms, Std: {std_time:.4f}ms, Memory: {memory_mb:.4f}MB\n")
    else:
        print("Skipping MACD (neither TA-Lib nor pandas_ta available)\n")

    # Bollinger Bands Benchmark
    print("Benchmarking Bollinger Bands (20, 2.0)...")
    avg_time, std_time, memory_mb = benchmark_function(numpy_bollinger_bands, prices, 20, 2.0, n_iterations=n_iterations)
    results.append({
        'indicator': 'Bollinger_Bands',
        'period': '20,2.0',
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'memory_mb': memory_mb
    })
    print(f"  Avg: {avg_time:.4f}ms, Std: {std_time:.4f}ms, Memory: {memory_mb:.4f}MB\n")

    return results


def save_results(results: List[dict], output_file: Path):
    """
    Save benchmark results to CSV file.

    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['indicator', 'period', 'avg_time_ms', 'std_time_ms', 'memory_mb']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {output_file}")


def print_summary(results: List[dict]):
    """Print a summary of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Indicator':<20} {'Period':<15} {'Avg Time (ms)':<15} {'Std Time (ms)':<15} {'Memory (MB)':<15}")
    print("-"*80)

    for result in results:
        print(f"{result['indicator']:<20} {str(result['period']):<15} "
              f"{result['avg_time_ms']:<15.4f} {result['std_time_ms']:<15.4f} "
              f"{result['memory_mb']:<15.4f}")

    print("="*80)


def main():
    """Main entry point for the benchmark script."""
    # Configuration
    N_POINTS = 10000  # ~40 years of daily data
    N_ITERATIONS = 100
    OUTPUT_FILE = Path(__file__).parent / "results" / "baseline.csv"

    # Generate price data
    print("Generating price data...")
    prices = generate_price_data(N_POINTS)
    print(f"Generated {len(prices)} price points\n")

    # Run benchmarks
    results = run_benchmarks(prices, n_iterations=N_ITERATIONS)

    # Save and display results
    save_results(results, OUTPUT_FILE)
    print_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
