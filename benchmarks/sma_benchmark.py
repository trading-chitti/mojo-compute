#!/usr/bin/env python3
"""
Comprehensive SMA Benchmark: Mojo vs Python
Compares performance across multiple data sizes and implementations.
"""

import time
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class SMAbenchmark:
  """Benchmark SMA implementations across Mojo and Python."""

  def __init__(self, data_sizes: List[int] = None, periods: List[int] = None):
    """Initialize benchmark with data sizes and periods to test.

    Args:
      data_sizes: List of data point counts to test (default: [100, 1000, 10000])
      periods: List of SMA periods to test (default: [5, 20, 50])
    """
    self.data_sizes = data_sizes or [100, 1000, 10000]
    self.periods = periods or [5, 20, 50]
    self.results = []
    self.mojo_src_path = Path(__file__).parent.parent / "src"

  def generate_price_data(self, n_points: int, seed: int = 42) -> np.ndarray:
    """Generate realistic price data for benchmarking.

    Args:
      n_points: Number of data points to generate
      seed: Random seed for reproducibility

    Returns:
      numpy array of price data
    """
    np.random.seed(seed)
    returns = np.random.normal(0.0005, 0.02, n_points)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices

  def python_sma_numpy(self, prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate SMA using NumPy (optimized Python).

    Args:
      prices: Price data array
      period: SMA period

    Returns:
      Array of SMA values
    """
    sma = np.convolve(prices, np.ones(period) / period, mode='valid')
    result = np.empty(len(prices))
    result[:period-1] = np.nan
    result[period-1:] = sma
    return result

  def python_sma_pure(self, prices: List[float], period: int) -> List[float]:
    """Calculate SMA using pure Python (baseline).

    Args:
      prices: List of price values
      period: SMA period

    Returns:
      List of SMA values
    """
    n = len(prices)
    result = [0.0] * n

    if period <= 0 or period > n:
      return result

    # Calculate first SMA value
    sum_val = sum(prices[:period])
    result[period - 1] = sum_val / period

    # Calculate remaining SMA values using sliding window
    for i in range(period, n):
      sum_val = sum_val - prices[i - period] + prices[i]
      result[i] = sum_val / period

    return result

  def benchmark_python(self, prices: np.ndarray, period: int,
                      n_iterations: int = 100) -> Tuple[float, float]:
    """Benchmark Python NumPy implementation.

    Args:
      prices: Price data array
      period: SMA period
      n_iterations: Number of iterations to run

    Returns:
      Tuple of (avg_time_ms, std_time_ms)
    """
    times = []

    # Warmup
    _ = self.python_sma_numpy(prices, period)

    # Benchmark
    for _ in range(n_iterations):
      start = time.perf_counter()
      _ = self.python_sma_numpy(prices, period)
      end = time.perf_counter()
      times.append((end - start) * 1000)

    return np.mean(times), np.std(times)

  def benchmark_python_pure(self, prices: List[float], period: int,
                           n_iterations: int = 100) -> Tuple[float, float]:
    """Benchmark pure Python implementation.

    Args:
      prices: List of price values
      period: SMA period
      n_iterations: Number of iterations to run

    Returns:
      Tuple of (avg_time_ms, std_time_ms)
    """
    times = []

    # Warmup
    _ = self.python_sma_pure(prices, period)

    # Benchmark
    for _ in range(n_iterations):
      start = time.perf_counter()
      _ = self.python_sma_pure(prices, period)
      end = time.perf_counter()
      times.append((end - start) * 1000)

    return np.mean(times), np.std(times)

  def benchmark_mojo(self, prices: List[float], period: int,
                    n_iterations: int = 100) -> Tuple[float, float]:
    """Benchmark Mojo implementation via subprocess.

    Args:
      prices: List of price values
      period: SMA period
      n_iterations: Number of iterations to run

    Returns:
      Tuple of (avg_time_ms, std_time_ms)
    """
    # Create temporary Mojo benchmark script
    mojo_code = f'''
from time import perf_counter
from indicators import sma

fn main() raises:
  # Price data
  var prices = List[Float64]()
  {chr(10).join(f"  prices.append({p})" for p in prices)}

  var times = List[Float64]()
  var period = {period}
  var n_iterations = {n_iterations}

  # Warmup
  _ = sma(prices, period)

  # Benchmark
  for i in range(n_iterations):
    var start = perf_counter()
    _ = sma(prices, period)
    var end = perf_counter()
    times.append((end - start) * 1000)

  # Calculate statistics
  var total: Float64 = 0.0
  for i in range(len(times)):
    total += times[i]
  var mean = total / Float64(len(times))

  var variance: Float64 = 0.0
  for i in range(len(times)):
    var diff = times[i] - mean
    variance += diff * diff
  var std = (variance / Float64(len(times))) ** 0.5

  print(mean)
  print(std)
'''

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mojo', delete=False) as f:
      f.write(mojo_code)
      temp_path = f.name

    try:
      # Run Mojo benchmark
      result = subprocess.run(
        ['mojo', temp_path],
        cwd=self.mojo_src_path,
        capture_output=True,
        text=True,
        timeout=30
      )

      if result.returncode != 0:
        print(f"Mojo benchmark failed: {result.stderr}")
        return 0.0, 0.0

      # Parse output
      lines = result.stdout.strip().split('\n')
      avg_time = float(lines[-2])
      std_time = float(lines[-1])

      return avg_time, std_time

    except Exception as e:
      print(f"Error running Mojo benchmark: {e}")
      return 0.0, 0.0

    finally:
      # Cleanup
      Path(temp_path).unlink(missing_ok=True)

  def run_benchmarks(self, n_iterations: int = 100):
    """Run all benchmarks across data sizes and periods.

    Args:
      n_iterations: Number of iterations per benchmark
    """
    print("=" * 80)
    print("SMA BENCHMARK: Mojo vs Python")
    print("=" * 80)
    print(f"Data sizes: {self.data_sizes}")
    print(f"Periods: {self.periods}")
    print(f"Iterations per test: {n_iterations}")
    print("=" * 80)
    print()

    for data_size in self.data_sizes:
      print(f"\n{'=' * 80}")
      print(f"Testing with {data_size} data points")
      print('=' * 80)

      # Generate data
      prices_np = self.generate_price_data(data_size)
      prices_list = prices_np.tolist()

      for period in self.periods:
        print(f"\nPeriod: {period}")
        print("-" * 40)

        # Benchmark Python NumPy
        print("  Benchmarking Python (NumPy)...", end=" ", flush=True)
        py_numpy_avg, py_numpy_std = self.benchmark_python(
          prices_np, period, n_iterations
        )
        print(f"{py_numpy_avg:.4f}ms ± {py_numpy_std:.4f}ms")

        # Benchmark Pure Python
        print("  Benchmarking Python (Pure)...", end=" ", flush=True)
        py_pure_avg, py_pure_std = self.benchmark_python_pure(
          prices_list, period, n_iterations
        )
        print(f"{py_pure_avg:.4f}ms ± {py_pure_std:.4f}ms")

        # Benchmark Mojo
        print("  Benchmarking Mojo...", end=" ", flush=True)
        mojo_avg, mojo_std = self.benchmark_mojo(
          prices_list, period, n_iterations
        )
        print(f"{mojo_avg:.4f}ms ± {mojo_std:.4f}ms")

        # Calculate speedup
        speedup_numpy = py_numpy_avg / mojo_avg if mojo_avg > 0 else 0
        speedup_pure = py_pure_avg / mojo_avg if mojo_avg > 0 else 0

        print(f"\n  Speedup vs NumPy: {speedup_numpy:.2f}x")
        print(f"  Speedup vs Pure Python: {speedup_pure:.2f}x")

        # Store results
        self.results.append({
          'data_size': data_size,
          'period': period,
          'python_numpy_avg_ms': py_numpy_avg,
          'python_numpy_std_ms': py_numpy_std,
          'python_pure_avg_ms': py_pure_avg,
          'python_pure_std_ms': py_pure_std,
          'mojo_avg_ms': mojo_avg,
          'mojo_std_ms': mojo_std,
          'speedup_vs_numpy': speedup_numpy,
          'speedup_vs_pure': speedup_pure
        })

  def save_results(self, output_path: Path):
    """Save benchmark results to JSON file.

    Args:
      output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
      json.dump({
        'benchmark': 'SMA',
        'data_sizes': self.data_sizes,
        'periods': self.periods,
        'results': self.results
      }, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")

  def generate_charts(self, output_dir: Path):
    """Generate performance comparison charts.

    Args:
      output_dir: Directory to save charts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: Performance by Data Size
    fig, axes = plt.subplots(1, len(self.periods), figsize=(15, 5))
    if len(self.periods) == 1:
      axes = [axes]

    for idx, period in enumerate(self.periods):
      ax = axes[idx]

      # Filter results for this period
      period_results = [r for r in self.results if r['period'] == period]
      data_sizes = [r['data_size'] for r in period_results]

      py_numpy_times = [r['python_numpy_avg_ms'] for r in period_results]
      py_pure_times = [r['python_pure_avg_ms'] for r in period_results]
      mojo_times = [r['mojo_avg_ms'] for r in period_results]

      x = np.arange(len(data_sizes))
      width = 0.25

      ax.bar(x - width, py_numpy_times, width, label='Python (NumPy)', color='#3776ab')
      ax.bar(x, py_pure_times, width, label='Python (Pure)', color='#ffd43b')
      ax.bar(x + width, mojo_times, width, label='Mojo', color='#ff4500')

      ax.set_xlabel('Data Size')
      ax.set_ylabel('Time (ms)')
      ax.set_title(f'SMA Performance - Period {period}')
      ax.set_xticks(x)
      ax.set_xticklabels(data_sizes)
      ax.legend()
      ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sma_performance_by_size.png', dpi=300)
    print(f"✅ Chart saved: {output_dir / 'sma_performance_by_size.png'}")
    plt.close()

    # Chart 2: Speedup Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(self.results))
    labels = [f"{r['data_size']}pts\nP{r['period']}" for r in self.results]

    speedup_numpy = [r['speedup_vs_numpy'] for r in self.results]
    speedup_pure = [r['speedup_vs_pure'] for r in self.results]

    width = 0.35
    ax.bar(x - width/2, speedup_numpy, width, label='vs NumPy', color='#2ecc71')
    ax.bar(x + width/2, speedup_pure, width, label='vs Pure Python', color='#e74c3c')

    ax.set_xlabel('Test Configuration')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Mojo Speedup over Python Implementations')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'sma_speedup_comparison.png', dpi=300)
    print(f"✅ Chart saved: {output_dir / 'sma_speedup_comparison.png'}")
    plt.close()

  def print_summary(self):
    """Print summary of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Calculate overall statistics
    avg_speedup_numpy = np.mean([r['speedup_vs_numpy'] for r in self.results])
    avg_speedup_pure = np.mean([r['speedup_vs_pure'] for r in self.results])
    max_speedup_numpy = max([r['speedup_vs_numpy'] for r in self.results])
    max_speedup_pure = max([r['speedup_vs_pure'] for r in self.results])

    print(f"\nAverage Speedup vs NumPy: {avg_speedup_numpy:.2f}x")
    print(f"Maximum Speedup vs NumPy: {max_speedup_numpy:.2f}x")
    print(f"\nAverage Speedup vs Pure Python: {avg_speedup_pure:.2f}x")
    print(f"Maximum Speedup vs Pure Python: {max_speedup_pure:.2f}x")

    print("\n" + "=" * 80)


def main():
  """Main entry point for SMA benchmark."""
  # Configuration
  data_sizes = [100, 1000, 10000]
  periods = [5, 20, 50]
  n_iterations = 100

  # Output paths
  results_dir = Path(__file__).parent / "results"
  output_json = results_dir / "sma_benchmark.json"
  charts_dir = results_dir / "charts"

  # Run benchmark
  benchmark = SMAbenchmark(data_sizes=data_sizes, periods=periods)
  benchmark.run_benchmarks(n_iterations=n_iterations)

  # Save results and generate charts
  benchmark.save_results(output_json)
  benchmark.generate_charts(charts_dir)
  benchmark.print_summary()

  return 0


if __name__ == "__main__":
  sys.exit(main())
