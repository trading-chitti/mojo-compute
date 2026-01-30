#!/usr/bin/env python3
"""
Comprehensive Benchmark: SMA, EMA, RSI - Mojo vs Python
Tests all three indicators with multiple data sizes.
"""

import time
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ComprehensiveBenchmark:
  """Benchmark technical indicators across Mojo and Python."""

  def __init__(self, data_sizes: List[int] = None):
    """Initialize benchmark with data sizes to test.

    Args:
      data_sizes: List of data point counts to test
    """
    self.data_sizes = data_sizes or [100, 1000, 10000]
    self.results = {
      'sma': [],
      'ema': [],
      'rsi': []
    }
    self.mojo_src_path = Path(__file__).parent.parent / "src"

  def generate_price_data(self, n_points: int, seed: int = 42) -> np.ndarray:
    """Generate realistic price data for benchmarking.

    Args:
      n_points: Number of data points
      seed: Random seed

    Returns:
      numpy array of prices
    """
    np.random.seed(seed)
    returns = np.random.normal(0.0005, 0.02, n_points)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices

  # ========== Python Implementations ==========

  def python_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate SMA using NumPy."""
    sma = np.convolve(prices, np.ones(period) / period, mode='valid')
    result = np.empty(len(prices))
    result[:period-1] = 0.0
    result[period-1:] = sma
    return result

  def python_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA using NumPy."""
    multiplier = 2.0 / (period + 1)
    ema = np.zeros(len(prices))

    # Start with SMA
    ema[period-1] = np.mean(prices[:period])

    # Calculate EMA
    for i in range(period, len(prices)):
      ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]

    return ema

  def python_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using NumPy."""
    n = len(prices)
    rsi = np.zeros(n)

    if period >= n:
      return rsi

    # Calculate price changes
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
      rsi[period] = 100.0
    else:
      rs = avg_gain / avg_loss
      rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Smoothed RSI
    for i in range(period + 1, n):
      avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
      avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

      if avg_loss == 0:
        rsi[i] = 100.0
      else:
        rs = avg_gain / avg_loss
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi

  # ========== Benchmark Functions ==========

  def benchmark_python_indicator(self, func, prices: np.ndarray, *args,
                                 n_iterations: int = 100) -> Tuple[float, float]:
    """Benchmark a Python indicator function.

    Args:
      func: Function to benchmark
      prices: Price data
      *args: Additional arguments for the function
      n_iterations: Number of iterations

    Returns:
      Tuple of (avg_time_ms, std_time_ms)
    """
    times = []

    # Warmup
    _ = func(prices, *args)

    # Benchmark
    for _ in range(n_iterations):
      start = time.perf_counter()
      _ = func(prices, *args)
      end = time.perf_counter()
      times.append((end - start) * 1000)

    return np.mean(times), np.std(times)

  def benchmark_mojo_indicator(self, indicator: str, prices: List[float],
                               period: int, n_iterations: int = 100) -> Tuple[float, float]:
    """Benchmark a Mojo indicator via subprocess.

    Args:
      indicator: Indicator name ('sma', 'ema', 'rsi')
      prices: List of prices
      period: Period parameter
      n_iterations: Number of iterations

    Returns:
      Tuple of (avg_time_ms, std_time_ms)
    """
    # Create Mojo benchmark script
    mojo_code = f'''
from time import perf_counter
from indicators import {indicator}

fn main() raises:
  var prices = List[Float64]()
  {chr(10).join(f"  prices.append({p})" for p in prices)}

  var times = List[Float64]()
  var period = {period}
  var n_iterations = {n_iterations}

  # Warmup
  _ = {indicator}(prices, period)

  # Benchmark
  for i in range(n_iterations):
    var start = perf_counter()
    _ = {indicator}(prices, period)
    var end = perf_counter()
    times.append((end - start) * 1000)

  # Statistics
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

    with tempfile.NamedTemporaryFile(mode='w', suffix='.mojo', delete=False) as f:
      f.write(mojo_code)
      temp_path = f.name

    try:
      result = subprocess.run(
        ['mojo', temp_path],
        cwd=self.mojo_src_path,
        capture_output=True,
        text=True,
        timeout=30
      )

      if result.returncode != 0:
        print(f"  Mojo {indicator.upper()} failed: {result.stderr}")
        return 0.0, 0.0

      lines = result.stdout.strip().split('\n')
      avg_time = float(lines[-2])
      std_time = float(lines[-1])

      return avg_time, std_time

    except Exception as e:
      print(f"  Error running Mojo {indicator.upper()}: {e}")
      return 0.0, 0.0

    finally:
      Path(temp_path).unlink(missing_ok=True)

  # ========== Main Benchmark Execution ==========

  def run_benchmarks(self, n_iterations: int = 100):
    """Run all benchmarks for all indicators.

    Args:
      n_iterations: Number of iterations per benchmark
    """
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK: SMA, EMA, RSI - Mojo vs Python")
    print("=" * 80)
    print(f"Data sizes: {self.data_sizes}")
    print(f"Iterations per test: {n_iterations}")
    print("=" * 80)
    print()

    # Test configurations
    configs = [
      ('sma', self.python_sma, 20),
      ('ema', self.python_ema, 12),
      ('rsi', self.python_rsi, 14)
    ]

    for data_size in self.data_sizes:
      print(f"\n{'=' * 80}")
      print(f"DATA SIZE: {data_size} points")
      print('=' * 80)

      # Generate data
      prices_np = self.generate_price_data(data_size)
      prices_list = prices_np.tolist()

      for indicator_name, python_func, period in configs:
        print(f"\n{indicator_name.upper()} (period={period})")
        print("-" * 40)

        # Benchmark Python
        print("  Python (NumPy)...", end=" ", flush=True)
        py_avg, py_std = self.benchmark_python_indicator(
          python_func, prices_np, period, n_iterations=n_iterations
        )
        print(f"{py_avg:.4f}ms ± {py_std:.4f}ms")

        # Benchmark Mojo
        print("  Mojo...", end=" ", flush=True)
        mojo_avg, mojo_std = self.benchmark_mojo_indicator(
          indicator_name, prices_list, period, n_iterations=n_iterations
        )
        print(f"{mojo_avg:.4f}ms ± {mojo_std:.4f}ms")

        # Calculate speedup
        speedup = py_avg / mojo_avg if mojo_avg > 0 else 0
        print(f"  Speedup: {speedup:.2f}x")

        # Store results
        self.results[indicator_name].append({
          'data_size': data_size,
          'period': period,
          'python_avg_ms': py_avg,
          'python_std_ms': py_std,
          'mojo_avg_ms': mojo_avg,
          'mojo_std_ms': mojo_std,
          'speedup': speedup
        })

  def save_results(self, output_path: Path):
    """Save all results to JSON.

    Args:
      output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
      json.dump({
        'benchmark': 'Comprehensive',
        'indicators': ['SMA', 'EMA', 'RSI'],
        'data_sizes': self.data_sizes,
        'results': self.results
      }, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")

  def generate_charts(self, output_dir: Path):
    """Generate comprehensive comparison charts.

    Args:
      output_dir: Directory to save charts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: Performance by Indicator and Data Size
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    indicators = ['sma', 'ema', 'rsi']
    titles = ['SMA', 'EMA', 'RSI']

    for idx, (indicator, title) in enumerate(zip(indicators, titles)):
      ax = axes[idx]
      results = self.results[indicator]

      data_sizes = [r['data_size'] for r in results]
      py_times = [r['python_avg_ms'] for r in results]
      mojo_times = [r['mojo_avg_ms'] for r in results]

      x = np.arange(len(data_sizes))
      width = 0.35

      ax.bar(x - width/2, py_times, width, label='Python', color='#3776ab')
      ax.bar(x + width/2, mojo_times, width, label='Mojo', color='#ff4500')

      ax.set_xlabel('Data Size')
      ax.set_ylabel('Time (ms)')
      ax.set_title(f'{title} Performance')
      ax.set_xticks(x)
      ax.set_xticklabels(data_sizes)
      ax.legend()
      ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300)
    print(f"✅ Chart saved: {output_dir / 'performance_comparison.png'}")
    plt.close()

    # Chart 2: Speedup Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    all_speedups = []
    all_labels = []

    for indicator in indicators:
      for result in self.results[indicator]:
        all_speedups.append(result['speedup'])
        all_labels.append(
          f"{indicator.upper()}\n{result['data_size']}pts"
        )

    x = np.arange(len(all_speedups))
    colors = ['#e74c3c', '#3498db', '#2ecc71'] * len(self.data_sizes)

    ax.bar(x, all_speedups, color=colors, alpha=0.8)
    ax.set_xlabel('Indicator & Data Size')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Mojo Speedup over Python NumPy')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
      Patch(facecolor='#e74c3c', alpha=0.8, label='SMA'),
      Patch(facecolor='#3498db', alpha=0.8, label='EMA'),
      Patch(facecolor='#2ecc71', alpha=0.8, label='RSI')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_comparison.png', dpi=300)
    print(f"✅ Chart saved: {output_dir / 'speedup_comparison.png'}")
    plt.close()

    # Chart 3: Scaling Analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    for indicator, color, label in zip(
      indicators,
      ['#e74c3c', '#3498db', '#2ecc71'],
      ['SMA', 'EMA', 'RSI']
    ):
      results = self.results[indicator]
      data_sizes = [r['data_size'] for r in results]
      speedups = [r['speedup'] for r in results]

      ax.plot(data_sizes, speedups, marker='o', linewidth=2,
             markersize=8, color=color, label=label)

    ax.set_xlabel('Data Size (points)', fontsize=12)
    ax.set_ylabel('Speedup (x)', fontsize=12)
    ax.set_title('Mojo Speedup Scaling with Data Size', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_analysis.png', dpi=300)
    print(f"✅ Chart saved: {output_dir / 'scaling_analysis.png'}")
    plt.close()

  def print_summary(self):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for indicator in ['sma', 'ema', 'rsi']:
      results = self.results[indicator]
      speedups = [r['speedup'] for r in results]

      avg_speedup = np.mean(speedups)
      max_speedup = max(speedups)
      min_speedup = min(speedups)

      print(f"\n{indicator.upper()}:")
      print(f"  Average Speedup: {avg_speedup:.2f}x")
      print(f"  Maximum Speedup: {max_speedup:.2f}x")
      print(f"  Minimum Speedup: {min_speedup:.2f}x")

    # Overall statistics
    all_speedups = []
    for indicator in ['sma', 'ema', 'rsi']:
      all_speedups.extend([r['speedup'] for r in self.results[indicator]])

    print(f"\nOVERALL:")
    print(f"  Average Speedup: {np.mean(all_speedups):.2f}x")
    print(f"  Maximum Speedup: {max(all_speedups):.2f}x")
    print(f"  Minimum Speedup: {min(all_speedups):.2f}x")

    print("\n" + "=" * 80)


def main():
  """Main entry point."""
  # Configuration
  data_sizes = [100, 1000, 10000]
  n_iterations = 100

  # Output paths
  results_dir = Path(__file__).parent / "results"
  output_json = results_dir / "comprehensive_benchmark.json"
  charts_dir = results_dir / "charts"

  # Run benchmark
  benchmark = ComprehensiveBenchmark(data_sizes=data_sizes)
  benchmark.run_benchmarks(n_iterations=n_iterations)

  # Save and visualize
  benchmark.save_results(output_json)
  benchmark.generate_charts(charts_dir)
  benchmark.print_summary()

  return 0


if __name__ == "__main__":
  sys.exit(main())
