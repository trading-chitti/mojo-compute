"""
Prometheus Metrics Module for Mojo Compute Service

Tracks key performance metrics:
- request_count: Total number of requests
- request_duration: Request processing time
- indicator_compute_time: Indicator calculation time
- error_count: Total errors by type
- active_connections: Current active connections
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from prometheus_client import generate_latest, REGISTRY
from functools import wraps
import time
from typing import Callable, Any
from contextlib import contextmanager


# ============================================================================
# Metric Definitions
# ============================================================================

# Request metrics
request_total = Counter(
  'mojo_compute_requests_total',
  'Total number of requests',
  ['action', 'status']
)

request_duration_seconds = Histogram(
  'mojo_compute_request_duration_seconds',
  'Request duration in seconds',
  ['action'],
  buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Indicator computation metrics
indicator_compute_time_seconds = Histogram(
  'mojo_compute_indicator_time_seconds',
  'Indicator computation time in seconds',
  ['indicator'],
  buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]
)

indicator_data_points = Summary(
  'mojo_compute_indicator_data_points',
  'Number of data points processed per indicator calculation',
  ['indicator']
)

# Error metrics
error_total = Counter(
  'mojo_compute_errors_total',
  'Total number of errors',
  ['error_type']
)

# Connection metrics
active_connections = Gauge(
  'mojo_compute_active_connections',
  'Number of active connections'
)

# System metrics
mojo_info = Info(
  'mojo_compute_info',
  'Mojo compute service information'
)

# Performance metrics
indicator_speedup = Gauge(
  'mojo_compute_indicator_speedup',
  'Speedup factor vs Python baseline',
  ['indicator']
)


# ============================================================================
# Metric Decorators
# ============================================================================

def track_request(action: str):
  """Decorator to track request metrics.

  Args:
    action: Action name (e.g., 'compute_sma')

  Usage:
    @track_request('compute_sma')
    async def compute_sma(request):
      # ...
  """
  def decorator(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
      start_time = time.perf_counter()
      status = 'success'

      try:
        result = await func(*args, **kwargs)
        return result

      except Exception as e:
        status = 'error'
        error_total.labels(error_type=type(e).__name__).inc()
        raise

      finally:
        duration = time.perf_counter() - start_time
        request_total.labels(action=action, status=status).inc()
        request_duration_seconds.labels(action=action).observe(duration)

    return wrapper
  return decorator


def track_indicator_compute(indicator: str):
  """Decorator to track indicator computation metrics.

  Args:
    indicator: Indicator name (e.g., 'sma', 'ema', 'rsi')

  Usage:
    @track_indicator_compute('sma')
    def compute_sma(prices, period):
      # ...
  """
  def decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(prices, *args, **kwargs) -> Any:
      start_time = time.perf_counter()

      try:
        result = func(prices, *args, **kwargs)

        # Track metrics
        duration = time.perf_counter() - start_time
        indicator_compute_time_seconds.labels(indicator=indicator).observe(duration)
        indicator_data_points.labels(indicator=indicator).observe(len(prices))

        return result

      except Exception as e:
        error_total.labels(error_type=f'{indicator}_{type(e).__name__}').inc()
        raise

    return wrapper
  return decorator


# ============================================================================
# Context Managers
# ============================================================================

@contextmanager
def track_time(metric_name: str, labels: dict = None):
  """Context manager to track execution time.

  Args:
    metric_name: Name of the metric to update
    labels: Optional labels dict

  Usage:
    with track_time('compute_sma', {'period': 20}):
      result = compute_sma(prices, 20)
  """
  start_time = time.perf_counter()

  try:
    yield

  finally:
    duration = time.perf_counter() - start_time

    # Update appropriate metric
    if metric_name == 'indicator_compute':
      indicator = labels.get('indicator', 'unknown')
      indicator_compute_time_seconds.labels(indicator=indicator).observe(duration)

    elif metric_name == 'request':
      action = labels.get('action', 'unknown')
      request_duration_seconds.labels(action=action).observe(duration)


@contextmanager
def track_connection():
  """Context manager to track active connections.

  Usage:
    with track_connection():
      await handle_client(socket)
  """
  active_connections.inc()

  try:
    yield

  finally:
    active_connections.dec()


# ============================================================================
# Helper Functions
# ============================================================================

def record_error(error_type: str):
  """Record an error occurrence.

  Args:
    error_type: Type of error (e.g., 'ValueError', 'TimeoutError')
  """
  error_total.labels(error_type=error_type).inc()


def record_indicator_speedup(indicator: str, speedup: float):
  """Record indicator speedup vs Python baseline.

  Args:
    indicator: Indicator name
    speedup: Speedup factor (e.g., 100.0 for 100x speedup)
  """
  indicator_speedup.labels(indicator=indicator).set(speedup)


def set_service_info(version: str, mojo_version: str, simd_enabled: bool):
  """Set service information metrics.

  Args:
    version: Service version
    mojo_version: Mojo compiler version
    simd_enabled: Whether SIMD optimization is enabled
  """
  mojo_info.info({
    'version': version,
    'mojo_version': mojo_version,
    'simd_enabled': str(simd_enabled),
    'simd_width': '4'  # AVX2 default
  })


def get_metrics() -> bytes:
  """Get current metrics in Prometheus exposition format.

  Returns:
    Metrics as bytes in Prometheus text format
  """
  return generate_latest(REGISTRY)


# ============================================================================
# Metrics Reset (for testing)
# ============================================================================

def reset_metrics():
  """Reset all metrics (for testing purposes only)."""
  # Note: Prometheus metrics are cumulative by design
  # This should only be used in test environments
  pass


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
  """Example usage of the metrics module."""

  # Set service info
  set_service_info(
    version='1.0.0',
    mojo_version='24.5.0',
    simd_enabled=True
  )

  # Simulate some requests
  with track_connection():
    with track_time('request', {'action': 'compute_sma'}):
      time.sleep(0.01)  # Simulate work
      request_total.labels(action='compute_sma', status='success').inc()

  # Simulate indicator computation
  with track_time('indicator_compute', {'indicator': 'sma'}):
    time.sleep(0.001)  # Simulate computation
    indicator_data_points.labels(indicator='sma').observe(10000)

  # Record speedup
  record_indicator_speedup('sma', 1000.0)

  # Record error
  record_error('ValueError')

  # Print metrics
  print("Prometheus Metrics:")
  print("=" * 80)
  print(get_metrics().decode('utf-8'))
