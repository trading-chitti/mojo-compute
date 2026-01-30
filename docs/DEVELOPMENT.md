# Development Guide

## Getting Started

### Prerequisites
- Python 3.11+
- Mojo SDK (see [INSTALLATION.md](./INSTALLATION.md) - to be created by SR. Dev Codex)
- macOS (M1 Pro ARM) or Linux

### Installation

```bash
# Clone repository
git clone https://github.com/trading-chitti/mojo-compute.git
cd mojo-compute

# Install Python dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install Mojo SDK (see INSTALLATION.md)
# ...
```

---

## Project Structure

```
mojo-compute/
├── mojo_compute/
│   ├── api/              # FastAPI server & schemas
│   │   ├── server.py     # Main FastAPI app
│   │   ├── schemas.py    # Pydantic models (DONE)
│   │   ├── bridge.py     # Python-Mojo FFI bridge
│   │   └── cache.py      # Redis caching
│   ├── indicators/       # Mojo indicator implementations
│   │   ├── sma.mojo      # Simple Moving Average
│   │   ├── rsi.mojo      # Relative Strength Index
│   │   ├── macd.mojo     # MACD
│   │   └── bollinger.mojo # Bollinger Bands
│   ├── backtesting/      # Mojo backtesting engine
│   │   ├── engine.mojo   # Core backtest loop
│   │   └── metrics.mojo  # Performance metrics
│   ├── ml/               # ML inference (future)
│   │   ├── matrix.mojo   # Matrix operations
│   │   ├── inference.mojo # Model inference
│   │   └── features.mojo # Feature engineering
│   └── tests/            # Test suite
│       ├── unit/
│       ├── integration/
│       └── benchmarks/
├── benchmarks/           # Performance benchmarks
│   ├── python_baseline.py # Python baseline (IN PROGRESS - SR. Dev Claude)
│   └── results/
├── docs/
│   ├── API_DESIGN.md     # API specification (DONE - Claude Master)
│   ├── INSTALLATION.md   # Installation guide (TODO - SR. Dev Codex)
│   ├── DEVELOPMENT.md    # This file
│   └── PERFORMANCE.md    # Performance report (TODO)
└── pyproject.toml        # Project configuration
```

---

## Development Workflow

### 1. Pick a Task
- Check [GitHub Issues](https://github.com/trading-chitti/mojo-compute/issues)
- Tasks labeled with `claude`, `codex`, or `master`
- Comment on issue to claim it

### 2. Create Branch
```bash
git checkout -b feature/issue-{number}-brief-description
# Example: git checkout -b feature/issue-5-sma-implementation
```

### 3. Implement
- Follow existing code style
- Add unit tests
- Update documentation if needed

### 4. Test
```bash
# Run all tests
pytest

# Run specific test file
pytest mojo_compute/tests/unit/test_sma.py

# Run with coverage
pytest --cov=mojo_compute

# Run benchmarks
pytest -m benchmark
```

### 5. Commit
```bash
git add .
git commit -m "Implement SMA in Mojo with SIMD optimization

- Created mojo_compute/indicators/sma.mojo
- Added unit tests with edge cases
- Benchmarked: 85x faster than NumPy

Closes #5

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### 6. Create Pull Request
```bash
gh pr create \
  --title "[Issue #5] SMA Implementation (Mojo)" \
  --body "Implements Simple Moving Average in Mojo with SIMD vectorization.

## Changes
- New file: \`mojo_compute/indicators/sma.mojo\`
- Tests: \`mojo_compute/tests/unit/test_sma.py\`
- Benchmark: 85x faster than NumPy

## Testing
- [x] Unit tests pass
- [x] Integration tests pass
- [x] Benchmark shows 80x+ speedup

## Checklist
- [x] Code follows Mojo best practices
- [x] Tests added
- [x] Documentation updated
- [x] Benchmark results documented

Closes #5" \
  --label "Ready for Deployment"
```

---

## Testing Guidelines

### Unit Tests
Test individual functions in isolation.

```python
# mojo_compute/tests/unit/test_sma.py
import pytest
import numpy as np
from mojo_compute.indicators import sma

def test_sma_basic():
    prices = [10.0, 11.0, 12.0, 13.0, 14.0]
    result = sma(prices, period=3)
    expected = [None, None, 11.0, 12.0, 13.0]
    assert result == expected

def test_sma_period_too_large():
    prices = [10.0, 11.0]
    with pytest.raises(ValueError):
        sma(prices, period=5)
```

### Integration Tests
Test API endpoints end-to-end.

```python
# mojo_compute/tests/integration/test_api.py
from fastapi.testclient import TestClient
from mojo_compute.api.server import app

client = TestClient(app)

def test_sma_endpoint():
    response = client.post("/compute/sma", json={
        "symbol": "TCS",
        "prices": [1234.5, 1240.0, 1235.5],
        "period": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "TCS"
    assert data["period"] == 2
    assert len(data["values"]) == 3
```

### Benchmark Tests
Compare Mojo vs Python performance.

```python
# mojo_compute/tests/benchmarks/test_performance.py
import pytest
import time
import numpy as np
from mojo_compute.indicators import sma as mojo_sma

@pytest.mark.benchmark
def test_sma_performance():
    prices = np.random.randn(10000).tolist()
    period = 20

    # Mojo version
    start = time.perf_counter()
    mojo_result = mojo_sma(prices, period)
    mojo_time = (time.perf_counter() - start) * 1000

    # NumPy version
    start = time.perf_counter()
    numpy_result = np.convolve(prices, np.ones(period)/period, mode='valid')
    numpy_time = (time.perf_counter() - start) * 1000

    speedup = numpy_time / mojo_time
    print(f"Mojo: {mojo_time:.2f}ms, NumPy: {numpy_time:.2f}ms, Speedup: {speedup:.1f}x")
    assert speedup > 50, f"Expected 50x+ speedup, got {speedup:.1f}x"
```

---

## Code Style

### Python
- Follow PEP 8
- Use type hints
- Use Pydantic for validation
- Keep functions small and focused

```python
# Good
def calculate_sma(prices: List[float], period: int) -> List[Optional[float]]:
    """Calculate Simple Moving Average.

    Args:
        prices: List of closing prices
        period: Number of periods for moving average

    Returns:
        List of SMA values (None for warmup period)

    Raises:
        ValueError: If period > len(prices)
    """
    if period > len(prices):
        raise ValueError(f"period ({period}) exceeds data length ({len(prices)})")
    # ...
```

### Mojo
- Use SIMD for vectorization
- Leverage strong typing
- Optimize for cache locality
- Document performance characteristics

```mojo
# Good
fn sma(prices: DynamicVector[Float64], period: Int) -> DynamicVector[Float64]:
    """Calculate Simple Moving Average using SIMD.

    Performance: ~100x faster than NumPy for 10K data points.

    Args:
        prices: Vector of closing prices
        period: Number of periods for moving average

    Returns:
        Vector of SMA values
    """
    # Use SIMD for vectorization
    let simd_width = 8  # Process 8 floats at once
    # ...
```

---

## Debugging

### Python Debugging
```bash
# Run with debugger
python -m pdb mojo_compute/api/server.py

# Or use ipdb
pip install ipdb
import ipdb; ipdb.set_trace()
```

### Mojo Debugging
```bash
# Print debugging (Mojo doesn't have full debugger yet)
print("Debug: prices length =", len(prices))

# Assertions
debug_assert(period > 0, "period must be positive")
```

---

## Performance Profiling

### Python
```bash
# Line profiler
pip install line_profiler
kernprof -l -v mojo_compute/api/server.py

# Memory profiler
pip install memory_profiler
python -m memory_profiler benchmarks/python_baseline.py
```

### Mojo
```bash
# Use Mojo's built-in profiler (when available)
mojo run --profile mojo_compute/indicators/sma.mojo
```

---

## Continuous Integration

GitHub Actions runs on every push:
- Linting (ruff)
- Type checking (mypy)
- Unit tests (pytest)
- Integration tests
- Benchmark regression tests

See `.github/workflows/test.yml` (to be created)

---

## Team Coordination

### Daily Standup (Async via GitHub Comments)
Each team member comments on their assigned issues:
- What I completed yesterday
- What I'm working on today
- Any blockers

Example:
```
**Update (2024-01-15)**:
- ✅ Completed SMA implementation with SIMD
- 🏗️ Working on RSI calculation
- ❌ Blocker: Need Mojo FFI bridge (Issue #11) before integration tests
```

### Code Review
- Claude Master reviews all PRs
- At least 1 approval required before merge
- Automated tests must pass

---

## Useful Resources

### Mojo
- [Mojo Docs](https://docs.modular.com/mojo/)
- [Mojo Playground](https://playground.modular.com/)
- [Mojo Standard Library](https://docs.modular.com/mojo/lib/)

### Python
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [NumPy Docs](https://numpy.org/doc/)

### Trading/Finance
- [TA-Lib](https://ta-lib.org/)
- [pandas-ta](https://github.com/twopirllc/pandas-ta)

---

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Issue Comments**: For task-specific questions
- **This Guide**: For general development questions

---

Happy coding! 🚀
