# Mojo Compute Service - API Design

## Overview
The Mojo Compute Service exposes high-performance indicator calculations, backtesting, and ML inference via a FastAPI REST API. Python services (news-nlp, signal-service) call this API to offload compute-intensive operations to Mojo.

**Architecture**: Python FastAPI → Mojo FFI → Mojo Compute → Response

---

## API Endpoints

### Health Check

#### `GET /health`
Check service health and Mojo availability.

**Response**:
```json
{
  "status": "healthy",
  "mojo_available": true,
  "version": "0.1.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### Indicator Computation

#### `POST /compute/sma`
Calculate Simple Moving Average.

**Request**:
```json
{
  "symbol": "TCS",
  "prices": [1234.5, 1240.0, 1235.5, ...],  // array of close prices
  "period": 20
}
```

**Response**:
```json
{
  "symbol": "TCS",
  "indicator": "sma",
  "period": 20,
  "values": [null, null, ..., 1235.8, 1236.2, ...],  // nulls for warmup
  "computation_time_ms": 0.15,
  "mojo_used": true
}
```

**Status Codes**:
- `200 OK`: Success
- `400 Bad Request`: Invalid input (e.g., period > len(prices))
- `500 Internal Server Error`: Computation failed

---

#### `POST /compute/rsi`
Calculate Relative Strength Index.

**Request**:
```json
{
  "symbol": "TCS",
  "prices": [1234.5, 1240.0, ...],
  "period": 14
}
```

**Response**:
```json
{
  "symbol": "TCS",
  "indicator": "rsi",
  "period": 14,
  "values": [null, ..., 45.2, 52.1, 58.3, ...],
  "computation_time_ms": 0.22,
  "mojo_used": true
}
```

---

#### `POST /compute/macd`
Calculate MACD indicator.

**Request**:
```json
{
  "symbol": "TCS",
  "prices": [1234.5, ...],
  "fast_period": 12,
  "slow_period": 26,
  "signal_period": 9
}
```

**Response**:
```json
{
  "symbol": "TCS",
  "indicator": "macd",
  "macd_line": [null, ..., 12.5, 13.2, ...],
  "signal_line": [null, ..., 11.8, 12.3, ...],
  "histogram": [null, ..., 0.7, 0.9, ...],
  "computation_time_ms": 0.35,
  "mojo_used": true
}
```

---

#### `POST /compute/bollinger`
Calculate Bollinger Bands.

**Request**:
```json
{
  "symbol": "TCS",
  "prices": [1234.5, ...],
  "period": 20,
  "std_dev": 2.0
}
```

**Response**:
```json
{
  "symbol": "TCS",
  "indicator": "bollinger_bands",
  "upper_band": [null, ..., 1250.5, ...],
  "middle_band": [null, ..., 1235.0, ...],
  "lower_band": [null, ..., 1219.5, ...],
  "computation_time_ms": 0.28,
  "mojo_used": true
}
```

---

### Batch Processing

#### `POST /compute/batch`
Compute multiple indicators for multiple symbols efficiently.

**Request**:
```json
{
  "requests": [
    {
      "symbol": "TCS",
      "prices": [1234.5, ...],
      "indicators": [
        {"type": "sma", "params": {"period": 20}},
        {"type": "rsi", "params": {"period": 14}},
        {"type": "macd", "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}}
      ]
    },
    {
      "symbol": "INFY",
      "prices": [1450.0, ...],
      "indicators": [
        {"type": "sma", "params": {"period": 50}},
        {"type": "bollinger", "params": {"period": 20, "std_dev": 2.0}}
      ]
    }
  ]
}
```

**Response**:
```json
{
  "results": [
    {
      "symbol": "TCS",
      "indicators": {
        "sma": {"period": 20, "values": [...], "computation_time_ms": 0.15},
        "rsi": {"period": 14, "values": [...], "computation_time_ms": 0.22},
        "macd": {"macd_line": [...], "signal_line": [...], "histogram": [...], "computation_time_ms": 0.35}
      }
    },
    {
      "symbol": "INFY",
      "indicators": {
        "sma": {"period": 50, "values": [...], "computation_time_ms": 0.16},
        "bollinger": {"upper_band": [...], "middle_band": [...], "lower_band": [...], "computation_time_ms": 0.29}
      }
    }
  ],
  "total_computation_time_ms": 1.17,
  "symbols_processed": 2,
  "indicators_computed": 5,
  "mojo_used": true
}
```

**Performance**:
- Parallel processing across symbols (Mojo workers)
- Shared memory for duplicate calculations
- Expected throughput: 10,000+ symbols/second

---

### Backtesting

#### `POST /backtest/run`
Run a strategy backtest.

**Request**:
```json
{
  "strategy_id": "sma_crossover",
  "symbol": "TCS",
  "prices": {
    "dates": ["2023-01-01", "2023-01-02", ...],
    "open": [1230.0, 1235.0, ...],
    "high": [1240.0, 1245.0, ...],
    "low": [1225.0, 1230.0, ...],
    "close": [1234.5, 1240.0, ...],
    "volume": [1000000, 1200000, ...]
  },
  "parameters": {
    "fast_period": 10,
    "slow_period": 20,
    "initial_capital": 100000,
    "position_size": 0.1,
    "transaction_cost": 0.001
  }
}
```

**Response**:
```json
{
  "run_id": "abc123-def456",
  "status": "completed",
  "results": {
    "total_return": 0.156,
    "sharpe_ratio": 1.42,
    "max_drawdown": -0.082,
    "win_rate": 0.58,
    "total_trades": 45,
    "profitable_trades": 26,
    "losing_trades": 19,
    "avg_win": 0.023,
    "avg_loss": -0.015,
    "profit_factor": 1.67
  },
  "equity_curve": [100000, 100500, 99800, ...],
  "trades": [
    {"date": "2023-01-15", "type": "buy", "price": 1245.0, "quantity": 8, "pnl": null},
    {"date": "2023-01-22", "type": "sell", "price": 1260.0, "quantity": 8, "pnl": 120.0},
    ...
  ],
  "computation_time_ms": 45.2,
  "mojo_used": true
}
```

**Async Version**: `POST /backtest/run?async=true`
- Returns `run_id` immediately
- Client polls `GET /backtest/results/{run_id}` for status

---

#### `GET /backtest/results/{run_id}`
Retrieve backtest results.

**Response**: Same as `/backtest/run` response

**Status**:
- `pending`: Backtest in progress
- `completed`: Results ready
- `failed`: Error occurred

---

### ML Inference

#### `POST /ml/predict`
Run ML model inference (future).

**Request**:
```json
{
  "model_id": "lgbm_direction_v2",
  "features": {
    "rsi_14": 45.2,
    "macd_line": 12.5,
    "momentum_20d": 0.035,
    "volume_surge": 1.8,
    "news_sentiment": 0.65,
    ...
  }
}
```

**Response**:
```json
{
  "model_id": "lgbm_direction_v2",
  "prediction": "bullish",
  "confidence": 0.78,
  "probabilities": {
    "bullish": 0.78,
    "bearish": 0.15,
    "neutral": 0.07
  },
  "inference_time_ms": 0.08,
  "mojo_used": true
}
```

---

## Pydantic Schemas

### Request Models

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional

class SMARequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    prices: List[float] = Field(..., min_length=1, max_length=1_000_000)
    period: int = Field(..., ge=2, le=500)

    @field_validator("prices")
    @classmethod
    def validate_prices(cls, v: List[float]) -> List[float]:
        if any(p <= 0 for p in v):
            raise ValueError("All prices must be positive")
        return v

class RSIRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    prices: List[float] = Field(..., min_length=1, max_length=1_000_000)
    period: int = Field(14, ge=2, le=100)

class MACDRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    prices: List[float] = Field(..., min_length=1, max_length=1_000_000)
    fast_period: int = Field(12, ge=2, le=100)
    slow_period: int = Field(26, ge=2, le=200)
    signal_period: int = Field(9, ge=2, le=50)

    @field_validator("slow_period")
    @classmethod
    def validate_slow_greater_than_fast(cls, v: int, info) -> int:
        if "fast_period" in info.data and v <= info.data["fast_period"]:
            raise ValueError("slow_period must be > fast_period")
        return v

class BollingerRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    prices: List[float] = Field(..., min_length=1, max_length=1_000_000)
    period: int = Field(20, ge=2, le=200)
    std_dev: float = Field(2.0, ge=0.1, le=5.0)

class IndicatorSpec(BaseModel):
    type: str = Field(..., pattern="^(sma|rsi|macd|bollinger|obv|mfi)$")
    params: Dict[str, float | int]

class BatchRequest(BaseModel):
    requests: List[Dict] = Field(..., min_length=1, max_length=1000)
```

### Response Models

```python
class SMAResponse(BaseModel):
    symbol: str
    indicator: str = "sma"
    period: int
    values: List[Optional[float]]
    computation_time_ms: float
    mojo_used: bool

class RSIResponse(BaseModel):
    symbol: str
    indicator: str = "rsi"
    period: int
    values: List[Optional[float]]
    computation_time_ms: float
    mojo_used: bool

class MACDResponse(BaseModel):
    symbol: str
    indicator: str = "macd"
    macd_line: List[Optional[float]]
    signal_line: List[Optional[float]]
    histogram: List[Optional[float]]
    computation_time_ms: float
    mojo_used: bool

class BacktestResults(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
```

---

## Data Transfer Optimization

### Memory-Efficient Strategies:

1. **NumPy Array Sharing**: Use Python buffer protocol for zero-copy transfer
   ```python
   # Python side
   import numpy as np
   prices_array = np.array(prices, dtype=np.float64)

   # Pass pointer to Mojo (via ctypes/cffi)
   result_ptr = mojo_sma(prices_array.ctypes.data, len(prices), period)
   ```

2. **Streaming for Large Datasets**: For 100K+ data points
   - Client sends data in chunks
   - Server streams results back
   - Reduces peak memory usage

3. **Compression**: For network transfer (optional)
   - Use gzip compression for large arrays
   - Trade CPU for bandwidth

---

## Error Handling

### Error Response Format:
```json
{
  "error": "invalid_input",
  "message": "period (200) exceeds data length (150)",
  "details": {
    "field": "period",
    "constraint": "period <= len(prices)"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Categories:
- `invalid_input`: Validation error (400)
- `computation_error`: Mojo calculation failed (500)
- `mojo_unavailable`: Fallback to Python (503)
- `timeout`: Computation took too long (504)

---

## Performance Targets

| Endpoint | Data Size | Mojo Target | Python Baseline | Speedup |
|----------|-----------|-------------|-----------------|---------|
| `/compute/sma` | 10K points | < 1ms | 80ms | 80x |
| `/compute/rsi` | 10K points | < 1.5ms | 120ms | 80x |
| `/compute/macd` | 10K points | < 2ms | 180ms | 90x |
| `/compute/bollinger` | 10K points | < 1.5ms | 150ms | 100x |
| `/compute/batch` | 1000 symbols | < 500ms | 30s | 60x |
| `/backtest/run` | 1 year daily | < 50ms | 3s | 60x |

---

## Caching Strategy

### Redis Cache:
- **Key Format**: `mojo:{indicator}:{symbol}:{params_hash}`
- **Example**: `mojo:sma:TCS:sha256(period=20,prices=...)`
- **TTL**: 1 hour (indicators rarely change for historical data)
- **Invalidation**: On new data arrival

### Cache Hit Benefits:
- Latency: ~5ms (Redis lookup) vs ~50ms (Mojo compute)
- Throughput: 10x higher for repeated requests

---

## Fallback Mechanism

If Mojo is unavailable:
1. Log warning: "Mojo unavailable, falling back to Python"
2. Use NumPy/pandas_ta/TA-Lib for computation
3. Response includes `"mojo_used": false`
4. Prometheus metric: `mojo_fallback_total` incremented

---

## Monitoring & Observability

### Prometheus Metrics:
```python
from prometheus_client import Counter, Histogram, Gauge

requests_total = Counter("mojo_compute_requests_total", "Total requests", ["endpoint", "status"])
computation_time = Histogram("mojo_computation_seconds", "Computation time", ["indicator"])
mojo_available = Gauge("mojo_available", "Is Mojo runtime available")
cache_hits = Counter("mojo_cache_hits_total", "Cache hits", ["indicator"])
cache_misses = Counter("mojo_cache_misses_total", "Cache misses", ["indicator"])
```

### Structured Logging:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "endpoint": "/compute/sma",
  "symbol": "TCS",
  "period": 20,
  "data_points": 10000,
  "computation_time_ms": 0.85,
  "mojo_used": true,
  "cache_hit": false
}
```

---

## Security Considerations

1. **Rate Limiting**: 100 requests/minute per IP
2. **Input Validation**: Strict Pydantic schemas
3. **Resource Limits**:
   - Max array size: 1M elements
   - Max batch size: 1000 symbols
   - Request timeout: 30 seconds
4. **Authentication** (future): API key or OAuth2

---

## Implementation Checklist

- [ ] FastAPI server setup (`api/server.py`)
- [ ] Pydantic request/response models (`api/schemas.py`)
- [ ] Mojo FFI bridge (`api/bridge.py`, `api/mojo_ffi.mojo`)
- [ ] Error handling middleware
- [ ] Redis caching layer (`api/cache.py`)
- [ ] Prometheus metrics
- [ ] Structured logging
- [ ] Health check endpoint
- [ ] Integration tests (`tests/integration/test_api.py`)
- [ ] Load testing (100K+ data points, 1000+ symbols)
- [ ] Documentation (OpenAPI/Swagger auto-generated)

---

## Next Steps

1. **SR. Dev Claude**: Implement FastAPI server (Issue #10)
2. **SR. Dev Codex**: Implement Mojo FFI bridge (Issue #11)
3. **Claude Master**: Integration testing (Issue #12)
