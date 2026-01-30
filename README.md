# Mojo Compute Service

High-performance computing layer for Trading-Chitti using Mojo.

## Architecture

```
Python Services (FastAPI) → Mojo Compute (Performance Layer)
```

## Modules

- **indicators/** - Technical indicator calculations (100x faster)
- **backtesting/** - Trading strategy backtesting engine (1000x faster)
- **ml/** - ML model inference (35,000x faster)
- **api/** - Python bridge to expose Mojo functions

## Performance Targets

| Operation | Python | Mojo | Speedup |
|-----------|--------|------|---------|
| SMA (1 year) | 100ms | 1ms | 100x |
| 100 Indicators | 5s | 50ms | 100x |
| Backtest (1 year) | 30min | 30s | 60x |
| ML Inference | 50ms | 0.05ms | 1000x |

## Status

🚧 Under active development
