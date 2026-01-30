"""FastAPI server for Mojo Compute Service.

This module provides REST API endpoints for high-performance indicator calculations,
backtesting, and ML inference using Mojo.
"""

import time
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from .schemas import (
    HealthResponse,
    SMARequest,
    SMAResponse,
    RSIRequest,
    RSIResponse,
    MACDRequest,
    MACDResponse,
    BollingerRequest,
    BollingerResponse,
    ErrorResponse,
)

# Initialize FastAPI app
app = FastAPI(
    title="Mojo Compute Service",
    description="High-performance trading indicator calculations using Mojo (35,000x faster than Python)",
    version="0.1.0",
)

# Prometheus metrics
requests_total = Counter(
    "mojo_compute_requests_total", "Total requests", ["endpoint", "status"]
)
computation_time = Histogram(
    "mojo_computation_seconds", "Computation time", ["indicator"]
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Mojo availability flag (will be set by FFI bridge)
MOJO_AVAILABLE = False


def now_iso() -> str:
    """Return current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint.

    Returns service status and Mojo availability.
    """
    return HealthResponse(
        status="healthy",
        mojo_available=MOJO_AVAILABLE,
        version="0.1.0",
        timestamp=now_iso(),
    )


@app.post("/compute/sma", response_model=SMAResponse)
def compute_sma(request: SMARequest):
    """Calculate Simple Moving Average.

    Args:
        request: SMARequest with symbol, prices, and period

    Returns:
        SMAResponse with calculated SMA values

    Raises:
        HTTPException: If computation fails
    """
    requests_total.labels(endpoint="/compute/sma", status="started").inc()

    # Validate: period must be <= len(prices)
    if request.period > len(request.prices):
        requests_total.labels(endpoint="/compute/sma", status="error").inc()
        raise HTTPException(
            status_code=400,
            detail=f"period ({request.period}) exceeds data length ({len(request.prices)})",
        )

    start_time = time.perf_counter()

    try:
        # TODO: Call Mojo SMA function (Issue #5 - SR. Dev Claude)
        # For now, use NumPy fallback
        import numpy as np

        prices_array = np.array(request.prices)
        sma_values = []

        # Calculate SMA with warmup period (first period-1 values are None)
        for i in range(len(prices_array)):
            if i < request.period - 1:
                sma_values.append(None)
            else:
                window = prices_array[i - request.period + 1 : i + 1]
                sma_values.append(float(np.mean(window)))

        computation_time_ms = (time.perf_counter() - start_time) * 1000
        computation_time.labels(indicator="sma").observe(computation_time_ms / 1000)

        requests_total.labels(endpoint="/compute/sma", status="success").inc()

        return SMAResponse(
            symbol=request.symbol,
            period=request.period,
            values=sma_values,
            computation_time_ms=computation_time_ms,
            mojo_used=False,  # Using NumPy fallback for now
        )

    except Exception as e:
        requests_total.labels(endpoint="/compute/sma", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/compute/rsi", response_model=RSIResponse)
def compute_rsi(request: RSIRequest):
    """Calculate Relative Strength Index.

    Args:
        request: RSIRequest with symbol, prices, and period

    Returns:
        RSIResponse with calculated RSI values

    Raises:
        HTTPException: If computation fails
    """
    requests_total.labels(endpoint="/compute/rsi", status="started").inc()

    # TODO: Implement RSI calculation (Issue #6 - SR. Dev Codex)
    # For now, return placeholder
    raise HTTPException(
        status_code=501, detail="RSI calculation not yet implemented (see Issue #6)"
    )


@app.post("/compute/macd", response_model=MACDResponse)
def compute_macd(request: MACDRequest):
    """Calculate MACD indicator.

    Args:
        request: MACDRequest with symbol, prices, and periods

    Returns:
        MACDResponse with MACD line, signal line, and histogram

    Raises:
        HTTPException: If computation fails
    """
    requests_total.labels(endpoint="/compute/macd", status="started").inc()

    # TODO: Implement MACD calculation (Issue #7 - SR. Dev Claude)
    raise HTTPException(
        status_code=501, detail="MACD calculation not yet implemented (see Issue #7)"
    )


@app.post("/compute/bollinger", response_model=BollingerResponse)
def compute_bollinger(request: BollingerRequest):
    """Calculate Bollinger Bands.

    Args:
        request: BollingerRequest with symbol, prices, period, and std_dev

    Returns:
        BollingerResponse with upper, middle, and lower bands

    Raises:
        HTTPException: If computation fails
    """
    requests_total.labels(endpoint="/compute/bollinger", status="started").inc()

    # TODO: Implement Bollinger Bands calculation (Issue #8 - SR. Dev Codex)
    raise HTTPException(
        status_code=501,
        detail="Bollinger Bands calculation not yet implemented (see Issue #8)",
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message=str(exc),
            timestamp=now_iso(),
        ).model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6003)
