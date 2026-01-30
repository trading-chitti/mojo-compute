"""Pydantic request/response schemas for Mojo Compute API."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional


class SMARequest(BaseModel):
    """Request model for SMA calculation."""

    symbol: str = Field(..., min_length=1, max_length=20, description="Stock symbol")
    prices: List[float] = Field(..., min_length=1, max_length=1_000_000, description="Close prices")
    period: int = Field(..., ge=2, le=500, description="SMA period")

    @field_validator("prices")
    @classmethod
    def validate_prices(cls, v: List[float]) -> List[float]:
        if any(p <= 0 for p in v):
            raise ValueError("All prices must be positive")
        return v


class SMAResponse(BaseModel):
    """Response model for SMA calculation."""

    symbol: str
    indicator: str = "sma"
    period: int
    values: List[Optional[float]]
    computation_time_ms: float
    mojo_used: bool


class RSIRequest(BaseModel):
    """Request model for RSI calculation."""

    symbol: str = Field(..., min_length=1, max_length=20)
    prices: List[float] = Field(..., min_length=1, max_length=1_000_000)
    period: int = Field(14, ge=2, le=100)

    @field_validator("prices")
    @classmethod
    def validate_prices(cls, v: List[float]) -> List[float]:
        if any(p <= 0 for p in v):
            raise ValueError("All prices must be positive")
        return v


class RSIResponse(BaseModel):
    """Response model for RSI calculation."""

    symbol: str
    indicator: str = "rsi"
    period: int
    values: List[Optional[float]]
    computation_time_ms: float
    mojo_used: bool


class MACDRequest(BaseModel):
    """Request model for MACD calculation."""

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

    @field_validator("prices")
    @classmethod
    def validate_prices(cls, v: List[float]) -> List[float]:
        if any(p <= 0 for p in v):
            raise ValueError("All prices must be positive")
        return v


class MACDResponse(BaseModel):
    """Response model for MACD calculation."""

    symbol: str
    indicator: str = "macd"
    macd_line: List[Optional[float]]
    signal_line: List[Optional[float]]
    histogram: List[Optional[float]]
    computation_time_ms: float
    mojo_used: bool


class BollingerRequest(BaseModel):
    """Request model for Bollinger Bands calculation."""

    symbol: str = Field(..., min_length=1, max_length=20)
    prices: List[float] = Field(..., min_length=1, max_length=1_000_000)
    period: int = Field(20, ge=2, le=200)
    std_dev: float = Field(2.0, ge=0.1, le=5.0)

    @field_validator("prices")
    @classmethod
    def validate_prices(cls, v: List[float]) -> List[float]:
        if any(p <= 0 for p in v):
            raise ValueError("All prices must be positive")
        return v


class BollingerResponse(BaseModel):
    """Response model for Bollinger Bands calculation."""

    symbol: str
    indicator: str = "bollinger_bands"
    upper_band: List[Optional[float]]
    middle_band: List[Optional[float]]
    lower_band: List[Optional[float]]
    computation_time_ms: float
    mojo_used: bool


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    mojo_available: bool
    version: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    message: str
    details: Optional[Dict] = None
    timestamp: str
