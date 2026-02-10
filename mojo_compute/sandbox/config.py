"""Sandbox Engine Configuration."""

import os
from datetime import timezone, timedelta

# Service
SERVICE_PORT = int(os.getenv("SANDBOX_PORT", "6009"))
SERVICE_HOST = os.getenv("SANDBOX_HOST", "0.0.0.0")

# Database
PG_DSN = os.getenv("TRADING_CHITTI_PG_DSN", "postgresql://hariprasath@localhost:6432/trading_chitti")

# Intraday Engine WebSocket (signal source)
INTRADAY_WS_URL = os.getenv("INTRADAY_WS_URL", "ws://localhost:6007/ws")

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Default session settings
DEFAULT_LEVERAGE = 10.0
DEFAULT_MAX_POSITION_PCT = 20.0
DEFAULT_MAX_OPEN_POSITIONS = 5
DEFAULT_MAX_DAILY_DRAWDOWN_PCT = 10.0
DEFAULT_MAX_SECTOR_EXPOSURE_PCT = 40.0
MIN_TRADE_CAPITAL = 100.0  # Minimum INR per trade

# ML Money Manager
ML_TRADE_THRESHOLD = 0.55  # Trade if ML score > this
ML_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Equity snapshot interval (seconds)
EQUITY_SNAPSHOT_INTERVAL = 30

# WebSocket reconnect
WS_RECONNECT_DELAY = 2  # seconds
WS_MAX_RECONNECT_DELAY = 30  # seconds
WS_PING_INTERVAL = 25  # seconds
