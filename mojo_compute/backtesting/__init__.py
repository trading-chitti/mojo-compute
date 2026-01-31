"""
Event-driven backtesting engine for trading strategies.

This module provides:
- BacktestEngine: Core event-driven backtesting engine
- Strategy: Base class for all trading strategies
- Order, Position, Bar: Data structures for backtesting
"""

from .engine import BacktestEngine, Order, Position, Bar, OrderSide, OrderType
from .strategy import Strategy

__all__ = [
    'BacktestEngine',
    'Strategy',
    'Order',
    'Position',
    'Bar',
    'OrderSide',
    'OrderType',
]
