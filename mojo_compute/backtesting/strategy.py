"""
Base strategy class for backtesting.

All trading strategies must inherit from this base class and implement
the on_bar() method to define their trading logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from .engine import Bar, Order, OrderSide, OrderType


class Strategy(ABC):
    """
    Base class for all trading strategies.

    Subclasses must implement:
    - on_bar(): Called for each bar event during backtesting
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize strategy with optional parameters.

        Args:
            params: Strategy-specific parameters (e.g., {'period': 20})
        """
        self.params = params or {}
        self.engine = None

    def initialize(self, engine):
        """
        Initialize strategy with backtesting engine.

        Called once at the start of backtesting.

        Args:
            engine: BacktestEngine instance
        """
        self.engine = engine

    @abstractmethod
    def on_bar(self, bars: Dict[str, Bar]):
        """
        Handle bar event - implement strategy logic here.

        Called for each time period during backtesting.

        Args:
            bars: Dict of symbol -> Bar data for current period
        """
        pass

    def finalize(self):
        """
        Cleanup after backtesting completes.

        Override to close any open positions or log final state.
        """
        pass

    # Helper methods for order submission

    def buy(self, symbol: str, quantity: float, price: Optional[float] = None):
        """
        Submit buy order.

        Args:
            symbol: Symbol to buy
            quantity: Number of shares/contracts
            price: Limit price (None for market order)
        """
        order = Order(
            order_id=f"order_{len(self.engine.orders)}",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET if price is None else OrderType.LIMIT,
            quantity=quantity,
            price=price,
        )
        self.engine.submit_order(order)

    def sell(self, symbol: str, quantity: float, price: Optional[float] = None):
        """
        Submit sell order.

        Args:
            symbol: Symbol to sell
            quantity: Number of shares/contracts
            price: Limit price (None for market order)
        """
        order = Order(
            order_id=f"order_{len(self.engine.orders)}",
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET if price is None else OrderType.LIMIT,
            quantity=quantity,
            price=price,
        )
        self.engine.submit_order(order)

    def get_position(self, symbol: str):
        """
        Get current position for a symbol.

        Args:
            symbol: Symbol to query

        Returns:
            Position object or None if no position
        """
        return self.engine.positions.get(symbol)

    def get_capital(self) -> float:
        """Get available capital."""
        return self.engine.capital

    def get_total_equity(self) -> float:
        """Get total equity (capital + unrealized P&L)."""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.engine.positions.values())
        return self.engine.capital + unrealized_pnl
