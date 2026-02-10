"""Baseline Tracker - "Trade all signals" parallel portfolio.

Tracks what would happen if every signal was traded with equal allocation,
providing a benchmark to measure ML Money Manager's alpha.
"""

import logging
from typing import Any, Dict, List

from .mojo_wrappers import (
    calculate_realized_pnl,
    calculate_realized_pnl_pct,
    calculate_quantity,
    calculate_position_value,
)

logger = logging.getLogger(__name__)


class BaselineTracker:
    """Simulates a naive strategy that trades every signal equally.

    No ML, no risk management - just equal-weight all signals.
    Used to measure ML alpha = (ML equity - baseline equity).
    """

    def __init__(self, initial_capital: float, leverage: float, max_positions: int = 5):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.leverage = leverage
        self.max_positions = max_positions

        self.open_trades: Dict[str, Dict[str, Any]] = {}  # signal_id -> trade info
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0

    @property
    def equity(self) -> float:
        return self.current_capital

    @property
    def pnl(self) -> float:
        return self.total_pnl

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100.0

    def on_new_signal(self, signal: Dict[str, Any]) -> None:
        """Process a new signal - trade it if under position limit."""
        if len(self.open_trades) >= self.max_positions:
            return

        signal_id = signal.get("signal_id", signal.get("id", ""))
        if signal_id in self.open_trades:
            return

        entry_price = float(signal.get("entry_price", 0))
        if entry_price <= 0:
            return

        # Equal allocation
        allocation_pct = 100.0 / self.max_positions
        allocated = self.current_capital * (allocation_pct / 100.0)

        quantity = calculate_quantity(allocated, self.leverage, entry_price)
        if quantity <= 0:
            return

        self.open_trades[signal_id] = {
            "signal_id": signal_id,
            "symbol": signal.get("symbol", ""),
            "signal_type": signal.get("signal_type", "CALL"),
            "entry_price": entry_price,
            "target_price": float(signal.get("target_price", 0)),
            "stop_loss": float(signal.get("stop_loss", 0)),
            "allocated_capital": allocated,
            "quantity": quantity,
        }

    def on_signal_closed(self, signal: Dict[str, Any], result: str) -> None:
        """Process a signal being closed."""
        signal_id = signal.get("signal_id", signal.get("id", ""))
        trade = self.open_trades.pop(signal_id, None)
        if not trade:
            return

        exit_price = float(signal.get("exit_price", signal.get("current_price", trade["entry_price"])))
        pnl = calculate_realized_pnl(
            trade["signal_type"], trade["entry_price"], exit_price, trade["quantity"]
        )

        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self.total_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get baseline performance summary."""
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": ((self.current_capital - self.initial_capital) / self.initial_capital * 100.0)
                             if self.initial_capital > 0 else 0.0,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "open_positions": len(self.open_trades),
        }
