"""
Donchian Channel Breakout Strategy.

Buy when price breaks above N-period high, sell when it breaks below N-period low.
Used in famous Turtle Trading system.
"""

from typing import Dict
from ..strategy import Strategy
from ..engine import Bar


class DonchianBreakoutStrategy(Strategy):
    """
    Donchian Channel Breakout: Buy N-period highs, sell N-period lows.

    Parameters:
        entry_period: Period for entry breakout (default: 20)
        exit_period: Period for exit (default: 10)
        position_size: Fraction of capital to use (default: 0.1)
    """

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.entry_period = self.params.get('entry_period', 20)
        self.exit_period = self.params.get('exit_period', 10)
        self.position_size = self.params.get('position_size', 0.1)

        # Store price history (highs and lows)
        self.highs = {}
        self.lows = {}

    def on_bar(self, bars: Dict[str, Bar]):
        """Handle each bar event."""
        for symbol, bar in bars.items():
            # Initialize price history
            if symbol not in self.highs:
                self.highs[symbol] = []
                self.lows[symbol] = []

            # Add current high and low
            self.highs[symbol].append(bar.high)
            self.lows[symbol].append(bar.low)

            # Need enough history
            if len(self.highs[symbol]) < self.entry_period:
                continue

            # Calculate Donchian channels
            entry_high = max(self.highs[symbol][-self.entry_period:-1])  # Exclude current bar
            entry_low = min(self.lows[symbol][-self.entry_period:-1])

            exit_high = max(self.highs[symbol][-self.exit_period:-1]) if len(self.highs[symbol]) >= self.exit_period else entry_high
            exit_low = min(self.lows[symbol][-self.exit_period:-1]) if len(self.lows[symbol]) >= self.exit_period else entry_low

            # Get current position
            position = self.get_position(symbol)
            current_qty = position.quantity if position else 0

            # Entry: Breakout above N-period high
            if bar.close > entry_high and current_qty == 0:
                capital = self.get_capital()
                quantity = (capital * self.position_size) / bar.close
                if quantity > 0:
                    self.buy(symbol, quantity)

            # Exit long: Price breaks below exit channel low
            elif bar.close < exit_low and current_qty > 0:
                self.sell(symbol, current_qty)

            # Alternative exit: If price breaks below entry channel low (stop loss)
            elif bar.close < entry_low and current_qty > 0:
                self.sell(symbol, current_qty)
