"""
Bollinger Bands Mean Reversion Strategy.

Buy when price touches lower band, sell when it touches upper band.
"""

from typing import Dict
from ..strategy import Strategy
from ..engine import Bar
import math


class BollingerReversionStrategy(Strategy):
    """
    Bollinger Bands Mean Reversion: Buy at lower band, sell at upper band.

    Parameters:
        period: Bollinger Bands period (default: 20)
        std_dev: Standard deviation multiplier (default: 2.0)
        position_size: Fraction of capital to use (default: 0.1)
    """

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.period = self.params.get('period', 20)
        self.std_dev = self.params.get('std_dev', 2.0)
        self.position_size = self.params.get('position_size', 0.1)

        # Store price history
        self.prices = {}

    def calculate_bollinger_bands(self, prices, period, std_dev):
        """Calculate Bollinger Bands (middle, upper, lower)."""
        if len(prices) < period:
            return None, None, None

        # Middle band (SMA)
        middle = sum(prices[-period:]) / period

        # Calculate standard deviation
        variance = sum((p - middle) ** 2 for p in prices[-period:]) / period
        std = math.sqrt(variance)

        # Upper and lower bands
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return middle, upper, lower

    def on_bar(self, bars: Dict[str, Bar]):
        """Handle each bar event."""
        for symbol, bar in bars.items():
            # Initialize price history
            if symbol not in self.prices:
                self.prices[symbol] = []

            # Add current close price
            self.prices[symbol].append(bar.close)

            # Need enough history
            if len(self.prices[symbol]) < self.period:
                continue

            # Calculate Bollinger Bands
            middle, upper, lower = self.calculate_bollinger_bands(
                self.prices[symbol], self.period, self.std_dev
            )

            if middle is None:
                continue

            # Get current position
            position = self.get_position(symbol)
            current_qty = position.quantity if position else 0
            current_price = bar.close

            # Buy when price touches or crosses below lower band
            if current_price <= lower and current_qty == 0:
                capital = self.get_capital()
                quantity = (capital * self.position_size) / bar.close
                if quantity > 0:
                    self.buy(symbol, quantity)

            # Sell when price touches or crosses above upper band
            elif current_price >= upper and current_qty > 0:
                self.sell(symbol, current_qty)

            # Also sell if price returns to middle band (take profit)
            elif current_price >= middle and current_qty > 0:
                if position.unrealized_pnl > 0:  # Only if profitable
                    self.sell(symbol, current_qty)
