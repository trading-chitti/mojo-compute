"""
RSI Reversal Strategy.

Mean reversion strategy: Buy when RSI < oversold, sell when RSI > overbought.
"""

from typing import Dict
from ..strategy import Strategy
from ..engine import Bar


class RSIReversalStrategy(Strategy):
    """
    RSI Mean Reversion: Buy oversold, sell overbought.

    Parameters:
        period: RSI calculation period (default: 14)
        oversold: Oversold threshold (default: 30)
        overbought: Overbought threshold (default: 70)
        position_size: Fraction of capital to use (default: 0.1)
    """

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.period = self.params.get('period', 14)
        self.oversold = self.params.get('oversold', 30)
        self.overbought = self.params.get('overbought', 70)
        self.position_size = self.params.get('position_size', 0.1)

        # Store price history
        self.prices = {}

    def calculate_rsi(self, prices, period):
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return None

        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(c, 0) for c in changes[-period:]]
        losses = [abs(min(c, 0)) for c in changes[-period:]]

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def on_bar(self, bars: Dict[str, Bar]):
        """Handle each bar event."""
        for symbol, bar in bars.items():
            # Initialize price history
            if symbol not in self.prices:
                self.prices[symbol] = []

            # Add current close price
            self.prices[symbol].append(bar.close)

            # Need enough history for RSI
            if len(self.prices[symbol]) < self.period + 1:
                continue

            # Calculate RSI
            rsi = self.calculate_rsi(self.prices[symbol], self.period)
            if rsi is None:
                continue

            # Get current position
            position = self.get_position(symbol)
            current_qty = position.quantity if position else 0

            # Buy when oversold
            if rsi < self.oversold and current_qty == 0:
                capital = self.get_capital()
                quantity = (capital * self.position_size) / bar.close
                if quantity > 0:
                    self.buy(symbol, quantity)

            # Sell when overbought
            elif rsi > self.overbought and current_qty > 0:
                self.sell(symbol, current_qty)
