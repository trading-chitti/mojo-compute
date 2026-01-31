"""
Opening Range Breakout (ORB) Strategy.

Buy when price breaks above opening range high, sell when it breaks below low.
Popular intraday strategy.
"""

from typing import Dict
from datetime import datetime, time
from ..strategy import Strategy
from ..engine import Bar


class OpeningRangeBreakoutStrategy(Strategy):
    """
    Opening Range Breakout: Trade breakouts from first N minutes of trading.

    Parameters:
        range_minutes: Opening range duration in minutes (default: 15)
        position_size: Fraction of capital to use (default: 0.1)
        use_stop_loss: Enable stop loss at opening range opposite end (default: True)
    """

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.range_minutes = self.params.get('range_minutes', 15)
        self.position_size = self.params.get('position_size', 0.1)
        self.use_stop_loss = self.params.get('use_stop_loss', True)

        # Track opening range for each symbol
        self.opening_ranges = {}  # {symbol: {'high': float, 'low': float, 'date': date}}
        self.bars_seen = {}  # {symbol: count}

    def on_bar(self, bars: Dict[str, Bar]):
        """Handle each bar event."""
        for symbol, bar in bars.items():
            # Initialize tracking
            if symbol not in self.opening_ranges:
                self.opening_ranges[symbol] = {'high': None, 'low': None, 'date': None}
                self.bars_seen[symbol] = 0

            current_date = bar.timestamp.date()

            # Check if new trading day
            if self.opening_ranges[symbol]['date'] != current_date:
                self.opening_ranges[symbol] = {
                    'high': bar.high,
                    'low': bar.low,
                    'date': current_date
                }
                self.bars_seen[symbol] = 1
                continue

            # Still building opening range
            if self.bars_seen[symbol] < self.range_minutes:
                self.opening_ranges[symbol]['high'] = max(
                    self.opening_ranges[symbol]['high'], bar.high
                )
                self.opening_ranges[symbol]['low'] = min(
                    self.opening_ranges[symbol]['low'], bar.low
                )
                self.bars_seen[symbol] += 1
                continue

            # Opening range established - look for breakouts
            or_high = self.opening_ranges[symbol]['high']
            or_low = self.opening_ranges[symbol]['low']

            # Get current position
            position = self.get_position(symbol)
            current_qty = position.quantity if position else 0

            # Breakout above opening range high
            if bar.close > or_high and current_qty == 0:
                capital = self.get_capital()
                quantity = (capital * self.position_size) / bar.close
                if quantity > 0:
                    self.buy(symbol, quantity)

            # Breakdown below opening range low
            elif bar.close < or_low and current_qty == 0:
                # Short not implemented in basic engine, so skip
                pass

            # Stop loss: if long and price breaks below OR low
            elif self.use_stop_loss and current_qty > 0 and bar.close < or_low:
                self.sell(symbol, current_qty)

            # Increment bar counter
            self.bars_seen[symbol] += 1

    def finalize(self):
        """Close all positions at end of day."""
        for symbol, position in self.engine.positions.items():
            if position.quantity > 0:
                # Get current bar to determine price
                if symbol in self.engine.current_bar:
                    bar = self.engine.current_bar[symbol]
                    self.sell(symbol, position.quantity)
