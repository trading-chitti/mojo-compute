"""
Event-driven backtesting engine.

This module implements a complete backtesting system with:
- Order execution with commission and slippage
- Position tracking
- P&L calculation
- Performance metrics (Sharpe ratio, max drawdown, win rate)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np


class OrderSide(Enum):
    """Order side: BUY or SELL."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass
class Order:
    """Order data structure."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    filled_quantity: float = 0.0
    status: str = "pending"  # pending, filled, cancelled, rejected


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    quantity: float
    avg_entry_price: float
    side: OrderSide
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Bar:
    """OHLCV bar data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Features:
    - Order execution with commission and slippage
    - Position tracking with P&L calculation
    - Performance metrics (Sharpe, drawdown, win rate)
    - Support for multiple symbols
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,   # 0.05%
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        self.current_bar: Dict[str, Bar] = {}
        self.bar_index = 0

    def run_backtest(
        self,
        strategy,
        data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """
        Run backtest for a strategy.

        Args:
            strategy: Strategy instance
            data: Dict of symbol -> DataFrame with OHLCV data
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Dict with metrics, equity_curve, trades
        """
        # Initialize strategy
        strategy.initialize(self)

        # Create event queue
        events = self._create_event_queue(data, start_date, end_date)

        # Process events
        for event in events:
            self.current_bar = event['bars']
            self.bar_index = event['index']

            # Process pending orders
            self._process_orders()

            # Update positions and equity
            self._update_positions()
            self._record_equity()

            # Call strategy on_bar
            strategy.on_bar(event['bars'])

        # Finalize
        strategy.finalize()

        # Calculate metrics
        metrics = self._calculate_metrics()

        return {
            'metrics': metrics,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'positions': {k: vars(v) for k, v in self.positions.items()},
        }

    def submit_order(self, order: Order):
        """Submit order to backtesting engine."""
        self.orders.append(order)

    def _process_orders(self):
        """Process pending orders against current bar."""
        for order in self.orders:
            if order.status != "pending":
                continue

            bar = self.current_bar.get(order.symbol)
            if not bar:
                continue

            # Execute order based on type
            if order.order_type == OrderType.MARKET:
                fill_price = self._get_fill_price(bar, order.side)
                self._execute_order(order, fill_price)

            elif order.order_type == OrderType.LIMIT:
                # Check if limit price hit
                if order.side == OrderSide.BUY and bar.low <= order.price:
                    self._execute_order(order, order.price)
                elif order.side == OrderSide.SELL and bar.high >= order.price:
                    self._execute_order(order, order.price)

    def _get_fill_price(self, bar: Bar, side: OrderSide) -> float:
        """Calculate fill price with slippage."""
        base_price = bar.close
        slippage_amount = base_price * self.slippage

        if side == OrderSide.BUY:
            return base_price + slippage_amount
        else:
            return base_price - slippage_amount

    def _execute_order(self, order: Order, fill_price: float):
        """Execute order and update positions."""
        # Calculate commission
        commission_cost = fill_price * order.quantity * self.commission

        # Update capital
        if order.side == OrderSide.BUY:
            cost = fill_price * order.quantity + commission_cost
            self.capital -= cost
        else:
            proceeds = fill_price * order.quantity - commission_cost
            self.capital += proceeds

        # Update position
        self._update_position(order, fill_price)

        # Record trade
        self.trades.append({
            'timestamp': self.current_bar[order.symbol].timestamp,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': fill_price,
            'commission': commission_cost,
            'bar_index': self.bar_index,
        })

        order.status = "filled"
        order.filled_quantity = order.quantity

    def _update_position(self, order: Order, fill_price: float):
        """Update position after order execution."""
        if order.symbol not in self.positions:
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=0.0,
                avg_entry_price=0.0,
                side=order.side,
            )

        pos = self.positions[order.symbol]

        if order.side == OrderSide.BUY:
            # Add to position
            if pos.quantity > 0:
                # Adding to existing long
                total_cost = pos.quantity * pos.avg_entry_price + order.quantity * fill_price
                pos.quantity += order.quantity
                pos.avg_entry_price = total_cost / pos.quantity if pos.quantity > 0 else 0
            else:
                # Opening new long or covering short
                if pos.quantity < 0:
                    # Covering short
                    pnl = (pos.avg_entry_price - fill_price) * min(abs(pos.quantity), order.quantity)
                    pos.realized_pnl += pnl
                    pos.quantity += order.quantity
                else:
                    # Opening new long
                    pos.quantity = order.quantity
                    pos.avg_entry_price = fill_price
            pos.side = OrderSide.BUY

        else:  # SELL
            # Reduce or reverse position
            if pos.quantity > 0:
                # Closing or reducing long position
                pnl = (fill_price - pos.avg_entry_price) * min(pos.quantity, order.quantity)
                pos.realized_pnl += pnl
                pos.quantity -= order.quantity

                if pos.quantity < 0:
                    # Reversed to short
                    pos.avg_entry_price = fill_price
                    pos.side = OrderSide.SELL
            else:
                # Opening or adding to short
                if pos.quantity == 0:
                    pos.quantity = -order.quantity
                    pos.avg_entry_price = fill_price
                else:
                    total_cost = abs(pos.quantity) * pos.avg_entry_price + order.quantity * fill_price
                    pos.quantity -= order.quantity
                    pos.avg_entry_price = total_cost / abs(pos.quantity) if pos.quantity != 0 else 0
                pos.side = OrderSide.SELL

    def _update_positions(self):
        """Update unrealized P&L for all positions."""
        for symbol, pos in self.positions.items():
            if pos.quantity == 0:
                continue

            bar = self.current_bar.get(symbol)
            if not bar:
                continue

            current_price = bar.close
            if pos.quantity > 0:
                # Long position
                pos.unrealized_pnl = (current_price - pos.avg_entry_price) * pos.quantity
            else:
                # Short position
                pos.unrealized_pnl = (pos.avg_entry_price - current_price) * abs(pos.quantity)

    def _record_equity(self):
        """Record current equity."""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        equity = self.capital + total_unrealized_pnl

        timestamp = list(self.current_bar.values())[0].timestamp if self.current_bar else None

        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'capital': self.capital,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
        })

    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics."""
        if not self.equity_curve:
            return {}

        equity_series = pd.Series([e['equity'] for e in self.equity_curve])

        # Total return
        total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital

        # Returns series
        returns = equity_series.pct_change().dropna()

        # Sharpe ratio (annualized, assuming daily bars)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5)
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        if self.trades:
            # Match buy/sell pairs to calculate P&L per trade
            winning_trades = 0
            total_trades = 0

            # Simple win rate based on realized P&L
            for pos in self.positions.values():
                if pos.realized_pnl > 0:
                    winning_trades += 1
                if pos.realized_pnl != 0:
                    total_trades += 1

            win_rate = winning_trades / total_trades if total_trades > 0 else 0
        else:
            win_rate = 0

        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino_ratio = (returns.mean() / negative_returns.std()) * (252 ** 0.5)
        else:
            sortino_ratio = 0.0

        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': len(self.trades),
            'final_equity': float(equity_series.iloc[-1]),
            'total_realized_pnl': float(sum(pos.realized_pnl for pos in self.positions.values())),
        }

    def _create_event_queue(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict]:
        """
        Create chronological event queue from data.

        Args:
            data: Dict of symbol -> DataFrame with OHLCV data
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            List of events with {timestamp, bars, index}
        """
        # Collect all unique timestamps
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index)

        # Sort timestamps
        sorted_timestamps = sorted([t for t in all_timestamps if start_date <= t <= end_date])

        # Create events
        events = []
        for i, timestamp in enumerate(sorted_timestamps):
            bars = {}
            for symbol, df in data.items():
                if timestamp in df.index:
                    row = df.loc[timestamp]
                    bars[symbol] = Bar(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=int(row['volume']) if 'volume' in row else 0,
                    )

            if bars:
                events.append({
                    'timestamp': timestamp,
                    'bars': bars,
                    'index': i,
                })

        return events
