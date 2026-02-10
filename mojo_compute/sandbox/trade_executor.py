"""Trade Executor - Paper trade execution with Mojo-accelerated P&L.

Opens new trades based on ML decisions, monitors open positions for
target/stoploss hits, and closes trades with realized P&L calculation.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras

from . import config
from .mojo_wrappers import (
    calculate_quantity,
    calculate_position_value,
    calculate_unrealized_pnl,
    calculate_unrealized_pnl_pct,
    calculate_realized_pnl,
    calculate_realized_pnl_pct,
    check_target_hit,
    check_stoploss_hit,
)

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Handles paper trade lifecycle: open, update, close."""

    def __init__(self):
        self._dsn = config.PG_DSN

    def _get_conn(self):
        return psycopg2.connect(self._dsn)

    # ------------------------------------------------------------------
    # Open a new trade
    # ------------------------------------------------------------------

    def open_trade(
        self,
        session_id: str,
        signal: Dict[str, Any],
        allocated_capital: float,
        leverage: float,
        ml_confidence: float,
    ) -> Optional[Dict[str, Any]]:
        """Open a new paper trade from a signal.

        Args:
            session_id: UUID of the active session.
            signal: Signal dict from intraday-engine WebSocket.
            allocated_capital: INR allocated by ML money manager.
            leverage: Leverage multiplier (e.g., 10.0).
            ml_confidence: ML model's confidence score.

        Returns:
            dict of the newly created trade row, or None on failure.
        """
        entry_price = float(signal.get("entry_price", 0))
        if entry_price <= 0:
            logger.warning("Cannot open trade: entry_price <= 0 for %s", signal.get("symbol"))
            return None

        signal_type = signal.get("signal_type", "CALL")
        position_value = calculate_position_value(allocated_capital, leverage)
        quantity = calculate_quantity(allocated_capital, leverage, entry_price)

        if quantity <= 0:
            logger.warning("Cannot open trade: quantity=0 for %s at %.2f", signal.get("symbol"), entry_price)
            return None

        trade_id = str(uuid.uuid4())
        signal_id = signal.get("signal_id", str(uuid.uuid4()))

        sql = """
            INSERT INTO sandbox.trades (
                trade_id, session_id, signal_id,
                symbol, stock_name, sector, signal_type,
                entry_price, entry_time,
                allocated_capital, leverage_used, position_value, quantity,
                target_price, stop_loss,
                current_price, unrealized_pnl, unrealized_pnl_pct,
                ml_confidence, signal_confidence,
                status
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, NOW(),
                %s, %s, %s, %s,
                %s, %s,
                %s, 0, 0,
                %s, %s,
                'OPEN'
            )
            RETURNING *
        """

        params = (
            trade_id, session_id, signal_id,
            signal.get("symbol", ""),
            signal.get("stock_name", signal.get("symbol", "")),
            signal.get("sector", ""),
            signal_type,
            entry_price,
            allocated_capital, leverage, position_value, quantity,
            float(signal.get("target_price", 0)),
            float(signal.get("stop_loss", 0)),
            entry_price,
            ml_confidence,
            float(signal.get("confidence", 0)),
        )

        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    conn.commit()
                    logger.info(
                        "📈 OPENED trade %s: %s %s @ %.2f | qty=%d | capital=%.0f | leverage=%.1fx",
                        trade_id[:8], signal_type, signal.get("symbol"),
                        entry_price, quantity, allocated_capital, leverage,
                    )
                    return dict(row)
        except Exception:
            logger.exception("Failed to open trade for %s", signal.get("symbol"))
            return None

    # ------------------------------------------------------------------
    # Update open positions with current price
    # ------------------------------------------------------------------

    def update_position(self, trade: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Update an open position with the latest price.

        Returns updated trade dict with unrealized P&L.
        """
        signal_type = trade["signal_type"]
        entry_price = float(trade["entry_price"])
        quantity = int(trade["quantity"])

        unrealized = calculate_unrealized_pnl(signal_type, entry_price, current_price, quantity)
        unrealized_pct = calculate_unrealized_pnl_pct(signal_type, entry_price, current_price)

        sql = """
            UPDATE sandbox.trades
            SET current_price = %s,
                unrealized_pnl = %s,
                unrealized_pnl_pct = %s,
                updated_at = NOW()
            WHERE trade_id = %s AND status = 'OPEN'
        """

        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (current_price, unrealized, unrealized_pct, trade["trade_id"]))
                    conn.commit()

            trade["current_price"] = current_price
            trade["unrealized_pnl"] = unrealized
            trade["unrealized_pnl_pct"] = unrealized_pct
            return trade
        except Exception:
            logger.exception("Failed to update position %s", trade["trade_id"])
            return trade

    # ------------------------------------------------------------------
    # Check if a trade should be closed
    # ------------------------------------------------------------------

    def check_exit_conditions(self, trade: Dict[str, Any], current_price: float) -> Optional[str]:
        """Check if a trade should be closed.

        Checks in order: target hit, trailing stop, time-based exit, stoploss hit.
        Returns exit_reason string or None if no exit condition met.
        """
        signal_type = trade["signal_type"]
        entry_price = float(trade["entry_price"])
        target_price = float(trade["target_price"])
        stop_loss = float(trade["stop_loss"])

        # 1. Check target hit first (highest priority)
        if check_target_hit(signal_type, current_price, target_price):
            return "HIT_TARGET"

        # 2. Apply trailing stop logic (updates trade dict in-place)
        self._apply_trailing_stop(trade, current_price)

        # 3. Check trailing stop hit
        trailing_sl = trade.get("trailing_stop_price")
        if trailing_sl and trade.get("trailing_stop_activated"):
            trailing_sl = float(trailing_sl)
            if check_stoploss_hit(signal_type, current_price, trailing_sl):
                return "TRAILING_STOP"

        # 4. Time-based exit (prevent stale trades)
        if self._check_time_exit(trade):
            return "TIME_EXIT"

        # 5. Original stoploss check
        if check_stoploss_hit(signal_type, current_price, stop_loss):
            return "HIT_STOPLOSS"

        return None

    def _apply_trailing_stop(self, trade: Dict[str, Any], current_price: float) -> None:
        """Apply trailing stop logic to protect profits.

        Stage 1: After 1% unrealized profit -> move SL to breakeven (entry +/- 0.1%)
        Stage 2: After 2% unrealized profit -> trail at 50% of max profit

        Updates trade dict in-place with trailing stop fields.
        """
        signal_type = trade["signal_type"]
        entry_price = float(trade["entry_price"])

        # Track highest/lowest price since entry
        highest = float(trade.get("highest_price_since_entry") or entry_price)
        lowest = float(trade.get("lowest_price_since_entry") or entry_price)

        if signal_type == "CALL":
            highest = max(highest, current_price)
            max_profit_pct = ((highest - entry_price) / entry_price) * 100
        else:  # PUT
            lowest = min(lowest, current_price)
            max_profit_pct = ((entry_price - lowest) / entry_price) * 100

        if max_profit_pct <= 0:
            return

        trailing_stop_price = trade.get("trailing_stop_price")
        trailing_activated = bool(trade.get("trailing_stop_activated", False))
        breakeven_reached = bool(trade.get("breakeven_reached", False))

        # Stage 1: Breakeven after 1% profit
        if max_profit_pct >= 1.0 and not breakeven_reached:
            breakeven_reached = True
            trailing_activated = True
            buffer = entry_price * 0.001  # 0.1% buffer
            if signal_type == "CALL":
                trailing_stop_price = entry_price + buffer
            else:
                trailing_stop_price = entry_price - buffer

        # Stage 2: Trail at 50% of max profit after 2% profit
        if max_profit_pct >= 2.0:
            trailing_activated = True
            trail_amount = (max_profit_pct * 0.5 / 100) * entry_price
            if signal_type == "CALL":
                new_trail = highest - trail_amount
                existing = float(trailing_stop_price) if trailing_stop_price else 0
                trailing_stop_price = max(new_trail, existing)
            else:
                new_trail = lowest + trail_amount
                existing = float(trailing_stop_price) if trailing_stop_price else float('inf')
                trailing_stop_price = min(new_trail, existing)

        # Persist state on trade dict
        trade["trailing_stop_activated"] = trailing_activated
        trade["breakeven_reached"] = breakeven_reached
        trade["highest_price_since_entry"] = highest
        trade["lowest_price_since_entry"] = lowest
        if trailing_stop_price is not None:
            trade["trailing_stop_price"] = trailing_stop_price

    def _check_time_exit(self, trade: Dict[str, Any]) -> bool:
        """Check if trade has exceeded maximum hold time (default: 120 min)."""
        entry_time = trade.get("entry_time")
        if not entry_time:
            return False

        now = datetime.now(config.IST)
        if isinstance(entry_time, str):
            from dateutil import parser as dt_parser
            entry_time = dt_parser.parse(entry_time)

        elapsed_minutes = (now - entry_time).total_seconds() / 60
        max_hold = float(trade.get("max_hold_minutes", 120))

        if elapsed_minutes > max_hold:
            logger.info(
                "⏰ TIME_EXIT: %s held for %.0f min (max=%d)",
                trade.get("symbol", "?"), elapsed_minutes, int(max_hold)
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Close a trade
    # ------------------------------------------------------------------

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[Dict[str, Any]]:
        """Close a trade and calculate realized P&L.

        Args:
            trade_id: UUID of the trade to close.
            exit_price: Price at which the trade is closed.
            exit_reason: One of HIT_TARGET, HIT_STOPLOSS, EXPIRED, MANUAL_CLOSE, etc.

        Returns:
            Updated trade dict with realized P&L, or None on failure.
        """
        # Fetch the trade first
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM sandbox.trades WHERE trade_id = %s AND status = 'OPEN'",
                        (trade_id,)
                    )
                    trade = cur.fetchone()
                    if not trade:
                        logger.warning("Trade %s not found or already closed", trade_id)
                        return None
                    trade = dict(trade)
        except Exception:
            logger.exception("Failed to fetch trade %s for closing", trade_id)
            return None

        signal_type = trade["signal_type"]
        entry_price = float(trade["entry_price"])
        quantity = int(trade["quantity"])

        realized = calculate_realized_pnl(signal_type, entry_price, exit_price, quantity)
        realized_pct = calculate_realized_pnl_pct(signal_type, entry_price, exit_price)

        entry_time = trade["entry_time"]
        now = datetime.now(config.IST)
        holding_minutes = int((now - entry_time).total_seconds() / 60) if entry_time else 0

        sql = """
            UPDATE sandbox.trades
            SET exit_price = %s,
                exit_time = NOW(),
                exit_reason = %s,
                realized_pnl = %s,
                realized_pnl_pct = %s,
                holding_duration_minutes = %s,
                current_price = %s,
                unrealized_pnl = 0,
                unrealized_pnl_pct = 0,
                status = 'CLOSED',
                updated_at = NOW()
            WHERE trade_id = %s AND status = 'OPEN'
            RETURNING *
        """

        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (
                        exit_price, exit_reason, realized, realized_pct,
                        holding_minutes, exit_price, trade_id
                    ))
                    row = cur.fetchone()
                    conn.commit()

                    if row:
                        icon = "✅" if realized > 0 else "❌"
                        logger.info(
                            "%s CLOSED trade %s: %s %s | entry=%.2f exit=%.2f | P&L=%.2f (%.2f%%) | %s | %dm",
                            icon, trade_id[:8], signal_type, trade["symbol"],
                            entry_price, exit_price, realized, realized_pct,
                            exit_reason, holding_minutes,
                        )
                        return dict(row)
                    return None
        except Exception:
            logger.exception("Failed to close trade %s", trade_id)
            return None

    # ------------------------------------------------------------------
    # Close trade by signal_id (when intraday-engine sends SIGNAL_CLOSED)
    # ------------------------------------------------------------------

    def close_trade_by_signal(
        self,
        signal_id: str,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[Dict[str, Any]]:
        """Close an open trade matching the given signal_id."""
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT trade_id FROM sandbox.trades WHERE signal_id = %s AND status = 'OPEN'",
                        (signal_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    return self.close_trade(str(row["trade_id"]), exit_price, exit_reason)
        except Exception:
            logger.exception("Failed to find trade for signal %s", signal_id)
            return None

    # ------------------------------------------------------------------
    # Bulk close (session end / max drawdown)
    # ------------------------------------------------------------------

    def close_all_open_trades(self, session_id: str, exit_reason: str) -> List[Dict[str, Any]]:
        """Close all open trades for a session."""
        open_trades = self.get_open_trades(session_id)
        closed = []
        for trade in open_trades:
            current_price = float(trade.get("current_price") or trade["entry_price"])
            result = self.close_trade(str(trade["trade_id"]), current_price, exit_reason)
            if result:
                closed.append(result)
        logger.info("Closed %d trades for session %s (%s)", len(closed), session_id[:8], exit_reason)
        return closed

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_open_trades(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all open trades for a session."""
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM sandbox.trades
                        WHERE session_id = %s AND status = 'OPEN'
                        ORDER BY entry_time ASC
                    """, (session_id,))
                    return [dict(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to get open trades for session %s", session_id)
            return []

    def get_trades(
        self,
        session_id: str,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get trades for a session, optionally filtered by status."""
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    if status:
                        cur.execute("""
                            SELECT * FROM sandbox.trades
                            WHERE session_id = %s AND status = %s
                            ORDER BY created_at DESC LIMIT %s
                        """, (session_id, status, limit))
                    else:
                        cur.execute("""
                            SELECT * FROM sandbox.trades
                            WHERE session_id = %s
                            ORDER BY created_at DESC LIMIT %s
                        """, (session_id, limit))
                    return [dict(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to get trades for session %s", session_id)
            return []

    def get_total_unrealized_pnl(self, session_id: str) -> float:
        """Get total unrealized P&L across all open positions."""
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COALESCE(SUM(unrealized_pnl), 0)
                        FROM sandbox.trades
                        WHERE session_id = %s AND status = 'OPEN'
                    """, (session_id,))
                    return float(cur.fetchone()[0])
        except Exception:
            logger.exception("Failed to get unrealized P&L for session %s", session_id)
            return 0.0

    # ------------------------------------------------------------------
    # ML Decision Logging
    # ------------------------------------------------------------------

    def log_ml_decision(
        self,
        session_id: str,
        signal_id: str,
        decision: str,
        ml_score: float,
        position_size_pct: Optional[float] = None,
        position_size_inr: Optional[float] = None,
        features: Optional[dict] = None,
        reasoning: Optional[dict] = None,
        trade_id: Optional[str] = None,
    ) -> None:
        """Log an ML decision (TRADE or SKIP) with reasoning."""
        import json

        sql = """
            INSERT INTO sandbox.ml_decisions (
                session_id, signal_id, decision, trade_id,
                ml_score, position_size_pct, position_size_inr,
                features, reasoning
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (
                        session_id, signal_id, decision, trade_id,
                        ml_score, position_size_pct, position_size_inr,
                        json.dumps(features or {}),
                        json.dumps(reasoning or {}),
                    ))
                    conn.commit()
        except Exception:
            logger.exception("Failed to log ML decision for signal %s", signal_id)

    def get_ml_decisions(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get ML decisions for a session."""
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM sandbox.ml_decisions
                        WHERE session_id = %s
                        ORDER BY created_at DESC LIMIT %s
                    """, (session_id, limit))
                    return [dict(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to get ML decisions for session %s", session_id)
            return []

    # ------------------------------------------------------------------
    # Equity Snapshots
    # ------------------------------------------------------------------

    def save_equity_snapshot(
        self,
        session_id: str,
        capital: float,
        unrealized_pnl: float,
        total_equity: float,
        open_positions: int,
        drawdown_pct: float,
        baseline_equity: Optional[float] = None,
        baseline_pnl: Optional[float] = None,
    ) -> None:
        """Save an equity snapshot for the equity curve."""
        sql = """
            INSERT INTO sandbox.equity_snapshots (
                session_id, capital, unrealized_pnl, total_equity,
                open_positions, drawdown_pct,
                baseline_equity, baseline_pnl
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (
                        session_id, capital, unrealized_pnl, total_equity,
                        open_positions, drawdown_pct,
                        baseline_equity, baseline_pnl,
                    ))
                    conn.commit()
        except Exception:
            logger.exception("Failed to save equity snapshot")

    def get_equity_curve(self, session_id: str) -> List[Dict[str, Any]]:
        """Get equity curve data for a session."""
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM sandbox.equity_snapshots
                        WHERE session_id = %s
                        ORDER BY snapshot_time ASC
                    """, (session_id,))
                    return [dict(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to get equity curve for session %s", session_id)
            return []
