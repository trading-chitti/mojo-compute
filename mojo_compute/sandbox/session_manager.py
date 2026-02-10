"""Session Manager for Sandbox Paper Trading Engine.

Handles creating, querying, and managing paper trading sessions
including lifecycle transitions and real-time stats updates.
"""

import logging
import uuid
from datetime import datetime, date
from typing import Optional

import psycopg2
import psycopg2.extras

from . import config

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages sandbox paper trading sessions.

    Provides CRUD operations and lifecycle management for trading sessions
    stored in the sandbox.sessions table.
    """

    def __init__(self):
        self._dsn = config.PG_DSN

    def _get_conn(self):
        """Create a new database connection."""
        return psycopg2.connect(self._dsn)

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(
        self,
        initial_capital: float,
        leverage: float = config.DEFAULT_LEVERAGE,
        max_position_pct: float = config.DEFAULT_MAX_POSITION_PCT,
        max_open_positions: int = config.DEFAULT_MAX_OPEN_POSITIONS,
        max_drawdown_pct: float = config.DEFAULT_MAX_DAILY_DRAWDOWN_PCT,
        max_sector_exposure_pct: float = config.DEFAULT_MAX_SECTOR_EXPOSURE_PCT,
    ) -> dict:
        """Create a new paper trading session.

        Args:
            initial_capital: Starting capital in INR.
            leverage: Leverage multiplier (e.g. 10.0 for 10x).
            max_position_pct: Maximum % of capital in a single position.
            max_open_positions: Maximum number of concurrent open trades.
            max_drawdown_pct: Maximum daily drawdown % before auto-stop.
            max_sector_exposure_pct: Maximum sector exposure %.

        Returns:
            dict with the newly created session row.
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(config.IST)
        session_date = now.date()

        sql = """
            INSERT INTO sandbox.sessions (
                session_id, session_date,
                initial_capital, current_capital, peak_capital,
                leverage_multiplier,
                max_position_pct, max_daily_drawdown_pct, max_sector_exposure_pct,
                max_open_positions,
                status,
                total_trades, winning_trades, losing_trades,
                total_pnl, total_pnl_pct, max_drawdown_pct,
                signals_received, signals_skipped,
                created_at, updated_at
            ) VALUES (
                %s, %s,
                %s, %s, %s,
                %s,
                %s, %s, %s,
                %s,
                'ACTIVE',
                0, 0, 0,
                0.00, 0.0000, 0.0000,
                0, 0,
                %s, %s
            )
            RETURNING *
        """

        params = (
            session_id, session_date,
            initial_capital, initial_capital, initial_capital,
            leverage,
            max_position_pct, max_drawdown_pct, max_sector_exposure_pct,
            max_open_positions,
            now, now,
        )

        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    conn.commit()
                    logger.info(
                        "Created session %s | capital=%.2f leverage=%.1f",
                        session_id, initial_capital, leverage,
                    )
                    return dict(row)
        except Exception:
            logger.exception("Failed to create session")
            raise

    def get_active_session(self) -> Optional[dict]:
        """Return the most recently created ACTIVE session, or None."""
        sql = """
            SELECT *
              FROM sandbox.sessions
             WHERE status = 'ACTIVE'
             ORDER BY created_at DESC
             LIMIT 1
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql)
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception:
            logger.exception("Failed to get active session")
            raise

    def get_all_sessions(self) -> list:
        """Return all sessions ordered by creation time (newest first)."""
        sql = "SELECT * FROM sandbox.sessions ORDER BY created_at DESC"
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql)
                    return [dict(r) for r in cur.fetchall()]
        except Exception:
            logger.exception("Failed to get all sessions")
            return []

    def get_session(self, session_id: str) -> Optional[dict]:
        """Return a specific session by its UUID."""
        sql = "SELECT * FROM sandbox.sessions WHERE session_id = %s"
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (session_id,))
                    row = cur.fetchone()
                    if row is None:
                        logger.warning("Session %s not found", session_id)
                    return dict(row) if row else None
        except Exception:
            logger.exception("Failed to get session %s", session_id)
            raise

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def _transition_status(
        self,
        session_id: str,
        from_status: str,
        to_status: str,
        stop_reason: Optional[str] = None,
    ) -> bool:
        """Atomically transition a session from one status to another.

        Returns True if exactly one row was updated, False otherwise.
        """
        now = datetime.now(config.IST)

        if stop_reason is not None:
            sql = """
                UPDATE sandbox.sessions
                   SET status = %s,
                       stop_reason = %s,
                       updated_at = %s
                 WHERE session_id = %s
                   AND status = %s
            """
            params = (to_status, stop_reason, now, session_id, from_status)
        else:
            sql = """
                UPDATE sandbox.sessions
                   SET status = %s,
                       updated_at = %s
                 WHERE session_id = %s
                   AND status = %s
            """
            params = (to_status, now, session_id, from_status)

        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    updated = cur.rowcount
                    conn.commit()
                    if updated == 1:
                        logger.info(
                            "Session %s transitioned %s -> %s",
                            session_id, from_status, to_status,
                        )
                        return True
                    else:
                        logger.warning(
                            "Session %s transition %s -> %s failed (rowcount=%d)",
                            session_id, from_status, to_status, updated,
                        )
                        return False
        except Exception:
            logger.exception(
                "Error transitioning session %s %s -> %s",
                session_id, from_status, to_status,
            )
            raise

    def pause_session(self, session_id: str) -> bool:
        """Pause an ACTIVE session."""
        return self._transition_status(session_id, "ACTIVE", "PAUSED")

    def resume_session(self, session_id: str) -> bool:
        """Resume a PAUSED session back to ACTIVE."""
        return self._transition_status(session_id, "PAUSED", "ACTIVE")

    def stop_session(self, session_id: str, reason: str) -> bool:
        """Stop an ACTIVE or PAUSED session with a reason."""
        # Try from ACTIVE first, then from PAUSED
        if self._transition_status(session_id, "ACTIVE", "STOPPED", stop_reason=reason):
            return True
        return self._transition_status(session_id, "PAUSED", "STOPPED", stop_reason=reason)

    def complete_session(self, session_id: str) -> bool:
        """Mark an ACTIVE session as COMPLETED (end of market day)."""
        return self._transition_status(session_id, "ACTIVE", "COMPLETED")

    # ------------------------------------------------------------------
    # Stats updates
    # ------------------------------------------------------------------

    def update_session_stats(
        self,
        session_id: str,
        current_capital: float,
        peak_capital: float,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        total_pnl_pct: float,
        max_drawdown_pct: float,
        signals_received: int,
        signals_skipped: int,
    ) -> bool:
        """Bulk-update session statistics.

        Called periodically by the engine after trades close or equity changes.

        Returns True if the row was updated.
        """
        now = datetime.now(config.IST)

        sql = """
            UPDATE sandbox.sessions
               SET current_capital   = %s,
                   peak_capital      = %s,
                   total_trades      = %s,
                   winning_trades    = %s,
                   losing_trades     = %s,
                   total_pnl         = %s,
                   total_pnl_pct     = %s,
                   max_drawdown_pct  = %s,
                   signals_received  = %s,
                   signals_skipped   = %s,
                   updated_at        = %s
             WHERE session_id = %s
        """

        params = (
            current_capital, peak_capital,
            total_trades, winning_trades, losing_trades,
            total_pnl, total_pnl_pct, max_drawdown_pct,
            signals_received, signals_skipped,
            now, session_id,
        )

        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    updated = cur.rowcount
                    conn.commit()
                    if updated == 1:
                        logger.debug(
                            "Updated stats for session %s | capital=%.2f pnl=%.2f trades=%d",
                            session_id, current_capital, total_pnl, total_trades,
                        )
                        return True
                    else:
                        logger.warning("Stats update for session %s matched 0 rows", session_id)
                        return False
        except Exception:
            logger.exception("Failed to update stats for session %s", session_id)
            raise

    # ------------------------------------------------------------------
    # Trade-level queries
    # ------------------------------------------------------------------

    def get_open_trade_count(self, session_id: str) -> int:
        """Return the number of currently open trades for a session."""
        sql = """
            SELECT COUNT(*) AS cnt
              FROM sandbox.trades
             WHERE session_id = %s
               AND status = 'OPEN'
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (session_id,))
                    row = cur.fetchone()
                    return int(row["cnt"]) if row else 0
        except Exception:
            logger.exception("Failed to get open trade count for session %s", session_id)
            raise

    def get_allocated_capital(self, session_id: str) -> float:
        """Return total capital allocated to open trades for a session."""
        sql = """
            SELECT COALESCE(SUM(allocated_capital), 0) AS total_allocated
              FROM sandbox.trades
             WHERE session_id = %s
               AND status = 'OPEN'
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (session_id,))
                    row = cur.fetchone()
                    return float(row["total_allocated"]) if row else 0.0
        except Exception:
            logger.exception("Failed to get allocated capital for session %s", session_id)
            raise
