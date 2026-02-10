"""Sandbox Engine Server - FastAPI app with WebSocket broadcaster and REST endpoints.

Connects to intraday-engine via WebSocket, processes signals through ML money manager,
executes paper trades, and broadcasts updates to the dashboard.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config
from .session_manager import SessionManager
from .signal_consumer import SignalConsumer
from .trade_executor import TradeExecutor
from .money_manager import MoneyManager
from .baseline_tracker import BaselineTracker
from .mojo_wrappers import (
    kelly_position_size_pct,
    current_drawdown,
    can_open_trade,
    get_rejection_reason,
    win_rate,
    profit_factor,
    max_drawdown as calc_max_drawdown,
    total_pnl_pct,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

def convert_for_json(obj: Any) -> Any:
    """Recursively convert Decimal, datetime, UUID to JSON-safe types."""
    import uuid as _uuid
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, _uuid.UUID):
        return str(obj)
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_for_json(i) for i in obj]
    return obj

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    initial_capital: float = 10000.0
    leverage: float = config.DEFAULT_LEVERAGE
    max_position_pct: float = config.DEFAULT_MAX_POSITION_PCT
    max_open_positions: int = config.DEFAULT_MAX_OPEN_POSITIONS
    max_daily_drawdown_pct: float = config.DEFAULT_MAX_DAILY_DRAWDOWN_PCT
    max_sector_exposure_pct: float = config.DEFAULT_MAX_SECTOR_EXPOSURE_PCT

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

session_mgr = SessionManager()
trade_exec = TradeExecutor()
signal_consumer = SignalConsumer()
money_mgr = MoneyManager()

# Baseline tracker (initialized per session)
_baseline: Optional[BaselineTracker] = None

# WebSocket broadcaster: connected dashboard clients
ws_clients: Set[WebSocket] = set()

# Equity snapshot task
_equity_task: Optional[asyncio.Task] = None

# ---------------------------------------------------------------------------
# WebSocket broadcaster
# ---------------------------------------------------------------------------

async def broadcast(msg_type: str, data: Dict[str, Any]) -> None:
    """Broadcast a message to all connected dashboard WebSocket clients."""
    if not ws_clients:
        return
    message = json.dumps(convert_for_json({"type": msg_type, **data}))
    dead: Set[WebSocket] = set()
    for ws in ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    for d in dead:
        ws_clients.discard(d)

# ---------------------------------------------------------------------------
# Signal handlers (called by SignalConsumer)
# ---------------------------------------------------------------------------

async def on_new_signal(signal: Dict[str, Any]) -> None:
    """Handle a new trading signal from intraday-engine.

    ML Money Manager pipeline:
    1. Risk guard rails (Mojo) - position limits, drawdown, capital
    2. ML evaluation - 25 features → score → TRADE or SKIP
    3. Kelly position sizing (Mojo) - confidence-aware allocation
    4. Execute paper trade
    5. Baseline tracker - track "trade all" benchmark
    """
    global _baseline

    session = session_mgr.get_active_session()
    if not session:
        return

    session_id = str(session["session_id"])
    signal_id = signal.get("signal_id", signal.get("id", ""))

    # Track in baseline (trade ALL signals for comparison)
    if _baseline is not None:
        _baseline.on_new_signal(signal)

    # Increment signals_received
    signals_received = session["signals_received"] + 1

    # ── Step 1: Risk guard rails (Mojo-accelerated) ──
    current_capital = float(session["current_capital"])
    peak_capital = float(session["peak_capital"])
    dd_pct = current_drawdown(current_capital, peak_capital)
    allocated = session_mgr.get_allocated_capital(session_id)
    open_count = session_mgr.get_open_trade_count(session_id)
    available = current_capital - allocated
    sector_exp = 0.0  # TODO: per-sector tracking

    trade_allowed = can_open_trade(
        dd_pct,
        float(session["max_daily_drawdown_pct"]),
        open_count,
        session["max_open_positions"],
        sector_exp,
        float(session["max_sector_exposure_pct"]),
        available,
        config.MIN_TRADE_CAPITAL,
        True,
    )

    if not trade_allowed:
        reason = get_rejection_reason(
            dd_pct,
            float(session["max_daily_drawdown_pct"]),
            open_count,
            session["max_open_positions"],
            sector_exp,
            float(session["max_sector_exposure_pct"]),
            available,
            config.MIN_TRADE_CAPITAL,
            True,
        )
        logger.info("⏭️ SKIP %s %s: %s (risk guard)", signal.get("symbol"), signal.get("signal_type"), reason)
        trade_exec.log_ml_decision(
            session_id, signal_id, "SKIP", 0.0,
            reasoning={"skip_reason": reason, "stage": "risk_guard"},
        )
        _update_session_counts(session_id, session, signals_received, session["signals_skipped"] + 1)
        await broadcast("SIGNAL_SKIPPED", {
            "symbol": signal.get("symbol"),
            "signal_type": signal.get("signal_type"),
            "reason": reason,
        })
        return

    # ── Step 2: ML Money Manager evaluation ──
    # Extracts 25 features (confidence, RSI, MACD, volume, R:R, drawdown,
    # win rate, time-of-day, etc.) and scores the signal.
    # High-confidence CALL with good R:R → high score → large position
    # Low-confidence PUT in drawdown → low score → SKIP
    decision, ml_score, position_pct, leverage, features, reasoning = money_mgr.evaluate_signal(
        signal, session, open_count, allocated,
    )

    if decision == "SKIP":
        logger.info(
            "⏭️ SKIP %s %s: ML score %.2f < threshold %.2f | R:R=%.1f conf=%.2f",
            signal.get("symbol"), signal.get("signal_type"),
            ml_score, money_mgr.trade_threshold,
            features.get("risk_reward", 0), features.get("confidence", 0),
        )
        trade_exec.log_ml_decision(
            session_id, signal_id, "SKIP", ml_score,
            features=features, reasoning=reasoning,
        )
        _update_session_counts(session_id, session, signals_received, session["signals_skipped"] + 1)
        await broadcast("SIGNAL_SKIPPED", {
            "symbol": signal.get("symbol"),
            "signal_type": signal.get("signal_type"),
            "reason": f"ML score {ml_score:.2f} < {money_mgr.trade_threshold}",
            "ml_score": ml_score,
            "factors": reasoning.get("factors", []),
        })
        return

    # ── Step 3: Calculate allocation (ML-decided) ──
    # position_pct varies by signal quality:
    #   - High confidence + good R:R → up to max_position_pct (e.g. 20%)
    #   - Medium confidence → ~10-12%
    #   - Low confidence but above threshold → min 5%
    #   - During drawdown → automatically reduced by Kelly dampening
    # leverage varies by ML score:
    #   - Score 0.55 → ~30% of max leverage (e.g. 3x of 10x)
    #   - Score 0.70 → ~65% of max leverage (e.g. 6.5x)
    #   - Score 0.85+ → full leverage (e.g. 10x)
    #   - PUT signals capped at 70% of max leverage
    #   - During drawdown → leverage further reduced
    allocated_capital = current_capital * (position_pct / 100.0)
    max_leverage = float(session["leverage_multiplier"])

    logger.info(
        "🧠 ML TRADE %s %s: score=%.2f → %.1f%% (%.0f INR) @ %.1fx leverage (max %.1fx) | R:R=%.1f conf=%.2f vol=%.1fx rsi=%.0f",
        signal.get("symbol"), signal.get("signal_type"),
        ml_score, position_pct, allocated_capital,
        leverage, max_leverage,
        features.get("risk_reward", 0),
        features.get("confidence", 0),
        features.get("volume_ratio", 1),
        features.get("rsi", 50),
    )

    # ── Step 4: Execute paper trade ──
    trade = trade_exec.open_trade(
        session_id, signal, allocated_capital, leverage, ml_score,
    )

    if trade:
        trade_exec.log_ml_decision(
            session_id, signal_id, "TRADE", ml_score,
            position_size_pct=position_pct,
            position_size_inr=allocated_capital,
            features=features,
            reasoning=reasoning,
            trade_id=str(trade["trade_id"]),
        )
        _update_session_counts(session_id, session, signals_received, session["signals_skipped"])
        await broadcast("NEW_TRADE", {"trade": convert_for_json(trade)})
    else:
        logger.warning("Failed to open trade for %s", signal.get("symbol"))


def _update_session_counts(session_id: str, session: dict, received: int, skipped: int) -> None:
    """Helper to update signal received/skipped counters."""
    session_mgr.update_session_stats(
        session_id,
        float(session["current_capital"]),
        float(session["peak_capital"]),
        session["total_trades"],
        session["winning_trades"],
        session["losing_trades"],
        float(session["total_pnl"]),
        float(session["total_pnl_pct"]),
        float(session["max_drawdown_pct"]),
        received,
        skipped,
    )


async def on_signal_closed(signal: Dict[str, Any], result: str) -> None:
    """Handle a signal being closed (target/stoploss hit)."""
    global _baseline

    session = session_mgr.get_active_session()
    if not session:
        return

    # Track in baseline
    if _baseline is not None:
        _baseline.on_signal_closed(signal, result)

    session_id = str(session["session_id"])
    signal_id = signal.get("signal_id", signal.get("id", ""))
    exit_price = float(signal.get("exit_price", signal.get("current_price", 0)))

    exit_reason = "HIT_TARGET" if result == "TARGET_HIT" else "HIT_STOPLOSS"
    closed_trade = trade_exec.close_trade_by_signal(signal_id, exit_price, exit_reason)

    if closed_trade:
        realized_pnl = float(closed_trade["realized_pnl"])

        # Update session capital
        session_mgr.update_session_stats(
            session_id,
            float(session["current_capital"]) + realized_pnl,
            max(float(session["peak_capital"]), float(session["current_capital"]) + realized_pnl),
            session["total_trades"] + 1,
            session["winning_trades"] + (1 if realized_pnl > 0 else 0),
            session["losing_trades"] + (1 if realized_pnl <= 0 else 0),
            float(session["total_pnl"]) + realized_pnl,
            total_pnl_pct(float(session["current_capital"]) + realized_pnl, float(session["initial_capital"])),
            float(session["max_drawdown_pct"]),  # will recalc in equity snapshot
            session["signals_received"],
            session["signals_skipped"],
        )

        await broadcast("TRADE_CLOSED", {"trade": convert_for_json(closed_trade)})

        # Check for max drawdown auto-stop
        new_capital = float(session["current_capital"]) + realized_pnl
        dd = current_drawdown(new_capital, float(session["peak_capital"]))
        if dd >= float(session["max_daily_drawdown_pct"]):
            logger.warning("⚠️ MAX DRAWDOWN reached (%.1f%%) - stopping session", dd)
            trade_exec.close_all_open_trades(session_id, "MAX_DRAWDOWN")
            session_mgr.stop_session(session_id, f"Max drawdown reached: {dd:.1f}%")
            await broadcast("SESSION_STOPPED", {"reason": f"Max drawdown reached: {dd:.1f}%"})


async def on_signal_update(signal: Dict[str, Any]) -> None:
    """Handle a signal price update - update open positions."""
    session = session_mgr.get_active_session()
    if not session:
        return

    session_id = str(session["session_id"])
    open_trades = trade_exec.get_open_trades(session_id)
    symbol = signal.get("symbol", "")
    current_price = float(signal.get("current_price", 0))

    if current_price <= 0:
        return

    for trade in open_trades:
        if trade["symbol"] == symbol:
            # Check exit conditions first
            exit_reason = trade_exec.check_exit_conditions(trade, current_price)
            if exit_reason:
                closed = trade_exec.close_trade(str(trade["trade_id"]), current_price, exit_reason)
                if closed:
                    realized_pnl = float(closed["realized_pnl"])
                    session = session_mgr.get_active_session()  # refresh
                    if session:
                        session_mgr.update_session_stats(
                            session_id,
                            float(session["current_capital"]) + realized_pnl,
                            max(float(session["peak_capital"]), float(session["current_capital"]) + realized_pnl),
                            session["total_trades"] + 1,
                            session["winning_trades"] + (1 if realized_pnl > 0 else 0),
                            session["losing_trades"] + (1 if realized_pnl <= 0 else 0),
                            float(session["total_pnl"]) + realized_pnl,
                            total_pnl_pct(float(session["current_capital"]) + realized_pnl, float(session["initial_capital"])),
                            float(session["max_drawdown_pct"]),
                            session["signals_received"],
                            session["signals_skipped"],
                        )
                    await broadcast("TRADE_CLOSED", {"trade": convert_for_json(closed)})
            else:
                # Update unrealized P&L
                updated = trade_exec.update_position(trade, current_price)
                await broadcast("TRADE_UPDATE", {"trade": convert_for_json(updated)})


# ---------------------------------------------------------------------------
# Equity snapshot task
# ---------------------------------------------------------------------------

async def equity_snapshot_loop() -> None:
    """Periodically save equity snapshots for the equity curve."""
    while True:
        await asyncio.sleep(config.EQUITY_SNAPSHOT_INTERVAL)
        try:
            session = session_mgr.get_active_session()
            if not session:
                continue

            session_id = str(session["session_id"])
            capital = float(session["current_capital"])
            peak = float(session["peak_capital"])
            unrealized = trade_exec.get_total_unrealized_pnl(session_id)
            total_equity = capital + unrealized
            open_count = session_mgr.get_open_trade_count(session_id)
            dd = current_drawdown(total_equity, peak)

            baseline_equity = _baseline.equity if _baseline else None
            baseline_pnl = _baseline.pnl if _baseline else None

            trade_exec.save_equity_snapshot(
                session_id, capital, unrealized, total_equity,
                open_count, dd,
                baseline_equity=baseline_equity,
                baseline_pnl=baseline_pnl,
            )

            await broadcast("EQUITY_UPDATE", {
                "capital": capital,
                "unrealized_pnl": unrealized,
                "total_equity": total_equity,
                "open_positions": open_count,
                "drawdown_pct": dd,
            })
        except Exception as e:
            logger.error("Equity snapshot error: %s", e)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _equity_task

    logger.info("🚀 Sandbox Engine starting on port %d", config.SERVICE_PORT)

    # Wire up signal handlers
    signal_consumer.set_handlers(on_new_signal, on_signal_closed, on_signal_update)

    # Start signal consumer in background
    consume_task = asyncio.create_task(signal_consumer.start())

    # Start equity snapshot loop
    _equity_task = asyncio.create_task(equity_snapshot_loop())

    yield

    # Shutdown
    logger.info("Shutting down sandbox engine...")
    await signal_consumer.stop()
    if _equity_task:
        _equity_task.cancel()
    consume_task.cancel()
    try:
        await consume_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Sandbox Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# WebSocket endpoint for dashboard
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    logger.info("Dashboard client connected (%d total)", len(ws_clients))
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data) if data else {}
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)
        logger.info("Dashboard client disconnected (%d remaining)", len(ws_clients))


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    session = session_mgr.get_active_session()
    from .mojo_wrappers import MOJO_AVAILABLE
    return {
        "status": "ok",
        "service": "sandbox-engine",
        "port": config.SERVICE_PORT,
        "mojo_accelerated": MOJO_AVAILABLE,
        "ws_connected": signal_consumer.connected,
        "dashboard_clients": len(ws_clients),
        "active_session": str(session["session_id"]) if session else None,
    }


@app.post("/sessions")
async def create_session(req: CreateSessionRequest):
    global _baseline
    session = session_mgr.create_session(
        initial_capital=req.initial_capital,
        leverage=req.leverage,
        max_position_pct=req.max_position_pct,
        max_open_positions=req.max_open_positions,
        max_drawdown_pct=req.max_daily_drawdown_pct,
        max_sector_exposure_pct=req.max_sector_exposure_pct,
    )
    # Initialize baseline tracker for this session
    _baseline = BaselineTracker(
        req.initial_capital, req.leverage, req.max_open_positions,
    )
    await broadcast("SESSION_STARTED", {"session": convert_for_json(session)})
    return convert_for_json(session)


@app.get("/sessions/active")
async def get_active_session():
    session = session_mgr.get_active_session()
    if not session:
        raise HTTPException(404, "No active session")
    return convert_for_json(session)


@app.get("/sessions")
async def list_sessions():
    sessions = session_mgr.get_all_sessions()
    return convert_for_json(sessions)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return convert_for_json(session)


@app.put("/sessions/{session_id}/pause")
async def pause_session(session_id: str):
    ok = session_mgr.pause_session(session_id)
    if not ok:
        raise HTTPException(400, "Cannot pause session")
    await broadcast("SESSION_PAUSED", {"session_id": session_id})
    return {"status": "paused"}


@app.put("/sessions/{session_id}/resume")
async def resume_session(session_id: str):
    ok = session_mgr.resume_session(session_id)
    if not ok:
        raise HTTPException(400, "Cannot resume session")
    await broadcast("SESSION_RESUMED", {"session_id": session_id})
    return {"status": "resumed"}


@app.put("/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    # Close all open trades first
    trade_exec.close_all_open_trades(session_id, "MANUAL_CLOSE")
    ok = session_mgr.stop_session(session_id, "Manual stop")
    if not ok:
        raise HTTPException(400, "Cannot stop session")
    await broadcast("SESSION_STOPPED", {"session_id": session_id, "reason": "Manual stop"})
    return {"status": "stopped"}


@app.get("/positions")
async def get_positions():
    session = session_mgr.get_active_session()
    if not session:
        return []
    trades = trade_exec.get_open_trades(str(session["session_id"]))
    return convert_for_json(trades)


@app.get("/trades")
async def get_trades(session_id: Optional[str] = None, status: Optional[str] = None):
    if not session_id:
        session = session_mgr.get_active_session()
        if not session:
            return []
        session_id = str(session["session_id"])
    trades = trade_exec.get_trades(session_id, status=status)
    return convert_for_json(trades)


@app.get("/decisions")
async def get_decisions(session_id: Optional[str] = None):
    if not session_id:
        session = session_mgr.get_active_session()
        if not session:
            return []
        session_id = str(session["session_id"])
    decisions = trade_exec.get_ml_decisions(session_id)
    return convert_for_json(decisions)


@app.get("/equity")
async def get_equity(session_id: Optional[str] = None):
    if not session_id:
        session = session_mgr.get_active_session()
        if not session:
            return []
        session_id = str(session["session_id"])
    snapshots = trade_exec.get_equity_curve(session_id)
    return convert_for_json(snapshots)


@app.get("/performance")
async def get_performance(session_id: Optional[str] = None):
    if not session_id:
        session = session_mgr.get_active_session()
        if not session:
            raise HTTPException(404, "No active session")
        session_id = str(session["session_id"])

    session = session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    total = session["total_trades"]
    wins = session["winning_trades"]
    losses = session["losing_trades"]

    return convert_for_json({
        "session_id": session_id,
        "initial_capital": session["initial_capital"],
        "current_capital": session["current_capital"],
        "peak_capital": session["peak_capital"],
        "total_pnl": session["total_pnl"],
        "total_pnl_pct": session["total_pnl_pct"],
        "total_trades": total,
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate": win_rate(wins, total),
        "max_drawdown_pct": session["max_drawdown_pct"],
        "signals_received": session["signals_received"],
        "signals_skipped": session["signals_skipped"],
        "drawdown_pct": current_drawdown(
            float(session["current_capital"]),
            float(session["peak_capital"]),
        ),
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    uvicorn.run(
        "mojo_compute.sandbox.server:app",
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
