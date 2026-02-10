"""
Python wrappers for Mojo sandbox computation modules.
Falls back to pure Python if Mojo is not available.
"""

import logging
import math

logger = logging.getLogger(__name__)

# Try to import Mojo modules
MOJO_AVAILABLE = False
try:
    import mojo.importer
    import sys
    from pathlib import Path
    mojo_src_dir = Path(__file__).parent.parent.parent / "src" / "sandbox"
    if str(mojo_src_dir) not in sys.path:
        sys.path.insert(0, str(mojo_src_dir))
    import position_sizer as mojo_position_sizer
    import pnl_calculator as mojo_pnl_calculator
    import performance_metrics as mojo_performance_metrics
    import risk_manager as mojo_risk_manager
    MOJO_AVAILABLE = True
    logger.info("✅ Mojo sandbox modules available (sub-ms computation)")
except ImportError as e:
    logger.warning(f"⚠️ Mojo not available, using Python fallback: {e}")


# =============================================================================
# Position Sizer
# =============================================================================

def kelly_position_size_pct(
    ml_score: float,
    signal_confidence: float,
    risk_reward_ratio: float,
    current_drawdown_pct: float,
    max_drawdown_pct: float = 10.0,
    min_position_pct: float = 5.0,
    max_position_pct: float = 20.0,
) -> float:
    """Calculate position size as percentage of capital using Half-Kelly."""
    if MOJO_AVAILABLE:
        return float(mojo_position_sizer.calculate_position_size_pct({
            "ml_score": ml_score,
            "confidence": signal_confidence,
            "rr": risk_reward_ratio,
            "dd": current_drawdown_pct,
            "max_dd": max_drawdown_pct,
            "min_pct": min_position_pct,
            "max_pct": max_position_pct,
        }))

    # Python fallback
    p = ml_score
    q = 1.0 - p
    b = risk_reward_ratio

    kelly = max(0.0, (p * b - q) / b)
    half_kelly = kelly * 0.5

    if current_drawdown_pct <= 0:
        dd_factor = 1.0
    elif current_drawdown_pct >= max_drawdown_pct:
        dd_factor = 0.2
    else:
        ratio = current_drawdown_pct / max_drawdown_pct
        dd_factor = 1.0 - (ratio * 0.8)

    raw_pct = half_kelly * dd_factor * signal_confidence * 100.0
    return max(min_position_pct, min(raw_pct, max_position_pct))


def calculate_quantity(allocated_capital: float, leverage: float, entry_price: float) -> int:
    """Calculate number of shares to buy."""
    if entry_price <= 0:
        return 0
    if MOJO_AVAILABLE:
        return int(mojo_position_sizer.calculate_quantity(allocated_capital, leverage, entry_price))
    return int((allocated_capital * leverage) / entry_price)


def calculate_position_value(allocated_capital: float, leverage: float) -> float:
    """Calculate total position value with leverage."""
    if MOJO_AVAILABLE:
        return float(mojo_position_sizer.calculate_position_value(allocated_capital, leverage))
    return allocated_capital * leverage


# =============================================================================
# P&L Calculator
# =============================================================================

def calculate_unrealized_pnl(signal_type: str, entry_price: float, current_price: float, quantity: int) -> float:
    """Calculate unrealized P&L for an open position."""
    if MOJO_AVAILABLE:
        return float(mojo_pnl_calculator.calculate_unrealized_pnl(
            signal_type, entry_price, current_price, quantity,
        ))

    if signal_type == "CALL":
        return quantity * (current_price - entry_price)
    else:
        return quantity * (entry_price - current_price)


def calculate_unrealized_pnl_pct(signal_type: str, entry_price: float, current_price: float) -> float:
    """Calculate unrealized P&L as percentage."""
    if MOJO_AVAILABLE:
        return float(mojo_pnl_calculator.calculate_unrealized_pnl_pct(
            signal_type, entry_price, current_price,
        ))

    if entry_price <= 0:
        return 0.0
    if signal_type == "CALL":
        return (current_price - entry_price) / entry_price * 100.0
    else:
        return (entry_price - current_price) / entry_price * 100.0


def calculate_realized_pnl(signal_type: str, entry_price: float, exit_price: float, quantity: int) -> float:
    """Calculate realized P&L when trade is closed."""
    if MOJO_AVAILABLE:
        return float(mojo_pnl_calculator.calculate_realized_pnl(
            signal_type, entry_price, exit_price, quantity,
        ))

    if signal_type == "CALL":
        return quantity * (exit_price - entry_price)
    else:
        return quantity * (entry_price - exit_price)


def calculate_realized_pnl_pct(signal_type: str, entry_price: float, exit_price: float) -> float:
    """Calculate realized P&L as percentage."""
    if MOJO_AVAILABLE:
        return float(mojo_pnl_calculator.calculate_realized_pnl_pct(
            signal_type, entry_price, exit_price,
        ))

    if entry_price <= 0:
        return 0.0
    if signal_type == "CALL":
        return (exit_price - entry_price) / entry_price * 100.0
    else:
        return (entry_price - exit_price) / entry_price * 100.0


def check_target_hit(signal_type: str, current_price: float, target_price: float) -> bool:
    """Check if price has hit the target."""
    if MOJO_AVAILABLE:
        return bool(mojo_pnl_calculator.check_target_hit(
            signal_type, current_price, target_price,
        ))

    if signal_type == "CALL":
        return current_price >= target_price
    else:
        return current_price <= target_price


def check_stoploss_hit(signal_type: str, current_price: float, stop_loss: float) -> bool:
    """Check if price has hit the stop loss."""
    if MOJO_AVAILABLE:
        return bool(mojo_pnl_calculator.check_stoploss_hit(
            signal_type, current_price, stop_loss,
        ))

    if signal_type == "CALL":
        return current_price <= stop_loss
    else:
        return current_price >= stop_loss


# =============================================================================
# Performance Metrics
# =============================================================================

def win_rate(winning_trades: int, total_trades: int) -> float:
    """Calculate win rate as percentage."""
    if total_trades == 0:
        return 0.0
    if MOJO_AVAILABLE:
        return float(mojo_performance_metrics.win_rate(winning_trades, total_trades))
    return winning_trades / total_trades * 100.0


def profit_factor(gross_profit: float, gross_loss: float) -> float:
    """Calculate profit factor."""
    if MOJO_AVAILABLE:
        return float(mojo_performance_metrics.profit_factor(gross_profit, gross_loss))

    if gross_loss == 0:
        return 999.99 if gross_profit > 0 else 0.0
    return gross_profit / abs(gross_loss)


def max_drawdown(equity_curve: list) -> float:
    """Calculate maximum drawdown percentage (Python-only, not hot-path)."""
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

    return max_dd


def current_drawdown(current_equity: float, peak_equity: float) -> float:
    """Calculate current drawdown percentage."""
    if MOJO_AVAILABLE:
        return float(mojo_performance_metrics.current_drawdown(current_equity, peak_equity))

    if peak_equity <= 0 or current_equity >= peak_equity:
        return 0.0
    return (peak_equity - current_equity) / peak_equity * 100.0


def sharpe_ratio(returns: list, risk_free_rate: float = 0.02, annualize_factor: float = 15.87) -> float:
    """Calculate Sharpe ratio (Python-only, not hot-path)."""
    n = len(returns)
    if n < 2:
        return 0.0

    mean_return = sum(returns) / n
    variance = sum((r - mean_return) ** 2 for r in returns) / (n - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0

    if std_dev == 0:
        return 0.0

    return ((mean_return - risk_free_rate) / std_dev) * annualize_factor


def total_pnl_pct(current_capital: float, initial_capital: float) -> float:
    """Calculate total P&L percentage."""
    if MOJO_AVAILABLE:
        return float(mojo_performance_metrics.total_pnl_pct(current_capital, initial_capital))

    if initial_capital <= 0:
        return 0.0
    return (current_capital - initial_capital) / initial_capital * 100.0


# =============================================================================
# Risk Manager
# =============================================================================

def can_open_trade(
    current_drawdown_pct: float,
    max_daily_drawdown_pct: float,
    open_positions: int,
    max_open_positions: int,
    sector_exposure_pct: float,
    max_sector_exposure_pct: float,
    available_capital: float,
    min_trade_capital: float,
    session_active: bool,
) -> bool:
    """Check all risk conditions before opening a trade."""
    if MOJO_AVAILABLE:
        return bool(mojo_risk_manager.can_open_trade({
            "dd": current_drawdown_pct,
            "max_dd": max_daily_drawdown_pct,
            "open_pos": open_positions,
            "max_pos": max_open_positions,
            "sector_exp": sector_exposure_pct,
            "max_sector": max_sector_exposure_pct,
            "avail": available_capital,
            "min_trade": min_trade_capital,
            "active": int(session_active),
        }))

    # Python fallback
    if not session_active:
        return False
    if current_drawdown_pct >= max_daily_drawdown_pct:
        return False
    if open_positions >= max_open_positions:
        return False
    if sector_exposure_pct >= max_sector_exposure_pct:
        return False
    if available_capital < min_trade_capital:
        return False
    return True


def get_rejection_reason(
    current_drawdown_pct: float,
    max_daily_drawdown_pct: float,
    open_positions: int,
    max_open_positions: int,
    sector_exposure_pct: float,
    max_sector_exposure_pct: float,
    available_capital: float,
    min_trade_capital: float,
    session_active: bool,
) -> str:
    """Return reason why trade was rejected (Python-only, string formatting)."""
    if not session_active:
        return "Session is not active"
    if current_drawdown_pct >= max_daily_drawdown_pct:
        return f"Max drawdown reached ({current_drawdown_pct:.1f}% >= {max_daily_drawdown_pct:.1f}%)"
    if open_positions >= max_open_positions:
        return f"Max positions reached ({open_positions}/{max_open_positions})"
    if sector_exposure_pct >= max_sector_exposure_pct:
        return f"Sector limit reached ({sector_exposure_pct:.1f}% >= {max_sector_exposure_pct:.1f}%)"
    if available_capital < min_trade_capital:
        return f"Insufficient capital ({available_capital:.0f} < {min_trade_capital:.0f})"
    return "OK"


def should_stop_session(current_drawdown_pct: float, max_daily_drawdown_pct: float) -> bool:
    """Check if session should auto-stop due to drawdown."""
    if MOJO_AVAILABLE:
        return bool(mojo_risk_manager.should_stop_session(current_drawdown_pct, max_daily_drawdown_pct))
    return current_drawdown_pct >= max_daily_drawdown_pct


def calculate_available_capital(current_capital: float, allocated_capital: float) -> float:
    """Calculate capital available for new trades."""
    if MOJO_AVAILABLE:
        return float(mojo_risk_manager.calculate_available_capital(current_capital, allocated_capital))
    available = current_capital - allocated_capital
    return max(0.0, available)


def calculate_sector_exposure_pct(sector_allocated: float, total_capital: float) -> float:
    """Calculate sector exposure as percentage of total capital."""
    if MOJO_AVAILABLE:
        return float(mojo_risk_manager.calculate_sector_exposure_pct(sector_allocated, total_capital))
    if total_capital <= 0:
        return 0.0
    return sector_allocated / total_capital * 100.0
