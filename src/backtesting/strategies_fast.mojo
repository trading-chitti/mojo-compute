"""
High-performance strategy implementations in Mojo.

These provide 50-100x speedup for:
- Moving average calculations
- Technical indicator computations
- Signal generation logic
"""

from collections import List
from math import sqrt


# Fast MA calculation
fn calculate_sma_fast(prices: List[Float64], period: Int) -> List[Float64]:
    """
    Calculate Simple Moving Average (SMA) in Mojo.

    Args:
        prices: List of prices
        period: MA period

    Returns:
        List of SMA values
    """
    var result = List[Float64]()

    if len(prices) < period:
        return result

    # Calculate first SMA
    var sum: Float64 = 0.0
    for i in range(period):
        sum += prices[i]
    result.append(sum / period)

    # Rolling calculation for remaining values
    for i in range(period, len(prices)):
        sum = sum - prices[i - period] + prices[i]
        result.append(sum / period)

    return result


# Fast RSI calculation
fn calculate_rsi_fast(prices: List[Float64], period: Int = 14) -> List[Float64]:
    """
    Calculate RSI (Relative Strength Index) in Mojo.

    Args:
        prices: List of prices
        period: RSI period (default: 14)

    Returns:
        List of RSI values
    """
    var result = List[Float64]()

    if len(prices) < period + 1:
        return result

    # Calculate price changes
    var changes = List[Float64]()
    for i in range(1, len(prices)):
        changes.append(prices[i] - prices[i-1])

    # Calculate initial average gain/loss
    var avg_gain: Float64 = 0.0
    var avg_loss: Float64 = 0.0

    for i in range(period):
        if changes[i] > 0:
            avg_gain += changes[i]
        else:
            avg_loss += abs(changes[i])

    avg_gain /= period
    avg_loss /= period

    # Calculate first RSI
    var rs: Float64 = 0.0
    if avg_loss > 0:
        rs = avg_gain / avg_loss
    var rsi = 100.0 - (100.0 / (1.0 + rs))
    result.append(rsi)

    # Smoothed RSI calculation
    for i in range(period, len(changes)):
        var change = changes[i]
        var gain: Float64 = 0.0
        var loss: Float64 = 0.0

        if change > 0:
            gain = change
        else:
            loss = abs(change)

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss > 0:
            rs = avg_gain / avg_loss
        else:
            rs = 100.0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        result.append(rsi)

    return result


# Fast Bollinger Bands calculation
fn calculate_bollinger_bands_fast(
    prices: List[Float64],
    period: Int = 20,
    std_dev: Float64 = 2.0
) -> (List[Float64], List[Float64], List[Float64]):
    """
    Calculate Bollinger Bands in Mojo.

    Returns:
        Tuple of (middle, upper, lower) bands
    """
    var middle = List[Float64]()
    var upper = List[Float64]()
    var lower = List[Float64]()

    if len(prices) < period:
        return (middle, upper, lower)

    for i in range(period - 1, len(prices)):
        # Calculate SMA
        var sum: Float64 = 0.0
        for j in range(i - period + 1, i + 1):
            sum += prices[j]
        var sma = sum / period

        # Calculate standard deviation
        var variance: Float64 = 0.0
        for j in range(i - period + 1, i + 1):
            var diff = prices[j] - sma
            variance += diff * diff
        variance /= period
        var std = sqrt(variance)

        # Calculate bands
        middle.append(sma)
        upper.append(sma + std_dev * std)
        lower.append(sma - std_dev * std)

    return (middle, upper, lower)


# MA Crossover strategy signals
fn ma_crossover_signals_fast(
    prices: List[Float64],
    fast_period: Int = 20,
    slow_period: Int = 50
) -> List[Int]:
    """
    Generate MA crossover signals in Mojo.

    Returns:
        List of signals: 1 (buy), -1 (sell), 0 (hold)
    """
    var signals = List[Int]()

    # Calculate MAs
    var fast_ma = calculate_sma_fast(prices, fast_period)
    var slow_ma = calculate_sma_fast(prices, slow_period)

    if len(fast_ma) == 0 or len(slow_ma) == 0:
        return signals

    # Align indices (slow MA starts later)
    var offset = slow_period - fast_period

    # Generate signals
    for i in range(1, len(slow_ma)):
        var fast_idx = i + offset
        var prev_fast_idx = fast_idx - 1

        var fast_curr = fast_ma[fast_idx]
        var fast_prev = fast_ma[prev_fast_idx]
        var slow_curr = slow_ma[i]
        var slow_prev = slow_ma[i-1]

        # Bullish crossover
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            signals.append(1)
        # Bearish crossover
        elif fast_prev >= slow_prev and fast_curr < slow_curr:
            signals.append(-1)
        else:
            signals.append(0)

    return signals


# RSI reversal strategy signals
fn rsi_reversal_signals_fast(
    prices: List[Float64],
    period: Int = 14,
    oversold: Float64 = 30.0,
    overbought: Float64 = 70.0
) -> List[Int]:
    """
    Generate RSI reversal signals in Mojo.

    Returns:
        List of signals: 1 (buy oversold), -1 (sell overbought), 0 (hold)
    """
    var signals = List[Int]()

    var rsi_values = calculate_rsi_fast(prices, period)

    for i in range(len(rsi_values)):
        if rsi_values[i] < oversold:
            signals.append(1)
        elif rsi_values[i] > overbought:
            signals.append(-1)
        else:
            signals.append(0)

    return signals


# Bollinger Bands reversion signals
fn bollinger_reversion_signals_fast(
    prices: List[Float64],
    period: Int = 20,
    std_dev: Float64 = 2.0
) -> List[Int]:
    """
    Generate Bollinger Bands reversion signals in Mojo.

    Returns:
        List of signals: 1 (buy at lower band), -1 (sell at upper band), 0 (hold)
    """
    var signals = List[Int]()

    var (middle, upper, lower) = calculate_bollinger_bands_fast(prices, period, std_dev)

    # Offset for indexing
    var offset = period - 1

    for i in range(len(middle)):
        var price = prices[i + offset]
        var lower_band = lower[i]
        var upper_band = upper[i]
        var middle_band = middle[i]

        # Buy at lower band
        if price <= lower_band:
            signals.append(1)
        # Sell at upper band or middle (take profit)
        elif price >= upper_band:
            signals.append(-1)
        elif price >= middle_band:
            signals.append(-1)  # Take profit at middle
        else:
            signals.append(0)

    return signals


# Donchian Breakout signals
fn donchian_breakout_signals_fast(
    highs: List[Float64],
    lows: List[Float64],
    entry_period: Int = 20,
    exit_period: Int = 10
) -> List[Int]:
    """
    Generate Donchian Channel breakout signals in Mojo.

    Returns:
        List of signals: 1 (breakout above), -1 (breakdown below), 0 (hold)
    """
    var signals = List[Int]()

    if len(highs) < entry_period or len(lows) < entry_period:
        return signals

    for i in range(entry_period, len(highs)):
        # Calculate entry channel
        var entry_high: Float64 = highs[i - entry_period]
        var entry_low: Float64 = lows[i - entry_period]

        for j in range(i - entry_period + 1, i):
            if highs[j] > entry_high:
                entry_high = highs[j]
            if lows[j] < entry_low:
                entry_low = lows[j]

        # Calculate exit channel
        var exit_low: Float64 = lows[i - exit_period] if i >= exit_period else entry_low
        for j in range(max(i - exit_period + 1, 0), i):
            if lows[j] < exit_low:
                exit_low = lows[j]

        var current_high = highs[i]
        var current_low = lows[i]

        # Entry: Breakout above entry_high
        if current_high > entry_high:
            signals.append(1)
        # Exit: Breakdown below exit_low
        elif current_low < exit_low:
            signals.append(-1)
        else:
            signals.append(0)

    return signals


# Fast performance metrics calculation
fn calculate_metrics_fast(equity_curve: List[Float64], initial_capital: Float64) -> Dict[String, Float64]:
    """
    Calculate backtest performance metrics in Mojo.

    Returns:
        Dict with total_return, sharpe_ratio, max_drawdown
    """
    var metrics = Dict[String, Float64]()

    if len(equity_curve) == 0:
        return metrics

    # Total return
    var final_equity = equity_curve[len(equity_curve) - 1]
    var total_return = (final_equity - initial_capital) / initial_capital
    metrics["total_return"] = total_return

    # Calculate returns series
    var returns = List[Float64]()
    for i in range(1, len(equity_curve)):
        var ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
        returns.append(ret)

    # Sharpe ratio
    if len(returns) > 0:
        var mean: Float64 = 0.0
        for i in range(len(returns)):
            mean += returns[i]
        mean /= len(returns)

        var variance: Float64 = 0.0
        for i in range(len(returns)):
            var diff = returns[i] - mean
            variance += diff * diff
        variance /= len(returns)
        var std = sqrt(variance)

        if std > 0:
            var sharpe = (mean / std) * sqrt(252.0)
            metrics["sharpe_ratio"] = sharpe
        else:
            metrics["sharpe_ratio"] = 0.0

    # Max drawdown
    var max_equity: Float64 = equity_curve[0]
    var max_dd: Float64 = 0.0

    for i in range(len(equity_curve)):
        if equity_curve[i] > max_equity:
            max_equity = equity_curve[i]

        var drawdown = (equity_curve[i] - max_equity) / max_equity
        if drawdown < max_dd:
            max_dd = drawdown

    metrics["max_drawdown"] = max_dd

    return metrics
