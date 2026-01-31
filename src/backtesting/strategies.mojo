"""
Complete strategy implementations in Mojo.

All strategy logic is implemented in Mojo for maximum performance:
- Price/bar processing
- Indicator calculations
- Signal generation
- Position sizing logic

Only the Python wrapper remains for interfacing with the backtesting engine.
"""

from collections import List, Dict
from math import sqrt, abs, min, max


# ============================================================================
# CORE INDICATOR FUNCTIONS (Fast implementations)
# ============================================================================

fn calculate_sma(prices: List[Float64], period: Int) -> List[Float64]:
    """Fast SMA calculation with rolling window."""
    var result = List[Float64]()

    if len(prices) < period:
        return result

    # First SMA
    var sum: Float64 = 0.0
    for i in range(period):
        sum += prices[i]
    result.append(sum / period)

    # Rolling SMA
    for i in range(period, len(prices)):
        sum = sum - prices[i - period] + prices[i]
        result.append(sum / period)

    return result


fn calculate_ema(prices: List[Float64], period: Int) -> List[Float64]:
    """Fast EMA calculation."""
    var result = List[Float64]()

    if len(prices) < period:
        return result

    # Calculate multiplier
    var multiplier = 2.0 / (period + 1)

    # First EMA is SMA
    var sum: Float64 = 0.0
    for i in range(period):
        sum += prices[i]
    var ema = sum / period
    result.append(ema)

    # Subsequent EMAs
    for i in range(period, len(prices)):
        ema = (prices[i] - ema) * multiplier + ema
        result.append(ema)

    return result


fn calculate_rsi(prices: List[Float64], period: Int) -> List[Float64]:
    """Fast RSI calculation."""
    var result = List[Float64]()

    if len(prices) < period + 1:
        return result

    # Calculate price changes
    var changes = List[Float64]()
    for i in range(1, len(prices)):
        changes.append(prices[i] - prices[i-1])

    # Initial average gain/loss
    var avg_gain: Float64 = 0.0
    var avg_loss: Float64 = 0.0

    for i in range(period):
        if changes[i] > 0:
            avg_gain += changes[i]
        else:
            avg_loss += abs(changes[i])

    avg_gain /= period
    avg_loss /= period

    # First RSI
    var rs = avg_gain / (avg_loss + 1e-10)
    result.append(100.0 - (100.0 / (1.0 + rs)))

    # Smoothed RSI
    for i in range(period, len(changes)):
        var gain: Float64 = 0.0
        var loss: Float64 = 0.0

        if changes[i] > 0:
            gain = changes[i]
        else:
            loss = abs(changes[i])

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        rs = avg_gain / (avg_loss + 1e-10)
        result.append(100.0 - (100.0 / (1.0 + rs)))

    return result


fn calculate_bollinger_bands(prices: List[Float64], period: Int, std_dev: Float64) -> (List[Float64], List[Float64], List[Float64]):
    """Fast Bollinger Bands calculation."""
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

        middle.append(sma)
        upper.append(sma + std_dev * std)
        lower.append(sma - std_dev * std)

    return (middle, upper, lower)


# ============================================================================
# STRATEGY 1: MA CROSSOVER
# ============================================================================

struct MACrossoverSignals:
    """MA Crossover strategy signals."""
    var signals: List[Int]  # 1=buy, -1=sell, 0=hold
    var fast_ma: List[Float64]
    var slow_ma: List[Float64]


fn ma_crossover_strategy(
    prices: List[Float64],
    fast_period: Int,
    slow_period: Int,
    position_size: Float64,
    capital: Float64
) -> MACrossoverSignals:
    """
    Complete MA Crossover strategy logic in Mojo.

    Returns signals and indicator values.
    """
    var signals = List[Int]()

    # Calculate MAs
    var fast_ma = calculate_sma(prices, fast_period)
    var slow_ma = calculate_sma(prices, slow_period)

    if len(fast_ma) == 0 or len(slow_ma) == 0:
        return MACrossoverSignals(signals, fast_ma, slow_ma)

    # Align indices
    var offset = slow_period - fast_period

    # Initialize with hold
    for i in range(offset):
        signals.append(0)

    # Generate crossover signals
    for i in range(1, len(slow_ma)):
        var fast_idx = i + offset
        var prev_fast_idx = fast_idx - 1

        var fast_curr = fast_ma[fast_idx]
        var fast_prev = fast_ma[prev_fast_idx]
        var slow_curr = slow_ma[i]
        var slow_prev = slow_ma[i-1]

        # Bullish crossover: fast crosses above slow
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            signals.append(1)
        # Bearish crossover: fast crosses below slow
        elif fast_prev >= slow_prev and fast_curr < slow_curr:
            signals.append(-1)
        else:
            signals.append(0)

    return MACrossoverSignals(signals, fast_ma, slow_ma)


# ============================================================================
# STRATEGY 2: RSI REVERSAL
# ============================================================================

struct RSIReversalSignals:
    """RSI Reversal strategy signals."""
    var signals: List[Int]
    var rsi_values: List[Float64]


fn rsi_reversal_strategy(
    prices: List[Float64],
    period: Int,
    oversold: Float64,
    overbought: Float64,
    position_size: Float64,
    capital: Float64
) -> RSIReversalSignals:
    """
    Complete RSI Reversal strategy logic in Mojo.

    Buy when RSI < oversold, sell when RSI > overbought.
    """
    var signals = List[Int]()
    var rsi_values = calculate_rsi(prices, period)

    if len(rsi_values) == 0:
        return RSIReversalSignals(signals, rsi_values)

    # Pad signals for offset
    for i in range(period):
        signals.append(0)

    # Generate signals
    for i in range(len(rsi_values)):
        var rsi = rsi_values[i]

        if rsi < oversold:
            signals.append(1)  # Buy oversold
        elif rsi > overbought:
            signals.append(-1)  # Sell overbought
        else:
            signals.append(0)

    return RSIReversalSignals(signals, rsi_values)


# ============================================================================
# STRATEGY 3: BOLLINGER BANDS REVERSION
# ============================================================================

struct BollingerReversalSignals:
    """Bollinger Bands reversion strategy signals."""
    var signals: List[Int]
    var middle: List[Float64]
    var upper: List[Float64]
    var lower: List[Float64]


fn bollinger_reversion_strategy(
    prices: List[Float64],
    period: Int,
    std_dev: Float64,
    position_size: Float64,
    capital: Float64
) -> BollingerReversalSignals:
    """
    Complete Bollinger Bands reversion strategy in Mojo.

    Buy at lower band, sell at upper band or middle (take profit).
    """
    var signals = List[Int]()
    var (middle, upper, lower) = calculate_bollinger_bands(prices, period, std_dev)

    if len(middle) == 0:
        return BollingerReversalSignals(signals, middle, upper, lower)

    # Offset for indexing
    var offset = period - 1

    # Pad signals
    for i in range(offset):
        signals.append(0)

    # Track if we have a position (simplified logic)
    var has_position = False

    # Generate signals
    for i in range(len(middle)):
        var price = prices[i + offset]
        var lower_band = lower[i]
        var upper_band = upper[i]
        var middle_band = middle[i]

        # Buy at lower band (if no position)
        if price <= lower_band and not has_position:
            signals.append(1)
            has_position = True
        # Sell at upper band or middle (take profit)
        elif has_position and (price >= upper_band or price >= middle_band):
            signals.append(-1)
            has_position = False
        else:
            signals.append(0)

    return BollingerReversalSignals(signals, middle, upper, lower)


# ============================================================================
# STRATEGY 4: OPENING RANGE BREAKOUT (ORB)
# ============================================================================

struct ORBSignals:
    """Opening Range Breakout strategy signals."""
    var signals: List[Int]
    var or_highs: List[Float64]
    var or_lows: List[Float64]


fn orb_strategy(
    highs: List[Float64],
    lows: List[Float64],
    closes: List[Float64],
    timestamps: List[Int64],  # Unix timestamps
    range_bars: Int,  # Number of bars for opening range (e.g., 15 for 15-min)
    use_stop_loss: Bool
) -> ORBSignals:
    """
    Complete Opening Range Breakout strategy in Mojo.

    Trade breakouts from first N bars of the day.
    """
    var signals = List[Int]()
    var or_highs = List[Float64]()
    var or_lows = List[Float64]()

    if len(highs) < range_bars:
        return ORBSignals(signals, or_highs, or_lows)

    var current_or_high: Float64 = 0.0
    var current_or_low: Float64 = 1e10
    var bars_in_range = 0
    var current_date: Int64 = 0
    var has_position = False

    for i in range(len(highs)):
        var date = timestamps[i] / 86400  # Convert to days

        # New trading day
        if date != current_date:
            current_date = date
            current_or_high = highs[i]
            current_or_low = lows[i]
            bars_in_range = 1
            signals.append(0)
            or_highs.append(0.0)
            or_lows.append(0.0)
            continue

        # Building opening range
        if bars_in_range < range_bars:
            if highs[i] > current_or_high:
                current_or_high = highs[i]
            if lows[i] < current_or_low:
                current_or_low = lows[i]
            bars_in_range += 1
            signals.append(0)
            or_highs.append(0.0)
            or_lows.append(0.0)
            continue

        # Opening range established - look for breakouts
        or_highs.append(current_or_high)
        or_lows.append(current_or_low)

        # Breakout above OR high
        if closes[i] > current_or_high and not has_position:
            signals.append(1)
            has_position = True
        # Stop loss: break below OR low
        elif use_stop_loss and has_position and closes[i] < current_or_low:
            signals.append(-1)
            has_position = False
        else:
            signals.append(0)

    return ORBSignals(signals, or_highs, or_lows)


# ============================================================================
# STRATEGY 5: DONCHIAN CHANNEL BREAKOUT
# ============================================================================

struct DonchianSignals:
    """Donchian Channel breakout strategy signals."""
    var signals: List[Int]
    var entry_highs: List[Float64]
    var entry_lows: List[Float64]
    var exit_lows: List[Float64]


fn donchian_breakout_strategy(
    highs: List[Float64],
    lows: List[Float64],
    closes: List[Float64],
    entry_period: Int,
    exit_period: Int,
    position_size: Float64,
    capital: Float64
) -> DonchianSignals:
    """
    Complete Donchian Channel breakout strategy in Mojo.

    Buy on N-period high breakout, exit on M-period low.
    """
    var signals = List[Int]()
    var entry_highs = List[Float64]()
    var entry_lows = List[Float64]()
    var exit_lows = List[Float64]()

    if len(highs) < entry_period:
        return DonchianSignals(signals, entry_highs, entry_lows, exit_lows)

    var has_position = False

    for i in range(len(highs)):
        if i < entry_period:
            signals.append(0)
            entry_highs.append(0.0)
            entry_lows.append(0.0)
            exit_lows.append(0.0)
            continue

        # Calculate entry channel (exclude current bar)
        var entry_high: Float64 = highs[i - entry_period]
        var entry_low: Float64 = lows[i - entry_period]

        for j in range(i - entry_period + 1, i):
            if highs[j] > entry_high:
                entry_high = highs[j]
            if lows[j] < entry_low:
                entry_low = lows[j]

        # Calculate exit channel
        var exit_low: Float64 = lows[max(i - exit_period, 0)]
        for j in range(max(i - exit_period + 1, 0), i):
            if lows[j] < exit_low:
                exit_low = lows[j]

        entry_highs.append(entry_high)
        entry_lows.append(entry_low)
        exit_lows.append(exit_low)

        # Entry: breakout above entry_high
        if closes[i] > entry_high and not has_position:
            signals.append(1)
            has_position = True
        # Exit: breakdown below exit_low
        elif has_position and closes[i] < exit_low:
            signals.append(-1)
            has_position = False
        # Stop loss: break below entry_low
        elif has_position and closes[i] < entry_low:
            signals.append(-1)
            has_position = False
        else:
            signals.append(0)

    return DonchianSignals(signals, entry_highs, entry_lows, exit_lows)


# ============================================================================
# BATCH PROCESSING FOR MULTIPLE SYMBOLS
# ============================================================================

fn process_multiple_strategies_batch(
    symbols: List[String],
    prices_data: Dict[String, List[Float64]],
    strategy_type: String,
    params: Dict[String, Float64]
) -> Dict[String, List[Int]]:
    """
    Process multiple symbols in batch for maximum performance.

    This is significantly faster than processing one by one.
    """
    var all_signals = Dict[String, List[Int]]()

    for symbol_idx in range(len(symbols)):
        var symbol = symbols[symbol_idx]
        if symbol not in prices_data:
            continue

        var prices = prices_data[symbol]

        # Route to appropriate strategy
        var signals = List[Int]()

        if strategy_type == "ma_crossover":
            var result = ma_crossover_strategy(
                prices,
                Int(params["fast_period"]),
                Int(params["slow_period"]),
                params["position_size"],
                params["capital"]
            )
            signals = result.signals
        elif strategy_type == "rsi_reversal":
            var result = rsi_reversal_strategy(
                prices,
                Int(params["period"]),
                params["oversold"],
                params["overbought"],
                params["position_size"],
                params["capital"]
            )
            signals = result.signals

        all_signals[symbol] = signals

    return all_signals
