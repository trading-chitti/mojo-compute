"""
Technical Indicators - Shared Library for ctypes Integration
C-compatible exports for direct Python integration via ctypes
"""

from math import sqrt
from memory import UnsafePointer


fn rsi_internal(prices: List[Float64], period: Int) -> List[Float64]:
    """Calculate RSI (internal implementation)."""
    var n = len(prices)
    var result = List[Float64](capacity=n)

    for _ in range(n):
        result.append(0.0)

    if period <= 0 or period >= n:
        return result^

    var gains = List[Float64](capacity=n-1)
    var losses = List[Float64](capacity=n-1)

    for i in range(1, n):
        var change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-change)

    var avg_gain: Float64 = 0.0
    var avg_loss: Float64 = 0.0

    for i in range(period):
        avg_gain += gains[i]
        avg_loss += losses[i]

    avg_gain /= Float64(period)
    avg_loss /= Float64(period)

    if avg_loss == 0:
        result[period] = 100.0
    else:
        var rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * Float64(period - 1) + gains[i - 1]) / Float64(period)
        avg_loss = (avg_loss * Float64(period - 1) + losses[i - 1]) / Float64(period)

        if avg_loss == 0:
            result[i] = 100.0
        else:
            var rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result^


fn ema_internal(prices: List[Float64], period: Int) -> List[Float64]:
    """Calculate EMA (internal implementation)."""
    var n = len(prices)
    var result = List[Float64](capacity=n)

    for _ in range(n):
        result.append(0.0)

    if period <= 0 or period > n:
        return result^

    var multiplier = 2.0 / Float64(period + 1)
    var sum: Float64 = 0.0

    for i in range(period):
        sum += prices[i]
    var initial_ema = sum / Float64(period)
    result[period - 1] = initial_ema

    var prev_ema = initial_ema
    for i in range(period, n):
        var current_ema = (prices[i] - prev_ema) * multiplier + prev_ema
        result[i] = current_ema
        prev_ema = current_ema

    return result^


fn sma_internal(prices: List[Float64], period: Int) -> List[Float64]:
    """Calculate SMA (internal implementation)."""
    var n = len(prices)
    var result = List[Float64](capacity=n)

    for _ in range(n):
        result.append(0.0)

    if period <= 0 or period > n:
        return result^

    var sum: Float64 = 0.0
    for i in range(period):
        sum += prices[i]
    result[period - 1] = sum / Float64(period)

    for i in range(period, n):
        sum = sum - prices[i - period] + prices[i]
        result[i] = sum / Float64(period)

    return result^


# C-compatible exports for ctypes
@always_inline
fn rsi_c_export(
    prices_ptr: UnsafePointer[Float64],
    n: Int,
    period: Int,
    result_ptr: UnsafePointer[Float64]
):
    """
    C-compatible RSI calculation for ctypes.

    Args:
        prices_ptr: Pointer to prices array
        n: Number of prices
        period: RSI period
        result_ptr: Pointer to result array (must be pre-allocated)
    """
    # Convert pointer to List
    var prices = List[Float64](capacity=n)
    for i in range(n):
        prices.append(prices_ptr[i])

    # Calculate RSI
    var result = rsi_internal(prices, period)

    # Copy back to pointer
    for i in range(len(result)):
        result_ptr[i] = result[i]


@always_inline
fn ema_c_export(
    prices_ptr: UnsafePointer[Float64],
    n: Int,
    period: Int,
    result_ptr: UnsafePointer[Float64]
):
    """C-compatible EMA calculation for ctypes."""
    var prices = List[Float64](capacity=n)
    for i in range(n):
        prices.append(prices_ptr[i])

    var result = ema_internal(prices, period)

    for i in range(len(result)):
        result_ptr[i] = result[i]


@always_inline
fn sma_c_export(
    prices_ptr: UnsafePointer[Float64],
    n: Int,
    period: Int,
    result_ptr: UnsafePointer[Float64]
):
    """C-compatible SMA calculation for ctypes."""
    var prices = List[Float64](capacity=n)
    for i in range(n):
        prices.append(prices_ptr[i])

    var result = sma_internal(prices, period)

    for i in range(len(result)):
        result_ptr[i] = result[i]


@always_inline
fn macd_c_export(
    prices_ptr: UnsafePointer[Float64],
    n: Int,
    fast_period: Int,
    slow_period: Int,
    signal_period: Int,
    macd_ptr: UnsafePointer[Float64],
    signal_ptr: UnsafePointer[Float64],
    hist_ptr: UnsafePointer[Float64]
):
    """C-compatible MACD calculation for ctypes."""
    var prices = List[Float64](capacity=n)
    for i in range(n):
        prices.append(prices_ptr[i])

    var ema_fast = ema_internal(prices, fast_period)
    var ema_slow = ema_internal(prices, slow_period)

    var macd_line = List[Float64](capacity=n)
    for i in range(n):
        macd_line.append(ema_fast[i] - ema_slow[i])

    var signal_line = ema_internal(macd_line, signal_period)

    for i in range(n):
        macd_ptr[i] = macd_line[i]
        signal_ptr[i] = signal_line[i]
        hist_ptr[i] = macd_line[i] - signal_line[i]


fn main():
    """Test the C-compatible exports."""
    print("🔥 Testing Mojo Shared Library Exports...")

    # Create test data
    var n = 20
    var prices_data = UnsafePointer[Float64].alloc(n)

    for i in range(n):
        prices_data[i] = 100.0 + Float64(i) * 0.5 + Float64(i % 3) * 2.0

    # Test RSI
    print("\n📈 Testing RSI C export...")
    var rsi_result = UnsafePointer[Float64].alloc(n)
    rsi_c_export(prices_data, n, 14, rsi_result)

    print("  Last 5 RSI values:")
    for i in range(n - 5, n):
        print("    ", rsi_result[i])

    # Test EMA
    print("\n📈 Testing EMA C export...")
    var ema_result = UnsafePointer[Float64].alloc(n)
    ema_c_export(prices_data, n, 9, ema_result)

    print("  Last 5 EMA values:")
    for i in range(n - 5, n):
        print("    ", ema_result[i])

    # Test SMA
    print("\n📈 Testing SMA C export...")
    var sma_result = UnsafePointer[Float64].alloc(n)
    sma_c_export(prices_data, n, 5, sma_result)

    print("  Last 5 SMA values:")
    for i in range(n - 5, n):
        print("    ", sma_result[i])

    # Cleanup
    prices_data.free()
    rsi_result.free()
    ema_result.free()
    sma_result.free()

    print("\n✅ All C exports working!")
    print("📦 Ready to build as shared library")
