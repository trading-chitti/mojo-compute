"""
Technical Indicators - Python FFI Module
Exposes Mojo functions to Python for direct integration
"""

from python import Python, PythonObject
from math import sqrt


fn rsi_mojo(prices: List[Float64], period: Int = 14) raises -> List[Float64]:
    """Calculate RSI in Mojo (internal function)."""
    var n = len(prices)
    var result = List[Float64](capacity=n)

    # Initialize with zeros
    for _ in range(n):
        result.append(0.0)

    if period <= 0 or period >= n:
        return result^

    # Calculate price changes
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

    # Calculate initial average gain/loss
    var avg_gain: Float64 = 0.0
    var avg_loss: Float64 = 0.0

    for i in range(period):
        avg_gain += gains[i]
        avg_loss += losses[i]

    avg_gain /= Float64(period)
    avg_loss /= Float64(period)

    # Calculate RSI for first period
    if avg_loss == 0:
        result[period] = 100.0
    else:
        var rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate RSI for remaining periods
    for i in range(period + 1, n):
        avg_gain = (avg_gain * Float64(period - 1) + gains[i - 1]) / Float64(period)
        avg_loss = (avg_loss * Float64(period - 1) + losses[i - 1]) / Float64(period)

        if avg_loss == 0:
            result[i] = 100.0
        else:
            var rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result^


fn rsi_from_python(prices_obj: PythonObject, period: Int = 14) raises -> PythonObject:
    """
    Calculate RSI from Python NumPy array.

    Args:
        prices_obj: NumPy array of prices
        period: RSI period (default 14)

    Returns:
        NumPy array of RSI values
    """
    # Import NumPy
    var np = Python.import_module("numpy")

    # Convert NumPy array to Mojo List
    var n = Int(prices_obj.__len__())
    var prices = List[Float64](capacity=n)

    for i in range(n):
        var price_val = prices_obj[i]
        # Use py= keyword argument (required in Mojo 0.26+)
        prices.append(Float64(py=price_val))

    # Calculate RSI
    var rsi_result = rsi_mojo(prices, period)

    # Convert back to NumPy array
    var result_list = Python.evaluate("[]")
    for i in range(len(rsi_result)):
        _ = result_list.append(rsi_result[i])

    return np.array(result_list)


fn ema_mojo(prices: List[Float64], period: Int) raises -> List[Float64]:
    """Calculate EMA in Mojo (internal function)."""
    var n = len(prices)
    var result = List[Float64](capacity=n)

    for _ in range(n):
        result.append(0.0)

    if period <= 0 or period > n:
        return result^

    var multiplier = 2.0 / Float64(period + 1)

    # Start with SMA for first value
    var sum: Float64 = 0.0
    for i in range(period):
        sum += prices[i]
    var initial_ema = sum / Float64(period)
    result[period - 1] = initial_ema

    # Calculate EMA
    var prev_ema = initial_ema
    for i in range(period, n):
        var current_ema = (prices[i] - prev_ema) * multiplier + prev_ema
        result[i] = current_ema
        prev_ema = current_ema

    return result^


fn ema_from_python(prices_obj: PythonObject, period: Int) raises -> PythonObject:
    """
    Calculate EMA from Python NumPy array.

    Args:
        prices_obj: NumPy array of prices
        period: EMA period

    Returns:
        NumPy array of EMA values
    """
    var np = Python.import_module("numpy")

    var n = Int(prices_obj.__len__())
    var prices = List[Float64](capacity=n)

    for i in range(n):
        # Use py= keyword argument (required in Mojo 0.26+)
        prices.append(Float64(py=prices_obj[i]))

    var ema_result = ema_mojo(prices, period)

    var result_list = Python.evaluate("[]")
    for i in range(len(ema_result)):
        _ = result_list.append(ema_result[i])

    return np.array(result_list)


fn sma_mojo(prices: List[Float64], period: Int) raises -> List[Float64]:
    """Calculate SMA in Mojo (internal function)."""
    var n = len(prices)
    var result = List[Float64](capacity=n)

    for _ in range(n):
        result.append(0.0)

    if period <= 0 or period > n:
        return result^

    # Calculate first SMA
    var sum: Float64 = 0.0
    for i in range(period):
        sum += prices[i]
    result[period - 1] = sum / Float64(period)

    # Calculate remaining SMAs using sliding window
    for i in range(period, n):
        sum = sum - prices[i - period] + prices[i]
        result[i] = sum / Float64(period)

    return result^


fn sma_from_python(prices_obj: PythonObject, period: Int) raises -> PythonObject:
    """
    Calculate SMA from Python NumPy array.

    Args:
        prices_obj: NumPy array of prices
        period: SMA period

    Returns:
        NumPy array of SMA values
    """
    var np = Python.import_module("numpy")

    var n = Int(prices_obj.__len__())
    var prices = List[Float64](capacity=n)

    for i in range(n):
        # Use py= keyword argument (required in Mojo 0.26+)
        prices.append(Float64(py=prices_obj[i]))

    var sma_result = sma_mojo(prices, period)

    var result_list = Python.evaluate("[]")
    for i in range(len(sma_result)):
        _ = result_list.append(sma_result[i])

    return np.array(result_list)


fn macd_from_python(
    prices_obj: PythonObject,
    fast_period: Int = 12,
    slow_period: Int = 26,
    signal_period: Int = 9
) raises -> PythonObject:
    """
    Calculate MACD from Python NumPy array.

    Args:
        prices_obj: NumPy array of prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram) as NumPy arrays
    """
    var np = Python.import_module("numpy")

    # Convert to Mojo List
    var n = Int(prices_obj.__len__())
    var prices = List[Float64](capacity=n)
    for i in range(n):
        # Use py= keyword argument (required in Mojo 0.26+)
        prices.append(Float64(py=prices_obj[i]))

    # Calculate MACD components
    var ema_fast = ema_mojo(prices, fast_period)
    var ema_slow = ema_mojo(prices, slow_period)

    # MACD line
    var macd_line = List[Float64](capacity=n)
    for i in range(n):
        macd_line.append(ema_fast[i] - ema_slow[i])

    # Signal line
    var signal_line = ema_mojo(macd_line, signal_period)

    # Histogram
    var histogram = List[Float64](capacity=n)
    for i in range(n):
        histogram.append(macd_line[i] - signal_line[i])

    # Convert to NumPy arrays
    var macd_list = Python.evaluate("[]")
    var signal_list = Python.evaluate("[]")
    var hist_list = Python.evaluate("[]")

    for i in range(n):
        _ = macd_list.append(macd_line[i])
        _ = signal_list.append(signal_line[i])
        _ = hist_list.append(histogram[i])

    # Return tuple of arrays - build list manually
    var result_list = Python.evaluate("[]")
    _ = result_list.append(np.array(macd_list))
    _ = result_list.append(np.array(signal_list))
    _ = result_list.append(np.array(hist_list))
    return Python.evaluate("tuple")(result_list)


fn bollinger_bands_from_python(
    prices_obj: PythonObject,
    period: Int = 20,
    std_dev: Float64 = 2.0
) raises -> PythonObject:
    """
    Calculate Bollinger Bands from Python NumPy array.

    Args:
        prices_obj: NumPy array of prices
        period: SMA period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band) as NumPy arrays
    """
    var np = Python.import_module("numpy")

    # Convert to Mojo List
    var n = Int(prices_obj.__len__())
    var prices = List[Float64](capacity=n)
    for i in range(n):
        # Use py= keyword argument (required in Mojo 0.26+)
        prices.append(Float64(py=prices_obj[i]))

    # Initialize bands
    var upper_band = List[Float64](capacity=n)
    var middle_band = List[Float64](capacity=n)
    var lower_band = List[Float64](capacity=n)

    for _ in range(n):
        upper_band.append(0.0)
        middle_band.append(0.0)
        lower_band.append(0.0)

    if period <= 0 or period > n:
        var upper_list = Python.evaluate("[]")
        var middle_list = Python.evaluate("[]")
        var lower_list = Python.evaluate("[]")

        for i in range(n):
            _ = upper_list.append(0.0)
            _ = middle_list.append(0.0)
            _ = lower_list.append(0.0)

        var result_list = Python.evaluate("[]")
        _ = result_list.append(np.array(upper_list))
        _ = result_list.append(np.array(middle_list))
        _ = result_list.append(np.array(lower_list))
        return Python.evaluate("tuple")(result_list)

    # Calculate SMA and standard deviation for each point
    for i in range(period - 1, n):
        var sum: Float64 = 0.0
        for j in range(i - period + 1, i + 1):
            sum += prices[j]
        var sma = sum / Float64(period)
        middle_band[i] = sma

        # Calculate standard deviation
        var variance: Float64 = 0.0
        for j in range(i - period + 1, i + 1):
            var diff = prices[j] - sma
            variance += diff * diff
        var std = sqrt(variance / Float64(period))

        upper_band[i] = sma + std_dev * std
        lower_band[i] = sma - std_dev * std

    # Convert to NumPy arrays
    var upper_list = Python.evaluate("[]")
    var middle_list = Python.evaluate("[]")
    var lower_list = Python.evaluate("[]")

    for i in range(n):
        _ = upper_list.append(upper_band[i])
        _ = middle_list.append(middle_band[i])
        _ = lower_list.append(lower_band[i])

    var final_list = Python.evaluate("[]")
    _ = final_list.append(np.array(upper_list))
    _ = final_list.append(np.array(middle_list))
    _ = final_list.append(np.array(lower_list))
    return Python.evaluate("tuple")(final_list)


fn main():
    """Test the FFI integration."""
    print("🔥 Testing Mojo FFI Integration...")

    try:
        var np = Python.import_module("numpy")

        # Create test data
        # Create test data using Python list first
        var py_list = Python.evaluate("[100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0, 111.0, 110.0, 112.0, 114.0, 113.0, 115.0, 117.0, 116.0, 118.0, 120.0]")
        var prices = np.array(py_list)

        print("\n📊 Test data:", prices)

        # Test RSI
        print("\n📈 Testing RSI...")
        var rsi_result = rsi_from_python(prices, 14)
        print("  RSI:", rsi_result)

        # Test EMA
        print("\n📈 Testing EMA...")
        var ema_result = ema_from_python(prices, 9)
        print("  EMA:", ema_result)

        # Test SMA
        print("\n📈 Testing SMA...")
        var sma_result = sma_from_python(prices, 5)
        print("  SMA:", sma_result)

        # Test MACD
        print("\n📈 Testing MACD...")
        var macd_result = macd_from_python(prices, 12, 26, 9)
        print("  MACD:", macd_result)

        # Test Bollinger Bands
        print("\n📈 Testing Bollinger Bands...")
        var bb_result = bollinger_bands_from_python(prices, 10, 2.0)
        print("  Bollinger Bands:", bb_result)

        print("\n✅ All FFI functions working!")
        print("📊 Ready for Python integration")

    except e:
        print("❌ Error:", e)
