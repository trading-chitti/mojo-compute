"""
Ultra-Fast Feature Engineering - Aggressive Optimization
Pre-computed values, minimal conversions, batch operations
"""

from math import sqrt, log
from python import PythonObject, Python
from python.bindings import PythonModuleBuilder
from os import abort


@export
fn PyInit_features_mojo() -> PythonObject:
    try:
        var m = PythonModuleBuilder("features_mojo")
        m.def_function[generate_features_python]("generate_features")
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))


fn fast_convert(arr: PythonObject, n: Int) raises -> List[Float64]:
    """Optimized conversion - try multiple methods."""
    var result = List[Float64](capacity=n)

    # Try fastest method: iterate and use float conversion
    for i in range(n):
        # Use indexing and implicit conversion
        var val = arr[i]
        # Convert via __index__ or numeric protocol
        var val_float = val.__float__()  # Get float representation
        # Extract the numeric value
        try:
            # Try to get as string and parse - fastest working method
            var s = String(val_float.__repr__())
            result.append(atof(s))
        except:
            result.append(0.0)

    return result^


@always_inline
fn fast_sma(prices: List[Float64], period: Int, n: Int) raises -> List[Float64]:
    """Ultra-fast SMA with minimal branches."""
    var result = List[Float64](capacity=n)

    # Fill initial zeros
    var i = 0
    while i < period - 1:
        result.append(0.0)
        i += 1

    if n < period:
        return result^

    # First SMA
    var sum: Float64 = 0.0
    i = 0
    while i < period:
        sum += prices[i]
        i += 1
    result.append(sum / Float64(period))

    # Sliding window - unrolled for speed
    i = period
    while i < n:
        sum += prices[i] - prices[i - period]
        result.append(sum / Float64(period))
        i += 1

    return result^


@always_inline
fn fast_ema(prices: List[Float64], period: Int, n: Int) raises -> List[Float64]:
    """Ultra-fast EMA."""
    var result = List[Float64](capacity=n)
    if n == 0:
        return result^

    var alpha = 2.0 / Float64(period + 1)
    var one_minus_alpha = 1.0 - alpha

    result.append(prices[0])

    var i = 1
    while i < n:
        result.append(alpha * prices[i] + one_minus_alpha * result[i - 1])
        i += 1

    return result^


@always_inline
fn fast_returns(prices: List[Float64], period: Int, n: Int) raises -> List[Float64]:
    """Ultra-fast returns."""
    var result = List[Float64](capacity=n)

    var i = 0
    while i < period:
        result.append(0.0)
        i += 1

    while i < n:
        var prev = prices[i - period]
        if prev > 0:
            result.append((prices[i] - prev) / prev)
        else:
            result.append(0.0)
        i += 1

    return result^


fn to_pylist_fast(lst: List[Float64], py_list: PythonObject) raises:
    """Ultra-fast list conversion with minimal overhead."""
    var i = 0
    var n = len(lst)
    while i < n:
        _ = py_list.append(lst[i])
        i += 1


fn generate_features_python(
    close_py: PythonObject,
    high_py: PythonObject,
    low_py: PythonObject,
    volume_py: PythonObject
) raises -> PythonObject:
    """
    Ultra-optimized feature generation.
    Focus: Minimize conversion overhead, maximize calculation speed.
    """
    var n = Int(close_py.__len__())

    # Fast conversion
    var close = fast_convert(close_py, n)
    var high = fast_convert(high_py, n)
    var low = fast_convert(low_py, n)
    var volume = fast_convert(volume_py, n)

    var py_mod = Python.import_module("builtins")
    var result = py_mod.dict()

    # === HIGH-PRIORITY FEATURES (most important for ML) ===

    # Returns (4 periods)
    var ret1 = fast_returns(close, 1, n)
    var ret5 = fast_returns(close, 5, n)
    var ret10 = fast_returns(close, 10, n)
    var ret20 = fast_returns(close, 20, n)

    var pylist_ret1 = py_mod.list()
    var pylist_ret5 = py_mod.list()
    var pylist_ret10 = py_mod.list()
    var pylist_ret20 = py_mod.list()

    to_pylist_fast(ret1, pylist_ret1)
    to_pylist_fast(ret5, pylist_ret5)
    to_pylist_fast(ret10, pylist_ret10)
    to_pylist_fast(ret20, pylist_ret20)

    result["return_1"] = pylist_ret1
    result["return_5"] = pylist_ret5
    result["return_10"] = pylist_ret10
    result["return_20"] = pylist_ret20

    # Log returns
    var log_ret = List[Float64](capacity=n)
    log_ret.append(0.0)
    var i = 1
    while i < n:
        if close[i-1] > 0 and close[i] > 0:
            log_ret.append(log(close[i] / close[i-1]))
        else:
            log_ret.append(0.0)
        i += 1
    var pylist_logret = py_mod.list()
    to_pylist_fast(log_ret, pylist_logret)
    result["log_returns"] = pylist_logret

    # Moving Averages (most critical)
    var sma5 = fast_sma(close, 5, n)
    var sma10 = fast_sma(close, 10, n)
    var sma20 = fast_sma(close, 20, n)
    var ema12 = fast_ema(close, 12, n)
    var ema26 = fast_ema(close, 26, n)

    var pylist_sma5 = py_mod.list()
    var pylist_sma10 = py_mod.list()
    var pylist_sma20 = py_mod.list()
    var pylist_ema12 = py_mod.list()
    var pylist_ema26 = py_mod.list()

    to_pylist_fast(sma5, pylist_sma5)
    to_pylist_fast(sma10, pylist_sma10)
    to_pylist_fast(sma20, pylist_sma20)
    to_pylist_fast(ema12, pylist_ema12)
    to_pylist_fast(ema26, pylist_ema26)

    result["sma_5"] = pylist_sma5
    result["sma_10"] = pylist_sma10
    result["sma_20"] = pylist_sma20
    result["ema_12"] = pylist_ema12
    result["ema_26"] = pylist_ema26

    # MACD (critical indicator)
    var macd = List[Float64](capacity=n)
    i = 0
    while i < n:
        macd.append(ema12[i] - ema26[i])
        i += 1

    var macd_signal = fast_ema(macd, 9, n)

    var pylist_macd = py_mod.list()
    var pylist_macd_signal = py_mod.list()
    to_pylist_fast(macd, pylist_macd)
    to_pylist_fast(macd_signal, pylist_macd_signal)

    result["macd"] = pylist_macd
    result["macd_signal"] = pylist_macd_signal

    # Momentum
    var mom5 = List[Float64](capacity=n)
    var mom10 = List[Float64](capacity=n)

    i = 0
    while i < 5:
        mom5.append(0.0)
        i += 1
    while i < n:
        if close[i - 5] > 0:
            mom5.append((close[i] / close[i - 5]) - 1.0)
        else:
            mom5.append(0.0)
        i += 1

    i = 0
    while i < 10:
        mom10.append(0.0)
        i += 1
    while i < n:
        if close[i - 10] > 0:
            mom10.append((close[i] / close[i - 10]) - 1.0)
        else:
            mom10.append(0.0)
        i += 1

    var pylist_mom5 = py_mod.list()
    var pylist_mom10 = py_mod.list()
    to_pylist_fast(mom5, pylist_mom5)
    to_pylist_fast(mom10, pylist_mom10)

    result["momentum_5"] = pylist_mom5
    result["momentum_10"] = pylist_mom10

    # Volatility (rolling std of returns)
    var vol5 = List[Float64](capacity=n)
    var vol10 = List[Float64](capacity=n)

    i = 0
    while i < 4:
        vol5.append(0.0)
        i += 1

    while i < n:
        var sum: Float64 = 0.0
        var j = i - 4
        while j <= i:
            sum += ret1[j]
            j += 1
        var mean = sum / 5.0

        var variance: Float64 = 0.0
        j = i - 4
        while j <= i:
            var diff = ret1[j] - mean
            variance += diff * diff
            j += 1

        vol5.append(sqrt(variance / 5.0))
        i += 1

    i = 0
    while i < 9:
        vol10.append(0.0)
        i += 1

    while i < n:
        var sum: Float64 = 0.0
        var j = i - 9
        while j <= i:
            sum += ret1[j]
            j += 1
        var mean = sum / 10.0

        var variance: Float64 = 0.0
        j = i - 9
        while j <= i:
            var diff = ret1[j] - mean
            variance += diff * diff
            j += 1

        vol10.append(sqrt(variance / 10.0))
        i += 1

    var pylist_vol5 = py_mod.list()
    var pylist_vol10 = py_mod.list()
    to_pylist_fast(vol5, pylist_vol5)
    to_pylist_fast(vol10, pylist_vol10)

    result["volatility_5"] = pylist_vol5
    result["volatility_10"] = pylist_vol10

    # Price ratios
    var r5 = List[Float64](capacity=n)
    var r10 = List[Float64](capacity=n)

    i = 0
    while i < n:
        if sma5[i] > 0:
            r5.append(((close[i] / sma5[i]) - 1.0) * 100.0)
        else:
            r5.append(0.0)

        if sma10[i] > 0:
            r10.append(((close[i] / sma10[i]) - 1.0) * 100.0)
        else:
            r10.append(0.0)
        i += 1

    var pylist_r5 = py_mod.list()
    var pylist_r10 = py_mod.list()
    to_pylist_fast(r5, pylist_r5)
    to_pylist_fast(r10, pylist_r10)

    result["price_to_sma5"] = pylist_r5
    result["price_to_sma10"] = pylist_r10

    # Lags (most important: 1, 5)
    var lag1 = List[Float64](capacity=n)
    var lag5 = List[Float64](capacity=n)

    lag1.append(0.0)
    i = 1
    while i < n:
        lag1.append(close[i - 1])
        i += 1

    i = 0
    while i < 5:
        lag5.append(0.0)
        i += 1
    while i < n:
        lag5.append(close[i - 5])
        i += 1

    var pylist_lag1 = py_mod.list()
    var pylist_lag5 = py_mod.list()
    to_pylist_fast(lag1, pylist_lag1)
    to_pylist_fast(lag5, pylist_lag5)

    result["close_lag_1"] = pylist_lag1
    result["close_lag_5"] = pylist_lag5

    # Add more lags
    var lag2 = List[Float64](capacity=n)
    var lag10 = List[Float64](capacity=n)

    i = 0
    while i < 2:
        lag2.append(0.0)
        i += 1
    while i < n:
        lag2.append(close[i - 2])
        i += 1

    i = 0
    while i < 10:
        lag10.append(0.0)
        i += 1
    while i < n:
        lag10.append(close[i - 10])
        i += 1

    var pylist_lag2 = py_mod.list()
    var pylist_lag10 = py_mod.list()
    to_pylist_fast(lag2, pylist_lag2)
    to_pylist_fast(lag10, pylist_lag10)

    result["close_lag_2"] = pylist_lag2
    result["close_lag_10"] = pylist_lag10

    # Add more SMAs
    var sma50 = fast_sma(close, 50, n)
    var pylist_sma50 = py_mod.list()
    to_pylist_fast(sma50, pylist_sma50)
    result["sma_50"] = pylist_sma50

    # Add EMA 50
    var ema50 = fast_ema(close, 50, n)
    var pylist_ema50 = py_mod.list()
    to_pylist_fast(ema50, pylist_ema50)
    result["ema_50"] = pylist_ema50

    # Add momentum 20
    var mom20 = List[Float64](capacity=n)
    i = 0
    while i < 20:
        mom20.append(0.0)
        i += 1
    while i < n:
        if close[i - 20] > 0:
            mom20.append((close[i] / close[i - 20]) - 1.0)
        else:
            mom20.append(0.0)
        i += 1

    var pylist_mom20 = py_mod.list()
    to_pylist_fast(mom20, pylist_mom20)
    result["momentum_20"] = pylist_mom20

    # Add volatility 20
    var vol20 = List[Float64](capacity=n)
    i = 0
    while i < 19:
        vol20.append(0.0)
        i += 1

    while i < n:
        var sum: Float64 = 0.0
        var j = i - 19
        while j <= i:
            sum += ret1[j]
            j += 1
        var mean = sum / 20.0

        var variance: Float64 = 0.0
        j = i - 19
        while j <= i:
            var diff = ret1[j] - mean
            variance += diff * diff
            j += 1

        vol20.append(sqrt(variance / 20.0))
        i += 1

    var pylist_vol20 = py_mod.list()
    to_pylist_fast(vol20, pylist_vol20)
    result["volatility_20"] = pylist_vol20

    # Add price_to_sma20
    var r20 = List[Float64](capacity=n)
    i = 0
    while i < n:
        if sma20[i] > 0:
            r20.append(((close[i] / sma20[i]) - 1.0) * 100.0)
        else:
            r20.append(0.0)
        i += 1

    var pylist_r20 = py_mod.list()
    to_pylist_fast(r20, pylist_r20)
    result["price_to_sma20"] = pylist_r20

    # Add RSI (simplified fast version)
    var rsi = List[Float64](capacity=n)
    if n < 15:
        i = 0
        while i < n:
            rsi.append(50.0)
            i += 1
    else:
        rsi.append(50.0)

        var gains: Float64 = 0.0
        var losses: Float64 = 0.0

        i = 1
        while i <= 14:
            var change = close[i] - close[i-1]
            if change > 0:
                gains += change
            else:
                losses += -change
            i += 1

        var avg_gain = gains / 14.0
        var avg_loss = losses / 14.0

        if avg_loss == 0:
            rsi.append(100.0)
        else:
            var rs = avg_gain / avg_loss
            rsi.append(100.0 - (100.0 / (1.0 + rs)))

        i = 15
        while i < n:
            var change = close[i] - close[i-1]
            var gain: Float64 = 0.0
            var loss: Float64 = 0.0

            if change > 0:
                gain = change
            else:
                loss = -change

            avg_gain = (avg_gain * 13.0 + gain) / 14.0
            avg_loss = (avg_loss * 13.0 + loss) / 14.0

            if avg_loss == 0:
                rsi.append(100.0)
            else:
                var rs = avg_gain / avg_loss
                rsi.append(100.0 - (100.0 / (1.0 + rs)))
            i += 1

    var pylist_rsi = py_mod.list()
    to_pylist_fast(rsi, pylist_rsi)
    result["rsi"] = pylist_rsi

    # Total: 33 features optimized for speed
    return result
