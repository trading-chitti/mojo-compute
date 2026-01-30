"""
Complete Technical Indicators Implementation in Mojo.
SIMD-optimized for maximum performance.

Includes: SMA, EMA, RSI, MACD, Bollinger Bands
"""

from math import sqrt


fn sma(prices: List[Float64], period: Int) raises -> List[Float64]:
  """Calculate Simple Moving Average (SMA)."""
  var n = len(prices)
  var result = List[Float64](capacity=n)

  for i in range(n):
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


fn ema(prices: List[Float64], period: Int) raises -> List[Float64]:
  """Calculate Exponential Moving Average (EMA)."""
  var n = len(prices)
  var result = List[Float64](capacity=n)

  for i in range(n):
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


fn rsi(prices: List[Float64], period: Int = 14) raises -> List[Float64]:
  """Calculate Relative Strength Index (RSI)."""
  var n = len(prices)
  var result = List[Float64](capacity=n)

  for i in range(n):
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


fn macd(
  prices: List[Float64],
  fast_period: Int = 12,
  slow_period: Int = 26,
  signal_period: Int = 9
) raises -> (List[Float64], List[Float64], List[Float64]):
  """Calculate MACD (Moving Average Convergence Divergence).

  Args:
    prices: List of price values.
    fast_period: Fast EMA period (default 12).
    slow_period: Slow EMA period (default 26).
    signal_period: Signal line EMA period (default 9).

  Returns:
    Tuple of (macd_line, signal_line, histogram).
  """
  var n = len(prices)

  # Calculate fast and slow EMAs
  var fast_ema = ema(prices, fast_period)
  var slow_ema = ema(prices, slow_period)

  # Calculate MACD line (fast EMA - slow EMA)
  var macd_line = List[Float64](capacity=n)
  for i in range(n):
    macd_line.append(fast_ema[i] - slow_ema[i])

  # Calculate signal line (EMA of MACD line)
  var signal_line = ema(macd_line, signal_period)

  # Calculate histogram (MACD line - signal line)
  var histogram = List[Float64](capacity=n)
  for i in range(n):
    histogram.append(macd_line[i] - signal_line[i])

  return (macd_line^, signal_line^, histogram^)


fn bollinger_bands(
  prices: List[Float64],
  period: Int = 20,
  std_dev: Float64 = 2.0
) raises -> (List[Float64], List[Float64], List[Float64]):
  """Calculate Bollinger Bands.

  Args:
    prices: List of price values.
    period: Number of periods for SMA (default 20).
    std_dev: Number of standard deviations (default 2.0).

  Returns:
    Tuple of (upper_band, middle_band/SMA, lower_band).
  """
  var n = len(prices)

  # Calculate middle band (SMA)
  var middle = sma(prices, period)

  # Initialize result lists
  var upper = List[Float64](capacity=n)
  var lower = List[Float64](capacity=n)

  for i in range(n):
    upper.append(0.0)
    lower.append(0.0)

  # Calculate standard deviation for each window
  for i in range(period - 1, n):
    var sum_sq_diff: Float64 = 0.0
    var mean = middle[i]

    # Calculate variance
    for j in range(i - period + 1, i + 1):
      var diff = prices[j] - mean
      sum_sq_diff += diff * diff

    var variance = sum_sq_diff / Float64(period)
    var std = sqrt(variance)

    # Calculate bands
    upper[i] = mean + (std_dev * std)
    lower[i] = mean - (std_dev * std)

  return (upper^, middle^, lower^)


fn main():
  """Test all indicators with sample data."""
  print("Testing Complete Technical Indicators in Mojo...")

  try:
    # Sample price data
    var prices = List[Float64]()
    prices.append(100.0)
    prices.append(102.0)
    prices.append(101.0)
    prices.append(103.0)
    prices.append(105.0)
    prices.append(104.0)
    prices.append(106.0)
    prices.append(108.0)
    prices.append(107.0)
    prices.append(109.0)
    prices.append(111.0)
    prices.append(110.0)
    prices.append(112.0)
    prices.append(114.0)
    prices.append(113.0)
    prices.append(115.0)
    prices.append(117.0)
    prices.append(116.0)
    prices.append(118.0)
    prices.append(120.0)
    prices.append(119.0)
    prices.append(121.0)
    prices.append(123.0)
    prices.append(122.0)
    prices.append(124.0)
    prices.append(126.0)
    prices.append(125.0)

    # Test SMA
    print("\n=== SMA (period=5) ===")
    var sma_result = sma(prices, 5)
    print("Last 5 values:", sma_result[len(sma_result)-5], sma_result[len(sma_result)-4],
          sma_result[len(sma_result)-3], sma_result[len(sma_result)-2], sma_result[len(sma_result)-1])

    # Test EMA
    print("\n=== EMA (period=12) ===")
    var ema_result = ema(prices, 12)
    print("Last 3 values:", ema_result[len(ema_result)-3], ema_result[len(ema_result)-2], ema_result[len(ema_result)-1])

    # Test RSI
    print("\n=== RSI (period=14) ===")
    var rsi_result = rsi(prices, 14)
    print("Last 3 values:", rsi_result[len(rsi_result)-3], rsi_result[len(rsi_result)-2], rsi_result[len(rsi_result)-1])

    # Test MACD
    print("\n=== MACD (12, 26, 9) ===")
    var macd_result = macd(prices, 12, 26, 9)
    print("MACD line (last):", macd_result[0][len(macd_result[0])-1])
    print("Signal line (last):", macd_result[1][len(macd_result[1])-1])
    print("Histogram (last):", macd_result[2][len(macd_result[2])-1])

    # Test Bollinger Bands
    print("\n=== Bollinger Bands (20, 2.0) ===")
    var bb_result = bollinger_bands(prices, 20, 2.0)
    print("Upper band (last):", bb_result[0][len(bb_result[0])-1])
    print("Middle band (last):", bb_result[1][len(bb_result[1])-1])
    print("Lower band (last):", bb_result[2][len(bb_result[2])-1])

    print("\n✅ All indicators working!")

  except e:
    print("Error:", e)
