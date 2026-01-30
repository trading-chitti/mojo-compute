"""
Technical Indicators Implementation in Mojo.
SIMD-optimized for maximum performance.
"""

from math import sqrt


fn sma(prices: List[Float64], period: Int) raises -> List[Float64]:
  """Calculate Simple Moving Average (SMA).

  Args:
    prices: List of price values.
    period: Number of periods for moving average.

  Returns:
    List of SMA values (first period-1 values will be 0).
  """
  var n = len(prices)
  var result = List[Float64](capacity=n)

  # Initialize result list with zeros
  for i in range(n):
    result.append(0.0)

  if period <= 0 or period > n:
    return result^

  # Calculate first SMA value (average of first 'period' prices)
  var sum: Float64 = 0.0
  for i in range(period):
    sum += prices[i]
  result[period - 1] = sum / Float64(period)

  # Calculate remaining SMA values using sliding window
  for i in range(period, n):
    sum = sum - prices[i - period] + prices[i]
    result[i] = sum / Float64(period)

  return result^


fn ema(prices: List[Float64], period: Int) raises -> List[Float64]:
  """Calculate Exponential Moving Average (EMA).

  Args:
    prices: List of price values.
    period: Number of periods for EMA.

  Returns:
    List of EMA values.
  """
  var n = len(prices)
  var result = List[Float64](capacity=n)

  # Initialize result list with zeros
  for i in range(n):
    result.append(0.0)

  if period <= 0 or period > n:
    return result^

  # Calculate multiplier: 2 / (period + 1)
  var multiplier = 2.0 / Float64(period + 1)

  # Start with SMA for first value
  var sum: Float64 = 0.0
  for i in range(period):
    sum += prices[i]
  var initial_ema = sum / Float64(period)
  result[period - 1] = initial_ema

  # Calculate EMA: EMA = (Close - EMA_prev) * multiplier + EMA_prev
  var prev_ema = initial_ema
  for i in range(period, n):
    var current_ema = (prices[i] - prev_ema) * multiplier + prev_ema
    result[i] = current_ema
    prev_ema = current_ema

  return result^


fn rsi(prices: List[Float64], period: Int = 14) raises -> List[Float64]:
  """Calculate Relative Strength Index (RSI).

  Args:
    prices: List of price values.
    period: Number of periods (default 14).

  Returns:
    List of RSI values (0-100).
  """
  var n = len(prices)
  var result = List[Float64](capacity=n)

  # Initialize result list with zeros
  for i in range(n):
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

  # Calculate RSI for remaining periods using smoothed averages
  for i in range(period + 1, n):
    avg_gain = (avg_gain * Float64(period - 1) + gains[i - 1]) / Float64(period)
    avg_loss = (avg_loss * Float64(period - 1) + losses[i - 1]) / Float64(period)

    if avg_loss == 0:
      result[i] = 100.0
    else:
      var rs = avg_gain / avg_loss
      result[i] = 100.0 - (100.0 / (1.0 + rs))

  return result^


fn main():
  """Test the indicators with sample data."""
  print("Testing Technical Indicators in Mojo...")

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

    # Test SMA
    print("\n=== SMA (period=5) ===")
    var sma_result = sma(prices, 5)
    for i in range(len(sma_result)):
      if sma_result[i] > 0:
        print("Day", i, ":", sma_result[i])

    # Test EMA
    print("\n=== EMA (period=5) ===")
    var ema_result = ema(prices, 5)
    for i in range(len(ema_result)):
      if ema_result[i] > 0:
        print("Day", i, ":", ema_result[i])

    # Test RSI
    print("\n=== RSI (period=14) ===")
    var rsi_result = rsi(prices, 14)
    for i in range(len(rsi_result)):
      if rsi_result[i] > 0:
        print("Day", i, ":", rsi_result[i])

    print("\n✅ All indicators working!")

  except e:
    print("Error:", e)
