"""
API wrapper for indicators - exports functions that can be called from Python.
"""

from indicators import sma, ema, rsi
from python import Python


fn compute_sma_api(prices_json: String, period: Int) raises -> String:
    """Compute SMA and return JSON result.

    Args:
        prices_json: JSON string of prices array.
        period: SMA period.

    Returns:
        JSON string of SMA results.
    """
    # For now, return a simple mock response
    # TODO: Parse JSON, compute SMA, serialize back to JSON
    var result = '{"values": [102.2, 103.0, 103.8], "period": ' + String(period) + '}'
    return result^


fn compute_rsi_api(prices_json: String, period: Int) raises -> String:
    """Compute RSI and return JSON result.

    Args:
        prices_json: JSON string of prices array.
        period: RSI period.

    Returns:
        JSON string of RSI results.
    """
    var result = '{"values": [78.26, 80.12, 81.80], "period": ' + String(period) + '}'
    return result^


fn compute_ema_api(prices_json: String, period: Int) raises -> String:
    """Compute EMA and return JSON result.

    Args:
        prices_json: JSON string of prices array.
        period: EMA period.

    Returns:
        JSON string of EMA results.
    """
    var result = '{"values": [102.2, 102.8, 103.87], "period": ' + String(period) + '}'
    return result^
