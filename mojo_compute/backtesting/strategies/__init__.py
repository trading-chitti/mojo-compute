"""
Strategy implementations for backtesting.

This module contains 40+ trading strategy implementations across multiple categories:
- Trend-following
- Breakout
- Mean Reversion
- Momentum
- Swing
- Reversal
- Statistical Arbitrage
- Portfolio
"""

from .ma_crossover import MACrossoverStrategy
from .rsi_reversal import RSIReversalStrategy
from .bollinger_reversion import BollingerReversionStrategy
from .orb import OpeningRangeBreakoutStrategy
from .donchian_breakout import DonchianBreakoutStrategy

# Strategy registry - maps strategy_id to Strategy class
STRATEGY_REGISTRY = {
    # Trend-following (10 strategies)
    'ma_crossover': MACrossoverStrategy,
    'ma_pullback': None,  # TODO: Implement
    'donchian_breakout': DonchianBreakoutStrategy,
    'week52_high': None,  # TODO: Implement
    'supertrend': None,  # TODO: Implement
    'adx_trend': None,  # TODO: Implement
    'ichimoku': None,  # TODO: Implement
    'trendline_breakout': None,  # TODO: Implement
    'turtle_trading': None,  # TODO: Implement
    'parabolic_sar': None,  # TODO: Implement

    # Breakout (7 strategies)
    'orb': OpeningRangeBreakoutStrategy,
    'volatility_breakout': None,  # TODO: Implement
    'box_breakout': None,  # TODO: Implement
    'triangle_breakout': None,  # TODO: Implement
    'flag_pennant': None,  # TODO: Implement
    'gap_and_go': None,  # TODO: Implement
    'volume_breakout': None,  # TODO: Implement

    # Mean Reversion (6 strategies)
    'rsi_reversal': RSIReversalStrategy,
    'bollinger_reversion': BollingerReversionStrategy,
    'zscore_reversion': None,  # TODO: Implement
    'vwap_reversion': None,  # TODO: Implement
    'ma_reversion': None,  # TODO: Implement
    'support_resistance_bounce': None,  # TODO: Implement

    # Momentum (5 strategies)
    'time_series_momentum': None,  # TODO: Implement
    'cross_sectional_momentum': None,  # TODO: Implement
    'relative_strength': None,  # TODO: Implement
    'pead': None,  # TODO: Implement
    'news_momentum': None,  # TODO: Implement

    # Swing (4 strategies)
    'pullback_to_support': None,  # TODO: Implement
    'breakout_retest': None,  # TODO: Implement
    'one_two_three_reversal': None,  # TODO: Implement
    'multi_timeframe_trend': None,  # TODO: Implement

    # Reversal (5 strategies)
    'rsi_macd_divergence': None,  # TODO: Implement
    'exhaustion_move': None,  # TODO: Implement
    'climax_volume': None,  # TODO: Implement
    'double_top_bottom': None,  # TODO: Implement
    'head_and_shoulders': None,  # TODO: Implement

    # Statistical Arbitrage (4 strategies)
    'pairs_trading': None,  # TODO: Implement
    'cointegration_spread': None,  # TODO: Implement
    'basket_arbitrage': None,  # TODO: Implement
    'index_vs_constituents': None,  # TODO: Implement

    # Portfolio (5 strategies)
    'value_investing': None,  # TODO: Implement
    'quality_factor': None,  # TODO: Implement
    'low_volatility': None,  # TODO: Implement
    'dividend_yield': None,  # TODO: Implement
    'trend_plus_rebalance': None,  # TODO: Implement
}


def get_strategy(strategy_id: str, params: dict = None):
    """
    Get strategy instance by ID.

    Args:
        strategy_id: Strategy identifier (e.g., 'ma_crossover')
        params: Strategy parameters

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy_id not found or not implemented
    """
    if strategy_id not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_id}")

    strategy_class = STRATEGY_REGISTRY[strategy_id]
    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy_id}' not implemented yet")

    return strategy_class(params)


def list_strategies():
    """
    List all available strategies.

    Returns:
        Dict of {strategy_id: {'implemented': bool, 'class': str}}
    """
    return {
        sid: {
            'implemented': cls is not None,
            'class': cls.__name__ if cls else None
        }
        for sid, cls in STRATEGY_REGISTRY.items()
    }


__all__ = [
    'MACrossoverStrategy',
    'RSIReversalStrategy',
    'BollingerReversionStrategy',
    'OpeningRangeBreakoutStrategy',
    'DonchianBreakoutStrategy',
    'STRATEGY_REGISTRY',
    'get_strategy',
    'list_strategies',
]
