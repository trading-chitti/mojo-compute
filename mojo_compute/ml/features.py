"""
Feature engineering pipeline for ML models.

Generates technical indicators and derived features from OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """
    Generate ML features from OHLCV data.

    Features include:
    - Price features: returns, momentum
    - Moving averages and ratios
    - Technical indicators: RSI, MACD, Bollinger Bands
    - Volume features
    - Volatility measures
    """

    def __init__(self):
        self.feature_columns = []

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive feature set from OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with all features added
        """
        df = df.copy()

        # Price features
        df = self._add_price_features(df)

        # Moving averages
        df = self._add_moving_averages(df)

        # Technical indicators
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)

        # Volume features
        df = self._add_volume_features(df)

        # Volatility
        df = self._add_volatility(df)

        # Lag features
        df = self._add_lag_features(df)

        # Target variable (next day return and direction)
        df['target_return_1d'] = df['close'].shift(-1) / df['close'] - 1
        df['target_direction'] = (df['target_return_1d'] > 0).astype(int)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        df['returns_1d'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Momentum (5-day and 10-day)
        df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['price_momentum_10'] = df['close'] / df['close'].shift(10) - 1

        # High-low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']

        # Close position in daily range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving averages and ratios."""
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1

        # EMA
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        return df

    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator."""
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()

        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df

    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands."""
        df[f'bb_middle_{period}'] = df['close'].rolling(period).mean()
        df[f'bb_std_{period}'] = df['close'].rolling(period).std()
        df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + std_dev * df[f'bb_std_{period}']
        df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - std_dev * df[f'bb_std_{period}']

        # BB position (where price is within bands)
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (
            df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10
        )

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving averages
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)

        # Volume momentum
        df['volume_momentum_5'] = df['volume'] / (df['volume'].shift(5) + 1e-10) - 1

        # Price-volume features
        df['pv_trend'] = df['returns_1d'] * df['volume_ratio']

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures."""
        # Historical volatility (rolling std of returns)
        df['volatility_10'] = df['returns_1d'].rolling(10).std()
        df['volatility_20'] = df['returns_1d'].rolling(20).std()

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        # Lag close prices
        for lag in [1, 2, 3, 5]:
            df[f'close_lag{lag}'] = df['close'].shift(lag)

        # Lag volume
        for lag in [1, 2]:
            df[f'volume_lag{lag}'] = df['volume'].shift(lag)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding OHLCV and target).

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        exclude = ['open', 'high', 'low', 'close', 'volume',
                   'target_return_1d', 'target_direction']

        feature_cols = [col for col in df.columns if col not in exclude]
        return feature_cols

    def prepare_for_training(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features and targets for model training.

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (X, y) where X is features, y is target
        """
        df = df.dropna()  # Remove rows with NaN

        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols]
        y = df['target_direction']

        return X, y
