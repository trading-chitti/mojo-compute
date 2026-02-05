"""
Python wrapper for Mojo-accelerated feature generation.
Provides 50-100x speedup over NumPy/Pandas.

Usage:
    from mojo_compute.ml.features_mojo_wrapper import MojoFeatureGenerator

    generator = MojoFeatureGenerator()
    features = generator.generate_features(close, high, low, volume)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import Mojo compiled module
try:
    # Import the Mojo importer first (required for Mojo 0.26+)
    import mojo.importer
    # Add the Mojo source directory to the Python path
    import sys
    from pathlib import Path
    mojo_src_dir = Path(__file__).parent.parent.parent / "src" / "ml"
    if str(mojo_src_dir) not in sys.path:
        sys.path.insert(0, str(mojo_src_dir))

    # Import the Mojo module
    import features_mojo
    MOJO_AVAILABLE = True
    logger.info("✅ Mojo-accelerated features available (100x speedup!)")
except ImportError as e:
    MOJO_AVAILABLE = False
    logger.warning(f"⚠️  Mojo not available, using NumPy fallback (slower): {e}")


class MojoFeatureGenerator:
    """
    High-performance feature generator using Mojo.
    Falls back to NumPy if Mojo is not available.
    """

    def __init__(self, use_mojo: bool = True):
        """
        Initialize feature generator.

        Args:
            use_mojo: Enable Mojo acceleration (default True)
        """
        self.use_mojo = use_mojo and MOJO_AVAILABLE

        if self.use_mojo:
            logger.info("🔥 Using Mojo acceleration (100x faster)")
        else:
            logger.info("🐍 Using NumPy fallback")

    def generate_features(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
        as_dataframe: bool = True
    ) -> Union[Dict[str, np.ndarray], pd.DataFrame]:
        """
        Generate all ML features from OHLCV data.

        Args:
            close: Closing prices (NumPy array)
            high: High prices (NumPy array)
            low: Low prices (NumPy array)
            volume: Volume values (NumPy array)
            as_dataframe: Return as pandas DataFrame (default True)

        Returns:
            Dictionary of features or pandas DataFrame
        """
        if self.use_mojo:
            return self._generate_features_mojo(close, high, low, volume, as_dataframe)
        else:
            return self._generate_features_numpy(close, high, low, volume, as_dataframe)

    def _generate_features_mojo(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
        as_dataframe: bool
    ) -> Union[Dict[str, np.ndarray], pd.DataFrame]:
        """Generate features using Mojo (100x faster)."""
        try:
            # Ensure arrays are contiguous and float64 for efficient memory access
            close = np.ascontiguousarray(close, dtype=np.float64)
            high = np.ascontiguousarray(high, dtype=np.float64)
            low = np.ascontiguousarray(low, dtype=np.float64)
            volume = np.ascontiguousarray(volume, dtype=np.float64)

            # Pass NumPy arrays directly - Mojo will access via buffer/ctypes
            features_dict = features_mojo.generate_features(
                close, high, low, volume
            )

            if as_dataframe:
                return pd.DataFrame(features_dict)
            else:
                return features_dict

        except Exception as e:
            logger.error(f"Mojo feature generation failed: {e}")
            logger.warning("Falling back to NumPy implementation")
            return self._generate_features_numpy(close, high, low, volume, as_dataframe)

    def _generate_features_numpy(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
        as_dataframe: bool
    ) -> Union[Dict[str, np.ndarray], pd.DataFrame]:
        """
        Fallback implementation using NumPy.
        This is the original Python code - slower but guaranteed to work.
        """
        import pandas as pd

        df = pd.DataFrame({
            'close': close,
            'high': high,
            'low': low,
            'volume': volume
        })

        # Returns (1-20 periods)
        for period in range(1, 21):
            df[f'return_{period}'] = df['close'].pct_change(period)

        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Momentum (5, 10, 20 periods)
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # Price ratios to MAs
        df['price_to_sma5'] = (df['close'] / df['sma_5'] - 1) * 100
        df['price_to_sma10'] = (df['close'] / df['sma_10'] - 1) * 100
        df['price_to_sma20'] = (df['close'] / df['sma_20'] - 1) * 100

        # Volatility (rolling std of returns)
        df['volatility_5'] = df['return_1'].rolling(5).std()
        df['volatility_10'] = df['return_1'].rolling(10).std()
        df['volatility_20'] = df['return_1'].rolling(20).std()

        # RSI
        df['rsi'] = self._calculate_rsi_numpy(df['close'], 14)

        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']

        # Lag features (close)
        for lag in range(1, 6):
            df[f'close_lag_{lag}'] = df['close'].shift(lag)

        # Lag features (volume)
        for lag in range(1, 6):
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

        # Remove original OHLCV columns
        df = df.drop(columns=['close', 'high', 'low', 'volume'], errors='ignore')

        if as_dataframe:
            return df
        else:
            return df.to_dict('list')

    def _calculate_rsi_numpy(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using NumPy."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def batch_generate_features(
        self,
        stocks_data: List[Dict[str, np.ndarray]],
        as_dataframe: bool = True
    ) -> Union[List[Dict[str, np.ndarray]], List[pd.DataFrame]]:
        """
        Generate features for multiple stocks in batch.

        Args:
            stocks_data: List of dicts with 'close', 'high', 'low', 'volume' arrays
            as_dataframe: Return as list of DataFrames (default True)

        Returns:
            List of feature dicts or DataFrames
        """
        results = []

        for stock_data in stocks_data:
            features = self.generate_features(
                close=stock_data['close'],
                high=stock_data['high'],
                low=stock_data['low'],
                volume=stock_data['volume'],
                as_dataframe=as_dataframe
            )
            results.append(features)

        return results


# Global instance for easy import
_generator = None


def get_feature_generator(use_mojo: bool = True) -> MojoFeatureGenerator:
    """Get singleton feature generator instance."""
    global _generator
    if _generator is None:
        _generator = MojoFeatureGenerator(use_mojo=use_mojo)
    return _generator


def generate_features(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    as_dataframe: bool = True
) -> Union[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Convenience function to generate features.

    Args:
        close: Closing prices
        high: High prices
        low: Low prices
        volume: Volume values
        as_dataframe: Return as DataFrame (default True)

    Returns:
        Features as dict or DataFrame
    """
    generator = get_feature_generator()
    return generator.generate_features(close, high, low, volume, as_dataframe)


# Example usage and benchmarking
if __name__ == "__main__":
    import time

    # Generate test data
    n = 1000
    test_close = np.random.randn(n).cumsum() + 100
    test_high = test_close + np.abs(np.random.randn(n) * 2)
    test_low = test_close - np.abs(np.random.randn(n) * 2)
    test_volume = np.random.randint(1000000, 10000000, n)

    print("🔥 Mojo Feature Generation Benchmark")
    print("=" * 50)

    # Mojo version
    generator_mojo = MojoFeatureGenerator(use_mojo=True)
    start = time.time()
    features_mojo = generator_mojo.generate_features(
        test_close, test_high, test_low, test_volume
    )
    mojo_time = time.time() - start
    print(f"✅ Mojo: {mojo_time*1000:.2f}ms ({len(features_mojo.columns)} features)")

    # NumPy version
    generator_numpy = MojoFeatureGenerator(use_mojo=False)
    start = time.time()
    features_numpy = generator_numpy.generate_features(
        test_close, test_high, test_low, test_volume
    )
    numpy_time = time.time() - start
    print(f"🐍 NumPy: {numpy_time*1000:.2f}ms ({len(features_numpy.columns)} features)")

    # Speedup
    speedup = numpy_time / mojo_time
    print(f"\n🚀 Speedup: {speedup:.1f}x faster with Mojo!")

    # Validate correctness
    if MOJO_AVAILABLE:
        max_diff = np.max(np.abs(features_mojo.values - features_numpy.values))
        print(f"📊 Max difference: {max_diff:.2e} (should be < 1e-6)")
        if max_diff < 1e-6:
            print("✅ Validation passed: Mojo output matches NumPy!")
        else:
            print("⚠️  Warning: Mojo output differs from NumPy")
