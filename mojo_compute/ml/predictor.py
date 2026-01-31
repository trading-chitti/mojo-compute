"""
Inference engine for price predictions.

Handles model loading, feature preparation, and prediction serving.
"""

import pandas as pd
from typing import Dict, Optional
from pathlib import Path
from .trainer import ModelTrainer
from .features import FeatureEngineer


class Predictor:
    """
    Prediction engine for price direction forecasting.

    Features:
    - Model loading and caching
    - Feature preparation
    - Batch and single predictions
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model file (.joblib)
        """
        self.model_path = model_path
        self.trainer: Optional[ModelTrainer] = None
        self.feature_engineer = FeatureEngineer()

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load trained model from disk.

        Args:
            model_path: Path to model file
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.trainer = ModelTrainer.load(model_path)
        self.model_path = model_path

    def predict(
        self,
        df: pd.DataFrame,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Predict price direction for OHLCV data.

        Args:
            df: DataFrame with OHLCV columns
            return_probabilities: Include prediction probabilities

        Returns:
            Dict with predictions and metadata
        """
        if self.trainer is None:
            raise ValueError("No model loaded. Call load_model() first.")

        # Generate features
        df_features = self.feature_engineer.generate_features(df)

        # Drop NaN rows
        df_features = df_features.dropna()

        if len(df_features) == 0:
            return {
                'error': 'No valid data after feature engineering',
                'predictions': []
            }

        # Get feature columns
        feature_cols = self.feature_engineer.get_feature_columns(df_features)
        X = df_features[feature_cols]

        # Predict
        predictions, probabilities = self.trainer.predict(X)

        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'index': int(i),
                'predicted_direction': 'up' if pred == 1 else 'down',
                'predicted_class': int(pred),
            }

            if return_probabilities:
                result['confidence'] = float(max(prob))
                result['prob_down'] = float(prob[0])
                result['prob_up'] = float(prob[1])

            results.append(result)

        return {
            'predictions': results,
            'model_algorithm': self.trainer.algorithm,
            'num_features': len(feature_cols),
        }

    def predict_latest(
        self,
        df: pd.DataFrame,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Predict direction for the most recent bar.

        Args:
            df: DataFrame with OHLCV columns
            return_probabilities: Include prediction probabilities

        Returns:
            Dict with single prediction
        """
        results = self.predict(df, return_probabilities)

        if 'error' in results:
            return results

        # Return only the last prediction
        if results['predictions']:
            return {
                'prediction': results['predictions'][-1],
                'model_algorithm': results['model_algorithm'],
            }
        else:
            return {'error': 'No predictions available'}

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dict with model metadata
        """
        if self.trainer is None:
            return {'error': 'No model loaded'}

        metrics = self.trainer.get_metrics()
        return {
            'model_path': self.model_path,
            'algorithm': self.trainer.algorithm,
            'feature_count': len(self.trainer.feature_columns),
            'training_metrics': metrics,
        }
