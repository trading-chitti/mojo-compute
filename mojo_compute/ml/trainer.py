"""
Model training pipeline for price prediction.

Supports multiple algorithms:
- Random Forest
- XGBoost
- LightGBM
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime


class ModelTrainer:
    """
    Train ML models for price direction prediction.

    Supports:
    - Time series cross-validation
    - Multiple algorithms
    - Hyperparameter tuning
    - Model evaluation
    """

    def __init__(self, algorithm: str = 'random_forest'):
        """
        Initialize trainer.

        Args:
            algorithm: Model algorithm ('random_forest', 'xgboost', 'lightgbm')
        """
        self.algorithm = algorithm
        self.model = None
        self.feature_columns = None
        self.training_metrics = {}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        cv_splits: int = 5
    ) -> Dict:
        """
        Train model with time series cross-validation.

        Args:
            X: Feature DataFrame
            y: Target series
            test_size: Test set size (default: 0.2)
            cv_splits: Number of CV splits (default: 5)

        Returns:
            Dict with training metrics
        """
        # Store feature columns
        self.feature_columns = list(X.columns)

        # Time series split (preserve order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Initialize model based on algorithm
        self.model = self._get_model()

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=tscv, scoring='accuracy'
        )

        # Train on full training set
        self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        # Store metrics
        self.training_metrics = {
            'algorithm': self.algorithm,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std_accuracy': float(cv_scores.std()),
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'trained_at': datetime.now().isoformat(),
        }

        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            self.training_metrics['feature_importance'] = feature_importance

        return self.training_metrics

    def _get_model(self):
        """Get model instance based on algorithm."""
        if self.algorithm == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif self.algorithm == 'xgboost':
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            except ImportError:
                raise ValueError("xgboost not installed. Use 'random_forest' or install xgboost.")
        elif self.algorithm == 'lightgbm':
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1
                )
            except ImportError:
                raise ValueError("lightgbm not installed. Use 'random_forest' or install lightgbm.")
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def save(self, filepath: str):
        """
        Save trained model to disk.

        Args:
            filepath: Path to save model (.joblib)
        """
        if self.model is None:
            raise ValueError("No model trained yet")

        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'algorithm': self.algorithm,
            'training_metrics': self.training_metrics,
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'ModelTrainer':
        """
        Load trained model from disk.

        Args:
            filepath: Path to model file (.joblib)

        Returns:
            ModelTrainer instance
        """
        data = joblib.load(filepath)

        trainer = cls(algorithm=data['algorithm'])
        trainer.model = data['model']
        trainer.feature_columns = data['feature_columns']
        trainer.training_metrics = data['training_metrics']

        return trainer

    def predict(self, X: pd.DataFrame) -> tuple:
        """
        Predict price direction.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("No model trained yet")

        # Ensure correct feature order
        X = X[self.feature_columns]

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return predictions, probabilities

    def get_metrics(self) -> Dict:
        """Get training metrics."""
        return self.training_metrics
