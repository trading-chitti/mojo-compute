#!/usr/bin/env python3
"""Train the ML Money Manager model from sandbox trade history.

Bootstrap from intraday.signal_history if sandbox data is insufficient.
Trains an XGBoost binary classifier: predict whether a signal will hit target.

Usage:
    python -m mojo_compute.sandbox.scripts.train_money_manager
"""

import logging
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import psycopg2
import psycopg2.extras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sandbox import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("train_money_manager")

PG_DSN = config.PG_DSN
MODEL_DIR = config.ML_MODEL_DIR

FEATURE_COLUMNS = [
    "confidence", "rsi", "macd", "bb_position", "volume_ratio",
    "atr_pct", "sentiment", "is_call", "risk_reward",
    "sl_pct", "target_pct",
]


def load_training_data():
    """Load labeled data from signal_history view + prediction_features."""
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Try sandbox trades first (labeled from actual paper trading)
    cur.execute("""
        SELECT COUNT(*) as cnt FROM sandbox.trades WHERE status = 'CLOSED'
    """)
    sandbox_count = cur.fetchone()["cnt"]

    if sandbox_count >= 200:
        logger.info("Using %d sandbox trades for training", sandbox_count)
        cur.execute("""
            SELECT
                t.signal_confidence as confidence,
                t.ml_confidence,
                t.signal_type,
                t.entry_price,
                t.target_price,
                t.stop_loss,
                t.exit_reason,
                t.realized_pnl,
                d.features
            FROM sandbox.trades t
            LEFT JOIN sandbox.ml_decisions d ON d.trade_id = t.trade_id
            WHERE t.status = 'CLOSED'
            ORDER BY t.created_at DESC
        """)
        rows = cur.fetchall()
    else:
        logger.info("Sandbox has %d trades (need 200+), bootstrapping from signal_history", sandbox_count)
        cur.execute("""
            SELECT
                s.confidence,
                s.signal_type,
                s.entry_price,
                s.target_price,
                s.stop_loss,
                s.status,
                s.prediction_features,
                s.recent_news_sentiment
            FROM intraday.signals s
            WHERE s.status IN ('HIT_TARGET', 'HIT_STOPLOSS', 'EXPIRED')
              AND s.prediction_features IS NOT NULL
            ORDER BY s.generated_at DESC
            LIMIT 5000
        """)
        rows = cur.fetchall()

    cur.close()
    conn.close()
    return rows, sandbox_count >= 200


def prepare_features(rows, from_sandbox: bool):
    """Convert raw rows to feature matrix X and label vector y."""
    X_list = []
    y_list = []

    for row in rows:
        try:
            if from_sandbox:
                features = row.get("features") or {}
                confidence = float(row.get("confidence", 0.5))
                is_call = 1 if row["signal_type"] == "CALL" else 0
                entry = float(row["entry_price"])
                target = float(row["target_price"])
                stop_loss = float(row["stop_loss"])

                # Label: 1 if trade was profitable
                label = 1 if float(row.get("realized_pnl", 0)) > 0 else 0

                feature_vec = [
                    confidence,
                    float(features.get("rsi", 50)),
                    float(features.get("macd", 0)),
                    float(features.get("bb_position", 0.5)),
                    float(features.get("volume_ratio", 1.0)),
                    float(features.get("atr_pct", 1.5)),
                    float(features.get("sentiment", 0)),
                    is_call,
                    float(features.get("risk_reward", 2.0)),
                    float(features.get("sl_pct", 1.5)),
                    float(features.get("target_pct", 3.0)),
                ]
            else:
                pred_features = row.get("prediction_features") or {}
                confidence = float(row.get("confidence", 0.5))
                is_call = 1 if row["signal_type"] == "CALL" else 0
                entry = float(row.get("entry_price", 0))
                target = float(row.get("target_price", 0))
                stop_loss = float(row.get("stop_loss", 0))

                # Label: 1 if signal hit target
                label = 1 if row["status"] == "HIT_TARGET" else 0

                # Risk:reward
                risk_reward = 2.0
                if entry > 0 and stop_loss > 0 and target > 0:
                    if is_call:
                        reward = target - entry
                        risk = entry - stop_loss
                    else:
                        reward = entry - target
                        risk = stop_loss - entry
                    if risk > 0:
                        risk_reward = reward / risk

                sl_pct = abs(entry - stop_loss) / entry * 100.0 if entry > 0 else 1.5
                target_pct = abs(target - entry) / entry * 100.0 if entry > 0 else 3.0

                feature_vec = [
                    confidence,
                    float(pred_features.get("rsi", 50)),
                    float(pred_features.get("macd", 0)),
                    float(pred_features.get("bb_position", 0.5)),
                    float(pred_features.get("volume_ratio", 1.0)),
                    float(pred_features.get("atr_pct", 1.5)),
                    float(row.get("recent_news_sentiment", 0) or 0),
                    is_call,
                    risk_reward,
                    sl_pct,
                    target_pct,
                ]

            X_list.append(feature_vec)
            y_list.append(label)
        except Exception as e:
            logger.warning("Skipping row: %s", e)
            continue

    return np.array(X_list), np.array(y_list)


def train_model(X, y):
    """Train XGBoost classifier."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info("Training set: %d samples, Validation: %d samples", len(X_train), len(X_val))
    logger.info("Label distribution: %.1f%% positive", y.mean() * 100)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    logger.info("Validation Accuracy: %.1f%%", acc * 100)
    logger.info("\n%s", classification_report(y_val, y_pred, target_names=["SKIP", "TRADE"]))

    # Feature importance
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("Feature Importance:")
    for feat, imp in sorted_imp:
        logger.info("  %-20s %.4f", feat, imp)

    return model, acc


def save_model(model, accuracy):
    """Save model to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"money_manager_{timestamp}.joblib"
    path = os.path.join(MODEL_DIR, filename)

    joblib.dump({
        "model": model,
        "features": FEATURE_COLUMNS,
        "accuracy": accuracy,
        "trained_at": datetime.now().isoformat(),
    }, path)

    logger.info("Model saved to %s", path)
    return path


def main():
    logger.info("=" * 60)
    logger.info("Training ML Money Manager")
    logger.info("=" * 60)

    rows, from_sandbox = load_training_data()
    logger.info("Loaded %d rows (source: %s)", len(rows), "sandbox" if from_sandbox else "signal_history")

    if len(rows) < 50:
        logger.error("Insufficient data: %d rows (need at least 50). Collect more trading data first.", len(rows))
        sys.exit(1)

    X, y = prepare_features(rows, from_sandbox)
    logger.info("Feature matrix: %s, Labels: %s", X.shape, y.shape)

    if len(np.unique(y)) < 2:
        logger.error("Only one class in labels. Need both winning and losing trades.")
        sys.exit(1)

    model, accuracy = train_model(X, y)
    path = save_model(model, accuracy)

    logger.info("=" * 60)
    logger.info("Training complete! Accuracy: %.1f%%", accuracy * 100)
    logger.info("Model: %s", path)
    logger.info("Restart sandbox-engine to load the new model.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
