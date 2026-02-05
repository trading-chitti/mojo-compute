"""
Daily Predictions Scheduler
Generates tomorrow's top gainers and losers predictions after market close
"""

import asyncio
import logging
import psycopg2
import pandas as pd
from datetime import datetime, time
import pytz
import os
import sys

# Import predictor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mojo_compute.ml.predictor import StockPredictor, predict_top_movers

logger = logging.getLogger(__name__)

PG_DSN = os.getenv('TRADING_CHITTI_PG_DSN', 'postgresql://hariprasath@localhost:6432/trading_chitti')


class PredictionsScheduler:
    """Scheduler for daily ML predictions."""

    def __init__(self):
        self.running = False
        self.prediction_time = time(16, 0)  # 4:00 PM IST (after market close)
        self.check_interval = 300  # Check every 5 minutes

    def should_generate_now(self) -> bool:
        """
        Check if it's time to generate predictions.

        Generates at 4:00 PM IST on weekdays (market closes at 3:30 PM).
        """
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)

        # Skip weekends
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check if it's past prediction time
        current_time = now.time()

        # Prediction window: 4:00 PM to 4:30 PM IST
        prediction_start = time(16, 0)
        prediction_end = time(16, 30)

        return prediction_start <= current_time <= prediction_end

    def fetch_historical_data(self) -> dict:
        """
        Fetch last 30 days of OHLCV data for prediction.

        Returns:
            Dict of symbol -> DataFrame with historical data
        """
        try:
            conn = psycopg2.connect(PG_DSN)
            cur = conn.cursor()

            # Get top 100 liquid stocks
            cur.execute("""
                SELECT DISTINCT symbol
                FROM md.stock_master
                WHERE is_active = true
                  AND exchange = 'NSE'
                ORDER BY symbol
                LIMIT 100
            """)

            symbols = [row[0] for row in cur.fetchall()]
            logger.info(f"Fetching data for {len(symbols)} symbols")

            stocks_data = {}

            for symbol in symbols:
                # Fetch last 30 days of EOD prices
                cur.execute("""
                    SELECT trade_date, open, high, low, close, volume
                    FROM md.eod_prices
                    WHERE exchange = 'NSE'
                      AND symbol = %s
                    ORDER BY trade_date DESC
                    LIMIT 30
                """, (symbol,))

                rows = cur.fetchall()

                if len(rows) >= 30:
                    # Convert to DataFrame
                    df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    df = df.sort_values('date')
                    df = df.set_index('date')

                    stocks_data[symbol] = df

            cur.close()
            conn.close()

            logger.info(f"Fetched historical data for {len(stocks_data)} stocks")
            return stocks_data

        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return {}

    def store_predictions(self, gainers: list, losers: list):
        """
        Store predictions in database.

        Args:
            gainers: List of top gainers predictions
            losers: List of top losers predictions
        """
        try:
            conn = psycopg2.connect(PG_DSN)
            cur = conn.cursor()

            prediction_date = datetime.now().date()

            # Clear existing predictions for today
            cur.execute("""
                DELETE FROM predictions.daily_predictions
                WHERE prediction_date = %s
            """, (prediction_date,))

            cur.execute("""
                DELETE FROM predictions.top_movers
                WHERE prediction_date = %s
            """, (prediction_date,))

            # Insert gainers
            for i, prediction in enumerate(gainers):
                # Insert into daily_predictions
                cur.execute("""
                    INSERT INTO predictions.daily_predictions (
                        symbol, prediction_date, predicted_price, current_price,
                        predicted_change_pct, stop_loss, target, confidence,
                        trend, reasoning, technical_summary
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    prediction['symbol'],
                    prediction_date,
                    prediction['predicted_price'],
                    prediction['current_price'],
                    prediction['predicted_change_pct'],
                    prediction['stop_loss'],
                    prediction['target'],
                    prediction['confidence'],
                    prediction['trend'],
                    prediction['reasoning'],
                    prediction['technical_summary']
                ))

                # Insert into top_movers
                cur.execute("""
                    INSERT INTO predictions.top_movers (
                        symbol, prediction_date, category, rank
                    )
                    VALUES (%s, %s, 'gainer', %s)
                """, (prediction['symbol'], prediction_date, i + 1))

            # Insert losers
            for i, prediction in enumerate(losers):
                # Insert into daily_predictions
                cur.execute("""
                    INSERT INTO predictions.daily_predictions (
                        symbol, prediction_date, predicted_price, current_price,
                        predicted_change_pct, stop_loss, target, confidence,
                        trend, reasoning, technical_summary
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, prediction_date) DO NOTHING
                """, (
                    prediction['symbol'],
                    prediction_date,
                    prediction['predicted_price'],
                    prediction['current_price'],
                    prediction['predicted_change_pct'],
                    prediction['stop_loss'],
                    prediction['target'],
                    prediction['confidence'],
                    prediction['trend'],
                    prediction['reasoning'],
                    prediction['technical_summary']
                ))

                # Insert into top_movers
                cur.execute("""
                    INSERT INTO predictions.top_movers (
                        symbol, prediction_date, category, rank
                    )
                    VALUES (%s, %s, 'loser', %s)
                """, (prediction['symbol'], prediction_date, i + 1))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"✅ Stored {len(gainers)} gainers and {len(losers)} losers predictions")

        except Exception as e:
            logger.error(f"Failed to store predictions: {e}")

    def generate_predictions(self):
        """Generate daily predictions."""
        logger.info("=" * 70)
        logger.info("🤖 GENERATING DAILY ML PREDICTIONS")
        logger.info("=" * 70)

        # Fetch historical data
        stocks_data = self.fetch_historical_data()

        if not stocks_data:
            logger.warning("No historical data available for predictions")
            return

        # Generate predictions
        logger.info("🧠 Running ML predictor...")
        top_gainers, top_losers = predict_top_movers(stocks_data, top_n=10)

        logger.info(f"✅ Generated {len(top_gainers)} gainers and {len(top_losers)} losers")

        # Store predictions
        self.store_predictions(top_gainers, top_losers)

        logger.info("=" * 70)
        logger.info("📊 DAILY PREDICTIONS COMPLETED")
        logger.info("=" * 70)

    async def run(self):
        """Main scheduler loop."""
        logger.info("🚀 Predictions scheduler started")
        logger.info(f"⏰ Will generate predictions daily at {self.prediction_time.strftime('%I:%M %p')} IST")

        self.running = True
        last_prediction_date = None

        while self.running:
            try:
                ist = pytz.timezone('Asia/Kolkata')
                now = datetime.now(ist)
                today_date = now.date()

                # Check if it's time to generate and we haven't generated today yet
                if self.should_generate_now() and last_prediction_date != today_date:
                    logger.info(f"📈 Generating predictions for tomorrow ({today_date})")

                    self.generate_predictions()
                    last_prediction_date = today_date

            except Exception as e:
                logger.error(f"Error in predictions scheduler: {e}", exc_info=True)

            # Wait before next check
            await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping predictions scheduler")
        self.running = False


# Standalone execution for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    scheduler = PredictionsScheduler()

    # For testing: run immediately
    logger.info("Testing predictions generator...")
    scheduler.generate_predictions()
