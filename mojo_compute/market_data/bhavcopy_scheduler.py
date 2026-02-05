"""
Bhavcopy Download Scheduler
Runs daily after market close (6:00 PM IST) to download NSE EOD data
"""

import asyncio
import logging
import subprocess
from datetime import datetime, time
import pytz
import os

logger = logging.getLogger(__name__)


class BhavcopyScheduler:
    """Scheduler for daily Bhavcopy downloads."""

    def __init__(self):
        self.running = False
        self.download_time = time(18, 0)  # 6:00 PM IST
        self.check_interval = 300  # Check every 5 minutes

    def should_download_now(self) -> bool:
        """
        Check if it's time to download Bhavcopy.

        Downloads at 6:00 PM IST on weekdays (market closes at 3:30 PM).
        """
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)

        # Skip weekends
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check if it's past download time
        current_time = now.time()

        # Download window: 6:00 PM to 6:30 PM IST
        download_start = time(18, 0)
        download_end = time(18, 30)

        return download_start <= current_time <= download_end

    def run_mojo_downloader(self) -> bool:
        """
        Execute the Mojo Bhavcopy downloader.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🚀 Starting Mojo Bhavcopy downloader...")

            # Path to Mojo executable
            mojo_file = os.path.join(
                os.path.dirname(__file__),
                'bhavcopy_downloader.mojo'
            )

            # Run Mojo program
            result = subprocess.run(
                ['mojo', 'run', mojo_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info("✅ Bhavcopy download successful")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ Bhavcopy download failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Bhavcopy download timed out (>5 minutes)")
            return False
        except Exception as e:
            logger.error(f"❌ Error running Mojo downloader: {e}")
            return False

    async def run(self):
        """Main scheduler loop."""
        logger.info("📅 Bhavcopy scheduler started")
        logger.info(f"⏰ Will download daily at {self.download_time.strftime('%I:%M %p')} IST")

        self.running = True
        last_download_date = None

        while self.running:
            try:
                ist = pytz.timezone('Asia/Kolkata')
                now = datetime.now(ist)
                today_date = now.date()

                # Check if it's time to download and we haven't downloaded today yet
                if self.should_download_now() and last_download_date != today_date:
                    logger.info(f"📊 Downloading Bhavcopy for {today_date}")

                    success = self.run_mojo_downloader()

                    if success:
                        last_download_date = today_date
                        logger.info("✅ Daily Bhavcopy download completed")
                    else:
                        logger.warning("⚠️ Bhavcopy download failed, will retry next check")

            except Exception as e:
                logger.error(f"Error in Bhavcopy scheduler: {e}", exc_info=True)

            # Wait before next check
            await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping Bhavcopy scheduler")
        self.running = False


# Standalone execution for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    scheduler = BhavcopyScheduler()

    # For testing: run immediately
    logger.info("Testing Bhavcopy downloader...")
    scheduler.run_mojo_downloader()
