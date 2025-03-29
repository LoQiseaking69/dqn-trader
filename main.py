import threading
import time
import logging
import signal
import sys
from coinbase_rustRNN import train_loop, start_data_fetching_thread

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Event to control process shutdown
shutdown_event = threading.Event()

# Function to handle graceful shutdown
def handle_shutdown_signal(signum, frame):
    """Handle termination signals for graceful shutdown."""
    logger.info("Shutdown signal received. Cleaning up...")
    shutdown_event.set()

# Function to manage the entire process
def start_process():
    """Starts the data fetching and training processes."""
    # Start the data fetching thread
    logger.info("Starting data fetching thread.")
    fetch_thread = threading.Thread(target=start_data_fetching_thread, daemon=True)
    fetch_thread.start()

    # Start the training loop in the main thread
    logger.info("Starting the training loop.")
    try:
        while not shutdown_event.is_set():
            train_loop()
            time.sleep(1)  # Small delay to prevent CPU hogging
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise
    finally:
        logger.info("Training loop terminated.")

# Main Execution
if __name__ == "__main__":
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    logger.info("Main process started.")
    start_process()

    # Wait for data thread to finish if necessary (it is daemonized)
    logger.info("Process ended gracefully.")