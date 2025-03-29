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
    try:
        # Start the data fetching thread
        logger.info("Starting data fetching thread.")
        fetch_thread = threading.Thread(target=start_data_fetching_thread, daemon=True)
        fetch_thread.start()

        # Start the training loop in the main thread
        logger.info("Starting the training loop.")
        while not shutdown_event.is_set():
            # Ensure that the training loop can be stopped gracefully
            if shutdown_event.is_set():
                logger.info("Shutdown signal detected. Terminating training loop.")
                break
            try:
                train_loop()  # Training process
                time.sleep(1)  # Small delay to prevent CPU hogging
            except Exception as e:
                logger.error(f"An error occurred during training: {e}")
                break

        logger.info("Training loop terminated.")
    except Exception as e:
        logger.error(f"Unexpected error in start_process: {e}")
        raise
    finally:
        logger.info("Main process clean-up done.")

# Main Execution
if __name__ == "__main__":
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    logger.info("Main process started.")
    try:
        start_process()  # Start both data fetching and training loops
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Ensure process ends gracefully
        logger.info("Process ended gracefully.")