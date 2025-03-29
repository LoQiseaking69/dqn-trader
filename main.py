import threading
import time
import logging
import signal
import sys
from coinbase_rustRNN import train_loop, start_data_fetching_thread

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flag to control process shutdown
shutdown_flag = False

# Function to handle graceful shutdown
def handle_shutdown_signal(signal, frame):
    """Handle termination signals for graceful shutdown."""
    global shutdown_flag
    logger.info("Shutdown signal received. Cleaning up...")
    shutdown_flag = True

# Function to manage the entire process
def start_process():
    """Starts the data fetching and training processes."""
    try:
        # Start the data fetching thread
        logger.info("Starting data fetching thread.")
        fetch_thread = threading.Thread(target=start_data_fetching_thread, daemon=True)
        fetch_thread.start()

        # Start the training loop
        logger.info("Starting the training loop.")
        while not shutdown_flag:
            train_loop()
            time.sleep(1)  # Add a small delay to prevent CPU hogging

    except Exception as e:
        logger.error(f"An error occurred while starting the process: {e}")
        raise

# Main function to manage threading and process control
if __name__ == "__main__":
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    logger.info("Main process started.")
    start_process()  # Start the process

    logger.info("Process ended.")