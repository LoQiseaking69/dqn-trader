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
            try:
                if shutdown_event.is_set():
                    logger.info("Shutdown signal detected. Terminating training loop.")
                    break

                # Start training process
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

# Modified version of the data-fetching thread with shutdown handling
def start_data_fetching_thread():
    """Fetch data and handle the shutdown signal properly."""
    while not shutdown_event.is_set():
        try:
            # Replace this with actual data fetching logic
            data = fetch_coinbase_data()  # Example: You should implement this function
            if data:
                cache_data(data)  # Example: Implement caching logic as needed
            time.sleep(60)  # Wait 60 seconds before fetching data again
        except Exception as e:
            logger.error(f"Error in data fetching thread: {e}")
            break

    logger.info("Data fetching thread terminated.")

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