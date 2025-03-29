import threading
import signal
import time
import logging
from src.coinbase_rustRNN import train_loop, data_fetching_thread

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    """Handle graceful shutdown on SIGINT or SIGTERM."""
    logger.info("Shutdown signal received. Stopping processes...")
    shutdown_event.set()

def main():
    """Main function to start training and data fetching in parallel."""
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start data fetching thread
    data_thread = threading.Thread(target=data_fetching_thread, daemon=True)
    data_thread.start()
    logger.info("Data fetching thread started.")

    # Start training in the main thread
    try:
        train_loop()
    except Exception as e:
        logger.error(f"Error in train_loop: {e}")

    # Wait for shutdown event
    while not shutdown_event.is_set():
        time.sleep(1)

    logger.info("Shutting down. Waiting for data thread to finish.")
    data_thread.join()
    logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()