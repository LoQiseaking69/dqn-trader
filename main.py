import threading
import time
import logging
import signal
import sys
from src.coinbase_rustRNN import train_loop, shutdown_event, data_fetching_thread, logger

# Set up logging for main.py
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Signal Handling for Graceful Shutdown
def handle_exit(sig, frame):
    """Handles system signals for graceful shutdown."""
    logger.info("Signal received. Initiating shutdown...")
    shutdown_event.set()  # Trigger shutdown for threads and training
    time.sleep(1)  # Allow processes to acknowledge shutdown
    logger.info("Shutdown process complete.")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

def main():
    """Main entry point to start data fetching and training."""
    logger.info("Starting main process...")

    # Start data fetching thread
    data_thread = threading.Thread(target=data_fetching_thread, daemon=True)
    try:
        data_thread.start()
        logger.info("Data fetching thread started.")
    except Exception as e:
        logger.error(f"Failed to start data fetching thread: {e}")
        handle_exit(None, None)

    # Start training loop
    try:
        train_loop()
    except Exception as e:
        logger.error(f"Error in training loop: {e}")
        handle_exit(None, None)

    # Keep main alive until shutdown is triggered
    while not shutdown_event.is_set():
        time.sleep(1)

    logger.info("Main process is shutting down. Waiting for threads to finish.")
    data_thread.join()
    logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
