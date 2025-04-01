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
    shutdown_event.set()  # Trigger shutdown

# Register signal handlers for SIGINT (Ctrl+C) and SIGTERM
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

def start_data_fetching_thread():
    """Start the data fetching thread with error handling."""
    try:
        data_thread = threading.Thread(target=data_fetching_thread, daemon=True)
        data_thread.start()
        logger.info("Data fetching thread started.")
        return data_thread
    except Exception as e:
        logger.error("Failed to start data fetching thread", exc_info=True)
        return None

def start_training_loop():
    """Start the training loop with error handling."""
    try:
        logger.info("Starting training loop...")
        train_loop()
    except Exception as e:
        logger.error("Error in training loop", exc_info=True)

def main():
    """Main entry point to start data fetching and training."""
    logger.info("Starting main process...")

    data_thread = start_data_fetching_thread()
    
    if data_thread:
        start_training_loop()

    # Keep main alive until shutdown is triggered
    while not shutdown_event.is_set():
        time.sleep(1)

    logger.info("Main process is shutting down.")
    
    if data_thread and data_thread.is_alive():
        logger.info("Waiting for data fetching thread to finish...")
        data_thread.join()

    logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()