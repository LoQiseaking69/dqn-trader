import threading
import time
import logging
import signal
import sys
from coinbase_rustRNN import train_loop, shutdown_event, data_fetching_thread, logger

# Set up logging for main.py
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Signal Handling for Graceful Shutdown
def handle_exit(sig, frame):
    logger.info("Signal received. Initiating shutdown...")
    shutdown_event.set()  # Set shutdown flag to stop threads and training loop
    sys.exit(0)

# Set up signal handling for graceful shutdown
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

def main():
    logger.info("Starting main process...")

    # Start data fetching thread
    data_thread = threading.Thread(target=data_fetching_thread, daemon=True)
    data_thread.start()
    logger.info("Data fetching thread started.")

    # Start training loop
    try:
        train_loop()
    except Exception as e:
        logger.error(f"Error in training loop: {e}")
    
    # Wait for shutdown signal to stop the program gracefully
    while not shutdown_event.is_set():
        time.sleep(1)

    logger.info("Main process is shutting down. Waiting for data thread to finish.")
    data_thread.join()
    logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()