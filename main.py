import threading
import time
import logging
import os
import sys

# Add the 'src' folder to the Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Now import coinbase_rustRNN from src
import coinbase_rustRNN

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start Data Fetching
def data_fetching_thread():
    """Run this function in a separate thread to fetch data periodically."""
    while True:
        try:
            # Call the function to fetch and cache data from coinbase_rustRNN.py
            coinbase_rustRNN.fetch_and_cache_data()  # Assuming this function exists in coinbase_rustRNN.py
        except Exception as e:
            logger.error(f"Error in data fetching thread: {e}")
        time.sleep(60)  # Fetch data every minute

# Main execution of the training loop (coinbase_rustRNN.py)
def main_training_loop():
    """Run the training loop for coinbase_rustRNN."""
    try:
        # Assuming there's a function to start the training loop in coinbase_rustRNN.py
        coinbase_rustRNN.train_loop()  # Adjust this to the function name in coinbase_rustRNN.py
    except Exception as e:
        logger.error(f"Error in main training loop: {e}")

# Main Execution
if __name__ == "__main__":
    try:
        # Start the data fetching thread
        data_thread = threading.Thread(target=data_fetching_thread)
        data_thread.daemon = True
        data_thread.start()

        # Run the training loop
        main_training_loop()

    except KeyboardInterrupt:
        logger.info("Training interrupted. Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")