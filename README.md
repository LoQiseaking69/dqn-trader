# DQN Trader - Deep Q-Network Based Trading Bot

This project implements a **Deep Q-Network (DQN)** for trading cryptocurrencies, specifically using data from **Coinbase's API**. The bot uses **reinforcement learning** to simulate trading strategies and aims to improve its performance through **Prioritized Experience Replay (PER)**.

## Features

- **Data Fetching**: Fetches real-time cryptocurrency price data from the **Coinbase API**.
- **Replay Buffer**: Uses a replay buffer with **Prioritized Experience Replay** to store experiences and prioritize high-impact experiences for training.
- **DQN Model**: Implements a deep Q-network for decision-making, with dropout layers for regularization to prevent overfitting.
- **Model Training**: The bot continuously trains on collected experiences, adapting its policy for trading actions (buy/sell).
- **Cache System**: Caches historical data locally for use in model training.
- **Asynchronous Execution**: Uses **async** operations for efficient data fetching and processing.

## Prerequisites

- **Rust**: Ensure you have Rust installed. You can install it from [here](https://www.rust-lang.org/tools/install).
- **Cargo**: Cargo is the Rust package manager and build system, which comes with Rust.
- **PyTorch**: This project uses **tch-rs** (PyTorch bindings for Rust) to define the neural network.

### Optional (For Logging):
- **Log and env_logger** for detailed runtime information during execution.

## Dependencies

- **Tokio**: For async tasks (fetching data and caching).
- **Reqwest**: For making HTTP requests to the Coinbase API.
- **Serde**: For serialization and deserialization of data.
- **Tch**: PyTorch bindings for Rust to create and train the DQN model.
- **Chrono**: For working with timestamps (future use).
- **Rand**: For random number generation (used in the epsilon-greedy strategy).
- **Serde_json**: For working with JSON data.
  
These dependencies are defined in the `Cargo.toml` file.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LoQiseaking69/dqn-trader.git
   cd dqn-trader
   ```

2. **Install Rust** (if you haven't already):
   - Visit https://www.rust-lang.org/tools/install to download and install the Rust toolchain.

3. **Build the Project**:
   ```bash
   cargo build
   ```

   This will download and compile the necessary dependencies.

4. **Run the Project**:
   To start the trading bot and begin training, run:
   ```bash
   cargo run
   ```

   The program will:
   - Fetch data from Coinbase.
   - Cache the data into a local file (`coinbase_data_cache.json`).
   - Train the DQN model using the cached data.
   - Output training logs and loss values during the process.

## Project Structure

```
dqn-trader/
â
âââ src/
â   âââ main.rs               # Main program logic
â   âââ replay_buffer.rs      # Defines the experience replay buffer and prioritized sampling
â   âââ dqn.rs                # Defines the DQN model (neural network)
â   âââ utils.rs              # Utility functions for data handling, including caching and preprocessing
â
âââ Cargo.toml                # Cargo configuration and dependencies
âââ README.md                 # Project documentation
```

## Usage

### Running the Training Loop

When you run the bot, it will perform the following:

1. **Fetch Data**: Real-time data from Coinbase will be fetched asynchronously.
2. **Cache Data**: The data will be saved locally in `coinbase_data_cache.json`.
3. **Model Training**: If enough data is available, the model will start training. The training loop will:
   - Sample a batch of experiences from the replay buffer.
   - Calculate Q-values for the current states and next states.
   - Compute the loss and backpropagate the error to update the model parameters.

### Training Parameters

- **Gamma (GAMMA)**: Discount factor for future rewards.
- **Learning Rate (LEARNING_RATE)**: Rate at which the model parameters are updated during training.
- **Epsilon (EPSILON)**: Exploration factor for epsilon-greedy strategy, decays over time.
- **Batch Size (BATCH_SIZE)**: Number of experiences sampled from the replay buffer for training.
- **Maximum Episodes (MAX_EPISODES)**: Maximum number of training episodes.
- **Target Update Frequency (TARGET_UPDATE_FREQUENCY)**: Frequency at which the target model is updated.

## Cache and Data Handling

- The bot caches the Coinbase data for later use. It stores data in the file `coinbase_data_cache.json`.
- If the cache exists, the bot will use it to train the model, simulating the environment by using past data.
- If thereâs not enough data, the bot will wait until sufficient data has been collected.

## Future Improvements

- **Integration with other cryptocurrency APIs**: Expand the bot to fetch data from other platforms (e.g., Binance, Kraken).
- **Hyperparameter Tuning**: Implement hyperparameter optimization for better model performance.
- **Visualization**: Add graphical visualization of training progress and data.
- **More Complex Reward Strategies**: Implement more sophisticated reward functions for the trading agent.

## Troubleshooting

- If the dependencies don't install properly, try running `cargo update` to fetch the latest versions of the dependencies.
- Ensure you have internet access when fetching data from Coinbase.
- If you encounter any issues, check the logs for errors or add `env_logger` to enable detailed logging.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
