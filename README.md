#dqnTrader - Coinbase USDC Trading Bot

## Overview
dqn-trader is a deep reinforcement learning trading bot for USDC crypto pairs on Coinbase. It leverages a Double DQN with batch normalization and Prioritized Experience Replay (PER) for short-interval grid profit targeting.

---

## Directory Structure
```
dqn-trader/
├── src/
│   └── coinbase_rustRNN.py
├── main.py
├── README.md
```

---

## Features
- **Double DQN** with Batch Normalization
- **Prioritized Experience Replay** for efficient learning
- **Robust Logging** for both training and data fetching processes
- **Signal Handling** for graceful shutdown
- **Model Checkpointing** and periodic evaluation
- **WebSocket Integration** for real-time Coinbase data

---

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/LoQiseaking69/dqn-trader.git
cd dqn-trader
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python main.py
```

---

## Components

### `src/coinbase_rustRNN.py`
- Implements the DQN model with Double DQN and PER
- Fetches data from Coinbase API with retry logic
- Includes replay buffer and model checkpointing

### `main.py`
- Starts data fetching and training processes
- Implements graceful shutdown with signal handling

---

## Logs and Checkpoints
- Model checkpoints are saved every 100 episodes.
- Training and fetching logs are saved to the console for real-time monitoring.


## License
This project is licensed under the MIT License.
