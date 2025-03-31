
# dqnTrader - Coinbase USDC Trading Bot

## Overview
dqn-trader is a deep reinforcement learning trading bot for USDC crypto pairs on Coinbase. It leverages a **Dueling Double DQN** with **Batch Normalization** and **Prioritized Experience Replay (PER)** for short-interval grid profit targeting.

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
- **Dueling Double DQN** with Batch Normalization
- **Prioritized Experience Replay** for efficient learning
- **Robust Logging** for both training and data fetching processes
- **Signal Handling** for graceful shutdown
- **Model Checkpointing** and periodic evaluation
- **WebSocket Integration** for real-time Coinbase data
- **Dynamic Price Fetching** from Coinbase API with retry logic
- **Cache Saving** for preserving training state

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
- Implements the **Dueling Double DQN** model with **Batch Normalization** and **Prioritized Experience Replay (PER)**
- Fetches data from the **Coinbase API** with retry logic and custom error handling
- Includes the **Replay Buffer** with priority sampling and **Model Checkpointing**

### `main.py`
- Starts the **data fetching** and **training processes** in parallel using **multithreading**
- Implements **graceful shutdown** with **signal handling** and saves the model state periodically
- Fetches real-time USDC prices from Coinbase and feeds them into the **DQN model**

---

## Logs and Checkpoints
- **Model checkpoints** are saved every **50 steps** during training and every **100 episodes**.
- **Training logs** are displayed on the console for real-time monitoring of the bot's actions.
- **Cache saving** allows for storing the current state of the replay buffer for continued training after a shutdown.

---

## License
This project is licensed under the MIT License.
