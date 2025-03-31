
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

<span style="color:red;">O</span><span style="color:orange;">b</span><span style="color:yellow;">v</span><span style="color:green;">i</span><span style="color:blue;">o</span><span style="color:indigo;">u</span><span style="color:violet;">s</span>, <span style="color:red;">c</span><span style="color:orange;">h</span><span style="color:yellow;">a</span><span style="color:green;">n</span><span style="color:blue;">g</span><span style="color:indigo;">i</span><span style="color:violet;">n</span>g <span style="color:red;">t</span><span style="color:orange;">h</span><span style="color:yellow;">e</span> <span style="color:green;">p</span><span style="color:blue;">a</span><span style="color:indigo;">i</span><span style="color:violet;">r</span><span style="color:red;">s</span> <span style="color:orange;">t</span><span style="color:yellow;">r</span><span style="color:green;">a</span><span style="color:blue;">c</span><span style="color:indigo;">k</span><span style="color:violet;">e</span>d; <span style="color:red;">h</span><span style="color:orange;">a</span><span style="color:yellow;">d</span> <span style="color:green;">a</span> <span style="color:blue;">b</span><span style="color:indigo;">r</span><span style="color:violet;">o</span><span style="color:red;">a</span><span style="color:orange;">d</span><span style="color:yellow;">e</span><span style="color:green;">r</span> <span style="color:blue;">t</span><span style="color:indigo;">h</span><span style="color:violet;">o</span><span style="color:red;">u</span><span style="color:orange;">g</span><span style="color:yellow;">h</span> <span style="color:green;">w</span><span style="color:blue;">h</span><span style="color:indigo;">e</span><span style="color:violet;">n</span> <span style="color:red;">c</span><span style="color:orange;">o</span><span style="color:yellow;">n</span><span style="color:green;">s</span><span style="color:blue;">i</span><span style="color:indigo;">d</span><span style="color:violet;">e</span><span style="color:red;">r</span>i<span style="color:orange;">n</span><span style="color:yellow;">g</span> <span style="color:green;">u</span><span style="color:blue;">s</span><span style="color:indigo;">d</span><span style="color:violet;">c</span>


---

## Logs and Checkpoints
- **Model checkpoints** are saved every **50 steps** during training and every **100 episodes**.
- **Training logs** are displayed on the console for real-time monitoring of the bot's actions.
- **Cache saving** allows for storing the current state of the replay buffer for continued training after a shutdown.

---

## License
This project is licensed under the MIT License.
