import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import requests
import json
import os
import threading
import time
import logging
from collections import deque
from typing import List, Tuple
from datetime import datetime

# Constants and Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-3
EPSILON = 0.1
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 32
MAX_EPISODES = 1000
TARGET_UPDATE_FREQUENCY = 10
CACHE_FILE = "coinbase_data_cache.json"
API_URL = "https://api.coinbase.com/v2/prices/spot?currency=USD"
SAVE_MODEL_FREQUENCY = 100  # Save the model every 100 episodes

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Architecture (DQN with Double DQN)
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # Two possible actions (buy/sell)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.layer_norm(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# Experience and Replay Buffer (with Prioritized Experience Replay)
class Experience:
    def __init__(self, state, action, reward, next_state, td_error):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.td_error = td_error

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, experience: Experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.popleft()
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        total_error = sum([exp.td_error for exp in self.buffer])
        probabilities = [exp.td_error / total_error for exp in self.buffer]
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        return [self.buffer[idx] for idx in indices]

    def len(self):
        return len(self.buffer)

# Data Caching and Fetching from Coinbase API
def fetch_coinbase_data() -> List[dict]:
    """Fetch live data from Coinbase API."""
    try:
        response = requests.get(API_URL)
        response.raise_for_status()  # Raise error for bad response
        data = response.json()
        return [data['data']]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from Coinbase API: {e}")
        return []

def cache_data(data: List[dict]):
    """Cache data in a JSON file."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f)
        logger.info(f"Data cached successfully.")
    except Exception as e:
        logger.error(f"Error caching data: {e}")

def load_cached_data() -> List[dict]:
    """Load cached data from file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            return []
    return []

# Helper Functions
def select_action(model: nn.Module, state: np.ndarray, epsilon: float) -> int:
    """Select an action based on epsilon-greedy policy."""
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    q_values = model(state_tensor)
    if random.random() < epsilon:
        return random.choice([0, 1])  # Random action (buy/sell)
    return q_values.argmax().item()  # Best action (highest Q-value)

def preprocess_data(data: List[dict]) -> np.ndarray:
    """Preprocess data to a usable format."""
    if not data:
        return np.zeros(4)
    prices = [float(d['price']) for d in data]
    return np.array([sum(prices), max(prices), min(prices), np.std(prices)])

def is_data_ready_for_training(data: List[dict], threshold: int = 100) -> bool:
    """Check if enough data is available for training."""
    return len(data) >= threshold

# Model Training
def train_dqn(model: nn.Module, target_model: nn.Module, replay_buffer: ReplayBuffer, optimizer: optim.Optimizer, batch_size: int):
    """Train the DQN model using a batch of experiences."""
    batch = replay_buffer.sample(batch_size)
    if len(batch) == 0:
        return

    states = torch.tensor([exp.state for exp in batch], dtype=torch.float32)
    actions = torch.tensor([exp.action for exp in batch], dtype=torch.int64)
    rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
    next_states = torch.tensor([exp.next_state for exp in batch], dtype=torch.float32)

    # Current Q values
    q_values = model(states)
    next_q_values = target_model(next_states)

    # Double DQN: Use main model to select actions, target model to compute target Q values
    next_actions = q_values.argmax(dim=1)
    max_next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

    # Compute target Q values
    target_q_values = rewards + GAMMA * max_next_q_values

    # Loss computation (MSE Loss)
    selected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = F.mse_loss(selected_q_values, target_q_values)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update TD errors in replay buffer
    for i in range(batch_size):
        replay_buffer.buffer[i].td_error = abs(target_q_values[i] - selected_q_values[i]).item()

    logger.info(f"Loss: {loss.item()}")

def update_target_model(model: nn.Module, target_model: nn.Module, tau: float = 0.005):
    """Soft update the target model."""
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# Model Checkpointing
def save_model(model: nn.Module, epoch: int):
    """Save the model periodically."""
    if epoch % SAVE_MODEL_FREQUENCY == 0:
        model_filename = f"dqn_model_{epoch}.pth"
        torch.save(model.state_dict(), model_filename)
        logger.info(f"Model saved at {model_filename}.")

# Main Training Loop
def train_loop():
    """Main function to initialize model, replay buffer, and start training."""
    model = DQN()
    target_model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(capacity=10000)

    # Fetch and cache data
    data = fetch_coinbase_data()
    cache_data(data)

    # Load cached data
    cached_data = load_cached_data()

    epsilon = EPSILON
    for epoch in range(MAX_EPISODES):
        state = preprocess_data(cached_data)
        action = select_action(model, state, epsilon)
        reward = 1.0 if action == 0 else -1.0  # Example reward (buy/sell)

        # Simulate next state (for simplicity, using same state here)
        next_state = preprocess_data(cached_data)
        experience = Experience(state, action, reward, next_state, 0.0)
        replay_buffer.push(experience)

        if replay_buffer.len() > BATCH_SIZE:
            train_dqn(model, target_model, replay_buffer, optimizer, BATCH_SIZE)

        if epoch % TARGET_UPDATE_FREQUENCY == 0:
            update_target_model(model, target_model)

        # Epsilon decay for exploration
        epsilon = max(epsilon * EPSILON_DECAY, MIN_EPSILON)

        # Output progress
        if epoch % 100 == 0:
            logger.info(f"Episode {epoch}/{MAX_EPISODES}, Epsilon: {epsilon:.4f}")

        # Save model periodically
        save_model(model, epoch)

    logger.info("Training completed.")
    save_model(model, MAX_EPISODES)  # Final model save

# Threaded Data Fetching
def data_fetching_thread():
    """Run this function in a separate thread to fetch data periodically."""
    while True:
        data = fetch_coinbase_data()
        if data:
            cache_data(data)
        time.sleep(60)  # Fetch data every minute

# Main Execution
if __name__ == "__main__":
    try:
        # Start data fetching in a separate thread
        data_thread = threading.Thread(target=data_fetching_thread)
        data_thread.daemon = True
        data_thread.start()

        # Run the main training loop
        train_loop()

    except KeyboardInterrupt:
        logger.info("Training interrupted. Shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
