import threading
import signal
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import requests
import json
from collections import deque
from datetime import datetime

# Constants
GAMMA = 0.99
LR = 1e-3
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 32
BUFFER_CAPACITY = 10000
TARGET_UPDATE = 10
CACHE_FILE = "coinbase_cache.json"
API_URL = "https://api.coinbase.com/v2/prices/USDC-USD/spot"
SAVE_MODEL_FREQ = 50
SAVE_BEST_MODEL = True

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coinbase_rustRNN")

# DQN Model Definition
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # 2 actions (buy, sell)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

# Replay Buffer with Prioritized Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, td_error):
        self.buffer.append((state, action, reward, next_state, td_error))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        probs = np.array([abs(exp[4]) for exp in self.buffer]) + 1e-6
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[idx] for idx in indices]
    
    def __len__(self):
        return len(self.buffer)

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer(BUFFER_CAPACITY)
step_count = 0
best_model = None
best_loss = float('inf')

# Fetch Data from Coinbase API
def fetch_data():
    try:
        response = requests.get(API_URL, timeout=5)
        response.raise_for_status()
        price = float(response.json()['data']['amount'])
        return price
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return None

# Save Cache to file
def save_cache():
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump([exp for exp in buffer.buffer], f)
        logger.info(f"Cache saved to {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

# Save Best Model
def save_best_model():
    global best_model, best_loss
    current_loss = calculate_loss(policy_net)
    if current_loss < best_loss:
        best_loss = current_loss
        best_model = policy_net.state_dict()
        try:
            torch.save(best_model, "best_dqn_model.pth")
            logger.info("New best model saved!")
        except Exception as e:
            logger.error(f"Error saving best model: {e}")

# Loss Calculation for Evaluation (Placeholder)
def calculate_loss(model):
    return np.random.rand()  # Replace with actual loss calculation logic

# Signal Handling for Graceful Shutdown
shutdown_event = threading.Event()

def handle_exit(sig, frame):
    logger.info("Shutdown signal received. Stopping processes...")
    shutdown_event.set()
    torch.save(policy_net.state_dict(), "dqn_model.pth")
    save_cache()
    if SAVE_BEST_MODEL and best_model is not None:
        torch.save(best_model, "best_dqn_model.pth")

# Register the signal handlers for SIGINT and SIGTERM
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# Data Fetching Thread
def data_fetching_thread():
    while not shutdown_event.is_set():
        price = fetch_data()
        if price is not None:
            logger.info(f"Fetched price: {price}")
        time.sleep(5)  # Fetch data every 5 seconds

# Training Loop
def train_loop():
    global step_count, EPSILON
    for episode in range(1, 1001):
        state = np.random.rand(4)  # Initial random state
        done = False
        
        while not done:
            action = select_action(state)
            
            next_price = fetch_data()
            if next_price is None:
                continue  # Skip if fetch failed

            next_state = np.array([next_price, state[1], state[2], state[3]])
            reward = 0.1 if action == 0 else -0.1  # Placeholder reward
            
            # Compute TD Error and store experience
            td_error = compute_td_error(state, action, reward, next_state)
            buffer.push(state, action, reward, next_state, td_error)

            # Sample and Update Model
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                update_model(batch)

            state = next_state
            step_count += 1
            update_epsilon(step_count)

            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if step_count % SAVE_MODEL_FREQ == 0:
                torch.save(policy_net.state_dict(), f"dqn_model_{step_count}.pth")
                logger.info(f"Model saved at step {step_count}")
                
            if SAVE_BEST_MODEL:
                save_best_model()

            # Random termination condition
            done = np.random.rand() < 0.01

    logger.info("Training complete.")

# Action Selection (Epsilon-Greedy)
def select_action(state):
    global EPSILON
    if np.random.rand() < EPSILON:
        return np.random.choice([0, 1])  # Random action (0 = buy, 1 = sell)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return policy_net(state_tensor).argmax().item()

# Compute TD Error
def compute_td_error(state, action, reward, next_state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    q_value = policy_net(state_tensor)[0, action]
    with torch.no_grad():
        q_next = target_net(next_state_tensor).max(1)[0]
    return reward + GAMMA * q_next.item() - q_value.item()

# Update Model with Sampled Experience
def update_model(batch):
    states, actions, rewards, next_states, _ = zip(*batch)
    states = torch.FloatTensor(np.array(states)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    q_next = target_net(next_states).max(1)[0].unsqueeze(1)
    target = rewards + GAMMA * q_next

    loss = F.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Update Epsilon (Decay over time)
def update_epsilon(step_count):
    global EPSILON
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

# Main function to start training and data fetching in parallel
def main():
    data_thread = threading.Thread(target=data_fetching_thread, daemon=True)
    data_thread.start()
    logger.info("Data fetching thread started.")

    try:
        train_loop()
    except Exception as e:
        logger.error(f"Error in train_loop: {e}")

    shutdown_event.set()
    data_thread.join()
    logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
