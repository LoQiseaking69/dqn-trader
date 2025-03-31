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
import random

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
logger = logging.getLogger("coinbase_dqn")

# Dueling Double DQN Model Definition
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # State value stream
        self.value_fc = nn.Linear(128, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(128, 2)  # Two actions: buy, sell
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        # Q-value = Value + Advantage - mean(Advantage)
        q_values = value + advantage - advantage.mean()
        return q_values

# Replay Buffer with Prioritized Experience Replay
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.buffer = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, td_error):
        self.buffer.append((state, action, reward, next_state, td_error))
        self.priorities.append(max(td_error, 1e-6))  # Avoid zero priority
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        
        # Calculate sampling probabilities based on priorities
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        
        # Update importance sampling weight
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability
        
        return batch, weights, indices
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = max(td_error, 1e-6)

    def __len__(self):
        return len(self.buffer)

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DuelingDQN().to(device)
target_net = DuelingDQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = PrioritizedReplayBuffer(BUFFER_CAPACITY)
step_count = 0
best_model = None
best_loss = float('inf')

# Fetch Data from Coinbase API
def fetch_data():
    try:
        response = requests.get(API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data.get('data', {}).get('amount', 0))
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
    current_loss = calculate_loss(torch.zeros(1, 4).to(device), 0, 0, torch.zeros(1, 4).to(device))
    if current_loss < best_loss:
        best_loss = current_loss
        best_model = policy_net.state_dict()
        try:
            torch.save(best_model, "best_dqn_model.pth")
            logger.info("New best model saved!")
        except Exception as e:
            logger.error(f"Error saving best model: {e}")

# Calculate Loss Function
def calculate_loss(states, actions, rewards, next_states):
    states = torch.FloatTensor(np.array(states)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)

    q_values = policy_net(states).gather(1, actions)
    q_next = target_net(next_states).max(1)[0].unsqueeze(1).detach()
    target = rewards + (GAMMA * q_next)

    loss = F.mse_loss(q_values, target)
    return loss

# Signal Handling for Graceful Shutdown
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    logger.info("Shutdown signal received. Stopping processes...")
    shutdown_event.set()
    torch.save(policy_net.state_dict(), "dqn_model.pth")
    save_cache()
    if SAVE_BEST_MODEL and best_model is not None:
        torch.save(best_model, "best_dqn_model.pth")
    exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Data Fetching Thread
def data_fetching_thread():
    while not shutdown_event.is_set():
        price = fetch_data()
        if price is not None:
            logger.info(f"Fetched price: {price}")
        time.sleep(5)

# Training Loop
def train_loop():
    global step_count, EPSILON
    for episode in range(1, 1001):
        state = np.random.rand(4)
        done = False
        
        while not done:
            action = select_action(state)
            
            next_price = fetch_data()
            if next_price is None:
                continue

            next_state = np.array([next_price, state[1], state[2], state[3]])
            reward = 0.1 if action == 0 else -0.1
            
            td_error = compute_td_error(state, action, reward, next_state)
            buffer.push(state, action, reward, next_state, td_error)

            if len(buffer) >= BATCH_SIZE:
                batch, weights, indices = buffer.sample(BATCH_SIZE)
                update_model(batch, weights)

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

            done = np.random.rand() < 0.01

    logger.info("Training complete.")

# Action Selection (Epsilon-Greedy)
def select_action(state):
    global EPSILON
    if np.random.rand() < EPSILON:
        return np.random.choice([0, 1])
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
        q_next = target_net(next_state_tensor).max(1)[0].detach()
    return reward + GAMMA * q_next.item() - q_value.item()

# Update Model with Sampled Experience
def update_model(batch, weights):
    states, actions, rewards, next_states, _ = zip(*batch)
    loss = calculate_loss(states, actions, rewards, next_states)
    
    # Weighted loss
    weights_tensor = torch.FloatTensor(weights).to(device)
    loss = (loss * weights_tensor).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Update Epsilon
def update_epsilon(step_count):
    global EPSILON
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

# Main Function to Start Training
def main():
    fetch_thread = threading.Thread(target=data_fetching_thread)
    fetch_thread.start()
    train_loop()

if __name__ == "__main__":
    main()
