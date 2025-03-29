import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import requests
import json
import os
import signal
import time
import logging
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

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coinbase_rustRNN")

# Model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
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

# Initialize
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer(BUFFER_CAPACITY)
step_count = 0

# Fetch Data
def fetch_data():
    try:
        response = requests.get(API_URL, timeout=5)
        response.raise_for_status()
        price = float(response.json()['data']['amount'])
        return price
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return None

# Save Cache
def save_cache():
    with open(CACHE_FILE, 'w') as f:
        json.dump([exp for exp in buffer.buffer], f)

# Signal Handling
def handle_exit(sig, frame):
    logger.info("Signal received. Saving model...")
    torch.save(policy_net.state_dict(), "dqn_model.pth")
    save_cache()
    exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# Training Loop
for episode in range(1, 1001):
    state = np.random.rand(4)  # Replace with meaningful state
    done = False
    
    while not done:
        # Epsilon-Greedy
        if np.random.rand() < EPSILON:
            action = np.random.choice([0, 1])
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = policy_net(state_tensor).argmax().item()
        
        # Fetch next state and reward
        next_price = fetch_data()
        if next_price is None:
            continue
        next_state = np.random.rand(4)  # Placeholder for next state
        reward = np.random.rand()  # Placeholder for reward
        
        # Compute TD Error
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        q_value = policy_net(state_tensor)[action]
        with torch.no_grad():
            q_next = target_net(next_state_tensor).max()
        td_error = reward + GAMMA * q_next - q_value.item()
        
        # Store in Replay Buffer
        buffer.push(state, action, reward, next_state, td_error)
        
        # Sample and Update
        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, _ = zip(*batch)
            states = torch.FloatTensor(states)
            next_states = torch.FloatTensor(next_states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            
            q_values = policy_net(states).gather(1, actions)
            q_next = target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + GAMMA * q_next
            
            loss = F.mse_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        state = next_state
        step_count += 1
        
        # Decay Epsilon
        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
        
        # Update Target Network
        if step_count % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Save Model Periodically
        if step_count % SAVE_MODEL_FREQ == 0:
            torch.save(policy_net.state_dict(), f"dqn_model_{step_count}.pth")
            logger.info(f"Model saved at step {step_count}")
        
        # End condition placeholder
        done = np.random.rand() < 0.01

logger.info("Training complete.")