# src/__init__.py
from .coinbase_rustRNN import (
    DuelingDQN,  # Changed DQN to DuelingDQN
    PrioritizedReplayBuffer as ReplayBuffer,  # Aliased PrioritizedReplayBuffer as ReplayBuffer
    fetch_data,
    save_cache,
    save_best_model,
    calculate_loss,
    signal_handler,
    data_fetching_thread,
    train_loop,
    main,
)

__all__ = [
    "DuelingDQN",  # Changed DQN to DuelingDQN
    "ReplayBuffer",  # Aliased PrioritizedReplayBuffer as ReplayBuffer
    "fetch_data",
    "save_cache",
    "save_best_model",
    "calculate_loss",
    "signal_handler",
    "data_fetching_thread",
    "train_loop",
    "main",
]
