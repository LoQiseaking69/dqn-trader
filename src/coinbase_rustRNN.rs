use rand::seq::SliceRandom;
use std::collections::VecDeque;
use std::fs::{self, File};
use std::path::Path;
use serde::{Deserialize, Serialize};
use reqwest::{Client, Error};
use chrono::prelude::*;
use tch::{nn, Device, Tensor};
use std::sync::{Arc, RwLock};
use tokio::{sync::Mutex, task};
use rand::Rng;

// -- Replay Buffer with Prioritized Experience Replay --

#[derive(Clone, Debug)]
pub struct Experience {
    state: Vec<f32>,
    action: usize,
    reward: f32,
    next_state: Vec<f32>,
    td_error: f32, // Temporal Difference Error
}

pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        let len = self.buffer.len();
        if len < batch_size {
            return Vec::new(); // Prevent sampling from empty or insufficient buffer
        }

        // Prioritized Sampling based on TD error
        let total_error: f32 = self.buffer.iter().map(|exp| exp.td_error).sum();
        let mut rng = rand::thread_rng();
        self.buffer
            .iter()
            .choose_multiple_weighted(&mut rng, batch_size, |exp| exp.td_error / total_error)
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

// -- DQN Model --

#[derive(Debug)]
struct DQN {
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
    dropout: nn::Dropout, // Adding dropout for regularization
}

impl DQN {
    fn new(vs: &nn::Path) -> DQN {
        let fc1 = nn::linear(vs, 4, 128, Default::default()); // 4 input features
        let fc2 = nn::linear(vs, 128, 128, Default::default());
        let fc3 = nn::linear(vs, 128, 2, Default::default()); // 2 possible actions (buy/sell)
        let dropout = nn::Dropout::new(0.3); // Dropout layer to prevent overfitting
        DQN { fc1, fc2, fc3, dropout }
    }

    fn forward(&self, input: Tensor) -> Tensor {
        let x = input.view([-1, 4]);
        let x = x.apply(&self.fc1).relu();
        let x = x.apply(&self.fc2).relu();
        let x = self.dropout.forward(&x); // Apply dropout
        self.fc3.forward(&x)
    }
}

fn create_model() -> DQN {
    let vs = nn::VarStore::new(Device::cuda_if_available());
    DQN::new(&vs.root())
}

// -- Training Loop with Adaptive Epsilon Decay and Advanced Reward --

const GAMMA: f32 = 0.99;
const LEARNING_RATE: f32 = 1e-3;
const EPSILON: f32 = 0.1; // Exploration factor
const EPSILON_DECAY: f32 = 0.995; // Decay rate for epsilon
const MIN_EPSILON: f32 = 0.01;
const BATCH_SIZE: usize = 32;
const MAX_EPISODES: usize = 1000;
const TARGET_UPDATE_FREQUENCY: usize = 10;

fn train_dqn(
    model: &DQN,
    target_model: &DQN,
    replay_buffer: &ReplayBuffer,
    optimizer: &mut nn::Optimizer<nn::Adam>,
    batch_size: usize,
) {
    let batch = replay_buffer.sample(batch_size);

    if batch.is_empty() {
        return; // Avoid training with empty batch
    }

    let mut states = Vec::new();
    let mut actions = Vec::new();
    let mut rewards = Vec::new();
    let mut next_states = Vec::new();
    let mut td_errors = Vec::new();

    for exp in batch {
        states.push(exp.state);
        actions.push(exp.action);
        rewards.push(exp.reward);
        next_states.push(exp.next_state);
        td_errors.push(exp.td_error);
    }

    let states_tensor = Tensor::of_slice(&states.concat()).view([batch_size as i64, 4]);
    let next_states_tensor = Tensor::of_slice(&next_states.concat()).view([batch_size as i64, 4]);

    // Calculate Q-values for current states
    let q_values = model.forward(states_tensor);

    // Calculate Q-values for next states using the target network
    let next_q_values = target_model.forward(next_states_tensor);

    // Compute the target Q-values using the Bellman equation
    let mut target_q_values = q_values.copy();
    for i in 0..batch_size {
        let max_next_q_value = next_q_values.i(i as i64).max_dim(0, false);
        target_q_values.i_mut(i as i64).index_fill_(
            0,
            &Tensor::of_slice(&[actions[i]]),
            rewards[i] + GAMMA * max_next_q_value,
        );
    }

    // Compute loss (Mean Squared Error)
    let loss = target_q_values.mse_loss(&q_values, tch::Reduction::Mean);

    // Backpropagate and optimize
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    // Update TD Errors in the Replay Buffer
    for i in 0..batch_size {
        replay_buffer.buffer[i].td_error = (target_q_values.i(i as i64).double_value(&[]) - q_values.i(i as i64).double_value(&[])) as f32;
    }

    println!("Loss: {}", loss.double_value(&[]));
}

// -- Data Caching --

#[derive(Deserialize, Debug, Serialize)]
struct CoinbaseData {
    time: String,
    price: String,
}

#[tokio::main]
async fn fetch_coinbase_data() -> Result<Vec<CoinbaseData>, Error> {
    let url = "https://api.coinbase.com/v2/prices/spot?currency=USD";
    let client = Client::new();
    let response = client.get(url).send().await?;
    let data: Vec<CoinbaseData> = response.json().await?;
    Ok(data)
}

async fn cache_data(data: Vec<CoinbaseData>, cache_file: &str) {
    let data_string = serde_json::to_string(&data).unwrap();
    if let Err(e) = fs::write(cache_file, data_string) {
        eprintln!("Error writing to cache file: {}", e);
    }
}

fn load_cached_data(cache_file: &str) -> Vec<CoinbaseData> {
    let path = Path::new(cache_file);
    if path.exists() {
        let data_string = fs::read_to_string(path).unwrap_or_else(|_| String::new());
        serde_json::from_str(&data_string).unwrap_or_else(|_| Vec::new())
    } else {
        Vec::new()
    }
}

// -- Helper Functions --

fn is_data_ready_for_training(data: &Vec<CoinbaseData>, threshold: usize) -> bool {
    data.len() >= threshold
}

fn select_action(model: &DQN, state: Vec<f32>, epsilon: f32) -> usize {
    let state_tensor = Tensor::of_slice(&state).view([1, 4]);
    let q_values = model.forward(state_tensor);
    let action = if rand::thread_rng().gen_bool(f64::from(epsilon)) {
        rand::thread_rng().gen_range(0..2) // Random action (buy/sell)
    } else {
        q_values.argmax(1, false).int64_value(&[0]) as usize // Best action (highest Q-value)
    };
    action
}

fn preprocess_data(data: Vec<CoinbaseData>) -> Vec<f32> {
    // Simple preprocessing to convert Coinbase data into state representation
    let prices: Vec<f32> = data.iter().map(|d| d.price.parse().unwrap_or(0.0)).collect();
    vec![prices.iter().sum(), prices.iter().fold(f32::MIN, |a, &b| a.max(b))]
}

// -- Target Model Update --

fn update_target_model(model: &DQN, target_model: &mut DQN, tau: f32) {
    for (target_param, model_param) in target_model.fc1.parameters_mut().iter_mut().zip(model.fc1.parameters().iter()) {
        *target_param = target_param * (1.0 - tau) + model_param * tau;
    }
    // Repeat for fc2 and fc3
}

// -- Main Function --

#[tokio::main]
async fn main() {
    let cache_file = "coinbase_data_cache.json";
    let replay_buffer = Arc::new(Mutex::new(ReplayBuffer::new(10000)));
    let model = create_model();
    let target_model = create_model();
    let mut optimizer = nn::Adam::default().build(&model.fc1, LEARNING_RATE).unwrap();

    // Fetch and cache data asynchronously
    match fetch_coinbase_data().await {
        Ok(processed_data) => {
            // Cache the data for later use asynchronously
            task::spawn(cache_data(processed_data.clone(), cache_file));

            // Load the cached data
            let cached_data = load_cached_data(cache_file);

            // Trigger the next phase when enough data is gathered
            let data_threshold = 100; // Set the threshold for how much data is needed
            if is_data_ready_for_training(&cached_data, data_threshold) {
                println!("Enough data collected, starting training...");

                let mut epsilon = EPSILON;
                for epoch in 0..MAX_EPISODES {
                    let state = preprocess_data(cached_data.clone()); // Simulate state
                    let action = select_action(&model, state.clone(), epsilon); // Choose action based on epsilon-greedy

                    let reward = if action == 0 { 1.0 } else { -1.0 }; // Reward based on action (buy or sell)

                    let next_state = preprocess_data(cached_data.clone());

                    let experience = Experience {
                        state,
                        action,
                        reward,
                        next_state,
                        td_error: 0.0, // Initialize TD error
                    };

                    let mut replay_buffer = replay_buffer.lock().await;
                    replay_buffer.push(experience);

                    if replay_buffer.len() > BATCH_SIZE {
                        train_dqn(&model, &target_model, &replay_buffer, &mut optimizer, BATCH_SIZE);
                    }

                    if epoch % TARGET_UPDATE_FREQUENCY == 0 {
                        update_target_model(&model, &mut target_model, 0.005); // Soft update
                    }

                    epsilon = (epsilon * EPSILON_DECAY).max(MIN_EPSILON);
                }
            } else {
                println!("Not enough data yet. Waiting...");
            }
        }
        Err(e) => {
            eprintln!("Failed to fetch data: {}", e);
        }
    }
}
