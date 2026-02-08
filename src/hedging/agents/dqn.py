"""Deep Q-Network agent for discrete hedging actions."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional, Tuple, List
from dataclasses import dataclass
import random

from ..policy import QNetwork


@dataclass
class Transition:
    """Single transition for experience replay."""
    
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int):
        """Initialize buffer with given capacity."""
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random batch from buffer."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for discrete hedging decisions."""
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: List[int] = None,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        double_dqn: bool = True,
        device: str = "auto",
    ):
        """Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            num_actions: Number of discrete actions
            hidden_dims: Hidden layer sizes
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            double_dqn: Use double DQN algorithm
            device: Computation device
        """
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        
        self.q_network = QNetwork(state_dim, num_actions, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, num_actions, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.training_steps = 0
        self.losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether to use exploration
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        transitions = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(self.device)
        dones = torch.FloatTensor([t.done for t in transitions]).to(self.device)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_network(next_states).max(dim=1)[0]
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def train(
        self,
        env,
        episodes: int,
        max_steps: int = 1000,
        verbose: bool = True,
    ) -> dict:
        """Train agent on environment.
        
        Args:
            env: Gymnasium environment
            episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Print training progress
            
        Returns:
            Training statistics
        """
        episode_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                self.store_transition(state, action, reward, next_state, done)
                loss = self.train_step()
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{episodes} | Avg Reward: {avg_reward:.4f} | Epsilon: {self.epsilon:.4f}")
        
        return {
            "episode_rewards": episode_rewards,
            "losses": self.losses,
            "final_epsilon": self.epsilon,
        }
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
