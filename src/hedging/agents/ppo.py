"""Proximal Policy Optimization agent for continuous hedging."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..policy import HedgingPolicy, ValueNetwork


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        """Clear all stored data."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Add transition to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE returns and advantages.
        
        Args:
            last_value: Value estimate for final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        num_steps = len(self.rewards)
        advantages = np.zeros(num_steps, dtype=np.float32)
        returns = np.zeros(num_steps, dtype=np.float32)
        
        last_gae = 0.0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.values[t]
        
        return returns, advantages
    
    def get_tensors(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Convert buffer data to tensors."""
        return (
            torch.FloatTensor(np.array(self.states)).to(device),
            torch.FloatTensor(np.array(self.actions)).to(device),
            torch.FloatTensor(self.log_probs).to(device),
        )


class PPOAgent:
    """Proximal Policy Optimization agent for continuous hedging."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dims: List[int] = None,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        device: str = "auto",
    ):
        """Initialize PPO agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer sizes
            learning_rate: Optimizer learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Gradient clipping threshold
            update_epochs: Number of optimization epochs per rollout
            batch_size: Minibatch size for updates
            device: Computation device
        """
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        self.policy = HedgingPolicy(state_dim, action_dim, hidden_dims).to(self.device)
        self.value_network = ValueNetwork(state_dim, hidden_dims).to(self.device)
        
        self.optimizer = optim.Adam([
            {"params": self.policy.parameters()},
            {"params": self.value_network.parameters()},
        ], lr=learning_rate)
        
        self.rollout_buffer = RolloutBuffer()
        
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
    
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> Tuple[np.ndarray, float, float]:
        """Select action from policy.
        
        Args:
            state: Current state observation
            training: Whether in training mode
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            action, log_prob = self.policy(state_tensor, deterministic=not training)
            value = self.value_network(state_tensor)
            
            action = action.cpu().numpy().squeeze(0)
            log_prob = log_prob.item() if log_prob is not None else 0.0
            value = value.item()
        
        return action, log_prob, value
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in rollout buffer."""
        self.rollout_buffer.add(state, action, reward, value, log_prob, done)
    
    def update(self, last_value: float) -> dict:
        """Perform PPO update on collected rollout.
        
        Args:
            last_value: Value estimate for final state
            
        Returns:
            Update statistics
        """
        returns, advantages = self.rollout_buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states, actions, old_log_probs = self.rollout_buffer.get_tensors(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        num_samples = len(states)
        indices = np.arange(num_samples)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                new_log_probs, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                new_values = self.value_network(batch_states)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(new_values, batch_returns)
                
                entropy_loss = -entropy.mean()
                
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_network.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1
        
        self.rollout_buffer.clear()
        
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_losses.append(avg_entropy_loss)
        
        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy_loss": avg_entropy_loss,
        }
    
    def train(
        self,
        env,
        episodes: int,
        rollout_steps: int = 2048,
        max_steps: int = 1000,
        verbose: bool = True,
    ) -> dict:
        """Train agent on environment.
        
        Args:
            env: Gymnasium environment
            episodes: Number of training episodes
            rollout_steps: Steps between policy updates
            max_steps: Maximum steps per episode
            verbose: Print training progress
            
        Returns:
            Training statistics
        """
        episode_rewards = []
        current_episode_reward = 0
        step_count = 0
        
        state, _ = env.reset()
        
        for episode in range(episodes):
            for _ in range(rollout_steps):
                action, log_prob, value = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                self.store_transition(state, action, reward, value, log_prob, done)
                
                current_episode_reward += reward
                step_count += 1
                
                if done:
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0
                    state, _ = env.reset()
                else:
                    state = next_state
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                last_value = self.value_network(state_tensor).item()
            
            update_stats = self.update(last_value)
            
            if verbose and (episode + 1) % 10 == 0:
                if episode_rewards:
                    avg_reward = np.mean(episode_rewards[-10:])
                else:
                    avg_reward = 0.0
                print(
                    f"Episode {episode + 1}/{episodes} | "
                    f"Avg Reward: {avg_reward:.4f} | "
                    f"Policy Loss: {update_stats['policy_loss']:.4f}"
                )
        
        return {
            "episode_rewards": episode_rewards,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropy_losses": self.entropy_losses,
        }
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            "policy": self.policy.state_dict(),
            "value_network": self.value_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
