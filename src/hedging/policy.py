"""Hedging policy network for deep RL agents."""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, List


class HedgingPolicy(nn.Module):
    """Neural network policy for determining hedge ratios."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        """Initialize policy network.
        
        Args:
            state_dim: Dimension of state observation
            action_dim: Dimension of action output
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        activation_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
        }.get(activation, nn.ReLU)
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        self.mean_head = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
            nn.Tanh(),
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        nn.init.orthogonal_(self.mean_head[0].weight, gain=0.01)
    
    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass to compute action.
        
        Args:
            state: Batch of state observations
            deterministic: If True, return mean action; else sample
            
        Returns:
            action: Selected action
            log_prob: Log probability of action (None if deterministic)
        """
        features = self.shared_layers(state)
        mean = self.mean_head(features)
        
        if deterministic:
            return mean, None
        
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for given state-action pairs.
        
        Args:
            state: Batch of states
            action: Batch of actions
            
        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of action distribution
        """
        features = self.shared_layers(state)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std)
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Value function for policy gradient methods."""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        """Initialize value network.
        
        Args:
            state_dim: Dimension of state observation
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        activation_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
        }.get(activation, nn.ReLU)
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute value estimate for state."""
        return self.network(state).squeeze(-1)


class QNetwork(nn.Module):
    """Q-function for DQN-based methods."""
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        dueling: bool = True,
    ):
        """Initialize Q-network.
        
        Args:
            state_dim: Dimension of state observation
            num_actions: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            dueling: Use dueling architecture
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.dueling = dueling
        
        activation_fn = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }.get(activation, nn.ReLU)
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        if dueling:
            self.value_stream = nn.Linear(prev_dim, 1)
            self.advantage_stream = nn.Linear(prev_dim, num_actions)
        else:
            self.q_head = nn.Linear(prev_dim, num_actions)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions."""
        features = self.shared_layers(state)
        
        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(features)
        
        return q_values
