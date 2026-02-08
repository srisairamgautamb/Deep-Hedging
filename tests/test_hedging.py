"""Unit tests for hedging agents and environment."""

import pytest
import numpy as np
import torch

from src.options.barrier import UpOutCall
from src.hedging.environment import HedgingEnvironment, DiscreteHedgingEnvironment
from src.hedging.agents import DQNAgent, PPOAgent


class TestHedgingEnvironment:
    """Test hedging environment functionality."""
    
    @pytest.fixture
    def option(self):
        return UpOutCall(strike=100, barrier=120, maturity=0.25)
    
    @pytest.fixture
    def env(self, option):
        return HedgingEnvironment(
            option=option,
            initial_spot=100.0,
            volatility=0.20,
            rate=0.05,
            cost_multiplier=0.001,
            seed=42,
        )
    
    def test_environment_reset(self, env):
        """Environment should reset to initial state."""
        obs, info = env.reset()
        
        assert obs.shape == env.observation_space.shape
        assert info["step"] == 0
        assert info["hedge_position"] == 0.0
    
    def test_environment_step(self, env):
        """Environment should process steps correctly."""
        env.reset()
        action = np.array([0.5])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert info["step"] == 1
    
    def test_action_affects_position(self, env):
        """Action should update hedge position."""
        env.reset()
        
        action = np.array([0.5])
        _, _, _, _, info = env.step(action)
        
        expected_position = 0.5 * env.max_position
        assert info["hedge_position"] == pytest.approx(expected_position, rel=0.01)
    
    def test_transaction_costs_incurred(self, env):
        """Trading should incur transaction costs."""
        env.reset()
        
        env.step(np.array([0.5]))
        env.step(np.array([-0.5]))
        
        assert env.total_cost > 0
    
    def test_terminal_at_maturity(self, env):
        """Episode should terminate at option maturity."""
        env.reset()
        
        terminated = False
        step = 0
        while not terminated and step < 1000:
            _, _, terminated, _, _ = env.step(np.array([0.0]))
            step += 1
        
        assert terminated


class TestDiscreteHedgingEnvironment:
    """Test discrete action space environment."""
    
    @pytest.fixture
    def option(self):
        return UpOutCall(strike=100, barrier=120, maturity=0.1)
    
    @pytest.fixture
    def env(self, option):
        return DiscreteHedgingEnvironment(
            option=option,
            num_actions=21,
            seed=42,
        )
    
    def test_discrete_action_space(self, env):
        """Should have discrete action space."""
        from gymnasium.spaces import Discrete
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 21
    
    def test_discrete_step(self, env):
        """Should process discrete actions."""
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(10)
        
        assert obs.shape == env.observation_space.shape


class TestDQNAgent:
    """Test DQN agent functionality."""
    
    @pytest.fixture
    def agent(self):
        return DQNAgent(
            state_dim=7,
            num_actions=21,
            hidden_dims=[64, 64],
            buffer_size=1000,
            batch_size=32,
        )
    
    def test_action_selection(self, agent):
        """Agent should select valid actions."""
        state = np.random.randn(7).astype(np.float32)
        
        action = agent.select_action(state, training=True)
        
        assert 0 <= action < agent.num_actions
    
    def test_store_transition(self, agent):
        """Agent should store transitions."""
        state = np.random.randn(7).astype(np.float32)
        next_state = np.random.randn(7).astype(np.float32)
        
        agent.store_transition(state, 5, 0.1, next_state, False)
        
        assert len(agent.replay_buffer) == 1
    
    def test_training_step(self, agent):
        """Agent should train when buffer has enough samples."""
        for _ in range(50):
            state = np.random.randn(7).astype(np.float32)
            next_state = np.random.randn(7).astype(np.float32)
            agent.store_transition(state, np.random.randint(21), np.random.randn(), next_state, False)
        
        loss = agent.train_step()
        
        assert loss is not None or len(agent.replay_buffer) < agent.batch_size


class TestPPOAgent:
    """Test PPO agent functionality."""
    
    @pytest.fixture
    def agent(self):
        return PPOAgent(
            state_dim=7,
            action_dim=1,
            hidden_dims=[64, 64],
        )
    
    def test_action_selection(self, agent):
        """Agent should select continuous actions."""
        state = np.random.randn(7).astype(np.float32)
        
        action, log_prob, value = agent.select_action(state, training=True)
        
        assert action.shape == (1,)
        assert -1 <= action[0] <= 1
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    def test_deterministic_action(self, agent):
        """Deterministic mode should give consistent actions."""
        state = np.random.randn(7).astype(np.float32)
        
        action1, _, _ = agent.select_action(state, training=False)
        action2, _, _ = agent.select_action(state, training=False)
        
        np.testing.assert_array_almost_equal(action1, action2)
    
    def test_store_and_update(self, agent):
        """Agent should update from rollout buffer."""
        for _ in range(100):
            state = np.random.randn(7).astype(np.float32)
            action = np.random.randn(1).astype(np.float32)
            agent.store_transition(state, action, np.random.randn(), np.random.randn(), np.random.rand(), False)
        
        stats = agent.update(last_value=0.0)
        
        assert "policy_loss" in stats
        assert "value_loss" in stats


class TestAgentCheckpointing:
    """Test agent save/load functionality."""
    
    def test_dqn_save_load(self, tmp_path):
        """DQN should save and load correctly."""
        agent = DQNAgent(state_dim=7, num_actions=21)
        path = tmp_path / "dqn.pt"
        
        agent.save(str(path))
        
        new_agent = DQNAgent(state_dim=7, num_actions=21)
        new_agent.load(str(path))
        
        state = np.random.randn(7).astype(np.float32)
        assert agent.select_action(state, training=False) == new_agent.select_action(state, training=False)
    
    def test_ppo_save_load(self, tmp_path):
        """PPO should save and load correctly."""
        agent = PPOAgent(state_dim=7, action_dim=1)
        path = tmp_path / "ppo.pt"
        
        agent.save(str(path))
        
        new_agent = PPOAgent(state_dim=7, action_dim=1)
        new_agent.load(str(path))
        
        state = np.random.randn(7).astype(np.float32)
        action1, _, _ = agent.select_action(state, training=False)
        action2, _, _ = new_agent.select_action(state, training=False)
        
        np.testing.assert_array_almost_equal(action1, action2)
