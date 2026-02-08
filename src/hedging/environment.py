"""Gymnasium-compatible hedging environment for deep RL training."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

from ..options.barrier import BarrierOption
from ..options.greeks import GreeksCalculator


@dataclass
class MarketState:
    """Current market state for hedging environment."""
    
    spot: float
    volatility: float
    rate: float
    time_to_maturity: float
    barrier_hit: bool = False
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array."""
        return np.array([
            self.spot,
            self.volatility,
            self.rate,
            self.time_to_maturity,
            float(self.barrier_hit),
        ], dtype=np.float32)


@dataclass
class PositionState:
    """Current position state for hedging."""
    
    option_position: float = 1.0
    hedge_position: float = 0.0
    cash: float = 0.0
    pnl_history: List[float] = field(default_factory=list)
    
    def portfolio_value(self, spot: float, option_value: float) -> float:
        """Calculate total portfolio value."""
        return self.option_position * option_value + self.hedge_position * spot + self.cash


class HedgingEnvironment(gym.Env):
    """Trading environment for learning optimal hedging strategies."""
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        option: BarrierOption,
        initial_spot: float = 100.0,
        volatility: float = 0.20,
        rate: float = 0.05,
        dt: float = 1.0 / 252.0,
        cost_multiplier: float = 0.001,
        slippage_bps: float = 1.0,
        max_position: float = 100.0,
        seed: Optional[int] = None,
    ):
        """Initialize hedging environment.
        
        Args:
            option: Barrier option to hedge
            initial_spot: Starting spot price
            volatility: Annualized volatility
            rate: Risk-free interest rate
            dt: Time step in years (trading days)
            cost_multiplier: Proportional transaction cost
            slippage_bps: Slippage in basis points
            max_position: Maximum hedge position size
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.option = option
        self.initial_spot = initial_spot
        self.volatility = volatility
        self.rate = rate
        self.dt = dt
        self.cost_multiplier = cost_multiplier
        self.slippage_bps = slippage_bps / 10000.0
        self.max_position = max_position
        
        self.greeks_calc = GreeksCalculator()
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0, 0.0, -max_position, -np.inf], dtype=np.float32),
            high=np.array([np.inf, 1.0, 1.0, option.maturity, 1.0, max_position, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        
        self._rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        self.market = MarketState(
            spot=self.initial_spot,
            volatility=self.volatility,
            rate=self.rate,
            time_to_maturity=self.option.maturity,
            barrier_hit=False,
        )
        
        initial_option_value = self.option.price(
            self.market.spot,
            self.market.rate,
            self.market.volatility,
            self.market.time_to_maturity,
        )
        
        self.position = PositionState(
            option_position=1.0,
            hedge_position=0.0,
            cash=-initial_option_value,
            pnl_history=[],
        )
        
        self.step_count = 0
        self.total_cost = 0.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step in the environment."""
        target_hedge = float(action[0]) * self.max_position
        
        trade_size = target_hedge - self.position.hedge_position
        trade_cost = self._compute_transaction_cost(trade_size, self.market.spot)
        
        self.position.hedge_position = target_hedge
        self.position.cash -= trade_size * self.market.spot + trade_cost
        self.total_cost += trade_cost
        
        old_spot = self.market.spot
        new_spot = self._simulate_price_step()
        self.market.spot = new_spot
        self.market.time_to_maturity = max(0, self.market.time_to_maturity - self.dt)
        
        if isinstance(self.option, (type(self.option),)):
            barrier = self.option.barrier
            if hasattr(self.option, '__class__') and 'Up' in self.option.__class__.__name__:
                if new_spot >= barrier:
                    self.market.barrier_hit = True
            elif hasattr(self.option, '__class__') and 'Down' in self.option.__class__.__name__:
                if new_spot <= barrier:
                    self.market.barrier_hit = True
        
        hedge_pnl = self.position.hedge_position * (new_spot - old_spot)
        option_value = self.option.price(
            self.market.spot,
            self.market.rate,
            self.market.volatility,
            self.market.time_to_maturity,
        )
        
        portfolio_value = self.position.portfolio_value(new_spot, option_value)
        self.position.pnl_history.append(portfolio_value)
        
        reward = self._compute_reward(hedge_pnl, trade_cost)
        
        self.step_count += 1
        terminated = self.market.time_to_maturity <= 0 or self.market.barrier_hit
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _simulate_price_step(self) -> float:
        """Simulate one step of geometric Brownian motion."""
        drift = (self.rate - 0.5 * self.volatility ** 2) * self.dt
        diffusion = self.volatility * np.sqrt(self.dt) * self._rng.standard_normal()
        return self.market.spot * np.exp(drift + diffusion)
    
    def _compute_transaction_cost(self, trade_size: float, spot: float) -> float:
        """Compute transaction cost for a trade."""
        notional = abs(trade_size) * spot
        proportional_cost = notional * self.cost_multiplier
        slippage_cost = notional * self.slippage_bps
        return proportional_cost + slippage_cost
    
    def _compute_reward(self, hedge_pnl: float, trade_cost: float) -> float:
        """Compute reward signal for the agent."""
        pnl_penalty = -abs(hedge_pnl) * 0.1
        cost_penalty = -trade_cost * 10.0
        return pnl_penalty + cost_penalty
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation array."""
        delta = self.greeks_calc.delta(
            self.option,
            self.market.spot,
            self.market.rate,
            self.market.volatility,
            self.market.time_to_maturity,
        )
        
        return np.array([
            self.market.spot / self.initial_spot,
            self.market.volatility,
            self.market.rate,
            self.market.time_to_maturity,
            float(self.market.barrier_hit),
            self.position.hedge_position / self.max_position,
            delta,
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary information."""
        return {
            "step": self.step_count,
            "spot": self.market.spot,
            "time_to_maturity": self.market.time_to_maturity,
            "barrier_hit": self.market.barrier_hit,
            "hedge_position": self.position.hedge_position,
            "total_cost": self.total_cost,
            "pnl_variance": np.var(self.position.pnl_history) if len(self.position.pnl_history) > 1 else 0.0,
        }


class DiscreteHedgingEnvironment(HedgingEnvironment):
    """Discrete action space variant for DQN agent."""
    
    def __init__(
        self,
        option: BarrierOption,
        num_actions: int = 21,
        **kwargs
    ):
        """Initialize with discrete action space.
        
        Args:
            option: Barrier option to hedge
            num_actions: Number of discrete hedge ratio choices
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(option=option, **kwargs)
        
        self.num_actions = num_actions
        self.action_space = spaces.Discrete(num_actions)
        
        self._action_map = np.linspace(-1.0, 1.0, num_actions)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step with discrete action."""
        continuous_action = np.array([self._action_map[action]], dtype=np.float32)
        return super().step(continuous_action)
