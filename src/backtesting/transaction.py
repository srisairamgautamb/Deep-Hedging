"""Transaction cost modeling for realistic backtesting."""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


class TransactionCostModel(ABC):
    """Base class for transaction cost models."""
    
    @abstractmethod
    def compute_cost(
        self,
        trade_size: float,
        price: float,
        spread: float = 0.0
    ) -> float:
        """Compute transaction cost for a trade.
        
        Args:
            trade_size: Number of units traded (positive or negative)
            price: Current market price
            spread: Bid-ask spread
            
        Returns:
            Total transaction cost
        """
        pass
    
    @abstractmethod
    def compute_slippage(
        self,
        trade_size: float,
        price: float,
        volatility: float = 0.0
    ) -> float:
        """Compute expected slippage for a trade.
        
        Args:
            trade_size: Number of units traded
            price: Current market price
            volatility: Current volatility
            
        Returns:
            Expected slippage amount
        """
        pass


@dataclass
class ProportionalCost(TransactionCostModel):
    """Proportional (linear) transaction cost model."""
    
    commission_rate: float = 0.001
    slippage_bps: float = 1.0
    min_commission: float = 0.0
    
    def compute_cost(
        self,
        trade_size: float,
        price: float,
        spread: float = 0.0
    ) -> float:
        """Compute proportional transaction cost."""
        notional = abs(trade_size) * price
        commission = max(self.min_commission, notional * self.commission_rate)
        spread_cost = abs(trade_size) * spread / 2
        return commission + spread_cost
    
    def compute_slippage(
        self,
        trade_size: float,
        price: float,
        volatility: float = 0.0
    ) -> float:
        """Compute slippage based on trade size."""
        notional = abs(trade_size) * price
        base_slippage = notional * self.slippage_bps / 10000
        vol_impact = notional * volatility * 0.01 if volatility > 0 else 0
        return base_slippage + vol_impact


@dataclass
class TieredCost(TransactionCostModel):
    """Tiered transaction cost model with volume breaks."""
    
    tiers: List[Tuple[float, float]] = None
    slippage_bps: float = 1.0
    impact_coefficient: float = 0.1
    
    def __post_init__(self):
        if self.tiers is None:
            self.tiers = [
                (0.0, 0.0020),
                (100000.0, 0.0015),
                (500000.0, 0.0010),
                (1000000.0, 0.0005),
            ]
    
    def compute_cost(
        self,
        trade_size: float,
        price: float,
        spread: float = 0.0
    ) -> float:
        """Compute tiered transaction cost."""
        notional = abs(trade_size) * price
        
        rate = self.tiers[0][1]
        for threshold, tier_rate in self.tiers:
            if notional >= threshold:
                rate = tier_rate
        
        commission = notional * rate
        spread_cost = abs(trade_size) * spread / 2
        
        return commission + spread_cost
    
    def compute_slippage(
        self,
        trade_size: float,
        price: float,
        volatility: float = 0.0
    ) -> float:
        """Compute slippage with market impact."""
        notional = abs(trade_size) * price
        
        linear_slippage = notional * self.slippage_bps / 10000
        
        market_impact = self.impact_coefficient * np.sqrt(notional) * np.sign(trade_size)
        
        return linear_slippage + abs(market_impact)


@dataclass
class MarketImpactModel:
    """Almgren-Chriss style market impact model."""
    
    eta: float = 0.01
    gamma: float = 0.1
    sigma: float = 0.0
    
    def temporary_impact(self, trade_rate: float) -> float:
        """Compute temporary market impact."""
        return self.eta * trade_rate
    
    def permanent_impact(self, trade_size: float) -> float:
        """Compute permanent market impact."""
        return self.gamma * trade_size
    
    def total_cost(
        self,
        trade_size: float,
        duration: float,
        price: float
    ) -> float:
        """Compute total execution cost."""
        trade_rate = abs(trade_size) / duration if duration > 0 else abs(trade_size)
        
        temp_impact = self.temporary_impact(trade_rate)
        perm_impact = self.permanent_impact(trade_size)
        
        notional = abs(trade_size) * price
        return notional * (temp_impact + 0.5 * perm_impact)


def estimate_optimal_execution(
    target_size: float,
    duration: float,
    volatility: float,
    eta: float = 0.01,
    gamma: float = 0.1,
    risk_aversion: float = 1.0
) -> np.ndarray:
    """Compute optimal execution trajectory using Almgren-Chriss.
    
    Args:
        target_size: Total position to execute
        duration: Time horizon in same units as volatility
        volatility: Annualized volatility
        eta: Temporary impact coefficient
        gamma: Permanent impact coefficient
        risk_aversion: Risk aversion parameter
        
    Returns:
        Array of optimal trade sizes at each step
    """
    num_steps = int(duration * 252)
    if num_steps < 1:
        num_steps = 1
    
    dt = duration / num_steps
    
    kappa = np.sqrt(risk_aversion * volatility ** 2 / eta)
    
    times = np.linspace(0, duration, num_steps + 1)
    trajectory = target_size * np.sinh(kappa * (duration - times)) / np.sinh(kappa * duration)
    
    trades = -np.diff(trajectory)
    
    return trades
