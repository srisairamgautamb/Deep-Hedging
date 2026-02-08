"""Vanilla option pricing using Black-Scholes model."""

import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class VanillaOption(ABC):
    """Base class for vanilla European options."""
    
    strike: float
    maturity: float
    
    @abstractmethod
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float
    ) -> float:
        """Calculate option price."""
        pass
    
    @abstractmethod
    def payoff(self, spot: float) -> float:
        """Calculate option payoff at expiry."""
        pass
    
    def _d1(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float
    ) -> float:
        """Calculate d1 parameter for Black-Scholes."""
        if time_to_maturity <= 0 or volatility <= 0:
            return 0.0
        numerator = np.log(spot / self.strike) + (rate + 0.5 * volatility ** 2) * time_to_maturity
        denominator = volatility * np.sqrt(time_to_maturity)
        return numerator / denominator
    
    def _d2(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float
    ) -> float:
        """Calculate d2 parameter for Black-Scholes."""
        return self._d1(spot, rate, volatility, time_to_maturity) - volatility * np.sqrt(time_to_maturity)


@dataclass
class Call(VanillaOption):
    """European call option."""
    
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float
    ) -> float:
        """Calculate call option price using Black-Scholes formula."""
        if time_to_maturity <= 0:
            return max(spot - self.strike, 0)
        
        d1 = self._d1(spot, rate, volatility, time_to_maturity)
        d2 = self._d2(spot, rate, volatility, time_to_maturity)
        
        discount = np.exp(-rate * time_to_maturity)
        return spot * norm.cdf(d1) - self.strike * discount * norm.cdf(d2)
    
    def payoff(self, spot: float) -> float:
        """Calculate call payoff at expiry."""
        return max(spot - self.strike, 0)


@dataclass
class Put(VanillaOption):
    """European put option."""
    
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float
    ) -> float:
        """Calculate put option price using Black-Scholes formula."""
        if time_to_maturity <= 0:
            return max(self.strike - spot, 0)
        
        d1 = self._d1(spot, rate, volatility, time_to_maturity)
        d2 = self._d2(spot, rate, volatility, time_to_maturity)
        
        discount = np.exp(-rate * time_to_maturity)
        return self.strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    def payoff(self, spot: float) -> float:
        """Calculate put payoff at expiry."""
        return max(self.strike - spot, 0)


def european_call_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float
) -> float:
    """Standalone Black-Scholes call pricing function."""
    return Call(strike=strike, maturity=time_to_maturity).price(
        spot, rate, volatility, time_to_maturity
    )


def european_put_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    time_to_maturity: float
) -> float:
    """Standalone Black-Scholes put pricing function."""
    return Put(strike=strike, maturity=time_to_maturity).price(
        spot, rate, volatility, time_to_maturity
    )
