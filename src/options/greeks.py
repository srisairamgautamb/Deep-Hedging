"""Greeks calculation for options using finite difference methods."""

import numpy as np
from typing import Protocol, Optional
from dataclasses import dataclass


class PriceableOption(Protocol):
    """Protocol for options that can be priced."""
    
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        **kwargs
    ) -> float:
        ...


@dataclass
class Greeks:
    """Container for option Greeks values."""
    
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


class GreeksCalculator:
    """Calculate option Greeks using finite difference methods."""
    
    def __init__(
        self,
        spot_bump: float = 0.01,
        vol_bump: float = 0.01,
        rate_bump: float = 0.0001,
        time_bump: float = 1.0 / 365.0,
    ):
        """Initialize calculator with bump sizes.
        
        Args:
            spot_bump: Relative bump for spot price (1% default)
            vol_bump: Absolute bump for volatility (1% default)
            rate_bump: Absolute bump for interest rate (1bp default)
            time_bump: Time decay in years (1 day default)
        """
        self.spot_bump = spot_bump
        self.vol_bump = vol_bump
        self.rate_bump = rate_bump
        self.time_bump = time_bump
    
    def delta(
        self,
        option: PriceableOption,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        **kwargs
    ) -> float:
        """Calculate delta using central difference."""
        bump = spot * self.spot_bump
        
        price_up = option.price(spot + bump, rate, volatility, time_to_maturity, **kwargs)
        price_down = option.price(spot - bump, rate, volatility, time_to_maturity, **kwargs)
        
        return (price_up - price_down) / (2 * bump)
    
    def gamma(
        self,
        option: PriceableOption,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        **kwargs
    ) -> float:
        """Calculate gamma using central difference."""
        bump = spot * self.spot_bump
        
        price_up = option.price(spot + bump, rate, volatility, time_to_maturity, **kwargs)
        price_mid = option.price(spot, rate, volatility, time_to_maturity, **kwargs)
        price_down = option.price(spot - bump, rate, volatility, time_to_maturity, **kwargs)
        
        return (price_up - 2 * price_mid + price_down) / (bump ** 2)
    
    def vega(
        self,
        option: PriceableOption,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        **kwargs
    ) -> float:
        """Calculate vega using central difference (per 1% vol move)."""
        bump = self.vol_bump
        
        price_up = option.price(spot, rate, volatility + bump, time_to_maturity, **kwargs)
        price_down = option.price(spot, rate, max(volatility - bump, 0.001), time_to_maturity, **kwargs)
        
        return (price_up - price_down) / 2
    
    def theta(
        self,
        option: PriceableOption,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        **kwargs
    ) -> float:
        """Calculate theta using forward difference (per day)."""
        if time_to_maturity <= self.time_bump:
            return 0.0
        
        price_now = option.price(spot, rate, volatility, time_to_maturity, **kwargs)
        price_later = option.price(spot, rate, volatility, time_to_maturity - self.time_bump, **kwargs)
        
        return price_later - price_now
    
    def rho(
        self,
        option: PriceableOption,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        **kwargs
    ) -> float:
        """Calculate rho using central difference (per 1% rate move)."""
        bump = self.rate_bump
        
        price_up = option.price(spot, rate + bump, volatility, time_to_maturity, **kwargs)
        price_down = option.price(spot, rate - bump, volatility, time_to_maturity, **kwargs)
        
        return (price_up - price_down) / (200 * bump)
    
    def calculate_all(
        self,
        option: PriceableOption,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        **kwargs
    ) -> Greeks:
        """Calculate all Greeks for an option."""
        return Greeks(
            delta=self.delta(option, spot, rate, volatility, time_to_maturity, **kwargs),
            gamma=self.gamma(option, spot, rate, volatility, time_to_maturity, **kwargs),
            vega=self.vega(option, spot, rate, volatility, time_to_maturity, **kwargs),
            theta=self.theta(option, spot, rate, volatility, time_to_maturity, **kwargs),
            rho=self.rho(option, spot, rate, volatility, time_to_maturity, **kwargs),
        )


def compute_hedge_ratio(
    option: PriceableOption,
    spot: float,
    rate: float,
    volatility: float,
    time_to_maturity: float,
    **kwargs
) -> float:
    """Compute delta hedge ratio for an option position."""
    calc = GreeksCalculator()
    return calc.delta(option, spot, rate, volatility, time_to_maturity, **kwargs)
