"""Barrier option pricing using analytical formulas."""

import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class BarrierOption(ABC):
    """Base class for single-barrier options."""
    
    strike: float
    barrier: float
    maturity: float
    rebate: float = 0.0
    
    @abstractmethod
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        dividend: float = 0.0
    ) -> float:
        """Calculate barrier option price."""
        pass
    
    @abstractmethod
    def payoff(self, spot: float, barrier_hit: bool) -> float:
        """Calculate option payoff considering barrier status."""
        pass
    
    def _compute_lambda(self, rate: float, dividend: float, volatility: float) -> float:
        """Compute lambda parameter for barrier pricing."""
        return (rate - dividend + 0.5 * volatility ** 2) / (volatility ** 2)
    
    def _compute_y(
        self,
        spot: float,
        rate: float,
        dividend: float,
        volatility: float,
        time_to_maturity: float
    ) -> float:
        """Compute y parameter for barrier pricing."""
        sigma_sqrt_t = volatility * np.sqrt(time_to_maturity)
        return np.log(self.barrier ** 2 / (spot * self.strike)) / sigma_sqrt_t + self._compute_lambda(rate, dividend, volatility) * sigma_sqrt_t
    
    def _compute_x1(
        self,
        spot: float,
        rate: float,
        dividend: float,
        volatility: float,
        time_to_maturity: float
    ) -> float:
        """Compute x1 parameter."""
        sigma_sqrt_t = volatility * np.sqrt(time_to_maturity)
        return np.log(spot / self.barrier) / sigma_sqrt_t + self._compute_lambda(rate, dividend, volatility) * sigma_sqrt_t
    
    def _compute_y1(
        self,
        spot: float,
        rate: float,
        dividend: float,
        volatility: float,
        time_to_maturity: float
    ) -> float:
        """Compute y1 parameter."""
        sigma_sqrt_t = volatility * np.sqrt(time_to_maturity)
        return np.log(self.barrier / spot) / sigma_sqrt_t + self._compute_lambda(rate, dividend, volatility) * sigma_sqrt_t


@dataclass
class UpOutCall(BarrierOption):
    """Up-and-out call option."""
    
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        dividend: float = 0.0
    ) -> float:
        """Price up-and-out call using analytical formula."""
        if time_to_maturity <= 0:
            return self.payoff(spot, spot >= self.barrier)
        
        if spot >= self.barrier:
            return self.rebate * np.exp(-rate * time_to_maturity)
        
        if self.barrier <= self.strike:
            return 0.0
        
        sigma = volatility
        sigma_sqrt_t = sigma * np.sqrt(time_to_maturity)
        discount = np.exp(-rate * time_to_maturity)
        dividend_discount = np.exp(-dividend * time_to_maturity)
        
        lam = self._compute_lambda(rate, dividend, sigma)
        
        d1 = (np.log(spot / self.strike) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
        
        d1_h = (np.log(spot / self.barrier) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
        d2_h = d1_h - sigma_sqrt_t
        
        d1_prime = (np.log(self.barrier ** 2 / (spot * self.strike)) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
        d2_prime = d1_prime - sigma_sqrt_t
        
        d1_h_prime = (np.log(self.barrier / spot) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
        d2_h_prime = d1_h_prime - sigma_sqrt_t
        
        a = spot * dividend_discount * (norm.cdf(d1) - norm.cdf(d1_h))
        b = self.strike * discount * (norm.cdf(d2) - norm.cdf(d2_h))
        
        ratio = self.barrier / spot
        c = spot * dividend_discount * (ratio ** (2 * lam)) * (norm.cdf(d1_prime) - norm.cdf(d1_h_prime))
        d = self.strike * discount * (ratio ** (2 * lam - 2)) * (norm.cdf(d2_prime) - norm.cdf(d2_h_prime))
        
        return a - b - c + d
    
    def payoff(self, spot: float, barrier_hit: bool) -> float:
        """Calculate payoff given barrier status."""
        if barrier_hit:
            return self.rebate
        return max(spot - self.strike, 0)


@dataclass
class DownOutCall(BarrierOption):
    """Down-and-out call option."""
    
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        dividend: float = 0.0
    ) -> float:
        """Price down-and-out call."""
        if time_to_maturity <= 0:
            return self.payoff(spot, spot <= self.barrier)
        
        if spot <= self.barrier:
            return self.rebate * np.exp(-rate * time_to_maturity)
        
        sigma = volatility
        sigma_sqrt_t = sigma * np.sqrt(time_to_maturity)
        discount = np.exp(-rate * time_to_maturity)
        dividend_discount = np.exp(-dividend * time_to_maturity)
        
        lam = self._compute_lambda(rate, dividend, sigma)
        ratio = self.barrier / spot
        
        if self.strike >= self.barrier:
            d1 = (np.log(spot / self.strike) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
            d2 = d1 - sigma_sqrt_t
            
            d1_prime = (np.log(self.barrier ** 2 / (spot * self.strike)) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
            d2_prime = d1_prime - sigma_sqrt_t
            
            vanilla = spot * dividend_discount * norm.cdf(d1) - self.strike * discount * norm.cdf(d2)
            reflection = (ratio ** (2 * lam)) * (spot * dividend_discount * norm.cdf(d1_prime) - self.strike * discount * (ratio ** (-2)) * norm.cdf(d2_prime))
            
            return vanilla - reflection
        else:
            d1_h = (np.log(spot / self.barrier) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
            d2_h = d1_h - sigma_sqrt_t
            
            d1_h_prime = (np.log(self.barrier / spot) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
            d2_h_prime = d1_h_prime - sigma_sqrt_t
            
            a = spot * dividend_discount * norm.cdf(d1_h) - self.strike * discount * norm.cdf(d2_h)
            b = (ratio ** (2 * lam)) * spot * dividend_discount * norm.cdf(d1_h_prime)
            c = (ratio ** (2 * lam - 2)) * self.strike * discount * norm.cdf(d2_h_prime)
            
            return a - b + c
    
    def payoff(self, spot: float, barrier_hit: bool) -> float:
        """Calculate payoff given barrier status."""
        if barrier_hit:
            return self.rebate
        return max(spot - self.strike, 0)


@dataclass
class UpOutPut(BarrierOption):
    """Up-and-out put option."""
    
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        dividend: float = 0.0
    ) -> float:
        """Price up-and-out put."""
        if time_to_maturity <= 0:
            return self.payoff(spot, spot >= self.barrier)
        
        if spot >= self.barrier:
            return self.rebate * np.exp(-rate * time_to_maturity)
        
        sigma = volatility
        sigma_sqrt_t = sigma * np.sqrt(time_to_maturity)
        discount = np.exp(-rate * time_to_maturity)
        dividend_discount = np.exp(-dividend * time_to_maturity)
        
        lam = self._compute_lambda(rate, dividend, sigma)
        ratio = self.barrier / spot
        
        if self.strike <= self.barrier:
            d1 = (np.log(spot / self.strike) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
            d2 = d1 - sigma_sqrt_t
            
            d1_prime = (np.log(self.barrier ** 2 / (spot * self.strike)) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
            d2_prime = d1_prime - sigma_sqrt_t
            
            vanilla = self.strike * discount * norm.cdf(-d2) - spot * dividend_discount * norm.cdf(-d1)
            reflection = (ratio ** (2 * lam - 2)) * self.strike * discount * norm.cdf(-d2_prime) - (ratio ** (2 * lam)) * spot * dividend_discount * norm.cdf(-d1_prime)
            
            return vanilla - reflection
        else:
            d1_h = (np.log(spot / self.barrier) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
            d2_h = d1_h - sigma_sqrt_t
            
            return self.strike * discount * norm.cdf(-d2_h) - spot * dividend_discount * norm.cdf(-d1_h)
    
    def payoff(self, spot: float, barrier_hit: bool) -> float:
        """Calculate payoff given barrier status."""
        if barrier_hit:
            return self.rebate
        return max(self.strike - spot, 0)


@dataclass
class DownOutPut(BarrierOption):
    """Down-and-out put option."""
    
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        dividend: float = 0.0
    ) -> float:
        """Price down-and-out put."""
        if time_to_maturity <= 0:
            return self.payoff(spot, spot <= self.barrier)
        
        if spot <= self.barrier:
            return self.rebate * np.exp(-rate * time_to_maturity)
        
        if self.barrier >= self.strike:
            return 0.0
        
        sigma = volatility
        sigma_sqrt_t = sigma * np.sqrt(time_to_maturity)
        discount = np.exp(-rate * time_to_maturity)
        dividend_discount = np.exp(-dividend * time_to_maturity)
        
        lam = self._compute_lambda(rate, dividend, sigma)
        ratio = self.barrier / spot
        
        d1 = (np.log(spot / self.strike) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
        
        d1_h = (np.log(spot / self.barrier) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
        d2_h = d1_h - sigma_sqrt_t
        
        d1_prime = (np.log(self.barrier ** 2 / (spot * self.strike)) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
        d2_prime = d1_prime - sigma_sqrt_t
        
        d1_h_prime = (np.log(self.barrier / spot) + (rate - dividend + 0.5 * sigma ** 2) * time_to_maturity) / sigma_sqrt_t
        d2_h_prime = d1_h_prime - sigma_sqrt_t
        
        a = self.strike * discount * (norm.cdf(-d2) - norm.cdf(-d2_h))
        b = spot * dividend_discount * (norm.cdf(-d1) - norm.cdf(-d1_h))
        c = (ratio ** (2 * lam - 2)) * self.strike * discount * (norm.cdf(-d2_prime) - norm.cdf(-d2_h_prime))
        d = (ratio ** (2 * lam)) * spot * dividend_discount * (norm.cdf(-d1_prime) - norm.cdf(-d1_h_prime))
        
        return a - b - c + d
    
    def payoff(self, spot: float, barrier_hit: bool) -> float:
        """Calculate payoff given barrier status."""
        if barrier_hit:
            return self.rebate
        return max(self.strike - spot, 0)


@dataclass
class UpInCall(BarrierOption):
    """Up-and-in call option (knock-in)."""
    
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        dividend: float = 0.0
    ) -> float:
        """Price up-and-in call using in-out parity."""
        from .vanilla import Call
        
        vanilla = Call(strike=self.strike, maturity=self.maturity)
        up_out = UpOutCall(strike=self.strike, barrier=self.barrier, maturity=self.maturity, rebate=0.0)
        
        vanilla_price = vanilla.price(spot, rate, volatility, time_to_maturity)
        up_out_price = up_out.price(spot, rate, volatility, time_to_maturity, dividend)
        
        return vanilla_price - up_out_price
    
    def payoff(self, spot: float, barrier_hit: bool) -> float:
        """Calculate payoff given barrier status."""
        if not barrier_hit:
            return self.rebate
        return max(spot - self.strike, 0)


@dataclass
class DownInCall(BarrierOption):
    """Down-and-in call option (knock-in)."""
    
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        dividend: float = 0.0
    ) -> float:
        """Price down-and-in call using in-out parity."""
        from .vanilla import Call
        
        vanilla = Call(strike=self.strike, maturity=self.maturity)
        down_out = DownOutCall(strike=self.strike, barrier=self.barrier, maturity=self.maturity, rebate=0.0)
        
        vanilla_price = vanilla.price(spot, rate, volatility, time_to_maturity)
        down_out_price = down_out.price(spot, rate, volatility, time_to_maturity, dividend)
        
        return vanilla_price - down_out_price
    
    def payoff(self, spot: float, barrier_hit: bool) -> float:
        """Calculate payoff given barrier status."""
        if not barrier_hit:
            return self.rebate
        return max(spot - self.strike, 0)


@dataclass
class UpInPut(BarrierOption):
    """Up-and-in put option (knock-in)."""
    
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        dividend: float = 0.0
    ) -> float:
        """Price up-and-in put using in-out parity."""
        from .vanilla import Put
        
        vanilla = Put(strike=self.strike, maturity=self.maturity)
        up_out = UpOutPut(strike=self.strike, barrier=self.barrier, maturity=self.maturity, rebate=0.0)
        
        vanilla_price = vanilla.price(spot, rate, volatility, time_to_maturity)
        up_out_price = up_out.price(spot, rate, volatility, time_to_maturity, dividend)
        
        return vanilla_price - up_out_price
    
    def payoff(self, spot: float, barrier_hit: bool) -> float:
        """Calculate payoff given barrier status."""
        if not barrier_hit:
            return self.rebate
        return max(self.strike - spot, 0)


@dataclass
class DownInPut(BarrierOption):
    """Down-and-in put option (knock-in)."""
    
    def price(
        self,
        spot: float,
        rate: float,
        volatility: float,
        time_to_maturity: float,
        dividend: float = 0.0
    ) -> float:
        """Price down-and-in put using in-out parity."""
        from .vanilla import Put
        
        vanilla = Put(strike=self.strike, maturity=self.maturity)
        down_out = DownOutPut(strike=self.strike, barrier=self.barrier, maturity=self.maturity, rebate=0.0)
        
        vanilla_price = vanilla.price(spot, rate, volatility, time_to_maturity)
        down_out_price = down_out.price(spot, rate, volatility, time_to_maturity, dividend)
        
        return vanilla_price - down_out_price
    
    def payoff(self, spot: float, barrier_hit: bool) -> float:
        """Calculate payoff given barrier status."""
        if not barrier_hit:
            return self.rebate
        return max(self.strike - spot, 0)
