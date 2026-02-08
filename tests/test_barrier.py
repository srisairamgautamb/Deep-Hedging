"""Unit tests for barrier option pricing."""

import pytest
import numpy as np

from src.options.barrier import (
    UpOutCall, DownOutCall, UpOutPut, DownOutPut,
    UpInCall, DownInCall, UpInPut, DownInPut,
)
from src.options.vanilla import Call, Put


class TestBarrierOptionPricing:
    """Test barrier option pricing accuracy."""
    
    def test_up_out_call_below_barrier(self):
        """Up-out call should have positive value when spot is below barrier."""
        option = UpOutCall(strike=100, barrier=120, maturity=1.0)
        price = option.price(spot=100, rate=0.05, volatility=0.20, time_to_maturity=1.0)
        
        assert price > 0
        assert price < 20
    
    def test_up_out_call_at_barrier_is_zero(self):
        """Up-out call should return rebate when spot hits barrier."""
        option = UpOutCall(strike=100, barrier=120, maturity=1.0, rebate=0.0)
        price = option.price(spot=120, rate=0.05, volatility=0.20, time_to_maturity=1.0)
        
        assert price == pytest.approx(0.0, abs=0.01)
    
    def test_down_out_call_above_barrier(self):
        """Down-out call should have positive value when spot is above barrier."""
        option = DownOutCall(strike=100, barrier=80, maturity=1.0)
        price = option.price(spot=100, rate=0.05, volatility=0.20, time_to_maturity=1.0)
        
        assert price > 0
    
    def test_down_out_call_at_barrier_is_zero(self):
        """Down-out call should return rebate when spot hits barrier."""
        option = DownOutCall(strike=100, barrier=80, maturity=1.0, rebate=0.0)
        price = option.price(spot=80, rate=0.05, volatility=0.20, time_to_maturity=1.0)
        
        assert price == pytest.approx(0.0, abs=0.01)
    
    def test_up_out_put_pricing(self):
        """Up-out put should price correctly."""
        option = UpOutPut(strike=100, barrier=120, maturity=1.0)
        price = option.price(spot=100, rate=0.05, volatility=0.20, time_to_maturity=1.0)
        
        assert price > 0
    
    def test_down_out_put_pricing(self):
        """Down-out put should price correctly."""
        option = DownOutPut(strike=100, barrier=80, maturity=1.0)
        price = option.price(spot=100, rate=0.05, volatility=0.20, time_to_maturity=1.0)
        
        assert price >= 0


class TestBarrierInOutParity:
    """Test in-out parity for barrier options."""
    
    def test_up_in_out_call_parity(self):
        """Up-in + up-out call should equal vanilla call."""
        spot, strike, barrier = 100.0, 100.0, 120.0
        rate, vol, ttm = 0.05, 0.20, 1.0
        
        vanilla = Call(strike=strike, maturity=ttm)
        up_out = UpOutCall(strike=strike, barrier=barrier, maturity=ttm, rebate=0.0)
        up_in = UpInCall(strike=strike, barrier=barrier, maturity=ttm, rebate=0.0)
        
        vanilla_price = vanilla.price(spot, rate, vol, ttm)
        up_out_price = up_out.price(spot, rate, vol, ttm)
        up_in_price = up_in.price(spot, rate, vol, ttm)
        
        assert up_out_price + up_in_price == pytest.approx(vanilla_price, rel=0.01)
    
    def test_down_in_out_call_parity(self):
        """Down-in + down-out call should equal vanilla call."""
        spot, strike, barrier = 100.0, 100.0, 80.0
        rate, vol, ttm = 0.05, 0.20, 1.0
        
        vanilla = Call(strike=strike, maturity=ttm)
        down_out = DownOutCall(strike=strike, barrier=barrier, maturity=ttm, rebate=0.0)
        down_in = DownInCall(strike=strike, barrier=barrier, maturity=ttm, rebate=0.0)
        
        vanilla_price = vanilla.price(spot, rate, vol, ttm)
        down_out_price = down_out.price(spot, rate, vol, ttm)
        down_in_price = down_in.price(spot, rate, vol, ttm)
        
        assert down_out_price + down_in_price == pytest.approx(vanilla_price, rel=0.01)
    
    def test_up_in_out_put_parity(self):
        """Up-in + up-out put should equal vanilla put."""
        spot, strike, barrier = 100.0, 100.0, 120.0
        rate, vol, ttm = 0.05, 0.20, 1.0
        
        vanilla = Put(strike=strike, maturity=ttm)
        up_out = UpOutPut(strike=strike, barrier=barrier, maturity=ttm, rebate=0.0)
        up_in = UpInPut(strike=strike, barrier=barrier, maturity=ttm, rebate=0.0)
        
        vanilla_price = vanilla.price(spot, rate, vol, ttm)
        up_out_price = up_out.price(spot, rate, vol, ttm)
        up_in_price = up_in.price(spot, rate, vol, ttm)
        
        assert up_out_price + up_in_price == pytest.approx(vanilla_price, rel=0.01)


class TestBarrierPayoffs:
    """Test barrier option payoffs."""
    
    def test_up_out_call_payoff_not_hit(self):
        """Up-out call payoff when barrier not hit."""
        option = UpOutCall(strike=100, barrier=120, maturity=1.0)
        
        assert option.payoff(spot=110, barrier_hit=False) == pytest.approx(10.0)
        assert option.payoff(spot=90, barrier_hit=False) == pytest.approx(0.0)
    
    def test_up_out_call_payoff_hit(self):
        """Up-out call payoff when barrier hit."""
        option = UpOutCall(strike=100, barrier=120, maturity=1.0, rebate=5.0)
        
        assert option.payoff(spot=130, barrier_hit=True) == pytest.approx(5.0)
    
    def test_down_out_put_payoff_not_hit(self):
        """Down-out put payoff when barrier not hit."""
        option = DownOutPut(strike=100, barrier=80, maturity=1.0)
        
        assert option.payoff(spot=90, barrier_hit=False) == pytest.approx(10.0)
        assert option.payoff(spot=110, barrier_hit=False) == pytest.approx(0.0)


class TestEdgeCases:
    """Test edge cases in barrier option pricing."""
    
    def test_zero_time_to_maturity(self):
        """Option at expiry should return intrinsic value."""
        option = UpOutCall(strike=100, barrier=120, maturity=1.0)
        
        price = option.price(spot=110, rate=0.05, volatility=0.20, time_to_maturity=0.0)
        
        assert price == pytest.approx(10.0, abs=0.01)
    
    def test_high_volatility(self):
        """High volatility should increase knock-out probability."""
        option = UpOutCall(strike=100, barrier=110, maturity=1.0)
        
        low_vol_price = option.price(spot=100, rate=0.05, volatility=0.10, time_to_maturity=1.0)
        high_vol_price = option.price(spot=100, rate=0.05, volatility=0.50, time_to_maturity=1.0)
        
        assert high_vol_price < low_vol_price
    
    def test_barrier_at_strike(self):
        """Barrier at strike level edge case."""
        option = UpOutCall(strike=100, barrier=100, maturity=1.0)
        price = option.price(spot=95, rate=0.05, volatility=0.20, time_to_maturity=1.0)
        
        assert price >= 0
