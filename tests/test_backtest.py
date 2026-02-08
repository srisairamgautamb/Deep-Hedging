"""Unit tests for backtesting engine."""

import pytest
import numpy as np
import pandas as pd

from src.options.barrier import UpOutCall
from src.backtesting.engine import BacktestEngine, run_monte_carlo_backtest
from src.backtesting.metrics import compute_sharpe, compute_max_drawdown, compute_all_metrics
from src.backtesting.transaction import ProportionalCost, TieredCost


class TestPerformanceMetrics:
    """Test performance metric calculations."""
    
    def test_sharpe_ratio_positive(self):
        """Sharpe should be positive for positive excess returns."""
        returns = np.array([0.01, 0.02, 0.01, 0.03, 0.01])
        sharpe = compute_sharpe(returns, risk_free_rate=0.0)
        
        assert sharpe > 0
    
    def test_sharpe_ratio_negative(self):
        """Sharpe should be negative for negative excess returns."""
        returns = np.array([-0.01, -0.02, -0.01, -0.03, -0.01])
        sharpe = compute_sharpe(returns, risk_free_rate=0.0)
        
        assert sharpe < 0
    
    def test_sharpe_ratio_zero_volatility(self):
        """Sharpe should be zero when volatility is zero."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = compute_sharpe(returns, risk_free_rate=0.0)
        
        assert sharpe == 0.0
    
    def test_max_drawdown(self):
        """Max drawdown should capture peak-to-trough decline."""
        equity = np.array([100, 110, 105, 120, 90, 100])
        mdd = compute_max_drawdown(equity)
        
        assert mdd == pytest.approx(0.25, rel=0.01)
    
    def test_max_drawdown_no_decline(self):
        """Max drawdown should be zero for monotonic increase."""
        equity = np.array([100, 110, 120, 130, 140])
        mdd = compute_max_drawdown(equity)
        
        assert mdd == 0.0
    
    def test_compute_all_metrics(self):
        """All metrics should compute without error."""
        returns = np.random.randn(252) * 0.02
        equity = 1000 * np.cumprod(1 + returns)
        pnl = np.diff(equity)
        
        metrics = compute_all_metrics(
            returns=returns,
            equity_curve=equity,
            pnl_series=pnl,
            transaction_costs=100.0,
            num_trades=50,
        )
        
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "max_drawdown")
        assert hasattr(metrics, "pnl_variance")


class TestTransactionCosts:
    """Test transaction cost models."""
    
    def test_proportional_cost(self):
        """Proportional cost should scale with notional."""
        model = ProportionalCost(commission_rate=0.001)
        
        cost1 = model.compute_cost(trade_size=100, price=10)
        cost2 = model.compute_cost(trade_size=200, price=10)
        
        assert cost2 == pytest.approx(2 * cost1, rel=0.01)
    
    def test_tiered_cost(self):
        """Tiered cost should decrease with volume."""
        model = TieredCost()
        
        small_trade_rate = model.compute_cost(100, 10) / (100 * 10)
        large_trade_rate = model.compute_cost(100000, 10) / (100000 * 10)
        
        assert large_trade_rate < small_trade_rate
    
    def test_slippage_positive(self):
        """Slippage should always be positive."""
        model = ProportionalCost()
        
        slippage = model.compute_slippage(trade_size=100, price=10)
        
        assert slippage > 0


class TestBacktestEngine:
    """Test backtesting engine."""
    
    @pytest.fixture
    def option(self):
        return UpOutCall(strike=100, barrier=120, maturity=0.25)
    
    @pytest.fixture
    def prices(self):
        np.random.seed(42)
        returns = np.random.randn(63) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))
        dates = pd.date_range(start="2024-01-01", periods=len(prices), freq="D")
        return pd.Series(prices, index=dates)
    
    def test_backtest_runs(self, option, prices):
        """Backtest should complete without error."""
        
        class DummyAgent:
            def select_action(self, state, training=False):
                return np.array([0.0]), 0.0, 0.0
        
        engine = BacktestEngine(
            agent=DummyAgent(),
            option=option,
            risk_free_rate=0.05,
            volatility=0.20,
        )
        
        result = engine.run(prices, progress=False)
        
        assert len(result.equity_curve) == len(prices)
        assert result.metrics is not None
    
    def test_backtest_result_structure(self, option, prices):
        """Backtest result should have expected fields."""
        
        class DummyAgent:
            def select_action(self, state, training=False):
                return np.array([0.0]), 0.0, 0.0
        
        engine = BacktestEngine(agent=DummyAgent(), option=option)
        result = engine.run(prices, progress=False)
        
        assert hasattr(result, "metrics")
        assert hasattr(result, "equity_curve")
        assert hasattr(result, "trades")
        assert hasattr(result, "variance_reduction")
        assert hasattr(result, "cost_savings")
    
    def test_backtest_trades_recorded(self, option, prices):
        """Trades should be recorded when agent changes position."""
        
        class TradingAgent:
            def __init__(self):
                self.step = 0
            
            def select_action(self, state, training=False):
                self.step += 1
                if self.step % 10 == 0:
                    return np.array([0.5]), 0.0, 0.0
                return np.array([-0.5]), 0.0, 0.0
        
        engine = BacktestEngine(agent=TradingAgent(), option=option)
        result = engine.run(prices, progress=False)
        
        assert len(result.trades) > 0
    
    def test_backtest_to_dataframe(self, option, prices):
        """Results should convert to DataFrame."""
        
        class DummyAgent:
            def select_action(self, state, training=False):
                return np.array([0.0]), 0.0, 0.0
        
        engine = BacktestEngine(agent=DummyAgent(), option=option)
        result = engine.run(prices, progress=False)
        
        df = result.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert "equity" in df.columns
        assert "pnl" in df.columns
