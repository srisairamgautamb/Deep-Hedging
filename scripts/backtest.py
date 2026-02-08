#!/usr/bin/env python
"""Backtesting script for trained hedging agents."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.options.barrier import UpOutCall, DownOutCall, UpOutPut, DownOutPut
from src.hedging.agents import DQNAgent, PPOAgent
from src.backtesting.engine import BacktestEngine, run_monte_carlo_backtest
from src.backtesting.transaction import ProportionalCost
from src.data.loader import CMEDataLoader, SyntheticDataGenerator
from src.utils.config import load_config


OPTION_CLASSES = {
    "up_out_call": UpOutCall,
    "down_out_call": DownOutCall,
    "up_out_put": UpOutPut,
    "down_out_put": DownOutPut,
}


def main():
    parser = argparse.ArgumentParser(description="Backtest trained hedging agent")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Agent checkpoint path")
    parser.add_argument("--data", type=str, default=None, help="Price data file (parquet)")
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo backtest")
    parser.add_argument("--paths", type=int, default=1000, help="Monte Carlo paths")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    option_cls = OPTION_CLASSES[config.option.option_type]
    option = option_cls(
        strike=config.option.strike,
        barrier=config.option.barrier,
        maturity=config.option.maturity,
    )
    
    state_dim = 7
    if config.hedging.agent_type == "ppo":
        agent = PPOAgent(state_dim=state_dim, action_dim=1)
    else:
        agent = DQNAgent(state_dim=state_dim, num_actions=21)
    
    agent.load(args.checkpoint)
    print(f"Loaded agent from {args.checkpoint}")
    
    cost_model = ProportionalCost(
        commission_rate=config.environment.cost_multiplier,
        slippage_bps=config.environment.slippage_bps,
    )
    
    if args.monte_carlo:
        print(f"Running Monte Carlo backtest with {args.paths} paths...")
        results = run_monte_carlo_backtest(
            agent=agent,
            option=option,
            initial_spot=config.option.strike,
            volatility=config.environment.volatility,
            rate=config.environment.risk_free_rate,
            num_paths=args.paths,
            cost_model=cost_model,
        )
        
        print("\nMonte Carlo Results:")
        print(f"  Variance Reduction: {results['mean_variance_reduction']:.2%} (+/- {results['std_variance_reduction']:.2%})")
        print(f"  Cost Savings:       {results['mean_cost_savings']:.2%} (+/- {results['std_cost_savings']:.2%})")
        print(f"  Sharpe Ratio:       {results['mean_sharpe']:.4f} (+/- {results['std_sharpe']:.4f})")
        
    else:
        if args.data:
            loader = CMEDataLoader(args.data)
            prices = loader.get_price_series()
        else:
            print("No data file provided, using synthetic prices...")
            generator = SyntheticDataGenerator(
                initial_price=config.option.strike,
                volatility=config.environment.volatility,
                drift=config.environment.risk_free_rate,
                seed=42,
            )
            prices = generator.generate_gbm_path(
                num_steps=int(config.option.maturity * 252),
                start_date=config.data.test_start,
            )
        
        engine = BacktestEngine(
            agent=agent,
            option=option,
            cost_model=cost_model,
            initial_capital=config.backtest.initial_capital,
            risk_free_rate=config.environment.risk_free_rate,
            volatility=config.environment.volatility,
        )
        
        print("Running backtest...")
        result = engine.run(prices)
        
        print(result.summary())
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result.to_dataframe().to_csv(output_dir / "backtest_results.csv", index=False)
        print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
