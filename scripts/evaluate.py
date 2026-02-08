#!/usr/bin/env python
"""Evaluation script for hedging performance analysis."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.options.barrier import UpOutCall
from src.hedging.environment import HedgingEnvironment
from src.hedging.agents import PPOAgent, DQNAgent
from src.backtesting.engine import BacktestEngine
from src.backtesting.transaction import ProportionalCost
from src.data.loader import SyntheticDataGenerator
from src.utils.config import load_config


def evaluate_agent(agent, env, num_episodes=100):
    """Evaluate agent over multiple episodes."""
    rewards = []
    pnl_variances = []
    total_costs = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if hasattr(agent, "select_action"):
                action, _, _ = agent.select_action(state, training=False)
            else:
                action = np.array([0.0])
            
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
        pnl_variances.append(info.get("pnl_variance", 0))
        total_costs.append(info.get("total_cost", 0))
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_pnl_variance": np.mean(pnl_variances),
        "mean_total_cost": np.mean(total_costs),
    }


def plot_results(backtest_result, output_path):
    """Generate performance visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(backtest_result.timestamps, backtest_result.equity_curve)
    ax1.set_title("Portfolio Equity Curve")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value")
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(backtest_result.timestamps, backtest_result.hedge_positions)
    ax2.set_title("Hedge Position Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Position Size")
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.hist(backtest_result.pnl_series, bins=50, edgecolor="black", alpha=0.7)
    ax3.axvline(x=0, color="red", linestyle="--")
    ax3.set_title("P&L Distribution")
    ax3.set_xlabel("P&L")
    ax3.set_ylabel("Frequency")
    
    ax4 = axes[1, 1]
    equity = backtest_result.equity_curve
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max
    ax4.fill_between(backtest_result.timestamps, drawdown, alpha=0.7)
    ax4.set_title("Drawdown")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Drawdown %")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate hedging agent")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="evaluation")
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    option = UpOutCall(
        strike=config.option.strike,
        barrier=config.option.barrier,
        maturity=config.option.maturity,
    )
    
    env = HedgingEnvironment(
        option=option,
        initial_spot=config.option.strike,
        volatility=config.environment.volatility,
        rate=config.environment.risk_free_rate,
        cost_multiplier=config.environment.cost_multiplier,
        slippage_bps=config.environment.slippage_bps,
        max_position=config.environment.max_position,
    )
    
    state_dim = env.observation_space.shape[0]
    if config.hedging.agent_type == "ppo":
        agent = PPOAgent(state_dim=state_dim, action_dim=1)
    else:
        agent = DQNAgent(state_dim=state_dim, num_actions=21)
    
    agent.load(args.checkpoint)
    
    print(f"Evaluating agent over {args.episodes} episodes...")
    eval_results = evaluate_agent(agent, env, args.episodes)
    
    print("\nEvaluation Results:")
    print(f"  Mean Reward:       {eval_results['mean_reward']:.4f} (+/- {eval_results['std_reward']:.4f})")
    print(f"  Mean P&L Variance: {eval_results['mean_pnl_variance']:.6f}")
    print(f"  Mean Total Cost:   {eval_results['mean_total_cost']:.4f}")
    
    generator = SyntheticDataGenerator(
        initial_price=config.option.strike,
        volatility=config.environment.volatility,
        seed=42,
    )
    prices = generator.generate_gbm_path(int(config.option.maturity * 252))
    
    engine = BacktestEngine(
        agent=agent,
        option=option,
        cost_model=ProportionalCost(commission_rate=config.environment.cost_multiplier),
    )
    result = engine.run(prices, progress=False)
    
    plot_path = output_dir / "performance_plots.png"
    plot_results(result, plot_path)
    print(f"\nPlots saved to {plot_path}")
    
    report = {
        "evaluation": eval_results,
        "backtest_metrics": result.metrics.to_dict(),
        "variance_reduction": result.variance_reduction,
        "cost_savings": result.cost_savings,
    }
    
    pd.DataFrame([report]).to_json(output_dir / "evaluation_report.json", orient="records", indent=2)
    print(f"Report saved to {output_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
