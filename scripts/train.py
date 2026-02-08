#!/usr/bin/env python
"""Training script for deep hedging agents."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.options.barrier import UpOutCall, DownOutCall, UpOutPut, DownOutPut
from src.hedging.environment import HedgingEnvironment, DiscreteHedgingEnvironment
from src.hedging.agents import DQNAgent, PPOAgent
from src.utils.config import load_config
from src.utils.logger import TrainingLogger


OPTION_CLASSES = {
    "up_out_call": UpOutCall,
    "down_out_call": DownOutCall,
    "up_out_put": UpOutPut,
    "down_out_put": DownOutPut,
}


def create_option(config):
    """Create option from configuration."""
    option_cls = OPTION_CLASSES.get(config.option.option_type)
    if option_cls is None:
        raise ValueError(f"Unknown option type: {config.option.option_type}")
    
    return option_cls(
        strike=config.option.strike,
        barrier=config.option.barrier,
        maturity=config.option.maturity,
    )


def create_agent(config, state_dim):
    """Create agent from configuration."""
    if config.hedging.agent_type == "ppo":
        return PPOAgent(
            state_dim=state_dim,
            action_dim=1,
            hidden_dims=config.hedging.hidden_dims,
            learning_rate=config.hedging.learning_rate,
            gamma=config.hedging.gamma,
            gae_lambda=config.hedging.gae_lambda,
            clip_epsilon=config.hedging.clip_epsilon,
            entropy_coef=config.hedging.entropy_coef,
            value_coef=config.hedging.value_coef,
            max_grad_norm=config.hedging.max_grad_norm,
            update_epochs=config.hedging.update_epochs,
            batch_size=config.hedging.batch_size,
        )
    elif config.hedging.agent_type == "dqn":
        return DQNAgent(
            state_dim=state_dim,
            num_actions=21,
            hidden_dims=config.hedging.hidden_dims,
            learning_rate=config.hedging.learning_rate,
            gamma=config.hedging.gamma,
            batch_size=config.hedging.batch_size,
        )
    else:
        raise ValueError(f"Unknown agent type: {config.hedging.agent_type}")


def main():
    parser = argparse.ArgumentParser(description="Train deep hedging agent")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes")
    parser.add_argument("--output", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.episodes:
        config.hedging.episodes = args.episodes
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = TrainingLogger(
        name="deep_hedging",
        log_dir=config.logging.log_dir,
        tensorboard=config.logging.tensorboard,
    )
    
    option = create_option(config)
    
    if config.hedging.agent_type == "dqn":
        env = DiscreteHedgingEnvironment(
            option=option,
            num_actions=21,
            initial_spot=config.option.strike,
            volatility=config.environment.volatility,
            rate=config.environment.risk_free_rate,
            cost_multiplier=config.environment.cost_multiplier,
            slippage_bps=config.environment.slippage_bps,
            max_position=config.environment.max_position,
            seed=args.seed,
        )
    else:
        env = HedgingEnvironment(
            option=option,
            initial_spot=config.option.strike,
            volatility=config.environment.volatility,
            rate=config.environment.risk_free_rate,
            cost_multiplier=config.environment.cost_multiplier,
            slippage_bps=config.environment.slippage_bps,
            max_position=config.environment.max_position,
            seed=args.seed,
        )
    
    state_dim = env.observation_space.shape[0]
    agent = create_agent(config, state_dim)
    
    print(f"Training {config.hedging.agent_type.upper()} agent for {config.hedging.episodes} episodes")
    print(f"Option: {config.option.option_type} K={config.option.strike} H={config.option.barrier}")
    
    stats = agent.train(
        env=env,
        episodes=config.hedging.episodes,
        verbose=True,
    )
    
    checkpoint_path = output_dir / f"{config.hedging.agent_type}_hedging.pt"
    agent.save(str(checkpoint_path))
    print(f"Saved checkpoint to {checkpoint_path}")
    
    logger.close()
    
    return stats


if __name__ == "__main__":
    main()
