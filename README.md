# Deep Hedging

A production-grade deep reinforcement learning framework for hedging exotic barrier options. Implements analytical pricing models for single-barrier options with continuous monitoring and trains RL agents to learn optimal hedging strategies that minimize transaction costs while managing portfolio risk.

## Overview

This system addresses the challenge of hedging path-dependent derivatives where traditional delta-hedging approaches are suboptimal due to gamma risk near barriers and discrete rebalancing constraints. The framework combines:

- Closed-form barrier option pricing using Reiner-Rubinstein formulas
- Greeks calculation for delta, gamma, vega, theta, and rho
- Gymnasium-compatible trading environments with realistic market dynamics
- PPO and DQN agents for learning optimal hedge ratios

## Features

### Option Pricing

- **Vanilla Options**: Black-Scholes pricing for European calls and puts
- **Barrier Options**: Full suite of single-barrier options
  - Up-and-out call/put (knock-out)
  - Down-and-out call/put (knock-out)
  - Up-and-in call/put (knock-in)
  - Down-and-in call/put (knock-in)
- **Greeks**: Analytical and numerical sensitivities for all option types
- **Dividend Support**: Continuous dividend yield modeling

### Hedging Environment

- **Gymnasium Interface**: Standard RL environment API
- **Market Simulation**: Geometric Brownian motion with configurable volatility
- **Transaction Costs**: Proportional costs and slippage modeling
- **Position Limits**: Configurable maximum hedge positions
- **Observation Space**: Spot price, volatility, time-to-maturity, Greeks, barrier status

### RL Agents

- **PPO Agent**: Proximal Policy Optimization with GAE advantage estimation
- **DQN Agent**: Deep Q-Network with experience replay and target networks
- **Continuous Actions**: Hedge ratio in [-1, 1] range
- **Discrete Actions**: Discretized hedge ratio for DQN compatibility

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy 1.21+
- SciPy 1.8+
- Gymnasium 0.28+
- Pandas 1.4+
- PyYAML 6.0+
- Matplotlib 3.5+

## Installation

```bash
git clone <repository-url>
cd deep-hedging
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Option Pricing

```python
from src.options.barrier import UpOutCall, DownOutPut
from src.options.greeks import GreeksCalculator

# Price an up-and-out call
option = UpOutCall(strike=100.0, barrier=120.0, maturity=1.0)
price = option.price(
    spot=100.0,
    rate=0.05,
    volatility=0.20,
    time_to_maturity=1.0
)

# Calculate Greeks
calculator = GreeksCalculator()
greeks = calculator.compute_all(option, spot=100.0, rate=0.05, volatility=0.20, ttm=1.0)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.6f}")
print(f"Vega: {greeks['vega']:.4f}")
```

### Training a Hedging Agent

```python
from src.options.barrier import UpOutCall
from src.hedging.environment import HedgingEnvironment
from src.hedging.agents.ppo import PPOHedgingAgent

# Configure option and environment
option = UpOutCall(strike=100.0, barrier=120.0, maturity=1.0)
env = HedgingEnvironment(
    option=option,
    initial_spot=100.0,
    volatility=0.20,
    rate=0.05,
    cost_multiplier=0.001
)

# Initialize and train agent
agent = PPOHedgingAgent(
    state_dim=env.observation_space.shape[0],
    learning_rate=0.0003,
    gamma=0.99
)
training_stats = agent.train(env, episodes=10000)
```

### Backtesting

```python
from scripts.backtest import run_backtest

results = run_backtest(
    agent=agent,
    option=option,
    num_paths=1000,
    initial_capital=1000000
)

print(f"Mean P&L: {results['mean_pnl']:.2f}")
print(f"P&L Std: {results['pnl_std']:.2f}")
print(f"Sharpe Ratio: {results['sharpe']:.4f}")
```

## Project Structure

```
deep-hedging/
├── configs/
│   └── default.yaml          # Default configuration parameters
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation and metrics
│   └── backtest.py           # Historical backtesting
├── src/
│   ├── backtesting/          # Backtesting engine and metrics
│   ├── data/                 # Data loading and preprocessing
│   ├── hedging/
│   │   ├── agents/           # PPO and DQN implementations
│   │   ├── environment.py    # Gymnasium hedging environment
│   │   └── policy.py         # Neural network policies
│   ├── options/
│   │   ├── barrier.py        # Barrier option pricing
│   │   ├── greeks.py         # Greeks calculation
│   │   └── vanilla.py        # Black-Scholes pricing
│   └── utils/                # Configuration and utilities
└── tests/
    ├── test_backtest.py      # Backtesting tests
    ├── test_barrier.py       # Barrier pricing tests
    └── test_hedging.py       # Environment tests
```

## Configuration

Configuration is managed via YAML files. Key parameters:

| Section | Parameter | Description | Default |
|---------|-----------|-------------|---------|
| hedging | agent_type | RL algorithm (ppo, dqn) | ppo |
| hedging | learning_rate | Optimizer learning rate | 0.0003 |
| hedging | gamma | Discount factor | 0.99 |
| hedging | clip_epsilon | PPO clipping parameter | 0.2 |
| environment | cost_multiplier | Transaction cost rate | 0.001 |
| environment | slippage_bps | Slippage in basis points | 1.0 |
| option | strike | Option strike price | 100.0 |
| option | barrier | Barrier level | 120.0 |
| option | maturity | Time to expiration (years) | 1.0 |

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## References

- Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. Quantitative Finance.
- Reiner, E., & Rubinstein, M. (1991). Breaking down the barriers. Risk Magazine.
- Hull, J. C. (2018). Options, Futures, and Other Derivatives. Pearson.

## License

MIT License
