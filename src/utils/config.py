"""Configuration management."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class HedgingConfig:
    """Hedging agent configuration."""
    
    agent_type: str = "ppo"
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    episodes: int = 10000
    batch_size: int = 64
    update_epochs: int = 10


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    
    cost_multiplier: float = 0.001
    slippage_bps: float = 1.0
    tick_size: float = 0.01
    max_position: float = 100.0
    risk_free_rate: float = 0.05
    volatility: float = 0.20


@dataclass
class OptionConfig:
    """Option specification."""
    
    strike: float = 100.0
    barrier: float = 120.0
    maturity: float = 1.0
    option_type: str = "up_out_call"


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    
    initial_capital: float = 1000000.0
    position_limit: float = 1000.0
    margin_requirement: float = 0.10


@dataclass
class DataConfig:
    """Data configuration."""
    
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2024-06-30"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    log_dir: str = "logs"
    tensorboard: bool = True


@dataclass
class Config:
    """Complete configuration container."""
    
    hedging: HedgingConfig = field(default_factory=HedgingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    option: OptionConfig = field(default_factory=OptionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(
            hedging=HedgingConfig(**data.get("hedging", {})),
            environment=EnvironmentConfig(**data.get("environment", {})),
            option=OptionConfig(**data.get("option", {})),
            backtest=BacktestConfig(**data.get("backtest", {})),
            data=DataConfig(**data.get("data", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "hedging": self.hedging.__dict__,
            "environment": self.environment.__dict__,
            "option": self.option.__dict__,
            "backtest": self.backtest.__dict__,
            "data": self.data.__dict__,
            "logging": self.logging.__dict__,
        }
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(path: str) -> Config:
    """Load configuration from YAML file.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        Config object
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path) as f:
        data = yaml.safe_load(f)
    
    return Config.from_dict(data)
