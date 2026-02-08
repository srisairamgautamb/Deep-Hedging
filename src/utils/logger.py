"""Logging utilities."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


_loggers = {}


def setup_logger(
    name: str = "deep_hedging",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """Set up and return a configured logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        console: Enable console output
        file: Enable file output
        
    Returns:
        Configured logger
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []
    
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"{name}_{timestamp}.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = "deep_hedging") -> logging.Logger:
    """Get existing logger or create default one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


class TrainingLogger:
    """Specialized logger for training progress."""
    
    def __init__(
        self,
        name: str = "training",
        log_dir: Optional[str] = None,
        tensorboard: bool = False,
    ):
        """Initialize training logger.
        
        Args:
            name: Logger name
            log_dir: Directory for logs
            tensorboard: Enable TensorBoard logging
        """
        self.logger = setup_logger(name, log_dir=log_dir)
        self.tensorboard = tensorboard
        self.writer = None
        
        if tensorboard and log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                self.logger.warning("TensorBoard not available")
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        loss: Optional[float] = None,
        **metrics
    ):
        """Log episode metrics.
        
        Args:
            episode: Episode number
            reward: Episode reward
            loss: Training loss
            **metrics: Additional metrics
        """
        msg = f"Episode {episode} | Reward: {reward:.4f}"
        if loss is not None:
            msg += f" | Loss: {loss:.4f}"
        
        for key, value in metrics.items():
            msg += f" | {key}: {value:.4f}"
        
        self.logger.info(msg)
        
        if self.writer:
            self.writer.add_scalar("reward", reward, episode)
            if loss is not None:
                self.writer.add_scalar("loss", loss, episode)
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, episode)
    
    def log_eval(
        self,
        step: int,
        sharpe: float,
        variance_reduction: float,
        cost_savings: float,
    ):
        """Log evaluation metrics.
        
        Args:
            step: Current step
            sharpe: Sharpe ratio
            variance_reduction: P&L variance reduction
            cost_savings: Transaction cost savings
        """
        self.logger.info(
            f"Eval @ {step} | Sharpe: {sharpe:.4f} | "
            f"Var Reduction: {variance_reduction:.2%} | "
            f"Cost Savings: {cost_savings:.2%}"
        )
        
        if self.writer:
            self.writer.add_scalar("eval/sharpe", sharpe, step)
            self.writer.add_scalar("eval/variance_reduction", variance_reduction, step)
            self.writer.add_scalar("eval/cost_savings", cost_savings, step)
    
    def close(self):
        """Close logger resources."""
        if self.writer:
            self.writer.close()
