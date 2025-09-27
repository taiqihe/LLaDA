"""Logging configuration for the LLaDA visualizer."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "llada_visualizer",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """Set up a logger with both console and file handlers."""

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name."""
    return logging.getLogger(name)


# Default logger for the visualizer
main_logger = setup_logger(
    name="llada_visualizer",
    level="INFO",
    log_file="logs/visualizer.log",
    console=True
)

# Component-specific loggers
diffusion_logger = setup_logger(
    name="llada_visualizer.diffusion",
    level="DEBUG",
    log_file="logs/diffusion.log",
    console=False
)

token_tracker_logger = setup_logger(
    name="llada_visualizer.token_tracker",
    level="DEBUG",
    log_file="logs/token_tracker.log",
    console=False
)

websocket_logger = setup_logger(
    name="llada_visualizer.websocket",
    level="INFO",
    log_file="logs/websocket.log",
    console=False
)

probability_logger = setup_logger(
    name="llada_visualizer.probability",
    level="DEBUG",
    log_file="logs/probability.log",
    console=False
)