"""
Logging configuration for the Car Troubleshooting Chatbot API
"""

import os
import sys

from loguru import logger

from backend_app.core.settings import get_settings


def setup_logging():
    """Configure logging for the application."""
    settings = get_settings()

    # Clear default logger configuration
    logger.remove()

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(settings.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Console logging
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # File logging
    logger.add(
        settings.log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="zip",
    )

    # Error file logging
    error_log_file = settings.log_file.replace(".log", "_errors.log")
    logger.add(
        error_log_file,
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="zip",
    )

    logger.info("Logging configuration completed")


def get_logger(name: str):
    """Get a logger instance for a specific module."""
    return logger.bind(name=name)
