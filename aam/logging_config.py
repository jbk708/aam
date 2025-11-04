"""Logging configuration for AAM (Attention-based Abundance Model).

This module provides utilities for setting up structured logging throughout
the AAM codebase, including file and console handlers.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    output_dir: str,
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """Set up a logger with file and console handlers.
    
    Args:
        name: Logger name (typically __name__)
        output_dir: Directory where log file will be written
        log_level: Logging level (default: logging.INFO)
        log_to_file: Whether to log to file (default: True)
        log_to_console: Whether to log to console (default: True)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # File handler
    if log_to_file:
        log_file = os.path.join(output_dir, "fit_sample_regressor.log")
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def sanitize_path(path: Optional[str]) -> str:
    """Sanitize file paths for logging (mask sensitive information).
    
    Args:
        path: File path to sanitize
        
    Returns:
        Sanitized path string (returns "None" if path is None)
    """
    if path is None:
        return "None"
    # For now, just return the path as-is
    # In the future, could mask user directories or sensitive paths
    return str(path)
