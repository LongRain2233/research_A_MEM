"""
Centralized logging configuration for PhaseForget-Zettel.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
) -> logging.Logger:
    """
    Configure the root logger for the PhaseForget system.

    Args:
        level:    Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for persistent log output.

    Returns:
        The configured root logger.
    """
    root_logger = logging.getLogger("phaseforget")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Guard: do not add duplicate handlers on repeated calls (e.g. tests)
    if root_logger.handlers:
        return root_logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
