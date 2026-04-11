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

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 移除旧的 FileHandler（多 trial 情况下切换到新日志文件）
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            root_logger.removeHandler(h)

    # 只在没有控制台 handler 时添加（避免重复输出）
    has_console = any(
        type(h) is logging.StreamHandler
        for h in root_logger.handlers
    )
    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler（每次 trial 都添加新的文件 handler）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
