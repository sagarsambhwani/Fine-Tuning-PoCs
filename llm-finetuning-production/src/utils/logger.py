"""
Structured Logging Utility
"""
import logging
import sys

def setup_logger(name: str = "llm_pipeline", level: int = logging.INFO) -> logging.Logger:
    """Configures a clean stream logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
