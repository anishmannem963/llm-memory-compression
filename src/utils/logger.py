import logging
import os
from pathlib import Path

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a named logger with consistent formatting."""
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_dir / "system.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
