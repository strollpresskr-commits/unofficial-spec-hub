"""Rich-based logging setup shared across pipeline steps."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def get_logger(name: str, log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    rich_handler = RichHandler(console=console, show_path=False, markup=True)
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
