from __future__ import annotations

"""Logging helpers using Rich and rotating files."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    handlers = [
        RichHandler(console=Console(), show_time=True),
        RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )
