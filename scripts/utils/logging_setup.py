"""Setup logging with rich handler."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(log_dir: str = "./artifacts/logs") -> Path:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = log_path / f"{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            RichHandler(rich_tracebacks=True),
            logging.FileHandler(file_path)
        ]
    )
    return file_path
