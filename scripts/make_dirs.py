from __future__ import annotations

"""Create required directories from config."""

import argparse
import sys

from pathlib import Path

from pathlib import Path as _P; sys.path.append(str(_P(__file__).resolve().parents[1]/"src"))
from core.config import load_config


def make_dirs(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    for p in [
        cfg.paths.data_dir,
        cfg.paths.raw_dir,
        cfg.paths.processed_dir,
        cfg.paths.artifacts_dir,
        cfg.paths.logs_dir,
        cfg.paths.reports_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    make_dirs(Path(args.config))
