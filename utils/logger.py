from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger("brats_hypergraph_demo")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    if not any(type(handler) is logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing_paths = {
            Path(handler.baseFilename).resolve()
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        }
        if log_path not in existing_paths:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger
