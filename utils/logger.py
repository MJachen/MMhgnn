from __future__ import annotations

import logging


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("brats_hypergraph_demo")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger
