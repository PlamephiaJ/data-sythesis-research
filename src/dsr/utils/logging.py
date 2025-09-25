"""Logging and filesystem helpers."""

from __future__ import annotations

import logging
import os
from typing import Iterable


def makedirs(dirname: str) -> None:
    os.makedirs(dirname, exist_ok=True)


def get_logger(
    logpath: str,
    package_files: Iterable[str] | None = None,
    displaying: bool = True,
    saving: bool = True,
    debug: bool = False,
) -> logging.Logger:
    logger = logging.getLogger()
    level = logging.DEBUG if debug else logging.INFO

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    if saving:
        file_handler = logging.FileHandler(logpath, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    package_files = package_files or []
    for path in package_files:
        logger.info(path)
        with open(path, "r", encoding="utf-8") as package_file:
            logger.info(package_file.read())

    return logger
