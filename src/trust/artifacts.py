"""Tiny JSON artifact store so Airflow tasks exchange results as files.

Tasks pass artifact *names* through XCom and read/write the payloads here,
keeping XCom small and making every intermediate result inspectable on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config


def save(name: str, obj) -> str:
    path = config.ARTIFACTS_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, default=str)
    return str(path)


def load(name: str):
    path = config.ARTIFACTS_DIR / f"{name}.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
