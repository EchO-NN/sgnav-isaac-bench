from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


class JsonlEpisodeLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: Mapping) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

