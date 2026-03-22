from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from app.storage.paths import workspace

M = TypeVar("M", bound=BaseModel)


class NormalizedStore:
    """
    Persiste blocos normalizados (MarketEarBlock, XTimelineItem) em JSONL.
    Uma linha por objeto — permite stream processing e append incremental.
    """

    def _path(self, source: str, run_id: str) -> Path:
        p = workspace.normalized_blocks_path(source, run_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def append(self, source: str, run_id: str, item: BaseModel) -> None:
        path = self._path(source, run_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(item.model_dump_json() + "\n")

    def write_all(self, source: str, run_id: str, items: list[BaseModel]) -> Path:
        path = self._path(source, run_id)
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(item.model_dump_json() + "\n")
        return path

    def read_all(self, source: str, run_id: str, model_class: type[M]) -> list[M]:
        path = workspace.normalized_blocks_path(source, run_id)
        if not path.exists():
            return []
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                items.append(model_class.model_validate_json(line))
        return items


normalized_store = NormalizedStore()
