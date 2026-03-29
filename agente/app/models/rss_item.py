from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.utils.timestamps import new_ulid


class RSSItem(BaseModel):
    id: str = Field(default_factory=new_ulid)
    source_name: str
    feed_url: str
    title: str
    summary: str = ""
    url: str = ""
    published_at: datetime | None = None
    author: str = ""
    tags: list[str] = Field(default_factory=list)
