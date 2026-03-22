from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    """Remove espaços múltiplos, tabs e quebras de linha redundantes."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text: str, max_chars: int = 500, suffix: str = "...") -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def word_count(text: str) -> int:
    return len(text.split())


def extract_urls(text: str) -> list[str]:
    pattern = r"https?://[^\s\"'>]+"
    return re.findall(pattern, text)


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:80]
