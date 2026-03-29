from __future__ import annotations

import httpx

from app.audit.logger import get_logger
from app.config.settings import settings

_log = get_logger("curation.llm_client")

_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class CurationLLMError(Exception):
    pass


def call_claude(
    prompt_system: str,
    prompt_user: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    key = settings.resolved_anthropic_api_key
    if not key:
        raise CurationLLMError("ANTHROPIC_API_KEY nao configurada")

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": prompt_system,
        "messages": [{"role": "user", "content": prompt_user}],
    }

    _log.debug("llm_call", model=model, prompt_chars=len(prompt_user))

    try:
        resp = httpx.post(
            _API_URL,
            json=payload,
            headers={
                "x-api-key": key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            timeout=300,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise CurationLLMError(f"API error {e.response.status_code}: {e.response.text[:300]}") from e
    except httpx.RequestError as e:
        raise CurationLLMError(f"Request error: {e}") from e

    data = resp.json()
    text = data["content"][0]["text"]
    usage = data.get("usage", {})
    _log.info("llm_done", model=model, input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
    return text
