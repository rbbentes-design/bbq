"""
Provider: Web Search (DuckDuckGo)

Busca web direcionada ao tema curado, para capturar notícias breaking
que o RSS pode ter perdido.

Requer: pip install duckduckgo-search
"""

from __future__ import annotations

from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.web_search")

SOURCE_NAME = "web_search"


def search(
    query: str,
    max_results: int = 8,
    region: str = "us-en",
    timelimit: str = "d",  # "d"=last day, "w"=last week, "m"=last month
) -> list[dict[str, Any]]:
    """
    Executa busca DuckDuckGo e retorna resultados relevantes.

    Args:
        query:      Termo de busca (ex: narrativa primária da curadoria).
        max_results: Número máximo de resultados.
        region:     Região da busca.
        timelimit:  Janela temporal ("d", "w", "m" ou None).

    Returns:
        Lista de dicts com: title, href, body
    """
    try:
        try:
            from ddgs import DDGS  # novo nome do pacote (>=8.x renomeado para ddgs)
        except ImportError:
            from duckduckgo_search import DDGS  # fallback para versão antiga
    except ImportError:
        _log.warning("ddgs_not_installed", hint="pip install ddgs")
        return []

    results: list[dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                region=region,
                timelimit=timelimit,
                max_results=max_results,
            ):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source": SOURCE_NAME,
                })
    except Exception as exc:
        _log.warning("web_search_error", query=query[:60], error=str(exc))

    _log.info("web_search_done", query=query[:60], results=len(results))
    return results


def format_for_context(results: list[dict[str, Any]]) -> str:
    """Formata resultados como bloco de texto para inclusão em prompt."""
    if not results:
        return ""
    lines = ["=== BUSCA WEB COMPLEMENTAR ==="]
    for r in results:
        lines.append(f"\n[{r['title']}]")
        if r.get("snippet"):
            lines.append(r["snippet"][:300])
        if r.get("url"):
            lines.append(f"URL: {r['url']}")
    return "\n".join(lines)
