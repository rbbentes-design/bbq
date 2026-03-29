"""
Provider: Polymarket Prediction Markets

Coleta mercados macro relevantes do Polymarket via API pública.
Usa o endpoint /events (agrupa mercados por tema) com filtro por volume e keywords.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.polymarket")

SOURCE_NAME = "polymarket"

_EVENTS_URL = (
    "https://gamma-api.polymarket.com/events"
    "?active=true&limit=100&order=volume&ascending=false"
)

# Volume mínimo do evento para ser considerado relevante (USD)
_MIN_EVENT_VOLUME = 5_000_000  # 5M USD — mercados com apostas reais

# Keywords macro — temas que movem mercado
_MACRO_KEYWORDS = [
    "fed ", "federal reserve", "rate cut", "rate hike", "interest rate", "fomc",
    "inflation", "cpi", "pce", "recession", "gdp", "unemployment", "payroll",
    "iran", "hormuz", "war ", "ceasefire", "sanctions", "ukraine", "russia",
    "china ", "taiwan", "tariff", "trade war",
    "oil ", "crude ", "opec", "natural gas",
    "gold ", "dollar", "treasury", "debt ceiling", "default",
    "s&p", "stock market", "market crash", "bear market",
    "bitcoin etf", "crypto regulation",
    "trump", "powell", "fed chair",
]

# Blocklist — exclui ruído independente de keyword
_BLOCKLIST = [
    "nba ", "nfl ", "nhl ", "mlb ", "soccer", " cup", "champion",
    "super bowl", "playoff", "match", " vs ", "beats ", "wins ",
    "oscar", "grammy", "emmy", "celebrity", "movie", "film",
    "counter-strike", "esport", "valorant",
    "mayoral", "governor", "senate seat", "house seat",
    "primary ", "nominee 20",  # eleições sem impacto macro imediato
]


def collect(max_results: int = 12) -> list[dict[str, Any]]:
    """
    Retorna top eventos macro do Polymarket com probabilidades ativas.

    Retorna lista de dicts com:
      - title: título do evento
      - question: pergunta do mercado mais representativo
      - probability: probabilidade do "Yes" (0-1)
      - volume_usd: volume total do evento em USD
      - end_date: data de resolução
    """
    try:
        req = urllib.request.Request(
            _EVENTS_URL,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        _log.warning("polymarket_fetch_error", error=str(exc))
        return []

    if not isinstance(data, list):
        data = data.get("events", data.get("results", []))

    results: list[dict[str, Any]] = []

    for event in data:
        title = (event.get("title") or "").lower()
        vol = float(event.get("volume") or 0)

        # Filtros básicos
        if vol < _MIN_EVENT_VOLUME:
            continue
        if any(bl in title for bl in _BLOCKLIST):
            continue
        if not any(kw in title for kw in _MACRO_KEYWORDS):
            continue

        # Pega o mercado mais representativo do evento (maior volume com prob ativa)
        markets = event.get("markets") or []
        best_market = _best_market(markets)
        if not best_market:
            continue

        results.append({
            "title": event.get("title", ""),
            "question": best_market["question"],
            "probability": best_market["prob"],
            "volume_usd": round(vol, 0),
            "end_date": best_market.get("end_date", event.get("endDate", "")),
        })

        if len(results) >= max_results:
            break

    _log.info("polymarket_done", events_checked=len(data), markets=len(results))
    return results


def _best_market(markets: list[dict]) -> dict | None:
    """Seleciona o mercado mais informativo do evento."""
    candidates = []
    for m in markets:
        if not m.get("active", True):
            continue
        try:
            prices = m.get("outcomePrices") or []
            if isinstance(prices, str):
                prices = json.loads(prices)
            prob = float(prices[0]) if prices else 0.0
        except (TypeError, ValueError, IndexError):
            continue

        # Só mercados com probabilidade não trivial (não resolvidos)
        if prob < 0.02 or prob > 0.98:
            continue

        vol = float(m.get("volume") or m.get("volumeNum") or 0)
        candidates.append({
            "question": m.get("question", ""),
            "prob": round(prob, 3),
            "volume": vol,
            "end_date": m.get("endDate", ""),
        })

    if not candidates:
        return None

    # Prefere o com maior volume
    candidates.sort(key=lambda x: x["volume"], reverse=True)
    return candidates[0]
