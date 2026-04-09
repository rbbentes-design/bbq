"""
Provider: Market Prices — Bloomberg primário, fallback em camadas

Hierarquia de fontes:
  1. Bloomberg DB  (query_layer — fonte oficial)
  2. IBKR snapshot (ib_insync)
  3. Alpha Vantage (ALPHA_VANTAGE_API_KEY)
  4. Twelve Data   (TWELVE_DATA_API_KEY)
  5. Finnhub       (FINNHUB_API_KEY)

Se o banco Bloomberg estiver vazio ou desatualizado, as camadas seguintes
preenchem os tickers em falta. Retorna {} apenas se todas as camadas falharem.

Para atualizar o banco Bloomberg:
    python -m core.bloomberg_main_agent
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.audit.logger import get_logger

_log = get_logger("providers.market_prices")

SOURCE_NAME = "market_prices"


def collect(
    tickers: dict[str, str] | None = None,
    period: str = "5d",
    include_ytd: bool = True,
) -> dict[str, Any]:
    """
    Retorna preços e retornos dos ativos diretamente do banco Bloomberg.

    Args:
        tickers:     Ignorado — o banco contém todos os tickers Bloomberg disponíveis.
        period:      Ignorado — os retornos são calculados a partir das referências no banco.
        include_ytd: Ignorado — o banco já armazena price_ytd quando disponível.

    Returns:
        {
          "^GSPC": {
            "name":          "S&P 500",
            "price":         5100.23,
            "daily_return":  -0.0122,
            "weekly_return": -0.034,
            "ytd_return":    -0.085,
            "source":        "bloomberg",
          },
          ...
        }
        Retorna {} se o banco não existir ou não tiver dados.

    Aviso:
        Se o banco estiver vazio ou desatualizado, imprime aviso operacional.
        NÃO tenta buscar dados de fontes alternativas.
    """
    try:
        from app.query_layer import BloombergQueryLayer
        ql = BloombergQueryLayer()

        # Carrega do banco independente de idade — banco é fonte de verdade
        if not ql.has_any_data():
            status = ql.get_last_ingestion_status()
            if not status:
                _log.warning(
                    "banco_bloomberg_vazio",
                    hint="Execute o Bloomberg Agent antes de usar o MacroDesk.",
                    action="python -m core.bloomberg_main_agent",
                )
            else:
                _log.warning(
                    "bloomberg_banco_sem_dados",
                    hint="Banco existe mas sem precos. Execute o Bloomberg Agent.",
                )
            return {}

        # Aviso não-bloqueante se dados desatualizados
        if not ql.is_data_available():
            snap = ql.get_snapshot_info()
            _log.info(
                "bloomberg_usando_snapshot",
                age_minutes=snap.get("age_minutes"),
                tickers=snap.get("tickers_count"),
                hint="Usando ultimo snapshot valido do banco.",
            )

        prices = ql.get_latest_prices()

        if prices:
            _log.info("market_prices_loaded", tickers=len(prices), source="bloomberg")
            # Enriquecimento yfinance DESABILITADO por padrão (era 5-10s sempre).
            # Se Bloomberg retornou daily_return=0/None pra alguns tickers,
            # eles ficam None mesmo. Set USE_FALLBACKS=1 para reativar.
            import os as _os
            if _os.environ.get("USE_FALLBACKS"):
                try:
                    import yfinance as yf
                    _need_ret = [t for t, d in prices.items() if not d.get("daily_return")]
                    if _need_ret:
                        _yf_data = yf.download(_need_ret, period="2d", progress=False, auto_adjust=True)
                        _close = _yf_data["Close"] if "Close" in _yf_data.columns else _yf_data
                        if hasattr(_close, "columns"):
                            for t in _need_ret:
                                if t in _close.columns:
                                    _s = _close[t].dropna()
                                    if len(_s) >= 2:
                                        _ret = (_s.iloc[-1] - _s.iloc[-2]) / _s.iloc[-2]
                                        prices[t]["daily_return"] = round(float(_ret), 6)
                                        prices[t]["price"] = round(float(_s.iloc[-1]), 4)
                    _log.info("market_prices_enriched_yf", enriched=len(_need_ret))
                except Exception as _exc_yf:
                    _log.debug("market_prices_yf_enrich_failed", error=str(_exc_yf))
            return prices

        _log.warning(
            "bloomberg_no_prices",
            hint="Banco existe mas não tem dados de preço. Tentando fallback.",
        )

    except FileNotFoundError as exc:
        _log.warning("banco_nao_encontrado", error=str(exc))
    except Exception as exc:
        _log.warning("market_prices_bbg_error", error=str(exc))

    # ── Fallback: IBKR → Alpha Vantage → Twelve Data → Finnhub ───────────────
    try:
        from app.providers.market_data_chain import collect_prices
        prices = collect_prices()
        if prices:
            _log.info("market_prices_from_chain", tickers=len(prices))
            return prices
    except Exception as exc:
        _log.warning("market_prices_chain_error", error=str(exc))

    return {}


def format_summary(prices: dict[str, Any]) -> str:
    """Formata retorno legível para inclusão no contexto do LLM."""
    if not prices:
        return (
            "=== PREÇOS BLOOMBERG ===\n"
            "  [Sem dados] Execute o Bloomberg Agent para atualizar o banco.\n"
        )
    lines = ["=== PREÇOS E RETORNOS DE MERCADO (Bloomberg) ==="]
    for sym, d in prices.items():
        name  = d.get("name", sym)
        price = d.get("price", "N/A")
        d1    = d.get("daily_return")
        w1    = d.get("weekly_return")
        ytd   = d.get("ytd_return")

        d1_str  = f"{d1:+.1%}"      if d1  is not None else ""
        w1_str  = f"{w1:+.1%}"      if w1  is not None else ""
        ytd_str = f"YTD {ytd:+.1%}" if ytd is not None else ""

        parts = [p for p in [d1_str, w1_str, ytd_str] if p]
        lines.append(f"  {name}: ${price}  {' | '.join(parts)}")
    return "\n".join(lines)
