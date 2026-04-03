"""
Provider: Market Prices — Bloomberg Only
=========================================

FONTE ÚNICA E OFICIAL: Banco SQLite macrodesk.db via query_layer.

Regra absoluta do ecossistema MacroDesk:
  - Apenas dados Bloomberg são usados
  - Sem Yahoo Finance
  - Sem Interactive Brokers
  - Sem fallback externo de nenhum tipo
  - Se o dado não está no banco, ele não existe

Se o banco estiver vazio ou desatualizado, retorna {} e loga aviso.
O MacroDesk deve verificar is_data_available() antes de renderizar dados.

Para atualizar o banco, execute o Bloomberg Agent:
    python -m core.bloomberg_main_agent
    ou
    launcher/run_bloomberg_agent.bat
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

        if not prices:
            _log.warning(
                "bloomberg_no_prices",
                hint="Banco existe mas não tem dados de preço. Verifique o export BQuant.",
            )
            return {}

        _log.info("market_prices_loaded", tickers=len(prices), source="bloomberg")
        return prices

    except FileNotFoundError as exc:
        _log.warning(
            "banco_nao_encontrado",
            error=str(exc),
            hint="Execute o Bloomberg Agent para criar o banco.",
        )
        return {}
    except Exception as exc:
        _log.error("market_prices_error", error=str(exc))
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
