"""
Provider: CheddarFlow — Unusual Options Flow

CheddarFlow.com agrega fluxo incomum de opções: sweeps, large blocks, golden sweeps.

Métricas coletadas:
  - Sweep orders (cruzam múltiplos exchanges — urgência)
  - Golden sweeps (sweep + premium > $1M)
  - Large blocks (size > 1000 contratos)
  - Sentiment por ativo: bullish/bearish baseado em call vs put flow
  - Premium total pago (proxy para conviction institucional)

Por que importa:
  - Sweeps indicam urgência — quem compra sweep aceita pagar mais para executar rápido
  - Golden sweep = institucional apostando > $1M num ativo específico
  - Fluxo concentrado em calls OTM de curto prazo = expectativa de move iminente

Autenticação: Playwright persistent context (login uma vez, reutiliza sessão).

Setup:
  python -m app.cli.auth login cheddarflow
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.audit.logger import get_logger
from app.config.settings import settings

_log = get_logger("providers.cheddarflow")

_BASE_URL    = "https://cheddarflow.com/flow"
_LOGIN_URL   = "https://cheddarflow.com/login"
_PROFILE_URL = "https://cheddarflow.com/flow"


@dataclass
class CheddarFlowOrder:
    """Uma ordem de fluxo incomum do CheddarFlow."""
    ticker:       str
    option_type:  str          # "call" | "put"
    strike:       float | None = None
    expiry:       str          = ""
    premium_usd:  float        = 0.0   # prêmio total pago
    contracts:    int          = 0
    sentiment:    str          = "neutral"   # "bullish" | "bearish"
    is_sweep:     bool         = False
    is_golden:    bool         = False
    timestamp:    str          = ""


@dataclass
class CheddarFlowResult:
    """Resultado agregado do CheddarFlow."""
    orders:        list[CheddarFlowOrder] = field(default_factory=list)
    by_ticker:     dict[str, dict]        = field(default_factory=dict)
    top_bullish:   list[str]              = field(default_factory=list)
    top_bearish:   list[str]              = field(default_factory=list)
    golden_sweeps: list[CheddarFlowOrder] = field(default_factory=list)
    total_call_premium_usd: float         = 0.0
    total_put_premium_usd:  float         = 0.0
    call_put_ratio:         float         = 0.0   # > 1 = bullish flow dominante
    timestamp:     str                    = ""
    source:        str                    = "cheddarflow"
    error:         str                    = ""


# ── Playwright scraper ────────────────────────────────────────────────────────

def _is_logged_in(page) -> bool:
    url = page.url.lower()
    return "login" not in url and "cheddarflow.com" in url


def _parse_flow_table(page) -> list[CheddarFlowOrder]:
    """
    Extrai ordens da tabela de fluxo do CheddarFlow.
    O site usa uma tabela com colunas: Time | Ticker | C/P | Strike | Expiry | Premium | Contracts | Type
    """
    orders = []
    try:
        # Tenta localizar tabela de fluxo
        rows = page.query_selector_all("table tr, [class*='flow-row'], [class*='order-row']")

        for row in rows[:200]:  # limita a 200 ordens recentes
            try:
                cells = row.query_selector_all("td")
                if len(cells) < 5:
                    continue

                texts = [c.inner_text().strip() for c in cells]

                # Parsing flexível — CheddarFlow pode mudar layout
                ticker = None
                option_type = "call"
                premium = 0.0
                contracts = 0
                is_sweep = False
                is_golden = False
                expiry = ""
                strike = None

                for i, t in enumerate(texts):
                    t_upper = t.upper()
                    # Ticker: 1-5 letras maiúsculas
                    if re.match(r'^[A-Z]{1,5}$', t_upper) and len(t_upper) <= 5 and ticker is None:
                        ticker = t_upper
                    # C/P
                    if t_upper in ('C', 'CALL', 'CALLS'):
                        option_type = "call"
                    elif t_upper in ('P', 'PUT', 'PUTS'):
                        option_type = "put"
                    # Premium (formato: $1.2M, $500K, $1,200,000)
                    pm = re.match(r'\$?([\d,]+\.?\d*)\s*([MKB]?)', t.replace(',', ''))
                    if pm and premium == 0:
                        val = float(pm.group(1))
                        mult = {'M': 1e6, 'K': 1e3, 'B': 1e9}.get(pm.group(2), 1)
                        if val > 0 and val * mult < 1e10:  # sanity check
                            premium = val * mult
                    # Contracts
                    if re.match(r'^\d{1,6}$', t) and contracts == 0:
                        c_val = int(t)
                        if 1 < c_val < 500_000:
                            contracts = c_val
                    # Sweep / Golden flags
                    if 'sweep' in t.lower():
                        is_sweep = True
                    if 'golden' in t.lower() or ('sweep' in t.lower() and premium >= 1_000_000):
                        is_golden = True
                    # Expiry (ex: 04/18/25, 2025-04-18)
                    if re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', t) or re.match(r'\d{4}-\d{2}-\d{2}', t):
                        expiry = t
                    # Strike
                    sm = re.match(r'^([\d]+\.?\d*)\s*[Cc]?$', t)
                    if sm and strike is None:
                        sv = float(sm.group(1))
                        if 1 < sv < 10_000:
                            strike = sv

                if ticker and premium > 0:
                    sentiment = "bullish" if option_type == "call" else "bearish"
                    orders.append(CheddarFlowOrder(
                        ticker=ticker, option_type=option_type, strike=strike,
                        expiry=expiry, premium_usd=premium, contracts=contracts,
                        sentiment=sentiment, is_sweep=is_sweep, is_golden=is_golden,
                        timestamp=datetime.now().isoformat()
                    ))
            except Exception:
                continue

    except Exception as exc:
        _log.debug("cheddarflow_parse_table_failed", error=str(exc)[:60])

    return orders


def _aggregate(orders: list[CheddarFlowOrder]) -> CheddarFlowResult:
    """Agrega ordens por ticker e calcula totais."""
    result = CheddarFlowResult(timestamp=datetime.now().isoformat())
    result.orders = orders

    ticker_data: dict[str, dict] = {}
    for o in orders:
        if o.ticker not in ticker_data:
            ticker_data[o.ticker] = {
                "call_premium": 0.0, "put_premium": 0.0,
                "n_sweeps": 0, "n_golden": 0,
                "n_call": 0, "n_put": 0,
            }
        td = ticker_data[o.ticker]
        if o.option_type == "call":
            td["call_premium"] += o.premium_usd
            td["n_call"] += 1
        else:
            td["put_premium"] += o.premium_usd
            td["n_put"] += 1
        if o.is_sweep:
            td["n_sweeps"] += 1
        if o.is_golden:
            td["n_golden"] += 1
            result.golden_sweeps.append(o)

        result.total_call_premium_usd += o.premium_usd if o.option_type == "call" else 0
        result.total_put_premium_usd  += o.premium_usd if o.option_type == "put"  else 0

    # Compute net score per ticker
    for tk, td in ticker_data.items():
        net_premium = td["call_premium"] - td["put_premium"]
        total       = td["call_premium"] + td["put_premium"]
        td["net_premium"] = net_premium
        td["total_premium"] = total
        td["sentiment"] = "bullish" if net_premium > 0 else ("bearish" if net_premium < 0 else "neutral")
        td["score"] = net_premium / total if total > 0 else 0.0

    result.by_ticker = ticker_data

    # Sort top bullish/bearish by net premium
    sorted_tickers = sorted(ticker_data.keys(),
                            key=lambda t: ticker_data[t]["net_premium"], reverse=True)
    result.top_bullish = [t for t in sorted_tickers if ticker_data[t]["net_premium"] > 0][:10]
    result.top_bearish = [t for t in reversed(sorted_tickers) if ticker_data[t]["net_premium"] < 0][:10]

    denom = result.total_put_premium_usd
    result.call_put_ratio = (result.total_call_premium_usd / denom) if denom > 0 else 0.0

    return result


def collect(page) -> CheddarFlowResult:
    """
    Coleta fluxo incomum de opções do CheddarFlow via Playwright.

    `page` deve ser uma Playwright Page em sessão autenticada.
    Retorna CheddarFlowResult com error preenchido se falhar.
    """
    result = CheddarFlowResult(timestamp=datetime.now().isoformat())

    try:
        page.goto(_PROFILE_URL, timeout=30_000)
        page.wait_for_load_state("domcontentloaded", timeout=15_000)

        if not _is_logged_in(page):
            _log.warning("cheddarflow_not_authenticated", url=page.url[:80])
            result.error = "Não autenticado — rode: python -m app.cli.auth login cheddarflow"
            return result

        try:
            page.wait_for_load_state("networkidle", timeout=20_000)
        except Exception:
            pass

        time.sleep(2)  # aguarda SPA renderizar

        orders = _parse_flow_table(page)
        result = _aggregate(orders)

        _log.info("cheddarflow_done",
                  n_orders=len(orders),
                  top_bullish=result.top_bullish[:3],
                  top_bearish=result.top_bearish[:3],
                  call_put_ratio=round(result.call_put_ratio, 2))

    except Exception as exc:
        result.error = str(exc)[:100]
        _log.warning("cheddarflow_failed", error=result.error)

    return result


def get_ticker_score(ticker: str, result: CheddarFlowResult) -> float:
    """
    Retorna score [-1, 1] de fluxo para um ticker.
    Positivo = call premium > put premium.
    """
    td = result.by_ticker.get(ticker)
    if td is None:
        return 0.0
    return float(td.get("score", 0.0))
