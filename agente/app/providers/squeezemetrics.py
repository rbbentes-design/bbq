"""
Provider: SqueezeMetrics — DIX (Dark Index) + GEX por ativo

SqueezeMetrics publica dados derivados de dark pool e opcoes a partir de dados publicos da CBOE.

  DIX (Dark Index): percentual do volume off-exchange para o SPX
    - DIX > 45% = institucional comprando no dark pool → bullish para SPX
    - DIX < 40% = distribuição → cautela

  GEX (Gamma Exposure): exposição gamma agregada dos dealers do SPX
    - GEX positivo = dealers long gamma → amortece moves (low vol)
    - GEX negativo = dealers short gamma → amplifica moves (high vol)

Fonte: https://squeezemetrics.com/monitor/dix
Autenticação: Playwright persistent context (login uma vez, reutiliza sessão).

Setup:
  python -m app.cli.auth login squeezemetrics
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.audit.logger import get_logger
from app.config.settings import settings

_log = get_logger("providers.squeezemetrics")

_BASE_URL  = "https://squeezemetrics.com/monitor/dix"
_LOGIN_URL = "https://squeezemetrics.com/monitor/login"


@dataclass
class SqueezeMetricsSnapshot:
    """Dados mais recentes do SqueezeMetrics."""
    dix:          float | None = None   # Dark Index [0,1]
    gex:          float | None = None   # Gamma Exposure ($B)
    dix_prev:     float | None = None   # DIX dia anterior
    gex_prev:     float | None = None
    dix_signal:   str  = "neutral"      # "bullish" | "bearish" | "neutral"
    gex_regime:   str  = "flat"         # "long" | "short" | "flat"
    date:         str  = ""
    rationale:    list[str] = field(default_factory=list)
    timestamp:    str  = ""
    source:       str  = "squeezemetrics"
    error:        str  = ""


# ── Playwright scraper ────────────────────────────────────────────────────────

def _is_logged_in(page) -> bool:
    """Verifica se a sessão está autenticada."""
    url = page.url.lower()
    return "login" not in url and "squeezemetrics.com/monitor" in url


def _parse_page(page) -> SqueezeMetricsSnapshot:
    snap = SqueezeMetricsSnapshot(timestamp=datetime.now().isoformat())
    try:
        body = page.inner_text("body")

        # Tenta extrair DIX e GEX do texto da página
        # SqueezeMetrics exibe valores numéricos próximos a labels DIX / GEX
        dix_match = re.search(
            r'DIX[^0-9]{0,20}([\d]+\.[\d]+)\s*%',
            body, re.I
        )
        if dix_match:
            snap.dix = float(dix_match.group(1)) / 100.0

        gex_match = re.search(
            r'GEX[^0-9\-]{0,20}([\-]?[\d]+\.?[\d]*)\s*[Bb]',
            body, re.I
        )
        if gex_match:
            snap.gex = float(gex_match.group(1))

        # Fallback: busca no HTML com BeautifulSoup se disponível
        if snap.dix is None or snap.gex is None:
            try:
                html = page.content()
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")

                # SqueezeMetrics usa tabela com id="dix-table" ou similar
                for td in soup.find_all("td"):
                    text = td.get_text(strip=True)
                    if snap.dix is None and re.match(r'0?\.\d{2,4}', text):
                        try:
                            v = float(text)
                            if 0.2 < v < 0.8:  # DIX sempre nesta faixa
                                snap.dix = v
                        except ValueError:
                            pass
            except ImportError:
                pass  # BeautifulSoup opcional

        # Sinais qualitativos
        reasons = []
        if snap.dix is not None:
            if snap.dix > 0.45:
                snap.dix_signal = "bullish"
                reasons.append(f"DIX {snap.dix:.1%} — dark pool buying elevado, institucional bullish")
            elif snap.dix < 0.38:
                snap.dix_signal = "bearish"
                reasons.append(f"DIX {snap.dix:.1%} — dark pool baixo, possível distribuição")
            else:
                snap.dix_signal = "neutral"
                reasons.append(f"DIX {snap.dix:.1%} — dark pool neutro")

        if snap.gex is not None:
            if snap.gex > 0:
                snap.gex_regime = "long"
                reasons.append(f"GEX +${snap.gex:.1f}B — dealers long gamma, vol suprimida")
            elif snap.gex < 0:
                snap.gex_regime = "short"
                reasons.append(f"GEX ${snap.gex:.1f}B — dealers short gamma, vol amplificada")
            else:
                snap.gex_regime = "flat"

        snap.rationale = reasons

    except Exception as exc:
        snap.error = str(exc)[:80]
        _log.debug("squeezemetrics_parse_failed", error=snap.error)

    return snap


def collect(page) -> SqueezeMetricsSnapshot:
    """
    Coleta DIX + GEX do SqueezeMetrics via Playwright.

    `page` deve ser uma Playwright Page já aberta em sessão autenticada
    (persistent context gerenciado por app.cli.auth).

    Retorna SqueezeMetricsSnapshot com error preenchido se falhar.
    """
    snap = SqueezeMetricsSnapshot(timestamp=datetime.now().isoformat())

    try:
        page.goto(_BASE_URL, timeout=30_000)
        page.wait_for_load_state("domcontentloaded", timeout=15_000)

        if not _is_logged_in(page):
            _log.warning("squeezemetrics_not_authenticated", url=page.url[:80])
            snap.error = "Não autenticado — rode: python -m app.cli.auth login squeezemetrics"
            return snap

        try:
            page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass  # SPA pode não atingir networkidle

        time.sleep(2)

        snap = _parse_page(page)
        _log.info("squeezemetrics_done",
                  dix=snap.dix, gex=snap.gex,
                  dix_signal=snap.dix_signal, gex_regime=snap.gex_regime)

    except Exception as exc:
        snap.error = str(exc)[:100]
        _log.warning("squeezemetrics_failed", error=snap.error)

    return snap


def get_dix_signal_for_portfolio(snap: SqueezeMetricsSnapshot) -> float:
    """
    Retorna score [-1, 1] baseado no DIX para uso no composite.
    DIX > 45% → +0.5, DIX < 38% → -0.5, neutro → 0.
    """
    if snap is None or snap.dix is None:
        return 0.0
    if snap.dix > 0.45:
        return min(1.0, (snap.dix - 0.40) * 10)
    if snap.dix < 0.38:
        return max(-1.0, (snap.dix - 0.40) * 10)
    return 0.0
