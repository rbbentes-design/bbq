"""
SpotGamma Live — Gamma Exposure em Tempo Real

Navega o dashboard SpotGamma via Playwright e extrai:
  1. GEX por ativo (Gamma Exposure em $B)
  2. Gamma Flip Level — nivel critico onde dealers mudam de hedging
  3. Zero Gamma Level
  4. Vol Trigger
  5. Call Wall / Put Wall
  6. Dark Pool Level (se disponivel)
  7. Key strikes com maior OI

O Gamma Flip Level e o sinal mais poderoso do SpotGamma:
  - Preco ACIMA do Flip: dealers SHORT gamma → vendem rallies, compram quedas (amortiza vol)
  - Preco ABAIXO do Flip: dealers LONG gamma → compram rallies, vendem quedas (amplifica vol)

Quando o SPX fica abaixo do gamma flip, os dealers precisam vender mais a cada queda.
Isso cria "convexity cascade" — quedas se auto-aceleram.

URLs:
  - Dashboard: https://dashboard.spotgamma.com
  - Tickers: https://dashboard.spotgamma.com/stock/{TICKER}
  - SPX: https://dashboard.spotgamma.com/stock/SPX

Autenticacao via perfil Playwright persistente (mesmo do spotgamma.py principal).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.spotgamma_live")

_BASE_URL = "https://dashboard.spotgamma.com"
_STOCK_URL = "https://dashboard.spotgamma.com/stock/{ticker}"

# Tickers com maior relevancia para monitorar
_PRIORITY_TICKERS = ["SPX", "SPY", "QQQ", "NVDA", "AAPL", "TSLA", "META", "AMZN", "MSFT"]


@dataclass
class SpotGammaLiveData:
    ticker: str
    price: float | None = None

    # Gamma levels
    gamma_flip: float | None = None       # Nivel critico
    zero_gamma: float | None = None       # Nivel onde GEX = 0
    vol_trigger: float | None = None      # Nivel de gatilho de volatilidade

    # Walls
    call_wall: float | None = None        # Maior resistencia (onde calls estao concentradas)
    put_wall: float | None = None         # Maior suporte (onde puts estao concentradas)

    # GEX
    total_gex_b: float | None = None      # GEX total em $B (positivo = dealers curtos gamma)
    gex_direction: str = "neutral"        # "positive" | "negative" | "neutral"

    # Posicionamento
    price_vs_flip: str = "unknown"        # "above" | "below" | "at"
    dealer_regime: str = "unknown"        # "short_gamma" | "long_gamma" | "neutral"

    # Dark pool / key levels
    hiro: float | None = None            # HIRO indicator se disponivel
    dark_pool_level: float | None = None

    # Sinal derivado
    sg_signal: float = 0.0               # [-1, 1] para alpha composite
    rationale: list[str] = field(default_factory=list)

    timestamp: str = ""
    source: str = "spotgamma_live"


@dataclass
class SpotGammaLiveResult:
    tickers: dict[str, SpotGammaLiveData] = field(default_factory=dict)
    spx_data: SpotGammaLiveData | None = None
    errors: list[str] = field(default_factory=list)
    timestamp: str = ""


def _extract_number(text: str, pattern: str) -> float | None:
    """Extrai numero de texto usando regex."""
    try:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(",", "").replace("$", "").strip()
            return float(num_str)
    except Exception:
        pass
    return None


def _parse_sg_text(text: str, ticker: str, price: float | None) -> SpotGammaLiveData:
    """
    Extrai dados do SpotGamma a partir do texto da pagina.
    Patterns baseados na estrutura do dashboard SpotGamma.
    """
    data = SpotGammaLiveData(ticker=ticker, timestamp=datetime.now().isoformat())
    data.price = price

    text_lower = text.lower()

    # Patterns para niveis chave
    # SpotGamma usa labels como "Gamma Flip", "Vol Trigger", "Call Wall", etc.
    patterns = {
        "gamma_flip":  r"gamma\s*flip[:\s]+\$?([\d,]+\.?\d*)",
        "zero_gamma":  r"zero\s*gamma[:\s]+\$?([\d,]+\.?\d*)",
        "vol_trigger": r"vol(?:atility)?\s*trigger[:\s]+\$?([\d,]+\.?\d*)",
        "call_wall":   r"call\s*wall[:\s]+\$?([\d,]+\.?\d*)",
        "put_wall":    r"put\s*wall[:\s]+\$?([\d,]+\.?\d*)",
        "total_gex":   r"total\s*gex[:\s]+\$?([-\d,]+\.?\d*)\s*[Bb]",
        "hiro":        r"hiro[:\s]+([-\d,]+\.?\d*)",
        "dark_pool":   r"dark\s*pool[:\s]+\$?([\d,]+\.?\d*)",
    }

    for field_name, pattern in patterns.items():
        val = _extract_number(text, pattern)
        if val is not None:
            setattr(data, field_name.replace("total_gex", "total_gex_b"), val)

    # Determina posicionamento vs gamma flip
    if data.price and data.gamma_flip:
        diff_pct = (data.price - data.gamma_flip) / data.gamma_flip
        if data.price > data.gamma_flip * 1.001:
            data.price_vs_flip = "above"
            data.dealer_regime = "short_gamma"  # dealers vendem rallies, compram dips
            data.rationale.append(
                f"Preco ${data.price:,.0f} ACIMA do Gamma Flip ${data.gamma_flip:,.0f} (+{diff_pct:.1%}): "
                f"dealers SHORT gamma → volatilidade amortizada, dips comprados"
            )
        elif data.price < data.gamma_flip * 0.999:
            data.price_vs_flip = "below"
            data.dealer_regime = "long_gamma"   # dealers amplificam movimentos
            data.rationale.append(
                f"Preco ${data.price:,.0f} ABAIXO do Gamma Flip ${data.gamma_flip:,.0f} ({diff_pct:.1%}): "
                f"dealers LONG gamma → movimentos AMPLIFICADOS, quedas auto-aceleram"
            )
        else:
            data.price_vs_flip = "at"
            data.dealer_regime = "neutral"
            data.rationale.append(f"Preco AT Gamma Flip — zona de alta volatilidade/instabilidade")

    # GEX direction
    if data.total_gex_b is not None:
        if data.total_gex_b > 0.5:
            data.gex_direction = "positive"
            data.rationale.append(f"GEX positivo (${data.total_gex_b:.1f}B) — dealers curtos gamma, vol suprimida")
        elif data.total_gex_b < -0.5:
            data.gex_direction = "negative"
            data.rationale.append(f"GEX negativo (${data.total_gex_b:.1f}B) — dealers longos gamma, vol pode explodir")

    # Call Wall e Put Wall como suporte/resistencia
    if data.price and data.call_wall:
        dist = (data.call_wall - data.price) / data.price
        if dist < 0.03:
            data.rationale.append(f"Call Wall proxima: ${data.call_wall:,.0f} ({dist:.1%} acima) → resistencia forte")
    if data.price and data.put_wall:
        dist = (data.price - data.put_wall) / data.price
        if dist < 0.03:
            data.rationale.append(f"Put Wall proxima: ${data.put_wall:,.0f} ({dist:.1%} abaixo) → suporte forte")

    # SG Signal: combina todos os dados
    sg = 0.0
    if data.dealer_regime == "short_gamma":
        sg += 0.20   # mercado mais estavel → leve bias de compra
    elif data.dealer_regime == "long_gamma":
        sg -= 0.25   # mercado instavel → bias de protecao

    if data.gex_direction == "positive":
        sg += 0.15
    elif data.gex_direction == "negative":
        sg -= 0.20

    # HIRO indicator (se disponivel)
    if data.hiro is not None:
        if data.hiro > 0:
            sg += min(0.20, data.hiro / 1000)
            data.rationale.append(f"HIRO={data.hiro:.0f} positivo — fluxo de call compradores")
        else:
            sg -= min(0.20, abs(data.hiro) / 1000)
            data.rationale.append(f"HIRO={data.hiro:.0f} negativo — fluxo de put compradores")

    data.sg_signal = max(-1.0, min(1.0, sg))

    return data


def collect_spotgamma_live(
    page,           # playwright Page
    tickers: list[str] | None = None,
) -> SpotGammaLiveResult:
    """
    Coleta dados live do SpotGamma via Playwright.

    page: Playwright Page ja autenticado
    tickers: lista de tickers a coletar (default: priority list)
    """
    result = SpotGammaLiveResult(timestamp=datetime.now().isoformat())
    tickers = tickers or _PRIORITY_TICKERS

    # Sempre coleta SPX primeiro (mercado geral)
    all_tickers = ["SPX"] + [t for t in tickers if t != "SPX"]

    for ticker in all_tickers[:8]:  # limite para nao demorar demais
        try:
            url = _STOCK_URL.format(ticker=ticker)
            page.goto(url, timeout=20_000)
            page.wait_for_load_state("domcontentloaded", timeout=10_000)
            time.sleep(2.5)

            # Checa autenticacao
            if "/login" in page.url:
                _log.warning("spotgamma_not_auth")
                result.errors.append("Nao autenticado no SpotGamma")
                break

            # Extrai preco do ativo
            price = None
            try:
                price_els = page.locator("[data-testid='price'], .stock-price, .current-price").all()
                for el in price_els[:3]:
                    txt = el.inner_text().strip().replace("$", "").replace(",", "")
                    try:
                        price = float(txt)
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

            # Extrai texto completo da pagina (dados renderizados pela SPA)
            body_text = page.inner_text("body")

            sg_data = _parse_sg_text(body_text, ticker, price)

            # Tenta extrair niveis diretamente de elementos especificos
            try:
                # SpotGamma usa labels como "Gamma Flip: $5,900"
                all_elements = page.locator("[class*='level'], [class*='strike'], [class*='key'], [data-level]").all()
                for el in all_elements[:20]:
                    label_text = el.inner_text()
                    for field_name, pattern in {
                        "gamma_flip":  r"flip.*?\$?([\d,]+)",
                        "call_wall":   r"call.*?wall.*?\$?([\d,]+)",
                        "put_wall":    r"put.*?wall.*?\$?([\d,]+)",
                        "vol_trigger": r"trigger.*?\$?([\d,]+)",
                    }.items():
                        val = _extract_number(label_text, pattern)
                        if val and not getattr(sg_data, field_name):
                            setattr(sg_data, field_name, val)
            except Exception:
                pass

            result.tickers[ticker] = sg_data
            if ticker == "SPX":
                result.spx_data = sg_data

            _log.info("spotgamma_live_collected",
                      ticker=ticker,
                      flip=sg_data.gamma_flip,
                      regime=sg_data.dealer_regime,
                      signal=sg_data.sg_signal)

        except Exception as exc:
            _log.warning("spotgamma_live_failed", ticker=ticker, error=str(exc)[:80])
            result.errors.append(f"{ticker}: {str(exc)[:60]}")

    return result


def get_sg_signal_for_ticker(ticker: str, sg_result: SpotGammaLiveResult) -> float:
    """
    Retorna o SpotGamma signal para uso no alpha_signals composite.
    Usa SPX como proxy para ativos sem dados individuais.
    """
    # Tenta dado direto do ticker
    data = sg_result.tickers.get(ticker)
    if data:
        return data.sg_signal

    # Fallback: SPX signal (gamma regime geral afeta todos os ativos)
    if sg_result.spx_data:
        return sg_result.spx_data.sg_signal * 0.6  # atenuado para ativos individuais

    return 0.0
