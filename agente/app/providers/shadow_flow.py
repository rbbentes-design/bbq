"""
Shadow Flow — Dark Pool & Unusual Options Activity

Coleta atividade de dark pool e opcoes incomuns que sinalizam posicionamento institucional.

Por que dark pool importa:
  - ~40% do volume americano passa por dark pools (off-exchange)
  - Large block prints = institucional acumulando/distribuindo
  - Quando um institucional compra 1M acoes no dark pool, ele sabe algo
  - Dark pool print ACIMA do preco de mercado = compra agressiva (bullish)
  - Dark pool print ABAIXO = venda agressiva (bearish)

Por que unusual options importa (Shadow Flow):
  - Calls OTM com expiracao curta e volume 5x o OI normal = smart money posicionando
  - Sweep orders (cruzam multiplos exchanges) = urgencia (nao quer esconder)
  - Put sweep em grande quantidade = hedge institucional (sabe de risco)

Fontes que o usuario tem acesso:
  1. Shadow Flow (shadowflow.io ou similar) — dark pool + unusual options
  2. Unusual Whales API (se disponivel)
  3. Finviz Dark Pool (scraping)
  4. SpotGamma HIRO (ja integrado no spotgamma_live.py)

Implementacao:
  - Provider via Playwright se autenticado
  - Fallback via yfinance volume ratio (volume atual / avg volume 20d > 3x = unusual)
  - Calcula "dark pool score" por ticker

Output: dict[ticker, DarkPoolSignal]
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.shadow_flow")


@dataclass
class DarkPoolPrint:
    """Um print individual de dark pool."""
    ticker: str
    price: float
    size: int               # numero de acoes
    value_usd: float        # valor total
    direction: str          # "buy" | "sell" | "neutral"
    premium: float          # preco vs mid (positivo = compra acima do mid)
    timestamp: str


@dataclass
class UnusualOption:
    """Uma opcao com atividade incomum."""
    ticker: str
    expiry: str
    strike: float
    option_type: str        # "call" | "put"
    volume: int
    open_interest: int
    volume_oi_ratio: float  # > 5x = muito incomum
    premium_paid: float     # $ total pago pelo sweep
    sentiment: str          # "bullish" | "bearish" | "neutral"
    is_sweep: bool          # se cruzou multiplos exchanges
    timestamp: str


@dataclass
class DarkPoolSignal:
    ticker: str

    # Dark pool summary
    total_dp_buy_value: float = 0.0     # $ comprado no dark pool hoje
    total_dp_sell_value: float = 0.0    # $ vendido no dark pool hoje
    dp_net_flow: float = 0.0            # buy - sell (positivo = acumulacao)
    dp_premium_avg: float = 0.0         # premium medio (acima/abaixo do mid)
    unusual_volume_ratio: float = 1.0   # vol atual / avg vol 20d

    # Unusual options
    call_premium_total: float = 0.0     # $ total em calls incomuns
    put_premium_total: float = 0.0      # $ total em puts incomuns
    options_sentiment: str = "neutral"  # "bullish" | "bearish" | "neutral"
    n_sweeps: int = 0                   # numero de sweep orders

    # Prints recentes
    recent_prints: list[DarkPoolPrint] = field(default_factory=list)
    unusual_options: list[UnusualOption] = field(default_factory=list)

    # Signal derivado
    dark_pool_score: float = 0.0        # [-1, 1]
    rationale: list[str] = field(default_factory=list)
    timestamp: str = ""
    source: str = "shadow_flow"


@dataclass
class ShadowFlowResult:
    signals: dict[str, DarkPoolSignal] = field(default_factory=dict)
    top_bullish: list[str] = field(default_factory=list)
    top_bearish: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp: str = ""


# ── Volume ratio fallback (sem acesso ao Shadow Flow) ─────────────────────────

def _compute_volume_signal(ticker: str, market_prices: dict) -> DarkPoolSignal:
    """
    Estima dark pool signal via volume ratio e opcoes.
    Usado quando nao ha acesso ao Shadow Flow.
    """
    sig = DarkPoolSignal(ticker=ticker, timestamp=datetime.now().isoformat())

    try:
        import yfinance as yf

        # Normaliza sufixos Bloomberg antes de chamar yfinance
        _yf_ticker = ticker
        for _sfx in (" US Equity", " US EQUITY", " Equity", " EQUITY",
                     " Index", " INDEX", " Comdty", " COMDTY", " Curncy", " CURNCY"):
            if _yf_ticker.endswith(_sfx):
                _yf_ticker = _yf_ticker[:-len(_sfx)].strip()
                break
        _yf_ticker = _yf_ticker.replace("/", "-")

        tk = yf.Ticker(_yf_ticker)
        hist = tk.history(period="25d", auto_adjust=True)

        if hist.empty or len(hist) < 5:
            return sig

        # Volume atual vs media 20d
        avg_vol = float(hist["Volume"].iloc[:-1].mean())
        cur_vol = float(hist["Volume"].iloc[-1])
        sig.unusual_volume_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

        # Preco vs VWAP aproximado
        cur_close = float(hist["Close"].iloc[-1])
        cur_open  = float(hist["Open"].iloc[-1])
        cur_high  = float(hist["High"].iloc[-1])
        cur_low   = float(hist["Low"].iloc[-1])
        vwap_approx = (cur_high + cur_low + cur_close) / 3

        # Volume forte + preco subindo = buyers agressivos
        daily_ret = float(hist["Close"].pct_change().iloc[-1])
        prev_ret  = float(hist["Close"].pct_change().iloc[-2]) if len(hist) > 2 else 0

        # Opcoes: usa fast_info para evitar chamadas pesadas de option_chain
        # (option_chain por si só gasta 2 requests por ticker — rate limit rápido)
        call_ratio = 0.0
        put_ratio = 0.0
        try:
            info = tk.fast_info
            # fast_info não tem put/call ratio diretamente — skip options aqui
            # Options flow é analisado separadamente no options_strategy.py
        except Exception:
            pass

        # Constroi sinal
        score = 0.0
        reasons = []

        # Volume unusual — thresholds calibrados para mercado normal
        if sig.unusual_volume_ratio > 2.0:
            if daily_ret > 0:
                score += 0.30
                reasons.append(f"Volume {sig.unusual_volume_ratio:.1f}x acima da media com preco em alta — acumulacao institucional")
            else:
                score -= 0.30
                reasons.append(f"Volume {sig.unusual_volume_ratio:.1f}x acima da media com preco em queda — distribuicao institucional")
        elif sig.unusual_volume_ratio > 1.3:
            score += 0.15 * (1 if daily_ret > 0 else -1)
            reasons.append(f"Volume {sig.unusual_volume_ratio:.1f}x acima da media")
        elif sig.unusual_volume_ratio < 0.5 and cur_vol > 0:
            # Volume muito baixo = mercado fechado ou sem liquidez — score neutro mas registra
            reasons.append(f"Volume muito baixo ({sig.unusual_volume_ratio:.2f}x avg) — possivel mercado fechado")

        # Momentum diário reforça o sinal
        if abs(daily_ret) > 0.015:  # movimento > 1.5%
            score += 0.20 * (1 if daily_ret > 0 else -1)
            reasons.append(f"Retorno diário {daily_ret:+.1%}")

        # Options ratio (calls vs puts)
        if call_ratio > 2.0 and call_ratio > put_ratio * 1.5:
            score += 0.20
            reasons.append(f"Call V/OI={call_ratio:.1f}x alto — fluxo bullish de opcoes")
            sig.options_sentiment = "bullish"
        elif put_ratio > 2.0 and put_ratio > call_ratio * 1.5:
            score -= 0.20
            reasons.append(f"Put V/OI={put_ratio:.1f}x alto — protecao institucional, bearish")
            sig.options_sentiment = "bearish"

        sig.call_premium_total = call_ratio * 100_000  # aproximacao
        sig.put_premium_total  = put_ratio  * 100_000
        sig.dark_pool_score = max(-1.0, min(1.0, score))
        sig.rationale = reasons

    except Exception as exc:
        _log.debug("volume_signal_failed", ticker=ticker, error=str(exc)[:60])

    return sig


def _scrape_shadow_flow(page, ticker: str) -> DarkPoolSignal | None:
    """
    Tenta scraping do Shadow Flow via Playwright.
    Retorna None se nao autenticado ou URL nao disponivel.
    """
    sig = DarkPoolSignal(ticker=ticker, timestamp=datetime.now().isoformat())
    sig.source = "shadow_flow_live"

    # Shadow Flow usa shadowflow.io ou similar
    urls_to_try = [
        f"https://app.shadowflow.io/ticker/{ticker}",
        f"https://shadowtrader.io/flow/{ticker}",
        f"https://unusualwhales.com/flow?ticker={ticker}",
    ]

    for url in urls_to_try[:1]:  # tenta apenas o primeiro por enquanto
        try:
            page.goto(url, timeout=15_000)
            page.wait_for_load_state("domcontentloaded", timeout=8_000)
            time.sleep(2)

            if "login" in page.url.lower() or "signin" in page.url.lower():
                return None

            body = page.inner_text("body")

            # Extrai flows (patterns genéricos para diferentes plataformas)
            import re

            # Bullish/bearish summary
            if re.search(r"bullish|bull\s*flow", body, re.I):
                sig.options_sentiment = "bullish"
                sig.dark_pool_score += 0.30
                sig.rationale.append("Shadow Flow: fluxo bullish detectado")
            elif re.search(r"bearish|bear\s*flow", body, re.I):
                sig.options_sentiment = "bearish"
                sig.dark_pool_score -= 0.30
                sig.rationale.append("Shadow Flow: fluxo bearish detectado")

            # Volume unusual
            vol_match = re.search(r"(\d+\.?\d*)x\s*(?:normal|average|avg)", body, re.I)
            if vol_match:
                ratio = float(vol_match.group(1))
                sig.unusual_volume_ratio = ratio
                if ratio > 3:
                    sig.rationale.append(f"Volume {ratio:.1f}x anormal detectado")

            sig.dark_pool_score = max(-1.0, min(1.0, sig.dark_pool_score))
            return sig

        except Exception as exc:
            _log.debug("shadow_flow_scrape_failed", url=url[:50], error=str(exc)[:60])

    return None


def collect_shadow_flow(
    tickers: list[str],
    market_prices: dict | None = None,
    page=None,  # Playwright page opcional
) -> ShadowFlowResult:
    """
    Coleta dados de dark pool e unusual options.

    Se `page` fornecido (Playwright), tenta Shadow Flow live.
    Fallback: volume ratio + options flow via yfinance.
    """
    result = ShadowFlowResult(timestamp=datetime.now().isoformat())
    market_prices = market_prices or {}

    for ticker in tickers[:30]:  # limita para nao demorar
        sig = None

        # Tenta Shadow Flow live primeiro
        if page is not None:
            try:
                sig = _scrape_shadow_flow(page, ticker)
            except Exception as exc:
                _log.debug("sf_live_failed", ticker=ticker, error=str(exc)[:40])

        # Fallback: volume ratio
        if sig is None:
            sig = _compute_volume_signal(ticker, market_prices)

        result.signals[ticker] = sig

        if sig.dark_pool_score > 0.25:
            result.top_bullish.append(ticker)
        elif sig.dark_pool_score < -0.25:
            result.top_bearish.append(ticker)

    # Sort por magnitude
    result.top_bullish.sort(
        key=lambda t: result.signals[t].dark_pool_score if t in result.signals else 0, reverse=True
    )
    result.top_bearish.sort(
        key=lambda t: result.signals[t].dark_pool_score if t in result.signals else 0
    )

    _log.info("shadow_flow_done",
              n=len(result.signals),
              bullish=result.top_bullish[:3],
              bearish=result.top_bearish[:3])

    return result


def get_dark_pool_signal_for_ticker(ticker: str, sf_result: ShadowFlowResult) -> float:
    """
    Retorna o dark pool signal para uso no alpha_signals composite.
    Score [-1, 1].
    """
    sig = sf_result.signals.get(ticker)
    if sig is None:
        return 0.0
    return sig.dark_pool_score
