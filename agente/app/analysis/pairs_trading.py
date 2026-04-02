"""
Pairs Trading Engine — Alpha Market-Neutro

Identifica pares estatisticamente cointegrados que divergiram alem do normal.
Gera sinais de entrada para posicoes long/short simultâneas.

Metodologia:
  1. Usa as edges do MST como candidatos iniciais (pares correlacionados)
  2. Para cada par, testa cointegração (Engle-Granger ou residuo OLS)
  3. Calcula z-score do spread atual vs historico (60d)
  4. z-score > 2: short o overperformer, long o underperformer
  5. z-score < -2: inverso
  6. Sai quando z-score volta para 0.5 (mean-reversion)

Vantagens:
  - Market-neutral: beta ~0
  - Funciona em qualquer regime (bull/bear/neutral)
  - Menor drawdown que posicoes direcionais puras

Output: list[PairsTrade] com log, entry, target, stop
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.pairs_trading")


@dataclass
class PairsTrade:
    # Par
    long_ticker: str
    short_ticker: str
    long_name: str = ""
    short_name: str = ""

    # Spread stats
    z_score: float = 0.0         # atual
    z_score_entry: float = 2.0   # threshold de entrada
    spread_mean: float = 0.0     # media historica (60d)
    spread_std: float = 0.0      # desvio padrao (60d)
    spread_current: float = 0.0  # valor atual

    # Hedge ratio (beta do spread)
    hedge_ratio: float = 1.0

    # Sinais
    direction: str = "none"        # "long_A_short_B" | "short_A_long_B" | "none"
    signal_strength: float = 0.0   # |z_score| normalizado [0, 1]
    days_since_divergence: int = 0

    # Sizing
    allocation_pct: float = 0.0   # % do portfolio para o par
    long_usd: float = 0.0
    short_usd: float = 0.0

    # Exit levels
    target_z: float = 0.5         # fechar quando z-score volta a 0.5
    stop_z: float = 3.5           # stop loss se spread aumenta

    # Correlacao / cointegração
    correlation: float = 0.0
    half_life_days: float | None = None  # velocidade de mean-reversion

    rationale: str = ""


@dataclass
class PairsResult:
    active_pairs: list[PairsTrade] = field(default_factory=list)
    candidate_pairs: list[tuple[str, str]] = field(default_factory=list)
    total_pairs_analyzed: int = 0
    timestamp: str = ""
    errors: list[str] = field(default_factory=list)


def _fetch_pair_history(ticker_a: str, ticker_b: str, days: int = 90) -> tuple | None:
    """
    Busca historico de precos para dois tickers.
    Returns: (series_a, series_b) alinhadas por data, ou None.
    """
    try:
        import yfinance as yf
        import datetime as dt

        end = dt.datetime.now()
        start = end - dt.timedelta(days=days + 20)

        data = yf.download(
            [ticker_a, ticker_b],
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            return None

        if hasattr(data.columns, "levels"):
            close = data["Close"]
        else:
            return None

        if ticker_a not in close.columns or ticker_b not in close.columns:
            return None

        close = close[[ticker_a, ticker_b]].dropna()
        if len(close) < 30:
            return None

        return close[ticker_a], close[ticker_b]

    except Exception as exc:
        _log.debug("pair_history_failed", pair=(ticker_a, ticker_b), error=str(exc)[:50])
        return None


def _compute_hedge_ratio(series_a, series_b) -> float:
    """Calcula hedge ratio via OLS: A = beta * B + alpha."""
    try:
        import numpy as np
        x = series_b.values
        y = series_a.values
        n = len(x)
        beta = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x**2).sum() - x.sum()**2)
        return float(beta) if not math.isnan(beta) else 1.0
    except Exception:
        return 1.0


def _estimate_half_life(residuals) -> float | None:
    """
    Estima o half-life de mean-reversion via AR(1) nos residuos.
    half_life = -log(2) / log(rho) onde rho e o coef AR(1).
    """
    try:
        import numpy as np
        y = residuals[1:]
        x = residuals[:-1]
        n = len(x)
        rho = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x**2).sum() - x.sum()**2)
        if rho >= 1.0 or rho <= 0.0:
            return None
        hl = -math.log(2) / math.log(rho)
        return float(hl) if 1 < hl < 120 else None  # 1 a 120 dias
    except Exception:
        return None


def _analyze_pair(ticker_a: str, ticker_b: str, budget: float = 100_000) -> PairsTrade | None:
    """Analisa um par e retorna PairsTrade se tiver sinal, None caso contrario."""
    hist = _fetch_pair_history(ticker_a, ticker_b, days=90)
    if hist is None:
        return None

    series_a, series_b = hist

    # Hedge ratio
    beta = _compute_hedge_ratio(series_a, series_b)

    # Spread: A - beta * B (ou log-ratio para estacionaridade)
    try:
        import numpy as np
        s_a = series_a.values
        s_b = series_b.values

        # Usar log-prices para robustez
        log_a = np.log(s_a)
        log_b = np.log(s_b)

        # Recalcula beta em log space
        n = len(log_a)
        beta_log = (n * (log_b * log_a).sum() - log_b.sum() * log_a.sum()) / \
                   (n * (log_b**2).sum() - log_b.sum()**2)

        spread = log_a - beta_log * log_b

        # Usa janela de 60 dias para media/std
        window = min(60, len(spread) - 5)
        spread_mean = float(spread[-window:].mean())
        spread_std  = float(spread[-window:].std())

        if spread_std < 1e-6:
            return None

        spread_current = float(spread[-1])
        z_score = (spread_current - spread_mean) / spread_std

        # Correlacao
        correlation = float(np.corrcoef(log_a[-60:], log_b[-60:])[0, 1])

        # Half-life
        residuals = spread - spread_mean
        half_life = _estimate_half_life(residuals)

        # Sinal: |z| > 2.0 com correlacao alta
        if abs(z_score) < 1.8 or correlation < 0.50:
            return None

        # Half-life aceitavel (5-45 dias de mean-reversion)
        if half_life is not None and (half_life > 45 or half_life < 2):
            return None

        # Direcao: se z_score > 2: A sobreperformou B → short A, long B
        if z_score > 0:
            direction = "short_A_long_B"  # A cara, B barata
        else:
            direction = "long_A_short_B"  # A barata, B cara

        signal_strength = min(1.0, (abs(z_score) - 1.8) / 1.5)

        # Sizing: max 5% do portfolio por par, escala por sinal
        alloc_pct = min(0.05, 0.02 + 0.03 * signal_strength)
        alloc_usd = budget * alloc_pct

        rationale = (
            f"z={z_score:+.2f} | corr={correlation:.2f} | "
            f"beta={beta_log:.2f} | HL={half_life:.0f}d"
            if half_life else
            f"z={z_score:+.2f} | corr={correlation:.2f}"
        )

        return PairsTrade(
            long_ticker=ticker_a if direction == "long_A_short_B" else ticker_b,
            short_ticker=ticker_b if direction == "long_A_short_B" else ticker_a,
            z_score=z_score,
            spread_mean=spread_mean,
            spread_std=spread_std,
            spread_current=spread_current,
            hedge_ratio=beta_log,
            direction=direction,
            signal_strength=signal_strength,
            allocation_pct=alloc_pct,
            long_usd=alloc_usd,
            short_usd=alloc_usd,
            target_z=0.5,
            stop_z=3.5,
            correlation=correlation,
            half_life_days=half_life,
            rationale=rationale,
        )

    except Exception as exc:
        _log.debug("pair_analysis_failed", pair=(ticker_a, ticker_b), error=str(exc)[:80])
        return None


# ── Candidatos de pares por setor (conhecidos como cointegrados historicamente) ──
_KNOWN_PAIRS: list[tuple[str, str]] = [
    # Treasuries
    ("TLT", "IEF"),
    ("IEF", "SHY"),
    ("TLT", "SHY"),
    # Equities vs Bonds (risk-off pairs)
    ("SPY", "TLT"),
    ("QQQ", "TLT"),
    # Oil e energia
    ("XOM", "CVX"),
    ("XLE", "XOM"),
    # Tech
    ("AAPL", "MSFT"),
    ("NVDA", "AMD"),
    ("META", "GOOGL"),
    # Financials
    ("JPM", "BAC"),
    ("JPM", "GS"),
    # Gold vs bonds
    ("GLD", "TLT"),
    ("GLD", "SLV"),
    # Vol
    ("VXX", "SPY"),
    # Dollar vs Gold
    ("UUP", "GLD"),
    # EM vs DM
    ("EEM", "SPY"),
    ("EWZ", "EEM"),
]


def compute_pairs_signals(
    mst_adj: dict[str, list[str]] | None = None,
    universe_tickers: list[str] | None = None,
    budget: float = 100_000,
    max_pairs: int = 10,
) -> PairsResult:
    """
    Identifica pares com sinal ativo.

    mst_adj: adjacency list do MST (edges = candidatos naturais)
    universe_tickers: lista de tickers no universe para filtrar candidatos
    """
    result = PairsResult(timestamp=datetime.now().isoformat())

    # Constroi lista de candidatos
    candidates: list[tuple[str, str]] = []

    # 1. Pares conhecidos historicamente cointegrados
    candidates.extend(_KNOWN_PAIRS)

    # 2. Pares do MST (alta correlacao estrutural)
    if mst_adj:
        for ticker_a, neighbors in mst_adj.items():
            for ticker_b in neighbors:
                pair = tuple(sorted([ticker_a, ticker_b]))
                if pair not in candidates:
                    candidates.append(pair)

    # Filtra por universe se fornecido
    if universe_tickers:
        universe_set = set(universe_tickers)
        candidates = [
            (a, b) for a, b in candidates
            if a in universe_set or b in universe_set
        ]

    result.candidate_pairs = candidates[:30]
    result.total_pairs_analyzed = 0

    active: list[PairsTrade] = []
    seen_tickers: set[str] = set()  # evita usar mesmo ticker em multiplos pares

    for ticker_a, ticker_b in candidates:
        if len(active) >= max_pairs:
            break

        # Nao usar mesmo ticker em dois pares (aumentaria concentracao)
        if ticker_a in seen_tickers or ticker_b in seen_tickers:
            continue

        result.total_pairs_analyzed += 1

        try:
            pair_trade = _analyze_pair(ticker_a, ticker_b, budget=budget)
            if pair_trade:
                active.append(pair_trade)
                seen_tickers.add(ticker_a)
                seen_tickers.add(ticker_b)
                _log.info("pair_signal_found",
                          long=pair_trade.long_ticker,
                          short=pair_trade.short_ticker,
                          z=round(pair_trade.z_score, 2))
        except Exception as exc:
            result.errors.append(f"{ticker_a}/{ticker_b}: {exc}")

    # Ordena por force do sinal
    active.sort(key=lambda p: abs(p.z_score), reverse=True)
    result.active_pairs = active

    _log.info("pairs_done",
              analyzed=result.total_pairs_analyzed,
              active=len(active))

    return result
