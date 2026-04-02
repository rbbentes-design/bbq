"""
CTA Positioning Model

Estima o posicionamento atual dos CTAs (Commodity Trading Advisors) —
fundos sistematicos que seguem momentum em futuros.

Por que isso importa:
  - CTAs gerenciam ~$400B em ativos sistematicamente
  - Seguem momentum: se um ativo subiu nos ultimos 3-12 meses, eles estao longos
  - Quando o momentum se inverte ou vol sobe, eles PRECISAM fechar posicoes
  - Esse fluxo forcado cria oportunidades: antecipar o squeeze

Metodologia (baseada em pesquisas do Goldman Sachs / DB / JPMorgan):
  1. Para cada ativo principal, calcula momentum 1m, 3m, 6m, 12m
  2. Estima posicionamento CTA via scoring ponderado
  3. Identifica ativos com posicionamento extremo (oversold CTAs)
  4. Gera sinal de "flow surprise": quanto os CTAs teriam que comprar/vender
     se o momentum reverter

Outputs por ativo:
  - cta_score: [-1, 1] — estimativa de posicionamento CTA
  - crowding: "extreme_short" | "short" | "neutral" | "long" | "extreme_long"
  - flow_surprise: se mercado virar, quantos $ de fluxo sao esperados
  - reversal_risk: probabilidade de squeeze nos proximos 5 dias

Fonte de dados: yfinance (retornos historicos)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.cta_positioning")


# Ativos que CTAs tradicionais operam (futuros proxies via ETFs)
_CTA_UNIVERSE: dict[str, str] = {
    # Equities
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "EEM": "EM Equities",
    "EWZ": "Brazil",
    "FXI": "China",
    # Fixed Income
    "TLT": "30Y Treasury",
    "IEF": "10Y Treasury",
    "HYG": "High Yield",
    # Commodities
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "Oil",
    "DBC": "Commodities Basket",
    "CORN": "Corn",
    "SOYB": "Soybeans",
    # FX (via ETFs)
    "UUP": "US Dollar",
    "FXE": "Euro",
    "FXY": "Yen",
    # Vol
    "VXX": "VIX Short-term",
}


@dataclass
class CTASignal:
    ticker: str
    name: str
    cta_score: float           # [-1, 1]: -1=extreme short, +1=extreme long
    crowding: str              # extreme_short | short | neutral | long | extreme_long
    momentum_1m: float | None = None
    momentum_3m: float | None = None
    momentum_6m: float | None = None
    momentum_12m: float | None = None
    flow_surprise_signal: float = 0.0  # [-1, 1]: potencial fluxo se reverter
    reversal_probability: float = 0.0  # [0, 1]
    rationale: str = ""


@dataclass
class CTAPositioningResult:
    signals: dict[str, CTASignal] = field(default_factory=dict)
    aggregate_equity_score: float = 0.0    # CTA exposure em equities
    aggregate_bond_score: float = 0.0
    aggregate_commodity_score: float = 0.0
    extreme_shorts: list[str] = field(default_factory=list)
    extreme_longs: list[str] = field(default_factory=list)
    squeeze_candidates: list[str] = field(default_factory=list)
    timestamp: str = ""
    errors: list[str] = field(default_factory=list)


def _fetch_returns(tickers: list[str]) -> dict[str, dict]:
    """Busca retornos historicos para calcular momentum CTA."""
    result: dict[str, dict] = {}
    try:
        import yfinance as yf
        import datetime as dt

        # Baixa 14 meses de dados
        end = dt.datetime.now()
        start = end - dt.timedelta(days=430)

        tickers_valid = [t for t in tickers if t]
        data = yf.download(
            tickers_valid,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            return result

        # Compatibilidade com MultiIndex e single-ticker
        if hasattr(data.columns, "levels"):
            close_df = data["Close"] if "Close" in data.columns else data
        else:
            close_df = data

        today = date.today()

        for ticker in tickers_valid:
            try:
                if ticker in close_df.columns:
                    series = close_df[ticker].dropna()
                elif len(tickers_valid) == 1:
                    series = close_df.dropna()
                else:
                    continue

                if len(series) < 20:
                    continue

                price_now = float(series.iloc[-1])

                def ret_n_days(n: int) -> float | None:
                    if len(series) <= n:
                        return None
                    p0 = float(series.iloc[-(n+1)])
                    if p0 <= 0:
                        return None
                    return (price_now / p0) - 1.0

                result[ticker] = {
                    "price": price_now,
                    "ret_1m":  ret_n_days(21),
                    "ret_3m":  ret_n_days(63),
                    "ret_6m":  ret_n_days(126),
                    "ret_12m": ret_n_days(252),
                }
            except Exception as exc:
                _log.debug("ticker_ret_failed", ticker=ticker, error=str(exc))

    except Exception as exc:
        _log.warning("cta_fetch_failed", error=str(exc))

    return result


def _compute_cta_signal(ticker: str, data: dict) -> CTASignal:
    """
    Calcula o sinal CTA para um ativo.

    Metodologia Goldman Sachs CTA scoring:
      - Peso 12m: 30%
      - Peso 6m:  25%
      - Peso 3m:  30%
      - Peso 1m:  15%
    Cada momentum normalizado como sign × log(1 + |ret|) para suavizar outliers.
    """
    name = _CTA_UNIVERSE.get(ticker, ticker)

    def norm_mom(r: float | None) -> float:
        if r is None:
            return 0.0
        return math.copysign(math.log1p(abs(r)), r)

    m1  = data.get("ret_1m")
    m3  = data.get("ret_3m")
    m6  = data.get("ret_6m")
    m12 = data.get("ret_12m")

    nm1  = norm_mom(m1)
    nm3  = norm_mom(m3)
    nm6  = norm_mom(m6)
    nm12 = norm_mom(m12)

    # Score ponderado
    raw_score = 0.15 * nm1 + 0.30 * nm3 + 0.25 * nm6 + 0.30 * nm12

    # Normaliza para [-1, 1] usando tanh
    cta_score = math.tanh(raw_score * 4)  # fator 4 para amplificar sinal tipico

    # Classificacao de crowding
    if cta_score < -0.70:
        crowding = "extreme_short"
    elif cta_score < -0.35:
        crowding = "short"
    elif cta_score < 0.35:
        crowding = "neutral"
    elif cta_score < 0.70:
        crowding = "long"
    else:
        crowding = "extreme_long"

    # Reversal probability:
    # Extreme positions + recente divergencia = alta probabilidade de squeeze
    recent_divergence = 0.0
    if m1 is not None and m3 is not None:
        # Se momentum recente (1m) diverge do posicionamento (3m+)
        recent_divergence = abs(nm1 - (0.5 * nm3 + 0.5 * nm12))

    reversal_prob = 0.0
    if crowding == "extreme_short" and m1 is not None and m1 > 0.02:
        # Posicionamento extremamente short mas preco subiu recentemente
        reversal_prob = min(0.70, 0.30 + recent_divergence * 2)
    elif crowding == "extreme_long" and m1 is not None and m1 < -0.02:
        # Posicionamento extremamente long mas preco caiu recentemente
        reversal_prob = min(0.70, 0.30 + recent_divergence * 2)
    else:
        reversal_prob = min(0.25, recent_divergence * 0.5)

    # Flow surprise: se posicionamento reverter completamente, qual o fluxo relativo?
    # Proporcional ao abs(cta_score) — ativos mais extremos tem mais fluxo potencial
    flow_surprise = -cta_score * reversal_prob  # sinal oposto ao posicionamento

    # Rationale
    parts = []
    if m12 is not None: parts.append(f"12m={m12:+.1%}")
    if m6  is not None: parts.append(f"6m={m6:+.1%}")
    if m3  is not None: parts.append(f"3m={m3:+.1%}")
    if m1  is not None: parts.append(f"1m={m1:+.1%}")
    rationale = f"CTA={cta_score:+.2f} ({crowding}) | {' · '.join(parts)}"
    if reversal_prob > 0.40:
        rationale += f" | SQUEEZE RISK={reversal_prob:.0%}"

    return CTASignal(
        ticker=ticker,
        name=name,
        cta_score=cta_score,
        crowding=crowding,
        momentum_1m=m1,
        momentum_3m=m3,
        momentum_6m=m6,
        momentum_12m=m12,
        flow_surprise_signal=flow_surprise,
        reversal_probability=reversal_prob,
        rationale=rationale,
    )


def compute_cta_positioning(
    tickers_extra: list[str] | None = None,
) -> CTAPositioningResult:
    """
    Calcula o posicionamento CTA para todos os ativos no universo.
    tickers_extra: ativos adicionais fora do universo padrao.
    """
    result = CTAPositioningResult(timestamp=datetime.now().isoformat())
    universe = list(_CTA_UNIVERSE.keys())
    if tickers_extra:
        universe = list(set(universe + tickers_extra))

    ret_data = _fetch_returns(universe)

    if not ret_data:
        result.errors.append("Sem dados de retorno — yfinance falhou")
        return result

    equity_scores: list[float] = []
    bond_scores: list[float] = []
    commodity_scores: list[float] = []

    _EQUITY_TICKERS = {"SPY", "QQQ", "IWM", "EEM", "EWZ", "FXI"}
    _BOND_TICKERS   = {"TLT", "IEF", "HYG"}
    _COMM_TICKERS   = {"GLD", "SLV", "USO", "DBC", "CORN", "SOYB"}

    for ticker, data in ret_data.items():
        sig = _compute_cta_signal(ticker, data)
        result.signals[ticker] = sig

        if ticker in _EQUITY_TICKERS:
            equity_scores.append(sig.cta_score)
        elif ticker in _BOND_TICKERS:
            bond_scores.append(sig.cta_score)
        elif ticker in _COMM_TICKERS:
            commodity_scores.append(sig.cta_score)

        if sig.crowding == "extreme_short":
            result.extreme_shorts.append(ticker)
        elif sig.crowding == "extreme_long":
            result.extreme_longs.append(ticker)

        if sig.reversal_probability > 0.45:
            result.squeeze_candidates.append(ticker)

    # Aggregate scores
    result.aggregate_equity_score    = sum(equity_scores) / len(equity_scores) if equity_scores else 0.0
    result.aggregate_bond_score      = sum(bond_scores) / len(bond_scores) if bond_scores else 0.0
    result.aggregate_commodity_score = sum(commodity_scores) / len(commodity_scores) if commodity_scores else 0.0

    _log.info("cta_positioning_done",
              n_signals=len(result.signals),
              equity_score=round(result.aggregate_equity_score, 3),
              extreme_shorts=result.extreme_shorts,
              squeeze_candidates=result.squeeze_candidates)

    return result


def get_cta_signal_for_ticker(
    ticker: str,
    cta_result: CTAPositioningResult,
) -> float:
    """
    Retorna o sinal CTA para uso no alpha_signals.
    Score [-1, 1]:
      +1: CTA extreme long (risco de selling pressure se virarem)
      -1: CTA extreme short (potencial squeeze / covering rally)
      0: neutro
    A perspectiva aqui e CONTRARIAN ao CTA:
      - Se CTAs estao extreme short -> sinal de compra (eles precisarao cobrir)
      - Se CTAs estao extreme long  -> sinal de venda (eles precisarao vender)
    Escala pelo flow_surprise e reversal_probability.
    """
    sig = cta_result.signals.get(ticker)
    if sig is None:
        return 0.0

    # Sinal contrarian pesado pela probabilidade de reversao
    contrarian = -sig.cta_score * sig.reversal_probability * 2
    return max(-1.0, min(1.0, contrarian))
