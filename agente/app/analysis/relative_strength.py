"""
Relative Strength Engine — RRG (Relative Rotation Graph)

Metodologia Julius de Kempenaer (criador do RRG, usado por Bloomberg, StockCharts, etc.)

Por que RS domina o mercado hoje:
  - CTAs, risk parity, momentum funds TODOS usam RS como sinal primario
  - Fluxo de capital segue o ativo mais forte vs benchmark
  - RS percentile > 75 = ativo atraindo capital sistematicamente
  - Rotating from Lagging → Improving → Leading = melhor timing de entrada

Calculos:
  1. RS bruto = Close_ativo / Close_benchmark (SPY ou setor)
  2. JdK RS-Ratio = RS normalizado via EMA(10)/EMA(26) — onde o ativo ESTA
  3. JdK RS-Momentum = ROC(1) do RS-Ratio — para onde o ativo VAI
  4. Quadrante RRG = (RS-Ratio vs 100, RS-Momentum vs 100)
  5. RS Percentile = rank no universo (0-100)
  6. RS Tail = vetor de direcao dos ultimos 5 periodos (trailing)

4 Quadrantes do RRG:
  - LEADING   (RS-Ratio>100, RS-Mom>100): forte e acelerando → LONG agora
  - WEAKENING (RS-Ratio>100, RS-Mom<100): forte mas desacelerando → preparar saida
  - LAGGING   (RS-Ratio<100, RS-Mom<100): fraco e piorando → SHORT
  - IMPROVING (RS-Ratio<100, RS-Mom>100): fraco mas recuperando → watch / entrada

Rotacao tipica: Lagging → Improving → Leading → Weakening → Lagging

Alpha sinal derivado:
  - Leading + RS%ile > 80  : composite boost +0.4
  - Improving + acelerando : composite boost +0.2 (entrada antecipada)
  - Weakening + RS%ile < 40: composite penalidade -0.2
  - Lagging + RS%ile < 20  : composite boost SHORT -0.4
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.relative_strength")


@dataclass
class RSSignal:
    ticker: str
    benchmark: str              # SPY normalmente

    # Metricas RS
    rs_ratio: float             # JdK RS-Ratio (pivot: 100)
    rs_momentum: float          # JdK RS-Momentum (pivot: 100)
    rs_raw: float               # RS bruto (preco relativo)
    rs_percentile: float        # 0-100: rank no universo

    # Quadrante RRG
    quadrant: str               # leading | weakening | lagging | improving
    rotation_direction: str     # clockwise | counter-clockwise | undefined

    # Tail (direcao recente)
    tail_rs_ratio: list[float]  # ultimos 5 valores de RS-Ratio
    tail_rs_momentum: list[float]

    # Performance relativa
    perf_1w_vs_bench: float | None = None    # outperformance 1w vs SPY
    perf_1m_vs_bench: float | None = None
    perf_3m_vs_bench: float | None = None
    perf_ytd_vs_bench: float | None = None

    # Alpha signal derivado
    rs_alpha_score: float = 0.0    # [-1, 1] para o composite
    conviction: str = "low"        # high | medium | low

    rationale: list[str] = field(default_factory=list)


@dataclass
class RRGResult:
    signals: dict[str, RSSignal] = field(default_factory=dict)
    benchmark: str = "SPY"

    # Quadrant summary
    leading:   list[str] = field(default_factory=list)
    improving: list[str] = field(default_factory=list)
    weakening: list[str] = field(default_factory=list)
    lagging:   list[str] = field(default_factory=list)

    # Top RS
    top_rs_long:  list[str] = field(default_factory=list)   # RS percentile > 75
    top_rs_short: list[str] = field(default_factory=list)   # RS percentile < 25

    timestamp: str = ""
    errors: list[str] = field(default_factory=list)


# ── EMA calculation ───────────────────────────────────────────────────────────

def _ema(values: list[float], period: int) -> list[float]:
    """Exponential Moving Average."""
    if len(values) < period:
        return [float("nan")] * len(values)
    k = 2.0 / (period + 1)
    result = [float("nan")] * len(values)
    # Seed com SMA dos primeiros `period` valores
    sma = sum(values[:period]) / period
    result[period - 1] = sma
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def _roc(values: list[float], period: int = 1) -> list[float]:
    """Rate of Change (%), sem nan padding."""
    result = [float("nan")] * len(values)
    for i in range(period, len(values)):
        if values[i - period] and not math.isnan(values[i - period]):
            result[i] = ((values[i] / values[i - period]) - 1.0) * 100.0
    return result


# ── Fetch price history ────────────────────────────────────────────────────────

def _fetch_prices(tickers: list[str], days: int = 280) -> dict[str, list[float]]:
    """Busca historico de precos. Returns dict ticker → list of close prices."""
    try:
        import yfinance as yf
        import datetime as dt

        end = dt.datetime.now()
        start = end - dt.timedelta(days=days)

        all_tickers = list(set(tickers))
        if not all_tickers:
            return {}

        data = yf.download(
            all_tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            return {}

        result: dict[str, list[float]] = {}

        if hasattr(data.columns, "levels"):
            close_df = data["Close"]
        else:
            close_df = data

        for ticker in all_tickers:
            if ticker in close_df.columns:
                series = close_df[ticker].dropna()
            elif len(all_tickers) == 1:
                series = close_df.squeeze().dropna()
            else:
                continue
            if len(series) >= 30:
                result[ticker] = [float(v) for v in series.values]

        return result

    except Exception as exc:
        _log.warning("rs_price_fetch_failed", error=str(exc)[:80])
        return {}


# ── JdK RS-Ratio and RS-Momentum ──────────────────────────────────────────────

def _compute_jdk_rs(
    asset_prices: list[float],
    bench_prices: list[float],
    period_short: int = 10,
    period_long: int = 26,
) -> tuple[list[float], list[float]]:
    """
    Calcula JdK RS-Ratio e RS-Momentum.

    RS-Ratio: normalizado para 100
      ratio = asset_close / bench_close
      rs_ratio_raw = EMA_short(ratio) / EMA_long(ratio) * 100

    RS-Momentum: taxa de mudanca do RS-Ratio
      rs_momentum = ROC(1, rs_ratio) normalizado para 100
    """
    # Alinha pelo comprimento menor
    n = min(len(asset_prices), len(bench_prices))
    if n < period_long + 10:
        return [], []

    asset_prices = asset_prices[-n:]
    bench_prices = bench_prices[-n:]

    # Raw RS
    rs_raw = [a / b if b > 0 else 1.0 for a, b in zip(asset_prices, bench_prices)]

    # EMA curta e longa do RS
    ema_short = _ema(rs_raw, period_short)
    ema_long  = _ema(rs_raw, period_long)

    # RS-Ratio: EMA_short / EMA_long * 100 (normalizado em torno de 100)
    rs_ratio = []
    for es, el in zip(ema_short, ema_long):
        if not math.isnan(es) and not math.isnan(el) and el > 0:
            rs_ratio.append(es / el * 100.0)
        else:
            rs_ratio.append(float("nan"))

    # RS-Momentum: ROC(1) do RS-Ratio, normalizado para 100
    rs_ratio_clean = [v for v in rs_ratio if not math.isnan(v)]
    if len(rs_ratio_clean) < 2:
        return rs_ratio, [100.0] * len(rs_ratio)

    # Normaliza RS-Momentum: 100 = sem mudanca, >100 = acelerando
    roc_raw = _roc(rs_ratio, 1)
    rs_momentum = []
    for r in roc_raw:
        if math.isnan(r):
            rs_momentum.append(float("nan"))
        else:
            rs_momentum.append(100.0 + r * 10)  # amplificado x10 para legibilidade

    return rs_ratio, rs_momentum


def _rotation_direction(
    tail_ratio: list[float], tail_mom: list[float]
) -> str:
    """
    Detecta direcao de rotacao no RRG (clockwise vs counter-clockwise).
    Clockwise = saudavel (Leading→Weakening→Lagging→Improving→Leading).
    """
    if len(tail_ratio) < 3:
        return "undefined"

    # Calcula cross product dos ultimos 3 pontos para detectar rotacao
    # Se cross product > 0: counter-clockwise | < 0: clockwise
    p1 = (tail_ratio[-3], tail_mom[-3])
    p2 = (tail_ratio[-2], tail_mom[-2])
    p3 = (tail_ratio[-1], tail_mom[-1])

    cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])

    if abs(cross) < 0.001:
        return "undefined"
    return "counter-clockwise" if cross > 0 else "clockwise"


def _quadrant(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "leading"
    elif rs_ratio >= 100 and rs_momentum < 100:
        return "weakening"
    elif rs_ratio < 100 and rs_momentum < 100:
        return "lagging"
    else:  # rs_ratio < 100 and rs_momentum >= 100
        return "improving"


def _rs_alpha_score(
    quadrant: str,
    rs_percentile: float,
    rotation: str,
    tail_ratio: list[float],
    tail_mom: list[float],
) -> tuple[float, str, list[str]]:
    """
    Score alpha derivado do RS para uso no composite signal.
    Returns: (score [-1,1], conviction, reasons)
    """
    reasons = []
    base = 0.0

    # Quadrant base score
    if quadrant == "leading":
        base = 0.50
        reasons.append(f"RRG: LEADING — RS forte e acelerando")
    elif quadrant == "improving":
        base = 0.20
        reasons.append(f"RRG: IMPROVING — RS fraco mas recuperando")
    elif quadrant == "weakening":
        base = -0.10
        reasons.append(f"RRG: WEAKENING — RS forte mas desacelerando")
    elif quadrant == "lagging":
        base = -0.45
        reasons.append(f"RRG: LAGGING — RS fraco e piorando")

    # RS Percentile adjustment
    if rs_percentile >= 80:
        base += 0.20
        reasons.append(f"RS%ile={rs_percentile:.0f} — top 20% do universo")
    elif rs_percentile >= 65:
        base += 0.10
        reasons.append(f"RS%ile={rs_percentile:.0f} — acima da media")
    elif rs_percentile <= 20:
        base -= 0.20
        reasons.append(f"RS%ile={rs_percentile:.0f} — bottom 20% do universo")
    elif rs_percentile <= 35:
        base -= 0.10

    # Rotation bonus: clockwise = saudavel
    if rotation == "clockwise" and quadrant in ("improving", "leading"):
        base += 0.10
        reasons.append("Rotacao clockwise: momentum crescente saudavel")
    elif rotation == "counter-clockwise" and quadrant in ("weakening", "lagging"):
        base -= 0.10
        reasons.append("Rotacao counter-clockwise: deterioracao confirmada")

    # Momentum acelerando dentro do quadrante
    if len(tail_mom) >= 3:
        mom_trend = tail_mom[-1] - tail_mom[-3]
        if mom_trend > 2 and quadrant == "improving":
            base += 0.15
            reasons.append(f"Improving acelerando: mom_trend=+{mom_trend:.1f}")
        elif mom_trend < -2 and quadrant == "weakening":
            base -= 0.15
            reasons.append(f"Weakening acelerando: mom_trend={mom_trend:.1f}")

    score = max(-1.0, min(1.0, base))

    # Conviction
    if abs(score) > 0.45:
        conviction = "high"
    elif abs(score) > 0.20:
        conviction = "medium"
    else:
        conviction = "low"

    return score, conviction, reasons


# ── Main compute function ──────────────────────────────────────────────────────

def compute_relative_strength(
    tickers: list[str],
    benchmark: str = "SPY",
    market_prices: dict | None = None,
) -> RRGResult:
    """
    Calcula RS/RRG para todos os tickers.

    tickers: lista de ativos a analisar
    benchmark: ativo de referencia (default: SPY)
    market_prices: usado para complementar se disponivel
    """
    result = RRGResult(benchmark=benchmark, timestamp=datetime.now().isoformat())

    # Busca precos historicos
    all_tickers = list(set(tickers + [benchmark]))
    price_data = _fetch_prices(all_tickers, days=280)

    bench_prices = price_data.get(benchmark)
    if not bench_prices or len(bench_prices) < 50:
        result.errors.append(f"Benchmark {benchmark} sem dados suficientes")
        return result

    # Coleta retornos relativos para calcular percentil
    rs_ratio_current: dict[str, float] = {}

    for ticker in tickers:
        if ticker == benchmark:
            continue

        asset_prices = price_data.get(ticker)
        if not asset_prices or len(asset_prices) < 50:
            continue

        try:
            rs_ratio_vals, rs_mom_vals = _compute_jdk_rs(asset_prices, bench_prices)

            if not rs_ratio_vals or len(rs_ratio_vals) < 10:
                continue

            # Valores mais recentes (ignorando NaN)
            clean_ratio = [(i, v) for i, v in enumerate(rs_ratio_vals) if not math.isnan(v)]
            clean_mom   = [(i, v) for i, v in enumerate(rs_mom_vals)   if not math.isnan(v)]

            if not clean_ratio or not clean_mom:
                continue

            current_ratio = clean_ratio[-1][1]
            current_mom   = clean_mom[-1][1]

            rs_ratio_current[ticker] = current_ratio

            # Tail: ultimos 5 valores
            tail_ratio = [v for _, v in clean_ratio[-5:]]
            tail_mom   = [v for _, v in clean_mom[-5:]]

            # Raw RS (ultimo valor)
            n = min(len(asset_prices), len(bench_prices))
            raw_rs = asset_prices[-1] / bench_prices[-1] if bench_prices[-1] > 0 else 1.0

            # Performance relativa
            def rel_perf(n_days: int) -> float | None:
                try:
                    a_n = asset_prices[-(n_days + 1)]
                    b_n = bench_prices[-(n_days + 1)]
                    a_0 = asset_prices[-1]
                    b_0 = bench_prices[-1]
                    if a_n <= 0 or b_n <= 0:
                        return None
                    return (a_0 / a_n - 1) - (b_0 / b_n - 1)
                except (IndexError, ZeroDivisionError):
                    return None

        except Exception as exc:
            _log.debug("rs_compute_failed", ticker=ticker, error=str(exc)[:60])
            continue

    # ── Calcula percentis agora que temos todos os RS-Ratio ────────────────────
    if not rs_ratio_current:
        result.errors.append("Nenhum ticker com RS calculado")
        return result

    sorted_values = sorted(rs_ratio_current.values())
    n_univ = len(sorted_values)

    def percentile_rank(val: float) -> float:
        """Percentile rank [0, 100] de val na distribuicao."""
        below = sum(1 for v in sorted_values if v < val)
        return (below / n_univ) * 100.0

    # ── Constroi RSSignal para cada ticker ─────────────────────────────────────
    for ticker in tickers:
        if ticker == benchmark or ticker not in rs_ratio_current:
            continue

        asset_prices = price_data.get(ticker)
        if not asset_prices:
            continue

        try:
            rs_ratio_vals, rs_mom_vals = _compute_jdk_rs(asset_prices, bench_prices)
            clean_ratio = [(i, v) for i, v in enumerate(rs_ratio_vals) if not math.isnan(v)]
            clean_mom   = [(i, v) for i, v in enumerate(rs_mom_vals)   if not math.isnan(v)]

            if not clean_ratio or not clean_mom:
                continue

            current_ratio = clean_ratio[-1][1]
            current_mom   = clean_mom[-1][1]
            tail_ratio    = [v for _, v in clean_ratio[-5:]]
            tail_mom      = [v for _, v in clean_mom[-5:]]

            raw_rs       = asset_prices[-1] / bench_prices[-1] if bench_prices[-1] > 0 else 1.0
            rs_pct       = percentile_rank(current_ratio)
            quad         = _quadrant(current_ratio, current_mom)
            rotation     = _rotation_direction(tail_ratio, tail_mom)
            alpha, conv, reasons = _rs_alpha_score(
                quad, rs_pct, rotation, tail_ratio, tail_mom
            )

            def rel_perf(n_days: int) -> float | None:
                try:
                    a_n = asset_prices[-(n_days + 1)]
                    b_n = bench_prices[-(n_days + 1)]
                    a_0 = asset_prices[-1]
                    b_0 = bench_prices[-1]
                    if a_n <= 0 or b_n <= 0:
                        return None
                    return (a_0 / a_n - 1) - (b_0 / b_n - 1)
                except (IndexError, ZeroDivisionError):
                    return None

            sig = RSSignal(
                ticker=ticker,
                benchmark=benchmark,
                rs_ratio=round(current_ratio, 2),
                rs_momentum=round(current_mom, 2),
                rs_raw=round(raw_rs, 4),
                rs_percentile=round(rs_pct, 1),
                quadrant=quad,
                rotation_direction=rotation,
                tail_rs_ratio=[round(v, 2) for v in tail_ratio],
                tail_rs_momentum=[round(v, 2) for v in tail_mom],
                perf_1w_vs_bench=rel_perf(5),
                perf_1m_vs_bench=rel_perf(21),
                perf_3m_vs_bench=rel_perf(63),
                perf_ytd_vs_bench=rel_perf(252),
                rs_alpha_score=round(alpha, 3),
                conviction=conv,
                rationale=reasons,
            )
            result.signals[ticker] = sig

            # Quadrant lists
            if quad == "leading":
                result.leading.append(ticker)
            elif quad == "improving":
                result.improving.append(ticker)
            elif quad == "weakening":
                result.weakening.append(ticker)
            elif quad == "lagging":
                result.lagging.append(ticker)

            if rs_pct >= 75:
                result.top_rs_long.append(ticker)
            elif rs_pct <= 25:
                result.top_rs_short.append(ticker)

        except Exception as exc:
            _log.debug("rs_signal_failed", ticker=ticker, error=str(exc)[:60])
            result.errors.append(f"{ticker}: {exc}")

    # Sort by RS-Ratio desc
    result.top_rs_long.sort(
        key=lambda t: result.signals[t].rs_ratio if t in result.signals else 0, reverse=True
    )
    result.top_rs_short.sort(
        key=lambda t: result.signals[t].rs_ratio if t in result.signals else 999
    )
    result.leading.sort(
        key=lambda t: result.signals[t].rs_percentile if t in result.signals else 0, reverse=True
    )
    result.lagging.sort(
        key=lambda t: result.signals[t].rs_percentile if t in result.signals else 100
    )

    _log.info("rrg_done",
              n_signals=len(result.signals),
              leading=len(result.leading),
              improving=len(result.improving),
              weakening=len(result.weakening),
              lagging=len(result.lagging),
              top_long=result.top_rs_long[:3],
              top_short=result.top_rs_short[:3])

    return result


def get_rs_signal_for_ticker(ticker: str, rrg: RRGResult) -> float:
    """
    Retorna o RS alpha score para uso no alpha_signals composite.
    Score [-1, 1].
    """
    sig = rrg.signals.get(ticker)
    if sig is None:
        return 0.0
    return sig.rs_alpha_score
