"""
Chart Regime Detector

Lê o gráfico como evidência primária de regime narrativo.
O gráfico não é enfeite — é a expressão final da disputa entre narrativa, fluxo e posicionamento.

Detecta:
  1. Aceleração de tendência (slope change)
  2. Compressão de range antes de rompimento
  3. Expansão de range pós-rompimento
  4. Falha de rompimento (breakout failure)
  5. Candle de exaustão (climax bar)
  6. Distância das médias de referência (overextension)
  7. Perda de eficiência do movimento (velocity decay)
  8. Mudança de comportamento pós-evento

Output: ChartRegime por ticker com regime_label e supporting scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.chart_regime")


# ── Regimes gráficos ──────────────────────────────────────────────────────────

class ChartRegimeLabel:
    TREND_ACCELERATION  = "tendência_acelerando"   # momentum crescente, entrada válida
    TREND_DECELERATION  = "tendência_desacelerando" # slope caindo, preparar saída
    COMPRESSION         = "compressão"              # range estreitando, rompimento iminente
    BREAKOUT_VALID      = "rompimento_válido"       # expansão de range + volume confirmando
    BREAKOUT_FAILURE    = "rompimento_falho"        # rompeu mas reverteu — perigoso
    OVEREXTENDED        = "sobreextendido"           # muito longe das médias, reversão possível
    CLIMAX_EXHAUSTION   = "exaustão_clímax"         # barra de climax, reversão provável
    VELOCITY_DECAY      = "decaimento_de_velocidade" # movimento perde eficiência
    RANGE_EXPANSION     = "expansão_de_range"        # volatilidade crescente, regime em mudança
    MEAN_REVERSION_ZONE = "zona_de_reversão"         # próximo das médias, suporte/resistência
    UNDEFINED           = "indefinido"


@dataclass
class ChartRegime:
    ticker: str
    regime_label: str = ChartRegimeLabel.UNDEFINED

    # Scores derivados [0, 1]
    trend_strength: float    = 0.0   # força da tendência atual
    acceleration: float      = 0.0   # aceleração (positivo = acelerando)
    compression_score: float = 0.0   # quanto range comprimido (maior = mais comprimido)
    overextension: float     = 0.0   # distância das médias (> 0.7 = sobreextendido)
    exhaustion_signal: float = 0.0   # sinal de exaustão gráfica (climax, failure)
    velocity_score: float    = 0.0   # eficiência do movimento (ret/vol ratio)

    # Flags binárias
    is_compressing: bool   = False
    is_breaking_out: bool  = False
    breakout_failed: bool  = False
    climax_bar: bool       = False
    is_overextended: bool  = False
    losing_efficiency: bool = False

    # Médias de referência (distância relativa)
    dist_ma20_pct: float   = 0.0
    dist_ma50_pct: float   = 0.0
    dist_ma200_pct: float  = 0.0

    # Contexto
    price_series_len: int  = 0
    rationale: str         = ""


# ── Motor principal ───────────────────────────────────────────────────────────

def detect_chart_regimes(
    market_prices: dict[str, Any] | None,
    lookback: int = 60,
) -> dict[str, ChartRegime]:
    """
    Detecta regime gráfico para cada ticker em market_prices.

    market_prices: {ticker: {prices: [float], volumes: [float], dates: [str]}}
    Returns: {ticker: ChartRegime}
    """
    if not market_prices:
        return {}

    results: dict[str, ChartRegime] = {}
    for ticker, data in market_prices.items():
        try:
            cr = _detect_single(ticker, data, lookback)
            results[ticker] = cr
        except Exception as exc:
            _log.warning("chart_regime_error", ticker=ticker, error=str(exc))
            results[ticker] = ChartRegime(ticker=ticker)

    _log.info("chart_regime_done",
              tickers=len(results),
              breaking_out=sum(1 for r in results.values() if r.is_breaking_out),
              compressing=sum(1 for r in results.values() if r.is_compressing),
              overextended=sum(1 for r in results.values() if r.is_overextended),
              exhaustion=sum(1 for r in results.values() if r.exhaustion_signal > 0.5))
    return results


def _detect_single(ticker: str, data: Any, lookback: int) -> ChartRegime:
    cr = ChartRegime(ticker=ticker)

    # Extrai séries de preço
    prices  = _extract_prices(data, lookback)
    volumes = _extract_volumes(data, lookback)

    if len(prices) < 10:
        return cr

    cr.price_series_len = len(prices)

    # ── Médias móveis ─────────────────────────────────────────────────────────
    p_last = prices[-1]
    ma20  = _sma(prices, 20)
    ma50  = _sma(prices, 50)
    ma200 = _sma(prices, 200)

    if ma20 and p_last:
        cr.dist_ma20_pct = (p_last - ma20) / ma20
    if ma50 and p_last:
        cr.dist_ma50_pct = (p_last - ma50) / ma50
    if ma200 and p_last:
        cr.dist_ma200_pct = (p_last - ma200) / ma200

    # ── Overextension ─────────────────────────────────────────────────────────
    # Usar distância da MA50 como proxy principal
    if ma50:
        dist = abs(cr.dist_ma50_pct)
        cr.overextension = min(dist / 0.20, 1.0)   # 20% = máximo
        cr.is_overextended = dist > 0.12

    # ── Retornos e velocidade ─────────────────────────────────────────────────
    rets = _returns(prices)
    if not rets:
        return cr

    # Slope recente (regressão linear simples sobre últimos 20 retornos)
    recent_rets = rets[-20:]
    slope_recent = _linear_slope(recent_rets)
    slope_full   = _linear_slope(rets[-40:]) if len(rets) >= 40 else slope_recent

    # Aceleração = mudança do slope
    acceleration = slope_recent - slope_full
    cr.acceleration = max(-1.0, min(1.0, acceleration * 100))

    # Força da tendência: média dos retornos recentes normalizada
    mean_ret = sum(recent_rets) / len(recent_rets) if recent_rets else 0
    vol_ret  = _std(recent_rets) or 0.001
    cr.trend_strength = min(abs(mean_ret / vol_ret) / 3, 1.0)  # sharpe ratio normalizado

    # Velocidade = ret/vol ratio (eficiência do movimento)
    cr.velocity_score = min(abs(mean_ret) / (vol_ret + 1e-8) / 2, 1.0)

    # Perda de eficiência: comparar velocity últimas 5 vs últimas 20 barras
    if len(rets) >= 10:
        vel_recent = _velocity(rets[-5:])
        vel_medium = _velocity(rets[-20:])
        if vel_medium > 0.01 and vel_recent < vel_medium * 0.5:
            cr.losing_efficiency = True

    # ── Compressão de range ───────────────────────────────────────────────────
    # ATR atual vs ATR histórico (40 barras)
    atr_recent = _atr(prices[-15:]) if len(prices) >= 15 else None
    atr_hist   = _atr(prices[-40:]) if len(prices) >= 40 else atr_recent
    if atr_recent and atr_hist and atr_hist > 0:
        compression_ratio = atr_recent / atr_hist
        cr.compression_score = max(0, 1.0 - compression_ratio)   # menor ATR = mais comprimido
        cr.is_compressing = compression_ratio < 0.6

    # ── Rompimento ────────────────────────────────────────────────────────────
    if len(prices) >= 20:
        high20 = max(prices[-21:-1])  # high das últimas 20 barras (excluindo a atual)
        low20  = min(prices[-21:-1])
        range20 = high20 - low20 if (high20 - low20) > 0 else 1
        current_atr = _atr(prices[-5:]) or (range20 * 0.02)

        if p_last > high20 + current_atr * 0.5:
            cr.is_breaking_out = True
        elif p_last < low20 - current_atr * 0.5:
            cr.is_breaking_out = True  # rompimento de suporte (bearish)

        # Falha de rompimento: preço voltou para dentro do range após romper
        if len(prices) >= 3:
            prev_high = max(prices[-22:-2]) if len(prices) >= 22 else high20
            if prices[-2] > prev_high and prices[-1] < prev_high:
                cr.breakout_failed = True

    # ── Candle de exaustão (climax bar) ──────────────────────────────────────
    if len(rets) >= 5 and volumes:
        last_ret = abs(rets[-1])
        avg_ret  = _std(rets[-20:]) or 0.001
        last_vol = volumes[-1] if volumes else 0
        avg_vol  = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else last_vol

        # Climax: retorno > 2x desvio padrão E volume > 2x médio
        if last_ret > avg_ret * 2.0 and avg_vol > 0 and last_vol > avg_vol * 1.8:
            cr.climax_bar = True

    # ── Exhaustion signal composto ────────────────────────────────────────────
    exhaust = 0.0
    if cr.is_overextended:
        exhaust += 0.35
    if cr.climax_bar:
        exhaust += 0.30
    if cr.breakout_failed:
        exhaust += 0.25
    if cr.losing_efficiency:
        exhaust += 0.10
    cr.exhaustion_signal = min(exhaust, 1.0)

    # ── Regime label final ────────────────────────────────────────────────────
    cr.regime_label = _classify_regime(cr)
    cr.rationale = _build_rationale(cr)

    return cr


# ── Classificação do regime ───────────────────────────────────────────────────

def _classify_regime(cr: ChartRegime) -> str:
    if cr.climax_bar and cr.is_overextended:
        return ChartRegimeLabel.CLIMAX_EXHAUSTION
    if cr.breakout_failed:
        return ChartRegimeLabel.BREAKOUT_FAILURE
    if cr.is_overextended and cr.losing_efficiency:
        return ChartRegimeLabel.VELOCITY_DECAY
    if cr.is_overextended:
        return ChartRegimeLabel.OVEREXTENDED
    if cr.is_breaking_out and not cr.is_compressing:
        return ChartRegimeLabel.BREAKOUT_VALID
    if cr.is_compressing:
        return ChartRegimeLabel.COMPRESSION
    if cr.acceleration > 0.15 and cr.trend_strength > 0.5:
        return ChartRegimeLabel.TREND_ACCELERATION
    if cr.acceleration < -0.15 and cr.trend_strength > 0.3:
        return ChartRegimeLabel.TREND_DECELERATION
    if abs(cr.dist_ma20_pct) < 0.02:
        return ChartRegimeLabel.MEAN_REVERSION_ZONE
    return ChartRegimeLabel.UNDEFINED


def _build_rationale(cr: ChartRegime) -> str:
    parts = []
    if cr.is_breaking_out:
        parts.append("Rompimento de range detectado.")
    if cr.breakout_failed:
        parts.append("Falha de rompimento — reversão para dentro do range.")
    if cr.is_compressing:
        parts.append(f"Range comprimido ({cr.compression_score:.0%} vs histórico) — rompimento iminente.")
    if cr.is_overextended:
        parts.append(f"Sobreextendido: {cr.dist_ma50_pct:+.1%} da MA50.")
    if cr.climax_bar:
        parts.append("Barra de clímax detectada — possível reversão.")
    if cr.losing_efficiency:
        parts.append("Movimento perdendo eficiência — momentum decaindo.")
    if cr.acceleration > 0.15:
        parts.append("Tendência acelerando.")
    elif cr.acceleration < -0.15:
        parts.append("Tendência desacelerando.")
    return " ".join(parts) if parts else "Sem sinal gráfico definido."


# ── Utilitários ───────────────────────────────────────────────────────────────

def _extract_prices(data: Any, lookback: int) -> list[float]:
    """Extrai série de preços do formato de market_prices."""
    if isinstance(data, dict):
        for key in ("prices", "close", "Close", "adjClose"):
            v = data.get(key)
            if v and isinstance(v, list):
                return [float(x) for x in v if x is not None][-lookback:]
        # Formato alternativo: {date: price}
        if data and not any(k in data for k in ("prices","close","Close","adjClose")):
            vals = list(data.values())
            if vals and isinstance(vals[0], (int, float)):
                return [float(x) for x in vals if x is not None][-lookback:]
    if isinstance(data, list):
        return [float(x) for x in data if x is not None][-lookback:]
    return []


def _extract_volumes(data: Any, lookback: int) -> list[float]:
    if isinstance(data, dict):
        for key in ("volumes", "volume", "Volume"):
            v = data.get(key)
            if v and isinstance(v, list):
                return [float(x) for x in v if x is not None][-lookback:]
    return []


def _sma(prices: list[float], period: int) -> float | None:
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def _returns(prices: list[float]) -> list[float]:
    return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] != 0]


def _std(data: list[float]) -> float:
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    return math.sqrt(sum((x - mean)**2 for x in data) / (len(data) - 1))


def _atr(prices: list[float]) -> float | None:
    if len(prices) < 2:
        return None
    ranges = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
    return sum(ranges) / len(ranges) if ranges else None


def _linear_slope(data: list[float]) -> float:
    """Slope da regressão linear simples (normalizado pelo std)."""
    n = len(data)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(data) / n
    num = sum((xs[i] - mx) * (data[i] - my) for i in range(n))
    den = sum((xs[i] - mx)**2 for i in range(n))
    if den == 0:
        return 0.0
    slope = num / den
    # Normaliza pelo std dos dados
    s = _std(data) or 1.0
    return slope / s


def _velocity(rets: list[float]) -> float:
    """Retorno médio / volatilidade (Sharpe-like)."""
    if not rets:
        return 0.0
    mean = sum(rets) / len(rets)
    vol  = _std(rets) or 0.001
    return abs(mean / vol)
