"""
TV Zone Filter — Ajuste fino via TradingView (Value Area, VWAP, RSI, Anchored VWAPs)

Princípio: o preço é a última palavra. Todos os sinais de macro, fluxo e opções
apontam o que DEVE acontecer. O TradingView mostra ONDE comprar/vender dentro
dessa tese — a zona de valor é o ponto de entrada com risco definido.

Para cada ativo com snapshot TV disponível, computa um ZoneSignal com:

  1. entry_quality: "ideal" | "acceptable" | "stretched" | "avoid"
     - ideal: preço dentro da Value Area, acima do POC (long) ou abaixo (short)
              + VWAP confirmando + RSI sem extremo
     - acceptable: preço próximo do VAL/VAH, zona de suporte/resistência confirmada
     - stretched: preço muito afastado da VA, overextended
     - avoid: condições adversas (preço abaixo de VAL para long, RSI 80+ etc.)

  2. position_bias: ajuste multiplicativo [-0.4, +0.3] ao composite do alpha_signal
     - +0.30: setup ideal + tendência confirmada + VWAP + volume
     - +0.10 a +0.20: setup acceptable
     - -0.10 a -0.20: stretched (pedir esperar por pullback)
     - -0.30 a -0.40: avoid (não adiciona posição agora)

  3. allocation_scalar: multiplicador [0.5, 1.3] para a alocação do optimizer
     - 1.25-1.30: ideal — full size
     - 1.00: acceptable — tamanho normal
     - 0.70-0.80: stretched — tamanho reduzido (aguardar retorno à VA)
     - 0.50: avoid — posição mínima ou zero

  4. refined_stop: stop ajustado pela estrutura do gráfico
     - Long: abaixo do VAL ou do suporte VWAP (o que for mais alto)
     - Short: acima do VAH ou da resistência VWAP (o que for mais baixo)

  5. first_target: primeiro alvo natural da estrutura
     - Long: POC → VAH → próximo anchor VWAP acima
     - Short: POC → VAL → próximo anchor VWAP abaixo

  6. regime_consistent: se o regime TV (bullish/bearish) é consistente com
     a direção do alpha_signal (composite > 0 = long)

Lógica central:
  - Value Area é a âncora de preço mais importante (70% do volume negociado)
  - Entrar dentro da VA = menor custo de oportunidade, stop definido
  - Entrar fora da VA (breakout) = válido mas precisa momentum forte (RSI > 55)
  - VWAP é separador institucional: acima = comprador dominante, abaixo = vendedor
  - Anchored VWAPs são suporte/resistência dinâmico de alta precisão
  - RSI < 40 em tendência de alta = oportunidade (momentum retornando)
  - RSI > 75 em tendência de alta = entry chasing, evitar
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.tv_zone_filter")


# ── Constantes de limiar ──────────────────────────────────────────────────────

_VA_PROXIMITY_PCT   = 0.005  # 0.5% de distância da borda da VA = "near"
_VWAP_PROXIMITY_PCT = 0.008  # 0.8% de distância do VWAP = "near"
_OVEREXTENDED_PCT   = 0.025  # > 2.5% acima da VA = stretched
_ANCHOR_PROXIMITY   = 0.006  # 0.6% de distância de um anchor VWAP


# ── Modelo ────────────────────────────────────────────────────────────────────

@dataclass
class ZoneSignal:
    ticker: str

    # Qualidade do setup de entrada
    entry_quality: str = "no_data"    # ideal | acceptable | stretched | avoid | no_data

    # Ajuste ao composite alpha (soma, não multiplica)
    position_bias: float = 0.0        # [-0.4, +0.3]

    # Multiplicador de alocação para o optimizer
    allocation_scalar: float = 1.0    # [0.5, 1.3]

    # Stop e alvo refinados pela estrutura TV
    refined_stop: float | None = None
    first_target: float | None = None

    # Consistência de direção
    regime_consistent: bool = True    # TV regime confirma direção do alpha_signal

    # Flags diagnósticas
    within_value_area: bool = False
    above_vwap: bool = False
    near_support: bool = False        # próximo de VAL / POC / anchor abaixo
    near_resistance: bool = False     # próximo de VAH / anchor acima
    rsi_oversold: bool = False        # RSI < 38
    rsi_overbought: bool = False      # RSI > 72
    momentum_confirming: bool = False # MACD cruzado + price > VWAP
    overextended: bool = False        # > 2.5% fora da VA

    # Valores capturados (para display no MacroDesk)
    price: float = 0.0
    vwap: float | None = None
    vah: float | None = None
    val: float | None = None
    poc: float | None = None
    rsi: float | None = None
    atr: float | None = None
    anchors: list[float] = field(default_factory=list)
    tv_regime: str = "neutral"        # bullish | bearish | neutral

    # Texto legível
    rationale: str = ""


# ── Motor principal ───────────────────────────────────────────────────────────

def compute_zone_signals(
    tv_map: dict[str, dict],
    signals: dict[str, Any] | None = None,    # dict[ticker, AssetSignal]
) -> dict[str, ZoneSignal]:
    """
    Para cada ticker com snapshot TV disponível, computa um ZoneSignal.

    tv_map: output de tradingview.collect_for_positions()
    signals: dict[ticker, AssetSignal] — usado para saber a direção esperada
             (composite > 0 = long bias, < 0 = short bias)

    Retorna {ticker: ZoneSignal}
    """
    results: dict[str, ZoneSignal] = {}

    for ticker, snap in (tv_map or {}).items():
        if not snap:
            continue
        try:
            # Direção esperada do alpha_signal
            direction_bias = "long"
            if signals:
                sig = signals.get(ticker)
                if sig is not None:
                    composite = getattr(sig, "composite", None)
                    if composite is not None and float(composite) < -0.05:
                        direction_bias = "short"

            zs = _compute_single(ticker, snap, direction_bias)
            results[ticker] = zs
        except Exception as exc:
            _log.warning("zone_signal_error", ticker=ticker, error=str(exc))

    _log.info(
        "tv_zone_filter_done",
        tickers=len(results),
        ideal=sum(1 for z in results.values() if z.entry_quality == "ideal"),
        acceptable=sum(1 for z in results.values() if z.entry_quality == "acceptable"),
        stretched=sum(1 for z in results.values() if z.entry_quality == "stretched"),
        avoid=sum(1 for z in results.values() if z.entry_quality == "avoid"),
    )
    return results


def _compute_single(ticker: str, snap: dict, direction: str = "long") -> ZoneSignal:
    zs = ZoneSignal(ticker=ticker)

    price   = float(snap.get("price") or 0)
    vwap    = snap.get("vwap")
    vah     = snap.get("vah")
    val     = snap.get("val")
    poc     = snap.get("poc")
    rsi     = snap.get("rsi")
    atr     = snap.get("atr")
    macd    = snap.get("macd")
    sig_    = snap.get("macd_signal")
    anchors = snap.get("anchors") or []
    tv_reg  = snap.get("regime", "neutral")

    if not price:
        return zs

    zs.price    = price
    zs.vwap     = vwap
    zs.vah      = vah
    zs.val      = val
    zs.poc      = poc
    zs.rsi      = rsi
    zs.atr      = atr
    zs.anchors  = anchors
    zs.tv_regime = tv_reg

    # ── Flags básicas ─────────────────────────────────────────────────────────
    zs.within_value_area = bool(vah and val and val <= price <= vah)
    zs.above_vwap = bool(vwap and price > vwap)

    if rsi is not None:
        zs.rsi_oversold   = rsi < 38
        zs.rsi_overbought = rsi > 72

    if macd is not None and sig_ is not None:
        zs.momentum_confirming = (
            (macd > sig_ and zs.above_vwap and direction == "long") or
            (macd < sig_ and not zs.above_vwap and direction == "short")
        )

    # Anchors: nearest below and above price
    anchors_below = sorted([a for a in anchors if a < price], reverse=True)
    anchors_above = sorted([a for a in anchors if a > price])

    # Near support: within _ANCHOR_PROXIMITY of VAL, POC (for longs) or anchor below
    near_support_levels: list[float] = []
    if val:
        near_support_levels.append(val)
    if poc and poc < price:
        near_support_levels.append(poc)
    if anchors_below:
        near_support_levels.append(anchors_below[0])
    if vwap and not zs.above_vwap:
        near_support_levels.append(vwap)

    zs.near_support = any(
        abs(price - lvl) / price < _ANCHOR_PROXIMITY
        for lvl in near_support_levels if lvl
    )

    # Near resistance: within proximity of VAH, POC (for shorts) or anchor above
    near_resistance_levels: list[float] = []
    if vah:
        near_resistance_levels.append(vah)
    if poc and poc > price:
        near_resistance_levels.append(poc)
    if anchors_above:
        near_resistance_levels.append(anchors_above[0])
    if vwap and zs.above_vwap:
        near_resistance_levels.append(vwap)

    zs.near_resistance = any(
        abs(price - lvl) / price < _ANCHOR_PROXIMITY
        for lvl in near_resistance_levels if lvl
    )

    # Overextended: > 2.5% above VAH (long) or below VAL (short)
    if vah and val:
        if direction == "long" and price > vah:
            zs.overextended = (price - vah) / vah > _OVEREXTENDED_PCT
        elif direction == "short" and price < val:
            zs.overextended = (val - price) / val > _OVEREXTENDED_PCT

    # Regime consistent: TV regime aligns with direction
    zs.regime_consistent = (
        (direction == "long" and tv_reg == "bullish") or
        (direction == "short" and tv_reg == "bearish") or
        tv_reg == "neutral"
    )

    # ── Entry quality ─────────────────────────────────────────────────────────
    zs.entry_quality = _classify_entry(zs, direction)

    # ── Stop refinado ─────────────────────────────────────────────────────────
    zs.refined_stop = _compute_stop(price, direction, vwap, vah, val, poc, anchors_below, anchors_above, atr)

    # ── Primeiro alvo ─────────────────────────────────────────────────────────
    zs.first_target = _compute_target(price, direction, vwap, vah, val, poc, anchors_below, anchors_above)

    # ── Position bias e allocation scalar ─────────────────────────────────────
    zs.position_bias, zs.allocation_scalar = _compute_adjustments(zs, direction)

    # ── Rationale ─────────────────────────────────────────────────────────────
    zs.rationale = _build_rationale(zs, direction)

    return zs


# ── Classificação ─────────────────────────────────────────────────────────────

def _classify_entry(zs: ZoneSignal, direction: str) -> str:
    price = zs.price
    vah   = zs.vah
    val   = zs.val
    poc   = zs.poc
    vwap  = zs.vwap

    # AVOID: condições adversas claras
    if direction == "long":
        if zs.rsi_overbought and not zs.within_value_area:
            return "avoid"
        if zs.overextended:
            return "avoid"
        if not zs.regime_consistent and zs.tv_regime == "bearish":
            return "avoid"
        # Preço muito abaixo do VAL sem suporte identificado
        if val and vwap and price < val and price < vwap and not zs.near_support:
            return "avoid"
    else:  # short
        if zs.rsi_oversold and not zs.within_value_area:
            return "avoid"
        if zs.overextended:
            return "avoid"
        if not zs.regime_consistent and zs.tv_regime == "bullish":
            return "avoid"

    # IDEAL: dentro da VA + VWAP confirmando + sem extremos RSI + momentum
    if zs.within_value_area:
        vwap_ok = (
            (direction == "long" and zs.above_vwap) or
            (direction == "short" and not zs.above_vwap) or
            vwap is None
        )
        rsi_ok = not zs.rsi_overbought and not zs.rsi_oversold
        poc_ok = bool(
            poc and (
                (direction == "long" and price >= poc) or
                (direction == "short" and price <= poc)
            )
        )
        if vwap_ok and rsi_ok and (poc_ok or zs.momentum_confirming):
            return "ideal"
        if vwap_ok and rsi_ok:
            return "acceptable"

    # IDEAL: pullback para suporte-chave (VAL/POC/anchor) em tendência correta
    if zs.near_support and direction == "long" and zs.regime_consistent:
        if not zs.rsi_overbought:
            return "ideal" if zs.momentum_confirming else "acceptable"

    if zs.near_resistance and direction == "short" and zs.regime_consistent:
        if not zs.rsi_oversold:
            return "ideal" if zs.momentum_confirming else "acceptable"

    # ACCEPTABLE: próximo da VA ou VWAP, sem overextension
    if vah and val:
        dist_vah = abs(price - vah) / price
        dist_val = abs(price - val) / price
        near_va = dist_vah < _VA_PROXIMITY_PCT or dist_val < _VA_PROXIMITY_PCT
        if near_va and not zs.overextended:
            return "acceptable"

    if vwap:
        dist_vwap = abs(price - vwap) / price
        if dist_vwap < _VWAP_PROXIMITY_PCT and not zs.overextended:
            return "acceptable"

    # STRETCHED: fora da VA mas sem ser overextended — pode comprar, tamanho menor
    if vah and val:
        if direction == "long" and price > vah:
            return "stretched"
        if direction == "short" and price < val:
            return "stretched"

    # Default: no_data se não tem informação suficiente
    has_data = any(v is not None for v in [vwap, vah, val, poc])
    return "acceptable" if has_data else "no_data"


# ── Stop e alvo ───────────────────────────────────────────────────────────────

def _compute_stop(
    price: float, direction: str,
    vwap, vah, val, poc,
    anchors_below: list[float], anchors_above: list[float],
    atr,
) -> float | None:
    if direction == "long":
        candidates: list[float] = []
        if val and val < price:
            candidates.append(val * 0.995)  # ligeiramente abaixo do VAL
        if poc and poc < price:
            candidates.append(poc * 0.997)
        if vwap and vwap < price:
            candidates.append(vwap * 0.992)
        if anchors_below:
            candidates.append(anchors_below[0] * 0.997)
        if candidates:
            best = max(candidates)  # mais alto = stop mais apertado = melhor R/R
            return round(best, 4)
        if atr:
            return round(price - 1.8 * atr, 4)
    else:  # short
        candidates_: list[float] = []
        if vah and vah > price:
            candidates_.append(vah * 1.005)
        if poc and poc > price:
            candidates_.append(poc * 1.003)
        if vwap and vwap > price:
            candidates_.append(vwap * 1.008)
        if anchors_above:
            candidates_.append(anchors_above[0] * 1.003)
        if candidates_:
            best_ = min(candidates_)  # mais baixo = stop mais apertado para short
            return round(best_, 4)
        if atr:
            return round(price + 1.8 * atr, 4)
    return None


def _compute_target(
    price: float, direction: str,
    vwap, vah, val, poc,
    anchors_below: list[float], anchors_above: list[float],
) -> float | None:
    if direction == "long":
        candidates: list[float] = []
        if poc and poc > price:
            candidates.append(poc)
        if vah and vah > price:
            candidates.append(vah)
        if vwap and vwap > price:
            candidates.append(vwap)
        if anchors_above:
            candidates.append(anchors_above[0])
        return round(min(candidates), 4) if candidates else None
    else:  # short
        candidates_: list[float] = []
        if poc and poc < price:
            candidates_.append(poc)
        if val and val < price:
            candidates_.append(val)
        if vwap and vwap < price:
            candidates_.append(vwap)
        if anchors_below:
            candidates_.append(anchors_below[0])
        return round(max(candidates_), 4) if candidates_ else None


# ── Bias e scalar ─────────────────────────────────────────────────────────────

def _compute_adjustments(zs: ZoneSignal, direction: str) -> tuple[float, float]:
    """Returns (position_bias, allocation_scalar)."""
    eq = zs.entry_quality

    if eq == "ideal":
        bias = 0.20
        scalar = 1.20
        if zs.momentum_confirming:
            bias += 0.08
            scalar += 0.08
        if zs.near_support and direction == "long":
            bias += 0.02
    elif eq == "acceptable":
        bias = 0.05
        scalar = 1.00
        if zs.momentum_confirming:
            bias += 0.05
    elif eq == "stretched":
        bias = -0.15
        scalar = 0.70
        if zs.regime_consistent and zs.momentum_confirming:
            bias += 0.05  # breakout válido mas incompleto
            scalar += 0.10
    elif eq == "avoid":
        bias = -0.30
        scalar = 0.50
    else:  # no_data
        bias = 0.0
        scalar = 1.0

    # RSI extremos penalizam/bonificam
    if direction == "long":
        if zs.rsi_oversold and eq in ("ideal", "acceptable"):
            bias += 0.08   # RSI baixo = oportunidade de entrada em tendência
            scalar += 0.05
        elif zs.rsi_overbought:
            bias -= 0.10
            scalar -= 0.15
    else:
        if zs.rsi_overbought and eq in ("ideal", "acceptable"):
            bias += 0.08
            scalar += 0.05
        elif zs.rsi_oversold:
            bias -= 0.10
            scalar -= 0.15

    # Regime inconsistente reduz
    if not zs.regime_consistent:
        bias -= 0.10
        scalar -= 0.15

    # Overextended
    if zs.overextended:
        bias -= 0.15
        scalar -= 0.20

    # Limites
    bias   = max(-0.40, min(0.30, bias))
    scalar = max(0.40, min(1.30, scalar))

    return round(bias, 3), round(scalar, 3)


# ── Rationale ─────────────────────────────────────────────────────────────────

_QUALITY_ICONS = {
    "ideal":       "✅",
    "acceptable":  "🟡",
    "stretched":   "⚠",
    "avoid":       "🔴",
    "no_data":     "⚪",
}


def _build_rationale(zs: ZoneSignal, direction: str) -> str:
    icon = _QUALITY_ICONS.get(zs.entry_quality, "")
    parts = [f"{icon} {zs.entry_quality.upper()}"]

    if zs.within_value_area:
        parts.append("dentro da VA")
    elif zs.vah and zs.val:
        if zs.price > zs.vah:
            pct = (zs.price - zs.vah) / zs.vah * 100
            parts.append(f"{pct:.1f}% acima da VA")
        else:
            pct = (zs.val - zs.price) / zs.val * 100
            parts.append(f"{pct:.1f}% abaixo da VA")

    if zs.vwap:
        pct_vwap = (zs.price - zs.vwap) / zs.vwap * 100
        parts.append(f"VWAP {'↑' if pct_vwap >= 0 else '↓'}{abs(pct_vwap):.1f}%")

    if zs.poc:
        parts.append(f"POC {zs.poc:,.2f}")

    if zs.rsi is not None:
        label = " (sobrevendido)" if zs.rsi_oversold else (" (sobrecomprado)" if zs.rsi_overbought else "")
        parts.append(f"RSI {zs.rsi:.0f}{label}")

    if zs.refined_stop:
        parts.append(f"stop {zs.refined_stop:,.2f}")
    if zs.first_target:
        parts.append(f"alvo {zs.first_target:,.2f}")

    if not zs.regime_consistent:
        parts.append(f"⚡ regime TV ({zs.tv_regime}) vs bias ({direction})")

    return " | ".join(parts)


# ── Helper para uso no optimizer ─────────────────────────────────────────────

def apply_zone_signals_to_signals(
    signals: dict[str, Any],
    zone_signals: dict[str, "ZoneSignal"],
) -> dict[str, Any]:
    """
    Aplica os ZoneSignals nos AssetSignals (ajuste de composite + conviction).
    Modifica in-place e retorna o mesmo dict.

    Usado NO PIPELINE antes do optimizer para que os ajustes TV
    entrem na função objetivo, não só no pós-processamento.
    """
    for ticker, sig in signals.items():
        zs = zone_signals.get(ticker)
        if zs is None:
            continue
        if zs.entry_quality == "no_data":
            continue

        # Ajuste do composite
        old_composite = float(getattr(sig, "composite", 0) or 0)
        new_composite = old_composite + zs.position_bias
        # Clamp entre -1 e 1
        new_composite = max(-1.0, min(1.0, new_composite))
        try:
            sig.composite = new_composite
        except AttributeError:
            pass  # dataclass frozen — ignora silenciosamente

        # Ajuste da conviction baseado no entry quality
        old_conv = getattr(sig, "conviction", "medium")
        if zs.entry_quality == "ideal" and old_conv in ("medium", "low"):
            try:
                sig.conviction = "high"
            except AttributeError:
                pass
        elif zs.entry_quality == "avoid" and old_conv == "high":
            try:
                sig.conviction = "medium"
            except AttributeError:
                pass
        elif zs.entry_quality == "stretched":
            try:
                sig.conviction = "low" if old_conv != "high" else "medium"
            except AttributeError:
                pass

        # Armazena o ZoneSignal no signal para uso posterior
        try:
            sig._zone_signal = zs
        except AttributeError:
            pass

    return signals
