"""
Provider: Positioning Models — CTA + Vol Control + Risk Parity (BofA/GS style)

Lê via `BloombergQueryLayer.get_positioning_models()` (que consulta `bql_latest`,
populado pelo fluxo BQuant zip → csv_parser → data_normalizer → db_writer).
Para cada ticker, expõe:
  - cta_score, cta_notional_b
  - volctrl_score, volctrl_notional_b
  - rp_score, rp_notional_b
  - flow_total_b, flow_direction

Agregados:
  - aggregate_flow_total_b: soma de notional dos 3 modelos no universo
  - extreme_long / extreme_short por modelo
  - top crowding (CTAs + VolCtrl + RP somados)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("providers.positioning_models")


@dataclass
class PositioningSignal:
    ticker: str
    cta_score:          float = 0.0
    cta_notional_b:     float = 0.0
    cta_leverage:       float = 0.0
    volctrl_score:      float = 0.0
    volctrl_notional_b: float = 0.0
    rp_score:           float = 0.0
    rp_notional_b:      float = 0.0
    flow_total_b:       float = 0.0
    flow_direction:     str = "flat"
    rv_30d:             float | None = None
    rv_60d:             float | None = None


@dataclass
class PositioningModelsResult:
    signals: dict[str, PositioningSignal] = field(default_factory=dict)
    aggregate_flow_total_b:  float = 0.0
    aggregate_cta_b:         float = 0.0
    aggregate_volctrl_b:     float = 0.0
    aggregate_rp_b:          float = 0.0
    extreme_longs_cta:       list[str] = field(default_factory=list)
    extreme_shorts_cta:      list[str] = field(default_factory=list)
    extreme_longs_volctrl:   list[str] = field(default_factory=list)
    extreme_shorts_volctrl:  list[str] = field(default_factory=list)
    top_long_combined:       list[str] = field(default_factory=list)
    top_short_combined:      list[str] = field(default_factory=list)
    timestamp:               str = ""

    def get(self, ticker: str) -> PositioningSignal | None:
        return self.signals.get(ticker)


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        if f != f:  # NaN
            return default
        return f
    except (TypeError, ValueError):
        return default


def load_positioning_models() -> PositioningModelsResult:
    """
    Carrega o CSV mais recente e devolve um PositioningModelsResult.
    Retorna result vazio se o CSV não existir.
    """
    result = PositioningModelsResult(timestamp=datetime.now().isoformat())

    try:
        from app.query_layer import BloombergQueryLayer
        ql = BloombergQueryLayer()
        raw = ql.get_positioning_models()
    except Exception as exc:
        _log.warning("positioning_models_load_failed", error=str(exc))
        return result

    if not raw:
        _log.info("positioning_models_empty")
        return result

    for tk, row in raw.items():
        sig = PositioningSignal(
            ticker             = tk,
            cta_score          = _safe(row.get("cta_score")),
            cta_notional_b     = _safe(row.get("cta_notional_b")),
            cta_leverage       = _safe(row.get("cta_leverage")),
            volctrl_score      = _safe(row.get("volctrl_score")),
            volctrl_notional_b = _safe(row.get("volctrl_notional_b")),
            rp_score           = _safe(row.get("rp_score")),
            rp_notional_b      = _safe(row.get("rp_notional_b")),
            flow_total_b       = _safe(row.get("flow_total_b")),
            flow_direction     = (row.get("flow_direction") or "flat"),
            rv_30d             = row.get("rv_30d"),
            rv_60d             = row.get("rv_60d"),
        )
        result.signals[tk] = sig
        result.aggregate_flow_total_b += sig.flow_total_b
        result.aggregate_cta_b        += sig.cta_notional_b
        result.aggregate_volctrl_b    += sig.volctrl_notional_b
        result.aggregate_rp_b         += sig.rp_notional_b

    # Extreme thresholds (BofA style: |score| > 0.6)
    for tk, sig in result.signals.items():
        if sig.cta_score >= 0.6:
            result.extreme_longs_cta.append(tk)
        elif sig.cta_score <= -0.6:
            result.extreme_shorts_cta.append(tk)
        if sig.volctrl_score >= 0.6:
            result.extreme_longs_volctrl.append(tk)
        elif sig.volctrl_score <= -0.6:
            result.extreme_shorts_volctrl.append(tk)

    # Top combinado (CTA + VolCtrl + RP)
    combined = sorted(
        result.signals.values(),
        key=lambda s: s.cta_score + s.volctrl_score + s.rp_score,
        reverse=True,
    )
    result.top_long_combined  = [s.ticker for s in combined[:10]]
    result.top_short_combined = [s.ticker for s in combined[-10:]][::-1]

    _log.info(
        "positioning_models_done",
        n=len(result.signals),
        cta_b=round(result.aggregate_cta_b, 2),
        volctrl_b=round(result.aggregate_volctrl_b, 2),
        rp_b=round(result.aggregate_rp_b, 2),
        ext_long_cta=len(result.extreme_longs_cta),
        ext_short_cta=len(result.extreme_shorts_cta),
    )
    return result


def get_positioning_signal_for_ticker(
    ticker: str, result: PositioningModelsResult
) -> float:
    """
    Retorna sinal composto [-1, 1] para alpha_signals: média dos 3 modelos.
    """
    sig = result.signals.get(ticker)
    if sig is None:
        return 0.0
    return max(-1.0, min(1.0, (sig.cta_score + sig.volctrl_score + sig.rp_score) / 3.0))
