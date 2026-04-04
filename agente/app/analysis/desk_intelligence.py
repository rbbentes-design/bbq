"""
Desk Intelligence — Motor de inferência de mercado regime-aware

Camada de inteligência sobre os sinais já computados pelo pipeline.
Não recomputa nada — apenas interpreta, agrega e classifica.

Outputs:
  DeskIntelligenceResult com:
    - Regime estendido (7 estados) com pesos dinâmicos por família de edge
    - Hidden Opportunity Score por ativo (preço atrasado vs opções/fluxo)
    - Fragility Score por ativo (preço forte mas deteriorando internamente)
    - Score composto regime-adjusted por ativo
    - RRG metadata (IV rank → cor bolha, skew → borda, conviction → tamanho)
    - Ranking de ativos priorizado por convergência de sinais
    - Explicações textuais legíveis por humano
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from app.audit.logger import get_logger

if TYPE_CHECKING:
    from app.analysis.alpha_signals import AssetSignal
    from app.analysis.relative_strength import RRGResult
    from app.analysis.vol_regime import VolRegimeResult
    from app.analysis.narrative_alpha import NarrativeAlphaResult
    from app.analysis.cta_positioning import CTAPositioningResult
    from app.providers.shadow_flow import ShadowFlowResult

_log = get_logger("analysis.desk_intelligence")


# ── Pesos por regime ────────────────────────────────────────────────────────────

_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "risk_on_momentum":   {"macro": 0.15, "technical": 0.35, "vol": 0.15, "flow": 0.20, "narrative": 0.15},
    "risk_off_defensive": {"macro": 0.35, "technical": 0.15, "vol": 0.30, "flow": 0.10, "narrative": 0.10},
    "vol_squeeze":        {"macro": 0.10, "technical": 0.25, "vol": 0.40, "flow": 0.15, "narrative": 0.10},
    "narrative_driven":   {"macro": 0.10, "technical": 0.20, "vol": 0.10, "flow": 0.15, "narrative": 0.45},
    "mechanical_passive": {"macro": 0.10, "technical": 0.10, "vol": 0.15, "flow": 0.50, "narrative": 0.15},
    "dispersed_rotation": {"macro": 0.20, "technical": 0.30, "vol": 0.20, "flow": 0.15, "narrative": 0.15},
    "stress":             {"macro": 0.40, "technical": 0.05, "vol": 0.40, "flow": 0.05, "narrative": 0.10},
}

_REGIME_LABELS: dict[str, str] = {
    "risk_on_momentum":   "Risk On — Momentum",
    "risk_off_defensive": "Risk Off — Defensive",
    "vol_squeeze":        "Vol Squeeze",
    "narrative_driven":   "Narrative Driven",
    "mechanical_passive": "Mechanical / Passive",
    "dispersed_rotation": "Dispersed Rotation",
    "stress":             "Stress / Crisis",
}


# ── Output dataclass ────────────────────────────────────────────────────────────

@dataclass
class DeskIntelligenceResult:
    # Regime estendido
    market_regime: str = "dispersed_rotation"
    market_regime_label: str = "Dispersed Rotation"
    regime_confidence: float = 0.5
    regime_drivers: list[str] = field(default_factory=list)
    regime_weights: dict[str, float] = field(default_factory=dict)

    # Scores por ativo
    opportunity_scores: dict[str, float] = field(default_factory=dict)   # Hidden Opportunity [-1,1]
    fragility_scores:   dict[str, float] = field(default_factory=dict)   # Fragility [0,1]
    contagion_scores:   dict[str, float] = field(default_factory=dict)   # [0,1]
    regime_adj_scores:  dict[str, float] = field(default_factory=dict)   # Composite regime-adjusted

    # Rankings
    ranked_assets:            list[str] = field(default_factory=list)
    top_opportunities:        list[str] = field(default_factory=list)   # 5 maiores
    top_fragile:              list[str] = field(default_factory=list)   # 5 maiores
    narrative_leaders:        list[str] = field(default_factory=list)
    laggards_with_conviction: list[str] = field(default_factory=list)  # improving + high conviction

    # Narrativa
    dominant_narrative: str = ""
    narrative_clusters: dict[str, list[str]] = field(default_factory=dict)  # theme → tickers

    # RRG metadata para plot turbinado
    rrg_meta: dict[str, dict] = field(default_factory=dict)
    # {ticker: {bubble_size, bubble_color, border_color, border_width, opacity, label_extra, flow_score}}

    # Explicações textuais por ativo
    explanations: dict[str, str] = field(default_factory=dict)

    # ── Motor de Narrativa (narrative_classifier) ──────────────────────────────
    # Estado narrativo por ativo: pre_narrative | emerging | mature | exhausted
    narrative_states:   dict[str, str]   = field(default_factory=dict)
    # Scores [0-100]
    asymmetry_scores:   dict[str, float] = field(default_factory=dict)
    crowdedness_scores: dict[str, float] = field(default_factory=dict)
    exhaustion_scores:  dict[str, float] = field(default_factory=dict)
    # Convicção operacional por ativo
    convictions:        dict[str, str]   = field(default_factory=dict)
    horizons:           dict[str, str]   = field(default_factory=dict)
    opportunity_types:  dict[str, str]   = field(default_factory=dict)
    # Racional narrativo por ativo
    narrative_rationales: dict[str, str] = field(default_factory=dict)
    # Alertas de consenso saturado
    consensus_overloads: dict[str, bool] = field(default_factory=dict)
    # Rankings narrativos
    top_asymmetry:      list[str] = field(default_factory=list)
    top_crowded:        list[str] = field(default_factory=list)
    top_exhausted_narr: list[str] = field(default_factory=list)
    emerging_themes:    list[str] = field(default_factory=list)
    short_candidates:   list[str] = field(default_factory=list)
    # Chart regimes por ativo
    chart_regimes:      dict[str, str]   = field(default_factory=dict)
    chart_rationales:   dict[str, str]   = field(default_factory=dict)
    # WSB top mentions + squeeze
    wsb_top_mentions:   list[str] = field(default_factory=list)
    top_squeeze:        list[str] = field(default_factory=list)

    # Vol Options Regime (Greeks Dashboard BBQ)
    vol_options_regime: Any | None = None   # VolOptionsRegime | None

    # Metadados de debug
    n_assets:   int = 0
    n_with_rrg: int = 0


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _safe(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _iv_rank_color(iv_pct: float | None) -> str:
    """IV percentile → cor hex. Verde=barata, vermelho=cara."""
    if iv_pct is None:
        return "#6b7280"
    # 0.0 (mais barata) → #22c55e verde
    # 0.5 (mediana)      → #f59e0b âmbar
    # 1.0 (mais cara)    → #ef4444 vermelho
    if iv_pct <= 0.35:
        return "#22c55e"
    if iv_pct <= 0.55:
        return "#84cc16"
    if iv_pct <= 0.70:
        return "#f59e0b"
    if iv_pct <= 0.85:
        return "#f97316"
    return "#ef4444"


def _skew_color(skew: float | None) -> str:
    """skew_5pct → cor borda. Negativo (calls baratas) = verde, positivo (put skew) = vermelho."""
    if skew is None:
        return "#6b7280"
    return "#22c55e" if skew <= -0.01 else ("#ef4444" if skew >= 0.02 else "#f59e0b")


# ── A. Regime classifier ─────────────────────────────────────────────────────────

def classify_market_regime(
    vol_regime: Any | None,
    network_result: dict | None,
    rrg_result: Any | None,
    narrative_result: Any | None,
    flow_pred: Any | None = None,
    cta_result: Any | None = None,
) -> tuple[str, float, list[str]]:
    """
    Classifica o regime de mercado em 7 estados.
    Retorna (regime_key, confidence, drivers).
    """
    drivers: list[str] = []

    # Extrai métricas
    stress   = _safe(getattr(vol_regime, "stress_score", None)) or 0.0
    vr       = getattr(vol_regime, "regime", "elevated") or "elevated"
    ts_ratio = _safe(getattr(vol_regime, "term_structure_ratio", None)) or 1.0

    net_reg  = (network_result or {}).get("regime", {})
    net_name = net_reg.get("regime", "unknown") or "unknown"
    entropy  = _safe(net_reg.get("corr_entropy")) or 1.5

    # Distribuição dos quadrantes RRG
    leading_pct = improving_pct = weakening_pct = lagging_pct = 0.0
    if rrg_result and hasattr(rrg_result, "signals") and rrg_result.signals:
        total = len(rrg_result.signals)
        n_lead = len(getattr(rrg_result, "leading", []))
        n_impr = len(getattr(rrg_result, "improving", []))
        n_weak = len(getattr(rrg_result, "weakening", []))
        n_lagg = len(getattr(rrg_result, "lagging", []))
        if total > 0:
            leading_pct  = n_lead / total
            improving_pct = n_impr / total
            weakening_pct = n_weak / total
            lagging_pct  = n_lagg / total

    # Intensidade narrativa
    avg_narrative = 0.0
    if narrative_result and hasattr(narrative_result, "signals"):
        vals = [_safe(getattr(s, "composite_narrative", None))
                for s in (narrative_result.signals or {}).values()]
        vals = [v for v in vals if v is not None]
        avg_narrative = sum(abs(v) for v in vals) / len(vals) if vals else 0.0

    # GEX / LETF magnitude
    gex_mag = 0.0
    if flow_pred and hasattr(flow_pred, "magnitude_bn"):
        gex_mag = abs(_safe(flow_pred.magnitude_bn) or 0.0)

    # CTA crowding extremo
    cta_extreme = False
    if cta_result and hasattr(cta_result, "signals"):
        extremes = sum(
            1 for s in (cta_result.signals or {}).values()
            if getattr(s, "crowding", "") in ("extreme_long", "extreme_short")
        )
        cta_extreme = extremes >= 2

    # ── Override em cascata ──────────────────────────────────────────────────────

    # 1. Stress: override imediato
    if stress > 0.65 or vr == "crisis":
        drivers.append(f"stress_score={stress:.2f}")
        drivers.append(f"vol_regime={vr}")
        return "stress", min(1.0, stress + 0.2), drivers

    # 2. Risk off defensive
    if stress > 0.40 or net_name == "risk_off":
        drivers.append(f"stress={stress:.2f}")
        if net_name == "risk_off":
            drivers.append("network=risk_off")
        conf = _clamp(0.5 + stress * 0.5 + (0.2 if net_name == "risk_off" else 0))
        return "risk_off_defensive", conf, drivers

    # 3. Mechanical / passive: GEX ou CTA extremo
    if gex_mag > 0.5 or cta_extreme:
        if gex_mag > 0.5:
            drivers.append(f"gex_magnitude={gex_mag:.1f}B")
        if cta_extreme:
            drivers.append("cta_extreme_crowding")
        return "mechanical_passive", 0.65, drivers

    # 4. Narrative driven: narrativa forte + vol calma
    if avg_narrative > 0.38 and stress < 0.30:
        drivers.append(f"avg_narrative={avg_narrative:.2f}")
        drivers.append(f"stress_low={stress:.2f}")
        return "narrative_driven", _clamp(0.5 + avg_narrative), drivers

    # 5. Vol squeeze: vol calma + contango steep
    if (vr == "calm" or stress < 0.18) and ts_ratio < 0.90:
        drivers.append(f"vol_regime={vr}")
        drivers.append(f"ts_ratio={ts_ratio:.2f}")
        return "vol_squeeze", 0.70, drivers

    # 6. Risk on momentum: leading dominante + rede positiva
    if leading_pct > 0.45 and stress < 0.28 and net_name in ("risk_on", "unknown"):
        drivers.append(f"leading_pct={leading_pct:.0%}")
        drivers.append(f"stress={stress:.2f}")
        if net_name == "risk_on":
            drivers.append("network=risk_on")
        conf = _clamp(0.5 + leading_pct * 0.5)
        return "risk_on_momentum", conf, drivers

    # 7. Dispersed rotation (default)
    if entropy > 2.2 or (improving_pct + weakening_pct) > leading_pct:
        drivers.append(f"entropy={entropy:.2f}")
        drivers.append(f"rotation_mix={improving_pct+weakening_pct:.0%}")
    else:
        drivers.append("no_dominant_signal")

    return "dispersed_rotation", 0.45, drivers


# ── B. Hidden Opportunity Score ──────────────────────────────────────────────────

def compute_opportunity_scores(
    signals: dict[str, "AssetSignal"],
    rrg_result: Any | None,
    options_map: dict[str, dict] | None,
    shadow_flow: Any | None,
    cta_result: Any | None,
) -> dict[str, float]:
    """
    Detecta ativos cujo preço ainda não lidera, mas onde opções/fluxo/skew
    já sinalizam liderança emergente.
    """
    rrg_sigs = {}
    if rrg_result and hasattr(rrg_result, "signals"):
        rrg_sigs = rrg_result.signals or {}

    dp_map: dict[str, float] = {}
    if shadow_flow and hasattr(shadow_flow, "signals"):
        for t, s in (shadow_flow.signals or {}).items():
            dp_map[t] = _safe(getattr(s, "dark_pool_score", None)) or 0.0

    cta_map: dict[str, tuple[str, float]] = {}
    if cta_result and hasattr(cta_result, "signals"):
        for t, s in (cta_result.signals or {}).items():
            cta_map[t] = (
                getattr(s, "crowding", "neutral"),
                _safe(getattr(s, "flow_surprise_signal", None)) or 0.0,
            )

    scores: dict[str, float] = {}
    for ticker, sig in signals.items():
        score = 0.0

        # RRG: improving com momentum crescendo
        rs = rrg_sigs.get(ticker)
        quadrant = getattr(rs, "quadrant", "") if rs else ""
        rs_mom   = _safe(getattr(rs, "rs_momentum", None)) if rs else None
        if quadrant == "improving":
            score += 0.25
            # Se tail do momentum está crescendo
            tail_mom = getattr(rs, "tail_rs_momentum", []) if rs else []
            if len(tail_mom) >= 2 and tail_mom[-1] > tail_mom[-2]:
                score += 0.10

        # Options: IV barata
        opt = (options_map or {}).get(ticker, {})
        iv_pct = _safe(sig.iv_percentile) or _safe(opt.get("iv_percentile"))
        if iv_pct is not None and iv_pct < 0.40:
            score += 0.20

        # Skew: calls baratas (risk reversal virando positivo)
        skew = _safe(sig.iv_skew) or _safe(opt.get("skew_5pct"))
        if skew is not None and skew < -0.02:
            score += 0.15

        # Dark pool acumulando
        dp = dp_map.get(ticker, _safe(getattr(sig, "dark_pool_score", None)) or 0.0)
        if dp > 0.20:
            score += 0.15

        # Contagion centrality alta em cluster de líderes
        katz = _safe(sig.contagion_katz) or 0.0
        if katz > 0.5 and quadrant in ("leading", "improving"):
            score += 0.15

        # Composite positivo mas ainda não liderando
        comp = _safe(sig.composite) or 0.0
        if comp > 0.15 and quadrant not in ("leading",):
            score += 0.10

        # CTA: posicionamento contrarian (potencial squeeze)
        crowding, cta_surprise = cta_map.get(ticker, ("neutral", 0.0))
        if cta_surprise > 0.30:
            score += 0.08

        scores[ticker] = math.tanh(score * 2.0)

    return scores


# ── C. Fragility Score ───────────────────────────────────────────────────────────

def compute_fragility_scores(
    signals: dict[str, "AssetSignal"],
    rrg_result: Any | None,
    options_map: dict[str, dict] | None,
    shadow_flow: Any | None,
    cta_result: Any | None,
) -> dict[str, float]:
    """
    Detecta ativos que parecem fortes no preço mas estão deteriorando internamente.
    """
    rrg_sigs = {}
    if rrg_result and hasattr(rrg_result, "signals"):
        rrg_sigs = rrg_result.signals or {}

    dp_map: dict[str, float] = {}
    if shadow_flow and hasattr(shadow_flow, "signals"):
        for t, s in (shadow_flow.signals or {}).items():
            dp_map[t] = _safe(getattr(s, "dark_pool_score", None)) or 0.0

    cta_map: dict[str, str] = {}
    if cta_result and hasattr(cta_result, "signals"):
        for t, s in (cta_result.signals or {}).items():
            cta_map[t] = getattr(s, "crowding", "neutral")

    scores: dict[str, float] = {}
    for ticker, sig in signals.items():
        score = 0.0

        rs = rrg_sigs.get(ticker)
        quadrant = getattr(rs, "quadrant", "") if rs else ""

        # RRG: weakening com momentum declinando
        if quadrant == "weakening":
            score += 0.25
            tail_mom = getattr(rs, "tail_rs_momentum", []) if rs else []
            if len(tail_mom) >= 2 and tail_mom[-1] < tail_mom[-2]:
                score += 0.10

        # Put skew elevado (mercado pagando prêmio por proteção)
        opt = (options_map or {}).get(ticker, {})
        skew = _safe(sig.iv_skew) or _safe(opt.get("skew_5pct"))
        if skew is not None and skew > 0.04:
            score += 0.20

        # IV cara (hedges custosos = stress interno)
        iv_pct = _safe(sig.iv_percentile) or _safe(opt.get("iv_percentile"))
        if iv_pct is not None and iv_pct > 0.75:
            score += 0.15

        # Dark pool vendendo
        dp = dp_map.get(ticker, _safe(getattr(sig, "dark_pool_score", None)) or 0.0)
        if dp < -0.20:
            score += 0.15

        # CTA extremamente long (vulnerável a unwind)
        if cta_map.get(ticker) == "extreme_long":
            score += 0.20

        # Composite positivo mas tail risk alto (distribuição fat-tailed)
        comp = _safe(sig.composite) or 0.0
        tail = _safe(sig.tail_score) or 0.0
        if comp > 0 and tail > 0.60:
            score += 0.10

        scores[ticker] = _clamp(score, 0.0, 1.0)

    return scores


# ── D. Contagion scores ──────────────────────────────────────────────────────────

def compute_contagion_scores(
    signals: dict[str, "AssetSignal"],
    rrg_result: Any | None,
    network_result: dict | None,
) -> dict[str, float]:
    """
    Score de contágio: combina centralidade Katz (já computada) com
    pertencimento ao cluster de líderes no RRG.
    """
    leading_set: set[str] = set()
    if rrg_result:
        leading_set = set(getattr(rrg_result, "leading", []) or [])
        leading_set.update(getattr(rrg_result, "improving", []) or [])

    # Hubs do MST por ticker
    hub_map: dict[str, float] = {}
    if network_result:
        for ticker, degree in (network_result.get("mst", {}).get("hubs", []) or []):
            hub_map[ticker] = float(degree)
    max_deg = max(hub_map.values(), default=1.0)

    scores: dict[str, float] = {}
    for ticker, sig in signals.items():
        katz = _clamp(_safe(sig.contagion_katz) or 0.0, 0.0, 1.0)
        hub_norm = hub_map.get(ticker, 0.0) / max_deg if max_deg > 0 else 0.0
        leader_bonus = 0.20 if ticker in leading_set else 0.0
        scores[ticker] = _clamp(katz * 0.60 + hub_norm * 0.20 + leader_bonus)

    return scores


# ── E. RRG metadata ──────────────────────────────────────────────────────────────

def compute_rrg_meta(
    signals: dict[str, "AssetSignal"],
    rrg_result: Any | None,
    options_map: dict[str, dict] | None,
    shadow_flow: Any | None,
    cta_result: Any | None,
) -> dict[str, dict]:
    """
    Metadata visual para o RRG turbinado:
      bubble_size   = conviction × base
      bubble_color  = IV rank (verde=barata → vermelho=cara)
      border_color  = skew direction
      border_width  = abs(skew)
      opacity       = signal confidence
      label_extra   = IV rank badge
      flow_score    = dark_pool + CTA surprise
    """
    rrg_sigs = {}
    if rrg_result and hasattr(rrg_result, "signals"):
        rrg_sigs = rrg_result.signals or {}

    dp_map: dict[str, float] = {}
    if shadow_flow and hasattr(shadow_flow, "signals"):
        for t, s in (shadow_flow.signals or {}).items():
            dp_map[t] = _safe(getattr(s, "dark_pool_score", None)) or 0.0

    cta_surp: dict[str, float] = {}
    if cta_result and hasattr(cta_result, "signals"):
        for t, s in (cta_result.signals or {}).items():
            cta_surp[t] = _safe(getattr(s, "flow_surprise_signal", None)) or 0.0

    meta: dict[str, dict] = {}
    for ticker, sig in signals.items():
        rs = rrg_sigs.get(ticker)
        opt = (options_map or {}).get(ticker, {})

        comp = abs(_safe(sig.composite) or 0.0)
        conv = sig.conviction or "low"
        iv_pct = _safe(sig.iv_percentile) or _safe(opt.get("iv_percentile"))
        skew   = _safe(sig.iv_skew) or _safe(opt.get("skew_5pct"))

        # Bubble size: conviction-based [12, 52]
        conv_scalar = {"high": 1.0, "medium": 0.65, "low": 0.35}.get(conv, 0.35)
        bubble_size = 12 + comp * 50 * conv_scalar

        # Opacity: signal confidence
        opacity = {"high": 1.0, "medium": 0.72, "low": 0.45}.get(conv, 0.45)

        # RS-Ratio para determinar se está no grafo
        rs_ratio  = _safe(getattr(rs, "rs_ratio", None)) if rs else None
        rs_mom    = _safe(getattr(rs, "rs_momentum", None)) if rs else None
        quadrant  = getattr(rs, "quadrant", "") if rs else ""
        rs_pct    = _safe(getattr(rs, "rs_percentile", None)) if rs else None
        tail_r    = list(getattr(rs, "tail_rs_ratio", []) or []) if rs else []
        tail_m    = list(getattr(rs, "tail_rs_momentum", []) or []) if rs else []

        # Flow score combinado
        dp = dp_map.get(ticker, 0.0)
        cta = cta_surp.get(ticker, 0.0)
        flow_score = dp * 0.6 + cta * 0.4

        meta[ticker] = {
            "bubble_size":  round(max(10.0, bubble_size), 1),
            "bubble_color": _iv_rank_color(iv_pct),
            "border_color": _skew_color(skew),
            "border_width": round(max(1.0, abs(skew or 0.0) * 30), 1),
            "opacity":      opacity,
            "label_extra":  f"IV:{iv_pct:.0%}" if iv_pct is not None else "",
            "flow_score":   round(flow_score, 3),
            "conviction":   conv,
            "composite":    round(_safe(sig.composite) or 0.0, 3),
            "quadrant":     quadrant,
            "rs_ratio":     round(rs_ratio, 2) if rs_ratio is not None else None,
            "rs_momentum":  round(rs_mom, 2) if rs_mom is not None else None,
            "rs_percentile": round(rs_pct, 1) if rs_pct is not None else None,
            "tail_rs_ratio":     [round(v, 2) for v in tail_r[-5:]],
            "tail_rs_momentum":  [round(v, 2) for v in tail_m[-5:]],
            "iv_percentile": round(iv_pct, 3) if iv_pct is not None else None,
            "skew":          round(skew, 4) if skew is not None else None,
        }

    return meta


# ── F. Regime-adjusted composite score ──────────────────────────────────────────

def compute_regime_adj_scores(
    signals: dict[str, "AssetSignal"],
    rrg_result: Any | None,
    narrative_result: Any | None,
    cta_result: Any | None,
    shadow_flow: Any | None,
    options_map: dict[str, dict] | None,
    regime_weights: dict[str, float],
) -> dict[str, float]:
    """
    Score composto ajustado ao regime atual.
    Pondera as 5 famílias de edge pelos pesos do regime.
    """
    rrg_sigs = {}
    if rrg_result and hasattr(rrg_result, "signals"):
        rrg_sigs = rrg_result.signals or {}

    narr_map: dict[str, float] = {}
    if narrative_result and hasattr(narrative_result, "signals"):
        for t, s in (narrative_result.signals or {}).items():
            narr_map[t] = _safe(getattr(s, "composite_narrative", None)) or 0.0

    cta_map: dict[str, float] = {}
    if cta_result and hasattr(cta_result, "signals"):
        for t, s in (cta_result.signals or {}).items():
            cta_map[t] = _safe(getattr(s, "flow_surprise_signal", None)) or 0.0

    dp_map: dict[str, float] = {}
    if shadow_flow and hasattr(shadow_flow, "signals"):
        for t, s in (shadow_flow.signals or {}).items():
            dp_map[t] = _safe(getattr(s, "dark_pool_score", None)) or 0.0

    w = regime_weights
    scores: dict[str, float] = {}

    for ticker, sig in signals.items():
        opt = (options_map or {}).get(ticker, {})
        rs  = rrg_sigs.get(ticker)

        # ── Família: macro ─────────────────────────────────────────────────
        regime_bull = _safe(sig.regime_bull) or 0.5
        direction_sign = 1.0 if sig.direction == "long" else (-1.0 if sig.direction == "short" else 0.0)
        macro_score = (
            (_safe(sig.momentum_score) or 0.0) * 0.30
            + (_safe(sig.mean_rev_score) or 0.0) * 0.20
            + (regime_bull - 0.5) * 2.0 * 0.50
        ) * (direction_sign if direction_sign != 0 else 1.0)

        # ── Família: technical ─────────────────────────────────────────────
        rs_alpha = _safe(getattr(rs, "rs_alpha_score", None)) if rs else None
        technical_score = (
            (rs_alpha or 0.0) * 0.60
            + (_safe(sig.momentum_score) or 0.0) * 0.40
        )

        # ── Família: vol ───────────────────────────────────────────────────
        iv_pct   = _safe(sig.iv_percentile) or _safe(opt.get("iv_percentile")) or 0.5
        tail_s   = _safe(sig.tail_score) or 0.0
        vol_edge = _safe(sig.vol_edge_score) or 0.0
        vol_score = (
            vol_edge * 0.50
            + (0.5 - iv_pct) * 2.0 * 0.30   # IV barata = positivo
            + (1.0 - tail_s) * 0.20
        )

        # ── Família: flow ──────────────────────────────────────────────────
        opt_flow = _safe(sig.options_flow_score) or 0.0
        dp       = dp_map.get(ticker, 0.0)
        cta      = cta_map.get(ticker, 0.0)
        flow_score = opt_flow * 0.40 + dp * 0.30 + cta * 0.30

        # ── Família: narrative ─────────────────────────────────────────────
        narr = narr_map.get(ticker, 0.0)
        rs_a = _safe(getattr(rs, "rs_alpha_score", None)) if rs else 0.0
        narrative_score = (narr or 0.0) * 0.70 + (rs_a or 0.0) * 0.30

        # ── Score final ponderado pelo regime ──────────────────────────────
        adj = (
            w.get("macro", 0.20)     * macro_score
            + w.get("technical", 0.25) * technical_score
            + w.get("vol", 0.20)       * vol_score
            + w.get("flow", 0.20)      * flow_score
            + w.get("narrative", 0.15) * narrative_score
        )
        scores[ticker] = round(math.tanh(adj * 1.5), 4)

    return scores


# ── G. Textual explanations ──────────────────────────────────────────────────────

def build_explanations(
    signals: dict[str, "AssetSignal"],
    rrg_result: Any | None,
    opportunity_scores: dict[str, float],
    fragility_scores:   dict[str, float],
    regime_adj_scores:  dict[str, float],
    options_map: dict[str, dict] | None,
    cta_result: Any | None,
    narrative_result: Any | None,
    dominant_narrative: str,
) -> dict[str, str]:
    """Gera explicação textual legível por humano para cada ativo."""
    rrg_sigs = {}
    if rrg_result and hasattr(rrg_result, "signals"):
        rrg_sigs = rrg_result.signals or {}

    narr_ticker_map: dict[str, list[str]] = {}
    if narrative_result and hasattr(narrative_result, "ticker_themes"):
        narr_ticker_map = narrative_result.ticker_themes or {}

    cta_map: dict[str, str] = {}
    if cta_result and hasattr(cta_result, "signals"):
        for t, s in (cta_result.signals or {}).items():
            cta_map[t] = getattr(s, "crowding", "neutral")

    _QUADRANT_PT = {
        "leading":   "liderança",
        "improving": "recuperação",
        "weakening": "enfraquecimento",
        "lagging":   "atraso",
    }

    explanations: dict[str, str] = {}
    for ticker, sig in signals.items():
        rs  = rrg_sigs.get(ticker)
        opt = (options_map or {}).get(ticker, {})

        quadrant  = getattr(rs, "quadrant", "") if rs else ""
        rs_ratio  = _safe(getattr(rs, "rs_ratio", None)) if rs else None
        rs_pct    = _safe(getattr(rs, "rs_percentile", None)) if rs else None
        iv_pct    = _safe(sig.iv_percentile) or _safe(opt.get("iv_percentile"))
        skew      = _safe(sig.iv_skew) or _safe(opt.get("skew_5pct"))
        comp      = _safe(sig.composite) or 0.0
        conv      = sig.conviction or "low"
        tail      = _safe(sig.tail_score) or 0.0
        opp       = opportunity_scores.get(ticker, 0.0)
        frag      = fragility_scores.get(ticker, 0.0)
        adj       = regime_adj_scores.get(ticker, 0.0)
        crowding  = cta_map.get(ticker, "neutral")
        themes    = narr_ticker_map.get(ticker, [])

        parts: list[str] = []

        # Header
        quad_pt = _QUADRANT_PT.get(quadrant, quadrant)
        if rs_ratio is not None:
            parts.append(f"{ticker} em fase de {quad_pt} (RS-Ratio {rs_ratio:.1f}, percentil {rs_pct:.0f}°)." if rs_pct else f"{ticker} em {quad_pt} (RS-Ratio {rs_ratio:.1f}).")
        else:
            parts.append(f"{ticker} — sem dados de força relativa.")

        # Flags de oportunidade
        if quadrant == "improving":
            parts.append("Força relativa em recuperação — sinal de entrada antecipada.")
        if iv_pct is not None and iv_pct < 0.40:
            parts.append(f"Volatilidade implícita barata ({iv_pct:.0%} percentil) — convexidade acessível para upside.")
        if skew is not None and skew < -0.02:
            parts.append("Risk reversal virando positivo (calls baratas vs puts) — opções apontando recuperação antes do preço.")
        if opp > 0.40:
            parts.append("Hidden Opportunity Score elevado — múltiplos sinais não-price apontando liderança emergente.")

        # Flags de fragilidade
        if quadrant == "weakening":
            parts.append("Ativo perdendo força relativa — monitorar deterioração.")
        if skew is not None and skew > 0.04:
            parts.append(f"Put skew elevado ({skew:.3f}) — mercado pagando prêmio por proteção.")
        if frag > 0.50:
            parts.append("Fragility Score alto — estrutura interna fraca apesar do preço.")
        if crowding == "extreme_long":
            parts.append("CTAs extremamente comprados — risco de unwind brusco.")

        # Narrativa
        if themes:
            parts.append(f"Ativo ligado aos temas: {', '.join(themes[:3])}.")
        elif dominant_narrative:
            parts.append(f"Sem posição clara na narrativa dominante ({dominant_narrative}).")

        # Scores finais
        parts.append(
            f"Score ajustado ao regime: {adj:+.2f} | "
            f"Composite: {comp:+.2f} ({conv}) | "
            f"Oportunidade: {opp:+.2f} | Fragilidade: {frag:.2f}."
        )

        explanations[ticker] = " ".join(parts)

    return explanations


# ── H. Narrativa e clusters ──────────────────────────────────────────────────────

def extract_narrative_clusters(
    narrative_result: Any | None,
    signals: dict[str, "AssetSignal"],
    rrg_result: Any | None,
) -> tuple[str, dict[str, list[str]]]:
    """
    Extrai narrativa dominante e clusters temáticos.
    Retorna (dominant_narrative, {theme: [tickers]}).
    """
    if narrative_result is None:
        return "", {}

    # Temas do DeepVue
    themes_parsed = getattr(narrative_result, "deepvue_themes_parsed", {}) or {}
    ticker_themes = getattr(narrative_result, "ticker_themes", {}) or {}

    if not themes_parsed and not ticker_themes:
        return "", {}

    # Tema com maior score absoluto
    dominant = ""
    best_score = 0.0
    for theme, score in themes_parsed.items():
        s = abs(_safe(score) or 0.0)
        if s > best_score:
            best_score = s
            dominant = theme

    # Clusters: inverte ticker_themes para theme → [tickers]
    clusters: dict[str, list[str]] = {}
    for ticker, t_list in ticker_themes.items():
        if ticker not in signals:
            continue
        for theme in (t_list or []):
            clusters.setdefault(theme, []).append(ticker)

    # Ordena cada cluster: leading/improving primeiro
    rrg_sigs = {}
    if rrg_result and hasattr(rrg_result, "signals"):
        rrg_sigs = rrg_result.signals or {}

    _Q_ORDER = {"leading": 0, "improving": 1, "weakening": 2, "lagging": 3, "": 4}
    for theme in clusters:
        clusters[theme].sort(
            key=lambda t: _Q_ORDER.get(
                getattr(rrg_sigs.get(t), "quadrant", ""),
                4,
            )
        )

    return dominant, clusters


# ── Main entry point ─────────────────────────────────────────────────────────────

def compute_desk_intelligence(
    signals: dict[str, "AssetSignal"],
    rrg_result: Any | None = None,
    vol_regime: Any | None = None,
    narrative_result: Any | None = None,
    cta_result: Any | None = None,
    shadow_flow: Any | None = None,
    options_map: dict[str, dict] | None = None,
    network_result: dict | None = None,
    market_prices: dict | None = None,
    flow_pred: Any | None = None,
    swaggy_result: Any | None = None,    # SwaggyResult (WSB + squeeze)
    options_snapshot: Any | None = None, # OptionsSnapshot (Greeks Dashboard BBQ)
) -> DeskIntelligenceResult:
    """
    Ponto de entrada principal.
    Agrega todos os sinais num DeskIntelligenceResult regime-aware.
    """
    if not signals:
        _log.warning("desk_intelligence_no_signals")
        return DeskIntelligenceResult()

    # 0. Vol Options Regime (BBQ Greeks data — maior autoridade em regime de vol)
    vol_options_regime = None
    try:
        from app.analysis.vol_options_regime import compute_vol_options_regime
        vol_options_regime = compute_vol_options_regime(options_snapshot, vol_regime)
    except Exception as exc:
        _log.warning("vol_options_regime_failed", error=str(exc))

    # A. Classificar regime
    market_regime, regime_conf, regime_drivers = classify_market_regime(
        vol_regime=vol_regime,
        network_result=network_result,
        rrg_result=rrg_result,
        narrative_result=narrative_result,
        flow_pred=flow_pred,
        cta_result=cta_result,
    )
    regime_weights = _REGIME_WEIGHTS.get(market_regime, _REGIME_WEIGHTS["dispersed_rotation"])

    # A1. Override de regime pelo VolOptionsRegime (GEX extremo sobrepõe tudo exceto crise)
    if vol_options_regime is not None:
        from app.analysis.vol_options_regime import GexRegime, VolDecision
        _vor = vol_options_regime
        if _vor.fragility_alert and _vor.hedge_signal > 0.65 and market_regime != "stress":
            # GEX + Squeeze/Tail indicam stress estrutural
            market_regime = "stress"
            regime_conf   = min(1.0, regime_conf + 0.15)
            regime_drivers.insert(0, f"options:tail={_vor.tail_score:.0f}+squeeze={_vor.squeeze_score:.0f}")
            regime_weights = _REGIME_WEIGHTS["stress"]
        elif _vor.gex_regime == GexRegime.SHORT_EXTREME and market_regime not in ("stress", "risk_off_defensive"):
            market_regime = "risk_off_defensive"
            regime_conf   = max(regime_conf, 0.70)
            regime_drivers.insert(0, f"options:gex_extreme={_vor.gex_net_bn:+.1f}B")
            regime_weights = _REGIME_WEIGHTS["risk_off_defensive"]
        elif _vor.vol_decision == VolDecision.SELL_VOL and market_regime not in ("stress", "risk_off_defensive", "mechanical_passive"):
            # GEX long + IV rica → vol_squeeze
            if market_regime not in ("risk_on_momentum", "vol_squeeze"):
                market_regime = "vol_squeeze"
                regime_conf   = max(regime_conf, 0.65)
                regime_drivers.insert(0, f"options:sell_vol gex={_vor.gex_net_bn:+.1f}B iv_rv={_vor.iv_rv_pp:+.1f}pp")
                regime_weights = _REGIME_WEIGHTS["vol_squeeze"]
        if _vor.amplification_regime:
            regime_drivers.append("options:gex_amplification")
        if _vor.dampening_regime:
            regime_drivers.append("options:gex_dampening")

    # B. Opportunity scores
    opp_scores = compute_opportunity_scores(
        signals=signals,
        rrg_result=rrg_result,
        options_map=options_map,
        shadow_flow=shadow_flow,
        cta_result=cta_result,
    )

    # C. Fragility scores
    frag_scores = compute_fragility_scores(
        signals=signals,
        rrg_result=rrg_result,
        options_map=options_map,
        shadow_flow=shadow_flow,
        cta_result=cta_result,
    )

    # D. Contagion scores
    cont_scores = compute_contagion_scores(
        signals=signals,
        rrg_result=rrg_result,
        network_result=network_result,
    )

    # E. RRG meta
    rrg_meta = compute_rrg_meta(
        signals=signals,
        rrg_result=rrg_result,
        options_map=options_map,
        shadow_flow=shadow_flow,
        cta_result=cta_result,
    )

    # F. Regime-adjusted scores
    adj_scores = compute_regime_adj_scores(
        signals=signals,
        rrg_result=rrg_result,
        narrative_result=narrative_result,
        cta_result=cta_result,
        shadow_flow=shadow_flow,
        options_map=options_map,
        regime_weights=regime_weights,
    )

    # Rankings
    ranked = sorted(adj_scores, key=lambda t: adj_scores[t], reverse=True)
    top_opp = sorted(opp_scores, key=lambda t: opp_scores[t], reverse=True)[:5]
    top_frag = sorted(frag_scores, key=lambda t: frag_scores[t], reverse=True)[:5]

    # Líderes narrativos: no cluster dominante com score positivo
    rrg_sigs = {}
    if rrg_result and hasattr(rrg_result, "signals"):
        rrg_sigs = rrg_result.signals or {}
    narrative_leaders = [
        t for t in ranked
        if getattr(rrg_sigs.get(t), "quadrant", "") == "leading"
        and adj_scores.get(t, 0) > 0.15
    ][:5]

    # Atrasados com convicção: improving + high conviction
    laggards_conv = [
        t for t in ranked
        if getattr(rrg_sigs.get(t), "quadrant", "") == "improving"
        and (signals[t].conviction == "high" or opp_scores.get(t, 0) > 0.35)
    ][:5]

    # Narrativa e clusters
    dominant_narrative, narrative_clusters = extract_narrative_clusters(
        narrative_result=narrative_result,
        signals=signals,
        rrg_result=rrg_result,
    )

    # G. Explicações textuais
    explanations = build_explanations(
        signals=signals,
        rrg_result=rrg_result,
        opportunity_scores=opp_scores,
        fragility_scores=frag_scores,
        regime_adj_scores=adj_scores,
        options_map=options_map,
        cta_result=cta_result,
        narrative_result=narrative_result,
        dominant_narrative=dominant_narrative,
    )

    n_with_rrg = sum(1 for t in signals if t in rrg_sigs and rrg_sigs[t] is not None)

    # H. Narrative Classifier (motor de assimetria/exaustão)
    narr_classifier_result = None
    try:
        from app.analysis.narrative_classifier import classify_narratives
        narr_classifier_result = classify_narratives(
            signals=signals,
            rrg_result=rrg_result,
            vol_regime=vol_regime,
            narrative_result=narrative_result,
            cta_result=cta_result,
            shadow_flow=shadow_flow,
            options_map=options_map,
            swaggy_result=swaggy_result,
            desk_intel=None,  # evita circularidade
        )
    except Exception as exc:
        _log.warning("narrative_classifier_failed", error=str(exc))

    # I. Chart Regime Detector
    chart_regime_map: dict = {}
    try:
        from app.analysis.chart_regime import detect_chart_regimes
        chart_regime_map = detect_chart_regimes(market_prices) or {}
    except Exception as exc:
        _log.warning("chart_regime_failed", error=str(exc))

    # Extrai campos do narrative_classifier
    ns_map = {}
    if narr_classifier_result:
        ns_map = narr_classifier_result.signals or {}

    result = DeskIntelligenceResult(
        market_regime=market_regime,
        market_regime_label=_REGIME_LABELS.get(market_regime, market_regime),
        regime_confidence=round(regime_conf, 3),
        regime_drivers=regime_drivers[:5],
        regime_weights=regime_weights,
        opportunity_scores=opp_scores,
        fragility_scores=frag_scores,
        contagion_scores=cont_scores,
        regime_adj_scores=adj_scores,
        ranked_assets=ranked,
        top_opportunities=top_opp,
        top_fragile=top_frag,
        narrative_leaders=narrative_leaders,
        laggards_with_conviction=laggards_conv,
        dominant_narrative=dominant_narrative,
        narrative_clusters=narrative_clusters,
        rrg_meta=rrg_meta,
        explanations=explanations,
        # Narrative engine
        narrative_states    ={t: s.narrative_state   for t, s in ns_map.items()},
        asymmetry_scores    ={t: s.asymmetry_score   for t, s in ns_map.items()},
        crowdedness_scores  ={t: s.crowdedness_score for t, s in ns_map.items()},
        exhaustion_scores   ={t: s.exhaustion_score  for t, s in ns_map.items()},
        convictions         ={t: s.conviction        for t, s in ns_map.items()},
        horizons            ={t: s.horizon           for t, s in ns_map.items()},
        opportunity_types   ={t: s.opportunity_type  for t, s in ns_map.items()},
        narrative_rationales={t: s.rationale         for t, s in ns_map.items()},
        consensus_overloads ={t: s.consensus_overload for t, s in ns_map.items()},
        top_asymmetry       =narr_classifier_result.top_asymmetry    if narr_classifier_result else [],
        top_crowded         =narr_classifier_result.top_crowded      if narr_classifier_result else [],
        top_exhausted_narr  =narr_classifier_result.top_exhausted    if narr_classifier_result else [],
        emerging_themes     =narr_classifier_result.emerging_themes  if narr_classifier_result else [],
        short_candidates    =narr_classifier_result.short_candidates if narr_classifier_result else [],
        # Chart regimes
        chart_regimes  ={t: cr.regime_label for t, cr in chart_regime_map.items()},
        chart_rationales={t: cr.rationale   for t, cr in chart_regime_map.items()},
        # WSB
        wsb_top_mentions=(getattr(swaggy_result, "top_mentions", None) or []) if swaggy_result else [],
        top_squeeze     =(getattr(swaggy_result, "top_squeeze",  None) or []) if swaggy_result else [],
        # Vol Options Regime
        vol_options_regime=vol_options_regime,
        n_assets=len(signals),
        n_with_rrg=n_with_rrg,
    )

    _log.info(
        "desk_intelligence_done",
        regime=market_regime,
        confidence=round(regime_conf, 2),
        n_assets=len(signals),
        n_rrg=n_with_rrg,
        top_opp=top_opp[:3],
        top_frag=top_frag[:3],
        emerging_themes=result.emerging_themes[:3],
        short_candidates=result.short_candidates[:3],
        top_asymmetry=result.top_asymmetry[:3],
    )

    return result
