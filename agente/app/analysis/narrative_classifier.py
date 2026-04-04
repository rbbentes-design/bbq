"""
Narrative Classifier — Motor de Narrativa, Assimetria e Exaustão

Classifica ativos e temas em 4 estados narrativos e computa 3 scores:

Estados:
  1. PRE_NARRATIVE    — desvio detectado, consenso ainda não verbalizou
  2. EMERGING         — fluxo acelerando, explicação se consolidando
  3. MATURE           — história disseminada, assimetria marginal caiu
  4. EXHAUSTED        — consenso excessivo, reversão provável

Scores [0-100]:
  - asymmetry_score   — distância entre oportunidade e consenso (maior = melhor assimetria)
  - crowdedness_score — saturação do trade (maior = mais crowded)
  - exhaustion_score  — probabilidade de reversão/exaustão (maior = mais perigoso)

Output por ativo:
  - NarrativeSignal com todos os scores + conviction + horizon + opportunity_type
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.narrative_classifier")

# ── Estados narrativos ────────────────────────────────────────────────────────

class NarrativeState:
    PRE_NARRATIVE = "pre_narrative"    # Pré-narrativa
    EMERGING      = "emerging"         # Emergente
    MATURE        = "mature"           # Matura
    EXHAUSTED     = "exhausted"        # Exaurida


# ── Convicção operacional ─────────────────────────────────────────────────────

class Conviction:
    AGGRESSIVE_LONG = "compra_agressiva"   # narrativa emergente + fluxo + confirmação técnica
    TACTICAL_LONG   = "compra_tatica"      # continuação saudável, payoff marginal ok
    REDUCE          = "reduzir"            # payoff piorando, consenso enchendo
    REALIZE         = "realizar"           # sair da posição
    AVOID           = "evitar"             # narrativa óbvia, r/r ruim
    TACTICAL_SHORT  = "short_tatico"       # exaurida + positioning lotado + falha técnica


# ── Horizonte ─────────────────────────────────────────────────────────────────

class Horizon:
    INTRADAY       = "intraday"
    SWING          = "swing"
    TACTICAL_DAYS  = "tático_dias"
    TACTICAL_WEEKS = "tático_semanas"
    STRUCTURAL     = "estrutural"


# ── Tipo de oportunidade ──────────────────────────────────────────────────────

class OpportunityType:
    BREAKOUT       = "rompimento_inicial"
    CONTINUATION   = "continuação"
    SQUEEZE        = "squeeze"
    REVERSAL       = "reversão"
    EXHAUSTION     = "exaustão"
    SHORT_CONSENSUS = "short_consenso"
    FLOW_ROTATION  = "rotação_de_fluxo"


# ── Data model de saída ───────────────────────────────────────────────────────

@dataclass
class NarrativeSignal:
    ticker: str

    # Estado e scores principais
    narrative_state: str   = NarrativeState.PRE_NARRATIVE
    asymmetry_score: float = 50.0   # [0-100] — maior = mais assimetria vs consenso
    crowdedness_score: float = 0.0  # [0-100] — maior = mais saturado
    exhaustion_score: float  = 0.0  # [0-100] — maior = mais próximo de reversão

    # Decisão
    conviction: str         = Conviction.AVOID
    horizon: str            = Horizon.TACTICAL_WEEKS
    opportunity_type: str   = OpportunityType.CONTINUATION

    # Texto de racional
    rationale: str = ""

    # Sinais intermediários (para debug/auditoria)
    attention_score: float  = 0.0   # WSB/social attention [0,1]
    flow_acceleration: float = 0.0  # aceleração de fluxo [0,1]
    technical_momentum: float = 0.0 # momentum técnico [-1,1]
    consensus_density: float  = 0.0 # densidade do consenso [0,1]

    # Alerta de consenso saturado
    consensus_overload: bool = False


@dataclass
class NarrativeClassifierResult:
    signals: dict[str, NarrativeSignal] = field(default_factory=dict)
    # Listas ranqueadas
    top_asymmetry: list[str]   = field(default_factory=list)  # maiores assimetrias
    top_crowded: list[str]     = field(default_factory=list)  # mais crowded
    top_exhausted: list[str]   = field(default_factory=list)  # mais exauridos
    emerging_themes: list[str] = field(default_factory=list)  # tickers em pré/emergente
    short_candidates: list[str]= field(default_factory=list)  # short tático


# ── Motor principal ───────────────────────────────────────────────────────────

def classify_narratives(
    signals: dict[str, Any],           # dict[ticker, AssetSignal]
    rrg_result: Any | None,            # RRGResult
    vol_regime: Any | None,            # VolRegimeResult
    narrative_result: Any | None,      # NarrativeAlphaResult
    cta_result: Any | None,            # CTAPositioningResult
    shadow_flow: Any | None,           # ShadowFlowResult
    options_map: dict[str, Any] | None,
    swaggy_result: Any | None,         # SwaggyResult (WSB + squeeze)
    desk_intel: Any | None,            # DeskIntelligenceResult (se disponível)
) -> NarrativeClassifierResult:
    """Classifica todos os ativos em estados narrativos e computa scores."""

    result = NarrativeClassifierResult()

    rrg_map  = _build_rrg_map(rrg_result)
    cta_map  = _build_cta_map(cta_result)
    flow_map = _build_flow_map(shadow_flow)
    narr_map = _build_narrative_map(narrative_result)

    for ticker, sig in signals.items():
        try:
            ns = _classify_ticker(
                ticker, sig, rrg_map, cta_map, flow_map, narr_map,
                options_map, swaggy_result, vol_regime, desk_intel,
            )
            result.signals[ticker] = ns
        except Exception as exc:
            _log.warning("narrative_classify_error", ticker=ticker, error=str(exc))

    # Rankings
    all_ns = list(result.signals.values())
    result.top_asymmetry   = _rank(all_ns, "asymmetry_score", top=10)
    result.top_crowded     = _rank(all_ns, "crowdedness_score", top=10)
    result.top_exhausted   = _rank(all_ns, "exhaustion_score", top=10)
    result.emerging_themes = [
        n.ticker for n in all_ns
        if n.narrative_state in (NarrativeState.PRE_NARRATIVE, NarrativeState.EMERGING)
        and n.asymmetry_score > 55
    ]
    result.short_candidates = [
        n.ticker for n in all_ns
        if n.conviction == Conviction.TACTICAL_SHORT
    ]

    _log.info("narrative_classifier_done",
              tickers=len(result.signals),
              emerging=len(result.emerging_themes),
              exhausted=len(result.top_exhausted),
              short_candidates=len(result.short_candidates))

    return result


# ── Classificação por ticker ──────────────────────────────────────────────────

def _classify_ticker(
    ticker: str,
    sig: Any,
    rrg_map: dict,
    cta_map: dict,
    flow_map: dict,
    narr_map: dict,
    options_map: dict | None,
    swaggy: Any | None,
    vol_regime: Any | None,
    desk_intel: Any | None,
) -> NarrativeSignal:

    ns = NarrativeSignal(ticker=ticker)

    # ── 1. Sinais de entrada ──────────────────────────────────────────────────
    composite      = float(getattr(sig, "composite", 0) or 0)
    momentum_score = float(getattr(sig, "momentum_score", 0) or 0)
    tail_score     = float(getattr(sig, "tail_score", 0) or 0)
    conv           = getattr(sig, "conviction", "low") or "low"
    direction      = getattr(sig, "direction", "neutral") or "neutral"
    dp_score       = float(flow_map.get(ticker, 0) or 0)        # dark pool
    cta_score      = float(cta_map.get(ticker, {}).get("score", 0) or 0)
    crowding       = str(cta_map.get(ticker, {}).get("crowding", "") or "")
    narr_comp      = float(narr_map.get(ticker, 0) or 0)        # narrative composite

    rrg            = rrg_map.get(ticker, {})
    quadrant       = str(rrg.get("quadrant", "lagging") or "lagging")
    rs_ratio       = float(rrg.get("rs_ratio", 100) or 100)
    rs_mom         = float(rrg.get("rs_momentum", 100) or 100)

    opts           = (options_map or {}).get(ticker, {}) or {}
    iv_pct         = opts.get("iv_percentile")
    skew           = opts.get("skew_5pct")

    attn           = _get_wsb_attention(swaggy, ticker)
    squeeze_score  = _get_squeeze_score(swaggy, ticker)

    opp_score      = float((getattr(desk_intel, "opportunity_scores", None) or {}).get(ticker, 0) or 0)
    frag_score     = float((getattr(desk_intel, "fragility_scores",   None) or {}).get(ticker, 0) or 0)

    vol_stress     = float(getattr(vol_regime, "stress_score", 0) or 0) if vol_regime else 0.0

    # ── 2. Attention / propagação ─────────────────────────────────────────────
    # Combina: WSB attention + narrative composite + dark pool
    attention = _w(attn, 0.35) + _w(_norm(narr_comp, -1, 1), 0.40) + _w(_norm(abs(dp_score), 0, 1), 0.25)
    ns.attention_score = attention

    # ── 3. Flow acceleration ──────────────────────────────────────────────────
    # Combina: dark pool + options flow + CTA
    opts_flow = float(getattr(sig, "options_flow_score", 0) or 0)
    flow_acc  = _w(_norm(abs(dp_score), 0, 0.5), 0.40) + _w(_norm(abs(opts_flow), 0, 1), 0.35) + _w(_norm(abs(cta_score), 0, 1), 0.25)
    ns.flow_acceleration = flow_acc

    # ── 4. Technical momentum ─────────────────────────────────────────────────
    tech = momentum_score * 0.5 + composite * 0.5
    ns.technical_momentum = tech

    # ── 5. Crowdedness ────────────────────────────────────────────────────────
    crowd = 0.0
    if crowding in ("extreme_long", "extreme"):
        crowd += 0.40
    elif crowding in ("crowded_long", "crowded"):
        crowd += 0.25
    if quadrant == "leading" and rs_ratio > 105:
        crowd += 0.20
    if attn > 0.5:
        crowd += 0.20   # WSB attention alta = varejo entrou
    if iv_pct is not None and iv_pct > 0.75:
        crowd += 0.10   # IV cara = medo já precificado
    if abs(narr_comp) > 0.4:
        crowd += 0.10   # narrativa saturada
    ns.crowdedness_score = round(min(crowd, 1.0) * 100, 1)

    # ── 6. Exhaustion ─────────────────────────────────────────────────────────
    exhaust = 0.0
    if quadrant == "weakening":
        exhaust += 0.25
    if tail_score and tail_score > 0.6:
        exhaust += 0.20
    if frag_score > 0.4:
        exhaust += 0.20
    if momentum_score < -0.1 and composite > 0.1:
        exhaust += 0.15   # divergência: composite positivo mas momentum caindo
    if skew is not None and skew > 0.05:
        exhaust += 0.10   # put skew elevado = mercado se protegendo
    if crowd > 0.5 and abs(dp_score) < 0.1:
        exhaust += 0.10   # muito crowded mas dark pool sem confirmação
    ns.exhaustion_score = round(min(exhaust, 1.0) * 100, 1)

    # ── 7. Asymmetry ──────────────────────────────────────────────────────────
    # Maior quando: opp_score alto + crowd baixo + atenção baixa (ainda não percebido)
    asym = 50.0  # base neutra
    asym += opp_score * 30           # oportunidade detectada pelo desk_intel
    asym -= ns.crowdedness_score * 0.25  # crowded penaliza assimetria
    asym += (1 - attention) * 15     # baixa atenção = ainda não precificado
    asym += flow_acc * 10            # fluxo acelerando = assimetria real
    if squeeze_score > 0.4:
        asym += squeeze_score * 15   # squeeze potencial = assimetria mecânica
    ns.asymmetry_score = round(max(0, min(100, asym)), 1)

    # ── 8. Consensus density (para alerta) ───────────────────────────────────
    consensus = (ns.crowdedness_score / 100) * 0.4 + (attn) * 0.3 + _norm(abs(narr_comp), 0, 1) * 0.3
    ns.consensus_density = consensus
    ns.consensus_overload = (
        ns.crowdedness_score > 65
        and ns.exhaustion_score > 50
        and attention > 0.5
        and momentum_score < 0.1
    )

    # ── 9. Estado narrativo ───────────────────────────────────────────────────
    ns.narrative_state = _classify_state(
        attention, flow_acc, ns.crowdedness_score, ns.exhaustion_score,
        quadrant, composite, dp_score,
    )

    # ── 10. Conviction + Horizon + Type ──────────────────────────────────────
    ns.conviction, ns.horizon, ns.opportunity_type = _compute_conviction(
        ns.narrative_state, ns.asymmetry_score, ns.crowdedness_score,
        ns.exhaustion_score, direction, quadrant, squeeze_score, vol_stress,
    )

    # ── 11. Racional ──────────────────────────────────────────────────────────
    ns.rationale = _build_rationale(ns, quadrant, crowding, attn, dp_score, squeeze_score)

    return ns


# ── Estado narrativo ──────────────────────────────────────────────────────────

def _classify_state(
    attention: float,
    flow_acc: float,
    crowdedness: float,
    exhaustion: float,
    quadrant: str,
    composite: float,
    dp_score: float,
) -> str:
    # Exaurida: consenso saturado + sinais de reversão
    if exhaustion > 55 and crowdedness > 55:
        return NarrativeState.EXHAUSTED

    # Matura: narrativa disseminada, fluxo ainda existe mas assimetria caiu
    if crowdedness > 45 and attention > 0.35:
        return NarrativeState.MATURE

    # Emergente: fluxo acelerando, atenção crescendo, ainda não saturado
    if flow_acc > 0.3 and attention > 0.15 and crowdedness < 40:
        return NarrativeState.EMERGING

    # Pré-narrativa: desvio de preço/fluxo, atenção ainda baixa
    if (abs(dp_score) > 0.15 or abs(composite) > 0.15) and attention < 0.15:
        return NarrativeState.PRE_NARRATIVE

    # Default: pré-narrativa se composite tem sinal, matura se tem atenção
    if attention > 0.25:
        return NarrativeState.MATURE
    return NarrativeState.PRE_NARRATIVE


# ── Conviction ────────────────────────────────────────────────────────────────

def _compute_conviction(
    state: str,
    asymmetry: float,
    crowdedness: float,
    exhaustion: float,
    direction: str,
    quadrant: str,
    squeeze_score: float,
    vol_stress: float,
) -> tuple[str, str, str]:
    """Retorna (conviction, horizon, opportunity_type)."""

    # Short tático: exaurida + crowded + quadrante weakening/lagging
    if state == NarrativeState.EXHAUSTED and crowdedness > 60 and exhaustion > 55:
        return Conviction.TACTICAL_SHORT, Horizon.TACTICAL_WEEKS, OpportunityType.SHORT_CONSENSUS

    # Squeeze: squeeze_score alto + narrativa emergente
    if squeeze_score > 0.5 and state in (NarrativeState.PRE_NARRATIVE, NarrativeState.EMERGING):
        return Conviction.AGGRESSIVE_LONG, Horizon.TACTICAL_DAYS, OpportunityType.SQUEEZE

    # Compra agressiva: pré/emergente + alta assimetria + direção long
    if state == NarrativeState.PRE_NARRATIVE and asymmetry > 65 and direction == "long":
        return Conviction.AGGRESSIVE_LONG, Horizon.TACTICAL_WEEKS, OpportunityType.BREAKOUT

    if state == NarrativeState.EMERGING and asymmetry > 60 and crowdedness < 35:
        return Conviction.AGGRESSIVE_LONG, Horizon.TACTICAL_WEEKS, OpportunityType.CONTINUATION

    # Compra tática: emergente/matura com payoff ainda ok
    if state in (NarrativeState.EMERGING, NarrativeState.MATURE) and asymmetry > 50 and crowdedness < 55:
        return Conviction.TACTICAL_LONG, Horizon.TACTICAL_WEEKS, OpportunityType.CONTINUATION

    # Rotação: improving no RRG = entrada antecipada
    if quadrant == "improving" and asymmetry > 55:
        return Conviction.TACTICAL_LONG, Horizon.TACTICAL_WEEKS, OpportunityType.FLOW_ROTATION

    # Reduzir: matura com crowdedness alto
    if state == NarrativeState.MATURE and crowdedness > 55:
        return Conviction.REDUCE, Horizon.SWING, OpportunityType.CONTINUATION

    # Realizar: exaurida
    if state == NarrativeState.EXHAUSTED:
        return Conviction.REALIZE, Horizon.SWING, OpportunityType.EXHAUSTION

    # Evitar: stress alto ou r/r ruim
    if vol_stress > 0.5 or (crowdedness > 50 and asymmetry < 45):
        return Conviction.AVOID, Horizon.TACTICAL_WEEKS, OpportunityType.CONTINUATION

    return Conviction.AVOID, Horizon.TACTICAL_WEEKS, OpportunityType.CONTINUATION


# ── Racional ──────────────────────────────────────────────────────────────────

def _build_rationale(
    ns: NarrativeSignal,
    quadrant: str,
    crowding: str,
    attn: float,
    dp_score: float,
    squeeze_score: float,
) -> str:
    parts = []

    state_labels = {
        NarrativeState.PRE_NARRATIVE: "pré-narrativa",
        NarrativeState.EMERGING:      "emergente",
        NarrativeState.MATURE:        "matura",
        NarrativeState.EXHAUSTED:     "exaurida",
    }
    parts.append(f"Narrativa {state_labels.get(ns.narrative_state, ns.narrative_state)}.")

    if ns.asymmetry_score > 65:
        parts.append(f"Assimetria alta ({ns.asymmetry_score:.0f}) — oportunidade ainda não precificada pelo consenso.")
    elif ns.asymmetry_score < 35:
        parts.append(f"Assimetria baixa ({ns.asymmetry_score:.0f}) — payoff marginal deteriorado.")

    if ns.crowdedness_score > 60:
        parts.append(f"Trade crowded ({ns.crowdedness_score:.0f}) — risco de liquidez de saída.")
    if ns.exhaustion_score > 55:
        parts.append(f"Sinais de exaustão ({ns.exhaustion_score:.0f}) — probabilidade de reversão crescente.")

    if quadrant == "improving":
        parts.append("RS-Ratio em recuperação — entrada antecipada possível.")
    elif quadrant == "weakening":
        parts.append("RS-Momentum deteriorando — preparar saída.")
    elif quadrant == "lagging":
        parts.append("Ativo lagging no universo — evitar long.")

    if abs(dp_score) > 0.2:
        direction = "acumulação" if dp_score > 0 else "distribuição"
        parts.append(f"Dark pool sinalizando {direction} (score {dp_score:+.2f}).")

    if squeeze_score > 0.4:
        parts.append(f"Squeeze score elevado ({squeeze_score:.2f}) — combustível mecânico para alta.")

    if ns.consensus_overload:
        parts.append("⚠ ALERTA: trade bonito demais. Mercado pode estar vendendo a saída para o público.")

    return " ".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rank(signals: list[NarrativeSignal], attr: str, top: int = 10) -> list[str]:
    return [s.ticker for s in sorted(signals, key=lambda x: getattr(x, attr, 0), reverse=True)[:top]]


def _norm(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def _w(v: float, weight: float) -> float:
    return v * weight


def _build_rrg_map(rrg_result: Any | None) -> dict[str, dict]:
    if not rrg_result:
        return {}
    out = {}
    for sig in (getattr(rrg_result, "signals", None) or []):
        t = getattr(sig, "ticker", None)
        if t:
            out[t] = {
                "quadrant":    getattr(sig, "quadrant", "lagging"),
                "rs_ratio":    getattr(sig, "rs_ratio", 100),
                "rs_momentum": getattr(sig, "rs_momentum", 100),
            }
    return out


def _build_cta_map(cta_result: Any | None) -> dict[str, dict]:
    if not cta_result:
        return {}
    out = {}
    for t, v in (getattr(cta_result, "by_ticker", None) or {}).items():
        out[t] = {
            "score":    getattr(v, "cta_score", 0) if hasattr(v, "cta_score") else (v.get("cta_score", 0) if isinstance(v, dict) else 0),
            "crowding": getattr(v, "crowding", "") if hasattr(v, "crowding") else (v.get("crowding", "") if isinstance(v, dict) else ""),
        }
    # Se cta_result tem campos diretos (não by_ticker)
    if not out and hasattr(cta_result, "cta_score"):
        pass  # CTA global, não por ticker
    return out


def _build_flow_map(shadow_flow: Any | None) -> dict[str, float]:
    if not shadow_flow:
        return {}
    by_ticker = getattr(shadow_flow, "by_ticker", None) or {}
    out = {}
    for t, v in by_ticker.items():
        score = getattr(v, "dark_pool_score", 0) if hasattr(v, "dark_pool_score") else (v.get("dark_pool_score", 0) if isinstance(v, dict) else 0)
        out[t] = float(score or 0)
    # Fallback: scores diretos
    if not out:
        scores = getattr(shadow_flow, "dark_pool_scores", None) or {}
        out = {t: float(v or 0) for t, v in scores.items()}
    return out


def _build_narrative_map(narrative_result: Any | None) -> dict[str, float]:
    if not narrative_result:
        return {}
    # composite_narrative pode ser por ticker ou global
    per_ticker = getattr(narrative_result, "per_ticker_composite", None) or {}
    if per_ticker:
        return {t: float(v or 0) for t, v in per_ticker.items()}
    # Fallback: mesma narrativa para todos
    global_comp = float(getattr(narrative_result, "composite_narrative", 0) or 0)
    return {}  # sem granularidade por ticker, retorna vazio


def _get_wsb_attention(swaggy: Any | None, ticker: str) -> float:
    if not swaggy:
        return 0.0
    m = (getattr(swaggy, "mention_map", None) or {}).get(ticker)
    return float(getattr(m, "attention_score", 0) or 0) if m else 0.0


def _get_squeeze_score(swaggy: Any | None, ticker: str) -> float:
    if not swaggy:
        return 0.0
    s = (getattr(swaggy, "squeeze_map", None) or {}).get(ticker)
    return float(getattr(s, "squeeze_score", 0) or 0) if s else 0.0
