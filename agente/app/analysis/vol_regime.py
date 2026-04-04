"""
Vol Regime Engine

Classifica o regime de volatilidade usando:
  1. VIX level (spot)
  2. VIX term structure: VIX / VIX3M ratio
     - Contango (< 0.90): mercado calmo, dealers vendendo vol
     - Flat (0.90-1.05): transicao
     - Backwardation (> 1.05): stress, dealers comprando vol
  3. VVIX (vol-of-vol): spike indica crise iminente
  4. SPX realized vol (10d) vs VIX: ratio mostra se IV justa ou premium

Output: VolRegime com score de stress [0, 1] e regime: calm/elevated/stress/crisis

Uso no portfolio:
  - calm: posicoes cheias, max leverage
  - elevated: posicoes 80%, hedge parcial
  - stress: posicoes 60%, hedge obrigatorio (VXX/puts)
  - crisis: cash 50%, apenas longs defensivos e vol long

Tambem calcula:
  - VIX flip direction (VIX subindo vs caindo nos ultimos 5d)
  - Vol risk premium (VIX - realized_vol): premio para vender vol ou comprar
  - Term structure slope (VIX / VIX3M): inversao sinaliza capitulacao
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

from app.audit.logger import get_logger

_log = get_logger("analysis.vol_regime")


@dataclass
class VolRegimeResult:
    # Core metrics
    vix: float | None = None
    vix3m: float | None = None
    vvix: float | None = None
    realized_vol_10d: float | None = None

    # Computed
    term_structure_ratio: float | None = None   # VIX / VIX3M
    vol_risk_premium: float | None = None       # VIX - realized_vol
    vix_5d_change: float | None = None          # VIX hoje vs 5d atras

    # Regime classification
    regime: str = "unknown"          # calm | elevated | stress | crisis
    stress_score: float = 0.5        # [0, 1] — 0=calm, 1=crisis
    vol_direction: str = "unknown"   # rising | falling | stable

    # Signals
    term_structure_signal: float = 0.0   # [-1, 1] — +1=contango (calm), -1=backwdation
    vrp_signal: float = 0.0              # [-1, 1] — +1=sell vol (IV rich), -1=buy vol
    crisis_indicator: bool = False

    # Sizing recommendations
    position_scalar: float = 1.0    # multiply all allocations by this
    hedge_required: bool = False
    hedge_asset: str = "BIL"        # fallback hedge when vol spikes

    # Rationale
    reasons: list[str] = field(default_factory=list)
    timestamp: str = ""


def _fetch_vix_data() -> dict:
    """Busca VIX, VIX3M, VVIX, SPX para calcular realized vol."""
    result = {}
    try:
        import yfinance as yf
        import datetime as dt

        end = dt.datetime.now()
        start = end - dt.timedelta(days=30)

        # VIX spot + historia 10d
        vix_tk = yf.Ticker("^VIX")
        vix_hist = vix_tk.history(period="15d")
        if not vix_hist.empty:
            result["vix"] = float(vix_hist["Close"].iloc[-1])
            if len(vix_hist) >= 6:
                result["vix_5d_ago"] = float(vix_hist["Close"].iloc[-6])

        # VIX3M
        vix3m_tk = yf.Ticker("^VIX3M")
        vix3m_hist = vix3m_tk.history(period="5d")
        if not vix3m_hist.empty:
            result["vix3m"] = float(vix3m_hist["Close"].iloc[-1])

        # VVIX
        vvix_tk = yf.Ticker("^VVIX")
        vvix_hist = vvix_tk.history(period="5d")
        if not vvix_hist.empty:
            result["vvix"] = float(vvix_hist["Close"].iloc[-1])

        # SPX realized vol (10d)
        spx_hist = yf.Ticker("^GSPC").history(period="20d")
        if len(spx_hist) >= 11:
            rets = spx_hist["Close"].pct_change().dropna().iloc[-10:]
            result["realized_vol_10d"] = float(rets.std() * math.sqrt(252)) * 100

    except Exception as exc:
        _log.debug("vix_fetch_error", error=str(exc))

    return result


def compute_vol_regime(
    market_prices: dict | None = None,
    options_snapshot: "Any | None" = None,   # OptionsSnapshot | None (BBQ data)
) -> VolRegimeResult:
    """
    Calcula o regime de volatilidade.
    market_prices: usado como fallback se fetch falhar.
    """
    res = VolRegimeResult(timestamp=datetime.now().isoformat())

    data = _fetch_vix_data()

    # Fallback: tenta pegar VIX do market_prices se fetch falhou
    if not data.get("vix") and market_prices:
        for k in ("^VIX", "VIX", "vix"):
            if k in market_prices and market_prices[k].get("price"):
                data["vix"] = float(market_prices[k]["price"])
                break

    vix    = data.get("vix")
    vix3m  = data.get("vix3m")
    vvix   = data.get("vvix")
    rv10   = data.get("realized_vol_10d")
    vix_5d = data.get("vix_5d_ago")

    res.vix = vix
    res.vix3m = vix3m
    res.vvix = vvix
    res.realized_vol_10d = rv10

    if vix is None:
        res.regime = "unknown"
        res.stress_score = 0.5
        res.position_scalar = 0.85  # conservador quando sem dados
        res.reasons.append("VIX nao disponivel — usando regime conservador")
        return res

    # ── Term structure ────────────────────────────────────────────────────────
    if vix3m and vix3m > 0:
        ts_ratio = vix / vix3m
        res.term_structure_ratio = ts_ratio
        # Contango: VIX < VIX3M (ratio < 1) → calmo
        # Backwardation: VIX > VIX3M (ratio > 1) → stress
        if ts_ratio < 0.90:
            ts_score = -0.5   # forte contango = calmo
            res.reasons.append(f"Forte contango VIX/VIX3M={ts_ratio:.2f} — mercado calmo")
        elif ts_ratio < 0.98:
            ts_score = -0.1
            res.reasons.append(f"Contango moderado VIX/VIX3M={ts_ratio:.2f}")
        elif ts_ratio < 1.05:
            ts_score = 0.2
            res.reasons.append(f"Term structure flat VIX/VIX3M={ts_ratio:.2f} — transicao")
        elif ts_ratio < 1.15:
            ts_score = 0.6
            res.reasons.append(f"Backwardation VIX/VIX3M={ts_ratio:.2f} — stress")
        else:
            ts_score = 1.0
            res.reasons.append(f"Forte backwardation VIX/VIX3M={ts_ratio:.2f} — CRISE")
        res.term_structure_signal = -ts_score  # invertido: +1=calmo, -1=stress
    else:
        ts_score = 0.3  # sem VIX3M, assume stress moderado

    # ── VIX level score ───────────────────────────────────────────────────────
    if vix < 13:
        vix_score = 0.0
        res.reasons.append(f"VIX={vix:.1f} — extremamente calmo (possivel complacencia)")
    elif vix < 18:
        vix_score = 0.15
        res.reasons.append(f"VIX={vix:.1f} — calmo")
    elif vix < 24:
        vix_score = 0.35
        res.reasons.append(f"VIX={vix:.1f} — elevado")
    elif vix < 32:
        vix_score = 0.60
        res.reasons.append(f"VIX={vix:.1f} — stress")
    elif vix < 45:
        vix_score = 0.80
        res.reasons.append(f"VIX={vix:.1f} — crise")
    else:
        vix_score = 1.0
        res.reasons.append(f"VIX={vix:.1f} — CRISE EXTREMA (similar a 2020/2008)")
        res.crisis_indicator = True

    # ── VVIX (vol-of-vol) ─────────────────────────────────────────────────────
    vvix_score = 0.0
    if vvix:
        if vvix > 130:
            vvix_score = 0.9
            res.reasons.append(f"VVIX={vvix:.0f} — vol-of-vol extrema, spike de crise")
            res.crisis_indicator = True
        elif vvix > 110:
            vvix_score = 0.5
            res.reasons.append(f"VVIX={vvix:.0f} — vol-of-vol elevada")
        elif vvix > 90:
            vvix_score = 0.25

    # ── Realized vol premium ──────────────────────────────────────────────────
    vrp = 0.0
    if rv10:
        vrp = vix - rv10
        res.vol_risk_premium = vrp
        if vrp > 10:
            vrp_signal = 1.0   # IV muito premium vs RV → vender vol
            res.reasons.append(f"VRP={vrp:.1f} — IV muito premium vs RV, oportunidade de vender vol")
        elif vrp > 4:
            vrp_signal = 0.5
        elif vrp > 0:
            vrp_signal = 0.1
        elif vrp > -3:
            vrp_signal = -0.2  # IV barata vs RV → comprar vol
        else:
            vrp_signal = -0.8
            res.reasons.append(f"VRP={vrp:.1f} — IV BARATA vs RV — comprar vol (VXX/straddles)")
        res.vrp_signal = vrp_signal * 0.5  # normalizado para [-0.5, 0.5]
    else:
        vrp_signal = 0.3

    # ── VIX direction ─────────────────────────────────────────────────────────
    if vix_5d:
        vix_change = (vix - vix_5d) / max(vix_5d, 1.0)
        res.vix_5d_change = vix_change
        if vix_change > 0.15:
            res.vol_direction = "rising_fast"
            res.reasons.append(f"VIX subiu {vix_change:+.0%} em 5d — tendencia de alta de vol")
        elif vix_change > 0.05:
            res.vol_direction = "rising"
        elif vix_change < -0.15:
            res.vol_direction = "falling_fast"
            res.reasons.append(f"VIX caiu {vix_change:+.0%} em 5d — vol se normalizando")
        elif vix_change < -0.05:
            res.vol_direction = "falling"
        else:
            res.vol_direction = "stable"
    else:
        res.vol_direction = "unknown"

    # ── Stress score composto ─────────────────────────────────────────────────
    stress = (
        0.40 * vix_score +
        0.30 * ts_score +
        0.15 * vvix_score +
        0.15 * (1 - vrp_signal / 2.0)  # vrp alto = calmo (vender vol = ganhar)
    )
    stress = max(0.0, min(1.0, stress))
    res.stress_score = stress

    # ── Regime classification ─────────────────────────────────────────────────
    if stress < 0.20:
        res.regime = "calm"
        res.position_scalar = 1.10    # slightly increase
        res.hedge_required = False
        res.hedge_asset = "BIL"
    elif stress < 0.40:
        res.regime = "elevated"
        res.position_scalar = 1.00
        res.hedge_required = False
    elif stress < 0.65:
        res.regime = "stress"
        res.position_scalar = 0.80
        res.hedge_required = True
        res.hedge_asset = "VXX"       # long vol ETF em stress
    else:
        res.regime = "crisis"
        res.position_scalar = 0.55
        res.hedge_required = True
        res.hedge_asset = "VXX"
        res.crisis_indicator = True

    # ── Enriquecimento com dados do Greeks Dashboard (BBG) ────────────────────
    if options_snapshot is not None:
        try:
            _enrich_from_options_snapshot(res, options_snapshot)
        except Exception as exc:
            _log.debug("vol_regime_options_enrich_failed", error=str(exc))

    _log.info("vol_regime_computed",
              regime=res.regime,
              stress=round(res.stress_score, 3),
              vix=res.vix,
              ts_ratio=res.term_structure_ratio)

    return res


def _enrich_from_options_snapshot(res: VolRegimeResult, snap: "Any") -> None:
    """
    Enriquece VolRegimeResult com dados BBG do Greeks Dashboard.

    Prioridade BBG > yfinance para VIX.
    GEX negativo aumenta stress (amplificação).
    Squeeze/Tail elevados elevam stress_score.
    Flow score positivo reduz levemente stress.
    """
    # VIX BBG é mais preciso que yfinance
    vix_bbg = float(snap.vix or 0)
    if vix_bbg > 0 and (res.vix is None or abs(vix_bbg - (res.vix or 0)) > 0.5):
        res.vix = vix_bbg
        res.reasons.append(f"VIX BBG={vix_bbg:.2f} (fonte Bloomberg)")

    # IV/RV premium — BBG é referência
    iv_rv = float(snap.iv_rv_pp or 0)
    if iv_rv > 0:
        res.vol_risk_premium = iv_rv
        if iv_rv > 6:
            res.vrp_signal = 0.5
            res.reasons.append(f"VRP BBG={iv_rv:+.1f}pp — IV muito premium, sell vol")
        elif iv_rv > 3:
            res.vrp_signal = 0.25
        elif iv_rv < -3:
            res.vrp_signal = -0.5
            res.reasons.append(f"VRP BBG={iv_rv:+.1f}pp — IV barata, buy vol")

    # GEX ajusta stress: short gamma → amplifica → mais stress
    gex = float(snap.gex_net_bn or 0)
    gex_adj = 0.0
    if gex < -3.0:
        gex_adj = 0.15
        res.reasons.append(f"GEX={gex:+.1f}B extremamente negativo — amplificação, +stress")
    elif gex < -0.5:
        gex_adj = 0.06
    elif gex > 3.0:
        gex_adj = -0.08  # long gamma forte → amortece → menos stress
        res.reasons.append(f"GEX={gex:+.1f}B muito positivo — amortecimento, -stress")
    elif gex > 0.5:
        gex_adj = -0.04

    # Squeeze + Tail
    sq   = float(snap.squeeze_score or 0)
    tail = float(snap.tail_score or 0)
    sq_adj   = min(sq / 100, 1.0) * 0.12
    tail_adj = min(tail / 100, 1.0) * 0.15

    if sq > 75:
        res.reasons.append(f"Squeeze={sq:.0f}/100 — compressão extrema, +stress")
    if tail > 65:
        res.reasons.append(f"Tail Risk={tail:.0f}/100 — risco de cauda alto, +stress")
        if not res.crisis_indicator and tail > 80:
            res.crisis_indicator = True

    # Flow score negativo adiciona stress
    flow = float(snap.flow_score_total or 50)
    flow_adj = 0.0
    if flow < 30:
        flow_adj = 0.05
    elif flow > 70:
        flow_adj = -0.03

    # Aplica ajuste ao stress_score
    new_stress = res.stress_score + gex_adj + sq_adj + tail_adj + flow_adj
    res.stress_score = max(0.0, min(1.0, new_stress))

    # Re-classifica regime se stress mudou
    s = res.stress_score
    if s >= 0.65:
        res.regime = "crisis" if s >= 0.80 else "stress"
        res.hedge_required = True
        res.position_scalar = 0.55 if s >= 0.80 else 0.75
        res.hedge_asset = "VXX"
    elif s >= 0.40:
        if res.regime not in ("stress", "crisis"):
            res.regime = "elevated"
            res.position_scalar = min(res.position_scalar, 1.00)
    elif s < 0.20:
        res.regime = "calm"
        res.position_scalar = min(res.position_scalar, 1.10)
