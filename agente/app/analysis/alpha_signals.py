"""
Alpha Signal Engine

Combina todos os sinais disponíveis por ativo num score composto de alpha,
ajustado por risco de contagio (DebtRank), tail risk e regime macro.

Sinais por ticker:
  1. Momentum        — HMM P(bull) × trend (weekly vs YTD velocity)
  2. Mean Reversion  — z-score vs rolling mean (oversold = oportunidade)
  3. Vol Edge        — IV premium vs RV (writer edge vs buyer edge)
  4. Options Flow    — GEX direction + PCR (dealer positioning pressure)
  5. Contagion       — Katz centrality no MST (propagação em cascata)
  6. Tail Risk       — tail_score penaliza retorno esperado

Score composto:
  alpha = 0.35*momentum + 0.20*mean_rev + 0.20*vol_edge + 0.25*options_flow
  final = alpha * (1 - 0.4*contagion) * (1 - 0.25*tail_risk)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.alpha_signals")


@dataclass
class AssetSignal:
    ticker: str
    name: str = ""

    # Raw market
    price: float | None = None
    daily_ret: float | None = None
    weekly_ret: float | None = None
    ytd_ret: float | None = None
    vol_ann: float | None = None
    pe_ratio: float | None = None
    mkt_cap: float | None = None

    # Probabilistic
    regime_bull: float | None = None   # P(bull) from HMM [0, 1]
    tail_score: float | None = None    # fat tails [0, 1]
    var_95: float | None = None
    cvar_95: float | None = None
    dominant_cycle: int | None = None

    # Options
    iv: float | None = None
    gex_b: float | None = None         # GEX in $B (neg = short gamma = momentum amp)
    pcr: float | None = None           # put/call ratio
    iv_skew: float | None = None       # 25d risk reversal (neg = put premium)
    iv_percentile: float | None = None

    # Network
    mst_degree: int = 0
    contagion_katz: float = 0.0        # Katz centrality [0, 1]
    cluster_id: str = ""

    # Computed signals [-1, 1]
    momentum_score: float = 0.0
    mean_rev_score: float = 0.0
    vol_edge_score: float = 0.0
    options_flow_score: float = 0.0

    # Penalties [0, 1]
    contagion_penalty: float = 0.0
    tail_penalty: float = 0.0

    # Final
    composite: float = 0.0
    expected_return_ann: float = 0.0   # estimated 1-year forward return
    risk_score: float = 0.0            # annualized risk measure
    sharpe_implied: float = 0.0

    direction: str = "neutral"         # long | short | neutral
    conviction: str = "low"            # high | medium | low
    rationale: list[str] = field(default_factory=list)


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _norm01(v: float | None, lo: float, hi: float) -> float:
    """Normaliza v para [0, 1] dado range esperado."""
    if v is None:
        return 0.5
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def _safe(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


# ── Signal 1: Momentum ─────────────────────────────────────────────────────────

def _compute_momentum(regime_bull: float | None,
                      daily: float | None,
                      weekly: float | None,
                      ytd: float | None) -> tuple[float, list[str]]:
    """
    Momentum = 0.5 × HMM_signal + 0.3 × short_trend + 0.2 × medium_trend

    HMM: regime_bull > 0.6 → bull signal (+1); < 0.4 → bear (-1)
    Short trend: weekly > 0 → +; < 0 → -
    Medium trend: YTD vs typical annual vol (z-score proxy)
    """
    reasons = []
    scores = []

    if regime_bull is not None:
        hmm = _clamp((regime_bull - 0.5) * 4)   # 0.5 → 0, 0.75 → 1, 0.25 → -1
        scores.append((0.5, hmm))
        if regime_bull > 0.65:
            reasons.append(f"Regime HMM bull {regime_bull:.0%}")
        elif regime_bull < 0.35:
            reasons.append(f"Regime HMM bear {regime_bull:.0%}")

    if weekly is not None:
        wk = _clamp(weekly / 0.03)  # 3% weekly = max signal
        scores.append((0.3, wk))
        if abs(weekly) > 0.015:
            reasons.append(f"Trend semana {weekly:+.1%}")

    if ytd is not None:
        ytd_sig = _clamp(ytd / 0.20)  # 20% YTD = max
        scores.append((0.2, ytd_sig))

    if not scores:
        return 0.0, reasons

    total_w = sum(w for w, _ in scores)
    raw = sum(w * s for w, s in scores) / total_w
    return round(raw, 4), reasons


# ── Signal 2: Mean Reversion ────────────────────────────────────────────────────

def _compute_mean_rev(daily: float | None,
                      weekly: float | None,
                      ytd: float | None,
                      vol_ann: float | None) -> tuple[float, list[str]]:
    """
    Mean reversion = -z_score de retorno recente vs típico

    Ativo que caiu muito (z negativo) → score positivo = oportunidade de compra
    Ativo que subiu muito (z positivo) → score negativo = overbought
    """
    reasons = []

    if ytd is None or vol_ann is None or vol_ann == 0:
        # Fallback: usa weekly
        if weekly is not None:
            z = _clamp(-weekly / 0.03)  # inversão: queda → compra
            if abs(weekly) > 0.02:
                reasons.append(f"Oversold semana {weekly:+.1%}" if weekly < -0.02
                                else f"Overbought semana {weekly:+.1%}")
            return round(z, 4), reasons
        return 0.0, reasons

    # YTD / vol anual = z-score aproximado vs distribuição de retornos
    # vol_ann já está em escala anual, YTD está em [0, 1y] — usamos como aproximação
    z_ytd = ytd / (vol_ann + 1e-9)
    mean_rev = _clamp(-z_ytd / 2.0)   # z > 2 → -1 (overbought); z < -2 → +1 (oversold)

    if z_ytd < -1.5:
        reasons.append(f"Oversold YTD {ytd:+.1%} (z={z_ytd:.1f}σ)")
    elif z_ytd > 1.5:
        reasons.append(f"Overbought YTD {ytd:+.1%} (z={z_ytd:.1f}σ)")

    return round(mean_rev, 4), reasons


# ── Signal 3: Vol Edge ──────────────────────────────────────────────────────────

def _compute_vol_edge(iv: float | None,
                      vol_ann: float | None,
                      iv_percentile: float | None,
                      gex_b: float | None) -> tuple[float, list[str]]:
    """
    IV premium sobre RV = edge para o lado vendedor de opções
    (e direção do momento para comprador quando GEX é negativo)

    iv_premium > 0  → IV caro → short vega ou follow trend (delta-neutral)
    iv_premium < 0  → IV barato → buy vol / oportunidade long
    """
    reasons = []
    scores = []

    if iv is not None and vol_ann is not None and vol_ann > 0.01:
        iv_premium = (iv - vol_ann) / vol_ann   # ex: 0.3 → IV 30% acima do RV
        # Positive premium = IV caro = sell vol = slightly negative for buyers
        # We invert: high premium = slight headwind for long (options will decay)
        iv_sig = _clamp(-iv_premium * 2)
        scores.append((0.5, iv_sig))
        if abs(iv_premium) > 0.2:
            reasons.append(f"IV {iv:.0%} vs RV {vol_ann:.0%} (premium={iv_premium:+.0%})")

    if iv_percentile is not None:
        # High IV percentile = fear/opportunity zone
        pct_sig = _clamp((iv_percentile - 0.5) * 2)  # 0.5 = neutral, 1.0 = max fear
        scores.append((0.3, -pct_sig))  # IV spike = potential reversal buy
        if iv_percentile > 0.8:
            reasons.append(f"IV percentile alto {iv_percentile:.0%} — potencial reversão")
        elif iv_percentile < 0.2:
            reasons.append(f"IV percentile baixo {iv_percentile:.0%} — complacência")

    if not scores:
        return 0.0, reasons

    total_w = sum(w for w, _ in scores)
    raw = sum(w * s for w, s in scores) / total_w
    return round(raw, 4), reasons


# ── Signal 4: Options Flow ──────────────────────────────────────────────────────

def _compute_options_flow(gex_b: float | None,
                          pcr: float | None,
                          iv_skew: float | None,
                          daily: float | None) -> tuple[float, list[str]]:
    """
    Dealer positioning signal:
      GEX < 0  → dealers estão short gamma → amplificam movimentos → follow momentum
      GEX > 0  → dealers long gamma → absorvem volatilidade → mean reversion tendency
      PCR > 1  → bearish put buying → contrarian buy signal (or confirm downturn)
      PCR < 0.7 → bullish call buying → slight overbought
    """
    reasons = []
    scores = []

    if gex_b is not None:
        # Negative GEX = short gamma = follow trend
        # We combine with daily to get directional
        gex_dir = math.copysign(1, gex_b) if gex_b != 0 else 0
        gex_mag = min(1.0, abs(gex_b) / 2.0)  # normalize: $2B = max signal

        if gex_b < -0.5:
            # Short gamma: follow daily direction
            flow_sig = (daily or 0) / 0.02 * gex_mag
            flow_sig = _clamp(flow_sig)
            reasons.append(f"GEX ${gex_b:.2f}B (short gamma — momentum amp)")
        elif gex_b > 0.5:
            # Long gamma: favor mean reversion
            flow_sig = _clamp(-(daily or 0) / 0.02 * 0.5)
            reasons.append(f"GEX ${gex_b:.2f}B (long gamma — absorção)")
        else:
            flow_sig = 0.0

        scores.append((0.6, flow_sig))

    if pcr is not None:
        # PCR > 1.2 → extreme fear → contrarian long
        # PCR < 0.6 → euphoria → contrarian short
        if pcr > 1.2:
            pcr_sig = min(1.0, (pcr - 1.0) * 1.5)
            reasons.append(f"PCR {pcr:.2f} — medo extremo, contrarian long")
        elif pcr < 0.6:
            pcr_sig = max(-1.0, -(0.6 - pcr) * 3)
            reasons.append(f"PCR {pcr:.2f} — euforia calls, contrarian short")
        else:
            pcr_sig = 0.0
        scores.append((0.4, pcr_sig))

    if iv_skew is not None:
        # Negative skew = put premium = market protection buying
        # Strong put skew = tail fear = slight negative signal
        skew_sig = _clamp(-iv_skew / 0.1)
        scores.append((0.2, skew_sig))

    if not scores:
        return 0.0, reasons

    total_w = sum(w for w, _ in scores)
    raw = sum(w * s for w, s in scores) / total_w
    return round(raw, 4), reasons


# ── DebtRank Contagion ──────────────────────────────────────────────────────────

def compute_debtrank(
    tickers: list[str],
    corr_matrix: Any,          # np.ndarray shape (n, n), or None
    mst_adj: dict[str, list[str]],
) -> dict[str, float]:
    """
    Contagion via Katz centrality no grafo de correlação MST.

    Katz_i = sum_{t=1}^{inf} alpha^t * A^t * ones
    onde A é a matriz de adjacência ponderada por |correlação|.

    Normalizado para [0, 1].

    Por que Katz e não apenas grau?
    - Captura cascata: se A está conectado a B e B está conectado a C,
      A choca B que choca C — efeito de 2ª ordem conta.
    - alpha controla o decaimento por hop (padrão: 0.1 para convergência).
    """
    try:
        import numpy as np
        import networkx as nx

        n = len(tickers)
        G = nx.Graph()
        G.add_nodes_from(tickers)

        # Adiciona arestas MST ponderadas por correlação
        for t, neighbors in mst_adj.items():
            for nbr in neighbors:
                if corr_matrix is not None and t in tickers and nbr in tickers:
                    i = tickers.index(t)
                    j = tickers.index(nbr)
                    w = float(abs(corr_matrix[i, j])) if corr_matrix is not None else 0.5
                else:
                    w = 0.5
                G.add_edge(t, nbr, weight=w)

        if G.number_of_edges() == 0:
            return {t: 0.0 for t in tickers}

        # Katz centrality com peso
        katz = nx.katz_centrality_numpy(G, alpha=0.1, weight="weight", normalized=True)

        # Normaliza para [0, 1]
        vals = list(katz.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {t: 0.0 for t in tickers}
        return {t: (katz.get(t, 0) - mn) / (mx - mn) for t in tickers}

    except Exception as exc:
        _log.warning("debtrank_fallback", error=str(exc))
        # Fallback: degree normalized
        degrees = {t: len(v) for t, v in mst_adj.items()}
        mx = max(degrees.values()) if degrees else 1
        return {t: degrees.get(t, 0) / max(mx, 1) for t in tickers}


# ── Cluster assignment via MST BFS ─────────────────────────────────────────────

def _assign_clusters(tickers: list[str], mst_adj: dict[str, list[str]]) -> dict[str, str]:
    """BFS no MST para atribuir cluster_id a cada ticker."""
    cluster = {}
    cluster_id = 0

    def bfs(start: str, cid: str) -> None:
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in cluster:
                continue
            cluster[node] = cid
            for nb in mst_adj.get(node, []):
                if nb not in cluster:
                    queue.append(nb)

    for t in tickers:
        if t not in cluster:
            bfs(t, f"C{cluster_id:02d}")
            cluster_id += 1

    return cluster


# ── Main compute function ───────────────────────────────────────────────────────

def compute_signals(
    market_prices: dict[str, Any],
    prob_map: dict[str, Any] | None = None,
    options_map: dict[str, Any] | None = None,
    mst_adj: dict[str, list[str]] | None = None,
    corr_matrix: Any = None,              # np.ndarray
    tickers_ordered: list[str] | None = None,
    vol_regime: Any = None,               # VolRegimeResult | None
    cta_result: Any = None,               # CTAPositioningResult | None
    rrg_result: Any = None,               # RRGResult | None
    shadow_flow: Any = None,              # ShadowFlowResult | None
    narrative_result: Any = None,         # NarrativeAlphaResult | None
) -> dict[str, AssetSignal]:
    """
    Computa sinais alpha para todos os tickers com dados de mercado.

    Returns:
        dict ticker → AssetSignal com composite score e direction
    """
    import numpy as np

    prob_map     = prob_map or {}
    options_map  = options_map or {}
    mst_adj      = mst_adj or {}

    # Filtra tickers com dados reais
    tickers = [
        t for t, v in market_prices.items()
        if not t.startswith("__") and isinstance(v, dict) and v.get("price")
    ]

    if not tickers:
        return {}

    # DebtRank contagion
    contagion_scores = compute_debtrank(
        tickers, corr_matrix,
        {t: mst_adj.get(t, []) for t in tickers}
    )

    # Cluster assignment
    clusters = _assign_clusters(tickers, mst_adj)

    results: dict[str, AssetSignal] = {}

    for ticker in tickers:
        mp = market_prices[ticker]
        prob = prob_map.get(ticker) or {}

        # Index → ETF proxy para options
        _INDEX_ETF = {"^GSPC": "SPY", "^NDX": "QQQ", "^RUT": "IWM", "^VIX": "VIXY"}
        opt_key = _INDEX_ETF.get(ticker, ticker)
        opts = options_map.get(opt_key) or {}

        sig = AssetSignal(
            ticker      = ticker,
            name        = mp.get("name", ticker),
            price       = _safe(mp.get("price")),
            daily_ret   = _safe(mp.get("daily_return")),
            weekly_ret  = _safe(mp.get("weekly_return")),
            ytd_ret     = _safe(mp.get("ytd_return")),
            vol_ann     = _safe(mp.get("vol_ann")) or _safe(mp.get("volatility")),
            pe_ratio    = _safe(mp.get("pe_ratio")),
            mkt_cap     = _safe(mp.get("market_cap")),
            regime_bull = _safe(prob.get("regime_prob_bull")),
            tail_score  = _safe(prob.get("tail_score")),
            var_95      = _safe(prob.get("var_95")),
            cvar_95     = _safe(prob.get("cvar_95")),
            dominant_cycle = prob.get("dominant_cycle"),
            iv            = _safe(opts.get("iv")),
            gex_b         = _safe(opts.get("gex_b")),
            pcr           = _safe(opts.get("pcr")),
            iv_skew       = _safe(opts.get("iv_skew")),
            iv_percentile = _safe(opts.get("iv_percentile")),
            mst_degree    = len(mst_adj.get(ticker, [])),
            contagion_katz = contagion_scores.get(ticker, 0.0),
            cluster_id    = clusters.get(ticker, "C00"),
        )

        # Compute sub-signals
        reasons: list[str] = []

        sig.momentum_score, r = _compute_momentum(
            sig.regime_bull, sig.daily_ret, sig.weekly_ret, sig.ytd_ret)
        reasons.extend(r)

        sig.mean_rev_score, r = _compute_mean_rev(
            sig.daily_ret, sig.weekly_ret, sig.ytd_ret, sig.vol_ann)
        reasons.extend(r)

        sig.vol_edge_score, r = _compute_vol_edge(
            sig.iv, sig.vol_ann, sig.iv_percentile, sig.gex_b)
        reasons.extend(r)

        sig.options_flow_score, r = _compute_options_flow(
            sig.gex_b, sig.pcr, sig.iv_skew, sig.daily_ret)
        reasons.extend(r)

        # Penalties
        sig.contagion_penalty = sig.contagion_katz   # already [0, 1]
        sig.tail_penalty      = sig.tail_score or 0.0

        # ── Relative Strength (RS/RRG) signal ───────────────────────────────
        rs_signal = 0.0
        if rrg_result:
            try:
                from app.analysis.relative_strength import get_rs_signal_for_ticker
                rs_signal = get_rs_signal_for_ticker(ticker, rrg_result)
                rs_sig_obj = rrg_result.signals.get(ticker)
                if rs_sig_obj and abs(rs_signal) > 0.1:
                    reasons.append(
                        f"RS: {rs_sig_obj.quadrant.upper()} | RS%ile={rs_sig_obj.rs_percentile:.0f} | "
                        f"RRG=({rs_sig_obj.rs_ratio:.1f},{rs_sig_obj.rs_momentum:.1f})"
                    )
            except Exception:
                pass

        # ── Dark pool / Shadow Flow signal ───────────────────────────────────
        dp_signal = 0.0
        if shadow_flow:
            try:
                from app.providers.shadow_flow import get_dark_pool_signal_for_ticker
                dp_signal = get_dark_pool_signal_for_ticker(ticker, shadow_flow)
                if abs(dp_signal) > 0.15:
                    sf_sig = shadow_flow.signals.get(ticker)
                    if sf_sig and sf_sig.unusual_volume_ratio > 2.0:
                        reasons.append(
                            f"Dark Pool: vol={sf_sig.unusual_volume_ratio:.1f}x | "
                            f"flow={sf_sig.options_sentiment}"
                        )
            except Exception:
                pass

        # ── Narrative alpha (DeepVue themes + X sentiment) ───────────────────
        narrative_signal = 0.0
        if narrative_result:
            try:
                from app.analysis.narrative_alpha import get_narrative_signal_for_ticker
                narrative_signal = get_narrative_signal_for_ticker(ticker, narrative_result)
                if abs(narrative_signal) > 0.10:
                    nsig = narrative_result.signals.get(ticker)
                    if nsig:
                        reasons.extend(nsig.rationale[:2])
            except Exception:
                pass

        # ── CTA flow signal ─────────────────────────────────────────────────
        cta_signal = 0.0
        if cta_result:
            try:
                from app.providers.cta_positioning import get_cta_signal_for_ticker
                cta_signal = get_cta_signal_for_ticker(ticker, cta_result)
                if abs(cta_signal) > 0.1:
                    direction = "compra (CTAs extreme short, risco de squeeze)" if cta_signal > 0 else "venda (CTAs extreme long, risco de selling)"
                    reasons.append(f"CTA flow={cta_signal:+.2f} — {direction}")
            except Exception:
                pass

        # ── Vol regime adjustment ─────────────────────────────────────────────
        vol_stress_scalar = 1.0
        if vol_regime:
            # Em stress/crisis, amplifica sinais bearish, atenua sinais bullish
            if vol_regime.regime in ("stress", "crisis"):
                stress = vol_regime.stress_score
                if sig.momentum_score < 0:   # bearish momentum
                    vol_stress_scalar = 1.0 + stress * 0.5  # amplifica short signals
                else:                         # bullish momentum
                    vol_stress_scalar = 1.0 - stress * 0.3  # atenua longs
                reasons.append(f"Vol regime={vol_regime.regime} (stress={vol_regime.stress_score:.2f})")
            elif vol_regime.regime == "calm" and sig.momentum_score > 0:
                vol_stress_scalar = 1.05  # slight boost para longs em ambiente calmo

        # Composite alpha — pesos revisados com RS como sinal central
        # RS e o sinal que todos os modelos sistematicos usam → 20% de peso
        # Narrative (DeepVue + X) = 10% — sinal qualitativo/tematico
        # CTA flow + Dark Pool = informacao de posicionamento
        alpha_raw = (
            0.23 * sig.momentum_score        # HMM + trend
            + 0.13 * sig.mean_rev_score       # z-score reversion
            + 0.11 * sig.vol_edge_score       # IV premium edge
            + 0.16 * sig.options_flow_score   # GEX + PCR + skew
            + 0.20 * rs_signal                # Relative Strength / RRG ← dominante
            + 0.10 * narrative_signal         # DeepVue themes + X confirmation
            + 0.05 * cta_signal               # CTA contrarian flow
            + 0.02 * dp_signal                # Dark pool institucional
        ) * vol_stress_scalar

        # Penalize by contagion (high contagion = systemic = avoid concentration)
        # and tail risk (fat tails = uncertain = reduce conviction)
        sig.composite = round(
            alpha_raw
            * (1.0 - 0.40 * sig.contagion_penalty)
            * (1.0 - 0.25 * sig.tail_penalty),
            4
        )

        # Direction
        if sig.composite > 0.08:
            sig.direction = "long"
        elif sig.composite < -0.08:
            sig.direction = "short"
        else:
            sig.direction = "neutral"

        # Conviction
        abs_comp = abs(sig.composite)
        if abs_comp > 0.25:
            sig.conviction = "high"
        elif abs_comp > 0.12:
            sig.conviction = "medium"
        else:
            sig.conviction = "low"

        # Expected return: signal × vol (rough Kelly-implied forward return)
        vol = sig.vol_ann or 0.20
        sig.expected_return_ann = round(sig.composite * vol * 2.5, 4)
        sig.risk_score = vol * (1 + sig.tail_penalty * 0.5)

        # Sharpe implied
        rf = 0.053  # ~5.3% risk-free rate (current)
        if sig.risk_score > 0:
            sig.sharpe_implied = round(
                (sig.expected_return_ann - rf) / sig.risk_score, 3)

        sig.rationale = reasons[:4]  # top 4 reasons
        results[ticker] = sig

    _log.info("alpha_signals_done",
              total=len(results),
              long=sum(1 for s in results.values() if s.direction == "long"),
              short=sum(1 for s in results.values() if s.direction == "short"),
              neutral=sum(1 for s in results.values() if s.direction == "neutral"))

    return results
