"""
MacroDesk — Graph Engine

Transforma o DailyIngestionBundle num grafo Cytoscape.js navegável:

  Nodes  → hierarquia de ativos (node_registry) enriquecida com preços/retornos
  Edges  → dois tipos:
           1. "hierarchy"   — parent→child da taxonomia (World→Asset Class→…)
           2. "mst"         — Minimum Spanning Tree de Mantegna (correlação real)
           3. "rmt"         — arestas do GraphicalLasso (precisão esparsa)

Output JSON → self-contained, pronto para Cytoscape.js init().
"""

from __future__ import annotations

import math
from typing import Any

from app.audit.logger import get_logger
from app.desk.node_registry import NODES, INTERNAL_LAYERS, INTERNAL_LAYER_LABELS

_log = get_logger("desk.graph_engine")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f}%"


def _return_color(v: float | None) -> str:
    """Cor de fundo com gradiente de intensidade — visível no canvas escuro (#060a12)."""
    if v is None:
        return "#1e293b"
    if v >= 0.04:
        return "#166534"   # verde muito forte  (>4%)
    if v >= 0.02:
        return "#15803d"   # verde forte         (2-4%)
    if v >= 0.01:
        return "#16a34a"   # verde médio         (1-2%)
    if v >= 0.003:
        return "#1a6b3c"   # verde fraco         (0.3-1%)
    if v >= -0.003:
        return "#1e293b"   # neutro
    if v >= -0.01:
        return "#991b1b"   # vermelho fraco      (-1 a -0.3%)
    if v >= -0.02:
        return "#b91c1c"   # vermelho médio      (-2 a -1%)
    if v >= -0.04:
        return "#dc2626"   # vermelho forte      (-4 a -2%)
    return "#ef4444"       # vermelho muito forte (< -4%)


def _border_color(v: float | None) -> str:
    if v is None:
        return "#6b7280"
    return "#22c55e" if v >= 0 else "#ef4444"


def _momentum_score(daily: float | None, weekly: float | None, ytd: float | None) -> float:
    """Score de momentum normalizado [-1, 1]."""
    parts = []
    if daily is not None:
        parts.append(math.tanh(daily * 20))   # 5% daily → ~tanh(1) ≈ 0.76
    if weekly is not None:
        parts.append(math.tanh(weekly * 8))   # 5% weekly → ~tanh(0.4) ≈ 0.38
    if ytd is not None:
        parts.append(math.tanh(ytd * 3))      # 20% ytd → ~tanh(0.6) ≈ 0.54
    return round(sum(parts) / len(parts), 3) if parts else 0.0


# ── Convexity helpers ─────────────────────────────────────────────────────────

def _convexity_halo_color(
    iv_pct: float | None,
    skew: float | None,
    fragility: float | None,
    hidden_opp: float | None,
) -> str | None:
    """Cor do halo de convexidade para o nó no grafo."""
    frag = fragility or 0.0
    opp  = hidden_opp or 0.0

    if iv_pct is not None:
        # IV data disponível — sinal completo
        if iv_pct > 0.75 and frag > 0.5:
            return "#ef4444"   # IV cara + frágil → vermelho
        if iv_pct > 0.75:
            return "#c084fc"   # IV cara → roxo
        if iv_pct < 0.35 and opp > 0.3:
            return "#22c55e"   # IV barata + oportunidade → verde
        if iv_pct < 0.35:
            return "#38bdf8"   # IV barata → azul
        if skew is not None and skew > 0.04:
            return "#f97316"   # Put skew elevado → laranja
        return None

    # Fallback: sem IV — usa apenas fragility/hidden_opp do desk_intel
    if frag > 0.5:
        return "#ef4444"   # alta fragilidade → vermelho
    if frag > 0.25:
        return "#f97316"   # fragilidade moderada → laranja
    if opp > 0.3:
        return "#22c55e"   # alta oportunidade → verde
    if opp > 0.15:
        return "#38bdf8"   # oportunidade moderada → azul
    return None


def _convexity_node_attrs(
    ticker: str | None,
    options: dict[str, Any] | None,
    desk_intel: Any | None,
) -> dict[str, Any] | None:
    """Monta atributos de convexidade para o nó."""
    if not ticker:
        return None
    o = options or {}
    iv_pct   = o.get("iv_percentile")
    skew     = o.get("skew_5pct")
    frag     = getattr(desk_intel, "fragility_scores",   {}).get(ticker) if desk_intel else None
    opp      = getattr(desk_intel, "opportunity_scores", {}).get(ticker) if desk_intel else None
    halo_col = _convexity_halo_color(iv_pct, skew, frag, opp)
    if iv_pct is None and (frag is None or frag == 0.0) and (opp is None or opp == 0.0):
        return None
    return {
        "iv_rank":    round(iv_pct, 3) if iv_pct is not None else None,
        "skew":       round(skew, 4)   if skew is not None else None,
        "fragility":  round(frag, 3)   if frag is not None else None,
        "hidden_opp": round(opp, 3)    if opp  is not None else None,
        "halo_color": halo_col,
        "halo_radius": 20 if halo_col else None,
    }


# ── Node builder ──────────────────────────────────────────────────────────────

def _build_node(
    node_id: str,
    node_def: dict[str, Any],
    market_prices: dict[str, Any],
    hub_set: set[str],
    scores: dict[str, float] | None = None,
    anatomy_map: dict[str, Any] | None = None,
    options_map: dict[str, Any] | None = None,
    prob_map: dict[str, Any] | None = None,
    desk_intel: Any | None = None,
    rrg_result: Any | None = None,
) -> dict[str, Any]:
    """Constrói um node Cytoscape.js com dados de mercado, anatomy, options e scores."""

    ticker = node_def.get("ticker") or (node_id if node_id in market_prices else None)
    mp = market_prices.get(ticker, {}) if ticker else {}

    price    = mp.get("price")
    daily    = mp.get("daily_return")
    weekly   = mp.get("weekly_return")
    ytd      = mp.get("ytd_return")
    name     = mp.get("name") or node_def.get("label", node_id)

    momentum = _momentum_score(daily, weekly, ytd)
    is_hub   = node_id in hub_set or ticker in hub_set

    level          = node_def.get("level", 5)
    weight         = node_def.get("weight")
    liquidity_w    = node_def.get("liquidity_weight")

    # Anatomy: fundamentais pré-coletados (P/E, beta, ROE, etc.)
    anatomy = (anatomy_map or {}).get(ticker) if ticker else None

    # Options: IV, skew, GEX — verifica proxy ETF para índices
    _INDEX_TO_ETF = {"^GSPC": "SPY", "^NDX": "QQQ", "^RUT": "IWM", "^VIX": "VIXY"}
    options_key = _INDEX_TO_ETF.get(ticker, ticker) if ticker else None
    options = (options_map or {}).get(options_key) if options_key else None

    # Probabilistic: VaR, CVaR, tail score, FFT cycle, regime P(bull)
    prob = (prob_map or {}).get(ticker) if ticker else None

    # Convexity layer attrs (IV rank, skew, fragility, hidden_opp, halo)
    convexity = _convexity_node_attrs(ticker, options, desk_intel)

    # RRG state (quadrant, rs_ratio, rs_momentum, alpha_score)
    rrg_quadrant = rrg_alpha = rrg_rs_ratio = rrg_rs_mom = None
    if rrg_result and ticker:
        _rrg_sigs = getattr(rrg_result, "signals", {}) or {}
        _rsig = _rrg_sigs.get(ticker)
        if _rsig:
            rrg_quadrant = getattr(_rsig, "quadrant", None)
            rrg_rs_ratio = getattr(_rsig, "rs_ratio", None)
            rrg_rs_mom   = getattr(_rsig, "rs_momentum", None)
            rrg_alpha    = getattr(_rsig, "rs_alpha_score", None)

    data: dict[str, Any] = {
        "id":         node_id,
        "label":      node_def.get("label", node_id),
        "level":      level,
        "parent_id":  node_def.get("parent"),
        "color":      node_def.get("color", "#6b7280"),
        "ticker":     ticker,
        "has_data":   bool(mp),
        # Mercado
        "price":      price,
        "daily":      daily,
        "weekly":     weekly,
        "ytd":        ytd,
        "daily_str":  _fmt_pct(daily),
        "weekly_str": _fmt_pct(weekly),
        "ytd_str":    _fmt_pct(ytd),
        # Visual
        "bg_color":     _return_color(daily),
        "border_color": _border_color(daily),
        "is_hub":       is_hub,
        "momentum":     momentum,
        # Fundamentais
        "anatomy":      anatomy,
        # Options (IV, skew, GEX)
        "options":      options,
        # Probabilistic (VaR, CVaR, tail, FFT, regime)
        "prob":         prob,
        # Scores (se vierem do investment agent)
        "agent_scores": scores,
        # Convexity layer (halo, IV rank, skew, fragility, hidden_opp)
        "convexity":    convexity,
        # RRG state (flow layer)
        "rrg_quadrant": rrg_quadrant,
        "rrg_rs_ratio": round(rrg_rs_ratio, 2) if rrg_rs_ratio is not None else None,
        "rrg_rs_mom":   round(rrg_rs_mom, 2)   if rrg_rs_mom   is not None else None,
        "rrg_alpha":    round(rrg_alpha, 3)     if rrg_alpha    is not None else None,
        # Tamanho: contribuição = peso × |retorno|; maior movimento/peso = esfera maior
        "size": _node_size_dynamic(level, is_hub, bool(mp), daily, weight, liquidity_w),
        "weight": weight,
        "liquidity_weight": liquidity_w,
    }
    return {"data": data}


def _node_size(level: int, is_hub: bool, has_data: bool) -> int:
    """Tamanho estático — usado apenas para níveis 0-2 (sem dados de mercado)."""
    base = {0: 80, 1: 60, 2: 48}.get(level, 28)
    if is_hub:
        base = int(base * 1.4)
    if not has_data:
        base = int(base * 0.85)
    return base


def _node_size_dynamic(
    level: int,
    is_hub: bool,
    has_data: bool,
    daily: float | None,
    weight: float | None,
    liquidity_weight: float | None,
) -> int:
    """
    Tamanho baseado na contribuição do ativo ao movimento do dia.

    Fórmula: contribution = weight × (1 + |daily| × 20)
      - weight    : peso do ativo no índice-pai (ex: AAPL = 7.5% do SPX)
      - |daily|   : retorno absoluto do dia
      - × 20      : amplifica retornos pequenos (1% = ×1.2, 3% = ×1.6)

    Efeito visual:
      - AAPL flat:     tamanho médio-grande  (peso 7.5%)
      - AAPL +3%:      esfera grande         (peso amplificado pela alta)
      - INTC flat:     esfera pequena        (peso 0.5%)
      - Tech flat:     sector grande         (peso 31%)
      - Energy flat:   sector pequeno        (peso 4%)
    """
    # Níveis estruturais (sem preços diretos): tamanho estático
    if level <= 2:
        return _node_size(level, is_hub, has_data)

    w = weight or liquidity_weight or 0.005
    ret = abs(daily or 0)
    contribution = w * (1.0 + ret * 20.0)

    if level == 3:       # índices
        norm, min_sz, max_sz = 0.80, 32, 72
    elif level == 4:     # setores
        norm, min_sz, max_sz = 0.60, 22, 60
    elif level == 5:     # stocks / macro assets
        norm, min_sz, max_sz = 0.15, 22, 52
    else:                # nível 7 (internal layers)
        norm, min_sz, max_sz = 0.10, 18, 32

    scale = min(contribution / norm, 1.0)
    sz = int(min_sz + scale * (max_sz - min_sz))

    if is_hub:
        sz = int(sz * 1.25)
    if not has_data:
        sz = int(sz * 0.80)

    return max(min_sz, sz)


# ── Edge builders ─────────────────────────────────────────────────────────────

def _hierarchy_edges(nodes_in_graph: set[str]) -> list[dict[str, Any]]:
    """Arestas parent→child da hierarquia estática."""
    edges = []
    for nid, nd in NODES.items():
        if nid not in nodes_in_graph:
            continue
        parent = nd.get("parent")
        if parent and parent in nodes_in_graph:
            edges.append({
                "data": {
                    "id":     f"h_{parent}_{nid}",
                    "source": parent,
                    "target": nid,
                    "type":   "hierarchy",
                    "layer":  "structure",
                    "weight": 1.0,
                    "color":  "#374151",
                    "width":  1,
                }
            })
    return edges


def _mst_edges(mst_data: dict[str, Any], nodes_in_graph: set[str]) -> list[dict[str, Any]]:
    """Arestas do MST de Mantegna."""
    edges = []
    for e in mst_data.get("edges", []):
        src = e.get("from")
        tgt = e.get("to")
        if not src or not tgt:
            continue
        # Resolve para node_id (ticker pode diferir do node_id)
        src_id = _ticker_to_node(src, nodes_in_graph)
        tgt_id = _ticker_to_node(tgt, nodes_in_graph)
        if not src_id or not tgt_id:
            continue
        rho = e.get("correlation", 0.0)
        color = "#22c55e" if rho >= 0 else "#ef4444"
        edges.append({
            "data": {
                "id":          f"mst_{src_id}_{tgt_id}",
                "source":      src_id,
                "target":      tgt_id,
                "type":        "mst",
                "layer":       "structure",
                "correlation": round(rho, 4),
                "distance":    e.get("distance", 0.0),
                "color":       color,
                "width":       max(1, int(abs(rho) * 4)),
                "weight":      abs(rho),
            }
        })
    return edges


def _synthetic_corr_edges(
    market_prices: dict[str, Any],
    nodes_in_graph: set[str],
    already_covered: set[str],
) -> list[dict[str, Any]]:
    """
    Arestas de correlação sintéticas baseadas em retornos snapshot (daily/weekly/ytd).
    Usadas quando MST não tem dados suficientes (IBKR offline, etc.).

    Metodologia:
    - Para cada par de nós com dados, calcula "pseudo-rho" pela concordância
      de sinal nos 3 horizontes: +1/3 por horizonte em que os retornos têm
      o mesmo sinal, −1/3 quando diferem.
    - Retorna apenas as N conexões mais fortes (para não poluir o grafo).
    """
    import math

    # Monta vetor de retornos por node_id
    vectors: dict[str, list[float | None]] = {}
    for nid, nd in __import__('app.desk.node_registry', fromlist=['NODES']).NODES.items():
        if nid not in nodes_in_graph:
            continue
        ticker = nd.get("ticker") or nid
        mp = market_prices.get(ticker, {})
        if not mp or not mp.get("price"):
            continue
        d = mp.get("daily_return")
        w = mp.get("weekly_return")
        y = mp.get("ytd_return")
        if d is None and w is None and y is None:
            continue
        vectors[nid] = [d, w, y]

    if len(vectors) < 2:
        return []

    # Pares — calcula pseudo-rho
    node_ids = list(vectors.keys())
    scored: list[tuple[float, str, str]] = []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            a, b = node_ids[i], node_ids[j]
            va, vb = vectors[a], vectors[b]
            score = 0.0
            count = 0
            for x, y in zip(va, vb):
                if x is None or y is None:
                    continue
                if x == 0 and y == 0:
                    continue
                sign_match = (x * y) > 0
                score += (1.0 if sign_match else -1.0) / 3.0
                count += 1
            if count < 2:
                continue
            rho = round(score, 3)
            # incluir correlações com algum sinal
            if abs(rho) >= 0.20:
                scored.append((abs(rho), a, b, rho))

    # Ordena por |rho| desc. Limite escala com nº de nós: ~1.2x cobertura.
    # Garantia: cada nó sem MST ganha pelo menos 1 aresta (até saturar).
    MAX_SYNTH = max(120, len(node_ids) * 2)
    scored.sort(key=lambda x: x[0], reverse=True)
    edges = []
    covered_nodes: set[str] = set(already_covered)
    for abs_rho, a, b, rho in scored:
        if len(edges) >= MAX_SYNTH and (a in covered_nodes and b in covered_nodes):
            continue
        color = "#22c55e" if rho >= 0 else "#ef4444"
        edges.append({
            "data": {
                "id":          f"syn_{a}_{b}",
                "source":      a,
                "target":      b,
                "type":        "mst",   # mesmo tipo para herdar estilo
                "correlation": rho,
                "distance":    round(1.0 - abs_rho, 3),
                "color":       color,
                "width":       max(1, int(abs_rho * 3)),
                "weight":      abs_rho,
                "synthetic":   True,
            }
        })
        covered_nodes.add(a)
        covered_nodes.add(b)

    _log.info("synthetic_corr_edges", n=len(edges), nodes_covered=len(covered_nodes))
    return edges


def _rrg_edges(rrg_result: Any, nodes_in_graph: set[str]) -> list[dict[str, Any]]:
    """
    Arestas direcionadas do RRG (Relative Rotation Graph).

    Cada ativo vs SPY benchmark — quadrante determina direção e cor:
      LEADING   (#22c55e) : ativo → sp500  (outperforming e acelerando)
      IMPROVING (#84cc16) : ativo → sp500  (fraco mas recuperando, dashed)
      WEAKENING (#f97316) : sp500 → ativo  (forte mas desacelerando, dashed)
      LAGGING   (#ef4444) : sp500 → ativo  (fraco e piorando)

    Espessura proporcional ao |RS-Ratio − 100| (distância do pivot).
    """
    if rrg_result is None:
        return []

    spx_id = "sp500"
    if spx_id not in nodes_in_graph:
        return []

    QUADRANT_STYLES: dict[str, dict] = {
        "leading":   {"color": "#22c55e", "width": 3, "dashed": False,  "to_spx": True},
        "improving": {"color": "#84cc16", "width": 2, "dashed": True,   "to_spx": True},
        "weakening": {"color": "#f97316", "width": 2, "dashed": True,   "to_spx": False},
        "lagging":   {"color": "#ef4444", "width": 3, "dashed": False,  "to_spx": False},
    }

    edges = []
    signals = getattr(rrg_result, "signals", {})
    for ticker, sig in signals.items():
        node_id = _ticker_to_node(ticker, nodes_in_graph)
        if not node_id or node_id == spx_id:
            continue

        quadrant = getattr(sig, "quadrant", "unknown")
        style = QUADRANT_STYLES.get(quadrant)
        if not style:
            continue

        rs_ratio    = getattr(sig, "rs_ratio", 100.0) or 100.0
        rs_mom      = getattr(sig, "rs_momentum", 100.0) or 100.0
        alpha_score = getattr(sig, "rs_alpha_score", 0.0) or 0.0

        # Espessura proporcional ao desvio do pivot (100)
        deviation = abs(rs_ratio - 100.0)
        width = max(1, min(style["width"], int(style["width"] + deviation / 15.0)))

        src = node_id if style["to_spx"] else spx_id
        tgt = spx_id  if style["to_spx"] else node_id

        edges.append({
            "data": {
                "id":          f"rrg_{node_id}",
                "source":      src,
                "target":      tgt,
                "type":        "rrg",
                "layer":       "flow",
                "quadrant":    quadrant,
                "rs_ratio":    round(rs_ratio, 2),
                "rs_momentum": round(rs_mom, 2),
                "alpha_score": round(alpha_score, 3),
                "color":       style["color"],
                "width":       width,
                "dashed":      style["dashed"],
                "weight":      round(deviation / 20.0, 3),
            }
        })

    _log.info("rrg_edges", n=len(edges))
    return edges


def _rmt_edges(rmt_data: dict[str, Any], nodes_in_graph: set[str]) -> list[dict[str, Any]]:
    """Arestas do GraphicalLasso (precision matrix — grafo esparso)."""
    edges = []
    for src, tgt, weight in rmt_data.get("edges", []):
        src_id = _ticker_to_node(src, nodes_in_graph)
        tgt_id = _ticker_to_node(tgt, nodes_in_graph)
        if not src_id or not tgt_id:
            continue
        edges.append({
            "data": {
                "id":     f"rmt_{src_id}_{tgt_id}",
                "source": src_id,
                "target": tgt_id,
                "type":   "rmt",
                "layer":  "structure",
                "weight": round(abs(weight), 4),
                "color":  "#818cf8",
                "width":  max(1, int(abs(weight) * 3)),
            }
        })
    return edges


# ── Ticker → node_id lookup ───────────────────────────────────────────────────

_TICKER_ALIAS: dict[str, str] = {
    # ETF proxy → índice canônico (como usado no node_registry)
    "SPY":  "^GSPC",
    "VOO":  "^GSPC",
    "IVV":  "^GSPC",
    "QQQ":  "^NDX",
    "TQQQ": "^NDX",
    "IWM":  "^RUT",
    "EFA":  "^EAFE",
    "EEM":  "EEM",
    "VNQ":  "VNQ",
    "DIA":  "^DJI",
    "IEF":  "^TNX",
    "SHY":  "^IRX",
    "VIXY": "^VIX",
    "VXX":  "^VIX",
    "UUP":  "DXY",
    "USO":  "CL",
    "BNO":  "CL",
    "BOIL": "NG",
    "IAU":  "GLD",
    "SLV":  "SI",
    "PDBC": "PDBC",
}


def _ticker_to_node(ticker: str, nodes_in_graph: set[str]) -> str | None:
    """Encontra o node_id que corresponde a um ticker.

    Resolve aliases de ETF → índice canônico antes de buscar no registry.
    """
    # Direto: node_id == ticker
    if ticker in nodes_in_graph:
        return ticker
    # Resolve alias ETF → ticker canônico
    canonical = _TICKER_ALIAS.get(ticker, ticker)
    if canonical != ticker and canonical in nodes_in_graph:
        return canonical
    # Busca pelo campo ticker nos NODES (suporta ^GSPC, ^NDX, etc.)
    for nid, nd in NODES.items():
        t = nd.get("ticker")
        if t and nid in nodes_in_graph and (t == ticker or t == canonical):
            return nid
    return None


# ── Main builder ──────────────────────────────────────────────────────────────

def build_graph(
    market_prices: dict[str, Any],
    network_result: dict[str, Any] | None = None,
    agent_scores: dict[str, float] | None = None,
    anatomy_map: dict[str, Any] | None = None,
    options_map: dict[str, Any] | None = None,
    vix_term: dict[str, Any] | None = None,
    prob_map: dict[str, Any] | None = None,
    flow_map: dict[str, Any] | None = None,
    rrg_result: Any | None = None,
    desk_intel: Any | None = None,
    max_level: int = 5,
    include_internal_layers: bool = False,
) -> dict[str, Any]:
    """
    Constrói o grafo completo para Cytoscape.js.

    Args:
        market_prices      : bundle.market_prices
        network_result     : output de analysis.network.analyze()
        agent_scores       : scores dos 5 motores do investment agent
        anatomy_map        : {ticker: {pe, ps, beta, roe, ...}} do node_anatomy.py
        max_level          : nível máximo de nós incluídos (0-7)
        include_internal_layers: se True, adiciona nível 7 (valuation, vol, etc.)

    Returns:
        {
          "elements": {"nodes": [...], "edges": [...]},
          "regime":   {...},
          "mst_meta": {...},
          "rmt_meta": {...},
          "stats":    {...},
        }
    """
    rmt_data    = (network_result or {}).get("rmt", {})
    mst_data    = (network_result or {}).get("mst", {})
    regime_data = (network_result or {}).get("regime", {})

    # ── Hub set (para destacar nós de alta conectividade) ────────────────────
    hub_set: set[str] = set()
    for ticker, _deg in mst_data.get("hubs", [])[:5]:
        hub_set.add(ticker)

    # ── Seleciona quais nodes incluir ─────────────────────────────────────────
    selected_ids: set[str] = set()
    for nid, nd in NODES.items():
        if nd.get("level", 99) <= max_level:
            selected_ids.add(nid)

    # ── Constrói nodes ────────────────────────────────────────────────────────
    cyto_nodes: list[dict] = []
    flow_map = flow_map or {}
    for nid in selected_ids:
        nd = NODES[nid]
        node = _build_node(nid, nd, market_prices, hub_set, agent_scores, anatomy_map, options_map, prob_map, desk_intel=desk_intel, rrg_result=rrg_result)
        # Injeta fluxo mecânico (GEX + LETF) no nó, se disponível
        ticker = nd.get("ticker", "")
        if ticker and ticker in flow_map:
            node["data"]["flow"] = flow_map[ticker]
        cyto_nodes.append(node)

    # ── FILTRO: remove nós sem dados de mercado ──────────────────────────────
    # Só ficam ativos com has_data=True. Os hubs (level 0/1) ficam sempre.
    cyto_nodes = [
        n for n in cyto_nodes
        if n["data"].get("has_data") or n["data"].get("level", 99) <= 1
    ]
    selected_ids = {n["data"]["id"] for n in cyto_nodes}

    # ── Internal layers (nível 7) — expandidos dinamicamente por asset ────────
    if include_internal_layers:
        # Para cada asset (level 5) com dados, adiciona sub-nós de camada interna
        for nid in selected_ids:
            nd = NODES[nid]
            if nd.get("level") != 5 or not nd.get("ticker"):
                continue
            if not market_prices.get(nd["ticker"]):
                continue
            for layer in INTERNAL_LAYERS:
                layer_id = f"{nid}__{layer}"
                layer_node: dict[str, Any] = {
                    "data": {
                        "id":        layer_id,
                        "label":     INTERNAL_LAYER_LABELS.get(layer, layer),
                        "level":     7,
                        "parent_id": nid,
                        "color":     "#6b7280",
                        "ticker":    nd["ticker"],
                        "layer":     layer,
                        "has_data":  False,
                        "size":      18,
                        "is_hub":    False,
                        "momentum":  0.0,
                    }
                }
                cyto_nodes.append(layer_node)
                selected_ids.add(layer_id)

    # ── Constrói arestas ──────────────────────────────────────────────────────
    cyto_edges: list[dict] = []

    # Hierarquia estática
    cyto_edges.extend(_hierarchy_edges(selected_ids))

    # Internal layer → parent
    if include_internal_layers:
        for nid in selected_ids:
            if "__" in nid:
                parent_id = nid.split("__")[0]
                if parent_id in selected_ids:
                    cyto_edges.append({
                        "data": {
                            "id":     f"il_{parent_id}_{nid}",
                            "source": parent_id,
                            "target": nid,
                            "type":   "internal",
                            "color":  "#4b5563",
                            "width":  1,
                        }
                    })

    # MST edges (se disponíveis)
    mst_node_ids: set[str] = set()
    if mst_data:
        mst_edges_list = _mst_edges(mst_data, selected_ids)
        cyto_edges.extend(mst_edges_list)
        for e in mst_edges_list:
            mst_node_ids.add(e["data"]["source"])
            mst_node_ids.add(e["data"]["target"])

    # RMT edges (precision matrix — mais esparso, complementar)
    if rmt_data:
        cyto_edges.extend(_rmt_edges(rmt_data, selected_ids))

    # RRG edges (rotação relativa vs benchmark — direcionado por quadrante)
    if rrg_result is not None:
        cyto_edges.extend(_rrg_edges(rrg_result, selected_ids))

    # Arestas sintéticas de correlação — proxy via retornos snapshot
    # Ativado quando MST cobre menos de 50% dos data nodes (ex: IBKR offline)
    data_node_ids = {n["data"]["id"] for n in cyto_nodes if n["data"].get("has_data")}
    if len(mst_node_ids) < len(data_node_ids) * 0.5:
        cyto_edges.extend(
            _synthetic_corr_edges(market_prices, selected_ids, mst_node_ids)
        )

    # ── Estatísticas ──────────────────────────────────────────────────────────
    n_with_data = sum(1 for n in cyto_nodes if n["data"].get("has_data"))
    n_edges_h   = sum(1 for e in cyto_edges if e["data"].get("type") == "hierarchy")
    n_edges_mst = sum(1 for e in cyto_edges if e["data"].get("type") == "mst")
    n_edges_rmt = sum(1 for e in cyto_edges if e["data"].get("type") == "rmt")
    n_edges_rrg = sum(1 for e in cyto_edges if e["data"].get("type") == "rrg")

    _log.info("graph_built",
              nodes=len(cyto_nodes),
              nodes_with_data=n_with_data,
              hierarchy_edges=n_edges_h,
              mst_edges=n_edges_mst,
              rmt_edges=n_edges_rmt,
              rrg_edges=n_edges_rrg,
              regime=regime_data.get("regime", "unknown"))

    return {
        "elements": {
            "nodes": cyto_nodes,
            "edges": cyto_edges,
        },
        "regime":   regime_data,
        "mst_meta": {
            "avg_corr":  mst_data.get("avg_corr"),
            "n_edges":   mst_data.get("n_edges"),
            "top_hubs":  mst_data.get("hubs", [])[:5],
        },
        "rmt_meta": {
            "n_signal_factors": rmt_data.get("n_signal_factors"),
            "lambda_threshold": rmt_data.get("lambda_threshold"),
            "edge_count":       rmt_data.get("edge_count"),
        },
        "agent_scores": agent_scores or {},
        "vix_term": vix_term or {},
        "flow_map": flow_map,
        "stats": {
            "total_nodes":     len(cyto_nodes),
            "nodes_with_data": n_with_data,
            "hierarchy_edges": n_edges_h,
            "mst_edges":       n_edges_mst,
            "rmt_edges":       n_edges_rmt,
            "rrg_edges":       n_edges_rrg,
        },
    }


def build_from_bundle(
    bundle: Any,
    curation_result: Any | None = None,
    max_level: int = 5,
    skip_anatomy: bool = False,
    skip_options: bool = False,
    skip_prob: bool = False,
    skip_network: bool = False,
    skip_flow: bool = False,
    cached_network: "dict | None" = None,
    cached_anatomy: "dict | None" = None,
    cached_options: "dict | None" = None,
    cached_prob: "dict | None" = None,
    cached_flow: "Any | None" = None,
    rrg_result: "Any | None" = None,
    desk_intel: "Any | None" = None,
) -> dict[str, Any]:
    """
    Ponto de entrada principal: constrói o grafo a partir do bundle do dia.

    Args:
        bundle          : DailyIngestionBundle
        curation_result : CurationResult (opcional, para scores do agente)
        max_level       : nível máximo de profundidade do grafo
        skip_anatomy    : pula coleta de fundamentais (para refreshes rápidos)
        skip_options    : pula coleta de opções
        skip_prob       : pula análise probabilística
        skip_network    : pula análise de rede (MST/RMT — usa Bloomberg/IBKR histórico)
        cached_network  : resultado de rede pré-computado (evita re-download)
        cached_anatomy  : {ticker: {pe, beta, roe, ...}} pré-coletado via BQL

    Returns:
        dict com elementos Cytoscape.js e metadados
    """
    market_prices: dict[str, Any] = {
        k: v for k, v in (bundle.market_prices or {}).items()
        if not k.startswith("__")   # ignora metadados como __refreshed_at__
    }

    # ── Scores do agente (se disponível) ─────────────────────────────────────
    agent_scores: dict[str, float] | None = None
    if curation_result and hasattr(curation_result, "investment_scores"):
        agent_scores = curation_result.investment_scores
    elif curation_result and isinstance(curation_result, dict):
        agent_scores = curation_result.get("investment_scores")

    # ── Análise de rede ────────────────────────────────────────────────────────
    network_result: dict[str, Any] = cached_network or {}
    if not skip_network and not cached_network and market_prices:
        try:
            from app.analysis.network import analyze as net_analyze
            network_result = net_analyze(market_prices) or {}
        except Exception as exc:
            _log.warning("network_analysis_failed", error=str(exc))

    # ── Anatomy: fundamentais dos nós (P/E, beta, ROE, etc.) ─────────────────
    anatomy_map: dict[str, Any] = cached_anatomy or {}
    if not skip_anatomy and not cached_anatomy:
        try:
            from app.providers.node_anatomy import collect_from_registry
            price_map = {
                sym: d.get("price")
                for sym, d in market_prices.items()
                if d.get("price") is not None
            }
            _collected = collect_from_registry(price_map=price_map)
            anatomy_map = (_collected[0] if isinstance(_collected, tuple) else _collected) or {}
            _log.info("anatomy_collected", count=len(anatomy_map))
        except Exception as exc:
            _log.warning("anatomy_failed", error=str(exc))

    # ── Options: IV, skew, GEX para TODOS os tickers da rede ─────────────────
    options_map: dict[str, Any] = cached_options or {}
    vix_term: dict[str, Any] = {}
    if not skip_options and not cached_options:
        try:
            from app.providers.options import collect as options_collect, vix_term_structure
            # Passa TODOS os tickers do market_prices, não só o OPTIONS_UNIVERSE default
            _opts_universe = sorted(set(market_prices.keys()))
            options_map = options_collect(tickers=_opts_universe) or {}
            vix_term    = vix_term_structure(market_prices)
            _log.info("options_collected", count=len(options_map))
        except Exception as exc:
            _log.warning("options_failed", error=str(exc))
    try:
        from app.providers.options import vix_term_structure
        vix_term = vix_term_structure(market_prices)
    except Exception:
        pass

    # ── Probabilistic: VaR, CVaR, tail score, FFT, regime ────────────────────
    prob_map: dict[str, Any] = cached_prob or {}
    if not skip_prob and not cached_prob:
        try:
            from app.analysis.probabilistic import analyze_from_registry
            prob_map = analyze_from_registry() or {}
            _log.info("prob_collected", count=len(prob_map))
        except Exception as exc:
            _log.warning("prob_failed", error=str(exc))

    # ── GEX + LETF Flow (Barbon et al.) ──────────────────────────────────────
    flow_pred = cached_flow
    if not skip_flow and cached_flow is None:
        try:
            from app.providers.gex_letf import collect as gex_letf_collect
            flow_pred = gex_letf_collect()
            _log.info("gex_letf_collected",
                      direction=flow_pred.direction,
                      conviction=flow_pred.conviction,
                      gamma_regime=flow_pred.gex.gamma_regime)
        except Exception as exc:
            _log.warning("gex_letf_failed", error=str(exc))

    # Monta flow_map por ticker para injeção nos nós
    flow_map: dict[str, dict] = {}
    if flow_pred is not None:
        for ticker, mf in (flow_pred.per_member or {}).items():
            flow_map[ticker] = {
                "letf_flow_usd": mf.letf_flow_usd if hasattr(mf, "letf_flow_usd") else mf.get("letf_flow_usd", 0),
                "gex_flow_usd":  mf.gex_flow_usd  if hasattr(mf, "gex_flow_usd")  else mf.get("gex_flow_usd",  0),
                "total_usd":     mf.total_usd      if hasattr(mf, "total_usd")      else mf.get("total", 0),
                "direction":     mf.direction      if hasattr(mf, "direction")      else mf.get("direction", "flat"),
            }

    graph_data = build_graph(
        market_prices=market_prices,
        network_result=network_result,
        agent_scores=agent_scores,
        anatomy_map=anatomy_map,
        options_map=options_map,
        vix_term=vix_term,
        prob_map=prob_map,
        flow_map=flow_map,
        rrg_result=rrg_result,
        desk_intel=desk_intel,
        max_level=max_level,
    )

    # ── Live Network: contagion, propagação, correlações rolling, SRI ─────────
    try:
        from app.analysis.network_live import analyze_live
        live = analyze_live(market_prices, graph_data)
        graph_data["live_network"] = live

        # Patch: embute contagion + propagated_shock em cada nó
        contagion_map = live.get("contagion", {})
        shock_map     = live.get("propagated_shock", {})
        for node in graph_data["elements"]["nodes"]:
            nid = node["data"]["id"]
            node["data"]["contagion"]        = contagion_map.get(nid)
            node["data"]["propagated_shock"] = shock_map.get(nid)

        _log.info("live_network_done",
                  sri=live.get("sri"),
                  label=live.get("sri_label"))
    except Exception as exc:
        _log.warning("live_network_failed", error=str(exc))
        graph_data.setdefault("live_network", {})

    # Embute flow_pred serializado para uso no HTML
    if flow_pred is not None:
        try:
            import dataclasses as _dc
            graph_data["flow_pred"] = _dc.asdict(flow_pred)
        except Exception:
            # Se flow_pred não for dataclass (e.g. dict), passa direto
            graph_data["flow_pred"] = flow_pred if isinstance(flow_pred, dict) else {}
    else:
        graph_data["flow_pred"] = {}

    return graph_data
