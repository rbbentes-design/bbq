"""
Analysis: Live Network

Camada de dinâmica de rede em tempo real:

  1. Contagion Score  — por nó: intensidade de estresse que está transmitindo
                         à rede via suas arestas MST
  2. Shock Propagation — difusão de choques 2 hops pelo grafo MST
                         (quanto cada nó seria impactado se outro tivesse -3%)
  3. Rolling Correlations — 20d vs 60d para pares macro-chave
                            detecta quebras de correlação (regime shifts)
  4. Systemic Risk Index — SRI ∈ [0,1]: índice agregado de estresse sistêmico

Entrada: market_prices + graph_data (com MST edges já calculadas)
Saída:   dict com todos os campos acima, pronto para embutir no graph_data
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.audit.logger import get_logger

_log = get_logger("analysis.network_live")

# Pares macro para monitorar quebras de correlação
_MACRO_PAIRS: list[tuple[str, str, str]] = [
    ("SPY",       "TLT",       "Equity ↔ Bonds"),
    ("SPY",       "GLD",       "Equity ↔ Gold"),
    ("SPY",       "HYG",       "Equity ↔ Credit"),
    ("DX-Y.NYB",  "GLD",       "Dollar ↔ Gold"),
    ("BTC-USD",   "SPY",       "Crypto ↔ Equity"),
    ("^VIX",      "SPY",       "Vol ↔ Equity"),
    ("TLT",       "GLD",       "Bonds ↔ Gold"),
    ("SPY",       "EEM",       "US ↔ EM"),
]


# ── Adjacency builder ─────────────────────────────────────────────────────────

def _build_adj(mst_edges: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """Constrói adjacência bidirecional {node_id: [(neighbor, corr), ...]}."""
    adj: dict[str, list[tuple[str, float]]] = {}
    for e in mst_edges:
        d = e.get("data", {})
        if d.get("type") != "mst":
            continue
        src  = d.get("source", "")
        tgt  = d.get("target", "")
        corr = abs(float(d.get("correlation") or d.get("weight") or 0.0))
        if not src or not tgt:
            continue
        adj.setdefault(src, []).append((tgt, corr))
        adj.setdefault(tgt, []).append((src, corr))
    return adj


# ── 1. Contagion Score ────────────────────────────────────────────────────────

def compute_contagion(
    market_prices: dict[str, Any],
    mst_edges: list[dict],
    node_ticker: dict[str, str],   # {node_id: ticker}
) -> dict[str, float]:
    """
    Contagion score por nó:

      contagion_i = |ret_i| × (1 + mean(|corr_ij| × |ret_j|) for j ∈ MST_neighbors_i)

    Mede o quanto o nó está ativamente transmitindo estresse via
    suas conexões de rede. Normalizado para [0, 1] via tanh.

    Returns {node_id: contagion_score}
    """
    adj = _build_adj(mst_edges)

    # ticker → daily_return
    ret: dict[str, float] = {}
    for sym, d in market_prices.items():
        r = d.get("daily_return")
        if r is not None:
            ret[sym] = abs(float(r))

    # node_id → ticker mapping (reverso também)
    ticker_to_node = {v: k for k, v in node_ticker.items()}

    scores: dict[str, float] = {}
    for nid, neighbors in adj.items():
        tk_i = node_ticker.get(nid, nid)
        ret_i = ret.get(tk_i, ret.get(nid, 0.0))

        nbr_stress = []
        for nbr_id, corr in neighbors:
            tk_j = node_ticker.get(nbr_id, nbr_id)
            ret_j = ret.get(tk_j, ret.get(nbr_id, 0.0))
            nbr_stress.append(corr * ret_j)

        avg_nbr = float(np.mean(nbr_stress)) if nbr_stress else 0.0
        raw = ret_i * (1.0 + avg_nbr * 10.0)  # amplifica 1% de ret vizinho
        scores[nid] = round(math.tanh(raw * 20.0), 4)  # normaliza [0,1]

    return scores


# ── 2. Shock Propagation ──────────────────────────────────────────────────────

def compute_propagated_shock(
    source_returns: dict[str, float],   # {node_id: actual_daily_return}
    mst_edges: list[dict],
    hops: int = 2,
    decay: float = 0.55,
) -> dict[str, float]:
    """
    Difusão de choques na rede MST.

    Para cada nó, estima o choque agregado recebido de seus vizinhos:
      shock_j = Σ_i (corr_ij × ret_i × decay^hop)

    Retorna {node_id: propagated_shock}  (com sinal — negativo = estresse)
    """
    adj = _build_adj(mst_edges)
    all_nodes = set(adj.keys()) | set(source_returns.keys())

    # Estado inicial: retorno real de cada nó
    state: dict[str, float] = {n: source_returns.get(n, 0.0) for n in all_nodes}
    propagated: dict[str, float] = {n: 0.0 for n in all_nodes}

    for hop in range(hops):
        new_state: dict[str, float] = {}
        for nid in all_nodes:
            nbrs = adj.get(nid, [])
            if not nbrs:
                new_state[nid] = 0.0
                continue
            inflow = sum(corr * state.get(nb, 0.0) for nb, corr in nbrs)
            new_state[nid] = inflow * (decay ** (hop + 1))
            propagated[nid] = propagated.get(nid, 0.0) + new_state[nid]
        state = new_state

    return {k: round(v, 6) for k, v in propagated.items()}


# ── 3. Rolling Correlations ───────────────────────────────────────────────────

def _fetch_returns_df(tickers: list[str], period: str = "90d") -> Any:
    """Baixa histórico e retorna DataFrame de log-retornos."""
    try:
        import yfinance as yf
        import pandas as pd
        raw = yf.download(tickers, period=period, auto_adjust=True,
                          progress=False, threads=True)
        if "Close" not in raw.columns:
            close = raw
        else:
            close = raw["Close"]
        ret = np.log(close / close.shift(1)).dropna(how="all")
        return ret
    except Exception as exc:
        _log.warning("rolling_fetch_failed", error=str(exc))
        return None


def rolling_corr_trends(
    pairs: list[tuple[str, str, str]] | None = None,
    period: str = "90d",
    w_short: int = 20,
    w_long:  int = 60,
) -> list[dict[str, Any]]:
    """
    Para cada par macro, calcula:
      - corr_short: correlação rolling {w_short} dias (mais recente)
      - corr_long:  correlação rolling {w_long}  dias (mais recente)
      - divergence: corr_short - corr_long
      - trend:      'rising' | 'falling' | 'stable'
      - signal:     descrição semântica (ex: 'Equity-Bond desacoplando')

    Returns lista de dicts ordenados por |divergence| decrescente.
    """
    pairs = pairs or _MACRO_PAIRS
    all_tickers = list({t for p in pairs for t in p[:2]})
    ret_df = _fetch_returns_df(all_tickers, period=period)
    if ret_df is None:
        return []

    results = []
    for t1, t2, label in pairs:
        try:
            if t1 not in ret_df.columns or t2 not in ret_df.columns:
                continue
            s1 = ret_df[t1].dropna()
            s2 = ret_df[t2].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) < w_long:
                continue
            s1 = s1.loc[common]
            s2 = s2.loc[common]

            roll_s = s1.rolling(w_short).corr(s2)
            roll_l = s1.rolling(w_long).corr(s2)

            corr_s = float(roll_s.iloc[-1])
            corr_l = float(roll_l.iloc[-1])
            div    = round(corr_s - corr_l, 4)

            if abs(div) > 0.12:
                trend = "rising" if div > 0 else "falling"
            else:
                trend = "stable"

            # Sinal semântico simples
            if abs(corr_s) < 0.2 and abs(corr_l) > 0.35:
                signal = f"{label}: correlação quebrando"
            elif corr_s > 0.6 and label in ("Equity ↔ Bonds", "Dollar ↔ Gold"):
                signal = f"{label}: correlação positiva incomum"
            elif corr_s < -0.6:
                signal = f"{label}: hedge forte ativo"
            else:
                signal = label

            results.append({
                "pair":       f"{t1}/{t2}",
                "label":      label,
                "corr_short": round(corr_s, 4),
                "corr_long":  round(corr_l, 4),
                "divergence": div,
                "trend":      trend,
                "signal":     signal,
            })
        except Exception as exc:
            _log.debug("pair_corr_error", pair=f"{t1}/{t2}", error=str(exc))

    results.sort(key=lambda x: abs(x["divergence"]), reverse=True)
    return results


# ── 4. Systemic Risk Index ────────────────────────────────────────────────────

def systemic_risk_index(
    contagion_scores: dict[str, float],
    market_prices: dict[str, Any],
    node_ticker: dict[str, str],
) -> float:
    """
    SRI ∈ [0, 1] — índice de risco sistêmico ponderado por liquidez.

    SRI = tanh(Σ_i(liq_weight_i × contagion_i) / Σ_i(liq_weight_i) × 5)

    Thresholds:
      < 0.25 → Verde  (baixo risco)
      0.25-0.50 → Amarelo (atenção)
      0.50-0.75 → Laranja (elevado)
      > 0.75 → Vermelho (crise)
    """
    try:
        from app.desk.node_registry import NODES
        total_w = 0.0
        total_s = 0.0
        for nid, score in contagion_scores.items():
            nd = NODES.get(nid, {})
            w  = nd.get("liquidity_weight") or nd.get("weight") or 0.005
            total_w += w
            total_s += w * score
        if total_w <= 0:
            return 0.0
        sri_raw = total_s / total_w
        return round(math.tanh(sri_raw * 5.0), 4)
    except Exception:
        if not contagion_scores:
            return 0.0
        return round(math.tanh(float(np.mean(list(contagion_scores.values()))) * 5.0), 4)


def _sri_label(sri: float) -> tuple[str, str]:
    """(label, color) para o SRI."""
    if sri < 0.25:
        return "Low",      "#4ade80"
    if sri < 0.50:
        return "Elevated", "#f59e0b"
    if sri < 0.75:
        return "High",     "#f97316"
    return "Critical",     "#ef4444"


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze_live(
    market_prices: dict[str, Any],
    graph_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Análise completa da rede em tempo real.

    Args:
        market_prices : bundle.market_prices
        graph_data    : saída do graph_engine.build_graph()
                        (contém elements.edges com MST)

    Returns dict com:
        contagion        : {node_id: score}
        propagated_shock : {node_id: shock_magnitude}
        rolling_corr     : lista de pares com tendência
        sri              : float [0,1]
        sri_label        : str
        sri_color        : str
        top_sources      : top 5 nós por contagion score
    """
    nodes = graph_data.get("elements", {}).get("nodes", [])
    edges = graph_data.get("elements", {}).get("edges", [])
    mst_edges = [e for e in edges if e.get("data", {}).get("type") == "mst"]

    # node_id → ticker map
    node_ticker: dict[str, str] = {}
    for n in nodes:
        d = n.get("data", {})
        tk = d.get("ticker")
        if tk:
            node_ticker[d["id"]] = tk

    # source_returns: node_id → daily return (com sinal)
    source_returns: dict[str, float] = {}
    for n in nodes:
        d = n.get("data", {})
        r = d.get("daily")
        if r is not None:
            source_returns[d["id"]] = float(r)

    # 1. Contagion
    contagion = compute_contagion(market_prices, mst_edges, node_ticker)

    # 2. Propagated shock
    prop_shock = compute_propagated_shock(source_returns, mst_edges)

    # 3. Rolling correlations (async-ish — não trava o run se falhar)
    rolling_corr: list[dict] = []
    try:
        rolling_corr = rolling_corr_trends()
    except Exception as exc:
        _log.warning("rolling_corr_failed", error=str(exc))

    # 4. SRI
    sri       = systemic_risk_index(contagion, market_prices, node_ticker)
    lbl, col  = _sri_label(sri)

    # Top 5 contagion sources
    top_sources = sorted(
        [{"node_id": k, "score": v, "ticker": node_ticker.get(k, k)}
         for k, v in contagion.items() if v > 0],
        key=lambda x: x["score"],
        reverse=True,
    )[:5]

    _log.info("live_network_done",
              sri=sri, sri_label=lbl,
              contagion_nodes=len(contagion),
              rolling_pairs=len(rolling_corr))

    return {
        "contagion":        contagion,
        "propagated_shock": prop_shock,
        "rolling_corr":     rolling_corr,
        "sri":              sri,
        "sri_label":        lbl,
        "sri_color":        col,
        "top_sources":      top_sources,
    }
