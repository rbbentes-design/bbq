"""
Analysis: Network / Graph Analysis of Financial Assets

Três camadas de análise inspiradas em Econophysics e Graphical Models:

1. RMT Cleaning (Random Matrix Theory)
   - Filtra ruído da matriz de correlação via lei de Marchenko-Pastur
   - Separa autovalores de "ruído" dos de "informação real"
   - Reconstrói correlação limpa + estima grafo esparso (GraphicalLasso)

2. Minimum Spanning Tree (Mantegna 1999)
   - Métrica de distância: d_ij = sqrt(2 * (1 - rho_ij))
   - MST = backbone mínimo de conexões entre ativos
   - Hub nodes = ativos mais conectados (líderes de cluster)

3. Regime Detection via Topologia do Grafo
   - Rastreia métricas do grafo ao longo do tempo
   - Detecta quebras estruturais: dispersão, concentração, colapso de correlação
   - Entropy score: quão desordenado está o sistema

Referências:
  - Mantegna (1999): Hierarchical structure in financial markets
  - Zhao et al. (2019): Gaussian graphical model for S&P 500
  - Marchenko & Pastur (1967): Distribution of eigenvalues in random matrices
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.audit.logger import get_logger

_log = get_logger("analysis.network")


# ── Universo SPX Core (~50 tickers) ───────────────────────────────────────────
# Indices, sector ETFs e top stocks por peso no S&P 500.
# Suficientemente pequeno para MST legível e rápido (yfinance sem rate-limit).
SPX_CORE: set[str] = {
    # Top 20 SPX por peso de mercado (abr/2026)
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "TSLA", "BRK-B", "AVGO", "JPM",
    "LLY", "UNH", "XOM", "COST", "V",
    "MA", "WMT", "NFLX", "JNJ", "PG",
    # Referências de índice e macro
    "^GSPC", "^VIX", "SPY", "QQQ",
    "GLD", "TLT", "HYG", "CL=F",
}


# ── 1. RMT — Random Matrix Theory ─────────────────────────────────────────────

def marchenko_pastur_threshold(n_assets: int, n_obs: int, sigma: float = 1.0) -> float:
    """
    Limite superior da lei de Marchenko-Pastur.

    Autovalores abaixo deste limiar são ruído puro (não carregam informação).

    Args:
        n_assets : número de ativos (p)
        n_obs    : número de observações (T)
        sigma    : variância da distribuição (1.0 para correlação)

    Returns:
        lambda_max da distribuição de Marchenko-Pastur
    """
    q = n_obs / n_assets  # razão T/p
    lambda_max = sigma**2 * (1 + 1 / q + 2 * np.sqrt(1 / q))
    return lambda_max


def rmt_clean_correlation(
    returns_df: Any,  # pd.DataFrame
    alpha: float = 0.1,
) -> dict[str, Any]:
    """
    Limpa a matriz de correlação via RMT (Marchenko-Pastur).

    Processo:
      1. Computa matriz de correlação empírica
      2. Decompõe em autovalores/autovetores
      3. Zera autovalores abaixo do limiar de Marchenko-Pastur (ruído)
      4. Reconstrói correlação limpa
      5. Aplica GraphicalLasso para estimar grafo esparso (precision matrix)

    Args:
        returns_df : pd.DataFrame com retornos (linhas=tempo, colunas=ativos)
        alpha      : penalidade do GraphicalLasso (maior = mais esparso)

    Returns:
        {
          "corr_raw"        : matriz de correlação bruta (n x n dict)
          "corr_clean"      : matriz de correlação limpa (n x n dict)
          "n_signal_factors": número de fatores acima do ruído
          "eigenvalues"     : autovalores ordenados
          "lambda_threshold": limiar de Marchenko-Pastur
          "precision_matrix": grafo esparso (precision matrix) — arestas com peso
          "edges"           : lista de arestas [(i, j, weight), ...]
          "tickers"         : lista de tickers
        }
    """
    try:
        import pandas as pd
        from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
    except ImportError as e:
        _log.error("rmt_import_error", error=str(e))
        return {}

    df = returns_df.dropna(axis=1, how="any").dropna(axis=0, how="any")
    if df.shape[0] < 20 or df.shape[1] < 3:
        _log.warning("rmt_insufficient_data", rows=df.shape[0], cols=df.shape[1])
        return {}

    T, p = df.shape
    tickers = list(df.columns)

    # ── Correlação bruta ────────────────────────────────────────────────────────
    corr_raw = df.corr().values  # (p, p)

    # ── Decomposição espectral ──────────────────────────────────────────────────
    eigenvalues, eigenvectors = np.linalg.eigh(corr_raw)
    # eigh retorna em ordem crescente; invertemos para decrescente
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    lambda_thresh = marchenko_pastur_threshold(p, T)
    n_signal = int(np.sum(eigenvalues > lambda_thresh))

    _log.info("rmt_eigenvalues",
              lambda_max=round(float(eigenvalues[0]), 3),
              lambda_thresh=round(lambda_thresh, 3),
              n_signal=n_signal,
              n_noise=p - n_signal)

    # ── Reconstrói correlação limpa (só fatores de sinal) ─────────────────────
    # Mantém os n_signal maiores autovalores, substitui o resto pela média
    evals_clean = eigenvalues.copy()
    noise_mask = evals_clean <= lambda_thresh
    # A variância total deve ser preservada — substituímos ruído pela média
    noise_mean = (np.trace(corr_raw) - np.sum(eigenvalues[~noise_mask])) / max(np.sum(noise_mask), 1)
    evals_clean[noise_mask] = noise_mean

    corr_clean = eigenvectors @ np.diag(evals_clean) @ eigenvectors.T
    # Normaliza diagonal para 1.0
    d = np.sqrt(np.diag(corr_clean))
    d[d == 0] = 1.0
    corr_clean = corr_clean / np.outer(d, d)
    np.fill_diagonal(corr_clean, 1.0)
    corr_clean = np.clip(corr_clean, -1.0, 1.0)

    # ── GraphicalLasso — precision matrix (grafo esparso) ─────────────────────
    precision = None
    edges: list[tuple[str, str, float]] = []
    try:
        gl = GraphicalLasso(alpha=alpha, max_iter=200, tol=1e-4)
        gl.fit(df.values)
        precision = gl.precision_

        # Extrai arestas: elementos off-diagonal não nulos
        for i in range(p):
            for j in range(i + 1, p):
                w = float(precision[i, j])
                if abs(w) > 1e-6:
                    edges.append((tickers[i], tickers[j], round(w, 4)))

        _log.info("graphical_lasso_done", edges=len(edges), alpha=alpha)
    except Exception as exc:
        _log.warning("graphical_lasso_failed", error=str(exc))

    # ── Converte matrizes para dict serializable ────────────────────────────────
    def matrix_to_dict(mat: np.ndarray) -> dict[str, dict[str, float]]:
        return {
            tickers[i]: {tickers[j]: round(float(mat[i, j]), 4) for j in range(p)}
            for i in range(p)
        }

    return {
        "tickers":          tickers,
        "n_assets":         p,
        "n_obs":            T,
        "lambda_threshold": round(lambda_thresh, 4),
        "n_signal_factors": n_signal,
        "eigenvalues":      [round(float(e), 4) for e in eigenvalues[:10]],
        "corr_raw":         matrix_to_dict(corr_raw),
        "corr_clean":       matrix_to_dict(corr_clean),
        "precision_matrix": matrix_to_dict(precision) if precision is not None else {},
        "edges":            edges,
        "edge_count":       len(edges),
    }


# ── 2. Minimum Spanning Tree (Mantegna 1999) ──────────────────────────────────

def mantegna_distance(rho: float) -> float:
    """d_ij = sqrt(2 * (1 - rho)) — métrica de distância de Mantegna."""
    rho_clipped = max(-1.0, min(1.0, rho))
    return float(np.sqrt(2.0 * (1.0 - rho_clipped)))


def minimum_spanning_tree(
    corr_matrix: dict[str, dict[str, float]],
    tickers: list[str],
) -> dict[str, Any]:
    """
    Constrói o Minimum Spanning Tree (MST) de Mantegna.

    O MST é o "esqueleto" mínimo da rede — conecta todos os ativos
    com o menor custo total de distância, revelando a estrutura hierárquica.

    Args:
        corr_matrix : correlações pairwise (pode ser raw ou cleaned)
        tickers     : lista de tickers

    Returns:
        {
          "edges"    : [(ticker_i, ticker_j, distance, correlation), ...]
          "hubs"     : [(ticker, degree), ...] — ativos mais conectados
          "clusters" : clusters identificados por componentes conectados
          "avg_distance": distância média das arestas do MST
        }
    """
    n = len(tickers)
    if n < 3:
        return {}

    # Matriz de distância
    dist = np.zeros((n, n))
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if i != j:
                rho = corr_matrix.get(ti, {}).get(tj, 0.0)
                dist[i, j] = mantegna_distance(rho)

    # Kruskal via union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True

    # Coleta todas as arestas (i < j) e ordena por distância
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((dist[i, j], i, j))
    all_edges.sort()

    mst_edges = []
    adjacency: dict[str, list[str]] = {t: [] for t in tickers}

    for d_val, i, j in all_edges:
        if len(mst_edges) == n - 1:
            break
        if union(i, j):
            rho = corr_matrix.get(tickers[i], {}).get(tickers[j], 0.0)
            mst_edges.append({
                "from":        tickers[i],
                "to":          tickers[j],
                "distance":    round(d_val, 4),
                "correlation": round(float(rho), 4),
            })
            adjacency[tickers[i]].append(tickers[j])
            adjacency[tickers[j]].append(tickers[i])

    # Hub nodes — maior grau no MST
    hubs = sorted(
        [(t, len(neighbors)) for t, neighbors in adjacency.items() if neighbors],
        key=lambda x: x[1],
        reverse=True,
    )

    avg_dist = float(np.mean([e["distance"] for e in mst_edges])) if mst_edges else 0.0
    avg_corr = float(np.mean([e["correlation"] for e in mst_edges])) if mst_edges else 0.0

    _log.info("mst_done",
              edges=len(mst_edges),
              top_hub=hubs[0][0] if hubs else None,
              avg_corr=round(avg_corr, 3))

    return {
        "edges":        mst_edges,
        "hubs":         hubs[:10],
        "avg_distance": round(avg_dist, 4),
        "avg_corr":     round(avg_corr, 4),
        "n_edges":      len(mst_edges),
    }


# ── 3. Regime Detection via Topologia do Grafo ────────────────────────────────

def graph_regime(
    corr_matrix: dict[str, dict[str, float]],
    tickers: list[str],
    corr_threshold: float = 0.5,
    mst_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Detecta o regime de mercado via métricas topológicas do grafo de correlação.

    Métricas calculadas:
      - avg_correlation     : correlação média entre todos os pares
      - correlation_entropy : quão dispersa é a distribuição de correlações
                             (alta entropia = mercado desordenado)
      - density             : fração de pares com |rho| > threshold
      - max_corr            : correlação máxima (sinal de crowding)
      - min_corr            : correlação mínima (sinal de hedge)
      - corr_dispersion     : desvio padrão das correlações (volatilidade de regime)
      - hub_dominance       : quanto o hub principal domina o MST (grau/n_ativos)
      - regime              : "risk_on" | "risk_off" | "transition" | "chaotic"
      - regime_confidence   : 0.0 a 1.0

    Args:
        corr_matrix     : matriz de correlação
        tickers         : lista de tickers
        corr_threshold  : limiar para considerar correlação "alta"
        mst_data        : output de minimum_spanning_tree() (opcional)

    Returns:
        dict com métricas e diagnóstico de regime
    """
    n = len(tickers)
    if n < 3:
        return {}

    # Coleta todos os pares off-diagonal
    corrs: list[float] = []
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if j <= i:
                continue
            rho = corr_matrix.get(ti, {}).get(tj, 0.0)
            corrs.append(float(rho))

    if not corrs:
        return {}

    arr = np.array(corrs)
    avg_corr      = float(np.mean(arr))
    std_corr      = float(np.std(arr))
    max_corr      = float(np.max(arr))
    min_corr      = float(np.min(arr))
    density_high  = float(np.mean(np.abs(arr) > corr_threshold))
    density_neg   = float(np.mean(arr < -corr_threshold))

    # Entropy de Shannon sobre histograma de correlações
    hist, _ = np.histogram(arr, bins=20, range=(-1, 1), density=True)
    hist = hist + 1e-10
    hist = hist / hist.sum()
    corr_entropy = float(-np.sum(hist * np.log(hist)))

    # Hub dominance via MST
    hub_dominance = 0.0
    if mst_data and mst_data.get("hubs"):
        top_hub_degree = mst_data["hubs"][0][1]
        hub_dominance = top_hub_degree / max(n - 1, 1)

    # ── Diagnóstico de regime ──────────────────────────────────────────────────
    #
    # Risk-ON  : correlações moderadas, bem estruturadas, baixa entropia
    #            → mercado "normal", setores se movendo por fundamento
    #
    # Risk-OFF : correlações altas e uniformes (tudo sobe/cai junto)
    #            → selloff ou rally de risk-on com compressão de dispersão
    #
    # Transition: correlações aumentando rapidamente, estrutura mudando
    #             → momento de ruptura de regime
    #
    # Chaotic  : correlações instáveis, alta dispersão, entropia máxima
    #            → mercado sem estrutura clara (evitar posições direcionais)

    regime: str
    confidence: float

    if avg_corr > 0.55 and std_corr < 0.15:
        regime = "risk_off"
        confidence = min(1.0, (avg_corr - 0.55) / 0.3 + (0.15 - std_corr) / 0.15)
    elif avg_corr > 0.35 and density_high > 0.5 and corr_entropy < 2.5:
        regime = "risk_on"
        confidence = min(1.0, density_high * 0.6 + (2.5 - corr_entropy) / 2.5 * 0.4)
    elif std_corr > 0.25 or corr_entropy > 2.8:
        regime = "chaotic"
        # Confiança baseada em quanto excede os limiares (sempre positiva)
        conf_std  = max(0.0, (std_corr - 0.25) / 0.2)
        conf_ent  = max(0.0, (corr_entropy - 2.8) / 0.5)
        confidence = min(1.0, 0.3 + conf_std * 0.5 + conf_ent * 0.5)
    else:
        regime = "transition"
        confidence = 0.4

    confidence = round(max(0.0, min(1.0, confidence)), 2)

    _log.info("regime_detected",
              regime=regime,
              confidence=confidence,
              avg_corr=round(avg_corr, 3),
              entropy=round(corr_entropy, 3))

    return {
        "regime":           regime,
        "confidence":       confidence,
        "avg_correlation":  round(avg_corr, 4),
        "std_correlation":  round(std_corr, 4),
        "max_correlation":  round(max_corr, 4),
        "min_correlation":  round(min_corr, 4),
        "density_high":     round(density_high, 4),
        "density_negative": round(density_neg, 4),
        "corr_entropy":     round(corr_entropy, 4),
        "hub_dominance":    round(hub_dominance, 4),
        "n_pairs":          len(corrs),
    }


# ── Pipeline completo ──────────────────────────────────────────────────────────

def analyze(
    market_prices_raw: dict[str, Any],
    lookback_days: int = 90,
    lasso_alpha: float = 0.1,
    corr_threshold: float = 0.5,
    universe: str = "spx",  # "spx" = usa SPX_CORE; "all" = todos os tickers
) -> dict[str, Any]:
    """
    Executa análise completa de rede financeira:
      1. Busca histórico de preços via yfinance
      2. Limpa correlação via RMT
      3. Constrói MST de Mantegna
      4. Detecta regime via topologia

    Args:
        market_prices_raw : output de market_prices.collect()
        lookback_days     : janela histórica em dias
        lasso_alpha       : penalidade do GraphicalLasso (0.05-0.3)
        corr_threshold    : limiar para densidade de arestas

    Returns:
        {
          "rmt"    : resultado do RMT cleaning
          "mst"    : minimum spanning tree
          "regime" : diagnóstico de regime
          "summary": resumo textual para o LLM
        }
    """
    try:
        import pandas as pd
    except ImportError as e:
        _log.error("network_import_error", error=str(e))
        return {}

    tickers = list(market_prices_raw.keys())
    if not tickers:
        return {}

    # ── Filtra universo ────────────────────────────────────────────────────────
    if universe == "spx":
        tickers = sorted(SPX_CORE)
        _log.info("network_universe_spx", tickers=len(tickers))

    # ── 1. Bloomberg CSV (BQuant export) — fonte primária ─────────────────────
    closes: dict[str, list[float]] = {}
    try:
        from app.providers.bql_csv import load_price_history
        bbg_hist = load_price_history()
        if bbg_hist:
            closes = bbg_hist
            _log.info("network_bloomberg_csv", tickers=len(closes))
    except Exception as exc:
        _log.warning("network_bloomberg_csv_failed", error=str(exc))

    # ── 2. IBKR — fallback ────────────────────────────────────────────────────
    if len(closes) < 4:
        try:
            from app.providers.ibkr import fetch_historical_closes
            closes = fetch_historical_closes(tickers, lookback_days=lookback_days)
        except Exception as exc:
            _log.warning("network_ibkr_failed", error=str(exc))

    # ── Fallback yfinance para tickers sem dados do IBKR ──────────────────────
    missing = [t for t in tickers if t not in closes]
    if missing:
        try:
            import yfinance as yf
            import pandas as pd
            raw = yf.download(missing, period=f"{lookback_days}d",
                              auto_adjust=True, progress=False, threads=True)
            close_df = raw["Close"] if "Close" in raw.columns else raw
            if isinstance(close_df, pd.Series):
                close_df = close_df.to_frame(missing[0])
            for sym in missing:
                if sym in close_df.columns:
                    s = close_df[sym].dropna()
                    if len(s) >= 10:
                        closes[sym] = [float(v) for v in s.values]
            _log.info("network_yf_fallback", filled=len(closes) - (len(tickers) - len(missing)))
        except Exception as exc:
            _log.warning("network_yf_fallback_failed", error=str(exc))

    if len(closes) < 4:
        _log.warning("network_too_few_tickers", n=len(closes))
        return {}

    # ── Alinha séries e calcula retornos ──────────────────────────────────────
    min_len = min(len(v) for v in closes.values())
    returns_dict = {}
    for sym, prices in closes.items():
        p = prices[-min_len:]
        rets = [(p[i] - p[i - 1]) / p[i - 1] for i in range(1, len(p))]
        returns_dict[sym] = rets

    df_returns = pd.DataFrame(returns_dict).dropna()
    valid_tickers = list(df_returns.columns)

    _log.info("network_data_ready",
              tickers=len(valid_tickers),
              observations=len(df_returns))

    # ── 1. RMT ─────────────────────────────────────────────────────────────────
    rmt_result = rmt_clean_correlation(df_returns, alpha=lasso_alpha)

    # ── 2. MST ─────────────────────────────────────────────────────────────────
    mst_result = {}
    if rmt_result.get("corr_clean"):
        mst_result = minimum_spanning_tree(rmt_result["corr_clean"], valid_tickers)

    # ── 3. Regime ──────────────────────────────────────────────────────────────
    regime_result = {}
    if rmt_result.get("corr_clean"):
        regime_result = graph_regime(
            rmt_result["corr_clean"],
            valid_tickers,
            corr_threshold=corr_threshold,
            mst_data=mst_result,
        )

    return {
        "rmt":    rmt_result,
        "mst":    mst_result,
        "regime": regime_result,
        "tickers": valid_tickers,
        "lookback_days": lookback_days,
    }


# ── Formatação para contexto do LLM ───────────────────────────────────────────

def format_summary(result: dict[str, Any], market_prices: dict[str, Any]) -> str:
    """Formata resultado da análise de rede para inclusão no diagnóstico do agente."""
    if not result:
        return ""

    lines = ["=== ANÁLISE DE REDE (Econophysics) ==="]

    rmt = result.get("rmt", {})
    if rmt:
        lines.append(
            f"  RMT: {rmt.get('n_signal_factors', '?')} fatores de sinal "
            f"de {rmt.get('n_assets', '?')} ativos "
            f"(limiar MP={rmt.get('lambda_threshold', '?')})"
        )
        if rmt.get("edge_count") is not None:
            lines.append(f"  Grafo esparso: {rmt['edge_count']} conexões reais")

    mst = result.get("mst", {})
    if mst:
        lines.append(f"  MST distância média: {mst.get('avg_distance', '?'):.3f}  "
                     f"correlação média: {mst.get('avg_corr', '?'):+.3f}")
        hubs = mst.get("hubs", [])[:4]
        if hubs:
            hub_str = " > ".join(
                f"{market_prices.get(t, {}).get('name', t)} (grau {d})"
                for t, d in hubs
            )
            lines.append(f"  Hubs MST: {hub_str}")

    regime = result.get("regime", {})
    if regime:
        r = regime.get("regime", "?")
        c = regime.get("confidence", 0)
        avg_rho = regime.get("avg_correlation", 0)
        entropy = regime.get("corr_entropy", 0)
        lines.append(
            f"  Regime de rede: {r.upper()} (confiança {c:.0%})"
            f"  | rho_medio={avg_rho:+.3f}"
            f"  | entropia={entropy:.2f}"
        )

    return "\n".join(lines)
