"""
Analysis: Probabilistic Layer

Por ticker, calcula:
  • Distribuição de retornos — ajuste t-Student (df, mu, sigma)
  • VaR / CVaR            — Value at Risk e Expected Shortfall (95% e 99%)
  • Tail Score            — índice composto [0-1] de fat tails + skewness
  • FFT Spectral          — ciclo dominante em dias de negociação
  • Regime HMM            — P(bull) via Hidden Markov 2-estado
                            Fallback: z-score rolling se hmmlearn ausente

Tickers cobertos: todos os registry level-5 com yfinance (stocks + ETFs).
Índices especiais (^) são ignorados se não tiverem histórico acessível.

Uso:
    from app.analysis.probabilistic import analyze_batch
    prob = analyze_batch(["AAPL", "SPY", "TLT"])
    # prob["AAPL"] → {var_95, cvar_95, var_99, cvar_99, skewness, kurtosis,
    #                  tail_score, dominant_cycle, regime_prob_bull, dist}
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.audit.logger import get_logger

_log = get_logger("analysis.probabilistic")

# Mínimo de pontos para análise confiável
_MIN_OBS = 60

# Tickers prioritários para análise completa (os demais usam versão rápida)
_PRIORITY = {
    "SPY", "QQQ", "IWM",
    "^GSPC", "^NDX", "^RUT",
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN",
    "GLD", "TLT", "HYG", "BTC-USD", "DX-Y.NYB",
    "^VIX",
}


# ── Helpers numéricos ─────────────────────────────────────────────────────────

def _safe(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _returns_from_prices(prices: np.ndarray) -> np.ndarray:
    """Log-retornos diários."""
    return np.diff(np.log(prices[prices > 0]))


# ── Distribuição t-Student ────────────────────────────────────────────────────

def _fit_t(returns: np.ndarray) -> dict[str, float | None]:
    """Ajusta t-Student via scipy. Retorna df, mu, sigma."""
    try:
        from scipy.stats import t as t_dist
        df, mu, sigma = t_dist.fit(returns, floc=None, fscale=None)
        return {"df": _safe(df), "mu": _safe(mu), "sigma": _safe(sigma)}
    except Exception:
        mu    = float(np.mean(returns))
        sigma = float(np.std(returns))
        return {"df": None, "mu": _safe(mu), "sigma": _safe(sigma)}


def _var_cvar_t(
    returns: np.ndarray,
    dist_params: dict[str, float | None],
    alpha: float,  # 0.05 → 95% VaR
) -> tuple[float | None, float | None]:
    """
    VaR e CVaR analíticos da t-Student.
    VaR: negativo = perda  (ex: -0.023 = 2.3% de perda no pior caso)
    CVaR: E[r | r < VaR]   (sempre ≤ VaR)
    """
    try:
        from scipy.stats import t as t_dist
        df    = dist_params.get("df")
        mu    = dist_params.get("mu") or float(np.mean(returns))
        sigma = dist_params.get("sigma") or float(np.std(returns))
        if df is None or df <= 2:
            # Sem df finito: usa normal
            from scipy.stats import norm
            var  = float(norm.ppf(alpha, mu, sigma))
            # CVaR normal: mu - sigma * phi(z) / alpha
            from scipy.stats import norm as _n
            z    = (var - mu) / sigma
            cvar = float(mu - sigma * _n.pdf(z) / alpha)
        else:
            var  = float(t_dist.ppf(alpha, df, mu, sigma))
            # CVaR t-Student analítico
            z    = (var - mu) / sigma
            t_pdf_z = t_dist.pdf(z, df)
            cvar = float(mu - sigma * (df + z**2) / (df - 1) * t_pdf_z / alpha)
        return _safe(var), _safe(cvar)
    except Exception:
        # Fallback empírico
        sorted_r = np.sort(returns)
        n = len(sorted_r)
        idx = max(1, int(alpha * n))
        var  = float(sorted_r[idx])
        cvar = float(np.mean(sorted_r[:idx]))
        return _safe(var), _safe(cvar)


# ── Tail Score ────────────────────────────────────────────────────────────────

def _tail_score(returns: np.ndarray) -> tuple[float, float, float]:
    """
    Retorna (skewness, excess_kurtosis, tail_score).

    tail_score ∈ [0, 1]:
      0 = distribuição normal pura
      1 = caudas muito gordas + alta assimetria (ex: BTC, small caps)

    Fórmula:
      tail_score = 0.5 × tanh(max(0, exc_kurt) / 3)
                 + 0.5 × tanh(|skewness| / 2)
    """
    try:
        from scipy.stats import skew, kurtosis
        sk  = float(skew(returns))
        ku  = float(kurtosis(returns))  # excess kurtosis (normal = 0)
    except Exception:
        sk = float(np.mean((returns - returns.mean())**3) / np.std(returns)**3)
        ku = float(np.mean((returns - returns.mean())**4) / np.std(returns)**4) - 3.0

    ts = 0.5 * math.tanh(max(0.0, ku) / 3.0) + 0.5 * math.tanh(abs(sk) / 2.0)
    return (_safe(sk) or 0.0), (_safe(ku) or 0.0), round(ts, 4)


# ── FFT Spectral ──────────────────────────────────────────────────────────────

def _dominant_cycle(returns: np.ndarray, min_period: int = 5) -> int | None:
    """
    Encontra o ciclo dominante via FFT dos retornos demediados.

    Retorna período em dias de negociação (ex: 21 ≈ ciclo mensal).
    Ignora frequências abaixo de min_period dias.
    """
    try:
        n = len(returns)
        if n < 2 * min_period:
            return None
        r = returns - returns.mean()
        fft_vals = np.abs(np.fft.rfft(r))
        freqs    = np.fft.rfftfreq(n)

        # Máscara: exclui DC (freq=0) e períodos muito curtos
        mask = (freqs > 0) & (freqs <= 1.0 / min_period)
        if not np.any(mask):
            return None

        dominant_freq = freqs[mask][np.argmax(fft_vals[mask])]
        period = int(round(1.0 / dominant_freq))
        return max(min_period, period)
    except Exception:
        return None


# ── Regime HMM ────────────────────────────────────────────────────────────────

def _regime_hmm(returns: np.ndarray) -> float | None:
    """
    Probabilidade de estar no estado 'bull' via HMM 2-estado.

    Usa hmmlearn.GaussianHMM se disponível.
    Fallback: z-score rolling (media 20d / std 252d).
    Identifica bull = estado com maior média.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
        X = returns.reshape(-1, 1)
        model = GaussianHMM(
            n_components=2, covariance_type="diag",
            n_iter=100, tol=1e-4, random_state=42,
        )
        model.fit(X)
        states = model.predict(X)
        probs  = model.predict_proba(X)

        # Estado 0 ou 1 é o "bull" (maior média)
        means = model.means_.flatten()
        bull_state = int(np.argmax(means))
        p_bull = float(probs[-1, bull_state])
        return _safe(p_bull)

    except ImportError:
        pass
    except Exception as exc:
        _log.debug("hmm_failed", error=str(exc))

    # ── Fallback z-score ──────────────────────────────────────────────────────
    try:
        if len(returns) < 40:
            return None
        window_fast = min(20, len(returns) // 3)
        window_slow = min(252, len(returns))
        recent_mean = float(np.mean(returns[-window_fast:]))
        long_std    = float(np.std(returns[-window_slow:]))
        if long_std <= 0:
            return None
        z = recent_mean / (long_std / math.sqrt(window_fast))
        # z → P(bull) via logistic
        p_bull = 1.0 / (1.0 + math.exp(-z * 1.5))
        return _safe(p_bull)
    except Exception:
        return None


# ── Coleta de dados ───────────────────────────────────────────────────────────

def _fetch_returns(ticker: str, period: str = "1y") -> np.ndarray | None:
    """
    Retorna log-retornos. Tenta primeiro Bloomberg DB (price_history); fallback
    yfinance só se USE_FALLBACKS=1 (yfinance era ~5s/ticker × 270 = 22 min).
    """
    # ── Tier 0: Bloomberg DB ──
    try:
        from app.providers.bql_csv import load_price_history
        bbg_hist = load_price_history()
        if bbg_hist:
            series = bbg_hist.get(ticker)
            if series and len(series) >= _MIN_OBS:
                return _returns_from_prices(np.array(series, dtype=float))
    except Exception:
        pass

    # ── Tier 1: yfinance (só se USE_FALLBACKS=1) ──
    import os as _os
    if not _os.environ.get("USE_FALLBACKS"):
        return None
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist.empty or len(hist) < _MIN_OBS:
            return None
        prices = hist["Close"].dropna().values.astype(float)
        return _returns_from_prices(prices)
    except Exception as exc:
        _log.debug("fetch_returns_error", ticker=ticker, error=str(exc))
        return None


# ── Análise individual ────────────────────────────────────────────────────────

def analyze_ticker(
    ticker: str,
    period: str = "1y",
) -> dict[str, Any] | None:
    """
    Perfil probabilístico completo de um ticker.

    Returns None se dados insuficientes.
    """
    returns = _fetch_returns(ticker, period)
    if returns is None or len(returns) < _MIN_OBS:
        return None

    dist_params = _fit_t(returns)
    var_95, cvar_95 = _var_cvar_t(returns, dist_params, 0.05)
    var_99, cvar_99 = _var_cvar_t(returns, dist_params, 0.01)
    sk, ku, ts      = _tail_score(returns)
    cycle           = _dominant_cycle(returns)
    p_bull          = _regime_hmm(returns)

    # Ann. vol
    ann_vol = _safe(float(np.std(returns)) * math.sqrt(252))

    return {
        "var_95":          var_95,
        "cvar_95":         cvar_95,
        "var_99":          var_99,
        "cvar_99":         cvar_99,
        "ann_vol":         ann_vol,
        "skewness":        _safe(sk),
        "excess_kurtosis": _safe(ku),
        "tail_score":      ts,
        "dominant_cycle":  cycle,
        "regime_prob_bull": p_bull,
        "dist": dist_params,
        "n_obs": len(returns),
    }


# ── Batch ─────────────────────────────────────────────────────────────────────

def analyze_batch(
    tickers: list[str],
    period: str = "1y",
    max_workers: int = 10,
) -> dict[str, dict[str, Any]]:
    """
    Analisa múltiplos tickers em paralelo.

    Priority tickers usam period="2y" para HMM mais robusto.
    Demais usam period padrão.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, dict[str, Any]] = {}
    _log.info("prob_batch_start", tickers=len(tickers))

    def _analyze_one(sym: str) -> tuple[str, dict | None]:
        try:
            p = "2y" if sym in _PRIORITY else period
            return sym, analyze_ticker(sym, period=p)
        except Exception as exc:
            _log.debug("prob_error", sym=sym, error=str(exc))
            return sym, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_analyze_one, sym): sym for sym in tickers}
        for fut in as_completed(futures):
            sym, res = fut.result()
            if res:
                results[sym] = res
                _log.debug("prob_ok", sym=sym,
                            var95=res.get("var_95"),
                            tail=res.get("tail_score"),
                            bull=res.get("regime_prob_bull"))

    _log.info("prob_batch_done", collected=len(results), of=len(tickers))
    return results


def analyze_from_registry(period: str = "1y") -> dict[str, dict[str, Any]]:
    """
    Analisa todos os tickers level-5 do node_registry.
    Entry point para graph_engine.build_from_bundle().
    """
    try:
        from app.desk.node_registry import NODES
        tickers = list({
            n.get("ticker")
            for n in NODES.values()
            if n.get("ticker") and n.get("level", 99) == 5
        })
    except Exception as exc:
        _log.warning("registry_load_failed", error=str(exc))
        return {}

    return analyze_batch(tickers, period=period)
