"""
Portfolio Optimizer — Alocação Autônoma de $100k

Teoria base: Black-Litterman-inspired Mean-Variance com overlay de regime.

Pipeline:
  1. Filtra universe: apenas ativos com direction != 'neutral' e conviction >= 'medium'
  2. Estima expected returns via signal × vol (views BL)
  3. Estima covariância via vol_ann × correlation (ou diagonal se sem corr)
  4. Maximiza Sharpe ajustado (scipy.optimize ou fallback analítico)
  5. Aplica constraints:
       - max_per_ticker  = 15%  (evita concentração)
       - max_per_cluster = 30%  (diversificação MST)
       - max_short_pct   = 20%  do portfolio (shorts via posição negativa)
       - min_position    = 2%   (sem micro-posições sem sentido)
  6. Overlay de regime:
       - Bear (regime_bull < 0.35): força max_equities = 40%, min_defensive = 30%
       - Bull (regime_bull > 0.65): libera equities até 70%
  7. Scaling para $100k nominal

Output: PortfolioResult com alocações, métricas e rationale
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.audit.logger import get_logger
from app.analysis.alpha_signals import AssetSignal

_log = get_logger("analysis.portfolio_optimizer")

# ── Classificação de asset class por ticker ────────────────────────────────────
_ASSET_CLASS: dict[str, str] = {
    # Equities US
    "SPY": "equity", "QQQ": "equity", "IWM": "equity", "DIA": "equity",
    "^GSPC": "equity", "^NDX": "equity", "^RUT": "equity",
    "AAPL": "equity", "MSFT": "equity", "NVDA": "equity", "TSLA": "equity",
    "META": "equity", "GOOGL": "equity", "AMZN": "equity", "JPM": "equity",
    "GS": "equity", "BAC": "equity", "AMD": "equity", "INTC": "equity",
    "NFLX": "equity", "XOM": "equity", "CVX": "equity",
    # Fixed income — tratados como CASH (excluídos do portfolio)
    "TLT": "bonds", "IEF": "bonds", "SHY": "bonds", "HYG": "bonds",
    "LQD": "bonds", "BND": "bonds", "AGG": "bonds", "BIL": "bonds",
    "IEI": "bonds", "TIP": "bonds", "GOVT": "bonds", "SHV": "bonds",
    "VGIT": "bonds", "VGLT": "bonds", "BNDX": "bonds",
    # Commodities / real assets
    "GLD": "commodities", "SLV": "commodities", "GC=F": "commodities",
    "CL=F": "commodities", "NG=F": "commodities",
    "DX-Y.NYB": "fx", "EURUSD=X": "fx", "USDJPY=X": "fx",
    "BTC-USD": "crypto", "ETH-USD": "crypto",
    # EM / International
    "EEM": "intl_equity", "EWZ": "intl_equity", "FXI": "intl_equity",
}

# Tickers de renda fixa pura = CASH. Excluídos como posicoes de portfolio.
# Se o regime é bear, esses ativos ficam como cash implícito (capital não alocado).
_CASH_EQUIVALENTS = frozenset({
    "BIL", "SHY", "IEF", "TLT", "AGG", "BND", "GOVT", "SHV",
    "VGIT", "VGLT", "BNDX", "IEI", "TIP",
})

# FX indices (DX, EURUSD etc.) — excluídos como posições diretas de equity
_FX_EXCLUDE = frozenset({
    "DX-Y.NYB", "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCAD=X", "AUDUSD=X",
    "DX=F", "6E=F", "6J=F",
})

_DEFENSIVE = {"bonds", "commodities"}
_RISK_ON   = {"equity", "crypto"}


def _asset_class(ticker: str) -> str:
    return _ASSET_CLASS.get(ticker, "equity")  # default = equity


def _compute_operational_levels(
    ticker: str,
    direction: str,
    entry_price: float,
    conviction: str,
    market_prices: dict,
) -> tuple[float, float, float, float, float]:
    """
    Calcula stop loss, take profit e R:R baseado em ATR ou percentual.

    Returns: (stop_loss, take_profit, stop_pct, target_pct, risk_reward)

    Metodologia:
      - ATR 14d via yfinance (se disponível)
      - Stop: 1.5x ATR (long: abaixo, short: acima)
      - Target: 3.0x ATR → mínimo 2:1 R/R
      - Fallback percentual por convicção:
          high:   stop 5%, target 15% (3:1)
          medium: stop 7%, target 14% (2:1)
          low:    stop 10%, target 15% (1.5:1)
    """
    if entry_price <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Percentuais padrão por convicção
    _stop_pcts = {"high": 0.05, "medium": 0.07, "low": 0.10}
    _target_mult = {"high": 3.0, "medium": 2.0, "low": 1.5}

    stop_pct = _stop_pcts.get(conviction, 0.07)
    rr_mult  = _target_mult.get(conviction, 2.0)

    # Tenta ATR 14d via dados já no market_prices (sem novas chamadas yfinance)
    try:
        prices = market_prices.get(ticker, {})
        hist = prices.get("_hist")  # injetado pelo pipeline se disponível

        # NÃO faz nova chamada yfinance — usa só o que já está em cache
        # (evita rate limit após 200+ chamadas do pipeline)
        if hist is not None and not hist.empty and len(hist) >= 5:
            high = hist["High"].values
            low  = hist["Low"].values
            close = hist["Close"].values
            n = len(high)
            tr = []
            for i in range(1, n):
                tr.append(max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                ))
            if tr:
                atr14 = sum(tr[-14:]) / min(14, len(tr[-14:]))
                atr_pct = atr14 / entry_price
                # Stop = 1.5x ATR, com mínimo de 3% e máximo de 15%
                stop_pct = max(0.03, min(0.15, 1.5 * atr_pct))
    except Exception:
        pass  # usa percentual padrão

    target_pct = stop_pct * rr_mult

    if direction == "long":
        stop_loss   = round(entry_price * (1 - stop_pct), 2)
        take_profit = round(entry_price * (1 + target_pct), 2)
    else:  # short
        stop_loss   = round(entry_price * (1 + stop_pct), 2)
        take_profit = round(entry_price * (1 - target_pct), 2)

    risk_reward = round(target_pct / stop_pct, 2) if stop_pct > 0 else 0.0

    return stop_loss, take_profit, round(stop_pct, 4), round(target_pct, 4), risk_reward


@dataclass
class PositionResult:
    ticker: str
    name: str
    direction: str          # long | short
    conviction: str
    allocation_pct: float   # [-1, 1] fraction of portfolio (negative = short)
    allocation_usd: float
    shares_approx: float | None
    expected_return_ann: float
    risk_score: float
    sharpe_implied: float
    composite: float
    asset_class: str
    cluster_id: str
    rationale: list[str]

    # ── Parâmetros operacionais ─────────────────────────────────────────────
    entry_price: float = 0.0     # preço de entrada (atual)
    stop_loss: float = 0.0       # nível de stop (preço)
    take_profit: float = 0.0     # alvo de lucro (preço)
    stop_pct: float = 0.0        # distância do stop em %
    target_pct: float = 0.0      # distância do target em %
    risk_reward: float = 0.0     # R:R (target_pct / stop_pct)


@dataclass
class PortfolioResult:
    budget: float
    regime_mode: str
    positions: list[PositionResult] = field(default_factory=list)

    # Portfolio-level metrics
    expected_return_ann: float = 0.0
    portfolio_vol: float = 0.0
    sharpe: float = 0.0
    max_drawdown_est: float = 0.0
    diversification_score: float = 0.0

    # Breakdowns
    by_asset_class: dict[str, float] = field(default_factory=dict)
    by_direction: dict[str, float] = field(default_factory=dict)
    by_conviction: dict[str, int] = field(default_factory=dict)

    # Alpha opportunities (highest |composite| neutrals — watchlist)
    watchlist: list[dict] = field(default_factory=list)

    summary_text: str = ""


def _detect_regime(signals: dict[str, AssetSignal]) -> str:
    """
    Detecta regime macro via média ponderada de P(bull) dos ativos de referência.
    """
    ref_tickers = ["SPY", "^GSPC", "QQQ", "^NDX", "IWM"]
    bulls = [
        s.regime_bull for t, s in signals.items()
        if t in ref_tickers and s.regime_bull is not None
    ]
    if not bulls:
        # Fallback: usa momentum médio de equities
        eq_mom = [
            s.momentum_score for s in signals.values()
            if _asset_class(s.ticker) == "equity"
        ]
        avg_mom = sum(eq_mom) / len(eq_mom) if eq_mom else 0.0
        return "bull" if avg_mom > 0.1 else ("bear" if avg_mom < -0.1 else "neutral")

    avg_bull = sum(bulls) / len(bulls)
    if avg_bull > 0.60:
        return "bull"
    elif avg_bull < 0.40:
        return "bear"
    return "neutral"


def _max_sharpe_weights(
    signals: list[AssetSignal],
    max_per_ticker: float,
    max_per_cluster: float,
    allow_short: bool,
) -> list[float]:
    """
    Maximiza Sharpe via scipy.optimize.minimize (SLSQP).
    Fallback: Score-proportional weights se scipy falhar.

    Retorna lista de pesos alinhada com `signals`.
    """
    import numpy as np

    n = len(signals)
    if n == 0:
        return []

    # Para o optimizer, mu é o retorno da POSIÇÃO (não do ativo)
    # Long: mu = expected_asset_return (positive for longs with composite > 0)
    # Short: mu = -expected_asset_return (positive for shorts with composite < 0)
    # Isso permite que o optimizer maximize retorno de posição diretamente com pesos positivos
    position_mu = np.array([
        max(0.0, min(0.8, abs(s.expected_return_ann)))   # sempre positivo (magnitude)
        for s in signals
    ])
    sig = np.array([max(0.05, s.risk_score) for s in signals])

    # Correlação simplificada: mesmos clusters = 0.6, diferentes = 0.2
    corr = np.full((n, n), 0.20)
    np.fill_diagonal(corr, 1.0)
    for i in range(n):
        for j in range(n):
            if i != j and signals[i].cluster_id == signals[j].cluster_id:
                corr[i, j] = 0.60

    # Covariância
    cov = corr * np.outer(sig, sig)
    rf = 0.053
    clusters = list({s.cluster_id for s in signals})

    try:
        from scipy.optimize import minimize

        # Pesos são MAGNITUDES [0, max_per_ticker] — aplicamos sinal depois
        # Objetivo: max Sharpe com leve penalidade de concentração (mais diversificado)
        lam = 0.01   # regularização L2

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(w @ position_mu)
            port_var = float(w @ cov @ w)
            port_vol = math.sqrt(max(port_var, 1e-8))
            reg = lam * float(w @ w)   # penalidade de concentração
            return -(port_ret - rf) / port_vol + reg

        def grad(w: np.ndarray) -> np.ndarray:
            eps = 1e-6
            g = np.zeros(n)
            base = neg_sharpe(w)
            for i in range(n):
                w2 = w.copy()
                w2[i] += eps
                g[i] = (neg_sharpe(w2) - base) / eps
            return g

        # Constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}]

        # Cluster constraints
        for cid in clusters:
            idx = [i for i, s in enumerate(signals) if s.cluster_id == cid]
            if idx:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, ix=idx: max_per_cluster - sum(abs(w[i]) for i in ix)
                })

        # Pesos sempre positivos [0, max_per_ticker] — sinal aplicado depois
        bounds = [(0.0, max_per_ticker)] * n

        # Initial guess: score-proportional com cap, spread para diversificação
        scores = np.array([abs(s.composite) for s in signals])
        scores = np.clip(scores, 1e-4, None)
        scores = scores / scores.sum()
        # Suaviza para forçar diversificação inicial
        uniform = np.ones(n) / n
        w0 = 0.6 * scores + 0.4 * uniform

        res = minimize(
            neg_sharpe, w0,
            method="SLSQP",
            jac=grad,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        if res.success or res.fun < 0:
            w = np.clip(np.array(res.x), 0, max_per_ticker)
            # Apply signs based on direction
            for i, s in enumerate(signals):
                if s.direction == "short":
                    w[i] = -abs(w[i])
                else:
                    w[i] = abs(w[i])
            total = np.sum(np.abs(w))
            if total > 0:
                w = w / total
            return list(w)

    except Exception as exc:
        _log.warning("scipy_failed", error=str(exc))

    # Fallback: composite-proportional
    raw = np.array([abs(s.composite) for s in signals])
    raw = np.clip(raw, 1e-4, None)
    raw = raw / raw.sum()
    # Apply cluster cap
    for cid in clusters:
        idx = [i for i, s in enumerate(signals) if s.cluster_id == cid]
        cluster_sum = sum(raw[i] for i in idx)
        if cluster_sum > max_per_cluster:
            scale = max_per_cluster / cluster_sum
            for i in idx:
                raw[i] *= scale
    raw = raw / raw.sum()
    # Apply per-ticker cap
    raw = np.clip(raw, 0, max_per_ticker)
    raw = raw / raw.sum()

    # Apply short signs
    result = []
    for i, s in enumerate(signals):
        w = float(raw[i])
        result.append(-w if s.direction == "short" else w)

    return result


def optimize_portfolio(
    signals: dict[str, AssetSignal],
    market_prices: dict[str, Any] | None = None,
    budget: float = 100_000.0,
    max_per_ticker: float = 0.15,
    max_per_cluster: float = 0.30,
    allow_short: bool = True,
    max_short_total: float = 0.20,
    min_position: float = 0.02,
    regime_override: str | None = None,
    vol_regime: Any = None,              # VolRegimeResult | None
) -> PortfolioResult:
    """
    Constrói portfolio maximizando Sharpe ajustado por regime.

    Args:
        signals         : output de alpha_signals.compute_signals()
        budget          : capital disponível (padrão: $100,000)
        max_per_ticker  : max alocação por ativo (15%)
        max_per_cluster : max por cluster MST (30%)
        allow_short     : permite posições short
        max_short_total : máximo % short no portfolio (20%)
        min_position    : mínimo % para incluir (2%)
        regime_override : força regime ("bull"|"bear"|"neutral"|None=auto)
    """
    import numpy as np

    market_prices = market_prices or {}

    # ── Detecta regime ────────────────────────────────────────────────────────
    regime = regime_override or _detect_regime(signals)
    _log.info("portfolio_regime", regime=regime)

    # ── Vol regime position scalar ────────────────────────────────────────────
    # Em stress/crisis: reduz alocacoes, pode adicionar VXX como hedge
    vol_position_scalar = 1.0
    vol_regime_note = ""
    if vol_regime:
        vol_position_scalar = vol_regime.position_scalar
        if vol_regime.regime == "stress":
            vol_regime_note = f"Vol STRESS ({vol_regime.vix:.0f}): posicoes reduzidas para {vol_position_scalar:.0%}"
        elif vol_regime.regime == "crisis":
            vol_regime_note = f"Vol CRISE ({vol_regime.vix:.0f}): posicoes reduzidas para {vol_position_scalar:.0%}, hedge obrigatorio"
        elif vol_regime.regime == "calm":
            vol_regime_note = f"Vol calma ({vol_regime.vix:.0f}): posicoes expandidas para {vol_position_scalar:.0%}"
        _log.info("vol_regime_overlay",
                  vol_regime=vol_regime.regime,
                  scalar=vol_position_scalar,
                  vix=vol_regime.vix)

    # ── Aplica overlay de regime ──────────────────────────────────────────────
    # Portfolio é puramente equity (longs + shorts). Cash = residual não alocado.
    # Bear: alavanca os shorts, longs são apenas convicção real (GLD, VXX etc.)
    regime_constraints: dict[str, float] = {}
    if regime == "bear":
        regime_constraints["max_equity_long"] = 0.35   # long equity limitado
        regime_constraints["max_shorts"] = 0.65        # shorts até 65% do capital
    elif regime == "bull":
        regime_constraints["max_equity_long"] = 0.80
        regime_constraints["max_shorts"] = 0.20
    else:  # neutral
        regime_constraints["max_equity_long"] = 0.55
        regime_constraints["max_shorts"] = 0.40

    # ── Seleciona candidatos ──────────────────────────────────────────────────
    # Equity-only: exclui renda fixa (cash) e índices FX puros
    candidates = [
        s for s in signals.values()
        if s.direction != "neutral"
        and s.conviction in ("high", "medium")
        and s.price is not None
        and s.price > 0
        and s.ticker not in _CASH_EQUIVALENTS   # bonds = cash, não posição
        and s.ticker not in _FX_EXCLUDE          # FX puro sem alpha de equity
    ]

    # Ordena por |composite| desc
    candidates.sort(key=lambda s: abs(s.composite), reverse=True)

    # Limite o universe a top 30 (mais que isso dilui demais)
    candidates = candidates[:30]

    # Watchlist = neutrals com |composite| > 0.05
    watchlist = [
        {"ticker": s.ticker, "name": s.name, "composite": s.composite,
         "direction": s.direction, "rationale": s.rationale[:2]}
        for s in sorted(signals.values(), key=lambda x: abs(x.composite), reverse=True)
        if s.direction == "neutral" and abs(s.composite) > 0.05
    ][:10]

    if not candidates:
        _log.warning("no_candidates_for_portfolio")
        return PortfolioResult(
            budget=budget, regime_mode=regime,
            watchlist=watchlist,
            summary_text="Sem candidatos com convicção suficiente."
        )

    # ── Separa longs e shorts ─────────────────────────────────────────────────
    longs  = [s for s in candidates if s.direction == "long"]
    shorts = [s for s in candidates if s.direction == "short"]

    # Em bear: se não há longs de media/alta convicção, injeta GLD/commodities como hedge.
    # Renda fixa (BIL, TLT, SHY, IEF) NÃO é posição — é cash implícito.
    if regime == "bear" and len(longs) < 2:
        real_havens = ["GLD", "SLV", "VXX", "UVXY"]  # ativos reais com alpha, não bonds
        import copy
        for sh_tk in real_havens:
            if sh_tk in signals and sh_tk not in {s.ticker for s in longs}:
                sh = signals[sh_tk]
                if sh.composite > -0.15:
                    sh_copy = copy.copy(sh)
                    sh_copy.direction = "long"
                    sh_copy.conviction = "medium"
                    sh_copy.expected_return_ann = max(0.05, abs(sh_copy.expected_return_ann))
                    longs.append(sh_copy)
                    if len(longs) >= 2:
                        break

    # Max short allocation (em bear: até 65%, outros regimes: 20%)
    if regime == "bear":
        max_short_total = 0.65
    short_budget_pct = min(max_short_total, len(shorts) * max_per_ticker)

    # Limita shorts ao budget e top K
    max_shorts_count = max(2, int(max_short_total / max_per_ticker))
    shorts = shorts[:max_shorts_count]

    # Universo para otimização: longs + shorts selecionados
    universe = longs[:12] + shorts[:18]  # cap: 12 longs + 18 shorts = 30 max

    if not universe:
        return PortfolioResult(
            budget=budget, regime_mode=regime,
            watchlist=watchlist,
            summary_text="Sem candidatos após filtros."
        )

    # ── Otimização ────────────────────────────────────────────────────────────
    raw_weights = _max_sharpe_weights(
        universe, max_per_ticker, max_per_cluster,
        allow_short=(allow_short and len(shorts) > 0)
    )

    # ── Aplica regime constraints ─────────────────────────────────────────────
    weights = list(raw_weights)
    max_eq_long = regime_constraints.get("max_equity_long", 1.0)
    max_shorts  = regime_constraints.get("max_shorts", 0.65)

    # Escala long equity se necessário
    long_eq_idx = [i for i, s in enumerate(universe)
                   if s.direction == "long" and _asset_class(s.ticker) in ("equity", "intl_equity")]
    long_eq_total = sum(weights[i] for i in long_eq_idx if weights[i] > 0)
    if long_eq_total > max_eq_long and long_eq_total > 0:
        scale = max_eq_long / long_eq_total
        for i in long_eq_idx:
            weights[i] *= scale

    # Escala shorts se necessário
    short_idx = [i for i, s in enumerate(universe) if s.direction == "short"]
    short_total = sum(abs(weights[i]) for i in short_idx)
    if short_total > max_shorts and short_total > 0:
        scale = max_shorts / short_total
        for i in short_idx:
            weights[i] *= scale

    # ── Remove posições abaixo do mínimo ──────────────────────────────────────
    weights_arr = np.array(weights)
    mask = np.abs(weights_arr) >= min_position
    weights_arr[~mask] = 0.0

    # Re-normaliza longs para soma <= 1 - short_exposure
    long_idx  = [i for i, s in enumerate(universe) if s.direction == "long"]
    short_idx = [i for i, s in enumerate(universe) if s.direction == "short"]

    short_exposure = sum(abs(weights_arr[i]) for i in short_idx)
    long_budget    = 1.0 - short_exposure

    long_sum = sum(weights_arr[i] for i in long_idx)
    if long_sum > 0 and long_sum > long_budget:
        scale = long_budget / long_sum
        for i in long_idx:
            weights_arr[i] *= scale

    total = float(np.sum(np.abs(weights_arr)))
    if total > 0:
        weights_arr = weights_arr / total

    # ── Constrói posições ─────────────────────────────────────────────────────
    positions: list[PositionResult] = []
    for i, sig in enumerate(universe):
        w = float(weights_arr[i])
        if abs(w) < min_position / 2:
            continue

        usd = w * budget
        price = (market_prices.get(sig.ticker) or {}).get("price") or sig.price

        # Retorno da POSIÇÃO: short inverte o sinal do ativo
        position_return = (
            sig.expected_return_ann if sig.direction == "long"
            else -sig.expected_return_ann   # short de ativo em queda = ganho
        )
        position_sharpe = (
            (abs(position_return) - 0.053) / sig.risk_score
            if sig.risk_score > 0 else 0.0
        )

        # ── Parâmetros operacionais: stop, target, R:R ──────────────────────
        stop_loss, take_profit, stop_pct, target_pct, rr = _compute_operational_levels(
            ticker=sig.ticker,
            direction=sig.direction,
            entry_price=price or 0.0,
            conviction=sig.conviction,
            market_prices=market_prices,
        )

        pos = PositionResult(
            ticker              = sig.ticker,
            name                = sig.name,
            direction           = sig.direction,
            conviction          = sig.conviction,
            allocation_pct      = round(w, 4),
            allocation_usd      = round(usd, 2),
            shares_approx       = round(abs(usd) / price, 2) if price else None,
            expected_return_ann = round(position_return, 4),
            risk_score          = sig.risk_score,
            sharpe_implied      = round(position_sharpe, 3),
            composite           = sig.composite,
            asset_class         = _asset_class(sig.ticker),
            cluster_id          = sig.cluster_id,
            rationale           = sig.rationale,
            entry_price         = round(price or 0.0, 2),
            stop_loss           = stop_loss,
            take_profit         = take_profit,
            stop_pct            = stop_pct,
            target_pct          = target_pct,
            risk_reward         = rr,
        )
        positions.append(pos)

    positions.sort(key=lambda p: abs(p.allocation_pct), reverse=True)

    # ── Vol regime overlay: scale positions + inject VXX hedge ────────────────
    if vol_regime and vol_position_scalar != 1.0:
        for p in positions:
            p.allocation_pct = round(p.allocation_pct * vol_position_scalar, 4)
            p.allocation_usd = round(p.allocation_usd * vol_position_scalar, 2)
            if p.shares_approx:
                p.shares_approx = round(p.shares_approx * vol_position_scalar, 2)

    # Injeta VXX em stress/crisis como hedge de vol (similar ao safe-haven em bear)
    if vol_regime and vol_regime.hedge_required:
        vxx_ticker = vol_regime.hedge_asset  # "VXX"
        vxx_exists = any(p.ticker == vxx_ticker for p in positions)
        if not vxx_exists:
            # Aloca 5% em VXX long (ETF de vol sobe quando mercado cai)
            vxx_alloc = 0.05 * vol_position_scalar
            vxx_price_data = market_prices.get(vxx_ticker) or market_prices.get("VXX") or {}
            vxx_price = vxx_price_data.get("price") if isinstance(vxx_price_data, dict) else None
            if not vxx_price or vxx_price <= 0:
                # Fetch live price
                try:
                    import yfinance as yf
                    vxx_price = float(yf.Ticker("VXX").fast_info["lastPrice"])
                except Exception:
                    vxx_price = 25.0
            vxx_conv = "high" if vol_regime.regime == "crisis" else "medium"
            vxx_usd = vxx_alloc * budget
            vxx_stop, vxx_target, vxx_stop_pct, vxx_tgt_pct, vxx_rr = _compute_operational_levels(
                ticker="VXX", direction="long", entry_price=vxx_price,
                conviction=vxx_conv, market_prices=market_prices,
            )
            vxx_pos = PositionResult(
                ticker="VXX",
                name="VIX Short-Term Futures ETN",
                direction="long",
                conviction=vxx_conv,
                allocation_pct=round(vxx_alloc, 4),
                allocation_usd=round(vxx_usd, 2),
                shares_approx=round(vxx_usd / vxx_price, 2),
                expected_return_ann=0.30,
                risk_score=0.60,
                sharpe_implied=0.40,
                composite=0.35,
                asset_class="volatility",
                cluster_id="VOL",
                rationale=[
                    f"Vol hedge: regime={vol_regime.regime}, stress={vol_regime.stress_score:.2f}",
                    f"VIX={vol_regime.vix:.0f}" + (f", VIX/VIX3M={vol_regime.term_structure_ratio:.2f}" if vol_regime.term_structure_ratio else ""),
                ],
                entry_price=round(vxx_price, 2),
                stop_loss=vxx_stop,
                take_profit=vxx_target,
                stop_pct=vxx_stop_pct,
                target_pct=vxx_tgt_pct,
                risk_reward=vxx_rr,
            )
            positions.append(vxx_pos)
            positions.sort(key=lambda p: abs(p.allocation_pct), reverse=True)
            _log.info("vxx_hedge_injected", alloc=vxx_alloc, vol_regime=vol_regime.regime)

    # ── Portfolio metrics ─────────────────────────────────────────────────────
    total_long  = sum(p.allocation_pct for p in positions if p.allocation_pct > 0)
    total_short = sum(abs(p.allocation_pct) for p in positions if p.allocation_pct < 0)

    # expected_return_ann já é o retorno da POSIÇÃO (corrigido para shorts)
    # Ponderado pelo peso absoluto
    exp_ret = sum(abs(p.allocation_pct) * p.expected_return_ann for p in positions)

    # Portfolio vol (simplified: sum of weighted individual vols × avg correlation)
    vols = np.array([p.risk_score * abs(p.allocation_pct) for p in positions])
    avg_corr = 0.35  # conservative estimate
    port_var = float(np.sum(vols**2)) + avg_corr * float(np.sum(vols))**2 - float(np.sum(vols**2))
    port_vol = math.sqrt(max(port_var, 0.001))

    rf = 0.053
    sharpe = (exp_ret - rf) / port_vol if port_vol > 0 else 0.0

    # Max drawdown estimate: 2 × port_vol (rough 95% CI daily → annual)
    mdd_est = min(0.99, 2.5 * port_vol)

    # By asset class
    by_class: dict[str, float] = {}
    for p in positions:
        ac = p.asset_class
        by_class[ac] = by_class.get(ac, 0.0) + abs(p.allocation_pct)

    # Diversification score: 1 - HHI of weights
    weights_abs = np.array([abs(p.allocation_pct) for p in positions])
    hhi = float(np.sum(weights_abs**2)) if len(weights_abs) > 0 else 1.0
    div_score = 1.0 - hhi

    # Summary text
    top3 = [p.ticker for p in positions[:3]]
    direction_counts = {
        "long": sum(1 for p in positions if p.direction == "long"),
        "short": sum(1 for p in positions if p.direction == "short"),
    }
    vol_note = f" | {vol_regime_note}" if vol_regime_note else ""
    summary = (
        f"Regime {regime.upper()} | "
        f"{len(positions)} posicoes ({direction_counts['long']}L/{direction_counts['short']}S) | "
        f"E[R]={exp_ret:+.1%} | Vol={port_vol:.1%} | Sharpe={sharpe:.2f} | "
        f"Top: {', '.join(top3)}"
        f"{vol_note}"
    )

    _log.info("portfolio_done",
              n_positions=len(positions),
              exp_return=round(exp_ret, 4),
              sharpe=round(sharpe, 3),
              regime=regime)

    return PortfolioResult(
        budget             = budget,
        regime_mode        = regime,
        positions          = positions,
        expected_return_ann = round(exp_ret, 4),
        portfolio_vol      = round(port_vol, 4),
        sharpe             = round(sharpe, 3),
        max_drawdown_est   = round(mdd_est, 4),
        diversification_score = round(div_score, 4),
        by_asset_class     = {k: round(v, 4) for k, v in by_class.items()},
        by_direction       = {
            "long_pct":  round(total_long, 4),
            "short_pct": round(total_short, 4),
            "cash_pct":  round(max(0, 1.0 - total_long - total_short), 4),
        },
        by_conviction      = {
            "high":   sum(1 for p in positions if p.conviction == "high"),
            "medium": sum(1 for p in positions if p.conviction == "medium"),
            "low":    sum(1 for p in positions if p.conviction == "low"),
        },
        watchlist   = watchlist,
        summary_text = summary,
    )
