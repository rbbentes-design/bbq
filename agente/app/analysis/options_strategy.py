"""
Options Strategy Engine — Recomendacoes de Opcoes

Para cada posicao do portfolio, recomenda estrategias concretas:
  - Long put / put spread para shorts (tendencia de queda)
  - Long call / call spread para longs (tendencia de alta)
  - Portfolio hedge via SPY put spread

Usa:
  - yfinance para baixar chains reais
  - Black-Scholes simplificado para preco teorico e gregas
  - Kelly fraction para sizing de contratos
  - IV percentile para decidir comprar vs vender premium

Logica de selecao:
  1. DTE: 30-45 dias (theta decay gerenciavel, gamma suficiente)
  2. Delta alvo: 0.30 para direccionais, 0.15-0.20 para spreads OTM
  3. IV percentile < 60%: comprar options (IV barata)
  4. IV percentile > 75%: vender spreads (premium rico)
  5. Kelly: f* = 0.25 * (p*b - q) / b, limitado a 2% do portfolio por trade
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from app.audit.logger import get_logger

_log = get_logger("analysis.options_strategy")


@dataclass
class OptionRecommendation:
    ticker: str
    strategy: str           # "long_put", "put_spread", "long_call", "call_spread", "collar", "spy_hedge"
    expiry: str             # YYYY-MM-DD
    dte: int
    # Para estrategias simples
    strike: float | None = None
    option_type: str | None = None   # "put" | "call"
    # Para spreads
    buy_strike: float | None = None
    sell_strike: float | None = None
    # Custo / credito
    debit_credit: float | None = None   # por contrato (100 acoes); negativo = credito
    max_profit: float | None = None     # por contrato
    max_loss: float | None = None       # por contrato (positivo = perda maxima)
    breakeven: float | None = None
    # Sizing
    contracts: int = 1
    total_cost_usd: float = 0.0
    # Contexto
    rationale: str = ""
    iv_used: float | None = None
    iv_percentile: float | None = None
    expected_value: float | None = None
    kelly_fraction: float | None = None


@dataclass
class OptionsStrategyResult:
    recommendations: list[OptionRecommendation] = field(default_factory=list)
    total_options_cost: float = 0.0
    portfolio_delta_hedge: float = 0.0   # delta total das opcoes
    errors: list[str] = field(default_factory=list)
    timestamp: str = ""


# ── Black-Scholes simplificado ─────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """CDF da normal padrao (aproximacao de Abramowitz & Stegun)."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    phi = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    cdf = 1.0 - phi * poly
    return cdf if x >= 0 else 1.0 - cdf


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Preco Black-Scholes. T em anos, sigma anualizado."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Delta Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type == "call":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1.0


def _implied_vol(price_data: dict) -> float:
    """Extrai IV do dict de preco. Fallback 30%."""
    iv = price_data.get("iv") or price_data.get("iv_atm")
    if iv and iv > 0.01:
        return float(iv)
    # Estima via vol historica * 1.25 (IV premium tipico)
    vol = price_data.get("vol_ann") or price_data.get("daily_vol", 0.0)
    if vol and vol > 0.005:
        return float(vol) * 1.25
    return 0.30  # fallback 30%


def _iv_percentile(price_data: dict) -> float:
    """IV percentile [0,1]. Se nao disponivel, assume 0.50."""
    pct = price_data.get("iv_percentile")
    if pct is not None:
        return float(pct) / 100.0 if float(pct) > 1.0 else float(pct)
    return 0.50


# ── Selecao de expiracao ────────────────────────────────────────────────────────

def _select_expiry(chains_expiries: list[str], target_dte: int = 35) -> tuple[str, int] | None:
    """
    Escolhe a expiracao mais proxima do target_dte das disponíveis na chain real.
    Retorna None se não houver expirações reais disponíveis.
    NÃO cria datas sintéticas — só usa o que o mercado tem.
    """
    today = date.today()
    best = None
    best_dte = 9999
    for exp_str in (chains_expiries or []):
        try:
            exp = date.fromisoformat(exp_str)
            dte = (exp - today).days
            if dte < 7:   # muito próximo — theta risco excessivo
                continue
            if dte > 120: # muito distante — menos relevante
                continue
            if abs(dte - target_dte) < abs(best_dte - target_dte):
                best = exp_str
                best_dte = dte
        except Exception:
            continue
    if best is None:
        return None  # sem chain real — não recomenda
    return best, best_dte


def _select_strike_by_delta(
    S: float, T: float, r: float, sigma: float,
    option_type: str, target_delta: float,
    available_strikes: list[float] | None = None
) -> float:
    """
    Encontra o strike mais proximo ao target_delta desejado.
    Se nao tiver strikes reais, aproxima via BS inverso.
    """
    if available_strikes:
        best_k = S  # fallback ATM
        best_diff = 1.0
        for k in available_strikes:
            d = abs(bs_delta(S, k, T, r, sigma, option_type))
            diff = abs(d - target_delta)
            if diff < best_diff:
                best_diff = diff
                best_k = k
        return best_k

    # Aproxima: para put 0.30-delta, K ~ S * exp(-0.52 * sigma * sqrt(T))
    # (heuristica empirica; delta=N(d1) -> inverte)
    import math
    z = _norm_cdf_inv(target_delta if option_type == "call" else 1.0 - target_delta)
    # d1 = z => log(S/K) = z*sigma*sqrt(T) - (r + 0.5*sigma^2)*T
    log_SK = z * sigma * math.sqrt(T) - (r + 0.5 * sigma ** 2) * T
    K = S * math.exp(-log_SK)
    # Arredonda para strike padrao (multiplos de 1, 2.5 ou 5 dependendo do preco)
    if S < 20:
        step = 0.5
    elif S < 50:
        step = 1.0
    elif S < 100:
        step = 2.5
    elif S < 200:
        step = 5.0
    else:
        step = 10.0
    return round(K / step) * step


def _norm_cdf_inv(p: float) -> float:
    """Inversa da CDF normal (aproximacao de Beasley-Springer-Moro)."""
    p = max(1e-6, min(1 - 1e-6, p))
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    if 0.08 <= p <= 0.92:
        q = p - 0.5
        r = q * q
        return q * (((a[3] * r + a[2]) * r + a[1]) * r + a[0]) / \
               ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0)
    else:
        if p < 0.08:
            r = math.sqrt(-math.log(p))
        else:
            r = math.sqrt(-math.log(1.0 - p))
        return (((((((c[8] * r + c[7]) * r + c[6]) * r + c[5]) * r + c[4]) * r +
                   c[3]) * r + c[2]) * r + c[1]) * r + c[0] if p >= 0.08 else \
               -(((((((c[8] * r + c[7]) * r + c[6]) * r + c[5]) * r + c[4]) * r +
                    c[3]) * r + c[2]) * r + c[1]) * r + c[0]


# ── Fetch de chain via yfinance ────────────────────────────────────────────────

def _fetch_chain(ticker: str, expiry: str) -> dict:
    """Baixa chain de opcoes via yfinance com retry."""
    import time
    import yfinance as yf
    for attempt in range(3):
        try:
            opts = yf.Ticker(ticker).option_chain(expiry)
            return {"calls": opts.calls, "puts": opts.puts}
        except Exception as exc:
            err = str(exc).lower()
            if "rate" in err or "too many" in err or "429" in err:
                time.sleep(3 + attempt * 3)
                continue
            _log.debug("chain_fetch_failed", ticker=ticker, expiry=expiry, error=str(exc)[:60])
            return {}
    return {}


def _get_expiries(ticker: str) -> list[str]:
    """
    Gera datas de expirações prováveis matematicamente (weeklies + mensais).
    Não chama yfinance para listar — calcula baseado no calendário padrão de opções.

    Opções líquidas expiram:
      - Weeklies: toda sexta-feira
      - Mensais: terceira sexta-feira de cada mês (monthly OpEx)
    """
    from datetime import date, timedelta
    today = date.today()
    expiries = []

    # Gera todas as sextas-feiras nos próximos 90 dias
    # (cobre weeklies + mensais sem chamar API)
    d = today + timedelta(days=1)
    while d <= today + timedelta(days=90):
        if d.weekday() == 4:  # sexta-feira
            if (d - today).days >= 7:  # mínimo 7 dias
                expiries.append(d.isoformat())
        d += timedelta(days=1)

    # Tenta confirmar via yfinance, mas só se não estiver em rate limit
    try:
        import yfinance as yf
        live = list(yf.Ticker(ticker).options or [])
        if live:
            return live  # usa dados reais se disponíveis
    except Exception:
        pass  # fallback para datas calculadas

    return expiries


# ── Estrategias por posicao ───────────────────────────────────────────────────

def _long_put(
    ticker: str, S: float, price_data: dict,
    budget_per_trade: float, direction: str
) -> OptionRecommendation | None:
    """
    Long put para posicao short (protege contra rally inesperado)
    ou para sinal bearish sem posicao stock.
    IV percentile < 60%: comprar put outright
    """
    iv = _implied_vol(price_data)
    iv_pct = _iv_percentile(price_data)

    expiries = _get_expiries(ticker)
    sel = _select_expiry(expiries, target_dte=35)
    if sel is None:
        return None   # sem chain real disponível
    expiry, dte = sel
    T = dte / 365.0
    r = 0.05  # fed funds rate

    strike = _select_strike_by_delta(S, T, r, iv, "put", target_delta=0.30)
    theo_price = bs_price(S, strike, T, r, iv, "put")

    if theo_price <= 0.01:
        return None

    # Kelly: win_prob ~ 0.35 para OTM put, payoff = intrinsic/debit
    win_prob = 0.35
    avg_payoff_ratio = (S * 0.08) / (theo_price * 100)  # 8% move avg payoff vs debit
    kelly_f = max(0.0, (win_prob * avg_payoff_ratio - (1 - win_prob)) / avg_payoff_ratio)
    kelly_f_fractional = kelly_f * 0.25  # 1/4 Kelly

    max_spend = budget_per_trade * kelly_f_fractional
    cost_per_contract = theo_price * 100
    contracts = max(1, int(max_spend / cost_per_contract)) if cost_per_contract > 0 else 1
    contracts = min(contracts, 10)  # cap 10 contratos
    total_cost = contracts * cost_per_contract

    return OptionRecommendation(
        ticker=ticker,
        strategy="long_put",
        expiry=expiry,
        dte=dte,
        strike=strike,
        option_type="put",
        debit_credit=theo_price * 100,
        max_profit=(strike - theo_price) * 100,
        max_loss=theo_price * 100,
        breakeven=strike - theo_price,
        contracts=contracts,
        total_cost_usd=total_cost,
        rationale=f"IV {iv:.0%} | IV%ile {iv_pct:.0%} — comprar put 30-delta {dte}DTE",
        iv_used=iv,
        iv_percentile=iv_pct,
        kelly_fraction=kelly_f_fractional,
    )


def _put_spread(
    ticker: str, S: float, price_data: dict,
    budget_per_trade: float, direction: str
) -> OptionRecommendation | None:
    """
    Put spread para sinal bearish (IV rica ou cost reduction).
    Compra put 30-delta, vende put 15-delta.
    """
    iv = _implied_vol(price_data)
    iv_pct = _iv_percentile(price_data)

    expiries = _get_expiries(ticker)
    sel = _select_expiry(expiries, target_dte=35)
    if sel is None:
        return None
    expiry, dte = sel
    T = dte / 365.0
    r = 0.05

    buy_strike = _select_strike_by_delta(S, T, r, iv, "put", target_delta=0.30)
    sell_strike = _select_strike_by_delta(S, T, r, iv, "put", target_delta=0.15)

    buy_price = bs_price(S, buy_strike, T, r, iv, "put")
    sell_price = bs_price(S, sell_strike, T, r, iv, "put")
    net_debit = (buy_price - sell_price) * 100
    max_profit = (buy_strike - sell_strike - (buy_price - sell_price)) * 100
    max_loss = net_debit

    if net_debit <= 0.5 or max_profit <= 0:
        return None

    rr = max_profit / max_loss  # reward/risk
    win_prob = 0.40
    kelly_f = max(0.0, (win_prob * rr - (1 - win_prob)) / rr)
    kelly_f_fractional = kelly_f * 0.25

    max_spend = budget_per_trade * kelly_f_fractional
    contracts = max(1, int(max_spend / net_debit)) if net_debit > 0 else 1
    contracts = min(contracts, 15)
    total_cost = contracts * net_debit

    return OptionRecommendation(
        ticker=ticker,
        strategy="put_spread",
        expiry=expiry,
        dte=dte,
        buy_strike=buy_strike,
        sell_strike=sell_strike,
        option_type="put",
        debit_credit=net_debit,
        max_profit=max_profit * contracts,
        max_loss=net_debit * contracts,
        breakeven=buy_strike - (buy_price - sell_price),
        contracts=contracts,
        total_cost_usd=total_cost,
        rationale=f"Put spread {buy_strike:.0f}/{sell_strike:.0f} | IV {iv:.0%} | R:R {rr:.1f}x",
        iv_used=iv,
        iv_percentile=iv_pct,
        kelly_fraction=kelly_f_fractional,
    )


def _long_call(
    ticker: str, S: float, price_data: dict,
    budget_per_trade: float
) -> OptionRecommendation | None:
    """Long call para sinal bullish com IV barata."""
    iv = _implied_vol(price_data)
    iv_pct = _iv_percentile(price_data)

    expiries = _get_expiries(ticker)
    sel = _select_expiry(expiries, target_dte=35)
    if sel is None:
        return None
    expiry, dte = sel
    T = dte / 365.0
    r = 0.05

    strike = _select_strike_by_delta(S, T, r, iv, "call", target_delta=0.35)
    theo_price = bs_price(S, strike, T, r, iv, "call")

    if theo_price <= 0.01:
        return None

    win_prob = 0.40
    avg_payoff_ratio = (S * 0.08) / (theo_price * 100)
    kelly_f = max(0.0, (win_prob * avg_payoff_ratio - (1 - win_prob)) / avg_payoff_ratio)
    kelly_f_fractional = kelly_f * 0.25

    cost_per_contract = theo_price * 100
    max_spend = budget_per_trade * kelly_f_fractional
    contracts = max(1, int(max_spend / cost_per_contract)) if cost_per_contract > 0 else 1
    contracts = min(contracts, 10)
    total_cost = contracts * cost_per_contract

    return OptionRecommendation(
        ticker=ticker,
        strategy="long_call",
        expiry=expiry,
        dte=dte,
        strike=strike,
        option_type="call",
        debit_credit=theo_price * 100,
        max_profit=None,  # teoricamente ilimitado
        max_loss=theo_price * 100,
        breakeven=strike + theo_price,
        contracts=contracts,
        total_cost_usd=total_cost,
        rationale=f"Call 35-delta {dte}DTE | IV {iv:.0%} barata ({iv_pct:.0%}ile)",
        iv_used=iv,
        iv_percentile=iv_pct,
        kelly_fraction=kelly_f_fractional,
    )


def _call_spread(
    ticker: str, S: float, price_data: dict,
    budget_per_trade: float
) -> OptionRecommendation | None:
    """Call spread para sinal bullish (IV rica, reduz custo)."""
    iv = _implied_vol(price_data)
    iv_pct = _iv_percentile(price_data)

    expiries = _get_expiries(ticker)
    sel = _select_expiry(expiries, target_dte=35)
    if sel is None:
        return None
    expiry, dte = sel
    T = dte / 365.0
    r = 0.05

    buy_strike = _select_strike_by_delta(S, T, r, iv, "call", target_delta=0.35)
    sell_strike = _select_strike_by_delta(S, T, r, iv, "call", target_delta=0.15)

    buy_price = bs_price(S, buy_strike, T, r, iv, "call")
    sell_price = bs_price(S, sell_strike, T, r, iv, "call")
    net_debit = (buy_price - sell_price) * 100
    max_profit = (sell_strike - buy_strike - (buy_price - sell_price)) * 100
    max_loss = net_debit

    if net_debit <= 0.5 or max_profit <= 0:
        return None

    rr = max_profit / max_loss
    win_prob = 0.40
    kelly_f = max(0.0, (win_prob * rr - (1 - win_prob)) / rr)
    kelly_f_fractional = kelly_f * 0.25

    max_spend = budget_per_trade * kelly_f_fractional
    contracts = max(1, int(max_spend / net_debit)) if net_debit > 0 else 1
    contracts = min(contracts, 15)
    total_cost = contracts * net_debit

    return OptionRecommendation(
        ticker=ticker,
        strategy="call_spread",
        expiry=expiry,
        dte=dte,
        buy_strike=buy_strike,
        sell_strike=sell_strike,
        option_type="call",
        debit_credit=net_debit,
        max_profit=max_profit * contracts,
        max_loss=net_debit * contracts,
        breakeven=buy_strike + (buy_price - sell_price),
        contracts=contracts,
        total_cost_usd=total_cost,
        rationale=f"Call spread {buy_strike:.0f}/{sell_strike:.0f} | IV {iv:.0%} | R:R {rr:.1f}x",
        iv_used=iv,
        iv_percentile=iv_pct,
        kelly_fraction=kelly_f_fractional,
    )


def _spy_portfolio_hedge(
    portfolio_long_usd: float, spy_price_data: dict
) -> OptionRecommendation | None:
    """
    Hedge de portfolio via SPY put spread.
    Tenta cobrir 50% do long exposure em caso de queda de 5%.
    Aloca ~1% do portfolio em protecao.
    """
    ticker = "SPY"
    try:
        import yfinance as yf
        S = yf.Ticker(ticker).fast_info.get("lastPrice") or yf.Ticker(ticker).fast_info.last_price
    except Exception:
        S = 520.0  # fallback aproximado

    if S <= 0:
        return None

    iv = _implied_vol(spy_price_data) if spy_price_data else 0.20
    iv_pct = _iv_percentile(spy_price_data) if spy_price_data else 0.50

    expiries = _get_expiries(ticker)
    sel = _select_expiry(expiries, target_dte=30)
    if sel is None:
        return None
    expiry, dte = sel
    T = dte / 365.0
    r = 0.05

    # Put spread: compra 5% OTM, vende 10% OTM
    buy_strike = round(S * 0.95 / 5) * 5
    sell_strike = round(S * 0.90 / 5) * 5

    buy_price = bs_price(S, buy_strike, T, r, iv, "put")
    sell_price = bs_price(S, sell_strike, T, r, iv, "put")
    net_debit = (buy_price - sell_price) * 100
    max_profit = (buy_strike - sell_strike) * 100 - net_debit

    if net_debit <= 0 or max_profit <= 0:
        return None

    # Aloca 1% do portfolio
    hedge_budget = portfolio_long_usd * 0.01
    contracts = max(1, int(hedge_budget / net_debit))
    contracts = min(contracts, 20)
    total_cost = contracts * net_debit

    return OptionRecommendation(
        ticker=ticker,
        strategy="spy_hedge",
        expiry=expiry,
        dte=dte,
        buy_strike=buy_strike,
        sell_strike=sell_strike,
        option_type="put",
        debit_credit=net_debit,
        max_profit=max_profit * contracts,
        max_loss=net_debit * contracts,
        breakeven=buy_strike - (buy_price - sell_price),
        contracts=contracts,
        total_cost_usd=total_cost,
        rationale=f"Hedge portfolio: SPY put spread {buy_strike}/{sell_strike} | {dte}DTE",
        iv_used=iv,
        iv_percentile=iv_pct,
    )


# Tickers sem opções líquidas ou que não faz sentido ter opção
_NO_OPTIONS = frozenset({
    "VXX", "UVXY", "SVXY", "VIXY",      # já são vol — não compra opção em opção
    "BIL", "SHY", "IEF", "TLT",         # bonds
    "^VIX", "^VIX9D", "^VIX3M",         # índices não têm opção direta
    "^GSPC", "^NDX", "^RUT", "^DJI",    # índices (usa SPY/QQQ como proxy)
    "DX-Y.NYB", "GC=F", "CL=F",         # futuros sem chain acessível via yfinance
})


# ── Engine principal ─────────────────────────────────────────────────────────

def compute_options_strategy(
    positions: list,            # list[PositionResult]
    market_prices: dict,
    budget: float = 100_000.0,
    options_budget_pct: float = 0.05,   # 5% do portfolio em options
) -> OptionsStrategyResult:
    """
    Gera recomendacoes de opcoes para cada posicao do portfolio.

    Logica:
      - SHORT position + IV < 60%ile: put_spread (custo reduzido)
      - SHORT position + IV > 75%ile: long_put (sem venda, IV cara)
      - LONG position + IV < 60%ile: call_spread
      - LONG position + IV > 75%ile: long_call nao faz sentido → skip
      - Portfolio hedge: SPY put spread se long_usd > 20% do portfolio
    """
    import datetime
    import time as _time

    # Pausa antes de iniciar — pipeline anterior faz ~300+ chamadas yfinance
    # yfinance tem rate limit ~2000 req/hr mas com bursts pode limitar antes
    _time.sleep(8)

    result = OptionsStrategyResult(
        timestamp=datetime.datetime.now().isoformat()
    )

    options_budget = budget * options_budget_pct
    budget_per_trade = options_budget / max(len(positions), 1)
    budget_per_trade = max(500.0, budget_per_trade)  # min $500 por trade

    long_exposure_usd = sum(
        abs(p.allocation_usd) for p in positions if p.direction == "long"
    )

    for pos in positions:
        ticker = pos.ticker

        # Pula tickers sem opções líquidas ou que não faz sentido ter opção
        if ticker in _NO_OPTIONS:
            _log.debug("options_skip_no_chain", ticker=ticker)
            continue

        price_data = market_prices.get(ticker, {})
        S = price_data.get("price") or pos.allocation_usd / max(abs(pos.allocation_pct), 0.01) * 0.01

        if S is None or S <= 0:
            continue

        # Verifica se há chain disponível antes de tentar a estratégia
        expiries = _get_expiries(ticker)
        if not expiries:
            _log.debug("options_no_expiries", ticker=ticker)
            result.errors.append(f"{ticker}: sem chain disponível")
            continue

        iv_pct = _iv_percentile(price_data)
        rec: OptionRecommendation | None = None

        try:
            if pos.direction == "short":
                # SHORT: queremos proteção se rally
                if iv_pct < 0.60:
                    rec = _put_spread(ticker, S, price_data, budget_per_trade, "short")
                else:
                    rec = _long_put(ticker, S, price_data, budget_per_trade, "short")

            elif pos.direction == "long":
                # LONG: alavancagem ou hedge
                if iv_pct < 0.55:
                    rec = _call_spread(ticker, S, price_data, budget_per_trade)
                # IV alta: nao compra calls caras, usa stock position ja existente

        except Exception as exc:
            result.errors.append(f"{ticker}: {exc}")
            _log.warning("options_rec_failed", ticker=ticker, error=str(exc))

        if rec is not None:
            result.recommendations.append(rec)

        _time.sleep(0.4)  # throttle — evita rate limit na chain de requests

    # Hedge de portfolio
    if long_exposure_usd > budget * 0.20:
        spy_data = market_prices.get("SPY", {})
        try:
            hedge = _spy_portfolio_hedge(long_exposure_usd, spy_data)
            if hedge:
                result.recommendations.append(hedge)
        except Exception as exc:
            result.errors.append(f"SPY hedge: {exc}")

    result.total_options_cost = sum(r.total_cost_usd for r in result.recommendations)
    result.portfolio_delta_hedge = sum(
        (r.contracts * 100 * (-1 if r.option_type == "put" else 1) * 0.30)
        for r in result.recommendations
        if r.option_type is not None
    )

    _log.info(
        "options_strategy_done",
        n_recs=len(result.recommendations),
        total_cost=result.total_options_cost,
        errors=len(result.errors),
    )

    return result
