"""
MARKET MAKER DASHBOARD — Versão Unificada v1.0
Consolidação completa de todas as análises anteriores de gregas, risco e posicionamento.

Organizado em seções:
  0. Imports e Configuração
  1. Motor de Gregas (Black-Scholes)
  2. Pipeline de Dados (BQL)
  3. Cálculos de Exposição e Curvas Modelo
  4. Modelos de Risco (VaR, Monte Carlo, P&L)
  5. Rebalanceamento de ETFs
  6. Matrizes de Sensibilidade
  7. Visualização (Gauges, Gráficos, Tabelas)
  8. Callback Principal e Montagem do Dashboard
  9. Interface de Widgets

Uso: Copie tudo para uma célula Jupyter no BQuant e execute.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 0 — IMPORTS E CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

import ipywidgets as wd
from IPython.display import display, clear_output, HTML
import pandas as pd
import numpy as np
from datetime import datetime
import math
import traceback
from scipy.stats import norm, t as student_t
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import bql
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

bq = bql.Service()
BQL_PARAMS = {'fill': 'prev'}
TRADING_DAYS = 252
FUTURES_TICKER = 'ES1 Index'
FUTURES_MULTIPLIER = 50

# Configuração dos 7 greeks para gráficos de exposição.
# Cada entrada define: nome, chave no dict de gregas, unidade de exibição,
# função de escala (converte greek*OI*100 → dólares), divisor de exibição,
# e operação para combinar call/put no total líquido.
GREEK_CONFIGS = [
    {'name': 'Delta',  'key': 'delta',  'unit': '$ Mn',            'scale': lambda L: L,              'div': 1e6, 'op': np.add},
    {'name': 'Gamma',  'key': 'gamma',  'unit': '$ Mn / 1% move',  'scale': lambda L: (L**2) * 0.01,  'div': 1e6, 'op': np.subtract},
    {'name': 'Vega',   'key': 'vega',   'unit': '$ Mn / 1% vol',   'scale': lambda L: 1,              'div': 1e6, 'op': np.add},
    {'name': 'Vanna',  'key': 'vanna',  'unit': '$ Mn',            'scale': lambda L: 1,              'div': 1e6, 'op': np.subtract},
    {'name': 'Theta',  'key': 'theta',  'unit': '$ Mn / dia',      'scale': lambda L: 1.0/TRADING_DAYS, 'div': 1e6, 'op': np.add},
    {'name': 'Zomma',  'key': 'zomma',  'unit': '$ Mn',            'scale': lambda L: 1,              'div': 1e6, 'op': np.subtract},
    {'name': 'Speed',  'key': 'speed',  'unit': '$ Mn',            'scale': lambda L: 1,              'div': 1e6, 'op': np.subtract},
]


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — MOTOR DE GREGAS (BLACK-SCHOLES)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_all_greeks(S, K, vol, T, option_types, r=0.0):
    """
    Calcula todas as gregas de Black-Scholes de forma vetorial.

    Args:
        S: Preço spot do ativo subjacente (escalar).
        K: Array de strikes.
        vol: Array de volatilidade implícita (decimal, ex: 0.20 para 20%).
        T: Array de tempo até expiração em anos.
        option_types: Array de strings 'Call' ou 'Put'.
        r: Taxa livre de risco (default 0).

    Returns:
        Dict com arrays: delta, gamma, vega, vanna, zomma, theta, charm, speed.
        Cada valor é o greek por opção (não multiplicado por OI).
    """
    K = np.asarray(K, dtype=float)
    vol = np.asarray(vol, dtype=float)
    T = np.asarray(T, dtype=float)
    option_types = np.asarray(option_types)

    mask = (vol <= 0) | (T <= 0) | np.isnan(vol) | (K <= 0) | (S <= 0)
    greeks = {k: np.zeros_like(vol) for k in
              ['delta', 'gamma', 'vega', 'vanna', 'zomma', 'theta', 'charm', 'speed']}
    d1 = np.zeros_like(vol)

    with np.errstate(divide='ignore', invalid='ignore'):
        v = ~mask
        if not np.any(v):
            return greeks

        K_v, vol_v, T_v = K[v], vol[v], T[v]
        types_v = option_types[v]
        sqrt_T = np.sqrt(T_v)

        d1_v = (np.log(S / K_v) + (r + 0.5 * vol_v**2) * T_v) / (vol_v * sqrt_T)
        d2_v = d1_v - vol_v * sqrt_T

        pdf_d1 = norm.pdf(d1_v)
        cdf_d1 = norm.cdf(d1_v)
        cdf_d2 = norm.cdf(d2_v)

        delta_call = cdf_d1
        gamma_v = pdf_d1 / (S * vol_v * sqrt_T)

        # Delta
        greeks['delta'][v] = np.where(types_v == 'Call', delta_call, delta_call - 1)

        # Gamma
        greeks['gamma'][v] = gamma_v

        # Vega (por 1% de mudança na vol — divide por 100)
        greeks['vega'][v] = S * pdf_d1 * sqrt_T / 100.0

        # Vanna (∂Delta/∂Vol)
        greeks['vanna'][v] = -pdf_d1 * d2_v / vol_v

        # Zomma (∂Gamma/∂Vol)
        greeks['zomma'][v] = gamma_v * ((d1_v * d2_v - 1) / vol_v)

        # Theta (decaimento anualizado do preço da opção)
        exp_rT = np.exp(-r * T_v)
        theta_call = -(S * pdf_d1 * vol_v) / (2 * sqrt_T) - r * K_v * exp_rT * cdf_d2
        theta_put = -(S * pdf_d1 * vol_v) / (2 * sqrt_T) + r * K_v * exp_rT * (1 - cdf_d2)
        greeks['theta'][v] = np.where(types_v == 'Call', theta_call, theta_put)

        # Charm (∂Delta/∂T — decaimento anualizado do delta)
        charm_num = 2 * r * T_v - d2_v * vol_v * sqrt_T
        charm_den = 2 * T_v * vol_v * sqrt_T
        greeks['charm'][v] = -pdf_d1 * charm_num / charm_den

        # Speed (∂Gamma/∂S)
        greeks['speed'][v] = -gamma_v / S * (d1_v / (vol_v * sqrt_T) + 1)

    return greeks


def calculate_flip(levels, curve):
    """Encontra o ponto onde a curva cruza zero (interpolação linear)."""
    sign_changes = np.where(np.diff(np.sign(curve)))[0]
    if sign_changes.size > 0:
        try:
            i = sign_changes[0]
            x1, y1 = levels[i], curve[i]
            x2, y2 = levels[i + 1], curve[i + 1]
            if (y2 - y1) == 0:
                return None
            return x1 - y1 * (x2 - x1) / (y2 - y1)
        except (IndexError, ZeroDivisionError):
            return None
    return None


def implied_move_pct(iv_annual, days=1):
    """Retorna o movimento implícito diário em % dado IV anualizada (decimal)."""
    return iv_annual * 100 * math.sqrt(days / TRADING_DAYS)


def fmt_value(value, decimals=2):
    """Formata valor grande em Bi/Mi/K."""
    if pd.isna(value):
        return "N/A"
    a = abs(value)
    if a >= 1e9:
        return f"{value / 1e9:,.{decimals}f} Bi"
    if a >= 1e6:
        return f"{value / 1e6:,.{decimals}f} Mi"
    if a >= 1e3:
        return f"{value / 1e3:,.{decimals}f} K"
    return f"{value:,.{decimals}f}"


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — PIPELINE DE DADOS (BQL)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_market_data(ticker):
    """Busca spot, IV 30d, RV 30d, skew, volume médio em dólares."""
    spot = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.px_last()}))[0].df()['Value'].iloc[-1],
        errors='coerce')

    iv_30d = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.implied_volatility(
            expiry='30D', pct_moneyness='100')}))[0].df()['Value'].iloc[-1],
        errors='coerce') / 100.0

    rv_30d = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.volatility_30d_calc()}))[0].df()['Value'].iloc[-1],
        errors='coerce') / 100.0

    put_iv = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.implied_volatility(
            expiry='30D', delta='25', put_call='PUT')}))[0].df()['Value'].iloc[-1],
        errors='coerce')
    call_iv = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.implied_volatility(
            expiry='30D', delta='25', put_call='CALL')}))[0].df()['Value'].iloc[-1],
        errors='coerce')
    skew = (put_iv - call_iv) / 100.0 if pd.notna(put_iv) and pd.notna(call_iv) else 0.0

    avg_dollar_volume_item = bq.func.avg(
        bq.data.px_volume(dates=bq.func.range('-30D', '0D')) *
        bq.data.px_last(dates=bq.func.range('-30D', '0D')))
    avg_dollar_volume = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': avg_dollar_volume_item}))[0].df()['Value'].iloc[-1],
        errors='coerce')

    return {
        'spot': spot, 'iv_30d': iv_30d, 'rv_30d': rv_30d,
        'skew': skew, 'avg_dollar_volume': avg_dollar_volume
    }


def fetch_options_chain(ticker, spot, min_dte, max_dte, mny_low, mny_high):
    """Busca a cadeia de opções e retorna DataFrame processado."""
    from_strike = (1 + mny_low) * spot
    to_strike = (1 + mny_high) * spot

    conditions = (
        bq.data.expire_dt() >= f'{min_dte}d'
    ).and_(
        bq.data.expire_dt() <= f'{max_dte}d'
    ).and_(
        bq.data.strike_px() > from_strike
    ).and_(
        bq.data.strike_px() < to_strike
    )
    univ = bq.univ.filter(bq.univ.options([ticker]), conditions)

    items = {
        'Expire': bq.data.expire_dt(),
        'Strike': bq.data.strike_px(),
        'Type':   bq.data.put_call(),
        'IV':     bq.data.ivol()['Value'],
        'OI':     bq.data.open_int()['Value'],
    }
    data = pd.concat(
        [r.df() for r in bq.execute(bql.Request(univ, items))], axis=1
    ).dropna()

    if data.empty:
        raise ValueError("Nenhum dado de opção encontrado para os filtros.")

    df = pd.DataFrame({
        'Exp':    pd.to_datetime(data['Expire']),
        'Strike': pd.to_numeric(data['Strike'], errors='coerce'),
        'Type':   data['Type'],
    })
    df['IV'] = pd.to_numeric(data['IV'], errors='coerce')
    df['OI'] = pd.to_numeric(data['OI'], errors='coerce')
    df.dropna(subset=['IV', 'OI', 'Strike'], inplace=True)
    df['IV'] /= 100.0

    today = np.datetime64(datetime.utcnow().date(), 'D')
    bus = np.busday_count(today, df.Exp.dt.normalize().values.astype('datetime64[D]'))
    df['Tte'] = np.maximum(bus, 1) / float(TRADING_DAYS)

    return df, from_strike, to_strike


def fetch_historical(ticker, period='-2Y'):
    """Busca preços históricos e retorna log-retornos."""
    hist_req = bql.Request(ticker, {'Value': bq.data.px_last(
        dates=bq.func.range(period, '0D'), fill='PREV')})
    prices = pd.to_numeric(
        bq.execute(hist_req)[0].df()['Value'], errors='coerce'
    ).dropna()
    if prices.empty:
        raise ValueError("Histórico de preços vazio.")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Retornos insuficientes.")
    return prices, log_returns


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — CÁLCULOS DE EXPOSIÇÃO E CURVAS MODELO
# ═══════════════════════════════════════════════════════════════════════════════

def compute_strike_exposures(df, greeks, spot):
    """
    Adiciona colunas de exposição por tipo (Call/Put) ao DataFrame
    e retorna DataFrame agregado por strike.
    """
    is_call = (df['Type'] == 'Call').values
    is_put = (df['Type'] == 'Put').values
    oi_100 = df['OI'].values * 100.0

    for cfg in GREEK_CONFIGS:
        key = cfg['key']
        raw = greeks[key]
        df[f'Call_{key}'] = np.where(is_call, raw * oi_100, 0.0)
        df[f'Put_{key}']  = np.where(is_put,  raw * oi_100, 0.0)

    # Agregar por strike — apenas colunas de exposição
    exp_cols = []
    for cfg in GREEK_CONFIGS:
        exp_cols += [f'Call_{cfg["key"]}', f'Put_{cfg["key"]}']
    agg = df.groupby('Strike')[exp_cols].sum()

    # Computar totais líquidos
    for cfg in GREEK_CONFIGS:
        key = cfg['key']
        scale_val = cfg['scale'](spot)
        call_scaled = agg[f'Call_{key}'] * scale_val
        put_scaled = agg[f'Put_{key}'] * scale_val
        agg[f'Total_{key}'] = cfg['op'](call_scaled, put_scaled) / cfg['div']

    return agg


def compute_model_curves(df, levels, configs=None):
    """
    Calcula curvas modelo (exposure vs. preço spot) para todas as gregas
    em uma única passagem por nível de preço.

    Retorna dict: {greek_name: np.array com a curva}.
    """
    if configs is None:
        configs = GREEK_CONFIGS

    results = {c['name']: [] for c in configs}
    strikes = df['Strike'].values
    ivs = df['IV'].values
    ttes = df['Tte'].values
    ois = df['OI'].values * 100.0
    types = df['Type'].values
    is_call = types == 'Call'
    is_put = types == 'Put'

    for L in levels:
        greeks = calculate_all_greeks(L, strikes, ivs, ttes, types)
        for cfg in configs:
            key = cfg['key']
            raw = greeks[key]
            call_exp = np.nansum(raw[is_call] * ois[is_call])
            put_exp = np.nansum(raw[is_put] * ois[is_put])
            total = cfg['op'](call_exp, put_exp) * cfg['scale'](L)
            results[cfg['name']].append(total)

    return {name: np.array(curve) for name, curve in results.items()}


def compute_walls(agg):
    """Identifica Call Wall e Put Wall (strikes com máxima concentração de gamma)."""
    call_wall = agg['Call_gamma'].idxmax() if 'Call_gamma' in agg.columns else None
    put_wall = agg['Put_gamma'].idxmax() if 'Put_gamma' in agg.columns else None
    return call_wall, put_wall


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — MODELOS DE RISCO (VaR, Monte Carlo, P&L)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_risk_model(log_returns):
    """Ajusta distribuição t-Student e calcula VaR/CVaR."""
    tdf, tloc, tscale = student_t.fit(log_returns)
    var_99 = student_t.ppf(0.01, tdf, tloc, tscale)
    cvar_99 = student_t.expect(
        lambda x: x, args=(tdf,), loc=tloc, scale=tscale,
        lb=-np.inf, ub=var_99) / 0.01
    return {'tdf': tdf, 'tloc': tloc, 'tscale': tscale,
            'var_99': var_99, 'cvar_99': cvar_99}


def run_monte_carlo(spot, df, risk_params, n_sims=10000):
    """Simula P&L do livro do market maker para o próximo dia."""
    greeks = calculate_all_greeks(spot, df.Strike.values, df.IV.values,
                                  df.Tte.values, df.Type.values)
    dex = (greeks['delta'] * df.OI.values * 100).sum()
    gex_per_pt = (greeks['gamma'] * np.where(df.Type.values == 'Call', 1, -1)
                  * df.OI.values * 100).sum()

    sim_returns = student_t.rvs(risk_params['tdf'], loc=risk_params['tloc'],
                                scale=risk_params['tscale'], size=n_sims)
    sim_prices = spot * (1 + sim_returns)
    delta_s = sim_prices - spot
    # P&L do dealer (posição inversa ao mercado)
    pnl = -(dex * delta_s + 0.5 * gex_per_pt * delta_s**2)
    return pnl, sim_prices


def compute_pnl_curves(greeks_now, df, spot, levels, skew):
    """
    Calcula curvas de P&L comparativas:
    - Simplificada (Delta + Gamma)
    - Completa (Delta + Gamma + Vega + Vanna + Zomma)
    - Market vs. Dealer
    """
    oi = df['OI'].values
    delta_vals = greeks_now['delta']
    gamma_vals = greeks_now['gamma']
    vega_vals = greeks_now['vega']
    vanna_vals = greeks_now['vanna']
    zomma_vals = greeks_now['zomma']

    pnl_simple = []
    pnl_complete = []
    market_pnl = []
    hedge_demand = []

    for s in levels:
        dS = s - spot
        dVol = -np.sign(dS) * abs(skew) if skew != 0 else 0

        # Simplificado
        p_s = np.nansum((delta_vals * dS + 0.5 * gamma_vals * dS**2) * oi * 100)

        # Completo (com efeito Zomma no gamma)
        gamma_adj = gamma_vals + zomma_vals * dVol
        p_c = np.nansum(
            (delta_vals * dS
             + 0.5 * gamma_adj * dS**2
             + vega_vals * (dVol * 100)
             + vanna_vals * dS * (dVol * 100)) * oi * 100)

        pnl_simple.append(p_s)
        pnl_complete.append(p_c)
        market_pnl.append(p_s)

        # Demanda de hedge (contratos futuros para ficar delta-neutro)
        greeks_at_s = calculate_all_greeks(s, df.Strike.values, df.IV.values,
                                           df.Tte.values, df.Type.values)
        mkt_delta = np.nansum(greeks_at_s['delta'] * oi * 100)
        hedge_demand.append(mkt_delta / FUTURES_MULTIPLIER)

    return {
        'simple': np.array(pnl_simple),
        'complete': np.array(pnl_complete),
        'market': np.array(market_pnl),
        'dealer': -np.array(market_pnl),
        'hedge_demand': np.array(hedge_demand),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — REBALANCEAMENTO DE ETFs
# ═══════════════════════════════════════════════════════════════════════════════

def compute_etf_rebalancing(ticker):
    """Calcula fluxo teórico de rebalanceamento dos ETFs que seguem o índice."""
    etfs = ['VOO US Equity', 'SPY US Equity', 'IVV US Equity']
    start_date = (pd.Timestamp('today') - pd.tseries.offsets.BMonthBegin(1)).strftime('%Y-%m-%d')

    def _float_weights(idx, as_of=None):
        uni = bq.univ.members([idx], dates=[as_of]) if as_of else bq.univ.members([idx])
        cap = bq.data.cur_mkt_cap(dates=[as_of]) if as_of else bq.data.cur_mkt_cap()
        df_cap = bq.execute(bql.Request(uni, [cap], with_params=BQL_PARAMS)
                            )[0].df().select_dtypes('number').rename(columns=lambda c: 'Cap')
        df_ff = bq.execute(bql.Request(bq.univ.members([idx]),
                                       [bq.data.eqy_free_float_pct()],
                                       with_params=BQL_PARAMS)
                           )[0].df().select_dtypes('number').rename(columns=lambda c: 'FF')
        df_ = df_cap.join(df_ff, how='inner')
        df_['FMC'] = df_['Cap'] * df_['FF'] / 100
        df_['Weight'] = df_['FMC'] / df_['FMC'].sum()
        return df_['Weight']

    def _adv5_usd(tickers_list):
        usd_val = (bq.data.px_volume(dates=bq.func.range('-5D', '-1D'))
                   * bq.data.px_last(dates=bq.func.range('-5D', '-1D')))
        adv_item = bq.func.avg(usd_val)
        return bq.execute(bql.Request(bq.univ.List(tickers_list),
                                      {'ADV5': adv_item},
                                      with_params=BQL_PARAMS))[0].df()['ADV5']

    w0 = _float_weights(ticker, start_date)
    w1 = _float_weights(ticker)
    dw = pd.DataFrame({'Start': w0, 'Now': w1}).dropna()
    dw['Delta'] = dw['Now'] - dw['Start']

    adv5 = _adv5_usd(dw.index.tolist()).fillna(1)
    aum = bq.execute(bql.Request(
        bq.univ.List(etfs), {'AUM': bq.data.fund_total_assets()},
        with_params=BQL_PARAMS))[0].df().select_dtypes('number').iloc[:, 0]

    flow = dw.copy()
    flow['Flow_$'] = flow['Delta'] * aum.sum()
    flow['PctADV'] = (flow['Flow_$'] / adv5.reindex(flow.index).fillna(1)) * 100
    flow = flow[flow['Flow_$'] != 0].sort_values('Flow_$', ascending=False)
    return flow, start_date


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — MATRIZES DE SENSIBILIDADE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sensitivity_matrices(df, spot):
    """
    Calcula matrizes de sensibilidade (preço × vol shift) para cada grega.
    Retorna dict: {greek_name: pd.DataFrame}.
    """
    spot_range = np.linspace(spot * 0.97, spot * 1.03, 7)
    vol_shifts = np.linspace(-0.03, 0.03, 5)

    cols = [f"{s:,.0f}" for s in spot_range]
    idx = [f"{vs:+.1%}" for vs in vol_shifts]

    greek_keys = ['delta', 'gamma', 'vega', 'vanna', 'theta', 'zomma', 'speed']
    matrices = {k: pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float) for k in greek_keys}

    strikes = df['Strike'].values
    base_ivs = df['IV'].values
    ttes = df['Tte'].values
    types = df['Type'].values
    ois = df['OI'].values

    for i, iv_shift in enumerate(vol_shifts):
        shifted_ivs = np.clip(base_ivs + iv_shift, 0.001, None)
        for j, s in enumerate(spot_range):
            greeks = calculate_all_greeks(s, strikes, shifted_ivs, ttes, types)
            oi_100 = ois * 100.0
            is_call = types == 'Call'

            matrices['delta'].iloc[i, j] = np.nansum(greeks['delta'] * oi_100 * s)
            matrices['gamma'].iloc[i, j] = np.nansum(
                greeks['gamma'] * np.where(is_call, 1, -1) * oi_100 * (s**2) * 0.01)
            matrices['vega'].iloc[i, j] = np.nansum(greeks['vega'] * oi_100)
            matrices['vanna'].iloc[i, j] = np.nansum(greeks['vanna'] * oi_100)
            matrices['theta'].iloc[i, j] = np.nansum(greeks['theta'] * oi_100)
            matrices['zomma'].iloc[i, j] = np.nansum(greeks['zomma'] * oi_100)
            matrices['speed'].iloc[i, j] = np.nansum(greeks['speed'] * oi_100)

    for k in matrices:
        matrices[k].index.name = 'Vol Shift'

    return matrices


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — VISUALIZAÇÃO (Gauges, Gráficos, Tabelas)
# ═══════════════════════════════════════════════════════════════════════════════

def create_gauge(value, title, range_min, range_max, bar_color, suffix,
                 steps=None, width=220, height=180):
    """Cria indicador gauge do Plotly."""
    if pd.isna(value):
        value = 0
    gauge_cfg = {
        'axis': {'range': [range_min, range_max]},
        'bar': {'color': bar_color},
    }
    if steps:
        gauge_cfg['steps'] = steps
    return go.FigureWidget(
        go.Indicator(
            mode="gauge+number", value=value,
            title={'text': title, 'font': {'size': 14}},
            number={'suffix': suffix, 'font': {'size': 18}, 'valueformat': '.2f'},
            gauge=gauge_cfg),
        layout=go.Layout(width=width, height=height,
                         margin=dict(l=15, r=15, t=40, b=15)))


def plot_exposure_charts(agg, df, spot, from_strike, to_strike,
                         levels, model_curves, flip_points,
                         call_wall, put_wall):
    """Gera todos os gráficos de exposição (Combo + Absolute + Model) para cada grega."""
    strikes = agg.index.values
    step = np.median(np.diff(np.sort(strikes))) if len(strikes) > 1 else 5
    bar_w = step * 0.8

    for cfg in GREEK_CONFIGS:
        name = cfg['name']
        key = cfg['key']
        unit = cfg['unit']
        div = cfg['div']
        scale_val = cfg['scale'](spot)
        combo_op = cfg['op']
        flip = flip_points.get(name)

        call_col = f'Call_{key}'
        put_col = f'Put_{key}'
        call_data = agg[call_col] * scale_val
        put_data = agg[put_col] * scale_val
        combo_data = combo_op(call_data, put_data) / div
        total_val = combo_data.sum()

        # ── Gráfico COMBO ──
        plt.figure(figsize=(18, 5))
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in combo_data.values]
        plt.bar(strikes, combo_data, width=bar_w, color=colors, edgecolor='k', lw=0.3)
        plt.axvline(spot, color='r', ls='--', lw=1.5, label=f'Spot {spot:,.0f}')
        if name == 'Gamma':
            if call_wall:
                plt.axvline(call_wall, color='g', ls=':', lw=2, label=f'Call Wall {call_wall:,.0f}')
            if put_wall:
                plt.axvline(put_wall, color='b', ls=':', lw=2, label=f'Put Wall {put_wall:,.0f}')
        if flip:
            plt.axvline(flip, color='orange', ls='--', lw=2.5, label=f'{name} Flip {flip:,.0f}')
        plt.title(f'COMBO {name.upper()} EXPOSURE = {total_val:,.2f} ({unit})',
                  fontsize=16, weight='bold')
        plt.ylabel(f'Exposure ({unit})', fontsize=12)
        plt.xlim(from_strike, to_strike)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ── Gráfico ABSOLUTE ──
        put_sign = 1 if combo_op is np.add else -1
        plt.figure(figsize=(18, 5))
        plt.bar(strikes, call_data / div, width=bar_w, color='#27ae60',
                edgecolor='k', lw=0.3, label=f'Call {name}')
        plt.bar(strikes, put_sign * put_data / div, width=bar_w, color='#c0392b',
                edgecolor='k', lw=0.3, label=f'Put {name}')
        plt.axvline(spot, color='r', ls='--', lw=1.5, label=f'Spot {spot:,.0f}')
        if name == 'Gamma':
            if call_wall:
                plt.axvline(call_wall, color='g', ls=':', lw=2, label=f'Call Wall {call_wall:,.0f}')
            if put_wall:
                plt.axvline(put_wall, color='b', ls=':', lw=2, label=f'Put Wall {put_wall:,.0f}')
        if flip:
            plt.axvline(flip, color='orange', ls='--', lw=2.5, label=f'{name} Flip {flip:,.0f}')
        plt.title(f'ABSOLUTE {name.upper()} EXPOSURE', fontsize=16, weight='bold')
        plt.ylabel(f'Exposure ({unit})', fontsize=12)
        plt.xlim(from_strike, to_strike)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ── Gráfico MODEL ──
        curve = model_curves[name] / div
        current_val = curve[np.argmin(np.abs(levels - spot))]
        plt.figure(figsize=(18, 5))
        plt.plot(levels, curve, lw=2, color='#2980b9')
        plt.fill_between(levels, 0, curve, where=curve >= 0,
                         alpha=0.15, color='green', interpolate=True)
        plt.fill_between(levels, 0, curve, where=curve <= 0,
                         alpha=0.15, color='red', interpolate=True)
        plt.axvline(spot, color='r', ls='--', lw=1.5, label=f'Spot {spot:,.0f}')
        if flip:
            plt.axvline(flip, color='orange', ls='--', lw=2.5, label=f'{name} Flip {flip:,.0f}')
        plt.axhline(0, color='k', lw=0.5)
        plt.title(f'{name.upper()} EXPOSURE MODEL = {current_val:,.2f} ({unit})',
                  fontsize=16, weight='bold')
        plt.xlabel('Preço do Ativo', fontsize=12)
        plt.ylabel(f'Exposure ({unit})', fontsize=12)
        plt.xlim(from_strike, to_strike)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def style_sensitivity_matrix(matrix_df, cmap='viridis'):
    """Aplica estilo visual a uma matriz de sensibilidade."""
    styled = matrix_df.style.format(fmt_value)
    styled = styled.set_properties(**{'width': '110px', 'text-align': 'center'})
    numeric = matrix_df.select_dtypes(include=np.number).dropna(how='all')
    if not numeric.empty and numeric.max().max() != numeric.min().min():
        styled = styled.background_gradient(cmap=cmap, axis=None)
    return styled.to_html()


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — CALLBACK PRINCIPAL E MONTAGEM DO DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

# Widget definitions (global)
ticker_w = wd.Text(value='SPX Index', description='Ativo:',
                   layout={'width': '250px'})
dte_w = wd.IntRangeSlider(value=[0, 30], min=0, max=90, step=1,
                          description='DTE (dias):',
                          layout={'width': '400px'})
mny_w = wd.FloatRangeSlider(value=[-0.05, 0.05], min=-0.30, max=0.30,
                            step=0.01, readout_format='.0%',
                            description='% MNY:',
                            layout={'width': '400px'})
run_btn = wd.Button(description='Gerar Análise Completa',
                    button_style='success', icon='cogs')
out_main = wd.Output()


def run_analysis(_):
    """Callback principal: busca dados, calcula tudo, monta dashboard."""
    with out_main:
        clear_output(wait=True)
        loading = wd.HTML("<h4>Inicializando...</h4>")
        display(loading)

        ticker = ticker_w.value.strip() or 'SPX Index'
        min_dte, max_dte = dte_w.value
        mny_low, mny_high = mny_w.value

        try:
            # ── 1. Dados de Mercado ──────────────────────────────────────
            loading.value = "<h4>1/9: Buscando dados de mercado...</h4>"
            mkt = fetch_market_data(ticker)
            spot = mkt['spot']
            iv_30d = mkt['iv_30d']
            rv_30d = mkt['rv_30d']
            skew = mkt['skew']
            avg_vol = mkt['avg_dollar_volume']

            if pd.isna(spot):
                raise ValueError(f"Spot inválido para {ticker}")

            # ── 2. Histórico + Modelo de Risco ───────────────────────────
            loading.value = "<h4>2/9: Modelagem de risco (t-Student)...</h4>"
            _, log_returns = fetch_historical(ticker)
            risk = fit_risk_model(log_returns)

            # ── 3. Cadeia de Opções ──────────────────────────────────────
            loading.value = "<h4>3/9: Buscando cadeia de opções...</h4>"
            df, from_strike, to_strike = fetch_options_chain(
                ticker, spot, min_dte, max_dte, mny_low, mny_high)

            # ── 4. Gregas + Exposições ───────────────────────────────────
            loading.value = f"<h4>4/9: Calculando gregas para {len(df)} opções...</h4>"
            greeks_now = calculate_all_greeks(
                spot, df.Strike.values, df.IV.values, df.Tte.values, df.Type.values)
            agg = compute_strike_exposures(df, greeks_now, spot)
            call_wall, put_wall = compute_walls(agg)

            if call_wall == put_wall and call_wall is not None:
                print(f"⚠️ Call Wall = Put Wall = {call_wall:,.0f}")

            # ── 5. Curvas Modelo ─────────────────────────────────────────
            loading.value = "<h4>5/9: Calculando curvas modelo (100 níveis × 7 gregas)...</h4>"
            levels = np.linspace(from_strike, to_strike, 100)
            model_curves = compute_model_curves(df, levels)

            # Flip points
            flip_points = {}
            for cfg in GREEK_CONFIGS:
                curve = model_curves[cfg['name']]
                flip_points[cfg['name']] = calculate_flip(levels, curve)

            gamma_flip = flip_points.get('Gamma')
            gamma_curve = model_curves['Gamma']

            # ── 6. Matrizes de Sensibilidade ─────────────────────────────
            loading.value = "<h4>6/9: Matrizes de sensibilidade (7×5×7)...</h4>"
            sens_matrices = compute_sensitivity_matrices(df, spot)

            # ── 7. Monte Carlo ───────────────────────────────────────────
            loading.value = "<h4>7/9: Simulação Monte Carlo (10k cenários)...</h4>"
            mc_pnl, mc_prices = run_monte_carlo(spot, df, risk)

            # ── 8. Curvas de P&L ─────────────────────────────────────────
            loading.value = "<h4>8/9: Curvas de P&L e hedge demand...</h4>"
            pnl_curves = compute_pnl_curves(greeks_now, df, spot, levels, skew)

            # ── 9. Rebalanceamento ETFs (opcional) ───────────────────────
            loading.value = "<h4>9/9: Rebalanceamento de ETFs...</h4>"
            try:
                etf_flow, etf_start = compute_etf_rebalancing(ticker)
                etf_ok = True
            except Exception:
                etf_flow, etf_start, etf_ok = None, None, False

            clear_output(wait=True)

            # ═════════════════════════════════════════════════════════════
            # MONTAGEM DAS ABAS DO DASHBOARD
            # ═════════════════════════════════════════════════════════════

            title_html = wd.HTML(
                f"<h2>Market Maker Dashboard: {ticker} @ {spot:,.2f}"
                f" ({datetime.now().strftime('%Y-%m-%d %H:%M')})</h2>")

            # ─── ABA 1: VISÃO GERAL ─────────────────────────────────────
            total_gex = gamma_curve[np.argmin(np.abs(levels - spot))]
            fragility = (abs(total_gex) / avg_vol) * 100 if pd.notna(avg_vol) and avg_vol > 0 else 0
            daily_move = implied_move_pct(iv_30d) if pd.notna(iv_30d) else 0
            vol_premium = (iv_30d - rv_30d) * 100 if pd.notna(iv_30d) and pd.notna(rv_30d) else 0

            g_frag = create_gauge(fragility, "Fragilidade (GEX/Vol)",
                                  0, 20, "#E74C3C", "%",
                                  steps=[{'range': [0, 5], 'color': '#c8e6c9'},
                                         {'range': [5, 12], 'color': '#fff9c4'},
                                         {'range': [12, 20], 'color': '#ffcdd2'}])
            g_vol = create_gauge(vol_premium, "Prêmio Vol (IV-RV)",
                                 -5, 5, "#F39C12", "%")
            g_skew = create_gauge(skew * 100, "Skew (P25-C25)",
                                  -5, 10, "#1ABC9C", "%")
            g_move = create_gauge(daily_move, "Mov. Esperado 1D",
                                  0, 5, "#2ECC71", "%")

            # GEX curve (Plotly)
            fig_gex = go.FigureWidget()
            fig_gex.add_trace(go.Scatter(
                x=levels, y=gamma_curve / 1e9, mode='lines',
                fill='tozeroy', line_color='#636EFA', name='GEX'))
            fig_gex.add_vline(x=spot, line_dash="dash", line_color="red",
                              annotation_text=f"Spot {spot:,.0f}")
            if gamma_flip:
                fig_gex.add_vline(x=gamma_flip, line_color="orange",
                                  annotation_text=f"G-Flip {gamma_flip:,.0f}")
            fig_gex.update_layout(title="Gamma Exposure (GEX)",
                                  yaxis_title="$ Bi / 1% move",
                                  height=350, template="plotly_white",
                                  margin=dict(t=35, b=25), showlegend=False)

            # Return distribution (Plotly)
            fig_dist = go.FigureWidget()
            fig_dist.add_trace(go.Histogram(
                x=log_returns, histnorm='probability density',
                name='Reais', marker_color='#636EFA', opacity=0.6,
                xbins=dict(size=log_returns.std() / 4)))
            x_pdf = np.linspace(log_returns.min(), log_returns.max(), 500)
            pdf_vals = student_t.pdf(x_pdf, risk['tdf'], risk['tloc'], risk['tscale'])
            fig_dist.add_trace(go.Scatter(
                x=x_pdf, y=pdf_vals, mode='lines',
                name='t-Student', line_color='orange'))
            fig_dist.add_vline(x=risk['var_99'], line_dash="dash",
                               line_color="darkred",
                               annotation_text=f"VaR 99% ({risk['var_99']:.2%})")
            fig_dist.update_layout(title="Distribuição de Retornos",
                                   yaxis_title="Prob.",
                                   xaxis_tickformat=".1%",
                                   height=350, template="plotly_white",
                                   margin=dict(t=35, b=25))

            # Sumário de vol e risco
            summary_html = f"""
            <div style='padding:10px; font-size:13px;'>
                <b>Condições de Volatilidade</b><br>
                IV 30d ATM: {iv_30d:.2%} | RV 30d: {rv_30d:.2%} |
                Prêmio: {vol_premium:+.2f}% | Skew: {skew:+.2%}<br><br>
                <b>Métricas de Risco Caudal</b><br>
                VaR 99%: {risk['var_99']:.2%} | CVaR 99%: {risk['cvar_99']:.2%}<br><br>
                <b>Posicionamento</b><br>
                Gamma Flip: ~{gamma_flip:,.0f if gamma_flip else 'N/A'} |
                Call Wall: {call_wall:,.0f if call_wall else 'N/A'} |
                Put Wall: {put_wall:,.0f if put_wall else 'N/A'}
            </div>"""

            tab1 = wd.VBox([
                wd.HBox([g_frag, g_vol, g_skew, g_move]),
                wd.HBox([fig_gex, fig_dist]),
                wd.HTML(summary_html)
            ])

            # ─── ABA 2: EXPOSIÇÕES POR STRIKE ───────────────────────────
            exp_output = wd.Output()
            with exp_output:
                plot_exposure_charts(
                    agg, df, spot, from_strike, to_strike,
                    levels, model_curves, flip_points,
                    call_wall, put_wall)
            tab2 = exp_output

            # ─── ABA 3: SENSIBILIDADE ────────────────────────────────────
            cmap_map = {
                'delta': 'RdBu_r', 'gamma': 'viridis', 'vega': 'YlGnBu',
                'vanna': 'PuOr', 'theta': 'Greens', 'zomma': 'plasma',
                'speed': 'coolwarm'
            }
            titles_map = {
                'delta': 'Delta Nocional', 'gamma': 'Gamma (GEX)',
                'vega': 'Vega', 'vanna': 'Vanna',
                'theta': 'Theta (Decaimento)', 'zomma': 'Zomma', 'speed': 'Speed'
            }
            sens_html_parts = []
            for key in ['delta', 'gamma', 'vega', 'vanna', 'theta', 'zomma', 'speed']:
                styled = style_sensitivity_matrix(sens_matrices[key], cmap_map[key])
                sens_html_parts.append(f"<h4>{titles_map[key]}</h4>{styled}<br>")
            tab3 = wd.VBox([
                wd.HTML("<h3>Matrizes de Sensibilidade (Preço × Vol Shift)</h3>"),
                wd.HTML("".join(sens_html_parts))
            ])

            # ─── ABA 4: ANÁLISE DE P&L ──────────────────────────────────
            # P&L comparativo
            fig_pnl = go.FigureWidget()
            fig_pnl.add_trace(go.Scatter(
                x=levels, y=pnl_curves['simple'] / 1e6,
                mode='lines', name='Simplificado (Δ+Γ)',
                line=dict(color='orange', dash='dot')))
            fig_pnl.add_trace(go.Scatter(
                x=levels, y=pnl_curves['complete'] / 1e6,
                mode='lines', name='Completo (+Vega+Vanna+Zomma)',
                line=dict(color='blue'), fill='tonexty',
                fillcolor='rgba(0,0,255,0.08)'))
            fig_pnl.add_vline(x=spot, line_dash="dash", line_color="red")
            fig_pnl.add_hline(y=0, line_width=0.5, line_color="black")
            fig_pnl.update_layout(title="P&L Comparativo: Modelo Completo vs. Simplificado",
                                  yaxis_title="P&L ($ Mi)", xaxis_title="Preço do Ativo",
                                  height=380, template="plotly_white")

            # Dealer P&L
            fig_dealer = go.FigureWidget()
            fig_dealer.add_trace(go.Scatter(
                x=levels, y=pnl_curves['dealer'] / 1e6,
                mode='lines', name='P&L Dealer', line_color='purple',
                fill='tozeroy', fillcolor='rgba(128,0,128,0.08)'))
            fig_dealer.add_vline(x=spot, line_dash="dash", line_color="red")
            fig_dealer.add_hline(y=0, line_width=0.5, line_color="black")
            fig_dealer.update_layout(title="P&L Estimado do Market Maker",
                                     yaxis_title="P&L ($ Mi)", xaxis_title="Preço do Ativo",
                                     height=380, template="plotly_white")

            # Hedge demand
            fig_hedge = go.FigureWidget()
            fig_hedge.add_trace(go.Scatter(
                x=levels, y=pnl_curves['hedge_demand'],
                mode='lines', line_color='teal', name='Contratos'))
            fig_hedge.add_vline(x=spot, line_dash="dash", line_color="red")
            fig_hedge.add_hline(y=0, line_width=0.5, line_color="black")
            fig_hedge.update_layout(
                title=f"Demanda de Hedge em Futuros ({FUTURES_TICKER})",
                yaxis_title="Número de Contratos",
                xaxis_title="Preço do Ativo",
                height=380, template="plotly_white")

            tab4 = wd.VBox([fig_pnl, wd.HBox([fig_dealer, fig_hedge])])

            # ─── ABA 5: MONTE CARLO ──────────────────────────────────────
            pnl_sorted = np.sort(mc_pnl)
            colors_mc = ['limegreen' if x > 0 else 'crimson' for x in pnl_sorted]
            fig_tornado = go.FigureWidget()
            fig_tornado.add_trace(go.Bar(
                y=np.arange(len(pnl_sorted)),
                x=pnl_sorted / 1e9,
                orientation='h', marker_color=colors_mc))
            fig_tornado.update_layout(
                title="Distribuição de P&L do Livro (10k Simulações, 1 Dia)",
                xaxis_title="P&L ($ Bi)", height=400,
                template="plotly_white",
                yaxis={'showticklabels': False, 'title': 'Cenários'})

            sim_var = np.percentile(mc_pnl, 1)
            sim_cvar = mc_pnl[mc_pnl <= sim_var].mean() if np.any(mc_pnl <= sim_var) else sim_var
            mc_table = pd.DataFrame({
                'Métrica': ['P&L Médio', 'VaR 99% (Sim.)', 'CVaR 99% (Sim.)'],
                'Valor': [f"${np.mean(mc_pnl)/1e9:,.4f} Bi",
                          f"${sim_var/1e9:,.4f} Bi",
                          f"${sim_cvar/1e9:,.4f} Bi"]
            }).to_html(classes='table table-sm', index=False, border=0)

            tab5 = wd.VBox([
                wd.HTML("<h3>Simulação Monte Carlo (t-Student)</h3>"),
                wd.HBox([fig_tornado, wd.HTML(
                    f"<div style='padding:20px;'>{mc_table}</div>")])
            ])

            # ─── ABA 6: REBALANCEAMENTO ETFs ────────────────────────────
            if etf_ok and etf_flow is not None and not etf_flow.empty:
                reb_html = (etf_flow.head(10)[['Flow_$', 'PctADV']]
                            .style.format({'Flow_$': '${:,.0f}', 'PctADV': '{:.1f}%'})
                            .background_gradient(cmap='RdYlGn', subset=['PctADV'])
                            .to_html())
                tab6 = wd.VBox([
                    wd.HTML(f"<h3>Fluxo de Rebalanceamento desde {etf_start}</h3>"),
                    wd.HTML(reb_html)
                ])
            else:
                tab6 = wd.VBox([wd.HTML(
                    "<h3>Rebalanceamento ETFs</h3>"
                    "<p>Dados de rebalanceamento não disponíveis para este ativo.</p>")])

            # ─── ABA 7: SIMULADOR INTERATIVO ─────────────────────────────
            vol_slider = wd.FloatSlider(
                value=0, min=-5, max=5, step=0.5,
                description='Shift Vol (%):', continuous_update=False,
                layout={'width': '400px'})
            dte_slider = wd.IntSlider(
                value=0, min=0, max=20, step=1,
                description='Dias a Frente:', continuous_update=False,
                layout={'width': '400px'})

            fig_sim_dex = go.FigureWidget()
            fig_sim_dex.add_trace(go.Scatter(x=levels, y=np.zeros(len(levels)),
                                             mode='lines', line_color='#3498DB',
                                             name='Delta'))
            fig_sim_dex.update_layout(title="Delta Nocional", yaxis_title="$ Bi",
                                      height=300, template="plotly_white",
                                      margin=dict(t=30, b=20))

            fig_sim_gex = go.FigureWidget()
            fig_sim_gex.add_trace(go.Scatter(x=levels, y=np.zeros(len(levels)),
                                             mode='lines', line_color='#E74C3C',
                                             name='Gamma'))
            fig_sim_gex.update_layout(title="Gamma (GEX)", yaxis_title="$ Bi / 1% move",
                                      height=300, template="plotly_white",
                                      margin=dict(t=30, b=20))

            fig_sim_vega = go.FigureWidget()
            fig_sim_vega.add_trace(go.Scatter(x=levels, y=np.zeros(len(levels)),
                                              mode='lines', line_color='#8E44AD',
                                              name='Vega'))
            fig_sim_vega.update_layout(title="Vega Nocional", yaxis_title="$ Mi",
                                       height=300, template="plotly_white",
                                       margin=dict(t=30, b=20))

            def _update_simulator(change=None):
                v_shift = vol_slider.value / 100.0
                d_shift = dte_slider.value
                sim_vol = np.clip(df.IV.values + v_shift, 0.001, None)
                sim_tte = np.clip(df.Tte.values - d_shift / TRADING_DAYS, 1.0 / TRADING_DAYS, None)
                types_arr = df.Type.values

                dex_c, gex_c, vex_c = [], [], []
                for L in levels:
                    g = calculate_all_greeks(L, df.Strike.values, sim_vol, sim_tte, types_arr)
                    oi_100 = df.OI.values * 100.0
                    dex_c.append(np.nansum(g['delta'] * oi_100 * L))
                    gex_c.append(np.nansum(g['gamma'] * np.where(types_arr == 'Call', 1, -1)
                                           * oi_100 * (L**2) * 0.01))
                    vex_c.append(np.nansum(g['vega'] * oi_100))

                with fig_sim_dex.batch_update():
                    fig_sim_dex.data[0].y = np.array(dex_c) / 1e9
                with fig_sim_gex.batch_update():
                    fig_sim_gex.data[0].y = np.array(gex_c) / 1e9
                with fig_sim_vega.batch_update():
                    fig_sim_vega.data[0].y = np.array(vex_c) / 1e6

            vol_slider.observe(_update_simulator, names='value')
            dte_slider.observe(_update_simulator, names='value')
            _update_simulator()

            tab7 = wd.VBox([
                wd.HTML("<h3>Simulador Interativo de Gregas</h3>"
                        "<p>Ajuste vol e tempo para ver como as exposições mudam.</p>"),
                wd.HBox([vol_slider, dte_slider]),
                wd.HBox([fig_sim_dex, fig_sim_gex]),
                fig_sim_vega
            ])

            # ─── ABA 8: RELATÓRIO DE RISCO ───────────────────────────────
            oi_100 = df['OI'].values * 100.0
            delta_notional = np.nansum(greeks_now['delta'] * oi_100) * spot
            total_vega = np.nansum(greeks_now['vega'] * oi_100)
            total_zomma = np.nansum(greeks_now['zomma'] * oi_100)
            total_speed = np.nansum(greeks_now['speed'] * oi_100)
            total_charm = np.nansum(greeks_now['charm'] * oi_100) / 365.0
            total_gex_val = np.nansum(
                greeks_now['gamma'] * np.where(df.Type.values == 'Call', 1, -1)
                * oi_100 * (spot**2) * 0.01)

            vanna_impact = np.nansum(greeks_now['vanna'] * oi_100) * spot
            vanna_action = "VENDER" if vanna_impact > 0 else "COMPRAR"
            zomma_action = "MAIS INSTÁVEL" if total_zomma < 0 else "MAIS ESTÁVEL"

            gamma_regime = "N/A"
            if gamma_flip:
                gamma_regime = ("ESTABILIDADE (acima do Flip)" if spot > gamma_flip
                                else "ACELERAÇÃO (abaixo do Flip)")

            hedge_contracts = -np.nansum(greeks_now['delta'] * oi_100) / FUTURES_MULTIPLIER
            hedge_action = "COMPRAR" if hedge_contracts > 0 else "VENDER"

            report_html = f"""
            <div style='font-family: monospace; font-size: 13px; padding: 15px;
                        background: #f8f9fa; border: 1px solid #ddd; border-radius: 8px;'>
            <h3>RELATÓRIO COMPLETO DE RISCO E POSICIONAMENTO</h3>
            <p>Análise: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Spot: ${spot:,.2f} |
               {len(df)} opções analisadas</p>
            <hr>

            <h4>1. HEDGE DIRECIONAL (DELTA)</h4>
            <p>Delta Nocional Total: <b>{fmt_value(delta_notional)}</b></p>
            <p>→ Para ficar delta-neutro: <b>{hedge_action} {abs(hedge_contracts):,.0f}</b>
               contratos {FUTURES_TICKER}</p>

            <h4>2. REGIME DE MERCADO (GAMMA)</h4>
            <p>GEX Total: <b>{fmt_value(total_gex_val)}</b> per 1% move</p>
            <p>Gamma Flip: <b>~{gamma_flip:,.0f if gamma_flip else 'N/A'}</b></p>
            <p>Regime atual: <b>{gamma_regime}</b></p>

            <h4>3. CENÁRIO: VOL +1%</h4>
            <p>Vanna → Dealers forçados a <b>{vanna_action}</b>
               ~{fmt_value(abs(vanna_impact))} do ativo</p>
            <p>Zomma → Regime de preços ficaria <b>{zomma_action}</b></p>

            <h4>4. EXPOSIÇÕES DE ORDEM SUPERIOR</h4>
            <table style='border-collapse: collapse; width: 100%;'>
            <tr><th style='text-align:left; padding:5px; border-bottom:1px solid #ccc;'>
                Exposição</th>
                <th style='text-align:right; padding:5px; border-bottom:1px solid #ccc;'>
                Valor</th>
                <th style='text-align:left; padding:5px; border-bottom:1px solid #ccc;'>
                Interpretação</th></tr>
            <tr><td style='padding:5px'>Vega</td>
                <td style='text-align:right; padding:5px'>{fmt_value(total_vega)}</td>
                <td style='padding:5px'>P&L por 1% de aumento na vol</td></tr>
            <tr><td style='padding:5px'>Zomma</td>
                <td style='text-align:right; padding:5px'>{fmt_value(total_zomma)}</td>
                <td style='padding:5px'>Mudança no Gamma por 1% vol</td></tr>
            <tr><td style='padding:5px'>Speed</td>
                <td style='text-align:right; padding:5px'>{fmt_value(total_speed)}</td>
                <td style='padding:5px'>Mudança no Gamma por $1 no spot</td></tr>
            <tr><td style='padding:5px'>Charm</td>
                <td style='text-align:right; padding:5px'>{fmt_value(total_charm)}</td>
                <td style='padding:5px'>Decaimento diário do delta</td></tr>
            </table>

            <h4>5. MÉTRICAS DE RISCO CAUDAL</h4>
            <p>VaR 99% (diário): <b>{risk['var_99']:.2%}</b> |
               CVaR 99%: <b>{risk['cvar_99']:.2%}</b></p>
            <p>Monte Carlo VaR 99%: <b>${np.percentile(mc_pnl, 1)/1e9:,.4f} Bi</b></p>
            </div>"""

            tab8 = wd.VBox([wd.HTML(report_html)])

            # ═════════════════════════════════════════════════════════════
            # MONTAGEM FINAL
            # ═════════════════════════════════════════════════════════════
            dashboard = wd.Tab()
            dashboard.children = [tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8]
            tab_names = [
                'Visão Geral', 'Exposições', 'Sensibilidade', 'Análise P&L',
                'Monte Carlo', 'Rebalanceamento', 'Simulador', 'Relatório'
            ]
            for i, name in enumerate(tab_names):
                dashboard.set_title(i, name)

            display(title_html, dashboard)

        except Exception as e:
            clear_output(wait=True)
            print(f"ERRO NA ANÁLISE: {e}")
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — INTERFACE DE WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

run_btn.on_click(run_analysis)
display(wd.VBox([
    wd.HBox([ticker_w, dte_w]),
    mny_w,
    run_btn,
    out_main
]))
