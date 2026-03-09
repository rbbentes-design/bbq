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
from functools import lru_cache
from typing import Optional, Dict, List, Tuple
import math
import traceback
from scipy.stats import norm, t as student_t
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import bql
import random as _random
import warnings

try:
    import ipydatagrid as ipd
    from ipydatagrid import DataGrid, BarRenderer, TextRenderer
    HAS_DATAGRID = True
except ImportError:
    HAS_DATAGRID = False

try:
    import bqplot as bqp
    HAS_BQPLOT = True
except ImportError:
    HAS_BQPLOT = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'font.size': 11,
})

bq = bql.Service()

# ── Design System (cores, CSS, template Plotly) ─────────────────────────────
_C = {
    'bg': '#0d1117', 'card': '#161b22', 'card2': '#1c2333',
    'border': '#30363d', 'border_light': '#21262d',
    'text': '#e6edf3', 'text_muted': '#8b949e', 'text_dim': '#484f58',
    'accent': '#58a6ff', 'teal': '#00d4aa', 'orange': '#f0883e',
    'green': '#3fb950', 'red': '#f85149', 'purple': '#bc8cff',
    'yellow': '#d29922', 'pink': '#f778ba',
}

DASH_CSS = f"""
<style>
.mm-dash {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; color: {_C['text']}; }}
.mm-title {{ font-size: 22px; font-weight: 600; color: {_C['text']}; padding: 12px 0 4px; border-bottom: 2px solid {_C['accent']}; margin-bottom: 8px; letter-spacing: -0.3px; }}
.mm-title small {{ font-size: 13px; color: {_C['text_muted']}; font-weight: 400; margin-left: 12px; }}
.mm-card {{ background: {_C['card']}; border: 1px solid {_C['border']}; border-radius: 8px; padding: 16px; margin: 6px 0; }}
.mm-card h3 {{ margin: 0 0 10px; font-size: 15px; font-weight: 600; color: {_C['accent']}; text-transform: uppercase; letter-spacing: 0.5px; }}
.mm-card h4 {{ margin: 8px 0 6px; font-size: 13px; font-weight: 600; color: {_C['teal']}; }}
.mm-card p, .mm-card span {{ font-size: 13px; color: {_C['text_muted']}; line-height: 1.6; }}
.mm-card b {{ color: {_C['text']}; }}
.mm-badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
.mm-badge-green {{ background: rgba(63,185,80,0.15); color: {_C['green']}; }}
.mm-badge-red {{ background: rgba(248,81,73,0.15); color: {_C['red']}; }}
.mm-badge-yellow {{ background: rgba(210,153,34,0.15); color: {_C['yellow']}; }}
.mm-badge-blue {{ background: rgba(88,166,255,0.15); color: {_C['accent']}; }}
.mm-table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; }}
.mm-table th {{ background: {_C['card2']}; color: {_C['text_muted']}; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; padding: 10px 12px; border-bottom: 2px solid {_C['border']}; text-align: left; }}
.mm-table td {{ padding: 8px 12px; border-bottom: 1px solid {_C['border_light']}; color: {_C['text']}; }}
.mm-table tr:hover td {{ background: rgba(88,166,255,0.04); }}
.mm-metric {{ display: inline-block; margin: 0 16px 8px 0; }}
.mm-metric .label {{ font-size: 11px; color: {_C['text_muted']}; text-transform: uppercase; letter-spacing: 0.3px; }}
.mm-metric .value {{ font-size: 16px; font-weight: 600; color: {_C['text']}; }}
.mm-section-label {{ font-size: 11px; font-weight: 700; color: {_C['text_dim']}; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 8px; padding-bottom: 4px; border-bottom: 1px solid {_C['border_light']}; }}
.mm-kpi-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0; }}
.mm-kpi {{ flex: 1; min-width: 140px; background: {_C['card2']}; border: 1px solid {_C['border']}; border-radius: 6px; padding: 10px 14px; text-align: center; }}
.mm-kpi .kpi-label {{ font-size: 10px; color: {_C['text_muted']}; text-transform: uppercase; letter-spacing: 0.5px; }}
.mm-kpi .kpi-value {{ font-size: 20px; font-weight: 700; margin: 4px 0; }}
.mm-loading {{ color: {_C['accent']}; font-size: 14px; font-weight: 500; padding: 12px; }}
.mm-loading .step {{ color: {_C['text_muted']}; font-size: 12px; }}
.mm-cot-label {{ background: {_C['card2']}; padding: 6px 10px; border-radius: 4px; font-size: 12px; display: inline-block; margin: 2px; }}
</style>
"""

# Template Plotly unificado (dark, elegante)
_PLOTLY_DARK = go.layout.Template()
_PLOTLY_DARK.layout = go.Layout(
    paper_bgcolor=_C['card'],
    plot_bgcolor=_C['bg'],
    font=dict(family='-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica', size=12, color=_C['text']),
    title=dict(font=dict(size=15, color=_C['text'])),
    xaxis=dict(gridcolor=_C['border_light'], zerolinecolor=_C['border'], linecolor=_C['border']),
    yaxis=dict(gridcolor=_C['border_light'], zerolinecolor=_C['border'], linecolor=_C['border']),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11, color=_C['text_muted'])),
    colorway=[_C['accent'], _C['teal'], _C['orange'], _C['green'], _C['red'],
              _C['purple'], _C['yellow'], _C['pink']],
    hoverlabel=dict(bgcolor=_C['card2'], font_size=12, font_color=_C['text']),
)
DASH_TEMPLATE = _PLOTLY_DARK
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

# ETFs passivos e alavancados
INDEX_PROXY = 'B500 Index'
PASSIVE_ETFS = ['VOO US Equity', 'SPY US Equity', 'IVV US Equity']
LEVERAGED_ETFS = [
    {'ticker': 'SPXL US Equity', 'name': 'SPXL', 'leverage': 3},
    {'ticker': 'UPRO US Equity', 'name': 'UPRO', 'leverage': 3},
    {'ticker': 'SSO US Equity',  'name': 'SSO',  'leverage': 2},
    {'ticker': 'SH US Equity',   'name': 'SH',   'leverage': -1},
    {'ticker': 'SDS US Equity',  'name': 'SDS',  'leverage': -2},
    {'ticker': 'SPXS US Equity', 'name': 'SPXS', 'leverage': -3},
    {'ticker': 'SPXU US Equity', 'name': 'SPXU', 'leverage': -3},
]

# ETFs alavancados estendidos (para flow predictor — inclui NDX e SOX)
LEVERAGED_ETFS_EXT = LEVERAGED_ETFS + [
    {'ticker': 'TQQQ US Equity', 'name': 'TQQQ', 'leverage': 3, 'under': 'NDX Index'},
    {'ticker': 'SQQQ US Equity', 'name': 'SQQQ', 'leverage': -3, 'under': 'NDX Index'},
    {'ticker': 'SOXL US Equity', 'name': 'SOXL', 'leverage': 3, 'under': 'SOX Index'},
    {'ticker': 'SOXS US Equity', 'name': 'SOXS', 'leverage': -3, 'under': 'SOX Index'},
]

# Mapeamento: ticker → futures com dados COT
COT_FUTURES_MAP = {
    'SPX Index': 'ES1 Index', 'NDX Index': 'NQ1 Index',
    'RTY Index': 'RA1 Index', 'INDU Index': 'DM1 Index',
    'SPXL US Equity': 'ES1 Index', 'SPXS US Equity': 'ES1 Index',
    'TQQQ US Equity': 'NQ1 Index', 'SQQQ US Equity': 'NQ1 Index',
    'NG1 Comdty': 'NG1 Comdty', 'CL1 Comdty': 'CL1 Comdty',
    'CO1 Comdty': 'CO1 Comdty', 'GC1 Comdty': 'GC1 Comdty',
    'SI1 Comdty': 'SI1 Comdty', 'HG1 Comdty': 'HG1 Comdty',
    'W 1 Comdty': 'W 1 Comdty', 'S 1 Comdty': 'S 1 Comdty',
    'C 1 Comdty': 'C 1 Comdty',
}

COT_CONTRACTS = {
    'Equity Indices': [
        ('S&P 500 E-mini', 'ES1 Index'),
        ('Nasdaq 100 E-mini', 'NQ1 Index'),
        ('Russell 2000 E-mini', 'RA1 Index'),
    ],
    'Energy': [
        ('WTI Crude', 'CL1 Comdty'),
        ('Brent Crude', 'CO1 Comdty'),
        ('Natural Gas', 'NG1 Comdty'),
    ],
    'Oil': [
        ('Brent', 'CO1 Comdty'),
        ('Gasoil', 'QS1 Comdty'),
        ('WTI', 'CL1 Comdty'),
        ('RBOB', 'XB1 Comdty'),
        ('ULSD', 'HO1 Comdty'),
    ],
    'Power': [
        ('German Power', 'DEBM1 Comdty'),
        ('French Power', 'FRBM1 Comdty'),
        ('Italian Power', 'ITBM1 Comdty'),
        ('ERCOT', 'ERHN1 Comdty'),
    ],
    'Metals': [
        ('Gold', 'GC1 Comdty'),
        ('Silver', 'SI1 Comdty'),
        ('Copper', 'HG1 Comdty'),
    ],
    'Agriculture': [
        ('Wheat', 'W 1 Comdty'),
        ('Soybeans', 'S 1 Comdty'),
        ('Corn', 'C 1 Comdty'),
    ],
}

# Estimativas padrão de AUM ($ USD) para cálculo rápido de fluxo
DEFAULT_AUM = {
    'SPXL': 5e9, 'UPRO': 3e9, 'SSO': 4e9,
    'SH': 2e9, 'SDS': 0.8e9, 'SPXS': 1e9, 'SPXU': 0.8e9,
    'TQQQ': 20e9, 'SQQQ': 5e9, 'SOXL': 10e9, 'SOXS': 3e9,
}

# Mapeamento: trader type → report type correto para COT
COT_TRADER_REPORT_MAP = {
    'Managed Money': 'CFTC_Disaggregated',
    'Swap Dealers': 'CFTC_Disaggregated',
    'Commercial': 'CFTC_Legacy',
    'Non-Commercial': 'CFTC_Legacy',
    'Asset Manager': 'CFTC_TFF',
    'Leveraged Funds': 'CFTC_TFF',
    'Total': None,
}

# Estimativa anual de buyback do SPX (fallback quando BQL não retorna)
SPX_ANNUAL_BUYBACK_EST = 1.0e12  # ~$1T USD

# Layout padrão Plotly (dark elegante para todos os gráficos)
FLOW_FIG_LAYOUT = {
    'template': DASH_TEMPLATE,
    'font': {'family': '-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica', 'size': 12},
    'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': -0.3,
               'xanchor': 'center', 'x': 0.5},
    'margin': {'t': 40, 'b': 40, 'l': 50, 'r': 20},
}


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
    """Ajusta distribuição t-Student e calcula VaR/CVaR a 95% e 99%."""
    tdf, tloc, tscale = student_t.fit(log_returns)
    var_95 = student_t.ppf(0.05, tdf, tloc, tscale)
    cvar_95 = student_t.expect(
        lambda x: x, args=(tdf,), loc=tloc, scale=tscale,
        lb=-np.inf, ub=var_95) / 0.05
    var_99 = student_t.ppf(0.01, tdf, tloc, tscale)
    cvar_99 = student_t.expect(
        lambda x: x, args=(tdf,), loc=tloc, scale=tscale,
        lb=-np.inf, ub=var_99) / 0.01
    return {'tdf': tdf, 'tloc': tloc, 'tscale': tscale,
            'var_95': var_95, 'cvar_95': cvar_95,
            'var_99': var_99, 'cvar_99': cvar_99}


def run_monte_carlo(spot, df, risk_params, n_sims=10000, n_days=5):
    """Simula P&L acumulado do livro do market maker ao longo de n_days."""
    greeks = calculate_all_greeks(spot, df.Strike.values, df.IV.values,
                                  df.Tte.values, df.Type.values)
    oi100 = df.OI.values * 100
    call_sign = np.where(df.Type.values == 'Call', 1, -1)
    dex = (greeks['delta'] * oi100).sum()
    gex_per_pt = (greeks['gamma'] * call_sign * oi100).sum()
    theta_tot = (greeks['theta'] * oi100).sum()

    cum_pnl = np.zeros(n_sims)
    cur_spot = np.full(n_sims, spot)
    for _ in range(n_days):
        day_rets = student_t.rvs(risk_params['tdf'],
                                 loc=risk_params['tloc'],
                                 scale=risk_params['tscale'], size=n_sims)
        new_spot = cur_spot * (1 + day_rets)
        ds = new_spot - cur_spot
        daily_pnl = -(dex * ds + 0.5 * gex_per_pt * ds ** 2) + theta_tot
        cum_pnl += daily_pnl
        cur_spot = new_spot
    return cum_pnl, cur_spot


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
# SEÇÃO 5 — REBALANCEAMENTO DE ETFs (passivos + alavancados)
# ═══════════════════════════════════════════════════════════════════════════════

def _last_spx_rebal_date():
    """Última data de rebalanceamento trimestral do S&P 500 (3ª sexta de Mar/Jun/Set/Dez)."""
    today = pd.Timestamp.now().normalize()
    rebal_months = [3, 6, 9, 12]
    candidates = []
    for year in [today.year - 1, today.year]:
        for month in rebal_months:
            first = pd.Timestamp(year, month, 1)
            dow = first.dayofweek  # Monday=0, Friday=4
            days_to_fri = (4 - dow) % 7
            first_friday = first + pd.Timedelta(days=days_to_fri)
            third_friday = first_friday + pd.Timedelta(weeks=2)
            if third_friday < today:
                candidates.append(third_friday)
    return max(candidates) if candidates else today - pd.Timedelta(days=90)


def _float_weights(idx, as_of=None):
    """Retorna pesos float-adjusted normalizados."""
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
    """Média dos últimos 5 dias de $-volume."""
    usd_val = (bq.data.px_volume(dates=bq.func.range('-5D', '-1D'))
               * bq.data.px_last(dates=bq.func.range('-5D', '-1D')))
    adv_item = bq.func.avg(usd_val)
    return bq.execute(bql.Request(bq.univ.List(tickers_list),
                                  {'ADV5': adv_item},
                                  with_params=BQL_PARAMS))[0].df()['ADV5']


def compute_full_etf_flows(index_proxy=INDEX_PROXY):
    """
    Calcula fluxos de rebalanceamento por ETF passivo + combinado.
    Retorna (flows_dict, summary_df, start_date).
    flows_dict: {'Combined': df, 'VOO US Equity': df, ...}
    """
    start_date = _last_spx_rebal_date().strftime('%Y-%m-%d')

    w0 = _float_weights(index_proxy, start_date)
    w1 = _float_weights(index_proxy)
    delta = pd.DataFrame({'Start': w0, 'Now': w1}).dropna()
    delta['Delta'] = delta['Now'] - delta['Start']

    adv5 = _adv5_usd(delta.index.tolist()).fillna(1)

    aum = bq.execute(bql.Request(
        bq.univ.List(PASSIVE_ETFS),
        {'AUM': bq.data.fund_total_assets()},
        with_params=BQL_PARAMS))[0].df().select_dtypes('number').iloc[:, 0]

    flows = {}
    for etf, aum_val in aum.items():
        t = delta.copy()
        t['Flow_$'] = t['Delta'] * aum_val
        t['PctADV'] = (t['Flow_$'] / adv5.reindex(t.index).fillna(1)) * 100
        flows[etf] = t[t['Flow_$'] != 0].sort_values('Flow_$', ascending=False)

    combo = delta.copy()
    combo['Flow_$'] = sum(flows[etf].reindex(delta.index, fill_value=0)['Flow_$']
                          for etf in flows)
    combo['PctADV'] = (combo['Flow_$'] / adv5.reindex(combo.index).fillna(1)) * 100
    flows['Combined'] = combo[combo['Flow_$'] != 0].sort_values('Flow_$', ascending=False)

    summary_rows = []
    for etf in PASSIVE_ETFS:
        df_f = flows.get(etf, pd.DataFrame(columns=['Flow_$']))
        buy = df_f.loc[df_f['Flow_$'] > 0, 'Flow_$'].sum() if not df_f.empty else 0
        sell = -df_f.loc[df_f['Flow_$'] < 0, 'Flow_$'].sum() if not df_f.empty else 0
        summary_rows.append({'ETF': etf.split()[0], 'Buy_$': buy, 'Sell_$': sell, 'Net_$': buy - sell})
    summary = pd.DataFrame(summary_rows).set_index('ETF')

    return flows, summary, start_date


def compute_leveraged_flows(daily_return):
    """
    Calcula fluxo de rebalanceamento end-of-day dos ETFs alavancados.
    Fórmula: Rebalance_$ = AUM × L × (L-1) × r / (1 + L×r)
    """
    tickers = [e['ticker'] for e in LEVERAGED_ETFS]
    print(f"[LEV] daily_return={daily_return:.6f}")
    aum_df = None
    try:
        aum_df = bq.execute(bql.Request(
            tickers,
            {'AUM': bq.data.fund_total_assets()},
            with_params=BQL_PARAMS))[0].df()
        print(f"[LEV] AUM fetched: {aum_df.shape}")
    except Exception as e:
        print(f"[LEV] AUM fetch failed: {e}")

    rows = []
    for etf in LEVERAGED_ETFS:
        tk = etf['ticker']
        L = etf['leverage']
        r = daily_return
        try:
            aum = pd.to_numeric(aum_df.loc[tk].iloc[0], errors='coerce') if aum_df is not None else np.nan
        except (KeyError, TypeError, IndexError):
            aum = np.nan
        if pd.isna(aum):
            aum = DEFAULT_AUM.get(etf['name'], 1e9)

        if pd.notna(aum) and (1 + L * r) != 0:
            rebal = aum * L * (L - 1) * r / (1 + L * r)
        else:
            rebal = np.nan

        rows.append({
            'ETF': etf['name'],
            'Leverage': f"{L:+d}x",
            'AUM_$': aum,
            'Rebalance_$': rebal,
            'Direção': ('COMPRAR' if pd.notna(rebal) and rebal > 0
                        else 'VENDER' if pd.notna(rebal) and rebal < 0
                        else 'N/A')
        })

    result = pd.DataFrame(rows).set_index('ETF')
    total_flow = result['Rebalance_$'].sum()
    print(f"[LEV] Total flow: ${total_flow:,.0f}")
    return result, total_flow


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5B — PREVISÃO DE REBALANCEAMENTO DO S&P 500 (inclusão/exclusão)
# ═══════════════════════════════════════════════════════════════════════════════

def build_spx_prediction():
    """
    Constrói modelo de previsão de inclusão/exclusão do S&P 500.
    Usa 4 variáveis: FMC, FALR (annual liquidity ratio), FREE_FLOAT_PCT, NET_INC_TTM.
    Retorna (top_in, top_out, model_df).
    Requer sklearn.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn não disponível. Instale com: pip install scikit-learn")

    def _fetch(u, item, name):
        df_ = bq.execute(bql.Request(u, item))[0].df()
        df_ = df_.select_dtypes('number').iloc[:, 0].to_frame(name)
        return df_.loc[~df_.index.duplicated(keep='first')]

    # 1 — Universos
    spx_ids = bq.execute(
        bql.Request(bq.univ.members(['SPX Index']), bq.data.id())
    )[0].df().index.tolist()

    # S&P 500: market cap mínimo ~$18B (regra atualizada)
    base_ids = bq.execute(
        bql.Request(
            bq.univ.filter(
                bq.univ.equitiesuniv(['active', 'primary']),
                bq.func.and_(
                    bq.data.cntry_of_domicile() == 'US',
                    bq.data.cur_mkt_cap() > 1.8e10)),
            bq.data.id())
    )[0].df().index.tolist()

    neg_ids = [t for t in base_ids if t not in spx_ids]
    _random.seed(42)
    neg_sample = _random.sample(neg_ids, k=min(len(neg_ids), len(spx_ids) * 2))
    universe = spx_ids + neg_sample

    # 2 — ADVT anual
    advt_series = (
        bq.data.px_volume(dates=bq.func.range('-365D', '0D')) *
        bq.data.px_last(dates=bq.func.range('-365D', '0D')))
    advt_item = bq.func.sum(advt_series)

    # 3 — Coleta em batches
    records = []
    batch_size = 50
    for i in range(0, len(universe), batch_size):
        u = bq.univ.List(universe[i:i + batch_size])
        dfb = pd.concat([
            _fetch(u, bq.data.cur_mkt_cap(), 'CUR_MKT_CAP'),
            _fetch(u, bq.data.eqy_free_float_pct(), 'FREE_FLOAT_PCT'),
            _fetch(u, advt_item, 'ADVT'),
            _fetch(u, bq.data.net_income(fa_period_offset='0'), 'NI_Q0'),
            _fetch(u, bq.data.net_income(fa_period_offset='1'), 'NI_Q1'),
            _fetch(u, bq.data.net_income(fa_period_offset='2'), 'NI_Q2'),
            _fetch(u, bq.data.net_income(fa_period_offset='3'), 'NI_Q3'),
        ], axis=1, join='inner')
        records.append(dfb)

    df_ = pd.concat(records)

    # 4 — Feature engineering
    df_['FMC'] = df_['CUR_MKT_CAP'] * df_['FREE_FLOAT_PCT'] / 100
    df_['FALR'] = df_['ADVT'] / df_['FMC']
    df_['NET_INC_TTM'] = df_[['NI_Q0', 'NI_Q1', 'NI_Q2', 'NI_Q3']].sum(axis=1)

    roots_all = df_.index.to_series().str.split().str[0]
    roots_spx = pd.Series(spx_ids).str.split().str[0]
    df_['IN_SPX'] = roots_all.isin(set(roots_spx)).astype(int)
    df_ = df_.dropna()

    # 4b — Flag de elegibilidade S&P 500
    # Regras: Free float >= 50%, lucro TTM > 0, último trimestre > 0, FALR >= 0.75
    df_['ELIGIBLE'] = (
        (df_['FREE_FLOAT_PCT'] >= 50) &
        (df_['NET_INC_TTM'] > 0) &
        (df_['NI_Q0'] > 0) &
        (df_['FALR'] >= 0.75) &
        (df_['CUR_MKT_CAP'] >= 1.8e10)
    ).astype(int)

    # 5 — Modelo LogisticRegression
    features = ['FMC', 'FALR', 'FREE_FLOAT_PCT', 'NET_INC_TTM']
    X, y = df_[features], df_['IN_SPX']
    pipe = SkPipeline([
        ('scaler', StandardScaler()),
        ('logit', LogisticRegression(solver='liblinear'))
    ])
    pipe.fit(X, y)
    df_['Prob_In'] = pipe.predict_proba(X)[:, 1]

    # 6 — ExitScore (penalidade por cap, FMC, lucro, liquidez)
    #     Thresholds atualizados conforme metodologia S&P
    cap_thr = 1.8e10   # mínimo de market cap para permanência
    fmc_thr = 0.9e10   # mínimo de float-adjusted market cap
    cap_pen = ((cap_thr - df_['CUR_MKT_CAP']).clip(lower=0)) / cap_thr
    fmc_pen = ((fmc_thr - df_['FMC']).clip(lower=0)) / fmc_thr
    earn_pen = ((df_['NET_INC_TTM'] <= 0) | (df_['NI_Q0'] <= 0)).astype(float)
    float_pen = ((50 - df_['FREE_FLOAT_PCT']).clip(lower=0)) / 50
    liq_pen = ((0.75 - df_['FALR']).clip(lower=0)) / 0.75
    df_['ExitScore'] = (cap_pen + fmc_pen + earn_pen + float_pen + liq_pen) / 5

    # 7 — Rankings
    allowed = df_.index.str.endswith((' US Equity', ' UW Equity', ' UQ Equity'))

    def _dedup_root(d, col):
        t = d.copy()
        t['root'] = t.index.str.split().str[0]
        return t.loc[t.groupby('root')[col].idxmax()]

    # Inclusão: somente elegíveis (FREE_FLOAT>=50, lucro positivo, FALR>=0.75)
    top_in = (df_[(df_['IN_SPX'] == 0) & allowed & (df_['ELIGIBLE'] == 1)]
              .pipe(_dedup_root, 'Prob_In')
              .sort_values('Prob_In', ascending=False)
              .head(30))

    top_out = (df_[(df_['IN_SPX'] == 1) & allowed]
               .pipe(_dedup_root, 'ExitScore')
               .sort_values('ExitScore', ascending=False)
               .head(30))

    return top_in, top_out, df_


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5C — COT ENGINE (Commitment of Traders)
# ═══════════════════════════════════════════════════════════════════════════════

def _fp_last_numeric_col(df_in):
    for c in df_in.columns[::-1]:
        if pd.api.types.is_numeric_dtype(df_in[c]):
            return c
    return None


def _fp_get_data(universe, data_items, with_params=None, preferences=None):
    """Executa BQL Request e retorna DataFrame com multi-index [ID, Date]."""
    if isinstance(universe, str):
        universe = [universe]
    req = bql.Request(universe, data_items,
                      with_params=with_params, preferences=preferences)
    response = bq.execute(req)
    df_r = pd.concat([di.df() for di in response], axis=1)
    df_r = df_r.loc[:, ~df_r.columns.duplicated()]
    if 'DATE' in df_r.columns:
        df_r = df_r.set_index('DATE', append=True)
    # Selecionar colunas disponíveis (ignorar as não retornadas)
    avail = [k for k in data_items.keys() if k in df_r.columns]
    if avail:
        df_r = df_r[avail]
    df_r.index.names = ['ID', 'Date']
    return df_r


# ── Helpers COT v4 ──
QUARTERLY_MONTHS = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}

SPEC_TRADER_TYPES = {
    'cftc_disaggregated': ['MANAGED_MONEY'],
    'cftc_tff': ['ASSET_MANAGER', 'LEVERAGED_FUNDS'],
    'cftc_legacy': ['NON_COMMERCIAL'],
}


def _concat_bql_response(resp):
    """Concat DataItems de um BQL response, usando integer index."""
    dfs = [di.df().reset_index() for di in resp]
    raw = pd.concat([d.reset_index(drop=True) for d in dfs], axis=1)
    return raw.loc[:, ~raw.columns.duplicated()]


def _has_positions_data(df):
    """Verifica se o BQL result tem dados reais de posição."""
    if df is None or df.empty:
        return False
    for c in df.columns:
        cl = str(c).lower()
        if 'value' in cl or 'position' in cl:
            vals = pd.to_numeric(df[c], errors='coerce')
            if vals.notna().any() and (vals != 0).any():
                return True
    return False


def resolve_specific_contract(generic_ticker):
    """ES1 Index → ['ESH6 Index', 'ESM6 Index'] baseado na data atual."""
    parts = generic_ticker.split()
    if len(parts) < 2:
        return []
    root_num = parts[0]
    suffix = ' '.join(parts[1:])
    root = root_num.rstrip('0123456789')
    if not root:
        return []
    now = pd.Timestamp.now()
    month, year = now.month, now.year
    sorted_qm = sorted(QUARTERLY_MONTHS.keys())
    candidates = []
    for qm in sorted_qm:
        if month <= qm:
            code = QUARTERLY_MONTHS[qm]
            candidates.append(f"{root}{code}{year % 10} {suffix}")
            if len(candidates) >= 2:
                break
    if len(candidates) < 2:
        next_year = (year + 1) % 10
        for qm in sorted_qm:
            code = QUARTERLY_MONTHS[qm]
            candidates.append(f"{root}{code}{next_year} {suffix}")
            if len(candidates) >= 2:
                break
    print(f"[resolve] {generic_ticker} → candidatos: {candidates}")
    return candidates


def _to_bql_date(s):
    """Converte data YYYYMMDD (do widget) para formato BQL relativo."""
    if not s or not isinstance(s, str):
        return s
    if s in ('0D', '0d') or (s.startswith('-') and s[-1] in 'YyDd'):
        return s
    try:
        dt = pd.to_datetime(s)
        days = (pd.Timestamp.now() - dt).days
        if days <= 0:
            return '0D'
        years = days / 365.25
        if years >= 1:
            return f'-{int(years) + 1}Y'
        return f'-{days}D'
    except Exception:
        return s


def _try_cot_query(ticker, rpt, start, end, with_dates=True):
    """Tenta uma query COT. Retorna DataFrame ou None."""
    try:
        bql_s, bql_e = _to_bql_date(start), _to_bql_date(end)
        if with_dates:
            q = f"""
let(#p = cot_position(report_type={rpt}, direction=all,
                      trader_type=all, commitment_type=futures,
                      dates=range({bql_s},{bql_e}));)
for('{ticker}') get(#p().date, #p().trader_type, #p().direction,
                    #p().value, #p().change)
"""
        else:
            q = f"""
let(#p = cot_position(report_type={rpt}, direction=all,
                      trader_type=all, commitment_type=futures);)
for('{ticker}') get(#p().date, #p().trader_type, #p().direction,
                    #p().value, #p().change)
"""
        label = f"{rpt} + {ticker}" + (" + dates" if with_dates else "")
        print(f"[COT] Tentando {label}…")
        resp = bq.execute(q)
        raw = _concat_bql_response(resp)
        print(f"[COT]   shape={raw.shape}")
        if not _has_positions_data(raw):
            print(f"[COT]   → sem dados de posição reais")
            return None
        print(f"[COT]   ✓ has real data")
        return raw
    except Exception as e:
        print(f"[COT]   ✗ {e}")
        return None


def has_cot(ticker):
    """Verifica se o ticker possui dados COT. Retorna (True, futures_ticker) ou (False, None)."""
    if ticker in COT_FUTURES_MAP:
        return True, COT_FUTURES_MAP[ticker]
    t = ticker.strip()
    if t.endswith('Comdty') or t.endswith('Index'):
        return True, ticker
    return False, None


def fetch_cot_data(futures_ticker, start='-2Y', end='0D'):
    """Busca COT histórico: tenta genérico + específico, filtra spec traders."""
    tickers = [futures_ticker] if isinstance(futures_ticker, str) else list(futures_ticker)

    # Build lista de tickers para tentar: genérico + contratos específicos
    tickers_to_try = list(tickers)
    for t in tickers:
        specific = resolve_specific_contract(t)
        tickers_to_try.extend(specific)

    raw = None
    used_rpt = ''
    used_ticker = ''

    for t in tickers_to_try:
        for rpt in ('cftc_disaggregated', 'cftc_tff', 'cftc_legacy'):
            result = _try_cot_query(t, rpt, start, end, with_dates=True)
            if result is None:
                result = _try_cot_query(t, rpt, start, end, with_dates=False)
            if result is not None:
                raw = result
                used_rpt = rpt
                used_ticker = t
                break
        if raw is not None:
            break

    if raw is None:
        print("[COT] Nenhum dado retornado")
        return pd.DataFrame()

    # ── Renomear colunas ──
    col_map = {}
    for c in raw.columns:
        cl = str(c).lower()
        if 'date' in cl:
            col_map[c] = 'Date'
        elif 'trader_type' in cl:
            col_map[c] = 'TraderType'
        elif 'direction' in cl:
            col_map[c] = 'Direction'
        elif 'change' in cl:
            col_map[c] = 'Pos_Chg'
        elif 'value' in cl:
            col_map[c] = 'Positions'
    raw = raw.rename(columns=col_map)
    if 'ID' not in raw.columns:
        raw['ID'] = used_ticker
    if 'Date' in raw.columns:
        raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
    for nc in ('Positions', 'Pos_Chg'):
        if nc in raw.columns:
            raw[nc] = pd.to_numeric(raw[nc], errors='coerce')

    if 'Direction' in raw.columns:
        raw['Direction'] = raw['Direction'].astype(str).str.strip().str.upper()
    if 'TraderType' in raw.columns:
        raw['TraderType'] = raw['TraderType'].astype(str).str.strip().str.upper()

    print(f"[COT] Rename: cols={list(raw.columns)}, report={used_rpt}, ticker={used_ticker}")

    # ── Filtrar trader types especulativos ──
    spec_types = SPEC_TRADER_TYPES.get(used_rpt, [])
    if 'TraderType' in raw.columns and spec_types:
        n_before = len(raw)
        raw = raw[raw['TraderType'].isin(spec_types)]
        print(f"[COT] Filtro spec ({spec_types}): {n_before} → {len(raw)} rows")
    if raw.empty:
        print("[COT] Vazio após filtro de trader types")
        return pd.DataFrame()

    # ── Drop rows com Positions NaN/zero (dias sem report) ──
    if 'Positions' in raw.columns:
        raw = raw.dropna(subset=['Positions'])
        raw = raw[raw['Positions'] != 0]
        print(f"[COT] Após drop zeros/NaN: {len(raw)} rows")
    if raw.empty:
        print("[COT] Vazio após drop zeros")
        return pd.DataFrame()

    # ── Filtrar datas ──
    def _pdt(s):
        if not s:
            return None
        s = str(s)
        if s in ('0D', '0d'):
            return pd.Timestamp.now()
        if s.startswith('-') and s[-1] in 'Yy':
            return pd.Timestamp.now() - pd.DateOffset(years=int(s[1:-1]))
        if s.startswith('-') and s[-1] in 'Dd':
            return pd.Timestamp.now() - pd.Timedelta(days=int(s[1:-1]))
        return pd.to_datetime(s, errors='coerce')
    if 'Date' in raw.columns:
        dt_s, dt_e = _pdt(start), _pdt(end)
        if dt_s is not None:
            raw = raw[raw['Date'] >= dt_s]
        if dt_e is not None:
            raw = raw[raw['Date'] <= dt_e]
    if raw.empty:
        print("[COT] Vazio após filtro de datas")
        return pd.DataFrame()

    # ── Pivot: spec traders por Date × Direction ──
    idx_cols = [c for c in ['ID', 'Date'] if c in raw.columns]
    if not idx_cols or 'Positions' not in raw.columns:
        print(f"[COT] Colunas insuficientes: {list(raw.columns)}")
        return pd.DataFrame()

    if 'Direction' in raw.columns:
        grouped = raw.groupby(idx_cols + ['Direction'])['Positions'].sum().reset_index()
        piv = grouped.pivot_table(index=idx_cols, columns='Direction',
                                  values='Positions', aggfunc='sum')
        piv.columns = [f'Positions - {c.title()}' for c in piv.columns]
        df = piv
    else:
        df = raw.groupby(idx_cols)['Positions'].sum().to_frame()

    # ── Compute Net = Long + Short ──
    if 'Positions - Net' in df.columns:
        df['Positions'] = df['Positions - Net']
    elif 'Positions - Long' in df.columns and 'Positions - Short' in df.columns:
        df['Positions'] = df['Positions - Long'] + df['Positions - Short']
    elif 'Positions - Long' in df.columns:
        df['Positions'] = df['Positions - Long']
    else:
        df['Positions'] = df.iloc[:, 0]

    df = df[df['Positions'] != 0]
    if df.empty:
        print("[COT] Vazio após drop net == 0")
        return pd.DataFrame()

    # ── Price & Open Interest (usa genérico para preço) ──
    price_ticker = futures_ticker if isinstance(futures_ticker, str) else futures_ticker[0]
    try:
        dates_bql = bq.func.range(_to_bql_date(start), _to_bql_date(end), frq='d')
        px_items = {'Price': bq.data.px_last(fill='prev'),
                    'Open Interest': bq.data.fut_aggte_open_int()}
        px_df = _fp_get_data(price_ticker, px_items,
                             with_params={'currency': 'usd', 'dates': dates_bql})
        px_sr = px_df.droplevel('ID').sort_index()
        cot_dates = df.index.get_level_values('Date')
        px_a = px_sr.reindex(cot_dates, method='ffill')
        for c in ['Price', 'Open Interest']:
            if c in px_a.columns:
                df[c] = px_a[c].values
    except Exception:
        df['Price'] = np.nan
        df['Open Interest'] = np.nan

    # ── week/year ──
    dt_idx = pd.to_datetime(df.index.get_level_values('Date'))
    iso = dt_idx.isocalendar()
    df['week'] = iso.week.values
    df['year'] = iso.year.values
    if 'Positions - Short' in df.columns:
        df['Positions - Short'] = df['Positions - Short'] * -1
    print(f"[COT] Final: {df.shape}, cols: {list(df.columns)}")
    return df.dropna(subset=['Positions'])


def aggregate_cot(df_cot):
    """Agrega COT por data (caso de múltiplos contratos)."""
    cols_skip = ['year', 'week', 'Price']
    aggs = {c: 'sum' for c in df_cot.columns if c not in cols_skip}
    out = df_cot.groupby('Date').agg(
        **{**{c: (c, agg) for c, agg in aggs.items()},
           'week': ('week', 'first'),
           'year': ('year', 'first'),
           'Price': ('Price', 'mean')})
    out['Basket Returns'] = (1 + out['Price'].pct_change()).cumprod() - 1
    return out


def cot_seasonality(df_cot):
    """Estatísticas semanais de sazonalidade COT."""
    cols = ['Positions', 'Open Interest', 'week']
    avail = [c for c in cols if c in df_cot.columns]
    return (df_cot[avail].groupby('week')
            .agg(['mean', 'max', 'sum', 'min'])
            .rename(columns=lambda x: x.title()))


def cot_summary_stats(df_cot):
    """Estatísticas resumo: último valor, WoW change, percentil, mediana, z-score."""
    summary = pd.Series(dtype=float)
    for col in ['Positions']:
        if col not in df_cot.columns:
            continue
        s = df_cot[col].dropna()
        if len(s) < 3:
            continue
        stats = pd.Series({
            col: s.iloc[-1],
            f'{col} WoW Change': s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else np.nan,
            f'{col} Percentile': (s.rank(pct=True).iloc[-1] * 100),
            f'{col} Median': s.median(),
            f'{col} Z-Score': (s.iloc[-1] - s.mean()) / max(s.std(), 1e-9),
        })
        summary = pd.concat([summary, stats])
    return summary


def safe_fetch_cot(ticker, start='-2Y', end='0D'):
    """Tenta buscar COT. Retorna None se não disponível."""
    ok, fut = has_cot(ticker)
    if not ok:
        return None
    try:
        df_r = fetch_cot_data(fut, start=start, end=end)
        if df_r is None or df_r.empty:
            return None
        return aggregate_cot(df_r)
    except Exception as e:
        print(f"⚠️ COT fetch error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5D — ESTIMATIVA DE BUYBACK
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=128)
def fetch_buyback_data(ticker):
    """Busca dados de buyback via BQL. Retorna dict com campos disponíveis."""
    fields_try = {
        'announced': 'ANNOUNCED_BUYBACK_AMT',
        'mkt_cap': 'CUR_MKT_CAP',
        'sh_out': 'EQY_SH_OUT',
        'px': 'PX_LAST',
        'adv20': 'VOLUME_AVG_20D',
    }
    result = {}
    for key, field_name in fields_try.items():
        try:
            fld = getattr(bq.data, field_name.lower(), None)
            if fld is None:
                continue
            req = bql.Request([ticker], fld())
            df_r = bq.execute(req)[0].df()
            if df_r is not None and not df_r.empty:
                vcol = _fp_last_numeric_col(df_r)
                if vcol:
                    val = pd.to_numeric(df_r[vcol], errors='coerce').dropna()
                    if len(val) > 0:
                        result[key] = float(val.iloc[-1])
        except Exception:
            continue
    return result


def estimate_buyback_flow(ticker, horizon_days=252):
    """
    Estima fluxo diário de buyback.
    Usa _adv5_usd para % ADV (mesma lógica do rebalanceamento).
    """
    data = fetch_buyback_data(ticker)
    announced = data.get('announced', 0)
    if not announced or announced <= 0:
        return {'daily_est': 0, 'pct_adv_est': 0,
                'confidence': 'none', 'announced': 0}
    execution_rate = 0.80
    daily_est = (announced * execution_rate) / max(horizon_days, 1)
    try:
        adv5 = _adv5_usd([ticker])
        adv_usd = float(adv5.iloc[0]) if len(adv5) > 0 else 0
    except Exception:
        adv_usd = 0
    pct_adv = (announced / adv_usd * 100) if adv_usd > 0 else np.nan
    return {'daily_est': daily_est, 'pct_adv_est': pct_adv,
            'confidence': 'low', 'announced': announced,
            'mkt_cap': data.get('mkt_cap', np.nan)}


def estimate_index_buyback_flow(index_ticker='SPX Index', top_n=50):
    """Estima fluxo de buyback para os maiores membros de um índice.

    Tenta buscar announced_buyback_amt via BQL. Se não disponível,
    faz fallback para estimativa baseada em market cap (~2% cap/ano).
    Usa _adv5_usd (mesmo ADV do rebalanceamento) para calcular % ADV.
    """
    try:
        uni = bq.univ.members(index_ticker)
        items = {'cap': bq.data.cur_mkt_cap()}

        # Tentar múltiplos campos de buyback
        bb_field_name = None
        for fname in ['announced_buyback_amt', 'bs_sh_repurchase']:
            try:
                fld = getattr(bq.data, fname, None)
                if fld is not None:
                    items['buyback_raw'] = fld()
                    bb_field_name = fname
                    break
            except Exception:
                continue

        req = bql.Request(uni, items, with_params=BQL_PARAMS)
        df_r = bq.execute(req)[0].df()
        if df_r is None or df_r.empty:
            return pd.DataFrame()

        # Renomear colunas robustamente
        rename = {}
        for c in df_r.columns:
            cl = str(c).lower()
            if 'mkt' in cl and 'cap' in cl:
                rename[c] = 'cap'
            elif 'buyback' in cl or 'repurchase' in cl:
                rename[c] = 'buyback'
        df_r = df_r.rename(columns=rename)

        if 'cap' in df_r.columns:
            df_r['cap'] = pd.to_numeric(df_r['cap'], errors='coerce')
            df_r = df_r.dropna(subset=['cap'])
            df_r = df_r.nlargest(top_n, 'cap')

        # Verificar se dados de buyback reais estão disponíveis
        has_bb = ('buyback' in df_r.columns and
                  pd.to_numeric(df_r['buyback'], errors='coerce').abs().sum() > 0)

        if has_bb:
            df_r['buyback'] = pd.to_numeric(df_r['buyback'], errors='coerce').fillna(0).abs()
            df_r['confidence'] = 'low'
        else:
            df_r['buyback'] = df_r['cap'].fillna(0) * 0.02
            df_r['confidence'] = 'estimated'

        df_r['daily_est'] = df_r['buyback'] * 0.80 / TRADING_DAYS

        # % ADV usando _adv5_usd (mesma lógica do rebalanceamento)
        try:
            tickers_list = df_r.index.get_level_values('ID').unique().tolist()
            adv5 = _adv5_usd(tickers_list)
            df_r['pct_adv_est'] = (df_r['buyback']
                                   / adv5.reindex(df_r.index).replace(0, np.nan)) * 100
        except Exception:
            df_r['pct_adv_est'] = np.nan

        out_cols = [c for c in ['cap', 'buyback', 'daily_est', 'pct_adv_est', 'confidence']
                    if c in df_r.columns]
        return df_r[out_cols]
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5E — FLOW SCORING & AGREGAÇÃO PREDITIVA
# ═══════════════════════════════════════════════════════════════════════════════

def flow_zscore(current, history):
    """Z-score do valor atual relativo ao histórico."""
    if history is None or len(history) < 5:
        return np.nan
    mu = history.mean()
    sigma = history.std()
    if sigma < 1e-12:
        return 0.0
    return (current - mu) / sigma


def compute_leveraged_flow_simple(daily_return, aum_estimates=None):
    """Calcula fluxo total de rebalanceamento sem BQL call."""
    if aum_estimates is None:
        aum_estimates = DEFAULT_AUM
    total = 0
    for etf in LEVERAGED_ETFS_EXT:
        L = etf['leverage']
        r = daily_return
        aum = aum_estimates.get(etf['name'], 1e9)
        denom = 1 + L * r
        if abs(denom) > 1e-12:
            total += aum * L * (L - 1) * r / denom
    return total


def build_flow_history(ticker='SPX Index', lookback=252):
    """Constrói série histórica de fluxo de rebalanceamento de ETFs alavancados."""
    item = bq.data.px_last(dates=bq.func.range(f'-{lookback + 10}D', '0D'), fill='PREV')
    try:
        df_r = bq.execute(bql.Request([ticker], item))[0].df()
        if df_r is None or df_r.empty:
            return pd.DataFrame()
        dcol = next((c for c in df_r.columns if 'date' in str(c).lower()), None)
        vcol = _fp_last_numeric_col(df_r)
        if dcol is None or vcol is None:
            return pd.DataFrame()
        px = df_r.set_index(dcol)[vcol].astype(float).sort_index()
        px.index = pd.to_datetime(px.index)
    except Exception:
        return pd.DataFrame()
    rets = px.pct_change().dropna()
    flows = rets.apply(compute_leveraged_flow_simple)
    flows.name = 'LevETF_Flow'
    return pd.DataFrame({'Return': rets, 'LevETF_Flow': flows})


def compute_flow_score(leveraged_flow, buyback_daily=0, cot_net_change=0,
                       passive_etf_flow=0, history_leveraged=None,
                       history_cot=None, dealer_flow=0, volctrl_flow=0,
                       cta_flow=0, rp_flow=0,
                       history_dealer=None, history_volctrl=None):
    """
    Computa score combinado de fluxo contratado — 8 componentes.
    Pesos: CTA 22%, Dealer 18%, VolCtrl 12%, Risk Parity 12%,
           ETFs Alav 12%, Buyback 8%, COT 10%, ETFs Passivos 6%.
    """
    z_lev = flow_zscore(leveraged_flow, history_leveraged) if history_leveraged is not None else 0
    z_cot = flow_zscore(cot_net_change, history_cot) if history_cot is not None else 0
    z_buyback = np.clip(buyback_daily / 1e8, -3, 3) if buyback_daily else 0
    z_passive = np.clip(passive_etf_flow / 1e9, -3, 3) if passive_etf_flow else 0
    z_dealer = np.clip(dealer_flow / 1e9, -3, 3) if dealer_flow else 0
    z_volctrl = np.clip(volctrl_flow / 1e9, -3, 3) if volctrl_flow else 0
    z_cta = np.clip(cta_flow / 1e9, -3, 3) if cta_flow else 0
    z_rp = np.clip(rp_flow / 1e9, -3, 3) if rp_flow else 0

    w_cta, w_deal, w_vc, w_rp = 0.22, 0.18, 0.12, 0.12
    w_lev, w_buy, w_cot, w_passive = 0.12, 0.08, 0.10, 0.06
    if history_cot is None or len(history_cot) < 5:
        w_cot = 0.0
        w_cta, w_deal, w_rp = 0.26, 0.20, 0.14
        w_vc, w_lev, w_buy, w_passive = 0.14, 0.14, 0.08, 0.04

    combined = (w_lev * z_lev + w_buy * z_buyback + w_cot * z_cot
                + w_passive * z_passive + w_deal * z_dealer + w_vc * z_volctrl
                + w_cta * z_cta + w_rp * z_rp)

    if combined > 0.5:
        direction = "BULLISH"
    elif combined < -0.5:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    prob_up = 1.0 / (1.0 + math.exp(-combined))
    return {
        'z_leveraged': z_lev, 'z_buyback': z_buyback,
        'z_cot': z_cot, 'z_passive_etf': z_passive,
        'z_dealer': z_dealer, 'z_volctrl': z_volctrl,
        'z_cta': z_cta, 'z_rp': z_rp,
        'combined_score': combined, 'direction': direction,
        'prob_up': prob_up, 'prob_down': 1.0 - prob_up,
        'weights': {'leveraged': w_lev, 'buyback': w_buy,
                    'cot': w_cot, 'passive_etf': w_passive,
                    'dealer': w_deal, 'volctrl': w_vc,
                    'cta': w_cta, 'rp': w_rp},
    }


def compute_dealer_hedging_flow(gex_per_pt, daily_price_change, spot):
    """
    Estima fluxo de hedging de dealers/market makers em opções.
    Quando dealers estão short gamma, um move de +1% os força a comprar
    (delta hedge), amplificando o movimento. Short gamma → pro-cíclico.
    Flow ≈ -GEX_per_pt × ΔS (dealers hedgeiam no sentido oposto à gamma).
    """
    if pd.isna(gex_per_pt) or pd.isna(daily_price_change) or pd.isna(spot):
        return 0
    flow = -gex_per_pt * daily_price_change
    return flow


# ── Volatility Control Fund Flows ─────────────────────────────────
# Vol-targeting strategies adjust equity exposure inversely to realized vol.
# When RV rises, they sell; when it falls, they buy.
# Typical global AUM: ~$350B-$500B across pension and systematic funds.
VOL_CTRL_AUM = {5: 100e9, 10: 150e9, 15: 100e9}  # target_vol% → AUM estimate
VOL_CTRL_MAX_LEV = 2.0  # Maximum leverage cap


def compute_vol_control_flow(rv_current, rv_prev, target_vols=None):
    """
    Estima fluxo dos fundos de controle de volatilidade.
    rv_current e rv_prev devem ser vol anualizada (ex: 0.15 = 15%).
    Quando vol sobe → exposure cai → fundos vendem (flow negativo).
    """
    if target_vols is None:
        target_vols = [5, 10, 15]
    if pd.isna(rv_current) or pd.isna(rv_prev) or rv_current < 1e-6 or rv_prev < 1e-6:
        return {'total': 0, 'detail': {}}

    detail = {}
    total = 0
    for tv in target_vols:
        tv_dec = tv / 100.0
        exp_new = min(tv_dec / rv_current, VOL_CTRL_MAX_LEV) if rv_current > 1e-6 else VOL_CTRL_MAX_LEV
        exp_old = min(tv_dec / rv_prev, VOL_CTRL_MAX_LEV) if rv_prev > 1e-6 else VOL_CTRL_MAX_LEV
        aum = VOL_CTRL_AUM.get(tv, 100e9)
        flow = aum * (exp_new - exp_old)
        detail[f'{tv}%'] = {'exposure_new': exp_new, 'exposure_old': exp_old,
                            'flow': flow, 'aum': aum}
        total += flow
    return {'total': total, 'detail': detail}


# ── Risk Parity Model ──────────────────────────────────────────────
# Baseado em BofA Systematic Flows Monitor:
# - 3 asset classes: Equities (SPX), Bonds (10Y UST), Commodities (GSCI)
# - Alocação inversamente proporcional à volatilidade (↓vol → ↑alocação)
# - Rebalanceamento mensal usando 3M de dados para vol e correlação
# - AUM estimado: $200B–$750B (BofA usa $200B nos exhibits)
# - Risk targets: 10%, 12%, 15% vol; max leverage 1.5x/1.5x/3.0x
RISK_PARITY_AUM = 200e9  # $200B total (BofA Exhibit 3)
RISK_PARITY_TARGETS = {
    10: {'aum_share': 0.33, 'max_lev': 1.5},
    12: {'aum_share': 0.34, 'max_lev': 1.5},
    15: {'aum_share': 0.33, 'max_lev': 3.0},
}


def compute_risk_parity_flow(rv_equity, rv_equity_prev,
                             rv_bonds=None, rv_bonds_prev=None,
                             rv_commod=None, rv_commod_prev=None):
    """
    Estima fluxo de risk parity para equities.
    Alocação equity ∝ 1/vol_equity. Quando vol equity sobe,
    alocação cai → vendem equities.
    Se vol de bonds/commodities disponível, usa pesos relativos;
    caso contrário, usa só equity vol como proxy.
    """
    if pd.isna(rv_equity) or pd.isna(rv_equity_prev) or rv_equity < 1e-6:
        return {'total': 0, 'detail': {}, 'eq_alloc_new': 0, 'eq_alloc_old': 0}

    def _eq_weight(rv_eq, rv_bd, rv_cm):
        """Peso equity = (1/vol_eq) / sum(1/vol_i)."""
        inv_eq = 1.0 / max(rv_eq, 0.01)
        inv_bd = 1.0 / max(rv_bd, 0.01) if rv_bd and rv_bd > 1e-6 else inv_eq * 2.5
        inv_cm = 1.0 / max(rv_cm, 0.01) if rv_cm and rv_cm > 1e-6 else inv_eq * 0.8
        total_inv = inv_eq + inv_bd + inv_cm
        return inv_eq / total_inv if total_inv > 0 else 0.33

    eq_w_new = _eq_weight(rv_equity, rv_bonds, rv_commod)
    eq_w_old = _eq_weight(rv_equity_prev, rv_bonds_prev, rv_commod_prev)

    detail = {}
    total = 0
    for tv, params in RISK_PARITY_TARGETS.items():
        aum = RISK_PARITY_AUM * params['aum_share']
        tv_dec = tv / 100.0
        # Portfolio vol ≈ weighted avg of component vols (simplificação)
        port_vol_new = rv_equity * eq_w_new + 0.05 * (1 - eq_w_new)
        port_vol_old = rv_equity_prev * eq_w_old + 0.05 * (1 - eq_w_old)
        lev_new = min(tv_dec / max(port_vol_new, 0.01), params['max_lev'])
        lev_old = min(tv_dec / max(port_vol_old, 0.01), params['max_lev'])
        eq_exp_new = lev_new * eq_w_new
        eq_exp_old = lev_old * eq_w_old
        flow = aum * (eq_exp_new - eq_exp_old)
        detail[f'{tv}%'] = {
            'aum': aum, 'leverage_new': lev_new, 'leverage_old': lev_old,
            'eq_alloc_new': eq_w_new, 'eq_alloc_old': eq_w_old,
            'eq_exposure_new': eq_exp_new, 'eq_exposure_old': eq_exp_old,
            'flow': flow}
        total += flow

    return {'total': total, 'detail': detail,
            'eq_alloc_new': eq_w_new, 'eq_alloc_old': eq_w_old}


# ── CTA Trend Following Model ─────────────────────────────────────
# Baseado em BofA Systematic Flows Monitor:
# - Trend strength via coleção de cruzamentos de médias móveis (near vs far)
# - Position sizing: trend_strength / volatility (trend/vol ratio)
# - AUM ~$340B (BarclayHedge 2024-Q4), ~25% alocação em equities
# - CTAs ajustam diariamente, são os mais rápidos a responder
CTA_AUM = 340e9
CTA_EQUITY_ALLOC = 0.25  # ~25% da carteira em equities
CTA_MA_PAIRS = [(5, 20), (5, 60), (10, 60), (20, 120), (20, 200)]


def compute_cta_trend_strength(prices, ma_pairs=None):
    """
    Calcula trend strength usando cruzamentos de médias móveis.
    Para cada par (curta, longa): +1 se MA curta > MA longa, -1 caso contrário.
    Média ponderada pela volatilidade relativa → score contínuo em [-1, +1].
    Ref: BofA "Trends aren't going out of fashion" (2017).
    """
    if ma_pairs is None:
        ma_pairs = CTA_MA_PAIRS
    if len(prices) < max(p[1] for p in ma_pairs):
        return 0
    scores = []
    for short_w, long_w in ma_pairs:
        ma_short = prices.rolling(short_w).mean().iloc[-1]
        ma_long = prices.rolling(long_w).mean().iloc[-1]
        if pd.isna(ma_short) or pd.isna(ma_long) or ma_long == 0:
            continue
        # Prorate pelo spread relativo (torna mais contínuo)
        spread = (ma_short - ma_long) / ma_long
        scores.append(np.clip(spread * 100, -1, 1))
    if not scores:
        return 0
    return float(np.mean(scores))


def compute_cta_flow(prices, rv_current, target_vol=0.10):
    """
    Estima fluxo de CTA trend followers para equities.
    Position = trend_strength × (target_vol / realized_vol), capped em [-2, +2].
    Flow = AUM × equity_alloc × Δposition.
    """
    if len(prices) < 201 or rv_current < 1e-6:
        return {'flow': 0, 'trend_today': 0, 'trend_prev': 0,
                'pos_today': 0, 'pos_prev': 0}

    trend_today = compute_cta_trend_strength(prices)
    trend_prev = compute_cta_trend_strength(prices.iloc[:-1])

    pos_today = np.clip(trend_today * (target_vol / rv_current), -2, 2)

    # RV do dia anterior (rolling 63d para capturar ~3M de vol)
    rets_prev = prices.iloc[:-1].pct_change().dropna()
    rv_prev = rets_prev.iloc[-63:].std() * np.sqrt(252) if len(rets_prev) >= 63 else rv_current
    pos_prev = np.clip(trend_prev * (target_vol / max(rv_prev, 1e-6)), -2, 2)

    flow = CTA_AUM * CTA_EQUITY_ALLOC * (pos_today - pos_prev)

    return {
        'flow': flow,
        'trend_today': trend_today,
        'trend_prev': trend_prev,
        'pos_today': pos_today,
        'pos_prev': pos_prev,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5F — VISUALIZAÇÕES DO FLOW PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

def _flow_border(fig_widget):
    return wd.VBox([fig_widget], layout={
        'border': f'1px solid {_C["border"]}', 'margin': '6px',
        'padding': '10px', 'width': '98%',
        'border_radius': '8px'})


def fp_plot_score_gauge(score):
    """Gauge chart do score combinado de fluxo."""
    prob = score.get('prob_up', 0.5)
    direction = score.get('direction', 'NEUTRAL')
    colors = {'BULLISH': _C['green'], 'BEARISH': _C['red'], 'NEUTRAL': _C['text_muted']}
    fig = go.FigureWidget(go.Indicator(
        mode="gauge+number+delta", value=prob * 100,
        title={'text': f"Flow Score: {direction}", 'font': {'color': _C['text_muted']}},
        number={'font': {'color': _C['text']}},
        delta={'reference': 50, 'increasing': {'color': _C['green']},
               'decreasing': {'color': _C['red']}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': _C['text_muted'],
                     'tickfont': {'color': _C['text_muted']}},
            'bar': {'color': colors.get(direction, _C['text_muted']), 'thickness': 0.3},
            'bgcolor': _C['card2'],
            'borderwidth': 1, 'bordercolor': _C['border'],
            'steps': [
                {'range': [0, 30], 'color': '#3a1a1a'},
                {'range': [30, 70], 'color': '#3a3520'},
                {'range': [70, 100], 'color': '#1a3a2a'}],
            'threshold': {'line': {'color': _C['text'], 'width': 3},
                          'thickness': 0.8, 'value': prob * 100}
        }))
    fig.update_layout(height=280, margin=dict(t=50, b=20, l=20, r=20),
                      **{k: v for k, v in FLOW_FIG_LAYOUT.items()
                         if k != 'margin'})
    return fig


def fp_plot_components_bar(score):
    """Barras dos componentes do flow score."""
    components = {
        'CTA': score.get('z_cta', 0),
        'Dealer/MM': score.get('z_dealer', 0),
        'Vol Ctrl': score.get('z_volctrl', 0),
        'Risk Parity': score.get('z_rp', 0),
        'ETFs Alav.': score.get('z_leveraged', 0),
        'ETFs Passivos': score.get('z_passive_etf', 0),
        'Buyback': score.get('z_buyback', 0),
        'COT': score.get('z_cot', 0),
    }
    weights = score.get('weights', {})
    names = list(components.keys())
    values = list(components.values())
    w_vals = [weights.get(k, 0) for k in ['cta', 'dealer', 'volctrl', 'rp', 'leveraged', 'passive_etf', 'buyback', 'cot']]
    colors_bar = [_C['accent'] if v >= 0 else _C['red'] for v in values]
    fig = go.FigureWidget()
    fig.add_trace(go.Bar(x=names, y=values, marker_color=colors_bar,
                         name='Z-Score', text=[f'{v:+.2f}' for v in values],
                         textposition='outside'))
    fig.add_trace(go.Scatter(x=names, y=w_vals, name='Peso',
                             yaxis='y2', mode='markers+text',
                             text=[f'{w:.0%}' for w in w_vals],
                             textposition='top center',
                             marker=dict(size=10, color=_C['text_muted'])))
    fig.update_layout(title='Componentes do Flow Score',
                      yaxis_title='Z-Score',
                      yaxis2=dict(overlaying='y', side='right',
                                  title='Peso', range=[0, 1]),
                      **FLOW_FIG_LAYOUT)
    return fig


def fp_plot_flow_history(flow_hist):
    """Série histórica de fluxo vs retorno."""
    if flow_hist.empty:
        fig = go.FigureWidget()
        fig.update_layout(title="Sem histórico de fluxo")
        return fig
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=flow_hist.index, y=flow_hist['LevETF_Flow'],
                             name='Lev ETF Flow', line=dict(color=_C['orange'], width=1.5)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=flow_hist.index, y=flow_hist['Return'],
                             name='Return', line=dict(color=_C['accent'], width=1)),
                  secondary_y=True)
    fig.update_layout(title='Fluxo ETFs Alavancados vs Retorno',
                      hovermode='x unified', **FLOW_FIG_LAYOUT)
    fig.update_yaxes(title_text='Flow ($)', secondary_y=False)
    fig.update_yaxes(title_text='Return', secondary_y=True)
    return go.FigureWidget(fig)


def fp_plot_positions_basket(df_cot, basket_col='Price', data_col='Positions'):
    """Dual-axis: positions + preço."""
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(
        x=df_cot.index, y=df_cot[data_col], name=data_col,
        yaxis='y1', line=dict(width=1.5), marker_color=_C['orange']))
    if basket_col in df_cot.columns:
        fig.add_trace(go.Scatter(
            x=df_cot.index, y=df_cot[basket_col], name=basket_col,
            yaxis='y2', line=dict(width=1), marker_color=_C['accent']))
    fig.update_yaxes(zeroline=False, showgrid=False)
    fig.layout.yaxis.update(title_text=data_col)
    fig.layout.yaxis2.update(title_text=basket_col)
    fig.update_layout(title='Positions & Basket Price', **FLOW_FIG_LAYOUT)
    return _flow_border(go.FigureWidget(fig))


def fp_plot_long_short_net(df_cot):
    """Barras Long/Short + linha Net."""
    fig = go.FigureWidget()
    for name, color in [('Long', _C['teal']), ('Short', _C['text_muted'])]:
        col = f'Positions - {name}'
        if col in df_cot.columns:
            fig.add_trace(go.Bar(x=df_cot.index, y=df_cot[col],
                                 name=name, marker_color=color))
    if 'Positions - Net' in df_cot.columns:
        fig.add_trace(go.Scatter(x=df_cot.index, y=df_cot['Positions - Net'],
                                 name='Net', marker_color=_C['orange']))
    fig.update_layout(barmode='relative',
                      title='Long, Short & Net Positions',
                      yaxis_title='Positions', hovermode='x unified',
                      **FLOW_FIG_LAYOUT)
    return _flow_border(fig)


def fp_plot_correlation(df_cot, window=26):
    """Rolling correlation entre preço e net positions."""
    if 'Price' not in df_cot.columns or 'Positions - Net' not in df_cot.columns:
        return wd.VBox([wd.HTML("<p>Sem dados para correlação.</p>")])
    corr = (df_cot[['Price', 'Positions - Net']].pct_change()
            .rolling(window).corr().unstack()[('Price', 'Positions - Net')])
    fig = go.FigureWidget(go.Scatter(
        x=corr.index, y=corr.values, name='Corr',
        line=dict(width=1.5), marker_color=_C['orange']))
    fig.update_yaxes(zeroline=False, showgrid=False)
    fig.layout.yaxis.update(title_text='Correlation')
    fig.update_layout(title='Rolling Correlation: Price Δ vs Net Length Δ',
                      **FLOW_FIG_LAYOUT)
    return _flow_border(fig)


def fp_plot_long_short_ratio(df_cot):
    """Long/Short ratio."""
    if 'Positions - Long' not in df_cot.columns or 'Positions - Short' not in df_cot.columns:
        return wd.VBox([wd.HTML("<p>Sem dados L/S.</p>")])
    ratio = df_cot['Positions - Long'] / df_cot['Positions - Short'] * -1
    fig = go.FigureWidget(go.Scatter(
        x=ratio.index, y=ratio.values, name='L/S Ratio',
        line=dict(width=1.5, color=_C['orange'])))
    fig.update_yaxes(zeroline=False, showgrid=False)
    fig.layout.yaxis.update(title_text='Ratio')
    fig.update_layout(title='Long/Short Ratio', **FLOW_FIG_LAYOUT)
    return _flow_border(fig)


def fp_plot_multi_year(df_cot):
    """Gráfico sazonal multi-ano."""
    if 'week' not in df_cot.columns or 'year' not in df_cot.columns:
        return wd.VBox([wd.HTML("<p>Sem dados semanais.</p>")])
    pivot = df_cot.pivot(columns='year', index='week', values='Positions')
    pivot = pivot.iloc[:, -6:]
    fig = go.FigureWidget()
    colors_yr = [_C['text_dim'], _C['text_muted'], '#8b949e', _C['yellow'], _C['orange'], _C['teal']]
    for col_name, color in zip(pivot.columns, colors_yr):
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[col_name], mode='lines',
            line=dict(color=color), name=str(col_name)))
    fig.update_layout(hovermode='x unified', yaxis_title='Positions',
                      title='Seasonal Analysis', **FLOW_FIG_LAYOUT)
    return _flow_border(fig)


def fp_plot_dispersion(seasonality_df, df_cot, col='Positions'):
    """Dispersion chart: min/max band + mean + current year."""
    seas = seasonality_df[col].copy() if col in seasonality_df.columns else pd.DataFrame()
    if seas.empty:
        return wd.VBox([wd.HTML("<p>Sem dados de sazonalidade.</p>")])
    current_year = pd.Timestamp.now().year
    yr_data = df_cot[df_cot.index.year == current_year] if hasattr(df_cot.index, 'year') else pd.DataFrame()
    if not yr_data.empty and 'week' in yr_data.columns and col in yr_data.columns:
        seas['Current'] = yr_data.set_index('week')[col]
    fig = go.FigureWidget()
    for name, fill in [('Max', None), ('Min', 'tonexty')]:
        if name in seas.columns:
            fig.add_trace(go.Scatter(
                x=seas.index, y=seas[name], mode='lines', name=name,
                fill=fill, fillcolor='rgba(88,166,255,0.08)',
                line=dict(color=_C['border'], width=0), showlegend=False))
    for name, color in [('Mean', _C['orange']), ('Current', _C['teal'])]:
        if name in seas.columns:
            fig.add_trace(go.Scatter(
                x=seas.index, y=seas[name], mode='lines+markers',
                name=name, line=dict(color=color)))
    fig.update_layout(title='5Y Dispersion', xaxis_title='Weeks',
                      yaxis_title=col, hovermode='x unified',
                      xaxis=dict(range=[1, 53]), **FLOW_FIG_LAYOUT)
    return _flow_border(fig)


def fp_bqp_flow_bar_line(flow_df, bar_col='LevETF_Flow', line_col='Return'):
    """Bar + line overlay usando bqplot."""
    if not HAS_BQPLOT or flow_df.empty:
        return wd.HTML("<p>bqplot não disponível ou sem dados.</p>")
    scale_x = bqp.DateScale()
    scale_y = bqp.LinearScale()
    mark_bar = bqp.Bars(
        x=flow_df.index, y=flow_df[bar_col],
        scales={'x': scale_x, 'y': scale_y}, colors=[_C['accent']],
        tooltip=bqp.Tooltip(fields=['y', 'x'], show_labels=False,
                            formats=['.0f', '%Y/%m/%d']))
    marks = [mark_bar]
    if line_col in flow_df.columns:
        scale_y2 = bqp.LinearScale()
        mark_line = bqp.Lines(
            x=flow_df.index, y=flow_df[line_col], stroke_width=2,
            scales={'x': scale_x, 'y': scale_y2}, colors=[_C['purple']])
        marks.append(mark_line)
    ax_y = bqp.Axis(scale=scale_y, orientation='vertical', label='Flow ($)')
    ax_x = bqp.Axis(scale=scale_x)
    return bqp.Figure(
        marks=marks, axes=[ax_x, ax_y],
        title='Fluxo de ETFs Alavancados',
        title_style={'font-size': '18px'}, padding_y=0,
        fig_margin={'top': 50, 'bottom': 50, 'left': 60, 'right': 50},
        layout={'width': 'auto', 'height': '400px'})


def fp_bqp_scatter(flow_df):
    """Scatter plot: flow vs return (bqplot)."""
    if not HAS_BQPLOT or flow_df.empty:
        return wd.HTML("<p>bqplot não disponível ou sem dados.</p>")
    scale_x = bqp.LinearScale()
    scale_y = bqp.LinearScale()
    tooltip = bqp.Tooltip(fields=['x', 'y'], labels=['Flow', 'Return'],
                          formats=['.0f', '.4f'])
    mark = bqp.Scatter(
        x=flow_df['LevETF_Flow'], y=flow_df['Return'],
        tooltip=tooltip, scales={'x': scale_x, 'y': scale_y},
        default_size=32, colors=[_C['orange']])
    ax_x = bqp.Axis(scale=scale_x, label='Flow ($)')
    ax_y = bqp.Axis(scale=scale_y, orientation='vertical', label='Return')
    return bqp.Figure(
        marks=[mark], axes=[ax_x, ax_y],
        title='Flow vs Return', title_style={'font-size': '18px'},
        padding_x=0.05, padding_y=0.05,
        layout={'width': '100%', 'height': '400px'})


def fp_grid_flow_score(score):
    """Tabela interativa do flow score."""
    if not HAS_DATAGRID:
        rows_html = [f"<tr><td>{k}</td><td>{v:.3f}</td></tr>"
                     for k, v in score.items() if isinstance(v, (int, float))]
        return wd.HTML(f"<table>{''.join(rows_html)}</table>")
    data = {
        'Componente': ['CTA Trend', 'Dealer/MM', 'Vol Control',
                        'Risk Parity', 'ETFs Alavancados', 'ETFs Passivos',
                        'Buyback', 'COT', 'Score Combinado'],
        'Z-Score': [score.get('z_cta', 0),
                    score.get('z_dealer', 0), score.get('z_volctrl', 0),
                    score.get('z_rp', 0),
                    score.get('z_leveraged', 0), score.get('z_passive_etf', 0),
                    score.get('z_buyback', 0),
                    score.get('z_cot', 0),
                    score.get('combined_score', 0)],
        'Peso (%)': [score.get('weights', {}).get('cta', 0) * 100,
                     score.get('weights', {}).get('dealer', 0) * 100,
                     score.get('weights', {}).get('volctrl', 0) * 100,
                     score.get('weights', {}).get('rp', 0) * 100,
                     score.get('weights', {}).get('leveraged', 0) * 100,
                     score.get('weights', {}).get('passive_etf', 0) * 100,
                     score.get('weights', {}).get('buyback', 0) * 100,
                     score.get('weights', {}).get('cot', 0) * 100,
                     100],
    }
    df_s = pd.DataFrame(data).set_index('Componente')
    try:
        linear_scale = bqp.LinearScale(min=-3, max=3)
        color_scale = bqp.ColorScale(min=-3, max=3,
                                     colors=[_C['red'], _C['card2'], _C['green']])
        renderers = {
            'Z-Score': BarRenderer(bar_value=linear_scale, format='.2f',
                                   bar_color=color_scale,
                                   horizontal_alignment='center'),
        }
    except Exception:
        renderers = {}
    return DataGrid(df_s, renderers=renderers, base_column_size=150,
                    layout={'height': '200px'})


def fp_grid_cot_stats(stats):
    """Tabela de estatísticas COT."""
    if stats.empty:
        return wd.HTML("<p>Sem dados COT.</p>")
    df_s = stats.to_frame('Value')
    if not HAS_DATAGRID:
        return wd.HTML(df_s.to_html())
    try:
        from ipydatagrid import VegaExpr
        renderers = {
            'Value': TextRenderer(
                background_color=VegaExpr(
                    f"cell.value < -1 ? '{_C['red']}' : "
                    f"cell.value > 1 ? '{_C['green']}' : 'transparent'"),
                format='.2f')
        }
    except Exception:
        renderers = {}
    return DataGrid(df_s, renderers=renderers, base_column_size=150,
                    base_row_header_size=200, layout={'height': '300px'})


def fp_grid_buyback(buyback_df):
    """Tabela de buyback por empresa."""
    if buyback_df.empty:
        return wd.HTML("<p>Sem dados de buyback.</p>")
    if not HAS_DATAGRID:
        return wd.HTML(buyback_df.head(20).to_html())
    try:
        scale = bqp.LinearScale(min=0, max=buyback_df['daily_est'].max())
        renderers = {
            'daily_est': BarRenderer(bar_value=scale, format='$,.0f',
                                     bar_color=_C['green'],
                                     horizontal_alignment='right'),
        }
    except Exception:
        renderers = {}
    return DataGrid(buyback_df.head(30), renderers=renderers,
                    base_column_size=120, base_row_header_size=140,
                    layout={'height': '400px'})


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
                 steps=None, width=220, height=190):
    """Cria indicador gauge do Plotly."""
    if pd.isna(value):
        value = 0
    gauge_cfg = {
        'axis': {'range': [range_min, range_max],
                 'tickcolor': _C['text_muted'], 'tickfont': {'color': _C['text_muted'], 'size': 10}},
        'bar': {'color': bar_color, 'thickness': 0.3},
        'bgcolor': _C['card2'],
        'borderwidth': 1,
        'bordercolor': _C['border'],
    }
    if steps:
        gauge_cfg['steps'] = steps
    return go.FigureWidget(
        go.Indicator(
            mode="gauge+number", value=value,
            title={'text': title, 'font': {'size': 13, 'color': _C['text_muted']}},
            number={'suffix': suffix, 'font': {'size': 20, 'color': _C['text']}, 'valueformat': '.2f'},
            gauge=gauge_cfg),
        layout=go.Layout(width=width, height=height,
                         paper_bgcolor=_C['card'],
                         font=dict(color=_C['text']),
                         margin=dict(l=18, r=18, t=42, b=12)))


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
        colors = [_C['green'] if v >= 0 else _C['red'] for v in combo_data.values]
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
        plt.bar(strikes, call_data / div, width=bar_w, color=_C['green'],
                edgecolor='k', lw=0.3, label=f'Call {name}')
        plt.bar(strikes, put_sign * put_data / div, width=bar_w, color=_C['red'],
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
        plt.plot(levels, curve, lw=2, color=_C['accent'])
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
    styled = styled.set_properties(**{
        'width': '110px', 'text-align': 'center', 'font-size': '12px',
        'padding': '6px 8px', 'border': f'1px solid {_C["border_light"]}',
        'color': _C['text'], 'background-color': _C['card'],
    })
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', _C['card2']), ('color', _C['text_muted']),
            ('font-size', '11px'), ('text-transform', 'uppercase'),
            ('letter-spacing', '0.5px'), ('padding', '8px'),
            ('border', f'1px solid {_C["border"]}')]},
    ])
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
spx_pred_w = wd.Checkbox(value=False, description='Incluir Previsão SPX',
                         indent=False, layout={'width': '250px'})
flow_pred_w = wd.Checkbox(value=True, description='Incluir Flow Predictor',
                          indent=False, layout={'width': '250px'})
cot_type_w = wd.Dropdown(
    options=list(COT_CONTRACTS.keys()),
    value='Equity Indices', description='COT Categoria:',
    layout={'width': '280px'})
cot_contract_w = wd.SelectMultiple(
    description='Contratos:', layout={'min_width': '300px', 'height': '80px'})
cot_trader_w = wd.Dropdown(
    options=['Total', 'Managed Money', 'Commercial', 'Non-Commercial',
             'Swap Dealers', 'Leveraged Funds', 'Asset Manager',
             'Dealer Intermediary', 'Other Reportables'],
    value='Total', description='Trader Type:',
    layout={'width': '280px'})
cot_start_w = wd.DatePicker(
    value=pd.Timestamp.now().date() - pd.Timedelta(days=2*365),
    description='COT Start:', layout={'width': '260px'})
cot_end_w = wd.DatePicker(
    value=pd.Timestamp.now().date(),
    description='COT End:', layout={'width': '260px'})


def _update_cot_contracts(change=None):
    opts = COT_CONTRACTS.get(cot_type_w.value, [])
    cot_contract_w.options = opts
    if opts:
        cot_contract_w.value = (opts[0][1],)

cot_type_w.observe(_update_cot_contracts, names='value')
_update_cot_contracts()

out_main = wd.Output()
out_cot_reload = wd.Output()

# Botões de recarga parcial
cot_reload_btn = wd.Button(description='⟳ Recarregar COT',
                           button_style='info', icon='refresh',
                           layout={'width': '180px'})
etf_reload_btn = wd.Button(description='⟳ Recarregar ETFs',
                           button_style='info', icon='refresh',
                           layout={'width': '180px'})


def _reload_cot(_):
    """Recarrega apenas dados COT e exibe resultado."""
    with out_cot_reload:
        clear_output(wait=True)
        display(wd.HTML(DASH_CSS + "<div class='mm-dash mm-loading'>⏳ Recarregando COT...</div>"))
        ticker = ticker_w.value.strip() or 'SPX Index'
        _cot_start = cot_start_w.value.strftime('%Y%m%d') if cot_start_w.value else '-2Y'
        _cot_end = cot_end_w.value.strftime('%Y%m%d') if cot_end_w.value else '0D'
        try:
            cot_df = safe_fetch_cot(ticker, start=_cot_start, end=_cot_end)
            sel_cots = list(cot_contract_w.value)
            sel_df = None
            if sel_cots:
                try:
                    raw = fetch_cot_data(
                        sel_cots[0] if len(sel_cots) == 1 else sel_cots,
                        start=_cot_start, end=_cot_end)
                    if raw is not None and not raw.empty:
                        sel_df = aggregate_cot(raw)
                except Exception as e:
                    print(f"⚠️ COT selected: {e}")
            clear_output(wait=True)
            children = [wd.HTML(
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<h3>COT — Recarga Rápida</h3>"
                f"<p>Dados agregados (todas trader types somadas). "
                f"Report type: auto (tenta disaggregated → tff → legacy)</p>"
                f"</div></div>")]
            ok, fut = has_cot(ticker)
            if ok and cot_df is not None and not cot_df.empty:
                stats = cot_summary_stats(cot_df)
                children.append(wd.HTML(f"<p>Futures: <b>{fut}</b> — {len(cot_df)} registros</p>"))
                children.append(fp_grid_cot_stats(stats))
                seas = cot_seasonality(cot_df)
                children.append(wd.HBox([
                    fp_plot_positions_basket(cot_df),
                    fp_plot_dispersion(seas, cot_df)]))
                children.append(fp_plot_long_short_net(cot_df))
            elif ok:
                children.append(wd.HTML(f"<p>COT disponível para {fut}, mas sem dados.</p>"))
            if sel_df is not None and not sel_df.empty:
                sel_label = ', '.join(sel_cots)
                children.append(wd.HTML(f"<hr><h4>COT: {sel_label} — {len(sel_df)} registros</h4>"))
                children.append(fp_plot_positions_basket(sel_df))
            display(wd.VBox(children))
        except Exception as e:
            clear_output(wait=True)
            print(f"Erro COT: {e}")
            import traceback; traceback.print_exc()


def _reload_etfs(_):
    """Recarrega apenas fluxo de ETFs alavancados."""
    with out_cot_reload:
        clear_output(wait=True)
        display(wd.HTML(DASH_CSS + "<div class='mm-dash mm-loading'>⏳ Recarregando ETFs...</div>"))
        ticker = ticker_w.value.strip() or 'SPX Index'
        try:
            prices, lr = fetch_historical(ticker)
            nz = lr[lr != 0]
            dr = float(nz.iloc[-1]) if len(nz) > 0 else 0
            flows, total = compute_leveraged_flows(dr)
            clear_output(wait=True)
            html = (
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<h4>ETFs Alavancados (Retorno diário: {dr*100:+.2f}%)</h4>" +
                flows[['Leverage', 'AUM_$', 'Rebalance_$', 'Direção']]
                .style.format({'AUM_$': '${:,.0f}', 'Rebalance_$': '${:,.0f}'})
                .to_html() +
                f"<p><b>Fluxo Direcional Total: ${total:,.0f}</b></p>"
                f"</div></div>")
            display(wd.HTML(html))
        except Exception as e:
            clear_output(wait=True)
            print(f"Erro ETF: {e}")


cot_reload_btn.on_click(_reload_cot)
etf_reload_btn.on_click(_reload_etfs)


def run_analysis(_):
    """Callback principal: busca dados, calcula tudo, monta dashboard."""
    with out_main:
        clear_output(wait=True)
        loading = wd.HTML(DASH_CSS + "<div class='mm-dash mm-loading'>⏳ Inicializando...</div>")
        display(loading)

        ticker = ticker_w.value.strip() or 'SPX Index'
        min_dte, max_dte = dte_w.value
        mny_low, mny_high = mny_w.value

        try:
            # ── 1. Dados de Mercado ──────────────────────────────────────
            loading.value = "<h4>1/16: Buscando dados de mercado...</h4>"
            mkt = fetch_market_data(ticker)
            spot = mkt['spot']
            iv_30d = mkt['iv_30d']
            rv_30d = mkt['rv_30d']
            skew = mkt['skew']
            avg_vol = mkt['avg_dollar_volume']

            if pd.isna(spot):
                raise ValueError(f"Spot inválido para {ticker}")

            # ── 2. Histórico + Modelo de Risco ───────────────────────────
            loading.value = "<h4>2/16: Modelagem de risco (t-Student)...</h4>"
            _, log_returns = fetch_historical(ticker)
            risk = fit_risk_model(log_returns)

            # ── 3. Cadeia de Opções ──────────────────────────────────────
            loading.value = "<h4>3/16: Buscando cadeia de opções...</h4>"
            df, from_strike, to_strike = fetch_options_chain(
                ticker, spot, min_dte, max_dte, mny_low, mny_high)

            # ── 4. Gregas + Exposições ───────────────────────────────────
            loading.value = f"<h4>4/16: Calculando gregas para {len(df)} opções...</h4>"
            greeks_now = calculate_all_greeks(
                spot, df.Strike.values, df.IV.values, df.Tte.values, df.Type.values)
            agg = compute_strike_exposures(df, greeks_now, spot)
            call_wall, put_wall = compute_walls(agg)

            if call_wall == put_wall and call_wall is not None:
                print(f"⚠️ Call Wall = Put Wall = {call_wall:,.0f}")

            # ── 5. Curvas Modelo ─────────────────────────────────────────
            loading.value = "<h4>5/16: Calculando curvas modelo (100 níveis × 7 gregas)...</h4>"
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
            loading.value = "<h4>6/16: Matrizes de sensibilidade (7×5×7)...</h4>"
            sens_matrices = compute_sensitivity_matrices(df, spot)

            # ── 7. Monte Carlo ───────────────────────────────────────────
            loading.value = "<h4>7/16: Simulação Monte Carlo (10k cenários, 5 dias)...</h4>"
            mc_n_days = 5
            mc_pnl, mc_prices = run_monte_carlo(spot, df, risk, n_days=mc_n_days)

            # ── 8. Curvas de P&L ─────────────────────────────────────────
            loading.value = "<h4>8/16: Curvas de P&L e hedge demand...</h4>"
            pnl_curves = compute_pnl_curves(greeks_now, df, spot, levels, skew)

            # ── 9. Rebalanceamento ETFs ──────────────────────────────────
            loading.value = "<h4>9/16: Fluxo de rebalanceamento ETFs passivos...</h4>"
            try:
                etf_flows, etf_summary, etf_start = compute_full_etf_flows()
                etf_ok = True
            except Exception:
                etf_flows, etf_summary, etf_start, etf_ok = {}, None, None, False

            # ── 10. ETFs Alavancados ─────────────────────────────────────
            loading.value = "<h4>10/16: Fluxo de ETFs alavancados...</h4>"
            try:
                # Usar último retorno não-zero (evitar 0 de weekend/feriado)
                nz_rets = log_returns[log_returns != 0]
                daily_ret = float(nz_rets.iloc[-1]) if len(nz_rets) > 0 else 0
                lev_flows, lev_total = compute_leveraged_flows(daily_ret)
                lev_ok = True
            except Exception:
                lev_flows, lev_total, lev_ok = None, 0, False
                daily_ret = 0

            # ── 11. Previsão SPX (opcional — pesado) ─────────────────────
            loading.value = "<h4>11/16: Previsão de rebalanceamento SPX...</h4>"
            spx_pred_ok = False
            top_in_spx, top_out_spx = None, None
            if spx_pred_w.value and HAS_SKLEARN:
                try:
                    top_in_spx, top_out_spx, _ = build_spx_prediction()
                    spx_pred_ok = True
                except Exception as pred_err:
                    print(f"⚠️ Previsão SPX falhou: {pred_err}")

            # ── 12. Flow Predictor — Histórico + Buyback ─────────────────
            fp_ok = False
            fp_flow_hist = pd.DataFrame()
            fp_buyback = {'daily_est': 0, 'pct_adv_est': 0,
                          'confidence': 'none', 'announced': 0}
            fp_score = None
            fp_cot_df = None
            fp_cot_stats = pd.Series(dtype=float)
            fp_selected_cot_df = None

            if flow_pred_w.value:
                loading.value = "<h4>12/16: Flow Predictor — histórico + buyback...</h4>"
                try:
                    fp_flow_hist = build_flow_history(ticker, lookback=252)
                    fp_today_flow = (compute_leveraged_flow_simple(
                        float(fp_flow_hist['Return'].iloc[-1]))
                        if not fp_flow_hist.empty else 0)
                except Exception as fp_err:
                    print(f"⚠️ Flow hist: {fp_err}")
                    fp_today_flow = 0

                # Buyback: para índices, usar agregação por membros
                try:
                    if ticker.strip().endswith('Index'):
                        bb_df = estimate_index_buyback_flow(ticker, top_n=30)
                        fp_buyback_daily = float(bb_df['daily_est'].sum()) if (
                            not bb_df.empty and 'daily_est' in bb_df.columns) else 0
                        fp_buyback = {
                            'daily_est': fp_buyback_daily,
                            'pct_adv_est': 0,
                            'confidence': 'estimated',
                            'announced': float(bb_df['buyback'].sum()) if (
                                not bb_df.empty and 'buyback' in bb_df.columns) else 0,
                        }
                    else:
                        fp_buyback = estimate_buyback_flow(ticker)
                        fp_buyback_daily = fp_buyback.get('daily_est', 0)
                except Exception as fp_err:
                    print(f"⚠️ Buyback: {fp_err}")
                    fp_buyback_daily = 0

                # ── 13. Flow Predictor — COT ─────────────────────────────
                loading.value = "<h4>13/16: Flow Predictor — COT...</h4>"
                cot_ok_fp, cot_fut_fp = has_cot(ticker)
                cot_net_change = 0
                history_cot = None

                _cot_start = (cot_start_w.value.strftime('%Y%m%d')
                              if cot_start_w.value else '-2Y')
                _cot_end = (cot_end_w.value.strftime('%Y%m%d')
                            if cot_end_w.value else '0D')

                if cot_ok_fp:
                    try:
                        fp_cot_df = safe_fetch_cot(ticker,
                                                   start=_cot_start,
                                                   end=_cot_end)
                        if fp_cot_df is not None and not fp_cot_df.empty:
                            fp_cot_stats = cot_summary_stats(fp_cot_df)
                            # Pega net change de qualquer coluna disponível
                            _net_col = None
                            for _nc in ('Positions - Net', 'Positions'):
                                if _nc in fp_cot_df.columns:
                                    _net_col = _nc
                                    break
                            if _net_col:
                                net = fp_cot_df[_net_col].dropna()
                                if len(net) >= 2:
                                    cot_net_change = float(
                                        net.iloc[-1] - net.iloc[-2])
                                    history_cot = net.diff().dropna()
                    except Exception:
                        pass

                # COT de contratos selecionados
                selected_cots = list(cot_contract_w.value)
                if selected_cots:
                    try:
                        sel_df = fetch_cot_data(
                            selected_cots[0] if len(selected_cots) == 1
                            else selected_cots,
                            start=_cot_start, end=_cot_end)
                        if not sel_df.empty:
                            fp_selected_cot_df = aggregate_cot(sel_df)
                    except Exception:
                        pass

                # ── 14. Flow Score Combinado ─────────────────────────────
                loading.value = "<h4>14/16: Calculando flow score...</h4>"

                # Dealer/MM hedging flow (baseado em GEX)
                fp_dealer_flow = 0
                try:
                    _gex_per_pt = (greeks_now['gamma']
                                   * np.where(df.Type.values == 'Call', 1, -1)
                                   * df['OI'].values * 100).sum()
                    _daily_chg = spot * daily_ret
                    fp_dealer_flow = compute_dealer_hedging_flow(
                        _gex_per_pt, _daily_chg, spot)
                    print(f"[FLOW] Dealer flow: ${fp_dealer_flow:,.0f} "
                          f"(GEX/pt={_gex_per_pt:,.0f}, ΔS={_daily_chg:,.1f})")
                except Exception as e:
                    print(f"⚠️ Dealer flow: {e}")

                # Vol control fund flows (5%, 10%, 15%)
                fp_volctrl = {'total': 0, 'detail': {}}
                try:
                    _rv_window = 21
                    _rets = log_returns.iloc[-_rv_window * 2:]
                    _rv_cur = _rets.iloc[-_rv_window:].std() * np.sqrt(252)
                    _rv_prev = _rets.iloc[-_rv_window * 2:-_rv_window].std() * np.sqrt(252)
                    fp_volctrl = compute_vol_control_flow(_rv_cur, _rv_prev)
                    print(f"[FLOW] Vol ctrl: ${fp_volctrl['total']:,.0f} "
                          f"(RV cur={_rv_cur:.2%}, prev={_rv_prev:.2%})")
                except Exception as e:
                    print(f"⚠️ Vol ctrl: {e}")

                # CTA trend following flow
                fp_cta = {'flow': 0, 'trend_today': 0, 'pos_today': 0, 'pos_prev': 0}
                try:
                    # Reconstruir preços a partir de log returns
                    _px_series = np.exp(np.cumsum(log_returns))
                    _px_series = _px_series * (spot / _px_series.iloc[-1])
                    _cta_rv = log_returns.iloc[-63:].std() * np.sqrt(252) if len(log_returns) >= 63 else rv_30d
                    fp_cta = compute_cta_flow(_px_series, _cta_rv)
                    print(f"[FLOW] CTA: ${fp_cta['flow']:,.0f} "
                          f"(trend={fp_cta['trend_today']:+.3f}, "
                          f"pos={fp_cta['pos_today']:+.3f}→{fp_cta['pos_prev']:+.3f})")
                except Exception as e:
                    print(f"⚠️ CTA flow: {e}")

                # Risk Parity flow
                fp_rp = {'total': 0, 'detail': {}, 'eq_alloc_new': 0, 'eq_alloc_old': 0}
                try:
                    _rv_window = 21
                    _rets_rp = log_returns.iloc[-_rv_window * 2:]
                    _rv_eq_cur = _rets_rp.iloc[-_rv_window:].std() * np.sqrt(252)
                    _rv_eq_prev = _rets_rp.iloc[-_rv_window * 2:-_rv_window].std() * np.sqrt(252)
                    fp_rp = compute_risk_parity_flow(_rv_eq_cur, _rv_eq_prev)
                    print(f"[FLOW] Risk Parity: ${fp_rp['total']:,.0f} "
                          f"(eq_alloc={fp_rp['eq_alloc_new']:.1%}→{fp_rp['eq_alloc_old']:.1%})")
                except Exception as e:
                    print(f"⚠️ Risk Parity: {e}")

                # Passive ETF flow — net rebalancing from VOO/SPY/IVV
                fp_passive_etf_flow = 0
                try:
                    if etf_ok and etf_flows:
                        combo = etf_flows.get('Combined', pd.DataFrame())
                        if not combo.empty and 'Flow_$' in combo.columns:
                            fp_passive_etf_flow = float(combo['Flow_$'].sum())
                            print(f"[FLOW] ETFs Passivos: ${fp_passive_etf_flow:,.0f} "
                                  f"(buy={combo.loc[combo['Flow_$']>0, 'Flow_$'].sum():,.0f}, "
                                  f"sell={combo.loc[combo['Flow_$']<0, 'Flow_$'].sum():,.0f})")
                except Exception as e:
                    print(f"⚠️ ETFs Passivos: {e}")

                try:
                    lev_history = (fp_flow_hist['LevETF_Flow']
                                   if not fp_flow_hist.empty else None)
                    fp_score = compute_flow_score(
                        leveraged_flow=fp_today_flow,
                        buyback_daily=fp_buyback_daily,
                        cot_net_change=cot_net_change,
                        passive_etf_flow=fp_passive_etf_flow,
                        history_leveraged=lev_history,
                        history_cot=history_cot,
                        dealer_flow=fp_dealer_flow,
                        volctrl_flow=fp_volctrl['total'],
                        cta_flow=fp_cta['flow'],
                        rp_flow=fp_rp['total'])
                    fp_ok = True
                except Exception as fp_err:
                    print(f"⚠️ Flow score: {fp_err}")

            clear_output(wait=True)

            # ═════════════════════════════════════════════════════════════
            # MONTAGEM DAS ABAS DO DASHBOARD
            # ═════════════════════════════════════════════════════════════

            title_html = wd.HTML(
                DASH_CSS +
                f"<div class='mm-dash'>"
                f"<div class='mm-title'>Market Maker Dashboard "
                f"<small>{ticker} @ {spot:,.2f} │ {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>"
                f"</div></div>")

            # ─── ABA 1: VISÃO GERAL ─────────────────────────────────────
            total_gex = gamma_curve[np.argmin(np.abs(levels - spot))]
            fragility = (abs(total_gex) / avg_vol) * 100 if pd.notna(avg_vol) and avg_vol > 0 else 0
            daily_move = implied_move_pct(iv_30d) if pd.notna(iv_30d) else 0
            vol_premium = (iv_30d - rv_30d) * 100 if pd.notna(iv_30d) and pd.notna(rv_30d) else 0

            g_frag = create_gauge(fragility, "Fragilidade (GEX/Vol)",
                                  0, 20, _C['red'], "%",
                                  steps=[{'range': [0, 5], 'color': '#1a3a2a'},
                                         {'range': [5, 12], 'color': '#3a3520'},
                                         {'range': [12, 20], 'color': '#3a1a1a'}])
            g_vol = create_gauge(vol_premium, "Prêmio Vol (IV-RV)",
                                 -5, 5, _C['orange'], "%")
            _skew_val = skew * 100
            _skew_hi = max(15, abs(_skew_val) * 1.3)
            g_skew = create_gauge(_skew_val, "Skew (P25-C25)",
                                  -_skew_hi, _skew_hi, _C['teal'], "%")
            _move_hi = max(5, daily_move * 1.3)
            g_move = create_gauge(daily_move, "Mov. Esperado 1D",
                                  0, _move_hi, _C['green'], "%")

            # GEX curve (Plotly)
            fig_gex = go.FigureWidget()
            fig_gex.add_trace(go.Scatter(
                x=levels, y=gamma_curve / 1e9, mode='lines',
                fill='tozeroy', line_color=_C['accent'],
                fillcolor='rgba(88,166,255,0.15)', name='GEX'))
            fig_gex.add_vline(x=spot, line_dash="dash", line_color=_C['red'],
                              annotation_text=f"Spot {spot:,.0f}")
            if gamma_flip:
                fig_gex.add_vline(x=gamma_flip, line_color=_C['orange'],
                                  annotation_text=f"G-Flip {gamma_flip:,.0f}")
            fig_gex.update_layout(title="Gamma Exposure (GEX)",
                                  yaxis_title="$ Bi / 1% move",
                                  height=350, template=DASH_TEMPLATE,
                                  margin=dict(t=35, b=25), showlegend=False)

            # Return distribution (Plotly)
            fig_dist = go.FigureWidget()
            fig_dist.add_trace(go.Histogram(
                x=log_returns, histnorm='probability density',
                name='Reais', marker_color=_C['accent'], opacity=0.6,
                xbins=dict(size=log_returns.std() / 4)))
            x_pdf = np.linspace(log_returns.min(), log_returns.max(), 500)
            pdf_vals = student_t.pdf(x_pdf, risk['tdf'], risk['tloc'], risk['tscale'])
            fig_dist.add_trace(go.Scatter(
                x=x_pdf, y=pdf_vals, mode='lines',
                name='t-Student', line_color=_C['orange']))
            fig_dist.add_vline(x=risk['var_95'], line_dash="dash",
                               line_color=_C['orange'],
                               annotation_text=f"VaR 95% ({risk['var_95']:.2%})")
            fig_dist.add_vline(x=risk['var_99'], line_dash="dash",
                               line_color=_C['red'],
                               annotation_text=f"VaR 99% ({risk['var_99']:.2%})")
            _xlo = max(log_returns.quantile(0.005), -0.08)
            _xhi = min(log_returns.quantile(0.995), 0.08)
            fig_dist.update_layout(title="Distribuição de Retornos",
                                   yaxis_title="Prob.",
                                   xaxis_tickformat=".1%",
                                   xaxis_range=[_xlo, _xhi],
                                   height=350, template=DASH_TEMPLATE,
                                   margin=dict(t=35, b=25))

            # Sumário de vol e risco
            summary_html = f"""
            <div class='mm-dash'><div class='mm-card'>
                <div class='mm-kpi-row'>
                    <div class='mm-kpi'><div class='kpi-label'>IV 30d ATM</div><div class='kpi-value' style='color:{_C["accent"]}'>{iv_30d:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>RV 30d</div><div class='kpi-value' style='color:{_C["teal"]}'>{rv_30d:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>Prêmio</div><div class='kpi-value' style='color:{_C["orange"]}'>{vol_premium:+.2f}%</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>Skew</div><div class='kpi-value' style='color:{_C["purple"]}'>{skew:+.2%}</div></div>
                </div>
                <div class='mm-section-label'>Risco Caudal</div>
                <div class='mm-kpi-row'>
                    <div class='mm-kpi'><div class='kpi-label'>VaR 95%</div><div class='kpi-value' style='color:{_C["yellow"]}'>{risk['var_95']:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>CVaR 95%</div><div class='kpi-value' style='color:{_C["orange"]}'>{risk['cvar_95']:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>VaR 99%</div><div class='kpi-value' style='color:{_C["red"]}'>{risk['var_99']:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>CVaR 99%</div><div class='kpi-value' style='color:{_C["red"]}'>{risk['cvar_99']:.2%}</div></div>
                </div>
                <div class='mm-section-label'>Posicionamento</div>
                <div class='mm-kpi-row'>
                    <div class='mm-kpi'><div class='kpi-label'>Gamma Flip</div><div class='kpi-value'>~{f'{gamma_flip:,.0f}' if gamma_flip else 'N/A'}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>Call Wall</div><div class='kpi-value' style='color:{_C["green"]}'>{f'{call_wall:,.0f}' if call_wall else 'N/A'}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>Put Wall</div><div class='kpi-value' style='color:{_C["red"]}'>{f'{put_wall:,.0f}' if put_wall else 'N/A'}</div></div>
                </div>
            </div></div>"""

            tab1 = wd.VBox([
                wd.HBox([g_frag, g_vol, g_skew, g_move],
                        layout={'justify_content': 'space-around'}),
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
                wd.HTML("<div class='mm-dash'><div class='mm-card'><h3>Matrizes de Sensibilidade (Preço × Vol Shift)</h3></div></div>"),
                wd.HTML("".join(sens_html_parts))
            ])

            # ─── ABA 4: ANÁLISE DE P&L ──────────────────────────────────
            # P&L comparativo
            fig_pnl = go.FigureWidget()
            fig_pnl.add_trace(go.Scatter(
                x=levels, y=pnl_curves['simple'] / 1e6,
                mode='lines', name='Simplificado (Δ+Γ)',
                line=dict(color=_C['orange'], dash='dot')))
            fig_pnl.add_trace(go.Scatter(
                x=levels, y=pnl_curves['complete'] / 1e6,
                mode='lines', name='Completo (+Vega+Vanna+Zomma)',
                line=dict(color=_C['accent']), fill='tonexty',
                fillcolor='rgba(88,166,255,0.08)'))
            fig_pnl.add_vline(x=spot, line_dash="dash", line_color=_C['red'])
            fig_pnl.add_hline(y=0, line_width=0.5, line_color=_C['text_dim'])
            fig_pnl.update_layout(title="P&L Comparativo: Modelo Completo vs. Simplificado",
                                  yaxis_title="P&L ($ Mi)", xaxis_title="Preço do Ativo",
                                  height=380, template=DASH_TEMPLATE)

            # Dealer P&L
            fig_dealer = go.FigureWidget()
            fig_dealer.add_trace(go.Scatter(
                x=levels, y=pnl_curves['dealer'] / 1e6,
                mode='lines', name='P&L Dealer', line_color=_C['purple'],
                fill='tozeroy', fillcolor='rgba(188,140,255,0.08)'))
            fig_dealer.add_vline(x=spot, line_dash="dash", line_color=_C['red'])
            fig_dealer.add_hline(y=0, line_width=0.5, line_color=_C['text_dim'])
            fig_dealer.update_layout(title="P&L Estimado do Market Maker",
                                     yaxis_title="P&L ($ Mi)", xaxis_title="Preço do Ativo",
                                     height=380, template=DASH_TEMPLATE)

            # Hedge demand
            fig_hedge = go.FigureWidget()
            fig_hedge.add_trace(go.Scatter(
                x=levels, y=pnl_curves['hedge_demand'],
                mode='lines', line_color=_C['teal'], name='Contratos'))
            fig_hedge.add_vline(x=spot, line_dash="dash", line_color=_C['red'])
            fig_hedge.add_hline(y=0, line_width=0.5, line_color=_C['text_dim'])
            fig_hedge.update_layout(
                title=f"Demanda de Hedge em Futuros ({FUTURES_TICKER})",
                yaxis_title="Número de Contratos",
                xaxis_title="Preço do Ativo",
                height=380, template=DASH_TEMPLATE)

            tab4 = wd.VBox([fig_pnl, wd.HBox([fig_dealer, fig_hedge])])

            # ─── ABA 5: MONTE CARLO ──────────────────────────────────────
            sim_var_95 = np.percentile(mc_pnl, 5)
            sim_cvar_95 = mc_pnl[mc_pnl <= sim_var_95].mean() if np.any(mc_pnl <= sim_var_95) else sim_var_95
            sim_var_99 = np.percentile(mc_pnl, 1)
            sim_cvar_99 = mc_pnl[mc_pnl <= sim_var_99].mean() if np.any(mc_pnl <= sim_var_99) else sim_var_99
            mc_p1 = np.percentile(mc_pnl, 1)
            mc_p99 = np.percentile(mc_pnl, 99)
            mc_iqr = mc_p99 - mc_p1

            fig_mc_hist = go.FigureWidget()
            fig_mc_hist.add_trace(go.Histogram(
                x=mc_pnl / 1e6, nbinsx=120,
                marker_color=_C['accent'], opacity=0.7, name='P&L'))
            fig_mc_hist.add_vline(x=sim_var_99 / 1e6, line_dash='dash',
                                  line_color=_C['red'],
                                  annotation_text=f'VaR 99% ${sim_var_99/1e6:,.1f}M',
                                  annotation_position='top left')
            fig_mc_hist.add_vline(x=sim_var_95 / 1e6, line_dash='dash',
                                  line_color=_C['orange'],
                                  annotation_text=f'VaR 95% ${sim_var_95/1e6:,.1f}M',
                                  annotation_position='bottom left')
            fig_mc_hist.add_vline(x=0, line_width=0.5, line_color=_C['text_dim'])
            mc_xlo = (mc_p1 - mc_iqr * 0.15) / 1e6
            mc_xhi = (mc_p99 + mc_iqr * 0.15) / 1e6
            fig_mc_hist.update_layout(
                title=f'Distribuição de P&L do Livro (10k Sim. t-Student, {mc_n_days} Dias)',
                xaxis_title='P&L ($ Mi)', yaxis_title='Frequência',
                xaxis_range=[mc_xlo, mc_xhi],
                height=420, template=DASH_TEMPLATE)

            mc_win_pct = (mc_pnl > 0).mean() * 100
            mc_max_loss = mc_pnl.min()
            mc_max_gain = mc_pnl.max()
            mc_table = pd.DataFrame({
                'Métrica': ['P&L Médio', 'P&L Mediano', '% Cenários Positivos',
                            'VaR 95% (Sim.)', 'CVaR 95% (Sim.)',
                            'VaR 99% (Sim.)', 'CVaR 99% (Sim.)',
                            'Perda Máxima', 'Ganho Máximo',
                            f'Horizonte ({mc_n_days}d)'],
                'Valor': [f'${np.mean(mc_pnl)/1e6:,.2f} Mi',
                          f'${np.median(mc_pnl)/1e6:,.2f} Mi',
                          f'{mc_win_pct:.1f}%',
                          f'${sim_var_95/1e6:,.2f} Mi',
                          f'${sim_cvar_95/1e6:,.2f} Mi',
                          f'${sim_var_99/1e6:,.2f} Mi',
                          f'${sim_cvar_99/1e6:,.2f} Mi',
                          f'${mc_max_loss/1e6:,.2f} Mi',
                          f'${mc_max_gain/1e6:,.2f} Mi',
                          f'{mc_n_days} dias úteis']
            }).to_html(classes='mm-table', index=False, border=0)

            tab5 = wd.VBox([
                wd.HTML("<div class='mm-dash'><div class='mm-card'>"
                        f"<h3>Simulação Monte Carlo (t-Student, {mc_n_days} Dias)</h3></div></div>"),
                wd.HBox([fig_mc_hist, wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>{mc_table}</div></div>")])
            ])

            # ─── ABA 6: REBALANCEAMENTO ETFs + ALAVANCADOS ─────────────
            if etf_ok and etf_flows:
                # Dropdown para selecionar ETF
                etf_dd = wd.Dropdown(
                    options=['Combined'] + PASSIVE_ETFS,
                    value='Combined', description='ETF:',
                    layout={'width': '300px'})
                flow_html = wd.HTML()
                summary_html = wd.HTML()

                def _render_etf(change=None):
                    key = etf_dd.value
                    df_f = etf_flows.get(key, pd.DataFrame())
                    if df_f.empty:
                        flow_html.value = "<p>Sem dados.</p>"
                        return
                    top = df_f.head(30)
                    flow_html.value = (
                        top[['Start', 'Now', 'Delta', 'Flow_$', 'PctADV']]
                        .style.format({
                            'Start': '{:.4f}', 'Now': '{:.4f}', 'Delta': '{:+.4f}',
                            'Flow_$': '${:,.0f}', 'PctADV': '{:.1f}%'})
                        .background_gradient(cmap='RdYlGn', subset=['Flow_$'])
                        .to_html())
                etf_dd.observe(_render_etf, names='value')
                _render_etf()

                if etf_summary is not None:
                    summary_html.value = (
                        "<h4>Resumo por ETF</h4>" +
                        etf_summary.style.format('${:,.0f}')
                            .background_gradient(cmap='RdYlGn', subset=['Net_$'])
                            .to_html())

                # Seção de ETFs alavancados
                if lev_ok and lev_flows is not None:
                    lev_html_str = (
                        f"<h4>ETFs Alavancados (Retorno diário: {daily_ret*100:+.2f}%)</h4>" +
                        lev_flows[['Leverage', 'AUM_$', 'Rebalance_$', 'Direção']]
                        .style.format({
                            'AUM_$': '${:,.0f}', 'Rebalance_$': '${:,.0f}'})
                        .to_html() +
                        f"<p><b>Fluxo Direcional Total: ${lev_total:,.0f}</b></p>")
                else:
                    lev_html_str = "<p>ETFs alavancados não disponíveis.</p>"

                # Top 30 que ganham fluxo de compra e top 30 que levam fluxo de venda
                combo_flow = etf_flows.get('Combined', pd.DataFrame())
                flow_ranking = wd.HTML()
                if not combo_flow.empty and 'Flow_$' in combo_flow.columns:
                    top30_buy = combo_flow.nlargest(30, 'Flow_$')
                    top30_sell = combo_flow.nsmallest(30, 'Flow_$')
                    fmt_dict = {'Delta': '{:+.4f}', 'Flow_$': '${:,.0f}', 'PctADV': '{:.1f}%'}
                    buy_html = (top30_buy[['Delta', 'Flow_$', 'PctADV']]
                        .style.format(fmt_dict)
                        .background_gradient(cmap='Greens', subset=['Flow_$'])
                        .to_html())
                    sell_html = (top30_sell[['Delta', 'Flow_$', 'PctADV']]
                        .style.format(fmt_dict)
                        .background_gradient(cmap='Reds', subset=['Flow_$'])
                        .to_html())
                    flow_ranking = wd.HBox([
                        wd.VBox([wd.HTML("<div class='mm-dash'><div class='mm-card'><h4>Top 30 — Fluxo de Compra</h4></div></div>"),
                                 wd.HTML(buy_html)]),
                        wd.VBox([wd.HTML("<div class='mm-dash'><div class='mm-card'><h4>Top 30 — Fluxo de Venda</h4></div></div>"),
                                 wd.HTML(sell_html)])
                    ])

                tab6 = wd.VBox([
                    wd.HTML(f"<div class='mm-dash'><div class='mm-card'><h3>Rebalanceamento ETFs Passivos desde {etf_start}</h3></div></div>"),
                    etf_dd, flow_html, summary_html,
                    flow_ranking,
                    wd.HTML(lev_html_str)
                ])
            else:
                tab6 = wd.VBox([wd.HTML(
                    "<h3>Rebalanceamento ETFs</h3>"
                    "<p>Dados de rebalanceamento não disponíveis.</p>")])

            # ─── ABA 7: PREVISÃO SPX ────────────────────────────────────
            if spx_pred_ok and top_in_spx is not None:
                cols_in = ['Prob_In', 'CUR_MKT_CAP', 'FMC', 'FALR', 'FREE_FLOAT_PCT', 'NET_INC_TTM']
                cols_out = ['ExitScore', 'CUR_MKT_CAP', 'FMC', 'FALR', 'FREE_FLOAT_PCT', 'NET_INC_TTM']
                in_html = (top_in_spx[[c for c in cols_in if c in top_in_spx.columns]]
                    .style.format({
                        'Prob_In': '{:.2%}', 'CUR_MKT_CAP': '${:,.0f}',
                        'FMC': '${:,.0f}', 'FALR': '{:.4f}',
                        'FREE_FLOAT_PCT': '{:.1f}%'})
                    .background_gradient(cmap='Greens', subset=['Prob_In'])
                    .to_html())
                out_html = (top_out_spx[[c for c in cols_out if c in top_out_spx.columns]]
                    .style.format({
                        'ExitScore': '{:.2%}', 'CUR_MKT_CAP': '${:,.0f}',
                        'FMC': '${:,.0f}', 'FALR': '{:.4f}',
                        'FREE_FLOAT_PCT': '{:.1f}%', 'NET_INC_TTM': '${:,.0f}'})
                    .background_gradient(cmap='Reds', subset=['ExitScore'])
                    .to_html())
                spx_rules_html = (
                    "<div class='mm-dash'><div class='mm-card' style='padding:10px;'>"
                    "<span class='mm-section-label'>Critérios S&P 500</span> "
                    f"<span class='mm-cot-label'>Market Cap ≥ $18B</span> "
                    f"<span class='mm-cot-label'>Free Float ≥ 50%</span> "
                    f"<span class='mm-cot-label'>FALR ≥ 0.75</span> "
                    f"<span class='mm-cot-label'>Lucro TTM > 0</span> "
                    f"<span class='mm-cot-label'>NI Q0 > 0</span>"
                    "</div></div>")
                tab7 = wd.VBox([
                    wd.HTML("<div class='mm-dash'><div class='mm-card'><h3>Previsão de Rebalanceamento S&P 500</h3></div></div>"),
                    wd.HTML(spx_rules_html),
                    wd.HBox([
                        wd.VBox([wd.HTML("<div class='mm-dash'><div class='mm-card'><h4>Top 30 — Candidatos a Entrar</h4></div></div>"),
                                 wd.HTML(in_html)]),
                        wd.VBox([wd.HTML("<div class='mm-dash'><div class='mm-card'><h4>Top 30 — Candidatos a Sair</h4></div></div>"),
                                 wd.HTML(out_html)])
                    ])
                ])
            else:
                reason = "Marque 'Incluir Previsão SPX' e rode novamente." if not spx_pred_w.value else (
                    "sklearn não disponível." if not HAS_SKLEARN else "Erro na execução.")
                tab7 = wd.VBox([wd.HTML(
                    f"<h3>Previsão SPX</h3><p>{reason}</p>")])

            # ─── ABA 8: SIMULADOR INTERATIVO ─────────────────────────────
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
                                             mode='lines', line_color=_C['accent'],
                                             name='Delta'))
            fig_sim_dex.update_layout(title="Delta Nocional", yaxis_title="$ Bi",
                                      height=300, template=DASH_TEMPLATE,
                                      margin=dict(t=30, b=20))

            fig_sim_gex = go.FigureWidget()
            fig_sim_gex.add_trace(go.Scatter(x=levels, y=np.zeros(len(levels)),
                                             mode='lines', line_color=_C['red'],
                                             name='Gamma'))
            fig_sim_gex.update_layout(title="Gamma (GEX)", yaxis_title="$ Bi / 1% move",
                                      height=300, template=DASH_TEMPLATE,
                                      margin=dict(t=30, b=20))

            fig_sim_vega = go.FigureWidget()
            fig_sim_vega.add_trace(go.Scatter(x=levels, y=np.zeros(len(levels)),
                                              mode='lines', line_color=_C['purple'],
                                              name='Vega'))
            fig_sim_vega.update_layout(title="Vega Nocional", yaxis_title="$ Mi",
                                       height=300, template=DASH_TEMPLATE,
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

            tab8 = wd.VBox([
                wd.HTML("<div class='mm-dash'><div class='mm-card'>"
                        "<h3>Simulador Interativo de Gregas</h3>"
                        "<p>Ajuste vol e tempo para ver como as exposições mudam.</p>"
                        "</div></div>"),
                wd.HBox([vol_slider, dte_slider]),
                wd.HBox([fig_sim_dex, fig_sim_gex]),
                fig_sim_vega
            ])

            # ─── ABA 9: RELATÓRIO DE RISCO ───────────────────────────────
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

            # ── Lógica de decisão do MM ──
            # Bias direcional
            if abs(delta_notional) < 50e6:
                delta_signal = '🟢 NEUTRO'
                delta_rec = 'Sem ajuste necessário.'
            elif delta_notional > 0:
                delta_signal = '🔴 COMPRADO'
                delta_rec = (f'VENDER {abs(hedge_contracts):,.0f} {FUTURES_TICKER} '
                             f'ou comprar puts para reduzir exposição.')
            else:
                delta_signal = '🔵 VENDIDO'
                delta_rec = (f'COMPRAR {abs(hedge_contracts):,.0f} {FUTURES_TICKER} '
                             f'ou comprar calls para reduzir exposição.')

            # Regime de gamma
            if total_gex_val > 0:
                gex_signal = '🟢 GAMMA POSITIVO'
                gex_rec = ('Mercado tende a se estabilizar. MM vende em alta, compra em '
                           'baixa → menor volatilidade realizada. Pode VENDER vol (straddles/strangles).')
            else:
                gex_signal = '🔴 GAMMA NEGATIVO'
                gex_rec = ('Mercado tende a se ACELERAR. MM compra em alta, vende em '
                           'baixa → maior volatilidade realizada. REDUZIR tamanho e '
                           'colocar stops mais apertados.')

            # Vol trade
            if vol_premium > 2:
                vol_signal = '🟢 IV > RV (prêmio alto)'
                vol_rec = 'Oportunidade de VENDER vol — spreads de crédito, iron condors.'
            elif vol_premium < -2:
                vol_signal = '🔴 RV > IV (vol barata)'
                vol_rec = 'Oportunidade de COMPRAR vol — straddles, calendars.'
            else:
                vol_signal = '🟡 VOL NEUTRA'
                vol_rec = 'Sem edge claro em vol. Preferir posições direcionais.'

            # Skew trade
            if skew > 0.03:
                skew_signal = '🔴 SKEW ELEVADO'
                skew_rec = ('Puts caras vs calls. Considerar risk reversals '
                            '(vender put OTM, comprar call OTM) ou put spreads.')
            elif skew < 0.01:
                skew_signal = '🟢 SKEW COMPRIMIDO'
                skew_rec = 'Proteção barata. Comprar puts OTM como hedge de cauda.'
            else:
                skew_signal = '🟡 SKEW NORMAL'
                skew_rec = 'Sem distorção significativa no skew.'

            # Vanna/Zomma trade
            if vanna_impact > 0:
                vanna_rec = ('Vol subindo → dealers precisam VENDER o ativo (pressão baixista). '
                             'Cuidado com posições compradas.')
            else:
                vanna_rec = ('Vol subindo → dealers precisam COMPRAR o ativo (pressão altista). '
                             'Suporte adicional em quedas.')

            # Theta vs Gamma
            daily_theta = np.nansum(greeks_now['theta'] * oi_100) / TRADING_DAYS
            gamma_pnl_1pct = total_gex_val
            theta_gamma_ratio = abs(daily_theta / gamma_pnl_1pct) if gamma_pnl_1pct != 0 else 0
            if theta_gamma_ratio > 1:
                tg_signal = '🟢 THETA DOMINANTE'
                tg_rec = ('Decaimento diário supera risco de gamma. '
                          'Posições vendidas em vol são favorecidas.')
            else:
                tg_signal = '🔴 GAMMA DOMINANTE'
                tg_rec = ('Risco de gamma supera o decaimento. '
                          'Movimentos grandes serão caros. Reduzir exposição vendida.')

            # Urgência geral
            urgency_score = 0
            if abs(delta_notional) > 200e6: urgency_score += 2
            if total_gex_val < 0: urgency_score += 2
            if abs(vol_premium) > 3: urgency_score += 1
            if skew > 0.04: urgency_score += 1
            if urgency_score >= 4:
                urgency = '🔴 ALTA — Ajustes imediatos recomendados'
            elif urgency_score >= 2:
                urgency = '🟡 MÉDIA — Monitorar e preparar ordens'
            else:
                urgency = '🟢 BAIXA — Posição confortável'

            report_html = f"""
            <div class='mm-dash'>
            <div class='mm-card'>
            <h3 style='font-size:18px;'>COCKPIT DO MARKET MAKER</h3>
            <p>Análise: {datetime.now().strftime('%Y-%m-%d %H:%M')} │ Spot: ${spot:,.2f} │
               {len(df)} opções │ {urgency}</p>
            </div>

            <div class='mm-card'>
            <h3>DECISÕES RECOMENDADAS</h3>
            <table class='mm-table'>
            <tr><th>Dimensão</th><th>Sinal</th><th>Ação Recomendada</th></tr>
            <tr><td><b>Delta (Direção)</b></td><td style='text-align:center;'>{delta_signal}</td><td>{delta_rec}</td></tr>
            <tr><td><b>Gamma (Regime)</b></td><td style='text-align:center;'>{gex_signal}</td><td>{gex_rec}</td></tr>
            <tr><td><b>Vol Premium</b></td><td style='text-align:center;'>{vol_signal}</td><td>{vol_rec}</td></tr>
            <tr><td><b>Skew</b></td><td style='text-align:center;'>{skew_signal}</td><td>{skew_rec}</td></tr>
            <tr><td><b>Vanna (Vol→Spot)</b></td><td style='text-align:center;'>{'🔴' if vanna_impact > 0 else '🟢'} {vanna_action}</td><td>{vanna_rec}</td></tr>
            <tr><td><b>Theta vs Gamma</b></td><td style='text-align:center;'>{tg_signal}</td><td>{tg_rec}</td></tr>
            </table>
            </div>

            <div class='mm-card'>
            <h3>POSIÇÃO ATUAL</h3>
            <table class='mm-table'>
            <tr><th>Exposição</th><th style='text-align:right;'>Valor</th><th>Interpretação</th></tr>
            <tr><td>Delta Nocional</td><td style='text-align:right;'>{fmt_value(delta_notional)}</td><td>Hedge: {hedge_action} {abs(hedge_contracts):,.0f} {FUTURES_TICKER}</td></tr>
            <tr><td>GEX Total</td><td style='text-align:right;'>{fmt_value(total_gex_val)}</td><td>Flip: ~{f'{gamma_flip:,.0f}' if gamma_flip else 'N/A'} │ {gamma_regime}</td></tr>
            <tr><td>Vega</td><td style='text-align:right;'>{fmt_value(total_vega)}</td><td>P&L por 1% de aumento na vol</td></tr>
            <tr><td>Vanna</td><td style='text-align:right;'>{fmt_value(vanna_impact)}</td><td>Fluxo de rebalanceamento se vol mudar</td></tr>
            <tr><td>Zomma</td><td style='text-align:right;'>{fmt_value(total_zomma)}</td><td>Regime ficaria {zomma_action} com vol</td></tr>
            <tr><td>Speed</td><td style='text-align:right;'>{fmt_value(total_speed)}</td><td>Aceleração do gamma por $1 no spot</td></tr>
            <tr><td>Charm (diário)</td><td style='text-align:right;'>{fmt_value(total_charm)}</td><td>Decaimento do delta overnight</td></tr>
            </table>
            </div>

            <div class='mm-card'>
            <h3>RISCO CAUDAL</h3>
            <table class='mm-table' style='width:70%;'>
            <tr><th>Métrica</th><th>Paramétrico</th><th>Monte Carlo</th></tr>
            <tr><td>VaR 95%</td><td style='color:{_C['yellow']}'>{risk['var_95']:.2%}</td><td style='color:{_C['yellow']}'>${sim_var_95/1e6:,.2f} Mi</td></tr>
            <tr><td>CVaR 95%</td><td style='color:{_C['orange']}'>{risk['cvar_95']:.2%}</td><td style='color:{_C['orange']}'>${sim_cvar_95/1e6:,.2f} Mi</td></tr>
            <tr><td>VaR 99%</td><td style='color:{_C['red']}'>{risk['var_99']:.2%}</td><td style='color:{_C['red']}'>${sim_var_99/1e6:,.2f} Mi</td></tr>
            <tr><td>CVaR 99%</td><td style='color:{_C['red']}'>{risk['cvar_99']:.2%}</td><td style='color:{_C['red']}'>${sim_cvar_99/1e6:,.2f} Mi</td></tr>
            </table>
            </div>

            <div class='mm-card'>
            <h3>NÍVEIS-CHAVE</h3>
            <div class='mm-kpi-row'>
                <div class='mm-kpi'><div class='kpi-label'>Call Wall</div><div class='kpi-value' style='color:{_C['green']}'>{f'{call_wall:,.0f}' if call_wall else 'N/A'}</div><div style='font-size:11px;color:{_C['text_muted']}'>Resistência forte</div></div>
                <div class='mm-kpi'><div class='kpi-label'>Put Wall</div><div class='kpi-value' style='color:{_C['red']}'>{f'{put_wall:,.0f}' if put_wall else 'N/A'}</div><div style='font-size:11px;color:{_C['text_muted']}'>Suporte forte</div></div>
                <div class='mm-kpi'><div class='kpi-label'>Gamma Flip</div><div class='kpi-value' style='color:{_C['yellow']}'>{f'{gamma_flip:,.0f}' if gamma_flip else 'N/A'}</div><div style='font-size:11px;color:{_C['text_muted']}'>Acima = estabilidade</div></div>
                <div class='mm-kpi'><div class='kpi-label'>Mov. Implícito 1D</div><div class='kpi-value' style='color:{_C['accent']}'>±{daily_move:.2f}%</div><div style='font-size:11px;color:{_C['text_muted']}'>±${spot * daily_move / 100:,.0f}</div></div>
            </div>
            </div>
            </div>"""

            tab9 = wd.VBox([wd.HTML(report_html)])

            # ─── ABA 10: FLOW PREDICTOR ──────────────────────────────────
            if fp_ok and fp_score is not None:
                # Score summary header
                fp_title = wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h3 style='font-size:18px;'>Flow Predictor — {ticker}</h3>"
                    f"<p>Direction: <b style='color:"
                    f"{_C['green'] if fp_score['direction'] == 'BULLISH' else _C['red'] if fp_score['direction'] == 'BEARISH' else _C['text_muted']}'>"
                    f"{fp_score['direction']}</b> │ "
                    f"P(Up): {fp_score['prob_up']:.1%} │ "
                    f"Score: {fp_score['combined_score']:+.2f}</p>"
                    f"</div></div>")

                # Sub-tab A: Score
                st_a = wd.VBox([
                    fp_title,
                    wd.HBox([fp_plot_score_gauge(fp_score),
                             fp_plot_components_bar(fp_score)]),
                    fp_grid_flow_score(fp_score)
                ])

                # Sub-tab B: Histórico
                st_b_children = [wd.HTML("<div class='mm-dash'><div class='mm-card'><h3>Histórico de Fluxo — ETFs Alavancados</h3></div></div>")]
                if not fp_flow_hist.empty:
                    st_b_children.append(fp_plot_flow_history(fp_flow_hist))
                    st_b_children.append(fp_bqp_flow_bar_line(fp_flow_hist.tail(60)))
                    st_b_children.append(fp_bqp_scatter(fp_flow_hist))
                st_b = wd.VBox(st_b_children)

                # Sub-tab C: Buyback
                st_c_children = [
                    wd.HTML("<div class='mm-dash'><div class='mm-card'>"
                            "<h3>Estimativa de Buyback</h3>"
                            "<p>⚠️ Confiança baixa: não temos % ADV executado"
                            " nem saldo restante.</p></div></div>")]
                bb_pct_adv = fp_buyback.get('pct_adv_est', 0)
                bb_pct_str = 'N/A' if (bb_pct_adv is None or pd.isna(bb_pct_adv)) else f'{bb_pct_adv:.2f}%'
                bb_html = (
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<table class='mm-table' style='width:auto;'>"
                    f"<tr><td>Anunciado:</td><td style='text-align:right;'>${fp_buyback.get('announced', 0):,.0f}</td></tr>"
                    f"<tr><td>Estimativa diária:</td><td style='text-align:right;'>${fp_buyback.get('daily_est', 0):,.0f}</td></tr>"
                    f"<tr><td>% ADV estimado:</td><td style='text-align:right;'>"
                    f"{bb_pct_str}</td></tr>"
                    f"<tr><td>Confiança:</td><td style='text-align:right;'>{fp_buyback.get('confidence', 'N/A')}</td></tr>"
                    f"</table></div></div>")
                st_c_children.append(wd.HTML(bb_html))
                try:
                    bb_df = estimate_index_buyback_flow(ticker, top_n=30)
                    if not bb_df.empty:
                        st_c_children.append(wd.HTML("<h4>Top Buybacks do Índice</h4>"))
                        st_c_children.append(fp_grid_buyback(bb_df))
                except Exception:
                    pass
                st_c = wd.VBox(st_c_children)

                # Sub-tab D: COT
                st_d_children = [wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h3>COT — Commitment of Traders</h3>"
                    f"<p>Dados agregados (soma de todas trader types). "
                    f"Report type: auto. "
                    f"<span class='mm-cot-label'>Period: <b>{cot_start_w.value} → {cot_end_w.value}</b></span></p>"
                    f"</div></div>")]
                cot_ok_fp2, cot_fut_fp2 = has_cot(ticker)
                if cot_ok_fp2 and fp_cot_df is not None and not fp_cot_df.empty:
                    st_d_children.append(
                        wd.HTML(f"<p>Futures: <b>{cot_fut_fp2}</b> — "
                                f"<b>{len(fp_cot_df)}</b> registros — "
                                f"WoW Δ Net: <b>{cot_net_change:+,.0f}</b></p>"))
                    st_d_children.append(fp_grid_cot_stats(fp_cot_stats))
                    seas = cot_seasonality(fp_cot_df)
                    st_d_children.append(wd.HBox([
                        fp_plot_positions_basket(fp_cot_df),
                        fp_plot_dispersion(seas, fp_cot_df)
                    ]))
                    st_d_children.append(fp_plot_long_short_net(fp_cot_df))
                    st_d_children.append(wd.HBox([
                        fp_plot_long_short_ratio(fp_cot_df),
                        fp_plot_correlation(fp_cot_df)
                    ]))
                    st_d_children.append(fp_plot_multi_year(fp_cot_df))
                elif cot_ok_fp2:
                    st_d_children.append(
                        wd.HTML(f"<p>COT disponível para {cot_fut_fp2}, "
                                "mas sem dados retornados.</p>"))
                else:
                    st_d_children.append(
                        wd.HTML(f"<p>{ticker} não possui dados COT vinculados. "
                                "Use os contratos selecionados abaixo.</p>"))

                if fp_selected_cot_df is not None and not fp_selected_cot_df.empty:
                    sel_label = ', '.join(selected_cots)
                    st_d_children.append(wd.HTML(f"<hr><h4>COT: {sel_label}</h4>"))
                    sel_stats = cot_summary_stats(fp_selected_cot_df)
                    st_d_children.append(fp_grid_cot_stats(sel_stats))
                    sel_seas = cot_seasonality(fp_selected_cot_df)
                    st_d_children.append(wd.HBox([
                        fp_plot_positions_basket(fp_selected_cot_df),
                        fp_plot_long_short_net(fp_selected_cot_df)
                    ]))
                    st_d_children.append(wd.HBox([
                        fp_plot_dispersion(sel_seas, fp_selected_cot_df),
                        fp_plot_multi_year(fp_selected_cot_df)
                    ]))
                st_d = wd.VBox(st_d_children)

                # Sub-tab E: Correlação
                st_e_children = [wd.HTML("<div class='mm-dash'><div class='mm-card'><h3>Análise de Correlação</h3></div></div>")]
                if not fp_flow_hist.empty:
                    df_test = fp_flow_hist.copy()
                    df_test['flow_signal'] = np.sign(df_test['LevETF_Flow'])
                    df_test['next_ret'] = df_test['Return'].shift(-1)
                    df_test['hit'] = (np.sign(df_test['flow_signal'])
                                      == np.sign(df_test['next_ret']))
                    hit_rate = df_test['hit'].mean()
                    st_e_children.append(wd.HTML(
                        f"<p><b>Hit Rate (flow signal vs next-day return):"
                        f"</b> {hit_rate:.1%} ({len(df_test)} obs)</p>"
                        f"<p><i>Nota: fluxo de rebalanceamento é contra-tendência"
                        f" (mean-reverting). Hit rate &lt; 50% é esperado.</i></p>"))
                st_e = wd.VBox(st_e_children)

                # Sub-tab F: Fluxos Sistemáticos (CTA + Dealer + Vol Control + Risk Parity)
                st_f_children = [wd.HTML(
                    "<div class='mm-dash'><div class='mm-card'>"
                    "<h3>Fluxos Sistemáticos — CTA, Dealer/MM, Vol Control, Risk Parity</h3>"
                    "<p><small>Ref: BofA Systematic Flows Monitor methodology</small></p>"
                    "</div></div>")]

                # CTA Trend Following
                _cta_flow = fp_cta.get('flow', 0)
                _cta_trend = fp_cta.get('trend_today', 0)
                _cta_pos = fp_cta.get('pos_today', 0)
                _cta_pos_prev = fp_cta.get('pos_prev', 0)
                _cta_color = _C['green'] if _cta_flow > 0 else _C['red'] if _cta_flow < 0 else _C['text_muted']
                _cta_dir = 'COMPRA' if _cta_flow > 0 else 'VENDA' if _cta_flow < 0 else 'FLAT'
                _trend_bar = '█' * max(1, int(abs(_cta_trend) * 10))
                _trend_color = _C['green'] if _cta_trend > 0 else _C['red'] if _cta_trend < 0 else _C['text_muted']
                st_f_children.append(wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h4>CTA / Trend Following (AUM: ~$340B, ~25% equity)</h4>"
                    f"<table class='mm-table' style='width:auto;'>"
                    f"<tr><td>Trend Strength:</td>"
                    f"<td style='color:{_trend_color}'><b>{_cta_trend:+.3f}</b> "
                    f"<span style='font-size:10px;'>{_trend_bar}</span></td></tr>"
                    f"<tr><td>Posição CTA:</td>"
                    f"<td>{_cta_pos_prev:+.3f}x → <b>{_cta_pos:+.3f}x</b></td></tr>"
                    f"<tr><td>Fluxo Estimado:</td>"
                    f"<td style='color:{_cta_color}'><b>${_cta_flow/1e9:,.2f}B</b> ({_cta_dir})</td></tr>"
                    f"</table>"
                    f"<p><small>Trend = média de MA crosses (5/20, 5/60, 10/60, 20/120, 20/200). "
                    f"Sizing = trend/vol. CTAs ajustam diariamente.</small></p>"
                    f"</div></div>"))

                # Dealer flow
                _dl_color = _C['green'] if fp_dealer_flow > 0 else _C['red'] if fp_dealer_flow < 0 else _C['text_muted']
                _dl_dir = 'COMPRA' if fp_dealer_flow > 0 else 'VENDA' if fp_dealer_flow < 0 else 'NEUTRO'
                st_f_children.append(wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h4>Dealer/Market Maker Delta Hedging</h4>"
                    f"<p>GEX-implied flow: <b style='color:{_dl_color}'>"
                    f"${fp_dealer_flow:,.0f}</b> ({_dl_dir})</p>"
                    f"<p><small>Baseado em GEX × ΔS. Dados de gamma derivados de "
                    f"signed volume (firms + MMs + flex). Short gamma → pro-cíclico.</small></p>"
                    f"</div></div>"))

                # Vol control
                vc_total = fp_volctrl.get('total', 0)
                _vc_color = _C['green'] if vc_total > 0 else _C['red'] if vc_total < 0 else _C['text_muted']
                vc_rows = ""
                for tv_k, tv_v in fp_volctrl.get('detail', {}).items():
                    tv_flow = tv_v.get('flow', 0)
                    tv_c = _C['green'] if tv_flow > 0 else _C['red'] if tv_flow < 0 else _C['text_muted']
                    vc_rows += (
                        f"<tr><td>Target {tv_k}</td>"
                        f"<td style='text-align:right;'>${tv_v.get('aum', 0)/1e9:,.0f}B</td>"
                        f"<td style='text-align:right;'>{tv_v.get('exposure_old', 0):.2f}x</td>"
                        f"<td style='text-align:right;'>{tv_v.get('exposure_new', 0):.2f}x</td>"
                        f"<td style='text-align:right;color:{tv_c}'>${tv_flow/1e9:,.2f}B</td></tr>")
                st_f_children.append(wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h4>Equity Vol Control (AUM: ~$300B)</h4>"
                    f"<p>Fluxo total estimado: <b style='color:{_vc_color}'>"
                    f"${vc_total/1e9:,.2f}B</b></p>"
                    f"<table class='mm-table'>"
                    f"<tr><th>Target Vol</th><th>AUM Est.</th>"
                    f"<th>Exp. Anterior</th><th>Exp. Atual</th><th>Fluxo</th></tr>"
                    f"{vc_rows}</table>"
                    f"<p><small>Leverage = target_vol / realized_vol (21d). "
                    f"Vol sobe → exposure cai → vendem. Ajuste em 1-2 dias.</small></p>"
                    f"</div></div>"))

                # Risk Parity
                rp_total = fp_rp.get('total', 0)
                _rp_color = _C['green'] if rp_total > 0 else _C['red'] if rp_total < 0 else _C['text_muted']
                rp_eq_new = fp_rp.get('eq_alloc_new', 0)
                rp_eq_old = fp_rp.get('eq_alloc_old', 0)
                rp_rows = ""
                for rp_k, rp_v in fp_rp.get('detail', {}).items():
                    rp_f = rp_v.get('flow', 0)
                    rp_c = _C['green'] if rp_f > 0 else _C['red'] if rp_f < 0 else _C['text_muted']
                    rp_rows += (
                        f"<tr><td>Target {rp_k}</td>"
                        f"<td style='text-align:right;'>${rp_v.get('aum', 0)/1e9:,.0f}B</td>"
                        f"<td style='text-align:right;'>{rp_v.get('leverage_old', 0):.2f}x</td>"
                        f"<td style='text-align:right;'>{rp_v.get('leverage_new', 0):.2f}x</td>"
                        f"<td style='text-align:right;'>{rp_v.get('eq_alloc_old', 0):.1%}</td>"
                        f"<td style='text-align:right;'>{rp_v.get('eq_alloc_new', 0):.1%}</td>"
                        f"<td style='text-align:right;color:{rp_c}'>${rp_f/1e9:,.2f}B</td></tr>")
                st_f_children.append(wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h4>Risk Parity (AUM: ~$200B, equity alloc: {rp_eq_old:.1%} → {rp_eq_new:.1%})</h4>"
                    f"<p>Fluxo equity estimado: <b style='color:{_rp_color}'>"
                    f"${rp_total/1e9:,.2f}B</b></p>"
                    f"<table class='mm-table'>"
                    f"<tr><th>Target Vol</th><th>AUM Est.</th>"
                    f"<th>Lev. Ant.</th><th>Lev. Atual</th>"
                    f"<th>Eq% Ant.</th><th>Eq% Atual</th><th>Fluxo</th></tr>"
                    f"{rp_rows}</table>"
                    f"<p><small>Alocação ∝ 1/vol por asset class (equities, bonds, commodities). "
                    f"Rebalanceamento mensal. Vol↑ → equity alloc↓ → vendem.</small></p>"
                    f"</div></div>"))
                st_f = wd.VBox(st_f_children)

                fp_tabs = wd.Tab()
                fp_tabs.children = [st_a, st_b, st_c, st_d, st_e, st_f]
                for idx_t, nm in enumerate(['Score', 'Histórico', 'Buyback',
                                            'COT', 'Correlação', 'Sistemáticos']):
                    fp_tabs.set_title(idx_t, nm)
                tab10 = fp_tabs
            else:
                reason = ("Marque 'Incluir Flow Predictor' e rode novamente."
                          if not flow_pred_w.value else "Erro na execução.")
                tab10 = wd.VBox([wd.HTML(
                    f"<h3>Flow Predictor</h3><p>{reason}</p>")])

            # ═════════════════════════════════════════════════════════════
            # MONTAGEM FINAL
            # ═════════════════════════════════════════════════════════════
            dashboard = wd.Tab()
            dashboard.children = [tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10]
            tab_names = [
                'Visão Geral', 'Exposições', 'Sensibilidade', 'Análise P&L',
                'Monte Carlo', 'Rebalanceamento', 'Previsão SPX',
                'Simulador', 'Relatório', 'Flow Predictor'
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

_ctrl_box_layout = wd.Layout(
    border=f'1px solid {_C["border"]}',
    border_radius='8px',
    padding='12px',
    margin='4px 0',
)

display(wd.HTML(DASH_CSS))
display(wd.VBox([
    wd.HTML(f"<div class='mm-dash'><div class='mm-section-label'>Parâmetros da Análise</div></div>"),
    wd.VBox([
        wd.HBox([ticker_w, dte_w]),
        mny_w,
        wd.HBox([run_btn, spx_pred_w, flow_pred_w]),
    ], layout=_ctrl_box_layout),
    wd.HTML(f"<div class='mm-dash'><div class='mm-section-label'>COT Controls</div></div>"),
    wd.VBox([
        wd.HBox([cot_type_w, cot_contract_w]),
        wd.HBox([cot_trader_w]),
        wd.HBox([cot_start_w, cot_end_w]),
        wd.HBox([cot_reload_btn, etf_reload_btn]),
    ], layout=_ctrl_box_layout),
    out_cot_reload,
    out_main
]))
