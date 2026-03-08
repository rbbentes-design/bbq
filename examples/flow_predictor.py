"""
FLOW PREDICTOR — Estatística Preditiva de Fluxo Contratado
===========================================================

Combina múltiplas fontes de fluxo em um score preditivo:
  1. Rebalanceamento de ETFs alavancados (mecânico, diário)
  2. Buyback corporativo (anunciado, execução incerta — estimado)
  3. Previsão de rebalanceamento do SPX (probabilístico)
  4. COT — Commitment of Traders (vinculado a futuros quando disponível)

Gera:
  • Z-score e percentil de cada componente
  • Score combinado de direção de fluxo
  • Probabilidade direcional (alta/baixa) baseada em padrões históricos
  • Visualizações: bqplot, plotly, ipydatagrid (nativo BQuant)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 0 — IMPORTS E CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

import math
import warnings
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import ipywidgets as wd
from IPython.display import display

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# bqplot (nativo BQuant)
try:
    import bqplot as bqp
    HAS_BQPLOT = True
except ImportError:
    HAS_BQPLOT = False

# ipydatagrid
try:
    from ipydatagrid import DataGrid, BarRenderer, TextRenderer
    HAS_DATAGRID = True
except ImportError:
    HAS_DATAGRID = False

try:
    import bql
except ImportError:
    raise ImportError("Requer Bloomberg BQL para Python.")

bq = bql.Service()

TRADING_DAYS = 252.0

# Layout padrão Plotly (estilo escuro)
FIG_LAYOUT = {
    'template': 'plotly_dark',
    'font': {'family': 'Nunito', 'size': 13},
    'legend': {'orientation': "h", 'yanchor': "bottom", 'y': -0.3,
               'xanchor': "center", 'x': 0.5},
    'margin': {'t': 40, 'b': 40, 'l': 10, 'r': 10},
}

# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — HELPERS BQL
# ═══════════════════════════════════════════════════════════════════════════════

def _last_numeric_col(df: pd.DataFrame):
    for c in df.columns[::-1]:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def _bql_to_series(ticker, item, name=None):
    """Executa BQL Request simples e retorna pd.Series indexada por data."""
    try:
        df = bq.execute(bql.Request([ticker], item))[0].df()
        if df is None or df.empty:
            return pd.Series(dtype=float, name=name or ticker)
        dcol = next((c for c in df.columns if "date" in str(c).lower()), None)
        vcol = _last_numeric_col(df)
        if dcol is None or vcol is None:
            return pd.Series(dtype=float, name=name or ticker)
        s = df.set_index(dcol)[vcol].astype(float).sort_index()
        s.index = pd.to_datetime(s.index)
        s.name = name or ticker
        return s
    except Exception:
        return pd.Series(dtype=float, name=name or ticker)


@lru_cache(maxsize=256)
def px_last_series(ticker: str, start="-500D", end="0D"):
    item = bq.data.px_last(dates=bq.func.range(start, end), fill='PREV')
    return _bql_to_series(ticker, item, name=ticker)


@lru_cache(maxsize=256)
def px_volume_series(ticker: str, start="-500D", end="0D"):
    item = bq.data.px_volume(dates=bq.func.range(start, end), fill='PREV')
    return _bql_to_series(ticker, item, name=ticker)


def get_data(universe, data_items, with_params=None, preferences=None):
    """Executa BQL Request e retorna DataFrame com multi-index [ID, Date]."""
    req = bql.Request(universe, data_items,
                      with_params=with_params, preferences=preferences)
    response = bq.execute(req)
    df = pd.concat([di.df() for di in response], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    if 'DATE' in df.columns:
        df = df.set_index('DATE', append=True)
    df = df.loc[:, data_items.keys()]
    df.index.names = ['ID', 'Date']
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — COT ENGINE (Commitment of Traders)
# ═══════════════════════════════════════════════════════════════════════════════

# Mapeamento: ticker → futures com dados COT
COT_FUTURES_MAP = {
    # Índices de ações
    'SPX Index': 'ES1 Comdty',
    'NDX Index': 'NQ1 Comdty',
    'RTY Index': 'RA1 Comdty',
    'INDU Index': 'DM1 Comdty',
    # ETFs alavancados → via índice subjacente
    'SPXL US Equity': 'ES1 Comdty',
    'SPXS US Equity': 'ES1 Comdty',
    'TQQQ US Equity': 'NQ1 Comdty',
    'SQQQ US Equity': 'NQ1 Comdty',
    # Commodities (já têm COT diretamente)
    'NG1 Comdty': 'NG1 Comdty',
    'CL1 Comdty': 'CL1 Comdty',
    'CO1 Comdty': 'CO1 Comdty',
    'GC1 Comdty': 'GC1 Comdty',
    'SI1 Comdty': 'SI1 Comdty',
    'HG1 Comdty': 'HG1 Comdty',
    'W 1 Comdty': 'W 1 Comdty',
    'S 1 Comdty': 'S 1 Comdty',
    'C 1 Comdty': 'C 1 Comdty',
}

# Categorias de contratos para o dropdown
COT_CONTRACTS = {
    'Equity Indices': [
        ('S&P 500 E-mini', 'ES1 Comdty'),
        ('Nasdaq 100 E-mini', 'NQ1 Comdty'),
        ('Russell 2000 E-mini', 'RA1 Comdty'),
    ],
    'Energy': [
        ('WTI Crude', 'CL1 Comdty'),
        ('Brent Crude', 'CO1 Comdty'),
        ('Natural Gas', 'NG1 Comdty'),
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


def has_cot(ticker: str) -> Tuple[bool, Optional[str]]:
    """Verifica se o ticker possui dados COT. Retorna (True, futures_ticker) ou (False, None)."""
    # Direto no mapa
    if ticker in COT_FUTURES_MAP:
        return True, COT_FUTURES_MAP[ticker]
    # Se termina em Comdty, assume que tem COT
    if ticker.strip().endswith('Comdty'):
        return True, ticker
    return False, None


def fetch_cot_data(futures_ticker: str, trader_type='Managed Money',
                   report_type='CFTC Disaggregated', start='-6Y',
                   end='0D') -> pd.DataFrame:
    """
    Busca dados COT do BQL para um contrato futuro.
    Retorna DataFrame com colunas: Traders, Positions, Price, Open Interest,
    Positions-Long, Positions-Short, Positions-Net, week, year.
    """
    dates = bq.func.range(start, end, frq='d')
    kwargs = {
        'report_type': report_type.replace(' ', '_'),
        'trader_type': trader_type.replace(' ', '_'),
    }

    data_items = {
        'Traders': bq.data.cot_traders(**kwargs),
        'Positions': bq.data.cot_position(**kwargs),
        'Price': bq.data.px_last(fill='prev'),
        'Open Interest': bq.data.fut_aggte_open_int(),
    }
    for direction in ['Long', 'Short', 'Net']:
        data_items[f'Positions - {direction}'] = (
            bq.data.cot_position(**kwargs)
            .with_updated_parameters(direction=direction)
        )
    with_params = {'currency': 'usd', 'dates': dates,
                   'crop_year': ['NA', 'combined']}
    df = get_data(futures_ticker, data_items,
                  with_params=with_params,
                  preferences={'unitscheck': 'ignore'})
    # Adiciona semana/ano ISO
    date_vals = pd.to_datetime(df.index.get_level_values('Date'))
    iso = date_vals.isocalendar()
    df['week'] = iso.week.values
    df['year'] = iso.year.values
    # Inverte short
    if 'Positions - Short' in df.columns:
        df['Positions - Short'] = df['Positions - Short'] * -1
    return df.dropna(subset=['Traders', 'Positions'])


def aggregate_cot(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega COT por data (caso de múltiplos contratos)."""
    cols_skip = ['year', 'week', 'Price']
    aggs = {c: 'sum' for c in df.columns if c not in cols_skip}
    out = df.groupby('Date').agg(
        **{**{c: (c, agg) for c, agg in aggs.items()},
           'week': ('week', 'first'),
           'year': ('year', 'first'),
           'Price': ('Price', 'mean')})
    out['Basket Returns'] = (1 + out['Price'].pct_change()).cumprod() - 1
    return out


def cot_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas semanais de sazonalidade COT."""
    cols = ['Traders', 'Positions', 'Open Interest', 'week']
    avail = [c for c in cols if c in df.columns]
    return (df[avail].groupby('week')
            .agg(['mean', 'max', 'sum', 'min'])
            .rename(columns=lambda x: x.title()))


def cot_summary_stats(df: pd.DataFrame) -> pd.Series:
    """Estatísticas resumo: último valor, WoW change, percentil 5Y, mediana, z-score."""
    summary = pd.Series(dtype=float)
    for col in ['Positions', 'Traders']:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 3:
            continue
        stats = pd.Series({
            col: s.iloc[-1],
            f'{col} Change': s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else np.nan,
            f'{col} 5Y Percentile': pd.cut(s, 100, labels=False).iloc[-1],
            f'{col} 5Y Median': s.median(),
            f'{col} 5Y Z-Score': (s.iloc[-1] - s.mean()) / max(s.std(), 1e-9),
        })
        summary = pd.concat([summary, stats])
    return summary


def safe_fetch_cot(ticker: str, **kwargs) -> Optional[pd.DataFrame]:
    """Tenta buscar COT. Retorna None se não disponível."""
    ok, fut = has_cot(ticker)
    if not ok:
        return None
    try:
        df = fetch_cot_data(fut, **kwargs)
        if df.empty:
            return None
        return aggregate_cot(df)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — ESTIMATIVA DE BUYBACK
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=128)
def fetch_buyback_data(ticker: str) -> Dict:
    """
    Busca dados de buyback via BQL. Retorna dict com campos disponíveis.
    Nota: não temos % ADV executado nem saldo restante — apenas anunciado.
    """
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
            df = bq.execute(req)[0].df()
            if df is not None and not df.empty:
                vcol = _last_numeric_col(df)
                if vcol:
                    val = pd.to_numeric(df[vcol], errors='coerce').dropna()
                    if len(val) > 0:
                        result[key] = float(val.iloc[-1])
        except Exception:
            continue
    return result


def estimate_buyback_flow(ticker: str, horizon_days: int = 252) -> Dict:
    """
    Estima fluxo diário de buyback.

    Modelo simplificado:
      - Usa ANNOUNCED_BUYBACK_AMT como total autorizado
      - Assume execução em ~252 dias úteis (1 ano)
      - Assume taxa de execução de ~80% (mercado típico)
      - Sem informação do saldo restante → usa valor total anualizado

    Retorna dict com:
      daily_est: fluxo diário estimado ($)
      pct_adv_est: % do ADV estimado
      confidence: 'low' (sempre — não sabemos o realizado)
      announced: valor anunciado total
    """
    data = fetch_buyback_data(ticker)
    announced = data.get('announced', 0)
    if not announced or announced <= 0:
        return {'daily_est': 0, 'pct_adv_est': 0,
                'confidence': 'none', 'announced': 0}

    execution_rate = 0.80
    daily_est = (announced * execution_rate) / max(horizon_days, 1)

    # % do ADV
    adv = data.get('adv20', 0)
    px = data.get('px', 0)
    adv_usd = adv * px if (adv and px) else 0
    pct_adv = (daily_est / adv_usd * 100) if adv_usd > 0 else np.nan

    return {
        'daily_est': daily_est,
        'pct_adv_est': pct_adv,
        'confidence': 'low',
        'announced': announced,
        'mkt_cap': data.get('mkt_cap', np.nan),
    }


def estimate_index_buyback_flow(index_ticker: str = 'SPX Index',
                                top_n: int = 50) -> pd.DataFrame:
    """
    Estima fluxo de buyback para os maiores membros de um índice.
    Retorna DataFrame com estimativas por nome.
    """
    try:
        uni = bq.univ.members(index_ticker)
        items = {
            'ticker': bq.data.id(),
            'cap': bq.data.cur_mkt_cap(),
        }
        # Somente campos que existem
        try:
            items['buyback'] = bq.data.announced_buyback_amt()
        except Exception:
            pass
        try:
            items['adv'] = bq.data.volume_avg_20d()
        except Exception:
            pass
        try:
            items['px'] = bq.data.px_last()
        except Exception:
            pass

        req = bql.Request(uni, items)
        df = bq.execute(req)[0].df()
        if df is None or df.empty:
            return pd.DataFrame()

        # Normaliza colunas
        rename = {}
        for c in df.columns:
            cl = str(c).lower()
            if 'id' in cl or 'ticker' in cl:
                rename[c] = 'ticker'
            elif 'cap' in cl and 'mkt' in cl:
                rename[c] = 'cap'
            elif 'buyback' in cl:
                rename[c] = 'buyback'
            elif 'volume' in cl or 'adv' in cl:
                rename[c] = 'adv'
            elif 'px_last' in cl:
                rename[c] = 'px'
        df = df.rename(columns=rename)

        if 'cap' in df.columns:
            df['cap'] = pd.to_numeric(df['cap'], errors='coerce')
            df = df.nlargest(top_n, 'cap')

        if 'buyback' not in df.columns:
            df['buyback'] = 0
        df['buyback'] = pd.to_numeric(df['buyback'], errors='coerce').fillna(0)

        # Estimativa diária
        df['daily_est'] = df['buyback'] * 0.80 / TRADING_DAYS
        if 'adv' in df.columns and 'px' in df.columns:
            adv_usd = (pd.to_numeric(df['adv'], errors='coerce')
                       * pd.to_numeric(df['px'], errors='coerce'))
            df['pct_adv_est'] = (df['daily_est'] / adv_usd.replace(0, np.nan)) * 100
        else:
            df['pct_adv_est'] = np.nan

        df['confidence'] = 'low'
        total = df['daily_est'].sum()
        if 'ticker' in df.columns:
            df = df.set_index('ticker')

        return df[['cap', 'buyback', 'daily_est', 'pct_adv_est', 'confidence']]

    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — REBALANCEAMENTO DE ETFs ALAVANCADOS (simplificado)
# ═══════════════════════════════════════════════════════════════════════════════

LEVERAGED_ETFS = [
    {'ticker': 'SPXL US Equity', 'name': 'SPXL', 'leverage': 3, 'under': 'SPX Index'},
    {'ticker': 'UPRO US Equity', 'name': 'UPRO', 'leverage': 3, 'under': 'SPX Index'},
    {'ticker': 'SSO US Equity',  'name': 'SSO',  'leverage': 2, 'under': 'SPX Index'},
    {'ticker': 'SH US Equity',   'name': 'SH',   'leverage': -1, 'under': 'SPX Index'},
    {'ticker': 'SDS US Equity',  'name': 'SDS',  'leverage': -2, 'under': 'SPX Index'},
    {'ticker': 'SPXS US Equity', 'name': 'SPXS', 'leverage': -3, 'under': 'SPX Index'},
    {'ticker': 'SPXU US Equity', 'name': 'SPXU', 'leverage': -3, 'under': 'SPX Index'},
    {'ticker': 'TQQQ US Equity', 'name': 'TQQQ', 'leverage': 3, 'under': 'NDX Index'},
    {'ticker': 'SQQQ US Equity', 'name': 'SQQQ', 'leverage': -3, 'under': 'NDX Index'},
    {'ticker': 'SOXL US Equity', 'name': 'SOXL', 'leverage': 3, 'under': 'SOX Index'},
    {'ticker': 'SOXS US Equity', 'name': 'SOXS', 'leverage': -3, 'under': 'SOX Index'},
]


def compute_leveraged_flows(daily_return: float) -> Tuple[pd.DataFrame, float]:
    """
    Calcula fluxo de rebalanceamento para ETFs alavancados.
    Fórmula: Rebal_$ = AUM × L × (L-1) × r / (1 + L×r)
    """
    rows = []
    for etf in LEVERAGED_ETFS:
        L = etf['leverage']
        r = daily_return
        try:
            req = bql.Request([etf['ticker']], bq.data.fund_total_assets())
            df = bq.execute(req)[0].df()
            aum = float(pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna().iloc[-1])
        except Exception:
            aum = 0
        denom = 1 + L * r
        rebal = aum * L * (L - 1) * r / denom if abs(denom) > 1e-12 else 0
        direction = "BUY" if rebal > 0 else "SELL" if rebal < 0 else "FLAT"
        rows.append({
            'ETF': etf['name'], 'Ticker': etf['ticker'],
            'Under': etf['under'], 'Leverage': L,
            'AUM': aum, 'Rebalance_$': rebal, 'Direção': direction
        })
    df = pd.DataFrame(rows)
    total = df['Rebalance_$'].sum()
    return df, total


def compute_leveraged_flow_simple(daily_return: float,
                                  aum_estimates: Optional[Dict] = None) -> float:
    """Versão rápida: calcula fluxo total de rebalanceamento sem BQL call."""
    if aum_estimates is None:
        # Estimativas padrão (bilhões USD)
        aum_estimates = {
            'SPXL': 5e9, 'UPRO': 3e9, 'SSO': 4e9,
            'SH': 2e9, 'SDS': 0.8e9, 'SPXS': 1e9, 'SPXU': 0.8e9,
            'TQQQ': 20e9, 'SQQQ': 5e9, 'SOXL': 10e9, 'SOXS': 3e9,
        }
    total = 0
    for etf in LEVERAGED_ETFS:
        L = etf['leverage']
        r = daily_return
        aum = aum_estimates.get(etf['name'], 1e9)
        denom = 1 + L * r
        if abs(denom) > 1e-12:
            total += aum * L * (L - 1) * r / denom
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — AGREGAÇÃO E ESTATÍSTICA PREDITIVA
# ═══════════════════════════════════════════════════════════════════════════════

def flow_zscore(current: float, history: pd.Series) -> float:
    """Z-score do valor atual relativo ao histórico."""
    if history is None or len(history) < 5:
        return np.nan
    mu = history.mean()
    sigma = history.std()
    if sigma < 1e-12:
        return 0.0
    return (current - mu) / sigma


def flow_percentile(current: float, history: pd.Series) -> float:
    """Percentil (0-100) do valor atual relativo ao histórico."""
    if history is None or len(history) < 5:
        return np.nan
    return float((history <= current).mean() * 100.0)


def compute_flow_score(leveraged_flow: float,
                       buyback_daily: float = 0,
                       cot_net_change: float = 0,
                       spx_rebal_prob: float = 0.5,
                       history_leveraged: Optional[pd.Series] = None,
                       history_cot: Optional[pd.Series] = None) -> Dict:
    """
    Computa score combinado de fluxo contratado.

    Fontes e pesos:
      - ETFs alavancados:   40% (mecânico, alta confiança)
      - Buyback:            20% (anunciado, confiança baixa)
      - COT net change:     20% (semanal, só quando disponível)
      - SPX rebalance prob: 20% (probabilístico)

    Retorna dict com scores individuais e combinado.
    """
    # Z-scores individuais
    z_lev = flow_zscore(leveraged_flow, history_leveraged) if history_leveraged is not None else 0
    z_cot = flow_zscore(cot_net_change, history_cot) if history_cot is not None else 0

    # Normaliza buyback como sinal direcional (sempre positivo = compra)
    z_buyback = np.clip(buyback_daily / 1e8, -3, 3) if buyback_daily else 0

    # SPX rebal: 0.5 = neutro, >0.5 = mais inclusões (bullish), <0.5 = exclusões
    z_spx = (spx_rebal_prob - 0.5) * 4  # escala para ±2

    # Pesos
    w_lev, w_buy, w_cot, w_spx = 0.40, 0.20, 0.20, 0.20
    if history_cot is None or len(history_cot) < 5:
        # Sem COT → redistribui peso
        w_lev, w_buy, w_spx = 0.50, 0.25, 0.25
        w_cot = 0.0

    combined = (w_lev * z_lev + w_buy * z_buyback
                + w_cot * z_cot + w_spx * z_spx)

    # Direção
    if combined > 0.5:
        direction = "BULLISH"
    elif combined < -0.5:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    # Probabilidade direcional (sigmoid)
    prob_up = 1.0 / (1.0 + math.exp(-combined))

    return {
        'z_leveraged': z_lev,
        'z_buyback': z_buyback,
        'z_cot': z_cot,
        'z_spx_rebal': z_spx,
        'combined_score': combined,
        'direction': direction,
        'prob_up': prob_up,
        'prob_down': 1.0 - prob_up,
        'weights': {'leveraged': w_lev, 'buyback': w_buy,
                    'cot': w_cot, 'spx_rebal': w_spx},
    }


def build_flow_history(ticker: str = 'SPX Index',
                       lookback: int = 252) -> pd.DataFrame:
    """
    Constrói série histórica de fluxo de rebalanceamento de ETFs alavancados
    baseado nos retornos diários do subjacente.
    """
    px = px_last_series(ticker, start=f"-{lookback + 10}D", end="0D")
    if px is None or len(px) < 10:
        return pd.DataFrame()
    rets = px.pct_change().dropna()
    flows = rets.apply(compute_leveraged_flow_simple)
    flows.name = 'LevETF_Flow'
    df = pd.DataFrame({'Return': rets, 'LevETF_Flow': flows})
    return df


def rolling_flow_vs_price_corr(flow_series: pd.Series,
                                price_series: pd.Series,
                                window: int = 26) -> pd.Series:
    """Correlação rolling entre mudança de fluxo e retorno do preço."""
    if flow_series is None or price_series is None:
        return pd.Series(dtype=float)
    idx = flow_series.index.intersection(price_series.index)
    if len(idx) < window + 5:
        return pd.Series(dtype=float)
    combined = pd.DataFrame({
        'flow_chg': flow_series.reindex(idx).pct_change(),
        'price_ret': price_series.reindex(idx).pct_change()
    }).dropna()
    return combined['flow_chg'].rolling(window).corr(combined['price_ret'])


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — VISUALIZAÇÃO: PLOTLY
# ═══════════════════════════════════════════════════════════════════════════════

def _add_border(fig_widget):
    """Container com borda para gráficos Plotly."""
    return wd.VBox([fig_widget], layout={
        'border': '1px solid white', 'margin': '10px',
        'padding': '10px', 'width': '98%'})


def plot_positions_basket(df: pd.DataFrame, basket_col='Price',
                          data_col='Positions') -> wd.VBox:
    """Dual-axis: positions (esquerda) + preço (direita)."""
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(
        x=df.index, y=df[data_col], name=data_col,
        yaxis='y1', line=dict(width=1), marker_color='orange'))
    if basket_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[basket_col], name=basket_col,
            yaxis='y2', line=dict(width=1), marker_color='white'))
    fig.update_yaxes(zeroline=False, showgrid=False)
    fig.layout.yaxis.update(title_text=data_col)
    fig.layout.yaxis2.update(title_text=basket_col)
    fig.update_layout(title='Positions & Basket Price', **FIG_LAYOUT)
    return _add_border(go.FigureWidget(fig))


def plot_long_short_net(df: pd.DataFrame) -> wd.VBox:
    """Barras Long/Short + linha Net."""
    fig = go.FigureWidget()
    for name, color in [('Long', 'grey'), ('Short', 'white')]:
        col = f'Positions - {name}'
        if col in df.columns:
            fig.add_trace(go.Bar(x=df.index, y=df[col],
                                 name=name, marker_color=color))
    if 'Positions - Net' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Positions - Net'],
                                 name='Net', marker_color='orange'))
    fig.update_layout(barmode='relative',
                      title='Long, Short & Net Positions',
                      yaxis_title='Positions', hovermode='x unified',
                      **FIG_LAYOUT)
    return _add_border(fig)


def plot_dispersion(seasonality: pd.DataFrame, df: pd.DataFrame,
                    col: str = 'Positions') -> wd.VBox:
    """Dispersion chart: min/max band + mean + current year."""
    seas = seasonality[col].copy() if col in seasonality.columns else pd.DataFrame()
    if seas.empty:
        return wd.VBox([wd.HTML("<p>Sem dados de sazonalidade.</p>")])
    current_year = pd.Timestamp.now().year
    yr_data = df[df.index.year == current_year] if hasattr(df.index, 'year') else pd.DataFrame()
    if not yr_data.empty and 'week' in yr_data.columns and col in yr_data.columns:
        seas['Current'] = yr_data.set_index('week')[col]

    fig = go.FigureWidget()
    for name, fill in [('Max', None), ('Min', 'tonexty')]:
        if name in seas.columns:
            fig.add_trace(go.Scatter(
                x=seas.index, y=seas[name], mode='lines', name=name,
                fill=fill, fillcolor='gray',
                line=dict(color='gray', width=0), showlegend=False))
    for name, color in [('Mean', 'orange'), ('Current', 'white')]:
        if name in seas.columns:
            fig.add_trace(go.Scatter(
                x=seas.index, y=seas[name], mode='lines+markers',
                name=name, line=dict(color=color)))
    fig.update_layout(title='5Y Dispersion', xaxis_title='Weeks',
                      yaxis_title=col, hovermode='x unified',
                      xaxis=dict(range=[1, 53]), **FIG_LAYOUT)
    return _add_border(fig)


def plot_multi_year(df: pd.DataFrame) -> wd.VBox:
    """Gráfico sazonal multi-ano de Positions por semana ISO."""
    if 'week' not in df.columns or 'year' not in df.columns:
        return wd.VBox([wd.HTML("<p>Sem dados semanais.</p>")])
    pivot = df.pivot(columns='year', index='week', values='Positions')
    pivot = pivot.iloc[:, -6:]
    fig = go.FigureWidget()
    colors = ['#ffffff', 'lightgrey', 'darkgrey', '#ffd793', 'orange', '#ec8100']
    for col_name, color in zip(pivot.columns, colors):
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[col_name], mode='lines',
            line=dict(color=color), name=str(col_name)))
    fig.update_layout(hovermode='x unified', yaxis_title='Positions',
                      title='Seasonal Analysis', **FIG_LAYOUT)
    return _add_border(fig)


def plot_correlation(df: pd.DataFrame, window: int = 26) -> wd.VBox:
    """Rolling correlation entre preço e net positions."""
    if 'Price' not in df.columns or 'Positions - Net' not in df.columns:
        return wd.VBox([wd.HTML("<p>Sem dados para correlação.</p>")])
    corr = (df[['Price', 'Positions - Net']].pct_change()
            .rolling(window).corr().unstack()[('Price', 'Positions - Net')])
    fig = go.FigureWidget(go.Scatter(
        x=corr.index, y=corr.values, name='Corr',
        line=dict(width=1), marker_color='orange'))
    fig.update_yaxes(zeroline=False, showgrid=False)
    fig.layout.yaxis.update(title_text='Correlation')
    fig.update_layout(title='Rolling Correlation: Price Δ vs Net Length Δ',
                      **FIG_LAYOUT)
    return _add_border(fig)


def plot_long_short_ratio(df: pd.DataFrame) -> wd.VBox:
    """Long/Short ratio ao longo do tempo."""
    if 'Positions - Long' not in df.columns or 'Positions - Short' not in df.columns:
        return wd.VBox([wd.HTML("<p>Sem dados L/S.</p>")])
    ratio = df['Positions - Long'] / df['Positions - Short'] * -1
    fig = go.FigureWidget(go.Scatter(
        x=ratio.index, y=ratio.values, name='L/S Ratio',
        line=dict(width=1, color='#ff991c')))
    fig.update_yaxes(zeroline=False, showgrid=False)
    fig.layout.yaxis.update(title_text='Ratio')
    fig.update_layout(title='Long/Short Ratio', **FIG_LAYOUT)
    return _add_border(fig)


def plot_flow_score_gauge(score: Dict) -> go.FigureWidget:
    """Gauge chart mostrando o score combinado de fluxo."""
    combined = score.get('combined_score', 0)
    prob = score.get('prob_up', 0.5)
    direction = score.get('direction', 'NEUTRAL')
    colors = {'BULLISH': 'green', 'BEARISH': 'red', 'NEUTRAL': 'gray'}
    fig = go.FigureWidget(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        title={'text': f"Flow Score: {direction}"},
        delta={'reference': 50, 'increasing': {'color': 'green'},
               'decreasing': {'color': 'red'}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': colors.get(direction, 'gray')},
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}],
            'threshold': {'line': {'color': 'black', 'width': 3},
                          'thickness': 0.8, 'value': prob * 100}
        }))
    fig.update_layout(height=280, margin=dict(t=50, b=20, l=20, r=20),
                      **{k: v for k, v in FIG_LAYOUT.items()
                         if k not in ('margin',)})
    return fig


def plot_flow_components_bar(score: Dict) -> go.FigureWidget:
    """Gráfico de barras dos componentes do flow score."""
    components = {
        'ETFs Alav.': score.get('z_leveraged', 0),
        'Buyback': score.get('z_buyback', 0),
        'COT': score.get('z_cot', 0),
        'SPX Rebal': score.get('z_spx_rebal', 0),
    }
    weights = score.get('weights', {})
    names = list(components.keys())
    values = list(components.values())
    w_vals = [weights.get(k.lower().replace(' ', '_').replace('.', ''), 0.25)
              for k in ['leveraged', 'buyback', 'cot', 'spx_rebal']]
    colors = ['#3498DB' if v >= 0 else '#E74C3C' for v in values]

    fig = go.FigureWidget()
    fig.add_trace(go.Bar(x=names, y=values, marker_color=colors,
                         name='Z-Score', text=[f'{v:+.2f}' for v in values],
                         textposition='outside'))
    fig.add_trace(go.Scatter(x=names, y=w_vals, name='Peso',
                             yaxis='y2', mode='markers+text',
                             text=[f'{w:.0%}' for w in w_vals],
                             textposition='top center',
                             marker=dict(size=10, color='white')))
    fig.update_layout(
        title='Componentes do Flow Score',
        yaxis_title='Z-Score', yaxis2=dict(overlaying='y', side='right',
                                           title='Peso', range=[0, 1]),
        **FIG_LAYOUT)
    return fig


def plot_flow_history_line(flow_hist: pd.DataFrame) -> go.FigureWidget:
    """Série histórica de fluxo de rebalanceamento vs retorno."""
    if flow_hist.empty:
        fig = go.FigureWidget()
        fig.update_layout(title="Sem histórico de fluxo")
        return fig
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=flow_hist.index, y=flow_hist['LevETF_Flow'],
                             name='Lev ETF Flow', line=dict(color='orange', width=1)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=flow_hist.index, y=flow_hist['Return'],
                             name='Return', line=dict(color='white', width=1)),
                  secondary_y=True)
    fig.update_layout(title='Fluxo ETFs Alavancados vs Retorno',
                      hovermode='x unified', **FIG_LAYOUT)
    fig.update_yaxes(title_text='Flow ($)', secondary_y=False)
    fig.update_yaxes(title_text='Return', secondary_y=True)
    return go.FigureWidget(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — VISUALIZAÇÃO: BQPLOT (nativo BQuant)
# ═══════════════════════════════════════════════════════════════════════════════

def bqp_flow_bar_line(flow_df: pd.DataFrame, bar_col='LevETF_Flow',
                      line_col='Return') -> object:
    """Bar + line overlay usando bqplot (BQuant nativo). Retorna Figure."""
    if not HAS_BQPLOT or flow_df.empty:
        return wd.HTML("<p>bqplot não disponível ou sem dados.</p>")
    scale_x = bqp.DateScale()
    scale_y = bqp.LinearScale()
    mark_bar = bqp.Bars(
        x=flow_df.index, y=flow_df[bar_col],
        scales={'x': scale_x, 'y': scale_y}, colors=['#1B84ED'],
        tooltip=bqp.Tooltip(fields=['y', 'x'], show_labels=False,
                            formats=['.0f', '%Y/%m/%d']))
    marks = [mark_bar]
    if line_col in flow_df.columns:
        scale_y2 = bqp.LinearScale()
        mark_line = bqp.Lines(
            x=flow_df.index, y=flow_df[line_col], stroke_width=2,
            scales={'x': scale_x, 'y': scale_y2}, colors=['#CF7DFF'])
        marks.append(mark_line)
    ax_y = bqp.Axis(scale=scale_y, orientation='vertical', label='Flow ($)')
    ax_x = bqp.Axis(scale=scale_x)
    fig = bqp.Figure(
        marks=marks, axes=[ax_x, ax_y],
        title='Fluxo de ETFs Alavancados',
        title_style={'font-size': '18px'}, padding_y=0,
        fig_margin={'top': 50, 'bottom': 50, 'left': 60, 'right': 50},
        layout={'width': 'auto', 'height': '400px'})
    return fig


def bqp_scatter_flow_return(flow_df: pd.DataFrame) -> object:
    """Scatter plot: flow vs return (bqplot)."""
    if not HAS_BQPLOT or flow_df.empty:
        return wd.HTML("<p>bqplot não disponível ou sem dados.</p>")
    scale_x = bqp.LinearScale()
    scale_y = bqp.LinearScale()
    tooltip = bqp.Tooltip(fields=['x', 'y'],
                          labels=['Flow', 'Return'],
                          formats=['.0f', '.4f'])
    mark = bqp.Scatter(
        x=flow_df['LevETF_Flow'], y=flow_df['Return'],
        tooltip=tooltip, scales={'x': scale_x, 'y': scale_y},
        default_size=32, colors=['#FF5A00'])
    ax_x = bqp.Axis(scale=scale_x, label='Flow ($)')
    ax_y = bqp.Axis(scale=scale_y, orientation='vertical', label='Return')
    fig = bqp.Figure(
        marks=[mark], axes=[ax_x, ax_y],
        title='Flow vs Return', title_style={'font-size': '18px'},
        padding_x=0.05, padding_y=0.05,
        layout={'width': '100%', 'height': '400px'})
    return fig


def bqp_box_plot_flows(monthly_flows: pd.DataFrame) -> object:
    """Box plot de fluxos mensais (bqplot)."""
    if not HAS_BQPLOT or monthly_flows.empty:
        return wd.HTML("<p>bqplot não disponível ou sem dados.</p>")
    # Pad columns to equal length so numpy gets a homogeneous 2D array
    max_len = max(monthly_flows[c].dropna().shape[0] for c in monthly_flows.columns)
    y_data = np.full((len(monthly_flows.columns), max_len), np.nan)
    for i, c in enumerate(monthly_flows.columns):
        vals = monthly_flows[c].dropna().values
        y_data[i, :len(vals)] = vals
    scale_x = bqp.OrdinalScale()
    scale_y = bqp.LinearScale()
    mark = bqp.Boxplot(
        x=list(monthly_flows.columns),
        y=y_data,
        colors=['#1B84ED'], scales={'x': scale_x, 'y': scale_y},
        stroke='white', outlier_fill_color='red')
    ax_x = bqp.Axis(scale=scale_x)
    ax_y = bqp.Axis(scale=scale_y, label='Flow ($)', orientation='vertical',
                     tick_format='0.2s')
    fig = bqp.Figure(
        marks=[mark], axes=[ax_x, ax_y],
        title='Distribuição Mensal de Fluxos',
        title_style={'font-size': '18px'},
        layout={'width': '100%', 'height': '400px'},
        fig_margin={'top': 50, 'bottom': 40, 'left': 80, 'right': 50})
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — IPYDATAGRID TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def grid_flow_score(score: Dict) -> object:
    """Tabela interativa do flow score com barras condicionais."""
    if not HAS_DATAGRID:
        # Fallback HTML
        rows = [f"<tr><td>{k}</td><td>{v:.3f}</td></tr>"
                for k, v in score.items() if isinstance(v, (int, float))]
        html = f"<table>{''.join(rows)}</table>"
        return wd.HTML(html)

    data = {
        'Componente': ['ETFs Alavancados', 'Buyback', 'COT', 'SPX Rebal',
                        'Score Combinado'],
        'Z-Score': [score.get('z_leveraged', 0), score.get('z_buyback', 0),
                    score.get('z_cot', 0), score.get('z_spx_rebal', 0),
                    score.get('combined_score', 0)],
        'Prob Up (%)': [np.nan, np.nan, np.nan, np.nan,
                        score.get('prob_up', 0.5) * 100],
    }
    df = pd.DataFrame(data).set_index('Componente')

    try:
        import bqplot as _bqp
        linear_scale = _bqp.LinearScale(min=-3, max=3)
        color_scale = _bqp.ColorScale(min=-3, max=3,
                                      colors=['#E74C3C', '#FFFFFF', '#2ECC71'])
        renderers = {
            'Z-Score': BarRenderer(bar_value=linear_scale, format='.2f',
                                   bar_color=color_scale,
                                   horizontal_alignment='center'),
        }
    except Exception:
        renderers = {}

    grid = DataGrid(df, renderers=renderers, base_column_size=150,
                    layout={'height': '200px'})
    return grid


def grid_buyback_table(buyback_df: pd.DataFrame) -> object:
    """Tabela de buyback por empresa usando ipydatagrid."""
    if buyback_df.empty:
        return wd.HTML("<p>Sem dados de buyback.</p>")
    if not HAS_DATAGRID:
        return wd.HTML(buyback_df.head(20).to_html())
    try:
        import bqplot as _bqp
        scale = _bqp.LinearScale(min=0, max=buyback_df['daily_est'].max())
        renderers = {
            'daily_est': BarRenderer(bar_value=scale, format='$,.0f',
                                     bar_color='#2ECC71',
                                     horizontal_alignment='right'),
        }
    except Exception:
        renderers = {}
    grid = DataGrid(buyback_df.head(30), renderers=renderers,
                    base_column_size=120, base_row_header_size=140,
                    layout={'height': '400px'})
    return grid


def grid_cot_stats(stats: pd.Series) -> object:
    """Tabela de estatísticas COT usando ipydatagrid."""
    if stats.empty:
        return wd.HTML("<p>Sem dados COT.</p>")
    df = stats.to_frame('Value')
    if not HAS_DATAGRID:
        return wd.HTML(df.to_html())
    # Formatação condicional: z-score coloring
    try:
        from ipydatagrid import VegaExpr
        renderers = {
            'Value': TextRenderer(
                background_color=VegaExpr(
                    "cell.value < -1 ? '#EE0D2D' : "
                    "cell.value > 1 ? '#20B020' : 'transparent'"),
                format='.2f')
        }
    except Exception:
        renderers = {}
    grid = DataGrid(df, renderers=renderers, base_column_size=150,
                    base_row_header_size=200,
                    layout={'height': '300px'})
    return grid


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — DASHBOARD INTERATIVO
# ═══════════════════════════════════════════════════════════════════════════════

# ---------- Widgets ----------
ticker_w = wd.Text(value='SPX Index', description='Ativo:',
                   layout={'width': '250px'})
lookback_w = wd.IntSlider(value=252, min=60, max=1000, step=10,
                          description='Lookback (dias):',
                          layout={'width': '400px'})

# COT
cot_type_w = wd.Dropdown(
    options=list(COT_CONTRACTS.keys()),
    value='Equity Indices', description='COT Categoria:',
    layout={'width': '300px'})
cot_contract_w = wd.SelectMultiple(
    description='Contratos:', layout={'min_width': '350px', 'height': '100px'})
cot_trader_w = wd.Dropdown(
    options=['Managed Money', 'Total', 'Commercial', 'Non-Commercial',
             'Swap Dealers', 'Leveraged Funds', 'Asset Manager'],
    value='Managed Money', description='Trader Type:',
    layout={'width': '300px'})
cot_report_w = wd.Dropdown(
    options=['CFTC Disaggregated', 'CFTC TFF', 'CFTC Legacy'],
    value='CFTC Disaggregated', description='Report Type:',
    layout={'width': '300px'})

run_btn = wd.Button(description='Calcular Flow Score',
                    button_style='success', icon='cogs')
spinner_w = wd.HTML(
    '<i class="fa fa-spinner fa-spin" style="font-size:18px"></i>',
    layout={'visibility': 'hidden', 'margin': '12px 0 0 10px'})
out_main = wd.Output(layout={'border': '1px solid #777'})


def _update_cot_contracts(change=None):
    opts = COT_CONTRACTS.get(cot_type_w.value, [])
    cot_contract_w.options = opts
    if opts:
        cot_contract_w.value = (opts[0][1],)

cot_type_w.observe(_update_cot_contracts, names='value')
_update_cot_contracts()


def run_analysis(_=None):
    """Callback principal: calcula tudo e monta output."""
    spinner_w.layout.visibility = 'visible'
    out_main.clear_output()

    with out_main:
        try:
            ticker = ticker_w.value.strip() or 'SPX Index'
            lb = lookback_w.value
            loading = wd.HTML("<h4>1/5: Construindo histórico de fluxo...</h4>")
            display(loading)

            # 1. Histórico de fluxo ETFs alavancados
            flow_hist = build_flow_history(ticker, lookback=lb)
            today_flow = (compute_leveraged_flow_simple(
                float(flow_hist['Return'].iloc[-1]))
                if not flow_hist.empty else 0)

            # 2. Buyback
            loading.value = "<h4>2/5: Estimando buyback...</h4>"
            buyback = estimate_buyback_flow(ticker)
            buyback_daily = buyback.get('daily_est', 0)

            # 3. COT (se disponível)
            loading.value = "<h4>3/5: Buscando COT (se disponível)...</h4>"
            cot_ok, cot_fut = has_cot(ticker)
            cot_df = None
            cot_stats = pd.Series(dtype=float)
            cot_net_change = 0
            history_cot = None

            if cot_ok:
                cot_df = safe_fetch_cot(ticker,
                                        trader_type=cot_trader_w.value,
                                        report_type=cot_report_w.value)
                if cot_df is not None and not cot_df.empty:
                    cot_stats = cot_summary_stats(cot_df)
                    if 'Positions - Net' in cot_df.columns:
                        net = cot_df['Positions - Net'].dropna()
                        if len(net) >= 2:
                            cot_net_change = float(net.iloc[-1] - net.iloc[-2])
                            history_cot = net.diff().dropna()

            # 4. COT de contratos selecionados (painel extra)
            loading.value = "<h4>4/5: COT contratos selecionados...</h4>"
            selected_cots = list(cot_contract_w.value)
            selected_cot_df = None
            if selected_cots:
                try:
                    sel_df = fetch_cot_data(
                        selected_cots[0] if len(selected_cots) == 1
                        else bq.univ.list(selected_cots),
                        trader_type=cot_trader_w.value,
                        report_type=cot_report_w.value)
                    if not sel_df.empty:
                        selected_cot_df = aggregate_cot(sel_df)
                except Exception:
                    selected_cot_df = None

            # 5. Score combinado
            loading.value = "<h4>5/5: Calculando flow score...</h4>"
            lev_history = (flow_hist['LevETF_Flow']
                           if not flow_hist.empty else None)
            score = compute_flow_score(
                leveraged_flow=today_flow,
                buyback_daily=buyback_daily,
                cot_net_change=cot_net_change,
                spx_rebal_prob=0.5,  # neutro se não calculado
                history_leveraged=lev_history,
                history_cot=history_cot)

            # ═════ MONTAGEM DO DASHBOARD ═════
            loading.value = ""

            # -- Score Summary --
            title_html = wd.HTML(
                f"<h2>Flow Predictor — {ticker}</h2>"
                f"<p>Lookback: {lb}d | "
                f"Direction: <b style='color:"
                f"{'green' if score['direction'] == 'BULLISH' else 'red' if score['direction'] == 'BEARISH' else 'gray'}'>"
                f"{score['direction']}</b> | "
                f"P(Up): {score['prob_up']:.1%} | "
                f"Score: {score['combined_score']:+.2f}</p>")

            # Tab 1: Score
            tab1 = wd.VBox([
                title_html,
                wd.HBox([
                    plot_flow_score_gauge(score),
                    plot_flow_components_bar(score)
                ]),
                grid_flow_score(score)
            ])

            # Tab 2: Histórico de Fluxo
            tab2_children = [wd.HTML("<h3>Histórico de Fluxo — ETFs Alavancados</h3>")]
            if not flow_hist.empty:
                tab2_children.append(plot_flow_history_line(flow_hist))
                tab2_children.append(bqp_flow_bar_line(flow_hist.tail(60)))
                tab2_children.append(bqp_scatter_flow_return(flow_hist))
                # Box plot mensal
                monthly = flow_hist['LevETF_Flow'].copy()
                monthly.index = pd.to_datetime(monthly.index)
                monthly_pivot = monthly.groupby(monthly.index.month).apply(
                    lambda g: g.values).to_frame('vals')
                if not monthly_pivot.empty:
                    mp = pd.DataFrame(
                        {str(m): pd.Series(v)
                         for m, v in zip(monthly_pivot.index,
                                         monthly_pivot['vals'])})
                    tab2_children.append(bqp_box_plot_flows(mp))
            tab2 = wd.VBox(tab2_children)

            # Tab 3: Buyback
            tab3_children = [
                wd.HTML("<h3>Estimativa de Buyback</h3>"
                        "<p><i>⚠️ Confiança baixa: não temos % ADV executado"
                        " nem saldo restante.</i></p>")]
            bb_html = (
                f"<table style='font-size:14px'>"
                f"<tr><td>Anunciado:</td><td>${buyback.get('announced', 0):,.0f}</td></tr>"
                f"<tr><td>Estimativa diária:</td><td>${buyback_daily:,.0f}</td></tr>"
                f"<tr><td>% ADV estimado:</td><td>"
                f"{buyback.get('pct_adv_est', 0):.2f}%</td></tr>"
                f"<tr><td>Confiança:</td><td>{buyback.get('confidence', 'N/A')}</td></tr>"
                f"</table>")
            tab3_children.append(wd.HTML(bb_html))
            # Tenta tabela de top buybacks do índice
            try:
                bb_df = estimate_index_buyback_flow(ticker, top_n=30)
                if not bb_df.empty:
                    tab3_children.append(
                        wd.HTML("<h4>Top Buybacks do Índice</h4>"))
                    tab3_children.append(grid_buyback_table(bb_df))
            except Exception:
                pass
            tab3 = wd.VBox(tab3_children)

            # Tab 4: COT
            tab4_children = [wd.HTML("<h3>COT — Commitment of Traders</h3>")]
            if cot_ok and cot_df is not None and not cot_df.empty:
                tab4_children.append(
                    wd.HTML(f"<p>Futures: <b>{cot_fut}</b> | "
                            f"Trader: {cot_trader_w.value}</p>"))
                tab4_children.append(grid_cot_stats(cot_stats))
                seas = cot_seasonality(cot_df)
                tab4_children.append(wd.HBox([
                    plot_positions_basket(cot_df),
                    plot_dispersion(seas, cot_df)
                ]))
                tab4_children.append(plot_long_short_net(cot_df))
                tab4_children.append(wd.HBox([
                    plot_long_short_ratio(cot_df),
                    plot_correlation(cot_df)
                ]))
                tab4_children.append(plot_multi_year(cot_df))
            elif cot_ok:
                tab4_children.append(
                    wd.HTML(f"<p>COT disponível para {cot_fut}, "
                            "mas sem dados retornados.</p>"))
            else:
                tab4_children.append(
                    wd.HTML(f"<p>{ticker} não possui dados COT vinculados. "
                            "Use o painel abaixo para análise direta.</p>"))

            # COT dos contratos selecionados
            if selected_cot_df is not None and not selected_cot_df.empty:
                sel_label = ', '.join(selected_cots)
                tab4_children.append(wd.HTML(f"<hr><h4>COT: {sel_label}</h4>"))
                sel_stats = cot_summary_stats(selected_cot_df)
                tab4_children.append(grid_cot_stats(sel_stats))
                sel_seas = cot_seasonality(selected_cot_df)
                tab4_children.append(wd.HBox([
                    plot_positions_basket(selected_cot_df),
                    plot_long_short_net(selected_cot_df)
                ]))
                tab4_children.append(wd.HBox([
                    plot_dispersion(sel_seas, selected_cot_df),
                    plot_multi_year(selected_cot_df)
                ]))
            tab4 = wd.VBox(tab4_children)

            # Tab 5: Correlação e Backtesting
            tab5_children = [
                wd.HTML("<h3>Análise de Correlação e Backtest</h3>")]
            if not flow_hist.empty:
                px = px_last_series(ticker, start=f"-{lb + 10}D", end="0D")
                corr_s = rolling_flow_vs_price_corr(
                    flow_hist['LevETF_Flow'], px, window=26)
                if len(corr_s) > 0:
                    fig_corr = go.FigureWidget(go.Scatter(
                        x=corr_s.index, y=corr_s.values,
                        name='Flow-Price Corr', line=dict(color='orange')))
                    fig_corr.update_layout(
                        title='Rolling Corr: Lev ETF Flow vs Price',
                        yaxis_title='Correlation',
                        hovermode='x unified', **FIG_LAYOUT)
                    tab5_children.append(_add_border(fig_corr))

                # Hit rate simples
                df_test = flow_hist.copy()
                df_test['flow_signal'] = np.sign(df_test['LevETF_Flow'])
                df_test['next_ret'] = df_test['Return'].shift(-1)
                df_test['hit'] = (np.sign(df_test['flow_signal'])
                                  == np.sign(df_test['next_ret']))
                hit_rate = df_test['hit'].mean()
                tab5_children.append(wd.HTML(
                    f"<p><b>Hit Rate (flow signal vs next-day return):"
                    f"</b> {hit_rate:.1%} ({len(df_test)} obs)</p>"
                    f"<p><i>Nota: fluxo de rebalanceamento é contra-tendência"
                    f" (mean-reverting). Hit rate < 50% é esperado.</i></p>"))
            tab5 = wd.VBox(tab5_children)

            # Montagem final
            dashboard = wd.Tab()
            dashboard.children = [tab1, tab2, tab3, tab4, tab5]
            for i, name in enumerate(['Flow Score', 'Histórico',
                                      'Buyback', 'COT', 'Correlação']):
                dashboard.set_title(i, name)
            display(dashboard)

        except Exception as e:
            import traceback
            print(f"ERRO: {e}")
            traceback.print_exc()
        finally:
            spinner_w.layout.visibility = 'hidden'


run_btn.on_click(run_analysis)

# ---------- Layout ----------
display(wd.VBox([
    wd.HBox([ticker_w, lookback_w]),
    wd.HBox([cot_type_w, cot_contract_w]),
    wd.HBox([cot_trader_w, cot_report_w]),
    wd.HBox([run_btn, spinner_w]),
    out_main
]))
