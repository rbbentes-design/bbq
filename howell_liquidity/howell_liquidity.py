"""
howell_liquidity.py — Two-Agent Liquidity Framework (v4)
================================================================
Single-file BQuant module. Segue o padrao de session_stats.py:
- BQL + ipywidgets + plotly
- Degradacao graciosa por ticker
- Output + ZIP export com state.json + memo.md + figs PNG
- BBG cap 20y automatico

Uso no BQuant:
    %run /path/to/howell_liquidity/howell_liquidity.py

Arquitetura:
  Agent A (Harvester) — puxa dados, constroi indicadores, renderiza 15
    charts, roda classifier + sine-fit, escreve state.json
  Agent B (Analyst)   — le state.json, aplica framework, responde 10
    questoes diagnosticas, gera memo.md

Framework references sao citadas nos comentarios como (§3.x).
"""

import sys
import os
import io
import json
import logging
import traceback
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ============================================================================
# 1. BQuant / BQL detection
# ============================================================================
log = logging.getLogger('howell')
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(message)s',
                                       datefmt='%H:%M:%S'))
    log.addHandler(h)

try:
    import bql
    bq = bql.Service()
    HAS_BQL = True
except Exception as e:
    log.warning(f'bql nao disponivel: {e} — rodara em modo mock')
    bq = None
    HAS_BQL = False

try:
    import ipywidgets as wd
    from IPython.display import display, clear_output, HTML as IPyHTML
    HAS_WIDGETS = True
except Exception:
    HAS_WIDGETS = False
    wd = None

import base64

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    from scipy.optimize import curve_fit
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# FRED fallback — usado quando BQL falhar (muitos series faltam entitlement)
try:
    from fredapi import Fred
    _fred_key = os.environ.get('FRED_API_KEY')
    fred = Fred(api_key=_fred_key) if _fred_key else None
    HAS_FRED = fred is not None
    if HAS_FRED:
        log.info(f'[fred] FRED API habilitada (key ****{_fred_key[-4:]})')
except Exception as e:
    log.warning(f'[fred] nao disponivel: {e}')
    fred = None
    HAS_FRED = False


# ============================================================================
# 2. Constants + Config
# ============================================================================

VERSION = 'v4.0'
AS_OF = datetime.now().strftime('%Y-%m-%d')

# Palette (GLI-style per spec §2)
PALETTE = {
    'orange':   '#E8742C',
    'black':    '#0B0E14',
    'red':      '#D64545',
    'beige':    '#C9A66B',
    'grey':     '#6C7280',
    'lightgrey':'#D7D9DB',
    'text':     '#cce8ff',
    'muted':    '#8b949e',
    'green':    '#7ae582',
    'blue':     '#00d4ff',
    'yellow':   '#ffb84d',
    'purple':   '#c77dff',
    'bg':       '#0B0E14',
    'card':     '#11151f',
}

DASH_TEMPLATE = {
    'layout': {
        'plot_bgcolor':  PALETTE['bg'],
        'paper_bgcolor': PALETTE['bg'],
        'font': {'color': PALETTE['text'], 'family': 'Arial, Helvetica, sans-serif',
                 'size': 12},
        'xaxis': {'gridcolor': '#1e2330', 'zerolinecolor': '#1e2330'},
        'yaxis': {'gridcolor': '#1e2330', 'zerolinecolor': '#1e2330'},
        'colorway': [PALETTE['orange'], PALETTE['blue'], PALETTE['yellow'],
                      PALETTE['green'], PALETTE['red'], PALETTE['purple'],
                      PALETTE['beige']],
        'margin': {'l': 50, 'r': 20, 't': 50, 'b': 40},
        'legend': {'orientation': 'h', 'y': -0.15, 'bgcolor': 'rgba(0,0,0,0)'},
    }
}

DASH_CSS = """
<style>
.how-root { font-family: Arial, Helvetica, sans-serif; color: #cce8ff; }
.how-card { background: #11151f; border: 1px solid #1e2330; border-radius: 6px;
             padding: 12px 16px; margin: 8px 0; }
.how-divider { background: linear-gradient(90deg, rgba(232,116,44,0.15), transparent);
                padding: 14px 20px; margin: 16px 0 10px 0; border-left: 3px solid #E8742C; }
.how-divider-title { color: #E8742C; font-size: 18px; font-weight: 700;
                       letter-spacing: 0.5px; }
.how-divider-sub { color: #8b949e; font-size: 11px; margin-top: 4px; }
.how-section { color: #E8742C; font-size: 13px; font-weight: 700; margin: 12px 0 6px 0;
                padding-bottom: 4px; border-bottom: 1px solid #1e2330; }
.how-metric-lbl { color: #8b949e; font-size: 11px; text-transform: uppercase;
                    letter-spacing: 1px; }
.how-flag { color: #ffb84d; font-weight: 700; margin: 4px 0; }
.how-table { border-collapse: collapse; width: auto; font-size: 12px; }
.how-table th, .how-table td { border: 1px solid #1e2330; padding: 6px 10px;
                                  text-align: left; }
.how-table th { background: #1a2030; color: #E8742C; font-weight: 700;
                 text-transform: uppercase; font-size: 11px; }
.how-badge { display: inline-block; padding: 2px 8px; border-radius: 3px;
              font-size: 11px; font-weight: 700; }
.how-badge-green { background: #1e4d2b; color: #7ae582; }
.how-badge-red { background: #4d1e1e; color: #ff6b6b; }
.how-badge-yellow { background: #4d3a1e; color: #ffb84d; }
.how-badge-blue { background: #1e3a4d; color: #00d4ff; }
.how-note { background: #161a24; border-left: 2px solid #8b949e; padding: 8px 12px;
             margin: 8px 0; font-size: 11px; color: #8b949e; }
.how-hr { border: 0; border-top: 1px solid #1e2330; margin: 16px 0; }
pre.how-memo { background: #0a0d14; color: #cce8ff; padding: 16px; border-radius: 4px;
                font-size: 11px; overflow-x: auto; white-space: pre-wrap; }
</style>
"""

# ----- Tickers config (§3.3) -----
# Tickers — multi-candidate: tenta em ordem, usa o 1o que funcionar.
# Se todos falharem via BBG, cai no FRED (FRED_ALIASES).
TICKERS_CB = {
    'FED':   ['FARBAST Index', 'H41RCASH Index'],
    'ECB':   ['EBBSTOTA Index', 'EUCBTOTA Index'],
    'BOJ':   ['BJACTOTL Index', 'JPNACBA Index'],
    # CNBMTTAS = China Monetary Authority Total Assets (confirmado no BBG)
    'PBOC':  ['CNBMTTAS Index', 'CNGFAS Index'],
    # .BOE_TOT_ = BOE UK Total Assets (custom BBG label)
    'BOE':   ['.BOE_TOT_ Index', 'APFAAPFA Index', 'APFTPFBH Index'],
}
TICKERS_BANK_CREDIT = {
    # ALCBBKCR = All Commercial Banks Bank Credit (US H.8 headline)
    'US':  ['ALCBBKCR Index', 'ALCBCBCT Index', 'H8TBNKCR Index'],
    'EUR': ['ECMSM3 Index', 'ECMAM3 Index'],
    # UKMSM4 = UK Money Supply M4 level (UKMSM41Y = YoY fallback)
    'UK':  ['UKMSM4 Index', 'UKMS Index', 'UKMSM41Y Index'],
    'JP':  ['JNLJAOYS Index', 'JMNSM2 Index'],
    # CNMSM2 = China Monthly Money Supply M2 (confirmado)
    'CN':  ['CNMSM2 Index', 'CNFFFAS Index'],
}
TICKERS_REPO = {
    'SOFR':   ['SOFRRATE Index', 'SOFR Index'],
    'IORB':   ['IORB Index', 'IOER Index'],
    # FARBRBFB = Federal Reserve Bank Reserves (confirmado)
    'WRESBAL':['FARBRBFB Index', 'WRESBAL Index', 'ARESDPIT Index'],
    # TOMOREPO = temp OMO repo (confirmado como proxy RRP)
    'RRP':    ['TOMOREPO Index', 'RRPONTSYD Index', 'RRPONTSYAWARD Index'],
    # USCBFDRA = Treasury General Account (TGA) — confirmado
    'TGA':    ['USCBFDRA Index', 'WTREGEN Index', 'WDTGAL Index'],
    'MOVE':   ['MOVE Index'],
}
TICKERS_WBCI_CORE4 = {
    'US_ISM':    ['NAPMPMI Index'],
    # JNTGALLI = Tankan All Enterprises Actual (confirmado)
    # JWCOOVRL = Japan overall business condition
    'JP_TANKAN': ['JNTGALLI Index', 'JWCOOVRL Index', 'JNTGMFG Index'],
    'DE_IFO':    ['GRIFPBUS Index'],
    # LTSBBANX = UK CBI (confirmado direto)
    'UK_CBI':    ['LTSBBANX Index', 'ECONUKCI Index'],
}
TICKERS_WBCI_EXT = {
    # NAPMNMAN = ISM Services (nao NAPMNMI)
    'US_ISM_SVC':  ['NAPMNMAN Index', 'NAPMNMI Index'],
    # ECSUSUUS = US business condition general
    'US_BUSINESS': ['ECSUSUUS Index'],
    # EMPRGBCI = Empire State Manufacturing
    'US_EMPIRE':   ['EMPRGBCI Index'],
    'EZ_PMI':      ['MPMIEZCA Index'],
    # SCCNSMEI = China business condition
    'CN_BUSINESS': ['SCCNSMEI Index', 'MPMICNMA Index'],
    'GLOBAL_PMI':  ['MPMIGLMA Index'],
    # Novos paises extended
    'FR_INSEE':    ['INSECOMP Index'],
    'IT_ISTAT':    ['ITESECSE Index'],
    'CA_BOC':      ['BCBSPC1 Index'],
}
# Chicago Fed Business Survey (additional context, §3.2)
TICKERS_CHICAGO_FED = {
    'CFSB_OVERALL':  ['CFSBACTI Index'],
    'CFSB_OUTLOOK':  ['CFSBOUTL Index'],
    'CFSB_MFG':      ['CFSBACMF Index'],
    'CFSB_NONMFG':   ['CFSBACNM Index'],
    'CFSB_HIRING':   ['CFSBHIRI Index'],
    'CFSB_CAPEX':    ['CFSBCAPS Index'],
}
TICKERS_REAL_ECON = {
    'OECD_CLI': 'OECDCLI Index',
    'WORLD_GDP':'EHGDUS Index',  # proxy; WGDPWRLD alt
}
TICKERS_SECTORS_CYC = {
    'ConsDisc':   'MXWO0CD Index',
    'Industrials':'MXWO0IN Index',
    'Materials':  'MXWO0MT Index',
    'Financials': 'MXWO0FN Index',
    'IT':         'MXWO0IT Index',
}
TICKERS_SECTORS_DEF = {
    'ConsStaples':'MXWO0CS Index',
    'HealthCare': 'MXWO0HC Index',
    'Utilities':  'MXWO0UT Index',
    'Telecom':    'MXWO0TC Index',
}
TICKERS_RATES = {
    'US2Y':   'USGG2YR Index',
    'US5Y':   'USGG5YR Index',
    'US10Y':  'USGG10YR Index',
    'US30Y':  'USGG30YR Index',
    'US3M':   'USGG3M Index',
    'TIPS10': 'USGGT10Y Index',
    'FED_FUNDS': 'FDTR Index',
    'TP_ACM10': 'ACMTP10 Index',  # ACM 10Y term premium
}
TICKERS_COMMODITIES = {
    'Gold':   'XAU Curncy',
    'WTI':    'CL1 Comdty',
    'Brent':  'CO1 Comdty',
    'Copper': 'HG1 Comdty',
    'BCOM':   'BCOM Index',
}
TICKERS_VOL_CREDIT = {
    'VIX':       'VIX Index',
    'VXV':       'VIX3M Index',
    'HY_OAS':    'LF98OAS Index',
    'IG_OAS':    'LUACOAS Index',
    'ITRAXX_EU': 'ITRXEUE CBIL Curncy',
}
TICKERS_RISK_APPETITE = {
    'DXY': ['DXY Curncy', 'DXY Index'],
    # USTWEME = US trade-weighted EM (confirmado, substitui EMCI)
    'EMFX':['USTWEME Index', 'EMCI Index', 'JPMVXYEM Index'],
    'PUTCALL':['PCUSEQTR Index', 'PCEQUSO Index'],
}
TICKERS_CRYPTO = {
    'BTC': ['XBT Curncy', 'XBTUSD Curncy'],
    # XETUSD = Ethereum/USD Cross (confirmado)
    'ETH': ['XETUSD Curncy', 'XETH Curncy'],
    # XSOUSD = Solana/USD Cross (confirmado)
    'SOL': ['XSOUSD Curncy', 'XSO Curncy', 'XSOL Curncy'],
}
TICKERS_WORLD_M2 = {
    'US_M2':  ['M2NS Index', 'M2SL Index'],
    'EUR_M3': ['ECMAM3 Index', 'ECMSM3 Index'],
    'JP_M2':  ['JMNSM2 Index', 'JNMSM2 Index'],
    # UKMSM4 = UK M4 level; UKMSM41Y = YoY (fallback)
    'UK_M4':  ['UKMSM4 Index', 'UKMSM41Y Index', 'UKMS Index'],
    # CNMSM2 = China M2 level (confirmado)
    'CN_M2':  ['CNMSM2 Index', 'CHM2 Index'],
}
TICKERS_EQ_INDICES = {
    'SPX':   'SPX Index',
    'SPY':   'SPY US Equity',
    'MSCI_WORLD': 'MXWO Index',
    'MSCI_EM':    'MXEF Index',
    'AGG':        'LBUSTRUU Index',
}

# ----- FRED series aliases (fallback quando BQL falhar) -----
# Mapeia o 'label' interno (chave do TICKERS_*) para o FRED series ID.
FRED_ALIASES = {
    # Central Banks
    'FED':      'WALCL',            # Fed balance sheet
    'ECB':      'ECBASSETSW',       # ECB total assets weekly
    'BOJ':      'JPNASSETS',        # BoJ total assets
    'BOE':      'BOGMBASE',         # fallback: US monetary base (se BoE ausente)
    # Bank credit / M
    'US':       'TOTBKCR',          # H.8 bank credit total
    'EUR':      'MYAGM3EZM196S',    # Euro M3 monthly
    'UK':       'MABMM101GBM189S',  # UK M4
    'JP':       'MYAGM2JPM189S',    # Japan M2
    'CN':       'MABMM201CNM189S',  # China M2
    'US_M2':    'M2SL',
    'EUR_M3':   'MYAGM3EZM196S',
    'UK_M4':    'MABMM101GBM189S',
    'JP_M2':    'MYAGM2JPM189S',
    'CN_M2':    'MABMM201CNM189S',
    # Rates
    'US2Y':     'DGS2',
    'US5Y':     'DGS5',
    'US10Y':    'DGS10',
    'US30Y':    'DGS30',
    'US3M':     'DGS3MO',
    'TIPS10':   'DFII10',
    'FED_FUNDS':'FEDFUNDS',
    'TP_ACM10': 'THREEFYTP10',      # Kim-Wright 10Y TP
    # Repo / reserves
    'SOFR':     'SOFR',
    'IORB':     'IORB',
    'WRESBAL':  'WRESBAL',
    'RRP':      'RRPONTSYD',
    # Vol / credit
    'VIX':      'VIXCLS',
    'VXV':      'VXVCLS',
    'HY_OAS':   'BAMLH0A0HYM2',
    'IG_OAS':   'BAMLC0A0CM',
    # Commodities
    'Gold':     'GOLDAMGBD228NLBM',
    'WTI':      'DCOILWTICO',
    'Brent':    'DCOILBRENTEU',
    'Copper':   'PCOPPUSDM',        # copper monthly
    # FX / EM
    'DXY':      'DTWEXBGS',
    'EMFX':     'DTWEXEMEGS',       # trade-weighted emerging
    # PMIs (via FRED onde tiver)
    'US_ISM':   'NAPM',             # ISM manuf (descontinuado no FRED; USMPMI alt)
    'GLOBAL_PMI':'MPMIGLMA',        # provavelmente ausente, log warning
    'OECD_CLI': 'OECDLOLITOAASTSAM',
    # Macro
    'cpi_lvl':  'CPIAUCSL',
    'cpi_yoy':  'CPIAUCSL',         # compute YoY downstream
    'ppi_lvl':  'PPIACO',
    'ppi_yoy':  'PPIACO',
    'umich':    'UMCSENT',
    'umich_cur':'UMCSENT',
    # Equity indices (fallback raro — BBG tem)
    'SPX':      'SP500',
    'NDX':      'NASDAQ100',
}


def _fred_get(series_id: str, years: int = 20) -> pd.Series:
    """Busca serie do FRED. Retorna vazia se falhar."""
    if not HAS_FRED or not series_id:
        return pd.Series(dtype=float)
    try:
        start = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
        s = fred.get_series(series_id, observation_start=start)
        if s is None or len(s) == 0:
            return pd.Series(dtype=float)
        s.index = pd.to_datetime(s.index)
        s = pd.to_numeric(s, errors='coerce').dropna()
        log.info(f'[fred] {series_id}: {len(s)} pts')
        return s
    except Exception as e:
        log.warning(f'[fred] {series_id} fail: {str(e)[:80]}')
        return pd.Series(dtype=float)


# Framework constants
HOWELL_CYCLE_MONTHS = 60  # 60-65 per §3.4
LIQ_LEAD_MONTHS = 18      # §3.4: 15-20m default 18
CRYPTO_WEIGHTS = {'BTC': 0.6, 'ETH': 0.3, 'SOL': 0.1}  # §3.11
RISK_APPETITE_THR = -1.0  # z below -1 = risk-off

# Phase thresholds (§4.2)
FOUR_DUCK_THRESHOLDS = {
    'wbci_peak': 0.5,         # core4 z above peak
    'liq_slope_neg': -0.15,   # liq 3m slope negative
    'curve_bull_flat': True,
    'risk_appetite_off': -1.0,
    'gold_oil_z': 1.0,
}


# ============================================================================
# 3. BQL helpers (reused from session_stats patterns)
# ============================================================================

def _bql_ts(resp_item, col='Value'):
    """Extrai serie temporal de um response item BQL."""
    df = resp_item.df()
    for date_col in ('DATE', 'date', 'Date'):
        if date_col in df.columns:
            df = df.reset_index(drop=True).set_index(date_col)
            break
    if col in df.columns:
        s = df[col]
    elif 'VALUE' in df.columns:
        s = df['VALUE']
    else:
        num = df.select_dtypes(include=[np.number])
        if len(num.columns) == 0:
            raise ValueError(f'sem coluna numerica: {df.columns.tolist()}')
        s = num.iloc[:, -1]
    s.index = pd.to_datetime(s.index, errors='coerce')
    s = s[s.index.notna()]
    return s


def _bql_one_field(ticker: str, field, period: str = '-20Y') -> pd.Series:
    """
    Busca 1 field historico. Cap 20y automatico. Try-except graceful.
    """
    if period.endswith('Y'):
        try:
            yrs = int(period.replace('-', '').replace('Y', ''))
            if yrs > 20:
                period = '-20Y'
        except Exception:
            pass
    if not HAS_BQL:
        raise RuntimeError('BQL nao disponivel')
    req = bql.Request(ticker, {'Value': field(
        dates=bq.func.range(period, '0D'), fill='PREV')})
    s = _bql_ts(bq.execute(req)[0], 'Value')
    s = pd.to_numeric(s, errors='coerce').dropna()
    return s


def safe_load(tickers, field=None, period: str = '-20Y',
               label: str = None) -> pd.Series:
    """
    Carrega serie com try/except. Aceita str (1 ticker) ou list (multi-candidate).
    Tenta cada ticker BBG em ordem; usa o 1o que retornar dados.
    Se tudo falhar, cai no FRED via FRED_ALIASES[label].
    """
    # Normaliza pra lista
    if isinstance(tickers, str):
        tickers = [tickers]

    # 1. Tenta cada candidate BBG em ordem
    if HAS_BQL:
        f = field if field is not None else bq.data.px_last
        for tk in tickers:
            try:
                s = _bql_one_field(tk, f, period)
                if len(s) > 0:
                    if len(tickers) > 1:
                        log.info(f'[bbg] {label or tk}: OK via {tk}')
                    return s
            except Exception as e:
                msg = str(e)[:60]
                if len(tickers) > 1:
                    log.debug(f'[bbg] {label} try {tk}: {msg}')

    # 2. Fallback FRED (via label)
    if HAS_FRED and label:
        fred_id = FRED_ALIASES.get(label)
        if fred_id:
            try:
                yrs = int(period.replace('-', '').replace('Y', ''))
            except Exception:
                yrs = 20
            s = _fred_get(fred_id, years=min(yrs, 20))
            if len(s) > 0:
                log.info(f'[fred fallback] {label} -> {fred_id}')
                return s

    # Todas falharam
    log.warning(f'{label or tickers[0]}: todas fontes vazias '
                  f'({len(tickers)} BBG + FRED)')
    return pd.Series(dtype=float)


def _clean(s):
    """Remove inf/-inf/NaN pra Plotly serializar em JSON."""
    if s is None:
        return pd.Series(dtype=float)
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    return s


def load_group(tickers: dict, period: str = '-20Y',
                field=None) -> dict:
    """Carrega um dict de tickers e retorna dict de series (so as que vieram).
    Registra em _loader_audit quais tickers tiveram sucesso/falha."""
    out = {}
    for name, tk in tickers.items():
        s = safe_load(tk, field, period, label=name)
        _loader_audit[name] = {
            'candidates': tk if isinstance(tk, list) else [tk],
            'got_data': len(s) > 0,
            'n_points': len(s),
            'fred_fallback': FRED_ALIASES.get(name, None),
        }
        if len(s) > 0:
            out[name] = s
    return out


# Audit global pra debug
_loader_audit: dict = {}


def debug_loader_report() -> str:
    """Gera relatorio HTML/text de quais tickers funcionaram."""
    rows = []
    ok_count = 0
    fail_count = 0
    for name, info in _loader_audit.items():
        status = '✓' if info['got_data'] else '✗'
        if info['got_data']:
            ok_count += 1
        else:
            fail_count += 1
        cands = ' | '.join(info['candidates'][:3])
        if len(info['candidates']) > 3:
            cands += f' +{len(info["candidates"])-3}'
        fred = info.get('fred_fallback') or '—'
        rows.append(f"<tr><td>{status}</td><td><b>{name}</b></td>"
                      f"<td>{cands}</td><td>{info['n_points']}</td>"
                      f"<td>{fred}</td></tr>")

    table = ("<table class='how-table'>"
              "<tr><th></th><th>Label</th><th>BBG candidates</th>"
              "<th>Points</th><th>FRED fallback</th></tr>"
              + ''.join(rows) + "</table>")
    summary = (f"<div class='how-card'><b>Loader Audit:</b> "
                f"<span class='how-badge how-badge-green'>{ok_count} OK</span> "
                f"<span class='how-badge how-badge-red'>{fail_count} failed</span> "
                f"| FRED enabled: {'✓' if HAS_FRED else '✗ (set FRED_API_KEY env var)'}"
                f"</div>")
    return summary + f"<div class='how-card'>{table}</div>"


# ============================================================================
# 4. Utility functions (z-scores, sine fit, lead-lag)
# ============================================================================

def rolling_z(s: pd.Series, window_years: int = 10) -> pd.Series:
    """Rolling z-score."""
    s = _clean(s)
    n = int(window_years * 252) if len(s) > 1000 else int(window_years * 12)
    n = min(n, max(24, len(s) // 2))
    return (s - s.rolling(n, min_periods=n // 2).mean()) / \
           s.rolling(n, min_periods=n // 2).std()


def yoy_pct(s: pd.Series, periods: int = 252) -> pd.Series:
    """YoY % change. periods=252 daily, 12 monthly, 52 weekly."""
    s = _clean(s)
    if len(s) < periods * 1.5:
        periods = max(12, len(s) // 4)
    return s.pct_change(periods) * 100


def mom_pct(s: pd.Series, periods: int = 21) -> pd.Series:
    """Month-over-month % change (21 business days approx)."""
    s = _clean(s)
    return s.pct_change(periods) * 100


def sine_wave_fit(s: pd.Series, t_min: int = 54, t_max: int = 72) -> dict:
    """
    Fit sine wave A*sin(2pi*t/T + phi) + slope*t + const ao momentum (§3.4).
    T em meses, bounded [54, 72]. Retorna params + proj 24m.
    """
    if not HAS_SCIPY:
        return {'error': 'scipy nao disponivel'}
    s = _clean(s)
    if len(s) < 60:
        return {'error': f'serie curta demais: {len(s)} pts'}

    # Resample pra mensal se for daily
    is_monthly = (s.index.to_series().diff().dt.days.median() or 1) > 20
    if not is_monthly:
        s = s.resample('M').last().dropna()

    y = s.values.astype(float)
    t = np.arange(len(y), dtype=float)

    def model(t, A, T, phi, slope, const):
        return A * np.sin(2 * np.pi * t / T + phi) + slope * t + const

    try:
        p0 = [np.std(y), 60.0, 0.0, 0.0, np.mean(y)]
        bounds = ([0, t_min, -2*np.pi, -np.inf, -np.inf],
                  [np.inf, t_max, 2*np.pi, np.inf, np.inf])
        popt, _ = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=5000)
        A, T, phi, slope, const = popt
    except Exception as e:
        return {'error': f'fit failed: {str(e)[:60]}'}

    # Projecao 24m
    t_proj = np.arange(len(y), len(y) + 24)
    y_proj = model(t_proj, *popt)

    # Prox inflection: zero-crossing da derivada
    t_dense = np.arange(len(y) - 12, len(y) + 36, 0.1)
    y_dense = model(t_dense, *popt)
    dy = np.diff(y_dense)
    sign_changes = np.where(np.diff(np.sign(dy)))[0]
    # Primeira inflection depois do ultimo ponto
    last_t = len(y)
    future_inflections = [t_dense[i + 1] for i in sign_changes
                           if t_dense[i + 1] > last_t]
    next_inflection_t = future_inflections[0] if future_inflections else None

    if next_inflection_t is not None:
        months_ahead = next_inflection_t - last_t
        last_date = s.index[-1]
        next_date = last_date + pd.DateOffset(months=int(round(months_ahead)))
    else:
        months_ahead = None
        next_date = None

    # Quadrant atual
    current_phase = (2 * np.pi * last_t / T + phi) % (2 * np.pi)
    if current_phase < np.pi / 2:
        quadrant = 'early_expansion'
    elif current_phase < np.pi:
        quadrant = 'late_expansion'
    elif current_phase < 3 * np.pi / 2:
        quadrant = 'early_contraction'
    else:
        quadrant = 'late_contraction'

    proj_index = pd.date_range(start=s.index[-1] + pd.DateOffset(months=1),
                                 periods=24, freq='M')
    proj_series = pd.Series(y_proj, index=proj_index)
    fit_series = pd.Series(model(t, *popt), index=s.index)

    return {
        'T_months': float(T),
        'amplitude': float(A),
        'phase_rad': float(phi),
        'slope': float(slope),
        'const': float(const),
        'fit_series': fit_series,
        'proj_series': proj_series,
        'next_inflection_months': float(months_ahead) if months_ahead else None,
        'next_inflection_date': next_date.strftime('%Y-%m-%d') if next_date else None,
        'current_quadrant': quadrant,
    }


def lead_lag_corr(leader: pd.Series, follower: pd.Series,
                    max_lag: int = 24, freq: str = 'M') -> dict:
    """
    Cross-correlation lead-lag. Retorna lag com max |correlacao|.
    freq 'M' monthly, 'D' daily.
    """
    ldr = _clean(leader)
    flw = _clean(follower)
    if freq == 'M':
        ldr = ldr.resample('M').last()
        flw = flw.resample('M').last()
    aligned = pd.concat([ldr, flw], axis=1, join='inner').dropna()
    if len(aligned) < max_lag * 2:
        return {'error': 'serie curta', 'best_lag': 0, 'best_corr': 0}
    aligned.columns = ['leader', 'follower']
    corrs = {}
    for lag in range(0, max_lag + 1):
        c = aligned['leader'].shift(lag).corr(aligned['follower'])
        if pd.notna(c):
            corrs[lag] = c
    if not corrs:
        return {'error': 'no valid corrs', 'best_lag': 0, 'best_corr': 0}
    best_lag = max(corrs, key=lambda k: abs(corrs[k]))
    return {
        'best_lag': int(best_lag),
        'best_corr': float(corrs[best_lag]),
        'all_corrs': {int(k): float(v) for k, v in corrs.items()},
    }


def gdp_weight(series_dict: dict, weights: dict = None) -> pd.Series:
    """
    Combina dict de series com pesos (defaults = GDP proxy).
    """
    if not series_dict:
        return pd.Series(dtype=float)
    default_w = {'US': 0.26, 'EUR': 0.18, 'CN': 0.18, 'JP': 0.05, 'UK': 0.03}
    w = weights or default_w
    aligned = {}
    for k, s in series_dict.items():
        aligned[k] = _clean(s).resample('M').last()
    df = pd.DataFrame(aligned).dropna(how='all')
    # Normaliza pesos por membros presentes
    active_w = {k: w.get(k, 1.0 / len(df.columns)) for k in df.columns}
    total = sum(active_w.values())
    active_w = {k: v / total for k, v in active_w.items()}
    weighted = sum(df[k] * active_w[k] for k in df.columns)
    return weighted


# ============================================================================
# 5. Indicator builders
# ============================================================================

def build_net_fed_liquidity(period: str = '-20Y') -> dict:
    """
    Net Fed Liquidity (formula BBG CIX do ripple):
        NFL = FARBAST - TOMOREPO*1000 - USCBFDRA*1000
             = Fed Balance Sheet - RRP - TGA

    Multiplicador 1000 porque TOMOREPO/USCBFDRA sao em millions e FARBAST em
    billions — normaliza pra mesma unidade. Resultado em USD bilhoes.
    """
    log.info('[nfl] carregando Fed BS, RRP, TGA...')
    fed = safe_load(TICKERS_CB['FED'], period=period, label='FED')
    rrp = safe_load(TICKERS_REPO['RRP'], period=period, label='RRP')
    tga = safe_load(TICKERS_REPO['TGA'], period=period, label='TGA')

    if len(fed) == 0:
        return {'error': 'Fed BS vazio'}

    # Alinha todas pela data (weekly usualy)
    fed_w = _clean(fed).resample('W').last()
    nfl = fed_w.copy()
    if len(rrp) > 0:
        rrp_w = _clean(rrp).resample('W').last()
        rrp_w, fed_aligned = rrp_w.align(fed_w, join='inner')
        nfl = fed_aligned - (rrp_w * 1000)
    if len(tga) > 0:
        tga_w = _clean(tga).resample('W').last()
        tga_w, nfl_aligned = tga_w.align(nfl, join='inner')
        nfl = nfl_aligned - (tga_w * 1000)

    # YoY + z
    nfl_yoy = yoy_pct(nfl, periods=52)
    nfl_z = rolling_z(nfl, window_years=5)

    return {
        'nfl_series': nfl,
        'fed_bs': fed_w,
        'rrp': _clean(rrp).resample('W').last() if len(rrp) else pd.Series(),
        'tga': _clean(tga).resample('W').last() if len(tga) else pd.Series(),
        'nfl_yoy_pct': nfl_yoy,
        'nfl_z': nfl_z,
        'latest_nfl': float(nfl.iloc[-1]) if len(nfl) else None,
        'latest_yoy_pct': float(nfl_yoy.iloc[-1]) if len(nfl_yoy.dropna()) else None,
        'latest_z': float(nfl_z.iloc[-1]) if len(nfl_z.dropna()) else None,
        'direction_13w': ('rising' if nfl.diff(13).iloc[-1] > 0 else 'falling')
                          if len(nfl) > 13 else 'unknown',
    }


def build_global_liquidity(period: str = '-20Y') -> dict:
    """
    Global Liquidity Index (§3.3). CB + Retail Bank Credit + Repo + Shadow.
    Retorna: series, YoY%, 10y z, fit sine, contribuicao CB vs Private.
    """
    log.info('[liquidity] carregando CBs + bank credit...')
    cbs = load_group(TICKERS_CB, period)
    bank = load_group(TICKERS_BANK_CREDIT, period)
    repo = load_group(TICKERS_REPO, period)

    # USD-sum (assume que todos ja sao em USD ou reasonable proxy)
    all_cb_sum = None
    for k, s in cbs.items():
        s_m = _clean(s).resample('M').last()
        all_cb_sum = s_m if all_cb_sum is None else all_cb_sum.add(s_m, fill_value=0)

    all_bank_sum = None
    for k, s in bank.items():
        s_m = _clean(s).resample('M').last()
        all_bank_sum = s_m if all_bank_sum is None else \
                       all_bank_sum.add(s_m, fill_value=0)

    # Total liquidity USD (proxy)
    total = pd.DataFrame({'cb': all_cb_sum, 'bank': all_bank_sum}).dropna(how='all')
    if total.empty or len(total) < 24:
        return {'error': 'dados de liquidity insuficientes',
                'cb_series': all_cb_sum, 'bank_series': all_bank_sum}
    total['total'] = total.sum(axis=1)

    # YoY + 3m slope + 10y z
    liq_yoy = yoy_pct(total['total'], periods=12)
    liq_z = rolling_z(liq_yoy, window_years=10)
    slope_3m = liq_z.diff(3)

    # Contribuicoes pct
    cb_contrib_pct = (total['cb'].iloc[-1] / total['total'].iloc[-1] * 100
                       if total['total'].iloc[-1] else 50)
    private_contrib_pct = 100 - cb_contrib_pct

    # Sine fit
    fit = sine_wave_fit(liq_z.dropna()) if len(liq_z.dropna()) > 60 else \
          {'error': 'curta'}

    return {
        'total_usd': total['total'],
        'cb_sum': total['cb'],
        'bank_sum': total['bank'],
        'yoy_pct': liq_yoy,
        'liq_z': liq_z,
        'slope_3m': slope_3m,
        'cb_contrib_pct': float(cb_contrib_pct),
        'private_contrib_pct': float(private_contrib_pct),
        'sine_fit': fit,
        'latest_yoy': float(liq_yoy.iloc[-1]) if len(liq_yoy) else None,
        'latest_z': float(liq_z.iloc[-1]) if len(liq_z.dropna()) else None,
        'latest_slope_3m': float(slope_3m.iloc[-1]) if len(slope_3m.dropna()) else None,
    }


def build_wbci(period: str = '-20Y') -> dict:
    """
    WBCI core-4 + extended (§10:44-11:15).
    """
    log.info('[wbci] carregando PMIs...')
    core = load_group(TICKERS_WBCI_CORE4, period)
    ext = load_group(TICKERS_WBCI_EXT, period)

    if not core:
        return {'error': 'nenhum PMI core disponivel'}

    core_df = pd.DataFrame({k: _clean(v).resample('M').last()
                              for k, v in core.items()}).dropna(how='all')
    core_z = core_df.apply(rolling_z)
    core4_composite = core_z.mean(axis=1)

    ext_all = {**core, **ext}
    ext_df = pd.DataFrame({k: _clean(v).resample('M').last()
                            for k, v in ext_all.items()}).dropna(how='all')
    ext_z = ext_df.apply(rolling_z)
    extended_composite = ext_z.mean(axis=1)

    return {
        'core4_z': core4_composite,
        'extended_z': extended_composite,
        'core4_raw': core_df,
        'ext_raw': ext_df,
        'latest_core4_z': float(core4_composite.iloc[-1])
                           if len(core4_composite.dropna()) else None,
        'latest_ext_z': float(extended_composite.iloc[-1])
                          if len(extended_composite.dropna()) else None,
        'direction_3m': 'rising' if core4_composite.diff(3).iloc[-1] > 0
                         else 'falling',
        'direction_12m': 'rising' if core4_composite.diff(12).iloc[-1] > 0
                          else 'falling',
    }


def build_cyc_def(period: str = '-20Y') -> dict:
    """
    Cyclicals vs Defensives (MSCI World). §3.2 lagging-indicator caveat.
    """
    log.info('[cyc_def] carregando setores MSCI...')
    cyc = load_group(TICKERS_SECTORS_CYC, period)
    dfn = load_group(TICKERS_SECTORS_DEF, period)

    if not cyc or not dfn:
        return {'error': 'setores faltando'}

    cyc_eq = pd.DataFrame({k: _clean(v).resample('M').last() for k, v in cyc.items()})
    dfn_eq = pd.DataFrame({k: _clean(v).resample('M').last() for k, v in dfn.items()})

    # Rebased 100 + equal-weight
    cyc_reb = cyc_eq.div(cyc_eq.iloc[0]).mean(axis=1) * 100
    dfn_reb = dfn_eq.div(dfn_eq.iloc[0]).mean(axis=1) * 100
    ratio = cyc_reb / dfn_reb

    return {
        'cyc_composite': cyc_reb,
        'def_composite': dfn_reb,
        'ratio': ratio,
        'ratio_12m_pct': float(ratio.pct_change(12).iloc[-1] * 100)
                          if len(ratio.dropna()) > 12 else None,
        'direction_3m': 'rising' if ratio.diff(3).iloc[-1] > 0 else 'falling',
    }


def build_term_premium(period: str = '-20Y') -> dict:
    """
    Term premium decomposition (§10:04-10:11). ACM 10Y do NY Fed via BQL.
    """
    log.info('[term_premium] carregando rates + TP...')
    rates = load_group(TICKERS_RATES, period)

    if 'TP_ACM10' not in rates:
        # Fallback: TP = 10Y - (avg forward SOFR expected) — aproximacao
        if 'US10Y' in rates:
            log.warning('[term_premium] ACM nao disponivel, usando US10Y puro')
            return {'us10y': rates['US10Y'],
                     'us10y_tp': None, 'direction_3m': 'unknown',
                     'world_10y_tp_bps': None}
        return {'error': 'rates insuficientes'}

    tp = _clean(rates['TP_ACM10'])
    us10y = _clean(rates.get('US10Y', pd.Series(dtype=float)))

    return {
        'us10y': us10y,
        'us10y_tp': tp,
        'latest_tp_bps': float(tp.iloc[-1] * 100) if len(tp) else None,
        'direction_3m': 'rising' if tp.diff(63).iloc[-1] > 0 else 'compressing',
        'terminal_policy_rate': None,  # requer OIS futures — stub
    }


def build_risk_appetite(period: str = '-20Y') -> dict:
    """
    Risk Appetite composite (Ch 13). MSCI/Agg, inv HY OAS, Cyc/Def,
    inv Gold/SPX, Put/Call, VIX/VXV.
    """
    log.info('[risk_appetite] carregando componentes...')
    eq = load_group(TICKERS_EQ_INDICES, period)
    vol = load_group(TICKERS_VOL_CREDIT, period)
    ra = load_group(TICKERS_RISK_APPETITE, period)
    comm = load_group(TICKERS_COMMODITIES, period)

    comps = {}

    # MSCI World / Agg (equity over bonds)
    if 'MSCI_WORLD' in eq and 'AGG' in eq:
        w, a = _clean(eq['MSCI_WORLD']).align(_clean(eq['AGG']), join='inner')
        if len(w) > 60:
            comps['eq_bond_ratio'] = rolling_z((w / a).resample('M').last())

    # -HY OAS (spread compression = risk on)
    if 'HY_OAS' in vol:
        comps['hy_oas_inv'] = -rolling_z(_clean(vol['HY_OAS']).resample('M').last())

    # -Gold/SPX
    if 'Gold' in comm and 'SPX' in eq:
        g, s = _clean(comm['Gold']).align(_clean(eq['SPX']), join='inner')
        if len(g) > 60:
            comps['gold_spx_inv'] = -rolling_z((g / s).resample('M').last())

    # -VIX/VXV (ratio > 1 = backwardation = stress)
    if 'VIX' in vol and 'VXV' in vol:
        v, x = _clean(vol['VIX']).align(_clean(vol['VXV']), join='inner')
        comps['vix_vxv_inv'] = -rolling_z((v / x).resample('M').last())

    # Put/Call
    if 'PUTCALL' in ra:
        comps['putcall_inv'] = -rolling_z(_clean(ra['PUTCALL']).resample('M').last())

    if not comps:
        return {'error': 'nenhum componente risk appetite disponivel'}

    df = pd.DataFrame(comps).dropna(how='all')
    composite = df.mean(axis=1)

    return {
        'composite_z': composite,
        'components': df,
        'latest_z': float(composite.iloc[-1]) if len(composite.dropna()) else None,
        'direction_3m': 'rising' if composite.diff(3).iloc[-1] > 0 else 'falling',
        'trough_flag': bool(composite.iloc[-1] < RISK_APPETITE_THR)
                        if len(composite.dropna()) else False,
    }


def build_gold_oil(period: str = '-20Y') -> dict:
    """Gold/Oil ratio (Ch 10). |z|>2 historico = late cycle / stress."""
    comm = load_group({'Gold': 'XAU Curncy', 'WTI': 'CL1 Comdty'}, period)
    if 'Gold' not in comm or 'WTI' not in comm:
        return {'error': 'gold ou wti faltando'}
    g, w = _clean(comm['Gold']).align(_clean(comm['WTI']), join='inner')
    ratio = (g / w).resample('M').last()
    z = rolling_z(ratio, window_years=20)
    regime = 'late_cycle' if z.iloc[-1] > 1 else 'risk_on' if z.iloc[-1] < -1 \
             else 'neutral'
    return {
        'ratio': ratio,
        'z_20y': z,
        'latest_ratio': float(ratio.iloc[-1]) if len(ratio) else None,
        'latest_z': float(z.iloc[-1]) if len(z.dropna()) else None,
        'regime': regime,
    }


def build_crypto_basket(period: str = '-10Y') -> dict:
    """
    Crypto basket 60/30/10 (§3.11). Barometer alta-freq de liquidez.
    """
    log.info('[crypto] carregando BTC/ETH/SOL...')
    cr = load_group(TICKERS_CRYPTO, period)
    if 'BTC' not in cr:
        return {'error': 'BTC nao disponivel'}

    weights = {'BTC': 0.6, 'ETH': 0.3, 'SOL': 0.1}
    parts = {}
    for k, w in weights.items():
        if k in cr and len(cr[k]) > 30:
            parts[k] = _clean(cr[k]).resample('D').last().ffill()

    if not parts:
        return {'error': 'sem pecas validas'}

    # Rebase 100 no primeiro dia comum
    df = pd.DataFrame(parts).dropna(how='any')
    if df.empty:
        # Fallback: so BTC
        df = pd.DataFrame({'BTC': parts.get('BTC', pd.Series())})
    base = df.iloc[0]
    rebased = df.div(base) * 100

    # Pesos ajustados se SOL/ETH faltarem
    total_w = sum(weights[k] for k in df.columns)
    active_w = {k: weights[k] / total_w for k in df.columns}
    basket = sum(rebased[k] * active_w[k] for k in df.columns)

    return {
        'basket': basket,
        'components_rebased': rebased,
        'latest_3m_pct': float(basket.pct_change(63).iloc[-1] * 100)
                          if len(basket) > 63 else None,
    }


def build_world_m2(period: str = '-20Y') -> dict:
    """World M2 = Σ USD of country M2 (§3.11)."""
    wm = load_group(TICKERS_WORLD_M2, period)
    if not wm:
        return {'error': 'M2 series faltando'}
    df = pd.DataFrame({k: _clean(v).resample('M').last()
                         for k, v in wm.items()}).dropna(how='all')
    if df.empty:
        return {'error': 'm2 vazio'}
    df['total'] = df.sum(axis=1)
    return {
        'total': df['total'],
        'components': df.drop(columns=['total']),
        'yoy_pct': yoy_pct(df['total'], periods=12),
    }


def compute_rate_paradox(period: str = '-20Y') -> dict:
    """
    Rate Paradox (§3.12) — transfer payments stimulative vs draining.
    Proxy: (FedFunds × Publicly Held Debt) - (avg corp spread × corp debt).
    Sem dados exatos de debt-held-by-public e corp debt, usamos um proxy:
    signal = sign(Δ FedFunds × current level) — stub explicativo.
    """
    rates = load_group({'FED_FUNDS': 'FDTR Index'}, period)
    if 'FED_FUNDS' not in rates:
        return {'error': 'fed funds faltando'}
    ff = _clean(rates['FED_FUNDS']).resample('M').last()
    if len(ff) < 12:
        return {'error': 'serie ff curta'}
    # Proxy simplificado: transfer flow proporcional a fed funds level
    # e direcao = derivada do nivel
    ff_yoy = ff.diff(12)
    latest = ff.iloc[-1]
    sign = ('positive_stimulative' if latest > 3 and ff_yoy.iloc[-1] > 0
            else 'negative_draining' if latest < 1 and ff_yoy.iloc[-1] < 0
            else 'neutral')
    # Proxy USD notional: assume ~25tn publicly held debt × ff/100
    transfer_usd_bn = 25000 * latest / 100
    return {
        'fed_funds': ff,
        'fed_funds_yoy_bps': ff_yoy * 100,
        'net_transfer_annual_usd_bn': float(transfer_usd_bn),
        'sign': sign,
        'direction_12m': 'rising' if ff_yoy.iloc[-1] > 0 else 'falling',
    }


def build_yield_curve_area(period: str = '-20Y') -> dict:
    """Curva 3M-30Y, area integrada + (Δnear, Δfar)."""
    log.info('[curve] carregando yields...')
    rates = load_group(TICKERS_RATES, period)
    # Curva composta: 3M, 2Y, 10Y, 30Y
    curve_points = {
        0.25: rates.get('US3M'),
        2:    rates.get('US2Y'),
        5:    rates.get('US5Y'),
        10:   rates.get('US10Y'),
        30:   rates.get('US30Y'),
    }
    curve_points = {k: _clean(v).resample('M').last()
                      for k, v in curve_points.items() if v is not None and len(v)}
    if len(curve_points) < 3:
        return {'error': 'curva insuficiente'}

    df = pd.DataFrame(curve_points).dropna()
    if df.empty:
        return {'error': 'curva vazia'}

    # Area: trapezio sobre pontos. Unidade = %-years.
    # Nao e "bps", e literalmente o integral do yield (em %) sobre tenor (yr).
    tenors = sorted(df.columns)
    df = df[tenors]
    area = df.apply(lambda row: np.trapz(row.values, tenors), axis=1)

    # Proxy: (Δnear, Δfar) via short-end vs long-end 3m delta
    near = df[tenors[0]]
    far = df[tenors[-1]]
    d_near_3m = near.diff(3)
    d_far_3m = far.diff(3)

    # Phase label heuristic (§3.5)
    def _phase_from_deltas(dn, df_):
        if pd.isna(dn) or pd.isna(df_):
            return 'unknown'
        if dn < 0 and df_ > 0:  # short falls, long rises
            return 'Rebound'    # bull steepening
        if dn > 0 and df_ > dn: # long rises more than short
            return 'Calm'       # bear steepening
        if dn > 0 and df_ < 0:  # short rises, long falls
            return 'Speculation'# bear flattening
        if dn < 0 and df_ < dn: # long falls more than short
            return 'Turbulence' # bull flattening
        return 'Neutral'

    phase = _phase_from_deltas(d_near_3m.iloc[-1] if len(d_near_3m) else 0,
                                 d_far_3m.iloc[-1] if len(d_far_3m) else 0)

    return {
        'curve_df': df,
        'area': area,
        'delta_near_3m': d_near_3m,
        'delta_far_3m': d_far_3m,
        'latest_area': float(area.iloc[-1]) if len(area) else None,
        'latest_d_near_bps': float(d_near_3m.iloc[-1] * 100)
                              if len(d_near_3m.dropna()) else None,
        'latest_d_far_bps': float(d_far_3m.iloc[-1] * 100)
                             if len(d_far_3m.dropna()) else None,
        'phase_label': phase,
    }


def build_debt_liquidity(liq: dict, period: str = '-20Y') -> dict:
    """
    Debt-Liquidity refinancing ratio (Ch 7). Proxy: BIS global credit / Liq total.
    Como BIS raro em BQL, usamos proxy com Treasury marketable + corp bond index.
    """
    # Proxy de debt: US Treasury marketable + LEGATRUU (Bloomberg Global Agg)
    # LUATTRUU = Bloomberg US Treasury Total Return (confirmado)
    # LGTRTRUU = Bloomberg Global Agg Treasuries
    debt = load_group({'TREAS': ['LUATTRUU Index', 'LEGATRUU Index'],
                        'GLOBAL_AGG': ['LGTRTRUU Index', 'LEGATRUU Index']}, period)

    if not debt or 'total_usd' not in liq:
        return {'error': 'dados de debt/liq insuficientes'}

    debt_sum = None
    for k, s in debt.items():
        sm = _clean(s).resample('M').last()
        debt_sum = sm if debt_sum is None else debt_sum.add(sm, fill_value=0)

    liq_total = liq['total_usd']
    aligned = pd.concat([debt_sum, liq_total], axis=1, join='inner').dropna()
    if aligned.empty:
        return {'error': 'sem overlap debt/liq'}
    aligned.columns = ['debt', 'liq']
    ratio = aligned['liq'] / aligned['debt']
    z = rolling_z(ratio, window_years=10)

    return {
        'ratio': ratio,
        'ratio_z_10y': z,
        'latest_ratio': float(ratio.iloc[-1]),
        'latest_z': float(z.iloc[-1]) if len(z.dropna()) else None,
        'regime': ('fragile' if z.iloc[-1] < -1 else
                    'stretched' if z.iloc[-1] > 1 else 'neutral'),
    }


# ============================================================================
# 6. Phase classifier (§4)
# ============================================================================

def classify_phase(state: dict) -> dict:
    """
    Phase classifier usando rules (§4.2) + 4-duck count (§3.10).
    """
    liq = state.get('liquidity', {})
    wbci = state.get('wbci', {})
    curve = state.get('yield_curve', {})
    cyc = state.get('cyclicals_defensives', {})
    gold_oil = state.get('gold_oil', {})
    risk = state.get('risk_appetite', {})
    move = state.get('move', {})

    # 4-duck checks
    duck_econ = wbci.get('direction_3m') == 'falling'
    duck_bonds = curve.get('phase_label') in ('Speculation', 'Turbulence') or \
                  (move.get('z_5y') or 0) > 1
    duck_equity = (cyc.get('ratio_12m_pct') or 0) < 0
    duck_liq = (liq.get('slope_3m') or 0) < 0 and (liq.get('yoy_pct') or 0) < 5

    ducks = {
        'economy': 'aligning_turbulence' if duck_econ else 'ok',
        'bonds': 'aligning_turbulence' if duck_bonds else 'ok',
        'equity_sectors': 'aligning_turbulence' if duck_equity else 'ok',
        'liquidity_metrics': 'aligning_turbulence' if duck_liq else 'ok',
    }
    duck_count = sum([duck_econ, duck_bonds, duck_equity, duck_liq])

    # Decision logic (§4.2)
    liq_slope = liq.get('slope_3m') or 0
    liq_z = liq.get('liq_z') or 0
    wbci_z = wbci.get('core4_z') or 0
    cyc_mom = cyc.get('ratio_12m_pct') or 0

    if duck_count >= 3:
        phase = 'Turbulence' if duck_count == 4 else 'Speculation'
    elif liq_slope > 0.1 and wbci_z < 0 and cyc_mom > 0:
        phase = 'Rebound'
    elif liq_slope > 0 and wbci_z > 0 and cyc_mom > 0:
        phase = 'Calm'
    elif liq_slope < 0 and wbci_z > 0:
        phase = 'Speculation'
    elif liq_slope < 0 and wbci_z < 0:
        phase = 'Turbulence'
    else:
        phase = 'Neutral'

    season_map = {'Rebound': 'Spring', 'Calm': 'Summer',
                   'Speculation': 'Autumn', 'Turbulence': 'Winter',
                   'Neutral': 'Unknown'}
    next_map = {'Rebound': 'Calm', 'Calm': 'Speculation',
                  'Speculation': 'Turbulence', 'Turbulence': 'Rebound',
                  'Neutral': 'Neutral'}

    confidence = duck_count / 4 if phase in ('Speculation', 'Turbulence') else \
                  0.6 if phase in ('Rebound', 'Calm') else 0.3

    drivers = sorted([
        {'name': 'liquidity_slope_3m', 'contribution': float(liq_slope)},
        {'name': 'wbci_core4_z', 'contribution': float(wbci_z)},
        {'name': 'cyc_def_12m', 'contribution': float(cyc_mom) / 10},
        {'name': 'risk_appetite_z', 'contribution': float(risk.get('z') or 0)},
    ], key=lambda x: abs(x['contribution']), reverse=True)[:3]

    return {
        'phase': phase,
        'season': season_map[phase],
        'confidence': round(confidence, 2),
        'next_expected_phase': next_map[phase],
        'drivers': drivers,
        'four_ducks': ducks,
        'duck_count': duck_count,
    }


# ============================================================================
# 7. Chart builders (15 charts per spec §5.2)
# ============================================================================

def _fig_base(title: str, height: int = 380) -> 'go.Figure':
    """Base figure com DASH_TEMPLATE."""
    fig = go.Figure()
    fig.update_layout(template=DASH_TEMPLATE, title=title, height=height)
    return fig


def chart_01_liquidity_vs_economy(liq: dict, real_econ: dict = None) -> 'go.Figure':
    """Ch 1: Global Liquidity vs Real Economy + sine-wave fit."""
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.update_layout(template=DASH_TEMPLATE, height=420,
                       title='Chart 1 — Global Liquidity YoY vs Real Economy')

    if 'yoy_pct' in liq:
        s = _clean(liq['yoy_pct'])
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines',
                                    name='Global Liquidity YoY %',
                                    line=dict(color=PALETTE['orange'], width=2)),
                        secondary_y=False)

    if liq.get('sine_fit') and 'fit_series' in liq['sine_fit']:
        fit_s = liq['sine_fit']['fit_series']
        proj_s = liq['sine_fit']['proj_series']
        fig.add_trace(go.Scatter(x=fit_s.index, y=fit_s.values, mode='lines',
                                    name=f"Sine fit (T={liq['sine_fit']['T_months']:.0f}m)",
                                    line=dict(color=PALETTE['red'], width=1.2,
                                               dash='dot')),
                        secondary_y=False)
        fig.add_trace(go.Scatter(x=proj_s.index, y=proj_s.values, mode='lines',
                                    name='Projection 24m',
                                    line=dict(color=PALETTE['red'], width=1.5,
                                               dash='dash')),
                        secondary_y=False)

    fig.update_yaxes(title_text='Liquidity YoY %', secondary_y=False)
    return fig


def chart_02_wbci(wbci: dict) -> 'go.Figure':
    """Ch 2: WBCI core-4 + extended."""
    fig = _fig_base('Chart 2 — World Business Cycle Index (WBCI) — z-score')
    if 'core4_z' in wbci:
        c = _clean(wbci['core4_z'])
        fig.add_trace(go.Scatter(x=c.index, y=c.values, mode='lines',
                                    name='WBCI core-4 (US+JP+DE+UK)',
                                    line=dict(color=PALETTE['orange'], width=2)))
    if 'extended_z' in wbci:
        e = _clean(wbci['extended_z'])
        fig.add_trace(go.Scatter(x=e.index, y=e.values, mode='lines',
                                    name='WBCI extended',
                                    line=dict(color=PALETTE['beige'],
                                               width=1.5, dash='dot')))
    fig.add_hline(y=0, line_color=PALETTE['grey'])
    fig.update_yaxes(title_text='z-score')
    return fig


def chart_03_growth_nowcast(period: str = '-20Y') -> 'go.Figure':
    """
    Ch 3: Growth Nowcast — composite z-score de predictors macro.
    Simples (sem ML) mas funcional. Predictors:
      +BCOM (commodity demand), +2s10s (curve steepness),
      -DXY (USD strength = tightening), -HY OAS (tight = confidence),
      -VIX (low vol = risk on).
    """
    fig = _fig_base('Chart 3 — Growth Nowcast (composite macro z-score)')
    predictors = {}

    # Carrega os componentes (reuso tickers ja configurados)
    dxy = safe_load('DXY Curncy', period=period, label='DXY')
    bcom = safe_load('BCOM Index', period=period, label='BCOM')
    hy = safe_load('LF98OAS Index', period=period, label='HY_OAS')
    vix = safe_load('VIX Index', period=period, label='VIX')
    us2 = safe_load('USGG2YR Index', period=period, label='US2Y')
    us10 = safe_load('USGG10YR Index', period=period, label='US10Y')

    def _z(s, invert=False):
        s = _clean(s).resample('M').last()
        z = rolling_z(s, window_years=10)
        return -z if invert else z

    if len(bcom) > 60:
        predictors['BCOM'] = _z(bcom)
    if len(us2) > 60 and len(us10) > 60:
        s2, s10 = _clean(us2).align(_clean(us10), join='inner')
        curve = (s10 - s2).resample('M').last()
        predictors['2s10s'] = _z(curve)
    if len(dxy) > 60:
        predictors['-DXY'] = _z(dxy, invert=True)
    if len(hy) > 60:
        predictors['-HY_OAS'] = _z(hy, invert=True)
    if len(vix) > 60:
        predictors['-VIX'] = _z(vix, invert=True)

    if not predictors:
        fig.add_annotation(text='Predictors ausentes',
                            xref='paper', yref='paper', x=0.5, y=0.5,
                            showarrow=False, font=dict(color=PALETTE['muted']))
        return fig

    df = pd.DataFrame(predictors).dropna(how='all')
    composite = df.mean(axis=1)

    fig.add_trace(go.Scatter(x=composite.index, y=composite.values, mode='lines',
                                name='Growth Pulse (composite z)',
                                line=dict(color=PALETTE['orange'], width=2.5),
                                fill='tozeroy',
                                fillcolor='rgba(232,116,44,0.15)'))
    for col in df.columns:
        s = _clean(df[col])
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines',
                                    name=col, visible='legendonly',
                                    line=dict(width=1)))
    fig.add_hline(y=0, line_color=PALETTE['grey'])
    fig.add_hline(y=-1, line_color=PALETTE['red'], line_dash='dash',
                   annotation_text='Contraction zone')
    fig.add_hline(y=1, line_color=PALETTE['green'], line_dash='dash',
                   annotation_text='Expansion zone')
    fig.update_yaxes(title_text='z-score')
    return fig


def chart_04_cyc_def(cyc: dict) -> 'go.Figure':
    """Ch 4: Cyclicals vs Defensives."""
    fig = _fig_base('Chart 4 — Cyclicals / Defensives (MSCI World)')
    if 'ratio' in cyc:
        r = _clean(cyc['ratio'])
        fig.add_trace(go.Scatter(x=r.index, y=r.values, mode='lines',
                                    name='Cyc/Def ratio',
                                    line=dict(color=PALETTE['orange'], width=2)))
        # 12m delta
        d = r.pct_change(12) * 100
        fig.add_trace(go.Scatter(x=d.index, y=d.values, mode='lines',
                                    name='12m Δ %', yaxis='y2',
                                    line=dict(color=PALETTE['beige'], width=1.2)))
        fig.update_layout(yaxis2=dict(title='12m Δ %', overlaying='y',
                                         side='right', gridcolor='rgba(0,0,0,0)'))
    return fig


def chart_05b_net_fed_liquidity(nfl: dict) -> 'go.Figure':
    """
    Net Fed Liquidity: FARBAST - RRP*1000 - TGA*1000
    Componentes no mesmo eixo + NFL composta como area laranja.
    """
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.update_layout(template=DASH_TEMPLATE, height=400,
                       title='Chart 5b — Net Fed Liquidity '
                             '(Fed BS − RRP − TGA)')
    if 'nfl_series' in nfl:
        n = _clean(nfl['nfl_series'])
        fig.add_trace(go.Scatter(x=n.index, y=n.values, mode='lines',
                                    name='Net Fed Liquidity',
                                    line=dict(color=PALETTE['orange'], width=2.5),
                                    fill='tozeroy',
                                    fillcolor='rgba(232,116,44,0.12)'),
                        secondary_y=False)
    if 'fed_bs' in nfl and len(nfl.get('fed_bs', [])) > 0:
        b = _clean(nfl['fed_bs'])
        fig.add_trace(go.Scatter(x=b.index, y=b.values, mode='lines',
                                    name='Fed BS (FARBAST)',
                                    line=dict(color=PALETTE['beige'], width=1.2,
                                               dash='dot')),
                        secondary_y=False)
    if 'rrp' in nfl and len(nfl.get('rrp', [])) > 0:
        r = _clean(nfl['rrp']) * 1000
        fig.add_trace(go.Scatter(x=r.index, y=r.values, mode='lines',
                                    name='RRP × 1000',
                                    line=dict(color=PALETTE['red'], width=1.2),
                                    visible='legendonly'),
                        secondary_y=False)
    if 'tga' in nfl and len(nfl.get('tga', [])) > 0:
        t = _clean(nfl['tga']) * 1000
        fig.add_trace(go.Scatter(x=t.index, y=t.values, mode='lines',
                                    name='TGA × 1000',
                                    line=dict(color=PALETTE['purple'], width=1.2),
                                    visible='legendonly'),
                        secondary_y=False)
    fig.update_yaxes(title_text='USD bn', secondary_y=False)
    return fig


def chart_05_stimulus(liq: dict, period: str = '-10Y') -> 'go.Figure':
    """Ch 5: Stimulus decomp (Fed QE / Not-QE / Treasury QE)."""
    fig = _fig_base('Chart 5 — Fed Stimulus Decomposition (Reserves, RRP, Treasury)')
    rates_repo = load_group(TICKERS_REPO, period)
    if 'WRESBAL' in rates_repo:
        r = _clean(rates_repo['WRESBAL'])
        fig.add_trace(go.Scatter(x=r.index, y=r.values, mode='lines',
                                    name='Reserve Balances',
                                    line=dict(color=PALETTE['orange'], width=2)))
    if 'RRP' in rates_repo:
        rr = _clean(rates_repo['RRP'])
        fig.add_trace(go.Scatter(x=rr.index, y=rr.values, mode='lines',
                                    name='ON RRP',
                                    line=dict(color=PALETTE['red'], width=1.5)))
    if 'cb_sum' in liq and liq['cb_sum'] is not None:
        c = _clean(liq['cb_sum'])
        fig.add_trace(go.Scatter(x=c.index, y=c.values, mode='lines',
                                    name='CB Total (Fed+ECB+BoJ+…)',
                                    line=dict(color=PALETTE['beige'], width=1.5,
                                               dash='dot')))
    return fig


def chart_06_repo_spread(period: str = '-10Y') -> 'go.Figure':
    """Ch 6: Repo spread vs Reserves + MOVE."""
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.update_layout(template=DASH_TEMPLATE,
                       title='Chart 6 — SOFR-IORB Spread vs MOVE Index', height=380)
    r = load_group(TICKERS_REPO, period)
    if 'SOFR' in r and 'IORB' in r:
        s, i = _clean(r['SOFR']).align(_clean(r['IORB']), join='inner')
        spread = (s - i) * 100  # bps
        fig.add_trace(go.Scatter(x=spread.index, y=spread.values, mode='lines',
                                    name='SOFR-IORB spread (bps)',
                                    line=dict(color=PALETTE['orange'], width=2)),
                        secondary_y=False)
    if 'MOVE' in r:
        m = _clean(r['MOVE'])
        fig.add_trace(go.Scatter(x=m.index, y=m.values, mode='lines',
                                    name='MOVE Index',
                                    line=dict(color=PALETTE['red'], width=1.5)),
                        secondary_y=True)
    fig.update_yaxes(title_text='bps', secondary_y=False)
    fig.update_yaxes(title_text='MOVE', secondary_y=True)
    return fig


def chart_07_debt_liquidity(dl: dict) -> 'go.Figure':
    """Ch 7: Debt-Liquidity ratio."""
    fig = _fig_base('Chart 7 — Debt / Liquidity Refinancing Capacity')
    if 'ratio' in dl:
        r = _clean(dl['ratio'])
        fig.add_trace(go.Scatter(x=r.index, y=r.values, mode='lines',
                                    name='Liquidity / Debt ratio',
                                    line=dict(color=PALETTE['orange'], width=2)))
    if 'ratio_z_10y' in dl:
        z = _clean(dl['ratio_z_10y'])
        fig.add_trace(go.Scatter(x=z.index, y=z.values, mode='lines',
                                    name='10y z-score',
                                    line=dict(color=PALETTE['beige'], width=1.2,
                                               dash='dot'),
                                    yaxis='y2'))
        fig.update_layout(yaxis2=dict(title='z-score', overlaying='y',
                                         side='right', gridcolor='rgba(0,0,0,0)'))
    return fig


def chart_08_term_premium(tp: dict) -> 'go.Figure':
    """Ch 8: US 10Y decomp — level vs term premium."""
    fig = _fig_base('Chart 8 — US 10Y Decomposition (Yield vs ACM Term Premium)')
    if 'us10y' in tp and tp['us10y'] is not None:
        y = _clean(tp['us10y'])
        fig.add_trace(go.Scatter(x=y.index, y=y.values, mode='lines',
                                    name='US 10Y Yield',
                                    line=dict(color=PALETTE['orange'], width=2)))
    if 'us10y_tp' in tp and tp.get('us10y_tp') is not None:
        t = _clean(tp['us10y_tp'])
        fig.add_trace(go.Scatter(x=t.index, y=t.values, mode='lines',
                                    name='ACM 10Y Term Premium',
                                    line=dict(color=PALETTE['red'], width=1.8)))
    fig.update_yaxes(title_text='%')
    return fig


def chart_09_curve_phase(curve: dict) -> 'go.Figure':
    """Ch 9: Yield curve area + phase label."""
    fig = _fig_base('Chart 9 — Yield Curve Area (3M-30Y integrated)')
    if 'area' in curve:
        a = _clean(curve['area'])
        fig.add_trace(go.Scatter(x=a.index, y=a.values, mode='lines',
                                    name='Curve area',
                                    line=dict(color=PALETTE['orange'], width=2)))
    if 'phase_label' in curve:
        fig.add_annotation(text=f"Phase: <b>{curve['phase_label']}</b>",
                            xref='paper', yref='paper', x=0.98, y=0.95,
                            xanchor='right', showarrow=False,
                            font=dict(color=PALETTE['red'], size=14))
    return fig


def chart_10_gold_oil(go_dict: dict) -> 'go.Figure':
    """Ch 10: Gold/Oil ratio."""
    fig = _fig_base('Chart 10 — Gold / Oil Ratio (late-cycle signal when z > +1)')
    if 'ratio' in go_dict:
        r = _clean(go_dict['ratio'])
        fig.add_trace(go.Scatter(x=r.index, y=r.values, mode='lines',
                                    name='Gold/Oil',
                                    line=dict(color=PALETTE['orange'], width=2)))
    if 'z_20y' in go_dict:
        z = _clean(go_dict['z_20y'])
        fig.add_trace(go.Scatter(x=z.index, y=z.values, mode='lines',
                                    name='20y z-score',
                                    line=dict(color=PALETTE['beige'], width=1.2,
                                               dash='dot'), yaxis='y2'))
        fig.update_layout(yaxis2=dict(title='z', overlaying='y', side='right',
                                         gridcolor='rgba(0,0,0,0)'))
    return fig


def chart_11_world_term_premia(period: str = '-15Y') -> 'go.Figure':
    """Ch 11: World term premia vs terminal policy rate (GLI replica)."""
    fig = _fig_base('Chart 11 — World Term Premium vs Terminal Policy Rate')
    rates = load_group({'US10Y_TP': 'ACMTP10 Index', 'US_FF': 'FDTR Index'},
                         period)
    if 'US10Y_TP' in rates:
        t = _clean(rates['US10Y_TP']).resample('M').last()
        fig.add_trace(go.Scatter(x=t.index, y=t.values, mode='lines',
                                    name='US ACM 10Y TP',
                                    line=dict(color=PALETTE['orange'], width=2)))
    if 'US_FF' in rates:
        f = _clean(rates['US_FF']).resample('M').last()
        fig.add_trace(go.Scatter(x=f.index, y=f.values, mode='lines',
                                    name='Fed Funds',
                                    line=dict(color=PALETTE['red'], width=1.5,
                                               dash='dot')))
    return fig


def chart_12_daily_liquidity_nowcast(period: str = '-10Y') -> 'go.Figure':
    """
    Ch 12: Daily Global Liquidity Nowcast.

    Composite de predictors diarios que antecipam liquidez global antes
    que o dado oficial mensal/semanal saia. Metodologia:

      + BCOM (commodity demand = credit expansion downstream)
      + Copper (industrial cycle lead)
      + Gold (safe-haven; sinal 'inverse' em alguns regimes — aqui neutro)
      - DXY (USD strength = global liquidity drain)
      - EMFX z (weak EM = global tightening)
      - HY OAS z (wider = credit tightening)
      - IG OAS z
      - MOVE z (bond vol = plumbing stress)

    Cada componente = rolling z 1y (diario). Composite = mean.
    Linha laranja solida = nowcast; linha tracejada = momentum 3m.
    """
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.update_layout(template=DASH_TEMPLATE, height=420,
                       title='Chart 12 — Daily Global Liquidity Nowcast '
                             '(FX + credit + commodities composite)')

    # Fetch (todos diarios)
    dxy = safe_load('DXY Curncy', period=period, label='DXY')
    emfx = safe_load(['USTWEME Index', 'EMCI Index'], period=period, label='EMFX')
    hy = safe_load('LF98OAS Index', period=period, label='HY_OAS')
    ig = safe_load('LUACOAS Index', period=period, label='IG_OAS')
    bcom = safe_load('BCOM Index', period=period, label='BCOM')
    copper = safe_load('HG1 Comdty', period=period, label='Copper')
    gold = safe_load('XAU Curncy', period=period, label='Gold')
    move = safe_load('MOVE Index', period=period, label='MOVE')

    def _dz(s, invert=False):
        s = _clean(s).asfreq('D').ffill()
        if len(s) < 250:
            return pd.Series(dtype=float)
        # Rolling 252d z-score (diario, janela 1y)
        mean = s.rolling(252, min_periods=100).mean()
        std = s.rolling(252, min_periods=100).std()
        z = (s - mean) / std
        return -z if invert else z

    components = {}
    if len(bcom) > 250:
        components['BCOM'] = _dz(bcom)
    if len(copper) > 250:
        components['Copper'] = _dz(copper)
    if len(dxy) > 250:
        components['-DXY'] = _dz(dxy, invert=True)
    if len(emfx) > 250:
        components['EMFX'] = _dz(emfx)  # USTWEME sobe = EM forte = liq+
    if len(hy) > 250:
        components['-HY OAS'] = _dz(hy, invert=True)
    if len(ig) > 250:
        components['-IG OAS'] = _dz(ig, invert=True)
    if len(move) > 250:
        components['-MOVE'] = _dz(move, invert=True)

    if not components:
        fig.add_annotation(text='Predictors diarios ausentes',
                            xref='paper', yref='paper', x=0.5, y=0.5,
                            showarrow=False, font=dict(color=PALETTE['muted']))
        return fig

    df = pd.DataFrame(components).dropna(how='all')
    df = df.resample('D').last().ffill(limit=5)
    composite = df.mean(axis=1)
    composite = _clean(composite)

    # Linha laranja solida = nowcast
    fig.add_trace(go.Scatter(x=composite.index, y=composite.values, mode='lines',
                                name='Daily Liquidity Nowcast (z)',
                                line=dict(color=PALETTE['orange'], width=2.2),
                                fill='tozeroy',
                                fillcolor='rgba(232,116,44,0.12)'),
                    secondary_y=False)

    # Momentum 3m (derivada suavizada) — mostra a progressive slide
    momentum = composite.diff(63)  # ~63 dias uteis = 3m
    momentum = _clean(momentum)
    fig.add_trace(go.Scatter(x=momentum.index, y=momentum.values, mode='lines',
                                name='3m Momentum',
                                line=dict(color=PALETTE['red'], width=1.2,
                                           dash='dot')),
                    secondary_y=True)

    # Componentes individuais em legendonly
    for col in df.columns:
        s = _clean(df[col])
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines',
                                    name=col, visible='legendonly',
                                    line=dict(width=1)),
                        secondary_y=False)

    fig.add_hline(y=0, line_color=PALETTE['grey'])
    fig.update_yaxes(title_text='Liquidity Nowcast (z)', secondary_y=False)
    fig.update_yaxes(title_text='3m Momentum (Δz)', secondary_y=True)
    return fig


def chart_13_risk_appetite(ra: dict) -> 'go.Figure':
    """Ch 13: Risk Appetite composite."""
    fig = _fig_base('Chart 13 — Risk Appetite Composite (z-score)')
    if 'composite_z' in ra:
        c = _clean(ra['composite_z'])
        fig.add_trace(go.Scatter(x=c.index, y=c.values, mode='lines',
                                    name='Risk Appetite z',
                                    line=dict(color=PALETTE['orange'], width=2),
                                    fill='tozeroy',
                                    fillcolor='rgba(232,116,44,0.15)'))
    fig.add_hline(y=-1, line_color=PALETTE['red'], line_dash='dash',
                   annotation_text='Risk-off threshold')
    fig.add_hline(y=1, line_color=PALETTE['green'], line_dash='dash',
                   annotation_text='Risk-on threshold')
    return fig


def chart_14_crypto_barometer(cr: dict, liq: dict, wm: dict = None) -> 'go.Figure':
    """Ch 14: Crypto basket + residual (GLI - WorldM2)."""
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.update_layout(template=DASH_TEMPLATE, height=380,
                       title='Chart 14 — Crypto Basket (60/30/10) vs Liquidity Residual')
    if 'basket' in cr:
        b = _clean(cr['basket'])
        fig.add_trace(go.Scatter(x=b.index, y=b.values, mode='lines',
                                    name='Crypto basket (rebased 100)',
                                    line=dict(color=PALETTE['orange'], width=2)),
                        secondary_y=False)

    # Residual = Liq YoY - WorldM2 YoY
    if wm and 'yoy_pct' in wm and 'yoy_pct' in liq:
        liq_y = _clean(liq['yoy_pct']).resample('M').last()
        wm_y = _clean(wm['yoy_pct']).resample('M').last()
        a = pd.concat([liq_y, wm_y], axis=1, join='inner').dropna()
        if not a.empty:
            a.columns = ['liq', 'wm']
            residual = a['liq'] - a['wm']
            fig.add_trace(go.Scatter(x=residual.index, y=residual.values, mode='lines',
                                        name='Liquidity − WorldM2 (residual %)',
                                        line=dict(color=PALETTE['red'], width=1.5,
                                                   dash='dot')),
                            secondary_y=True)
    return fig


def chart_15_transmission_chain(liq: dict, tp: dict, ra: dict,
                                   wbci: dict) -> 'go.Figure':
    """Ch 15: Transmission chain Liquidity -> TP -> RA -> WBCI."""
    fig = _fig_base('Chart 15 — Transmission Chain (all z-scores)')
    if 'liq_z' in liq:
        s = _clean(liq['liq_z'])
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines',
                                    name='Liquidity z',
                                    line=dict(color=PALETTE['orange'], width=2)))
    if 'us10y_tp' in tp and tp.get('us10y_tp') is not None:
        t = _clean(tp['us10y_tp'])
        fig.add_trace(go.Scatter(x=t.index, y=t.values, mode='lines',
                                    name='Term Premium',
                                    line=dict(color=PALETTE['red'], width=1.5)))
    if 'composite_z' in ra:
        r = _clean(ra['composite_z'])
        fig.add_trace(go.Scatter(x=r.index, y=r.values, mode='lines',
                                    name='Risk Appetite z',
                                    line=dict(color=PALETTE['beige'], width=1.5)))
    if 'core4_z' in wbci:
        w = _clean(wbci['core4_z'])
        fig.add_trace(go.Scatter(x=w.index, y=w.values, mode='lines',
                                    name='WBCI core-4 z',
                                    line=dict(color=PALETTE['green'], width=1.5,
                                               dash='dot')))
    return fig


# ============================================================================
# 8. Agent A — Harvester
# ============================================================================

def run_harvest(years: int = 20) -> dict:
    """
    Agent A: pulla dados, constroi indicadores, renderiza charts,
    roda classifier + sine-fit, retorna state + figs.
    """
    years = min(years, 20)
    period = f'-{years}Y'
    log.info(f'[harvester] iniciando com {years}y de historia')

    # Indicators
    liq = build_global_liquidity(period)
    nfl = build_net_fed_liquidity(period)
    wbci = build_wbci(period)
    cyc = build_cyc_def(period)
    tp = build_term_premium(period)
    ra = build_risk_appetite(period)
    gold_oil = build_gold_oil(period)
    cr = build_crypto_basket(period='-10Y')
    wm = build_world_m2(period)
    rate_p = compute_rate_paradox(period)
    curve = build_yield_curve_area(period)
    dl = build_debt_liquidity(liq, period)
    move_data = load_group({'MOVE': 'MOVE Index'}, period)
    move_s = _clean(move_data.get('MOVE', pd.Series()))
    move_z = rolling_z(move_s, window_years=5) if len(move_s) else pd.Series()

    # Assemble raw state
    state = {
        'as_of_date': AS_OF,
        'version': VERSION,
        'cycle': {
            'length_months_sine_fit': liq.get('sine_fit', {}).get('T_months'),
            'next_inflection_date': liq.get('sine_fit', {}).get('next_inflection_date'),
            'months_to_next_inflection': liq.get('sine_fit', {}).get('next_inflection_months'),
            'current_quadrant': liq.get('sine_fit', {}).get('current_quadrant'),
        },
        'liquidity': {
            'liq_z': liq.get('latest_z'),
            'yoy_pct': liq.get('latest_yoy'),
            'slope_3m': liq.get('latest_slope_3m'),
            'contribution_central_bank_pct': liq.get('cb_contrib_pct'),
            'contribution_private_sector_pct': liq.get('private_contrib_pct'),
        },
        'net_fed_liquidity': {
            'nfl_usd_bn': nfl.get('latest_nfl'),
            'yoy_pct': nfl.get('latest_yoy_pct'),
            'z_5y': nfl.get('latest_z'),
            'direction_13w': nfl.get('direction_13w'),
        },
        'wbci': {
            'core4_z': wbci.get('latest_core4_z'),
            'extended_z': wbci.get('latest_ext_z'),
            'direction_3m': wbci.get('direction_3m'),
            'direction_12m': wbci.get('direction_12m'),
        },
        'yield_curve': {
            'area_bps': curve.get('latest_area_bps'),
            'delta_near_3m_bps': curve.get('latest_d_near_bps'),
            'delta_far_3m_bps': curve.get('latest_d_far_bps'),
            'phase_label': curve.get('phase_label'),
        },
        'term_premium': {
            'us_10y_bps': tp.get('latest_tp_bps'),
            'direction_3m': tp.get('direction_3m'),
        },
        'risk_appetite': {
            'z': ra.get('latest_z'),
            'direction_3m': ra.get('direction_3m'),
            'trough_flag': ra.get('trough_flag'),
        },
        'cyclicals_defensives': {
            'ratio_12m_pct': cyc.get('ratio_12m_pct'),
            'direction_3m': cyc.get('direction_3m'),
        },
        'gold_oil': {
            'ratio': gold_oil.get('latest_ratio'),
            'z_20y': gold_oil.get('latest_z'),
            'regime': gold_oil.get('regime'),
        },
        'debt_liquidity': {
            'refinancing_capacity': dl.get('latest_ratio'),
            'z_10y': dl.get('latest_z'),
            'regime': dl.get('regime'),
        },
        'move': {
            'level': float(move_s.iloc[-1]) if len(move_s) else None,
            'z_5y': float(move_z.iloc[-1]) if len(move_z.dropna()) else None,
            'regime': ('elevated' if len(move_z.dropna()) and move_z.iloc[-1] > 1
                        else 'normal'),
        },
        'crypto': {
            'basket_3m_pct': cr.get('latest_3m_pct'),
        },
        'rate_paradox': {
            'net_transfer_annual_usd_bn': rate_p.get('net_transfer_annual_usd_bn'),
            'sign': rate_p.get('sign'),
            'direction_12m': rate_p.get('direction_12m'),
        },
        'data_quality': {
            'missing': [k for k, v in {
                'liquidity': liq.get('error'),
                'wbci': wbci.get('error'),
                'yield_curve': curve.get('error'),
                'term_premium': tp.get('error'),
                'risk_appetite': ra.get('error'),
                'crypto': cr.get('error'),
                'debt_liquidity': dl.get('error'),
                'rate_paradox': rate_p.get('error'),
            }.items() if v],
        },
    }

    # Classifier
    state['classifier_output'] = classify_phase(state)
    state['four_ducks'] = state['classifier_output']['four_ducks']
    state['four_ducks']['alignment_count'] = state['classifier_output']['duck_count']

    # Transmission lags — compute on-the-fly
    lags = {}
    if 'liq_z' in liq and tp.get('us10y_tp') is not None:
        r1 = lead_lag_corr(liq['liq_z'], tp['us10y_tp'], max_lag=12)
        lags['liquidity_to_term_premium_months'] = r1.get('best_lag')
    if tp.get('us10y_tp') is not None and 'composite_z' in ra:
        r2 = lead_lag_corr(tp['us10y_tp'], ra['composite_z'], max_lag=12)
        lags['term_premium_to_risk_appetite_months'] = r2.get('best_lag')
    if 'composite_z' in ra and 'core4_z' in wbci:
        r3 = lead_lag_corr(ra['composite_z'], wbci['core4_z'], max_lag=18)
        lags['risk_appetite_to_wbci_months'] = r3.get('best_lag')
    if lags:
        lags['end_to_end_months'] = sum(v for v in lags.values() if v)
    state['transmission_lags'] = lags

    # Figs
    figs = {
        'chart_01': chart_01_liquidity_vs_economy(liq),
        'chart_02': chart_02_wbci(wbci),
        'chart_03': chart_03_growth_nowcast(period),
        'chart_04': chart_04_cyc_def(cyc),
        'chart_05':  chart_05_stimulus(liq, period),
        'chart_05b': chart_05b_net_fed_liquidity(nfl),
        'chart_06': chart_06_repo_spread(period),
        'chart_07': chart_07_debt_liquidity(dl),
        'chart_08': chart_08_term_premium(tp),
        'chart_09': chart_09_curve_phase(curve),
        'chart_10': chart_10_gold_oil(gold_oil),
        'chart_11': chart_11_world_term_premia(period),
        'chart_12': chart_12_daily_liquidity_nowcast(period),
        'chart_13': chart_13_risk_appetite(ra),
        'chart_14': chart_14_crypto_barometer(cr, liq, wm),
        'chart_15': chart_15_transmission_chain(liq, tp, ra, wbci),
    }

    return {
        'state': state,
        'figs': figs,
        'raw': {
            'liquidity': liq, 'wbci': wbci, 'cyc_def': cyc,
            'term_premium': tp, 'risk_appetite': ra, 'gold_oil': gold_oil,
            'crypto': cr, 'world_m2': wm, 'rate_paradox': rate_p,
            'curve': curve, 'debt_liquidity': dl,
        },
    }


# ============================================================================
# 9. Agent B — Analyst (memo writer)
# ============================================================================

def run_analyst(state: dict) -> str:
    """
    Agent B: aplica framework ao state e gera memo markdown.
    Nao re-pulla dados.
    """
    s = state
    cls = s.get('classifier_output', {})
    liq = s.get('liquidity', {})
    wbci = s.get('wbci', {})
    curve = s.get('yield_curve', {})
    cyc = s.get('cyclicals_defensives', {})
    tp = s.get('term_premium', {})
    ra = s.get('risk_appetite', {})
    go_ = s.get('gold_oil', {})
    dl = s.get('debt_liquidity', {})
    mv = s.get('move', {})
    cr = s.get('crypto', {})
    rp = s.get('rate_paradox', {})
    cyc_ = s.get('cycle', {})
    ducks = s.get('four_ducks', {})
    lags = s.get('transmission_lags', {})

    def _fmt(v, unit='', dec=2):
        if v is None:
            return 'N/A'
        if isinstance(v, float):
            return f'{v:.{dec}f}{unit}'
        return f'{v}{unit}'

    def _badge(val, good, bad):
        if val is None:
            return '<span class="how-badge how-badge-yellow">N/A</span>'
        cls_ = 'green' if val == good else ('red' if val == bad else 'yellow')
        return f'<span class="how-badge how-badge-{cls_}">{val}</span>'

    phase = cls.get('phase', 'Unknown')
    season = cls.get('season', 'Unknown')
    conf = cls.get('confidence', 0)
    next_p = cls.get('next_expected_phase', 'Unknown')
    duck_cnt = ducks.get('alignment_count', 0)
    months_to_next = cyc_.get('months_to_next_inflection')

    # Q1: phase vs yield-curve label sanity check
    yc_label = curve.get('phase_label')
    phase_mismatch = (yc_label and yc_label != phase and yc_label != 'Neutral')

    # Q5: Private vs CB
    priv_pct = liq.get('contribution_private_sector_pct') or 0
    priv_dom = priv_pct > 60

    # Q6: fragility
    fragile = (dl.get('z_10y') or 0) < -1 or (mv.get('z_5y') or 0) > 1

    # Q7: MOVE/repo
    move_stress = (mv.get('z_5y') or 0) > 1
    # repo: nao temos z direto mas sinalizamos se regime elevated
    repo_stress = move_stress  # proxy simples

    # Q8: rate paradox
    paradox = rp.get('sign') == 'positive_stimulative' and \
               rp.get('direction_12m') == 'rising'

    # Q9: crypto gap — nao computado direto, mas usamos 3m
    crypto_3m = cr.get('basket_3m_pct') or 0

    memo = f"""# Howell Liquidity Framework — Market Phase Memo
**As of**: {s.get('as_of_date')}  |  **Version**: {s.get('version', 'v4')}

## Executive Summary
- **Phase**: {phase} ({season})
- **Confidence**: {_fmt(conf, '', 2)} ({duck_cnt}/4 ducks aligned)
- **Next expected phase**: {next_p} in ~{_fmt(months_to_next, ' months', 0)}
- **Single-sentence call**: {_one_liner(phase, next_p, priv_dom, fragile, move_stress)}

## 1. Where We Are in the Cycle (§3.4, §3.5)
- Classifier phase: **{phase}** (season: {season})
- Yield-curve-derived label: **{yc_label}** { '⚠ MISMATCH with classifier' if phase_mismatch else '✓ agrees' }
- Sine-fit cycle length: **{_fmt(cyc_.get('length_months_sine_fit'), ' months', 0)}** (Howell prior: 60-65m)
- Current quadrant: **{cyc_.get('current_quadrant', 'unknown')}**
- Next inflection projected: **{cyc_.get('next_inflection_date', 'unknown')}** ({_fmt(months_to_next, 'm', 0)} ahead)

## 2. Four-Duck Scorecard (§3.10)
| Duck | Status | Reading |
|------|--------|---------|
| Economy (WBCI core-4) | {ducks.get('economy', 'n/a')} | z={_fmt(wbci.get('core4_z'), '', 2)}, 3m {wbci.get('direction_3m', 'n/a')} |
| Bond markets (curve + MOVE) | {ducks.get('bonds', 'n/a')} | area {_fmt(curve.get('area_bps'), 'bps', 0)}, MOVE z {_fmt(mv.get('z_5y'), '', 2)} |
| Equity sectors (Cyc/Def) | {ducks.get('equity_sectors', 'n/a')} | 12m ratio {_fmt(cyc.get('ratio_12m_pct'), '%', 1)} |
| Liquidity (GLI YoY + slope) | {ducks.get('liquidity_metrics', 'n/a')} | YoY {_fmt(liq.get('yoy_pct'), '%', 1)}, slope_3m {_fmt(liq.get('slope_3m'), '', 2)} |

{_four_ducks_summary(duck_cnt)}

## 3. Transmission Chain (§3.6)
Current empirical lags (from lead-lag cross-correlation):
- Liquidity → Term Premium: **{_fmt(lags.get('liquidity_to_term_premium_months'), ' months', 0)}** (Howell prior: 1-3m)
- Term Premium → Risk Appetite: **{_fmt(lags.get('term_premium_to_risk_appetite_months'), ' months', 0)}** (prior: 3-6m)
- Risk Appetite → WBCI: **{_fmt(lags.get('risk_appetite_to_wbci_months'), ' months', 0)}** (prior: 6-12m)
- **End-to-end**: ~{_fmt(lags.get('end_to_end_months'), ' months', 0)} (prior: 15-20m)

Reading: { _transmission_reading(liq, tp, ra, wbci, lags) }

## 4. Liquidity — Private vs Central Bank (§3.3)
- **Private sector contribution**: {_fmt(priv_pct, '%', 1)} {'✓ dominante (regime moderno)' if priv_dom else '⚠ CB ainda swing factor'}
- CB contribution: {_fmt(liq.get('contribution_central_bank_pct'), '%', 1)}
- Per §3.3, Howell stresses private-sector dominance; retail bank credit + shadow + repo devem superar CB balance sheets materialmente.

## 5. Fragility — Debt/Liquidity + MOVE + Repo (§3.7, §3.9)
- Liquidity/Debt ratio z: **{_fmt(dl.get('z_10y'), '', 2)}** ({dl.get('regime', 'n/a')})
- MOVE level: **{_fmt(mv.get('level'), '', 1)}** (z_5y: {_fmt(mv.get('z_5y'), '', 2)}, regime {mv.get('regime', 'n/a')})
- { _fragility_reading(fragile, move_stress, repo_stress) }

## 6. Rate Paradox (§3.12) — CHALLENGES CONSENSUS
- Net transfer payments annual: **${_fmt(rp.get('net_transfer_annual_usd_bn'), 'bn', 0)}**
- Sign: **{rp.get('sign', 'n/a')}** | Direction 12m: **{rp.get('direction_12m', 'n/a')}**
- { _paradox_reading(paradox) }

## 7. Crypto Cross-Check (§3.11)
- Basket 3m return: **{_fmt(crypto_3m, '%', 1)}**
- { _crypto_reading(crypto_3m, liq) }

## 8. What I'd Watch Next (triggers)
- MOVE breaking > 120 (current {_fmt(mv.get('level'), '', 0)}) → forced Fed intervention probable (§3.7)
- Risk Appetite z crossing below -1.5 (current {_fmt(ra.get('z'), '', 2)}) → confirms Turbulence entry
- Curve area flipping sign (current {_fmt(curve.get('area_bps'), 'bps', 0)}) → phase transition trigger
- Liquidity YoY crossing below -5% (current {_fmt(liq.get('yoy_pct'), '%', 1)}) → deep contraction signal
- Crypto basket 3m < -15% (current {_fmt(crypto_3m, '%', 1)}) → liquidity leading risk

## 9. Data Gaps / Caveats
{_data_gaps(s.get('data_quality', {}))}

---
_Framework reference: §§3.1-3.12. Proxy indices: global_liquidity, world_term_premium, world_m2. Phase classifier per §4; sine-wave fit per §3.4._
"""
    return memo


def _one_liner(phase, next_p, priv_dom, fragile, move_stress):
    if phase == 'Turbulence':
        return ('Sistema em Turbulence: 4 ducks alinhados, bull flattening, '
                'risk appetite < -1. Defensive posture.')
    if phase == 'Speculation':
        return (f'Speculation/Autumn — liquidity inflecting, curve flattening, '
                f'proxima parada Turbulence. {"MOVE stress elevada." if move_stress else ""}')
    if phase == 'Calm':
        return ('Calm/Summer — liquidity + WBCI rising, bear steepening, risk-on.')
    if phase == 'Rebound':
        return ('Rebound/Spring — liquidity leaving trough, bull steepening. '
                'Buy bonds early, equities late.')
    return f'Phase {phase}, transitioning to {next_p}.'


def _four_ducks_summary(count):
    if count == 4:
        return ('> **High-confidence Turbulence call** — all four ducks aligned (§3.10).')
    if count == 3:
        return ('> Three ducks lining up — Speculation/Turbulence transition likely, '
                'watch the 4th (§3.10).')
    if count == 2:
        return '> Two ducks aligned — mixed signal, no conviction call yet.'
    return '> Fewer than 2 ducks — insufficient alignment for downturn call.'


def _transmission_reading(liq, tp, ra, wbci, lags):
    slope = liq.get('slope_3m') or 0
    if slope < -0.1:
        return (f'Liquidity slope negative ({slope:.2f}); curve should flatten within '
                f'{lags.get("liquidity_to_term_premium_months", 2)}m per empirical lag; '
                f'Risk Appetite deterioration expected '
                f'{lags.get("term_premium_to_risk_appetite_months", 4)}m later.')
    if slope > 0.1:
        return (f'Liquidity slope positive ({slope:.2f}); curve should steepen; '
                f'Risk Appetite recovery expected within '
                f'{(lags.get("liquidity_to_term_premium_months") or 0) + (lags.get("term_premium_to_risk_appetite_months") or 0)}m.')
    return 'Liquidity slope near zero — transmission chain in transition.'


def _fragility_reading(fragile, move_stress, repo_stress):
    flags = []
    if fragile:
        flags.append('⚠ Debt/Liq ratio stretched (z < -1)')
    if move_stress:
        flags.append('⚠ MOVE z > 1')
    if repo_stress:
        flags.append('⚠ Repo plumbing stress')
    if flags:
        return ('Fragility flags: ' + ' | '.join(flags) +
                ' — Fed intervention probability elevated (§3.7).')
    return 'No acute fragility signals — plumbing functioning normally.'


def _paradox_reading(paradox):
    if paradox:
        return ('**Warning**: rate paradox signal positive_stimulative AND rising. '
                'Consensus reading that "Fed cutting = liquidity-positive" may be '
                'inverted this cycle. Check §3.12 — with government as dominant '
                'debtor, rate cuts reduce transfer payments to private sector.')
    return 'Paradox signal neutral — consensus rate interpretation likely valid.'


def _crypto_reading(cr3m, liq):
    liq_yoy = liq.get('yoy_pct') or 0
    if cr3m < -10 and liq_yoy < 0:
        return ('Crypto + Liquidity both contracting — confirmation of risk-off '
                '(§3.11 barometer working correctly).')
    if cr3m > 10 and liq_yoy < 0:
        return ('Crypto rallying while liquidity contracts — anomaly; check for '
                'idiosyncratic factors (ETF flows, halving cycle, regulatory).')
    if cr3m > 0 and liq_yoy > 0:
        return 'Crypto + Liquidity both expanding — confirmation of risk-on regime.'
    return 'Crypto signal mixed — not a strong cross-check at this juncture.'


def _data_gaps(dq):
    missing = dq.get('missing', []) or []
    if not missing:
        return 'No critical data gaps.'
    return ('Missing sections: **' + ', '.join(missing) +
            '**. Framework calls affected may be lower-confidence. '
            'Ver `daily.log` (se existir) pra mensagens BQL exatas.')


# ============================================================================
# 10. ZIP export
# ============================================================================

def build_zip(result: dict) -> bytes:
    """
    Bundle state.json + memo.md + figs PNG num zip.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        # state.json
        state_json = json.dumps(result['state'], indent=2, default=str)
        z.writestr(f"state_{AS_OF}.json", state_json)
        # memo.md
        memo = run_analyst(result['state'])
        z.writestr(f"memo_{AS_OF}.md", memo)
        # figs HTML (plotly to_html)
        for name, fig in result.get('figs', {}).items():
            try:
                html = fig.to_html(full_html=True, include_plotlyjs='cdn')
                z.writestr(f"charts/{name}.html", html)
            except Exception as e:
                log.warning(f'zip {name}: {e}')
        # raw metadata
        z.writestr('README.txt',
                    f"Howell Liquidity Framework {VERSION}\n"
                    f"Generated: {AS_OF}\n\n"
                    f"Files:\n"
                    f"  state_{AS_OF}.json — Agent A output (structured data)\n"
                    f"  memo_{AS_OF}.md    — Agent B output (framework memo)\n"
                    f"  charts/*.html      — interactive Plotly figs (15 charts)\n")
    buf.seek(0)
    return buf.getvalue()


# ============================================================================
# 11. UI — widgets + run button + ZIP export
# ============================================================================

_result_cache = {'result': None}


def _build_section_widgets(result: dict) -> list:
    """Converte result em lista de widgets pra display."""
    sec = []
    state = result['state']
    cls = state.get('classifier_output', {})

    # Header
    sec.append(wd.HTML(
        DASH_CSS +
        f"<div class='how-divider'>"
        f"<div class='how-divider-title'>HOWELL LIQUIDITY FRAMEWORK {VERSION}</div>"
        f"<div class='how-divider-sub'>As of {state['as_of_date']} | "
        f"Two-agent liquidity framework — Harvester + Analyst</div>"
        f"</div>"))

    # Executive summary card
    phase = cls.get('phase', 'Unknown')
    season = cls.get('season', 'Unknown')
    next_p = cls.get('next_expected_phase', 'Unknown')
    conf = cls.get('confidence', 0)
    duck_cnt = state.get('four_ducks', {}).get('alignment_count', 0)

    phase_colors = {'Rebound': 'green', 'Calm': 'blue',
                     'Speculation': 'yellow', 'Turbulence': 'red',
                     'Neutral': 'yellow', 'Unknown': 'yellow'}
    pc = phase_colors.get(phase, 'yellow')

    sec.append(wd.HTML(
        f"<div class='how-card how-root'>"
        f"<div class='how-section'>EXECUTIVE SUMMARY</div>"
        f"<div style='font-size:16px; margin-bottom:10px;'>"
        f"Phase: <span class='how-badge how-badge-{pc}'>{phase}</span> "
        f"(Season: <b>{season}</b>) | "
        f"Confidence: <b>{conf}</b> | "
        f"Ducks aligned: <b>{duck_cnt}/4</b> | "
        f"Next: <b>{next_p}</b>"
        f"</div>"
        f"<div style='font-size:12px; color:#8b949e;'>"
        f"Sine-fit T: {state.get('cycle', {}).get('length_months_sine_fit', 'N/A')}m | "
        f"Next inflection: {state.get('cycle', {}).get('next_inflection_date', 'N/A')} | "
        f"Quadrant: {state.get('cycle', {}).get('current_quadrant', 'N/A')}"
        f"</div>"
        f"</div>"))

    # Loader audit (debug de tickers)
    sec.append(wd.HTML("<div class='how-divider'><div class='how-divider-title'>"
                         "LOADER AUDIT (debug) — quais BBG tickers funcionaram</div>"
                         "<div class='how-divider-sub'>Use pra identificar tickers "
                         "quebrados e adicionar FRED_API_KEY env var se faltando</div>"
                         "</div>"))
    sec.append(wd.HTML(debug_loader_report()))

    # Charts (all 15)
    sec.append(wd.HTML("<div class='how-divider'><div class='how-divider-title'>"
                         "AGENT A — HARVESTED CHARTS (15)</div></div>"))
    for name in sorted(result['figs'].keys()):
        fig = result['figs'][name]
        try:
            sec.append(go.FigureWidget(fig))
        except Exception:
            # fallback via Output+display
            out = wd.Output()
            with out:
                display(fig)
            sec.append(out)

    # State.json dump (collapsible)
    state_pretty = json.dumps(state, indent=2, default=str)
    sec.append(wd.HTML("<div class='how-divider'><div class='how-divider-title'>"
                         "AGENT A — STATE.JSON (Handoff Contract)</div>"
                         "<div class='how-divider-sub'>Schema per §5.3 — "
                         "consumed by Agent B</div></div>"))
    sec.append(wd.HTML(
        f"<div class='how-card'><pre class='how-memo' style='max-height:400px;overflow:auto;'>"
        f"{state_pretty[:8000]}{'...[truncated]' if len(state_pretty) > 8000 else ''}"
        f"</pre></div>"))

    # Agent B memo
    sec.append(wd.HTML("<div class='how-divider'><div class='how-divider-title'>"
                         "AGENT B — ANALYST MEMO</div>"
                         "<div class='how-divider-sub'>Framework-driven "
                         "interpretation (§§3.1-3.12)</div></div>"))
    memo = run_analyst(state)
    sec.append(wd.HTML(
        f"<div class='how-card'><pre class='how-memo'>{memo}</pre></div>"))

    return sec


def display_dashboard():
    """Entry point chamado na ultima celula."""
    if not HAS_WIDGETS:
        log.warning('ipywidgets nao disponivel — rodando modo console')
        result = run_harvest(years=20)
        memo = run_analyst(result['state'])
        print(memo)
        return

    years_w = wd.IntSlider(value=20, min=5, max=20, step=1,
                             description='Years:',
                             layout=wd.Layout(width='320px'))
    run_btn = wd.Button(description='▶ Run Harvester + Analyst',
                          button_style='success', icon='play',
                          layout=wd.Layout(width='260px'))
    zip_btn = wd.Button(description='📦 Export ZIP',
                          button_style='info', icon='download',
                          layout=wd.Layout(width='180px'), disabled=True)
    out_main = wd.Output()
    out_zip = wd.Output()

    def _on_run(_):
        zip_btn.disabled = True
        with out_main:
            clear_output(wait=True)
            display(wd.HTML(DASH_CSS +
                              "<div class='how-card'>⏳ Harvesting BQL data "
                              "(may take 30-60s for all 15 charts)...</div>"))
            try:
                years = int(years_w.value)
                result = run_harvest(years=years)
                _result_cache['result'] = result
                clear_output(wait=True)
                sections = _build_section_widgets(result)
                for s in sections:
                    display(s)
                zip_btn.disabled = False
            except Exception as e:
                clear_output(wait=True)
                display(wd.HTML(DASH_CSS +
                                  f"<div class='how-card'><p class='how-flag'>"
                                  f"❌ Harvest failed:</p>"
                                  f"<p>{e}</p>"
                                  f"<pre style='color:#8b949e;font-size:10px;'>"
                                  f"{traceback.format_exc()}</pre></div>"))

    def _on_zip(_):
        with out_zip:
            clear_output(wait=True)
            if not _result_cache['result']:
                display(wd.HTML(DASH_CSS +
                                  "<div class='how-card'><p class='how-flag'>"
                                  "⚠ Run harvester primeiro.</p></div>"))
                return
            try:
                blob = build_zip(_result_cache['result'])
                fname = f'howell_{AS_OF}.zip'
                b64 = base64.b64encode(blob).decode('ascii')
                # Trigger download via JS (funciona no BQuant sandbox,
                # mesmo padrao do session_stats.py)
                js = (f"var a=document.createElement('a');"
                       f"a.href='data:application/zip;base64,{b64}';"
                       f"a.download='{fname}';"
                       f"document.body.appendChild(a);a.click();"
                       f"document.body.removeChild(a);")
                display(IPyHTML(f"<script>{js}</script>"))
                display(wd.HTML(DASH_CSS +
                                  f"<div class='how-card'>"
                                  f"<p>✅ ZIP gerado: <b>{fname}</b> "
                                  f"({len(blob)/1024:.0f} KB)</p>"
                                  f"<p class='how-note'>state.json + memo.md + "
                                  f"charts/*.html</p></div>"))
            except Exception as e:
                display(wd.HTML(DASH_CSS +
                                  f"<div class='how-card'>"
                                  f"<p class='how-flag'>❌ ZIP fail: {e}</p>"
                                  f"<pre style='color:#8b949e;font-size:10px'>"
                                  f"{traceback.format_exc()}</pre></div>"))

    run_btn.on_click(_on_run)
    zip_btn.on_click(_on_zip)

    display(wd.VBox([
        wd.HTML(DASH_CSS +
                  f"<div class='how-root' style='padding:10px;'>"
                  f"<h2 style='color:#E8742C;margin:0;'>Howell Liquidity Framework "
                  f"{VERSION}</h2>"
                  f"<p style='color:#8b949e;margin:4px 0;font-size:12px;'>"
                  f"Two-agent BQuant system — 15 charts + phase classifier + "
                  f"sine-wave fit + Agent A/B memo</p></div>"),
        wd.HBox([years_w, run_btn, zip_btn]),
        out_main,
        out_zip,
    ]))


# ============================================================================
# ENTRYPOINT
# ============================================================================
if __name__ == '__main__' or True:
    try:
        display_dashboard()
    except NameError:
        # Nao estamos em Jupyter; rodar modo console
        log.info('Modo console (ipywidgets nao disponivel)')
        try:
            r = run_harvest(years=20)
            print(run_analyst(r['state']))
        except Exception as e:
            log.error(f'Console run fail: {e}')
            traceback.print_exc()
