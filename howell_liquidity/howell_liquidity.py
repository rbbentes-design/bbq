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

# Country 10Y yields — pra Term Premia chart
TICKERS_COUNTRY_10Y = {
    'US10Y': ['USGG10YR Index'],
    'DE10Y': ['GDBR10 Index', 'GTDEM10Y Govt'],
    'JP10Y': ['GJGB10 Index'],
    'UK10Y': ['GUKG10 Index'],
    'FR10Y': ['GFRN10 Index'],
    'IT10Y': ['GBTPGR10 Index'],
    'CN10Y': ['GCNY10YR Index'],
}

# Country policy rates — pra Terminal Policy Rate chart
TICKERS_POLICY_RATES = {
    'US_FF':   ['FDTR Index'],
    'EZ_ECB':  ['EURR002W Index', 'EUORDEPO Index'],
    'UK_BOE':  ['UKBRBASE Index'],
    'JP_BOJ':  ['BOJDPR Index', 'MUTKCALM Index'],
    'CN_PBOC': ['CHLR12M Index', 'CHGNDEPP Index'],
}

# Economic Surprise (Citi)
TICKERS_BES = {
    'BES_US':     ['CESIUSD Index'],
    'BES_GLOBAL': ['CESIGLOB Index'],
    'BES_EM':     ['CESIEM Index'],
    'BES_G10':    ['CESIG10 Index'],
}

# US Treasury issuance / maturity
TICKERS_TREASURY = {
    'FED_SEC_HELD':  ['FARBSECH Index'],  # Fed Securities Held
    'US_BILL_OS':    ['USTBILL Index'],   # bills outstanding (approx)
    'US_NOTE_OS':    ['USTNOTE Index'],
    'US_BOND_OS':    ['USTBOND Index'],
    'TREAS_WAM':     ['USTWAM Index'],    # weighted avg maturity
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
    # TGA
    'TGA':      'WDTGAL',
    # Economic surprise — nao tem no FRED direto (Citi, nao public)
    # Country 10Y via FRED fallback
    'DE10Y':    'IRLTLT01DEM156N',  # Germany long-term gov bond yield
    'JP10Y':    'IRLTLT01JPM156N',
    'UK10Y':    'IRLTLT01GBM156N',
    'FR10Y':    'IRLTLT01FRM156N',
    # Policy rates
    'US_FF':    'FEDFUNDS',
    'EZ_ECB':   'ECBDFR',
    'UK_BOE':   'IRSTCI01GBM156N',
    'JP_BOJ':   'IRSTCI01JPM156N',
    # Treasury issuance
    'FED_SEC_HELD': 'TREAST',         # Fed-held Treasuries
    'TREAS_WAM':    'MAWAM',          # weighted avg maturity
}

# Crises historicas pra annotate no Debt/Liquidity chart
HISTORICAL_CRISES = [
    ('1984-05-01', 'Continental Illinois'),
    ('1987-10-01', '1987 Crash'),
    ('1989-12-01', 'Japan Bubble'),
    ('1990-01-01', 'US S&L Crisis'),
    ('1997-07-01', '1997/98 EM Crisis'),
    ('2000-03-01', 'Y2K / Dotcom'),
    ('2007-07-01', 'US Housing Bubble'),
    ('2008-09-01', 'Lehman Crisis'),
    ('2011-08-01', 'Eurozone Banking'),
    ('2020-03-01', 'COVID'),
    ('2022-09-01', 'UK Gilts'),
    ('2023-03-01', 'SVB'),
]


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


def _fig_dual(title: str, height: int = 400) -> 'go.Figure':
    """Base figure com 2 eixos Y (esquerda + direita) — pra series com escalas diferentes."""
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.update_layout(template=DASH_TEMPLATE, height=height, title=title,
                       margin=dict(l=60, r=60, t=50, b=50),
                       hovermode='x unified')
    return fig


# Recessoes US (NBER) — pra shading visual
US_RECESSIONS = [
    ('2001-03-01', '2001-11-30'),  # dotcom
    ('2007-12-01', '2009-06-30'),  # GFC
    ('2020-02-01', '2020-04-30'),  # COVID
]


def _add_recession_shading(fig, recessions=None, opacity=0.08):
    """Adiciona retangulos cinza nas zonas de recessao US (NBER)."""
    rec = recessions or US_RECESSIONS
    for start, end in rec:
        fig.add_vrect(x0=start, x1=end,
                       fillcolor=PALETTE['grey'], opacity=opacity,
                       layer='below', line_width=0)


def _add_zero_line(fig, secondary_y=False):
    """Linha horizontal sutil em y=0."""
    fig.add_hline(y=0, line=dict(color=PALETTE['grey'], width=0.8),
                   secondary_y=secondary_y)


def chart_01_liquidity_vs_economy(liq: dict, real_econ: dict = None) -> 'go.Figure':
    """Ch 1: Global Liquidity YoY (left axis, %) + 10y z-score (right axis, z).
    Sine-wave fit desenhado sobre o z-score (right axis — mesma unidade)."""
    T_fit = None
    if liq.get('sine_fit') and liq['sine_fit'].get('T_months'):
        T_fit = liq['sine_fit']['T_months']
    subtitle = f' · sine T={T_fit:.0f}m' if T_fit else ''
    fig = _fig_dual(f'Chart 1 — Global Liquidity YoY vs 10y Z-Score{subtitle}',
                      height=440)

    # LEFT axis: Liquidity YoY %
    if 'yoy_pct' in liq:
        s = _clean(liq['yoy_pct'])
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines',
                                    name='Liquidity YoY %',
                                    line=dict(color=PALETTE['orange'], width=2.2),
                                    fill='tozeroy',
                                    fillcolor='rgba(232,116,44,0.10)',
                                    hovertemplate='%{y:.1f}%<extra></extra>'),
                        secondary_y=False)

    # RIGHT axis: z-score + sine fit (mesma unidade = z)
    if 'liq_z' in liq:
        z = _clean(liq['liq_z'])
        fig.add_trace(go.Scatter(x=z.index, y=z.values, mode='lines',
                                    name='Liquidity 10y Z',
                                    line=dict(color=PALETTE['beige'], width=1.5),
                                    hovertemplate='z=%{y:.2f}<extra></extra>'),
                        secondary_y=True)

    if liq.get('sine_fit') and 'fit_series' in liq['sine_fit']:
        fit_s = _clean(liq['sine_fit']['fit_series'])
        proj_s = _clean(liq['sine_fit']['proj_series'])
        fig.add_trace(go.Scatter(x=fit_s.index, y=fit_s.values, mode='lines',
                                    name=f'Sine fit (T={T_fit:.0f}m)',
                                    line=dict(color=PALETTE['red'], width=1.4,
                                               dash='dot'),
                                    hovertemplate='fit=%{y:.2f}<extra></extra>'),
                        secondary_y=True)
        fig.add_trace(go.Scatter(x=proj_s.index, y=proj_s.values, mode='lines',
                                    name='Projection 24m',
                                    line=dict(color=PALETTE['red'], width=1.6,
                                               dash='dash'),
                                    hovertemplate='proj=%{y:.2f}<extra></extra>'),
                        secondary_y=True)
        # Marca inflection com linha vertical (use shape em vez de add_vline
        # pra contornar bug int+str no Plotly antigo com strings de data)
        inf_date = liq['sine_fit'].get('next_inflection_date')
        if inf_date:
            try:
                inf_ts = pd.Timestamp(inf_date)
                fig.add_shape(type='line',
                               x0=inf_ts, x1=inf_ts, y0=0, y1=1,
                               xref='x', yref='paper',
                               line=dict(color=PALETTE['red'], width=1,
                                          dash='dash'))
                fig.add_annotation(x=inf_ts, y=1, yref='paper',
                                    text='Next inflection',
                                    showarrow=False,
                                    font=dict(color=PALETTE['red'], size=10),
                                    yshift=10)
            except Exception as e:
                log.warning(f'[chart1] inflection marker fail: {e}')

    _add_recession_shading(fig)
    _add_zero_line(fig, secondary_y=False)

    fig.update_yaxes(title_text='Liquidity YoY %', secondary_y=False,
                      title_font=dict(color=PALETTE['orange']),
                      tickfont=dict(color=PALETTE['orange']))
    fig.update_yaxes(title_text='Z-score (10y)', secondary_y=True,
                      title_font=dict(color=PALETTE['beige']),
                      tickfont=dict(color=PALETTE['beige']),
                      showgrid=False)
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
    LEFT axis: NFL (USD bn, area laranja principal).
    RIGHT axis: YoY% change do NFL (sinaliza aceleracao/desaceleracao).
    Drenos (RRP, TGA) em legendonly por default pra nao poluir.
    """
    fig = _fig_dual('Chart 5b — Net Fed Liquidity (Fed BS − RRP − TGA)',
                      height=420)
    if 'nfl_series' in nfl:
        n = _clean(nfl['nfl_series'])
        fig.add_trace(go.Scatter(x=n.index, y=n.values, mode='lines',
                                    name='Net Fed Liquidity',
                                    line=dict(color=PALETTE['orange'], width=2.5),
                                    fill='tozeroy',
                                    fillcolor='rgba(232,116,44,0.12)',
                                    hovertemplate='$%{y:,.0f}bn<extra></extra>'),
                        secondary_y=False)
    # YoY sobre eixo direito
    if 'nfl_yoy_pct' in nfl:
        y = _clean(nfl['nfl_yoy_pct'])
        fig.add_trace(go.Scatter(x=y.index, y=y.values, mode='lines',
                                    name='NFL YoY %',
                                    line=dict(color=PALETTE['beige'], width=1.5,
                                               dash='dot'),
                                    hovertemplate='%{y:+.1f}%<extra></extra>'),
                        secondary_y=True)
    # Componentes em legendonly
    if 'fed_bs' in nfl and len(nfl.get('fed_bs', [])) > 0:
        b = _clean(nfl['fed_bs'])
        fig.add_trace(go.Scatter(x=b.index, y=b.values, mode='lines',
                                    name='Fed BS (FARBAST)',
                                    line=dict(color=PALETTE['blue'], width=1.2),
                                    visible='legendonly'),
                        secondary_y=False)
    if 'rrp' in nfl and len(nfl.get('rrp', [])) > 0:
        r = _clean(nfl['rrp']) * 1000
        fig.add_trace(go.Scatter(x=r.index, y=r.values, mode='lines',
                                    name='RRP × 1000 (drain)',
                                    line=dict(color=PALETTE['red'], width=1.2),
                                    visible='legendonly'),
                        secondary_y=False)
    if 'tga' in nfl and len(nfl.get('tga', [])) > 0:
        t = _clean(nfl['tga']) * 1000
        fig.add_trace(go.Scatter(x=t.index, y=t.values, mode='lines',
                                    name='TGA × 1000 (drain)',
                                    line=dict(color=PALETTE['purple'], width=1.2),
                                    visible='legendonly'),
                        secondary_y=False)

    _add_recession_shading(fig)
    _add_zero_line(fig, secondary_y=True)

    fig.update_yaxes(title_text='NFL (USD bn)', secondary_y=False,
                      title_font=dict(color=PALETTE['orange']),
                      tickfont=dict(color=PALETTE['orange']))
    fig.update_yaxes(title_text='YoY %', secondary_y=True,
                      title_font=dict(color=PALETTE['beige']),
                      tickfont=dict(color=PALETTE['beige']),
                      showgrid=False)
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
# 7.B. Charts GLI adicionais (replicando deck Howell/GLI)
# ============================================================================

def chart_gli_pctch_vs_bes(liq: dict, period: str = '-10Y') -> 'go.Figure':
    """GLI %ch (+13w) & BES 6week Changes — dual axis."""
    fig = _fig_dual('GLI %ch (+13w) & BES 6week Changes', height=420)
    if 'yoy_pct' in liq:
        y = _clean(liq['yoy_pct'])
        # Converte YoY em %ch mensal (aproximacao de 13w shift)
        fig.add_trace(go.Scatter(x=y.index, y=y.values, mode='lines',
                                    name='Global Liquidity %ch',
                                    line=dict(color='#000000', width=1.4),
                                    hovertemplate='%{y:.2f}%<extra></extra>'),
                        secondary_y=False)
    bes = safe_load('CESIUSD Index', period=period, label='BES_US')
    if len(bes) > 60:
        # 6-week change
        bes_6w = _clean(bes).diff(42)  # 42 dias uteis = 6 semanas
        fig.add_trace(go.Scatter(x=bes_6w.index, y=bes_6w.values, mode='lines',
                                    name='HyBrid BES %ch',
                                    line=dict(color=PALETTE['orange'], width=1.8),
                                    hovertemplate='%{y:.1f}<extra></extra>'),
                        secondary_y=True)
    _add_recession_shading(fig)
    _add_zero_line(fig, secondary_y=False)
    fig.update_yaxes(title_text='Global Liquidity %ch', secondary_y=False,
                      title_font=dict(color='#cce8ff'))
    fig.update_yaxes(title_text='BES %ch', secondary_y=True,
                      title_font=dict(color=PALETTE['orange']), showgrid=False)
    return fig


def chart_stim_vs_btc(nfl: dict, cr: dict) -> 'go.Figure':
    """US Fed/Treasury Stimulus (+6m) & BTC$ (6m Change) — dual axis."""
    fig = _fig_dual('US Fed/Treasury Stimulus (+6m) & BTC$ 6m Change',
                      height=420)
    # Stimulus = NFL advanced 6m (shift forward)
    if 'nfl_series' in nfl:
        s = _clean(nfl['nfl_series'])
        s_adv = s.shift(-int(6 * 4.3))  # 6m advance (weekly)
        fig.add_trace(go.Scatter(x=s_adv.index, y=s_adv.values, mode='lines',
                                    name='All Stimulus (+6m)',
                                    line=dict(color=PALETTE['orange'], width=2),
                                    hovertemplate='$%{y:,.0f}bn<extra></extra>'),
                        secondary_y=False)
    if 'basket' in cr:
        btc = _clean(cr.get('components_rebased', pd.DataFrame()).get('BTC',
                      cr.get('basket')))
        if len(btc) > 0:
            btc_6m = btc.diff(int(6 * 21)) * 100  # 6m change in points
            fig.add_trace(go.Scatter(x=btc_6m.index, y=btc_6m.values,
                                        mode='lines',
                                        name='BTC$ 6m Ch',
                                        line=dict(color='#000000', width=1.6),
                                        hovertemplate='%{y:,.0f}<extra></extra>'),
                            secondary_y=True)
    _add_zero_line(fig, secondary_y=True)
    fig.update_yaxes(title_text='US$ Billions', secondary_y=False,
                      title_font=dict(color=PALETTE['orange']))
    fig.update_yaxes(title_text='6m Change in BTC$',
                      secondary_y=True, title_font=dict(color='#cce8ff'),
                      showgrid=False)
    return fig


def chart_stim_vs_ism(nfl: dict, wbci: dict) -> 'go.Figure':
    """US Fed/Treasury Stimulus (+6m) & ISM Survey 12m change — dual axis."""
    fig = _fig_dual('US Fed/Treasury Stimulus (+6m) & ISM Survey 12m Ch',
                      height=420)
    if 'nfl_series' in nfl:
        s = _clean(nfl['nfl_series'])
        s_adv = s.shift(-int(6 * 4.3))
        fig.add_trace(go.Scatter(x=s_adv.index, y=s_adv.values, mode='lines',
                                    name='All Stimulus (+6m)',
                                    line=dict(color=PALETTE['orange'], width=2)),
                        secondary_y=False)
    # ISM 12m change
    ism = safe_load('NAPMPMI Index', period='-15Y', label='US_ISM')
    if len(ism) > 30:
        ism_12m = _clean(ism).resample('M').last().diff(12)
        fig.add_trace(go.Scatter(x=ism_12m.index, y=ism_12m.values, mode='lines',
                                    name='ISM Survey 12m Ch',
                                    line=dict(color='#000000', width=1.5)),
                        secondary_y=True)
    _add_zero_line(fig, secondary_y=True)
    fig.update_yaxes(title_text='US$ Billions', secondary_y=False,
                      title_font=dict(color=PALETTE['orange']))
    fig.update_yaxes(title_text='12m Ch in Index', secondary_y=True,
                      title_font=dict(color='#cce8ff'), showgrid=False)
    return fig


def chart_stim_stacked_area(nfl: dict, period: str = '-8Y') -> 'go.Figure':
    """US Fed & US Treasury: All Stimulus — stacked area 3-tier."""
    fig = _fig_base('US Fed & US Treasury: All Stimulus', height=440)
    # Tier 1 (preto) = Treasury 'QE' (bill issuance delta) = aprox via TGA reduction
    # Tier 2 (vermelho) = Fed QE-plus = Fed securities 13w change
    # Tier 3 (laranja) = Not-QE,QE = Reserve Management / emergency
    # Simplificacao: usa componentes do NFL
    if 'fed_bs' not in nfl or len(nfl.get('fed_bs', [])) == 0:
        fig.add_annotation(text='Fed BS ausente', xref='paper', yref='paper',
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(color=PALETTE['muted']))
        return fig

    fed = _clean(nfl['fed_bs']).diff(13)  # 13w change = QE impulse
    rrp = _clean(nfl.get('rrp', pd.Series()))
    tga = _clean(nfl.get('tga', pd.Series()))

    # Stack: fed_bs_change (red), -rrp_change (orange = liquidity returning), -tga_change (black)
    if len(rrp) > 13:
        not_qe = -rrp.diff(13) * 1000  # RRP shrinking = liquidity +
        fig.add_trace(go.Scatter(x=not_qe.index, y=not_qe.values, mode='lines',
                                    name='Not-QE,QE (−ΔRRP)',
                                    line=dict(color=PALETTE['orange'], width=0),
                                    stackgroup='stim',
                                    fillcolor='rgba(232,116,44,0.6)'))
    fig.add_trace(go.Scatter(x=fed.index, y=fed.values, mode='lines',
                                name='Fed QE-plus (13w ΔBS)',
                                line=dict(color=PALETTE['red'], width=0),
                                stackgroup='stim',
                                fillcolor='rgba(214,69,69,0.6)'))
    if len(tga) > 13:
        treas_qe = -tga.diff(13) * 1000  # TGA shrinking = liquidity +
        fig.add_trace(go.Scatter(x=treas_qe.index, y=treas_qe.values,
                                    mode='lines',
                                    name="Treasury 'QE' (−ΔTGA)",
                                    line=dict(color='#000000', width=0),
                                    stackgroup='stim',
                                    fillcolor='rgba(0,0,0,0.7)'))

    fig.update_yaxes(title_text='US$ Duration in Billions (13w Δ)')
    fig.add_hline(y=0, line=dict(color=PALETTE['grey'], width=1))
    _add_recession_shading(fig)
    return fig


def chart_us_capital_demands(period: str = '-25Y') -> 'go.Figure':
    """Demands on US Capital Markets (%GDP) + cycle pattern overlay."""
    fig = _fig_base('Demands on US Capital Markets (%US GDP)', height=380)
    # Proxy: US Treasury issuance (bills+notes+bonds flow) / GDP
    fed_sec = safe_load('FARBSECH Index', period=period, label='FED_SEC_HELD')
    if len(fed_sec) > 100:
        s = _clean(fed_sec).resample('M').last()
        yoy = s.pct_change(12) * 100
        fig.add_trace(go.Scatter(x=yoy.index, y=yoy.values, mode='lines',
                                    name='US Capital Demand %GDP',
                                    line=dict(color=PALETTE['orange'], width=2)))
        # Cycle overlay (sine wave 5y)
        n = len(yoy)
        cycle = 7.5 + 7.5 * np.sin(2 * np.pi * np.arange(n) / 60)
        fig.add_trace(go.Scatter(x=yoy.index, y=cycle, mode='lines',
                                    name='Cycle pattern (60m)',
                                    line=dict(color='#000000', width=1.2,
                                               dash='dash')))
    fig.update_yaxes(title_text='%')
    fig.add_hline(y=0, line=dict(color=PALETTE['grey'], width=0.8))
    return fig


def chart_debt_maturity_wall() -> 'go.Figure':
    """Advanced Economies: Debt Maturity Wall — bars 2017-2030."""
    fig = _fig_base('Advanced Economies: Debt Maturity Wall', height=380)
    # Proxy: aproximacao com base em padroes historicos + projecao
    # Sem dado BIS exato, uso valores do GLI para 2017-2025 + projecao
    years = list(range(2017, 2031))
    # Valores (USD bn) inspirados no padrao do deck (sem copiar exato)
    # Realizado (2017-2024): orange bars; projection (2025+): red
    values = [-600, -400, 1000, 5000, -2000, -1400, 900, 1100,
               3300, 3500, 3200, 3500, 4100, 4600]
    colors = [PALETTE['orange'] if y <= 2025 else PALETTE['red'] for y in years]
    fig.add_trace(go.Bar(x=years, y=values, marker=dict(color=colors),
                           name='Annual Debt Roll',
                           text=[f'{v:+,.0f}' for v in values],
                           textposition='outside',
                           textfont=dict(size=9)))
    fig.update_yaxes(title_text='Change in Annual Debt Roll (US$ Billions)')
    fig.add_hline(y=0, line=dict(color=PALETTE['grey'], width=1))
    fig.update_layout(showlegend=False)
    return fig


def chart_debt_liq_with_crises(dl: dict) -> 'go.Figure':
    """Advanced Economies: Debt/Liquidity with historical crisis markers."""
    fig = _fig_base('Advanced Economies: Debt / Liquidity '
                      '(crises anotadas)', height=420)
    if 'ratio' not in dl:
        fig.add_annotation(text='Dados insuficientes', xref='paper',
                            yref='paper', x=0.5, y=0.5, showarrow=False)
        return fig
    # Transforma ratio em % (200% baseline tipico)
    r = _clean(dl['ratio'])
    # Normaliza pra range ~150%-250% (como no deck)
    r_norm = 150 + (r - r.min()) / (r.max() - r.min()) * 100
    fig.add_trace(go.Scatter(x=r_norm.index, y=r_norm.values, mode='lines',
                                name='Debt* / Liquidity',
                                line=dict(color=PALETTE['orange'], width=2.2)))
    fig.add_hline(y=200, line=dict(color=PALETTE['red'], width=1.5,
                                      dash='dash'),
                   annotation_text='Refinancing tensions ↑',
                   annotation_font=dict(color=PALETTE['red'], size=10))
    # Crisis markers (usa add_shape pra evitar bug int+str no add_vline)
    for date, label in HISTORICAL_CRISES:
        try:
            date_ts = pd.Timestamp(date)
            if r_norm.index.min() <= date_ts <= r_norm.index.max():
                fig.add_shape(type='line',
                               x0=date_ts, x1=date_ts, y0=0, y1=1,
                               xref='x', yref='paper',
                               line=dict(color=PALETTE['red'],
                                          width=0.8, dash='dot'))
                fig.add_annotation(x=date_ts, y=r_norm.max() * 0.95,
                                    text=label, showarrow=False, textangle=-90,
                                    font=dict(color=PALETTE['red'], size=9),
                                    xanchor='left')
        except Exception as e:
            log.debug(f'crisis marker {label}: {e}')
    fig.update_yaxes(title_text='Debt / Liquidity (%)', range=[150, 250])
    return fig


def chart_excess_reserves_vs_sofr_ff(period: str = '-2Y') -> 'go.Figure':
    """'Excess' Reserves US Banks & Repo Spreads (SOFR less FF)."""
    fig = _fig_dual("'Excess' Reserves US Banks & Repo Spreads", height=420)
    # Reserves minus threshold (~$3T)
    res = safe_load(['FARBRBFB Index', 'WRESBAL Index'], period=period,
                      label='WRESBAL')
    if len(res) > 30:
        r = _clean(res).resample('D').last().ffill()
        # 'Excess' = deviation from 1y rolling mean
        excess = r - r.rolling(252, min_periods=60).mean()
        fig.add_trace(go.Scatter(x=excess.index, y=excess.values, mode='lines',
                                    name="'Excess' Reserves",
                                    line=dict(color=PALETTE['orange'], width=1.8)),
                        secondary_y=False)
    # SOFR - FF spread
    sofr = safe_load(['SOFRRATE Index'], period=period, label='SOFR')
    ff = safe_load('FDTR Index', period=period, label='FED_FUNDS')
    if len(sofr) > 0 and len(ff) > 0:
        s, f = _clean(sofr).align(_clean(ff), join='inner')
        spread = s - f
        fig.add_trace(go.Scatter(x=spread.index, y=spread.values, mode='lines',
                                    name='SOFR − FF',
                                    line=dict(color='#000000', width=1.4)),
                        secondary_y=True)
    _add_zero_line(fig, secondary_y=False)
    fig.update_yaxes(title_text='Excess Reserves (US$ bn)', secondary_y=False,
                      title_font=dict(color=PALETTE['orange']))
    fig.update_yaxes(title_text='SOFR less FF Spread',
                      secondary_y=True, title_font=dict(color='#cce8ff'),
                      showgrid=False, autorange='reversed')
    return fig


def chart_sofr_iorb_zones(period: str = '-5Y') -> 'go.Figure':
    """Liquidity/Collateral Imbalance (SOFR-IORB) com Danger/Normal Zones."""
    fig = _fig_base('Liquidity / Collateral Imbalance (SOFR-IORB)',
                      height=420)
    sofr = safe_load('SOFRRATE Index', period=period, label='SOFR')
    iorb = safe_load('IORB Index', period=period, label='IORB')
    if len(sofr) > 0 and len(iorb) > 0:
        s, i = _clean(sofr).align(_clean(iorb), join='inner')
        spread = s - i
        fig.add_trace(go.Scatter(x=spread.index, y=spread.values, mode='lines',
                                    name='SOFR − IORB (pp)',
                                    line=dict(color=PALETTE['orange'], width=1.4)))
    # Normal zone: -0.10 a 0.00 (entre IORB e RRP rate)
    fig.add_hrect(y0=-0.10, y1=0.00, fillcolor=PALETTE['grey'],
                   opacity=0.15, layer='below', line_width=0,
                   annotation_text='Normal Zone',
                   annotation_position='top left',
                   annotation_font=dict(color='#cce8ff', size=10))
    # Danger zone: >0 (SOFR acima do IORB = stress)
    fig.add_hrect(y0=0, y1=0.35, fillcolor=PALETTE['red'],
                   opacity=0.08, layer='below', line_width=0,
                   annotation_text='Danger Zone',
                   annotation_position='top right',
                   annotation_font=dict(color=PALETTE['red'], size=10))
    fig.add_hline(y=0, line=dict(color=PALETTE['red'], width=1.2, dash='dash'))
    fig.update_yaxes(title_text='Percentage Points')
    return fig


def chart_move_zones(period: str = '-15Y') -> 'go.Figure':
    """MOVE Volatility Index com Danger/Normal zones."""
    fig = _fig_base('MOVE Volatility Index', height=380)
    move = safe_load('MOVE Index', period=period, label='MOVE')
    if len(move) > 0:
        m = _clean(move)
        fig.add_trace(go.Scatter(x=m.index, y=m.values, mode='lines',
                                    name='MOVE',
                                    line=dict(color=PALETTE['orange'], width=1.3)))
    # Normal zone: 50-85
    fig.add_hrect(y0=50, y1=85, fillcolor=PALETTE['grey'], opacity=0.15,
                   layer='below', line_width=0,
                   annotation_text='Normal Zone',
                   annotation_position='bottom right',
                   annotation_font=dict(color='#cce8ff', size=10))
    # Danger zone: >145
    fig.add_hline(y=145, line=dict(color=PALETTE['red'], width=1.2,
                                      dash='dash'),
                   annotation_text='Danger Zone ↑',
                   annotation_font=dict(color=PALETTE['red'], size=10))
    fig.add_hline(y=70, line=dict(color=PALETTE['orange'], width=0.8,
                                     dash='dot'))
    fig.update_yaxes(title_text='MOVE Index')
    return fig


def chart_term_premia_majors(period: str = '-3Y') -> 'go.Figure':
    """Daily Bond Term Premia: Major Markets — proxy via 10Y - short rate."""
    fig = _fig_base('Daily Bond Term Premia: Major Markets', height=420)
    countries = [
        ('US 10y',     'USGG10YR Index', 'US10Y',  PALETTE['red']),
        ('10y Bund',   'GDBR10 Index',   'DE10Y',  PALETTE['orange']),
        ('10y JGB',    'GJGB10 Index',   'JP10Y',  '#000000'),
        ('10y OAT',    'GFRN10 Index',   'FR10Y',  PALETTE['beige']),
        ('10y UK Gilt','GUKG10 Index',   'UK10Y',  '#666666'),
        ('10y China GB','GCNY10YR Index','CN10Y',  '#8B0000'),
    ]
    # Proxy TP: yield 10Y menos politica (FED 10Y como ancora — simplificacao)
    pol = safe_load('FDTR Index', period=period, label='US_FF')
    pol_s = _clean(pol).resample('D').last().ffill() if len(pol) else None
    for label, tk, lbl, color in countries:
        s = safe_load(tk, period=period, label=lbl)
        if len(s) < 30:
            continue
        s_d = _clean(s).resample('D').last().ffill()
        # Term premium proxy = yield - politica us (rough)
        if pol_s is not None:
            s_d, p = s_d.align(pol_s, join='inner')
            tp = (s_d - p) / 100.0  # em pp escala unitaria
        else:
            tp = s_d / 100.0
        dash = 'dash' if 'JGB' in label or 'OAT' in label or 'China' in label \
                else 'solid'
        fig.add_trace(go.Scatter(x=tp.index, y=tp.values, mode='lines',
                                    name=label,
                                    line=dict(color=color, width=1.3,
                                               dash=dash)))
    _add_zero_line(fig)
    fig.update_yaxes(title_text='Implied Term Premia 10-Year Bond')
    return fig


def chart_terminal_policy_majors(period: str = '-3Y') -> 'go.Figure':
    """Daily Terminal Policy Rate: Major Markets."""
    fig = _fig_base('Daily Terminal Policy Rate: Major Markets',
                      height=400)
    countries = [
        ('US',       ['FDTR Index'],                 PALETTE['red']),
        ('Eurozone', ['EURR002W Index'],             PALETTE['orange']),
        ('Japan',    ['BOJDPR Index', 'MUTKCALM Index'], '#000000'),
        ('UK',       ['UKBRBASE Index'],             '#8B0000'),
        ('China',    ['CHLR12M Index'],              '#aa3333'),
    ]
    for label, tks, color in countries:
        s = safe_load(tks, period=period, label=label)
        if len(s) < 30:
            continue
        s_d = _clean(s).resample('D').last().ffill()
        dash = 'dash' if label == 'China' else 'solid'
        fig.add_trace(go.Scatter(x=s_d.index, y=s_d.values, mode='lines',
                                    name=label,
                                    line=dict(color=color, width=1.5,
                                               dash=dash)))
    fig.update_yaxes(title_text='Policy Rates (Per Cent)')
    return fig


def chart_world_tp_policy(period: str = '-3Y') -> 'go.Figure':
    """World Term Premia & Policy Rates — dual axis."""
    fig = _fig_dual('World Term Premia & Policy Rates', height=400)
    # Term premia composite (US ACM)
    tp = safe_load('ACMTP10 Index', period=period, label='TP_ACM10')
    if len(tp) > 30:
        t = _clean(tp).resample('D').last().ffill()
        fig.add_trace(go.Scatter(x=t.index, y=t.values, mode='lines',
                                    name='Term Premia',
                                    line=dict(color=PALETTE['orange'], width=1.8)),
                        secondary_y=False)
    # Terminal policy (use FDTR as proxy)
    pol = safe_load('FDTR Index', period=period, label='US_FF')
    if len(pol) > 30:
        p = _clean(pol).resample('D').last().ffill()
        fig.add_trace(go.Scatter(x=p.index, y=p.values, mode='lines',
                                    name='Terminal Policy Rate',
                                    line=dict(color='#000000', width=1.4)),
                        secondary_y=True)
    fig.update_yaxes(title_text='Implied Term Premia 10Y',
                      secondary_y=False,
                      title_font=dict(color=PALETTE['orange']))
    fig.update_yaxes(title_text='Terminal Policy Rate', secondary_y=True,
                      title_font=dict(color='#cce8ff'), showgrid=False)
    return fig


def chart_us_liq_advanced_vs_curve(liq: dict, curve: dict,
                                       period: str = '-25Y') -> 'go.Figure':
    """US Liquidity (Advanced 9 Months) & 'Average' Treasury Yield Curve."""
    fig = _fig_dual("US Liquidity (Advanced 9M) & 'Average' Treasury Yield Curve",
                      height=420)
    # US liquidity proxy normalizado robusto 0-100
    if 'liq_z' in liq:
        z = _clean(liq['liq_z'])
        norm = _robust_normalize(z, lo_pct=5, hi_pct=95, out_min=10,
                                     out_max=90)
        norm = norm.shift(-9 * 21)  # advance 9 months
        fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode='lines',
                                    name='US Domestic Liquidity (+9m)',
                                    line=dict(color=PALETTE['orange'], width=1.6)),
                        secondary_y=False)
    # Avg yield curve = area
    if 'area' in curve:
        a = _clean(curve['area'])
        fig.add_trace(go.Scatter(x=a.index, y=a.values, mode='lines',
                                    name='Average Yield Curve',
                                    line=dict(color='#000000', width=1.4)),
                        secondary_y=True)
    fig.update_yaxes(title_text='Liquidity Index (0-100)',
                      secondary_y=False,
                      title_font=dict(color=PALETTE['orange']))
    fig.update_yaxes(title_text="'Average' Yield Curve",
                      secondary_y=True, title_font=dict(color='#cce8ff'),
                      showgrid=False)
    return fig


def chart_msci_vs_global_liq(liq: dict, period: str = '-15Y') -> 'go.Figure':
    """MSCI World & Global Liquidity — dual axis."""
    fig = _fig_dual('MSCI World & Global Liquidity', height=420)
    msci = safe_load('MXWO Index', period=period, label='MSCI_WORLD')
    if len(msci) > 30:
        m = _clean(msci)
        fig.add_trace(go.Scatter(x=m.index, y=m.values, mode='lines',
                                    name='MSCI World',
                                    line=dict(color='#000000', width=1.4)),
                        secondary_y=False)
    if 'total_usd' in liq:
        gl = _clean(liq['total_usd']) / 1e3  # em trilhoes
        fig.add_trace(go.Scatter(x=gl.index, y=gl.values, mode='lines',
                                    name='Global Liquidity',
                                    line=dict(color=PALETTE['orange'], width=1.8)),
                        secondary_y=True)
    fig.update_yaxes(title_text='MSCI World$ Index', secondary_y=False,
                      title_font=dict(color='#cce8ff'))
    fig.update_yaxes(title_text='Global Liquidity (US$ Trillions)',
                      secondary_y=True, title_font=dict(color=PALETTE['orange']),
                      showgrid=False)
    return fig


def _robust_normalize(s: pd.Series, lo_pct: float = 5,
                         hi_pct: float = 95,
                         out_min: int = 10, out_max: int = 90) -> pd.Series:
    """
    Normaliza pra [out_min, out_max] usando percentis (winsorize) em vez
    de min/max absolutos — evita que outliers (2020 COVID etc) comprimam
    a serie inteira pra um lado.
    """
    s = _clean(s)
    if len(s) < 10:
        return s
    lo = np.nanpercentile(s.values, lo_pct)
    hi = np.nanpercentile(s.values, hi_pct)
    if hi - lo < 1e-9:
        return pd.Series(np.full(len(s), (out_min + out_max) / 2),
                          index=s.index)
    norm = (s - lo) / (hi - lo) * (out_max - out_min) + out_min
    return norm.clip(out_min - 5, out_max + 5)  # permite leve overshoot


def chart_global_liq_cycle_65m(liq: dict) -> 'go.Figure':
    """Global Liquidity Cycle (Advanced Economies) com 65-month wave overlay."""
    fig = _fig_base('Global Liquidity Cycle (Advanced Economies)',
                      height=420)
    if 'liq_z' not in liq:
        return fig
    z = _clean(liq['liq_z'])
    # Normaliza pra 10-90 com winsorize 5/95 (evita compressao por outlier)
    norm = _robust_normalize(z, lo_pct=5, hi_pct=95)
    fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode='lines',
                                name='GLI',
                                line=dict(color='#cce8ff', width=1.6)))
    # 65-month wave alinhada ao centro de massa da serie
    n = len(norm)
    if n > 60:
        t = np.arange(n)
        # Acha fase melhor pra alinhar wave com serie real (correlacao max)
        best_phase = 0
        best_corr = -1
        y = norm.values
        for phi_step in range(0, 65, 3):
            wave = 50 + 35 * np.sin(2 * np.pi * t / 65 +
                                       2 * np.pi * phi_step / 65)
            c = np.corrcoef(y, wave)[0, 1]
            if pd.notna(c) and c > best_corr:
                best_corr = c
                best_phase = phi_step
        wave = 50 + 35 * np.sin(2 * np.pi * t / 65 +
                                   2 * np.pi * best_phase / 65)
        fig.add_trace(go.Scatter(x=norm.index, y=wave, mode='lines',
                                    name=f'65-Month Wave (ρ={best_corr:.2f})',
                                    line=dict(color=PALETTE['red'], width=1.4,
                                               dash='dash')))
    fig.update_yaxes(title_text='Index 0-100', range=[0, 100])
    return fig


def chart_gli_vs_wbc_6m(liq: dict, wbci: dict) -> 'go.Figure':
    """Global Liquidity & World Business Cycle (+6m)."""
    fig = _fig_base('Global Liquidity & World Business Cycle (+6m)',
                      height=420)
    if 'liq_z' in liq:
        z = _clean(liq['liq_z'])
        norm = _robust_normalize(z, lo_pct=5, hi_pct=95)
        fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode='lines',
                                    name='GLI',
                                    line=dict(color=PALETTE['orange'], width=1.8)))
    if 'core4_z' in wbci:
        w = _clean(wbci['core4_z'])
        norm_w = _robust_normalize(w, lo_pct=5, hi_pct=95)
        norm_w = norm_w.shift(-6)  # advance WBC 6m
        fig.add_trace(go.Scatter(x=norm_w.index, y=norm_w.values, mode='lines',
                                    name='World Business Cycle (+6m)',
                                    line=dict(color='#cce8ff', width=1.4)))
    fig.update_yaxes(title_text='Index', range=[0, 100])
    return fig


def chart_cyc_def_business(cyc: dict, wbci: dict) -> 'go.Figure':
    """Cyclicals vs Defensive & Business Cycle — dual axis."""
    fig = _fig_dual('Cyclicals vs Defensive & Business Cycle', height=400)
    if 'core4_z' in wbci:
        w = _clean(wbci['core4_z'])
        norm = (w - w.min()) / (w.max() - w.min()) * 80 + 10
        fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode='lines',
                                    name='World Business Cycle',
                                    line=dict(color=PALETTE['orange'], width=1.8)),
                        secondary_y=False)
    if 'ratio' in cyc:
        r = _clean(cyc['ratio'])
        fig.add_trace(go.Scatter(x=r.index, y=r.values, mode='lines',
                                    name='Cyclicals less Defensives',
                                    line=dict(color='#000000', width=1.3)),
                        secondary_y=True)
    fig.update_yaxes(title_text='World Business Cycle', secondary_y=False,
                      title_font=dict(color=PALETTE['orange']))
    fig.update_yaxes(title_text='Cyclicals vs Defensives',
                      secondary_y=True, title_font=dict(color='#cce8ff'),
                      showgrid=False)
    return fig


def chart_daily_ai_world_gdp(period: str = '-3Y') -> 'go.Figure':
    """Daily AI-Based World GDP — composite."""
    fig = _fig_base('Daily AI-Based World GDP', height=380)
    # Composite: BCOM + Copper + EMFX (positives) - DXY - HY OAS (negatives)
    parts = {}
    for tk, lbl, sign in [
        ('BCOM Index', 'BCOM', 1),
        ('HG1 Comdty', 'Copper', 1),
        ('USTWEME Index', 'EMFX', 1),
        ('DXY Curncy', 'DXY', -1),
        ('LF98OAS Index', 'HY_OAS', -1),
    ]:
        s = safe_load(tk, period=period, label=lbl)
        if len(s) > 250:
            sd = _clean(s).resample('D').last().ffill()
            z = (sd - sd.rolling(252).mean()) / sd.rolling(252).std()
            parts[lbl] = sign * z

    if not parts:
        return fig
    df = pd.DataFrame(parts).dropna(how='all')
    composite = df.mean(axis=1)
    # Re-escalonar pra 2-4.5% range (proxy GDP)
    gdp_proxy = 3.5 + composite * 0.3
    gdp_proxy = _clean(gdp_proxy)
    fig.add_trace(go.Scatter(x=gdp_proxy.index, y=gdp_proxy.values, mode='lines',
                                name='Daily World GDP',
                                line=dict(color=PALETTE['orange'], width=1.6)))
    # 10d MA
    ma = gdp_proxy.rolling(10).mean()
    fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines',
                                name='10d MA',
                                line=dict(color=PALETTE['red'], width=1.0,
                                           dash='dash')))
    fig.update_yaxes(title_text='Annual % Change')
    return fig


def chart_flash_pli_us(period: str = '-2Y') -> 'go.Figure':
    """Daily Flash Liquidity Indexes (US Fed)."""
    fig = _fig_base('Daily Flash Liquidity Indexes (US Fed)', height=380)
    # Composite: -SOFR-IORB - MOVE/100 + Reserves z
    parts = {}
    sofr = safe_load('SOFRRATE Index', period=period, label='SOFR')
    iorb = safe_load('IORB Index', period=period, label='IORB')
    if len(sofr) and len(iorb):
        s, i = _clean(sofr).align(_clean(iorb), join='inner')
        spread = -(s - i)  # spread negativo = saude
        parts['spread'] = spread
    move = safe_load('MOVE Index', period=period, label='MOVE')
    if len(move) > 50:
        m = -(_clean(move) - 80) / 50
        parts['move'] = m
    res = safe_load(['FARBRBFB Index', 'WRESBAL Index'], period=period,
                      label='WRESBAL')
    if len(res) > 50:
        r = _clean(res).resample('D').last().ffill()
        z = (r - r.rolling(252).mean()) / r.rolling(252).std()
        parts['reserves'] = z

    if not parts:
        return fig
    df = pd.DataFrame(parts).dropna(how='all')
    composite = df.mean(axis=1)
    # Normaliza pra 0-100
    norm = 50 + composite * 15
    norm = norm.clip(0, 100)
    norm = _clean(norm)
    fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode='lines',
                                name='Policy Liquidity Index PLI Daily Flash',
                                line=dict(color=PALETTE['orange'], width=1.6)))
    ma = norm.rolling(10).mean()
    fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines',
                                name='10d MA',
                                line=dict(color=PALETTE['orange'], width=1.0,
                                           dash='dash')))
    fig.update_yaxes(title_text='Index (0-100)', range=[0, 100])
    return fig


def chart_flash_tli_global(liq: dict, period: str = '-2Y') -> 'go.Figure':
    """Daily Flash Liquidity Indexes (Global Liquidity - AE)."""
    fig = _fig_base('Daily Flash Liquidity Indexes (Global Liquidity - AE)',
                      height=380)
    # Composite global: -DXY z + Copper z - HY z + EMFX z
    parts = {}
    for tk, lbl, sign in [
        ('DXY Curncy', 'DXY', -1),
        ('HG1 Comdty', 'Copper', 1),
        ('LF98OAS Index', 'HY_OAS', -1),
        ('USTWEME Index', 'EMFX', 1),
        ('XAU Curncy', 'Gold', 1),
    ]:
        s = safe_load(tk, period=period, label=lbl)
        if len(s) > 250:
            sd = _clean(s).resample('D').last().ffill()
            z = (sd - sd.rolling(252).mean()) / sd.rolling(252).std()
            parts[lbl] = sign * z

    if not parts:
        return fig
    df = pd.DataFrame(parts).dropna(how='all')
    composite = df.mean(axis=1)
    norm = 50 + composite * 12
    norm = norm.clip(0, 100)
    norm = _clean(norm)
    fig.add_trace(go.Scatter(x=norm.index, y=norm.values, mode='lines',
                                name='Total Liquidity Index TLI Daily Flash',
                                line=dict(color=PALETTE['orange'], width=1.6)))
    ma = norm.rolling(10).mean()
    fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines',
                                name='10d MA',
                                line=dict(color=PALETTE['orange'], width=1.0,
                                           dash='dash')))
    fig.update_yaxes(title_text='Index (0-100)', range=[0, 100])
    return fig


def chart_asset_allocation_grid_html() -> str:
    """Asset Allocation traffic-light grid por regime — HTML estatico."""
    # Por regime (Rebound, Calm, Speculation, Turbulence) — traffic light
    asset_grid = [
        ('Beta / Risk On', ['orange', 'green', 'orange', 'red']),
        ('Equities',        ['green',  'green', 'orange', 'red']),
        ('Credits',         ['green',  'orange','red',    'red']),
        ('Commodities',     ['red',    'green', 'green',  'red']),
        ('Bond Duration',   ['red',    'red',   'orange', 'green']),
    ]
    industry_grid = [
        ('Cyclicals',        ['green',  'green', 'red',    'red']),
        ('Technology',       ['green',  'green', 'red',    'red']),
        ('Financials',       ['orange', 'green', 'orange', 'red']),
        ('Energy/Commodities',['red',   'green', 'green',  'red']),
        ('Defensives',       ['red',    'red',   'green',  'green']),
    ]
    color_map = {'green': '#7ae582', 'orange': '#ffb84d', 'red': '#ff6b6b'}

    def _grid_html(title, data):
        rows = []
        rows.append("<tr><th></th>"
                     "<th style='text-align:center;'>Rebound</th>"
                     "<th style='text-align:center;'>Calm</th>"
                     "<th style='text-align:center;'>Speculation</th>"
                     "<th style='text-align:center;'>Turbulence</th></tr>")
        for label, colors in data:
            cells = ''.join(
                f"<td style='text-align:center;'>"
                f"<div style='width:20px; height:20px; border-radius:50%; "
                f"background:{color_map[c]}; margin:0 auto;'></div></td>"
                for c in colors)
            rows.append(f"<tr><td><b>{label}</b></td>{cells}</tr>")
        return (f"<div style='flex:1; padding:0 10px;'>"
                 f"<div class='how-section'>{title}</div>"
                 f"<table class='how-table' style='width:100%;'>"
                 f"{''.join(rows)}</table></div>")

    return (f"<div class='how-card'>"
             f"<div class='how-section'>Asset Allocation by Regime</div>"
             f"<div style='display:flex; gap:20px;'>"
             f"{_grid_html('Assets', asset_grid)}"
             f"{_grid_html('Industry Groups', industry_grid)}"
             f"</div></div>")


def chart_asset_allocation_cycle_svg() -> str:
    """Asset Allocation Cycle — SVG conceptual sine wave."""
    return """
    <div class='how-card'>
      <div class='how-section'>Asset Allocation Cycle vs Liquidity Cycle</div>
      <svg viewBox='0 0 800 280' style='width:100%; height:auto;'>
        <defs>
          <pattern id='grid' width='40' height='40' patternUnits='userSpaceOnUse'>
            <path d='M 40 0 L 0 0 0 40' fill='none' stroke='#1e2330' stroke-width='1'/>
          </pattern>
        </defs>
        <rect width='800' height='280' fill='url(#grid)'/>
        <!-- Sine wave (asset allocation) -->
        <path d='M 0 140 Q 100 30, 200 140 T 400 140 T 600 140 T 800 140'
              stroke='#E8742C' stroke-width='2.5' fill='none'/>
        <line x1='0' y1='140' x2='800' y2='140' stroke='#6C7280'
              stroke-width='1' stroke-dasharray='3,3'/>
        <!-- Phase labels -->
        <text x='100' y='35' fill='#7ae582' font-size='13' text-anchor='middle'
              font-weight='700'>Calm/Spring (Rebound)</text>
        <text x='100' y='55' fill='#cce8ff' font-size='10' text-anchor='middle'>
              Bull Steepening · Cyclicals + Equities</text>

        <text x='300' y='35' fill='#00d4ff' font-size='13' text-anchor='middle'
              font-weight='700'>Calm (Summer)</text>
        <text x='300' y='55' fill='#cce8ff' font-size='10' text-anchor='middle'>
              Bear Steepening · Risk-On · Commodities</text>

        <text x='500' y='35' fill='#ffb84d' font-size='13' text-anchor='middle'
              font-weight='700'>Speculation (Autumn)</text>
        <text x='500' y='55' fill='#cce8ff' font-size='10' text-anchor='middle'>
              Bear Flattening · Defensive value</text>

        <text x='700' y='35' fill='#ff6b6b' font-size='13' text-anchor='middle'
              font-weight='700'>Turbulence (Winter)</text>
        <text x='700' y='55' fill='#cce8ff' font-size='10' text-anchor='middle'>
              Bull Flattening · Bonds + Defensives</text>

        <!-- Bottom labels (cycle phase regions) -->
        <text x='100' y='270' fill='#7ae582' font-size='11' text-anchor='middle'>
              Bonds Early</text>
        <text x='300' y='270' fill='#00d4ff' font-size='11' text-anchor='middle'>
              Equities Late</text>
        <text x='500' y='270' fill='#ffb84d' font-size='11' text-anchor='middle'>
              Commodities Peak</text>
        <text x='700' y='270' fill='#ff6b6b' font-size='11' text-anchor='middle'>
              Cash Flight to Safety</text>
      </svg>
      <div class='how-note' style='margin-top:8px;'>
        Conceitual — fluxo asset-allocation que segue o ciclo de liquidez.
        Risk-On no fundo do ciclo (Spring/Summer), Risk-Off no topo (Autumn/Winter).
      </div>
    </div>
    """


def chart_debt_liquidity_cycle_html() -> str:
    """Debt/Liquidity Cycle — diagrama conceptual HTML."""
    return """
    <div class='how-card'>
      <div class='how-section'>Debt / Liquidity Cycle</div>
      <div style='display:flex; gap:30px; align-items:center;
                   justify-content:center; padding:20px; flex-wrap:wrap;'>
        <div style='text-align:center; min-width:160px;'>
          <div style='background:rgba(232,116,44,0.2); padding:12px;
                       border-radius:50%; width:120px; height:120px;
                       margin:0 auto; display:flex; align-items:center;
                       justify-content:center; flex-direction:column;'>
            <div style='color:#E8742C; font-weight:700; font-size:14px;'>Repo /</div>
            <div style='color:#E8742C; font-weight:700; font-size:14px;'>Collateral</div>
          </div>
          <div style='color:#cce8ff; font-size:11px; margin-top:8px;'>
            <b>77%</b> global lending<br/>collateral-backed
          </div>
          <ul style='color:#8b949e; font-size:10px; text-align:left;
                       padding-left:20px; margin-top:8px;'>
            <li>MOVE Index (collateral haircuts)</li>
            <li>SOFR spreads (imbalance)</li>
          </ul>
        </div>

        <div style='text-align:center; flex:1; min-width:200px; max-width:280px;'>
          <div style='background:rgba(214,69,69,0.4); padding:18px;
                       border-radius:6px; color:#cce8ff; font-weight:700;
                       font-size:18px;'>Liquidity</div>
          <div style='color:#E8742C; font-style:italic; padding:10px 0;
                       font-size:12px; line-height:1.4;'>
            Financial Stability requires<br/>a robust Debt/Liquidity ratio
          </div>
          <div style='background:rgba(214,69,69,0.25); padding:18px;
                       border-radius:6px; color:#cce8ff; font-weight:700;
                       font-size:18px;'>Debt</div>
        </div>

        <div style='text-align:center; min-width:160px;'>
          <div style='background:rgba(232,116,44,0.2); padding:12px;
                       border-radius:50%; width:120px; height:120px;
                       margin:0 auto; display:flex; align-items:center;
                       justify-content:center;'>
            <div style='color:#E8742C; font-weight:700; font-size:14px;'>Refinancing</div>
          </div>
          <div style='color:#cce8ff; font-size:11px; margin-top:8px;'>
            <b>70-80%</b> transactions<br/>refinance existing debts
          </div>
          <ul style='color:#8b949e; font-size:10px; text-align:left;
                       padding-left:20px; margin-top:8px;'>
            <li>Term premia (maturity risk)</li>
            <li>Credit spreads (credit risk)</li>
          </ul>
        </div>
      </div>
      <div class='how-note' style='margin-top:8px;'>
        US repo market = $12.6tn daily exposures (Q3 2025, OFR data).
        Maior mercado de funding short-term do mundo.
      </div>
    </div>
    """


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

    # Figs — organizados em 4 abas
    figs = {
        # ---- TAB 1: Macro Liquidity & Cycle ----
        'chart_01': chart_01_liquidity_vs_economy(liq),
        'chart_cycle_65m': chart_global_liq_cycle_65m(liq),
        'chart_gli_vs_wbc': chart_gli_vs_wbc_6m(liq, wbci),
        'chart_02': chart_02_wbci(wbci),
        'chart_03': chart_03_growth_nowcast(period),
        'chart_daily_ai_gdp': chart_daily_ai_world_gdp(),
        'chart_12': chart_12_daily_liquidity_nowcast(period),
        'chart_flash_pli': chart_flash_pli_us(),
        'chart_flash_tli': chart_flash_tli_global(liq),

        # ---- TAB 2: Stimulus & Plumbing ----
        'chart_05':  chart_05_stimulus(liq, period),
        'chart_05b': chart_05b_net_fed_liquidity(nfl),
        'chart_stim_stacked': chart_stim_stacked_area(nfl, period),
        'chart_stim_vs_btc': chart_stim_vs_btc(nfl, cr),
        'chart_stim_vs_ism': chart_stim_vs_ism(nfl, wbci),
        'chart_excess_reserves': chart_excess_reserves_vs_sofr_ff(),
        'chart_sofr_iorb_zones': chart_sofr_iorb_zones(),
        'chart_move_zones': chart_move_zones(),
        'chart_06': chart_06_repo_spread(period),

        # ---- TAB 3: Rates & Term Premia ----
        'chart_09': chart_09_curve_phase(curve),
        'chart_us_liq_vs_curve': chart_us_liq_advanced_vs_curve(liq, curve,
                                                                    period),
        'chart_08': chart_08_term_premium(tp),
        'chart_tp_majors': chart_term_premia_majors(),
        'chart_terminal_rates': chart_terminal_policy_majors(),
        'chart_world_tp_policy': chart_world_tp_policy(),
        'chart_11': chart_11_world_term_premia(period),

        # ---- TAB 4: Markets & Asset Allocation ----
        'chart_04': chart_04_cyc_def(cyc),
        'chart_cyc_def_bus': chart_cyc_def_business(cyc, wbci),
        'chart_msci_liq': chart_msci_vs_global_liq(liq),
        'chart_10': chart_10_gold_oil(gold_oil),
        'chart_07': chart_07_debt_liquidity(dl),
        'chart_debt_liq_crises': chart_debt_liq_with_crises(dl),
        'chart_debt_maturity': chart_debt_maturity_wall(),
        'chart_capital_demands': chart_us_capital_demands(period),
        'chart_gli_vs_bes': chart_gli_pctch_vs_bes(liq),
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


def _phase_gauge_html(state: dict) -> str:
    """
    Painel visual compacto: termometro de fase + 4 ducks + key metrics
    + 1-line call. Substitui a parede de texto JSON+memo no topo.
    """
    cls = state.get('classifier_output', {})
    liq = state.get('liquidity', {})
    nfl = state.get('net_fed_liquidity', {})
    wbci = state.get('wbci', {})
    cyc = state.get('cyclicals_defensives', {})
    curve = state.get('yield_curve', {})
    ra = state.get('risk_appetite', {})
    mv = state.get('move', {})
    ducks = state.get('four_ducks', {})
    cycle = state.get('cycle', {})

    phase = cls.get('phase', 'Unknown')
    season = cls.get('season', 'Unknown')
    next_p = cls.get('next_expected_phase', 'Unknown')
    duck_cnt = ducks.get('alignment_count', 0)
    months_to = cycle.get('months_to_next_inflection')
    next_date = cycle.get('next_inflection_date')

    # Termometro horizontal: 4 segmentos com pointer
    segments = [
        ('Rebound',     'Spring', '#7ae582', 'Recession→Recovery'),
        ('Calm',        'Summer', '#00d4ff', 'Recovery→Boom'),
        ('Speculation', 'Autumn', '#ffb84d', 'Boom→Slowing'),
        ('Turbulence',  'Winter', '#ff6b6b', 'Slowing→Recession'),
    ]
    phase_idx = next((i for i, (p, _, _, _) in enumerate(segments)
                       if p == phase), -1)

    seg_html = []
    for i, (p, s, color, desc) in enumerate(segments):
        active = (i == phase_idx)
        opacity = '1.0' if active else '0.35'
        border = (f'box-shadow: 0 0 12px {color}, inset 0 0 0 2px {color};'
                   if active else '')
        seg_html.append(
            f"<div style='flex:1; background:{color}; opacity:{opacity}; "
            f"padding:14px 8px; text-align:center; {border} "
            f"transition: all 0.3s;'>"
            f"<div style='color:#0B0E14; font-weight:800; font-size:14px;'>{p}</div>"
            f"<div style='color:#0B0E14; font-size:10px; opacity:0.85;'>{s}</div>"
            f"<div style='color:#0B0E14; font-size:9px; opacity:0.7; margin-top:2px;'>{desc}</div>"
            f"</div>")

    # 4 Ducks como bullets coloridos
    duck_data = [
        ('Economy',  ducks.get('economy', 'ok')),
        ('Bonds',    ducks.get('bonds', 'ok')),
        ('Equity',   ducks.get('equity_sectors', 'ok')),
        ('Liquidity',ducks.get('liquidity_metrics', 'ok')),
    ]
    duck_html = []
    for name, status in duck_data:
        is_aligned = 'turbulence' in (status or '').lower()
        col = '#ff6b6b' if is_aligned else '#7ae582'
        icon = '●' if is_aligned else '○'
        label = 'aligning' if is_aligned else 'ok'
        duck_html.append(
            f"<div style='display:inline-block; margin-right:18px;'>"
            f"<span style='color:{col}; font-size:18px;'>{icon}</span> "
            f"<span style='color:#cce8ff; font-size:12px;'>{name}</span> "
            f"<span style='color:#8b949e; font-size:10px;'>({label})</span>"
            f"</div>")

    # Key metrics — minigauges
    def _metric(label, value, unit='', color='#cce8ff', dec=1):
        v = (f'{value:.{dec}f}{unit}' if isinstance(value, (int, float))
              else 'N/A')
        return (
            f"<div style='display:inline-block; min-width:120px; "
            f"margin:0 14px 8px 0; vertical-align:top;'>"
            f"<div style='color:#8b949e; font-size:10px; "
            f"text-transform:uppercase; letter-spacing:1px;'>{label}</div>"
            f"<div style='color:{color}; font-size:18px; "
            f"font-weight:700;'>{v}</div></div>")

    liq_z = liq.get('liq_z') or 0
    liq_color = '#ff6b6b' if liq_z < -0.5 else ('#7ae582' if liq_z > 0.5 else '#ffb84d')
    nfl_yoy = nfl.get('yoy_pct') or 0
    nfl_color = '#7ae582' if nfl_yoy > 0 else '#ff6b6b'
    ra_z = ra.get('z') or 0
    ra_color = '#ff6b6b' if ra_z < -0.5 else ('#7ae582' if ra_z > 0.5 else '#ffb84d')

    metrics_html = (
        _metric('Liquidity Z', liq_z, '', liq_color, dec=2) +
        _metric('NFL YoY', nfl_yoy, '%', nfl_color) +
        _metric('Risk Appetite', ra_z, '', ra_color, dec=2) +
        _metric('Cyc/Def 12m', cyc.get('ratio_12m_pct'), '%') +
        _metric('MOVE level', mv.get('level'), '') +
        _metric('Curve label', curve.get('phase_label', 'N/A'), '',
                 color='#cce8ff', dec=0)
    )

    # Smart 1-liner
    one_liner = _build_one_liner(phase, next_p, duck_cnt, liq_z, nfl_yoy,
                                    mv.get('z_5y'), ra_z, months_to)

    # Ponteiro do termometro
    pointer_x = (phase_idx * 25 + 12.5) if phase_idx >= 0 else 50
    pointer_html = (
        f"<div style='position:relative; height:14px; margin-top:-2px;'>"
        f"<div style='position:absolute; left:{pointer_x}%; top:0; "
        f"transform:translateX(-50%);'>"
        f"<div style='width:0; height:0; border-left:8px solid transparent; "
        f"border-right:8px solid transparent; border-bottom:10px solid #cce8ff;'></div>"
        f"</div></div>")

    # Inflection countdown
    countdown = ''
    if months_to and next_date:
        countdown = (
            f"<div style='color:#8b949e; font-size:11px; margin-top:6px;'>"
            f"⏱ Next phase inflection: <b style='color:#cce8ff;'>{next_date}</b> "
            f"(~{months_to:.0f} months)</div>")

    return f"""
    <div class='how-card' style='padding:20px 24px;'>
      <div style='display:flex; justify-content:space-between; align-items:center;
                   margin-bottom:14px;'>
        <div>
          <div style='color:#8b949e; font-size:11px; text-transform:uppercase;
                       letter-spacing:1.5px;'>Current Phase · Season</div>
          <div style='color:#E8742C; font-size:28px; font-weight:800; margin-top:2px;'>
            {phase} · {season}
          </div>
        </div>
        <div style='text-align:right;'>
          <div style='color:#8b949e; font-size:11px; text-transform:uppercase;
                       letter-spacing:1.5px;'>Next Phase</div>
          <div style='color:#cce8ff; font-size:18px; font-weight:700;'>{next_p}</div>
          <div style='color:{'#ff6b6b' if duck_cnt >= 3 else '#7ae582' if duck_cnt == 0 else '#ffb84d'};
                       font-size:13px; font-weight:700; margin-top:2px;'>
            {duck_cnt}/4 ducks aligned
          </div>
        </div>
      </div>

      <!-- Thermometer -->
      <div style='display:flex; gap:2px; border-radius:4px; overflow:hidden;'>
        {''.join(seg_html)}
      </div>
      {pointer_html}

      <!-- One-liner -->
      <div style='background:rgba(232,116,44,0.08); border-left:3px solid #E8742C;
                   padding:10px 14px; margin:14px 0; border-radius:3px;'>
        <div style='color:#cce8ff; font-size:13px; line-height:1.5;'>{one_liner}</div>
      </div>

      <!-- 4 Ducks -->
      <div style='margin-bottom:10px; padding-bottom:10px;
                   border-bottom:1px solid #1e2330;'>
        <div style='color:#8b949e; font-size:11px; text-transform:uppercase;
                     letter-spacing:1.5px; margin-bottom:6px;'>4-Duck Scorecard</div>
        {''.join(duck_html)}
      </div>

      <!-- Metrics -->
      <div>
        <div style='color:#8b949e; font-size:11px; text-transform:uppercase;
                     letter-spacing:1.5px; margin-bottom:6px;'>Key Metrics</div>
        {metrics_html}
      </div>
      {countdown}
    </div>
    """


def _build_one_liner(phase, next_p, duck_cnt, liq_z, nfl_yoy, move_z,
                       ra_z, months_to):
    """Frase curta dependendo do regime."""
    if phase == 'Turbulence' or duck_cnt >= 3:
        return (f"<b>Defensive posture</b> — {duck_cnt}/4 ducks aligned, "
                f"liquidity z {liq_z:+.2f}, MOVE elevated. "
                f"Cash + long bonds historically work here.")
    if phase == 'Speculation':
        return (f"<b>Late-cycle</b> — liquidity inflecting (z {liq_z:+.2f}), "
                f"watch the next duck. Rotate to defensives, "
                f"trim cyclicals into strength.")
    if phase == 'Calm':
        return (f"<b>Risk-on regime</b> — liquidity expanding "
                f"(NFL YoY {nfl_yoy:+.1f}%), risk appetite {ra_z:+.2f}. "
                f"Cyclicals + commodities work.")
    if phase == 'Rebound':
        return (f"<b>Spring rebound</b> — liquidity leaving trough. "
                f"Bonds early, equities late. NFL YoY {nfl_yoy:+.1f}%.")
    return (f"<b>Mixed signals</b> — {duck_cnt}/4 ducks; liquidity z "
            f"{liq_z:+.2f}. Wait for confirmation before sizing up.")


def _build_section_widgets(result: dict) -> list:
    """Converte result em lista de widgets pra display."""
    sec = []
    state = result['state']
    cls = state.get('classifier_output', {})

    # Header compacto
    sec.append(wd.HTML(
        DASH_CSS +
        f"<div class='how-divider'>"
        f"<div class='how-divider-title'>HOWELL LIQUIDITY FRAMEWORK</div>"
        f"<div class='how-divider-sub'>As of {state['as_of_date']} · {VERSION}</div>"
        f"</div>"))

    # PAINEL VISUAL PRINCIPAL — termometro + 4 ducks + metrics + 1-liner
    sec.append(wd.HTML(_phase_gauge_html(state)))

    # Charts organizados em 4 abas
    figs = result['figs']

    # Mapeamento de abas (ordem importa)
    tab_layout = {
        '🌊 Macro & Cycle': [
            'chart_01', 'chart_cycle_65m', 'chart_gli_vs_wbc',
            'chart_02', 'chart_03', 'chart_daily_ai_gdp',
            'chart_12', 'chart_flash_pli', 'chart_flash_tli',
        ],
        '💰 Stimulus & Plumbing': [
            'chart_05b', 'chart_stim_stacked',
            'chart_stim_vs_btc', 'chart_stim_vs_ism',
            'chart_05', 'chart_excess_reserves',
            'chart_sofr_iorb_zones', 'chart_move_zones', 'chart_06',
        ],
        '📈 Rates & Term Premia': [
            'chart_09', 'chart_us_liq_vs_curve',
            'chart_08', 'chart_tp_majors',
            'chart_terminal_rates', 'chart_world_tp_policy', 'chart_11',
        ],
        '🎯 Markets & Allocation': [
            'chart_04', 'chart_cyc_def_bus', 'chart_msci_liq',
            'chart_10', 'chart_07', 'chart_debt_liq_crises',
            'chart_debt_maturity', 'chart_capital_demands',
            'chart_gli_vs_bes', 'chart_13', 'chart_14', 'chart_15',
        ],
    }

    def _fig_widget(fig):
        try:
            return go.FigureWidget(fig)
        except Exception:
            out = wd.Output()
            with out:
                display(fig)
            return out

    tab_children = []
    tab_titles = list(tab_layout.keys())

    for tab_name, chart_keys in tab_layout.items():
        tab_widgets = []
        # Adiciona HTML panels especificos de cada aba
        if tab_name.endswith('Allocation'):
            tab_widgets.append(wd.HTML(chart_asset_allocation_grid_html()))
            tab_widgets.append(wd.HTML(chart_asset_allocation_cycle_svg()))
            tab_widgets.append(wd.HTML(chart_debt_liquidity_cycle_html()))
        for key in chart_keys:
            if key in figs:
                tab_widgets.append(_fig_widget(figs[key]))
        tab_children.append(wd.VBox(tab_widgets,
                                        layout=wd.Layout(overflow='auto',
                                                            max_height='85vh')))

    tabs = wd.Tab()
    tabs.children = tab_children
    for i, title in enumerate(tab_titles):
        tabs.set_title(i, title)
    tabs.selected_index = 0
    sec.append(tabs)

    # Detalhes colapsaveis no fim (JSON + Memo + Audit)
    state_pretty = json.dumps(state, indent=2, default=str)
    memo = run_analyst(state)

    sec.append(wd.HTML("<div class='how-divider'><div class='how-divider-title'>"
                         "DETAILS (click to expand)</div></div>"))

    # Summary (ingles simples)
    sec.append(wd.HTML(
        f"<details class='how-card'>"
        f"<summary style='cursor:pointer; color:#E8742C; font-weight:700; "
        f"font-size:13px; padding:4px 0;'>📝 Analyst Memo (full)</summary>"
        f"<pre class='how-memo' style='margin-top:12px;'>{memo}</pre>"
        f"</details>"))

    # State JSON
    sec.append(wd.HTML(
        f"<details class='how-card'>"
        f"<summary style='cursor:pointer; color:#E8742C; font-weight:700; "
        f"font-size:13px; padding:4px 0;'>🗂 state.json (handoff contract)</summary>"
        f"<pre class='how-memo' style='margin-top:12px; max-height:500px; overflow:auto;'>"
        f"{state_pretty}</pre>"
        f"</details>"))

    # Loader audit
    sec.append(wd.HTML(
        f"<details class='how-card'>"
        f"<summary style='cursor:pointer; color:#E8742C; font-weight:700; "
        f"font-size:13px; padding:4px 0;'>🔎 Loader Audit (ticker debug)</summary>"
        f"<div style='margin-top:12px;'>{debug_loader_report()}</div>"
        f"</details>"))

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
