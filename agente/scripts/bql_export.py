# ═══════════════════════════════════════════════════════════════
#  BQL Export — cola e roda no BQuant Notebook
#
#  Cria botões no notebook:
#    ⬇ Exportar agora  → snapshot do dia (fund, IV, GEX, prices, macro)
#    ⬇ Bulk 252d       → tudo + 252 dias de histórico (preços + IV)
#    ▶ Loop 3min       → re-exporta a cada 3 minutos
#
#  Salva CSVs em ~/bql_data/ (ou C:\Users\rafael bentes\bbg\agente\bql_data
#  se a pasta do projeto existir) e dispara download do ZIP no browser.
# ═══════════════════════════════════════════════════════════════
import bql
import pandas as pd
import numpy as np
import zipfile, io, base64, time, threading
from pathlib import Path
from datetime import date, datetime
from scipy.stats import norm
from IPython.display import display, HTML
import ipywidgets as widgets

# ── Config ────────────────────────────────────────────────────────────────
# Sempre o mesmo lugar — não muda entre BQuant / local / qualquer ambiente.
OUT = Path.home() / "bql_data"
INTERVAL     = 180          # segundos entre execuções no modo loop
TRADING_DAYS = 252

bq           = bql.Service()
OUT.mkdir(parents=True, exist_ok=True)
hoje         = date.today().isoformat()

# ── Helpers ────────────────────────────────────────────────────────────────
def _bql(univ, items):
    """Executa BQL e retorna DataFrame deduplicado por índice e coluna."""
    resp   = bq.execute(bql.Request(univ, items))
    frames = []
    for r in resp:
        s = r.df()[r.name]
        s = s[~s.index.duplicated(keep='last')]
        frames.append(s)
    df = pd.concat(frames, axis=1)
    return df.loc[:, ~df.columns.duplicated()]


def _safe_items(spec):
    """
    Constrói o dict de items do BQL pulando fields que não existem.

    spec: lista de tuplas (alias, callable)
      - callable: lambda: bq.data.foo()  → tenta chamar; se erro, pula
    Retorna: dict {alias: bql_item} apenas com fields que existiram.
    """
    items = {}
    for alias, item in spec:
        try:
            items[alias] = item()
        except Exception as e:
            _log(f'  field skipped {alias}: {type(e).__name__}')
    return items


def _to_num(v):
    """Converte valor BQL para float, retorna '' se NaN."""
    try:
        n = pd.to_numeric(v, errors='coerce')
        return round(float(n), 4) if not pd.isna(n) else ''
    except Exception:
        return ''

def _log(msg): print(f'  {msg}')

def calc_gamma(S, K, vol, T):
    """Black-Scholes gamma."""
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        g  = norm.pdf(d1) / (S * vol * np.sqrt(T))
        return np.where(np.isfinite(g), g, 0.0)


# ── Universos ─────────────────────────────────────────────────────────────
# Tudo é ETF agora — cada vértice da rede neural é um ETF que representa
# o setor / duration / moeda / commodity inteiro. Nada de mega-caps individuais.
#
# SECTOR_ETFS = ETFs de equity (P/E, P/B, dividend yield fazem sentido)
# RATES_ETFS  = renda fixa (yield, duration, OAS — sem P/E)
# FX_ETFS     = moedas (yield, NAV)
# COMMODITY_ETFS = commodities (NAV, expense ratio)
SECTOR_ETFS = [
    # ── 11 SPDR Sector ETFs (cobre o SPX inteiro) ────────────────────────
    ('XLK',  'Technology'),
    ('XLF',  'Financials'),
    ('XLV',  'Health Care'),
    ('XLY',  'Consumer Discretionary'),
    ('XLP',  'Consumer Staples'),
    ('XLE',  'Energy'),
    ('XLI',  'Industrials'),
    ('XLB',  'Materials'),
    ('XLRE', 'Real Estate'),
    ('XLU',  'Utilities'),
    ('XLC',  'Communication Services'),
    # ── Nasdaq setoriais (sub-temas) ──────────────────────────────────────
    ('SOXX', 'Semiconductors'),
    ('IGV',  'Software'),
    ('IBB',  'Biotech'),
    ('QCLN', 'Clean Energy'),
    # ── Broad index ETFs ─────────────────────────────────────────────────
    ('SPY',  'S&P 500'),
    ('QQQ',  'Nasdaq 100'),
    ('IWM',  'Russell 2000'),
    ('MDY',  'S&P MidCap 400'),
    # ── International equity ─────────────────────────────────────────────
    ('EEM',  'EM Equity'),
    ('EFA',  'Developed ex-US'),
    ('VWO',  'EM Vanguard'),
    ('FXI',  'China'),
    ('EWJ',  'Japan'),
    ('EWZ',  'Brazil'),
    ('INDA', 'India'),
    # ── Miners (commodity equity) ─────────────────────────────────────────
    ('GDX',  'Gold Miners'),
    ('GDXJ', 'Gold Junior Miners'),
    ('SIL',  'Silver Miners'),
    ('XME',  'Metals & Mining'),
]

# ── Mega-caps individuais — cada uma é 1 nó único na rede ────────────────
# (sem duplicação: ticker aparece UMA vez só, mesmo que pertença a múltiplos
#  setores como XLK e SOXX no caso de NVDA)
MEGA_CAPS = [
    # Tech / AI
    ('AAPL',  'Apple'),
    ('MSFT',  'Microsoft'),
    ('NVDA',  'NVIDIA'),
    ('GOOGL', 'Alphabet'),
    ('META',  'Meta Platforms'),
    ('AMZN',  'Amazon'),
    ('AVGO',  'Broadcom'),
    ('TSLA',  'Tesla'),
    ('NFLX',  'Netflix'),
    # Financials
    ('JPM',   'JPMorgan'),
    ('BRK-B', 'Berkshire Hathaway'),
    ('V',     'Visa'),
    ('MA',    'Mastercard'),
    # Health / Pharma
    ('LLY',   'Eli Lilly'),
    ('UNH',   'UnitedHealth'),
    ('JNJ',   'Johnson & Johnson'),
    # Consumer / Energy / Staples
    ('XOM',   'Exxon Mobil'),
    ('COST',  'Costco'),
    ('WMT',   'Walmart'),
    ('PG',    'Procter & Gamble'),
]
MEGA_CAP_TICKERS = [f'{tk} US Equity' for tk, _ in MEGA_CAPS]
# BRK-B no Bloomberg é BRK/B
MEGA_CAP_TICKERS = [t.replace('BRK-B US Equity', 'BRK/B US Equity') for t in MEGA_CAP_TICKERS]
MEGA_CAP_LABELS  = {}
for tk, label in MEGA_CAPS:
    bbg = 'BRK/B US Equity' if tk == 'BRK-B' else f'{tk} US Equity'
    MEGA_CAP_LABELS[bbg] = label

RATES_ETFS = [
    # Treasuries duration ladder
    ('BIL',  'T-bills 1-3M',     'rates'),
    ('SHV',  'Treasury 0-1Y',    'rates'),
    ('SHY',  'Treasury 1-3Y',    'rates'),
    ('IEI',  'Treasury 3-7Y',    'rates'),
    ('IEF',  'Treasury 7-10Y',   'rates'),
    ('TLH',  'Treasury 10-20Y',  'rates'),
    ('TLT',  'Treasury 20Y+',    'rates'),
    ('EDV',  'Treasury 25Y+',    'rates'),
    ('GOVT', 'Treasuries broad', 'rates'),
    # Credit
    ('LQD',  'IG Corporate',     'credit'),
    ('HYG',  'HY Corporate',     'credit'),
    ('JNK',  'HY Corporate alt', 'credit'),
    ('EMB',  'EM USD bonds',     'credit'),
    ('BNDX', 'International bonds','credit'),
    # Inflação / TIPS
    ('TIP',  'TIPS broad',       'tips'),
    ('STIP', 'TIPS short',       'tips'),
    ('LTPZ', 'TIPS long',        'tips'),
    # Floating rate / convertibles / preferred
    ('BKLN', 'Senior loans',     'credit'),
    ('CWB',  'Convertibles',     'hybrid'),
    ('PFF',  'Preferred stock',  'hybrid'),
]

FX_ETFS = [
    ('UUP', 'USD Long'),
    ('UDN', 'USD Short'),
    ('FXE', 'EUR'),
    ('FXB', 'GBP'),
    ('FXY', 'JPY'),
    ('FXC', 'CAD'),
    ('FXA', 'AUD'),
    ('FXF', 'CHF'),
    ('CYB', 'CNY'),
    ('CEW', 'EM Currencies'),
]

COMMODITY_ETFS = [
    # Broad
    ('DBC',  'Broad commodities', 'broad'),
    ('GSG',  'GSCI broad',        'broad'),
    ('PDBC', 'Broad no K-1',      'broad'),
    # Precious metals
    ('GLD',  'Gold',              'precious'),
    ('SLV',  'Silver',            'precious'),
    ('PPLT', 'Platinum',          'precious'),
    ('PALL', 'Palladium',         'precious'),
    # Energy
    ('USO',  'WTI Oil',           'energy'),
    ('BNO',  'Brent Oil',         'energy'),
    ('UNG',  'Natural Gas',       'energy'),
    ('UGA',  'Gasoline',          'energy'),
    # Industrial / agro
    ('CPER', 'Copper',            'industrial'),
    ('DBA',  'Agriculture broad', 'agro'),
    ('CORN', 'Corn',              'agro'),
    ('WEAT', 'Wheat',             'agro'),
    ('SOYB', 'Soybeans',          'agro'),
]

VOL_ETFS = [
    ('VIXY', 'VIX short-term'),
    ('UVXY', 'VIX 1.5x'),
    ('SVXY', 'Short VIX'),
]

# Equity-style fundamentals (P/E faz sentido aqui)
# Inclui SECTOR_ETFS + 6 mega-caps individuais — cada uma é 1 nó único.
FUND_TICKERS = [f'{tk} US Equity' for tk, _ in SECTOR_ETFS] + MEGA_CAP_TICKERS
SECTOR_LABELS = {f'{tk} US Equity': label for tk, label in SECTOR_ETFS}
SECTOR_LABELS.update(MEGA_CAP_LABELS)

# Bond/FX/Commodity ETFs — fundamentals reduzidos (sem P/E)
RATES_FUND_TICKERS     = [f'{tk} US Equity' for tk, _, _ in RATES_ETFS]
FX_FUND_TICKERS        = [f'{tk} US Equity' for tk, _    in FX_ETFS]
COMMODITY_FUND_TICKERS = [f'{tk} US Equity' for tk, _, _ in COMMODITY_ETFS]
VOL_FUND_TICKERS       = [f'{tk} US Equity' for tk, _    in VOL_ETFS]

NON_EQUITY_FUND_TICKERS = (
    RATES_FUND_TICKERS + FX_FUND_TICKERS + COMMODITY_FUND_TICKERS + VOL_FUND_TICKERS
)
NON_EQUITY_LABELS = {}
NON_EQUITY_CATEGORY = {}
for tk, label, cat in RATES_ETFS:
    NON_EQUITY_LABELS[f'{tk} US Equity']   = label
    NON_EQUITY_CATEGORY[f'{tk} US Equity'] = cat
for tk, label in FX_ETFS:
    NON_EQUITY_LABELS[f'{tk} US Equity']   = label
    NON_EQUITY_CATEGORY[f'{tk} US Equity'] = 'fx'
for tk, label, cat in COMMODITY_ETFS:
    NON_EQUITY_LABELS[f'{tk} US Equity']   = label
    NON_EQUITY_CATEGORY[f'{tk} US Equity'] = cat
for tk, label in VOL_ETFS:
    NON_EQUITY_LABELS[f'{tk} US Equity']   = label
    NON_EQUITY_CATEGORY[f'{tk} US Equity'] = 'vol'


# ── Universo completo da rede neural ─────────────────────────────────────
# Cobre todos os vértices: setores SPX, mega-caps individuais (1 nó cada),
# fixed income por duration, FX por moeda, commodities por tipo, vol,
# internacional.
# REGRA: cada ticker aparece UMA VEZ só (sem duplicação).
_YF_TO_BBG = {
    # ══════ MEGA-CAPS INDIVIDUAIS (1 nó cada, sem duplicar) ══════
    'AAPL':  'AAPL US Equity',  'MSFT':  'MSFT US Equity',
    'NVDA':  'NVDA US Equity',  'GOOGL': 'GOOGL US Equity',
    'META':  'META US Equity',  'AMZN':  'AMZN US Equity',
    'AVGO':  'AVGO US Equity',  'TSLA':  'TSLA US Equity',
    'NFLX':  'NFLX US Equity',  'JPM':   'JPM US Equity',
    'BRK-B': 'BRK/B US Equity', 'V':     'V US Equity',
    'MA':    'MA US Equity',    'LLY':   'LLY US Equity',
    'UNH':   'UNH US Equity',   'JNJ':   'JNJ US Equity',
    'XOM':   'XOM US Equity',   'COST':  'COST US Equity',
    'WMT':   'WMT US Equity',   'PG':    'PG US Equity',

    # ══════ EQUITY ETFs ══════
    # ── Broad index ETFs ─────────────────────────────────────────────────
    'SPY': 'SPY US Equity',  'QQQ': 'QQQ US Equity',
    'IWM': 'IWM US Equity',  'MDY': 'MDY US Equity',
    # ── 11 SPDR Sector ETFs (SPX completo) ────────────────────────────────
    'XLK':  'XLK US Equity',  'XLF':  'XLF US Equity',  'XLV':  'XLV US Equity',
    'XLY':  'XLY US Equity',  'XLP':  'XLP US Equity',  'XLE':  'XLE US Equity',
    'XLI':  'XLI US Equity',  'XLB':  'XLB US Equity',  'XLRE': 'XLRE US Equity',
    'XLU':  'XLU US Equity',  'XLC':  'XLC US Equity',
    # ── Nasdaq setoriais (sub-temas) ──────────────────────────────────────
    'SOXX': 'SOXX US Equity', 'IGV':  'IGV US Equity',
    'IBB':  'IBB US Equity',  'QCLN': 'QCLN US Equity',
    # ── International equity ─────────────────────────────────────────────
    'EEM':  'EEM US Equity',  'EFA':  'EFA US Equity',
    'VWO':  'VWO US Equity',  'FXI':  'FXI US Equity',  # EM broad / China
    'EWJ':  'EWJ US Equity',  'EWZ':  'EWZ US Equity',  # Japão / Brasil
    'INDA': 'INDA US Equity',                            # India

    # ══════ FIXED INCOME — por vértice de duração ══════
    # Treasuries duration ladder
    'BIL':  'BIL US Equity',   # 1-3M T-bills
    'SHV':  'SHV US Equity',   # 0-1Y
    'SHY':  'SHY US Equity',   # 1-3Y
    'IEI':  'IEI US Equity',   # 3-7Y
    'IEF':  'IEF US Equity',   # 7-10Y
    'TLH':  'TLH US Equity',   # 10-20Y
    'TLT':  'TLT US Equity',   # 20Y+
    'EDV':  'EDV US Equity',   # 25Y+ extended duration
    'GOVT': 'GOVT US Equity',  # broad treasuries
    # Credit
    'LQD':  'LQD US Equity',   # IG corporate
    'HYG':  'HYG US Equity',   # HY corporate
    'JNK':  'JNK US Equity',   # HY corporate (alternative)
    'EMB':  'EMB US Equity',   # EM USD bonds
    'BNDX': 'BNDX US Equity',  # international bonds
    # Inflação / TIPS
    'TIP':  'TIP US Equity',   # TIPS broad
    'STIP': 'STIP US Equity',  # short TIPS
    'LTPZ': 'LTPZ US Equity',  # long TIPS
    # Bank loans / convertibles / preferred
    'BKLN': 'BKLN US Equity',  # senior loans (floating rate)
    'CWB':  'CWB US Equity',   # convertibles
    'PFF':  'PFF US Equity',   # preferred stock

    # ══════ FX — por moeda ══════
    'UUP':  'UUP US Equity',   # USD long
    'UDN':  'UDN US Equity',   # USD short
    'FXE':  'FXE US Equity',   # EUR
    'FXB':  'FXB US Equity',   # GBP
    'FXY':  'FXY US Equity',   # JPY
    'FXC':  'FXC US Equity',   # CAD
    'FXA':  'FXA US Equity',   # AUD
    'FXF':  'FXF US Equity',   # CHF
    'CYB':  'CYB US Equity',   # CNY
    'CEW':  'CEW US Equity',   # EM currencies basket

    # ══════ COMMODITIES — por tipo ══════
    # Broad
    'DBC':  'DBC US Equity',   # broad commodities
    'GSG':  'GSG US Equity',   # GSCI commodities
    'PDBC': 'PDBC US Equity',  # broad (no K-1)
    # Precious metals
    'GLD':  'GLD US Equity',   # Gold
    'SLV':  'SLV US Equity',   # Silver
    'PPLT': 'PPLT US Equity',  # Platinum
    'PALL': 'PALL US Equity',  # Palladium
    # Energy
    'USO':  'USO US Equity',   # WTI Oil
    'BNO':  'BNO US Equity',   # Brent Oil
    'UNG':  'UNG US Equity',   # Natural Gas
    'UGA':  'UGA US Equity',   # Gasoline
    # Industrial / agro
    'CPER': 'CPER US Equity',  # Copper
    'DBA':  'DBA US Equity',   # Agriculture broad
    'CORN': 'CORN US Equity',  # Corn
    'WEAT': 'WEAT US Equity',  # Wheat
    'SOYB': 'SOYB US Equity',  # Soybeans
    # Miners (proxy de commodity exposure alavancado)
    'GDX':  'GDX US Equity',   # Gold miners
    'GDXJ': 'GDXJ US Equity',  # Gold junior miners
    'SIL':  'SIL US Equity',   # Silver miners
    'XME':  'XME US Equity',   # Metals & mining

    # ══════ VOL & TAIL ══════
    'VIXY': 'VIXY US Equity',  # VIX short-term
    'UVXY': 'UVXY US Equity',  # VIX 1.5x
    'SVXY': 'SVXY US Equity',  # short VIX

    # ══════ ÍNDICES / FUTURES / FX / CRYPTO (não-ETF) ══════
    '^GSPC':    'SPX Index',
    '^NDX':     'NDX Index',
    '^RUT':     'RTY Index',
    '^VIX':     'VIX Index',
    'CL=F':     'CL1 Comdty',
    'GC=F':     'GC1 Comdty',
    'DX-Y.NYB': 'DXY Curncy',
    'BTC-USD':  'XBT Curncy',
}

LETFS = [
    'UPRO US Equity', 'SPXU US Equity', 'TQQQ US Equity',
    'SQQQ US Equity', 'TNA US Equity',  'TZA US Equity',
]

MACRO_TICKERS = [
    # Curva de juros EUA
    ('USGG1M Index',   'US Treasury 1M',   'rates_usd'),
    ('USGG3M Index',   'US Treasury 3M',   'rates_usd'),
    ('USGG6M Index',   'US Treasury 6M',   'rates_usd'),
    ('USGG1YR Index',  'US Treasury 1Y',   'rates_usd'),
    ('USGG2YR Index',  'US Treasury 2Y',   'rates_usd'),
    ('USGG5YR Index',  'US Treasury 5Y',   'rates_usd'),
    ('USGG10YR Index', 'US Treasury 10Y',  'rates_usd'),
    ('USGG30YR Index', 'US Treasury 30Y',  'rates_usd'),
    # Volatilidade
    ('VIX Index',      'VIX Spot',         'volatility'),
    ('VIX9D Index',    'VIX 9-Day',        'volatility'),
    ('VIX3M Index',    'VIX 3-Month',      'volatility'),
    ('VVIX Index',     'Vol of VIX',       'volatility'),
    ('MOVE Index',     'MOVE (bond vol)',  'volatility'),
    # Spreads de crédito
    ('LUACOAS Index',  'IG OAS Spread',    'credit_spread'),
    ('LF98OAS Index',  'HY OAS Spread',    'credit_spread'),
    # FX / monetary / inflation
    ('DXY Curncy',     'Dollar Index',     'fx'),
    ('SOFRRATE Index', 'SOFR Rate',        'monetary'),
    ('USGGBE10 Index', 'US 10Y Breakeven', 'inflation'),
]


# ── Funções de export ─────────────────────────────────────────────────────

def export_fundamentals():
    """
    Snapshot de fundamentals dos ETFs setoriais.
    Cada ETF reflete a média ponderada das holdings — pe_ratio, dividend_yield,
    beta etc. já são "do setor inteiro".
    """
    univ  = bq.univ.list(FUND_TICKERS)
    t_str = ', '.join(f'"{t}"' for t in FUND_TICKERS)
    items = {
        'PE_RATIO':            bq.data.pe_ratio(),
        'PX_TO_BOOK':          bq.data.px_to_book_ratio(),
        'PX_TO_SALES':         bq.data.px_to_sales_ratio(),
        'CUR_MKT_CAP':         bq.data.cur_mkt_cap(),       # AUM efetivo do ETF
        'BETA':                bq.data.beta(),
        'PROF_MARGIN':         bq.data.prof_margin(),
        'RETURN_COM_EQY':      bq.data.return_com_eqy(),
        'EQY_DVD_YLD_IND':     bq.data.eqy_dvd_yld_ind(),
        'TOT_DEBT_TO_TOT_EQY': bq.data.tot_debt_to_tot_eqy(),
        'EXPENSE_RATIO':       bq.data.fund_expense_ratio(),
        'FUND_TOTAL_ASSETS':   bq.data.fund_total_assets(),
        'PX_LAST':             bq.data.px_last(),
    }
    df = _bql(univ, items)
    df.rename(columns={
        'PE_RATIO':          'pe',
        'PX_TO_BOOK':        'pb',
        'PX_TO_SALES':       'ps',
        'CUR_MKT_CAP':       'mktcap_b',
        'BETA':              'beta',
        'PROF_MARGIN':       'profit_margin',
        'RETURN_COM_EQY':    'roe',
        'EQY_DVD_YLD_IND':   'dividend_yield',
        'TOT_DEBT_TO_TOT_EQY':'debt_equity',
        'EXPENSE_RATIO':     'expense_ratio',
        'FUND_TOTAL_ASSETS': 'aum_b',
        'PX_LAST':           'price',
    }, inplace=True)
    df['mktcap_b']       = pd.to_numeric(df['mktcap_b'],       errors='coerce') / 1e9
    df['aum_b']          = pd.to_numeric(df['aum_b'],          errors='coerce') / 1e9
    df['profit_margin']  = pd.to_numeric(df['profit_margin'],  errors='coerce') / 100
    df['roe']            = pd.to_numeric(df['roe'],            errors='coerce') / 100
    df['dividend_yield'] = pd.to_numeric(df['dividend_yield'], errors='coerce') / 100
    df['expense_ratio']  = pd.to_numeric(df['expense_ratio'],  errors='coerce') / 100

    # 52w high/low
    try:
        resp2 = bq.execute(
            f'get(PX_HIGH(dates=range(-365D,0D),frq=Y),'
            f'PX_LOW(dates=range(-365D,0D),frq=Y)) for([{t_str}])'
        )
        df2 = pd.concat([r.df()[r.name] for r in resp2], axis=1)
        df2 = df2.loc[:, ~df2.columns.duplicated()]
        hi  = next((c for c in df2.columns if 'HIGH' in c.upper()), None)
        lo  = next((c for c in df2.columns if 'LOW'  in c.upper()), None)
        if hi: df['hi_52w'] = pd.to_numeric(df2[hi], errors='coerce')
        if lo: df['lo_52w'] = pd.to_numeric(df2[lo], errors='coerce')
        if hi: df['drawdown_52w'] = (df['price'] - df['hi_52w']) / df['hi_52w']
    except Exception as e:
        _log(f'52w warn: {e}')

    # Sector label (humano)
    df['sector'] = df.index.map(SECTOR_LABELS)

    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity', '', regex=False)
    df.to_csv(OUT / f'fundamentals_{hoje}.csv')
    _log(f'fundamentals — {len(df)} ETFs setoriais')


def export_fundamentals_history():
    """
    Histórico de 252 dias dos fundamentals que mudam ao longo do tempo:
    pe_ratio, dividend_yield, px_to_book, px_to_sales, beta.
    Cobre: SECTOR_ETFS + MEGA_CAPS individuais.
    Permite computar pe_percentile, pe_zscore, valuation expansion/compression.
    """
    rows = []
    for bbg_tk in FUND_TICKERS:
        sector = SECTOR_LABELS.get(bbg_tk, '')
        try:
            resp = bq.execute(
                f'get('
                f'PE_RATIO(dates=range(-252D,0D),frq=D,fill=PREV),'
                f'PX_TO_BOOK_RATIO(dates=range(-252D,0D),frq=D,fill=PREV),'
                f'PX_TO_SALES_RATIO(dates=range(-252D,0D),frq=D,fill=PREV),'
                f'EQY_DVD_YLD_IND(dates=range(-252D,0D),frq=D,fill=PREV),'
                f'BETA(dates=range(-252D,0D),frq=D,fill=PREV)'
                f') for(["{bbg_tk}"])'
            )
            frames = []
            for r in resp:
                s = r.df()[r.name]
                s = s[~s.index.duplicated(keep='last')]
                frames.append(s.rename(r.name))
            df_h = pd.concat(frames, axis=1)
            df_h.index = pd.to_datetime(df_h.index)
            ticker_short = bbg_tk.replace(' US Equity', '').replace('/', '-')
            for dt, row in df_h.iterrows():
                pe   = pd.to_numeric(row.get('PE_RATIO'),         errors='coerce')
                pb   = pd.to_numeric(row.get('PX_TO_BOOK_RATIO'), errors='coerce')
                ps   = pd.to_numeric(row.get('PX_TO_SALES_RATIO'),errors='coerce')
                dy   = pd.to_numeric(row.get('EQY_DVD_YLD_IND'),  errors='coerce')
                beta = pd.to_numeric(row.get('BETA'),             errors='coerce')
                if pd.isna(pe) and pd.isna(pb) and pd.isna(ps):
                    continue
                rows.append({
                    'date':           dt.date().isoformat(),
                    'ticker':         ticker_short,
                    'sector':         sector,
                    'pe':             round(float(pe),   4) if not pd.isna(pe)   else '',
                    'pb':             round(float(pb),   4) if not pd.isna(pb)   else '',
                    'ps':             round(float(ps),   4) if not pd.isna(ps)   else '',
                    'dividend_yield': round(float(dy)/100,6) if not pd.isna(dy)  else '',
                    'beta':           round(float(beta), 4) if not pd.isna(beta) else '',
                })
        except Exception as e:
            _log(f'fund_hist warn {bbg_tk}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'fundamentals_history_{hoje}.csv', index=False)
        _log(f'fundamentals_history — {len(rows)} linhas ({len(FUND_TICKERS)} tickers × 252d)')


# Cada bond ETF tem duração implícita pelo próprio nome/mandato.
# Usamos um mapa estático em vez de tentar puxar fund_effective_duration
# (que não existe como campo BQL — duration é uma característica do produto).
RATES_DURATION = {
    'BIL':  0.15,   # 1-3M T-bills (~0.1y)
    'SHV':  0.40,   # 0-1Y (~0.4y)
    'SHY':  1.85,   # 1-3Y (~1.9y)
    'IEI':  4.50,   # 3-7Y (~4.5y)
    'IEF':  7.50,   # 7-10Y (~7.5y)
    'TLH':  13.50,  # 10-20Y
    'TLT':  17.00,  # 20Y+ (~17y)
    'EDV':  24.00,  # 25Y+ extended duration
    'GOVT': 6.00,   # broad treasuries (~6y)
    'LQD':  8.50,   # IG corporate (~8.5y)
    'HYG':  3.50,   # HY corporate (~3.5y)
    'JNK':  3.50,   # HY corporate alt
    'EMB':  7.00,   # EM USD bonds
    'BNDX': 7.50,   # international bonds
    'TIP':  6.50,   # TIPS broad
    'STIP': 2.50,   # TIPS short
    'LTPZ': 19.00,  # TIPS long
    'BKLN': 0.10,   # senior loans (floating rate)
    'CWB':  3.00,   # convertibles
    'PFF':  6.00,   # preferred stock
}


def export_bond_etf_fundamentals():
    """
    Snapshot de bond ETFs: NAV, expense ratio, AUM, dividend yield (proxy de yield),
    YTD return.

    Duration vem do mapa estático RATES_DURATION (cada ETF tem duração implícita
    pelo seu mandato — TLT=20Y+, SHY=1-3Y, etc — não é um campo dinâmico no BQL).
    """
    univ  = bq.univ.list(RATES_FUND_TICKERS)
    items = {
        'PX_LAST':           bq.data.px_last(),
        'EXPENSE_RATIO':     bq.data.fund_expense_ratio(),
        'FUND_TOTAL_ASSETS': bq.data.fund_total_assets(),
        'DVD_YLD':           bq.data.eqy_dvd_yld_ind(),  # proxy para yield do ETF
    }
    df = _bql(univ, items)
    df.rename(columns={
        'PX_LAST':           'price',
        'EXPENSE_RATIO':     'expense_ratio',
        'FUND_TOTAL_ASSETS': 'aum_b',
        'DVD_YLD':           'yield',
    }, inplace=True)
    df['aum_b']         = pd.to_numeric(df['aum_b'],         errors='coerce') / 1e9
    df['expense_ratio'] = pd.to_numeric(df['expense_ratio'], errors='coerce') / 100
    df['yield']         = pd.to_numeric(df['yield'],         errors='coerce') / 100
    df['label']    = df.index.map(NON_EQUITY_LABELS)
    df['category'] = df.index.map(NON_EQUITY_CATEGORY)
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity', '', regex=False)
    df['duration'] = df.index.map(RATES_DURATION)  # mapeia depois do strip
    df.to_csv(OUT / f'bond_etf_fundamentals_{hoje}.csv')
    _log(f'bond_etf_fundamentals — {len(df)} linhas')


def export_fx_etf_fundamentals():
    """Snapshot de FX ETFs: NAV, expense ratio, AUM. Retornos vêm de export_prices."""
    univ  = bq.univ.list(FX_FUND_TICKERS)
    items = {
        'PX_LAST':           bq.data.px_last(),
        'EXPENSE_RATIO':     bq.data.fund_expense_ratio(),
        'FUND_TOTAL_ASSETS': bq.data.fund_total_assets(),
    }
    df = _bql(univ, items)
    df.rename(columns={
        'PX_LAST':           'price',
        'EXPENSE_RATIO':     'expense_ratio',
        'FUND_TOTAL_ASSETS': 'aum_b',
    }, inplace=True)
    df['aum_b']         = pd.to_numeric(df['aum_b'],         errors='coerce') / 1e9
    df['expense_ratio'] = pd.to_numeric(df['expense_ratio'], errors='coerce') / 100
    df['label']    = df.index.map(NON_EQUITY_LABELS)
    df['category'] = df.index.map(NON_EQUITY_CATEGORY)
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity', '', regex=False)
    df.to_csv(OUT / f'fx_etf_fundamentals_{hoje}.csv')
    _log(f'fx_etf_fundamentals — {len(df)} linhas')


def export_commodity_etf_fundamentals():
    """Snapshot de commodity ETFs: NAV, expense ratio, AUM. Retornos vêm de export_prices."""
    univ  = bq.univ.list(COMMODITY_FUND_TICKERS + VOL_FUND_TICKERS)
    items = {
        'PX_LAST':           bq.data.px_last(),
        'EXPENSE_RATIO':     bq.data.fund_expense_ratio(),
        'FUND_TOTAL_ASSETS': bq.data.fund_total_assets(),
    }
    df = _bql(univ, items)
    df.rename(columns={
        'PX_LAST':           'price',
        'EXPENSE_RATIO':     'expense_ratio',
        'FUND_TOTAL_ASSETS': 'aum_b',
    }, inplace=True)
    df['aum_b']         = pd.to_numeric(df['aum_b'],         errors='coerce') / 1e9
    df['expense_ratio'] = pd.to_numeric(df['expense_ratio'], errors='coerce') / 100
    df['label']    = df.index.map(NON_EQUITY_LABELS)
    df['category'] = df.index.map(NON_EQUITY_CATEGORY)
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity', '', regex=False)
    df.to_csv(OUT / f'commodity_etf_fundamentals_{hoje}.csv')
    _log(f'commodity_etf_fundamentals — {len(df)} linhas')


def export_bond_etf_history():
    """
    Histórico de 252 dias dos bond ETFs.
    Como duration é estática (vem de RATES_DURATION) e OAS/yield_to_maturity
    não existem como fields BQL agregados, usamos:
      - PX_LAST: preço (que reflete movimento de yield via NAV)
      - EQY_DVD_YLD_IND: dividend yield indicado (proxy de yield do ETF)
    Permite track de yield drift, duration risk e movimento de NAV.
    """
    rows = []
    for bbg_tk in RATES_FUND_TICKERS:
        label = NON_EQUITY_LABELS.get(bbg_tk, '')
        cat   = NON_EQUITY_CATEGORY.get(bbg_tk, '')
        ticker_short = bbg_tk.replace(' US Equity', '')
        duration = RATES_DURATION.get(ticker_short, '')
        try:
            resp = bq.execute(
                f'get('
                f'PX_LAST(dates=range(-252D,0D),frq=D,fill=PREV),'
                f'EQY_DVD_YLD_IND(dates=range(-252D,0D),frq=D,fill=PREV)'
                f') for(["{bbg_tk}"])'
            )
            frames = []
            for r in resp:
                s = r.df()[r.name]
                s = s[~s.index.duplicated(keep='last')]
                frames.append(s.rename(r.name))
            df_h = pd.concat(frames, axis=1)
            df_h.index = pd.to_datetime(df_h.index)
            for dt, row in df_h.iterrows():
                px  = pd.to_numeric(row.get('PX_LAST'),         errors='coerce')
                yld = pd.to_numeric(row.get('EQY_DVD_YLD_IND'), errors='coerce')
                if pd.isna(px) and pd.isna(yld):
                    continue
                rows.append({
                    'date':     dt.date().isoformat(),
                    'ticker':   ticker_short,
                    'label':    label,
                    'category': cat,
                    'duration': duration,
                    'price':    round(float(px),     4) if not pd.isna(px)  else '',
                    'yield':    round(float(yld)/100,6) if not pd.isna(yld) else '',
                })
        except Exception as e:
            _log(f'bond_hist warn {bbg_tk}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'bond_etf_history_{hoje}.csv', index=False)
        _log(f'bond_etf_history — {len(rows)} linhas ({len(RATES_FUND_TICKERS)} ETFs × 252d)')


def export_options_iv():
    univ  = bq.univ.list(FUND_TICKERS)
    items = _safe_items([
        ('atm_iv', lambda: bq.data.implied_volatility(expiry='30D', pct_moneyness='100')),
        ('put25',  lambda: bq.data.implied_volatility(expiry='30D', delta='25', put_call='PUT')),
        ('call25', lambda: bq.data.implied_volatility(expiry='30D', delta='25', put_call='CALL')),
        ('pcr_oi', lambda: bq.data.put_call_open_interest_ratio()),
    ])
    if not items:
        _log('options_iv: nenhum field disponível')
        return
    try:
        df = _bql(univ, items)
        if 'atm_iv' in df.columns:
            df['atm_iv'] = pd.to_numeric(df['atm_iv'], errors='coerce') / 100
        if 'put25' in df.columns and 'call25' in df.columns:
            df['skew_25d'] = (pd.to_numeric(df['put25'], errors='coerce') -
                              pd.to_numeric(df['call25'], errors='coerce')) / 100
        cols = [c for c in ['atm_iv', 'skew_25d', 'pcr_oi'] if c in df.columns]
        if cols:
            df = df[cols]
        df.index.name = 'ticker'
        df.index = df.index.str.replace(' US Equity', '', regex=False)
        df.to_csv(OUT / f'options_iv_{hoje}.csv')
        _log(f'options_iv — {len(df)} linhas, fields: {list(df.columns)}')
    except Exception as e:
        _log(f'options_iv warn: {e}')


def fetch_spx_chain(spot):
    lo, hi = spot * 0.95, spot * 1.05
    conditions = (
        bq.data.expire_dt() >= '0d'
    ).and_(bq.data.expire_dt() <= '45d'
    ).and_(bq.data.strike_px() > lo
    ).and_(bq.data.strike_px() < hi
    ).and_(bq.data.open_int() > 0
    ).and_(bq.data.ivol() > 0)
    univ  = bq.univ.filter(bq.univ.options(['SPX Index']), conditions)
    items = {
        'expiry':   bq.data.expire_dt(),
        'strike':   bq.data.strike_px(),
        'put_call': bq.data.put_call(),
        'open_int': bq.data.open_int(),
        'ivol':     bq.data.ivol(),
    }
    resp = bq.execute(bql.Request(univ, items))
    df   = pd.concat([r.df()[r.name] for r in resp], axis=1)
    df   = df.loc[:, ~df.columns.duplicated()]
    df['ivol']     = pd.to_numeric(df['ivol'],     errors='coerce') / 100
    df['strike']   = pd.to_numeric(df['strike'],   errors='coerce')
    df['open_int'] = pd.to_numeric(df['open_int'], errors='coerce')
    return df.dropna(subset=['ivol'])


def export_gex_spx():
    spot = float(_bql(bq.univ.list(['SPX Index']), {'px': bq.data.px_last()})['px'].iloc[0])
    _log(f'SPX: {spot:,.0f}')
    df = fetch_spx_chain(spot)
    _log(f'Chain: {len(df)} contratos')
    T = (pd.to_datetime(df['expiry']) - pd.Timestamp.now()).dt.days / TRADING_DAYS
    T = T.clip(lower=1 / TRADING_DAYS)
    g = calc_gamma(spot, df['strike'].values, df['ivol'].values, T.values)
    df['gamma'] = g
    is_call = df['put_call'].str.upper().str.startswith('C')
    df['gex_bn'] = np.where(
        is_call,
         g * df['open_int'] * 100 * spot / 1e9,
        -g * df['open_int'] * 100 * spot / 1e9,
    )
    df.index.name = 'ticker'
    df.to_csv(OUT / f'gex_spx_{hoje}.csv')
    gex_total = df['gex_bn'].sum()
    pd.DataFrame([{
        'date': hoje, 'spot': spot,
        'gex_total_bn': round(gex_total, 3),
        'gex_call_bn':  round(df[is_call]['gex_bn'].sum(), 3),
        'gex_put_bn':   round(df[~is_call]['gex_bn'].sum(), 3),
        'direction':    'long' if gex_total > 0 else 'short',
        'gamma_regime': 'positive' if gex_total > 0 else 'negative',
        'n_options':    len(df),
    }]).to_csv(OUT / f'gex_summary_{hoje}.csv', index=False)
    _log(f'gex_spx — GEX={gex_total:+.2f}B')


# ═══════════════════════════════════════════════════════════════════════════
#  COBERTURA COMPLETA — funções adicionais para fechar 100% via BQL
# ═══════════════════════════════════════════════════════════════════════════

def export_options_greeks_full():
    """
    Chain completa + Greeks dealer-aggregated por mega-cap individual.
    Hoje só SPX tem GEX. Aqui replicamos para AAPL, NVDA, TSLA, etc.

    Para cada underlying:
      - Pega chain ±10% spot, expiries 0-45d, OI > 0
      - Computa gamma BS local
      - Agrega: gex_net_bn, gex_call_bn, gex_put_bn, call_wall, put_wall,
                gamma_flip (zero-crossing do GEX cumulativo)
    """
    UNDERLYINGS = [
        ('AAPL US Equity',  'AAPL'),
        ('MSFT US Equity',  'MSFT'),
        ('NVDA US Equity',  'NVDA'),
        ('GOOGL US Equity', 'GOOGL'),
        ('META US Equity',  'META'),
        ('AMZN US Equity',  'AMZN'),
        ('AVGO US Equity',  'AVGO'),
        ('TSLA US Equity',  'TSLA'),
        ('NFLX US Equity',  'NFLX'),
        ('SPY US Equity',   'SPY'),
        ('QQQ US Equity',   'QQQ'),
        ('IWM US Equity',   'IWM'),
        ('NDX Index',       'NDX'),
        ('RTY Index',       'RUT'),
    ]

    summary_rows = []
    for bbg_und, short in UNDERLYINGS:
        try:
            spot = float(_bql(bq.univ.list([bbg_und]), {'px': bq.data.px_last()})['px'].iloc[0])
        except Exception as e:
            _log(f'greeks_full warn {short} spot: {e}')
            continue

        try:
            lo, hi = spot * 0.90, spot * 1.10
            conditions = (
                bq.data.expire_dt() >= '0d'
            ).and_(bq.data.expire_dt() <= '45d'
            ).and_(bq.data.strike_px() > lo
            ).and_(bq.data.strike_px() < hi
            ).and_(bq.data.open_int() > 0
            ).and_(bq.data.ivol() > 0)
            univ  = bq.univ.filter(bq.univ.options([bbg_und]), conditions)
            # 'volume' por contrato não existe via bq.data — só fica OI.
            items = {
                'expiry':   bq.data.expire_dt(),
                'strike':   bq.data.strike_px(),
                'put_call': bq.data.put_call(),
                'open_int': bq.data.open_int(),
                'ivol':     bq.data.ivol(),
            }
            resp = bq.execute(bql.Request(univ, items))
            df   = pd.concat([r.df()[r.name] for r in resp], axis=1)
            df   = df.loc[:, ~df.columns.duplicated()]
            df['ivol']     = pd.to_numeric(df['ivol'],     errors='coerce') / 100
            df['strike']   = pd.to_numeric(df['strike'],   errors='coerce')
            df['open_int'] = pd.to_numeric(df['open_int'], errors='coerce')
            df = df.dropna(subset=['ivol', 'strike', 'open_int'])
            if df.empty:
                _log(f'greeks_full warn {short}: chain vazia')
                continue

            T = (pd.to_datetime(df['expiry']) - pd.Timestamp.now()).dt.days / TRADING_DAYS
            T = T.clip(lower=1 / TRADING_DAYS)
            g = calc_gamma(spot, df['strike'].values, df['ivol'].values, T.values)
            df['gamma']  = g
            is_call = df['put_call'].str.upper().str.startswith('C')
            df['is_call'] = is_call
            df['gex_bn'] = np.where(
                is_call,
                 g * df['open_int'] * 100 * spot / 1e9,
                -g * df['open_int'] * 100 * spot / 1e9,
            )

            # Salva chain raw por ticker
            df.to_csv(OUT / f'chain_{short}_{hoje}.csv')

            # Agrega
            gex_total = float(df['gex_bn'].sum())
            gex_call  = float(df[is_call]['gex_bn'].sum())
            gex_put   = float(df[~is_call]['gex_bn'].sum())

            # Walls: strike com max OI por put/call
            calls = df[is_call]
            puts  = df[~is_call]
            call_wall = float(calls.loc[calls['open_int'].idxmax(), 'strike']) if not calls.empty else 0.0
            put_wall  = float(puts.loc[puts['open_int'].idxmax(), 'strike'])  if not puts.empty  else 0.0

            # Gamma flip: zero-crossing do GEX cumulativo por strike
            gex_by_strike = df.groupby('strike')['gex_bn'].sum().sort_index()
            gamma_flip = 0.0
            cum = gex_by_strike.cumsum()
            sign_change = (cum.shift() * cum) < 0
            if sign_change.any():
                gamma_flip = float(cum[sign_change].index[0])

            # P/C OI ratio
            pc_oi = float(puts['open_int'].sum() / calls['open_int'].sum()) if not calls.empty and calls['open_int'].sum() > 0 else 0.0

            # OI agregados (volume por contrato não disponível via bq.data)
            total_call_oi  = float(calls['open_int'].sum())
            total_put_oi   = float(puts['open_int'].sum())

            summary_rows.append({
                'ticker':         short,
                'bbg':            bbg_und,
                'spot':           round(spot, 4),
                'gex_total_bn':   round(gex_total, 4),
                'gex_call_bn':    round(gex_call,  4),
                'gex_put_bn':     round(gex_put,   4),
                'call_wall':      round(call_wall, 2),
                'put_wall':       round(put_wall,  2),
                'gamma_flip':     round(gamma_flip, 2),
                'pc_oi':          round(pc_oi, 4),
                'total_call_oi':  int(total_call_oi),
                'total_put_oi':   int(total_put_oi),
                'n_contracts':    len(df),
                'direction':      'long' if gex_total > 0 else 'short',
                'gamma_regime':   'positive' if gex_total > 0 else 'negative',
            })
            _log(f'greeks_full {short}: GEX={gex_total:+.2f}B chain={len(df)}')
        except Exception as e:
            _log(f'greeks_full warn {short}: {e}')

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(OUT / f'greeks_per_ticker_{hoje}.csv', index=False)
        _log(f'greeks_per_ticker — {len(summary_rows)} underlyings')


def export_iv_term_structure():
    """
    IV ATM por vencimento (term structure) — 30d, 60d, 90d, 180d, 360d.
    Permite detectar contango / backwardation, vol curve steepness.
    """
    universe = list(set(FUND_TICKERS))  # equity ETFs + mega-caps
    univ  = bq.univ.list(universe)
    items = {
        'iv_30d':  bq.data.implied_volatility(expiry='30D',  pct_moneyness='100'),
        'iv_60d':  bq.data.implied_volatility(expiry='60D',  pct_moneyness='100'),
        'iv_90d':  bq.data.implied_volatility(expiry='90D',  pct_moneyness='100'),
        'iv_180d': bq.data.implied_volatility(expiry='180D', pct_moneyness='100'),
        'iv_360d': bq.data.implied_volatility(expiry='360D', pct_moneyness='100'),
    }
    df = _bql(univ, items)
    for c in ['iv_30d', 'iv_60d', 'iv_90d', 'iv_180d', 'iv_360d']:
        df[c] = pd.to_numeric(df[c], errors='coerce') / 100
    df['contango_60_30']  = df['iv_60d']  - df['iv_30d']
    df['contango_90_30']  = df['iv_90d']  - df['iv_30d']
    df['contango_180_30'] = df['iv_180d'] - df['iv_30d']
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity', '', regex=False).str.replace('/', '-')
    df.to_csv(OUT / f'iv_term_{hoje}.csv')
    _log(f'iv_term — {len(df)} tickers')


def export_skew_tails():
    """
    Skew completo: 25-delta E 10-delta (tails finos).
    25d = skew "normal", 10d = tail risk verdadeiro.
    """
    universe = list(set(FUND_TICKERS))
    univ  = bq.univ.list(universe)
    items = {
        'put25':  bq.data.implied_volatility(expiry='30D', delta='25', put_call='PUT'),
        'call25': bq.data.implied_volatility(expiry='30D', delta='25', put_call='CALL'),
        'put10':  bq.data.implied_volatility(expiry='30D', delta='10', put_call='PUT'),
        'call10': bq.data.implied_volatility(expiry='30D', delta='10', put_call='CALL'),
    }
    df = _bql(univ, items)
    for c in ['put25', 'call25', 'put10', 'call10']:
        df[c] = pd.to_numeric(df[c], errors='coerce') / 100
    df['skew_25d'] = df['put25'] - df['call25']
    df['skew_10d'] = df['put10'] - df['call10']
    df['tail_premium'] = df['skew_10d'] - df['skew_25d']  # quanto a tail é mais cara que o skew normal
    df = df[['put25', 'call25', 'skew_25d', 'put10', 'call10', 'skew_10d', 'tail_premium']]
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity', '', regex=False).str.replace('/', '-')
    df.to_csv(OUT / f'skew_tails_{hoje}.csv')
    _log(f'skew_tails — {len(df)} tickers')


def export_volume_flows():
    """
    Volume + dollar volume + short interest + ETF flows (creation/redemption).
    Usa _safe_items: pula fields que não existem no BQL desta versão.
    """
    universe = list(set(FUND_TICKERS))
    univ  = bq.univ.list(universe)
    items = _safe_items([
        ('volume',          lambda: bq.data.px_volume()),
        ('price',           lambda: bq.data.px_last()),
        # Short interest — variantes possíveis
        ('short_int',       lambda: bq.data.short_int()),
        ('short_int_ratio', lambda: bq.data.short_int_ratio()),
        ('si_pct_float',    lambda: bq.data.short_int_ratio_pct_of_float()),
        ('days_to_cover',   lambda: bq.data.short_int_days_to_cover()),
        # ETF flows
        ('fund_flow',       lambda: bq.data.fund_net_creation()),
    ])
    if not items:
        _log('volume_flows: nenhum field disponível')
        return
    try:
        df = _bql(univ, items)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        if 'volume' in df.columns and 'price' in df.columns:
            df['dollar_volume'] = df['volume'] * df['price']
        for c in ('short_int', 'short_int_ratio', 'si_pct_float',
                  'days_to_cover', 'fund_flow'):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df.index.name = 'ticker'
        df.index = df.index.str.replace(' US Equity', '', regex=False).str.replace('/', '-')
        df.to_csv(OUT / f'volume_flows_{hoje}.csv')
        _log(f'volume_flows — {len(df)} tickers, fields: {list(df.columns)}')
    except Exception as e:
        _log(f'volume_flows warn: {e}')


def export_earnings_calendar():
    """
    Próximas datas de earnings + estimates por mega-cap.
    Usa _safe_items: pula fields que não existem nesta versão BQL.
    """
    universe = list(set(MEGA_CAP_TICKERS))
    items = _safe_items([
        ('next_dt', lambda: bq.data.next_announcement_date()),
        ('eps_est', lambda: bq.data.eps_estimate()),
        ('rev_est', lambda: bq.data.sales_estimate()),
        ('growth',  lambda: bq.data.eps_growth_estimate()),
    ])
    if not items:
        _log('earnings_calendar: nenhum field disponível')
        return
    rows = []
    for bbg_tk in universe:
        try:
            r = bq.execute(bql.Request(bq.univ.list([bbg_tk]), items))
            df_e = pd.concat([x.df()[x.name] for x in r], axis=1)
            ticker_short = bbg_tk.replace(' US Equity', '').replace('/', '-')
            if len(df_e) > 0:
                row = df_e.iloc[0]
                rows.append({
                    'ticker':         ticker_short,
                    'next_earn_date': str(row.get('next_dt', ''))[:10] if 'next_dt' in items else '',
                    'eps_estimate':   _to_num(row.get('eps_est'))      if 'eps_est' in items else '',
                    'rev_estimate':   _to_num(row.get('rev_est'))      if 'rev_est' in items else '',
                    'eps_growth_est': _to_num(row.get('growth'))       if 'growth'  in items else '',
                })
        except Exception as e:
            _log(f'earnings warn {bbg_tk}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'earnings_calendar_{hoje}.csv', index=False)
        _log(f'earnings_calendar — {len(rows)} mega-caps')


def export_index_members():
    """
    Membros + pesos dos índices principais (SPX, NDX, RUT).
    Permite top contributors, breadth, concentration analysis.
    """
    INDICES = [('SPX Index', 'SPX'), ('NDX Index', 'NDX'), ('RTY Index', 'RUT')]
    rows = []
    for bbg_idx, short in INDICES:
        try:
            members_univ = bq.univ.members(bbg_idx)
            r = bq.execute(bql.Request(members_univ, {
                'wt':    bq.data.id_index_weight(),
                'mcap':  bq.data.cur_mkt_cap(),
                'price': bq.data.px_last(),
            }))
            df_m = pd.concat([x.df()[x.name] for x in r], axis=1)
            df_m = df_m.loc[:, ~df_m.columns.duplicated()]
            df_m['index'] = short
            df_m['wt']    = pd.to_numeric(df_m['wt'],    errors='coerce') / 100
            df_m['mcap']  = pd.to_numeric(df_m['mcap'],  errors='coerce') / 1e9
            df_m.index.name = 'member'
            for member, row in df_m.iterrows():
                rows.append({
                    'index':    short,
                    'member':   member.replace(' US Equity', '').replace('/', '-'),
                    'weight':   _to_num(row.get('wt')),
                    'mcap_b':   _to_num(row.get('mcap')),
                    'price':    _to_num(row.get('price')),
                })
            _log(f'index_members {short}: {len(df_m)}')
        except Exception as e:
            _log(f'index_members warn {short}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'index_members_{hoje}.csv', index=False)
        _log(f'index_members — {len(rows)} linhas')


def export_etf_holdings():
    """Top holdings dos sector ETFs principais (SPDR + Nasdaq sectoriais)."""
    target_etfs = [f'{tk} US Equity' for tk, _ in SECTOR_ETFS if tk.startswith('XL')] + [
        'SOXX US Equity', 'IGV US Equity', 'IBB US Equity', 'QCLN US Equity',
        'GDX US Equity',  'GDXJ US Equity',
    ]
    rows = []
    for bbg_etf in target_etfs:
        short = bbg_etf.replace(' US Equity', '')
        try:
            r = bq.execute(bql.Request(bq.univ.list([bbg_etf]), {
                'top_holdings': bq.data.fund_top_holdings(),
            }))
            df_h = r[0].df()
            for idx, row in df_h.iterrows():
                rows.append({
                    'etf':      short,
                    'holding':  str(idx).replace(' US Equity', '').replace('/', '-'),
                    'value':    str(row.iloc[0]) if len(row) > 0 else '',
                })
        except Exception as e:
            _log(f'holdings warn {short}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'etf_holdings_{hoje}.csv', index=False)
        _log(f'etf_holdings — {len(rows)} linhas')


def export_borrow_rate():
    """Borrow rate / cost to borrow para mega-caps (squeeze risk)."""
    universe = list(set(MEGA_CAP_TICKERS))
    universe = list(set(MEGA_CAP_TICKERS))
    items = _safe_items([
        ('borrow',   lambda: bq.data.eqy_short_int_rate()),
        ('sl_avail', lambda: bq.data.sec_lend_avail()),
    ])
    if not items:
        _log('borrow_rate: nenhum field disponível')
        return
    rows = []
    for bbg_tk in universe:
        try:
            r = bq.execute(bql.Request(bq.univ.list([bbg_tk]), items))
            df_b = pd.concat([x.df()[x.name] for x in r], axis=1)
            if len(df_b) > 0:
                row = df_b.iloc[0]
                rows.append({
                    'ticker':       bbg_tk.replace(' US Equity', '').replace('/', '-'),
                    'borrow_rate':  _to_num(row.get('borrow'))   if 'borrow'   in items else '',
                    'sl_available': _to_num(row.get('sl_avail')) if 'sl_avail' in items else '',
                })
        except Exception as e:
            _log(f'borrow warn {bbg_tk}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'borrow_rate_{hoje}.csv', index=False)
        _log(f'borrow_rate — {len(rows)} mega-caps')


def export_dividends():
    """Próximas datas de dividendos (gap risk)."""
    universe = list(set(FUND_TICKERS))
    items = _safe_items([
        ('next_div', lambda: bq.data.next_dvd_pay_dt()),
        ('ex_div',   lambda: bq.data.dvd_ex_dt()),
        ('div_amt',  lambda: bq.data.dvd_sh_last()),
        ('yld_ind',  lambda: bq.data.eqy_dvd_yld_ind()),
    ])
    if not items:
        _log('dividends: nenhum field disponível')
        return
    rows = []
    for bbg_tk in universe:
        try:
            r = bq.execute(bql.Request(bq.univ.list([bbg_tk]), items))
            df_d = pd.concat([x.df()[x.name] for x in r], axis=1)
            if len(df_d) > 0:
                row = df_d.iloc[0]
                rows.append({
                    'ticker':         bbg_tk.replace(' US Equity', '').replace('/', '-'),
                    'next_div_date':  str(row.get('next_div', ''))[:10] if 'next_div' in items else '',
                    'ex_div_date':    str(row.get('ex_div', ''))[:10]   if 'ex_div'   in items else '',
                    'div_amount':     _to_num(row.get('div_amt'))       if 'div_amt'  in items else '',
                    'div_yield':      _to_num(row.get('yld_ind'))       if 'yld_ind'  in items else '',
                })
        except Exception as e:
            _log(f'div warn {bbg_tk}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'dividends_{hoje}.csv', index=False)
        _log(f'dividends — {len(rows)} tickers')


def export_eps_revisions():
    """EPS estimate revisions (momentum fundamental)."""
    universe = list(set(MEGA_CAP_TICKERS) | set(FUND_TICKERS))
    items = _safe_items([
        ('eps_est',    lambda: bq.data.eps_estimate()),
        ('eps_3m_ago', lambda: bq.data.eps_estimate(dates='-90D')),
        ('eps_up',     lambda: bq.data.eps_est_up_count_30d()),
        ('eps_down',   lambda: bq.data.eps_est_down_count_30d()),
    ])
    if not items:
        _log('eps_revisions: nenhum field disponível')
        return
    rows = []
    for bbg_tk in universe:
        try:
            r = bq.execute(bql.Request(bq.univ.list([bbg_tk]), items))
            df_r = pd.concat([x.df()[x.name] for x in r], axis=1)
            if len(df_r) > 0:
                row = df_r.iloc[0]
                eps_now = _to_num(row.get('eps_est'))    if 'eps_est'    in items else ''
                eps_old = _to_num(row.get('eps_3m_ago')) if 'eps_3m_ago' in items else ''
                rev_pct = ''
                try:
                    if eps_now != '' and eps_old != '' and eps_old != 0:
                        rev_pct = round((eps_now - eps_old) / eps_old, 6)
                except Exception:
                    pass
                rows.append({
                    'ticker':       bbg_tk.replace(' US Equity', '').replace('/', '-'),
                    'eps_est':      eps_now,
                    'eps_3m_ago':   eps_old,
                    'eps_rev_3m':   rev_pct,
                    'eps_up_30d':   _to_num(row.get('eps_up'))   if 'eps_up'   in items else '',
                    'eps_down_30d': _to_num(row.get('eps_down')) if 'eps_down' in items else '',
                })
        except Exception as e:
            _log(f'eps_rev warn {bbg_tk}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'eps_revisions_{hoje}.csv', index=False)
        _log(f'eps_revisions — {len(rows)} tickers')


def export_realized_vol():
    """RV (realized volatility) 30d, 60d, 90d, 252d via px_volatility BQL."""
    universe = list(set(FUND_TICKERS))
    univ  = bq.univ.list(universe)
    items = _safe_items([
        ('rv_30d',  lambda: bq.data.px_volatility(period='30D')),
        ('rv_60d',  lambda: bq.data.px_volatility(period='60D')),
        ('rv_90d',  lambda: bq.data.px_volatility(period='90D')),
        ('rv_252d', lambda: bq.data.px_volatility(period='260D')),
    ])
    if not items:
        _log('realized_vol: nenhum field disponível')
        return
    try:
        df = _bql(univ, items)
        for c in list(items.keys()):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce') / 100
        df.index.name = 'ticker'
        df.index = df.index.str.replace(' US Equity', '', regex=False).str.replace('/', '-')
        df.to_csv(OUT / f'realized_vol_{hoje}.csv')
        _log(f'realized_vol — {len(df)} tickers')
    except Exception as e:
        _log(f'realized_vol warn: {e}')


# ═══════════════════════════════════════════════════════════════════════════


def export_letf():
    items = {
        'nav':      bq.data.px_last(),
        'nav_prev': bq.data.px_last(dates=bq.func.range('-5D', '-1D'), fill='PREV'),
        'aum_b':    bq.data.fund_total_assets(),
    }
    df = _bql(bq.univ.list(LETFS), items)
    df['aum_b'] = pd.to_numeric(df['aum_b'], errors='coerce') / 1e9
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity', '', regex=False)
    df['leverage'] = df.index.map({'UPRO': 3, 'SPXU': -3, 'TQQQ': 3, 'SQQQ': -3, 'TNA': 3, 'TZA': -3})
    df['index']    = df.index.map(lambda x: 'SPX' if x in ['UPRO', 'SPXU']
                                            else 'NDX' if x in ['TQQQ', 'SQQQ']
                                            else 'RUT')
    df.to_csv(OUT / f'letf_flows_{hoje}.csv')
    _log(f'letf_flows — {len(df)} linhas')


def export_prices():
    """
    Snapshot atual de preços + retornos para todos os tickers.

    Como bq.data.chg_pct_* não existe no BQL, computamos os retornos
    client-side puxando uma série de px_last (-260D até hoje) por ticker
    e calculando daily/weekly/ytd em pandas.
    """
    bbg_tickers = list(_YF_TO_BBG.values())
    yf_by_bbg   = {v: k for k, v in _YF_TO_BBG.items()}
    rows = []
    today_dt = pd.Timestamp.today().normalize()
    year_start = pd.Timestamp(today_dt.year, 1, 1)

    for bbg_tk in bbg_tickers:
        yf_tk = yf_by_bbg.get(bbg_tk, bbg_tk.split()[0])
        try:
            resp = bq.execute(
                f'get(PX_LAST(dates=range(-260D,0D),frq=D,fill=PREV)) for(["{bbg_tk}"])'
            )
            df_h = resp[0].df()
            df_h.columns = ['price']
            df_h = df_h.dropna()
            if df_h.empty:
                continue
            df_h.index = pd.to_datetime(df_h.index)
            df_h = df_h.sort_index()
            prices = df_h['price'].astype(float)

            p_now = float(prices.iloc[-1])

            # Daily return: vs último fechamento anterior
            dr = ''
            if len(prices) >= 2:
                p_prev = float(prices.iloc[-2])
                if p_prev > 0:
                    dr = round((p_now - p_prev) / p_prev, 6)

            # Weekly return: vs ~5 sessões atrás
            wr = ''
            if len(prices) >= 6:
                p_5 = float(prices.iloc[-6])
                if p_5 > 0:
                    wr = round((p_now - p_5) / p_5, 6)

            # YTD return: vs último preço antes do início do ano
            ydr = ''
            ytd_slice = prices[prices.index < year_start]
            if not ytd_slice.empty:
                p_ys = float(ytd_slice.iloc[-1])
                if p_ys > 0:
                    ydr = round((p_now - p_ys) / p_ys, 6)

            rows.append({
                'yf_ticker':     yf_tk,
                'bbg_ticker':    bbg_tk,
                'name':          bbg_tk.split()[0],
                'price':         round(p_now, 4),
                'daily_return':  dr,
                'weekly_return': wr,
                'ytd_return':    ydr,
            })
        except Exception as e:
            _log(f'prices warn {yf_tk}: {e}')

    pd.DataFrame(rows).to_csv(OUT / f'prices_{hoje}.csv', index=False)
    _log(f'prices — {len(rows)} tickers')


def export_price_history():
    """Snapshot do dia — banco acumula o histórico."""
    rows = []
    for yf_tk, bbg_tk in _YF_TO_BBG.items():
        try:
            r  = bq.execute(bql.Request(bq.univ.list([bbg_tk]), {'p': bq.data.px_last()}))[0].df()
            px = float(r.select_dtypes('number').iloc[-1, 0])
            rows.append({'date': hoje, 'yf_ticker': yf_tk, 'price': round(px, 4)})
        except Exception as e:
            _log(f'hist warn {yf_tk}: {e}')
    pd.DataFrame(rows).to_csv(OUT / f'price_history_{hoje}.csv', index=False)
    _log(f'price_history — {len(rows)} tickers')


def export_price_history_bulk():
    """252 dias de histórico para todos os tickers — para correlações de rede."""
    rows = []
    for yf_tk, bbg_tk in _YF_TO_BBG.items():
        try:
            resp = bq.execute(
                f'get(PX_LAST(dates=range(-252D,0D),frq=D,fill=PREV)) for(["{bbg_tk}"])'
            )
            df2 = resp[0].df()
            df2.columns = ['price']
            df2 = df2.dropna()
            df2.index = pd.to_datetime(df2.index)
            for dt, row in df2.iterrows():
                px = float(row['price'])
                if px > 0:
                    rows.append({'date': dt.date().isoformat(), 'yf_ticker': yf_tk, 'price': round(px, 4)})
        except Exception as e:
            _log(f'hist_bulk warn {yf_tk}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'price_history_bulk_{hoje}.csv', index=False)
        _log(f'price_history_bulk — {len(rows)} linhas')


def export_iv_history():
    """252 dias de IV ATM para calcular iv_percentile."""
    universe_iv = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'META', 'GOOGL', 'AMZN', 'AVGO',
                   'JPM', 'XLF', 'XLE', 'XOM', 'GLD', 'TLT', 'IEF',
                   'HYG', 'EEM', 'VIXY', 'SPY', 'QQQ', 'IWM']
    rows = []
    for yf_tk in universe_iv:
        bbg_tk = _YF_TO_BBG.get(yf_tk)
        if not bbg_tk:
            continue
        try:
            resp = bq.execute(
                f'get(IVOL_MID_ATM(expiry="30D", dates=range(-252D,0D), frq=D)) for(["{bbg_tk}"])'
            )
            df2 = resp[0].df()
            df2.columns = ['iv']
            df2 = df2.dropna()
            df2['iv'] = pd.to_numeric(df2['iv'], errors='coerce') / 100
            df2.index = pd.to_datetime(df2.index)
            for dt, row in df2.iterrows():
                iv = float(row['iv'])
                if iv > 0:
                    rows.append({'date': dt.date().isoformat(), 'yf_ticker': yf_tk, 'iv': round(iv, 4)})
        except Exception as e:
            _log(f'iv_hist warn {yf_tk}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'iv_history_{hoje}.csv', index=False)
        _log(f'iv_history — {len(rows)} linhas')


def export_macro():
    """Curva de juros, vol, spreads, FX. Inclui derivados (term-structure, spreads)."""
    rows   = []
    px_map = {}
    for tk, desc, cat in MACRO_TICKERS:
        try:
            r  = bq.execute(bql.Request(bq.univ.list([tk]), {'px': bq.data.px_last()}))[0].df()
            px = float(r.select_dtypes('number').iloc[-1, 0])
            rows.append({'bbg_ticker': tk, 'description': desc,
                         'category': cat, 'px_last': round(px, 4)})
            px_map[tk] = px
        except Exception as e:
            _log(f'macro warn {tk}: {e}')

    y2  = px_map.get('USGG2YR Index')
    y5  = px_map.get('USGG5YR Index')
    y10 = px_map.get('USGG10YR Index')
    y30 = px_map.get('USGG30YR Index')
    vix = px_map.get('VIX Index')
    v9d = px_map.get('VIX9D Index')
    v3m = px_map.get('VIX3M Index')

    if y2 and y10:
        rows.append({'bbg_ticker': 'US_2Y10Y_SPREAD', 'description': '2Y-10Y Spread',
                     'category': 'rates_derived', 'px_last': round(y10 - y2, 4)})
    if y5 and y30:
        rows.append({'bbg_ticker': 'US_5Y30Y_SPREAD', 'description': '5Y-30Y Spread',
                     'category': 'rates_derived', 'px_last': round(y30 - y5, 4)})
    if vix and v9d and vix > 0:
        rows.append({'bbg_ticker': 'VIX_TERM_9D_SP', 'description': 'VIX 9D/Spot ratio',
                     'category': 'volatility_derived', 'px_last': round(v9d / vix, 4)})
    if vix and v3m and vix > 0:
        rows.append({'bbg_ticker': 'VIX_TERM_3M_SP', 'description': 'VIX 3M/Spot ratio',
                     'category': 'volatility_derived', 'px_last': round(v3m / vix, 4)})

    pd.DataFrame(rows).to_csv(OUT / f'macro_series_{hoje}.csv', index=False)
    _log(f'macro_series — {len(rows)} séries')


def export_meta():
    pd.DataFrame([{'generated_at': datetime.now().isoformat()}])\
      .to_csv(OUT / f'meta_{hoje}.csv', index=False)


# ── Auto-download via browser ─────────────────────────────────────────────

def auto_download():
    """Empacota CSVs do dia e dispara download no browser."""
    arquivos = sorted(OUT.glob(f'*_{hoje}*.csv'))
    if not arquivos:
        return
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in arquivos:
            zf.write(f, f.name)
    buf.seek(0)
    b64  = base64.b64encode(buf.read()).decode()
    nome = f'bql_data_{hoje}.zip'
    uid  = str(int(time.time()))
    display(HTML(
        f'<a id="dl{uid}" href="data:application/zip;base64,{b64}" download="{nome}"></a>'
        f'<script>document.getElementById("dl{uid}").click();</script>'
    ))
    _log(f'Download: {nome} ({len(arquivos)} arquivos)')


# ── Ciclos completos ──────────────────────────────────────────────────────

def export_all():
    """Snapshot do dia: TUDO via BQL — equity, bond, FX, commodity, options, macro."""
    print(f'\n=== [{time.strftime("%H:%M:%S")}] Export ===')

    # ── Fundamentals snapshot ─────────────────────────────────────────────
    print('Fundamentais (equity)...');     export_fundamentals()
    print('Bond ETFs...');                 export_bond_etf_fundamentals()
    print('FX ETFs...');                   export_fx_etf_fundamentals()
    print('Commodity / Vol ETFs...');      export_commodity_etf_fundamentals()

    # ── Options chain & greeks ───────────────────────────────────────────
    print('Options IV ATM 30d...');        export_options_iv()
    print('IV term structure 30/60/90/180/360...'); export_iv_term_structure()
    print('Skew 25d + 10d (tails)...');    export_skew_tails()
    print('GEX SPX...')
    try:
        export_gex_spx()
    except Exception as e:
        _log(f'GEX warn: {e}')
    print('Greeks per ticker (mega-caps)...')
    try:
        export_options_greeks_full()
    except Exception as e:
        _log(f'greeks_full warn: {e}')

    # ── Volume / fluxos / short interest ─────────────────────────────────
    print('Volume + dark flows + short interest...'); export_volume_flows()

    # ── LETF / earnings / borrow / dividends / revisions ──────────────────
    print('LETF flows...');                export_letf()
    print('Earnings calendar...');         export_earnings_calendar()
    print('Borrow rate (mega-caps)...');   export_borrow_rate()
    print('Dividends...');                 export_dividends()
    print('EPS revisions...');             export_eps_revisions()

    # ── Prices / macro ───────────────────────────────────────────────────
    print('Realized vol (30/60/90/252d)...'); export_realized_vol()
    print('Prices snapshot...');           export_prices()
    print('Price history (snapshot)...');  export_price_history()
    print('Macro series...');              export_macro()

    export_meta()
    auto_download()
    print('Pronto.')


def export_all_bulk():
    """Tudo + 252 dias de histórico (preços + IV + fundamentals + bonds + index members). ~10 min."""
    print(f'\n=== BULK [{time.strftime("%H:%M:%S")}] ===')

    # Snapshot
    export_all()

    # ── Histórico 252d ────────────────────────────────────────────────────
    print('Fundamentals history (252d)...');   export_fundamentals_history()
    print('Bond ETF history (252d)...');       export_bond_etf_history()
    print('Price history bulk (252d)...');     export_price_history_bulk()
    print('IV history (252d)...');             export_iv_history()

    # ── Estrutura de mercado (members + holdings) ─────────────────────────
    print('Index members + weights (SPX/NDX/RUT)...'); export_index_members()
    print('ETF holdings (sectors)...');                export_etf_holdings()

    export_meta()
    auto_download()
    print('Bulk pronto.')


# ── UI dos botões ─────────────────────────────────────────────────────────

_running = False

def _loop():
    while _running:
        export_all()
        time.sleep(INTERVAL)

btn_run  = widgets.Button(description='⬇ Exportar agora', button_style='success',
                          layout=widgets.Layout(width='160px'))
btn_bulk = widgets.Button(description='⬇ Bulk 252d', button_style='warning',
                          layout=widgets.Layout(width='130px'),
                          tooltip='Exporta histórico completo (~5 min).')
btn_loop = widgets.ToggleButton(description='▶ Loop 3min', button_style='info',
                                layout=widgets.Layout(width='120px'))
out_w    = widgets.Output()

def on_run(_):
    with out_w:
        export_all()

def on_bulk(_):
    with out_w:
        export_all_bulk()

def on_loop(change):
    global _running
    _running = change['new']
    btn_loop.description = '⏹ Parar' if _running else '▶ Loop 3min'
    if _running:
        threading.Thread(target=_loop, daemon=True).start()

btn_run.on_click(on_run)
btn_bulk.on_click(on_bulk)
btn_loop.observe(on_loop, names='value')

display(widgets.HBox([btn_run, btn_bulk, btn_loop]), out_w)
