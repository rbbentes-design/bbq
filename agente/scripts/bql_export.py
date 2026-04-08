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
_PROJECT_OUT = Path(r"C:\Users\rafael bentes\bbg\agente\bql_data")
OUT          = _PROJECT_OUT if _PROJECT_OUT.parent.exists() else (Path.home() / "bql_data")
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


def export_bond_etf_fundamentals():
    """
    Snapshot de bond ETFs: yield, effective duration, expense ratio, AUM, NAV.
    Cobre toda a duration ladder (BIL→TLT→EDV) + credit + TIPS.
    """
    univ  = bq.univ.list(RATES_FUND_TICKERS)
    items = {
        'PX_LAST':              bq.data.px_last(),
        'EXPENSE_RATIO':        bq.data.fund_expense_ratio(),
        'FUND_TOTAL_ASSETS':    bq.data.fund_total_assets(),
        # Bond-specific
        'YIELD':                bq.data.yield_to_maturity(),  # YTM agregado da carteira
        'EFFECTIVE_DURATION':   bq.data.fund_effective_duration(),
        'AVG_MATURITY':         bq.data.fund_avg_maturity(),
        'OAS_SPREAD':           bq.data.oas_spread_to_govt(),
    }
    df = _bql(univ, items)
    df.rename(columns={
        'PX_LAST':            'price',
        'EXPENSE_RATIO':      'expense_ratio',
        'FUND_TOTAL_ASSETS':  'aum_b',
        'YIELD':              'yield',
        'EFFECTIVE_DURATION': 'duration',
        'AVG_MATURITY':       'avg_maturity',
        'OAS_SPREAD':         'oas',
    }, inplace=True)
    df['aum_b']         = pd.to_numeric(df['aum_b'],         errors='coerce') / 1e9
    df['expense_ratio'] = pd.to_numeric(df['expense_ratio'], errors='coerce') / 100
    df['yield']         = pd.to_numeric(df['yield'],         errors='coerce') / 100
    df['oas']           = pd.to_numeric(df['oas'],           errors='coerce')
    df['label']    = df.index.map(NON_EQUITY_LABELS)
    df['category'] = df.index.map(NON_EQUITY_CATEGORY)
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity', '', regex=False)
    df.to_csv(OUT / f'bond_etf_fundamentals_{hoje}.csv')
    _log(f'bond_etf_fundamentals — {len(df)} linhas')


def export_fx_etf_fundamentals():
    """Snapshot de FX ETFs: NAV, expense ratio, AUM."""
    univ  = bq.univ.list(FX_FUND_TICKERS)
    items = {
        'PX_LAST':             bq.data.px_last(),
        'EXPENSE_RATIO':       bq.data.fund_expense_ratio(),
        'FUND_TOTAL_ASSETS':   bq.data.fund_total_assets(),
        'CHG_PCT_YTD':         bq.data.chg_pct_ytd(),
    }
    df = _bql(univ, items)
    df.rename(columns={
        'PX_LAST':           'price',
        'EXPENSE_RATIO':     'expense_ratio',
        'FUND_TOTAL_ASSETS': 'aum_b',
        'CHG_PCT_YTD':       'ytd_return',
    }, inplace=True)
    df['aum_b']         = pd.to_numeric(df['aum_b'],         errors='coerce') / 1e9
    df['expense_ratio'] = pd.to_numeric(df['expense_ratio'], errors='coerce') / 100
    df['ytd_return']    = pd.to_numeric(df['ytd_return'],    errors='coerce') / 100
    df['label']    = df.index.map(NON_EQUITY_LABELS)
    df['category'] = df.index.map(NON_EQUITY_CATEGORY)
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity', '', regex=False)
    df.to_csv(OUT / f'fx_etf_fundamentals_{hoje}.csv')
    _log(f'fx_etf_fundamentals — {len(df)} linhas')


def export_commodity_etf_fundamentals():
    """Snapshot de commodity ETFs: NAV, expense ratio, AUM, retorno YTD."""
    univ  = bq.univ.list(COMMODITY_FUND_TICKERS + VOL_FUND_TICKERS)
    items = {
        'PX_LAST':             bq.data.px_last(),
        'EXPENSE_RATIO':       bq.data.fund_expense_ratio(),
        'FUND_TOTAL_ASSETS':   bq.data.fund_total_assets(),
        'CHG_PCT_YTD':         bq.data.chg_pct_ytd(),
        'CHG_PCT_1D':          bq.data.chg_pct_1d(),
    }
    df = _bql(univ, items)
    df.rename(columns={
        'PX_LAST':           'price',
        'EXPENSE_RATIO':     'expense_ratio',
        'FUND_TOTAL_ASSETS': 'aum_b',
        'CHG_PCT_YTD':       'ytd_return',
        'CHG_PCT_1D':        'daily_return',
    }, inplace=True)
    df['aum_b']         = pd.to_numeric(df['aum_b'],         errors='coerce') / 1e9
    df['expense_ratio'] = pd.to_numeric(df['expense_ratio'], errors='coerce') / 100
    df['ytd_return']    = pd.to_numeric(df['ytd_return'],    errors='coerce') / 100
    df['daily_return']  = pd.to_numeric(df['daily_return'],  errors='coerce') / 100
    df['label']    = df.index.map(NON_EQUITY_LABELS)
    df['category'] = df.index.map(NON_EQUITY_CATEGORY)
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity', '', regex=False)
    df.to_csv(OUT / f'commodity_etf_fundamentals_{hoje}.csv')
    _log(f'commodity_etf_fundamentals — {len(df)} linhas')


def export_bond_etf_history():
    """
    Histórico de 252 dias de yield + duration dos bond ETFs.
    Permite computar yield curve over time, duration drift, OAS percentile.
    """
    rows = []
    for bbg_tk in RATES_FUND_TICKERS:
        label = NON_EQUITY_LABELS.get(bbg_tk, '')
        cat   = NON_EQUITY_CATEGORY.get(bbg_tk, '')
        try:
            resp = bq.execute(
                f'get('
                f'YIELD_TO_MATURITY(dates=range(-252D,0D),frq=D,fill=PREV),'
                f'FUND_EFFECTIVE_DURATION(dates=range(-252D,0D),frq=D,fill=PREV),'
                f'OAS_SPREAD_TO_GOVT(dates=range(-252D,0D),frq=D,fill=PREV)'
                f') for(["{bbg_tk}"])'
            )
            frames = []
            for r in resp:
                s = r.df()[r.name]
                s = s[~s.index.duplicated(keep='last')]
                frames.append(s.rename(r.name))
            df_h = pd.concat(frames, axis=1)
            df_h.index = pd.to_datetime(df_h.index)
            ticker_short = bbg_tk.replace(' US Equity', '')
            for dt, row in df_h.iterrows():
                yld = pd.to_numeric(row.get('YIELD_TO_MATURITY'),         errors='coerce')
                dur = pd.to_numeric(row.get('FUND_EFFECTIVE_DURATION'),   errors='coerce')
                oas = pd.to_numeric(row.get('OAS_SPREAD_TO_GOVT'),        errors='coerce')
                if pd.isna(yld) and pd.isna(dur):
                    continue
                rows.append({
                    'date':     dt.date().isoformat(),
                    'ticker':   ticker_short,
                    'label':    label,
                    'category': cat,
                    'yield':    round(float(yld)/100, 6) if not pd.isna(yld) else '',
                    'duration': round(float(dur),     4) if not pd.isna(dur) else '',
                    'oas':      round(float(oas),     4) if not pd.isna(oas) else '',
                })
        except Exception as e:
            _log(f'bond_hist warn {bbg_tk}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'bond_etf_history_{hoje}.csv', index=False)
        _log(f'bond_etf_history — {len(rows)} linhas ({len(RATES_FUND_TICKERS)} ETFs × 252d)')


def export_options_iv():
    univ  = bq.univ.list(FUND_TICKERS)
    items = {
        'atm_iv': bq.data.implied_volatility(expiry='30D', pct_moneyness='100'),
        'put25':  bq.data.implied_volatility(expiry='30D', delta='25', put_call='PUT'),
        'call25': bq.data.implied_volatility(expiry='30D', delta='25', put_call='CALL'),
        'pcr_oi': bq.data.put_call_open_interest_ratio(),
    }
    df = _bql(univ, items)
    df['atm_iv']   = pd.to_numeric(df['atm_iv'],  errors='coerce') / 100
    df['skew_25d'] = (pd.to_numeric(df['put25'], errors='coerce') -
                      pd.to_numeric(df['call25'], errors='coerce')) / 100
    df = df[['atm_iv', 'skew_25d', 'pcr_oi']]
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity', '', regex=False)
    df.to_csv(OUT / f'options_iv_{hoje}.csv')
    _log(f'options_iv — {len(df)} linhas')


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
    Usa CHG_PCT_1D (Bloomberg nativo) para evitar daily_return=0 fora do horário.
    """
    bbg_tickers = list(_YF_TO_BBG.values())
    yf_by_bbg   = {v: k for k, v in _YF_TO_BBG.items()}
    try:
        univ  = bq.univ.list(bbg_tickers)
        items = {
            'price':   bq.data.px_last(),
            'chg_1d':  bq.data.chg_pct_1d(),
            'chg_ytd': bq.data.chg_pct_ytd(),
            'chg_5d':  bq.data.chg_pct_5d(),
        }
        df = _bql(univ, items)
        df['chg_1d']  = pd.to_numeric(df['chg_1d'],  errors='coerce') / 100
        df['chg_ytd'] = pd.to_numeric(df['chg_ytd'], errors='coerce') / 100
        df['chg_5d']  = pd.to_numeric(df['chg_5d'],  errors='coerce') / 100
        rows = []
        for bbg_tk, row in df.iterrows():
            yf_tk = yf_by_bbg.get(bbg_tk, bbg_tk.split()[0])
            p = pd.to_numeric(row.get('price'), errors='coerce')
            if pd.isna(p):
                continue
            dr  = row.get('chg_1d')
            ydr = row.get('chg_ytd')
            wr  = row.get('chg_5d')
            rows.append({
                'yf_ticker':     yf_tk,
                'bbg_ticker':    bbg_tk,
                'name':          bbg_tk.split()[0],
                'price':         round(float(p), 4),
                'daily_return':  round(float(dr),  6) if not pd.isna(dr)  else '',
                'ytd_return':    round(float(ydr), 6) if not pd.isna(ydr) else '',
                'weekly_return': round(float(wr),  6) if not pd.isna(wr)  else '',
            })
    except Exception as e:
        _log(f'prices batch warn: {e}')
        # Fallback: coleta individual
        rows = []
        for yf_tk, bbg_tk in _YF_TO_BBG.items():
            try:
                r = bq.execute(bql.Request(bq.univ.list([bbg_tk]), {
                    'p':   bq.data.px_last(),
                    'dr':  bq.data.chg_pct_1d(),
                    'ydr': bq.data.chg_pct_ytd(),
                }))[0].df()
                p   = float(r.select_dtypes('number')['p'].iloc[-1])
                dr  = float(r.select_dtypes('number')['dr'].iloc[-1])  / 100
                ydr = float(r.select_dtypes('number')['ydr'].iloc[-1]) / 100
                rows.append({'yf_ticker': yf_tk, 'bbg_ticker': bbg_tk,
                             'name': bbg_tk.split()[0], 'price': round(p, 4),
                             'daily_return': round(dr, 6), 'ytd_return': round(ydr, 6)})
            except Exception as e2:
                _log(f'prices warn {yf_tk}: {e2}')
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
    """Snapshot do dia: fundamentais (equity + bond/fx/commodity), IV, GEX, LETF, prices, macro."""
    print(f'\n=== [{time.strftime("%H:%M:%S")}] Export ===')
    print('Fundamentais (equity)...');     export_fundamentals()
    print('Bond ETFs...');                 export_bond_etf_fundamentals()
    print('FX ETFs...');                   export_fx_etf_fundamentals()
    print('Commodity / Vol ETFs...');      export_commodity_etf_fundamentals()
    print('Options IV...');                export_options_iv()
    print('GEX SPX...')
    try:
        export_gex_spx()
    except Exception as e:
        _log(f'GEX warn: {e}')
    print('LETF...');                      export_letf()
    print('Prices...');                    export_prices()
    print('Price history...');             export_price_history()
    print('Macro series...');              export_macro()
    export_meta()
    auto_download()
    print('Pronto.')


def export_all_bulk():
    """Tudo + 252 dias de histórico (preços + IV + fundamentals + bonds). ~7 min."""
    print(f'\n=== BULK [{time.strftime("%H:%M:%S")}] ===')
    print('Fundamentais (equity)...');     export_fundamentals()
    print('Bond ETFs...');                 export_bond_etf_fundamentals()
    print('FX ETFs...');                   export_fx_etf_fundamentals()
    print('Commodity / Vol ETFs...');      export_commodity_etf_fundamentals()
    print('Options IV...');                export_options_iv()
    print('GEX SPX...')
    try:
        export_gex_spx()
    except Exception as e:
        _log(f'GEX warn: {e}')
    print('LETF...');                          export_letf()
    print('Prices...');                        export_prices()
    print('Price history...');                 export_price_history()
    print('Macro series...');                  export_macro()
    print('Fundamentals history (252d)...');   export_fundamentals_history()
    print('Bond ETF history (252d)...');       export_bond_etf_history()
    print('Price history bulk (252d)...');     export_price_history_bulk()
    print('IV history (252d)...');             export_iv_history()
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
