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

    # ══════ S&P 500 / NDX / RUT MEMBERS — restantes do node_registry ══════
    # Adiciona ~172 ações que estão no registry da rede neural mas
    # não eram puxadas pelo BQL (small/mid-caps + nomes setoriais).
    'ABBV': 'ABBV US Equity',   'ABNB': 'ABNB US Equity',   'ABT':  'ABT US Equity',
    'ACAD': 'ACAD US Equity',   'ACMR': 'ACMR US Equity',   'ADBE': 'ADBE US Equity',
    'AEP':  'AEP US Equity',    'AMAT': 'AMAT US Equity',   'AMCX': 'AMCX US Equity',
    'AMD':  'AMD US Equity',    'AMGN': 'AMGN US Equity',   'AMT':  'AMT US Equity',
    'APD':  'APD US Equity',    'ARWR': 'ARWR US Equity',   'ATI':  'ATI US Equity',
    'AVAV': 'AVAV US Equity',   'AWK':  'AWK US Equity',    'AXP':  'AXP US Equity',
    'BA':   'BA US Equity',     'BAC':  'BAC US Equity',    'BANF': 'BANF US Equity',
    'BIIB': 'BIIB US Equity',   'BKNG': 'BKNG US Equity',   'BOOT': 'BOOT US Equity',
    'BRT':  'BRT US Equity',    'CAT':  'CAT US Equity',    'CCI':  'CCI US Equity',
    'CDNS': 'CDNS US Equity',   'CEG':  'CEG US Equity',    'CENT': 'CENT US Equity',
    'CF':   'CF US Equity',     'CHTR': 'CHTR US Equity',   'CHUY': 'CHUY US Equity',
    'CL':   'CL US Equity',     'CMCSA':'CMCSA US Equity',  'COHU': 'COHU US Equity',
    'COP':  'COP US Equity',    'CRM':  'CRM US Equity',    'CTAS': 'CTAS US Equity',
    'CTVA': 'CTVA US Equity',   'CVBF': 'CVBF US Equity',   'CVX':  'CVX US Equity',
    'DBJP': 'DBJP US Equity',   'DE':   'DE US Equity',     'DHR':  'DHR US Equity',
    'DIS':  'DIS US Equity',    'DLR':  'DLR US Equity',    'DNLI': 'DNLI US Equity',
    'DOW':  'DOW US Equity',    'DUK':  'DUK US Equity',    'DXCM': 'DXCM US Equity',
    'EA':   'EA US Equity',     'ECL':  'ECL US Equity',    'ED':   'ED US Equity',
    'EE':   'EE US Equity',     'EOG':  'EOG US Equity',    'EQIX': 'EQIX US Equity',
    'ESAB': 'ESAB US Equity',   'ETN':  'ETN US Equity',    'EWG':  'EWG US Equity',
    'EWH':  'EWH US Equity',    'EWQ':  'EWQ US Equity',    'EWU':  'EWU US Equity',
    'EXC':  'EXC US Equity',    'FAST': 'FAST US Equity',   'FCX':  'FCX US Equity',
    'FEZ':  'FEZ US Equity',    'GE':   'GE US Equity',     'GILD': 'GILD US Equity',
    'GIS':  'GIS US Equity',    'GM':   'GM US Equity',     'GMS':  'GMS US Equity',
    'GOGO': 'GOGO US Equity',   'GS':   'GS US Equity',     'GTY':  'GTY US Equity',
    'HAIN': 'HAIN US Equity',   'HD':   'HD US Equity',     'HIMS': 'HIMS US Equity',
    'HON':  'HON US Equity',    'IDXX': 'IDXX US Equity',   'INGR': 'INGR US Equity',
    'INTC': 'INTC US Equity',   'ISRG': 'ISRG US Equity',   'KALU': 'KALU US Equity',
    'KHC':  'KHC US Equity',    'KLAC': 'KLAC US Equity',   'KLIC': 'KLIC US Equity',
    'KMB':  'KMB US Equity',    'KMI':  'KMI US Equity',    'KO':   'KO US Equity',
    'KTOS': 'KTOS US Equity',   'LGND': 'LGND US Equity',   'LIN':  'LIN US Equity',
    'LMT':  'LMT US Equity',    'LOW':  'LOW US Equity',    'LRCX': 'LRCX US Equity',
    'LULU': 'LULU US Equity',   'MCD':  'MCD US Equity',    'MCHI': 'MCHI US Equity',
    'MDLZ': 'MDLZ US Equity',   'MGEE': 'MGEE US Equity',   'MMM':  'MMM US Equity',
    'MNST': 'MNST US Equity',   'MO':   'MO US Equity',     'MPC':  'MPC US Equity',
    'MRK':  'MRK US Equity',    'MS':   'MS US Equity',     'MSGS': 'MSGS US Equity',
    'MTDR': 'MTDR US Equity',   'MU':   'MU US Equity',     'NBTB': 'NBTB US Equity',
    'NDAQ': 'NDAQ US Equity',   'NEE':  'NEE US Equity',    'NEM':  'NEM US Equity',
    'NKE':  'NKE US Equity',    'NUE':  'NUE US Equity',    'NXRT': 'NXRT US Equity',
    'O':    'O US Equity',      'ODFL': 'ODFL US Equity',   'OMG':  'OMG US Equity',
    'ON':   'ON US Equity',     'ORCL': 'ORCL US Equity',   'ORLY': 'ORLY US Equity',
    'OSIS': 'OSIS US Equity',   'OXY':  'OXY US Equity',    'PAYX': 'PAYX US Equity',
    'PEG':  'PEG US Equity',    'PEP':  'PEP US Equity',    'PLAY': 'PLAY US Equity',
    'PLD':  'PLD US Equity',    'PM':   'PM US Equity',     'PRGO': 'PRGO US Equity',
    'PSA':  'PSA US Equity',    'PSX':  'PSX US Equity',    'PYPL': 'PYPL US Equity',
    'QCOM': 'QCOM US Equity',   'REGN': 'REGN US Equity',   'RES':  'RES US Equity',
    'ROLL': 'ROLL US Equity',   'RTX':  'RTX US Equity',    'SANM': 'SANM US Equity',
    'SBUX': 'SBUX US Equity',   'SHAK': 'SHAK US Equity',   'SHW':  'SHW US Equity',
    'SLB':  'SLB US Equity',    'SM':   'SM US Equity',     'SMTC': 'SMTC US Equity',
    'SNPS': 'SNPS US Equity',   'SO':   'SO US Equity',     'SPG':  'SPG US Equity',
    'SPGI': 'SPGI US Equity',   'SPWH': 'SPWH US Equity',   'SRE':  'SRE US Equity',
    'SSD':  'SSD US Equity',    'SXT':  'SXT US Equity',    'T':    'T US Equity',
    'TJX':  'TJX US Equity',    'TMO':  'TMO US Equity',    'TMUS': 'TMUS US Equity',
    'TREX': 'TREX US Equity',   'TXN':  'TXN US Equity',    'UPS':  'UPS US Equity',
    'VICI': 'VICI US Equity',   'VLO':  'VLO US Equity',    'VRSK': 'VRSK US Equity',
    'VRTX': 'VRTX US Equity',   'VZ':   'VZ US Equity',     'WBD':  'WBD US Equity',
    'WELL': 'WELL US Equity',   'WFC':  'WFC US Equity',    'WSFS': 'WSFS US Equity',
    'XEL':  'XEL US Equity',

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
    DESATIVADO: PE_RATIO, PX_TO_BOOK, PX_TO_SALES, BETA são métricas
    estáticas no BQL — não suportam dates=range. Snapshot diário só.

    Para histórico de valuation, use o snapshot diário cumulativo
    (cada run grava 1 linha; banco acumula).
    """
    _log('fundamentals_history: BQL não expõe PE/PB/PS histórico — pulando')
    return


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
            # Só PX_LAST 252d (yield_ind histórico não funciona via dates=range)
            resp = bq.execute(
                f'get(PX_LAST(dates=range(-252D,0D),frq=D,fill=PREV)) for(["{bbg_tk}"])'
            )
            r = resp[0]
            df_h = r.df()
            if 'DATE' in df_h.columns:
                dates = pd.to_datetime(df_h['DATE'])
                vals  = pd.to_numeric(df_h[r.name], errors='coerce')
            else:
                s = r.df()[r.name]
                dates = pd.to_datetime(s.index)
                vals  = pd.to_numeric(s.values, errors='coerce')
            for d, px in zip(dates, vals):
                if pd.notna(px) and float(px) > 0:
                    rows.append({
                        'date':     d.date().isoformat(),
                        'ticker':   ticker_short,
                        'label':    label,
                        'category': cat,
                        'duration': duration,
                        'price':    round(float(px), 4),
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


def fetch_chain(bbg_und, spot, lo_pct=0.95, hi_pct=1.05):
    """
    Pega chain de opções de QUALQUER underlying (índice ou ação) — mesma lógica
    que o Greeks Dashboard usa para SPX. Funciona pra SPX/NDX/RUT/AAPL/etc.
    """
    lo, hi = spot * lo_pct, spot * hi_pct
    conditions = (
        bq.data.expire_dt() >= '0d'
    ).and_(bq.data.expire_dt() <= '45d'
    ).and_(bq.data.strike_px() > lo
    ).and_(bq.data.strike_px() < hi
    ).and_(bq.data.open_int() > 0
    ).and_(bq.data.ivol() > 0)
    univ  = bq.univ.filter(bq.univ.options([bbg_und]), conditions)
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


# Wrapper de compatibilidade
def fetch_spx_chain(spot):
    return fetch_chain('SPX Index', spot)


def export_gex_spx():
    spot = float(_bql(bq.univ.list(['SPX Index']), {'px': bq.data.px_last()})['px'].iloc[0])
    _log(f'SPX: {spot:,.0f}')
    df = fetch_chain('SPX Index', spot).copy()
    _log(f'Chain: {len(df)} contratos')
    T = (pd.to_datetime(df['expiry']) - pd.Timestamp.now()).dt.days / TRADING_DAYS
    T = T.clip(lower=1 / TRADING_DAYS)
    g = calc_gamma(spot, df['strike'].values, df['ivol'].values, T.values)
    df.loc[:, 'gamma'] = g
    is_call = df['put_call'].str.upper().str.startswith('C')
    df.loc[:, 'gex_bn'] = np.where(
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
    Chain completa + Greeks dealer-aggregated por underlying.
    USA EXATAMENTE A MESMA LÓGICA do export_gex_spx — só muda o underlying.
    Reutiliza fetch_chain() (mesmo padrão do Greeks Dashboard pro SPX).
    """
    UNDERLYINGS = [
        # Índices (têm chain líquida via SPX-style)
        ('SPX Index',       'SPX'),
        ('NDX Index',       'NDX'),
        ('RTY Index',       'RUT'),
        # ETFs broad (chain líquida)
        ('SPY US Equity',   'SPY'),
        ('QQQ US Equity',   'QQQ'),
        ('IWM US Equity',   'IWM'),
        # Mega-caps (chain líquida)
        ('AAPL US Equity',  'AAPL'),
        ('MSFT US Equity',  'MSFT'),
        ('NVDA US Equity',  'NVDA'),
        ('GOOGL US Equity', 'GOOGL'),
        ('META US Equity',  'META'),
        ('AMZN US Equity',  'AMZN'),
        ('AVGO US Equity',  'AVGO'),
        ('TSLA US Equity',  'TSLA'),
        ('NFLX US Equity',  'NFLX'),
    ]

    summary_rows = []
    for bbg_und, short in UNDERLYINGS:
        try:
            # Spot — mesma chamada do SPX
            spot = float(_bql(bq.univ.list([bbg_und]),
                              {'px': bq.data.px_last()})['px'].iloc[0])

            # Chain — mesma função fetch_chain (idêntica ao SPX, só muda o underlying)
            df = fetch_chain(bbg_und, spot).copy()
            if df.empty:
                _log(f'greeks_full warn {short}: chain vazia')
                continue

            # Cálculo idêntico ao export_gex_spx
            T = (pd.to_datetime(df['expiry']) - pd.Timestamp.now()).dt.days / TRADING_DAYS
            T = T.clip(lower=1 / TRADING_DAYS)
            g = calc_gamma(spot, df['strike'].values, df['ivol'].values, T.values)
            df.loc[:, 'gamma'] = g
            is_call = df['put_call'].str.upper().str.startswith('C')
            df.loc[:, 'gex_bn'] = np.where(
                is_call,
                 g * df['open_int'] * 100 * spot / 1e9,
                -g * df['open_int'] * 100 * spot / 1e9,
            )

            # Salva chain raw por ticker (mesmo formato gex_spx_*.csv)
            df.to_csv(OUT / f'chain_{short}_{hoje}.csv')

            # Agrega — mesma estrutura do gex_summary do SPX
            gex_total = float(df['gex_bn'].sum())
            gex_call  = float(df[is_call]['gex_bn'].sum())
            gex_put   = float(df[~is_call]['gex_bn'].sum())

            # Walls: strike com maior OI por put/call
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
            pc_oi = (float(puts['open_int'].sum() / calls['open_int'].sum())
                     if not calls.empty and calls['open_int'].sum() > 0 else 0.0)
            total_call_oi = float(calls['open_int'].sum())
            total_put_oi  = float(puts['open_int'].sum())

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
            _log(f'greeks_full {short}: spot={spot:,.0f} GEX={gex_total:+.2f}B chain={len(df)}')
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
    Skew snapshot completo (todos os tickers, todos os tenors):

    Por tenor (30D, 90D, 180D), captura:
      - atm                          (IV ATM = 100% moneyness)
      - put25, call25, put10, call10  (4 deltas)
      - skew_25d   = put25 - call25      (skew clássico)
      - skew_10d   = put10 - call10      (tail skew)
      - call_skew  = call25 / atm        (cobertura para o lado direito)
      - put_skew   = put25  / atm        (cobertura para o lado esquerdo)
      - rr_25d     = put25 - call25      (risk reversal 25-delta)
      - tail_premium = skew_10d - skew_25d
    """
    # Inclui MEGA_CAPS + FUND_TICKERS (antes so FUND_TICKERS — sem SPY/QQQ/AAPL/etc)
    universe = list(set(FUND_TICKERS) | set(MEGA_CAP_TICKERS))
    univ  = bq.univ.list(universe)

    TENORS = ['30D', '90D', '180D']
    items = _safe_items([
        # snapshot por tenor
        *[
            (f'atm_{t}', (lambda t=t: bq.data.implied_volatility(expiry=t, pct_moneyness='100')))
            for t in TENORS
        ],
        *[
            (f'put25_{t}', (lambda t=t: bq.data.implied_volatility(expiry=t, delta='25', put_call='PUT')))
            for t in TENORS
        ],
        *[
            (f'call25_{t}', (lambda t=t: bq.data.implied_volatility(expiry=t, delta='25', put_call='CALL')))
            for t in TENORS
        ],
        *[
            (f'put10_{t}', (lambda t=t: bq.data.implied_volatility(expiry=t, delta='10', put_call='PUT')))
            for t in TENORS
        ],
        *[
            (f'call10_{t}', (lambda t=t: bq.data.implied_volatility(expiry=t, delta='10', put_call='CALL')))
            for t in TENORS
        ],
    ])
    if not items:
        _log('skew_tails: nenhum field disponível')
        return

    try:
        df = _bql(univ, items)
        # Converte tudo para decimal (IV vem em pct)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce') / 100

        # Calcula derivados por tenor
        for t in TENORS:
            atm = df.get(f'atm_{t}')
            p25 = df.get(f'put25_{t}')
            c25 = df.get(f'call25_{t}')
            p10 = df.get(f'put10_{t}')
            c10 = df.get(f'call10_{t}')
            if atm is not None and c25 is not None:
                df[f'call_skew_{t}'] = c25 / atm    # ratio call25/ATM (right-tail hedge)
            if atm is not None and p25 is not None:
                df[f'put_skew_{t}'] = p25 / atm     # ratio put25/ATM (left-tail hedge)
            if p25 is not None and c25 is not None:
                df[f'skew_25d_{t}'] = p25 - c25     # skew 25d clássico
                df[f'rr_25d_{t}']   = p25 - c25     # risk reversal 25d
            if p10 is not None and c10 is not None:
                df[f'skew_10d_{t}'] = p10 - c10     # tail skew 10d
            if (f'skew_10d_{t}' in df.columns and f'skew_25d_{t}' in df.columns):
                df[f'tail_premium_{t}'] = df[f'skew_10d_{t}'] - df[f'skew_25d_{t}']

        df.index.name = 'ticker'
        df.index = df.index.str.replace(' US Equity', '', regex=False).str.replace('/', '-')
        df.to_csv(OUT / f'skew_tails_{hoje}.csv')
        _log(f'skew_tails — {len(df)} tickers, {len(df.columns)} fields ({len(TENORS)} tenors)')
    except Exception as e:
        _log(f'skew_tails warn: {e}')


# ── Universo de ETFs temáticos (substitui DeepVue scraping) ─────────────────
# Cada tema é representado pelo ETF mais líquido. Permite track de performance
# por tema (Today/1W/1M/YTD) direto do Bloomberg, sem depender de scraping.
THEMATIC_ETFS = [
    # ── Tech / AI / Innovation ─────────────────────────────────────────────
    ('IGV',  'Software'),
    ('SOXX', 'Semiconductors'),
    ('SMH',  'Semiconductors (alt)'),
    ('CIBR', 'Cybersecurity'),
    ('ARKK', 'Innovation / Disruptive'),
    ('BOTZ', 'Robotics & AI'),
    ('IRBO', 'Robotics & AI (alt)'),
    ('AIQ',  'AI Multisector'),
    ('SKYY', 'Cloud'),
    ('CLOU', 'Cloud (alt)'),
    ('FINX', 'Fintech'),
    ('IPAY', 'Payments'),
    ('SOCL', 'Social Media'),
    ('HACK', 'Cybersecurity (alt)'),
    # ── Crypto ──────────────────────────────────────────────────────────────
    ('IBIT', 'Bitcoin spot ETF'),
    ('BITO', 'Bitcoin futures ETF'),
    ('BITQ', 'Crypto industry'),
    # ── Health / Bio ────────────────────────────────────────────────────────
    ('IBB',  'Biotech'),
    ('XBI',  'Biotech (small cap)'),
    ('IHI',  'Medical Devices'),
    ('GNOM', 'Genomics'),
    ('IDNA', 'Genomics & Immunology'),
    ('PSCH', 'Healthcare small cap'),
    # ── Clean Energy / ESG / Climate ────────────────────────────────────────
    ('QCLN', 'Clean Energy'),
    ('ICLN', 'Clean Energy (alt)'),
    ('TAN',  'Solar'),
    ('FAN',  'Wind'),
    ('LIT',  'Lithium / Battery'),
    ('URA',  'Uranium'),
    # ── Defense / Aerospace ─────────────────────────────────────────────────
    ('ITA',  'Aerospace & Defense'),
    ('PPA',  'Aerospace & Defense (alt)'),
    # ── Industrials thematic ────────────────────────────────────────────────
    ('XHB',  'Home Construction'),
    ('ITB',  'Home Builders'),
    ('IYT',  'Transports'),
    # ── Consumer thematic ───────────────────────────────────────────────────
    ('XRT',  'Retail'),
    ('PEJ',  'Leisure & Entertainment'),
    ('BJK',  'Casinos / Gaming'),
    ('PBJ',  'Food & Beverage'),
    # ── Financials sub-industries ───────────────────────────────────────────
    ('KBE',  'Banks'),
    ('KRE',  'Regional Banks'),
    ('IAI',  'Broker-Dealers'),
    ('KIE',  'Insurance'),
    # ── Real Estate ─────────────────────────────────────────────────────────
    ('REZ',  'Residential REIT'),
    ('PSR',  'Active REIT'),
    # ── Materials thematic ──────────────────────────────────────────────────
    ('XME',  'Metals & Mining'),
    ('PICK', 'Industrial Metals'),
    ('REMX', 'Rare Earth & Critical Materials'),
    ('COPX', 'Copper Miners'),
    # ── Utilities thematic ──────────────────────────────────────────────────
    ('GRID', 'Smart Grid'),
    ('XLU',  'Utilities broad'),
]

THEMATIC_TICKERS = [f'{tk} US Equity' for tk, _ in THEMATIC_ETFS]
THEMATIC_LABELS  = {f'{tk} US Equity': label for tk, label in THEMATIC_ETFS}


def export_thematic_flow():
    """
    Performance por tema — substituto Bloomberg do scraping DeepVue.

    Para cada ETF temático puxa price + retornos via histórico (260d).
    Computa retornos client-side: 1d, 5d, 21d (1M), 63d (3M), YTD.
    Resultado: dataset 'thematic_flow_*.csv' com 1 linha por tema.
    """
    rows = []
    today_dt = pd.Timestamp.today().normalize()
    year_start = pd.Timestamp(today_dt.year, 1, 1)

    for bbg_tk in THEMATIC_TICKERS:
        label = THEMATIC_LABELS.get(bbg_tk, '')
        ticker_short = bbg_tk.replace(' US Equity', '')
        try:
            resp = bq.execute(
                f'get(PX_LAST(dates=range(-260D,0D),frq=D,fill=PREV)) for(["{bbg_tk}"])'
            )
            r = resp[0]
            df_h = r.df()
            if 'DATE' in df_h.columns:
                dates_idx = pd.to_datetime(df_h['DATE'])
                vals      = pd.to_numeric(df_h[r.name], errors='coerce')
                prices    = pd.Series(vals.values, index=dates_idx).dropna().sort_index()
            else:
                s = r.df()[r.name]
                prices = pd.Series(
                    pd.to_numeric(s.values, errors='coerce'),
                    index=pd.to_datetime(s.index),
                ).dropna().sort_index()
            if prices.empty or len(prices) < 2:
                continue
            prices = prices.astype(float)
            p_now = float(prices.iloc[-1])

            def _ret(n):
                if len(prices) < n + 1:
                    return ''
                p_past = float(prices.iloc[-(n + 1)])
                if p_past <= 0:
                    return ''
                return round((p_now - p_past) / p_past, 6)

            ydr = ''
            ytd_slice = prices[prices.index < year_start]
            if not ytd_slice.empty:
                p_ys = float(ytd_slice.iloc[-1])
                if p_ys > 0:
                    ydr = round((p_now - p_ys) / p_ys, 6)

            rows.append({
                'ticker':    ticker_short,
                'theme':     label,
                'price':     round(p_now, 4),
                'ret_1d':    _ret(1),
                'ret_5d':    _ret(5),
                'ret_21d':   _ret(21),     # 1M
                'ret_63d':   _ret(63),     # 3M
                'ret_ytd':   ydr,
            })
        except Exception as e:
            _log(f'thematic warn {ticker_short}: {e}')

    if rows:
        # Ordena por retorno 1d desc — leaderboard automático
        df_out = pd.DataFrame(rows)
        try:
            df_out = df_out.sort_values('ret_1d', ascending=False, na_position='last')
        except Exception:
            pass
        df_out.to_csv(OUT / f'thematic_flow_{hoje}.csv', index=False)
        _log(f'thematic_flow — {len(rows)} temas')


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
    Membros dos índices principais (SPX/NDX/RUT) — só com mcap + price.
    O field id_index_weight() não existe nesta versão BQL; pesos podem
    ser calculados client-side via mcap_member / mcap_total.
    """
    INDICES = [('SPX Index', 'SPX'), ('NDX Index', 'NDX'), ('RTY Index', 'RUT')]
    rows = []
    for bbg_idx, short in INDICES:
        try:
            members_univ = bq.univ.members(bbg_idx)
            items = _safe_items([
                ('mcap',  lambda: bq.data.cur_mkt_cap()),
                ('price', lambda: bq.data.px_last()),
            ])
            if not items:
                continue
            r = bq.execute(bql.Request(members_univ, items))
            df_m = pd.concat([x.df()[x.name] for x in r], axis=1)
            df_m = df_m.loc[:, ~df_m.columns.duplicated()]
            if 'mcap' in df_m.columns:
                df_m['mcap'] = pd.to_numeric(df_m['mcap'], errors='coerce') / 1e9
            df_m.index.name = 'member'
            # Pesos calculados a partir do mcap
            total_mcap = float(df_m['mcap'].sum()) if 'mcap' in df_m.columns else 0.0
            for member, row in df_m.iterrows():
                mcap = row.get('mcap')
                wt = (float(mcap) / total_mcap) if (pd.notna(mcap) and total_mcap > 0) else ''
                rows.append({
                    'index':    short,
                    'member':   member.replace(' US Equity', '').replace('/', '-'),
                    'weight':   round(wt, 6) if wt != '' else '',
                    'mcap_b':   _to_num(mcap),
                    'price':    _to_num(row.get('price')),
                })
            _log(f'index_members {short}: {len(df_m)}')
        except Exception as e:
            _log(f'index_members warn {short}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'index_members_{hoje}.csv', index=False)
        _log(f'index_members — {len(rows)} linhas')


def export_etf_holdings():
    """
    DESATIVADO: bq.data.fund_top_holdings() não existe nesta versão BQL.
    Para holdings, use bq.univ.holdings('XLK US Equity') que retorna o
    universo de membros — mas requer outro padrão de query.
    """
    _log('etf_holdings: bq.data.fund_top_holdings() não existe — pulando')
    return


def export_borrow_rate():
    """Borrow rate / cost to borrow para mega-caps (squeeze risk)."""
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


def export_quarterly_financials():
    """
    Financials trimestrais (9 trimestres) para infográfico de earnings.
    Exporta: receita, lucro líquido, LPA, margens por mega-cap.
    CSV: quarterly_financials_YYYY-MM-DD.csv
    """
    universe = list(set(MEGA_CAP_TICKERS))

    # Periods: últimos 9 trimestres fiscais
    # BQL: is_comp_sales, is_net_income, is_oper_income, trail_12m_eps
    # Usamos fund_data com periodicidade trimestral
    quarters = []
    ref = date.today()
    for i in range(9):
        # ~3 meses por trimestre, volta do mais recente
        offset_days = i * 91
        d = ref - pd.Timedelta(days=offset_days)
        quarters.append(d.strftime('%Y-%m-%d'))

    rows = []
    for bbg_tk in universe:
        ticker_short = bbg_tk.replace(' US Equity', '').replace('/', '-')
        try:
            # Tenta buscar dados fundamentais trimestrais
            # BQL fund_data: is_comp_sales (receita), is_net_income, is_oper_income
            items = _safe_items([
                ('revenue',        lambda: bq.data.is_comp_sales(fa_period_type='Q', fa_period_offset=0)),
                ('revenue_1q',     lambda: bq.data.is_comp_sales(fa_period_type='Q', fa_period_offset=-1)),
                ('revenue_2q',     lambda: bq.data.is_comp_sales(fa_period_type='Q', fa_period_offset=-2)),
                ('revenue_3q',     lambda: bq.data.is_comp_sales(fa_period_type='Q', fa_period_offset=-3)),
                ('revenue_4q',     lambda: bq.data.is_comp_sales(fa_period_type='Q', fa_period_offset=-4)),
                ('revenue_5q',     lambda: bq.data.is_comp_sales(fa_period_type='Q', fa_period_offset=-5)),
                ('revenue_6q',     lambda: bq.data.is_comp_sales(fa_period_type='Q', fa_period_offset=-6)),
                ('revenue_7q',     lambda: bq.data.is_comp_sales(fa_period_type='Q', fa_period_offset=-7)),
                ('revenue_8q',     lambda: bq.data.is_comp_sales(fa_period_type='Q', fa_period_offset=-8)),
                ('net_income',     lambda: bq.data.is_net_income(fa_period_type='Q', fa_period_offset=0)),
                ('net_income_1q',  lambda: bq.data.is_net_income(fa_period_type='Q', fa_period_offset=-1)),
                ('net_income_2q',  lambda: bq.data.is_net_income(fa_period_type='Q', fa_period_offset=-2)),
                ('net_income_3q',  lambda: bq.data.is_net_income(fa_period_type='Q', fa_period_offset=-3)),
                ('net_income_4q',  lambda: bq.data.is_net_income(fa_period_type='Q', fa_period_offset=-4)),
                ('net_income_5q',  lambda: bq.data.is_net_income(fa_period_type='Q', fa_period_offset=-5)),
                ('net_income_6q',  lambda: bq.data.is_net_income(fa_period_type='Q', fa_period_offset=-6)),
                ('net_income_7q',  lambda: bq.data.is_net_income(fa_period_type='Q', fa_period_offset=-7)),
                ('net_income_8q',  lambda: bq.data.is_net_income(fa_period_type='Q', fa_period_offset=-8)),
                ('oper_income',    lambda: bq.data.is_oper_income(fa_period_type='Q', fa_period_offset=0)),
                ('oper_income_1q', lambda: bq.data.is_oper_income(fa_period_type='Q', fa_period_offset=-1)),
                ('oper_income_2q', lambda: bq.data.is_oper_income(fa_period_type='Q', fa_period_offset=-2)),
                ('oper_income_3q', lambda: bq.data.is_oper_income(fa_period_type='Q', fa_period_offset=-3)),
                ('oper_income_4q', lambda: bq.data.is_oper_income(fa_period_type='Q', fa_period_offset=-4)),
                ('oper_income_5q', lambda: bq.data.is_oper_income(fa_period_type='Q', fa_period_offset=-5)),
                ('oper_income_6q', lambda: bq.data.is_oper_income(fa_period_type='Q', fa_period_offset=-6)),
                ('oper_income_7q', lambda: bq.data.is_oper_income(fa_period_type='Q', fa_period_offset=-7)),
                ('oper_income_8q', lambda: bq.data.is_oper_income(fa_period_type='Q', fa_period_offset=-8)),
                ('gross_profit',   lambda: bq.data.is_gross_profit(fa_period_type='Q', fa_period_offset=0)),
                ('gross_profit_1q',lambda: bq.data.is_gross_profit(fa_period_type='Q', fa_period_offset=-1)),
                ('gross_profit_2q',lambda: bq.data.is_gross_profit(fa_period_type='Q', fa_period_offset=-2)),
                ('gross_profit_3q',lambda: bq.data.is_gross_profit(fa_period_type='Q', fa_period_offset=-3)),
                ('gross_profit_4q',lambda: bq.data.is_gross_profit(fa_period_type='Q', fa_period_offset=-4)),
                ('gross_profit_5q',lambda: bq.data.is_gross_profit(fa_period_type='Q', fa_period_offset=-5)),
                ('gross_profit_6q',lambda: bq.data.is_gross_profit(fa_period_type='Q', fa_period_offset=-6)),
                ('gross_profit_7q',lambda: bq.data.is_gross_profit(fa_period_type='Q', fa_period_offset=-7)),
                ('gross_profit_8q',lambda: bq.data.is_gross_profit(fa_period_type='Q', fa_period_offset=-8)),
                ('eps',            lambda: bq.data.is_diluted_eps(fa_period_type='Q', fa_period_offset=0)),
                ('eps_1q',         lambda: bq.data.is_diluted_eps(fa_period_type='Q', fa_period_offset=-1)),
                ('eps_2q',         lambda: bq.data.is_diluted_eps(fa_period_type='Q', fa_period_offset=-2)),
                ('eps_3q',         lambda: bq.data.is_diluted_eps(fa_period_type='Q', fa_period_offset=-3)),
                ('eps_4q',         lambda: bq.data.is_diluted_eps(fa_period_type='Q', fa_period_offset=-4)),
                ('eps_5q',         lambda: bq.data.is_diluted_eps(fa_period_type='Q', fa_period_offset=-5)),
                ('eps_6q',         lambda: bq.data.is_diluted_eps(fa_period_type='Q', fa_period_offset=-6)),
                ('eps_7q',         lambda: bq.data.is_diluted_eps(fa_period_type='Q', fa_period_offset=-7)),
                ('eps_8q',         lambda: bq.data.is_diluted_eps(fa_period_type='Q', fa_period_offset=-8)),
                # Estimates (consensus) para o trimestre atual
                ('eps_est',        lambda: bq.data.best_eps()),
                ('rev_est',        lambda: bq.data.best_sales()),
                # Fiscal period info
                ('fq_end',         lambda: bq.data.fiscal_quarter_end_date(fa_period_type='Q', fa_period_offset=0)),
                ('fq_end_1q',      lambda: bq.data.fiscal_quarter_end_date(fa_period_type='Q', fa_period_offset=-1)),
                ('fq_end_2q',      lambda: bq.data.fiscal_quarter_end_date(fa_period_type='Q', fa_period_offset=-2)),
                ('fq_end_3q',      lambda: bq.data.fiscal_quarter_end_date(fa_period_type='Q', fa_period_offset=-3)),
                ('fq_end_4q',      lambda: bq.data.fiscal_quarter_end_date(fa_period_type='Q', fa_period_offset=-4)),
                ('fq_end_5q',      lambda: bq.data.fiscal_quarter_end_date(fa_period_type='Q', fa_period_offset=-5)),
                ('fq_end_6q',      lambda: bq.data.fiscal_quarter_end_date(fa_period_type='Q', fa_period_offset=-6)),
                ('fq_end_7q',      lambda: bq.data.fiscal_quarter_end_date(fa_period_type='Q', fa_period_offset=-7)),
                ('fq_end_8q',      lambda: bq.data.fiscal_quarter_end_date(fa_period_type='Q', fa_period_offset=-8)),
            ])
            if not items:
                _log(f'quarterly_fin: sem fields para {ticker_short}')
                continue

            r = bq.execute(bql.Request(bq.univ.list([bbg_tk]), items))
            df_q = pd.concat([x.df()[x.name] for x in r], axis=1)

            if len(df_q) > 0:
                row = df_q.iloc[0]
                entry = {'ticker': ticker_short}
                for field in items.keys():
                    entry[field] = _to_num(row.get(field)) if field in items else ''
                rows.append(entry)
        except Exception as e:
            _log(f'quarterly_fin warn {ticker_short}: {e}')

    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'quarterly_financials_{hoje}.csv', index=False)
        _log(f'quarterly_financials — {len(rows)} mega-caps, {len(rows[0])} fields each')


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


def export_positioning_models():
    """
    Modelo CTA + Risk Parity + Vol Control para TODOS os tickers.

    Replica a metodologia BofA/GS usada no Greeks Dashboard:

    CTA (Trend Follower):
      Sinal por janela: tanh((P_now - P_avg_window) / (vol * sqrt(window/252)))
      Janelas: 20d (curto), 60d (médio), 120d (longo)
      Score final = média dos 3 sinais (range [-1, +1])
      Notional estimado = score × AUM_universo × vol_target / vol_realizada

    Vol Control / Vol Targeting:
      Position weight = vol_target / max(vol_realizada, vol_floor)
      Score = +1 se vol cai (re-leveraging), -1 se vol sobe (de-leveraging)
      Notional flow = -delta_vol * AUM * leverage_sensitivity

    Risk Parity:
      Position weight ∝ 1 / vol_realizada
      Sinal de re-balance: derivada da vol em janela curta
      Score = -delta_vol_5d_pct (vol caindo = comprador)
    """
    universe = list(set(FUND_TICKERS))

    # AUM assumido por classe (universo CTA + RP + VolCtrl em USD bilhões)
    AUM_CTA       = 350.0   # CTAs total ~$350B em equity (BofA estimate)
    AUM_VOLCTRL   = 220.0   # Vol target funds (~$220B)
    AUM_RP        = 180.0   # Risk Parity (~$180B)
    VOL_TARGET    = 0.10    # 10% target vol (BofA standard)
    VOL_FLOOR     = 0.05    # piso pra evitar leverage absurdo

    rows = []
    for bbg_tk in universe:
        ticker_short = bbg_tk.replace(' US Equity', '').replace('/', '-')
        try:
            # Pega 260d de PX_LAST
            resp = bq.execute(
                f'get(PX_LAST(dates=range(-260D,0D),frq=D,fill=PREV)) for(["{bbg_tk}"])'
            )
            r = resp[0]
            df_h = r.df()
            if 'DATE' in df_h.columns:
                dates = pd.to_datetime(df_h['DATE'])
                vals  = pd.to_numeric(df_h[r.name], errors='coerce')
                prices = pd.Series(vals.values, index=dates).dropna().sort_index()
            else:
                s = r.df()[r.name]
                prices = pd.Series(
                    pd.to_numeric(s.values, errors='coerce'),
                    index=pd.to_datetime(s.index),
                ).dropna().sort_index()
            if len(prices) < 130:  # precisa pelo menos 130 dias pro 120d window
                continue
            prices = prices.astype(float)
            p_now = float(prices.iloc[-1])

            # Retornos log
            rets = np.log(prices / prices.shift(1)).dropna()

            # Vol realizada anualizada
            def _ann_vol(window):
                if len(rets) < window:
                    return None
                return float(rets.iloc[-window:].std() * np.sqrt(252))

            vol_30  = _ann_vol(30)
            vol_60  = _ann_vol(60)
            vol_5   = _ann_vol(5)
            vol_5_prev = _ann_vol(10) if len(rets) > 10 else None

            # ── CTA: trend signal por janela (BofA/GS) ──────────────────────
            def _trend_signal(window):
                if len(prices) < window + 1:
                    return None
                p_past = float(prices.iloc[-(window + 1)])
                if p_past <= 0:
                    return None
                ret_window = (p_now - p_past) / p_past
                # Normaliza pelo risk-budget esperado da janela
                vol_w = _ann_vol(window) or 0.20
                expected_move = vol_w * np.sqrt(window / 252.0)
                if expected_move == 0:
                    return None
                return float(np.tanh(ret_window / expected_move))

            sig_20  = _trend_signal(20)
            sig_60  = _trend_signal(60)
            sig_120 = _trend_signal(120)
            cta_signals = [s for s in (sig_20, sig_60, sig_120) if s is not None]
            cta_score = float(np.mean(cta_signals)) if cta_signals else 0.0

            # CTA notional estimado
            vol_for_cta = vol_60 or 0.20
            cta_leverage = VOL_TARGET / max(vol_for_cta, VOL_FLOOR)
            cta_notional_b = cta_score * AUM_CTA * cta_leverage / len(universe) * 100  # spread

            # ── Vol Control / Vol Targeting ────────────────────────────────
            # Score: vol caindo = re-leveraging (positivo), vol subindo = de-leveraging (negativo)
            volctrl_score = 0.0
            if vol_5 is not None and vol_5_prev is not None and vol_5_prev > 0:
                # Negativo do delta de vol — quanto mais vol cai, mais positivo
                volctrl_score = float(np.tanh(-5.0 * (vol_5 - vol_5_prev) / vol_5_prev))
            volctrl_leverage = VOL_TARGET / max(vol_30 or 0.15, VOL_FLOOR)
            volctrl_notional_b = volctrl_score * AUM_VOLCTRL * volctrl_leverage / len(universe) * 100

            # ── Risk Parity ─────────────────────────────────────────────────
            # Position weight ∝ 1/vol; rebalance puxa pra cima quando vol cai
            rp_weight = 1.0 / max(vol_60 or 0.20, VOL_FLOOR)
            rp_score = volctrl_score  # mesmo proxy de delta vol
            rp_notional_b = rp_score * AUM_RP * rp_weight / 30  # normalização leve

            rows.append({
                'ticker':            ticker_short,
                'price':             round(p_now, 4),
                # ── Vol realizada ──
                'rv_5d':             round(vol_5, 4)  if vol_5  is not None else '',
                'rv_30d':            round(vol_30, 4) if vol_30 is not None else '',
                'rv_60d':            round(vol_60, 4) if vol_60 is not None else '',
                # ── CTA model ──
                'cta_sig_20d':       round(sig_20, 4)  if sig_20  is not None else '',
                'cta_sig_60d':       round(sig_60, 4)  if sig_60  is not None else '',
                'cta_sig_120d':      round(sig_120, 4) if sig_120 is not None else '',
                'cta_score':         round(cta_score, 4),
                'cta_leverage':      round(cta_leverage, 3),
                'cta_notional_b':    round(cta_notional_b, 4),
                # ── Vol Control ──
                'volctrl_score':     round(volctrl_score, 4),
                'volctrl_leverage':  round(volctrl_leverage, 3),
                'volctrl_notional_b':round(volctrl_notional_b, 4),
                # ── Risk Parity ──
                'rp_score':          round(rp_score, 4),
                'rp_weight':         round(rp_weight, 3),
                'rp_notional_b':     round(rp_notional_b, 4),
                # ── Posicionamento líquido (todos os 3 modelos somados) ──
                'flow_total_b':      round(cta_notional_b + volctrl_notional_b + rp_notional_b, 4),
                # Direção macro
                'flow_direction':    ('long' if (cta_score + volctrl_score + rp_score) > 0.1
                                      else 'short' if (cta_score + volctrl_score + rp_score) < -0.1
                                      else 'flat'),
            })
        except Exception as e:
            _log(f'positioning warn {ticker_short}: {e}')

    if rows:
        df_out = pd.DataFrame(rows)
        try:
            df_out = df_out.sort_values('flow_total_b', ascending=False, na_position='last')
        except Exception:
            pass
        df_out.to_csv(OUT / f'positioning_models_{hoje}.csv', index=False)
        _log(f'positioning_models — {len(rows)} tickers (CTA + VolCtrl + RP)')


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
            r = resp[0]
            df_h = r.df()
            # df pode vir com colunas (DATE, PX_LAST, CURRENCY) e ticker no index
            if 'DATE' in df_h.columns:
                dates_idx = pd.to_datetime(df_h['DATE'])
                vals      = pd.to_numeric(df_h[r.name], errors='coerce')
                prices    = pd.Series(vals.values, index=dates_idx).dropna().sort_index()
            else:
                s = r.df()[r.name]
                prices = pd.Series(
                    pd.to_numeric(s.values, errors='coerce'),
                    index=pd.to_datetime(s.index),
                ).dropna().sort_index()
            if prices.empty:
                continue
            prices = prices.astype(float)

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
    """
    252 dias de histórico para todos os tickers — base da rede de correlação.
    O BQL retorna o df com colunas ['DATE', 'PX_LAST', 'CURRENCY'] (ou similar) e
    o índice = ticker. Pegamos a coluna numérica e usamos DATE como index.
    """
    rows = []
    for yf_tk, bbg_tk in _YF_TO_BBG.items():
        try:
            resp = bq.execute(
                f'get(PX_LAST(dates=range(-252D,0D),frq=D,fill=PREV)) for(["{bbg_tk}"])'
            )
            r = resp[0]
            df2 = r.df()
            # df2 vem com várias colunas (DATE, PX_LAST, CURRENCY...) e ticker no index
            # Usa o método .df()[r.name] que retorna a Series numérica direto
            s = r.df()[r.name]
            # Reset index pra obter as datas (se vier no índice ou em coluna DATE)
            if 'DATE' in df2.columns:
                dates = pd.to_datetime(df2['DATE'])
                vals  = pd.to_numeric(df2[r.name], errors='coerce')
            else:
                dates = pd.to_datetime(s.index)
                vals  = pd.to_numeric(s.values, errors='coerce')
            for d, px in zip(dates, vals):
                if pd.notna(px) and float(px) > 0:
                    rows.append({
                        'date':      d.date().isoformat(),
                        'yf_ticker': yf_tk,
                        'price':     round(float(px), 4),
                    })
        except Exception as e:
            _log(f'hist_bulk warn {yf_tk}: {e}')
    if rows:
        pd.DataFrame(rows).to_csv(OUT / f'price_history_bulk_{hoje}.csv', index=False)
        _log(f'price_history_bulk — {len(rows)} linhas')


def export_iv_history():
    """
    DESATIVADO: IVOL_MID_ATM com (expiry, dates=range) não é aceito como
    chamada histórica no BQL. Snapshot diário é o único disponível
    (export_options_iv grava 1 linha por dia; banco acumula).
    """
    _log('iv_history: BQL não aceita IVOL_MID_ATM(expiry, dates=range) — pulando')
    return


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

def _safe(label, fn):
    """Roda uma export envolvida em try/except — um crash não para o resto."""
    try:
        fn()
    except Exception as e:
        _log(f'{label} CRASHED: {type(e).__name__}: {e}')


def export_all(_download=True):
    """
    Snapshot do dia: TUDO via BQL.
    Se _download=False, não dispara o download (usado pelo bulk pra evitar
    baixar 2 ZIPs).
    """
    print(f'\n=== [{time.strftime("%H:%M:%S")}] Export ===')

    # ── Fundamentals snapshot ─────────────────────────────────────────────
    print('Fundamentais (equity)...');           _safe('fundamentals', export_fundamentals)
    print('Bond ETFs...');                       _safe('bond_etf', export_bond_etf_fundamentals)
    print('FX ETFs...');                         _safe('fx_etf', export_fx_etf_fundamentals)
    print('Commodity / Vol ETFs...');            _safe('commodity_etf', export_commodity_etf_fundamentals)

    # ── Options chain & greeks ───────────────────────────────────────────
    print('Options IV ATM 30d...');              _safe('options_iv', export_options_iv)
    print('IV term structure 30/60/90/180/360...'); _safe('iv_term', export_iv_term_structure)
    print('Skew tails (30/90/180D, todos tickers)...'); _safe('skew_tails', export_skew_tails)
    print('GEX SPX...');                         _safe('gex_spx', export_gex_spx)
    print('Greeks per ticker (mega-caps)...');   _safe('greeks_full', export_options_greeks_full)

    # ── Flow temático (substitui DeepVue scraping) ────────────────────────
    print('Thematic flow (50+ ETFs por tema)...'); _safe('thematic', export_thematic_flow)

    # ── Volume / fluxos / short interest ─────────────────────────────────
    print('Volume + dark flows + short interest...'); _safe('volume_flows', export_volume_flows)

    # ── LETF / earnings / borrow / dividends / revisions ──────────────────
    print('LETF flows...');                _safe('letf', export_letf)
    print('Earnings calendar...');         _safe('earnings', export_earnings_calendar)
    print('Borrow rate (mega-caps)...');   _safe('borrow', export_borrow_rate)
    print('Dividends...');                 _safe('dividends', export_dividends)
    print('EPS revisions...');             _safe('eps_revisions', export_eps_revisions)
    print('Quarterly financials (9Q)...');  _safe('quarterly_fin', export_quarterly_financials)

    # ── Prices / macro ───────────────────────────────────────────────────
    print('Realized vol (30/60/90/252d)...');   _safe('realized_vol', export_realized_vol)
    print('Positioning models (CTA+VolCtrl+RP)...'); _safe('positioning', export_positioning_models)
    print('Prices snapshot...');                _safe('prices', export_prices)
    print('Price history (snapshot)...');       _safe('price_history', export_price_history)
    print('Macro series...');                   _safe('macro', export_macro)

    export_meta()
    if _download:
        auto_download()
    print('Pronto.')


def export_all_bulk():
    """Tudo + 252 dias de histórico (preços + IV + fundamentals + bonds + index members). ~10 min."""
    print(f'\n=== BULK [{time.strftime("%H:%M:%S")}] ===')

    # Snapshot — sem baixar (download único no fim do bulk)
    export_all(_download=False)

    # ── Histórico 252d ────────────────────────────────────────────────────
    print('Fundamentals history (252d)...');   _safe('fund_history', export_fundamentals_history)
    print('Bond ETF history (252d)...');       _safe('bond_history', export_bond_etf_history)
    print('Price history bulk (252d)...');     _safe('price_bulk', export_price_history_bulk)
    print('IV history (252d)...');             _safe('iv_history', export_iv_history)

    # ── Estrutura de mercado (members + holdings) ─────────────────────────
    print('Index members + weights (SPX/NDX/RUT)...'); _safe('index_members', export_index_members)
    print('ETF holdings (sectors)...');                _safe('etf_holdings', export_etf_holdings)

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
