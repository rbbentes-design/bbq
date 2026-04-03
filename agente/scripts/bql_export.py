# ═══════════════════════════════════════════════════════
#  BQL Export + Auto-Download  |  cola e roda no BQuant
# ═══════════════════════════════════════════════════════
import bql
import pandas as pd
import numpy as np
import zipfile, io, base64, time, threading
from pathlib import Path
from datetime import date, datetime
from scipy.stats import norm
from IPython.display import display, HTML
import ipywidgets as widgets

OUT          = Path.home() / 'bql_data'
INTERVAL     = 180
TRADING_DAYS = 252
bq           = bql.Service()
OUT.mkdir(parents=True, exist_ok=True)
hoje = date.today().isoformat()

# ── helpers ──────────────────────────────────────────────
def _bql(univ, items):
    resp = bq.execute(bql.Request(univ, items))
    df   = pd.concat([r.df()[r.name] for r in resp], axis=1)
    return df.loc[:, ~df.columns.duplicated()]

def _log(msg): print(f'  {msg}')

def calc_gamma(S, K, vol, T):
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S/K) + 0.5*vol**2*T) / (vol*np.sqrt(T))
        g  = norm.pdf(d1) / (S*vol*np.sqrt(T))
        return np.where(np.isfinite(g), g, 0.0)

# ── universos ─────────────────────────────────────────────
FUND_TICKERS = [
    'AAPL US Equity','AMZN US Equity','MSFT US Equity','TSLA US Equity',
    'META US Equity','NVDA US Equity','GOOGL US Equity','AVGO US Equity',
    'JPM US Equity','LLY US Equity','UNH US Equity','XOM US Equity',
    'COST US Equity','V US Equity','MA US Equity','WMT US Equity',
    'NFLX US Equity','JNJ US Equity','PG US Equity',
]

# yfinance → Bloomberg
_YF_TO_BBG = {
    'AAPL':'AAPL US Equity','MSFT':'MSFT US Equity','NVDA':'NVDA US Equity',
    'AMZN':'AMZN US Equity','META':'META US Equity','GOOGL':'GOOGL US Equity',
    'TSLA':'TSLA US Equity','AVGO':'AVGO US Equity','JPM':'JPM US Equity',
    'LLY':'LLY US Equity','UNH':'UNH US Equity','XOM':'XOM US Equity',
    'COST':'COST US Equity','V':'V US Equity','MA':'MA US Equity',
    'WMT':'WMT US Equity','NFLX':'NFLX US Equity','JNJ':'JNJ US Equity',
    'PG':'PG US Equity','BRK-B':'BRK/B US Equity',
    'SPY':'SPY US Equity','QQQ':'QQQ US Equity',
    'TLT':'TLT US Equity','HYG':'HYG US Equity','GLD':'GLD US Equity',
    'IWM':'IWM US Equity','EEM':'EEM US Equity','EFA':'EFA US Equity',
    '^GSPC':'SPX Index','^NDX':'NDX Index','^RUT':'RTY Index',
    '^VIX':'VIX Index','CL=F':'CL1 Comdty','GC=F':'GC1 Comdty',
    'DX-Y.NYB':'DXY Curncy','BTC-USD':'XBT Curncy',
}

# ── fundamentals ──────────────────────────────────────────
def export_fundamentals():
    univ  = bq.univ.list(FUND_TICKERS)
    t_str = ', '.join(f'"{t}"' for t in FUND_TICKERS)
    items = {
        'PE_RATIO':            bq.data.pe_ratio(),
        'CUR_MKT_CAP':         bq.data.cur_mkt_cap(),
        'BETA':                bq.data.beta(),
        'PROF_MARGIN':         bq.data.prof_margin(),
        'TOT_DEBT_TO_TOT_EQY': bq.data.tot_debt_to_tot_eqy(),
        'RETURN_COM_EQY':      bq.data.return_com_eqy(),
        'EQY_DVD_YLD_IND':     bq.data.eqy_dvd_yld_ind(),
        'PX_LAST':             bq.data.px_last(),
    }
    df = _bql(univ, items)
    df.rename(columns={
        'PE_RATIO':'pe','CUR_MKT_CAP':'mktcap_b','BETA':'beta',
        'PROF_MARGIN':'profit_margin','TOT_DEBT_TO_TOT_EQY':'debt_equity',
        'RETURN_COM_EQY':'roe','EQY_DVD_YLD_IND':'dividend_yield','PX_LAST':'price'
    }, inplace=True)
    df['mktcap_b']       = pd.to_numeric(df['mktcap_b'],       errors='coerce') / 1e9
    df['profit_margin']  = pd.to_numeric(df['profit_margin'],  errors='coerce') / 100
    df['roe']            = pd.to_numeric(df['roe'],            errors='coerce') / 100
    df['dividend_yield'] = pd.to_numeric(df['dividend_yield'], errors='coerce') / 100
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
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity','', regex=False)
    df.to_csv(OUT / f'fundamentals_{hoje}.csv')
    _log(f'fundamentals — {len(df)} linhas')

# ── options iv ────────────────────────────────────────────
def export_options_iv():
    univ  = bq.univ.list(FUND_TICKERS)
    items = {
        'atm_iv': bq.data.implied_volatility(expiry='30D', pct_moneyness='100'),
        'put25':  bq.data.implied_volatility(expiry='30D', delta='25', put_call='PUT'),
        'call25': bq.data.implied_volatility(expiry='30D', delta='25', put_call='CALL'),
        'pcr_oi': bq.data.put_call_open_interest_ratio(),
    }
    df = _bql(univ, items)
    df['atm_iv']   = pd.to_numeric(df['atm_iv'], errors='coerce') / 100
    df['skew_25d'] = (pd.to_numeric(df['put25'], errors='coerce') -
                      pd.to_numeric(df['call25'], errors='coerce')) / 100
    df = df[['atm_iv','skew_25d','pcr_oi']]
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity','', regex=False)
    df.to_csv(OUT / f'options_iv_{hoje}.csv')
    _log(f'options_iv — {len(df)} linhas')

# ── gex spx ───────────────────────────────────────────────
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
    T = T.clip(lower=1/TRADING_DAYS)
    g = calc_gamma(spot, df['strike'].values, df['ivol'].values, T.values)
    df['gamma']  = g
    is_call = df['put_call'].str.upper().str.startswith('C')
    df['gex_bn'] = np.where(is_call,
         g * df['open_int'] * 100 * spot / 1e9,
        -g * df['open_int'] * 100 * spot / 1e9)
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

# ── letf ──────────────────────────────────────────────────
LETFS = ['UPRO US Equity','SPXU US Equity','TQQQ US Equity',
         'SQQQ US Equity','TNA US Equity','TZA US Equity']

def export_letf():
    items = {'nav': bq.data.px_last(), 'aum_b': bq.data.fund_total_assets()}
    df = _bql(bq.univ.list(LETFS), items)
    df['aum_b'] = pd.to_numeric(df['aum_b'], errors='coerce') / 1e9
    df.index.name = 'ticker'
    df.index = df.index.str.replace(' US Equity','', regex=False)
    df['leverage'] = df.index.map({'UPRO':3,'SPXU':-3,'TQQQ':3,'SQQQ':-3,'TNA':3,'TZA':-3})
    df['index']    = df.index.map(lambda x: 'SPX' if x in ['UPRO','SPXU'] else 'NDX' if x in ['TQQQ','SQQQ'] else 'RUT')
    df.to_csv(OUT / f'letf_flows_{hoje}.csv')
    _log(f'letf_flows — {len(df)} linhas')

# ── prices (snapshot atual de todos os tickers) ───────────
def export_prices():
    rows = []
    for yf_tk, bbg_tk in _YF_TO_BBG.items():
        try:
            r    = bq.execute(bql.Request(bq.univ.list([bbg_tk]), {'p': bq.data.px_last()}))[0].df()
            prev = bq.execute(bql.Request(bq.univ.list([bbg_tk]), {
                'p0': bq.data.px_last(dates=bq.func.range('-2D','-1D'), fill='PREV')
            }))[0].df()
            ytd  = bq.execute(bql.Request(bq.univ.list([bbg_tk]), {
                'y0': bq.data.px_last(dates=bq.func.range(f'{date.today().year}-01-01','0D'), fill='PREV')
            }))[0].df()
            p   = float(r.select_dtypes('number').iloc[-1, 0])
            p0  = float(prev.select_dtypes('number').iloc[-1, 0]) if not prev.empty else None
            y0  = float(ytd.select_dtypes('number').iloc[0, 0])  if not ytd.empty  else None
            rows.append({
                'yf_ticker':    yf_tk,
                'bbg_ticker':   bbg_tk,
                'name':         bbg_tk.split()[0],
                'price':        round(p, 4),
                'daily_return': round((p-p0)/p0, 6) if p0 else '',
                'ytd_return':   round((p-y0)/y0, 6) if y0  else '',
            })
        except Exception as e:
            _log(f'prices warn {yf_tk}: {e}')
    pd.DataFrame(rows).to_csv(OUT / f'prices_{hoje}.csv', index=False)
    _log(f'prices — {len(rows)} tickers')

# ── price history (só hoje — banco acumula) ───────────────
def export_price_history():
    rows = []
    for yf_tk, bbg_tk in _YF_TO_BBG.items():
        try:
            r   = bq.execute(bql.Request(bq.univ.list([bbg_tk]), {'p': bq.data.px_last()}))[0].df()
            px  = float(r.select_dtypes('number').iloc[-1, 0])
            rows.append({'date': hoje, 'yf_ticker': yf_tk, 'price': round(px, 4)})
        except Exception as e:
            _log(f'hist warn {yf_tk}: {e}')
    pd.DataFrame(rows).to_csv(OUT / f'price_history_{hoje}.csv', index=False)
    _log(f'price_history — {len(rows)} tickers')

# ── macro series (curva de juros, VIX, spreads, FX) ──────
# Mesmo padrão de export_prices: uma chamada BQL por ticker, try/except por ticker
MACRO_TICKERS = [
    # Curva de juros EUA
    ('USGG1M Index',   'US Treasury 1M',    'rates_usd'),
    ('USGG3M Index',   'US Treasury 3M',    'rates_usd'),
    ('USGG6M Index',   'US Treasury 6M',    'rates_usd'),
    ('USGG1YR Index',  'US Treasury 1Y',    'rates_usd'),
    ('USGG2YR Index',  'US Treasury 2Y',    'rates_usd'),
    ('USGG5YR Index',  'US Treasury 5Y',    'rates_usd'),
    ('USGG10YR Index', 'US Treasury 10Y',   'rates_usd'),
    ('USGG30YR Index', 'US Treasury 30Y',   'rates_usd'),
    # Volatilidade
    ('VIX Index',      'VIX Spot',          'volatility'),
    ('VIX9D Index',    'VIX 9-Day',         'volatility'),
    ('VIX3M Index',    'VIX 3-Month',       'volatility'),
    ('VVIX Index',     'Vol of VIX',        'volatility'),
    ('MOVE Index',     'MOVE (bond vol)',    'volatility'),
    # Spreads de crédito
    ('LUACOAS Index',  'IG OAS Spread',     'credit_spread'),
    ('LF98OAS Index',  'HY OAS Spread',     'credit_spread'),
    # FX
    ('DXY Curncy',     'Dollar Index',      'fx'),
    # Juros de curto prazo
    ('SOFRRATE Index', 'SOFR Rate',         'monetary'),
    # Inflação implícita
    ('USGGBE10 Index', 'US 10Y Breakeven',  'inflation'),
]

def export_macro():
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

    # Derivados calculados localmente a partir dos valores BQL já coletados
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

# ── meta ──────────────────────────────────────────────────
def export_meta():
    pd.DataFrame([{'generated_at': datetime.now().isoformat()}])\
      .to_csv(OUT / f'meta_{hoje}.csv', index=False)

# ── auto-download zip ─────────────────────────────────────
def auto_download():
    arquivos = sorted(OUT.glob(f'*_{hoje}.csv'))
    if not arquivos: return
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in arquivos:
            zf.write(f, f.name)
    buf.seek(0)
    b64  = base64.b64encode(buf.read()).decode()
    nome = f'bql_data_{hoje}.zip'
    uid  = str(int(time.time()))
    display(HTML(f'<a id="dl{uid}" href="data:application/zip;base64,{b64}" download="{nome}"></a>'
                 f'<script>document.getElementById("dl{uid}").click();</script>'))
    _log(f'Download: {nome} ({len(arquivos)} arquivos)')

# ── ciclo completo ────────────────────────────────────────
def export_all():
    print(f'\n=== [{time.strftime("%H:%M:%S")}] ===')
    print('Fundamentais...');    export_fundamentals()
    print('Options IV...');      export_options_iv()
    print('GEX SPX...');         export_gex_spx()
    print('LETF...');            export_letf()
    print('Prices...');          export_prices()
    print('Price history...');   export_price_history()
    print('Macro series...');    export_macro()
    export_meta()
    auto_download()
    print('Pronto.')

# ── UI ───────────────────────────────────────────────────
_running = False

def _loop():
    while _running:
        export_all()
        time.sleep(INTERVAL)

btn_run  = widgets.Button(description='⬇ Exportar agora', button_style='success',
                           layout=widgets.Layout(width='160px'))
btn_loop = widgets.ToggleButton(description='▶ Loop 3min', button_style='info',
                                 layout=widgets.Layout(width='120px'))
out_w    = widgets.Output()

def on_run(_):
    with out_w: export_all()

def on_loop(change):
    global _running
    _running = change['new']
    btn_loop.description = '⏹ Parar' if _running else '▶ Loop 3min'
    if _running:
        threading.Thread(target=_loop, daemon=True).start()

btn_run.on_click(on_run)
btn_loop.observe(on_loop, names='value')
display(widgets.HBox([btn_run, btn_loop]), out_w)
