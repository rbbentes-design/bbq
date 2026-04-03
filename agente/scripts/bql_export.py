"""
BQL Export — MacroDesk
======================
Cole este script inteiro em uma celula do Jupyter (BQuant) e rode.
Exporta fundamentais, IV e LETF para CSV a cada 3 minutos.

Kernel -> Interrupt (■) para parar o loop.

Arquivos gerados (na pasta home do BQuant):
    ~/bql_data/fundamentals_2026-04-02.csv
    ~/bql_data/options_iv_2026-04-02.csv
    ~/bql_data/letf_flows_2026-04-02.csv
    ~/bql_data/meta_2026-04-02.csv

Depois baixe os CSVs do Jupyter (botao direito -> Download)
e coloque em: C:\\Users\\rafael bentes\\bbg\\agente\\bql_data\\
"""

import bql
import pandas as pd
import csv, math, time
from datetime import date, datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACAO
# ─────────────────────────────────────────────────────────────────────────────

# Pasta de saida — Windows (bqlData.bat) salva direto no projeto; Linux (BQuant cloud) salva no home
import platform as _platform
if _platform.system() == 'Windows':
    OUT = Path(r'C:\Users\rafael bentes\bbg\agente\bql_data')
else:
    OUT = Path.home() / 'bql_data'
OUT.mkdir(parents=True, exist_ok=True)
print(f'Sistema: {_platform.system()} | Salvando em: {OUT}')

INTERVAL = 180  # segundos entre exports (3 min)

STOCKS = [
    'AAPL US Equity', 'MSFT US Equity', 'NVDA US Equity', 'AMZN US Equity', 'META US Equity',
    'GOOGL US Equity', 'TSLA US Equity', 'BRK/B US Equity', 'AVGO US Equity', 'JPM US Equity',
    'LLY US Equity', 'UNH US Equity', 'XOM US Equity', 'COST US Equity', 'V US Equity',
    'MA US Equity', 'WMT US Equity', 'NFLX US Equity', 'JNJ US Equity', 'PG US Equity',
    'SPY US Equity', 'QQQ US Equity', 'GLD US Equity', 'TLT US Equity', 'HYG US Equity',
]

TICKER = {s: s.replace(' US Equity', '').replace('BRK/B', 'BRK-B') for s in STOCKS}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _v(row, col):
    try:
        v = float(row[col])
        return None if math.isnan(v) else v
    except:
        return None

def _bql(req):
    """Substitui bql.combined_df (deprecated)."""
    return pd.concat([x.df()[x.name] for x in req], axis=1)

def _norm(df):
    """Remove parenteses: PE_RATIO() -> PE_RATIO."""
    df.columns = [c.split('(')[0].strip() for c in df.columns]
    return df

def _csv(name, rows, fields):
    """Salva CSV com data no nome: fundamentals_2026-04-02.csv"""
    filename = f"{name}_{date.today().isoformat()}.csv"
    path = OUT / filename
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'  OK: {path}  ({len(rows)} linhas)')

# ─────────────────────────────────────────────────────────────────────────────
# 1. FUNDAMENTAIS
# ─────────────────────────────────────────────────────────────────────────────

def export_fundamentals():
    print('Fundamentais...')
    tstr = ', '.join(f'"{s}"' for s in STOCKS)
    try:
        req = bq.execute(
            f'get(PE_RATIO, PX_TO_BOOK_RATIO, CUR_MKT_CAP, BETA, PROF_MARGIN,'
            f' TOT_DEBT_TO_TOT_EQY, RETURN_COM_EQY, EQY_DVD_YLD_IND, PX_LAST)'
            f' for([{tstr}])'
        )
        df = _norm(_bql(req).groupby(level=0).last())

        req2 = bq.execute(
            f'get(PX_HIGH(dates=range(-365D,0D),frq=Y),'
            f'    PX_LOW(dates=range(-365D,0D),frq=Y))'
            f' for([{tstr}])'
        )
        df2 = _norm(_bql(req2).groupby(level=0).last())

        rows = []
        for s in STOCKS:
            t  = TICKER[s]
            r1 = df.loc[s]  if s in df.index  else pd.Series(dtype=float)
            r2 = df2.loc[s] if s in df2.index else pd.Series(dtype=float)
            px = _v(r1, 'PX_LAST')
            hi = _v(r2, 'PX_HIGH')
            mc = _v(r1, 'CUR_MKT_CAP')
            rows.append({
                'ticker':         t,
                'pe':             round(_v(r1,'PE_RATIO'), 2)              if _v(r1,'PE_RATIO') else '',
                'pb':             round(_v(r1,'PX_TO_BOOK_RATIO'), 2)      if _v(r1,'PX_TO_BOOK_RATIO') else '',
                'beta':           round(_v(r1,'BETA'), 3)                  if _v(r1,'BETA') else '',
                'mktcap_b':       round(mc / 1e9, 2)                       if mc else '',
                'roe':            round(_v(r1,'RETURN_COM_EQY') / 100, 4)  if _v(r1,'RETURN_COM_EQY') else '',
                'profit_margin':  round(_v(r1,'PROF_MARGIN') / 100, 4)    if _v(r1,'PROF_MARGIN') else '',
                'debt_equity':    round(_v(r1,'TOT_DEBT_TO_TOT_EQY'), 2)  if _v(r1,'TOT_DEBT_TO_TOT_EQY') else '',
                'dividend_yield': round(_v(r1,'EQY_DVD_YLD_IND') / 100, 4) if _v(r1,'EQY_DVD_YLD_IND') else '',
                'price':          round(px, 4)                             if px else '',
                'hi_52w':         round(hi, 4)                             if hi else '',
                'lo_52w':         round(_v(r2,'PX_LOW'), 4)               if _v(r2,'PX_LOW') else '',
                'drawdown_52w':   round((px - hi) / hi, 4)                if px and hi and hi > 0 else '',
            })
        _csv('fundamentals', rows,
             ['ticker','pe','pb','beta','mktcap_b','roe',
              'profit_margin','debt_equity','dividend_yield',
              'price','hi_52w','lo_52w','drawdown_52w'])
    except Exception as e:
        print(f'  [ERRO] fundamentals: {e}')

# ─────────────────────────────────────────────────────────────────────────────
# 2. OPTIONS IV / SKEW / PCR
# ─────────────────────────────────────────────────────────────────────────────

def export_options_iv():
    print('Options IV...')
    eq = [s for s in STOCKS if 'Equity' in s]
    try:
        req = bq.execute(bql.Request(eq, {
            'atm_iv': bq.data.implied_volatility(expiry='30D', pct_moneyness='100'),
            'put25':  bq.data.implied_volatility(expiry='30D', delta='25', put_call='PUT'),
            'call25': bq.data.implied_volatility(expiry='30D', delta='25', put_call='CALL'),
        }))
        df = _bql(req).groupby(level=0).last()

        try:
            req_pcr = bq.execute(bql.Request(eq, bq.data.put_call_open_interest_ratio()))
            df_pcr  = _bql(req_pcr).groupby(level=0).last()
            pcr_col = df_pcr.columns[0]
        except:
            df_pcr  = None
            pcr_col = None

        rows = []
        for s in eq:
            t   = TICKER[s]
            row = df.loc[s] if s in df.index else pd.Series(dtype=float)
            va  = _v(row, 'atm_iv')
            vp  = _v(row, 'put25')
            vc  = _v(row, 'call25')
            pcr = _v(df_pcr.loc[s], pcr_col) if (df_pcr is not None and s in df_pcr.index and pcr_col) else None
            rows.append({
                'ticker':   t,
                'atm_iv':   round(va / 100, 4)        if va else '',
                'skew_25d': round((vp - vc) / 100, 4) if vp and vc else '',
                'pcr_oi':   round(pcr, 3)              if pcr else '',
            })
        _csv('options_iv', rows, ['ticker','atm_iv','skew_25d','pcr_oi'])
    except Exception as e:
        print(f'  [ERRO] options_iv: {e}')

# ─────────────────────────────────────────────────────────────────────────────
# 3. PRECOS E RETORNOS (todas as classes de ativos)
# ─────────────────────────────────────────────────────────────────────────────

# Bloomberg → (yf_ticker, friendly_name)
_ALL_PRICES = {
    # Indices
    'SPX Index':   ('^GSPC',     'S&P 500'),
    'NDX Index':   ('^NDX',      'Nasdaq 100'),
    'RTY Index':   ('^RUT',      'Russell 2000'),
    'VIX Index':   ('^VIX',      'VIX'),
    # ETFs de renda fixa e commodities
    'TLT US Equity': ('TLT',     'Treasury 20yr'),
    'HYG US Equity': ('HYG',     'High Yield'),
    'GLD US Equity': ('GLD',     'Gold ETF'),
    'SPY US Equity': ('SPY',     'S&P 500 ETF'),
    'QQQ US Equity': ('QQQ',     'Nasdaq 100 ETF'),
    # Commodities
    'GC1 Comdty':  ('GC=F',      'Gold Futures'),
    'CL1 Comdty':  ('CL=F',      'WTI Crude'),
    # Crypto
    'XBT Curncy':  ('BTC-USD',   'Bitcoin'),
    # FX
    'DXY Curncy':  ('DX-Y.NYB',  'USD Index'),
    # Equities (Mag7 + blue chips)
    'AAPL US Equity':  ('AAPL',  'Apple'),
    'MSFT US Equity':  ('MSFT',  'Microsoft'),
    'NVDA US Equity':  ('NVDA',  'NVIDIA'),
    'AMZN US Equity':  ('AMZN',  'Amazon'),
    'META US Equity':  ('META',  'Meta'),
    'GOOGL US Equity': ('GOOGL', 'Alphabet'),
    'TSLA US Equity':  ('TSLA',  'Tesla'),
    'BRK/B US Equity': ('BRK-B', 'Berkshire B'),
    'AVGO US Equity':  ('AVGO',  'Broadcom'),
    'JPM US Equity':   ('JPM',   'JPMorgan'),
    'LLY US Equity':   ('LLY',   'Eli Lilly'),
    'UNH US Equity':   ('UNH',   'UnitedHealth'),
    'XOM US Equity':   ('XOM',   'ExxonMobil'),
    'COST US Equity':  ('COST',  'Costco'),
    'V US Equity':     ('V',     'Visa'),
    'MA US Equity':    ('MA',    'Mastercard'),
    'WMT US Equity':   ('WMT',   'Walmart'),
    'NFLX US Equity':  ('NFLX',  'Netflix'),
    'JNJ US Equity':   ('JNJ',   'Johnson & Johnson'),
    'PG US Equity':    ('PG',    'Procter & Gamble'),
}


def export_prices():
    print('Precos e retornos...')
    bbg_list   = list(_ALL_PRICES.keys())
    today_dt   = date.today()
    ytd_start  = f"{today_dt.year}-01-02"   # primeiro dia util do ano

    try:
        items = {
            'px_now':  bq.data.px_last(),
            'px_prev': bq.data.px_last(dates=bq.func.range('-2D', '-1D'), fill='PREV'),
            'px_week': bq.data.px_last(dates=bq.func.range('-7D', '-6D'), fill='PREV'),
        }
        req = bq.execute(bql.Request(bbg_list, items))
        df  = _bql(req).groupby(level=0).last()
        _norm(df)

        # YTD start (primeira sessao de janeiro)
        px_ytd_map = {}
        try:
            ytd_item  = bq.data.px_last(dates=bq.func.range(ytd_start, ytd_start), fill='PREV')
            req_ytd   = bq.execute(bql.Request(bbg_list, ytd_item))
            df_ytd    = _bql(req_ytd).groupby(level=0).last()
            ytd_col   = df_ytd.columns[0] if len(df_ytd.columns) else None
            if ytd_col:
                for t, row in df_ytd.iterrows():
                    v = _v(row, ytd_col)
                    if v:
                        px_ytd_map[t] = v
        except Exception as e:
            print(f'  YTD fallback: {e}')

        rows = []
        for bbg, (yf_tk, name) in _ALL_PRICES.items():
            if bbg not in df.index:
                continue
            row     = df.loc[bbg]
            px_now  = _v(row, 'px_now')
            px_prev = _v(row, 'px_prev')
            px_wk   = _v(row, 'px_week')
            px_ytd  = px_ytd_map.get(bbg)

            if not px_now:
                continue

            daily  = round((px_now - px_prev) / px_prev, 4) if px_prev and px_prev > 0 else ''
            weekly = round((px_now - px_wk)   / px_wk,   4) if px_wk   and px_wk   > 0 else ''
            ytd    = round((px_now - px_ytd)  / px_ytd,  4) if px_ytd  and px_ytd  > 0 else ''

            rows.append({
                'yf_ticker':     yf_tk,
                'name':          name,
                'price':         round(px_now, 4),
                'daily_return':  daily,
                'weekly_return': weekly,
                'ytd_return':    ytd,
            })

        _csv('prices', rows, ['yf_ticker', 'name', 'price', 'daily_return', 'weekly_return', 'ytd_return'])
    except Exception as e:
        print(f'  [ERRO] prices: {e}')


def export_price_history():
    """Exporta o fechamento de HOJE para todos os tickers. O banco local acumula dia a dia."""
    print('Historico de precos (hoje)...')
    bbg_list  = list(_ALL_PRICES.keys())
    today_str = date.today().isoformat()

    try:
        req = bq.execute(bql.Request(bbg_list, bq.data.px_last()))
        df  = _bql(req).groupby(level=0).last()
        col = df.columns[0] if len(df.columns) else None
        if not col:
            return

        rows = []
        for bbg, (yf_tk, _) in _ALL_PRICES.items():
            if bbg not in df.index:
                continue
            px = _v(df.loc[bbg], col)
            if px:
                rows.append({'date': today_str, 'yf_ticker': yf_tk, 'price': round(px, 4)})

        _csv('price_history', rows, ['date', 'yf_ticker', 'price'])
    except Exception as e:
        print(f'  [ERRO] price_history: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# 4. GEX — Maiores empresas dos EUA (AAPL MSFT NVDA AMZN META GOOGL TSLA)
#    Gamma nao existe como campo BQL — calcula via Black-Scholes com VIX como IV
# ─────────────────────────────────────────────────────────────────────────────

BIG_TECH = [
    'AAPL US Equity', 'MSFT US Equity', 'NVDA US Equity', 'AMZN US Equity',
    'META US Equity', 'GOOGL US Equity', 'TSLA US Equity',
]

def _bs_gamma(S, K, T, vol):
    """Gamma Black-Scholes — identico para call e put."""
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        import math as _math
        d1 = (_math.log(S / K) + 0.5 * vol * vol * T) / (vol * _math.sqrt(T))
        return _math.exp(-0.5 * d1 * d1) / (_math.sqrt(2 * _math.pi) * S * vol * _math.sqrt(T))
    except:
        return 0.0

def export_gex_bigtech():
    print('GEX maiores empresas...')

    # VIX como proxy de IV (evita buscar ivol() para cada opcao)
    vix_iv = 0.20
    try:
        vix_df = _norm(_bql(bq.execute('get(PX_LAST) for(["VIX Index"])')).groupby(level=0).last())
        vix_val = _v(vix_df.iloc[0], 'PX_LAST')
        if vix_val: vix_iv = vix_val / 100.0
        print(f'  VIX: {vix_iv:.1%}')
    except Exception as e:
        print(f'  VIX fallback 20%: {e}')

    # Precos spot
    try:
        tstr  = ', '.join(f'"{s}"' for s in BIG_TECH)
        spots = _norm(_bql(bq.execute(f'get(PX_LAST) for([{tstr}])')).groupby(level=0).last())
    except Exception as e:
        print(f'  [ERRO] GEX spot: {e}'); return

    rows_all = []; summary = []
    today = date.today()

    for s in BIG_TECH:
        t = s.replace(' US Equity', '')
        try:
            px = _v(spots.loc[s], 'PX_LAST') if s in spots.index else None
            if not px: continue
            lo, hi = px * 0.95, px * 1.05

            # Busca chain: OI + metadados (STRIKE_PX, PUT_CALL, EXPIRE_DT vem do universo)
            req_g  = bq.execute(bql.Request(bq.univ.options([s]), {'OI': bq.data.open_int()}))
            df_g   = _bql(req_g).reset_index()
            df_g.columns = [c.split('(')[0].strip() for c in df_g.columns]

            # Renomeia colunas padrao do universo de opcoes
            renames = {}
            for c in df_g.columns:
                cu = c.upper()
                if 'STRIKE' in cu: renames[c] = 'Strike'
                elif 'PUT_CALL' in cu or 'TYPE' in cu: renames[c] = 'Type'
                elif 'EXPIRE' in cu or ('DATE' in cu and c != 'date'): renames[c] = 'Expire'
            df_g.rename(columns=renames, inplace=True)

            ticker_gex = 0.0
            for _, r in df_g.iterrows():
                stk = r.get('Strike')
                pc  = str(r.get('Type', '')).upper()
                oi  = r.get('OI')
                exp = r.get('Expire')

                if stk is None or oi is None or not pc: continue
                stk = float(stk); oi = float(oi)
                if oi <= 0 or not (lo <= stk <= hi): continue

                # DTE em anos
                try:
                    exp_date = exp.date() if hasattr(exp, 'date') else date.fromisoformat(str(exp)[:10])
                    T = max((exp_date - today).days, 0) / 365.0
                except:
                    T = 30 / 365.0  # fallback 30 dias

                gm  = _bs_gamma(px, stk, T, vix_iv)
                gex = oi * gm * px**2 / 1e9 * (1 if pc == 'CALL' else -1)
                ticker_gex += gex
                rows_all.append({'ticker': t, 'strike': round(stk, 2), 'put_call': pc,
                                 'open_int': int(oi), 'gamma': round(gm, 6), 'gex_bn': round(gex, 4)})

            direction = 'buy' if ticker_gex > 0 else 'sell'
            summary.append({'ticker': t, 'spot': round(px, 2), 'gex_bn': round(ticker_gex, 4), 'direction': direction})
            print(f'    {t}: GEX={ticker_gex:+.3f}B ({direction})')
        except Exception as e:
            print(f'    [ERRO] GEX {t}: {e}')

    if rows_all: _csv('gex_bigtech', rows_all, ['ticker','strike','put_call','open_int','gamma','gex_bn'])
    if summary:  _csv('gex_bigtech_summary', summary, ['ticker','spot','gex_bn','direction'])

    # ── Escreve gex_summary e gex_spx no formato que bql_csv.py espera ────────
    if summary:
        total_gex = sum(s['gex_bn'] for s in summary)
        call_gex  = sum(r['gex_bn'] for r in rows_all if r['put_call'] == 'CALL') if rows_all else 0.0
        put_gex   = sum(r['gex_bn'] for r in rows_all if r['put_call'] == 'PUT')  if rows_all else 0.0
        direction = 'buy' if total_gex > 0.1 else ('sell' if total_gex < -0.1 else 'flat')
        gamma_reg = 'long' if total_gex > 0.5 else ('short' if total_gex < -0.5 else 'flat')
        # Pega preco spot do SPX (aproximado pelo maior ticker da lista)
        spot_spx  = next((s['spot'] for s in summary if s['ticker'] in ('AAPL','MSFT')), 0)
        _csv('gex_summary', [{
            'date':          str(date.today()),
            'spot':          spot_spx,
            'gex_total_bn':  round(total_gex, 3),
            'gex_call_bn':   round(call_gex,  3),
            'gex_put_bn':    round(put_gex,   3),
            'direction':     direction,
            'gamma_regime':  gamma_reg,
            'n_options':     len(rows_all),
        }], ['date','spot','gex_total_bn','gex_call_bn','gex_put_bn','direction','gamma_regime','n_options'])

    # gex_spx: usa as mesmas linhas de todas as empresas, expiry vazio (nao temos expiry individual)
    if rows_all:
        gex_spx_rows = [
            {'expiry': '', 'strike': r['strike'], 'put_call': r['put_call'],
             'open_int': r['open_int'], 'gamma': r['gamma'], 'gex_bn': r['gex_bn']}
            for r in rows_all
        ]
        _csv('gex_spx', gex_spx_rows, ['expiry','strike','put_call','open_int','gamma','gex_bn'])

# ─────────────────────────────────────────────────────────────────────────────
# 4. LETF FLOWS
# ─────────────────────────────────────────────────────────────────────────────

def export_letf():
    print('LETF flows...')
    LF = {
        'TQQQ US Equity': ('TQQQ',  3, 'NDX'),
        'SQQQ US Equity': ('SQQQ', -3, 'NDX'),
        'UPRO US Equity': ('UPRO',  3, 'SPX'),
        'SPXS US Equity': ('SPXS', -3, 'SPX'),
        'SOXL US Equity': ('SOXL',  3, 'SOX'),
        'SOXS US Equity': ('SOXS', -3, 'SOX'),
    }
    try:
        lstr  = ', '.join(f'"{t}"' for t in LF)
        req   = bq.execute(f'get(PX_LAST, FUND_TOTAL_ASSETS, FUND_NET_ASSET_VAL) for([{lstr}])')
        dl    = _norm(_bql(req).groupby(level=0).last())
        rows  = []
        for b, (sym, lev, idx) in LF.items():
            r   = dl.loc[b] if b in dl.index else pd.Series(dtype=float)
            aum = _v(r, 'FUND_TOTAL_ASSETS')
            nav = _v(r, 'FUND_NET_ASSET_VAL') or _v(r, 'PX_LAST')
            rows.append({
                'ticker':   sym,
                'leverage': lev,
                'index':    idx,
                'nav':      round(nav, 4)       if nav else '',
                'aum_b':    round(aum / 1e9, 4) if aum else '',
            })
        _csv('letf_flows', rows, ['ticker','leverage','index','nav','aum_b'])
    except Exception as e:
        print(f'  [ERRO] letf: {e}')

# ─────────────────────────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def export_all():
    export_fundamentals()
    export_options_iv()
    export_gex_bigtech()
    export_letf()
    export_prices()
    export_price_history()
    _csv('meta',
         [{'generated_at': datetime.now().isoformat(), 'date': str(date.today())}],
         ['generated_at','date'])

# ─────────────────────────────────────────────────────────────────────────────
print(f'Pasta de saida: {OUT}')
print(f'Intervalo: {INTERVAL}s | Kernel Interrupt (■) para parar\n')

bq = bql.Service()
print('BQL conectado.\n')

cycle = 0
while True:
    cycle += 1
    t0 = time.time()
    print(f'=== Ciclo {cycle} [{datetime.now().strftime("%H:%M:%S")}] ===')
    export_all()
    elapsed = time.time() - t0
    sleep   = max(10, INTERVAL - elapsed)
    print(f'Pronto em {elapsed:.0f}s — proxima em {sleep:.0f}s\n')
    time.sleep(sleep)
