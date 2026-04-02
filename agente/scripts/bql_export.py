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

# Pasta de saida — Path.home() funciona tanto no BQuant cloud quanto local
OUT = Path.home() / 'bql_data'
OUT.mkdir(parents=True, exist_ok=True)

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
# 3. GEX — MAG 7 (options individuais, muito mais rapido que SPX inteiro)
# ─────────────────────────────────────────────────────────────────────────────

MAG7 = [
    'AAPL US Equity', 'MSFT US Equity', 'NVDA US Equity', 'AMZN US Equity',
    'META US Equity', 'GOOGL US Equity', 'TSLA US Equity',
]
MAG7_TICKER = {s: s.replace(' US Equity', '') for s in MAG7}

def export_gex_mag7():
    print('GEX Mag7...')
    rows_all = []
    summary  = []
    try:
        spot_req = bq.execute(
            'get(PX_LAST) for(["' + '","'.join(MAG7) + '"])'
        )
        spots = _norm(_bql(spot_req).groupby(level=0).last())
    except Exception as e:
        print(f'  [ERRO] GEX spot: {e}')
        return

    for s in MAG7:
        t = MAG7_TICKER[s]
        try:
            px = float(spots.loc[s, 'PX_LAST']) if s in spots.index else None
            if not px:
                continue
            lo, hi = px * 0.95, px * 1.05  # ±5% do spot

            univ  = bq.univ.options(s)
            # bql.Request em vez de string — evita o FutureExt
            req_g = bq.execute(bql.Request(univ, {
                'OPEN_INT': bq.data.open_interest(),
                'GAMMA':    bq.data.gamma(),
            }))
            df_g  = _norm(_bql(req_g).reset_index())

            # Detecta colunas de data, put_call e strike (nomes variam no BQuant)
            dc  = next((c for c in df_g.columns if 'DATE'     in c.upper() or 'EXPIR' in c.upper()), None)
            pcc = next((c for c in df_g.columns if 'PUT_CALL' in c.upper() or 'TYPE'  in c.upper()), None)
            skc = next((c for c in df_g.columns if 'STRIKE'   in c.upper()), None)
            oic = next((c for c in df_g.columns if 'OPEN_INT' in c.upper() or c == 'OPEN_INT'), 'OPEN_INT')
            gmc = next((c for c in df_g.columns if c == 'GAMMA' or 'GAMMA' in c.upper()), 'GAMMA')

            ticker_gex = 0.0
            for _, r in df_g.iterrows():
                stk = float(r.get(skc) or 0) if skc else 0
                if skc and not (lo <= stk <= hi):
                    continue
                oi  = float(r.get(oic) or 0)
                gm  = float(r.get(gmc) or 0)
                pc  = str(r.get(pcc, '')).upper() if pcc else ''
                gex = oi * gm * px ** 2 / 1e9 * (1 if pc == 'CALL' else -1)
                ticker_gex += gex
                rows_all.append({
                    'ticker':   t,
                    'strike':   round(stk, 2),
                    'put_call': pc,
                    'open_int': int(oi),
                    'gamma':    round(gm, 6),
                    'gex_bn':   round(gex, 4),
                })
            direction = 'buy' if ticker_gex > 0 else 'sell'
            summary.append({'ticker': t, 'spot': round(px, 2),
                            'gex_bn': round(ticker_gex, 4), 'direction': direction})
            print(f'    {t}: GEX={ticker_gex:+.3f}B ({direction})')
        except Exception as e:
            print(f'    [ERRO] GEX {t}: {e}')

    if rows_all:
        _csv('gex_mag7', rows_all, ['ticker','strike','put_call','open_int','gamma','gex_bn'])
    if summary:
        _csv('gex_mag7_summary', summary, ['ticker','spot','gex_bn','direction'])

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
    export_gex_mag7()
    export_letf()
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
