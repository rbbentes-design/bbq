"""
COT (Commitment of Traders) — Script standalone para testes no BQuant.
v2: resolve contrato ativo, split position/trader queries, evita concat error.
"""

import pandas as pd
import numpy as np

# ── BQL Init ──
import bql
bq = bql.Service()

BQL_PARAMS = {'fill': 'prev'}

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES COT
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════════

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
    avail = [k for k in data_items.keys() if k in df_r.columns]
    if avail:
        df_r = df_r[avail]
    df_r.index.names = ['ID', 'Date']
    return df_r


def _concat_bql_response(resp):
    """Concat DataItems de um BQL response, usando integer index para evitar
    erro de 'Reindexing only valid with uniquely valued Index objects'."""
    dfs = [di.df().reset_index() for di in resp]
    raw = pd.concat([d.reset_index(drop=True) for d in dfs], axis=1)
    return raw.loc[:, ~raw.columns.duplicated()]


def _has_real_data(df):
    """Verifica se o DataFrame tem dados reais (não só NaN/NaT/None)."""
    if df is None or df.empty:
        return False
    non_id = df.drop(columns=['ID'], errors='ignore')
    return not non_id.isna().all().all()


# Trader types especulativos por report type (sinal útil para fluxo)
SPEC_TRADER_TYPES = {
    'cftc_disaggregated': ['MANAGED_MONEY'],
    'cftc_tff': ['ASSET_MANAGER', 'LEVERAGED_FUNDS'],
    'cftc_legacy': ['NON_COMMERCIAL'],
}


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES COT
# ═══════════════════════════════════════════════════════════════════════════════

def has_cot(ticker):
    """Verifica se o ticker possui dados COT. Retorna (True, futures_ticker) ou (False, None)."""
    if ticker in COT_FUTURES_MAP:
        return True, COT_FUTURES_MAP[ticker]
    t = ticker.strip()
    if t.endswith('Comdty') or t.endswith('Index'):
        return True, ticker
    return False, None


def fetch_cot_data(futures_ticker, start='-2Y', end='0D'):
    """Busca COT histórico: filtra trader types especulativos, retorna série semanal."""
    tickers = [futures_ticker] if isinstance(futures_ticker, str) else list(futures_ticker)

    raw = pd.DataFrame()
    used_rpt = ''
    used_ticker = ''

    for t in tickers:
        for rpt in ('cftc_disaggregated', 'cftc_tff', 'cftc_legacy'):
            try:
                # Query com dates=range para histórico completo
                q = f"""
let(#p = cot_position(report_type={rpt}, direction=all,
                      trader_type=all, commitment_type=futures,
                      dates=range({start},{end}));)
for('{t}') get(#p().date, #p().trader_type, #p().direction,
               #p().value, #p().change)
"""
                print(f"[COT] Tentando {rpt} com {t} + dates=range({start},{end})…")
                resp = bq.execute(q)
                pos_raw = _concat_bql_response(resp)
                print(f"[COT]   shape={pos_raw.shape}, cols={list(pos_raw.columns)}")

                if not _has_real_data(pos_raw):
                    print(f"[COT]   → sem dados reais")
                    continue

                print(f"[COT]   ✓ head:\n{pos_raw.head(3)}")
                raw = pos_raw
                used_rpt = rpt
                used_ticker = t
                break
            except Exception as e:
                # Fallback: sem dates=range (retorna só últimos dados)
                print(f"[COT]   ✗ com dates: {e}")
                try:
                    q2 = f"""
let(#p = cot_position(report_type={rpt}, direction=all,
                      trader_type=all, commitment_type=futures);)
for('{t}') get(#p().date, #p().trader_type, #p().direction,
               #p().value, #p().change)
"""
                    print(f"[COT]   Fallback {rpt} sem dates…")
                    resp2 = bq.execute(q2)
                    pos_raw2 = _concat_bql_response(resp2)
                    if _has_real_data(pos_raw2):
                        print(f"[COT]   ✓ fallback: shape={pos_raw2.shape}")
                        print(f"[COT]   head:\n{pos_raw2.head(3)}")
                        raw = pos_raw2
                        used_rpt = rpt
                        used_ticker = t
                        break
                except Exception as e2:
                    print(f"[COT]   ✗ fallback: {e2}")

        if not raw.empty:
            break

    if raw.empty:
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

    # ── Normalizar Direction ──
    if 'Direction' in raw.columns:
        raw['Direction'] = raw['Direction'].astype(str).str.strip().str.upper()
    if 'TraderType' in raw.columns:
        raw['TraderType'] = raw['TraderType'].astype(str).str.strip().str.upper()

    print(f"[COT] Rename: cols={list(raw.columns)}, report={used_rpt}")
    print(f"[COT] TraderTypes: {raw['TraderType'].unique().tolist() if 'TraderType' in raw.columns else 'N/A'}")
    print(f"[COT] Directions: {raw['Direction'].unique().tolist() if 'Direction' in raw.columns else 'N/A'}")
    print(f"[COT] Dates: {raw['Date'].nunique() if 'Date' in raw.columns else 'N/A'} únicas")

    # ── Filtrar trader types especulativos ──
    spec_types = SPEC_TRADER_TYPES.get(used_rpt, [])
    if 'TraderType' in raw.columns and spec_types:
        spec_mask = raw['TraderType'].isin(spec_types)
        n_before = len(raw)
        raw = raw[spec_mask]
        print(f"[COT] Filtro spec ({spec_types}): {n_before} → {len(raw)} rows")
    if raw.empty:
        print("[COT] Vazio após filtro de trader types")
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

    # ── Pivot: agregar spec traders por Date × Direction ──
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

    # ── Compute Net = Long - Short (se ambos existem) ──
    if 'Positions - Net' in df.columns:
        df['Positions'] = df['Positions - Net']
    elif 'Positions - Long' in df.columns and 'Positions - Short' in df.columns:
        df['Positions'] = df['Positions - Long'] + df['Positions - Short']
    elif 'Positions - Long' in df.columns:
        df['Positions'] = df['Positions - Long']
    else:
        df['Positions'] = df.iloc[:, 0]

    # ── Price & Open Interest ──
    try:
        dates_bql = bq.func.range(start, end, frq='d')
        px_items = {'Price': bq.data.px_last(fill='prev'),
                    'Open Interest': bq.data.fut_aggte_open_int()}
        px_df = _fp_get_data(used_ticker, px_items,
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
    print(f"[COT] Dates range: {dt_idx.min()} → {dt_idx.max()}")
    print(f"[COT] Positions (last): {df['Positions'].iloc[-1]:,.0f}")
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
            f'{col} Change': s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else np.nan,
            f'{col} Percentile': pd.cut(s, 100, labels=False).iloc[-1],
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
# TESTES
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    test_tickers = [
        ('ES1 Index', 'S&P 500 E-mini'),
        ('GC1 Comdty', 'Gold'),
        ('CL1 Comdty', 'WTI Crude'),
    ]

    for ticker, name in test_tickers:
        print('\n' + '=' * 70)
        print(f'TESTE: {name} ({ticker})')
        print('=' * 70)

        ok, fut = has_cot(ticker)
        print(f'has_cot → ok={ok}, fut={fut}')

        df = fetch_cot_data(ticker, start='-1Y', end='0D')
        if df is None or df.empty:
            print(f'⚠️  Sem dados para {ticker}')
            continue

        print(f'\nfetch_cot_data → shape={df.shape}')
        print(f'colunas: {list(df.columns)}')
        print(f'index names: {df.index.names}')
        print(f'dtypes:\n{df.dtypes}')
        print(f'\nhead(5):\n{df.head(5)}')
        print(f'\ntail(3):\n{df.tail(3)}')

        agg = aggregate_cot(df)
        print(f'\naggregate_cot → shape={agg.shape}')
        print(f'colunas: {list(agg.columns)}')
        print(f'tail(3):\n{agg.tail(3)}')

        stats = cot_summary_stats(agg)
        print(f'\ncot_summary_stats:\n{stats}')

        seas = cot_seasonality(agg)
        print(f'\ncot_seasonality → shape={seas.shape}')
        print(f'head(5):\n{seas.head(5)}')

    # ── Teste via safe_fetch_cot ──
    print('\n' + '=' * 70)
    print('TESTE safe_fetch_cot com SPX Index (mapeia → ES1)')
    print('=' * 70)
    result = safe_fetch_cot('SPX Index', start='-1Y', end='0D')
    if result is not None:
        print(f'shape={result.shape}')
        print(f'colunas: {list(result.columns)}')
        print(f'tail(3):\n{result.tail(3)}')
    else:
        print('⚠️  safe_fetch_cot retornou None')
