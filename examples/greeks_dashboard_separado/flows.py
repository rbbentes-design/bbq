"""Flow analysis: ETF rebalancing, COT, buybacks, flow scoring, CTA, dealer hedging, vol control, risk parity."""

import numpy as np
import pandas as pd
import traceback
import warnings
import math
import random as _random
from datetime import datetime, timedelta
from functools import lru_cache
from scipy.stats import norm
from scipy.optimize import minimize as sp_minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import bql

try:
    from .config import (bq, _C, INDEX_PROXY, LEVERAGED_ETFS, LEVERAGED_ETFS_EXT,
                         HAS_SKLEARN, HAS_BQPLOT, DASH_TEMPLATE, FLOW_FIG_LAYOUT,
                         PASSIVE_ETFS, COT_FUTURES_MAP, COT_CONTRACTS, DEFAULT_AUM,
                         COT_TRADER_REPORT_MAP, SPX_ANNUAL_BUYBACK_EST, TRADING_DAYS,
                         _greek_cache, _snapshot, wd, BQL_PARAMS,
                         MM_VOLUME_SHARES, OPTIONS_TOTAL_ADC)
except ImportError:
    import sys, os as _os
    _dir = _os.path.dirname(_os.path.abspath(__file__)) if '__file__' in dir() else _os.getcwd()
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from config import (bq, _C, INDEX_PROXY, LEVERAGED_ETFS, LEVERAGED_ETFS_EXT,
                        HAS_SKLEARN, HAS_BQPLOT, DASH_TEMPLATE, FLOW_FIG_LAYOUT,
                        PASSIVE_ETFS, COT_FUTURES_MAP, COT_CONTRACTS, DEFAULT_AUM,
                        COT_TRADER_REPORT_MAP, SPX_ANNUAL_BUYBACK_EST, TRADING_DAYS,
                        _greek_cache, _snapshot, wd, BQL_PARAMS,
                        MM_VOLUME_SHARES, OPTIONS_TOTAL_ADC)

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline
except ImportError:
    pass


def _last_spx_rebal_date():
    """Última data de rebalanceamento trimestral do S&P 500 (3ª sexta de Mar/Jun/Set/Dez)."""
    today = pd.Timestamp.now().normalize()
    rebal_months = [3, 6, 9, 12]
    candidates = []
    for year in [today.year - 1, today.year]:
        for month in rebal_months:
            first = pd.Timestamp(year, month, 1)
            dow = first.dayofweek  # Monday=0, Friday=4
            days_to_fri = (4 - dow) % 7
            first_friday = first + pd.Timedelta(days=days_to_fri)
            third_friday = first_friday + pd.Timedelta(weeks=2)
            if third_friday < today:
                candidates.append(third_friday)
    return max(candidates) if candidates else today - pd.Timedelta(days=90)


def _float_weights(idx, as_of=None):
    """Retorna pesos float-adjusted normalizados."""
    uni = bq.univ.members([idx], dates=[as_of]) if as_of else bq.univ.members([idx])
    cap = bq.data.cur_mkt_cap(dates=[as_of]) if as_of else bq.data.cur_mkt_cap()
    df_cap = bq.execute(bql.Request(uni, [cap], with_params=BQL_PARAMS)
                        )[0].df().select_dtypes('number').rename(columns=lambda c: 'Cap')
    df_ff = bq.execute(bql.Request(bq.univ.members([idx]),
                                   [bq.data.eqy_free_float_pct()],
                                   with_params=BQL_PARAMS)
                       )[0].df().select_dtypes('number').rename(columns=lambda c: 'FF')
    df_ = df_cap.join(df_ff, how='inner')
    df_['FMC'] = df_['Cap'] * df_['FF'] / 100
    df_['Weight'] = df_['FMC'] / df_['FMC'].sum()
    return df_['Weight']


def _adv5_usd(tickers_list):
    """Média dos últimos 5 dias de $-volume."""
    usd_val = (bq.data.px_volume(dates=bq.func.range('-5D', '-1D'))
               * bq.data.px_last(dates=bq.func.range('-5D', '-1D')))
    adv_item = bq.func.avg(usd_val)
    return bq.execute(bql.Request(bq.univ.List(tickers_list),
                                  {'ADV5': adv_item},
                                  with_params=BQL_PARAMS))[0].df()['ADV5']


def compute_full_etf_flows(index_proxy=INDEX_PROXY, start_date_override=None):
    """
    Calcula fluxos de rebalanceamento por ETF passivo + combinado.
    Retorna (flows_dict, summary_df, start_date).
    flows_dict: {'Combined': df, 'VOO US Equity': df, ...}
    """
    if start_date_override:
        start_date = pd.Timestamp(start_date_override).strftime('%Y-%m-%d')
    else:
        start_date = _last_spx_rebal_date().strftime('%Y-%m-%d')

    w0 = _float_weights(index_proxy, start_date)
    w1 = _float_weights(index_proxy)
    delta = pd.DataFrame({'Start': w0, 'Now': w1}).dropna()
    delta['Delta'] = delta['Now'] - delta['Start']

    adv5 = _adv5_usd(delta.index.tolist()).fillna(1)

    aum = bq.execute(bql.Request(
        bq.univ.List(PASSIVE_ETFS),
        {'AUM': bq.data.fund_total_assets()},
        with_params=BQL_PARAMS))[0].df().select_dtypes('number').iloc[:, 0]

    flows = {}
    for etf, aum_val in aum.items():
        t = delta.copy()
        t['Flow_$'] = t['Delta'] * aum_val
        t['PctADV'] = (t['Flow_$'] / adv5.reindex(t.index).fillna(1)) * 100
        flows[etf] = t[t['Flow_$'] != 0].sort_values('Flow_$', ascending=False)

    combo = delta.copy()
    combo['Flow_$'] = sum(flows[etf].reindex(delta.index, fill_value=0)['Flow_$']
                          for etf in flows)
    combo['PctADV'] = (combo['Flow_$'] / adv5.reindex(combo.index).fillna(1)) * 100
    flows['Combined'] = combo[combo['Flow_$'] != 0].sort_values('Flow_$', ascending=False)

    summary_rows = []
    for etf in PASSIVE_ETFS:
        df_f = flows.get(etf, pd.DataFrame(columns=['Flow_$']))
        buy = df_f.loc[df_f['Flow_$'] > 0, 'Flow_$'].sum() if not df_f.empty else 0
        sell = -df_f.loc[df_f['Flow_$'] < 0, 'Flow_$'].sum() if not df_f.empty else 0
        summary_rows.append({'ETF': etf.split()[0], 'Buy_$': buy, 'Sell_$': sell, 'Net_$': buy - sell})
    summary = pd.DataFrame(summary_rows).set_index('ETF')

    return flows, summary, start_date


def compute_leveraged_flows(daily_return):
    """
    Calcula fluxo de rebalanceamento end-of-day dos ETFs alavancados.
    Fórmula: Rebalance_$ = AUM × L × (L-1) × r / (1 + L×r)
    """
    tickers = [e['ticker'] for e in LEVERAGED_ETFS]
    print(f"[LEV] daily_return={daily_return:.6f}")
    aum_df = None
    try:
        aum_df = bq.execute(bql.Request(
            tickers,
            {'AUM': bq.data.fund_total_assets()},
            with_params=BQL_PARAMS))[0].df()
        print(f"[LEV] AUM fetched: {aum_df.shape}")
    except Exception as e:
        print(f"[LEV] AUM fetch failed: {e}")

    rows = []
    for etf in LEVERAGED_ETFS:
        tk = etf['ticker']
        L = etf['leverage']
        r = daily_return
        try:
            aum = pd.to_numeric(aum_df.loc[tk].iloc[0], errors='coerce') if aum_df is not None else np.nan
        except (KeyError, TypeError, IndexError):
            aum = np.nan
        if pd.isna(aum):
            aum = DEFAULT_AUM.get(etf['name'], 1e9)

        if pd.notna(aum) and (1 + L * r) != 0:
            rebal = aum * L * (L - 1) * r / (1 + L * r)
        else:
            rebal = np.nan

        rows.append({
            'ETF': etf['name'],
            'Leverage': f"{L:+d}x",
            'AUM_$': aum,
            'Rebalance_$': rebal,
            'Direção': ('COMPRAR' if pd.notna(rebal) and rebal > 0
                        else 'VENDER' if pd.notna(rebal) and rebal < 0
                        else 'N/A')
        })

    result = pd.DataFrame(rows).set_index('ETF')
    total_flow = result['Rebalance_$'].sum()
    print(f"[LEV] Total flow: ${total_flow:,.0f}")
    return result, total_flow


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5B — PREVISÃO DE REBALANCEAMENTO DO S&P 500 (inclusão/exclusão)
# ═══════════════════════════════════════════════════════════════════════════════

def build_spx_prediction():
    """
    Constrói modelo de previsão de inclusão/exclusão do S&P 500.
    Usa 4 variáveis: FMC, FALR (annual liquidity ratio), FREE_FLOAT_PCT, NET_INC_TTM.
    Retorna (top_in, top_out, model_df).
    Requer sklearn.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn não disponível. Instale com: pip install scikit-learn")

    def _fetch(u, item, name):
        df_ = bq.execute(bql.Request(u, item))[0].df()
        df_ = df_.select_dtypes('number').iloc[:, 0].to_frame(name)
        return df_.loc[~df_.index.duplicated(keep='first')]

    # 1 — Universos
    spx_ids = bq.execute(
        bql.Request(bq.univ.members(['SPX Index']), bq.data.id())
    )[0].df().index.tolist()

    # S&P 500: market cap mínimo ~$18B (regra atualizada)
    base_ids = bq.execute(
        bql.Request(
            bq.univ.filter(
                bq.univ.equitiesuniv(['active', 'primary']),
                bq.func.and_(
                    bq.data.cntry_of_domicile() == 'US',
                    bq.data.cur_mkt_cap() > 1.8e10)),
            bq.data.id())
    )[0].df().index.tolist()

    neg_ids = [t for t in base_ids if t not in spx_ids]
    _random.seed(42)
    neg_sample = _random.sample(neg_ids, k=min(len(neg_ids), len(spx_ids) * 2))
    universe = spx_ids + neg_sample

    # 2 — ADVT anual
    advt_series = (
        bq.data.px_volume(dates=bq.func.range('-365D', '0D')) *
        bq.data.px_last(dates=bq.func.range('-365D', '0D')))
    advt_item = bq.func.sum(advt_series)

    # 3 — Coleta em batches
    records = []
    batch_size = 50
    for i in range(0, len(universe), batch_size):
        u = bq.univ.List(universe[i:i + batch_size])
        dfb = pd.concat([
            _fetch(u, bq.data.cur_mkt_cap(), 'CUR_MKT_CAP'),
            _fetch(u, bq.data.eqy_free_float_pct(), 'FREE_FLOAT_PCT'),
            _fetch(u, advt_item, 'ADVT'),
            _fetch(u, bq.data.net_income(fa_period_offset='0'), 'NI_Q0'),
            _fetch(u, bq.data.net_income(fa_period_offset='1'), 'NI_Q1'),
            _fetch(u, bq.data.net_income(fa_period_offset='2'), 'NI_Q2'),
            _fetch(u, bq.data.net_income(fa_period_offset='3'), 'NI_Q3'),
        ], axis=1, join='inner')
        records.append(dfb)

    df_ = pd.concat(records)

    # 4 — Feature engineering
    df_['FMC'] = df_['CUR_MKT_CAP'] * df_['FREE_FLOAT_PCT'] / 100
    df_['FALR'] = df_['ADVT'] / df_['FMC']
    df_['NET_INC_TTM'] = df_[['NI_Q0', 'NI_Q1', 'NI_Q2', 'NI_Q3']].sum(axis=1)

    roots_all = df_.index.to_series().str.split().str[0]
    roots_spx = pd.Series(spx_ids).str.split().str[0]
    df_['IN_SPX'] = roots_all.isin(set(roots_spx)).astype(int)
    df_ = df_.dropna()

    # 4b — Flag de elegibilidade S&P 500
    # Regras: Free float >= 50%, lucro TTM > 0, último trimestre > 0, FALR >= 0.75
    df_['ELIGIBLE'] = (
        (df_['FREE_FLOAT_PCT'] >= 50) &
        (df_['NET_INC_TTM'] > 0) &
        (df_['NI_Q0'] > 0) &
        (df_['FALR'] >= 0.75) &
        (df_['CUR_MKT_CAP'] >= 1.8e10)
    ).astype(int)

    # 5 — Modelo LogisticRegression
    features = ['FMC', 'FALR', 'FREE_FLOAT_PCT', 'NET_INC_TTM']
    X, y = df_[features], df_['IN_SPX']
    pipe = SkPipeline([
        ('scaler', StandardScaler()),
        ('logit', LogisticRegression(solver='liblinear'))
    ])
    pipe.fit(X, y)
    df_['Prob_In'] = pipe.predict_proba(X)[:, 1]

    # 6 — ExitScore (penalidade por cap, FMC, lucro, liquidez)
    #     Thresholds atualizados conforme metodologia S&P
    cap_thr = 1.8e10   # mínimo de market cap para permanência
    fmc_thr = 0.9e10   # mínimo de float-adjusted market cap
    cap_pen = ((cap_thr - df_['CUR_MKT_CAP']).clip(lower=0)) / cap_thr
    fmc_pen = ((fmc_thr - df_['FMC']).clip(lower=0)) / fmc_thr
    earn_pen = ((df_['NET_INC_TTM'] <= 0) | (df_['NI_Q0'] <= 0)).astype(float)
    float_pen = ((50 - df_['FREE_FLOAT_PCT']).clip(lower=0)) / 50
    liq_pen = ((0.75 - df_['FALR']).clip(lower=0)) / 0.75
    df_['ExitScore'] = (cap_pen + fmc_pen + earn_pen + float_pen + liq_pen) / 5

    # 7 — Rankings
    allowed = df_.index.str.endswith((' US Equity', ' UW Equity', ' UQ Equity'))

    def _dedup_root(d, col):
        t = d.copy()
        t['root'] = t.index.str.split().str[0]
        return t.loc[t.groupby('root')[col].idxmax()]

    # Inclusão: somente elegíveis (FREE_FLOAT>=50, lucro positivo, FALR>=0.75)
    top_in = (df_[(df_['IN_SPX'] == 0) & allowed & (df_['ELIGIBLE'] == 1)]
              .pipe(_dedup_root, 'Prob_In')
              .sort_values('Prob_In', ascending=False)
              .head(30))

    top_out = (df_[(df_['IN_SPX'] == 1) & allowed]
               .pipe(_dedup_root, 'ExitScore')
               .sort_values('ExitScore', ascending=False)
               .head(30))

    return top_in, top_out, df_


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5C — COT ENGINE (Commitment of Traders)
# ═══════════════════════════════════════════════════════════════════════════════

def _fp_last_numeric_col(df_in):
    for c in df_in.columns[::-1]:
        if pd.api.types.is_numeric_dtype(df_in[c]):
            return c
    return None


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
    # Selecionar colunas disponíveis (ignorar as não retornadas)
    avail = [k for k in data_items.keys() if k in df_r.columns]
    if avail:
        df_r = df_r[avail]
    df_r.index.names = ['ID', 'Date']
    return df_r


# ── Helpers COT v4 ──
QUARTERLY_MONTHS = {3: 'H', 6: 'M', 9: 'U', 12: 'Z'}

SPEC_TRADER_TYPES = {
    'cftc_disaggregated': ['MANAGED_MONEY'],
    'cftc_tff': ['ASSET_MANAGER', 'LEVERAGED_FUNDS'],
    'cftc_legacy': ['NON_COMMERCIAL'],
}


def _concat_bql_response(resp):
    """Concat DataItems de um BQL response, usando integer index."""
    dfs = [di.df().reset_index() for di in resp]
    raw = pd.concat([d.reset_index(drop=True) for d in dfs], axis=1)
    return raw.loc[:, ~raw.columns.duplicated()]


def _has_positions_data(df):
    """Verifica se o BQL result tem dados reais de posição."""
    if df is None or df.empty:
        return False
    for c in df.columns:
        cl = str(c).lower()
        if 'value' in cl or 'position' in cl:
            vals = pd.to_numeric(df[c], errors='coerce')
            if vals.notna().any() and (vals != 0).any():
                return True
    return False


def resolve_specific_contract(generic_ticker):
    """ES1 Index → ['ESH6 Index', 'ESM6 Index'] baseado na data atual."""
    parts = generic_ticker.split()
    if len(parts) < 2:
        return []
    root_num = parts[0]
    suffix = ' '.join(parts[1:])
    root = root_num.rstrip('0123456789')
    if not root:
        return []
    now = pd.Timestamp.now()
    month, year = now.month, now.year
    sorted_qm = sorted(QUARTERLY_MONTHS.keys())
    candidates = []
    for qm in sorted_qm:
        if month <= qm:
            code = QUARTERLY_MONTHS[qm]
            candidates.append(f"{root}{code}{year % 10} {suffix}")
            if len(candidates) >= 2:
                break
    if len(candidates) < 2:
        next_year = (year + 1) % 10
        for qm in sorted_qm:
            code = QUARTERLY_MONTHS[qm]
            candidates.append(f"{root}{code}{next_year} {suffix}")
            if len(candidates) >= 2:
                break
    print(f"[resolve] {generic_ticker} → candidatos: {candidates}")
    return candidates


def _to_bql_date(s):
    """Converte data YYYYMMDD (do widget) para formato BQL relativo."""
    if not s or not isinstance(s, str):
        return s
    if s in ('0D', '0d') or (s.startswith('-') and s[-1] in 'YyDd'):
        return s
    try:
        dt = pd.to_datetime(s)
        days = (pd.Timestamp.now() - dt).days
        if days <= 0:
            return '0D'
        years = days / 365.25
        if years >= 1:
            return f'-{int(years) + 1}Y'
        return f'-{days}D'
    except Exception:
        return s


def _try_cot_query(ticker, rpt, start, end, with_dates=True):
    """Tenta uma query COT. Retorna DataFrame ou None."""
    try:
        bql_s, bql_e = _to_bql_date(start), _to_bql_date(end)
        if with_dates:
            q = f"""
let(#p = cot_position(report_type={rpt}, direction=all,
                      trader_type=all, commitment_type=futures,
                      dates=range({bql_s},{bql_e}));)
for('{ticker}') get(#p().date, #p().trader_type, #p().direction,
                    #p().value, #p().change)
"""
        else:
            q = f"""
let(#p = cot_position(report_type={rpt}, direction=all,
                      trader_type=all, commitment_type=futures);)
for('{ticker}') get(#p().date, #p().trader_type, #p().direction,
                    #p().value, #p().change)
"""
        label = f"{rpt} + {ticker}" + (" + dates" if with_dates else "")
        print(f"[COT] Tentando {label}…")
        resp = bq.execute(q)
        raw = _concat_bql_response(resp)
        print(f"[COT]   shape={raw.shape}")
        if not _has_positions_data(raw):
            print(f"[COT]   → sem dados de posição reais")
            return None
        print(f"[COT]   ✓ has real data")
        return raw
    except Exception as e:
        print(f"[COT]   ✗ {e}")
        return None


def has_cot(ticker):
    """Verifica se o ticker possui dados COT. Retorna (True, futures_ticker) ou (False, None)."""
    if ticker in COT_FUTURES_MAP:
        return True, COT_FUTURES_MAP[ticker]
    t = ticker.strip()
    if t.endswith('Comdty') or t.endswith('Index'):
        return True, ticker
    return False, None


def fetch_cot_data(futures_ticker, start='-2Y', end='0D'):
    """Busca COT histórico: tenta genérico + específico, filtra spec traders."""
    tickers = [futures_ticker] if isinstance(futures_ticker, str) else list(futures_ticker)

    # Build lista de tickers para tentar: genérico + contratos específicos
    tickers_to_try = list(tickers)
    for t in tickers:
        specific = resolve_specific_contract(t)
        tickers_to_try.extend(specific)

    raw = None
    used_rpt = ''
    used_ticker = ''

    for t in tickers_to_try:
        for rpt in ('cftc_disaggregated', 'cftc_tff', 'cftc_legacy'):
            result = _try_cot_query(t, rpt, start, end, with_dates=True)
            if result is None:
                result = _try_cot_query(t, rpt, start, end, with_dates=False)
            if result is not None:
                raw = result
                used_rpt = rpt
                used_ticker = t
                break
        if raw is not None:
            break

    if raw is None:
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

    if 'Direction' in raw.columns:
        raw['Direction'] = raw['Direction'].astype(str).str.strip().str.upper()
    if 'TraderType' in raw.columns:
        raw['TraderType'] = raw['TraderType'].astype(str).str.strip().str.upper()

    print(f"[COT] Rename: cols={list(raw.columns)}, report={used_rpt}, ticker={used_ticker}")

    # ── Filtrar trader types especulativos ──
    spec_types = SPEC_TRADER_TYPES.get(used_rpt, [])
    if 'TraderType' in raw.columns and spec_types:
        n_before = len(raw)
        raw = raw[raw['TraderType'].isin(spec_types)]
        print(f"[COT] Filtro spec ({spec_types}): {n_before} → {len(raw)} rows")
    if raw.empty:
        print("[COT] Vazio após filtro de trader types")
        return pd.DataFrame()

    # ── Drop rows com Positions NaN/zero (dias sem report) ──
    if 'Positions' in raw.columns:
        raw = raw.dropna(subset=['Positions'])
        raw = raw[raw['Positions'] != 0]
        print(f"[COT] Após drop zeros/NaN: {len(raw)} rows")
    if raw.empty:
        print("[COT] Vazio após drop zeros")
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

    # ── Pivot: spec traders por Date × Direction ──
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

    # ── Compute Net = Long + Short ──
    if 'Positions - Net' in df.columns:
        df['Positions'] = df['Positions - Net']
    elif 'Positions - Long' in df.columns and 'Positions - Short' in df.columns:
        df['Positions'] = df['Positions - Long'] + df['Positions - Short']
    elif 'Positions - Long' in df.columns:
        df['Positions'] = df['Positions - Long']
    else:
        df['Positions'] = df.iloc[:, 0]

    df = df[df['Positions'] != 0]
    if df.empty:
        print("[COT] Vazio após drop net == 0")
        return pd.DataFrame()

    # ── Price & Open Interest (usa genérico para preço) ──
    price_ticker = futures_ticker if isinstance(futures_ticker, str) else futures_ticker[0]
    try:
        dates_bql = bq.func.range(_to_bql_date(start), _to_bql_date(end), frq='d')
        px_items = {'Price': bq.data.px_last(fill='prev'),
                    'Open Interest': bq.data.fut_aggte_open_int()}
        px_df = _fp_get_data(price_ticker, px_items,
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
            f'{col} WoW Change': s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else np.nan,
            f'{col} Percentile': (s.rank(pct=True).iloc[-1] * 100),
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
# SEÇÃO 5D — ESTIMATIVA DE BUYBACK + BLACKOUT WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

# Blackout: empresas não podem recomprar ações de ~28 dias antes do balanço
# até ~2 dias úteis após a divulgação. Quando muitas estão em blackout, o
# fluxo de buyback cai significativamente.
BLACKOUT_DAYS_BEFORE = 28  # ~4 semanas antes do earnings
BLACKOUT_DAYS_AFTER = 2    # ~2 dias após divulgação


def fetch_earnings_dates(index_ticker='SPX Index'):
    """Busca datas de earnings esperadas dos membros do índice via BQL.
    Retorna DataFrame com coluna 'earn_dt' (datetime)."""
    try:
        uni = bq.univ.members(index_ticker)
        # Tentar EXPECTED_REPORT_DT primeiro, depois EARN_ANN_DT_NEXT_ACTUAL
        for fld_name in ['expected_report_dt', 'earn_ann_dt_next_actual',
                         'next_announce_dt']:
            try:
                fld = getattr(bq.data, fld_name, None)
                if fld is None:
                    continue
                req = bql.Request(uni, {'earn_dt': fld()})
                resp = bq.execute(req)
                df_r = resp[0].df()
                if df_r is not None and not df_r.empty:
                    # Encontrar coluna de data
                    for c in df_r.columns:
                        if 'earn' in str(c).lower() or 'dt' in str(c).lower() or 'announce' in str(c).lower():
                            df_r['earn_dt'] = pd.to_datetime(df_r[c], errors='coerce')
                            break
                    else:
                        df_r['earn_dt'] = pd.to_datetime(df_r.iloc[:, 0], errors='coerce')
                    df_r = df_r.dropna(subset=['earn_dt'])
                    if len(df_r) > 10:
                        return df_r[['earn_dt']]
            except Exception:
                continue
    except Exception:
        pass
    return pd.DataFrame()


def compute_blackout_curve(earnings_df, n_days_forward=365):
    """Calcula curva de blackout: para os próximos N dias, quantas empresas do
    SPX estão em janela de restrição de buyback.

    BQL retorna apenas o PRÓXIMO earnings de cada empresa. Para projetar o ano
    inteiro, replicamos cada data trimestralmente (~91 dias) para frente.

    Returns:
        DataFrame com colunas: date, n_blackout, pct_blackout
    """
    if earnings_df.empty:
        return pd.DataFrame()

    today = pd.Timestamp.now().normalize()
    horizon = today + pd.Timedelta(days=n_days_forward)

    # Projetar earnings trimestrais a partir da data conhecida
    projected = []
    for dt in earnings_df['earn_dt']:
        if pd.isna(dt):
            continue
        d = pd.Timestamp(dt).normalize()
        # Gerar datas trimestrais para trás e para frente
        for q in range(-1, 5):  # -1 trimestre atrás até +4 à frente
            qd = d + pd.Timedelta(days=91 * q)
            if (today - pd.Timedelta(days=60)) <= qd <= horizon:
                projected.append(qd.to_datetime64())
    earn_dates_all = np.array(projected)
    total_companies = len(earnings_df)

    dates = pd.date_range(today - pd.Timedelta(days=30),
                          today + pd.Timedelta(days=n_days_forward))
    records = []
    for d in dates:
        # Uma empresa está em blackout se: earn_dt - 28 <= d <= earn_dt + 2
        in_blackout = np.sum(
            (earn_dates_all >= (d - pd.Timedelta(days=BLACKOUT_DAYS_AFTER)).to_datetime64()) &
            (earn_dates_all <= (d + pd.Timedelta(days=BLACKOUT_DAYS_BEFORE)).to_datetime64())
        )
        # Limitar ao total (pode ter duplicatas por projeção)
        in_blackout = min(int(in_blackout), total_companies)
        records.append({
            'date': d,
            'n_blackout': in_blackout,
            'pct_blackout': in_blackout / total_companies if total_companies > 0 else 0
        })
    return pd.DataFrame(records)


def blackout_pct_today(earnings_df):
    """Retorna % de empresas atualmente em blackout."""
    if earnings_df.empty:
        return 0.0, 0, 0
    today = pd.Timestamp.now().normalize()
    earn_dates = earnings_df['earn_dt'].values
    total = len(earnings_df)
    in_blackout = int(np.sum(
        (earn_dates >= (today - pd.Timedelta(days=BLACKOUT_DAYS_AFTER)).to_datetime64()) &
        (earn_dates <= (today + pd.Timedelta(days=BLACKOUT_DAYS_BEFORE)).to_datetime64())
    ))
    in_blackout = min(in_blackout, total)
    return in_blackout / total if total > 0 else 0, in_blackout, total

@lru_cache(maxsize=128)
def fetch_buyback_data(ticker):
    """Busca dados de buyback via BQL. Retorna dict com campos disponíveis."""
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
            df_r = bq.execute(req)[0].df()
            if df_r is not None and not df_r.empty:
                vcol = _fp_last_numeric_col(df_r)
                if vcol:
                    val = pd.to_numeric(df_r[vcol], errors='coerce').dropna()
                    if len(val) > 0:
                        result[key] = float(val.iloc[-1])
        except Exception:
            continue
    return result


def estimate_buyback_flow(ticker, horizon_days=252):
    """
    Estima fluxo diário de buyback.
    Usa _adv5_usd para % ADV (mesma lógica do rebalanceamento).
    """
    data = fetch_buyback_data(ticker)
    announced = data.get('announced', 0)
    if not announced or announced <= 0:
        return {'daily_est': 0, 'pct_adv_est': 0,
                'confidence': 'none', 'announced': 0}
    execution_rate = 0.80
    daily_est = (announced * execution_rate) / max(horizon_days, 1)
    try:
        adv5 = _adv5_usd([ticker])
        adv_usd = float(adv5.iloc[0]) if len(adv5) > 0 else 0
    except Exception:
        adv_usd = 0
    pct_adv = (announced / adv_usd * 100) if adv_usd > 0 else np.nan
    return {'daily_est': daily_est, 'pct_adv_est': pct_adv,
            'confidence': 'low', 'announced': announced,
            'mkt_cap': data.get('mkt_cap', np.nan)}


def estimate_index_buyback_flow(index_ticker='SPX Index', top_n=50):
    """Estima fluxo de buyback para os maiores membros de um índice.

    Tenta buscar announced_buyback_amt via BQL. Se não disponível,
    faz fallback para estimativa baseada em market cap (~2% cap/ano).
    Usa _adv5_usd (mesmo ADV do rebalanceamento) para calcular % ADV.
    """
    try:
        uni = bq.univ.members(index_ticker)
        items = {'cap': bq.data.cur_mkt_cap()}

        # Tentar múltiplos campos de buyback
        bb_field_name = None
        for fname in ['announced_buyback_amt', 'bs_sh_repurchase']:
            try:
                fld = getattr(bq.data, fname, None)
                if fld is not None:
                    items['buyback_raw'] = fld()
                    bb_field_name = fname
                    break
            except Exception:
                continue

        req = bql.Request(uni, items, with_params=BQL_PARAMS)
        df_r = bq.execute(req)[0].df()
        if df_r is None or df_r.empty:
            return pd.DataFrame()

        # Renomear colunas robustamente
        rename = {}
        for c in df_r.columns:
            cl = str(c).lower()
            if 'mkt' in cl and 'cap' in cl:
                rename[c] = 'cap'
            elif 'buyback' in cl or 'repurchase' in cl:
                rename[c] = 'buyback'
        df_r = df_r.rename(columns=rename)

        if 'cap' in df_r.columns:
            df_r['cap'] = pd.to_numeric(df_r['cap'], errors='coerce')
            df_r = df_r.dropna(subset=['cap'])
            df_r = df_r.nlargest(top_n, 'cap')

        # Verificar se dados de buyback reais estão disponíveis
        has_bb = ('buyback' in df_r.columns and
                  pd.to_numeric(df_r['buyback'], errors='coerce').abs().sum() > 0)

        if has_bb:
            df_r['buyback'] = pd.to_numeric(df_r['buyback'], errors='coerce').fillna(0).abs()
            df_r['confidence'] = 'low'
        else:
            df_r['buyback'] = df_r['cap'].fillna(0) * 0.02
            df_r['confidence'] = 'estimated'

        df_r['daily_est'] = df_r['buyback'] * 0.80 / TRADING_DAYS

        # % ADV usando _adv5_usd (mesma lógica do rebalanceamento)
        try:
            tickers_list = df_r.index.get_level_values('ID').unique().tolist()
            adv5 = _adv5_usd(tickers_list)
            df_r['pct_adv_est'] = (df_r['buyback']
                                   / adv5.reindex(df_r.index).replace(0, np.nan)) * 100
        except Exception:
            df_r['pct_adv_est'] = np.nan

        out_cols = [c for c in ['cap', 'buyback', 'daily_est', 'pct_adv_est', 'confidence']
                    if c in df_r.columns]
        return df_r[out_cols]
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5E — FLOW SCORING & AGREGAÇÃO PREDITIVA
# ═══════════════════════════════════════════════════════════════════════════════

def flow_zscore(current, history):
    """Z-score do valor atual relativo ao histórico."""
    if history is None or len(history) < 5:
        return np.nan
    mu = history.mean()
    sigma = history.std()
    if sigma < 1e-12:
        return 0.0
    return (current - mu) / sigma


def compute_leveraged_flow_simple(daily_return, aum_estimates=None):
    """Calcula fluxo total de rebalanceamento sem BQL call."""
    if aum_estimates is None:
        aum_estimates = DEFAULT_AUM
    total = 0
    for etf in LEVERAGED_ETFS_EXT:
        L = etf['leverage']
        r = daily_return
        aum = aum_estimates.get(etf['name'], 1e9)
        denom = 1 + L * r
        if abs(denom) > 1e-12:
            total += aum * L * (L - 1) * r / denom
    return total


def build_flow_history(ticker='SPX Index', lookback=252):
    """Constrói série histórica de fluxo de rebalanceamento de ETFs alavancados."""
    item = bq.data.px_last(dates=bq.func.range(f'-{lookback + 10}D', '0D'), fill='PREV')
    try:
        df_r = bq.execute(bql.Request([ticker], item))[0].df()
        if df_r is None or df_r.empty:
            return pd.DataFrame()
        dcol = next((c for c in df_r.columns if 'date' in str(c).lower()), None)
        vcol = _fp_last_numeric_col(df_r)
        if dcol is None or vcol is None:
            return pd.DataFrame()
        px = df_r.set_index(dcol)[vcol].astype(float).sort_index()
        px.index = pd.to_datetime(px.index)
    except Exception:
        return pd.DataFrame()
    rets = px.pct_change().dropna()
    flows = rets.apply(compute_leveraged_flow_simple)
    flows.name = 'LevETF_Flow'
    return pd.DataFrame({'Return': rets, 'LevETF_Flow': flows})


def compute_flow_score(leveraged_flow, buyback_daily=0, cot_net_change=0,
                       passive_etf_flow=0, history_leveraged=None,
                       history_cot=None, dealer_flow=0, volctrl_flow=0,
                       cta_flow=0, rp_flow=0,
                       history_dealer=None, history_volctrl=None):
    """
    Computa score combinado de fluxo contratado — 8 componentes.
    Pesos: CTA 22%, Dealer 18%, VolCtrl 12%, Risk Parity 12%,
           ETFs Alav 12%, Buyback 8%, COT 10%, ETFs Passivos 6%.
    """
    z_lev = flow_zscore(leveraged_flow, history_leveraged) if history_leveraged is not None else 0
    z_cot = flow_zscore(cot_net_change, history_cot) if history_cot is not None else 0
    z_buyback = np.clip(buyback_daily / 1e8, -3, 3) if buyback_daily else 0
    z_passive = np.clip(passive_etf_flow / 1e9, -3, 3) if passive_etf_flow else 0
    z_dealer = np.clip(dealer_flow / 1e9, -3, 3) if dealer_flow else 0
    z_volctrl = np.clip(volctrl_flow / 1e9, -3, 3) if volctrl_flow else 0
    z_cta = np.clip(cta_flow / 1e9, -3, 3) if cta_flow else 0
    z_rp = np.clip(rp_flow / 1e9, -3, 3) if rp_flow else 0

    w_cta, w_deal, w_vc, w_rp = 0.22, 0.18, 0.12, 0.12
    w_lev, w_buy, w_cot, w_passive = 0.12, 0.08, 0.10, 0.06
    if history_cot is None or len(history_cot) < 5:
        w_cot = 0.0
        w_cta, w_deal, w_rp = 0.26, 0.20, 0.14
        w_vc, w_lev, w_buy, w_passive = 0.14, 0.14, 0.08, 0.04

    combined = (w_lev * z_lev + w_buy * z_buyback + w_cot * z_cot
                + w_passive * z_passive + w_deal * z_dealer + w_vc * z_volctrl
                + w_cta * z_cta + w_rp * z_rp)

    if combined > 0.5:
        direction = "BULLISH"
    elif combined < -0.5:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    prob_up = 1.0 / (1.0 + math.exp(-combined))
    score_0_100 = round(prob_up * 100, 1)
    return {
        'z_leveraged': z_lev, 'z_buyback': z_buyback,
        'z_cot': z_cot, 'z_passive_etf': z_passive,
        'z_dealer': z_dealer, 'z_volctrl': z_volctrl,
        'z_cta': z_cta, 'z_rp': z_rp,
        'combined_score': combined, 'direction': direction,
        'prob_up': prob_up, 'prob_down': 1.0 - prob_up,
        'score': score_0_100, 'score_total': score_0_100,
        'weights': {'leveraged': w_lev, 'buyback': w_buy,
                    'cot': w_cot, 'passive_etf': w_passive,
                    'dealer': w_deal, 'volctrl': w_vc,
                    'cta': w_cta, 'rp': w_rp},
    }


def compute_dealer_hedging_flow(gex_per_pt, daily_price_change, spot):
    """
    Estima fluxo de hedging de dealers/market makers em opções.
    Quando dealers estão short gamma, um move de +1% os força a comprar
    (delta hedge), amplificando o movimento. Short gamma → pro-cíclico.
    Flow ≈ -GEX_per_pt × ΔS (dealers hedgeiam no sentido oposto à gamma).
    """
    if pd.isna(gex_per_pt) or pd.isna(daily_price_change) or pd.isna(spot):
        return 0
    flow = -gex_per_pt * daily_price_change
    return flow


def fetch_options_volume_bql(ticker='SPX Index'):
    """
    Busca volume total de opções e P/C ratio via BQL.
    P/C ratio: usa PCUSEQTR Index (código BBG oficial do Put/Call ratio).
    Retorna dict com total_adc, put_vol, call_vol, pc_ratio.
    """
    bq = bql.Service()
    # P/C ratio — PCUSEQTR Index é o código oficial no terminal Bloomberg
    pcr = 0.0
    try:
        pc_req = bql.Request('PCUSEQTR Index', {'v': bq.data.px_last(fill='PREV')})
        pc_resp = bq.execute(pc_req)
        pcr = float(pc_resp[0].df()['v'].iloc[-1] or 0)
        print(f"[PC] PCUSEQTR Index = {pcr:.2f}")
    except Exception as _pce:
        print(f"⚠️ PCUSEQTR fetch: {_pce}")

    # Volume de calls/puts — tenta campos BQL alternativos
    cv, pv = 0.0, 0.0
    try:
        req = bql.Request(ticker, {
            'call_vol': bq.data.opt_call_volume(),
            'put_vol':  bq.data.opt_put_volume(),
        })
        resp = bq.execute(req)
        _df = pd.concat([r.df()[r.name] for r in resp], axis=1)
        _df = _df.loc[:, ~_df.columns.duplicated()]
        cv = float(_df['call_vol'].iloc[0] or 0)
        pv = float(_df['put_vol'].iloc[0]  or 0)
    except Exception as _ve:
        print(f"⚠️ Options volume fetch (opt_call_volume): {_ve}")
        # fallback: opt_volume separado por type
        try:
            _cond_c = bq.data.put_call() == 'Call'
            _cond_p = bq.data.put_call() == 'Put'
            _univ_c = bq.univ.filter(bq.univ.options([ticker]), _cond_c)
            _univ_p = bq.univ.filter(bq.univ.options([ticker]), _cond_p)
            _rc = bq.execute(bql.Request(_univ_c, {'v': bq.data.opt_volume()}))
            _rp = bq.execute(bql.Request(_univ_p, {'v': bq.data.opt_volume()}))
            cv = float(_rc[0].df()['v'].sum() or 0)
            pv = float(_rp[0].df()['v'].sum() or 0)
        except Exception as _ve2:
            print(f"⚠️ Options volume fetch (fallback): {_ve2}")

    total = cv + pv
    return {
        'total_adc': total if total > 0 else OPTIONS_TOTAL_ADC,
        'call_vol': cv,
        'put_vol': pv,
        'pc_ratio': pcr,
        'source': 'BQL' if pcr > 0 else 'fallback',
    }


def estimate_mm_var_by_book(gex_per_pt, spot, risk_params, oi_total):
    """
    Estima VaR 95%/99% por market maker, proporcional ao volume share.
    Usa distribuição t-Student calibrada dos log returns.
    Para cada MM: assume que detém fração proporcional do GEX e OI.
    VaR_mm = share × VaR_total (linear porque gamma exposure escala com OI).
    Retorna lista de dicts com VaR por MM.
    """
    var_95 = risk_params.get('var_95', 0)
    var_99 = risk_params.get('var_99', 0)
    cvar_95 = risk_params.get('cvar_95', 0)

    # Total portfolio 1-day VaR in $ terms
    ds_95 = spot * abs(var_95)
    ds_99 = spot * abs(var_99)
    ds_cvar = spot * abs(cvar_95)

    # Total P&L at VaR level: delta-neutral MM → mostly gamma exposure
    # PnL ≈ 0.5 × GEX × ΔS²  (dealers are short the book → negative)
    pnl_var95 = abs(0.5 * gex_per_pt * ds_95 ** 2)
    pnl_var99 = abs(0.5 * gex_per_pt * ds_99 ** 2)
    pnl_cvar95 = abs(0.5 * gex_per_pt * ds_cvar ** 2)

    results = []
    for mm_name, share in MM_VOLUME_SHARES.items():
        mm_gex = gex_per_pt * share
        mm_oi = oi_total * share
        results.append({
            'name': mm_name,
            'share': share,
            'gex_per_pt': mm_gex,
            'oi_contracts': mm_oi,
            'var_95': pnl_var95 * share,
            'var_99': pnl_var99 * share,
            'cvar_95': pnl_cvar95 * share,
            'daily_theta': 0,  # Will be filled later if available
        })
    return results, {'pnl_var95': pnl_var95, 'pnl_var99': pnl_var99,
                     'pnl_cvar95': pnl_cvar95}


# ── Volatility Control Fund Flows ─────────────────────────────────
# Vol-targeting strategies adjust equity exposure inversely to realized vol.
# When RV rises, they sell; when it falls, they buy.
# Typical global AUM: ~$350B-$500B across pension and systematic funds.
VOL_CTRL_AUM = {5: 100e9, 10: 150e9, 15: 100e9}  # target_vol% → AUM estimate
VOL_CTRL_MAX_LEV = 2.0  # Maximum leverage cap
VOL_CTRL_MIN_EXP = 0.20  # Piso mínimo de exposição (fundos não vão a 0%)
VOL_CTRL_DAILY_ADJ = 0.25  # Ajuste máximo por dia (~25% do delta)


def _vc_exposure(target_dec, rv):
    """Calcula exposure com piso e teto."""
    if rv < 1e-6:
        return VOL_CTRL_MAX_LEV
    return max(min(target_dec / rv, VOL_CTRL_MAX_LEV), VOL_CTRL_MIN_EXP)


def compute_vol_control_flow(rv_current, rv_prev, target_vols=None):
    """
    Estima fluxo dos fundos de controle de volatilidade.
    rv_current e rv_prev devem ser vol anualizada (ex: 0.15 = 15%).
    Quando vol sobe → exposure cai → fundos vendem (flow negativo).
    Inclui piso de exposição (20%) e ajuste gradual (25%/dia).
    """
    if target_vols is None:
        target_vols = [5, 10, 15]
    if pd.isna(rv_current) or pd.isna(rv_prev) or rv_current < 1e-6 or rv_prev < 1e-6:
        return {'total': 0, 'detail': {}}

    detail = {}
    total = 0
    for tv in target_vols:
        tv_dec = tv / 100.0
        exp_new = _vc_exposure(tv_dec, rv_current)
        exp_old = _vc_exposure(tv_dec, rv_prev)
        aum = VOL_CTRL_AUM.get(tv, 100e9)
        # Fluxo total necessário vs fluxo diário (ajuste gradual)
        full_flow = aum * (exp_new - exp_old)
        daily_flow = full_flow * VOL_CTRL_DAILY_ADJ
        detail[f'{tv}%'] = {'exposure_new': exp_new, 'exposure_old': exp_old,
                            'flow': full_flow, 'daily_flow': daily_flow, 'aum': aum}
        total += full_flow
    daily_total = total * VOL_CTRL_DAILY_ADJ
    return {'total': total, 'daily_total': daily_total, 'detail': detail}


def compute_vol_control_scenarios(rv_current, target_vols=None):
    """
    Cenários de stress: quanto vendem se vol sobe.
    Retorna lista com total flow e daily flow (ajuste gradual).
    """
    if target_vols is None:
        target_vols = [5, 10, 15]
    if pd.isna(rv_current) or rv_current < 1e-6:
        return []
    shock_vols = [0.15, 0.20, 0.25, 0.30, 0.40]
    scenarios = []
    for sv in shock_vols:
        if sv <= rv_current * 1.05:
            continue
        total = 0
        for tv in target_vols:
            tv_dec = tv / 100.0
            exp_cur = _vc_exposure(tv_dec, rv_current)
            exp_shock = _vc_exposure(tv_dec, sv)
            aum = VOL_CTRL_AUM.get(tv, 100e9)
            total += aum * (exp_shock - exp_cur)
        scenarios.append({'rv_shock': sv, 'flow': total,
                          'daily_flow': total * VOL_CTRL_DAILY_ADJ})
    return scenarios


def compute_combined_flow_scenarios(rv_current, prices=None, gex_per_pt=0,
                                    spot=0, vanna_notional=0, vega_notional=0,
                                    charm_notional=0):
    """
    Cenários combinados: para cada nível de vol shock, estima fluxo de
    Vol Control + Risk Parity + CTA + Dealer + Vanna + Charm.
    vanna_notional = sum(vanna * OI * 100) * spot  ($ de delta por 1% vol)
    vega_notional = sum(vega * OI * 100)           ($ por 1% vol)
    charm_notional = sum(charm * OI * 100) * spot / 365  ($ de delta decay diário)
    """
    if pd.isna(rv_current) or rv_current < 1e-6:
        return []
    scenarios = [
        ('Leve (-3%, vol 18%)', -0.03, 0.18),
        ('Moderado (-5%, vol 22%)', -0.05, 0.22),
        ('Forte (-8%, vol 28%)', -0.08, 0.28),
        ('Crash (-12%, vol 35%)', -0.12, 0.35),
        ('Pânico (-20%, vol 50%)', -0.20, 0.50),
    ]
    results = []
    for name, spx_move, rv_shock in scenarios:
        if rv_shock <= rv_current * 1.05:
            continue
        # Vol Control (com piso)
        vc = 0
        for tv in [5, 10, 15]:
            tv_dec = tv / 100.0
            exp_cur = _vc_exposure(tv_dec, rv_current)
            exp_shock = _vc_exposure(tv_dec, rv_shock)
            vc += VOL_CTRL_AUM.get(tv, 100e9) * (exp_shock - exp_cur)

        # Risk Parity
        rp_result = compute_risk_parity_flow(rv_shock, rv_current)
        rp = rp_result['total']

        # CTA (trend reversal)
        cta = 0
        if prices is not None and len(prices) > 200:
            trend_now = compute_cta_trend_strength(prices)
            shocked = prices.copy()
            for i in range(1, 6):
                shocked.iloc[-i] = shocked.iloc[-i] * (1 + spx_move / 5)
            trend_shock = compute_cta_trend_strength(shocked)
            cta_rv = max(rv_shock, rv_current)
            pos_now = np.clip(trend_now * (0.10 / rv_current), -2, 2)
            pos_shock = np.clip(trend_shock * (0.10 / cta_rv), -2, 2)
            cta = CTA_AUM * CTA_EQUITY_ALLOC * (pos_shock - pos_now)

        # Dealer (short gamma amplifica sell-off)
        dealer = 0
        if gex_per_pt != 0 and spot > 0:
            daily_chg = spot * spx_move
            dealer = -gex_per_pt * daily_chg

        # Vanna flow: quando vol sobe + spot cai, dealers com vanna positivo
        # precisam vender delta. Flow ≈ -vanna_notional × Δvol_pts
        # vanna_notional já é em $ de delta por 1% vol
        vol_chg_pts = (rv_shock - rv_current) * 100  # em pontos de vol
        vanna = -vanna_notional * vol_chg_pts if vanna_notional != 0 else 0

        # Charm flow: decay diário do delta — dealers precisam rebalancear overnight.
        # charm_notional já é em $ de delta por dia (positivo = dealers precisam vender)
        charm = -charm_notional if charm_notional != 0 else 0

        total = vc + rp + cta + dealer + vanna + charm
        results.append({
            'name': name, 'spx_move': spx_move, 'rv_shock': rv_shock,
            'vol_ctrl': vc, 'risk_parity': rp, 'cta': cta,
            'dealer': dealer, 'vanna': vanna, 'charm': charm, 'total': total,
        })
    return results


# ── Risk Parity Model ──────────────────────────────────────────────
# Baseado em BofA Systematic Flows Monitor:
# - 3 asset classes: Equities (SPX), Bonds (10Y UST), Commodities (GSCI)
# - Alocação inversamente proporcional à volatilidade (↓vol → ↑alocação)
# - Rebalanceamento mensal usando 3M de dados para vol e correlação
# - AUM estimado: $200B–$750B (BofA usa $200B nos exhibits)
# - Risk targets: 10%, 12%, 15% vol; max leverage 1.5x/1.5x/3.0x
RISK_PARITY_AUM = 200e9  # $200B total (BofA Exhibit 3)
RISK_PARITY_TARGETS = {
    10: {'aum_share': 0.33, 'max_lev': 1.5},
    12: {'aum_share': 0.34, 'max_lev': 1.5},
    15: {'aum_share': 0.33, 'max_lev': 3.0},
}


def compute_risk_parity_flow(rv_equity, rv_equity_prev,
                             rv_bonds=None, rv_bonds_prev=None,
                             rv_commod=None, rv_commod_prev=None):
    """
    Estima fluxo de risk parity para equities.
    Alocação equity ∝ 1/vol_equity. Quando vol equity sobe,
    alocação cai → vendem equities.
    Se vol de bonds/commodities disponível, usa pesos relativos;
    caso contrário, usa só equity vol como proxy.
    """
    if pd.isna(rv_equity) or pd.isna(rv_equity_prev) or rv_equity < 1e-6:
        return {'total': 0, 'detail': {}, 'eq_alloc_new': 0, 'eq_alloc_old': 0}

    def _eq_weight(rv_eq, rv_bd, rv_cm):
        """Peso equity = (1/vol_eq) / sum(1/vol_i)."""
        inv_eq = 1.0 / max(rv_eq, 0.01)
        inv_bd = 1.0 / max(rv_bd, 0.01) if rv_bd and rv_bd > 1e-6 else inv_eq * 2.5
        inv_cm = 1.0 / max(rv_cm, 0.01) if rv_cm and rv_cm > 1e-6 else inv_eq * 0.8
        total_inv = inv_eq + inv_bd + inv_cm
        return inv_eq / total_inv if total_inv > 0 else 0.33

    eq_w_new = _eq_weight(rv_equity, rv_bonds, rv_commod)
    eq_w_old = _eq_weight(rv_equity_prev, rv_bonds_prev, rv_commod_prev)

    detail = {}
    total = 0
    _bv_new = rv_bonds if rv_bonds and rv_bonds > 1e-6 else 0.05
    _bv_old = rv_bonds_prev if rv_bonds_prev and rv_bonds_prev > 1e-6 else 0.05
    for tv, params in RISK_PARITY_TARGETS.items():
        aum = RISK_PARITY_AUM * params['aum_share']
        tv_dec = tv / 100.0
        # Portfolio vol ≈ weighted avg of component vols (simplificação)
        port_vol_new = rv_equity * eq_w_new + _bv_new * (1 - eq_w_new)
        port_vol_old = rv_equity_prev * eq_w_old + _bv_old * (1 - eq_w_old)
        lev_new = min(tv_dec / max(port_vol_new, 0.01), params['max_lev'])
        lev_old = min(tv_dec / max(port_vol_old, 0.01), params['max_lev'])
        eq_exp_new = lev_new * eq_w_new
        eq_exp_old = lev_old * eq_w_old
        flow = aum * (eq_exp_new - eq_exp_old)
        detail[f'{tv}%'] = {
            'aum': aum, 'leverage_new': lev_new, 'leverage_old': lev_old,
            'eq_alloc_new': eq_w_new, 'eq_alloc_old': eq_w_old,
            'eq_exposure_new': eq_exp_new, 'eq_exposure_old': eq_exp_old,
            'flow': flow}
        total += flow

    return {'total': total, 'detail': detail,
            'eq_alloc_new': eq_w_new, 'eq_alloc_old': eq_w_old}


# ── CTA Trend Following Model ─────────────────────────────────────
# Baseado em BofA Systematic Flows Monitor:
# - Trend strength via coleção de cruzamentos de médias móveis (near vs far)
# - Position sizing: trend_strength / volatility (trend/vol ratio)
# - AUM ~$340B (BarclayHedge 2024-Q4), ~25% alocação em equities
# - CTAs ajustam diariamente, são os mais rápidos a responder
CTA_AUM = 340e9
CTA_EQUITY_ALLOC = 0.25  # ~25% da carteira em equities
CTA_MA_PAIRS = [(5, 20), (5, 60), (10, 60), (20, 120), (20, 200)]
# Pesos por janela: sinais longos movem mais notional (estilo GS/BofA)
CTA_MA_WEIGHTS = {(5, 20): 0.10, (5, 60): 0.15, (10, 60): 0.20,
                  (20, 120): 0.25, (20, 200): 0.30}


def compute_cta_trend_strength(prices, ma_pairs=None, use_weights=None):
    """
    Calcula trend strength usando cruzamentos de médias móveis.
    Para cada par (curta, longa): spread contínuo em [-1, +1].
    use_weights=True → pesos por janela (sinais longos têm mais peso, estilo GS/BofA).
    use_weights=False → média simples (igual peso por janela).
    use_weights=None → lê o toggle cta_weight_w se disponível.
    Ref: BofA "Trends aren't going out of fashion" (2017).
    """
    if ma_pairs is None:
        ma_pairs = CTA_MA_PAIRS
    if use_weights is None:
        try:
            use_weights = cta_weight_w.value
        except NameError:
            use_weights = False
    if len(prices) < max(p[1] for p in ma_pairs):
        return 0
    scores, weights = [], []
    for short_w, long_w in ma_pairs:
        ma_short = prices.rolling(short_w).mean().iloc[-1]
        ma_long = prices.rolling(long_w).mean().iloc[-1]
        if pd.isna(ma_short) or pd.isna(ma_long) or ma_long == 0:
            continue
        spread = (ma_short - ma_long) / ma_long
        scores.append(np.clip(spread * 100, -1, 1))
        weights.append(CTA_MA_WEIGHTS.get((short_w, long_w), 1.0))
    if not scores:
        return 0
    if use_weights and len(weights) > 0:
        w = np.array(weights)
        return float(np.average(scores, weights=w / w.sum()))
    return float(np.mean(scores))


def compute_cta_flow(prices, rv_current, target_vol=0.10):
    """
    Estima fluxo de CTA trend followers para equities.
    Position = trend_strength × (target_vol / realized_vol), capped em [-2, +2].
    Flow = AUM × equity_alloc × Δposition.
    """
    if len(prices) < 201 or rv_current < 1e-6:
        return {'flow': 0, 'trend_today': 0, 'trend_prev': 0,
                'pos_today': 0, 'pos_prev': 0}

    trend_today = compute_cta_trend_strength(prices)
    trend_prev = compute_cta_trend_strength(prices.iloc[:-1])

    pos_today = np.clip(trend_today * (target_vol / rv_current), -2, 2)

    # RV do dia anterior (rolling 63d para capturar ~3M de vol)
    rets_prev = prices.iloc[:-1].pct_change().dropna()
    rv_prev = rets_prev.iloc[-63:].std() * np.sqrt(252) if len(rets_prev) >= 63 else rv_current
    pos_prev = np.clip(trend_prev * (target_vol / max(rv_prev, 1e-6)), -2, 2)

    flow = CTA_AUM * CTA_EQUITY_ALLOC * (pos_today - pos_prev)

    return {
        'flow': flow,
        'trend_today': trend_today,
        'trend_prev': trend_prev,
        'pos_today': pos_today,
        'pos_prev': pos_prev,
    }

def _trend_from_array(vals, use_weights=None):
    """Fast trend strength from raw numpy array (no pandas overhead)."""
    if use_weights is None:
        try:
            use_weights = cta_weight_w.value
        except NameError:
            use_weights = False
    n = len(vals)
    scores, weights = [], []
    for short_w, long_w in CTA_MA_PAIRS:
        if n < long_w:
            continue
        ms = vals[-short_w:].mean()
        ml = vals[-long_w:].mean()
        if ml == 0:
            continue
        spread = (ms - ml) / ml
        scores.append(np.clip(spread * 100, -1, 1))
        weights.append(CTA_MA_WEIGHTS.get((short_w, long_w), 1.0))
    if not scores:
        return 0.0
    if use_weights:
        w = np.array(weights)
        return float(np.average(scores, weights=w / w.sum()))
    return float(np.mean(scores))


def compute_cta_scenario_flows(prices, rv_current, spot, horizon_days=5,
                                annualized_vol=None):
    """
    Calcula fluxos de CTA em diferentes cenários (Flat / Up / Down) ao estilo GS.
    Versão rápida: usa numpy arrays, sem pd.concat nem pd.Timedelta.
    """
    if len(prices) < 201 or rv_current < 1e-6:
        return []
    if annualized_vol is None:
        annualized_vol = rv_current

    daily_vol = annualized_vol / np.sqrt(252)
    move_1sigma = daily_vol * np.sqrt(horizon_days)

    scenarios = [
        ('Flat', 0.0),
        ('Up 1\u03c3', move_1sigma),
        ('Up 2\u03c3', 2 * move_1sigma),
        ('Down 1\u03c3', -move_1sigma),
        ('Down 2\u03c3', -2 * move_1sigma),
        ('Down 2.5\u03c3', -2.5 * move_1sigma),
    ]

    vals = np.asarray(prices.values, dtype=float)
    trend_now = _trend_from_array(vals)
    pos_now = np.clip(trend_now * (0.10 / rv_current), -2, 2)
    current_notional = CTA_AUM * CTA_EQUITY_ALLOC * pos_now

    results = []
    for name, pct_move in scenarios:
        daily_step = pct_move / max(horizon_days, 1)
        sim = np.empty(len(vals) + horizon_days, dtype=float)
        sim[:len(vals)] = vals
        for d in range(horizon_days):
            sim[len(vals) + d] = sim[len(vals) + d - 1] * (1 + daily_step)

        trend_end = _trend_from_array(sim)
        rv_end = rv_current * (1 + max(0, -pct_move) * 3)
        pos_end = np.clip(trend_end * (0.10 / max(rv_end, 1e-6)), -2, 2)
        end_notional = CTA_AUM * CTA_EQUITY_ALLOC * pos_end
        flow = end_notional - current_notional

        results.append({
            'name': name,
            'spx_end': spot * (1 + pct_move),
            'pct_move': pct_move,
            'flow_total': flow,
            'trend_end': trend_end,
            'pos_end': pos_end,
            'pos_now': pos_now,
        })
    return results


def compute_cta_pivot_levels(prices, spot, rv_current):
    """
    Calcula níveis de preço onde sinais de trend MA flip (pivot levels).
    Usa numpy arrays para robustez (sem dependência de index type).
    """
    vals = np.asarray(prices.values, dtype=float)
    if len(vals) < 201:
        return []

    pivots = []
    labels = {
        (5, 20): 'Curto prazo',
        (5, 60): 'Curto-médio',
        (10, 60): 'Médio prazo',
        (20, 120): 'Médio-longo',
        (20, 200): 'Longo prazo',
    }

    for short_w, long_w in CTA_MA_PAIRS:
        ma_short_now = vals[-short_w:].mean()
        ma_long_now = vals[-long_w:].mean()

        if np.isnan(ma_short_now) or np.isnan(ma_long_now):
            continue

        above = ma_short_now > ma_long_now

        # Pivot: price X such that new MA_short = MA_long
        sum_recent = vals[-short_w + 1:].sum() if short_w > 1 else 0.0
        pivot_px = ma_long_now * short_w - sum_recent

        if pivot_px <= 0 or pivot_px > spot * 2:
            continue

        dist_pct = (pivot_px - spot) / spot
        signal_type = 'SELL trigger' if above else 'BUY trigger'
        label = labels.get((short_w, long_w), f'MA{short_w}/{long_w}')

        pivots.append({
            'label': label,
            'ma_pair': f'{short_w}/{long_w}',
            'level': pivot_px,
            'type': signal_type,
            'distance_pct': dist_pct,
            'above_now': above,
        })

    # Sort by proximity to spot
    pivots.sort(key=lambda x: abs(x['distance_pct']))
    return pivots


def compute_cta_historical_positions(prices, rv_series=None, lookback=252):
    """
    Calcula série histórica de trend strength, position sizing e notional.
    Versão vetorizada com numpy → rápido.
    """
    vals = np.asarray(prices.values, dtype=float)
    if len(vals) < 201:
        return pd.DataFrame()

    # Pre-compute all rolling MAs as numpy arrays
    ma_arrays = {}
    for short_w, long_w in CTA_MA_PAIRS:
        ms = pd.Series(vals).rolling(short_w).mean().values
        ml = pd.Series(vals).rolling(long_w).mean().values
        ma_arrays[(short_w, long_w)] = (ms, ml)

    # Rolling realized vol (63d)
    rets = np.diff(vals) / vals[:-1]
    rets = np.concatenate([[np.nan], rets])
    rv_arr = pd.Series(rets).rolling(63).std().values * np.sqrt(252)

    start = max(201, len(vals) - lookback)
    idx_range = range(start, len(vals))

    # Try to get dates from prices index
    try:
        dates = prices.index
    except Exception:
        dates = list(range(len(vals)))

    records = []
    for i in idx_range:
        scores = []
        for short_w, long_w in CTA_MA_PAIRS:
            ms_v, ml_v = ma_arrays[(short_w, long_w)]
            msv = ms_v[i]
            mlv = ml_v[i]
            if np.isnan(msv) or np.isnan(mlv) or mlv == 0:
                continue
            spread = (msv - mlv) / mlv
            scores.append(np.clip(spread * 100, -1, 1))
        trend = float(np.mean(scores)) if scores else 0.0

        rv = rv_arr[i] if not np.isnan(rv_arr[i]) else 0.15
        pos = np.clip(trend * (0.10 / max(rv, 1e-6)), -2, 2)
        notional = CTA_AUM * CTA_EQUITY_ALLOC * pos
        records.append({
            'date': dates[i],
            'trend': trend,
            'position': pos,
            'notional': notional,
        })
    return pd.DataFrame(records)


def build_cta_gs_chart(fp_cta_hist, fp_cta_scenarios_1w, fp_cta_scenarios_1m,
                       spot):
    """
    Constroi chart: histórico de posição CTA com fan de
    cenários projetados para 1W e 1M à frente.
    Retorna go.Figure (converter para FigureWidget externamente).
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=[
            'CTA Estimates — S&P 500 (Notional $B)',
            'CTA Scenario Flows ($B) — 1 Week vs 1 Month',
        ])

    # ── Top panel: Historical notional + scenario fan ──
    if not fp_cta_hist.empty and len(fp_cta_hist) > 5:
        hist_dates = fp_cta_hist['date']
        hist_notional = fp_cta_hist['notional'] / 1e9

        # Historical line
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_notional,
            name='CTA Notional (Hist)',
            mode='lines',
            line=dict(color='#4A90D9', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(74,144,217,0.10)'),
            row=1, col=1)

        # Scenario fan from last date
        last_date = pd.Timestamp(hist_dates.iloc[-1])
        last_notional = hist_notional.iloc[-1]

        # Build forward points for 1W and 1M scenarios
        scenario_colors = {
            'Flat': '#AAAAAA',
            'Up 1\u03c3': '#00C853',
            'Up 2\u03c3': '#00E676',
            'Down 1\u03c3': '#FF5252',
            'Down 2\u03c3': '#FF1744',
            'Down 2.5\u03c3': '#D50000',
        }

        for s1w, s1m in zip(fp_cta_scenarios_1w, fp_cta_scenarios_1m):
            name = s1w['name']
            end_notional_1w = CTA_AUM * CTA_EQUITY_ALLOC * s1w['pos_end'] / 1e9
            end_notional_1m = CTA_AUM * CTA_EQUITY_ALLOC * s1m['pos_end'] / 1e9
            color = scenario_colors.get(name, '#888888')

            try:
                d1w = last_date + pd.Timedelta(days=7)
                d1m = last_date + pd.Timedelta(days=30)
            except Exception:
                d1w = last_date + pd.Timedelta(days=5)
                d1m = last_date + pd.Timedelta(days=21)

            fig.add_trace(go.Scatter(
                x=[last_date, d1w, d1m],
                y=[last_notional, end_notional_1w, end_notional_1m],
                name=name,
                mode='lines+markers',
                line=dict(color=color, width=2, dash='dot'),
                marker=dict(size=6, color=color),
                legendgroup=name),
                row=1, col=1)

        fig.add_hline(y=0, line_dash='dash', line_color='rgba(150,150,150,0.5)',
                      row=1, col=1)

    # ── Bottom panel: Scenario flow bar chart (grouped 1W vs 1M) ──
    if fp_cta_scenarios_1w and fp_cta_scenarios_1m:
        names = [s['name'] for s in fp_cta_scenarios_1w]
        flows_1w = [s['flow_total'] / 1e9 for s in fp_cta_scenarios_1w]
        flows_1m = [s['flow_total'] / 1e9 for s in fp_cta_scenarios_1m]

        bar_colors_1w = ['#00C853' if f > 0 else '#FF5252' for f in flows_1w]
        bar_colors_1m = ['#00E676' if f > 0 else '#FF1744' for f in flows_1m]

        fig.add_trace(go.Bar(
            x=names, y=flows_1w, name='1 Week Flow',
            marker_color=bar_colors_1w,
            text=[f'${f:+.1f}B' for f in flows_1w],
            textposition='outside',
            textfont=dict(size=10)),
            row=2, col=1)

        fig.add_trace(go.Bar(
            x=names, y=flows_1m, name='1 Month Flow',
            marker_color=bar_colors_1m,
            text=[f'${f:+.1f}B' for f in flows_1m],
            textposition='outside',
            textfont=dict(size=10),
            opacity=0.7),
            row=2, col=1)

        fig.add_hline(y=0, line_dash='dash', line_color='rgba(150,150,150,0.5)',
                      row=2, col=1)

    fig.update_layout(
        paper_bgcolor=_C['card'], plot_bgcolor=_C['card'],
        font=dict(color=_C['text'], size=11),
        legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center',
                    font=dict(color=_C['text_muted'], size=10)),
        height=620, margin=dict(l=55, r=30, t=40, b=40),
        barmode='group')
    for ax in ['yaxis', 'yaxis2', 'xaxis', 'xaxis2']:
        fig.update_layout(**{ax: dict(
            gridcolor=_C['border'], zerolinecolor=_C['border'],
            tickfont=dict(color=_C['text_muted']))})
    # Make subplot titles use theme color
    for ann in fig.layout.annotations:
        ann.font.color = _C['text']
        ann.font.size = 13

    return fig
