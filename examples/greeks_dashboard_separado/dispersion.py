"""Dispersion analysis: correlation, implied/realized, pairs, straddles, ML model."""

import numpy as np
import pandas as pd
import traceback
import warnings
import os
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import minimize as sp_minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import bql

try:
    from .config import (bq, _C, DASH_TEMPLATE, FLOW_FIG_LAYOUT, HAS_SKLEARN,
                         HAS_DATAGRID, HAS_BQPLOT, wd, INDEX_PROXY, TRADING_DAYS,
                         DISP_COR1M, DISP_DSPX, DISP_VIXEQ, DISP_TOP_N, DISP_EXCLUDE,
                         MAG7, MAG8, GAMMA_HISTORY_PATH, DISP_CORR_WINDOWS, DISP_SPXSK3)
    from .data import _bql_ts
    from .ui import _hud_panel
except ImportError:
    from config import (bq, _C, DASH_TEMPLATE, FLOW_FIG_LAYOUT, HAS_SKLEARN,
                        HAS_DATAGRID, HAS_BQPLOT, wd, INDEX_PROXY, TRADING_DAYS,
                        DISP_COR1M, DISP_DSPX, DISP_VIXEQ, DISP_TOP_N, DISP_EXCLUDE,
                        MAG7, MAG8, GAMMA_HISTORY_PATH, DISP_CORR_WINDOWS, DISP_SPXSK3)
    from data import _bql_ts
    from ui import _hud_panel

try:
    from ipydatagrid import DataGrid, TextRenderer
except ImportError:
    pass

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline
except ImportError:
    pass


def _bql_fetch_member_data(index_ticker='SPX Index', lookback_days=252):
    """
    Busca preços e IV dos membros de um índice via BQL.
    Retorna (prices_df, iv_df, weights_dict).
    """
    bq = bql.Service()
    univ = bq.univ.members(index_ticker)
    dt_range = bq.func.range('-{}d'.format(lookback_days), '0d')
    req_px = bql.Request(univ, {
        'px': bq.data.px_last(fill='PREV', dates=dt_range),
    })
    req_iv = bql.Request(univ, {
        'iv': bq.data.implied_volatility(fill='PREV', dates=dt_range),
    })

    # Pesos: tentar idx_wt_val, senão cur_mkt_cap como proxy
    try:
        req_wt = bql.Request(univ, {
            'wt': bq.data.idx_wt_val(fill='PREV'),
        })
        resp_wt = bq.execute(req_wt)
        wt_df = resp_wt[0].df()
    except Exception:
        try:
            req_wt = bql.Request(univ, {
                'wt': bq.data.cur_mkt_cap(),
            })
            resp_wt = bq.execute(req_wt)
            wt_df = resp_wt[0].df()
        except Exception:
            wt_df = pd.DataFrame()

    resp_px = bq.execute(req_px)
    resp_iv = bq.execute(req_iv)

    df_px = resp_px[0].df().reset_index()
    # Encontrar coluna de data para usar como index do pivot
    _dt_col_px = next((c for c in df_px.columns if str(c).upper() == 'DATE'), None)
    if _dt_col_px:
        df_px[_dt_col_px] = pd.to_datetime(df_px[_dt_col_px])
        prices_df = df_px.pivot(index=_dt_col_px, columns='ID', values='px')
    else:
        prices_df = df_px.pivot(columns='ID', values='px')

    df_iv = resp_iv[0].df().reset_index()
    _dt_col_iv = next((c for c in df_iv.columns if str(c).upper() == 'DATE'), None)
    if _dt_col_iv:
        df_iv[_dt_col_iv] = pd.to_datetime(df_iv[_dt_col_iv])
        iv_df = df_iv.pivot(index=_dt_col_iv, columns='ID', values='iv')
    else:
        iv_df = df_iv.pivot(columns='ID', values='iv')

    weights = {}
    if not wt_df.empty:
        if 'ID' not in wt_df.columns:
            wt_df = wt_df.reset_index()
        total_wt = 0
        for _, row in wt_df.iterrows():
            ticker = row.get('ID', '')
            w = row.get('wt', 0.0)
            if ticker and w and not np.isnan(w):
                weights[ticker] = float(w)
                total_wt += float(w)
        if total_wt > 0:
            for k in weights:
                weights[k] = weights[k] / total_wt
    return prices_df, iv_df, weights


def _bql_fetch_index_iv(index_ticker='SPX Index', lookback_days=252):
    """Busca IV histórica do índice."""
    bq = bql.Service()
    dt_range = bq.func.range('-{}d'.format(lookback_days), '0d')
    req = bql.Request(index_ticker, {
        'iv': bq.data.implied_volatility(fill='PREV', dates=dt_range),
    })
    resp = bq.execute(req)
    return _bql_ts(resp[0], 'iv')


def _bql_fetch_impl_corr(lookback_days=252):
    """Busca o CBOE S&P 500 3M Implied Correlation Index (.SPXSK3 G Index)."""
    bq = bql.Service()
    dt_range = bq.func.range('-{}d'.format(lookback_days), '0d')
    try:
        req = bql.Request(DISP_SPXSK3, {
            'px': bq.data.px_last(fill='PREV', dates=dt_range),
        })
        resp = bq.execute(req)
        return _bql_ts(resp[0], 'px')
    except Exception:
        return pd.Series(dtype=float)


def _fetch_vol_of_vol_indicators(lookback_days=252):
    """
    Busca indicadores de Vol-of-Vol e Tail Risk para o Gamma Squeeze panel.

    Retorna dict com séries históricas e valores spot:
      vvix_hist, vvix_cur          — VVIX Index (vol da vol)
      vix_call_oi, vix_put_oi     — VIX Total Call/Put OI (spot, milhões)
      sdex_hist, sdex_cur         — CBOE SDEX (downside tail)
      tdex_hist, tdex_cur         — CBOE TDEX (tail risk)
      vix_skew_c25, vix_skew_p25  — VIX 25d Call/Put IV vs ATM (ratio)
      vrp_hist, vrp_cur            — VRP = VIX − RV10D SPX (vol risk premium, %)
      rv10_cur                     — SPX realized vol 10D atual (%)
      axwa_hist, axwa_cur          — AXWA Index: SPX equity funding spread (bps)
      fedpsor1_hist, fedpsor1_cur  — FEDPSOR1 Index: Primary Dealer equity repo ($B)
      es_bid_ask_cur               — ES1 bid-ask spread atual (ticks)
      lagidbma_hist, lagidbma_cur  — LAGIDBMA Index: Conference Board margin level
      spx_iv_hist, spx_iv_cur      — SPX 30d ATM IV histórico
      spx_iv_pct                   — IV Percentile 1Y (0–100)
      spx_skew_hist, spx_skew_cur  — SPX skew 30d (ATM − 90% moneyness)
      spx_skew_pct                 — Skew Percentile 1Y
      rv_multi                     — dict {10:val, 15:val, 21:val, 30:val} RV SPX annualised %
      splv5ute_hist/cur/pct        — SPLV5UTE Index: vol-ctrl 5% equity exposure
      splv10te_hist/cur/pct        — SPLV10TE Index: vol-ctrl 10%
      splv12te_hist/cur/pct        — SPLV12TE Index: vol-ctrl 12%
      splv15te_hist/cur/pct        — SPLV15TE Index: vol-ctrl 15%
    """
    out = {k: None for k in [
        'vvix_hist','vvix_cur',
        'vix_call_oi','vix_put_oi',
        'sdex_hist','sdex_cur',
        'tdex_hist','tdex_cur',
        'vix_skew_c25','vix_skew_p25',
        'vrp_hist','vrp_cur','rv10_cur',
        'rv_multi',
        'axwa_hist','axwa_cur',
        'fedpsor1_hist','fedpsor1_cur',
        'es_bid_ask_cur',
        'lagidbma_hist','lagidbma_cur',
        'spx_iv_hist','spx_iv_cur','spx_iv_pct',
        'spx_skew_hist','spx_skew_cur','spx_skew_pct',
        'splv5ute_hist','splv5ute_cur','splv5ute_pct',
        'splv10te_hist','splv10te_cur','splv10te_pct',
        'splv12te_hist','splv12te_cur','splv12te_pct',
        'splv15te_hist','splv15te_cur','splv15te_pct',
    ]}
    dt_range = bq.func.range(f'-{lookback_days}d', '0d')

    # ── SPX 30d ATM IV + Skew history + percentiles ───────────────────────────
    try:
        _r = bq.execute(bql.Request('SPX Index', {
            'iv_atm':  bq.data.implied_volatility(
                           expiry='30d', pct_moneyness='100',
                           dates=dt_range),
            'iv_90':   bq.data.implied_volatility(
                           expiry='30d', pct_moneyness='90',
                           dates=dt_range),
        }))
        _df_spx = pd.concat([r.df()[r.name] for r in _r], axis=1)
        _df_spx = _df_spx.loc[:, ~_df_spx.columns.duplicated()]
        _iv_s   = pd.to_numeric(_df_spx['iv_atm'], errors='coerce').dropna()
        _sk_s   = pd.to_numeric(_df_spx['iv_atm'] - _df_spx['iv_90'],
                                errors='coerce').dropna()
        if not _iv_s.empty:
            out['spx_iv_hist'] = _iv_s
            out['spx_iv_cur']  = round(float(_iv_s.iloc[-1]), 2)
            out['spx_iv_pct']  = round(float(np.mean(_iv_s.values < _iv_s.values[-1])) * 100, 1)
        if not _sk_s.empty:
            out['spx_skew_hist'] = _sk_s
            out['spx_skew_cur']  = round(float(_sk_s.iloc[-1]), 2)
            out['spx_skew_pct']  = round(float(np.mean(_sk_s.values < _sk_s.values[-1])) * 100, 1)
    except Exception as _e:
        print(f'[SPX IV hist] {_e}')

    # ── VVIX ─────────────────────────────────────────────────────────────────
    try:
        _r = bq.execute(bql.Request('VVIX Index',
                {'px': bq.data.px_last(fill='PREV', dates=dt_range)}))
        _s = _bql_ts(_r[0], 'px').dropna()
        out['vvix_hist'] = _s
        out['vvix_cur']  = float(_s.iloc[-1]) if not _s.empty else None
    except Exception as _e:
        print(f'[VVIX] {_e}')

    # ── SDEX ─────────────────────────────────────────────────────────────────
    for _sdex_ticker in ['SPXSDEX Index', '.SDEX G Index', 'SDEX Index']:
        try:
            _r = bq.execute(bql.Request(_sdex_ticker,
                    {'px': bq.data.px_last(fill='PREV', dates=dt_range)}))
            _s = _bql_ts(_r[0], 'px').dropna()
            if not _s.empty:
                out['sdex_hist'] = _s
                out['sdex_cur']  = float(_s.iloc[-1])
                break
        except Exception:
            continue

    # ── TDEX ─────────────────────────────────────────────────────────────────
    for _tdex_ticker in ['SPXTDEX Index', '.TDEX G Index', 'TDEX Index']:
        try:
            _r = bq.execute(bql.Request(_tdex_ticker,
                    {'px': bq.data.px_last(fill='PREV', dates=dt_range)}))
            _s = _bql_ts(_r[0], 'px').dropna()
            if not _s.empty:
                out['tdex_hist'] = _s
                out['tdex_cur']  = float(_s.iloc[-1])
                break
        except Exception:
            continue

    # ── VIX OI (Call + Put) — via universe filter por tipo ───────────────────
    try:
        _univ_c = bq.univ.filter(bq.univ.options(['VIX Index']),
                                  bq.data.put_call() == 'Call')
        _univ_p = bq.univ.filter(bq.univ.options(['VIX Index']),
                                  bq.data.put_call() == 'Put')
        _rc = bq.execute(bql.Request(_univ_c, {'oi': bq.data.open_int()}))
        _rp = bq.execute(bql.Request(_univ_p, {'oi': bq.data.open_int()}))
        out['vix_call_oi'] = round(float(_rc[0].df()['oi'].sum()) / 1e6, 2)
        out['vix_put_oi']  = round(float(_rp[0].df()['oi'].sum()) / 1e6, 2)
    except Exception as _e:
        print(f'[VIX OI] {_e}')

    # ── VIX 25-delta Call/Put skew vs ATM ─────────────────────────────────
    # implied_volatility() direto no VIX Index — mais limpo e confiável
    try:
        # VIX não suporta delta negativo nem put_call — usa pct_moneyness como proxy
        # 110% moneyness ≈ 25d call, 90% moneyness ≈ 25d put no VIX
        _r = bq.execute(bql.Request('VIX Index', {
            'iv_atm': bq.data.implied_volatility(expiry='30D', pct_moneyness='100'),
            'iv_c25': bq.data.implied_volatility(expiry='30D', pct_moneyness='110'),
            'iv_p25': bq.data.implied_volatility(expiry='30D', pct_moneyness='90'),
        }))
        _df_iv = pd.concat([r.df()[r.name] for r in _r], axis=1)
        _df_iv = _df_iv.loc[:, ~_df_iv.columns.duplicated()]
        _atm = float(_df_iv['iv_atm'].dropna().iloc[-1])
        if _atm and _atm > 0:
            out['vix_skew_c25'] = round(float(_df_iv['iv_c25'].dropna().iloc[-1]) / _atm, 3)
            out['vix_skew_p25'] = round(float(_df_iv['iv_p25'].dropna().iloc[-1]) / _atm, 3)
    except Exception as _e:
        print(f'[VIX skew] {_e}')

    # ── VRP = VIX − RV 10D (vol risk premium) ────────────────────────────────
    # VIX já está em % anualizado; RV10D = std 10d dos retornos SPX × sqrt(252) × 100
    try:
        _dt_vrp = bq.func.range(f'-{lookback_days + 20}d', '0d')  # +20 warm-up
        _r_vix  = bq.execute(bql.Request('VIX Index',
                    {'px': bq.data.px_last(fill='PREV', dates=_dt_vrp)}))
        _r_spx  = bq.execute(bql.Request('SPX Index',
                    {'px': bq.data.px_last(fill='PREV', dates=_dt_vrp)}))
        _vix_s  = _bql_ts(_r_vix[0], 'px').dropna()
        _spx_s  = _bql_ts(_r_spx[0], 'px').dropna()
        _rv10   = (_spx_s.pct_change()
                         .rolling(10).std()
                         .mul(np.sqrt(252) * 100)  # → % anualizado, igual ao VIX
                         .dropna())
        _common = _vix_s.index.intersection(_rv10.index)
        _vrp    = (_vix_s.reindex(_common) - _rv10.reindex(_common)).dropna()
        # recorta ao lookback solicitado
        _vrp    = _vrp.iloc[-lookback_days:]
        _rv10   = _rv10.iloc[-lookback_days:]
        out['vrp_hist'] = _vrp
        out['vrp_cur']  = round(float(_vrp.iloc[-1]),  2) if not _vrp.empty  else None
        out['rv10_cur'] = round(float(_rv10.iloc[-1]), 2) if not _rv10.empty else None
        # RV multi-window (10, 15, 21, 30d) — mesma série SPX já carregada
        _log_ret = _spx_s.pct_change().dropna()
        _rv_multi = {}
        for _w in (10, 15, 21, 30):
            _rv_w = _log_ret.rolling(_w).std().mul(np.sqrt(252) * 100).dropna()
            if not _rv_w.empty:
                _rv_multi[_w] = round(float(_rv_w.iloc[-1]), 2)
        out['rv_multi'] = _rv_multi if _rv_multi else None
    except Exception as _e:
        print(f'[VRP] {_e}')

    # ── Vol-Control Fund Exposure (SPLV*TE) ───────────────────────────────────
    for _splv_ticker, _splv_key in [
        ('SPLV5UTE Index',  'splv5ute'),
        ('SPLV10TE Index',  'splv10te'),
        ('SPLV12TE Index',  'splv12te'),
        ('SPLV15TE Index',  'splv15te'),
    ]:
        try:
            _r = bq.execute(bql.Request(_splv_ticker,
                    {'px': bq.data.px_last(fill='PREV', dates=dt_range)}))
            _s = _bql_ts(_r[0], 'px').dropna()
            if not _s.empty:
                out[f'{_splv_key}_hist'] = _s
                out[f'{_splv_key}_cur']  = round(float(_s.iloc[-1]), 1)
                out[f'{_splv_key}_pct']  = round(float(np.mean(_s.values < _s.values[-1])) * 100, 1)
        except Exception as _e:
            print(f'[{_splv_ticker}] {_e}')

    # ── AXWA Index — SPX equity funding spread (financing cost) ──────────────
    # Generic 2nd 'AXW' SPX Funding Future → spread em ticks/bps
    try:
        _r = bq.execute(bql.Request('AXWA Index',
                {'px':  bq.data.px_last(fill='PREV', dates=dt_range),
                 'vol': bq.data.px_volume(fill='PREV', dates=dt_range)}))
        _s_px  = _bql_ts(_r[0], 'px').dropna()
        _s_vol = _bql_ts(_r[1], 'vol').dropna()
        out['axwa_hist'] = _s_px
        out['axwa_cur']  = round(float(_s_px.iloc[-1]),  2) if not _s_px.empty  else None
        out['axwa_vol']  = round(float(_s_vol.iloc[-1]), 0) if not _s_vol.empty else None
    except Exception as _e:
        print(f'[AXWA] {_e}')

    # ── FEDPSOR1 Index — Primary Dealer equity repo outstanding ($B) ──────────
    try:
        _r = bq.execute(bql.Request('FEDPSOR1 Index',
                {'px': bq.data.px_last(fill='PREV', dates=dt_range)}))
        _s = _bql_ts(_r[0], 'px').dropna()
        out['fedpsor1_hist'] = _s
        out['fedpsor1_cur']  = round(float(_s.iloc[-1]), 1) if not _s.empty else None
    except Exception as _e:
        print(f'[FEDPSOR1] {_e}')

    # ── LAGIDBMA Index — Conference Board margin level ───────────────────────
    try:
        _r = bq.execute(bql.Request('LAGIDBMA Index',
                {'px': bq.data.px_last(fill='PREV', dates=dt_range)}))
        _s = _bql_ts(_r[0], 'px').dropna()
        out['lagidbma_hist'] = _s
        out['lagidbma_cur']  = round(float(_s.iloc[-1]), 2) if not _s.empty else None
    except Exception as _e:
        print(f'[LAGIDBMA] {_e}')

    # ── ES1 bid-ask spread — average_bid_ask_spread() em pts, converte p/ ticks ─
    try:
        _r = bq.execute(bql.Request('ES1 Index',
                {'ba': bq.data.average_bid_ask_spread()}))
        _ba_pts = float(_r[0].df()['ba'].iloc[0])
        # ES1 tick = 0.25 pts
        out['es_bid_ask_cur'] = round(_ba_pts / 0.25, 2)
    except Exception as _e:
        print(f'[ES bid-ask] {_e}')

    return out


def compute_realized_correlation(prices_df, windows=None):
    """
    Calcula correlação realizada média (pairwise) para cada janela.
    Retorna DataFrame: index=datas, cols=window labels.
    """
    if windows is None:
        windows = DISP_CORR_WINDOWS
    log_rets = np.log(prices_df / prices_df.shift(1)).dropna()
    result = {}
    for label, w in windows.items():
        corrs = []
        dates_out = []
        for end in range(w, len(log_rets)):
            window_rets = log_rets.iloc[end - w:end]
            corr_mat = window_rets.corr()
            n = len(corr_mat)
            if n < 2:
                continue
            upper = corr_mat.values[np.triu_indices(n, k=1)]
            avg_corr = float(np.nanmean(upper))
            corrs.append(avg_corr)
            dates_out.append(log_rets.index[end])
        result[label] = pd.Series(corrs, index=dates_out)
    return pd.DataFrame(result)


def compute_implied_correlation(sigma_idx, sigmas_i, weights):
    """
    Calcula correlação implícita (CBOE methodology).
    ρ_impl = (σ²_idx - Σ wi² σi²) / (Σ_{i≠j} wi wj σi σj)
    Inputs: escalares (para um dia).
    """
    tickers = list(weights.keys())
    n = len(tickers)
    if n < 2:
        return np.nan
    w = np.array([weights[t] for t in tickers])
    s = np.array([sigmas_i.get(t, 0.0) for t in tickers])
    var_idx = sigma_idx ** 2
    sum_wi2_si2 = np.sum(w ** 2 * s ** 2)
    cross = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cross += w[i] * w[j] * s[i] * s[j]
    denom = 2.0 * cross
    if abs(denom) < 1e-12:
        return np.nan
    rho = (var_idx - sum_wi2_si2) / denom
    return float(np.clip(rho, -1.0, 1.0))


def compute_implied_corr_series(index_iv, member_iv_df, weights):
    """
    Calcula série temporal de correlação implícita.
    index_iv: Series (dates → IV), member_iv_df: DataFrame (dates × tickers).
    """
    common_dates = index_iv.index.intersection(member_iv_df.index)
    impl_corrs = []
    dates_out = []
    for dt in common_dates:
        sig_idx = index_iv.loc[dt]
        if np.isnan(sig_idx) or sig_idx <= 0:
            continue
        sigmas = {}
        for t in weights:
            if t in member_iv_df.columns:
                val = member_iv_df.loc[dt, t]
                if not np.isnan(val) and val > 0:
                    sigmas[t] = val / 100.0
        sub_w = {t: weights[t] for t in sigmas if t in weights}
        total_w = sum(sub_w.values())
        if total_w < 0.5:
            continue
        sub_w = {t: v / total_w for t, v in sub_w.items()}
        rho = compute_implied_correlation(sig_idx / 100.0, sigmas, sub_w)
        if not np.isnan(rho):
            impl_corrs.append(rho)
            dates_out.append(dt)
    return pd.Series(impl_corrs, index=dates_out, name='impl_corr')


def compute_dispersion_signal(impl_corr_series, real_corr_df, window='3M'):
    """
    Gera sinal de dispersion trade.
    Spread = Impl Corr - Realized Corr → positivo = vender vol do índice.
    Retorna DataFrame com colunas: impl_corr, real_corr, spread, z_score, signal.
    """
    real_col = real_corr_df[window] if window in real_corr_df.columns else real_corr_df.iloc[:, 0]
    common = impl_corr_series.index.intersection(real_col.index)
    if len(common) < 20:
        return pd.DataFrame()
    impl = impl_corr_series.reindex(common)
    real = real_col.reindex(common)
    spread = impl - real
    roll_mean = spread.rolling(63, min_periods=20).mean()
    roll_std = spread.rolling(63, min_periods=20).std()
    z = (spread - roll_mean) / roll_std.replace(0, np.nan)
    signal = pd.Series('NEUTRAL', index=common)
    signal[z > 1.0] = 'SHORT INDEX VOL'
    signal[z < -1.0] = 'LONG INDEX VOL'
    return pd.DataFrame({
        'impl_corr': impl, 'real_corr': real,
        'spread': spread, 'z_score': z, 'signal': signal,
    })


def optimize_tracking_basket(index_prices, member_prices, n_stocks=10):
    """
    Seleciona N ações que melhor replicam o índice (minimize tracking error).
    Retorna (selected_tickers, weights_dict, tracking_error).
    """
    idx_ret = np.log(index_prices / index_prices.shift(1)).dropna()
    mem_ret = np.log(member_prices / member_prices.shift(1)).dropna()
    common = idx_ret.index.intersection(mem_ret.index)
    idx_ret = idx_ret.reindex(common).values
    mem_ret = mem_ret.reindex(common)
    tickers = list(mem_ret.columns)
    mem_arr = mem_ret.values

    corrs = np.array([np.corrcoef(idx_ret, mem_arr[:, i])[0, 1]
                       for i in range(mem_arr.shape[1])])
    valid = ~np.isnan(corrs)
    top_k = min(max(n_stocks * 3, 30), int(valid.sum()))
    top_idx = np.argsort(-np.abs(np.where(valid, corrs, 0)))[:top_k]
    sub_tickers = [tickers[i] for i in top_idx]
    sub_arr = mem_arr[:, top_idx]

    def objective(w):
        port_ret = sub_arr @ w
        te = np.std(port_ret - idx_ret)
        return te

    n = len(sub_tickers)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    res = sp_minimize(objective, w0, method='SLSQP',
                      bounds=bounds, constraints=cons,
                      options={'maxiter': 500, 'ftol': 1e-10})
    opt_w = res.x
    ranked = np.argsort(-opt_w)[:n_stocks]
    sel_tickers = [sub_tickers[i] for i in ranked]
    sel_weights = {sub_tickers[i]: float(opt_w[i]) for i in ranked}
    total = sum(sel_weights.values())
    if total > 0:
        sel_weights = {t: v / total for t, v in sel_weights.items()}
    te_final = float(objective(opt_w))
    return sel_tickers, sel_weights, te_final


def compute_mag7_dispersion(prices_df):
    """
    Calcula dispersão entre pares Mag7.
    Retorna DataFrame com IV spread e correlation para cada par.
    """
    mag7_in_df = [t for t in MAG7 if t in prices_df.columns]
    if len(mag7_in_df) < 2:
        return pd.DataFrame()
    log_rets = np.log(prices_df[mag7_in_df] / prices_df[mag7_in_df].shift(1)).dropna()
    corr_mat = log_rets.tail(63).corr()
    pairs = []
    for i in range(len(mag7_in_df)):
        for j in range(i + 1, len(mag7_in_df)):
            t1 = mag7_in_df[i]
            t2 = mag7_in_df[j]
            short_1 = t1.split(' ')[0]
            short_2 = t2.split(' ')[0]
            corr_val = corr_mat.loc[t1, t2]
            vol1 = float(log_rets[t1].std() * np.sqrt(252) * 100)
            vol2 = float(log_rets[t2].std() * np.sqrt(252) * 100)
            pairs.append({
                'Par': '{}/{}'.format(short_1, short_2),
                'Corr 3M': round(corr_val, 3),
                'RVol1 (%)': round(vol1, 1),
                'RVol2 (%)': round(vol2, 1),
                'Vol Spread': round(abs(vol1 - vol2), 1),
            })
    df = pd.DataFrame(pairs)
    df = df.sort_values('Vol Spread', ascending=False).reset_index(drop=True)
    return df


def find_best_2x2_dispersion(prices_df, iv_df=None):
    """
    Encontra melhor combo NxN entre Mag7 para dispersion trade.
    Testa 2x2, 3x3, ... até floor(len/2) x floor(len/2).
    Para cada tamanho N: top-N high vol vs bottom-N low vol.
    Retorna lista com melhor combo de cada tamanho + o ótimo geral.
    """
    from itertools import combinations

    mag7_in = [t for t in MAG7 if t in prices_df.columns]
    if len(mag7_in) < 4:
        return []
    log_rets = np.log(prices_df[mag7_in] / prices_df[mag7_in].shift(1)).dropna()
    vols = {}
    for t in mag7_in:
        vols[t] = float(log_rets[t].tail(63).std() * np.sqrt(252))

    if iv_df is not None:
        for t in mag7_in:
            if t in iv_df.columns:
                last_iv = iv_df[t].dropna()
                if len(last_iv) > 0:
                    vols[t] = float(last_iv.iloc[-1]) / 100.0

    sorted_by_vol = sorted(mag7_in, key=lambda t: vols.get(t, 0), reverse=True)
    max_n = len(mag7_in) // 2  # max group size

    combos = []
    best_spread = -np.inf
    best_combo = None

    for n in range(2, max_n + 1):
        # Try all combinations of n from high-vol pool vs n from low-vol pool
        # For efficiency: top 2*n candidates, test all combos of n from each half
        pool_high = sorted_by_vol[:min(2 * n, len(sorted_by_vol))]
        pool_low = sorted_by_vol[max(0, len(sorted_by_vol) - 2 * n):]

        local_best_spread = -np.inf
        local_best_high = sorted_by_vol[:n]
        local_best_low = sorted_by_vol[-n:]

        for high_combo in combinations(pool_high, n):
            remaining = [t for t in mag7_in if t not in high_combo]
            if len(remaining) < n:
                continue
            for low_combo in combinations(remaining, n):
                avg_high = np.mean([vols[t] for t in high_combo])
                avg_low = np.mean([vols[t] for t in low_combo])
                spread = avg_high - avg_low
                if spread > local_best_spread:
                    local_best_spread = spread
                    local_best_high = list(high_combo)
                    local_best_low = list(low_combo)

        combo_entry = {
            'Combo': f'{n}x{n}',
            'Long Vol (Buy)': ', '.join(t.split(' ')[0] for t in local_best_high),
            'Short Vol (Sell)': ', '.join(t.split(' ')[0] for t in local_best_low),
            'Avg IV High': round(np.mean([vols[t] for t in local_best_high]) * 100, 1),
            'Avg IV Low': round(np.mean([vols[t] for t in local_best_low]) * 100, 1),
            'Spread (pp)': round(local_best_spread * 100, 1),
        }
        combos.append(combo_entry)

        if local_best_spread > best_spread:
            best_spread = local_best_spread
            best_combo = combo_entry

    # Mark the best overall combo
    if best_combo:
        for c in combos:
            c['Ótimo'] = '⭐' if c['Combo'] == best_combo['Combo'] else ''

    return combos


def find_best_pair_combos(prices_df, iv_df=None, straddle_data=None, max_pairs=3):
    """
    Encontra melhores combinações de até max_pairs pares para dispersion trade.
    Cada par = (long vol ticker, short vol ticker).
    Score = soma dos vol spreads * (1 - abs(corr)) para diversificação.
    Retorna list of dicts com detalhes de cada combo (1-par, 2-pares, 3-pares).
    """
    from itertools import combinations

    mag7_in = [t for t in MAG7 if t in prices_df.columns]
    if len(mag7_in) < 2:
        return []
    log_rets = np.log(prices_df[mag7_in] / prices_df[mag7_in].shift(1)).dropna()
    corr_mat = log_rets.tail(63).corr()
    vols = {}
    for t in mag7_in:
        vols[t] = float(log_rets[t].tail(63).std() * np.sqrt(252))
    if iv_df is not None:
        for t in mag7_in:
            if t in iv_df.columns:
                last_iv = iv_df[t].dropna()
                if len(last_iv) > 0:
                    vols[t] = float(last_iv.iloc[-1]) / 100.0

    # Build all possible pairs
    all_pairs = []
    for t1, t2 in combinations(mag7_in, 2):
        v1, v2 = vols[t1], vols[t2]
        if v1 < v2:
            t1, t2 = t2, t1
            v1, v2 = v2, v1
        spread = v1 - v2
        corr = corr_mat.loc[t1, t2] if t1 in corr_mat.index and t2 in corr_mat.index else 0
        # Straddle cost info
        s1_pct = straddle_data.get(t1, {}).get('straddle_pct', np.nan) if straddle_data else np.nan
        s2_pct = straddle_data.get(t2, {}).get('straddle_pct', np.nan) if straddle_data else np.nan
        all_pairs.append({
            'long': t1, 'short': t2,
            'spread': spread, 'corr': corr,
            'score': spread * (1 - abs(corr)),
            'straddle_long': s1_pct, 'straddle_short': s2_pct,
        })
    all_pairs.sort(key=lambda x: x['score'], reverse=True)

    results = []
    for n_pairs in range(1, min(max_pairs + 1, len(all_pairs) + 1)):
        # For n_pairs: find best non-overlapping combination
        best_score = -np.inf
        best_selection = None
        # Try top candidates (limit search space for speed)
        top_pool = all_pairs[:min(15, len(all_pairs))]
        for combo in combinations(range(len(top_pool)), n_pairs):
            selected = [top_pool[i] for i in combo]
            # Ensure no ticker overlap
            tickers_used = set()
            overlap = False
            for p in selected:
                if p['long'] in tickers_used or p['short'] in tickers_used:
                    overlap = True
                    break
                tickers_used.add(p['long'])
                tickers_used.add(p['short'])
            if overlap:
                continue
            total_score = sum(p['score'] for p in selected)
            if total_score > best_score:
                best_score = total_score
                best_selection = selected

        if best_selection is None:
            continue

        pair_strs = []
        for p in best_selection:
            l_short = p['long'].split(' ')[0]
            s_short = p['short'].split(' ')[0]
            pair_strs.append(f"{l_short}/{s_short}")

        avg_spread = np.mean([p['spread'] for p in best_selection])
        avg_corr = np.mean([p['corr'] for p in best_selection])
        straddle_costs = []
        for p in best_selection:
            if not np.isnan(p['straddle_long']) and not np.isnan(p['straddle_short']):
                straddle_costs.append(p['straddle_long'] + p['straddle_short'])

        entry = {
            'N Pares': n_pairs,
            'Pares': ' + '.join(pair_strs),
            'Spread Médio (pp)': round(avg_spread * 100, 1),
            'Corr Média': round(avg_corr, 3),
            'Score': round(best_score * 100, 2),
        }
        if straddle_costs:
            entry['Custo Straddle (%)'] = round(np.mean(straddle_costs), 2)
        results.append(entry)

    return results


# ── Tail Risk (EVT + Conditional Expectations) ──

def compute_tail_risk(log_returns, threshold_pct=5):
    """
    Calcula métricas de risco caudal usando EVT simplificada.
    - Exceedances abaixo do percentil threshold_pct
    - Hill estimator para tail index
    - Conditional Tail Expectation (CTE)
    - Expected Shortfall beyond threshold
    Retorna dict com métricas.
    """
    rets = np.asarray(log_returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    n = len(rets)
    if n < 50:
        return {}

    threshold = np.percentile(rets, threshold_pct)
    exceedances = rets[rets <= threshold]
    n_exceed = len(exceedances)

    cte = float(np.mean(exceedances)) if n_exceed > 0 else np.nan

    abs_exceed = np.abs(exceedances - threshold)
    abs_exceed = abs_exceed[abs_exceed > 0]
    if len(abs_exceed) >= 5:
        log_excess = np.log(abs_exceed / abs_exceed.min())
        hill_alpha = float(1.0 / np.mean(log_excess)) if np.mean(log_excess) > 0 else np.nan
    else:
        hill_alpha = np.nan

    max_loss = float(np.min(rets))
    p1 = np.percentile(rets, 1)
    p5 = np.percentile(rets, 5)
    below_1 = rets[rets <= p1]
    cte_1 = float(np.mean(below_1)) if len(below_1) > 0 else np.nan

    kurtosis_val = float(
        np.mean((rets - np.mean(rets)) ** 4) / (np.std(rets) ** 4)
    ) if np.std(rets) > 0 else np.nan
    skew_val = float(
        np.mean((rets - np.mean(rets)) ** 3) / (np.std(rets) ** 3)
    ) if np.std(rets) > 0 else np.nan

    return {
        'threshold_pct': threshold_pct,
        'threshold_ret': round(float(threshold) * 100, 2),
        'n_exceedances': n_exceed,
        'pct_exceedances': round(n_exceed / n * 100, 1),
        'CTE (5%)': round(float(cte) * 100, 2),
        'CTE (1%)': round(float(cte_1) * 100, 2),
        'Max Loss': round(float(max_loss) * 100, 2),
        'Hill Alpha': round(hill_alpha, 2) if not np.isnan(hill_alpha) else 'N/A',
        'Kurtosis': round(kurtosis_val, 2),
        'Skewness': round(skew_val, 3),
        'VaR 1%': round(float(p1) * 100, 2),
        'VaR 5%': round(float(p5) * 100, 2),
    }


# ── Dispersion Dashboard Visualizations ──

def build_dispersion_chart(disp_signal_df, impl_corr_cboe=None):
    """
    Gráfico principal de dispersion: Impl Corr vs Realized Corr + Spread.
    """
    if disp_signal_df.empty:
        fig = go.Figure()
        fig.add_annotation(text='Sem dados de dispersão', x=0.5, y=0.5,
                           xref='paper', yref='paper', showarrow=False,
                           font=dict(color='white', size=16))
        fig.update_layout(template='plotly_dark',
                          paper_bgcolor='#0d1117', plot_bgcolor='#161b22')
        return go.FigureWidget(fig)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45],
                        subplot_titles=['Correlação Implícita vs Realizada',
                                        'Spread (Impl - Real) + Z-Score'],
                        vertical_spacing=0.14)

    fig.add_trace(go.Scatter(
        x=disp_signal_df.index, y=disp_signal_df['impl_corr'],
        name='Impl Corr (calc)', line=dict(color='#f0883e', width=2.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=disp_signal_df.index, y=disp_signal_df['real_corr'],
        name='Real Corr 3M', line=dict(color='#58a6ff', width=2.5),
    ), row=1, col=1)

    if impl_corr_cboe is not None and len(impl_corr_cboe) > 0:
        cboe_scaled = impl_corr_cboe / 100.0
        fig.add_trace(go.Scatter(
            x=cboe_scaled.index, y=cboe_scaled,
            name='CBOE Impl Corr', line=dict(color='#d29922',
                                              width=1.5, dash='dot'),
        ), row=1, col=1)

    # --- Row 2: Spread bars + Z-score on secondary y ---
    colors = ['#238636' if v > 0 else '#da3633'
              for v in disp_signal_df['spread'].values]
    fig.add_trace(go.Bar(
        x=disp_signal_df.index, y=disp_signal_df['spread'],
        name='Spread', marker_color=colors, opacity=0.55,
    ), row=2, col=1)

    # Z-Score on secondary y-axis for row 2
    fig.add_trace(go.Scatter(
        x=disp_signal_df.index, y=disp_signal_df['z_score'],
        name='Z-Score', line=dict(color='#bc8cff', width=2),
    ), row=2, col=1)

    fig.add_hline(y=1.0, line_dash='dash', line_color='#238636',
                  opacity=0.5, row=2, col=1)
    fig.add_hline(y=-1.0, line_dash='dash', line_color='#da3633',
                  opacity=0.5, row=2, col=1)

    # --- Layout ---
    fig.update_layout(
        template='plotly_dark', height=660,
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', size=12),
        legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center',
                    font=dict(size=12), bgcolor='rgba(13,17,23,0.7)'),
        margin=dict(l=60, r=60, t=60, b=30),
    )
    # Axis labels
    fig.update_yaxes(title_text='Correlação', row=1, col=1, title_font_size=12)
    fig.update_yaxes(title_text='Spread / Z-Score', row=2, col=1, title_font_size=12)
    fig.update_xaxes(tickfont=dict(size=10), row=1, col=1)
    fig.update_xaxes(tickfont=dict(size=10), row=2, col=1)

    return go.FigureWidget(fig)


def build_corr_regime_chart(real_corr_df):
    """Gráfico de correlação realizada em múltiplos timeframes."""
    if real_corr_df.empty:
        fig = go.Figure()
        fig.update_layout(template='plotly_dark',
                          paper_bgcolor='#0d1117', plot_bgcolor='#161b22')
        return go.FigureWidget(fig)

    palette = {'1M': '#58a6ff', '3M': '#f0883e', '6M': '#bc8cff'}
    fig = go.Figure()
    for col in real_corr_df.columns:
        fig.add_trace(go.Scatter(
            x=real_corr_df.index, y=real_corr_df[col],
            name='Corr {}'.format(col),
            line=dict(color=palette.get(col, '#8b949e'), width=1.8),
        ))
    fig.add_hline(y=0.5, line_dash='dash', line_color='#8b949e', opacity=0.4)
    fig.update_layout(
        template='plotly_dark', height=300,
        title='Correlação Realizada Média (Pairwise)',
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', size=11),
        yaxis_title='Avg Correlation',
        legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
        margin=dict(l=60, r=40, t=50, b=30),
    )
    return go.FigureWidget(fig)


def build_tail_risk_chart(log_returns, tail_metrics):
    """Histograma de retornos com cauda marcada + métricas."""
    rets = np.asarray(log_returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets * 100, nbinsx=80, name='Retornos (%)',
        marker_color='#58a6ff', opacity=0.7,
    ))
    var5 = tail_metrics.get('VaR 5%', None)
    var1 = tail_metrics.get('VaR 1%', None)
    if var5 is not None:
        fig.add_vline(x=var5, line_dash='dash', line_color='#f0883e',
                      annotation_text='VaR 5%', annotation_font_color='#f0883e')
    if var1 is not None:
        fig.add_vline(x=var1, line_dash='dash', line_color='#da3633',
                      annotation_text='VaR 1%', annotation_font_color='#da3633')
    fig.update_layout(
        template='plotly_dark', height=300,
        title='Distribuição de Retornos + Tail Risk',
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9', size=11),
        xaxis_title='Retorno (%)', yaxis_title='Frequência',
        margin=dict(l=60, r=40, t=50, b=30),
    )
    return go.FigureWidget(fig)


def _disp_table_widget(df, title='', height='250px'):
    """Helper: cria widget de tabela para dispersion data."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return wd.HTML('<p style="color:#8b949e;">{}: Sem dados</p>'.format(title))
    html = '<div style="color:#c9d1d9; font-size:12px;">'
    html += '<b>{}</b>'.format(title)
    html += '<table style="border-collapse:collapse; width:100%; margin-top:5px;">'
    html += '<tr>'
    for c in df.columns:
        html += '<th style="border-bottom:1px solid #30363d; padding:4px 8px; '
        html += 'text-align:left; color:#58a6ff;">{}</th>'.format(c)
    html += '</tr>'
    for _, row in df.iterrows():
        html += '<tr>'
        for c in df.columns:
            val = row[c]
            html += '<td style="padding:3px 8px; border-bottom:1px solid #21262d; '
            html += 'color:#c9d1d9;">{}</td>'.format(val)
        html += '</tr>'
    html += '</table></div>'
    return wd.HTML(html)


# ── Gamma History Database ──────────────────────────────────────────────────
def load_gamma_history(path=None):
    """
    Carrega histórico de gamma do CSV e calcula RV 21d a partir do Ref Px.
    Retorna DataFrame com colunas: date, px, gamma, delta, call_wall, put_wall,
    vol_trigger, rv21d.
    """
    fpath = path or GAMMA_HISTORY_PATH
    print(f"[GAMMA DB] Loading from: {fpath} (exists={os.path.exists(fpath)})")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_csv(fpath)
    print(f"[GAMMA DB] CSV columns: {list(df.columns)}, shape={df.shape}")
    # Normalize column names (strip whitespace, handle variations)
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(' ', '_')
        if cl in ('trade_date', 'date'):
            col_map[c] = 'date'
        elif cl in ('ref_px', 'price', 'px', 'spot'):
            col_map[c] = 'px'
        elif cl in ('net_gamma', 'gamma', 'gamma_index'):
            col_map[c] = 'gamma'
        elif cl in ('net_delta', 'delta'):
            col_map[c] = 'delta'
        elif cl in ('call_wall',):
            col_map[c] = 'call_wall'
        elif cl in ('put_wall',):
            col_map[c] = 'put_wall'
        elif cl in ('vol_trigger',):
            col_map[c] = 'vol_trigger'
        elif cl in ('data_release',):
            col_map[c] = 'data_release'
    df = df.rename(columns=col_map)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    # Remove linhas corrompidas: px ou gamma não numérico (linhas mescladas no CSV)
    for _col in ['px', 'gamma']:
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors='coerce')
    df = df.dropna(subset=[c for c in ['px', 'gamma'] if c in df.columns])
    df = df.sort_values('date').reset_index(drop=True)
    # RV 21d (annualized) from Ref Px
    if 'px' in df.columns:
        df['px'] = pd.to_numeric(df['px'], errors='coerce')
        log_ret = np.log(df['px'] / df['px'].shift(1))
        df['rv21d'] = log_ret.rolling(21).std() * np.sqrt(252)
    print(f"[GAMMA DB] After processing: {len(df)} rows, cols={list(df.columns)}")
    return df


def append_gamma_snapshot(spot, gamma_idx, net_delta, call_wall, put_wall,
                          vol_trigger, trade_date=None, path=None):
    """Append today's snapshot to gamma history CSV.
    Same day → replaces last row for that date. New day → appends new row.
    Numbers formatted with fixed decimal places to avoid CSV corruption."""
    fpath = path or GAMMA_HISTORY_PATH
    dt = trade_date or datetime.now().strftime('%Y-%m-%d')
    new_row = (f"{dt},{int(round(spot))},{float(gamma_idx):.4f},"
               f"{int(round(net_delta))},{int(round(call_wall))},"
               f"{int(round(put_wall))},{int(round(vol_trigger))},{dt}\n")
    header = "Trade Date,Ref Px,Net Gamma,Net Delta,Call Wall,Put Wall,Vol Trigger,Data Release\n"

    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            lines = f.readlines()
        # Ensure every line ends with \n to prevent row merging on write
        lines = [ln if ln.endswith('\n') else ln + '\n' for ln in lines]
        kept = [lines[0]] if lines else [header]  # keep header
        for line in lines[1:]:
            # Keep only rows that are NOT today's date
            if not line.startswith(dt + ','):
                kept.append(line)
        kept.append(new_row)
        with open(fpath, 'w') as f:
            f.writelines(kept)
    else:
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, 'w') as f:
            f.write(header)
            f.write(new_row)
    print(f"[GAMMA DB] Snapshot saved for {dt}")


def build_rv_gamma_chart(gamma_hist, current_gamma=None, current_rv=None):
    """
    Scatter: RV 21d (y) vs Gamma Index (x) com regressão OLS + teste de hipótese.
    Colore por regime: gamma positivo (verde), gamma negativo (vermelho).
    Destaca ponto atual + forecasted RV.
    """
    from scipy import stats as sp_stats

    df = gamma_hist.dropna(subset=['rv21d', 'gamma']).copy()
    if df.empty:
        return wd.HTML('<p style="color:#8b949e;">Sem dados históricos de gamma.</p>')

    pos = df[df['gamma'] >= 0]
    neg = df[df['gamma'] < 0]

    fig = go.Figure()

    # Positive gamma regime (green)
    if not pos.empty:
        fig.add_trace(go.Scatter(
            x=pos['gamma'], y=pos['rv21d'] * 100,
            mode='markers', name='γ ≥ 0 (Longo Gamma)',
            marker=dict(color='rgba(63,185,80,0.45)', size=5),
            text=pos['date'].dt.strftime('%Y-%m-%d') + '<br>Px: ' + pos['px'].astype(str),
            hovertemplate='<b>%{text}</b><br>Gamma: %{x:.2f}<br>RV21d: %{y:.1f}%<extra></extra>',
        ))

    # Negative gamma regime (red)
    if not neg.empty:
        fig.add_trace(go.Scatter(
            x=neg['gamma'], y=neg['rv21d'] * 100,
            mode='markers', name='γ < 0 (Curto Gamma)',
            marker=dict(color='rgba(248,81,73,0.45)', size=5),
            text=neg['date'].dt.strftime('%Y-%m-%d') + '<br>Px: ' + neg['px'].astype(str),
            hovertemplate='<b>%{text}</b><br>Gamma: %{x:.2f}<br>RV21d: %{y:.1f}%<extra></extra>',
        ))

    # ── OLS Regression line + hypothesis test ──
    x_all = df['gamma'].values.astype(float)
    y_all = (df['rv21d'] * 100).values.astype(float)
    slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x_all, y_all)
    r2 = r_val ** 2
    n = len(x_all)
    f_stat = (r2 / 1) / ((1 - r2) / (n - 2)) if r2 < 1 and n > 2 else np.inf
    f_pval = 1 - sp_stats.f.cdf(f_stat, 1, n - 2) if n > 2 else 1.0

    x_fit = np.linspace(x_all.min(), x_all.max(), 200)
    y_fit = slope * x_fit + intercept
    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode='lines', name=f'OLS (R²={r2:.3f}, p={p_val:.2e})',
        line=dict(color='#58a6ff', width=2.5, dash='solid'),
    ))

    # ── Confidence band (95%) ──
    x_mean = x_all.mean()
    se_fit = std_err * np.sqrt(1/n + (x_fit - x_mean)**2 / ((x_all - x_mean)**2).sum())
    t_crit = sp_stats.t.ppf(0.975, n - 2)
    y_upper = y_fit + t_crit * se_fit
    y_lower = y_fit - t_crit * se_fit
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fit, x_fit[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself', fillcolor='rgba(88,166,255,0.12)',
        line=dict(width=0), showlegend=False,
        hoverinfo='skip',
    ))

    # Forecasted RV at current gamma
    forecasted_rv = None
    if current_gamma is not None:
        forecasted_rv = slope * current_gamma + intercept
        fig.add_trace(go.Scatter(
            x=[current_gamma], y=[forecasted_rv],
            mode='markers', name=f'Forecast: {forecasted_rv:.1f}%',
            marker=dict(color='#58a6ff', size=12, symbol='diamond',
                        line=dict(width=2, color='white')),
            hovertemplate=f'<b>FORECAST</b><br>Gamma: {current_gamma:.2f}<br>'
                          f'RV Prevista: {forecasted_rv:.1f}%<extra></extra>',
        ))

    # Último dia histórico do CSV (marcador destacado)
    last_row = df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_row['gamma']], y=[last_row['rv21d'] * 100],
        mode='markers+text', name=f"Último ({last_row['date'].strftime('%Y-%m-%d')})",
        marker=dict(color='#ffffff', size=13, symbol='circle',
                    line=dict(width=2.5, color='#f0883e')),
        text=[last_row['date'].strftime('%m/%d')],
        textposition='top center',
        textfont=dict(size=9, color='#f0883e'),
        hovertemplate=(f"<b>ÚLTIMO HISTÓRICO</b><br>"
                       f"Data: {last_row['date'].strftime('%Y-%m-%d')}<br>"
                       f"Gamma: %{{x:.2f}}<br>RV21d: %{{y:.1f}}%<extra></extra>"),
    ))

    # Current point
    if current_gamma is not None and current_rv is not None:
        fig.add_trace(go.Scatter(
            x=[current_gamma], y=[current_rv * 100],
            mode='markers', name='HOJE (Realizado)',
            marker=dict(color='#f0883e', size=14, symbol='star',
                        line=dict(width=2, color='white')),
            hovertemplate='<b>HOJE</b><br>Gamma: %{x:.2f}<br>RV21d: %{y:.1f}%<extra></extra>',
        ))

    # Add zero gamma vertical line
    fig.add_vline(x=0, line=dict(color='#8b949e', dash='dash', width=1),
                  annotation_text='Gamma Flip', annotation_position='top')

    # RV percentiles as horizontal lines
    all_rv = df['rv21d'].dropna() * 100
    for pct, lbl, clr in [(0.75, '75th', '#d29922'), (0.90, '90th', '#f85149')]:
        val = all_rv.quantile(pct)
        fig.add_hline(y=val, line=dict(color=clr, dash='dot', width=1),
                      annotation_text=f'RV {lbl}: {val:.0f}%',
                      annotation_position='right')

    # Annotation with regression stats
    sig_txt = '✅ Significativo' if p_val < 0.05 else '❌ Não significativo'
    ann_text = (f"OLS: RV = {slope:.3f}·γ + {intercept:.1f}<br>"
                f"R² = {r2:.4f} | p = {p_val:.2e}<br>"
                f"F = {f_stat:.1f} (p={f_pval:.2e}) | n = {n}<br>"
                f"{sig_txt}")
    if forecasted_rv is not None and current_rv is not None:
        delta = current_rv * 100 - forecasted_rv
        ann_text += f"<br>Forecast: {forecasted_rv:.1f}% | Δ: {delta:+.1f}pp"
    fig.add_annotation(
        x=0.02, y=0.98, xref='paper', yref='paper',
        text=ann_text, showarrow=False,
        font=dict(size=10, color='#c9d1d9'),
        bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
        borderwidth=1, borderpad=6, align='left',
        xanchor='left', yanchor='top')

    fig.update_layout(
        title='RV Realizada 21d vs Gamma Index — Regressão + Teste de Hipótese',
        template=DASH_TEMPLATE,
        height=480,
        margin=dict(l=50, r=80, t=45, b=40),
        xaxis_title='Gamma Index (Net GEX Bn)',
        yaxis_title='Realized Vol 21d (%)',
        legend=dict(orientation='h', yanchor='bottom', y=-0.22,
                    xanchor='center', x=0.5),
    )
    return go.FigureWidget(fig)


def build_gamma_ts_chart(gamma_hist):
    """
    Time-series: Gamma Index + Call Wall + Put Wall + Vol Trigger + Ref Px.
    Dual axis: left = gamma/walls, right = price.
    """
    df = gamma_hist.dropna(subset=['gamma']).copy()
    if df.empty:
        return wd.HTML('<p style="color:#8b949e;">Sem dados históricos de gamma.</p>')

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['gamma'],
        name='Gamma Index', line=dict(color=_C['accent'], width=1.5),
        fill='tozeroy', fillcolor='rgba(88,166,255,0.08)'),
        secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['call_wall'],
        name='Call Wall', line=dict(color=_C['green'], width=1, dash='dot')),
        secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['put_wall'],
        name='Put Wall', line=dict(color=_C['red'], width=1, dash='dot')),
        secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['vol_trigger'],
        name='Vol Trigger (Gamma Flip)', line=dict(color=_C['yellow'], width=1.2, dash='dash')),
        secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['px'],
        name='SPX', line=dict(color='#ffffff', width=1)),
        secondary_y=True)

    fig.update_layout(
        title='Gamma Index + Walls + Vol Trigger vs SPX',
        template=DASH_TEMPLATE,
        height=420,
        margin=dict(l=50, r=60, t=45, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25,
                    xanchor='center', x=0.5),
    )
    fig.update_yaxes(title_text='Gamma Index (Bn)', secondary_y=False)
    fig.update_yaxes(title_text='SPX Level', secondary_y=True)

    return go.FigureWidget(fig)


def build_dispersion_index_chart(cor1m, dspx, vixeq):
    """
    Gráfico estilo Bloomberg G 1059: COR1M + DSPX + VIXEQ (single stock vol premium).
    3 séries no mesmo chart, eixo Y compartilhado.
    """
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=1)

    if not cor1m.empty:
        fig.add_trace(go.Scatter(
            x=cor1m.index, y=cor1m.values,
            name='Impl Corr 1M 50Δ (COR1M)',
            line=dict(color='#FFFFFF', width=1.5)))
    if not dspx.empty:
        fig.add_trace(go.Scatter(
            x=dspx.index, y=dspx.values,
            name='CBOE S&P 500 Dispersion (DSPX)',
            line=dict(color='#58a6ff', width=1.5)))
    if not vixeq.empty:
        fig.add_trace(go.Scatter(
            x=vixeq.index, y=vixeq.values,
            name='Single Stock Vol Premium (VIXEQ)',
            line=dict(color='#f0883e', width=1.5)))

    fig.update_layout(
        title='Low Implied Correlation and Rising Dispersion for S&P 500',
        template=DASH_TEMPLATE,
        height=380,
        margin=dict(l=50, r=50, t=40, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25,
                    xanchor='center', x=0.5),
        yaxis_title='Index Level',
    )
    return go.FigureWidget(fig)


def _tail_metrics_widget(metrics):
    """Widget de métricas de tail risk."""
    if not metrics:
        return wd.HTML('<p style="color:#8b949e;">Sem dados de tail risk.</p>')
    html = '<div style="color:#c9d1d9; font-size:12px;">'
    html += '<b>Tail Risk Metrics (EVT)</b>'
    html += '<table style="border-collapse:collapse; margin-top:5px;">'
    for k, v in metrics.items():
        color = '#da3633' if 'Loss' in str(k) or 'CTE' in str(k) else '#c9d1d9'
        html += '<tr>'
        html += '<td style="padding:3px 10px; color:#8b949e;">{}</td>'.format(k)
        html += '<td style="padding:3px 10px; color:{}; font-weight:600;">'.format(color)
        html += '{}</td>'.format(v)
        html += '</tr>'
    html += '</table></div>'
    return wd.HTML(html)


# ── Multi-Window Correlation Matrix + Dispersion Trade Engine ─────────────

def compute_multi_window_correlations(prices_df, windows=None):
    """
    Calcula matrizes de correlação para múltiplas janelas temporais.
    windows: dict {label: n_days} — e.g., {'1D': 1, '5D': 5, '1M_roll5D': (21,5)}
    Para tupla (W, R): rolling(R) sobre retornos de W dias.
    Retorna dict {label: corr_matrix (DataFrame)}.
    """
    if windows is None:
        windows = {'Intraday (1D)': 1, '5D': 5, '1M rolling 5D': (21, 5)}

    log_rets_1d = np.log(prices_df / prices_df.shift(1)).dropna()
    result = {}

    for label, spec in windows.items():
        if isinstance(spec, tuple):
            # Rolling window: compute W-day returns, then rolling R-day corr
            w_days, r_days = spec
            rets_w = np.log(prices_df / prices_df.shift(w_days)).dropna()
            if len(rets_w) >= r_days:
                result[label] = rets_w.tail(r_days * 4).corr()
        elif spec == 1:
            # Last N days of daily returns for intraday proxy
            result[label] = log_rets_1d.tail(21).corr()
        else:
            # Simple N-day return correlation
            rets_n = np.log(prices_df / prices_df.shift(spec)).dropna()
            if len(rets_n) >= 21:
                result[label] = rets_n.tail(63).corr()

    return result


def build_correlation_heatmap(corr_matrix, title='Correlation Matrix'):
    """Heatmap interativo de correlação com Plotly."""
    tickers = [t.split(' ')[0] for t in corr_matrix.columns]
    z = corr_matrix.values

    fig = go.Figure(data=go.Heatmap(
        z=z, x=tickers, y=tickers,
        colorscale=[
            [0.0, '#da3633'], [0.25, '#f85149'],
            [0.5, '#161b22'], [0.75, '#238636'],
            [1.0, '#3fb950']],
        zmin=-1, zmax=1,
        text=np.round(z, 2), texttemplate='%{text}',
        textfont=dict(size=9),
        hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=title,
        template=DASH_TEMPLATE,
        height=400, width=450,
        margin=dict(l=60, r=20, t=40, b=60),
        xaxis=dict(side='bottom', tickangle=45),
    )
    return go.FigureWidget(fig)


def find_dispersion_pairs(corr_matrices, iv_latest, n_pairs=8):
    """
    Identifica os melhores pares para dispersion trade via straddle/strangle.
    Critérios:
    1. Baixa correlação (ou negativa) entre os ativos
    2. Alto diferencial de IV
    3. Consistência entre janelas temporais
    Retorna DataFrame com pares rankeados.
    """
    # Collect pairwise metrics across windows
    pair_scores = {}
    tickers = None

    for label, corr in corr_matrices.items():
        if tickers is None:
            tickers = list(corr.columns)
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                t1, t2 = tickers[i], tickers[j]
                key = (t1, t2)
                if key not in pair_scores:
                    pair_scores[key] = {'corrs': [], 'labels': []}
                pair_scores[key]['corrs'].append(corr.iloc[i, j])
                pair_scores[key]['labels'].append(label)

    if not pair_scores:
        return pd.DataFrame()

    rows = []
    for (t1, t2), info in pair_scores.items():
        avg_corr = np.mean(info['corrs'])
        min_corr = np.min(info['corrs'])
        iv1 = iv_latest.get(t1, np.nan)
        iv2 = iv_latest.get(t2, np.nan)
        iv_spread = abs(iv1 - iv2) if not (np.isnan(iv1) or np.isnan(iv2)) else 0

        # Score: low correlation + high IV spread = better dispersion opportunity
        # Normalize: corr contributes negatively (lower = better), IV spread positively
        disp_score = (1 - avg_corr) * 0.4 + (1 - min_corr) * 0.3 + iv_spread * 100 * 0.3

        rows.append({
            'Pair': f"{t1.split(' ')[0]}/{t2.split(' ')[0]}",
            'Ticker1': t1, 'Ticker2': t2,
            'Avg Corr': round(avg_corr, 3),
            'Min Corr': round(min_corr, 3),
            'IV1 (%)': round(iv1 * 100, 1) if not np.isnan(iv1) else np.nan,
            'IV2 (%)': round(iv2 * 100, 1) if not np.isnan(iv2) else np.nan,
            'IV Spread (pp)': round(iv_spread * 100, 1),
            'Disp Score': round(disp_score, 2),
        })

    df = pd.DataFrame(rows).sort_values('Disp Score', ascending=False).head(n_pairs)
    return df.reset_index(drop=True)


def fetch_straddle_prices(tickers, expiry_range='30D'):
    """
    Busca preços de straddle ATM (50-delta) + strangle 25-delta para tickers via BQL.
    Para cada ticker: cadeia de opções → ATM call+put (straddle) + 25Δ put + 25Δ call (strangle).
    Retorna dict {ticker: {call_iv, put_iv, straddle_iv, straddle_px, straddle_pct,
                           strangle_iv, strangle_px, strangle_pct, p25_iv, c25_iv, ...}}.
    """
    bq = bql.Service()
    results = {}

    for tk in tickers:
        try:
            # Spot price
            spot_req = bql.Request(tk, {'px': bq.data.px_last()})
            spot_val = bq.execute(spot_req)[0].df().reset_index()
            spot_v = float(spot_val.iloc[0]['px']) if len(spot_val) > 0 else np.nan
            if np.isnan(spot_v):
                continue

            # Options universe: 15-45 DTE
            opt_univ = bq.univ.filter(
                bq.univ.options(tk),
                bq.func.and_(
                    bq.data.expire_dt() >= bq.func.today() + '15D',
                    bq.data.expire_dt() <= bq.func.today() + '45D',
                ))

            req = bql.Request(opt_univ, {
                'strike': bq.data.strike_px(),
                'pc': bq.data.put_call(),
                'iv': bq.data.ivol(),
                'bid': bq.data.px_bid(),
                'ask': bq.data.px_ask(),
                'delta': bq.data.delta(),
                'expiry': bq.data.expire_dt(),
            })
            resp = bq.execute(req)

            # Build DataFrame from all response items
            dfs = []
            for item in resp:
                _d = item.df().reset_index()
                if not _d.empty:
                    dfs.append(_d)
            if not dfs:
                continue

            df = pd.concat(dfs, axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.dropna(subset=['strike', 'pc'])

            # Ensure numeric delta
            df['delta'] = pd.to_numeric(df['delta'], errors='coerce')

            # ── ATM Straddle ──
            df['dist'] = (df['strike'] - spot_v).abs()
            atm_strike = df.loc[df['dist'].idxmin(), 'strike']

            atm = df[df['strike'] == atm_strike]
            calls_atm = atm[atm['pc'] == 'Call']
            puts_atm = atm[atm['pc'] == 'Put']

            if calls_atm.empty or puts_atm.empty:
                continue

            call_row = calls_atm.iloc[0]
            put_row = puts_atm.iloc[0]

            call_mid = (call_row.get('bid', 0) + call_row.get('ask', 0)) / 2
            put_mid = (put_row.get('bid', 0) + put_row.get('ask', 0)) / 2
            call_iv = float(call_row.get('iv', np.nan))
            put_iv = float(put_row.get('iv', np.nan))

            straddle_px = call_mid + put_mid
            straddle_iv = (call_iv + put_iv) / 2 if not (np.isnan(call_iv) or np.isnan(put_iv)) else np.nan

            # ── 25-Delta Strangle ──
            calls_all = df[df['pc'] == 'Call'].copy()
            puts_all = df[df['pc'] == 'Put'].copy()

            c25_iv, p25_iv, strangle_px, strangle_iv = np.nan, np.nan, np.nan, np.nan
            c25_mid, p25_mid = 0.0, 0.0

            # 25-delta call: delta closest to 0.25
            if not calls_all.empty and calls_all['delta'].notna().any():
                calls_all['d25_dist'] = (calls_all['delta'].abs() - 0.25).abs()
                c25_row = calls_all.loc[calls_all['d25_dist'].idxmin()]
                c25_iv = float(c25_row.get('iv', np.nan))
                c25_mid = (c25_row.get('bid', 0) + c25_row.get('ask', 0)) / 2

            # 25-delta put: delta closest to -0.25
            if not puts_all.empty and puts_all['delta'].notna().any():
                puts_all['d25_dist'] = (puts_all['delta'].abs() - 0.25).abs()
                p25_row = puts_all.loc[puts_all['d25_dist'].idxmin()]
                p25_iv = float(p25_row.get('iv', np.nan))
                p25_mid = (p25_row.get('bid', 0) + p25_row.get('ask', 0)) / 2

            strangle_px = c25_mid + p25_mid
            if not (np.isnan(c25_iv) or np.isnan(p25_iv)):
                strangle_iv = (c25_iv + p25_iv) / 2

            results[tk] = {
                'spot': spot_v,
                'strike': float(atm_strike),
                'expiry': str(call_row.get('expiry', '')),
                'call_iv': call_iv,
                'put_iv': put_iv,
                'straddle_iv': straddle_iv,
                'call_mid': call_mid,
                'put_mid': put_mid,
                'straddle_px': straddle_px,
                'straddle_pct': straddle_px / spot_v * 100 if spot_v > 0 else np.nan,
                'c25_iv': c25_iv,
                'p25_iv': p25_iv,
                'strangle_iv': strangle_iv,
                'strangle_px': strangle_px,
                'strangle_pct': strangle_px / spot_v * 100 if spot_v > 0 else np.nan,
            }
        except Exception as e:
            print(f"⚠️ Straddle {tk}: {e}")

    return results


def fetch_historical_straddle_iv(tickers, lookback=252):
    """
    Busca IV ATM histórica para avaliar se straddle está caro/barato.
    Usa implied_volatility(expiry='30D', pct_moneyness='100').
    Retorna DataFrame com IV histórica de cada ticker.
    """
    bq = bql.Service()
    dt_range = bq.func.range(f'-{lookback}d', '0d')
    iv_hist = {}

    for tk in tickers:
        try:
            req = bql.Request(tk, {
                'iv': bq.data.implied_volatility(
                    expiry='30D', pct_moneyness='100', fill='PREV', dates=dt_range),
            })
            s = _bql_ts(bq.execute(req)[0], 'iv')
            if not s.empty:
                iv_hist[tk] = s
        except Exception:
            pass

    if not iv_hist:
        return pd.DataFrame()
    return pd.DataFrame(iv_hist)


def compute_straddle_richness(straddle_data, iv_hist_df, rv_df=None):
    """
    Avalia se cada straddle está caro ou barato.
    - Percentil da IV atual vs histórico
    - IV vs RV spread (se rv_df disponível)
    Retorna DataFrame com métricas de richness.
    """
    rows = []
    for tk, data in straddle_data.items():
        iv_now = data.get('straddle_iv', np.nan)
        if np.isnan(iv_now):
            continue

        short_name = tk.split(' ')[0]

        # Percentil vs histórico
        hist = iv_hist_df[tk].dropna() if tk in iv_hist_df.columns else pd.Series(dtype=float)
        if len(hist) > 20:
            pct = (hist < iv_now).sum() / len(hist) * 100
            avg_iv = hist.mean()
            std_iv = hist.std()
            z_score = (iv_now - avg_iv) / std_iv if std_iv > 0 else 0
        else:
            pct, avg_iv, std_iv, z_score = np.nan, np.nan, np.nan, np.nan

        # IV-RV spread
        rv_now = np.nan
        if rv_df is not None and tk in rv_df.columns:
            rv_series = rv_df[tk].dropna()
            if len(rv_series) > 0:
                rv_now = rv_series.iloc[-1]

        iv_rv_spread = iv_now - rv_now if not np.isnan(rv_now) else np.nan

        rows.append({
            'Ticker': short_name,
            'IV Atual (%)': round(iv_now * 100, 1) if iv_now < 1 else round(iv_now, 1),
            'IV Média (%)': round(avg_iv * 100, 1) if not np.isnan(avg_iv) and avg_iv < 1 else round(avg_iv, 1) if not np.isnan(avg_iv) else np.nan,
            'Percentil': round(pct, 0) if not np.isnan(pct) else np.nan,
            'Z-Score': round(z_score, 2) if not np.isnan(z_score) else np.nan,
            'RV 21d (%)': round(rv_now * 100, 1) if not np.isnan(rv_now) and rv_now < 1 else round(rv_now, 1) if not np.isnan(rv_now) else np.nan,
            'IV-RV (pp)': round(iv_rv_spread * 100, 1) if not np.isnan(iv_rv_spread) and abs(iv_rv_spread) < 1 else round(iv_rv_spread, 1) if not np.isnan(iv_rv_spread) else np.nan,
            'Straddle (%)': round(data['straddle_pct'], 2),
            'Sinal': 'CARO' if not np.isnan(pct) and pct > 75 else ('BARATO' if not np.isnan(pct) and pct < 25 else 'NEUTRO'),
        })
    return pd.DataFrame(rows)


def build_atm_vol_matrix(straddle_data):
    """
    Constrói matriz de ATM vol: Call/Put IV, Straddle IV, Strangle 25Δ IV,
    Straddle %, Strangle % para Mag8 + SPX.
    Retorna (DataFrame, FigureWidget).
    """
    rows = []
    for tk, data in straddle_data.items():
        short = tk.split(' ')[0]
        call_iv = data.get('call_iv', np.nan)
        put_iv = data.get('put_iv', np.nan)
        strl_iv = data.get('straddle_iv', np.nan)
        c25_iv = data.get('c25_iv', np.nan)
        p25_iv = data.get('p25_iv', np.nan)
        strg_iv = data.get('strangle_iv', np.nan)
        straddle_pct = data.get('straddle_pct', np.nan)
        strangle_pct = data.get('strangle_pct', np.nan)
        # Normalize to % if value is in decimal form (< 1)
        if not np.isnan(call_iv) and call_iv < 1:
            call_iv *= 100
        if not np.isnan(put_iv) and put_iv < 1:
            put_iv *= 100
        if not np.isnan(strl_iv) and strl_iv < 1:
            strl_iv *= 100
        if not np.isnan(c25_iv) and c25_iv < 1:
            c25_iv *= 100
        if not np.isnan(p25_iv) and p25_iv < 1:
            p25_iv *= 100
        if not np.isnan(strg_iv) and strg_iv < 1:
            strg_iv *= 100
        skew = put_iv - call_iv if not (np.isnan(put_iv) or np.isnan(call_iv)) else np.nan
        rows.append({
            'Ticker': short,
            'Call IV (%)': round(call_iv, 1) if not np.isnan(call_iv) else np.nan,
            'Put IV (%)': round(put_iv, 1) if not np.isnan(put_iv) else np.nan,
            'Straddle IV (%)': round(strl_iv, 1) if not np.isnan(strl_iv) else np.nan,
            'Straddle (%)': round(straddle_pct, 2) if not np.isnan(straddle_pct) else np.nan,
            '25Δ Call IV (%)': round(c25_iv, 1) if not np.isnan(c25_iv) else np.nan,
            '25Δ Put IV (%)': round(p25_iv, 1) if not np.isnan(p25_iv) else np.nan,
            'Strangle IV (%)': round(strg_iv, 1) if not np.isnan(strg_iv) else np.nan,
            'Strangle (%)': round(strangle_pct, 2) if not np.isnan(strangle_pct) else np.nan,
            'Skew (pp)': round(skew, 1) if not np.isnan(skew) else np.nan,
        })
    matrix_df = pd.DataFrame(rows)
    if matrix_df.empty:
        return matrix_df, wd.HTML('<p style="color:#8b949e;">Sem dados de ATM vol.</p>')

    # Build grouped bar chart: Straddle IV vs Strangle IV + skew overlay
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=matrix_df['Ticker'], y=matrix_df['Straddle IV (%)'],
        name='Straddle IV', marker_color='rgba(63,185,80,0.75)',
        text=matrix_df['Straddle IV (%)'], textposition='outside',
        textfont=dict(size=10),
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=matrix_df['Ticker'], y=matrix_df['Strangle IV (%)'],
        name='Strangle 25Δ IV', marker_color='rgba(88,166,255,0.75)',
        text=matrix_df['Strangle IV (%)'], textposition='outside',
        textfont=dict(size=10),
    ), secondary_y=False)

    # Add straddle % as spot-normalized cost
    fig.add_trace(go.Scatter(
        x=matrix_df['Ticker'], y=matrix_df['Straddle (%)'],
        name='Straddle Cost (%Spot)', mode='markers+lines',
        marker=dict(color=_C['purple'], size=9, symbol='circle'),
        line=dict(color=_C['purple'], width=1.5, dash='dot'),
    ), secondary_y=True)

    # Skew diamonds
    fig.add_trace(go.Scatter(
        x=matrix_df['Ticker'], y=matrix_df['Skew (pp)'],
        name='Put-Call Skew', mode='markers',
        marker=dict(color=_C['yellow'], size=10, symbol='diamond'),
    ), secondary_y=True)

    fig.update_layout(
        title='ATM Vol Matrix — Straddle & Strangle (Mag8 + SPX)',
        template=DASH_TEMPLATE,
        height=450,
        barmode='group',
        margin=dict(l=50, r=60, t=50, b=40),
        xaxis_title='',
        legend=dict(orientation='h', yanchor='bottom', y=-0.20,
                    xanchor='center', x=0.5, font=dict(size=11)),
    )
    fig.update_yaxes(title_text='IV (%)', secondary_y=False)
    fig.update_yaxes(title_text='Cost (%) / Skew (pp)', secondary_y=True, showgrid=False)
    return matrix_df, go.FigureWidget(fig)


def build_intraday_mag8_chart(prices_df, index_ticker='SPX Index'):
    """
    Gráfico intraday (última sessão) de Mag8 + SPX normalizados como % change.
    Usa dados BQL de preço intraday 1-min bars.
    Se intraday não disponível, usa últimos 5 dias normalized.
    """
    tickers = [t for t in MAG8 if t in prices_df.columns]
    if index_ticker in prices_df.columns:
        tickers.append(index_ticker)

    if not tickers or len(prices_df) < 2:
        return wd.HTML('<p style="color:#8b949e;">Sem dados intraday.</p>')

    # Use last 5 trading days for near-intraday view
    df = prices_df[tickers].tail(5).copy()
    # Normalize: % change from start of window
    base = df.iloc[0]
    pct_change = (df / base - 1) * 100

    fig = go.Figure()
    colors = ['#3fb950', '#f85149', '#58a6ff', '#bc8cff',
              '#d29922', '#f0883e', '#8b949e', '#da3633', 'white']

    for i, tk in enumerate(tickers):
        short = tk.split(' ')[0]
        is_index = 'Index' in tk
        fig.add_trace(go.Scatter(
            x=pct_change.index, y=pct_change[tk],
            name=short,
            line=dict(color=colors[i % len(colors)],
                      width=3 if is_index else 1.5,
                      dash='solid' if is_index else 'solid'),
            opacity=1.0 if is_index else 0.8,
        ))

    fig.add_hline(y=0, line=dict(color='#30363d', width=1))

    fig.update_layout(
        title='Mag8 + SPX — Retorno Recente (% Change, últimos 5D)',
        template=DASH_TEMPLATE,
        height=400,
        margin=dict(l=50, r=30, t=45, b=30),
        xaxis_title='', yaxis_title='% Change',
        legend=dict(orientation='h', yanchor='bottom', y=-0.22,
                    xanchor='center', x=0.5),
    )
    return go.FigureWidget(fig)


def build_dispersion_trade_recommendations(pair_df, richness_df, straddle_data):
    """
    Gera recomendações de trade de dispersão com interpretação clara e detalhada.
    Para cada par: operação exata, direção, preços, IV, percentil.
    """
    if pair_df.empty or richness_df.empty:
        return pd.DataFrame(), ''

    richness_map = {}
    for _, row in richness_df.iterrows():
        richness_map[row['Ticker']] = row

    trades = []
    interp_lines = []
    for idx, (_, pair) in enumerate(pair_df.iterrows()):
        t1_short = pair['Ticker1'].split(' ')[0]
        t2_short = pair['Ticker2'].split(' ')[0]
        r1 = richness_map.get(t1_short, {})
        r2 = richness_map.get(t2_short, {})

        if not r1 or not r2:
            continue

        pct1 = r1.get('Percentil', 50)
        pct2 = r2.get('Percentil', 50)
        if pd.isna(pct1):
            pct1 = 50
        if pd.isna(pct2):
            pct2 = 50

        # Straddle data for each leg
        s1 = straddle_data.get(pair['Ticker1'], {})
        s2 = straddle_data.get(pair['Ticker2'], {})

        # Determine direction: buy cheap vol, sell expensive vol
        if pct1 > pct2:
            long_leg, short_leg = t2_short, t1_short
            long_pct, short_pct = pct2, pct1
            long_s, short_s = s2, s1
            long_r, short_r = r2, r1
        else:
            long_leg, short_leg = t1_short, t2_short
            long_pct, short_pct = pct1, pct2
            long_s, short_s = s1, s2
            long_r, short_r = r1, r2

        spread_pct = short_pct - long_pct
        edge = pair['Disp Score'] * (spread_pct / 100) if spread_pct > 0 else 0

        # Build detailed interpretation
        long_iv = long_r.get('IV Atual (%)', '?')
        short_iv = short_r.get('IV Atual (%)', '?')
        long_strd_pct = long_s.get('straddle_pct', 0)
        short_strd_pct = short_s.get('straddle_pct', 0)
        long_expiry = long_s.get('expiry', '30D')
        short_expiry = short_s.get('expiry', '30D')

        op_tipo = 'DISPERSÃO' if pair['Avg Corr'] < 0.5 else 'RELATIVE VALUE'

        # Clear leg descriptions
        long_desc = f"COMPRAR Straddle ATM {long_leg} (IV={long_iv}%, pctl={long_pct:.0f}th — BARATO)"
        short_desc = f"VENDER Straddle ATM {short_leg} (IV={short_iv}%, pctl={short_pct:.0f}th — CARO)"

        trades.append({
            'Operação': f'{long_leg}/{short_leg}',
            'Leg 1 (Long)': long_desc,
            'Leg 2 (Short)': short_desc,
            'Corr': pair['Avg Corr'],
            'Long IV Pctl': round(long_pct, 0),
            'Short IV Pctl': round(short_pct, 0),
            'Straddle Long (%)': round(long_strd_pct, 2) if long_strd_pct else np.nan,
            'Straddle Short (%)': round(short_strd_pct, 2) if short_strd_pct else np.nan,
            'Edge Score': round(edge, 2),
            'Tipo': op_tipo,
        })

        if idx < 3:  # Top 3 detailed interpretation
            interp_lines.append(
                f"<b style='color:#58a6ff;'>Trade {idx + 1}: {op_tipo} — {long_leg} vs {short_leg}</b><br>"
                f"&nbsp;&nbsp;📈 {long_desc}<br>"
                f"&nbsp;&nbsp;📉 {short_desc}<br>"
                f"&nbsp;&nbsp;↔ Correlação: {pair['Avg Corr']:.3f} | Edge: {edge:.2f} | "
                f"Spread Pctl: {spread_pct:.0f}pp"
            )

    df = pd.DataFrame(trades).sort_values('Edge Score', ascending=False)
    df = df.head(8).reset_index(drop=True)

    # Build interpretation HTML
    interpretation = ''
    if interp_lines:
        interpretation = (
            "<div style='background:#161b22; padding:12px; border-radius:6px; "
            "margin:5px 0; border-left:3px solid #58a6ff;'>"
            "<b style='color:#f0883e; font-size:13px;'>🎯 Interpretação — "
            "O que fazer exatamente:</b><br><br>"
            + "<br><br>".join(interp_lines)
            + "<br><br><small style='color:#8b949e;'>"
            "Long straddle = comprar ATM call + ATM put (aposta que vol sobe ou "
            "ativo se move). Short straddle = vender ATM call + ATM put (aposta "
            "que vol cai ou ativo fica parado). Spread de dispersão lucra quando "
            "a vol individual diverge da vol do índice.</small></div>"
        )

    return df, interpretation


def train_dispersion_model(prices_df, iv_df, lookback=126):
    """
    Modelo de ML simples (Logistic Regression) para prever dispersão futura.
    Features: IV spread, correlação rolling, RV ratio, momentum divergence.
    Target: retorno da dispersão foi positivo nos próximos 5 dias?
    Retorna (model, feature_names, accuracy, feature_importance).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit

    tickers = list(prices_df.columns)
    if len(tickers) < 2:
        return None, [], 0, {}

    log_rets = np.log(prices_df / prices_df.shift(1)).dropna()

    # Build features for each day
    features_list = []
    targets = []

    for end_idx in range(63, len(log_rets) - 5):
        window = log_rets.iloc[end_idx - 21:end_idx]
        window_long = log_rets.iloc[end_idx - 63:end_idx]

        # Feature 1: Average pairwise correlation (21d)
        corr_21 = window.corr()
        mask = np.triu(np.ones(corr_21.shape, dtype=bool), k=1)
        avg_corr = corr_21.values[mask].mean()

        # Feature 2: Std of individual vols (cross-sectional vol dispersion)
        vol_cs = window.std() * np.sqrt(252)
        vol_disp = vol_cs.std()

        # Feature 3: Correlation change (21d vs 63d)
        corr_63 = window_long.corr()
        avg_corr_63 = corr_63.values[mask[:corr_63.shape[0], :corr_63.shape[1]]].mean() \
            if corr_63.shape == corr_21.shape else avg_corr
        corr_chg = avg_corr - avg_corr_63

        # Feature 4: Mean IV level (if available)
        mean_iv = 0
        if iv_df is not None and not iv_df.empty:
            iv_slice = iv_df.iloc[min(end_idx, len(iv_df) - 1)]
            mean_iv = iv_slice.mean() if not iv_slice.isna().all() else 0

        # Feature 5: Return dispersion (cross-sectional std of returns)
        ret_disp = window.iloc[-1].std()

        # Feature 6: Momentum divergence (range of cumulative returns)
        cum_21 = window.sum()
        mom_div = cum_21.max() - cum_21.min()

        features_list.append([avg_corr, vol_disp, corr_chg, mean_iv, ret_disp, mom_div])

        # Target: was cross-sectional vol dispersion higher in next 5 days?
        future = log_rets.iloc[end_idx:end_idx + 5]
        future_disp = future.std().std() * np.sqrt(252)
        current_disp = vol_disp
        targets.append(1 if future_disp > current_disp else 0)

    if len(features_list) < 50:
        return None, [], 0, {}

    X = np.array(features_list)
    y = np.array(targets)

    # Remove NaN rows
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]

    if len(X) < 50:
        return None, [], 0, {}

    feature_names = ['Avg Corr 21d', 'Vol Dispersion', 'Corr Change',
                     'Mean IV', 'Return Dispersion', 'Momentum Divergence']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []
    for train_idx, test_idx in tscv.split(X_scaled):
        model = LogisticRegression(max_iter=500, C=0.1, random_state=42)
        model.fit(X_scaled[train_idx], y[train_idx])
        acc = model.score(X_scaled[test_idx], y[test_idx])
        accuracies.append(acc)

    # Train final model on all data
    final_model = LogisticRegression(max_iter=500, C=0.1, random_state=42)
    final_model.fit(X_scaled, y)

    importance = dict(zip(feature_names, np.abs(final_model.coef_[0])))

    # Store scaler for prediction
    final_model._scaler = scaler
    final_model._feature_names = feature_names

    return final_model, feature_names, np.mean(accuracies), importance


def predict_dispersion(model, prices_df, iv_df=None):
    """
    Usa modelo treinado para prever probabilidade de dispersão futura.
    Retorna (probability, features_dict).
    """
    if model is None:
        return 0.5, {}

    log_rets = np.log(prices_df / prices_df.shift(1)).dropna()
    if len(log_rets) < 63:
        return 0.5, {}

    window21 = log_rets.tail(21)
    window63 = log_rets.tail(63)

    corr_21 = window21.corr()
    mask = np.triu(np.ones(corr_21.shape, dtype=bool), k=1)
    avg_corr = corr_21.values[mask].mean()

    vol_cs = window21.std() * np.sqrt(252)
    vol_disp = vol_cs.std()

    corr_63 = window63.corr()
    mask_63 = np.triu(np.ones(corr_63.shape, dtype=bool), k=1)
    avg_corr_63 = corr_63.values[mask_63].mean()
    corr_chg = avg_corr - avg_corr_63

    mean_iv = 0
    if iv_df is not None and not iv_df.empty:
        iv_last = iv_df.iloc[-1]
        mean_iv = iv_last.mean() if not iv_last.isna().all() else 0

    ret_disp = window21.iloc[-1].std()
    cum_21 = window21.sum()
    mom_div = cum_21.max() - cum_21.min()

    X = np.array([[avg_corr, vol_disp, corr_chg, mean_iv, ret_disp, mom_div]])
    X_scaled = model._scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]

    features_dict = dict(zip(model._feature_names, X[0]))
    return float(prob), features_dict


def build_kde_distribution_chart(prices_df, weights=None):
    """
    Chart KDE de distribuição de retornos dos constituintes — estilo Tier1 Alpha.
    Dual display: distribuição individual (equi-weighted, red/green fill) +
    ponderada por peso no índice (yellow line on twin axis).
    Top-10 labels posicionados por peso (y = weight_adj * scale) no eixo secundário.
    Retorna (FigureWidget, HTML_interpretacao).
    """
    from scipy.stats import gaussian_kde

    log_rets_1d = (prices_df.iloc[-1] / prices_df.iloc[-2] - 1) * 100
    rets = log_rets_1d.dropna()

    if len(rets) < 5:
        return wd.HTML('<p style="color:#8b949e;">Dados insuficientes para KDE.</p>'), ''
    if rets.std() == 0:
        return wd.HTML('<p style="color:#8b949e;">Retornos sem variação (possível feriado/dia sem pregão).</p>'), ''

    n_down = int((rets < 0).sum())
    n_up = int((rets >= 0).sum())
    avg_all = rets.mean()
    avg_up = rets[rets >= 0].mean() if (rets >= 0).any() else 0
    avg_dn = rets[rets < 0].mean() if (rets < 0).any() else 0

    # ── Equi-weighted KDE ──
    kde = gaussian_kde(rets)
    x = np.linspace(rets.min() - 2, rets.max() + 2, 1000)
    y = kde(x)

    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ── Fill areas: red (neg) / green (pos) — equi-weighted on primary y ──
    neg_mask = x <= 0
    pos_mask = x >= 0

    # Outline (grey, thin)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines',
        line=dict(color='rgba(150,150,150,0.5)', width=1),
        showlegend=False), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x[neg_mask], y=y[neg_mask],
        fill='tozeroy', fillcolor='rgba(220,38,38,0.50)',
        line=dict(width=0),
        name=f'Down: {n_down} stocks',
        showlegend=False), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x[pos_mask], y=y[pos_mask],
        fill='tozeroy', fillcolor='rgba(34,197,94,0.50)',
        line=dict(width=0),
        name=f'Up: {n_up} stocks',
        showlegend=False), secondary_y=False)

    # ── Weighted KDE (yellow line on secondary y) ──
    has_weighted = False
    if weights:
        w_arr = np.array([weights.get(t, 0) for t in rets.index])
        w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr
        non_zero = w_arr > 0
        if non_zero.sum() > 3 and rets[non_zero].std() > 0:
            try:
                kde_w = gaussian_kde(rets[non_zero], weights=w_arr[non_zero])
                y_w = kde_w(x)
                fig.add_trace(go.Scatter(
                    x=x, y=y_w,
                    line=dict(color='#fbbf24', width=2.5),
                    name='Weighted distribution',
                    opacity=0.95), secondary_y=True)
                has_weighted = True
            except Exception:
                pass

    # ── Average lines ──
    fig.add_vline(x=0, line=dict(color='#6b7280', dash='solid', width=1))
    fig.add_vline(x=avg_all, line=dict(color='white', dash='dash', width=1.5))
    fig.add_vline(x=avg_up, line=dict(color='#22c55e', dash='dash', width=2))
    fig.add_vline(x=avg_dn, line=dict(color='#dc2626', dash='dash', width=2))

    # ── Top-10 by weight: scatter on SECONDARY y using weight as y-position ──
    # This naturally separates labels vertically (heavier stocks = higher y)
    interp_heavy = []
    if weights:
        w_series = pd.Series(weights)
        top10 = w_series.nlargest(10)
        # Compute weight-adjusted position like the reference: w_adj * scale_factor
        # w_adj = (price * w) / sum(price * w) for proper positioning
        top10_data = []
        for tk, w in top10.items():
            if tk in rets.index:
                top10_data.append((tk, w, rets[tk]))
        if top10_data:
            # Scale y: map weight to secondary-axis range for visibility
            max_y_secondary = max(y) * 1.1 if not has_weighted else 0
            if has_weighted:
                max_y_secondary = max(kde_w(x)) if 'kde_w' in dir() else max(y)
            scale = max_y_secondary * 3.5  # scale factor like reference code

            for tk, w, ret_val in top10_data:
                short_name = tk.split(' ')[0]
                # Remove common suffixes
                for sfx in ['UW', 'UQ', 'UN', 'US']:
                    short_name = short_name.replace(sfx, '')
                y_pos = w * scale
                fig.add_trace(go.Scatter(
                    x=[ret_val], y=[y_pos],
                    mode='markers+text',
                    marker=dict(color='#7dd3fc', size=7,
                                line=dict(width=0.5, color='white')),
                    text=[short_name],
                    textposition='top center',
                    textfont=dict(size=12, color='white'),
                    showlegend=False,
                    hovertemplate=f'<b>{short_name}</b><br>Ret: {ret_val:+.1f}%<br>'
                                  f'Peso: {w*100:.1f}%<extra></extra>',
                ), secondary_y=True)

                # Mean-reversion analysis
                w_pct = w * 100
                if tk in prices_df.columns and len(prices_df) > 21:
                    recent = (prices_df[tk].iloc[-21:] / prices_df[tk].iloc[-22:-1].values - 1) * 100
                    r_mean = recent.mean()
                    r_std = recent.std()
                    z = (ret_val - r_mean) / r_std if r_std > 0 else 0
                    if abs(z) > 1.5:
                        direction = 'ABAIXO da média 21d' if z < -1.5 else 'ACIMA da média 21d'
                        interp_heavy.append(
                            f"<b>{short_name}</b> (peso {w_pct:.1f}%): ret 1D = {ret_val:+.1f}%, "
                            f"Z-score 21d = {z:.1f} → {direction} — "
                            f"<b style='color:#fbbf24;'>possível reversão à média</b>")

    # ── Legend entries for stats ──
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='white', dash='dash', width=1.5),
        name=f'Avg. Daily Return: {avg_all:+.1f}%'), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='#22c55e', dash='dash', width=2),
        name=f'Avg. Gain: {avg_up:+.1f}%'), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='#dc2626', dash='dash', width=2),
        name=f'Avg. Decline: {avg_dn:+.1f}%'), secondary_y=False)

    # ── Stats text (upper left) ──
    fig.add_annotation(
        x=0.01, y=0.98, xref='paper', yref='paper',
        text=(f"Stocks Up: {n_up}<br>Stocks Down: {n_down}"),
        showarrow=False, font=dict(size=14, color='white'),
        bgcolor='rgba(0,0,0,0.0)', align='left',
        xanchor='left', yanchor='top')

    # ── Date annotation (bottom center) ──
    _date_str = pd.Timestamp.now().strftime('%Y/%m/%d')
    fig.add_annotation(
        x=0.5, y=-0.08, xref='paper', yref='paper',
        text=f'Date: {_date_str}',
        showarrow=False, font=dict(size=12, color='white'),
        xanchor='center', yanchor='top')

    title_text = ('1D% Return Distribution For SPX, Individual Constituents')

    # Dynamic x range: pad by 2 or at least ±8%
    _x_lo = min(rets.min() - 2, -8)
    _x_hi = max(rets.max() + 2, 8)

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16, color='white')),
        template=DASH_TEMPLATE,
        height=650,
        margin=dict(l=50, r=50, t=50, b=60),
        xaxis=dict(
            title='', zeroline=False,
            tickformat='.1f', ticksuffix='%',
            gridcolor='rgba(128,128,128,0.2)', gridwidth=0.5,
            range=[_x_lo, _x_hi],
            dtick=2.5,
        ),
        yaxis=dict(showgrid=False, title='', showticklabels=True),
        yaxis2=dict(showgrid=False, title='', showticklabels=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1,
                    font=dict(size=11, color='white'),
                    bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
    )

    chart_widget = go.FigureWidget(fig)

    # ── Build interpretation HTML ──
    interp_html = ''
    if interp_heavy or weights:
        parts = []
        parts.append(
            f"<b style='color:#58a6ff;'>📊 Análise de Distribuição</b><br>"
            f"Total: {n_down + n_up} stocks | ↓ {n_down} caindo ({100*n_down/(n_down+n_up):.0f}%) "
            f"| ↑ {n_up} subindo ({100*n_up/(n_down+n_up):.0f}%)<br>"
            f"Média geral: {avg_all:+.2f}% | Média ganhos: +{avg_up:.2f}% "
            f"| Média perdas: {avg_dn:.2f}%"
        )
        if interp_heavy:
            parts.append(
                "<br><br><b style='color:#fbbf24;'>🔄 Candidatos à Reversão à Média "
                "(Heavy Stocks com Z > 1.5):</b><br>"
                + "<br>".join(interp_heavy)
            )
        interp_html = (
            "<div style='background:#161b22; padding:12px; border-radius:6px; "
            "margin:5px 0; border-left:3px solid #fbbf24;'>"
            + "<br>".join(parts)
            + "</div>"
        )

    return chart_widget, interp_html


def build_straddle_richness_chart(richness_df):
    """Bar chart horizontal: IV percentil de cada ativo (caro vs barato)."""
    if richness_df.empty:
        return wd.HTML('<p style="color:#8b949e;">Sem dados de straddle.</p>')

    df = richness_df.sort_values('Percentil', ascending=True)
    colors = ['#da3633' if p > 75 else '#3fb950' if p < 25 else '#8b949e'
              for p in df['Percentil']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Ticker'], x=df['Percentil'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p:.0f}th | IV: {iv}" for p, iv in zip(df['Percentil'], df['IV Atual (%)'])],
        textposition='outside',
        hovertemplate='%{y}: IV Pctl %{x:.0f}th<extra></extra>',
    ))

    fig.add_vline(x=50, line=dict(color='#8b949e', dash='dash', width=1))
    fig.add_vline(x=25, line=dict(color=_C['green'], dash='dot', width=1),
                  annotation_text='Barato', annotation_position='top')
    fig.add_vline(x=75, line=dict(color=_C['red'], dash='dot', width=1),
                  annotation_text='Caro', annotation_position='top')

    fig.update_layout(
        title='Straddle ATM — Percentil de IV (Caro vs Barato)',
        template=DASH_TEMPLATE,
        height=max(250, len(df) * 35),
        margin=dict(l=80, r=60, t=45, b=30),
        xaxis_title='Percentil IV (%)',
        xaxis=dict(range=[0, 105]),
    )
    return go.FigureWidget(fig)


def build_dispersion_ml_widget(model_accuracy, feature_importance, disp_prob, features_dict):
    """Widget HTML com output do modelo ML de dispersão."""
    prob_color = _C['green'] if disp_prob > 0.6 else (_C['red'] if disp_prob < 0.4 else _C['yellow'])
    signal = 'LONG DISPERSÃO' if disp_prob > 0.6 else ('SHORT DISPERSÃO' if disp_prob < 0.4 else 'NEUTRO')

    html = (
        f"<div style='background:#161b22; padding:12px; border-radius:6px; margin:5px 0;'>"
        f"<b style='color:#58a6ff; font-size:14px;'>🤖 ML Dispersion Model</b><br>"
        f"<span style='color:#c9d1d9; font-size:13px;'>"
        f"Acurácia CV: <b>{model_accuracy:.1%}</b> │ "
        f"P(dispersão ↑): <b style='color:{prob_color}'>{disp_prob:.1%}</b> │ "
        f"Sinal: <b style='color:{prob_color}'>{signal}</b></span><br>"
        f"<div style='margin-top:6px;'>"
    )

    # Feature importance bars
    if feature_importance:
        sorted_fi = sorted(feature_importance.items(), key=lambda x: -x[1])
        max_fi = max(v for _, v in sorted_fi) if sorted_fi else 1
        for fname, fval in sorted_fi:
            bar_w = int(fval / max_fi * 150)
            current = features_dict.get(fname, 0)
            html += (
                f"<div style='display:flex; align-items:center; margin:2px 0;'>"
                f"<span style='color:#8b949e; font-size:11px; width:160px;'>{fname}</span>"
                f"<div style='background:#58a6ff; height:10px; width:{bar_w}px; "
                f"border-radius:3px; margin:0 8px;'></div>"
                f"<span style='color:#c9d1d9; font-size:11px;'>{current:.4f}</span>"
                f"</div>"
            )
    html += "</div></div>"
    return wd.HTML(html)


def run_dispersion_analysis(index_ticker='SPX Index', lookback=252):
    """
    Executa análise completa de dispersion trade.
    Usa top-N membros por peso (Mag8 + próximos maiores) para correlação mais robusta.
    Retorna dict com todos os resultados.
    """
    result = {
        'error': None,
        'disp_signal': pd.DataFrame(),
        'real_corr': pd.DataFrame(),
        'impl_corr_cboe': pd.Series(dtype=float),
        'mag7_pairs': pd.DataFrame(),
        'best_2x2': [],
        'best_pairs': [],
        'optimal_basket': {},
        'tail_risk': {},
        'index_returns': np.array([]),
        'hyp_test': {},
        'cor1m': pd.Series(dtype=float),
        'dspx': pd.Series(dtype=float),
        'vixeq': pd.Series(dtype=float),
    }

    try:
        prices_df, iv_df, weights = _bql_fetch_member_data(index_ticker, lookback)
    except Exception as e:
        result['error'] = 'Erro ao buscar membros: {}'.format(str(e))
        return result

    # ── Filtrar top-N membros por peso (mais robusto que 500 stocks) ──
    # Excluir tickers indesejados (ex: BRK/B)
    if weights:
        weights = {t: w for t, w in weights.items() if t not in DISP_EXCLUDE}
    if weights and len(weights) > DISP_TOP_N:
        top_tickers = sorted(weights, key=lambda t: -weights[t])[:DISP_TOP_N]
        top_weights = {t: weights[t] for t in top_tickers}
        tw_sum = sum(top_weights.values())
        if tw_sum > 0:
            top_weights = {t: v / tw_sum for t, v in top_weights.items()}
        top_cols = [c for c in top_tickers if c in prices_df.columns]
        prices_top = prices_df[top_cols] if top_cols else prices_df
        iv_top = iv_df[[c for c in top_cols if c in iv_df.columns]] if top_cols else iv_df
        weights_top = top_weights
        print(f"[DISP] Usando top-{len(top_cols)} membros por peso: "
              f"{[t.split(' ')[0] for t in top_cols]}")
    else:
        prices_top = prices_df
        iv_top = iv_df
        weights_top = weights

    try:
        index_iv = _bql_fetch_index_iv(index_ticker, lookback)
    except Exception as e:
        result['error'] = 'Erro ao buscar IV do índice: {}'.format(str(e))
        return result

    try:
        result['real_corr'] = compute_realized_correlation(prices_top)
    except Exception:
        pass

    try:
        impl_corr_ts = compute_implied_corr_series(index_iv, iv_top, weights_top)
        if not impl_corr_ts.empty and not result['real_corr'].empty:
            result['disp_signal'] = compute_dispersion_signal(
                impl_corr_ts, result['real_corr'], window='3M')
    except Exception:
        pass

    try:
        result['impl_corr_cboe'] = _bql_fetch_impl_corr(lookback)
    except Exception:
        pass

    # ── Fetch Bloomberg dispersion indices: COR1M, DSPX, VIXEQ ──
    bq_svc = bql.Service()
    dt_rng = bq_svc.func.range('-{}d'.format(lookback), '0d')
    for tk, key in [(DISP_COR1M, 'cor1m'), (DISP_DSPX, 'dspx'), (DISP_VIXEQ, 'vixeq')]:
        try:
            req = bql.Request(tk, {'px': bq_svc.data.px_last(fill='PREV', dates=dt_rng)})
            result[key] = _bql_ts(bq_svc.execute(req)[0], 'px').dropna()
        except Exception as _idx_err:
            print(f"⚠️ Disp index {tk}: {_idx_err}")

    try:
        result['mag7_pairs'] = compute_mag7_dispersion(prices_df)
    except Exception:
        pass

    try:
        result['best_2x2'] = find_best_2x2_dispersion(prices_df, iv_df)
    except Exception:
        pass

    try:
        idx_prices = prices_top.mean(axis=1)
        sel, wts, te = optimize_tracking_basket(idx_prices, prices_top, n_stocks=min(10, len(prices_top.columns)))
        result['optimal_basket'] = {'tickers': sel, 'weights': wts,
                                    'tracking_error': te}
    except Exception:
        pass

    # ── Hypothesis test: F-test para R² do modelo de dispersão ──
    try:
        if not result['disp_signal'].empty:
            ds = result['disp_signal'].dropna(subset=['impl_corr', 'real_corr'])
            if len(ds) > 30:
                from scipy import stats as sp_stats
                y = ds['real_corr'].values
                x = ds['impl_corr'].values
                slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x, y)
                r2 = r_val ** 2
                n = len(x)
                k = 1  # 1 regressor
                f_stat = (r2 / k) / ((1 - r2) / (n - k - 1)) if r2 < 1 else np.inf
                f_pval = 1 - sp_stats.f.cdf(f_stat, k, n - k - 1)
                result['hyp_test'] = {
                    'R²': round(r2, 4),
                    'R² adj': round(1 - (1 - r2) * (n - 1) / (n - k - 1), 4),
                    'F-stat': round(f_stat, 2),
                    'p-value': f'{f_pval:.2e}',
                    'slope': round(slope, 4),
                    'intercept': round(intercept, 4),
                    'n_obs': n,
                    'significant': p_val < 0.05,
                }
    except Exception:
        pass

    try:
        log_rets = np.log(prices_top.mean(axis=1) /
                          prices_top.mean(axis=1).shift(1)).dropna().values
        result['index_returns'] = log_rets
        result['tail_risk'] = compute_tail_risk(log_rets)
    except Exception:
        pass

    # ── Multi-window correlation matrices ──
    try:
        corr_matrices = compute_multi_window_correlations(prices_top)
        result['corr_matrices'] = corr_matrices

        # Latest IV for pair scoring
        iv_latest = {}
        if iv_top is not None and not iv_top.empty:
            for col in iv_top.columns:
                last_val = iv_top[col].dropna()
                if len(last_val) > 0:
                    iv_latest[col] = float(last_val.iloc[-1])
                    if iv_latest[col] > 1:
                        iv_latest[col] /= 100.0

        result['dispersion_pairs'] = find_dispersion_pairs(corr_matrices, iv_latest)
        print(f"[DISP] {len(result['dispersion_pairs'])} dispersion pairs identified")
    except Exception as _mwc_err:
        print(f"⚠️ Multi-window corr: {_mwc_err}")
        result['corr_matrices'] = {}
        result['dispersion_pairs'] = pd.DataFrame()

    # ── Straddle pricing (Mag8 + SPX) ──
    # Always include all MAG8 + index — do NOT filter by prices_df.columns
    straddle_tickers = list(MAG8) + [index_ticker]
    try:
        result['straddle_data'] = fetch_straddle_prices(straddle_tickers)
        print(f"[DISP] Straddle prices for {len(result['straddle_data'])} tickers")
        # Fallback: for any MAG8 ticker missing, try BQL implied_volatility directly
        _bq_sd = bql.Service()
        for _mag_tk in MAG8:
            if _mag_tk not in result['straddle_data']:
                try:
                    _iv_req = bql.Request(_mag_tk, {
                        'atm_iv': _bq_sd.data.implied_volatility(
                            expiry='30D', pct_moneyness='100'),
                        'px': _bq_sd.data.px_last(),
                    })
                    _iv_resp = _bq_sd.execute(_iv_req)
                    _iv_df = pd.concat([r.df() for r in _iv_resp], axis=1).reset_index()
                    _atm = float(_iv_df.iloc[0]['atm_iv'])
                    _px = float(_iv_df.iloc[0]['px'])
                    if not np.isnan(_atm) and not np.isnan(_px):
                        result['straddle_data'][_mag_tk] = {
                            'spot': _px, 'strike': _px, 'expiry': '30D',
                            'call_iv': _atm, 'put_iv': _atm,
                            'straddle_iv': _atm,
                            'call_mid': 0, 'put_mid': 0,
                            'straddle_px': 0, 'straddle_pct': 0,
                            'c25_iv': np.nan, 'p25_iv': np.nan,
                            'strangle_iv': np.nan, 'strangle_px': 0,
                            'strangle_pct': 0,
                        }
                        print(f"[DISP] Fallback IV for {_mag_tk}: {_atm:.4f}")
                except Exception:
                    pass
    except Exception as _strd_err:
        print(f"⚠️ Straddle prices: {_strd_err}")
        result['straddle_data'] = {}

    # ── Best pair combos (1-pair, 2-pair, 3-pair) ──
    try:
        result['best_pairs'] = find_best_pair_combos(
            prices_df, iv_df,
            straddle_data=result.get('straddle_data'),
            max_pairs=3)
    except Exception:
        result['best_pairs'] = []

    # ── Historical IV for richness ──
    try:
        iv_hist = fetch_historical_straddle_iv(straddle_tickers, lookback=lookback)
        # RV from prices for IV-RV spread
        rv_df = pd.DataFrame()
        for tk in straddle_tickers:
            if tk in prices_df.columns:
                p = prices_df[tk].dropna()
                lr = np.log(p / p.shift(1))
                rv_df[tk] = lr.rolling(21).std() * np.sqrt(252)

        result['straddle_richness'] = compute_straddle_richness(
            result['straddle_data'], iv_hist, rv_df)
        print(f"[DISP] Richness computed for {len(result['straddle_richness'])} tickers")
    except Exception as _rich_err:
        print(f"⚠️ Straddle richness: {_rich_err}")
        result['straddle_richness'] = pd.DataFrame()

    # ── Trade recommendations ──
    try:
        recs_df, recs_interp = build_dispersion_trade_recommendations(
            result.get('dispersion_pairs', pd.DataFrame()),
            result.get('straddle_richness', pd.DataFrame()),
            result.get('straddle_data', {}))
        result['trade_recs'] = recs_df
        result['trade_interp'] = recs_interp
    except Exception:
        result['trade_recs'] = pd.DataFrame()
        result['trade_interp'] = ''

    # ── ML Dispersion Model ──
    try:
        model, feat_names, accuracy, feat_imp = train_dispersion_model(
            prices_top, iv_top, lookback=min(lookback, 126))
        if model is not None:
            disp_prob, feat_dict = predict_dispersion(model, prices_top, iv_top)
            result['ml_model'] = {
                'accuracy': accuracy,
                'feature_importance': feat_imp,
                'disp_prob': disp_prob,
                'features': feat_dict,
            }
            print(f"[DISP] ML model: accuracy={accuracy:.1%}, P(disp↑)={disp_prob:.1%}")
    except Exception as _ml_err:
        print(f"⚠️ ML Dispersion: {_ml_err}")

    # ── KDE distribution data ──
    result['prices_df'] = prices_df
    result['weights'] = weights

    # ── ATM vol matrix (from straddle data already fetched) ──
    if result.get('straddle_data'):
        try:
            atm_matrix_df, atm_matrix_chart = build_atm_vol_matrix(result['straddle_data'])
            result['atm_vol_matrix'] = atm_matrix_df
            result['atm_vol_chart'] = atm_matrix_chart
        except Exception as _atm_err:
            print(f"⚠️ ATM Vol Matrix: {_atm_err}")

    return result
