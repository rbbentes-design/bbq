"""BQL data pipeline: fetch market data, options chains, historical prices."""

import pandas as pd
import numpy as np
import traceback
from datetime import datetime

import bql

try:
    from .config import bq, _C, TRADING_DAYS
except ImportError:
    import sys, os as _os
    _dir = _os.path.dirname(_os.path.abspath(__file__)) if '__file__' in dir() else _os.getcwd()
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from config import bq, _C, TRADING_DAYS


def _bql_ts(resp_item, field):
    """
    Extrai série temporal de um BQL response item para single-ticker.
    Usa reset_index() para robustez contra variações de MultiIndex.
    Retorna pd.Series com DatetimeIndex.
    """
    df = resp_item.df().reset_index()
    # Encontrar coluna de data
    date_col = None
    for c in df.columns:
        if str(c).upper() == 'DATE':
            date_col = c
            break
    if date_col is None:
        # Tentar detectar coluna com datas
        for c in df.columns:
            if c == field or str(c).upper() == 'ID':
                continue
            try:
                sample = df[c].dropna().head(5)
                if len(sample) > 0:
                    pd.to_datetime(sample)
                    date_col = c
                    break
            except Exception:
                continue
    if date_col is not None:
        s = df.set_index(date_col)[field]
        s.index = pd.to_datetime(s.index)
    else:
        # Fallback: tentar converter index original
        s = resp_item.df()[field]
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            if isinstance(s.index, pd.MultiIndex):
                s.index = s.index.droplevel(0)
                s.index = pd.to_datetime(s.index)
    return s


def _bql_ts_df(resp_item):
    """
    Extrai DataFrame de um BQL response item para single-ticker.
    Retorna DataFrame com DatetimeIndex (sem coluna ID).
    """
    df = resp_item.df().reset_index()
    date_col = None
    for c in df.columns:
        if str(c).upper() == 'DATE':
            date_col = c
            break
    if date_col is not None:
        id_col = [c for c in df.columns if str(c).upper() == 'ID']
        if id_col:
            df = df.drop(columns=id_col)
        df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index)
    else:
        orig = resp_item.df()
        if isinstance(orig.index, pd.MultiIndex):
            orig.index = orig.index.droplevel(0)
        orig.index = pd.to_datetime(orig.index)
        df = orig
    return df


def fetch_market_data(ticker):
    """Busca spot, IV 30d, RV 30d, skew, volume médio em dólares, risk-free rate, MOVE Index."""
    spot = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.px_last()}))[0].df()['Value'].iloc[-1],
        errors='coerce')

    iv_30d = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.implied_volatility(
            expiry='30D', pct_moneyness='100')}))[0].df()['Value'].iloc[-1],
        errors='coerce') / 100.0

    rv_30d = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.volatility_30d_calc()}))[0].df()['Value'].iloc[-1],
        errors='coerce') / 100.0

    put_iv = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.implied_volatility(
            expiry='30D', delta='25', put_call='PUT')}))[0].df()['Value'].iloc[-1],
        errors='coerce')
    call_iv = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': bq.data.implied_volatility(
            expiry='30D', delta='25', put_call='CALL')}))[0].df()['Value'].iloc[-1],
        errors='coerce')
    skew = (put_iv - call_iv) / 100.0 if pd.notna(put_iv) and pd.notna(call_iv) else 0.0

    avg_dollar_volume_item = bq.func.avg(
        bq.data.px_volume(dates=bq.func.range('-30D', '0D')) *
        bq.data.px_last(dates=bq.func.range('-30D', '0D')))
    avg_dollar_volume = pd.to_numeric(
        bq.execute(bql.Request(ticker, {'Value': avg_dollar_volume_item}))[0].df()['Value'].iloc[-1],
        errors='coerce')

    # Risk-free rate: 3M US T-Bill yield
    rfr = 0.0
    try:
        rfr = pd.to_numeric(
            bq.execute(bql.Request('USGG3M Index', {'Value': bq.data.px_last()}))[0].df()['Value'].iloc[-1],
            errors='coerce') / 100.0  # BQL retorna em %
        if pd.isna(rfr):
            rfr = 0.0
    except Exception:
        pass

    # MOVE Index (bond vol proxy) para Risk Parity
    move_idx = np.nan
    try:
        move_idx = pd.to_numeric(
            bq.execute(bql.Request('MOVE Index', {'Value': bq.data.px_last()}))[0].df()['Value'].iloc[-1],
            errors='coerce')
    except Exception:
        pass

    return {
        'spot': spot, 'iv_30d': iv_30d, 'rv_30d': rv_30d,
        'skew': skew, 'avg_dollar_volume': avg_dollar_volume,
        'risk_free_rate': rfr, 'move_index': move_idx,
    }


def fetch_options_chain(ticker, spot, min_dte, max_dte, mny_low, mny_high,
                        monthly_only=None):
    """Busca a cadeia de opções e retorna DataFrame processado.

    monthly_only: filtra só expiries mensais (3ª sexta, dia 15-21 do mês).
      None = auto (True quando ticker for ES1/ES futures, False caso contrário).
      Reduz drasticamente o tamanho da chain para futuros tipo ES1.
    """
    # Auto-detect: ES futures têm chain enorme → force monthly
    if monthly_only is None:
        monthly_only = any(k in ticker.upper() for k in ('ES1', 'ES2', 'ES '))

    from_strike = (1 + mny_low) * spot
    to_strike = (1 + mny_high) * spot

    conditions = (
        bq.data.expire_dt() >= f'{min_dte}d'
    ).and_(
        bq.data.expire_dt() <= f'{max_dte}d'
    ).and_(
        bq.data.strike_px() > from_strike
    ).and_(
        bq.data.strike_px() < to_strike
    ).and_(
        bq.data.open_int() > 0
    ).and_(
        bq.data.ivol() > 0
    )

    # monthly_only via BQL (expiration_periodicity) — mais correto que filtro por dia
    if monthly_only:
        conditions = conditions.and_(
            bq.data.expiration_periodicity() == 'monthly'
        )

    univ = bq.univ.filter(bq.univ.options([ticker]), conditions)

    items = {
        'Expire': bq.data.expire_dt(),
        'Strike': bq.data.strike_px(),
        'Type':   bq.data.put_call(),
        'IV':     bq.data.ivol(),
        'OI':     bq.data.open_int(),
    }
    resp = bq.execute(bql.Request(univ, items))
    data = pd.concat(
        [r.df()[r.name] for r in resp], axis=1
    )
    data = data.loc[:, ~data.columns.duplicated()]

    if data.empty:
        raise ValueError("Nenhum dado de opção encontrado para os filtros.")

    df = pd.DataFrame({
        'Exp':    pd.to_datetime(data['Expire']),
        'Strike': pd.to_numeric(data['Strike'], errors='coerce'),
        'Type':   data['Type'],
        'IV':     pd.to_numeric(data['IV'], errors='coerce'),
        'OI':     pd.to_numeric(data['OI'], errors='coerce'),
    })
    df.dropna(subset=['IV', 'OI', 'Strike'], inplace=True)
    df['IV'] /= 100.0

    today = np.datetime64(datetime.utcnow().date(), 'D')
    bus = np.busday_count(today, df.Exp.dt.normalize().values.astype('datetime64[D]'))
    df['Tte'] = np.maximum(bus, 1) / float(TRADING_DAYS)

    return df, from_strike, to_strike


def fetch_historical(ticker, period='-2Y'):
    """Busca preços históricos e retorna log-retornos."""
    hist_req = bql.Request(ticker, {'Value': bq.data.px_last(
        dates=bq.func.range(period, '0D'), fill='PREV')})
    prices = pd.to_numeric(
        _bql_ts(bq.execute(hist_req)[0], 'Value'), errors='coerce'
    ).dropna()
    if prices.empty:
        raise ValueError("Histórico de preços vazio.")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Retornos insuficientes.")
    return prices, log_returns
