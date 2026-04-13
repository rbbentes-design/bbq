"""Exposure calculations, model curves, walls, risk models (VaR, Monte Carlo, P&L)."""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

try:
    from .config import GREEK_CONFIGS, FUTURES_MULTIPLIER
    from .greeks import calculate_all_greeks, black_scholes_price_vec
except ImportError:
    import sys, os as _os
    _dir = _os.path.dirname(_os.path.abspath(__file__)) if '__file__' in dir() else _os.getcwd()
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from config import GREEK_CONFIGS, FUTURES_MULTIPLIER
    from greeks import calculate_all_greeks, black_scholes_price_vec


def compute_strike_exposures(df, greeks, spot):
    """
    Adiciona colunas de exposição por tipo (Call/Put) ao DataFrame
    e retorna DataFrame agregado por strike.
    """
    is_call = (df['Type'] == 'Call').values
    is_put = (df['Type'] == 'Put').values
    oi_100 = df['OI'].values * 100.0

    for cfg in GREEK_CONFIGS:
        key = cfg['key']
        raw = greeks[key]
        df[f'Call_{key}'] = np.where(is_call, raw * oi_100, 0.0)
        df[f'Put_{key}']  = np.where(is_put,  raw * oi_100, 0.0)

    # OI bruto por tipo — necessário para identificar Call/Put Wall por concentração de OI
    df['Call_OI'] = np.where(is_call, df['OI'].values, 0.0)
    df['Put_OI']  = np.where(is_put,  df['OI'].values, 0.0)

    # Agregar por strike — exposições de gregas + OI bruto por tipo
    exp_cols = []
    for cfg in GREEK_CONFIGS:
        exp_cols += [f'Call_{cfg["key"]}', f'Put_{cfg["key"]}']
    exp_cols += ['Call_OI', 'Put_OI']
    agg = df.groupby('Strike')[exp_cols].sum()

    # Computar totais líquidos
    for cfg in GREEK_CONFIGS:
        key = cfg['key']
        scale_val = cfg['scale'](spot)
        call_scaled = agg[f'Call_{key}'] * scale_val
        put_scaled = agg[f'Put_{key}'] * scale_val
        agg[f'Total_{key}'] = cfg['op'](call_scaled, put_scaled) / cfg['div']

    return agg


def compute_model_curves(df, levels, configs=None, r=0.0):
    """
    Calcula curvas modelo (exposure vs. preço spot) para todas as gregas
    em uma única passagem por nível de preço.

    Retorna dict: {greek_name: np.array com a curva}.
    """
    if configs is None:
        configs = GREEK_CONFIGS

    results = {c['name']: [] for c in configs}
    strikes = df['Strike'].values
    ivs = df['IV'].values
    ttes = df['Tte'].values
    ois = df['OI'].values * 100.0
    types = df['Type'].values
    is_call = types == 'Call'
    is_put = types == 'Put'

    for L in levels:
        greeks = calculate_all_greeks(L, strikes, ivs, ttes, types, r=r)
        for cfg in configs:
            key = cfg['key']
            raw = greeks[key]
            call_exp = np.nansum(raw[is_call] * ois[is_call])
            put_exp = np.nansum(raw[is_put] * ois[is_put])
            total = cfg['op'](call_exp, put_exp) * cfg['scale'](L)
            results[cfg['name']].append(total)

    return {name: np.array(curve) for name, curve in results.items()}


def compute_walls(agg):
    """
    Identifica Call Wall e Put Wall pelo maior Open Interest por strike.
    GEX (gamma) é usado apenas como critério secundário em caso de empate
    ou ambiguidade entre strikes com OI muito próximo (dentro de 2% do máximo).
    """
    def _wall(oi_col, gamma_col):
        # Fallback para gamma se OI não estiver disponível
        if oi_col not in agg.columns or agg[oi_col].max() <= 0:
            return agg[gamma_col].idxmax() if gamma_col in agg.columns else None
        max_oi = agg[oi_col].max()
        # Zona de ambiguidade: strikes dentro de 2% do OI máximo
        candidates = agg[agg[oi_col] >= max_oi * 0.98]
        if len(candidates) == 1 or gamma_col not in agg.columns:
            return int(candidates[oi_col].idxmax())
        # Tiebreaker: maior GEX entre os candidatos empatados
        return int(candidates[gamma_col].idxmax())

    call_wall = _wall('Call_OI', 'Call_gamma')
    put_wall  = _wall('Put_OI',  'Put_gamma')
    return call_wall, put_wall


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — MODELOS DE RISCO (VaR, Monte Carlo, P&L)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_risk_model(log_returns):
    """Ajusta distribuição t-Student e calcula VaR/CVaR a 95% e 99%."""
    arr = np.asarray(log_returns, dtype=float)
    tdf, tloc, tscale = student_t.fit(arr)
    print(f"[RISK] t-Student fit: df={tdf:.2f}, loc={tloc:.6f}, scale={tscale:.6f}, n={len(arr)}")

    # Sanity check: se tscale for degenerado, usa parâmetros empíricos
    if tscale < 1e-6 or not np.isfinite(tscale):
        print(f"[RISK] ⚠️ tscale degenerado ({tscale}), usando fallback empírico")
        tloc = float(np.mean(arr))
        tscale = float(np.std(arr))
        tdf = 4.0  # fat-tail conservador

    var_95 = student_t.ppf(0.05, tdf, tloc, tscale)
    cvar_95 = student_t.expect(
        lambda x: x, args=(tdf,), loc=tloc, scale=tscale,
        lb=-np.inf, ub=var_95) / 0.05
    var_99 = student_t.ppf(0.01, tdf, tloc, tscale)
    cvar_99 = student_t.expect(
        lambda x: x, args=(tdf,), loc=tloc, scale=tscale,
        lb=-np.inf, ub=var_99) / 0.01
    print(f"[RISK] VaR 95%={var_95:.4f} ({var_95:.2%}), VaR 99%={var_99:.4f} ({var_99:.2%})")
    return {'tdf': tdf, 'tloc': tloc, 'tscale': tscale,
            'var_95': var_95, 'cvar_95': cvar_95,
            'var_99': var_99, 'cvar_99': cvar_99}


def run_monte_carlo(spot, df, risk_params, n_sims=10000, n_days=5, r=0.0):
    """Simula P&L acumulado do livro do market maker ao longo de n_days."""
    greeks = calculate_all_greeks(spot, df.Strike.values, df.IV.values,
                                  df.Tte.values, df.Type.values, r=r)
    oi100 = df.OI.values * 100
    call_sign = np.where(df.Type.values == 'Call', 1, -1)
    dex = (greeks['delta'] * oi100).sum()
    gex_per_pt = (greeks['gamma'] * call_sign * oi100).sum()
    theta_tot = (greeks['theta'] * oi100).sum()

    cum_pnl = np.zeros(n_sims)
    cur_spot = np.full(n_sims, spot)
    for _ in range(n_days):
        day_rets = student_t.rvs(risk_params['tdf'],
                                 loc=risk_params['tloc'],
                                 scale=risk_params['tscale'], size=n_sims)
        new_spot = cur_spot * (1 + day_rets)
        ds = new_spot - cur_spot
        daily_pnl = -(dex * ds + 0.5 * gex_per_pt * ds ** 2) + theta_tot
        cum_pnl += daily_pnl
        cur_spot = new_spot
    return cum_pnl, cur_spot


def compute_pnl_curves(greeks_now, df, spot, levels, skew, r=0.0):
    """
    Calcula curvas de P&L comparativas:
    - Simplificada (Delta + Gamma)
    - Completa (Delta + Gamma + Vega + Vanna + Zomma)
    - Market vs. Dealer
    """
    oi = df['OI'].values
    delta_vals = greeks_now['delta']
    gamma_vals = greeks_now['gamma']
    vega_vals = greeks_now['vega']
    vanna_vals = greeks_now['vanna']
    zomma_vals = greeks_now['zomma']

    pnl_simple = []
    pnl_complete = []
    market_pnl = []
    hedge_demand = []

    for s in levels:
        dS = s - spot
        dVol = -np.sign(dS) * abs(skew) if skew != 0 else 0

        # Simplificado
        p_s = np.nansum((delta_vals * dS + 0.5 * gamma_vals * dS**2) * oi * 100)

        # Completo (com efeito Zomma no gamma)
        gamma_adj = gamma_vals + zomma_vals * dVol
        p_c = np.nansum(
            (delta_vals * dS
             + 0.5 * gamma_adj * dS**2
             + vega_vals * (dVol * 100)
             + vanna_vals * dS * (dVol * 100)) * oi * 100)

        pnl_simple.append(p_s)
        pnl_complete.append(p_c)
        market_pnl.append(p_s)

        # Demanda de hedge (contratos futuros para ficar delta-neutro)
        greeks_at_s = calculate_all_greeks(s, df.Strike.values, df.IV.values,
                                           df.Tte.values, df.Type.values, r=r)
        mkt_delta = np.nansum(greeks_at_s['delta'] * oi * 100)
        hedge_demand.append(mkt_delta / FUTURES_MULTIPLIER)

    return {
        'simple': np.array(pnl_simple),
        'complete': np.array(pnl_complete),
        'market': np.array(market_pnl),
        'dealer': -np.array(market_pnl),
        'hedge_demand': np.array(hedge_demand),
    }
