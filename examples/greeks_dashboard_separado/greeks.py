"""Black-Scholes Greeks engine (vectorized)."""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


def calculate_all_greeks(S, K, vol, T, option_types, r=0.0):
    """
    Calcula todas as gregas de Black-Scholes de forma vetorial.

    Args:
        S: Preço spot do ativo subjacente (escalar).
        K: Array de strikes.
        vol: Array de volatilidade implícita (decimal, ex: 0.20 para 20%).
        T: Array de tempo até expiração em anos.
        option_types: Array de strings 'Call' ou 'Put'.
        r: Taxa livre de risco (default 0).

    Returns:
        Dict com arrays: delta, gamma, vega, vanna, zomma, theta, charm, speed.
        Cada valor é o greek por opção (não multiplicado por OI).
    """
    K = np.asarray(K, dtype=float)
    vol = np.asarray(vol, dtype=float)
    T = np.asarray(T, dtype=float)
    option_types = np.asarray(option_types)

    mask = (vol <= 0) | (T <= 0) | np.isnan(vol) | (K <= 0) | (S <= 0)
    greeks = {k: np.zeros_like(vol) for k in
              ['delta', 'gamma', 'vega', 'vanna', 'zomma', 'theta', 'charm', 'speed']}
    d1 = np.zeros_like(vol)

    with np.errstate(divide='ignore', invalid='ignore'):
        v = ~mask
        if not np.any(v):
            return greeks

        K_v, vol_v, T_v = K[v], vol[v], T[v]
        types_v = option_types[v]
        sqrt_T = np.sqrt(T_v)

        d1_v = (np.log(S / K_v) + (r + 0.5 * vol_v**2) * T_v) / (vol_v * sqrt_T)
        d2_v = d1_v - vol_v * sqrt_T

        pdf_d1 = norm.pdf(d1_v)
        cdf_d1 = norm.cdf(d1_v)
        cdf_d2 = norm.cdf(d2_v)

        delta_call = cdf_d1
        gamma_v = pdf_d1 / (S * vol_v * sqrt_T)

        # Delta
        greeks['delta'][v] = np.where(types_v == 'Call', delta_call, delta_call - 1)

        # Gamma
        greeks['gamma'][v] = gamma_v

        # Vega (por 1% de mudança na vol — divide por 100)
        greeks['vega'][v] = S * pdf_d1 * sqrt_T / 100.0

        # Vanna (∂Delta/∂Vol)
        greeks['vanna'][v] = -pdf_d1 * d2_v / vol_v

        # Zomma (∂Gamma/∂Vol)
        greeks['zomma'][v] = gamma_v * ((d1_v * d2_v - 1) / vol_v)

        # Theta (decaimento anualizado do preço da opção)
        exp_rT = np.exp(-r * T_v)
        theta_call = -(S * pdf_d1 * vol_v) / (2 * sqrt_T) - r * K_v * exp_rT * cdf_d2
        theta_put = -(S * pdf_d1 * vol_v) / (2 * sqrt_T) + r * K_v * exp_rT * (1 - cdf_d2)
        greeks['theta'][v] = np.where(types_v == 'Call', theta_call, theta_put)

        # Charm (∂Delta/∂T — decaimento anualizado do delta)
        charm_num = 2 * r * T_v - d2_v * vol_v * sqrt_T
        charm_den = 2 * T_v * vol_v * sqrt_T
        greeks['charm'][v] = -pdf_d1 * charm_num / charm_den

        # Speed (∂Gamma/∂S)
        greeks['speed'][v] = -gamma_v / S * (d1_v / (vol_v * sqrt_T) + 1)

    return greeks


def black_scholes_price_vec(S, K, vol, T, option_types, r=0.0):
    """
    Preço Black-Scholes vetorizado (europeu).
    Mesma fórmula d1/d2 de calculate_all_greeks — retorna array de preços.
    Opções com T <= 0 retornam valor intrínseco (max(S-K,0) ou max(K-S,0)).
    """
    K_a    = np.asarray(K,            dtype=float)
    vol_a  = np.asarray(vol,          dtype=float)
    T_a    = np.asarray(T,            dtype=float)
    types_a = np.asarray(option_types)
    prices = np.zeros(len(K_a))

    # opções expiradas → valor intrínseco
    expired = T_a <= 0
    if expired.any():
        prices[expired] = np.where(
            types_a[expired] == 'Call',
            np.maximum(S - K_a[expired], 0.0),
            np.maximum(K_a[expired] - S, 0.0))

    v = (vol_a > 0) & (T_a > 0) & (K_a > 0) & (S > 0) & ~np.isnan(vol_a)
    if not v.any():
        return prices
    with np.errstate(divide='ignore', invalid='ignore'):
        sq   = np.sqrt(T_a[v])
        d1   = (np.log(S / K_a[v]) + (r + 0.5 * vol_a[v] ** 2) * T_a[v]) / (vol_a[v] * sq)
        d2   = d1 - vol_a[v] * sq
        disc = np.exp(-r * T_a[v])
        c_px = S * norm.cdf(d1) - K_a[v] * disc * norm.cdf(d2)
        p_px = K_a[v] * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
        prices[v] = np.where(types_a[v] == 'Call', c_px, p_px)
    return prices


def calculate_flip(levels, curve):
    """Encontra o ponto onde a curva cruza zero (interpolação linear)."""
    sign_changes = np.where(np.diff(np.sign(curve)))[0]
    if sign_changes.size > 0:
        try:
            i = sign_changes[0]
            x1, y1 = levels[i], curve[i]
            x2, y2 = levels[i + 1], curve[i + 1]
            if (y2 - y1) == 0:
                return None
            return x1 - y1 * (x2 - x1) / (y2 - y1)
        except (IndexError, ZeroDivisionError):
            return None
    return None


def implied_move_pct(iv_annual, days=1):
    """Retorna o movimento implícito diário em % dado IV anualizada (decimal)."""
    return iv_annual * 100 * math.sqrt(days / TRADING_DAYS)


def fmt_value(value, decimals=2):
    """Formata valor grande em Bi/Mi/K."""
    if pd.isna(value):
        return "N/A"
    a = abs(value)
    if a >= 1e9:
        return f"{value / 1e9:,.{decimals}f} Bi"
    if a >= 1e6:
        return f"{value / 1e6:,.{decimals}f} Mi"
    if a >= 1e3:
        return f"{value / 1e3:,.{decimals}f} K"
    return f"{value:,.{decimals}f}"
