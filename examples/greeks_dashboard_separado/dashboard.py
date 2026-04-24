"""Dashboard assembly: overview, exposure charts, main callback, UI widgets."""

import numpy as np
import pandas as pd
import math
import os
import json
import traceback
import warnings
import base64 as _b64
import io
import zipfile
from datetime import datetime, timedelta
from functools import lru_cache
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import ipywidgets as wd
from IPython.display import display, clear_output, HTML

try:
    from .config import *
    from .ui import _hud_panel, _svg_ring_html, create_gauge, create_symmetric_gauge
    from .greeks import calculate_all_greeks, black_scholes_price_vec, calculate_flip, implied_move_pct, fmt_value
    from .data import fetch_market_data, fetch_options_chain, fetch_historical, _bql_ts
    from .exposure import compute_strike_exposures, compute_model_curves, compute_walls, fit_risk_model, run_monte_carlo, compute_pnl_curves
    from .flow_charts import *
    from . import flows
    from . import flow_charts
    from . import dispersion
    from . import charts
    from .flows import *
    from .charts import *
    from .dispersion import *
except ImportError:
    import sys, os as _os
    _dir = _os.path.dirname(_os.path.abspath(__file__)) if '__file__' in dir() else _os.getcwd()
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from config import *
    from ui import _hud_panel, _svg_ring_html, create_gauge, create_symmetric_gauge
    from greeks import calculate_all_greeks, black_scholes_price_vec, calculate_flip, implied_move_pct, fmt_value
    from data import fetch_market_data, fetch_options_chain, fetch_historical, _bql_ts
    from exposure import compute_strike_exposures, compute_model_curves, compute_walls, fit_risk_model, run_monte_carlo, compute_pnl_curves
    from flow_charts import *
    import flows
    import flow_charts
    import dispersion
    import charts
    from flows import *
    from charts import *
    from dispersion import *


def build_greek_overview(greeks_now, df, spot, etf_flows=None):
    """
    Seção de termômetros das gregas + fluxo por ação (Mag8) para a Visão Geral.
    Gregas em $Bn usando a mesma escala das abas de exposição (GREEK_CONFIGS).
    etf_flows: dict de DataFrames com Flow_$ e PctADV por ação (compute_full_etf_flows).
    """
    oi_100  = df['OI'].values * 100.0
    is_call = df['Type'].values == 'Call'
    is_put  = df['Type'].values == 'Put'

    # ── Gregas em $Bn — mesma escala do GREEK_CONFIGS ─────────────
    # Delta:  op=add,      scale=L (spot)
    delta_bn = float(np.nansum(greeks_now['delta'] * oi_100) * spot / 1e9)
    # Gamma:  op=subtract, scale=L²×0.01
    gamma_bn = float((np.nansum(greeks_now['gamma'][is_call] * oi_100[is_call]) -
                      np.nansum(greeks_now['gamma'][is_put]  * oi_100[is_put])) * spot**2 * 0.01 / 1e9)
    # Vanna:  op=subtract, scale=1  (sem spot)
    vanna_bn = float((np.nansum(greeks_now['vanna'][is_call] * oi_100[is_call]) -
                      np.nansum(greeks_now['vanna'][is_put]  * oi_100[is_put])) / 1e9)
    # Charm:  op=add,      scale=L/365
    charm_bn = float(np.nansum(greeks_now['charm'] * oi_100) * spot / 365.0 / 1e9)

    # Cache module-level para exportação JARVIS (div/10 = escala BBG)
    _greek_cache['delta_bn'] = delta_bn / 10
    _greek_cache['vanna_bn'] = vanna_bn
    _greek_cache['charm_bn'] = charm_bn / 10

    # Escala dinâmica mínima por grega (SPX típico)
    g_delta = create_symmetric_gauge(delta_bn, 'Δ Delta Nocional',  max(5.0,  abs(delta_bn) * 1.5))
    g_gamma = create_symmetric_gauge(gamma_bn, 'Γ Gamma (GEX Net)', max(0.5,  abs(gamma_bn) * 1.5))
    g_vanna = create_symmetric_gauge(vanna_bn, 'V Vanna',           max(2.0,  abs(vanna_bn) * 1.5))
    g_charm = create_symmetric_gauge(charm_bn, 'C Charm (diário)',  max(0.5,  abs(charm_bn) * 1.5))

    # ── Interpretação textual ──────────────────────────────────────
    def _badge(positive, txt_pos, txt_neg, val, thr=0.1):
        if abs(val) < thr:
            return f"<span style='color:#8b949e;'>⚪ Neutro</span>"
        color = _C['green'] if (val > 0) == positive else _C['red']
        txt   = txt_pos if val > 0 else txt_neg
        return f"<span style='color:{color};'>{txt}</span>"

    interp_html = (
        f"<div class='mm-dash'><div class='mm-card' style='padding:10px 16px;min-width:280px;'>"
        f"<div class='mm-section-label' style='margin-top:0;'>Leitura das Gregas</div>"
        f"<p style='margin:4px 0;font-size:12px;'><b>Δ Delta:</b> "
        + _badge(False, '🔴 Dealers comprados → venda', '🟢 Dealers vendidos → compra', delta_bn, 0.5)
        + f"</p><p style='margin:4px 0;font-size:12px;'><b>Γ Gamma:</b> "
        + _badge(True, '🟢 GEX+ estabiliza mercado', '🔴 GEX− acelera movimentos', gamma_bn, 0.3)
        + f"</p><p style='margin:4px 0;font-size:12px;'><b>V Vanna:</b> "
        + _badge(False, '🔴 Vol↑ → dealers vendem', '🟢 Vol↑ → dealers compram', vanna_bn, 0.2)
        + f"</p><p style='margin:4px 0;font-size:12px;'><b>C Charm:</b> "
        + _badge(False, '🔴 Delta decai → desfaz hedge', '🟢 Delta decai → reforça hedge', charm_bn, 0.05)
        + "</p></div></div>"
    )

    row_gauges = wd.HBox(
        [g_delta, g_gamma, g_vanna, g_charm, wd.HTML(interp_html)],
        layout={'justify_content': 'flex-start', 'align_items': 'center', 'flex_wrap': 'wrap'})

    # ── Fluxo por ação — dados reais do ETF rebalanceamento ────────
    combo = pd.DataFrame()
    if etf_flows and 'Combined' in etf_flows:
        combo = etf_flows['Combined'].copy()
        # Normalizar ticker: 'AAPL US Equity' → 'AAPL'
        combo.index = combo.index.str.split().str[0]

    def _stock_bar(name, val_bn, pct_adv, flow_max_bn):
        pct_bar = min(abs(val_bn) / flow_max_bn * 100, 100) if flow_max_bn > 0 else 0
        color   = _C['green'] if val_bn >= 0 else _C['red']
        arrow   = '▲' if val_bn >= 0 else '▼'
        adv_str = f' | {pct_adv:.1f}% ADV' if pd.notna(pct_adv) else ''
        return (
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
            f"<span style='font-size:12px;font-weight:700;color:{_C['text']};width:48px;'>{name}</span>"
            f"<div style='flex:1;background:{_C['card2']};border-radius:3px;height:14px;'>"
            f"<div style='width:{pct_bar:.0f}%;height:100%;background:{color};"
            f"border-radius:3px;opacity:0.75;'></div></div>"
            f"<span style='font-size:11px;color:{color};min-width:110px;text-align:right;'>"
            f"{arrow} ${val_bn:.2f}Bn{adv_str}</span>"
            f"</div>"
        )

    if not combo.empty and 'Flow_$' in combo.columns:
        combo['Flow_Bn'] = combo['Flow_$'] / 1e9
        buys_df  = combo[combo['Flow_Bn'] >= 0].nlargest(4, 'Flow_Bn')
        sells_df = combo[combo['Flow_Bn'] <  0].nsmallest(4, 'Flow_Bn')
        flow_max = max(combo['Flow_Bn'].abs().max(), 1.0)

        buy_rows  = ''.join(_stock_bar(s, row['Flow_Bn'],
                            row.get('PctADV', np.nan), flow_max)
                            for s, row in buys_df.iterrows())
        sell_rows = ''.join(_stock_bar(s, row['Flow_Bn'],
                            row.get('PctADV', np.nan), flow_max)
                            for s, row in sells_df.iterrows())
        data_note = 'Dados reais do rebalanceamento de ETFs passivos (VOO/SPY/IVV).'

        # ── Overhang — mesmos dados das barras, top 2 por %ADV ───
        # Usa buys_df/sells_df (já filtrados e normalizados) para
        # garantir que os valores batam exatamente com as barras acima.
        overhang_section = wd.HTML('')
        if 'PctADV' in combo.columns:
            # Limpa inf/nan nas colunas PctADV das mesmas tabelas usadas acima
            buys_adv  = buys_df[buys_df['PctADV'].replace([np.inf,-np.inf],np.nan).notna()].copy()
            sells_adv = sells_df[sells_df['PctADV'].replace([np.inf,-np.inf],np.nan).notna()].copy()

            top2_buy  = buys_adv.nlargest(2, 'PctADV')   # maior %ADV positivo
            top2_sell = sells_adv.nsmallest(2, 'PctADV') # menor %ADV (mais negativo)

            all_adv_vals = (list(top2_buy['PctADV']) + list(top2_sell['PctADV']))
            adv_scale = max((abs(v) for v in all_adv_vals if pd.notna(v)), default=5.0) * 1.15

            overhang_gauges = []
            for tkr, row in list(top2_buy.iterrows()) + list(top2_sell.iterrows()):
                pct = float(row['PctADV'])
                overhang_gauges.append(
                    create_symmetric_gauge(
                        round(pct, 2), tkr,
                        adv_scale, unit='% ADV',
                        width=200, height=178))

            overhang_row   = wd.HBox(overhang_gauges,
                                     layout={'justify_content': 'flex-start',
                                             'flex_wrap': 'wrap'})
            overhang_title = wd.HTML(
                f"<div class='mm-section-label' style='margin:10px 0 4px;padding:0 8px;'>"
                f"Overhang — Impacto do Rebalanceamento (% ADV)</div>"
                f"<div style='font-size:11px;color:{_C['text_dim']};padding:0 8px 6px;'>"
                f"Top 2 maiores compra e Top 2 maiores venda — "
                f"quanto o flow representa do volume médio diário do ativo (5d ADV). "
                f"Mesmo dado das barras acima (BQL).</div>")
            overhang_section = wd.VBox([overhang_title, overhang_row])
    else:
        # Fallback sem dados ETF
        buy_rows = sell_rows = "<p style='color:#8b949e;font-size:12px;'>Dados ETF não disponíveis.</p>"
        data_note = 'Execute com ETF rebalancing ativo para ver dados reais.'
        overhang_section = wd.HTML('')

    flow_html = (
        f"<div class='mm-dash'><div class='mm-card' style='padding:12px 16px;'>"
        f"<div class='mm-section-label' style='margin:0 0 8px;'>"
        f"Rebalanceamento ETF Passivo (VOO / SPY / IVV) — Top 4 Compra + Top 4 Venda</div>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;'>"
        f"<div>"
        f"<div style='font-size:11px;font-weight:700;color:{_C['green']};margin-bottom:6px;"
        f"text-transform:uppercase;letter-spacing:0.8px;'>▲ Fluxo de Compra</div>"
        f"{buy_rows}</div>"
        f"<div>"
        f"<div style='font-size:11px;font-weight:700;color:{_C['red']};margin-bottom:6px;"
        f"text-transform:uppercase;letter-spacing:0.8px;'>▼ Fluxo de Venda</div>"
        f"{sell_rows}</div>"
        f"</div>"
        f"<p style='font-size:10px;color:{_C['text_dim']};margin:8px 0 0;'>{data_note}</p>"
        f"</div></div>"
    )

    return wd.VBox([
        row_gauges,
        wd.HBox([wd.HTML(flow_html), overhang_section],
                layout={'align_items': 'flex-start', 'flex_wrap': 'wrap'}),
    ])


def plot_exposure_charts(agg, df, spot, from_strike, to_strike,
                         levels, model_curves, flip_points,
                         call_wall, put_wall):
    """Gera todos os gráficos de exposição (Combo + Absolute + Model) para cada grega."""
    strikes = agg.index.values
    step = np.median(np.diff(np.sort(strikes))) if len(strikes) > 1 else 5
    bar_w = step * 0.8

    for cfg in GREEK_CONFIGS:
        name = cfg['name']
        key = cfg['key']
        unit = cfg['unit']
        div = cfg['div']
        scale_val = cfg['scale'](spot)
        combo_op = cfg['op']
        flip = flip_points.get(name)

        call_col = f'Call_{key}'
        put_col = f'Put_{key}'
        call_data = agg[call_col] * scale_val
        put_data = agg[put_col] * scale_val
        combo_data = combo_op(call_data, put_data) / div
        total_val = combo_data.sum()

        # ── Gráfico COMBO ──
        plt.figure(figsize=(18, 5))
        colors = [_C['green'] if v >= 0 else _C['red'] for v in combo_data.values]
        plt.bar(strikes, combo_data, width=bar_w, color=colors, edgecolor='k', lw=0.3)
        plt.axvline(spot, color='r', ls='--', lw=1.5, label=f'Spot {spot:,.0f}')
        if name == 'Gamma':
            if call_wall:
                plt.axvline(call_wall, color='g', ls=':', lw=2, label=f'Call Wall {call_wall:,.0f}')
            if put_wall:
                plt.axvline(put_wall, color='b', ls=':', lw=2, label=f'Put Wall {put_wall:,.0f}')
        if flip:
            plt.axvline(flip, color='orange', ls='--', lw=2.5, label=f'{name} Flip {flip:,.0f}')
        plt.title(f'COMBO {name.upper()} EXPOSURE = {total_val:,.2f} ({unit})',
                  fontsize=16, weight='bold')
        plt.ylabel(f'Exposure ({unit})', fontsize=12)
        plt.xlim(from_strike, to_strike)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ── Gráfico ABSOLUTE ──
        put_sign = 1 if combo_op is np.add else -1
        plt.figure(figsize=(18, 5))
        plt.bar(strikes, call_data / div, width=bar_w, color=_C['green'],
                edgecolor='k', lw=0.3, label=f'Call {name}')
        plt.bar(strikes, put_sign * put_data / div, width=bar_w, color=_C['red'],
                edgecolor='k', lw=0.3, label=f'Put {name}')
        plt.axvline(spot, color='r', ls='--', lw=1.5, label=f'Spot {spot:,.0f}')
        if name == 'Gamma':
            if call_wall:
                plt.axvline(call_wall, color='g', ls=':', lw=2, label=f'Call Wall {call_wall:,.0f}')
            if put_wall:
                plt.axvline(put_wall, color='b', ls=':', lw=2, label=f'Put Wall {put_wall:,.0f}')
        if flip:
            plt.axvline(flip, color='orange', ls='--', lw=2.5, label=f'{name} Flip {flip:,.0f}')
        plt.title(f'ABSOLUTE {name.upper()} EXPOSURE', fontsize=16, weight='bold')
        plt.ylabel(f'Exposure ({unit})', fontsize=12)
        plt.xlim(from_strike, to_strike)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ── Gráfico MODEL ──
        curve = model_curves[name] / div
        current_val = curve[np.argmin(np.abs(levels - spot))]
        plt.figure(figsize=(18, 5))
        plt.plot(levels, curve, lw=2, color=_C['accent'])
        plt.fill_between(levels, 0, curve, where=curve >= 0,
                         alpha=0.15, color='green', interpolate=True)
        plt.fill_between(levels, 0, curve, where=curve <= 0,
                         alpha=0.15, color='red', interpolate=True)
        plt.axvline(spot, color='r', ls='--', lw=1.5, label=f'Spot {spot:,.0f}')
        if flip:
            plt.axvline(flip, color='orange', ls='--', lw=2.5, label=f'{name} Flip {flip:,.0f}')
        plt.axhline(0, color='k', lw=0.5)
        plt.title(f'{name.upper()} EXPOSURE MODEL = {current_val:,.2f} ({unit})',
                  fontsize=16, weight='bold')
        plt.xlabel('Preço do Ativo', fontsize=12)
        plt.ylabel(f'Exposure ({unit})', fontsize=12)
        plt.xlim(from_strike, to_strike)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def style_sensitivity_matrix(matrix_df, cmap='viridis'):
    """Aplica estilo visual a uma matriz de sensibilidade."""
    styled = matrix_df.style.format(fmt_value)
    styled = styled.set_properties(**{
        'width': '110px', 'text-align': 'center', 'font-size': '12px',
        'padding': '6px 8px', 'border': f'1px solid {_C["border_light"]}',
        'color': _C['text'], 'background-color': _C['card'],
    })
    styled = styled.set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', _C['card2']), ('color', _C['text_muted']),
            ('font-size', '11px'), ('text-transform', 'uppercase'),
            ('letter-spacing', '0.5px'), ('padding', '8px'),
            ('border', f'1px solid {_C["border"]}')]},
    ])
    numeric = matrix_df.select_dtypes(include=np.number).dropna(how='all')
    if not numeric.empty and numeric.max().max() != numeric.min().min():
        styled = styled.background_gradient(cmap=cmap, axis=None)
    return styled.to_html()


_JARVIS_EXPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>J.A.R.V.I.S — Intelligence Core</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#020810;
  --c:#00d4e8;              /* cyan — primary/neutral */
  --c60:rgba(0,212,232,.88);
  --c40:rgba(0,212,232,.72);
  --c20:rgba(0,212,232,.52);
  --c08:rgba(0,212,232,.16);
  --c04:rgba(0,212,232,.08);
  --brd:rgba(0,212,232,.25);
  --dim:#041420;
  --txt:rgba(0,212,232,.92);
  --lbl:rgba(0,212,232,.62);
  --a:#f5a623;              /* amber — warning/caution */
  --a60:rgba(245,166,35,.88);
  --a40:rgba(245,166,35,.72);
  --a20:rgba(245,166,35,.5);
  --a08:rgba(245,166,35,.16);
  --r:#f85149;              /* red — danger/bearish */
  --r60:rgba(248,81,73,.88);
  --r40:rgba(248,81,73,.72);
  --r20:rgba(248,81,73,.5);
  --r08:rgba(248,81,73,.16);
}
html,body{width:100%;height:100vh;overflow:hidden;background:var(--bg);
  font-family:'Share Tech Mono',monospace;color:var(--txt)}
canvas#bg{position:fixed;inset:0;z-index:0;opacity:.5}
.scl{position:fixed;inset:0;z-index:2;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.04) 3px,rgba(0,0,0,.04) 4px)}

/* ── BOOT ── */
#boot{position:fixed;inset:0;z-index:999;background:var(--bg);
  display:flex;flex-direction:column;align-items:center;justify-content:center;transition:opacity .7s}
#boot.gone{opacity:0;pointer-events:none}
.brw{position:relative;width:68px;height:68px;display:flex;align-items:center;justify-content:center;margin-bottom:20px}
.bri{position:absolute;border-radius:50%;border:1px solid transparent}
.bri1{inset:0;border-top-color:var(--c);border-right-color:var(--c);animation:sp 1s linear infinite}
.bri2{inset:10px;border-bottom-color:var(--c40);border-left-color:var(--c40);animation:sp 1.8s linear reverse infinite}
.bri3{inset:20px;border-top-color:var(--c20);animation:sp 3s linear infinite}
.bcore{width:8px;height:8px;border-radius:50%;background:var(--c);box-shadow:0 0 10px var(--c),0 0 20px var(--c)}
.btitle{font-family:'Orbitron',sans-serif;font-size:9px;letter-spacing:6px;color:var(--c);margin-bottom:14px}
#blog{font-size:9px;color:var(--c40);line-height:2.2;text-align:center;min-height:60px}
@keyframes sp{to{transform:rotate(360deg)}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.1}}

/* ── APP ── */
#app{position:relative;z-index:10;width:100vw;height:100vh;
  display:flex;flex-direction:column;opacity:0;transition:opacity .8s}
#app.on{opacity:1}

/* ── CMD STRIP ── */
.cmd{display:flex;align-items:center;gap:8px;padding:6px 24px;
  background:rgba(0,4,10,.98);border-bottom:1px solid var(--c08);flex-shrink:0}
.cmd-id{font-family:'Orbitron',sans-serif;font-size:10px;font-weight:700;letter-spacing:3px;
  color:var(--c);display:flex;align-items:center;gap:6px;white-space:nowrap;margin-right:6px}
.cmd-id::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--c);
  box-shadow:0 0 8px var(--c);animation:blink 1.4s ease-in-out infinite;flex-shrink:0}
.dvd{width:1px;height:18px;background:var(--c08);flex-shrink:0;margin:0 6px}
.cs{display:flex;flex-direction:column;gap:1px;flex-shrink:0}
.csl{font-size:9px;letter-spacing:2px;color:rgba(0,200,220,.65)}
.csv{font-family:'Orbitron',sans-serif;font-size:13px;font-weight:700;color:var(--c)}

/* ── HEADER ── */
.hdr{display:flex;align-items:center;padding:8px 24px;gap:16px;
  background:linear-gradient(180deg,rgba(0,8,18,.98),rgba(0,4,10,.9));
  border-bottom:1px solid var(--c08);flex-shrink:0;position:relative}
.hdr::after{content:'';position:absolute;bottom:0;left:0;width:15%;height:1px;
  background:linear-gradient(90deg,transparent,var(--c60),transparent);animation:hln 7s linear infinite}
@keyframes hln{0%{left:-15%}100%{left:115%}}
.reactor{width:30px;height:30px;position:relative;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.ri{position:absolute;border-radius:50%;border:1px solid transparent}
.ri1{inset:0;border-top-color:var(--c);border-right-color:var(--c);animation:sp 2s linear infinite}
.ri2{inset:5px;border-bottom-color:var(--c40);border-left-color:var(--c40);animation:sp 3.5s linear reverse infinite}
.ri3{inset:11px;border-top-color:var(--c20);animation:sp 5s linear infinite}
.rcore{width:6px;height:6px;border-radius:50%;background:var(--c);
  box-shadow:0 0 6px var(--c),0 0 14px var(--c);animation:rpc 2.2s ease-in-out infinite}
@keyframes rpc{0%,100%{box-shadow:0 0 6px var(--c),0 0 12px var(--c)}
  50%{box-shadow:0 0 10px var(--c),0 0 22px var(--c),0 0 36px var(--c20)}}
.brand{flex-shrink:0}
.bn{font-family:'Orbitron',sans-serif;font-size:18px;font-weight:900;letter-spacing:5px;
  color:var(--c);text-shadow:0 0 18px var(--c20)}
.bs{font-size:9px;color:var(--lbl);letter-spacing:3px;margin-top:2px}
.tabs{display:flex;align-items:stretch;gap:0;flex:1;justify-content:center}
.tb{font-family:'Orbitron',sans-serif;font-size:11px;font-weight:600;letter-spacing:3px;
  color:var(--lbl);padding:0 20px;cursor:pointer;border:none;background:none;
  border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap}
.tb:hover{color:var(--c40)}
.tb.act{color:var(--c);border-bottom-color:var(--c)}
.clkw{text-align:right;flex-shrink:0}
.cll{font-size:9px;color:var(--lbl);letter-spacing:2px}
.clv{font-family:'Orbitron',sans-serif;font-size:15px;font-weight:700;color:var(--c)}

/* ── CONTENT ── */
.content{flex:1;overflow-y:auto;overflow-x:hidden;min-height:0;
  scrollbar-width:thin;scrollbar-color:var(--c08) transparent}
.content::-webkit-scrollbar{width:4px}
.content::-webkit-scrollbar-thumb{background:var(--c20)}
.tp{display:none;flex-direction:column;gap:16px;padding:20px 28px}
.tp.act{display:flex}

/* ── PANEL ── */
.p{background:linear-gradient(145deg,rgba(0,10,20,.98),rgba(0,4,10,.99));
  border:1px solid var(--brd);position:relative;
  clip-path:polygon(0 0,calc(100% - 12px) 0,100% 12px,100% 100%,12px 100%,0 calc(100% - 12px))}
.p::before,.p::after,.p .cb,.p .ct{content:'';position:absolute;width:9px;height:9px;pointer-events:none}
.p::before{top:-1px;left:-1px;border-top:1px solid var(--c40);border-left:1px solid var(--c40)}
.p::after{bottom:-1px;right:-1px;border-bottom:1px solid var(--c40);border-right:1px solid var(--c40)}
.p .cb{bottom:-1px;left:-1px;border-bottom:1px solid var(--c40);border-left:1px solid var(--c40)}
.p .ct{top:-1px;right:-1px;border-top:1px solid var(--c40);border-right:1px solid var(--c40)}
.ph{font-family:'Orbitron',sans-serif;font-size:11px;font-weight:700;letter-spacing:3px;
  color:var(--c);padding-bottom:10px;border-bottom:1px solid var(--c08);margin-bottom:14px;
  display:flex;align-items:center;gap:9px}
.phd{width:5px;height:5px;background:var(--c);box-shadow:0 0 7px var(--c);flex-shrink:0}

/* ── ARC GAUGE ── */
.gc{clip-path:polygon(0 0,calc(100% - 8px) 0,100% 8px,100% 100%,8px 100%,0 calc(100% - 8px));
  background:linear-gradient(145deg,rgba(0,10,20,.98),rgba(0,4,10,.99));
  border:1px solid var(--brd);padding:14px 10px 12px;
  display:flex;flex-direction:column;align-items:center}
.gw{position:relative}
.gv{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.gn{font-family:'Orbitron',sans-serif;font-weight:900;line-height:1}
.gst{font-size:10px;letter-spacing:2px;margin-top:4px}
.gl{font-family:'Orbitron',sans-serif;font-size:10px;font-weight:600;letter-spacing:2px;
  text-transform:uppercase;margin-top:6px;color:var(--lbl);text-align:center;line-height:1.4}
.gmm{display:flex;justify-content:space-between;font-size:9px;color:var(--c08);margin-top:2px}

/* ── SEMI GAUGE ── */
.sc{clip-path:polygon(0 0,calc(100% - 7px) 0,100% 7px,100% 100%,0 100%);
  background:linear-gradient(145deg,rgba(0,10,20,.98),rgba(0,4,10,.99));
  border:1px solid var(--brd);padding:12px 12px 14px;
  display:flex;flex-direction:column;align-items:center;gap:4px}
.sl2{font-family:'Orbitron',sans-serif;font-size:10px;letter-spacing:3px;color:var(--lbl);text-align:center}
.sv{font-family:'Orbitron',sans-serif;font-size:16px;font-weight:700;text-align:center;line-height:1.2;margin-top:4px}

/* ── CARD ── */
.card{clip-path:polygon(0 0,calc(100% - 6px) 0,100% 6px,100% 100%,0 100%);
  background:linear-gradient(145deg,rgba(0,10,20,.98),rgba(0,4,10,.99));
  border:1px solid var(--brd);padding:13px 16px;display:flex;flex-direction:column;gap:5px}
.cdl{font-size:10px;letter-spacing:3px;color:var(--lbl);text-transform:uppercase}
.cdv{font-family:'Orbitron',sans-serif;font-size:22px;font-weight:700;line-height:1.1;color:var(--c)}
.secl{font-family:'Orbitron',sans-serif;font-size:10px;letter-spacing:4px;color:var(--lbl);
  padding:4px 0 3px;display:flex;align-items:center;gap:8px}
.secl::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--c08),transparent)}

/* ── RISK TABLE ── */
.rt{width:100%;border-collapse:collapse}
.rt th{font-family:'Orbitron',sans-serif;font-size:10px;letter-spacing:2px;color:var(--lbl);
  padding:5px 8px;border-bottom:1px solid var(--c08);text-align:left}
.rt td{font-size:12px;padding:7px 8px;border-bottom:1px solid var(--c04);color:var(--txt)}
.rt td:last-child{text-align:right;font-family:'Orbitron',sans-serif;font-size:13px;font-weight:700;color:var(--c)}

/* ── COMP BARS ── */
.cbar{margin-bottom:12px}
.cbh{display:flex;justify-content:space-between;margin-bottom:5px}
.cbn{font-size:11px;color:var(--txt)}
.cbs{font-family:'Orbitron',sans-serif;font-size:12px;font-weight:700;color:var(--c)}
.cbt{height:5px;background:var(--c04);border-radius:2px;overflow:hidden}
.cbf{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--c08),var(--c))}

/* ── SCORE BAR ── */
.sbw{margin-top:14px}
.sbl2{display:flex;justify-content:space-between;font-size:11px;color:var(--lbl);margin-bottom:6px}
.sbt{height:5px;background:var(--c04);border-radius:2px;overflow:hidden}
.sbf{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--c08),var(--c));
  box-shadow:0 0 10px var(--c20)}

/* ── FLOW ITEMS ── */
.flow-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.fhdr{font-family:'Orbitron',sans-serif;font-size:11px;letter-spacing:3px;color:var(--c60);
  padding-bottom:8px;border-bottom:1px solid var(--c08);margin-bottom:10px;display:flex;align-items:center;gap:6px}
.fi{display:flex;align-items:center;gap:10px;padding:8px 12px;
  background:var(--c04);border:1px solid var(--brd);margin-bottom:6px}
.fi-t{font-family:'Orbitron',sans-serif;font-size:13px;font-weight:700;color:var(--c);width:52px;flex-shrink:0}
.fi-bar{flex:1;height:5px;background:var(--c08);border-radius:2px;overflow:hidden}
.fi-fill{height:100%;border-radius:2px}
.fi-v{font-family:'Orbitron',sans-serif;font-size:12px;font-weight:700;color:var(--c);white-space:nowrap;margin-left:8px}
.fi-s{font-size:10px;color:var(--lbl);white-space:nowrap;margin-left:4px}

/* ── LEITURA ── */
.li{display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid var(--c08)}
.lg{font-family:'Orbitron',sans-serif;font-size:14px;font-weight:700;color:var(--c);width:18px;flex-shrink:0}
.ld{width:9px;height:9px;border-radius:50%;flex-shrink:0;background:var(--c)}
.lt{font-size:12px;color:var(--txt);line-height:1.5}

/* ── CHART WRAP ── */
.cw{position:relative;width:100%}

/* ── TICKER ── */
.ticker{flex-shrink:0;border-top:1px solid var(--c08);background:rgba(0,2,6,.9);overflow:hidden;padding:6px 0}
.ti{display:inline-block;white-space:nowrap;animation:tck 42s linear infinite}
.ti s{font-size:11px;color:var(--c40);margin:0 22px;text-decoration:none}
.up{color:rgba(0,212,232,1);text-shadow:0 0 6px rgba(0,212,232,.4)}.dn{color:rgba(248,81,73,.9);text-shadow:0 0 6px rgba(248,81,73,.3)}
@keyframes tck{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
</style>
<script src="https://unpkg.com/zdog@1/dist/zdog.dist.min.js"></script>
</head>
<body>
<canvas id="bg"></canvas>
<div class="scl"></div>

<!-- BOOT -->
<div id="boot">
  <canvas id="boot-reactor" width="80" height="80" style="display:block;margin-bottom:16px"></canvas>
  <div class="btitle">INICIALIZANDO J.A.R.V.I.S</div>
  <div id="blog"></div>
</div>

<!-- APP -->
<div id="app">

  <!-- CMD: aparece UMA vez, resumo topo -->
  <div class="cmd">
    <div class="cmd-id">SPX MARKET COMMAND</div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">SPOT</div><div class="csv" style="color:rgba(0,212,232,1);text-shadow:0 0 8px rgba(0,212,232,.4)">__JV_SPOT__</div></div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">GAMMA FLIP</div><div class="csv" style="color:var(--a60)">__JV_FLIP__</div></div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">GEX NET</div><div class="csv" style="color:__JV_C_GEX__">__JV_GEX__</div></div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">P/C RATIO</div><div class="csv" style="color:var(--a60)">__JV_PC__</div></div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">IV−RV</div><div class="csv" style="color:__JV_C_IVRV__">__JV_IVRV__</div></div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">SQUEEZE RISK</div><div class="csv" style="color:__JV_C_SQ__">__JV_SQ__</div></div>
    <div class="dvd"></div>
    <div class="cs"><div class="csl">TAIL RISK</div><div class="csv" style="color:__JV_C_TAIL__">__JV_TAIL__</div></div>
  </div>

  <!-- HEADER -->
  <div class="hdr">
    <canvas id="hdr-reactor" width="32" height="32" style="flex-shrink:0;cursor:grab"></canvas>
    <div class="brand">
      <div class="bn">J.A.R.V.I.S</div>
      <div class="bs">JUST A RATHER VERY INTELLIGENT SYSTEM  ·  OPTIONS CORE  ·  v4.2</div>
    </div>
    <div class="tabs">
      <button class="tb act" data-t="painel">PAINEL</button>
      <button class="tb" data-t="risco">RISCO</button>
      <button class="tb" data-t="gregas">GREGAS</button>
      <button class="tb" data-t="estrutura">ESTRUTURA</button>
      <button class="tb" data-t="cta">CTA</button>
    </div>
    <div class="clkw" style="text-align:right;flex-shrink:0;min-width:180px">
      <div id="clk-session" style="font-family:'Orbitron',sans-serif;font-size:9px;letter-spacing:2px;color:var(--lbl);margin-bottom:2px">● CARREGANDO...</div>
      <div style="font-family:'Orbitron',sans-serif;font-size:20px;font-weight:700;color:var(--c);line-height:1;letter-spacing:2px" id="clk-time">--:--:--</div>
      <div style="display:flex;justify-content:flex-end;gap:14px;margin-top:3px">
        <div style="font-size:9px;color:var(--lbl);letter-spacing:1px" id="clk-date">---</div>
        <div style="font-size:9px;color:var(--a40);letter-spacing:1px" id="clk-ny">NY --:--</div>
      </div>
      <div style="font-size:8px;color:rgba(0,212,232,.3);letter-spacing:1px;margin-top:2px">DATA: __JV_TS__</div>
    </div>
  </div>

  <!-- CONTENT -->
  <div class="content">

    <!-- ══ PAINEL ══ -->
    <div class="tp act" id="tab-painel">
      <!-- 4 gauges top -->
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px" id="gr1"></div>
      <!-- 3 gauges bottom -->
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px" id="gr2"></div>
      <!-- Vol metrics — PRÊMIO IV-RV e SKEW já estão nos gauges acima -->
      <div class="secl">Volatilidade Implícita</div>
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px">
        <div class="card"><div class="cdl">IV 30D ATM</div><div class="cdv">__JV_IV30__</div></div>
        <div class="card"><div class="cdl">RV 30D REALIZADA</div><div class="cdv" style="opacity:.7">__JV_RV30__</div></div>
      </div>
      <!-- Key levels -->
      <div class="secl">Níveis Chave</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px">
        <div class="card"><div class="cdl">GAMMA FLIP</div><div class="cdv">~__JV_FLIP__</div></div>
        <div class="card"><div class="cdl">CALL WALL</div><div class="cdv">__JV_CW__</div></div>
        <div class="card"><div class="cdl">PUT WALL</div><div class="cdv" style="opacity:.55">__JV_PW__</div></div>
      </div>
      <div style="height:8px"></div>
    </div>

    <!-- ══ RISCO ══ -->
    <div class="tp" id="tab-risco">
      <div style="display:grid;grid-template-columns:1fr 1.2fr 0.9fr;gap:16px">

        <!-- Tail Risk table -->
        <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
          <div class="cb"></div><div class="ct"></div>
          <div class="ph"><div class="phd"></div>TAIL RISK — __JV_TAIL_INT__/100</div>
          <div style="font-size:11px;color:var(--lbl);margin-bottom:12px;letter-spacing:1px">ELEVADO — Monitorar sinais de stress</div>
          <div style="font-size:11px;color:var(--lbl);margin:8px 0 12px;line-height:1.8">
            Score calculado via BQL: excess kurtosis, skew, IV/RV ratio, put skew e risk reversal da superfície de vol do SPX.
          </div>
          <div class="sbw" style="margin-top:auto">
            <div class="sbl2"><span>SCORE TOTAL</span><span style="font-family:'Orbitron',sans-serif;font-size:16px;color:var(--c);font-weight:700">__JV_TAIL_NUM__ / 100</span></div>
            <div class="sbt"><div class="sbf" style="width:__JV_TAIL_PCT__%"></div></div>
          </div>
        </div>

        <!-- Flow Z-Score chart -->
        <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
          <div class="cb"></div><div class="ct"></div>
          <div class="ph"><div class="phd"></div>COMPONENTES FLOW SCORE — Z-SCORE</div>
          <div class="cw" style="flex:1;min-height:520px"><canvas id="flowChart"></canvas></div>
        </div>

        <!-- Gamma Squeeze -->
        <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
          <div class="cb"></div><div class="ct"></div>
          <div class="ph"><div class="phd" style="animation:blink 1s infinite"></div>GAMMA SQUEEZE — __JV_SQ_NUM__/100</div>
          <div style="font-size:11px;color:var(--lbl);margin-bottom:14px;letter-spacing:1px">RISCO MUITO ALTO — condições extremas</div>
          <div id="gc" style="flex:1"></div>
          <div class="sbw" style="border-top:1px solid var(--c08);padding-top:10px">
            <div class="sbl2"><span>SCORE TOTAL</span><span style="font-family:'Orbitron',sans-serif;font-size:16px;color:var(--c);font-weight:700">__JV_SQ_NUM__ / 100</span></div>
            <div class="sbt"><div class="sbf" style="width:__JV_SQ_PCT__%"></div></div>
          </div>
        </div>
      </div>
      <div style="height:8px"></div>
    </div>

    <!-- ══ GREGAS ══ -->
    <div class="tp" id="tab-gregas">

      <!-- 4 semi gauges + leitura -->
      <div style="display:grid;grid-template-columns:repeat(4,1fr) 260px;gap:16px">
        <div id="sgr" style="display:contents"></div>
        <div class="p" style="padding:18px 20px">
          <div class="cb"></div><div class="ct"></div>
          <div class="ph"><div class="phd"></div>LEITURA DAS GREGAS</div>
          <div style="display:flex;flex-direction:column;gap:1px">
            <div class="li"><span class="lg">Δ</span><span class="ld" style="opacity:1"></span><span class="lt">Dealers vendidos → compra</span></div>
            <div class="li"><span class="lg">Γ</span><span class="ld" style="opacity:.35"></span><span class="lt">GEX− acelera movimentos</span></div>
            <div class="li"><span class="lg">V</span><span class="ld" style="opacity:.2"></span><span class="lt">Vanna: Neutro</span></div>
            <div class="li"><span class="lg">C</span><span class="ld" style="opacity:.9"></span><span class="lt">Delta decai → reforça hedge</span></div>
          </div>
        </div>
      </div>


      <div style="height:8px"></div>
    </div>

    <!-- ══ ESTRUTURA ══ -->
    <div class="tp" id="tab-estrutura">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
          <div class="cb"></div><div class="ct"></div>
          <div class="ph"><div class="phd"></div>GAMMA EXPOSURE (GEX) — POR STRIKE</div>
          <div style="font-size:11px;color:var(--lbl);margin-bottom:8px">$ Bi por 1% de movimento — Spot __JV_SPOT__ | G-Flip __JV_FLIP__</div>
          <div class="cw" style="flex:1;min-height:380px"><canvas id="gexChart"></canvas></div>
        </div>
        <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
          <div class="cb"></div><div class="ct"></div>
          <div class="ph"><div class="phd"></div>NÍVEIS DE GAMMA — SPOT vs REFERÊNCIAS</div>
          <div style="font-size:11px;color:var(--lbl);margin-bottom:8px;line-height:1.7">
            <span style="color:rgba(0,212,232,.9);font-weight:700">▲ Call Wall</span> = strike com maior Open Interest de Calls (resistência técnica) &nbsp;·&nbsp;
            <span style="color:rgba(245,166,35,.95);font-weight:700">◆ Gamma Flip</span> = divisor crítico: acima=vol amortecida/mercado pinado, abaixo=vol amplificada &nbsp;·&nbsp;
            <span style="color:rgba(0,212,232,.5);font-weight:700">▼ Put Wall</span> = strike com maior Open Interest de Puts (suporte técnico) &nbsp;·&nbsp;
            linhas tracejadas finas = projeção de movimento esperado pela IV em 5 dias
          </div>
          <div class="cw" style="flex:1;min-height:380px"><canvas id="levChart"></canvas></div>
        </div>
      </div>
      <div style="height:8px"></div>
    </div>

    <!-- ══ CTA ══ -->
    <div class="tp" id="tab-cta">
      <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
        <div class="cb"></div><div class="ct"></div>
        <div class="ph"><div class="phd"></div>CTA ESTIMATES — S&P 500 (NOTIONAL $B)</div>
        <div style="font-size:11px;color:var(--lbl);margin-bottom:8px">
          Dados do modelo CTA — estimativas baseadas em posicionamento histórico via BQL.
          CTAs (Commodity Trading Advisors) são fundos sistemáticos que seguem tendência.
          Posição em $B: <span style="color:rgba(0,212,232,.9)">positivo = comprado</span> · <span style="color:rgba(248,81,73,.9)">negativo = vendido</span>.
        </div>
        <div class="cw" style="min-height:380px"><canvas id="ctaLine"></canvas></div>
      </div>
      <div class="p" style="padding:18px 20px;display:flex;flex-direction:column">
        <div class="cb"></div><div class="ct"></div>
        <div class="ph"><div class="phd"></div>CTA SCENARIO FLOWS ($B) — 1 SEMANA vs 1 MÊS</div>
        <div class="cw" style="min-height:360px"><canvas id="ctaBar"></canvas></div>
      </div>
      <div style="height:8px"></div>
    </div>

  </div><!-- /content -->

  <div class="ticker"><div class="ti" id="ti"></div></div>
</div><!-- /app -->

<script>
// ── PARTICLES
const cv=document.getElementById('bg'),ctx2=cv.getContext('2d');
let W,H,pts=[];
function rsz(){W=cv.width=innerWidth;H=cv.height=innerHeight;
  pts=[];const n=Math.floor(W*H/18000);
  for(let i=0;i<n;i++)pts.push({x:Math.random()*W,y:Math.random()*H,
    vx:(Math.random()-.5)*.1,vy:(Math.random()-.5)*.1,r:Math.random()*.8+.3,a:Math.random()*.25+.08})}
function drawBg(){ctx2.clearRect(0,0,W,H);
  ctx2.strokeStyle='rgba(0,60,80,.04)';ctx2.lineWidth=.5;
  const s=52;for(let x=0;x<W;x+=s)for(let y=0;y<H;y+=s)ctx2.strokeRect(x,y,s,s);
  pts.forEach((p,i)=>{p.x+=p.vx;p.y+=p.vy;
    if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;
    ctx2.beginPath();ctx2.arc(p.x,p.y,p.r,0,Math.PI*2);
    ctx2.fillStyle=`rgba(0,180,210,${p.a})`;ctx2.fill();
    for(let j=i+1;j<pts.length;j++){
      const dx=pts[j].x-p.x,dy=pts[j].y-p.y,d=Math.sqrt(dx*dx+dy*dy);
      if(d<80){ctx2.beginPath();ctx2.moveTo(p.x,p.y);ctx2.lineTo(pts[j].x,pts[j].y);
        ctx2.strokeStyle=`rgba(0,140,180,${.05*(1-d/80)})`;ctx2.lineWidth=.4;ctx2.stroke()}}});
  requestAnimationFrame(drawBg)}
window.addEventListener('resize',rsz);rsz();drawBg();

// ── CLOCK
function _upClock(){
  const now=new Date();
  document.getElementById('clk-time').textContent=
    now.toLocaleTimeString('pt-BR',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  document.getElementById('clk-date').textContent=
    now.toLocaleDateString('pt-BR',{weekday:'short',day:'2-digit',month:'short'});
  const nyFmt=new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',
    hour:'2-digit',minute:'2-digit',hour12:false});
  const nyStr=nyFmt.format(now);
  document.getElementById('clk-ny').textContent='NY '+nyStr;
  const nyD=new Date(now.toLocaleString('en-US',{timeZone:'America/New_York'}));
  const tot=nyD.getHours()*60+nyD.getMinutes(),wd=nyD.getDay();
  let ses,sc;
  if(wd===0||wd===6){ses='■ FIM DE SEMANA';sc='rgba(0,212,232,.35)'}
  else if(tot<240){ses='■ FECHADO';sc='rgba(0,212,232,.35)'}
  else if(tot<570){ses='◐ PRÉ-MERCADO';sc='rgba(245,166,35,.85)'}
  else if(tot<960){ses='● AO VIVO — NYSE/NASDAQ';sc='rgba(0,212,232,1)'}
  else if(tot<1200){ses='◑ PÓS-MERCADO';sc='rgba(245,166,35,.75)'}
  else{ses='■ FECHADO';sc='rgba(0,212,232,.35)'}
  const sel=document.getElementById('clk-session');
  sel.textContent=ses;sel.style.color=sc;
}
setInterval(_upClock,1000);_upClock();

// ── TABS
document.querySelectorAll('.tb').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.tb').forEach(x=>x.classList.remove('act'));
  document.querySelectorAll('.tp').forEach(x=>x.classList.remove('act'));
  b.classList.add('act');
  document.getElementById('tab-'+b.dataset.t).classList.add('act');
}));

// ── ZDOG REACTORS ──────────────────────────────────────────────────────────
(function(){
  if(typeof Zdog==='undefined') return;
  function mkR(id,sz,drag){
    const cvs=document.getElementById(id); if(!cvs) return;
    cvs.width=sz; cvs.height=sz;
    const illo=new Zdog.Illustration({element:'#'+id,resize:false,zoom:sz/90,dragRotate:!!drag});
    const c1='rgba(0,212,232,.95)',c2='rgba(0,212,232,.55)',c3='rgba(0,212,232,.22)';
    new Zdog.Ellipse({addTo:illo,diameter:62,stroke:3,color:c1,fill:false,rotate:{x:Zdog.TAU/4}});
    new Zdog.Ellipse({addTo:illo,diameter:44,stroke:2.2,color:c2,fill:false,rotate:{x:Zdog.TAU/6,y:Zdog.TAU/8}});
    new Zdog.Ellipse({addTo:illo,diameter:26,stroke:1.5,color:c3,fill:false,rotate:{x:-Zdog.TAU/5,y:-Zdog.TAU/6}});
    new Zdog.Shape({addTo:illo,stroke:10,color:'rgba(0,212,232,1)',translate:{z:5}});
    new Zdog.Shape({addTo:illo,stroke:20,color:'rgba(0,212,232,.1)',translate:{z:3}});
    let t=0;
    (function anim(){t+=0.007;illo.rotate.y=t;illo.updateRenderGraph();requestAnimationFrame(anim)})();
  }
  mkR('boot-reactor',80,false);
  mkR('hdr-reactor',32,true);
})();

// ── BOOT VOICE ────────────────────────────────────────────────────────────────
function _jvSpeak(txt){
  if(!('speechSynthesis' in window)) return;
  speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(txt);
  u.lang='en-US'; u.pitch=0.72; u.rate=0.88; u.volume=0.72;
  function go(){speechSynthesis.speak(u);}
  if(speechSynthesis.getVoices().length>0) go();
  else { speechSynthesis.onvoiceschanged=function(){speechSynthesis.onvoiceschanged=null;go();};
         setTimeout(go,250); }
}

// ── BOOT
const BL=['Inicializando núcleo de risco...','Carregando superfície de vol...','Conectando feed OI...','Compilando GEX matrix...','Calibrando modelos de cauda...','Sincronizando posicionamento CTA...','Sistema operacional — ONLINE'];
let bi=0;const bel=document.getElementById('blog');
(function nb(){if(bi<BL.length){bel.innerHTML+=BL[bi++]+'<br>';setTimeout(nb,260)}
 else setTimeout(()=>{document.getElementById('boot').classList.add('gone');
   document.getElementById('app').classList.add('on');buildAll();_jvSpeak('Welcome trader. J A R V I S online.')},500);})();

// ── ARC GAUGE — intensity via opacity only (monochromatic)
function arcGauge(container,{v,mn,mx,label,unit='',state='',intensity=1,size=140}){
  const el=document.createElement('div');el.className='gc';
  const alpha=0.42+intensity*0.58;
  const col=intensity>0.78?`rgba(248,81,73,${alpha})`:
             intensity>0.48?`rgba(245,166,35,${alpha})`:
             `rgba(0,212,232,${alpha})`;
  const glw=intensity>0.78?`rgba(248,81,73,${alpha*.5})`:
             intensity>0.48?`rgba(245,166,35,${alpha*.5})`:
             `rgba(0,212,232,${alpha*.5})`;
  const R=size*.43,CX=size/2,CY=size/2,circ=2*Math.PI*R,sw=.75,tr=sw*circ;
  const pct=Math.max(0,Math.min(1,(v-mn)/(mx-mn)));
  const fill=pct*tr;const id='a'+Math.random().toString(36).slice(2,8);
  el.innerHTML=`
    <div class="gw" style="width:${size}px;height:${size}px">
      <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
        <defs><filter id="${id}"><feGaussianBlur stdDeviation="2.5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
        <circle cx="${CX}" cy="${CY}" r="${R+8}" fill="none" stroke="rgba(0,212,232,.06)" stroke-width="1" stroke-dasharray="3 8">
          <animateTransform attributeName="transform" type="rotate" from="0 ${CX} ${CY}" to="360 ${CX} ${CY}" dur="${14+pct*9}s" repeatCount="indefinite"/>
        </circle>
        <circle cx="${CX}" cy="${CY}" r="${R}" fill="none" stroke="rgba(0,30,44,.95)" stroke-width="7"
          stroke-dasharray="${tr} ${circ-tr}" stroke-dashoffset="0" transform="rotate(-225 ${CX} ${CY})" stroke-linecap="round"/>
        <circle cx="${CX}" cy="${CY}" r="${R}" fill="none" stroke="${col}" stroke-width="10" opacity=".1"
          stroke-dasharray="${fill} ${circ-fill}" stroke-dashoffset="0" transform="rotate(-225 ${CX} ${CY})" stroke-linecap="round"/>
        <circle cx="${CX}" cy="${CY}" r="${R}" fill="none" stroke="${col}" stroke-width="6"
          stroke-dasharray="${fill} ${circ-fill}" stroke-dashoffset="0" transform="rotate(-225 ${CX} ${CY})" stroke-linecap="round" filter="url(#${id})"/>
        <circle cx="${CX}" cy="${CY}" r="${R*.6}" fill="none" stroke="rgba(0,212,232,.1)" stroke-width="1" stroke-dasharray="2 6">
          <animateTransform attributeName="transform" type="rotate" from="0 ${CX} ${CY}" to="-360 ${CX} ${CY}" dur="11s" repeatCount="indefinite"/>
        </circle>
      </svg>
      <div class="gv">
        <span class="gn" style="color:${col};text-shadow:0 0 14px ${glw};font-size:${size*.17}px">${unit==='%'?v+'%':v}</span>
        ${state?`<span class="gst" style="color:${col};opacity:.8">${state}</span>`:''}
      </div>
    </div>
    <div class="gmm" style="width:${size}px"><span>${mn}${unit==='%'?'%':''}</span><span>${mx}${unit==='%'?'%':''}</span></div>
    <div class="gl">${label}</div>`;
  container.appendChild(el);
}

// ── SEMI GAUGE
function semiGauge(container,{v,mn,mx,label,unit='$Bn',intensity=0.7}){
  const el=document.createElement('div');el.className='sc';
  const alpha=0.42+intensity*0.58;
  const col=intensity>0.78?`rgba(248,81,73,${alpha})`:
             intensity>0.48?`rgba(245,166,35,${alpha})`:
             `rgba(0,212,232,${alpha})`;
  const W2=180,H2=100,R=70,CX=90,CY=90;
  const pct=Math.max(0,Math.min(1,(v-mn)/(mx-mn)));
  const sa=Math.PI,ea=2*Math.PI,fa=sa+pct*(ea-sa);
  const tx=a=>CX+R*Math.cos(a),ty=a=>CY+R*Math.sin(a);
  const nz=pct>0.001;
  const trackD=`M ${tx(sa)} ${ty(sa)} A ${R} ${R} 0 1 1 ${tx(ea)} ${ty(ea)}`;
  const fillD=nz?`M ${tx(sa)} ${ty(sa)} A ${R} ${R} 0 0 1 ${tx(fa)} ${ty(fa)}`:'';
  const id='s'+Math.random().toString(36).slice(2,8);
  el.innerHTML=`
    <div class="sl2">${label}</div>
    <svg width="${W2}" height="${H2}" viewBox="0 0 ${W2} ${H2}" style="display:block">
      <defs><filter id="${id}"><feGaussianBlur stdDeviation="2" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
      <path d="${trackD}" fill="none" stroke="rgba(0,30,44,.9)" stroke-width="8" stroke-linecap="round"/>
      ${nz?`<path d="${fillD}" fill="none" stroke="${col}" stroke-width="8" stroke-linecap="round" filter="url(#${id})"/>
            <path d="${fillD}" fill="none" stroke="${col}" stroke-width="14" stroke-linecap="round" opacity=".1"/>`:``}
      <text x="4" y="96" font-size="9" fill="rgba(0,100,130,.5)" font-family="Share Tech Mono">${mn}</text>
      <text x="${W2-12}" y="96" font-size="9" fill="rgba(0,100,130,.5)" font-family="Share Tech Mono" text-anchor="end">${mx}</text>
    </svg>
    <div class="sv" style="color:${col};text-shadow:0 0 10px rgba(0,212,232,${alpha*.4})">${v} <span style="font-size:12px;opacity:.6">${unit}</span></div>`;
  container.appendChild(el);
}

// ── CHART DEFAULTS
Chart.defaults.color='rgba(0,140,170,.65)';
Chart.defaults.font={family:"'Share Tech Mono',monospace",size:11};
const G='rgba(0,50,70,.25)';
const TT={backgroundColor:'rgba(0,4,10,.97)',borderColor:'rgba(0,80,100,.3)',borderWidth:1,
  titleColor:'rgba(0,200,220,.8)',bodyColor:'rgba(0,140,170,.85)',padding:8};

function buildAll(){

  // PAINEL gauges row 1
  const g1=document.getElementById('gr1');
  [{v:__JV_V_FRAG__,mn:0,mx:20,label:'FRAGILIDADE',unit:'%',state:'ALTO',intensity:0.95},
   {v:__JV_V_IVRV__,mn:0,mx:10,label:'PRÊMIO VOL',state:'pp',intensity:0.55},
   {v:__JV_V_SKEW__,mn:-15,mx:15,label:'SKEW P25-C25',unit:'%',intensity:0.65},
   {v:__JV_V_MOVE__,mn:0,mx:5,label:'MOV ESP 10',unit:'%',intensity:0.35}
  ].forEach(g=>arcGauge(g1,g));

  const g2=document.getElementById('gr2');
  [{v:__JV_V_TAIL__,mn:0,mx:100,label:'TAIL RISK',state:'ELEVADO',intensity:0.65},
   {v:__JV_V_FLOW__,mn:0,mx:100,label:'FLOW SCORE',state:'NEUTRO',intensity:0.5},
   {v:__JV_V_SQ__,mn:0,mx:100,label:'GAMMA SQUEEZE',state:'CRÍTICO',intensity:1}
  ].forEach(g=>arcGauge(g2,g));

  // RISCO — gamma comps
  __JV_SQ_COMPS__.forEach(c=>{const p=(c.s/c.m)*100;
    document.getElementById('gc').innerHTML+=`
    <div class="cbar">
      <div class="cbh">
        <span class="cbn">${c.n}</span>
        <span class="cbs" style="opacity:${c.i}">${c.s}<span style="opacity:.4;font-size:10px">/${c.m}</span></span>
      </div>
      <div class="cbt"><div class="cbf" style="width:${p}%;opacity:${c.i}"></div></div>
    </div>`});

  // RISCO — Flow Z-Score
  new Chart(document.getElementById('flowChart'),{
    type:'bar',
    data:{
      labels:['CTA','Dealer/MM','Vol Ctrl',['Risk','Parity'],['ETFs','Alav.'],['ETFs','Passivos'],'Buyback','COT'],
      datasets:[{
        label:'Z-Score',
        data:[__JV_FLOW_DATA__],
        backgroundColor:d=>d.raw>=0?'rgba(88,166,255,.78)':'rgba(248,81,73,.82)',
        borderColor:d=>d.raw>=0?'rgba(120,200,255,1)':'rgba(255,120,120,1)',
        borderWidth:1,borderRadius:2,
        order:2
      },{
        type:'line',
        label:'Peso',
        data:[__JV_FLOW_W_DATA__],
        yAxisID:'y1',
        showLine:false,
        pointRadius:4,
        pointHoverRadius:4,
        pointBackgroundColor:'rgba(180,190,205,.95)',
        pointBorderColor:'rgba(40,55,70,.95)',
        pointBorderWidth:1,
        order:1
      }]
    },
    options:{responsive:true,maintainAspectRatio:false,
      layout:{padding:{top:6,bottom:12,left:6,right:6}},
      plugins:{
        legend:{
          display:true,
          position:'bottom',
          labels:{color:'rgba(180,200,220,.85)',boxWidth:10,font:{size:9}}
        },
        tooltip:TT
      },
      scales:{
        x:{
          grid:{color:G},
          ticks:{
            color:'rgba(0,200,220,.9)',
            font:{size:9,family:"'Orbitron',sans-serif"},
            autoSkip:false,
            maxRotation:0,
            minRotation:0,
            padding:8
          },
          border:{color:'rgba(0,80,100,.2)'}
        },
        y:{grid:{color:G},ticks:{color:'rgba(0,180,200,.7)'},min:-3.5,max:4,
          title:{display:true,text:'Z-Score',color:'rgba(0,120,150,.5)',font:{size:8}}},
        y1:{
          position:'right',
          min:0,max:1,
          grid:{drawOnChartArea:false},
          ticks:{
            color:'rgba(170,180,190,.8)',
            callback:(v)=>`${Math.round(v*100)}%`
          },
          title:{display:true,text:'Peso',color:'rgba(140,150,165,.65)',font:{size:8}}
        }
      }
    }
  });

  // GREGAS — semis
  const sg=document.getElementById('sgr');
  [{v:__JV_V_DELTA__,mn:__JV_V_DELTA_MIN__,mx:__JV_V_DELTA_MAX__,label:'Δ DELTA NOCIONAL',intensity:1},
   {v:__JV_V_GEX_SEMI__,mn:-40,mx:40,label:'Γ GAMMA (GEX NET)',intensity:0.4},
   {v:__JV_V_VANNA__,mn:__JV_V_VANNA_MIN__,mx:__JV_V_VANNA_MAX__,label:'V VANNA',intensity:0.25},
   {v:__JV_V_CHARM__,mn:__JV_V_CHARM_MIN__,mx:__JV_V_CHARM_MAX__,label:'C CHARM (DIÁRIO)',intensity:0.8}
  ].forEach(g=>semiGauge(sg,g));

  // ESTRUTURA — GEX curve
  const strikes=[],gex=[];
  for(let s=6400;s<=7000;s+=10){
    strikes.push(s);
    const x=(s-__JV_FLIP_NUM__)/200;
    gex.push(40*(2/(1+Math.exp(-x*3))-1));
  }
  new Chart(document.getElementById('gexChart'),{
    type:'line',
    data:{labels:strikes,datasets:[
      {label:'GEX $Bi/1%',data:gex,
       borderColor:'rgba(0,212,232,.8)',backgroundColor:'rgba(0,80,120,.1)',
       borderWidth:1.5,fill:true,pointRadius:0,tension:0.4},
      {label:'Spot __JV_SPOT__',
       data:strikes.map((s,i)=>Math.abs(s-__JV_SPOT_R10__)<6?gex[i]:null),
       type:'scatter',pointRadius:6,
       pointBackgroundColor:'rgba(0,212,232,1)',pointBorderColor:'rgba(0,212,232,.3)',pointBorderWidth:3},
      {label:'G-Flip __JV_FLIP__',
       data:strikes.map((s,i)=>Math.abs(s-__JV_FLIP_R10__)<6?gex[i]:null),
       type:'scatter',pointRadius:6,
       pointBackgroundColor:'rgba(0,212,232,.4)',pointBorderColor:'rgba(0,212,232,.2)',pointBorderWidth:3}
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'rgba(0,140,170,.7)',boxWidth:8,font:{size:8}}},tooltip:TT},
      scales:{
        x:{grid:{color:G},ticks:{color:'rgba(0,140,170,.5)',maxTicksLimit:8},
          title:{display:true,text:'SPX Strike',color:'rgba(0,120,150,.4)',font:{size:8}}},
        y:{grid:{color:G},ticks:{color:'rgba(0,140,170,.5)'},
          title:{display:true,text:'$ Bi / 1% move',color:'rgba(0,120,150,.4)',font:{size:8}}}
      }
    }
  });

  // ESTRUTURA — Gamma Levels (snapshot — sem dados históricos falsos)
  const lvSpot=__JV_SPOT_NUM__;
  const lvCW=__JV_CW_NUM__;
  const lvPW=__JV_PW_NUM__;
  const lvFlip=__JV_FLIP_NUM__;
  const lvEstUp=__JV_EST_UP_NUM__;
  const lvEstDn=__JV_EST_DN_NUM__;
  const lvRange=Math.max(200,(lvCW-lvPW)*0.35,Math.abs(lvEstUp-lvSpot)*1.8);
  const lvMin=Math.min(lvPW,lvEstDn,lvSpot)-lvRange;
  const lvMax=Math.max(lvCW,lvEstUp,lvSpot)+lvRange;

  // Determina a zona atual do SPX
  const zoneLabel=lvSpot>lvCW?'ACIMA CALL WALL — Mercado empurrado':
                  lvSpot>lvFlip?'ACIMA GAMMA FLIP — Vol baixa, dealers compram dip':
                  lvSpot>lvPW?'ABAIXO FLIP — Vol pode acelerar, zona de risco':
                  'ABAIXO PUT WALL — Suporte em risco, vol extrema';
  const zoneColor=lvSpot>lvFlip?'rgba(0,212,232,.9)':'rgba(248,81,73,.9)';

  // Plugin: desenha zonas coloridas + labels diretamente no gráfico
  const lvDraw={id:'lvDraw',afterDraw(chart){
    const {ctx,chartArea:{left,right,top,bottom},scales:{y}}=chart;
    // Zonas de fundo coloridas por região
    [
      [Math.max(lvCW,lvEstUp),lvMax,'rgba(0,212,232,.05)'],   // acima call wall
      [lvFlip,Math.max(lvCW,lvEstUp),'rgba(0,212,232,.09)'],  // entre flip e call wall
      [lvPW,lvFlip,'rgba(245,166,35,.06)'],                    // entre put wall e flip
      [Math.min(lvPW,lvEstDn),lvPW,'rgba(248,81,73,.09)'],    // abaixo put wall
    ].forEach(([lo,hi,c])=>{
      const y1=Math.min(bottom,Math.max(top,y.getPixelForValue(hi)));
      const y2=Math.min(bottom,Math.max(top,y.getPixelForValue(lo)));
      if(y1>=y2) return;
      ctx.save();ctx.fillStyle=c;ctx.fillRect(left,y1,right-left,y2-y1);ctx.restore();
    });
    // Labels das linhas de referência — lado direito
    const lblPad=right-8;
    [
      [lvCW,  '▲ CALL WALL  '+lvCW.toLocaleString('pt-BR'),  'rgba(0,212,232,.95)', 'bold 11px'],
      [lvFlip,'◆ GAMMA FLIP  '+lvFlip.toLocaleString('pt-BR'),'rgba(245,166,35,1)', 'bold 12px'],
      [lvPW,  '▼ PUT WALL  '+lvPW.toLocaleString('pt-BR'),   'rgba(0,212,232,.65)', 'bold 11px'],
      [lvEstUp,'↑ +5d IV  '+lvEstUp.toLocaleString('pt-BR'), 'rgba(0,212,232,.55)', '10px'],
      [lvEstDn,'↓ −5d IV  '+lvEstDn.toLocaleString('pt-BR'), 'rgba(248,81,73,.55)', '10px'],
      [lvSpot, '● SPX  '+lvSpot.toLocaleString('pt-BR'),     'rgba(0,212,232,1)',   'bold 13px'],
    ].forEach(([v,lbl,col,fnt])=>{
      const yp=y.getPixelForValue(v);
      if(yp<top-2||yp>bottom+2) return;
      const yc=Math.max(top+10,Math.min(bottom-10,yp));
      ctx.save();
      ctx.fillStyle=col;
      ctx.font=`${fnt} 'Share Tech Mono',monospace`;
      ctx.textAlign='right';ctx.textBaseline='middle';
      // fundo semi-transparente para legibilidade
      const w=ctx.measureText(lbl).width;
      ctx.fillStyle='rgba(2,8,16,.7)';
      ctx.fillRect(lblPad-w-6,yc-8,w+12,16);
      ctx.fillStyle=col;
      ctx.fillText(lbl,lblPad,yc);
      ctx.restore();
    });
    // Status da zona — canto superior esquerdo
    ctx.save();
    ctx.fillStyle='rgba(2,8,16,.75)';ctx.fillRect(left+6,top+6,420,22);
    ctx.fillStyle=zoneColor;
    ctx.font=`bold 11px 'Share Tech Mono',monospace`;
    ctx.textAlign='left';ctx.textBaseline='middle';
    ctx.fillText('ZONA ATUAL: '+zoneLabel,left+12,top+17);
    ctx.restore();
  }};

  new Chart(document.getElementById('levChart'),{
    type:'line',
    data:{labels:['HOJE','PROJ +5d'],datasets:[
      // Fan de projeção — zona preenchida
      {data:[lvSpot,lvEstUp],backgroundColor:'rgba(0,212,232,.07)',
       borderColor:'transparent',borderWidth:0,pointRadius:0,fill:'+1',tension:0},
      {data:[lvSpot,lvEstDn],backgroundColor:'rgba(248,81,73,.05)',
       borderColor:'transparent',borderWidth:0,pointRadius:0,fill:false,tension:0},
      // Linhas de referência — horizontais
      {data:[lvCW,lvCW],  borderColor:'rgba(0,212,232,.9)',   borderWidth:2.5,borderDash:[14,6],pointRadius:0,fill:false},
      {data:[lvFlip,lvFlip],borderColor:'rgba(245,166,35,1)', borderWidth:3,  borderDash:[10,5],pointRadius:0,fill:false},
      {data:[lvPW,lvPW],  borderColor:'rgba(0,212,232,.5)',   borderWidth:2,  borderDash:[14,6],pointRadius:0,fill:false},
      // Projeção +/− 5d
      {data:[lvSpot,lvEstUp],borderColor:'rgba(0,212,232,.5)',borderWidth:2,borderDash:[5,6],
       pointRadius:[0,8],pointBackgroundColor:'rgba(0,212,232,.7)',fill:false,tension:0},
      {data:[lvSpot,lvEstDn],borderColor:'rgba(248,81,73,.5)',borderWidth:2,borderDash:[5,6],
       pointRadius:[0,8],pointBackgroundColor:'rgba(248,81,73,.7)',fill:false,tension:0},
      // Spot
      {data:[lvSpot,null],type:'scatter',
       pointRadius:16,pointHoverRadius:18,
       pointBackgroundColor:'rgba(0,212,232,1)',
       pointBorderColor:'rgba(0,212,232,.2)',pointBorderWidth:6}
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{display:false},
        tooltip:{...TT,callbacks:{
          title:()=>'',
          label:ctx=>{
            const v=ctx.parsed.y; if(v==null) return null;
            return `${v.toLocaleString('pt-BR')}`;
          }
        }}
      },
      scales:{
        x:{grid:{color:G},
          ticks:{color:'rgba(0,200,220,.85)',font:{size:14,family:"'Orbitron',sans-serif"},
            padding:10}},
        y:{grid:{color:G},
          ticks:{color:'rgba(0,140,170,.7)',font:{size:11},
            callback:v=>v.toLocaleString('pt-BR')},
          min:lvMin,max:lvMax}
      }
    },
    plugins:[lvDraw]
  });

  // CTA — Line
  const ctaDates=['Dez 1','Dez 8','Dez 15','Dez 22','Jan 1','Jan 8','Jan 15','Jan 22','Fev 1','Fev 8','Fev 15','Mar 1','Mar 8','Mar 15'];
  const ctaHist=[65,22,-15,8,28,52,72,83,88,90,86,35,16,-68];
  const projDates=[...ctaDates,'Mar 22','Mar 29','Abr 5','Abr 12'];
  const nh=ctaDates.length; // 14
  // pad: 13 nulls + 5-point array = 18 items matching projDates
  const pad=arr=>[...Array(nh-1).fill(null),...arr];

  // inline plugin: $B labels on bars
  const barLabels={id:'barLabels',afterDatasetsDraw(chart){
    const {ctx}=chart;
    chart.data.datasets.forEach((ds,i)=>{
      chart.getDatasetMeta(i).data.forEach((bar,j)=>{
        const v=ds.data[j]; if(v==null) return;
        const lbl=(v>=0?'+':'')+v.toFixed(1)+'B';
        ctx.save(); ctx.fillStyle='rgba(0,212,232,1)';
        ctx.font='bold 11px monospace'; ctx.textAlign='center';
        ctx.textBaseline=v>=0?'bottom':'top';
        ctx.fillText(lbl,bar.x,v>=0?bar.y-4:bar.y+4); ctx.restore();
      });
    });
  }};

  new Chart(document.getElementById('ctaLine'),{
    type:'line',
    data:{labels:projDates,datasets:[
      {label:'CTA Notional (Hist)',
       data:[...ctaHist,...Array(4).fill(null)],
       borderColor:'rgba(0,212,232,.9)',backgroundColor:'rgba(0,60,100,.12)',
       borderWidth:2,fill:true,pointRadius:0,tension:0.35},
      {label:'Flat',
       data:pad([-68,-68,-68,-68,-68]),
       borderColor:'rgba(0,212,232,.3)',borderWidth:1.5,borderDash:[5,4],
       pointRadius:0,fill:false,spanGaps:true},
      {label:'Up 1σ  →+37B',
       data:pad([-68,-38,-8,18,37]),
       borderColor:'rgba(0,212,232,1)',borderWidth:2.5,borderDash:[5,3],
       pointRadius:[...Array(nh-1).fill(0),5,3,3,3,8],
       pointBackgroundColor:'rgba(0,212,232,1)',
       fill:false,tension:0.3,spanGaps:true},
      {label:'Up 2σ  →+75B',
       data:pad([-68,-20,15,48,75]),
       borderColor:'rgba(0,212,232,.6)',borderWidth:2,borderDash:[5,3],
       pointRadius:[...Array(nh-1).fill(0),4,3,3,3,7],
       pointBackgroundColor:'rgba(0,212,232,.6)',
       fill:false,tension:0.3,spanGaps:true},
      {label:'Down 1σ  →-75B',
       data:pad([-68,-70,-73,-74,-75]),
       borderColor:'rgba(248,81,73,.9)',borderWidth:2,borderDash:[5,3],
       pointRadius:[...Array(nh-1).fill(0),4,3,3,3,7],
       pointBackgroundColor:'rgba(248,81,73,.9)',
       fill:false,tension:0.2,spanGaps:true},
      {label:'Down 2.5σ  →-85B',
       data:pad([-68,-73,-78,-82,-85]),
       borderColor:'rgba(248,81,73,.45)',borderWidth:1.5,borderDash:[3,5],
       pointRadius:[...Array(nh-1).fill(0),3,2,2,2,6],
       pointBackgroundColor:'rgba(248,81,73,.45)',
       fill:false,tension:0.2,spanGaps:true},
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{
        legend:{
          display:true,position:'top',
          labels:{color:'rgba(0,212,232,.85)',boxWidth:20,padding:14,
            font:{size:11,family:"'Share Tech Mono',monospace"}}
        },
        tooltip:{...TT,callbacks:{
          label:ctx=>`${ctx.dataset.label}: $${ctx.parsed.y}B`
        }}
      },
      scales:{
        x:{grid:{color:G},ticks:{color:'rgba(0,212,232,.55)',maxTicksLimit:12,font:{size:10}}},
        y:{grid:{color:G},ticks:{color:'rgba(0,212,232,.55)',font:{size:10}},
          title:{display:true,text:'$B Notional (CTA positioning)',color:'rgba(0,212,232,.5)',font:{size:11}}}
      }
    }
  });

  // CTA — Scenario Bar
  new Chart(document.getElementById('ctaBar'),{
    type:'bar',
    plugins:[barLabels],
    data:{
      labels:['Flat','Up 1σ','Up 2σ','Down 1σ','Down 2σ','Down 2.5σ'],
      datasets:[
        {label:'1 Semana',data:[0.5,14.5,38.3,-6.8,-7.9,-6.4],
         backgroundColor:d=>d.raw>=0?'rgba(0,212,232,.55)':'rgba(0,212,232,.18)',
         borderColor:'rgba(0,212,232,.8)',borderWidth:1,borderRadius:2},
        {label:'1 Mês',data:[1.5,105.8,143.8,-7.7,-2.0,0.5],
         backgroundColor:d=>d.raw>=0?'rgba(0,212,232,.3)':'rgba(0,212,232,.1)',
         borderColor:'rgba(0,212,232,.5)',borderWidth:1,borderRadius:2},
      ]
    },
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'rgba(0,212,232,1)',boxWidth:8,font:{size:8}}},tooltip:TT},
      scales:{
        x:{grid:{color:G},ticks:{color:'rgba(0,212,232,.5)'}},
        y:{grid:{color:G},ticks:{color:'rgba(0,212,232,.5)'},
          title:{display:true,text:'$B',color:'rgba(0,212,232,.4)',font:{size:8}}}
      }
    }
  });
}

// TICKER
const td=[['SPX','__JV_SPOT__',1],['GEX','__JV_GEX_T__',0],['GAMMA FLIP','__JV_FLIP__',0],['VIX','__JV_VIX__',0],
  ['IV 30D','__JV_IV30__',0],['RV 30D','__JV_RV30__',1],['P/C','__JV_PC_T__',0],['CTA','−$68B',0],
  ['PUT WALL','__JV_PW__',0],['CALL WALL','__JV_CW__',1],['TAIL RISK','__JV_TAIL_NUM__',0],['SQUEEZE','__JV_SQ_T__',0]];
const th=td.map(([n,v,u])=>`<s>${n} <span class="${u?'up':'dn'}">${u?'▲':'▼'} ${v}</span></s>`).join('');
document.getElementById('ti').innerHTML=th+th;
</script>
</body>
</html>

"""

def _export_dashboard_html():
    """Exporta JARVIS HUD HTML standalone — 100% fiel ao jarvis_final v2."""
    if not _snapshot.get('ts'):
        return None

    m    = _snapshot.get('metrics', {})
    spot = _snapshot['spot']

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _f(k, default=0):
        v = m.get(k, default)
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def _sym_range(v, factor=1.5, min_abs=1.0):
        """Symmetric range centred on 0 for semi-gauges."""
        mag = max(min_abs, abs(v) * factor)
        return round(-mag, 1), round(mag, 1)

    # ── Metric strings ────────────────────────────────────────────────────────
    _spot_s      = f"{spot:,.0f}"
    _flip_raw    = _f('gamma_flip')
    _flip_s      = f"{_flip_raw:,.0f}" if _flip_raw else "N/A"
    _flip_num    = round(_flip_raw)
    _spot_r10    = round(spot / 10) * 10
    _flip_r10    = round(_flip_num / 10) * 10

    _gex_raw     = _f('gex_net_bn')
    _gex_sign    = "\u2212" if _gex_raw < 0 else "+"
    _gex_s       = f"{_gex_sign}{abs(_gex_raw):.1f}B"
    _gex_t       = f"{_gex_sign}${abs(_gex_raw):.1f}B"

    _pc_raw      = _f('pc_ratio')
    _pc_s        = f"{_pc_raw:.2f}\u00d7" if _pc_raw > 0 else 'N/D'

    _ivrv_raw    = _f('iv_rv_pp')
    _ivrv_s      = f"{_ivrv_raw:+.1f}pp"
    _ivrv_prem_s = f"{_ivrv_raw:+.2f}%"

    _sq_raw      = _f('squeeze_score')
    _sq_s        = f"{_sq_raw:.0f}/100"
    _sq_int_s    = f"{_sq_raw:.0f}"

    _tail_raw    = _f('tail_score')
    _tail_s      = f"{_tail_raw:.1f}/100"
    _tail_num_s  = f"{_tail_raw:.1f}"
    _tail_int_s  = f"{_tail_raw:.0f}"

    _iv30_raw    = _f('iv_30d')
    _rv30_raw    = _f('rv_30d')
    _iv30_s      = f"{_iv30_raw*100:.2f}%"
    _rv30_s      = f"{_rv30_raw*100:.2f}%"

    _cw_raw      = _f('call_wall')
    _pw_raw      = _f('put_wall')
    _cw_s        = f"{_cw_raw:,.0f}" if _cw_raw else "N/A"
    _pw_s        = f"{_pw_raw:,.0f}" if _pw_raw else "N/A"

    # ── JS gauge values ───────────────────────────────────────────────────────
    _frag_raw    = _f('fragility')
    _frag_v      = round(_frag_raw * 100, 2) if abs(_frag_raw) <= 1.0 else round(abs(_frag_raw), 2)
    _move_raw    = _f('daily_move')
    _move_v      = round(abs(_move_raw) * 100, 2) if abs(_move_raw) <= 1.0 else round(abs(_move_raw), 2)
    _ivrv_v      = round(abs(_ivrv_raw), 2)
    _sq_v        = round(_sq_raw, 1)
    _tail_v      = round(_tail_raw, 1)

    # ── Greek semi-gauge values (real BBG) ────────────────────────────────────
    _delta_v     = round(_f('delta_bn'), 2)
    _delta_min, _delta_max = _sym_range(_delta_v, factor=1.5, min_abs=5.0)
    _vanna_v     = round(_f('vanna_bn'), 3)
    _vanna_min, _vanna_max = _sym_range(_vanna_v, factor=2.0, min_abs=0.5)
    _charm_v     = round(_f('charm_bn'), 3)
    _charm_min, _charm_max = _sym_range(_charm_v, factor=2.0, min_abs=0.2)
    _gex_semi    = round(_gex_raw, 2)

    # ── Apply replacements ────────────────────────────────────────────────────
    _html = _JARVIS_EXPORT_TEMPLATE
    _html = _html.replace('__JV_SPOT__',       _spot_s)
    _html = _html.replace('__JV_FLIP__',       _flip_s)
    _html = _html.replace('__JV_GEX__',        _gex_s)
    _html = _html.replace('__JV_PC__',         _pc_s)
    _html = _html.replace('__JV_IVRV__',       _ivrv_s)
    _html = _html.replace('__JV_SQ__',         _sq_s)
    _html = _html.replace('__JV_TAIL__',       _tail_s)
    _html = _html.replace('__JV_IV30__',       _iv30_s)
    _html = _html.replace('__JV_RV30__',       _rv30_s)
    _html = _html.replace('__JV_IVRV_PREM__',  _ivrv_prem_s)
    _html = _html.replace('__JV_CW__',         _cw_s)
    _html = _html.replace('__JV_PW__',         _pw_s)
    _html = _html.replace('__JV_TAIL_INT__',   _tail_int_s)
    _html = _html.replace('__JV_TAIL_NUM__',   _tail_num_s)
    _html = _html.replace('__JV_TAIL_PCT__',   _tail_num_s)
    _html = _html.replace('__JV_SQ_NUM__',     _sq_int_s)
    _html = _html.replace('__JV_SQ_PCT__',     _sq_int_s)
    _html = _html.replace('__JV_V_FRAG__',     str(_frag_v))
    _html = _html.replace('__JV_V_IVRV__',     str(_ivrv_v))
    _html = _html.replace('__JV_V_MOVE__',     str(_move_v))
    _html = _html.replace('__JV_V_TAIL__',     str(_tail_v))
    _html = _html.replace('__JV_V_SQ__',       str(_sq_v))
    _html = _html.replace('__JV_FLIP_NUM__',   str(_flip_num))
    _html = _html.replace('__JV_SPOT_R10__',   str(_spot_r10))
    _html = _html.replace('__JV_FLIP_R10__',   str(_flip_r10))
    _html = _html.replace('__JV_GEX_T__',      _gex_t)
    _html = _html.replace('__JV_PC_T__',       _pc_s)
    _html = _html.replace('__JV_SQ_T__',       _sq_s)
    # ── Semantic colors for CMD strip ─────────────────────────────────────────
    _c_gex   = 'rgba(0,212,232,1)' if _gex_raw >= 0 else 'rgba(248,81,73,.9)'
    _c_ivrv  = 'rgba(248,81,73,.95)' if _ivrv_raw > 2 else ('rgba(245,166,35,.9)' if _ivrv_raw > 0 else 'rgba(0,212,232,.85)')
    _c_sq    = 'rgba(248,81,73,1)' if _sq_raw > 70 else ('rgba(245,166,35,.95)' if _sq_raw > 40 else 'rgba(0,212,232,.85)')
    _c_tail  = 'rgba(248,81,73,1)' if _tail_raw > 70 else ('rgba(245,166,35,.95)' if _tail_raw > 40 else 'rgba(0,212,232,.85)')
    _html = _html.replace('__JV_C_GEX__',    _c_gex)
    _html = _html.replace('__JV_C_IVRV__',   _c_ivrv)
    _html = _html.replace('__JV_C_SQ__',     _c_sq)
    _html = _html.replace('__JV_C_TAIL__',   _c_tail)
    # Timestamp
    import datetime as _dt
    _ts_str = _snapshot.get('ts', _dt.datetime.now().strftime('%Y-%m-%d %H:%M'))
    _html = _html.replace('__JV_TS__', str(_ts_str)[:16])
    # Greek semi-gauges (real BBG)
    _html = _html.replace('__JV_V_GEX_SEMI__',  str(_gex_semi))
    _html = _html.replace('__JV_V_DELTA__',      str(_delta_v))
    _html = _html.replace('__JV_V_DELTA_MIN__',  str(_delta_min))
    _html = _html.replace('__JV_V_DELTA_MAX__',  str(_delta_max))
    _html = _html.replace('__JV_V_VANNA__',      str(_vanna_v))
    _html = _html.replace('__JV_V_VANNA_MIN__',  str(_vanna_min))
    _html = _html.replace('__JV_V_VANNA_MAX__',  str(_vanna_max))
    _html = _html.replace('__JV_V_CHARM__',      str(_charm_v))
    _html = _html.replace('__JV_V_CHARM_MIN__',  str(_charm_min))
    _html = _html.replace('__JV_V_CHARM_MAX__',  str(_charm_max))
    # ── Gamma levels chart numeric values ─────────────────────────────────────
    _spot_num    = round(spot)
    _cw_num      = round(_cw_raw) if _cw_raw else round(spot * 1.03)
    _pw_num      = round(_pw_raw) if _pw_raw else round(spot * 0.97)
    _est_move_5d = round(spot * (_iv30_raw if _iv30_raw > 0 else 0.15) * (5/252)**0.5)
    _est_up_num  = round(spot) + _est_move_5d
    _est_dn_num  = round(spot) - _est_move_5d
    _html = _html.replace('__JV_SPOT_NUM__',   str(_spot_num))
    _html = _html.replace('__JV_CW_NUM__',     str(_cw_num))
    _html = _html.replace('__JV_PW_NUM__',     str(_pw_num))
    _html = _html.replace('__JV_EST_UP_NUM__', str(_est_up_num))
    _html = _html.replace('__JV_EST_DN_NUM__', str(_est_dn_num))

    # Flow score — 8 real BBG components
    import json as _json
    _flow_data = _json.dumps([
        round(_f('z_cta'), 2),
        round(_f('z_dealer'), 2),
        round(_f('z_volctrl'), 2),
        round(_f('z_rp'), 2),
        round(_f('z_leveraged'), 2),
        round(_f('z_passive_etf'), 2),
        round(_f('z_buyback'), 2),
        round(_f('z_cot'), 2),
    ])
    _flow_w_data = _json.dumps([
        round(_f('w_cta', 0), 4),
        round(_f('w_dealer', 0), 4),
        round(_f('w_volctrl', 0), 4),
        round(_f('w_rp', 0), 4),
        round(_f('w_leveraged', 0), 4),
        round(_f('w_passive_etf', 0), 4),
        round(_f('w_buyback', 0), 4),
        round(_f('w_cot', 0), 4),
    ])
    _html = _html.replace('[__JV_FLOW_DATA__]', _flow_data)
    _html = _html.replace('[__JV_FLOW_W_DATA__]', _flow_w_data)

    # ── SKEW 25d (put-call IV spread, real BBG) ───────────────────────────────
    _skew_v = round(_f('skew_25d', 0), 2)
    _html = _html.replace('__JV_V_SKEW__', str(_skew_v))

    # ── FLOW SCORE total (from fp_score via BBG) ──────────────────────────────
    _flow_score_v = round(_f('flow_score_total', 50), 1)
    _html = _html.replace('__JV_V_FLOW__', str(_flow_score_v))

    # ── VIX current (fetched from BBG time series, last value) ────────────────
    _vix_raw = _f('vix', 0)
    _vix_s = f"{_vix_raw:.1f}" if _vix_raw > 0 else 'N/D'
    _html = _html.replace('__JV_VIX__', _vix_s)

    # ── Gamma Squeeze component bars (real BBG-derived) ───────────────────────
    import json as _json2
    _sq_comps_raw = m.get('squeeze_components', {})
    _sq_comps_list = []
    _comp_order = [
        ('vol_premium',     'Prêmio de Vol (IV−RV)', 20),
        ('flip_proximity',  'Distância Gamma Flip',   25),
        ('pc_ratio',        'P/C OI Ratio',           25),
        ('gex',             'GEX Negatividade',       30),
    ]
    for _ck, _cn, _cm in _comp_order:
        _cv = _sq_comps_raw.get(_ck, {})
        _cs = float(_cv.get('score', 0)) if isinstance(_cv, dict) else 0.0
        _ci = min(1.0, max(0.1, _cs / _cm)) if _cm > 0 else 0.5
        _sq_comps_list.append({'n': _cn, 's': round(_cs, 1), 'm': _cm, 'i': round(_ci, 2)})
    _html = _html.replace('__JV_SQ_COMPS__', _json2.dumps(_sq_comps_list))

    return _html



def _collect_widget_content(widget):
    """Recursivamente extrai conteúdo (Plotly figs + HTML) de um widget tree."""
    items = []
    if isinstance(widget, go.FigureWidget):
        items.append({'type': 'plotly', 'data': go.Figure(widget)})
    elif isinstance(widget, wd.HTML):
        val = widget.value.strip()
        if val:
            items.append({'type': 'html', 'data': val})
    elif isinstance(widget, wd.Output):
        # Output widgets contem matplotlib — não é fácil capturar retroativamente
        items.append({'type': 'html', 'data': '<p><i>[Conteúdo de Output widget — ver aba Exposições no dashboard]</i></p>'})
    elif hasattr(widget, 'children'):
        for child in widget.children:
            items.extend(_collect_widget_content(child))
    return items


def _capture_matplotlib_figures(func, *args, **kwargs):
    """Executa func que gera plt.show() e captura todas as figuras como base64 PNGs."""
    import io, base64
    plt.close('all')
    _orig_show = plt.show
    _figs_collected = []

    def _capture_show(*a, **kw):
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        _figs_collected.append(base64.b64encode(buf.read()).decode())
        plt.close(fig)

    plt.show = _capture_show
    try:
        func(*args, **kwargs)
    finally:
        plt.show = _orig_show
    return [{'type': 'matplotlib', 'data': b64} for b64 in _figs_collected]


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — CALLBACK PRINCIPAL E MONTAGEM DO DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

# Widget definitions (global)
ticker_w = wd.Text(value='SPX Index', description='Ativo:',
                   placeholder='ex: SPX Index | HWAA Index (Micro SPX)',
                   layout={'width': '300px'})
dte_w = wd.IntRangeSlider(value=[0, 30], min=0, max=90, step=1,
                          description='DTE (dias):',
                          layout={'width': '400px'})
mny_w = wd.FloatRangeSlider(value=[-0.05, 0.05], min=-0.30, max=0.30,
                            step=0.01, readout_format='.0%',
                            description='% MNY:',
                            layout={'width': '400px'})
run_btn = wd.Button(description='Gerar Análise Completa',
                    button_style='success', icon='cogs')
spx_pred_w = wd.Checkbox(value=False, description='Incluir Previsão SPX',
                         indent=False, layout={'width': '250px'})
flow_pred_w = wd.Checkbox(value=True, description='Incluir Flow Predictor',
                          indent=False, layout={'width': '250px'})
disp_w = wd.Checkbox(value=False, description='Incluir Dispersão',
                     indent=False, layout={'width': '250px'})
cta_weight_w = wd.ToggleButton(value=False, description='CTA: Peso por Janela',
                                tooltip='Ativo: sinais longos têm mais peso (estilo GS/BofA). '
                                        'Inativo: peso igual entre todas as janelas.',
                                button_style='info', icon='balance-scale',
                                layout={'width': '220px'})
cot_type_w = wd.Dropdown(
    options=list(COT_CONTRACTS.keys()),
    value='Equity Indices', description='COT Categoria:',
    layout={'width': '280px'})
cot_contract_w = wd.SelectMultiple(
    description='Contratos:', layout={'min_width': '300px', 'height': '80px'})
cot_trader_w = wd.Dropdown(
    options=['Total', 'Managed Money', 'Commercial', 'Non-Commercial',
             'Swap Dealers', 'Leveraged Funds', 'Asset Manager',
             'Dealer Intermediary', 'Other Reportables'],
    value='Total', description='Trader Type:',
    layout={'width': '280px'})
cot_start_w = wd.DatePicker(
    value=pd.Timestamp.now().date() - pd.Timedelta(days=2*365),
    description='COT Start:', layout={'width': '260px'})
cot_end_w = wd.DatePicker(
    value=pd.Timestamp.now().date(),
    description='COT End:', layout={'width': '260px'})

rebal_date_w = wd.DatePicker(
    value=_last_spx_rebal_date().date(),
    description='Último Rebalance:', layout={'width': '300px'})


def _update_cot_contracts(change=None):
    opts = COT_CONTRACTS.get(cot_type_w.value, [])
    cot_contract_w.options = opts
    if opts:
        cot_contract_w.value = (opts[0][1],)

cot_type_w.observe(_update_cot_contracts, names='value')
_update_cot_contracts()

out_main = wd.Output()
out_cot_reload = wd.Output()

# Botões de recarga parcial
cot_reload_btn = wd.Button(description='⟳ Recarregar COT',
                           button_style='info', icon='refresh',
                           layout={'width': '180px'})
etf_reload_btn = wd.Button(description='⟳ Recarregar ETFs',
                           button_style='info', icon='refresh',
                           layout={'width': '180px'})


def _reload_cot(_):
    """Recarrega apenas dados COT e exibe resultado."""
    with out_cot_reload:
        clear_output(wait=True)
        display(wd.HTML(DASH_CSS + "<div class='mm-dash mm-loading'>⏳ Recarregando COT...</div>"))
        ticker = ticker_w.value.strip() or 'SPX Index'
        _cot_start = cot_start_w.value.strftime('%Y%m%d') if cot_start_w.value else '-2Y'
        _cot_end = cot_end_w.value.strftime('%Y%m%d') if cot_end_w.value else '0D'
        try:
            cot_df = safe_fetch_cot(ticker, start=_cot_start, end=_cot_end)
            sel_cots = list(cot_contract_w.value)
            sel_df = None
            if sel_cots:
                try:
                    raw = fetch_cot_data(
                        sel_cots[0] if len(sel_cots) == 1 else sel_cots,
                        start=_cot_start, end=_cot_end)
                    if raw is not None and not raw.empty:
                        sel_df = aggregate_cot(raw)
                except Exception as e:
                    print(f"⚠️ COT selected: {e}")
            clear_output(wait=True)
            children = [wd.HTML(
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<h3>COT — Recarga Rápida</h3>"
                f"<p>Dados agregados (todas trader types somadas). "
                f"Report type: auto (tenta disaggregated → tff → legacy)</p>"
                f"</div></div>")]
            ok, fut = has_cot(ticker)
            if ok and cot_df is not None and not cot_df.empty:
                stats = cot_summary_stats(cot_df)
                children.append(wd.HTML(f"<p>Futures: <b>{fut}</b> — {len(cot_df)} registros</p>"))
                children.append(fp_grid_cot_stats(stats))
                seas = cot_seasonality(cot_df)
                children.append(wd.HBox([
                    fp_plot_positions_basket(cot_df),
                    fp_plot_dispersion(seas, cot_df)]))
                children.append(fp_plot_long_short_net(cot_df))
            elif ok:
                children.append(wd.HTML(f"<p>COT disponível para {fut}, mas sem dados.</p>"))
            if sel_df is not None and not sel_df.empty:
                sel_label = ', '.join(sel_cots)
                children.append(wd.HTML(f"<hr><h4>COT: {sel_label} — {len(sel_df)} registros</h4>"))
                children.append(fp_plot_positions_basket(sel_df))
            display(wd.VBox(children))
        except Exception as e:
            clear_output(wait=True)
            print(f"Erro COT: {e}")
            import traceback as _tb_mod; _tb_mod.print_exc()


def _reload_etfs(_):
    """Recarrega apenas fluxo de ETFs alavancados."""
    with out_cot_reload:
        clear_output(wait=True)
        display(wd.HTML(DASH_CSS + "<div class='mm-dash mm-loading'>⏳ Recarregando ETFs...</div>"))
        ticker = ticker_w.value.strip() or 'SPX Index'
        try:
            prices, lr = fetch_historical(ticker)
            nz = lr[lr != 0]
            dr = float(nz.iloc[-1]) if len(nz) > 0 else 0
            flows, total = compute_leveraged_flows(dr)
            clear_output(wait=True)
            html = (
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<h4>ETFs Alavancados (Retorno diário: {dr*100:+.2f}%)</h4>" +
                flows[['Leverage', 'AUM_$', 'Rebalance_$', 'Direção']]
                .style.format({'AUM_$': '${:,.0f}', 'Rebalance_$': '${:,.0f}'})
                .to_html() +
                f"<p><b>Fluxo Direcional Total: ${total:,.0f}</b></p>"
                f"</div></div>")
            display(wd.HTML(html))
        except Exception as e:
            clear_output(wait=True)
            print(f"Erro ETF: {e}")


cot_reload_btn.on_click(_reload_cot)
etf_reload_btn.on_click(_reload_etfs)


def run_analysis(_):
    """Callback principal: busca dados, calcula tudo, monta dashboard."""
    with out_main:
        clear_output(wait=True)
        loading = wd.HTML(DASH_CSS + "<div class='mm-dash mm-loading'>⏳ Inicializando...</div>")
        display(loading)

        ticker = ticker_w.value.strip() or 'SPX Index'
        min_dte, max_dte = dte_w.value
        mny_low, mny_high = mny_w.value

        try:
            # ── 1. Dados de Mercado ──────────────────────────────────────
            loading.value = "<h4>1/16: Buscando dados de mercado...</h4>"
            mkt = fetch_market_data(ticker)
            spot = mkt['spot']
            iv_30d = mkt['iv_30d']
            rv_30d = mkt['rv_30d']
            skew = mkt['skew']
            avg_vol = mkt['avg_dollar_volume']
            rfr = mkt.get('risk_free_rate', 0.0)
            move_index = mkt.get('move_index', np.nan)
            print(f"[MKT] risk-free rate = {rfr:.4f} (USGG3M), MOVE = {move_index}")

            if pd.isna(spot):
                raise ValueError(f"Spot inválido para {ticker}")

            # ── 2. Histórico + Modelo de Risco ───────────────────────────
            loading.value = "<h4>2/16: Modelagem de risco (t-Student)...</h4>"
            prices, log_returns = fetch_historical(ticker)
            risk = fit_risk_model(log_returns)

            # ── 3. Cadeia de Opções ──────────────────────────────────────
            loading.value = "<h4>3/16: Buscando cadeia de opções...</h4>"
            df, from_strike, to_strike = fetch_options_chain(
                ticker, spot, min_dte, max_dte, mny_low, mny_high)

            # ── 4. Gregas + Exposições ───────────────────────────────────
            loading.value = f"<h4>4/16: Calculando gregas para {len(df)} opções...</h4>"
            greeks_now = calculate_all_greeks(
                spot, df.Strike.values, df.IV.values, df.Tte.values, df.Type.values, r=rfr)
            agg = compute_strike_exposures(df, greeks_now, spot)
            call_wall, put_wall = compute_walls(agg)

            if call_wall == put_wall and call_wall is not None:
                print(f"⚠️ Call Wall = Put Wall = {call_wall:,.0f}")

            # ── 5. Curvas Modelo ─────────────────────────────────────────
            loading.value = "<h4>5/16: Calculando curvas modelo (100 níveis × 7 gregas)...</h4>"
            levels = np.linspace(from_strike, to_strike, 100)
            model_curves = compute_model_curves(df, levels, r=rfr)

            # Flip points
            flip_points = {}
            for cfg in GREEK_CONFIGS:
                curve = model_curves[cfg['name']]
                flip_points[cfg['name']] = calculate_flip(levels, curve)

            gamma_flip = flip_points.get('Gamma')
            gamma_curve = model_curves['Gamma']

            # ── 6. Matrizes de Sensibilidade ─────────────────────────────
            loading.value = "<h4>6/16: Matrizes de sensibilidade (7×5×7)...</h4>"
            sens_matrices = compute_sensitivity_matrices(df, spot, r=rfr)

            # ── 7. Monte Carlo ───────────────────────────────────────────
            loading.value = "<h4>7/16: Simulação Monte Carlo (10k cenários, 5 dias)...</h4>"
            mc_n_days = 5
            mc_pnl, mc_prices = run_monte_carlo(spot, df, risk, n_days=mc_n_days, r=rfr)

            # ── 8. Curvas de P&L ─────────────────────────────────────────
            loading.value = "<h4>8/16: Curvas de P&L e hedge demand...</h4>"
            pnl_curves = compute_pnl_curves(greeks_now, df, spot, levels, skew, r=rfr)

            # ── 9. Rebalanceamento ETFs ──────────────────────────────────
            loading.value = "<h4>9/16: Fluxo de rebalanceamento ETFs passivos...</h4>"
            try:
                _rebal_dt = rebal_date_w.value
                etf_flows, etf_summary, etf_start = compute_full_etf_flows(
                    start_date_override=_rebal_dt)
                etf_ok = True
            except Exception:
                etf_flows, etf_summary, etf_start, etf_ok = {}, None, None, False

            # ── 10. ETFs Alavancados ─────────────────────────────────────
            loading.value = "<h4>10/16: Fluxo de ETFs alavancados...</h4>"
            try:
                # Usar último retorno não-zero (evitar 0 de weekend/feriado)
                nz_rets = log_returns[log_returns != 0]
                daily_ret = float(nz_rets.iloc[-1]) if len(nz_rets) > 0 else 0
                lev_flows, lev_total = compute_leveraged_flows(daily_ret)
                lev_ok = True
            except Exception:
                lev_flows, lev_total, lev_ok = None, 0, False
                daily_ret = 0

            # ── 11. Previsão SPX (opcional — pesado) ─────────────────────
            loading.value = "<h4>11/16: Previsão de rebalanceamento SPX...</h4>"
            spx_pred_ok = False
            top_in_spx, top_out_spx = None, None
            if spx_pred_w.value and HAS_SKLEARN:
                try:
                    top_in_spx, top_out_spx, _ = build_spx_prediction()
                    spx_pred_ok = True
                except Exception as pred_err:
                    print(f"⚠️ Previsão SPX falhou: {pred_err}")

            # ── 12. Flow Predictor — Histórico + Buyback ─────────────────
            fp_ok = False
            fp_flow_hist = pd.DataFrame()
            fp_buyback = {'daily_est': 0, 'pct_adv_est': 0,
                          'confidence': 'none', 'announced': 0}
            fp_score = None
            fp_cot_df = None
            fp_cot_stats = pd.Series(dtype=float)
            fp_selected_cot_df = None

            if flow_pred_w.value:
                loading.value = "<h4>12/16: Flow Predictor — histórico + buyback...</h4>"
                try:
                    fp_flow_hist = build_flow_history(ticker, lookback=252)
                    fp_today_flow = (compute_leveraged_flow_simple(
                        float(fp_flow_hist['Return'].iloc[-1]))
                        if not fp_flow_hist.empty else 0)
                except Exception as fp_err:
                    print(f"⚠️ Flow hist: {fp_err}")
                    fp_today_flow = 0

                # Buyback: para índices, usar agregação por membros
                try:
                    if ticker.strip().endswith('Index'):
                        bb_df = estimate_index_buyback_flow(ticker, top_n=30)
                        fp_buyback_daily = float(bb_df['daily_est'].sum()) if (
                            not bb_df.empty and 'daily_est' in bb_df.columns) else 0
                        fp_buyback = {
                            'daily_est': fp_buyback_daily,
                            'pct_adv_est': 0,
                            'confidence': 'estimated',
                            'announced': float(bb_df['buyback'].sum()) if (
                                not bb_df.empty and 'buyback' in bb_df.columns) else 0,
                        }
                    else:
                        fp_buyback = estimate_buyback_flow(ticker)
                        fp_buyback_daily = fp_buyback.get('daily_est', 0)
                except Exception as fp_err:
                    print(f"⚠️ Buyback: {fp_err}")
                    fp_buyback_daily = 0

                # Blackout window: buscar datas de earnings e calcular % em restrição
                fp_earnings_df = pd.DataFrame()
                fp_blackout_pct = 0.0
                fp_blackout_n = 0
                fp_blackout_total = 0
                fp_blackout_curve = pd.DataFrame()
                try:
                    loading.value = "<h4>12b/16: Blackout window — earnings dates...</h4>"
                    fp_earnings_df = fetch_earnings_dates(ticker if ticker.strip().endswith('Index') else 'SPX Index')
                    if not fp_earnings_df.empty:
                        fp_blackout_pct, fp_blackout_n, fp_blackout_total = blackout_pct_today(fp_earnings_df)
                        fp_blackout_curve = compute_blackout_curve(fp_earnings_df, n_days_forward=365)
                        # Ajustar buyback diário pela janela de blackout
                        fp_buyback['blackout_pct'] = fp_blackout_pct
                        fp_buyback['blackout_n'] = fp_blackout_n
                        fp_buyback['blackout_total'] = fp_blackout_total
                        fp_buyback['daily_est_open'] = fp_buyback.get('daily_est', 0)
                        fp_buyback['daily_est'] = fp_buyback['daily_est_open'] * (1 - fp_blackout_pct)
                        fp_buyback_daily = fp_buyback['daily_est']
                except Exception as bo_err:
                    print(f"⚠️ Blackout: {bo_err}")

                # ── 13. Flow Predictor — COT ─────────────────────────────
                loading.value = "<h4>13/16: Flow Predictor — COT...</h4>"
                cot_ok_fp, cot_fut_fp = has_cot(ticker)
                cot_net_change = 0
                history_cot = None

                _cot_start = (cot_start_w.value.strftime('%Y%m%d')
                              if cot_start_w.value else '-2Y')
                _cot_end = (cot_end_w.value.strftime('%Y%m%d')
                            if cot_end_w.value else '0D')

                if cot_ok_fp:
                    try:
                        fp_cot_df = safe_fetch_cot(ticker,
                                                   start=_cot_start,
                                                   end=_cot_end)
                        if fp_cot_df is not None and not fp_cot_df.empty:
                            fp_cot_stats = cot_summary_stats(fp_cot_df)
                            # Pega net change de qualquer coluna disponível
                            _net_col = None
                            for _nc in ('Positions - Net', 'Positions'):
                                if _nc in fp_cot_df.columns:
                                    _net_col = _nc
                                    break
                            if _net_col:
                                net = fp_cot_df[_net_col].dropna()
                                if len(net) >= 2:
                                    cot_net_change = float(
                                        net.iloc[-1] - net.iloc[-2])
                                    history_cot = net.diff().dropna()
                    except Exception:
                        pass

                # COT de contratos selecionados
                selected_cots = list(cot_contract_w.value)
                if selected_cots:
                    try:
                        sel_df = fetch_cot_data(
                            selected_cots[0] if len(selected_cots) == 1
                            else selected_cots,
                            start=_cot_start, end=_cot_end)
                        if not sel_df.empty:
                            fp_selected_cot_df = aggregate_cot(sel_df)
                    except Exception:
                        pass

                # ── 14. Flow Score Combinado ─────────────────────────────
                loading.value = "<h4>14/16: Calculando flow score...</h4>"

                # Dealer/MM hedging flow (baseado em GEX)
                fp_dealer_flow = 0
                try:
                    _gex_per_pt = (greeks_now['gamma']
                                   * np.where(df.Type.values == 'Call', 1, -1)
                                   * df['OI'].values * 100).sum()
                    _daily_chg = spot * daily_ret
                    fp_dealer_flow = compute_dealer_hedging_flow(
                        _gex_per_pt, _daily_chg, spot)
                    print(f"[FLOW] Dealer flow: ${fp_dealer_flow:,.0f} "
                          f"(GEX/pt={_gex_per_pt:,.0f}, ΔS={_daily_chg:,.1f})")
                except Exception as e:
                    print(f"⚠️ Dealer flow: {e}")

                # Market Maker VaR by individual book
                fp_mm_var = []
                fp_mm_var_totals = {}
                fp_vol_data = {'total_adc': OPTIONS_TOTAL_ADC, 'source': 'fallback',
                               'call_vol': 0, 'put_vol': 0, 'pc_ratio': 0}
                try:
                    fp_vol_data = fetch_options_volume_bql(ticker)
                    print(f"[FLOW] Options volume: {fp_vol_data['total_adc']:,.0f} "
                          f"(source={fp_vol_data['source']})")
                except Exception as e:
                    print(f"⚠️ Options volume BQL: {e}")
                try:
                    _oi_total = df['OI'].sum()
                    _theta_total = (greeks_now['theta'] * df['OI'].values * 100).sum()
                    fp_mm_var, fp_mm_var_totals = estimate_mm_var_by_book(
                        _gex_per_pt, spot, risk, _oi_total)
                    # Fill theta per MM
                    for mm in fp_mm_var:
                        mm['daily_theta'] = _theta_total * mm['share']
                    print(f"[FLOW] MM VaR: {len(fp_mm_var)} MMs, "
                          f"VaR95=${fp_mm_var_totals.get('pnl_var95', 0):,.0f}")
                except Exception as e:
                    print(f"⚠️ MM VaR: {e}")

                # Vol control fund flows (5%, 10%, 15%)
                fp_volctrl = {'total': 0, 'detail': {}}
                fp_vc_scenarios = []
                fp_combined_scenarios = []
                try:
                    _rv_window = 21
                    _rets = log_returns.iloc[-_rv_window * 2:]
                    _rv_cur = _rets.iloc[-_rv_window:].std() * np.sqrt(252)
                    _rv_prev = _rets.iloc[-_rv_window * 2:-_rv_window].std() * np.sqrt(252)
                    fp_volctrl = compute_vol_control_flow(_rv_cur, _rv_prev)
                    fp_vc_scenarios = compute_vol_control_scenarios(_rv_cur)
                    print(f"[FLOW] Vol ctrl: ${fp_volctrl['total']:,.0f} "
                          f"(RV cur={_rv_cur:.2%}, prev={_rv_prev:.2%})")
                except Exception as e:
                    print(f"⚠️ Vol ctrl: {e}")

                # CTA trend following flow
                fp_cta = {'flow': 0, 'trend_today': 0, 'pos_today': 0, 'pos_prev': 0}
                fp_cta_scenarios_1w = []
                fp_cta_scenarios_1m = []
                fp_cta_pivots = []
                fp_cta_hist = pd.DataFrame()
                try:
                    # Reconstruir preços a partir de log returns
                    _px_series = np.exp(np.cumsum(log_returns))
                    _px_series = _px_series * (spot / _px_series.iloc[-1])
                    _cta_rv = log_returns.iloc[-63:].std() * np.sqrt(252) if len(log_returns) >= 63 else rv_30d
                    fp_cta = compute_cta_flow(_px_series, _cta_rv)
                    print(f"[FLOW] CTA: ${fp_cta['flow']:,.0f} "
                          f"(trend={fp_cta['trend_today']:+.3f}, "
                          f"pos={fp_cta['pos_today']:+.3f}→{fp_cta['pos_prev']:+.3f})")
                except Exception as e:
                    print(f"⚠️ CTA flow: {e}\n{traceback.format_exc()}")

                # CTA: scenarios, pivots, historical
                try:
                    if len(_px_series) < 201:
                        _px_series = np.exp(np.cumsum(log_returns))
                        _px_series = _px_series * (spot / _px_series.iloc[-1])
                except Exception:
                    _px_series = np.exp(np.cumsum(log_returns))
                    _px_series = _px_series * (spot / _px_series.iloc[-1])
                try:
                    if '_cta_rv' not in dir() or _cta_rv < 1e-6:
                        _cta_rv = log_returns.iloc[-63:].std() * np.sqrt(252) if len(log_returns) >= 63 else rv_30d
                except Exception:
                    _cta_rv = rv_30d
                try:
                    print(f"[FLOW] CTA GS: px_series len={len(_px_series)}, rv={_cta_rv:.4f}")
                    fp_cta_scenarios_1w = compute_cta_scenario_flows(
                        _px_series, _cta_rv, spot, horizon_days=5)
                    print(f"[FLOW] CTA scenarios 1W: {len(fp_cta_scenarios_1w)}")

                    fp_cta_scenarios_1m = compute_cta_scenario_flows(
                        _px_series, _cta_rv, spot, horizon_days=21)
                    print(f"[FLOW] CTA scenarios 1M: {len(fp_cta_scenarios_1m)}")

                    fp_cta_pivots = compute_cta_pivot_levels(_px_series, spot, _cta_rv)
                    print(f"[FLOW] CTA pivots: {len(fp_cta_pivots)}")

                    loading.value = "<h4>14b/16: CTA — histórico de posições...</h4>"
                    fp_cta_hist = compute_cta_historical_positions(_px_series, lookback=126)
                    print(f"[FLOW] CTA hist: {len(fp_cta_hist)} rows")
                except Exception as e:
                    print(f"⚠️ CTA: {e}\n{traceback.format_exc()}")

                # Risk Parity flow
                fp_rp = {'total': 0, 'detail': {}, 'eq_alloc_new': 0, 'eq_alloc_old': 0}
                try:
                    _rv_window = 21
                    _rets_rp = log_returns.iloc[-_rv_window * 2:]
                    _rv_eq_cur = _rets_rp.iloc[-_rv_window:].std() * np.sqrt(252)
                    _rv_eq_prev = _rets_rp.iloc[-_rv_window * 2:-_rv_window].std() * np.sqrt(252)
                    # Bond vol from MOVE Index (yield vol → price vol, ~7yr duration)
                    _bond_vol = (move_index / 10000) * 7 if move_index and move_index > 0 else None
                    fp_rp = compute_risk_parity_flow(_rv_eq_cur, _rv_eq_prev,
                                                     rv_bonds=_bond_vol, rv_bonds_prev=_bond_vol)
                    print(f"[FLOW] Risk Parity: ${fp_rp['total']:,.0f} "
                          f"(eq_alloc={fp_rp['eq_alloc_new']:.1%}→{fp_rp['eq_alloc_old']:.1%})")
                except Exception as e:
                    print(f"⚠️ Risk Parity: {e}")

                # Combined flow scenarios (vol spike → all components)
                try:
                    _px_s = _px_series if '_px_series' in dir() else None
                    _gex = _gex_per_pt if '_gex_per_pt' in dir() else 0
                    _oi_100 = df['OI'].values * 100.0
                    _vanna_not = np.nansum(greeks_now['vanna'] * _oi_100) * spot
                    _vega_not = np.nansum(greeks_now['vega'] * _oi_100)
                    _charm_not = np.nansum(greeks_now['charm'] * _oi_100) * spot / 365.0
                    fp_combined_scenarios = compute_combined_flow_scenarios(
                        _rv_cur, prices=_px_s, gex_per_pt=_gex, spot=spot,
                        vanna_notional=_vanna_not, vega_notional=_vega_not,
                        charm_notional=_charm_not)
                except Exception as e:
                    print(f"⚠️ Combined scenarios: {e}")

                # Passive ETF flow — net rebalancing from VOO/SPY/IVV
                fp_passive_etf_flow = 0
                try:
                    if etf_ok and etf_flows:
                        combo = etf_flows.get('Combined', pd.DataFrame())
                        if not combo.empty and 'Flow_$' in combo.columns:
                            fp_passive_etf_flow = float(combo['Flow_$'].sum())
                            print(f"[FLOW] ETFs Passivos: ${fp_passive_etf_flow:,.0f} "
                                  f"(buy={combo.loc[combo['Flow_$']>0, 'Flow_$'].sum():,.0f}, "
                                  f"sell={combo.loc[combo['Flow_$']<0, 'Flow_$'].sum():,.0f})")
                except Exception as e:
                    print(f"⚠️ ETFs Passivos: {e}")

                try:
                    lev_history = (fp_flow_hist['LevETF_Flow']
                                   if not fp_flow_hist.empty else None)
                    fp_score = compute_flow_score(
                        leveraged_flow=fp_today_flow,
                        buyback_daily=fp_buyback_daily,
                        cot_net_change=cot_net_change,
                        passive_etf_flow=fp_passive_etf_flow,
                        history_leveraged=lev_history,
                        history_cot=history_cot,
                        dealer_flow=fp_dealer_flow,
                        volctrl_flow=fp_volctrl['total'],
                        cta_flow=fp_cta['flow'],
                        rp_flow=fp_rp['total'],
                        buyback_blackout_pct=fp_buyback.get('blackout_pct', 0))
                    fp_ok = True
                except Exception as fp_err:
                    print(f"⚠️ Flow score: {fp_err}")

            # ── P/C OI Ratio (standalone, always fetched from BBG) ───────
            # fp_vol_data may already be populated by Flow Predictor block;
            # if not (FP disabled or query failed), fetch it now so that
            # pc_ratio is always a real BBG value, never hardcoded.
            if not (isinstance(fp_vol_data, dict) and fp_vol_data.get('pc_ratio', 0) > 0):
                try:
                    _pc_tmp = fetch_options_volume_bql(ticker)
                    if _pc_tmp.get('pc_ratio', 0) > 0:
                        fp_vol_data = _pc_tmp
                        print(f"[PC] P/C OI ratio (standalone): {_pc_tmp['pc_ratio']:.2f}")
                except Exception as _pc_err:
                    print(f"⚠️ P/C ratio standalone fetch: {_pc_err}")

            # ── 15. Dispersion Trade + Tail Risk ─────────────────────────
            disp_result = {
                'error': None, 'disp_signal': pd.DataFrame(),
                'real_corr': pd.DataFrame(),
                'impl_corr_cboe': pd.Series(dtype=float),
                'mag7_pairs': pd.DataFrame(), 'best_2x2': [], 'best_pairs': [],
                'optimal_basket': {}, 'tail_risk': {},
                'index_returns': np.array([]),
                'hyp_test': {},
                'cor1m': pd.Series(dtype=float),
                'dspx': pd.Series(dtype=float),
                'vixeq': pd.Series(dtype=float),
            }
            disp_ok = False
            if disp_w.value:
                loading.value = "<h4>15/16: Dispersion Trade + Tail Risk (BQL)...</h4>"
                try:
                    disp_result = run_dispersion_analysis(
                        index_ticker=ticker, lookback=252)
                    if disp_result['error'] is None:
                        disp_ok = True
                    else:
                        print(f"⚠️ Dispersion: {disp_result['error']}")
                except Exception as disp_err:
                    print(f"⚠️ Dispersion: {disp_err}")

            # ── 16. Advanced Analytics (Skew, Tail Gauge, Dealer MC, OPEX) ──
            analytics = {
                'skew_df': pd.DataFrame(),
                'skew_summary': {},
                'spot_vol_up': {'current_streak': 0, 'max_streak': 0, 'total_days': 0, 'pct_up_up': 0, 'history': pd.Series(dtype=int)},
                'vix_reg': {},
                'dealer_mc': {},
                'opex_stats': {},
                'tail_score': 50, 'tail_components': {}, 'tail_interp': '',
                'dealer_scenarios': pd.DataFrame(),
                'mag8_scenarios': pd.DataFrame(),
                'vol_rebal': pd.DataFrame(),
                'gamma_vol': {},
            }
            try:
                loading.value = "<h4>16/16: Advanced Analytics...</h4>"

                # Skew monitor
                try:
                    analytics['skew_df'] = fetch_skew_metrics(ticker, lookback=252)
                    analytics['skew_summary'] = compute_skew_summary(analytics['skew_df'])
                except Exception as _sk_err:
                    print(f"⚠️ Skew: {_sk_err}")

                # Spot-Up-Vol-Up: fetch VIX
                try:
                    bq = bql.Service()
                    dt_rng = bq.func.range('-504d', '0d')
                    vix_req = bql.Request('VIX Index', {
                        'px': bq.data.px_last(fill='PREV', dates=dt_rng),
                    })
                    vix_resp = bq.execute(vix_req)
                    vix_s = _bql_ts(vix_resp[0], 'px').dropna()
                    vix_changes = vix_s.diff().dropna()
                    analytics['spot_vol_up'] = compute_spot_up_vol_up(log_returns, vix_changes)
                    analytics['vix_reg'] = compute_vix_spx_regression(log_returns, vix_changes)
                except Exception as _vix_err:
                    print(f"⚠️ VIX/Spot-Vol-Up: {_vix_err}")

                # Per-dealer MC
                try:
                    analytics['dealer_mc'] = run_dealer_monte_carlo(
                        spot, df, risk, n_sims=10000, n_days=mc_n_days, r=rfr)
                except Exception as _dmc_err:
                    print(f"⚠️ Dealer MC: {_dmc_err}")

                # OPEX stats
                try:
                    analytics['opex_stats'] = compute_opex_stats(log_returns, lookback_years=5)
                except Exception as _opex_err:
                    print(f"⚠️ OPEX: {_opex_err}")

                # Dealer scenario matrices
                try:
                    analytics['dealer_scenarios'] = compute_dealer_scenario_matrix(
                        spot, df, greeks_now)
                    analytics['mag8_scenarios'] = compute_mag8_dealer_scenarios(
                        spot, df, greeks_now)
                except Exception as _dsm_err:
                    print(f"⚠️ Dealer Scenarios: {_dsm_err}")

                # Vol Control rebalance projection
                oi100_vc = df.OI.values * 100
                cs_vc = np.where(df.Type.values == 'Call', 1, -1)
                _gex_pt = (greeks_now['gamma'] * cs_vc * oi100_vc).sum()
                _dex_vc = (greeks_now['delta'] * oi100_vc).sum()
                _vanna_not = float(np.nansum(greeks_now['vanna'] * oi100_vc) * spot)
                try:
                    analytics['vol_rebal'] = compute_vol_rebalance_projection(
                        rv_30d if pd.notna(rv_30d) else 0.15, spot,
                        gex_per_pt=_gex_pt, vanna_notional=_vanna_not, dex=_dex_vc)
                except Exception as _vr_err:
                    print(f"⚠️ Vol Rebal: {_vr_err}")

                # Tail risk gauge
                suvu_streak = analytics['spot_vol_up'].get('current_streak', 0)
                analytics['tail_score'], analytics['tail_components'], analytics['tail_interp'] = \
                    compute_tail_risk_gauge(
                        log_returns,
                        iv_30d=iv_30d if pd.notna(iv_30d) else None,
                        rv_30d=rv_30d if pd.notna(rv_30d) else None,
                        skew_summary=analytics['skew_summary'],
                        spot_vol_up_streak=suvu_streak)

            except Exception as _analytics_err:
                print(f"⚠️ Analytics: {_analytics_err}")

            # ── Gamma History (CSV database) ─────────────────────────────
            gamma_hist = pd.DataFrame()
            try:
                gamma_hist = load_gamma_history()
                if not gamma_hist.empty:
                    print(f"[GAMMA DB] {len(gamma_hist)} rows loaded "
                          f"({gamma_hist['date'].iloc[0].strftime('%Y-%m-%d')} → "
                          f"{gamma_hist['date'].iloc[-1].strftime('%Y-%m-%d')})")
                else:
                    print(f"[GAMMA DB] Empty — path={GAMMA_HISTORY_PATH}, exists={os.path.exists(GAMMA_HISTORY_PATH)}")
            except Exception as _gh_err:
                print(f"⚠️ Gamma History load: {_gh_err}")
            # Atualização do banco de dados desativada — preencher manualmente no CSV.

            clear_output(wait=True)

            # ═════════════════════════════════════════════════════════════
            # MONTAGEM DAS ABAS DO DASHBOARD
            # ═════════════════════════════════════════════════════════════

            title_html = wd.HTML(
                DASH_CSS +
                f"<div class='mm-dash'>"
                f"<div class='mm-title'>Market Maker Dashboard "
                f"<small>{ticker} @ {spot:,.2f} │ {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>"
                f"</div></div>")

            # ─── ABA 1: VISÃO GERAL ─────────────────────────────────────
            total_gex = gamma_curve[np.argmin(np.abs(levels - spot))]
            # Fragilidade = GEX / (notional diário de opções SPX + futuros ES)
            # Opções: total de contratos ADV × 100 shares × spot
            _opt_adc = fp_vol_data.get('total_adc', OPTIONS_TOTAL_ADC)
            # options_notional: volume ADC em lotes × spot (sem ×100 — ADC já em unidades de share equiv.)
            _opt_notional = _opt_adc * spot                  # ~$145B @ SPX=6700, ADC=21.7M
            # Futuros ES: ADV ~400k contratos × $50/ponto × spot = ~$134B @ SPX=6700
            _fut_notional = 400_000 * 50 * spot
            _daily_flow_cap = _opt_notional + _fut_notional
            # Fragilidade = GEX ($ / 1%move) como % do fluxo diário total (opções + futuros)
            fragility = (abs(total_gex) / _daily_flow_cap * 100
                         if _daily_flow_cap > 0 else 0)

            # Diagnóstico GEX modelo completo (todos vencimentos)
            _gc_is_call = df.Type.values == 'Call'
            _gc_is_put  = df.Type.values == 'Put'
            _gc_oi100   = df['OI'].values * 100.0
            _gc_g       = greeks_now['gamma']
            _gc_calls_abs = float(np.nansum(_gc_g[_gc_is_call] * _gc_oi100[_gc_is_call] * spot**2 * 0.01)) / 1e9
            _gc_puts_abs  = float(np.nansum(_gc_g[_gc_is_put]  * _gc_oi100[_gc_is_put]  * spot**2 * 0.01)) / 1e9
            print(f"[GEX FULL] OI calls: {df[_gc_is_call].OI.sum():,.0f} | OI puts: {df[_gc_is_put].OI.sum():,.0f} | "
                  f"Expirations: {df['Tte'].nunique() if 'Tte' in df.columns else '?'} | "
                  f"Options total: {len(df)}")
            print(f"[GEX FULL] Gamma absoluto (bbg-like): {_gc_calls_abs + _gc_puts_abs:+.3f}Bn | "
                  f"Calls: {_gc_calls_abs:+.3f}Bn | Puts: {_gc_puts_abs:+.3f}Bn | "
                  f"NET (calls-puts): {_gc_calls_abs - _gc_puts_abs:+.3f}Bn | "
                  f"Curva no spot: {total_gex/1e9:+.3f}Bn")
            daily_move = implied_move_pct(iv_30d) if pd.notna(iv_30d) else 0
            vol_premium = (iv_30d - rv_30d) * 100 if pd.notna(iv_30d) and pd.notna(rv_30d) else 0

            _frag_max = max(20, round(fragility * 1.5 / 5) * 5)  # arredonda p/ múltiplo de 5
            g_frag = create_gauge(fragility, "Fragilidade GEX/Fluxo",
                                  0, _frag_max, _C['red'], "%",
                                  steps=[
                                      {'range': [0,           _frag_max * 0.25], 'color': '#1a3a2a'},
                                      {'range': [_frag_max * 0.25, _frag_max * 0.6],  'color': '#3a3520'},
                                      {'range': [_frag_max * 0.6,  _frag_max],        'color': '#3a1a1a'},
                                  ])
            _vol_hi = max(5, round(abs(vol_premium) * 1.4))
            g_vol = create_gauge(vol_premium, "Prêmio Vol (IV-RV)",
                                 -_vol_hi, _vol_hi, _C['orange'], "%")
            _skew_raw = skew * 100
            # Clamp outliers de BQL (dados anômalos excedem ±25pp)
            _skew_val = float(np.clip(_skew_raw, -25, 25))
            _skew_hi = max(15, abs(_skew_val) * 1.3)
            g_skew = create_gauge(_skew_val, "Skew (P25-C25)",
                                  -_skew_hi, _skew_hi, _C['teal'], "%")
            _move_hi = max(5, daily_move * 1.3)
            g_move = create_gauge(daily_move, "Mov. Esperado 1D",
                                  0, _move_hi, _C['green'], "%")

            # GEX curve (Plotly)
            fig_gex = go.FigureWidget()
            fig_gex.add_trace(go.Scatter(
                x=levels, y=gamma_curve / 1e10, mode='lines',
                fill='tozeroy', line_color=_C['accent'],
                fillcolor='rgba(88,166,255,0.15)', name='GEX'))
            fig_gex.add_vline(x=spot, line_dash="dash", line_color=_C['red'],
                              annotation_text=f"Spot {spot:,.0f}")
            if gamma_flip:
                fig_gex.add_vline(x=gamma_flip, line_color=_C['orange'],
                                  annotation_text=f"G-Flip {gamma_flip:,.0f}")
            fig_gex.update_layout(title="Gamma Exposure (GEX)",
                                  yaxis_title="$ Bi / 1% move",
                                  height=350, width=480, template=DASH_TEMPLATE,
                                  margin=dict(t=35, b=25), showlegend=False)

            # Return distribution (Plotly)
            fig_dist = go.FigureWidget()
            fig_dist.add_trace(go.Histogram(
                x=log_returns, histnorm='probability density',
                name='Reais', marker_color=_C['accent'], opacity=0.6,
                xbins=dict(size=log_returns.std() / 4)))
            x_pdf = np.linspace(log_returns.min(), log_returns.max(), 500)
            pdf_vals = student_t.pdf(x_pdf, risk['tdf'], risk['tloc'], risk['tscale'])
            fig_dist.add_trace(go.Scatter(
                x=x_pdf, y=pdf_vals, mode='lines',
                name='t-Student', line_color=_C['orange']))
            fig_dist.add_vline(x=risk['var_95'], line_dash="dash",
                               line_color=_C['orange'],
                               annotation_text=f"VaR 95% ({risk['var_95']:.2%})")
            fig_dist.add_vline(x=risk['var_99'], line_dash="dash",
                               line_color=_C['red'],
                               annotation_text=f"VaR 99% ({risk['var_99']:.2%})")
            _xlo = max(log_returns.quantile(0.005), -0.08)
            _xhi = min(log_returns.quantile(0.995), 0.08)
            fig_dist.update_layout(title="Distribuição de Retornos",
                                   yaxis_title="Prob.",
                                   xaxis_tickformat=".1%",
                                   xaxis_range=[_xlo, _xhi],
                                   height=350, width=480, template=DASH_TEMPLATE,
                                   margin=dict(t=35, b=25))

            # Sumário de vol e risco
            summary_html = f"""
            <div class='mm-dash'><div class='mm-card'>
                <div class='mm-kpi-row'>
                    <div class='mm-kpi'><div class='kpi-label'>IV 30d ATM</div><div class='kpi-value' style='color:{_C["accent"]}'>{iv_30d:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>RV 30d</div><div class='kpi-value' style='color:{_C["teal"]}'>{rv_30d:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>Prêmio</div><div class='kpi-value' style='color:{_C["orange"]}'>{vol_premium:+.2f}%</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>Skew</div><div class='kpi-value' style='color:{_C["purple"]}'>{skew:+.2%}</div></div>
                </div>
                <div class='mm-section-label'>Risco Caudal</div>
                <div class='mm-kpi-row'>
                    <div class='mm-kpi'><div class='kpi-label'>VaR 95%</div><div class='kpi-value' style='color:{_C["yellow"]}'>{risk['var_95']:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>CVaR 95%</div><div class='kpi-value' style='color:{_C["orange"]}'>{risk['cvar_95']:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>VaR 99%</div><div class='kpi-value' style='color:{_C["red"]}'>{risk['var_99']:.2%}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>CVaR 99%</div><div class='kpi-value' style='color:{_C["red"]}'>{risk['cvar_99']:.2%}</div></div>
                </div>
                <div class='mm-section-label'>Posicionamento</div>
                <div class='mm-kpi-row'>
                    <div class='mm-kpi'><div class='kpi-label'>Gamma Flip</div><div class='kpi-value'>~{f'{gamma_flip:,.0f}' if gamma_flip else 'N/A'}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>Call Wall</div><div class='kpi-value' style='color:{_C["green"]}'>{f'{call_wall:,.0f}' if call_wall else 'N/A'}</div></div>
                    <div class='mm-kpi'><div class='kpi-label'>Put Wall</div><div class='kpi-value' style='color:{_C["red"]}'>{f'{put_wall:,.0f}' if put_wall else 'N/A'}</div></div>
                </div>
            </div></div>"""

            # ══ Dimensões fixas — NUNCA alterar sem redesign completo ═
            _GW, _GH   = 210, 190   # gauge  width × height
            _DH        = 250        # detail panel height

            # Helper: envolve qualquer widget numa célula de tamanho fixo
            def _cell(widget, w, h=None):
                kw = dict(width=f'{w}px', min_width=f'{w}px', max_width=f'{w}px',
                          overflow='hidden')
                if h:
                    kw.update(height=f'{h}px', min_height=f'{h}px', max_height=f'{h}px')
                return wd.Box([widget], layout=wd.Layout(**kw))

            # ── Tail Risk ──────────────────────────────────────────────
            _home_tail_gauge = build_tail_gauge(
                analytics.get('tail_score', 50),
                analytics.get('tail_interp', ''))
            _home_tail_gauge.update_layout(width=_GW, height=_GH,
                                           margin=dict(t=40, b=8, l=15, r=15))
            _tail_score_val = analytics.get('tail_score', 50)
            _tail_interp    = analytics.get('tail_interp', '')
            _tail_rows = []
            for _ck, _cv in analytics.get('tail_components', {}).items():
                _tail_rows.append(
                    f"<tr><td style='color:{_C['text_muted']};padding:3px 8px;font-size:11px;'>"
                    f"{_cv.get('label', _ck)}</td>"
                    f"<td style='color:{_C['text']};padding:3px 8px;font-size:11px;font-weight:700;'>"
                    f"{_cv.get('value', 0)}</td>"
                    f"<td style='color:{_C['yellow']};padding:3px 8px;font-size:11px;'>"
                    f"{_cv.get('score', 0):.1f}</td></tr>")
            _tail_detail_html = (
                f"<div class='mm-dash'><div class='mm-card' style='min-width:280px;'>"
                f"<h3>Tail Risk — {_tail_score_val:.0f}/100</h3>"
                f"<p style='margin:0 0 8px;'>{_tail_interp}</p>"
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<tr><th style='font-size:9px;color:{_C['text_dim']};padding:3px 8px;"
                f"text-align:left;border-bottom:1px solid {_C['border']};'>Componente</th>"
                f"<th style='font-size:9px;color:{_C['text_dim']};padding:3px 8px;"
                f"border-bottom:1px solid {_C['border']};'>Valor</th>"
                f"<th style='font-size:9px;color:{_C['text_dim']};padding:3px 8px;"
                f"border-bottom:1px solid {_C['border']};'>Score</th></tr>"
                + ''.join(_tail_rows)
                + f"</table></div></div>")

            # ── Flow Predictor ─────────────────────────────────────────
            _fp_gauge_w = wd.HTML(
                f"<div class='mm-dash'><div class='mm-card' style='width:{_GW}px;height:{_GH}px;"
                f"display:flex;align-items:center;justify-content:center;'>"
                f"<p style='color:{_C['text_muted']};font-size:11px;'>Flow N/A</p></div></div>")
            _fp_comps_w = wd.HTML(
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<p style='color:{_C['text_muted']};font-size:11px;'>Ative o Flow Predictor</p>"
                f"</div></div>")
            if fp_ok and fp_score is not None:
                try:
                    _fp_gauge_w = fp_plot_score_gauge(fp_score)
                    _fp_gauge_w.update_layout(width=_GW, height=_GH,
                                              margin=dict(t=40, b=8, l=15, r=15))
                    _fp_comps_w = fp_plot_components_bar(fp_score)
                    _fp_comps_w.update_layout(height=_DH,
                                              margin=dict(t=32, b=40, l=10, r=20))
                except Exception:
                    pass

            # ── CTA Chart ─────────────────────────────────────────────
            _home_cta = wd.HTML(
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<p style='color:{_C['text_muted']};'>CTA: ative o Flow Predictor</p>"
                f"</div></div>")
            if fp_ok and not fp_cta_hist.empty:
                try:
                    _cta_fig = build_cta_gs_chart(
                        fp_cta_hist, fp_cta_scenarios_1w, fp_cta_scenarios_1m, spot)
                    _home_cta = wd.Output()
                    with _home_cta:
                        _cta_fig.show()
                except Exception:
                    pass

            # ── Gregas + ETF flow ──────────────────────────────────────
            try:
                _greek_overview = build_greek_overview(
                    greeks_now, df, spot,
                    etf_flows=etf_flows if etf_ok else {})
            except Exception as _go_err:
                print(f"⚠️ Greek overview: {_go_err}")
                _greek_overview = wd.HTML('')

            # ── Gamma Squeeze ──────────────────────────────────────────
            _sq_gauge_w    = wd.HTML('')
            _sq_comps_w    = wd.HTML('')
            _sq_badge_w    = wd.HTML('')
            _sq_score_disp = 'N/A'
            _sq_ac         = _C['text_muted']
            try:
                _sq_pc_v1  = fp_vol_data.get('pc_ratio', 0) or 0  # 0 = BBG unavailable
                _sq_gex_v1 = total_gex_val / 1e9 if 'total_gex_val' in dir() else (
                              total_gex / 1e9    if 'total_gex'     in dir() else 0)
                _sq_result_v1 = compute_gamma_squeeze_score(
                    net_gex_bn=_sq_gex_v1, pc_ratio=_sq_pc_v1,
                    iv_30d=iv_30d, rv_30d=rv_30d, gamma_flip=gamma_flip,
                    spot=spot, skew=skew, put_wall=put_wall, call_wall=call_wall)
                _sq_gauge_w, _sq_comps_w, _sq_badge_str, _sq_ac = \
                    build_squeeze_mini_panel(_sq_result_v1, _C)
                _sq_gauge_w.update_layout(width=_GW, height=_GH,
                                          margin=dict(t=38, b=8, l=18, r=18))
                _sq_comps_w.update_layout(height=_DH,
                                          margin=dict(t=30, b=20, l=5, r=60))
                _sq_badge_w    = wd.HTML(_sq_badge_str)
                _sq_score_disp = f"{_sq_result_v1['score']:.0f}"
            except Exception as _sqm_err:
                print(f"⚠️ Squeeze mini: {_sqm_err}")

            # ── Status bar ─────────────────────────────────────────────
            _flip_str = f"{gamma_flip:,.0f}"  if gamma_flip      else "N/A"
            _gex_disp = _sq_gex_v1 * 0.1 if '_sq_gex_v1' in dir() else None
            _gex_str  = f"{_gex_disp:+.1f}B" if _gex_disp is not None else "N/A"
            _pc_str   = (f"{_sq_pc_v1:.2f}×" if ('_sq_pc_v1' in dir() and _sq_pc_v1 > 0) else "N/D")
            _ivrv_str = f"{(iv_30d - rv_30d)*100:+.1f}pp"

            def _stat(label, value, color):
                return (f"<div class='mm-stat-item'>"
                        f"<span class='mm-stat-label'>{label}</span>"
                        f"<span class='mm-stat-value' style='color:{color};'>{value}</span>"
                        f"</div>")

            _status_bar = wd.HTML(
                f"<div class='mm-dash mm-statusbar'>"
                "<div class='jarvis-reactor'>"
                "<svg width='42' height='42' viewBox='0 0 42 42'>"
                "<defs><filter id='rfglow'><feGaussianBlur stdDeviation='2' result='b'/>"
                "<feMerge><feMergeNode in='b'/><feMergeNode in='SourceGraphic'/></feMerge></filter></defs>"
                "<circle cx='21' cy='21' r='18' fill='none' stroke='rgba(0,200,255,.22)' stroke-width='1' stroke-dasharray='6 4' class='jarvis-r1'/>"
                "<circle cx='21' cy='21' r='13' fill='none' stroke='rgba(0,200,255,.42)' stroke-width='1' stroke-dasharray='4 3' class='jarvis-r2'/>"
                "<circle cx='21' cy='21' r='7'  fill='none' stroke='rgba(0,200,255,.62)' stroke-width='1' stroke-dasharray='2 2' class='jarvis-r3'/>"
                "<circle cx='21' cy='21' r='3.5' fill='rgba(0,200,255,.95)' filter='url(#rfglow)'/>"
                "</svg></div>"
                f"<span class='mm-cmd-title'>⬡ SPX&nbsp;MARKET&nbsp;COMMAND</span>"
                f"<div style='display:flex;flex-wrap:wrap;align-items:stretch;'>"
                + _stat('Spot',            f"{spot:,.0f}",        _C['text'])
                + _stat('Gamma&nbsp;Flip', _flip_str,             _C['orange'])
                + _stat('GEX&nbsp;Net',    _gex_str,              _C['accent'])
                + _stat('P/C&nbsp;Ratio',  _pc_str,               _C['purple'])
                + _stat('IV−RV',           _ivrv_str,             _C['yellow'])
                + _stat('Squeeze&nbsp;Risk', f"{_sq_score_disp}/100", _sq_ac)
                + f"</div></div>")

            # ── Section header helper ──────────────────────────────────
            def _sh(title, sub=''):
                _s = f"<span class='mm-hdr-sub'>· {sub}</span>" if sub else ''
                return wd.HTML(
                    f"<div class='mm-dash mm-section-hdr'>"
                    f"<div class='mm-dot'></div>"
                    f"<span class='mm-hdr-title'>{title}</span>{_s}"
                    f"</div>")

            # ══ LAYOUT TAB 1 ═══════════════════════════════════════════
            # ── 4-column symmetric ring grid (2 rows × 4 cols) ─────────
            _GW, _GH = 260, 272   # Stark arc reactor ring size

            # -- Dynamic gradient colors per ring --
            def _risk_grad(score):
                """Return (color1, color2) based on 0-100 risk score."""
                if score < 30:   return '#2ed573', '#00d4ff'
                if score < 60:   return '#ffd32a', '#ffa502'
                if score < 80:   return '#ffa502', '#ff6b35'
                return '#ff4757', '#b84040'

            _tail_g1, _tail_g2 = _risk_grad(_tail_score_val)
            _tail_svg_lbl2 = (_tail_interp.split('—')[0].strip()[:16]
                              if _tail_interp else '')

            _fp_svg_val   = 50.0
            _fp_svg_lbl2  = ''
            _fp_g1, _fp_g2 = '#00d4ff', '#7efff5'
            if fp_ok and fp_score is not None:
                try:
                    _fp_svg_val  = float(fp_score.get('score', 50)
                                         if isinstance(fp_score, dict) else 50)
                    _fp_svg_lbl2 = (str(fp_score.get('interpretation', '')).split()[0][:14]
                                    if isinstance(fp_score, dict) else '')
                    _fp_g1, _fp_g2 = _risk_grad(100 - _fp_svg_val)  # invert: high flow = good
                except Exception:
                    pass

            _sq_svg_val  = 50.0
            _sq_svg_lbl2 = ''
            _sq_g1, _sq_g2 = '#b44aff', '#ff6b9d'
            try:
                _sq_svg_val  = float(_sq_result_v1.get('score', 50))
                _sq_svg_lbl2 = str(_sq_result_v1.get('label', ''))[:16]
                _sq_g1, _sq_g2 = _risk_grad(_sq_svg_val)
            except Exception:
                pass

            # Row 1: 4 market metric rings
            _r_frag = wd.HTML(_svg_ring_html(
                fragility,    0, _frag_max,  'Fragilidade',   '%', '#ff4757', '#ff7843', _GW, _GH))
            _r_vol  = wd.HTML(_svg_ring_html(
                vol_premium, -_vol_hi, _vol_hi, 'Premio Vol', '%', '#ffa502', '#ff6b81', _GW, _GH))
            _r_skew = wd.HTML(_svg_ring_html(
                _skew_val,   -_skew_hi, _skew_hi, 'Skew P25-C25','%','#00d4ff','#7efff5',_GW,_GH))
            _r_move = wd.HTML(_svg_ring_html(
                daily_move,  0, _move_hi,   'Mov Esp 1D',    '%', '#2ed573', '#00d4ff', _GW, _GH))

            # Row 2: 3 risk score rings + 1 compact KPI panel
            _r_tail = wd.HTML(_svg_ring_html(
                _tail_score_val, 0, 100, 'Tail Risk', '',
                _tail_g1, _tail_g2, _GW, _GH, label2=_tail_svg_lbl2))
            _r_fp   = wd.HTML(_svg_ring_html(
                _fp_svg_val, 0, 100, 'Flow Score', '',
                _fp_g1, _fp_g2, _GW, _GH, label2=_fp_svg_lbl2))
            _r_sq   = wd.HTML(_svg_ring_html(
                _sq_svg_val, 0, 100, 'Gamma Squeeze', '',
                _sq_g1, _sq_g2, _GW, _GH, label2=_sq_svg_lbl2))

            # 4th slot row 2: compact KPI status card (pure SVG)
            _kpi_svg = (
                f"<svg viewBox='0 0 {_GW} {_GH}' width='{_GW}' height='{_GH}' "
                f"style='background:linear-gradient(145deg,#04081e,#080c28);display:block;'>"
                # corner brackets
                f"<path d='M 2,15 L 2,2 L 15,2' fill='none' stroke='#00d4ff' stroke-width='2' opacity='0.7'/>"
                f"<path d='M {_GW-15},2 L {_GW-2},2 L {_GW-2},15' fill='none' stroke='#00d4ff' stroke-width='2' opacity='0.7'/>"
                f"<path d='M 2,{_GH-15} L 2,{_GH-2} L 15,{_GH-2}' fill='none' stroke='#00d4ff' stroke-width='2' opacity='0.7'/>"
                f"<path d='M {_GW-15},{_GH-2} L {_GW-2},{_GH-2} L {_GW-2},{_GH-15}' fill='none' stroke='#00d4ff' stroke-width='2' opacity='0.7'/>"
                # title
                f"<text x='{_GW//2}' y='28' text-anchor='middle' "
                f"font-family=\"'Courier New',monospace\" font-size='9' font-weight='700' "
                f"letter-spacing='2.5' fill='rgba(0,212,255,0.55)'>MARKET STATUS</text>"
                f"<line x1='20' y1='38' x2='{_GW-20}' y2='38' stroke='rgba(0,212,255,0.18)' stroke-width='1'/>"
                # Row A: SPOT | GAMMA FLIP
                f"<text x='62' y='68' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='8' fill='rgba(0,212,255,0.4)' letter-spacing='1'>SPOT</text>"
                f"<text x='62' y='90' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='20' font-weight='700' fill='#e0f0ff'>{spot:,.0f}</text>"
                f"<text x='{_GW-62}' y='68' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='8' fill='rgba(0,212,255,0.4)' letter-spacing='1'>GAMMA FLIP</text>"
                f"<text x='{_GW-62}' y='90' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='20' font-weight='700' fill='#ffa502'>{_flip_str}</text>"
                f"<line x1='20' y1='104' x2='{_GW-20}' y2='104' stroke='rgba(0,212,255,0.08)' stroke-width='1'/>"
                # Row B: IV-RV | P/C RATIO
                f"<text x='62' y='130' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='8' fill='rgba(0,212,255,0.4)' letter-spacing='1'>IV-RV</text>"
                f"<text x='62' y='152' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='20' font-weight='700' fill='#ffd32a'>{_ivrv_str}</text>"
                f"<text x='{_GW-62}' y='130' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='8' fill='rgba(0,212,255,0.4)' letter-spacing='1'>P/C RATIO</text>"
                f"<text x='{_GW-62}' y='152' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='20' font-weight='700' fill='#b44aff'>{_pc_str}</text>"
                f"<line x1='20' y1='166' x2='{_GW-20}' y2='166' stroke='rgba(0,212,255,0.08)' stroke-width='1'/>"
                # Row C: GEX NET | SQUEEZE
                f"<text x='62' y='193' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='8' fill='rgba(0,212,255,0.4)' letter-spacing='1'>GEX NET</text>"
                f"<text x='62' y='215' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='20' font-weight='700' fill='#00d4ff'>{_gex_str}</text>"
                f"<text x='{_GW-62}' y='193' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='8' fill='rgba(0,212,255,0.4)' letter-spacing='1'>SQUEEZE</text>"
                f"<text x='{_GW-62}' y='215' text-anchor='middle' font-family=\"'Courier New',monospace\" font-size='20' font-weight='700' fill='{_sq_g1}'>{_sq_score_disp}/100</text>"
                f"</svg>"
            )
            _r_kpi = wd.HTML(_kpi_svg)

            # 4×2 grid: wraps naturally into 2 rows of 4
            _gauge_grid = wd.GridBox(
                [_cell(_r_frag, _GW, _GH), _cell(_r_vol,  _GW, _GH),
                 _cell(_r_skew, _GW, _GH), _cell(_r_move, _GW, _GH),
                 _cell(_r_tail, _GW, _GH), _cell(_r_fp,   _GW, _GH),
                 _cell(_r_sq,   _GW, _GH), _cell(_r_kpi,  _GW, _GH)],
                layout=wd.Layout(
                    grid_template_columns=f'repeat(4, {_GW}px)',
                    gap='6px',
                    width='fit-content'))

            # ── Linha 2: 3 painéis de detalhe em GridBox ───────────────
            _tail_html_w  = wd.HTML(_tail_detail_html)
            _sq_detail_vb = wd.VBox([_sq_badge_w, _sq_comps_w])
            _detail_grid  = wd.GridBox(
                [_cell(_tail_html_w,  None, _DH),
                 _cell(_fp_comps_w,   None, _DH),
                 _cell(_sq_detail_vb, None, _DH)],
                layout=wd.Layout(
                    grid_template_columns='repeat(3, minmax(420px, 1fr))',
                    grid_template_rows=f'{_DH}px',
                    gap='6px',
                    width='100%',
                    overflow_x='auto'))

            tab1 = wd.VBox([
                _status_bar,
                _sh('Painel de Controle',
                    'Fragilidade · Vol · Skew · Move · Tail Risk · Flow Score · Gamma Squeeze'),
                _gauge_grid,
                _detail_grid,
                _sh('Exposição das Gregas',
                    'Delta · Gamma · Vanna · Charm + Rebalanceamento ETF Passivo'),
                _greek_overview,
                _sh('Estrutura de Mercado',
                    'GEX por Strike · Distribuição de Retornos'),
                wd.HBox([fig_gex, fig_dist],
                        layout={'flex_wrap': 'nowrap', 'align_items': 'flex-start', 'overflow_x': 'auto', 'width': '100%'}),
                _sh('CTA Estimado & Resumo Narrativo'),
                _home_cta,
                wd.HTML(summary_html),
            ])

            # ─── ABA 2: EXPOSIÇÕES POR STRIKE ───────────────────────────
            exp_output = wd.Output()
            with exp_output:
                plot_exposure_charts(
                    agg, df, spot, from_strike, to_strike,
                    levels, model_curves, flip_points,
                    call_wall, put_wall)
            tab2 = exp_output

            # ─── ABA 3: SENSIBILIDADE ────────────────────────────────────
            cmap_map = {
                'delta': 'RdBu_r', 'gamma': 'viridis', 'vega': 'YlGnBu',
                'vanna': 'PuOr', 'theta': 'Greens', 'charm': 'RdYlGn',
                'zomma': 'plasma', 'speed': 'coolwarm'
            }
            titles_map = {
                'delta': 'Delta Nocional ($ Mn)',
                'gamma': 'Gamma — GEX NET ($ Mn / 1% move)',
                'vega': 'Vega ($ Mn / 1 vol pt)',
                'vanna': 'Vanna ($ Mn / 1pt × 1 vol pt)',
                'theta': 'Theta — Decaimento ($ Mn / dia)',
                'charm': 'Charm — Decay do Delta ($ Mn / dia)',
                'zomma': 'Zomma ($ Mn / 1% move²)',
                'speed': 'Speed ($ Mn / 1pt)',
            }
            sens_html_parts = []
            for key in ['delta', 'gamma', 'vega', 'vanna', 'theta', 'charm', 'zomma', 'speed']:
                styled = style_sensitivity_matrix(sens_matrices[key], cmap_map[key])
                sens_html_parts.append(f"<h4>{titles_map[key]}</h4>{styled}<br>")
            tab3 = wd.VBox([
                wd.HTML("<div class='mm-dash'><div class='mm-card'><h3>Matrizes de Sensibilidade (Preço × Vol Shift)</h3></div></div>"),
                wd.HTML("".join(sens_html_parts))
            ])

            # ─── ABA 4: ANÁLISE DE P&L ──────────────────────────────────
            # P&L comparativo
            fig_pnl = go.FigureWidget()
            fig_pnl.add_trace(go.Scatter(
                x=levels, y=pnl_curves['simple'] / 1e6,
                mode='lines', name='Simplificado (Δ+Γ)',
                line=dict(color=_C['orange'], dash='dot')))
            fig_pnl.add_trace(go.Scatter(
                x=levels, y=pnl_curves['complete'] / 1e6,
                mode='lines', name='Completo (+Vega+Vanna+Zomma)',
                line=dict(color=_C['accent']), fill='tonexty',
                fillcolor='rgba(88,166,255,0.08)'))
            fig_pnl.add_vline(x=spot, line_dash="dash", line_color=_C['red'])
            fig_pnl.add_hline(y=0, line_width=0.5, line_color=_C['text_dim'])
            fig_pnl.update_layout(title="P&L Comparativo: Modelo Completo vs. Simplificado",
                                  yaxis_title="P&L ($ Mi)", xaxis_title="Preço do Ativo",
                                  height=380, template=DASH_TEMPLATE)

            # Dealer P&L
            fig_dealer = go.FigureWidget()
            fig_dealer.add_trace(go.Scatter(
                x=levels, y=pnl_curves['dealer'] / 1e6,
                mode='lines', name='P&L Dealer (Total)', line_color=_C['purple'],
                line=dict(width=2.5),
                fill='tozeroy', fillcolor='rgba(188,140,255,0.08)'))
            # Per-dealer P&L curves
            _dealer_colors = ['#58a6ff', '#3fb950', '#f0883e', '#da3633',
                              '#d29922', '#bc8cff', '#8b949e', '#e6edf3']
            for _di, (_mm, _share) in enumerate(MM_VOLUME_SHARES.items()):
                _dlr_pnl = pnl_curves['dealer'] * _share / 1e6
                fig_dealer.add_trace(go.Scatter(
                    x=levels, y=_dlr_pnl,
                    mode='lines', name=_mm,
                    line=dict(color=_dealer_colors[_di % len(_dealer_colors)],
                              width=1, dash='dot'),
                    visible='legendonly'))
            fig_dealer.add_vline(x=spot, line_dash="dash", line_color=_C['red'])
            fig_dealer.add_hline(y=0, line_width=0.5, line_color=_C['text_dim'])
            fig_dealer.update_layout(title="P&L Estimado — Total + Por Dealer",
                                     yaxis_title="P&L ($ Mi)", xaxis_title="Preço do Ativo",
                                     height=420, template=DASH_TEMPLATE,
                                     legend=dict(font=dict(size=9), y=1.02, orientation='h'))

            # Dealer book summary table
            _oi100_pnl = df.OI.values * 100
            _cs_pnl = np.where(df.Type.values == 'Call', 1, -1)
            _total_dex = float(np.nansum(greeks_now['delta'] * _oi100_pnl))
            _total_gex = float(np.nansum(greeks_now['gamma'] * _cs_pnl * _oi100_pnl))
            _total_theta = float(np.nansum(greeks_now['theta'] * _oi100_pnl))
            _total_vanna = float(np.nansum(greeks_now['vanna'] * _oi100_pnl))
            _book_rows = []
            for _mm, _share in list(MM_VOLUME_SHARES.items()) + [('TOTAL', 1.0)]:
                _book_rows.append({
                    'Dealer': _mm,
                    'Share': '{:.0%}'.format(_share),
                    'DEX ($M)': '{:,.1f}'.format(_total_dex * _share * spot / 1e6),
                    'GEX/pt ($M)': '{:,.1f}'.format(_total_gex * _share / 1e6),
                    'Theta ($K/d)': '{:,.0f}'.format(_total_theta * _share / 1e3),
                    'Vanna': '{:,.0f}'.format(_total_vanna * _share),
                })
            _book_html = pd.DataFrame(_book_rows).to_html(
                classes='mm-table', index=False, border=0)
            dealer_summary_w = wd.HTML(
                "<div class='mm-dash'><div class='mm-card'>"
                "<h3>Book dos Dealers (Estimado)</h3>"
                "{}</div></div>".format(_book_html))

            # Hedge demand — per dealer
            fig_hedge = go.FigureWidget()
            _dealer_colors_h = ['#58a6ff', '#3fb950', '#f0883e', '#da3633',
                                '#d29922', '#bc8cff', '#8b949e', '#e6edf3']
            # Total hedge demand (main curve)
            fig_hedge.add_trace(go.Scatter(
                x=levels, y=pnl_curves['hedge_demand'],
                mode='lines', line=dict(color=_C['teal'], width=2.5),
                name='Total'))
            # Per-dealer hedge demand curves
            for _dhi, (_mm_h, _share_h) in enumerate(MM_VOLUME_SHARES.items()):
                _dlr_hedge = pnl_curves['hedge_demand'] * _share_h
                fig_hedge.add_trace(go.Scatter(
                    x=levels, y=_dlr_hedge,
                    mode='lines', name=_mm_h,
                    line=dict(color=_dealer_colors_h[_dhi % len(_dealer_colors_h)],
                              width=1, dash='dot'),
                    visible='legendonly'))
            fig_hedge.add_vline(x=spot, line_dash="dash", line_color=_C['red'])
            fig_hedge.add_hline(y=0, line_width=0.5, line_color=_C['text_dim'])
            fig_hedge.update_layout(
                title=f"Demanda de Hedge em Futuros ({FUTURES_TICKER})",
                yaxis_title="Número de Contratos",
                xaxis_title="Preço do Ativo",
                height=380, template=DASH_TEMPLATE)

            tab4 = wd.VBox([fig_pnl, wd.HBox([fig_dealer, fig_hedge]),
                            dealer_summary_w])

            # ─── ABA 5: MONTE CARLO ──────────────────────────────────────
            sim_var_95 = np.percentile(mc_pnl, 5)
            sim_cvar_95 = mc_pnl[mc_pnl <= sim_var_95].mean() if np.any(mc_pnl <= sim_var_95) else sim_var_95
            sim_var_99 = np.percentile(mc_pnl, 1)
            sim_cvar_99 = mc_pnl[mc_pnl <= sim_var_99].mean() if np.any(mc_pnl <= sim_var_99) else sim_var_99
            mc_p1 = np.percentile(mc_pnl, 1)
            mc_p99 = np.percentile(mc_pnl, 99)
            mc_iqr = mc_p99 - mc_p1

            fig_mc_hist = go.FigureWidget()
            fig_mc_hist.add_trace(go.Histogram(
                x=mc_pnl / 1e6, nbinsx=120,
                marker_color=_C['accent'], opacity=0.7, name='P&L'))
            fig_mc_hist.add_vline(x=sim_var_99 / 1e6, line_dash='dash',
                                  line_color=_C['red'],
                                  annotation_text=f'VaR 99% ${sim_var_99/1e6:,.1f}M',
                                  annotation_position='top left')
            fig_mc_hist.add_vline(x=sim_var_95 / 1e6, line_dash='dash',
                                  line_color=_C['orange'],
                                  annotation_text=f'VaR 95% ${sim_var_95/1e6:,.1f}M',
                                  annotation_position='bottom left')
            fig_mc_hist.add_vline(x=0, line_width=0.5, line_color=_C['text_dim'])
            mc_xlo = (mc_p1 - mc_iqr * 0.15) / 1e6
            mc_xhi = (mc_p99 + mc_iqr * 0.15) / 1e6
            fig_mc_hist.update_layout(
                title=f'Distribuição de P&L do Livro (10k Sim. t-Student, {mc_n_days} Dias)',
                xaxis_title='P&L ($ Mi)', yaxis_title='Frequência',
                xaxis_range=[mc_xlo, mc_xhi],
                height=420, template=DASH_TEMPLATE)

            mc_win_pct = (mc_pnl > 0).mean() * 100
            mc_max_loss = mc_pnl.min()
            mc_max_gain = mc_pnl.max()
            mc_table = pd.DataFrame({
                'Métrica': ['P&L Médio', 'P&L Mediano', '% Cenários Positivos',
                            'VaR 95% (Sim.)', 'CVaR 95% (Sim.)',
                            'VaR 99% (Sim.)', 'CVaR 99% (Sim.)',
                            'Perda Máxima', 'Ganho Máximo',
                            f'Horizonte ({mc_n_days}d)'],
                'Valor': [f'${np.mean(mc_pnl)/1e6:,.2f} Mi',
                          f'${np.median(mc_pnl)/1e6:,.2f} Mi',
                          f'{mc_win_pct:.1f}%',
                          f'${sim_var_95/1e6:,.2f} Mi',
                          f'${sim_cvar_95/1e6:,.2f} Mi',
                          f'${sim_var_99/1e6:,.2f} Mi',
                          f'${sim_cvar_99/1e6:,.2f} Mi',
                          f'${mc_max_loss/1e6:,.2f} Mi',
                          f'${mc_max_gain/1e6:,.2f} Mi',
                          f'{mc_n_days} dias úteis']
            }).to_html(classes='mm-table', index=False, border=0)

            # Per-dealer MC table
            dealer_mc_html = ''
            if analytics.get('dealer_mc') and len(analytics['dealer_mc']) > 1:
                dmc = analytics['dealer_mc']
                dmc_rows = []
                for mm_name in list(MM_VOLUME_SHARES.keys()) + ['TOTAL']:
                    if mm_name in dmc:
                        d = dmc[mm_name]
                        dmc_rows.append({
                            'Dealer': mm_name,
                            'Share': '{:.0%}'.format(d.get('share', 0)),
                            'Mean P&L ($M)': '{:,.1f}'.format(d['mean_pnl'] / 1e6),
                            'VaR 95% ($M)': '{:,.1f}'.format(d['var_95'] / 1e6),
                            'VaR 99% ($M)': '{:,.1f}'.format(d['var_99'] / 1e6),
                            'CVaR 95% ($M)': '{:,.1f}'.format(d['cvar_95'] / 1e6),
                            'Win %': '{:.1f}'.format(d['win_pct']),
                            'Max Loss ($M)': '{:,.1f}'.format(d['max_loss'] / 1e6),
                        })
                if dmc_rows:
                    dealer_mc_html = pd.DataFrame(dmc_rows).to_html(
                        classes='mm-table', index=False, border=0)

            mc_dealer_widget = wd.HTML(
                "<div class='mm-dash'><div class='mm-card'>"
                "<h3>Monte Carlo por Dealer</h3>"
                "{}</div></div>".format(dealer_mc_html if dealer_mc_html else
                                        '<p style="color:#8b949e;">Sem dados de dealer MC</p>'))

            # Per-dealer histograms (overlay top 4 dealers)
            fig_mc_dealers = go.FigureWidget()
            _mc_d_colors = ['#58a6ff', '#3fb950', '#f0883e', '#da3633']
            _mc_top4 = list(MM_VOLUME_SHARES.keys())[:4]
            dmc = analytics.get('dealer_mc', {})
            for _mci, _mmn in enumerate(_mc_top4):
                if _mmn in dmc and 'mc_pnl' in dmc[_mmn]:
                    fig_mc_dealers.add_trace(go.Histogram(
                        x=dmc[_mmn]['mc_pnl'] / 1e6, nbinsx=80,
                        marker_color=_mc_d_colors[_mci], opacity=0.5,
                        name=_mmn))
            fig_mc_dealers.update_layout(
                title='Distribuição P&L por Dealer (Top 4)',
                xaxis_title='P&L ($ Mi)', yaxis_title='Frequência',
                barmode='overlay', height=350, template=DASH_TEMPLATE)

            tab5 = wd.VBox([
                wd.HTML("<div class='mm-dash'><div class='mm-card'>"
                        f"<h3>Simulação Monte Carlo (t-Student, {mc_n_days} Dias)</h3></div></div>"),
                wd.HBox([fig_mc_hist, wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>{mc_table}</div></div>")]),
                mc_dealer_widget,
                fig_mc_dealers,
            ])

            # ─── ABA 6: REBALANCEAMENTO ETFs + ALAVANCADOS ─────────────
            if etf_ok and etf_flows:
                # Dropdown para selecionar ETF
                etf_dd = wd.Dropdown(
                    options=['Combined'] + PASSIVE_ETFS,
                    value='Combined', description='ETF:',
                    layout={'width': '300px'})
                flow_html = wd.HTML()
                summary_html = wd.HTML()

                def _render_etf(change=None):
                    key = etf_dd.value
                    df_f = etf_flows.get(key, pd.DataFrame())
                    if df_f.empty:
                        flow_html.value = "<p>Sem dados.</p>"
                        return
                    top = df_f.head(30)
                    flow_html.value = (
                        top[['Start', 'Now', 'Delta', 'Flow_$', 'PctADV']]
                        .style.format({
                            'Start': '{:.4f}', 'Now': '{:.4f}', 'Delta': '{:+.4f}',
                            'Flow_$': '${:,.0f}', 'PctADV': '{:.1f}%'})
                        .background_gradient(cmap='RdYlGn', subset=['Flow_$'])
                        .to_html())
                etf_dd.observe(_render_etf, names='value')
                _render_etf()

                if etf_summary is not None:
                    summary_html.value = (
                        "<h4>Resumo por ETF</h4>" +
                        etf_summary.style.format('${:,.0f}')
                            .background_gradient(cmap='RdYlGn', subset=['Net_$'])
                            .to_html())

                # Seção de ETFs alavancados
                if lev_ok and lev_flows is not None:
                    lev_html_str = (
                        f"<h4>ETFs Alavancados (Retorno diário: {daily_ret*100:+.2f}%)</h4>" +
                        lev_flows[['Leverage', 'AUM_$', 'Rebalance_$', 'Direção']]
                        .style.format({
                            'AUM_$': '${:,.0f}', 'Rebalance_$': '${:,.0f}'})
                        .to_html() +
                        f"<p><b>Fluxo Direcional Total: ${lev_total:,.0f}</b></p>")
                else:
                    lev_html_str = "<p>ETFs alavancados não disponíveis.</p>"

                # Top 30 que ganham fluxo de compra e top 30 que levam fluxo de venda
                combo_flow = etf_flows.get('Combined', pd.DataFrame())
                flow_ranking = wd.HTML()
                if not combo_flow.empty and 'Flow_$' in combo_flow.columns:
                    top30_buy = combo_flow.nlargest(30, 'Flow_$')
                    top30_sell = combo_flow.nsmallest(30, 'Flow_$')
                    fmt_dict = {'Delta': '{:+.4f}', 'Flow_$': '${:,.0f}', 'PctADV': '{:.1f}%'}
                    buy_html = (top30_buy[['Delta', 'Flow_$', 'PctADV']]
                        .style.format(fmt_dict)
                        .background_gradient(cmap='Greens', subset=['Flow_$'])
                        .to_html())
                    sell_html = (top30_sell[['Delta', 'Flow_$', 'PctADV']]
                        .style.format(fmt_dict)
                        .background_gradient(cmap='Reds', subset=['Flow_$'])
                        .to_html())
                    flow_ranking = wd.HBox([
                        wd.VBox([wd.HTML("<div class='mm-dash'><div class='mm-card'><h4>Top 30 — Fluxo de Compra</h4></div></div>"),
                                 wd.HTML(buy_html)]),
                        wd.VBox([wd.HTML("<div class='mm-dash'><div class='mm-card'><h4>Top 30 — Fluxo de Venda</h4></div></div>"),
                                 wd.HTML(sell_html)])
                    ])

                tab6 = wd.VBox([
                    wd.HTML(f"<div class='mm-dash'><div class='mm-card'><h3>Rebalanceamento ETFs Passivos desde {etf_start}</h3></div></div>"),
                    etf_dd, flow_html, summary_html,
                    flow_ranking,
                    wd.HTML(lev_html_str)
                ])
            else:
                tab6 = wd.VBox([wd.HTML(
                    "<h3>Rebalanceamento ETFs</h3>"
                    "<p>Dados de rebalanceamento não disponíveis.</p>")])

            # ─── ABA 7: PREVISÃO SPX ────────────────────────────────────
            if spx_pred_ok and top_in_spx is not None:
                cols_in = ['Prob_In', 'CUR_MKT_CAP', 'FMC', 'FALR', 'FREE_FLOAT_PCT', 'NET_INC_TTM']
                cols_out = ['ExitScore', 'CUR_MKT_CAP', 'FMC', 'FALR', 'FREE_FLOAT_PCT', 'NET_INC_TTM']
                in_html = (top_in_spx[[c for c in cols_in if c in top_in_spx.columns]]
                    .style.format({
                        'Prob_In': '{:.2%}', 'CUR_MKT_CAP': '${:,.0f}',
                        'FMC': '${:,.0f}', 'FALR': '{:.4f}',
                        'FREE_FLOAT_PCT': '{:.1f}%'})
                    .background_gradient(cmap='Greens', subset=['Prob_In'])
                    .to_html())
                out_html = (top_out_spx[[c for c in cols_out if c in top_out_spx.columns]]
                    .style.format({
                        'ExitScore': '{:.2%}', 'CUR_MKT_CAP': '${:,.0f}',
                        'FMC': '${:,.0f}', 'FALR': '{:.4f}',
                        'FREE_FLOAT_PCT': '{:.1f}%', 'NET_INC_TTM': '${:,.0f}'})
                    .background_gradient(cmap='Reds', subset=['ExitScore'])
                    .to_html())
                spx_rules_html = (
                    "<div class='mm-dash'><div class='mm-card' style='padding:10px;'>"
                    "<span class='mm-section-label'>Critérios S&P 500</span> "
                    f"<span class='mm-cot-label'>Market Cap ≥ $18B</span> "
                    f"<span class='mm-cot-label'>Free Float ≥ 50%</span> "
                    f"<span class='mm-cot-label'>FALR ≥ 0.75</span> "
                    f"<span class='mm-cot-label'>Lucro TTM > 0</span> "
                    f"<span class='mm-cot-label'>NI Q0 > 0</span>"
                    "</div></div>")
                tab7 = wd.VBox([
                    wd.HTML("<div class='mm-dash'><div class='mm-card'><h3>Previsão de Rebalanceamento S&P 500</h3></div></div>"),
                    wd.HTML(spx_rules_html),
                    wd.HBox([
                        wd.VBox([wd.HTML("<div class='mm-dash'><div class='mm-card'><h4>Top 30 — Candidatos a Entrar</h4></div></div>"),
                                 wd.HTML(in_html)]),
                        wd.VBox([wd.HTML("<div class='mm-dash'><div class='mm-card'><h4>Top 30 — Candidatos a Sair</h4></div></div>"),
                                 wd.HTML(out_html)])
                    ])
                ])
            else:
                reason = "Marque 'Incluir Previsão SPX' e rode novamente." if not spx_pred_w.value else (
                    "sklearn não disponível." if not HAS_SKLEARN else "Erro na execução.")
                tab7 = wd.VBox([wd.HTML(
                    f"<h3>Previsão SPX</h3><p>{reason}</p>")])

            # ─── ABA 8: SIMULADOR INTERATIVO ─────────────────────────────
            vol_slider = wd.FloatSlider(
                value=0, min=-10, max=10, step=1,
                description='Shift Vol (pts):', continuous_update=False,
                layout={'width': '350px'})
            dte_slider = wd.IntSlider(
                value=0, min=0, max=20, step=1,
                description='Dias a Frente:', continuous_update=False,
                layout={'width': '350px'})
            spot_slider = wd.FloatSlider(
                value=0, min=-10, max=10, step=0.5,
                description='Spot Move (%):', continuous_update=False,
                layout={'width': '350px'})

            fig_sim_dex = go.FigureWidget()
            fig_sim_dex.add_trace(go.Scatter(x=levels, y=np.zeros(len(levels)),
                                             mode='lines', line_color=_C['accent'],
                                             name='Delta'))
            fig_sim_dex.add_trace(go.Scatter(x=[spot], y=[0], mode='markers',
                                             marker=dict(color=_C['red'], size=10, symbol='x'),
                                             name='Spot'))
            fig_sim_dex.update_layout(title="Delta Nocional", yaxis_title="$ Bi",
                                      height=320, template=DASH_TEMPLATE,
                                      margin=dict(t=30, b=20))

            fig_sim_gex = go.FigureWidget()
            fig_sim_gex.add_trace(go.Scatter(x=levels, y=np.zeros(len(levels)),
                                             mode='lines', line_color=_C['red'],
                                             name='Gamma'))
            fig_sim_gex.add_trace(go.Scatter(x=[spot], y=[0], mode='markers',
                                             marker=dict(color='#d29922', size=10, symbol='diamond'),
                                             name='Flip'))
            fig_sim_gex.update_layout(title="Gamma (GEX) + Gamma Flip",
                                      yaxis_title="$ Bi / 1% move",
                                      height=320, template=DASH_TEMPLATE,
                                      margin=dict(t=30, b=20))

            fig_sim_vega = go.FigureWidget()
            fig_sim_vega.add_trace(go.Scatter(x=levels, y=np.zeros(len(levels)),
                                              mode='lines', line_color=_C['purple'],
                                              name='Vega'))
            fig_sim_vega.update_layout(title="Vega Nocional", yaxis_title="$ Mi",
                                       height=320, template=DASH_TEMPLATE,
                                       margin=dict(t=30, b=20))

            # Flow adjustment chart (how vol ctrl / RP / CTA / dealer adjust)
            fig_sim_flows = go.FigureWidget()
            fig_sim_flows.add_trace(go.Bar(x=['Vol Ctrl', 'Risk Parity', 'CTA', 'Dealer', 'Vanna', 'Charm'],
                                           y=[0, 0, 0, 0, 0, 0],
                                           marker_color=[_C['accent'], _C['teal'], _C['orange'],
                                                         _C['purple'], _C['pink'], _C['yellow']],
                                           name='Flow ($B)'))
            fig_sim_flows.update_layout(
                title="Fluxo Estimado por Componente ($B)",
                yaxis_title="$ Bi", height=320, template=DASH_TEMPLATE,
                margin=dict(t=30, b=20))

            sim_info = wd.HTML('')

            def _update_simulator(change=None):
                v_shift = vol_slider.value / 100.0
                d_shift = dte_slider.value
                s_move = spot_slider.value / 100.0
                sim_vol = np.clip(df.IV.values + v_shift, 0.001, None)
                sim_tte = np.clip(df.Tte.values - d_shift / TRADING_DAYS, 1.0 / TRADING_DAYS, None)
                types_arr = df.Type.values
                new_spot = spot * (1 + s_move)

                dex_c, gex_c, vex_c = [], [], []
                _flip_level = None
                _prev_gex = None
                for L in levels:
                    g = calculate_all_greeks(L, df.Strike.values, sim_vol, sim_tte, types_arr, r=rfr)
                    oi_100 = df.OI.values * 100.0
                    dex_c.append(np.nansum(g['delta'] * oi_100 * L))
                    _gex_val = np.nansum(g['gamma'] * np.where(types_arr == 'Call', 1, -1)
                                         * oi_100 * (L**2) * 0.01)
                    gex_c.append(_gex_val)
                    vex_c.append(np.nansum(g['vega'] * oi_100))
                    if _prev_gex is not None and _flip_level is None:
                        if (_prev_gex < 0 and _gex_val >= 0) or (_prev_gex > 0 and _gex_val <= 0):
                            _flip_level = L
                    _prev_gex = _gex_val

                dex_arr = np.array(dex_c) / 1e9
                gex_arr = np.array(gex_c) / 1e9
                vex_arr = np.array(vex_c) / 1e6

                with fig_sim_dex.batch_update():
                    fig_sim_dex.data[0].y = dex_arr
                    fig_sim_dex.data[1].x = [new_spot]
                    idx_ns = np.argmin(np.abs(levels - new_spot))
                    fig_sim_dex.data[1].y = [dex_arr[idx_ns]] if idx_ns < len(dex_arr) else [0]
                with fig_sim_gex.batch_update():
                    fig_sim_gex.data[0].y = gex_arr
                    if _flip_level is not None:
                        fig_sim_gex.data[1].x = [_flip_level]
                        fig_sim_gex.data[1].y = [0]
                    else:
                        fig_sim_gex.data[1].x = [new_spot]
                        idx_ns2 = np.argmin(np.abs(levels - new_spot))
                        fig_sim_gex.data[1].y = [gex_arr[idx_ns2]] if idx_ns2 < len(gex_arr) else [0]
                with fig_sim_vega.batch_update():
                    fig_sim_vega.data[0].y = vex_arr

                # Compute combined flow for that scenario
                ds = new_spot - spot
                rv_now = rv_30d if pd.notna(rv_30d) else 0.15
                # Vol slider shifts IV; treat it as realized vol shock too
                vol_shift_dec = v_shift  # already in decimal (slider / 100)
                rv_shock = max(rv_now + vol_shift_dec, rv_now * (1 + max(0, -s_move) * 5))
                rv_shock = min(rv_shock, 0.80)

                # Vol Control: responds to BOTH spot crash and vol rise
                vc_flow = 0
                for _tv in [5, 10, 15]:
                    _tv_d = _tv / 100.0
                    _e0 = _vc_exposure(_tv_d, rv_now)
                    _e1 = _vc_exposure(_tv_d, rv_shock)
                    vc_flow += VOL_CTRL_AUM.get(_tv, 100e9) * (_e1 - _e0)

                # Risk Parity: responds to vol changes (inverse-vol allocation)
                rp_result = compute_risk_parity_flow(rv_shock, rv_now)
                rp_flow = rp_result['total']

                # CTA: responds to spot trends
                cta_flow = 0
                if s_move < -0.03:
                    cta_flow = s_move * CTA_AUM * CTA_EQUITY_ALLOC * 0.5
                elif s_move > 0.03:
                    cta_flow = s_move * CTA_AUM * CTA_EQUITY_ALLOC * 0.3

                _oi_sim = df.OI.values * 100
                _cs_sim = np.where(types_arr == 'Call', 1, -1)
                g_new = calculate_all_greeks(new_spot, df.Strike.values, sim_vol, sim_tte, types_arr, r=rfr)
                _gex_new = float(np.nansum(g_new['gamma'] * _cs_sim * _oi_sim))
                _dex_new = float(np.nansum(g_new['delta'] * _oi_sim))
                dealer_flow = -(_dex_new * ds + 0.5 * _gex_new * ds ** 2)

                vanna_not = float(np.nansum(g_new['vanna'] * _oi_sim) * new_spot)
                vol_chg_dec = rv_shock - rv_now  # decimal (0.02 = 2 pts)
                vanna_flow = -vanna_not * vol_chg_dec if vanna_not != 0 else 0

                charm_not_sim = float(np.nansum(g_new['charm'] * _oi_sim) * new_spot / 365.0)
                charm_flow = -charm_not_sim  # dealers unwind decayed delta overnight

                flows = [vc_flow / 1e9, rp_flow / 1e9, cta_flow / 1e9,
                         dealer_flow / 1e9, vanna_flow / 1e9, charm_flow / 1e9]
                total_flow = sum(flows)

                with fig_sim_flows.batch_update():
                    fig_sim_flows.data[0].y = flows

                # Info panel
                regime = 'ESTABILIDADE (GEX+)' if _flip_level and new_spot > _flip_level else 'ACELERAÇÃO (GEX−)'
                flip_str = '{:,.0f}'.format(_flip_level) if _flip_level else 'N/A'
                sim_info.value = (
                    "<div class='mm-dash'><div class='mm-card'>"
                    "<h4>Cenário: SPX {:+.1f}% → {:,.0f} | Vol {:+.0f} pts | "
                    "{} dias à frente</h4>"
                    "<p>Gamma Flip: <b>{}</b> | Regime: <b>{}</b></p>"
                    "<p>Fluxo Total Estimado: <b style='color:{};'>${:+.1f}B</b> "
                    "(VC: ${:.1f}B, RP: ${:.1f}B, CTA: ${:.1f}B, "
                    "Dealer: ${:.1f}B, Vanna: ${:.1f}B, Charm: ${:.1f}B)</p>"
                    "</div></div>".format(
                        s_move * 100, new_spot, vol_slider.value, d_shift,
                        flip_str, regime,
                        '#3fb950' if total_flow > 0 else '#da3633', total_flow,
                        flows[0], flows[1], flows[2], flows[3], flows[4], flows[5]))

            vol_slider.observe(_update_simulator, names='value')
            dte_slider.observe(_update_simulator, names='value')
            spot_slider.observe(_update_simulator, names='value')
            _update_simulator()

            tab8 = wd.VBox([
                wd.HTML("<div class='mm-dash'><div class='mm-card'>"
                        "<h3>Simulador Interativo — Cenários + Fluxos</h3>"
                        "<p>Ajuste spot, vol e tempo para ver inflexões, "
                        "gamma flip e como dealers/vol-ctrl/RP/CTA ajustam.</p>"
                        "</div></div>"),
                wd.HBox([spot_slider, vol_slider, dte_slider]),
                sim_info,
                wd.HBox([fig_sim_dex, fig_sim_gex]),
                wd.HBox([fig_sim_vega, fig_sim_flows]),
            ])

            # ─── ABA 9: RELATÓRIO DE RISCO ───────────────────────────────
            oi_100 = df['OI'].values * 100.0
            delta_notional = np.nansum(greeks_now['delta'] * oi_100) * spot
            total_vega = np.nansum(greeks_now['vega'] * oi_100)
            total_zomma = np.nansum(greeks_now['zomma'] * oi_100)
            total_speed = np.nansum(greeks_now['speed'] * oi_100)
            total_charm = np.nansum(greeks_now['charm'] * oi_100) / 365.0
            total_gex_val = np.nansum(
                greeks_now['gamma'] * np.where(df.Type.values == 'Call', 1, -1)
                * oi_100 * (spot**2) * 0.01)

            vanna_impact = np.nansum(greeks_now['vanna'] * oi_100) * spot
            vanna_action = "VENDER" if vanna_impact > 0 else "COMPRAR"
            zomma_action = "MAIS INSTÁVEL" if total_zomma < 0 else "MAIS ESTÁVEL"

            gamma_regime = "N/A"
            if gamma_flip:
                gamma_regime = ("ESTABILIDADE (acima do Flip)" if spot > gamma_flip
                                else "ACELERAÇÃO (abaixo do Flip)")

            hedge_contracts = -np.nansum(greeks_now['delta'] * oi_100) / FUTURES_MULTIPLIER
            hedge_action = "COMPRAR" if hedge_contracts > 0 else "VENDER"

            # ── Lógica de decisão do MM ──
            # Bias direcional
            if abs(delta_notional) < 50e6:
                delta_signal = '🟢 NEUTRO'
                delta_rec = 'Sem ajuste necessário.'
            elif delta_notional > 0:
                delta_signal = '🔴 COMPRADO'
                delta_rec = (f'VENDER {abs(hedge_contracts):,.0f} {FUTURES_TICKER} '
                             f'ou comprar puts para reduzir exposição.')
            else:
                delta_signal = '🔵 VENDIDO'
                delta_rec = (f'COMPRAR {abs(hedge_contracts):,.0f} {FUTURES_TICKER} '
                             f'ou comprar calls para reduzir exposição.')

            # Regime de gamma
            if total_gex_val > 0:
                gex_signal = '🟢 GAMMA POSITIVO'
                gex_rec = ('Mercado tende a se estabilizar. MM vende em alta, compra em '
                           'baixa → menor volatilidade realizada. Pode VENDER vol (straddles/strangles).')
            else:
                gex_signal = '🔴 GAMMA NEGATIVO'
                gex_rec = ('Mercado tende a se ACELERAR. MM compra em alta, vende em '
                           'baixa → maior volatilidade realizada. REDUZIR tamanho e '
                           'colocar stops mais apertados.')

            # Vol trade
            if vol_premium > 2:
                vol_signal = '🟢 IV > RV (prêmio alto)'
                vol_rec = 'Oportunidade de VENDER vol — spreads de crédito, iron condors.'
            elif vol_premium < -2:
                vol_signal = '🔴 RV > IV (vol barata)'
                vol_rec = 'Oportunidade de COMPRAR vol — straddles, calendars.'
            else:
                vol_signal = '🟡 VOL NEUTRA'
                vol_rec = 'Sem edge claro em vol. Preferir posições direcionais.'

            # Skew trade
            if skew > 0.03:
                skew_signal = '🔴 SKEW ELEVADO'
                skew_rec = ('Puts caras vs calls. Considerar risk reversals '
                            '(vender put OTM, comprar call OTM) ou put spreads.')
            elif skew < 0.01:
                skew_signal = '🟢 SKEW COMPRIMIDO'
                skew_rec = 'Proteção barata. Comprar puts OTM como hedge de cauda.'
            else:
                skew_signal = '🟡 SKEW NORMAL'
                skew_rec = 'Sem distorção significativa no skew.'

            # Vanna/Zomma trade
            if vanna_impact > 0:
                vanna_rec = ('Vol subindo → dealers precisam VENDER o ativo (pressão baixista). '
                             'Cuidado com posições compradas.')
            else:
                vanna_rec = ('Vol subindo → dealers precisam COMPRAR o ativo (pressão altista). '
                             'Suporte adicional em quedas.')

            # Theta vs Gamma
            daily_theta = np.nansum(greeks_now['theta'] * oi_100) / TRADING_DAYS
            gamma_pnl_1pct = total_gex_val
            theta_gamma_ratio = abs(daily_theta / gamma_pnl_1pct) if gamma_pnl_1pct != 0 else 0
            if theta_gamma_ratio > 1:
                tg_signal = '🟢 THETA DOMINANTE'
                tg_rec = ('Decaimento diário supera risco de gamma. '
                          'Posições vendidas em vol são favorecidas.')
            else:
                tg_signal = '🔴 GAMMA DOMINANTE'
                tg_rec = ('Risco de gamma supera o decaimento. '
                          'Movimentos grandes serão caros. Reduzir exposição vendida.')

            # Urgência geral
            urgency_score = 0
            if abs(delta_notional) > 200e6: urgency_score += 2
            if total_gex_val < 0: urgency_score += 2
            if abs(vol_premium) > 3: urgency_score += 1
            if skew > 0.04: urgency_score += 1
            if urgency_score >= 4:
                urgency = '🔴 ALTA — Ajustes imediatos recomendados'
            elif urgency_score >= 2:
                urgency = '🟡 MÉDIA — Monitorar e preparar ordens'
            else:
                urgency = '🟢 BAIXA — Posição confortável'

            report_html = f"""
            <div class='mm-dash'>
            <div class='mm-card'>
            <h3 style='font-size:18px;'>COCKPIT DO MARKET MAKER</h3>
            <p>Análise: {datetime.now().strftime('%Y-%m-%d %H:%M')} │ Spot: ${spot:,.2f} │
               {len(df)} opções │ {urgency}</p>
            </div>

            <div class='mm-card'>
            <h3>DECISÕES RECOMENDADAS</h3>
            <table class='mm-table'>
            <tr><th>Dimensão</th><th>Sinal</th><th>Ação Recomendada</th></tr>
            <tr><td><b>Delta (Direção)</b></td><td style='text-align:center;'>{delta_signal}</td><td>{delta_rec}</td></tr>
            <tr><td><b>Gamma (Regime)</b></td><td style='text-align:center;'>{gex_signal}</td><td>{gex_rec}</td></tr>
            <tr><td><b>Vol Premium</b></td><td style='text-align:center;'>{vol_signal}</td><td>{vol_rec}</td></tr>
            <tr><td><b>Skew</b></td><td style='text-align:center;'>{skew_signal}</td><td>{skew_rec}</td></tr>
            <tr><td><b>Vanna (Vol→Spot)</b></td><td style='text-align:center;'>{'🔴' if vanna_impact > 0 else '🟢'} {vanna_action}</td><td>{vanna_rec}</td></tr>
            <tr><td><b>Theta vs Gamma</b></td><td style='text-align:center;'>{tg_signal}</td><td>{tg_rec}</td></tr>
            </table>
            </div>

            <div class='mm-card'>
            <h3>POSIÇÃO ATUAL</h3>
            <table class='mm-table'>
            <tr><th>Exposição</th><th style='text-align:right;'>Valor</th><th>Interpretação</th></tr>
            <tr><td>Delta Nocional</td><td style='text-align:right;'>{fmt_value(delta_notional)}</td><td>Hedge: {hedge_action} {abs(hedge_contracts):,.0f} {FUTURES_TICKER}</td></tr>
            <tr><td>GEX Total</td><td style='text-align:right;'>{fmt_value(total_gex_val)}</td><td>Flip: ~{f'{gamma_flip:,.0f}' if gamma_flip else 'N/A'} │ {gamma_regime}</td></tr>
            <tr><td>Vega</td><td style='text-align:right;'>{fmt_value(total_vega)}</td><td>P&L por 1% de aumento na vol</td></tr>
            <tr><td>Vanna</td><td style='text-align:right;'>{fmt_value(vanna_impact)}</td><td>Fluxo de rebalanceamento se vol mudar</td></tr>
            <tr><td>Zomma</td><td style='text-align:right;'>{fmt_value(total_zomma)}</td><td>Regime ficaria {zomma_action} com vol</td></tr>
            <tr><td>Speed</td><td style='text-align:right;'>{fmt_value(total_speed)}</td><td>Aceleração do gamma por $1 no spot</td></tr>
            <tr><td>Charm (diário)</td><td style='text-align:right;'>{fmt_value(total_charm)}</td><td>Decaimento do delta overnight</td></tr>
            </table>
            </div>

            <div class='mm-card'>
            <h3>RISCO CAUDAL</h3>
            <table class='mm-table' style='width:70%;'>
            <tr><th>Métrica</th><th>Paramétrico</th><th>Monte Carlo</th></tr>
            <tr><td>VaR 95%</td><td style='color:{_C['yellow']}'>{risk['var_95']:.2%}</td><td style='color:{_C['yellow']}'>${sim_var_95/1e6:,.2f} Mi</td></tr>
            <tr><td>CVaR 95%</td><td style='color:{_C['orange']}'>{risk['cvar_95']:.2%}</td><td style='color:{_C['orange']}'>${sim_cvar_95/1e6:,.2f} Mi</td></tr>
            <tr><td>VaR 99%</td><td style='color:{_C['red']}'>{risk['var_99']:.2%}</td><td style='color:{_C['red']}'>${sim_var_99/1e6:,.2f} Mi</td></tr>
            <tr><td>CVaR 99%</td><td style='color:{_C['red']}'>{risk['cvar_99']:.2%}</td><td style='color:{_C['red']}'>${sim_cvar_99/1e6:,.2f} Mi</td></tr>
            </table>
            </div>

            <div class='mm-card'>
            <h3>NÍVEIS-CHAVE</h3>
            <div class='mm-kpi-row'>
                <div class='mm-kpi'><div class='kpi-label'>Call Wall</div><div class='kpi-value' style='color:{_C['green']}'>{f'{call_wall:,.0f}' if call_wall else 'N/A'}</div><div style='font-size:11px;color:{_C['text_muted']}'>Resistência forte</div></div>
                <div class='mm-kpi'><div class='kpi-label'>Put Wall</div><div class='kpi-value' style='color:{_C['red']}'>{f'{put_wall:,.0f}' if put_wall else 'N/A'}</div><div style='font-size:11px;color:{_C['text_muted']}'>Suporte forte</div></div>
                <div class='mm-kpi'><div class='kpi-label'>Gamma Flip</div><div class='kpi-value' style='color:{_C['yellow']}'>{f'{gamma_flip:,.0f}' if gamma_flip else 'N/A'}</div><div style='font-size:11px;color:{_C['text_muted']}'>Acima = estabilidade</div></div>
                <div class='mm-kpi'><div class='kpi-label'>Mov. Implícito 1D</div><div class='kpi-value' style='color:{_C['accent']}'>±{daily_move:.2f}%</div><div style='font-size:11px;color:{_C['text_muted']}'>±${spot * daily_move / 100:,.0f}</div></div>
            </div>
            </div>
            </div>"""

            tab9 = wd.VBox([wd.HTML(report_html)])

            # ─── ABA 10: FLOW PREDICTOR ──────────────────────────────────
            try:
              if fp_ok and fp_score is not None:
                # Score summary header
                fp_title = wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h3 style='font-size:18px;'>Flow Predictor — {ticker}</h3>"
                    f"<p>Direction: <b style='color:"
                    f"{_C['green'] if fp_score['direction'] == 'BULLISH' else _C['red'] if fp_score['direction'] == 'BEARISH' else _C['text_muted']}'>"
                    f"{fp_score['direction']}</b> │ "
                    f"P(Up): {fp_score['prob_up']:.1%} │ "
                    f"Score: {fp_score['combined_score']:+.2f}</p>"
                    f"</div></div>")

                # Sub-tab A: Score
                st_a = wd.VBox([
                    fp_title,
                    wd.HBox([fp_plot_score_gauge(fp_score),
                             fp_plot_components_bar(fp_score)]),
                    fp_grid_flow_score(fp_score)
                ])

                # Sub-tab B: Histórico
                st_b_children = [wd.HTML("<div class='mm-dash'><div class='mm-card'><h3>Histórico de Fluxo — ETFs Alavancados</h3></div></div>")]
                if not fp_flow_hist.empty:
                    st_b_children.append(fp_plot_flow_history(fp_flow_hist))
                    st_b_children.append(fp_bqp_flow_bar_line(fp_flow_hist.tail(60)))
                    st_b_children.append(fp_bqp_scatter(fp_flow_hist))
                st_b = wd.VBox(st_b_children)

                # Sub-tab C: Buyback + Blackout
                _bo_pct = fp_buyback.get('blackout_pct', 0)
                _bo_n = fp_buyback.get('blackout_n', 0)
                _bo_total = fp_buyback.get('blackout_total', 0)
                _bo_open = 1 - _bo_pct
                _bo_color = _C['red'] if _bo_pct > 0.4 else _C['yellow'] if _bo_pct > 0.2 else _C['green']
                _bo_signal = ('🔴 BLACKOUT PESADO' if _bo_pct > 0.4
                              else '🟡 BLACKOUT MODERADO' if _bo_pct > 0.2
                              else '🟢 JANELA ABERTA')

                st_c_children = [
                    wd.HTML(
                        f"<div class='mm-dash'><div class='mm-card'>"
                        f"<h3>Estimativa de Buyback + Blackout Window</h3>"
                        f"<p>{_bo_signal} — <b style='color:{_bo_color}'>"
                        f"{_bo_pct:.0%}</b> das empresas em blackout "
                        f"({_bo_n}/{_bo_total})</p>"
                        f"<p><small>Empresas não podem recomprar ações ~{BLACKOUT_DAYS_BEFORE} dias "
                        f"antes do balanço até ~{BLACKOUT_DAYS_AFTER} dias após divulgação. "
                        f"Quando muitas estão em restrição, o fluxo de buyback cai "
                        f"significativamente.</small></p>"
                        f"</div></div>")]

                bb_pct_adv = fp_buyback.get('pct_adv_est', 0)
                bb_pct_str = 'N/A' if (bb_pct_adv is None or pd.isna(bb_pct_adv)) else f'{bb_pct_adv:.2f}%'
                _daily_open = fp_buyback.get('daily_est_open', fp_buyback.get('daily_est', 0))
                _daily_adj = fp_buyback.get('daily_est', 0)
                bb_html = (
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<table class='mm-table' style='width:auto;'>"
                    f"<tr><td>Anunciado:</td><td style='text-align:right;'>"
                    f"${fp_buyback.get('announced', 0):,.0f}</td></tr>"
                    f"<tr><td>Estimativa diária (sem blackout):</td>"
                    f"<td style='text-align:right;'>${_daily_open:,.0f}</td></tr>"
                    f"<tr><td>Blackout ({_bo_pct:.0%} restrito):</td>"
                    f"<td style='text-align:right;color:{_bo_color}'>-{_bo_pct:.0%}</td></tr>"
                    f"<tr><td><b>Estimativa diária ajustada:</b></td>"
                    f"<td style='text-align:right;font-weight:bold;'>${_daily_adj:,.0f}</td></tr>"
                    f"<tr><td>% ADV estimado:</td><td style='text-align:right;'>"
                    f"{bb_pct_str}</td></tr>"
                    f"<tr><td>Confiança:</td><td style='text-align:right;'>"
                    f"{fp_buyback.get('confidence', 'N/A')}</td></tr>"
                    f"</table></div></div>")
                st_c_children.append(wd.HTML(bb_html))

                # Blackout curve chart
                if not fp_blackout_curve.empty:
                    _bb_annual = fp_buyback.get('announced', 0)
                    if not _bb_annual or _bb_annual <= 0:
                        _bb_annual = SPX_ANNUAL_BUYBACK_EST
                    st_c_children.append(
                        build_buyback_blackout_chart(
                            fp_blackout_curve, fp_earnings_df,
                            buyback_annual=_bb_annual))

                try:
                    bb_df = estimate_index_buyback_flow(ticker, top_n=30)
                    if not bb_df.empty:
                        st_c_children.append(wd.HTML("<h4>Top Buybacks do Índice</h4>"))
                        st_c_children.append(fp_grid_buyback(bb_df))
                except Exception:
                    pass
                st_c = wd.VBox(st_c_children)

                # Sub-tab D: COT
                st_d_children = [wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h3>COT — Commitment of Traders</h3>"
                    f"<p>Dados agregados (soma de todas trader types). "
                    f"Report type: auto. "
                    f"<span class='mm-cot-label'>Period: <b>{cot_start_w.value} → {cot_end_w.value}</b></span></p>"
                    f"</div></div>")]
                cot_ok_fp2, cot_fut_fp2 = has_cot(ticker)
                if cot_ok_fp2 and fp_cot_df is not None and not fp_cot_df.empty:
                    st_d_children.append(
                        wd.HTML(f"<p>Futures: <b>{cot_fut_fp2}</b> — "
                                f"<b>{len(fp_cot_df)}</b> registros — "
                                f"WoW Δ Net: <b>{cot_net_change:+,.0f}</b></p>"))
                    st_d_children.append(fp_grid_cot_stats(fp_cot_stats))
                    seas = cot_seasonality(fp_cot_df)
                    st_d_children.append(wd.HBox([
                        fp_plot_positions_basket(fp_cot_df),
                        fp_plot_dispersion(seas, fp_cot_df)
                    ]))
                    st_d_children.append(fp_plot_long_short_net(fp_cot_df))
                    st_d_children.append(wd.HBox([
                        fp_plot_long_short_ratio(fp_cot_df),
                        fp_plot_correlation(fp_cot_df)
                    ]))
                    st_d_children.append(fp_plot_multi_year(fp_cot_df))
                elif cot_ok_fp2:
                    st_d_children.append(
                        wd.HTML(f"<p>COT disponível para {cot_fut_fp2}, "
                                "mas sem dados retornados.</p>"))
                else:
                    st_d_children.append(
                        wd.HTML(f"<p>{ticker} não possui dados COT vinculados. "
                                "Use os contratos selecionados abaixo.</p>"))

                if fp_selected_cot_df is not None and not fp_selected_cot_df.empty:
                    sel_label = ', '.join(selected_cots)
                    st_d_children.append(wd.HTML(f"<hr><h4>COT: {sel_label}</h4>"))
                    sel_stats = cot_summary_stats(fp_selected_cot_df)
                    st_d_children.append(fp_grid_cot_stats(sel_stats))
                    sel_seas = cot_seasonality(fp_selected_cot_df)
                    st_d_children.append(wd.HBox([
                        fp_plot_positions_basket(fp_selected_cot_df),
                        fp_plot_long_short_net(fp_selected_cot_df)
                    ]))
                    st_d_children.append(wd.HBox([
                        fp_plot_dispersion(sel_seas, fp_selected_cot_df),
                        fp_plot_multi_year(fp_selected_cot_df)
                    ]))
                st_d = wd.VBox(st_d_children)

                # Sub-tab E: Regressão OLS — LevETF Flow vs Retorno D+1
                from scipy import stats as _scipy_stats
                st_e_children = []
                if not fp_flow_hist.empty and len(fp_flow_hist) >= 20:
                    _df_reg = fp_flow_hist[['Return', 'LevETF_Flow']].dropna().copy()
                    _y_all = _df_reg['Return'].shift(-1) * 100   # next-day return %
                    _x_all = _df_reg['LevETF_Flow'] / 1e9         # $B
                    _mask  = _y_all.notna() & _x_all.notna()
                    _x_reg = _x_all[_mask].values
                    _y_reg = _y_all[_mask].values

                    _slope, _intc, _r, _pval, _se = _scipy_stats.linregress(_x_reg, _y_reg)
                    _r2    = _r ** 2
                    _tstat = _slope / (_se + 1e-10)
                    _sig   = '***' if _pval < 0.01 else '**' if _pval < 0.05 else '*' if _pval < 0.10 else 'n.s.'
                    _sig_color = _C['green'] if _pval < 0.05 else _C['orange'] if _pval < 0.10 else _C['text_muted']

                    # Scatter + regression line
                    _x_line = np.linspace(_x_reg.min(), _x_reg.max(), 100)
                    _y_line = _slope * _x_line + _intc
                    _scatter_fig = go.FigureWidget()
                    _scatter_fig.add_trace(go.Scatter(
                        x=_x_reg, y=_y_reg, mode='markers',
                        marker=dict(color=_C['accent'], size=4, opacity=0.5),
                        name='Observações'))
                    _scatter_fig.add_trace(go.Scatter(
                        x=_x_line, y=_y_line, mode='lines',
                        line=dict(color=_C['red'], width=2),
                        name=f'OLS fit (R²={_r2:.3f})'))
                    _scatter_fig.add_hline(y=0, line_color=_C['border'], line_width=1)
                    _scatter_fig.add_vline(x=0, line_color=_C['border'], line_width=1)
                    _scatter_fig.update_layout(
                        title='LevETF Flow ($B) [t] vs Retorno SPX D+1 (%)',
                        xaxis_title='Fluxo LevETF ($B) — dia t',
                        yaxis_title='Retorno SPX (%) — dia t+1',
                        height=380, template=DASH_TEMPLATE,
                        margin=dict(t=40, b=30),
                        legend=dict(font=dict(size=10)))

                    # Stats card
                    _stats_html = (
                        "<div class='mm-dash'><div class='mm-card'>"
                        "<h3>Regressão OLS: LevETF Flow → Retorno SPX D+1</h3>"
                        "<p><b>Variáveis:</b> "
                        "X = fluxo estimado de ETFs alavancados no dia t ($B) | "
                        "Y = retorno do SPX no dia seguinte t+1 (%)</p>"
                        "<table class='mm-table' style='width:auto;font-size:13px;'>"
                        f"<tr><td>N observações</td><td><b>{len(_x_reg)}</b></td></tr>"
                        f"<tr><td>R²</td><td><b>{_r2:.4f}</b> "
                        f"({_r2*100:.1f}% da variância explicada)</td></tr>"
                        f"<tr><td>Coeficiente β (slope)</td><td><b>{_slope:+.4f}</b> "
                        f"pp por $B de fluxo</td></tr>"
                        f"<tr><td>Intercepto α</td><td><b>{_intc:+.4f}%</b></td></tr>"
                        f"<tr><td>t-estatístico</td><td><b>{_tstat:+.2f}</b></td></tr>"
                        f"<tr><td>p-valor</td>"
                        f"<td><b style='color:{_sig_color}'>{_pval:.4f} ({_sig})</b></td></tr>"
                        f"<tr><td>Correlação (r)</td><td><b>{_r:+.3f}</b></td></tr>"
                        "</table>"
                        "<p><small>Sinal negativo no β é esperado: fluxo de LevETF é "
                        "contra-tendência (mean-reverting). ETF compra quando mercado caiu "
                        "→ retorno D+1 tende a ser positivo.</small></p>"
                        "</div></div>")
                    st_e_children.append(wd.HTML(_stats_html))
                    st_e_children.append(_scatter_fig)
                else:
                    st_e_children.append(wd.HTML(
                        "<p style='color:#8b949e;'>Histórico insuficiente (&lt;20 obs).</p>"))
                st_e = wd.VBox(st_e_children)

                # Sub-tab F: Fluxos Sistemáticos (CTA + Dealer + Vol Control + Risk Parity)
                print("[UI] st_f: INICIO")
                st_f_children = [wd.HTML(
                    "<div class='mm-dash'><div class='mm-card'>"
                    "<h3>Fluxos Sistemáticos — CTA, Dealer/MM, Vol Control, Risk Parity</h3>"
                    "<p><small>Ref: BofA Systematic Flows Monitor methodology</small></p>"
                    "</div></div>")]

                # CTA Trend Following
                print(f"[UI] CTA data: flow={fp_cta.get('flow',0):.0f}, "
                      f"scenarios_1w={len(fp_cta_scenarios_1w)}, "
                      f"scenarios_1m={len(fp_cta_scenarios_1m)}, "
                      f"pivots={len(fp_cta_pivots)}, "
                      f"hist={len(fp_cta_hist) if not fp_cta_hist.empty else 0}")
                try:
                    _cta_flow = fp_cta.get('flow', 0)
                    _cta_trend = fp_cta.get('trend_today', 0)
                    _cta_pos = fp_cta.get('pos_today', 0)
                    _cta_pos_prev = fp_cta.get('pos_prev', 0)
                    _cta_color = _C['green'] if _cta_flow > 0 else _C['red'] if _cta_flow < 0 else _C['text_muted']
                    _cta_dir = 'COMPRA' if _cta_flow > 0 else 'VENDA' if _cta_flow < 0 else 'FLAT'
                    _trend_bar = '█' * max(1, int(abs(_cta_trend) * 10))
                    _trend_color = _C['green'] if _cta_trend > 0 else _C['red'] if _cta_trend < 0 else _C['text_muted']

                    # ── 1. Current status summary ──
                    _cta_html = (
                        f"<div class='mm-dash'><div class='mm-card'>"
                        f"<h4>CTA / Trend Following (AUM: ~$340B, ~25% equity = ~$85B)</h4>"
                        f"<table class='mm-table' style='width:auto;'>"
                        f"<tr><td>Trend Strength:</td>"
                        f"<td style='color:{_trend_color}'><b>{_cta_trend:+.3f}</b> "
                        f"<span style='font-size:10px;'>{_trend_bar}</span></td></tr>"
                        f"<tr><td>Posição CTA:</td>"
                        f"<td>{_cta_pos_prev:+.3f}x → <b>{_cta_pos:+.3f}x</b></td></tr>"
                        f"<tr><td>Fluxo Estimado (hoje):</td>"
                        f"<td style='color:{_cta_color}'><b>${_cta_flow/1e9:,.2f}B</b> ({_cta_dir})</td></tr>"
                        f"</table>")

                    # ── 2. Scenario table (1W + 1M) ──
                    if fp_cta_scenarios_1w and fp_cta_scenarios_1m:
                        _card2 = _C['card2']
                        _border = _C['border']
                        _cta_html += (
                            f"<h4 style='margin-top:16px;'>📊 CTA Estimated Flows by Scenario</h4>"
                            f"<table class='mm-table' style='width:100%;font-size:12px;'>"
                            f"<tr style='background:{_card2};'>"
                            f"<th rowspan='2'>Cenário</th>"
                            f"<th colspan='3' style='text-align:center;border-bottom:1px solid {_border};'>1 Week</th>"
                            f"<th colspan='3' style='text-align:center;border-bottom:1px solid {_border};'>1 Month</th></tr>"
                            f"<tr style='background:{_card2};'>"
                            f"<th>SPX End</th><th>Flow ($B)</th><th>Pos End</th>"
                            f"<th>SPX End</th><th>Flow ($B)</th><th>Pos End</th></tr>")
                        for s1w, s1m in zip(fp_cta_scenarios_1w, fp_cta_scenarios_1m):
                            _f1w = s1w['flow_total']
                            _f1m = s1m['flow_total']
                            _fc1w = _C['green'] if _f1w > 0 else _C['red'] if _f1w < 0 else _C['text_muted']
                            _fc1m = _C['green'] if _f1m > 0 else _C['red'] if _f1m < 0 else _C['text_muted']
                            _row_bg = ''
                            if 'Down' in s1w['name']:
                                _row_bg = f" style='background:rgba(255,77,77,0.08);'"
                            elif 'Up' in s1w['name']:
                                _row_bg = f" style='background:rgba(0,200,100,0.08);'"
                            _cta_html += (
                                f"<tr{_row_bg}>"
                                f"<td><b>{s1w['name']}</b></td>"
                                f"<td>{s1w['spx_end']:,.0f} ({s1w['pct_move']:+.1%})</td>"
                                f"<td style='color:{_fc1w}'><b>${_f1w/1e9:,.1f}B</b></td>"
                                f"<td>{s1w['pos_end']:+.2f}x</td>"
                                f"<td>{s1m['spx_end']:,.0f} ({s1m['pct_move']:+.1%})</td>"
                                f"<td style='color:{_fc1m}'><b>${_f1m/1e9:,.1f}B</b></td>"
                                f"<td>{s1m['pos_end']:+.2f}x</td>"
                                f"</tr>")
                        _cta_html += f"</table>"

                    # ── 3. Pivot levels ──
                    if fp_cta_pivots:
                        _card2 = _C['card2']
                        _cta_html += (
                            f"<h4 style='margin-top:16px;'>🎯 CTA Pivot Levels — Trigger Thresholds</h4>"
                            f"<table class='mm-table' style='width:auto;font-size:12px;'>"
                            f"<tr style='background:{_card2};'>"
                            f"<th>Horizonte</th><th>MA Pair</th><th>Nível</th>"
                            f"<th>Tipo</th><th>Distância</th><th>Posição Atual</th></tr>")
                        for pv in fp_cta_pivots:
                            _pv_color = _C['red'] if 'SELL' in pv['type'] else _C['green']
                            _pv_icon = '🔻' if 'SELL' in pv['type'] else '🔺'
                            # Posição atual: above_now=True → fast > slow → COMPRADO
                            _pos_now = pv.get('above_now', None)
                            if _pos_now is True:
                                _pos_label = '🟢 COMPRADO'
                                _pos_color = _C['green']
                            elif _pos_now is False:
                                _pos_label = '🔴 VENDIDO'
                                _pos_color = _C['red']
                            else:
                                _pos_label = '—'
                                _pos_color = _C['text']
                            _cta_html += (
                                f"<tr>"
                                f"<td><b>{pv['label']}</b></td>"
                                f"<td>{pv['ma_pair']}</td>"
                                f"<td><b>{pv['level']:,.0f}</b></td>"
                                f"<td style='color:{_pv_color}'>{_pv_icon} {pv['type']}</td>"
                                f"<td>{pv['distance_pct']:+.1%}</td>"
                                f"<td style='color:{_pos_color};font-weight:bold'>{_pos_label}</td>"
                                f"</tr>")
                        _cta_html += (
                            f"</table>"
                            f"<p><small>Nível de preço que causaria flip do sinal de MA cross. "
                            f"Posição Atual = lado corrente do CTA. "
                            f"Mais próximo do spot = maior risco de trigger.</small></p>")

                    _cta_html += (
                        f"<p><small>Trend = média de MA crosses (5/20, 5/60, 10/60, 20/120, 20/200). "
                        f"Sizing = trend/vol. CTAs ajustam diariamente. "
                        f"Vol spike em cenários Down: RV_end = RV × (1 + |move| × 3).</small></p>"
                        f"</div></div>")
                    st_f_children.append(wd.HTML(_cta_html))

                    # ── 4. CTA chart: historical + scenario fan ──
                    _has_hist = (not fp_cta_hist.empty and len(fp_cta_hist) > 5)
                    _has_scen = bool(fp_cta_scenarios_1w and fp_cta_scenarios_1m)
                    if _has_hist or _has_scen:
                        _gs_fig = build_cta_gs_chart(
                            fp_cta_hist, fp_cta_scenarios_1w,
                            fp_cta_scenarios_1m, spot)
                        st_f_children.append(_flow_border(go.FigureWidget(_gs_fig)))
                except Exception as _cta_err:
                    _tb = traceback.format_exc()
                    print(f"⚠️ CTA rendering: {_cta_err}\n{_tb}")
                    st_f_children.append(wd.HTML(
                        f"<div class='mm-dash'><div class='mm-card'>"
                        f"<h4>CTA / Trend Following</h4>"
                        f"<p style='color:red'>Erro ao renderizar CTA: {_cta_err}</p>"
                        f"</div></div>"))

                # Dealer flow + MM VaR by book + Options volume
                _dl_color = _C['green'] if fp_dealer_flow > 0 else _C['red'] if fp_dealer_flow < 0 else _C['text_muted']
                _dl_dir = 'COMPRA' if fp_dealer_flow > 0 else 'VENDA' if fp_dealer_flow < 0 else 'NEUTRO'
                _card2 = _C['card2']

                _dl_html = (
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h4>Dealer/Market Maker Delta Hedging</h4>"
                    f"<p>GEX-implied flow: <b style='color:{_dl_color}'>"
                    f"${fp_dealer_flow:,.0f}</b> ({_dl_dir})</p>"
                    f"<p><small>Baseado em GEX × ΔS. Dados de gamma derivados de "
                    f"signed volume (firms + MMs + flex). Short gamma → pro-cíclico.</small></p>")

                # ── MM VaR by book ──
                if fp_mm_var:
                    _v95_total = fp_mm_var_totals.get('pnl_var95', 0)
                    _v99_total = fp_mm_var_totals.get('pnl_var99', 0)
                    _dl_html += (
                        f"<h4 style='margin-top:16px;'>📊 VaR por Market Maker (proporcional ao volume)</h4>"
                        f"<p><small>VaR total (gamma exposure): "
                        f"<b>95%: ${_v95_total/1e6:,.1f}M</b> | "
                        f"<b>99%: ${_v99_total/1e6:,.1f}M</b></small></p>"
                        f"<table class='mm-table' style='width:100%;font-size:12px;'>"
                        f"<tr style='background:{_card2};'>"
                        f"<th>Market Maker</th>"
                        f"<th>Share</th>"
                        f"<th>OI (K cts)</th>"
                        f"<th>GEX/pt</th>"
                        f"<th>VaR 95%</th>"
                        f"<th>VaR 99%</th>"
                        f"<th>CVaR 95%</th>"
                        f"<th>Theta/dia</th></tr>")
                    for mm in fp_mm_var:
                        _mm_theta = mm['daily_theta']
                        _th_c = _C['green'] if _mm_theta > 0 else _C['red']
                        _dl_html += (
                            f"<tr>"
                            f"<td><b>{mm['name']}</b></td>"
                            f"<td style='text-align:center;'>{mm['share']:.0%}</td>"
                            f"<td style='text-align:right;'>{mm['oi_contracts']/1e3:,.0f}</td>"
                            f"<td style='text-align:right;'>{mm['gex_per_pt']:,.0f}</td>"
                            f"<td style='text-align:right;color:{_C['yellow']}'>"
                            f"${mm['var_95']/1e6:,.1f}M</td>"
                            f"<td style='text-align:right;color:{_C['red']}'>"
                            f"${mm['var_99']/1e6:,.1f}M</td>"
                            f"<td style='text-align:right;color:{_C['red']}'>"
                            f"${mm['cvar_95']/1e6:,.1f}M</td>"
                            f"<td style='text-align:right;color:{_th_c}'>"
                            f"${_mm_theta/1e6:,.2f}M</td>"
                            f"</tr>")
                    _dl_html += (
                        f"</table>"
                        f"<p><small>Volume shares: OVME (Bloomberg). "
                        f"VaR = 0.5 × GEX × ΔS² na pior perda (t-Student). "
                        f"Proporcional ao share de cada MM.</small></p>")

                # ── Options Volume by Trade Description ──
                _total_adc = fp_vol_data.get('total_adc', OPTIONS_TOTAL_ADC)
                _vol_src = fp_vol_data.get('source', 'fallback')
                _cv = fp_vol_data.get('call_vol', 0)
                _pv = fp_vol_data.get('put_vol', 0)
                _pcr = fp_vol_data.get('pc_ratio', 0)
                _vol_extra = ''
                if _vol_src == 'BQL':
                    _vol_extra = (
                        f" │ Call: {_cv:,.0f} │ Put: {_pv:,.0f}"
                        f" │ P/C: {_pcr:.2f}")
                _dl_html += (
                    f"<h4 style='margin-top:16px;'>📈 Mercado de Opções — Volume por Trade Type</h4>"
                    f"<p>Total Avg Daily Contracts: <b>{_total_adc:,.0f}</b>"
                    f" <small>({_vol_src}{_vol_extra})</small></p>"
                    f"<table class='mm-table' style='width:100%;font-size:12px;'>"
                    f"<tr style='background:{_card2};'>"
                    f"<th>Trade Description</th>"
                    f"<th>%</th>"
                    f"<th>Avg Daily Contracts</th></tr>")
                for _td_name, _td_pct in OPTIONS_TRADE_DESC:
                    _td_vol = _total_adc * _td_pct / 100
                    _bar_w = max(1, int(_td_pct * 2))
                    _dl_html += (
                        f"<tr>"
                        f"<td>{_td_name}</td>"
                        f"<td style='text-align:right;'><b>{_td_pct:.2f}%</b></td>"
                        f"<td style='text-align:right;'>{_td_vol:,.0f}"
                        f"<span style='display:inline-block;width:{_bar_w}px;"
                        f"height:10px;background:{_C['green']};margin-left:6px;"
                        f"vertical-align:middle;border-radius:2px;'></span></td>"
                        f"</tr>")
                _dl_html += (
                    f"</table>"
                    f"<p><small>Fonte: Bloomberg OVME. Electronic inclui 0DTE + 10-14DTE. "
                    f"Multi-leg = spreads, combos, butterflies.</small></p>")

                _dl_html += f"</div></div>"
                st_f_children.append(wd.HTML(_dl_html))

                # Vol control
                vc_total = fp_volctrl.get('total', 0)
                _vc_color = _C['green'] if vc_total > 0 else _C['red'] if vc_total < 0 else _C['text_muted']
                vc_rows = ""
                for tv_k, tv_v in fp_volctrl.get('detail', {}).items():
                    tv_flow = tv_v.get('flow', 0)
                    tv_c = _C['green'] if tv_flow > 0 else _C['red'] if tv_flow < 0 else _C['text_muted']
                    _tv_daily = tv_v.get('daily_flow', 0)
                    _tv_dc = _C['green'] if _tv_daily > 0 else _C['red'] if _tv_daily < 0 else _C['text_muted']
                    vc_rows += (
                        f"<tr><td>Target {tv_k}</td>"
                        f"<td style='text-align:right;'>${tv_v.get('aum', 0)/1e9:,.0f}B</td>"
                        f"<td style='text-align:right;'>{tv_v.get('exposure_old', 0):.2f}x</td>"
                        f"<td style='text-align:right;'>{tv_v.get('exposure_new', 0):.2f}x</td>"
                        f"<td style='text-align:right;color:{tv_c}'>${tv_flow/1e9:,.2f}B</td>"
                        f"<td style='text-align:right;color:{_tv_dc}'>${_tv_daily/1e9:,.2f}B</td></tr>")
                _vc_daily_total = fp_volctrl.get('daily_total', 0)
                _vc_dtc = _C['green'] if _vc_daily_total > 0 else _C['red'] if _vc_daily_total < 0 else _C['text_muted']
                st_f_children.append(wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h4>Equity Vol Control (AUM: ~$300B)</h4>"
                    f"<p>Fluxo total estimado: <b style='color:{_vc_color}'>"
                    f"${vc_total/1e9:,.2f}B</b> | Fluxo/dia: <b style='color:{_vc_dtc}'>"
                    f"${_vc_daily_total/1e9:,.2f}B</b></p>"
                    f"<table class='mm-table'>"
                    f"<tr><th>Target Vol</th><th>AUM Est.</th>"
                    f"<th>Exp. Anterior</th><th>Exp. Atual</th><th>Fluxo Total</th><th>Fluxo/Dia</th></tr>"
                    f"{vc_rows}</table>"
                    f"<p><small>Leverage = target_vol / realized_vol (21d). Piso mínimo 20% exposição. "
                    f"Vol sobe → exposure cai → vendem. Ajuste ~25%/dia (~4 dias para completar).</small></p>"
                    f"</div></div>"))

                # Vol control stress scenarios
                if fp_vc_scenarios:
                    sc_rows = ""
                    for sc in fp_vc_scenarios:
                        sc_c = _C['red'] if sc['flow'] < 0 else _C['green']
                        _sc_df = sc.get('daily_flow', 0)
                        _sc_dc = _C['red'] if _sc_df < 0 else _C['green']
                        sc_rows += (
                            f"<tr><td style='text-align:center;'>{sc['rv_shock']:.0%}</td>"
                            f"<td style='text-align:right;color:{sc_c}'>"
                            f"${sc['flow']/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_sc_dc}'>"
                            f"${_sc_df/1e9:,.1f}B</td></tr>")
                    st_f_children.append(wd.HTML(
                        f"<div class='mm-dash'><div class='mm-card'>"
                        f"<h4>⚡ Vol Spike Scenarios — Venda Forçada</h4>"
                        f"<p><small>Se realized vol subir para esses níveis, "
                        f"quanto os fundos vol-control precisam vender:</small></p>"
                        f"<table class='mm-table'>"
                        f"<tr><th>RV Spike Para</th><th>Fluxo Total</th><th>Fluxo/Dia</th></tr>"
                        f"{sc_rows}</table>"
                        f"<p><small>Piso mínimo 20% exposição. "
                        f"Ajuste gradual ~25%/dia (~4 dias para completar).</small></p>"
                        f"</div></div>"))

                # Risk Parity
                rp_total = fp_rp.get('total', 0)
                _rp_color = _C['green'] if rp_total > 0 else _C['red'] if rp_total < 0 else _C['text_muted']
                rp_eq_new = fp_rp.get('eq_alloc_new', 0)
                rp_eq_old = fp_rp.get('eq_alloc_old', 0)
                rp_rows = ""
                for rp_k, rp_v in fp_rp.get('detail', {}).items():
                    rp_f = rp_v.get('flow', 0)
                    rp_c = _C['green'] if rp_f > 0 else _C['red'] if rp_f < 0 else _C['text_muted']
                    rp_rows += (
                        f"<tr><td>Target {rp_k}</td>"
                        f"<td style='text-align:right;'>${rp_v.get('aum', 0)/1e9:,.0f}B</td>"
                        f"<td style='text-align:right;'>{rp_v.get('leverage_old', 0):.2f}x</td>"
                        f"<td style='text-align:right;'>{rp_v.get('leverage_new', 0):.2f}x</td>"
                        f"<td style='text-align:right;'>{rp_v.get('eq_alloc_old', 0):.1%}</td>"
                        f"<td style='text-align:right;'>{rp_v.get('eq_alloc_new', 0):.1%}</td>"
                        f"<td style='text-align:right;color:{rp_c}'>${rp_f/1e9:,.2f}B</td></tr>")
                st_f_children.append(wd.HTML(
                    f"<div class='mm-dash'><div class='mm-card'>"
                    f"<h4>Risk Parity (AUM: ~$200B, equity alloc: {rp_eq_old:.1%} → {rp_eq_new:.1%})</h4>"
                    f"<p>Fluxo equity estimado: <b style='color:{_rp_color}'>"
                    f"${rp_total/1e9:,.2f}B</b></p>"
                    f"<table class='mm-table'>"
                    f"<tr><th>Target Vol</th><th>AUM Est.</th>"
                    f"<th>Lev. Ant.</th><th>Lev. Atual</th>"
                    f"<th>Eq% Ant.</th><th>Eq% Atual</th><th>Fluxo</th></tr>"
                    f"{rp_rows}</table>"
                    f"<p><small>Alocação ∝ 1/vol por asset class (equities, bonds, commodities). "
                    f"Rebalanceamento mensal. Vol↑ → equity alloc↓ → vendem.</small></p>"
                    f"</div></div>"))

                # Combined flow scenarios table
                if fp_combined_scenarios:
                    cs_rows = ""
                    for cs in fp_combined_scenarios:
                        _cs_color = _C['red'] if cs['total'] < 0 else _C['green']
                        _cs_vanna = cs.get('vanna', 0)
                        _cs_charm = cs.get('charm', 0)
                        cs_rows += (
                            f"<tr>"
                            f"<td>{cs['name']}</td>"
                            f"<td style='text-align:center;'>{cs['spx_move']:+.0%}</td>"
                            f"<td style='text-align:center;'>{cs['rv_shock']:.0%}</td>"
                            f"<td style='text-align:right;color:{_C['red'] if cs['vol_ctrl'] < 0 else _C['text_muted']}'>"
                            f"${cs['vol_ctrl']/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_C['red'] if cs['risk_parity'] < 0 else _C['text_muted']}'>"
                            f"${cs['risk_parity']/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_C['red'] if cs['cta'] < 0 else _C['text_muted']}'>"
                            f"${cs['cta']/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_C['red'] if cs['dealer'] < 0 else _C['text_muted']}'>"
                            f"${cs['dealer']/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_C['red'] if _cs_vanna < 0 else _C['text_muted']}'>"
                            f"${_cs_vanna/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_C['red'] if _cs_charm < 0 else _C['text_muted']}'>"
                            f"${_cs_charm/1e9:,.1f}B</td>"
                            f"<td style='text-align:right;color:{_cs_color};font-weight:bold'>"
                            f"${cs['total']/1e9:,.1f}B</td>"
                            f"</tr>")
                    st_f_children.append(wd.HTML(
                        f"<div class='mm-dash'><div class='mm-card'>"
                        f"<h4>⚡ Cenários de Stress — Fluxo Combinado por Componente</h4>"
                        f"<p><small>Estimativa de venda forçada se SPX cair e vol subir. "
                        f"Mostra fluxo de cada estratégia sistemática + vanna de opções:</small></p>"
                        f"<table class='mm-table'>"
                        f"<tr><th>Cenário</th><th>SPX</th><th>RV</th>"
                        f"<th>Vol Ctrl</th><th>Risk Parity</th>"
                        f"<th>CTA</th><th>Dealer</th><th>Vanna</th><th>Charm</th>"
                        f"<th>TOTAL</th></tr>"
                        f"{cs_rows}</table>"
                        f"<p><small>Vol Ctrl/RP: ajuste ~25%/dia (~4d). CTA: ajustam diariamente. "
                        f"Dealer: instantâneo (delta hedge). Vanna: -vanna_notional × ΔVol. "
                        f"Charm: -charm_notional (decay diário do delta, constante). "
                        f"Valores estimados, não garantidos.</small></p>"
                        f"</div></div>"))

                print(f"[UI] st_f: {len(st_f_children)} children")
                st_f = wd.VBox(st_f_children)

                # ─── Sub-tab Dispersão (st_g) ────────────────────────────
                st_g_children = []
                if disp_ok:
                    try:
                        # Dispersion signal chart
                        disp_chart = build_dispersion_chart(
                            disp_result['disp_signal'],
                            disp_result.get('impl_corr_cboe'))
                        st_g_children.append(disp_chart)

                        # Correlation regime chart
                        if not disp_result['real_corr'].empty:
                            corr_chart = build_corr_regime_chart(
                                disp_result['real_corr'])
                            st_g_children.append(corr_chart)

                        # Current signal
                        if not disp_result['disp_signal'].empty:
                            last_row = disp_result['disp_signal'].iloc[-1]
                            sig_color = '#238636' if 'SHORT' in str(last_row.get('signal', '')) else (
                                '#da3633' if 'LONG' in str(last_row.get('signal', '')) else '#8b949e')
                            _ic = last_row.get('impl_corr', 0)
                            _rc = last_row.get('real_corr', 0)
                            _sp = last_row.get('spread', 0)
                            _zs = last_row.get('z_score', 0)
                            _sg = last_row.get('signal', 'N/A')
                            st_g_children.append(wd.HTML(
                                f"<div style='background:#161b22; padding:10px; "
                                f"border-radius:6px; margin:5px 0;'>"
                                f"<b style='color:{sig_color}; font-size:14px;'>"
                                f"Sinal Atual: {_sg}</b><br>"
                                f"<span style='color:#c9d1d9; font-size:12px;'>"
                                f"Impl Corr: {_ic:.3f} │ Real Corr: {_rc:.3f} │ "
                                f"Spread: {_sp:.3f} │ Z-Score: {_zs:.2f}</span>"
                                f"</div>"))

                        # Mag7 pairs dispersion table
                        if not disp_result['mag7_pairs'].empty:
                            st_g_children.append(_disp_table_widget(
                                disp_result['mag7_pairs'],
                                title='Mag7 — Dispersão por Par'))

                        # Best NxN dispersion trade (2x2, 3x3, etc.)
                        if disp_result['best_2x2']:
                            b22_df = pd.DataFrame(disp_result['best_2x2'])
                            st_g_children.append(_disp_table_widget(
                                b22_df,
                                title='Melhor Trade NxN (Mag7) — ⭐ = Combo Ótimo'))

                        # Best pair combos (1-pair, 2-pair, 3-pair)
                        if disp_result.get('best_pairs'):
                            bp_df = pd.DataFrame(disp_result['best_pairs'])
                            st_g_children.append(_disp_table_widget(
                                bp_df,
                                title='Melhores Combinações de Pares (até 3)'))

                        # Optimal tracking basket
                        ob = disp_result.get('optimal_basket', {})
                        if ob.get('weights'):
                            ob_df = pd.DataFrame([
                                {'Ticker': t.split(' ')[0],
                                 'Peso': round(w * 100, 1)}
                                for t, w in sorted(ob['weights'].items(),
                                                   key=lambda x: -x[1])
                            ])
                            te_val = ob.get('tracking_error', 0)
                            st_g_children.append(_disp_table_widget(
                                ob_df,
                                title='Basket Ótimo ({} stocks, TE={:.4f})'.format(
                                    len(ob_df), te_val)))

                        # Hypothesis test for R² (quality of dispersion model)
                        ht = disp_result.get('hyp_test', {})
                        if ht:
                            _r2 = ht.get('R²', 0)
                            _r2_color = _C['green'] if _r2 >= 0.5 else (
                                '#f0883e' if _r2 >= 0.3 else _C['red'])
                            _sig_txt = '✅ Significativo' if ht.get('significant') else '❌ Não significativo'
                            _sig_color = _C['green'] if ht.get('significant') else _C['red']
                            st_g_children.append(wd.HTML(
                                f"<div style='background:#161b22; padding:10px; "
                                f"border-radius:6px; margin:5px 0;'>"
                                f"<b style='color:#58a6ff;'>📊 Teste de Hipótese — Modelo de Dispersão</b><br>"
                                f"<span style='color:#c9d1d9; font-size:12px;'>"
                                f"R² = <b style='color:{_r2_color}'>{_r2:.4f}</b> │ "
                                f"R² adj = {ht.get('R² adj', 0):.4f} │ "
                                f"F-stat = {ht.get('F-stat', 0):.2f} │ "
                                f"p-value = {ht.get('p-value', 'N/A')} │ "
                                f"<b style='color:{_sig_color}'>{_sig_txt}</b><br>"
                                f"n = {ht.get('n_obs', 0)} │ "
                                f"slope = {ht.get('slope', 0):.4f} │ "
                                f"intercept = {ht.get('intercept', 0):.4f}</span>"
                                f"<br><small style='color:#8b949e;'>Top-{DISP_TOP_N} membros por peso. "
                                f"R² &lt; 0.30 → modelo fraco, considere usar Mag8 / top 5.</small>"
                                f"</div>"))

                        # Bloomberg-style COR1M + DSPX + VIXEQ chart
                        _cor1m = disp_result.get('cor1m', pd.Series(dtype=float))
                        _dspx = disp_result.get('dspx', pd.Series(dtype=float))
                        _vixeq = disp_result.get('vixeq', pd.Series(dtype=float))
                        if not _cor1m.empty or not _dspx.empty or not _vixeq.empty:
                            st_g_children.append(build_dispersion_index_chart(
                                _cor1m, _dspx, _vixeq))

                        # ── Multi-window correlation heatmaps ──
                        corr_mats = disp_result.get('corr_matrices', {})
                        if corr_mats:
                            heatmap_widgets = []
                            for lbl, cmat in corr_mats.items():
                                if cmat is not None and not cmat.empty:
                                    heatmap_widgets.append(
                                        build_correlation_heatmap(cmat, title=f'Correlação — {lbl}'))
                            if heatmap_widgets:
                                st_g_children.append(wd.HBox(heatmap_widgets))

                        # ── Dispersion pairs (multi-window scoring) ──
                        dp = disp_result.get('dispersion_pairs', pd.DataFrame())
                        if not dp.empty:
                            st_g_children.append(_disp_table_widget(
                                dp[['Pair', 'Avg Corr', 'Min Corr', 'IV1 (%)',
                                    'IV2 (%)', 'IV Spread (pp)', 'Disp Score']],
                                title='Top Pares para Dispersão (Multi-Window)'))

                        # ── Straddle richness chart ──
                        rich_df = disp_result.get('straddle_richness', pd.DataFrame())
                        if not rich_df.empty:
                            st_g_children.append(build_straddle_richness_chart(rich_df))
                            st_g_children.append(_disp_table_widget(
                                rich_df,
                                title='Straddle ATM — Caro vs Barato (Mag8 + SPX)'))

                        # ── ATM Vol Matrix (Call vs Put IV) ──
                        atm_chart = disp_result.get('atm_vol_chart')
                        atm_matrix = disp_result.get('atm_vol_matrix')
                        if atm_chart is not None:
                            st_g_children.append(atm_chart)
                        if atm_matrix is not None and not atm_matrix.empty:
                            st_g_children.append(_disp_table_widget(
                                atm_matrix,
                                title='ATM Vol Matrix — Call IV vs Put IV (Mag8 + SPX)'))

                        # ── Intraday / Recent Price Chart (Mag8 + SPX) ──
                        _px_df_all = disp_result.get('prices_df', pd.DataFrame())
                        if not _px_df_all.empty and len(_px_df_all) >= 5:
                            st_g_children.append(build_intraday_mag8_chart(_px_df_all))

                        # ── Trade recommendations + interpretation ──
                        trade_recs = disp_result.get('trade_recs', pd.DataFrame())
                        trade_interp = disp_result.get('trade_interp', '')
                        if trade_interp:
                            st_g_children.append(wd.HTML(trade_interp))
                        if not trade_recs.empty:
                            st_g_children.append(_disp_table_widget(
                                trade_recs,
                                title='🎯 Recomendações de Dispersão — Straddle/Strangle'))

                        # ── ML Dispersion Model ──
                        ml = disp_result.get('ml_model', {})
                        if ml:
                            st_g_children.append(build_dispersion_ml_widget(
                                ml['accuracy'], ml['feature_importance'],
                                ml['disp_prob'], ml['features']))

                        # ── Breadth Summary Cards + KDE return distribution ──
                        _px_df = disp_result.get('prices_df', pd.DataFrame())
                        _wts = disp_result.get('weights', {})
                        if not _px_df.empty and len(_px_df) >= 2:
                            # Breadth summary cards
                            try:
                                st_g_children.append(
                                    build_mbad_summary_cards(_px_df, _wts))
                            except Exception:
                                pass

                            # KDE distribution
                            _kde_result = build_kde_distribution_chart(_px_df, _wts)
                            if isinstance(_kde_result, tuple):
                                kde_chart, kde_interp = _kde_result
                                st_g_children.append(kde_chart)
                                if kde_interp:
                                    st_g_children.append(wd.HTML(kde_interp))
                            else:
                                st_g_children.append(_kde_result)

                        # ── RV 1M × Gamma Scatter ──
                        if not gamma_hist.empty:
                            try:
                                _cur_rv_d = rv_30d if pd.notna(rv_30d) else None
                                _cur_gex_d = total_gex_val / 1e9 if 'total_gex_val' in dir() else None
                                st_g_children.append(
                                    build_rv_gamma_chart(gamma_hist,
                                                         current_gamma=_cur_gex_d,
                                                         current_rv=_cur_rv_d))
                            except Exception:
                                pass

                        # ── SPY Intraday Candlestick ──
                        try:
                            st_g_children.append(
                                build_spy_intraday_candlestick('SPY US Equity', lookback_days=5))
                        except Exception:
                            pass

                    except Exception as _dsp_err:
                        st_g_children.append(wd.HTML(
                            f"<p style='color:#da3633;'>Erro na UI de dispersão: "
                            f"{_dsp_err}</p>"))
                else:
                    if disp_w.value:
                        _de = disp_result.get('error', 'Erro desconhecido')
                        st_g_children.append(wd.HTML(
                            f"<p style='color:#8b949e;'>Dispersão: {_de}</p>"))
                    else:
                        st_g_children.append(wd.HTML(
                            "<p style='color:#8b949e;'>Marque 'Incluir Dispersão' "
                            "para analisar.</p>"))
                st_g = wd.VBox(st_g_children)

                fp_tabs = wd.Tab()
                fp_tabs.children = [st_a, st_b, st_c, st_d, st_e, st_f, st_g]
                for idx_t, nm in enumerate(['Score', 'Histórico', 'Buyback',
                                            'COT', 'Correlação', 'Sistemáticos',
                                            'Dispersão']):
                    fp_tabs.set_title(idx_t, nm)
                tab10 = fp_tabs
              else:
                reason = ("Marque 'Incluir Flow Predictor' e rode novamente."
                          if not flow_pred_w.value else "Erro na execução.")
                tab10 = wd.VBox([wd.HTML(
                    f"<h3>Flow Predictor</h3><p>{reason}</p>")])
            except Exception as _fp_ui_err:
                _fp_tb = traceback.format_exc()
                print(f"⚠️ Flow Predictor UI: {_fp_ui_err}\n{_fp_tb}")
                tab10 = wd.VBox([wd.HTML(
                    f"<h3>Flow Predictor</h3>"
                    f"<p style='color:red;'>Erro ao montar UI: {_fp_ui_err}</p>"
                    f"<pre style='font-size:10px;color:#aaa;white-space:pre-wrap;'>{_fp_tb}</pre>")])

            # ─── ABA 11: ANALYTICS AVANÇADO ─────────────────────────────
            try:
                analytics_children = []

                # ── Row 1: Tail Risk Gauge + Skew Summary ──
                tail_gauge = build_tail_gauge(
                    analytics['tail_score'], analytics['tail_interp'])
                tail_info_parts = ['<h3>Tail Risk</h3>']
                tail_info_parts.append(
                    '<p><b>Score:</b> {:.0f}/100 — {}</p>'.format(
                        analytics['tail_score'], analytics['tail_interp']))
                for comp_key, comp_val in analytics['tail_components'].items():
                    label = comp_val.get('label', comp_key)
                    val = comp_val.get('value', 0)
                    scr = comp_val.get('score', 0)
                    tail_info_parts.append(
                        '<p style="margin:2px 0;font-size:12px;">'
                        '{}: <b>{}</b> (contrib: {:.1f})</p>'.format(label, val, scr))
                tail_info_html = wd.HTML(
                    "<div class='mm-dash'><div class='mm-card'>"
                    "{}</div></div>".format(''.join(tail_info_parts)))

                skew_summary_parts = ['<h3>Skew Monitor</h3>']
                sk = analytics['skew_summary']
                if sk:
                    for key in ['atm_iv', 'put25d', 'call25d', 'risk_reversal', 'put_skew', 'call_skew']:
                        if key in sk:
                            pctile_key = '{}_pctile'.format(key)
                            if pctile_key in sk:
                                raw_pct = sk[pctile_key]
                                # call_skew: percentil invertido (baixo = calls baratas = extremo)
                                disp_pct = 100 - raw_pct if key == 'call_skew' else raw_pct
                                pctile_str = ' (pctile: {:.0f}%)'.format(disp_pct)
                            else:
                                pctile_str = ''
                            skew_summary_parts.append(
                                '<p style="margin:2px 0;font-size:12px;">'
                                '{}: <b>{}</b>{}</p>'.format(key.replace('_', ' ').title(), sk[key], pctile_str))
                else:
                    skew_summary_parts.append('<p style="color:#8b949e;">Sem dados de skew</p>')
                skew_summary_widget = wd.HTML(
                    "<div class='mm-dash'><div class='mm-card'>"
                    "{}</div></div>".format(''.join(skew_summary_parts)))

                analytics_children.append(wd.HBox([tail_gauge, tail_info_html, skew_summary_widget]))

                # ── Row 2: Skew 4-Panel Chart ──
                skew_chart = build_skew_chart(analytics['skew_df'])
                analytics_children.append(skew_chart)

                # ── Row 3: Spot-Up-Vol-Up + VIX Regression ──
                suvu = analytics['spot_vol_up']
                suvu_parts = [
                    '<h3>Spot Up / Vol Up</h3>',
                    '<p>Streak Atual: <b style="color:#f0883e;">{}</b> dias</p>'.format(
                        suvu.get('current_streak', 0)),
                    '<p>Streak Máximo: <b>{}</b></p>'.format(suvu.get('max_streak', 0)),
                    '<p>Total dias Up/Up: {} ({:.1f}%)</p>'.format(
                        suvu.get('total_days', 0), suvu.get('pct_up_up', 0)),
                ]
                suvu_widget = wd.HTML(
                    "<div class='mm-dash'><div class='mm-card'>"
                    "{}</div></div>".format(''.join(suvu_parts)))

                vix_reg = analytics['vix_reg']
                vix_parts = ['<h3>VIX vs SPX Regressão (1M)</h3>']
                if vix_reg:
                    vix_parts.append(
                        '<p>R²: <b>{:.1%}</b></p>'.format(vix_reg.get('r2', 0)))
                    vix_parts.append(
                        '<p>Slope: {} | Intercept: {}</p>'.format(
                            vix_reg.get('slope', 0), vix_reg.get('intercept', 0)))
                    vix_parts.append(
                        '<p>SPX 1M: {:.1f}% → VIX pred: <b>{:+.1f} pts</b></p>'.format(
                            vix_reg.get('last_spx_1m', 0), vix_reg.get('predicted_vix_move', 0)))
                else:
                    vix_parts.append('<p style="color:#8b949e;">Sem dados VIX</p>')
                vix_widget = wd.HTML(
                    "<div class='mm-dash'><div class='mm-card'>"
                    "{}</div></div>".format(''.join(vix_parts)))

                # VIX scatter chart
                if vix_reg and 'x' in vix_reg and len(vix_reg['x']) > 10:
                    fig_vix_sc = go.FigureWidget()
                    fig_vix_sc.add_trace(go.Scatter(
                        x=vix_reg['x'], y=vix_reg['y'],
                        mode='markers', name='1M obs',
                        marker=dict(color=_C['accent'], size=4, opacity=0.5)))
                    x_line = np.array([min(vix_reg['x']), max(vix_reg['x'])])
                    y_line = vix_reg['slope'] * x_line + vix_reg['intercept']
                    fig_vix_sc.add_trace(go.Scatter(
                        x=x_line, y=y_line, mode='lines',
                        line=dict(color=_C['red'], width=2), name='Reg'))
                    fig_vix_sc.update_layout(
                        title='VIX Move vs SPX Move (1M, R²={:.1%})'.format(vix_reg['r2']),
                        xaxis_title='SPX 1M (%)', yaxis_title='VIX 1M (pts)',
                        height=320, template=DASH_TEMPLATE)
                    analytics_children.append(wd.HBox([suvu_widget, vix_widget, fig_vix_sc]))
                else:
                    analytics_children.append(wd.HBox([suvu_widget, vix_widget]))

                # ── Row 4: OPEX Analysis ──
                opex = analytics['opex_stats']
                if opex:
                    opex_parts = [
                        '<h3>OPEX Price Action</h3>',
                        '<p>Total OPEX analisados: <b>{}</b></p>'.format(opex.get('total_opex', 0)),
                        '<p>Performance Flip: <b>{:.1f}%</b> ({} de {})</p>'.format(
                            opex.get('flip_pct', 0), opex.get('flip_count', 0), opex.get('total_opex', 0)),
                        '<p>Avg Ret Antes 5d: {:.2f}% | Depois 5d: {:.2f}%</p>'.format(
                            opex.get('avg_ret_before', 0), opex.get('avg_ret_after', 0)),
                        '<h4>Realized Vol Impact</h4>',
                        '<p>RV 5d Into OPEX: {:.1f}% | Out: {:.1f}%</p>'.format(
                            opex.get('rv5_delta_into', 0), opex.get('rv5_delta_out', 0)),
                        '<p>RV 10d Into: {:.1f}% | Out: {:.1f}%</p>'.format(
                            opex.get('rv10_delta_into', 0), opex.get('rv10_delta_out', 0)),
                    ]
                    opex_widget = wd.HTML(
                        "<div class='mm-dash'><div class='mm-card'>"
                        "{}</div></div>".format(''.join(opex_parts)))

                    # OPEX RV bar chart
                    _opex_fig = go.FigureWidget()
                    _opex_cats = ['RV 5d', 'RV 10d']
                    _opex_fig.add_trace(go.Bar(
                        x=_opex_cats,
                        y=[opex.get('rv5_delta_into', 0), opex.get('rv10_delta_into', 0)],
                        name='Into OPEX', marker_color=_C['accent']))
                    _opex_fig.add_trace(go.Bar(
                        x=_opex_cats,
                        y=[opex.get('rv5_delta_out', 0), opex.get('rv10_delta_out', 0)],
                        name='Out of OPEX', marker_color=_C['orange']))
                    _opex_fig.update_layout(
                        title='Realized Vol: Into vs Out of OPEX',
                        yaxis_title='RV Anualizada (%)', barmode='group',
                        height=300, template=DASH_TEMPLATE,
                        margin=dict(t=35, b=25))
                    analytics_children.append(wd.HBox([opex_widget, _opex_fig]))

                # ── Row 5: Dealer Scenario Matrix ──
                if not analytics['dealer_scenarios'].empty:
                    _ds_df = analytics['dealer_scenarios']
                    ds_html = _ds_df.to_html(
                        classes='mm-table', index=False, border=0)

                    # Dealer scenario bar chart
                    _ds_fig = go.FigureWidget()
                    _ds_fig.add_trace(go.Bar(
                        x=_ds_df['Move'], y=_ds_df['Dealer Gamma ($B)'],
                        name='Gamma', marker_color=_C['accent']))
                    _ds_fig.add_trace(go.Bar(
                        x=_ds_df['Move'], y=_ds_df['Vanna Flow ($B)'],
                        name='Vanna', marker_color=_C['purple']))
                    _ds_fig.add_trace(go.Bar(
                        x=_ds_df['Move'], y=_ds_df['Charm Flow ($B)'],
                        name='Charm', marker_color=_C['yellow']))
                    _ds_fig.add_trace(go.Scatter(
                        x=_ds_df['Move'], y=_ds_df['Total ($B)'],
                        name='Total', mode='lines+markers',
                        line=dict(color=_C['red'], width=2)))
                    _ds_fig.update_layout(
                        title='Dealer Scenario: Gamma + Vanna Flow',
                        yaxis_title='$B', barmode='group',
                        height=340, template=DASH_TEMPLATE,
                        margin=dict(t=35, b=25))

                    analytics_children.append(wd.HTML(
                        "<div class='mm-dash'><div class='mm-card'>"
                        "<h3>Dealer Scenario Matrix</h3>"
                        "{}</div></div>".format(ds_html)))
                    analytics_children.append(_ds_fig)

                # ── Row 6: Mag8 Rebalance Projection ──
                if not analytics['mag8_scenarios'].empty:
                    _m8_df = analytics['mag8_scenarios']
                    m8_html = _m8_df.to_html(
                        classes='mm-table', index=False, border=0)

                    # Mag8 bar chart (stacked by stock)
                    _m8_fig = go.FigureWidget()
                    _m8_stocks = [c for c in _m8_df.columns if c not in ('Move', 'Total')]
                    _m8_colors = ['#58a6ff', '#3fb950', '#f0883e', '#da3633',
                                  '#d29922', '#bc8cff', '#8b949e', '#e6edf3']
                    for _si, _stk in enumerate(_m8_stocks):
                        _m8_fig.add_trace(go.Bar(
                            x=_m8_df['Move'], y=_m8_df[_stk],
                            name=_stk, marker_color=_m8_colors[_si % len(_m8_colors)]))
                    _m8_fig.update_layout(
                        title='Mag8 Dealer Rebalance by Stock ($B)',
                        yaxis_title='$B', barmode='relative',
                        height=340, template=DASH_TEMPLATE,
                        margin=dict(t=35, b=25),
                        legend=dict(font=dict(size=9), orientation='h', y=1.05))

                    analytics_children.append(wd.HTML(
                        "<div class='mm-dash'><div class='mm-card'>"
                        "<h3>Mag8 Dealer Rebalance Projection ($B)</h3>"
                        "{}</div></div>".format(m8_html)))
                    analytics_children.append(_m8_fig)

                # ── Row 7: Vol Control Rebalance Projection ──
                if not analytics['vol_rebal'].empty:
                    _vr_df = analytics['vol_rebal']
                    vr_html = _vr_df.to_html(
                        classes='mm-table', index=False, border=0)

                    # Vol rebal stacked bar
                    _vr_fig = go.FigureWidget()
                    _vr_fig.add_trace(go.Bar(
                        x=_vr_df['Move'], y=_vr_df['Vol Ctrl ($B)'],
                        name='Vol Ctrl', marker_color=_C['accent']))
                    _vr_fig.add_trace(go.Bar(
                        x=_vr_df['Move'], y=_vr_df['Dealer ($B)'],
                        name='Dealer', marker_color=_C['purple']))
                    _vr_fig.add_trace(go.Bar(
                        x=_vr_df['Move'], y=_vr_df['Lev ETF ($B)'],
                        name='Lev ETF', marker_color=_C['orange']))
                    _vr_fig.add_trace(go.Scatter(
                        x=_vr_df['Move'], y=_vr_df['Total ($B)'],
                        name='Total', mode='lines+markers',
                        line=dict(color=_C['red'], width=2)))
                    _vr_fig.update_layout(
                        title='Vol Ctrl + Dealer + LevETF Rebalance',
                        yaxis_title='$B', barmode='relative',
                        height=340, template=DASH_TEMPLATE,
                        margin=dict(t=35, b=25))

                    analytics_children.append(wd.HTML(
                        "<div class='mm-dash'><div class='mm-card'>"
                        "<h3>Vol Ctrl + Dealer + LevETF Rebalance Projection</h3>"
                        "{}</div></div>".format(vr_html)))
                    analytics_children.append(_vr_fig)

                # ── Row 8: Tail Risk EVT (movido da Dispersão) ──
                if disp_ok and disp_result.get('tail_risk'):
                    analytics_children.append(
                        _tail_metrics_widget(disp_result['tail_risk']))
                    if len(disp_result.get('index_returns', [])) > 50:
                        analytics_children.append(
                            build_tail_risk_chart(
                                disp_result['index_returns'],
                                disp_result['tail_risk']))

                # ── Row 9: COR1M + DSPX + VIXEQ Bloomberg-style chart ──
                if disp_ok:
                    _cor1m_a = disp_result.get('cor1m', pd.Series(dtype=float))
                    _dspx_a = disp_result.get('dspx', pd.Series(dtype=float))
                    _vixeq_a = disp_result.get('vixeq', pd.Series(dtype=float))
                    if not _cor1m_a.empty or not _dspx_a.empty or not _vixeq_a.empty:
                        analytics_children.append(
                            build_dispersion_index_chart(_cor1m_a, _dspx_a, _vixeq_a))

                # ── Row 10: RV 21d vs Gamma Index (scatter) ──
                if not gamma_hist.empty:
                    _cur_rv = rv_30d if pd.notna(rv_30d) else None
                    # Net GEX em bilhões — clamp ao range do histórico CSV para
                    # evitar que o ponto apareça fora do scatter quando o CSV foi
                    # construído com 0DTE (escala menor que o GEX full-chain)
                    if 'total_gex_val' in dir() and total_gex_val is not None:
                        _cur_gex_bn = total_gex_val / 1e9
                        _hist_gex = gamma_hist['gamma'].dropna()
                        if not _hist_gex.empty:
                            _gex_lo = _hist_gex.quantile(0.02)
                            _gex_hi = _hist_gex.quantile(0.98)
                            _cur_gex_bn = float(np.clip(_cur_gex_bn, _gex_lo, _gex_hi))
                    else:
                        _cur_gex_bn = None
                    analytics_children.append(
                        build_rv_gamma_chart(gamma_hist,
                                             current_gamma=_cur_gex_bn,
                                             current_rv=_cur_rv))

                # ── Row 11: Gamma Index + Walls time-series ──
                if not gamma_hist.empty:
                    analytics_children.append(build_gamma_ts_chart(gamma_hist))

                # ── Row 12: SPX vs Equal-Weight SPX Rolling Correlation ──
                try:
                    _ew_corr, _ew_chart = fetch_spx_eq_weight_correlation(lookback=2520)
                    analytics_children.append(_ew_chart)
                except Exception as _ew_err:
                    print(f"⚠️ EW Correlation: {_ew_err}")

                # ── Row 13: 0DTE Volume as % of Total ──
                try:
                    _odte_ratio, _odte_chart = fetch_odte_volume_pct(lookback=2000)
                    analytics_children.append(_odte_chart)
                except Exception as _odte_err:
                    print(f"⚠️ 0DTE Volume: {_odte_err}")

                if not analytics_children:
                    analytics_children.append(wd.HTML(
                        '<p style="color:#8b949e;">Sem dados de analytics.</p>'))

                tab11 = wd.VBox(analytics_children)

            except Exception as _an_ui_err:
                _an_tb = traceback.format_exc()
                print(f"⚠️ Analytics UI: {_an_ui_err}\n{_an_tb}")
                tab11 = wd.VBox([wd.HTML(
                    "<h3>Analytics</h3>"
                    f"<p style='color:red;'>Erro: {_an_ui_err}</p>"
                    f"<pre style='font-size:10px;color:#aaa;white-space:pre-wrap;'>"
                    f"{_an_tb}</pre>")])

            # ─── ABA 12: GAMMA SQUEEZE MODEL ──────────────────────────
            try:
                _sq_pc = fp_vol_data.get('pc_ratio', 1.5) or 1.5
                _sq_gex_bn = total_gex_val / 1e9 if 'total_gex_val' in dir() else (
                    total_gex / 1e9 if 'total_gex' in dir() else 0)

                # ── Fetch SPXSK3 history (SPX Skew Ratio 30d) ────────────
                _spxsk3_hist = pd.Series(dtype=float)
                _spxsk3_cur  = None
                try:
                    _spxsk3_hist = _bql_fetch_impl_corr(504)  # 2 anos
                    if not _spxsk3_hist.empty:
                        _spxsk3_cur = float(_spxsk3_hist.dropna().iloc[-1])
                except Exception as _sk3e:
                    print(f"[SPXSK3] {_sk3e}")

                # ── Fetch P/C ratio history (PUTCALLRAT Index) ────────────
                _pc_hist = pd.Series(dtype=float)
                try:
                    _dt_pc = bq.func.range('-504d', '0d')
                    _pc_req = bql.Request('PUTCALLRAT Index',
                                          {'px': bq.data.px_last(fill='PREV', dates=_dt_pc)})
                    _pc_resp = bq.execute(_pc_req)
                    _pc_hist = _bql_ts(_pc_resp[0], 'px').dropna()
                except Exception as _pce:
                    print(f"[PC hist] {_pce}")

                # ── Fetch VIX9D current ────────────────────────────────────
                _vix9d_val = None
                try:
                    _vix9d_req = bql.Request('VIX9D Index',
                                             {'px': bq.data.px_last()})
                    _vix9d_resp = bq.execute(_vix9d_req)
                    _vix9d_val = float(_vix9d_resp[0].df()['px'].iloc[0])
                except Exception as _v9e:
                    print(f"[VIX9D] {_v9e}")

                # ── Fetch Vol-of-Vol indicators (VVIX, SDEX, TDEX, VIX skew) ─
                _vvol_data = {}
                try:
                    _vvol_data = _fetch_vol_of_vol_indicators(lookback_days=252)
                except Exception as _vve:
                    print(f"[Vol-of-Vol] {_vve}")

                _sq_result = compute_gamma_squeeze_score(
                    net_gex_bn=_sq_gex_bn,
                    pc_ratio=_sq_pc,
                    iv_30d=iv_30d,
                    rv_30d=rv_30d,
                    gamma_flip=gamma_flip,
                    spot=spot,
                    skew=skew,
                    put_wall=put_wall,
                    call_wall=call_wall,
                    spxsk3_current=_spxsk3_cur,
                    spxsk3_hist=_spxsk3_hist.values if not _spxsk3_hist.empty else None,
                    pc_hist=_pc_hist.values if not _pc_hist.empty else None,
                    vix9d=_vix9d_val,
                    vvix=_vvol_data.get('vvix_cur'),
                    vix_skew_c25=_vvol_data.get('vix_skew_c25'),
                    vix_skew_p25=_vvol_data.get('vix_skew_p25'),
                    sdex_cur=_vvol_data.get('sdex_cur'),
                    tdex_cur=_vvol_data.get('tdex_cur'),
                    vix_call_oi=_vvol_data.get('vix_call_oi'),
                    vix_put_oi=_vvol_data.get('vix_put_oi'))
                tab12 = build_squeeze_tab(
                    _sq_result, _sq_gex_bn, spot, gamma_flip,
                    iv_30d, rv_30d, _sq_pc, _C, vvol_data=_vvol_data)
            except Exception as _sq_err:
                _sq_tb = traceback.format_exc()
                print(f"⚠️ Gamma Squeeze tab: {_sq_err}\n{_sq_tb}")
                tab12 = wd.VBox([wd.HTML(
                    f"<h3>Gamma Squeeze</h3>"
                    f"<p style='color:red;'>Erro: {_sq_err}</p>")])

            # ── Tab 13: Ajuste Dinâmico do Book ──────────────────────────
            try:
                # dealer_aum_bn: delta_bn raw (antes do /10 de escala BBG)
                # = sum(delta × OI × 100) × spot / 1e9 — proxy do notional total do livro
                _dealer_aum = abs(_greek_cache.get('delta_bn', 0)) * 10  # reverte escala BBG
                tab13 = build_dynamic_book_tab(df, spot, rfr, ticker=ticker,
                                               dealer_aum_bn=_dealer_aum)
            except Exception as _db_err:
                print(f"⚠️ Ajuste Dinâmico tab: {_db_err}")
                tab13 = wd.VBox([wd.HTML(
                    f"<h3 style='color:#00d4e8;'>Ajuste Dinâmico do Book</h3>"
                    f"<p style='color:#f85149;'>Erro: {_db_err}</p>")])

            # ── Tab 14: Decision Engine (0DTE Intraday) ───────────────────
            try:
                _ext_scores = {
                    'flow_score':    float(fp_score.get('score', 50)) if isinstance(fp_score, dict) else 50.0,
                    'squeeze_score': float(_sq_result_v1['score']) if '_sq_result_v1' in dir() and _sq_result_v1 else 0.0,
                    'tail_score':    float(analytics.get('tail_score', 0)) if analytics else 0.0,
                    'iv_rv_spread':  float((iv_30d - rv_30d) * 100) if pd.notna(iv_30d) and pd.notna(rv_30d) else 0.0,
                    'skew_level':    float(iv_30d * 100) if pd.notna(iv_30d) else 0.0,
                }
                tab14 = _build_decision_engine_tab_inline(df, spot, rfr, ticker, _ext_scores)
            except Exception as _de_err:
                print(f"⚠️ Decision Engine tab: {_de_err}\n{traceback.format_exc()}")
                tab14 = wd.VBox([wd.HTML(
                    f"<h3 style='color:#00d4e8;'>Decision Engine</h3>"
                    f"<p style='color:#f85149;'>Erro: {_de_err}</p>"
                    f"<pre style='font-size:10px;color:#aaa;'>{traceback.format_exc()}</pre>")])

            # ═════════════════════════════════════════════════════════════
            # MONTAGEM FINAL
            # ═════════════════════════════════════════════════════════════
            dashboard = wd.Tab()
            dashboard.children = [tab1, tab2, tab3, tab4, tab5, tab6, tab7,
                                   tab8, tab9, tab10, tab11, tab12, tab13, tab14]
            tab_names = [
                'Visão Geral', 'Exposições', 'Sensibilidade', 'Análise P&L',
                'Monte Carlo', 'Rebalanceamento', 'Previsão SPX',
                'Simulador', 'Relatório', 'Flow Predictor', 'Analytics',
                'Gamma Squeeze', 'Ajuste Dinâmico', 'Decision Engine',
            ]
            for i, name in enumerate(tab_names):
                dashboard.set_title(i, name)

            # ── Snapshot: captura conteúdo de todas as abas ──
            _snapshot['ticker'] = ticker
            _snapshot['spot'] = spot
            _snapshot['ts'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            _snapshot['sections'] = []
            _snapshot['metrics'] = {
                'gamma_flip':    gamma_flip,
                'gex_net_bn':    (_sq_gex_v1 * 0.1) if '_sq_gex_v1' in dir() else 0,
                'pc_ratio':      _sq_pc_v1  if '_sq_pc_v1'  in dir() else 0,
                'iv_rv_pp':      (iv_30d - rv_30d) * 100 if pd.notna(iv_30d) and pd.notna(rv_30d) else 0,
                'iv_30d':        iv_30d if pd.notna(iv_30d) else 0,
                'rv_30d':        rv_30d if pd.notna(rv_30d) else 0,
                'squeeze_score': (_sq_result_v1['score'] if '_sq_result_v1' in dir() and _sq_result_v1 else 0),
                'tail_score':    analytics.get('tail_score', 0) if analytics else 0,
                'call_wall':     call_wall,
                'put_wall':      put_wall,
                'daily_move':    daily_move if 'daily_move' in dir() else 0,
                'fragility':     fragility  if 'fragility'  in dir() else 0,
                'delta_bn':      _greek_cache.get('delta_bn', 0),
                'vanna_bn':      _greek_cache.get('vanna_bn', 0),
                'charm_bn':      _greek_cache.get('charm_bn', 0),
                # Flow score z-components (real BBG)
                'z_cta':         fp_score.get('z_cta', 0)        if isinstance(fp_score, dict) else 0,
                'z_dealer':      fp_score.get('z_dealer', 0)     if isinstance(fp_score, dict) else 0,
                'z_volctrl':     fp_score.get('z_volctrl', 0)    if isinstance(fp_score, dict) else 0,
                'z_rp':          fp_score.get('z_rp', 0)         if isinstance(fp_score, dict) else 0,
                'z_leveraged':   fp_score.get('z_leveraged', 0)  if isinstance(fp_score, dict) else 0,
                'z_passive_etf': fp_score.get('z_passive_etf', 0) if isinstance(fp_score, dict) else 0,
                'z_buyback':     fp_score.get('z_buyback', 0)    if isinstance(fp_score, dict) else 0,
                'z_cot':         fp_score.get('z_cot', 0)        if isinstance(fp_score, dict) else 0,
                'w_cta':         fp_score.get('weights', {}).get('cta', 0) if isinstance(fp_score, dict) else 0,
                'w_dealer':      fp_score.get('weights', {}).get('dealer', 0) if isinstance(fp_score, dict) else 0,
                'w_volctrl':     fp_score.get('weights', {}).get('volctrl', 0) if isinstance(fp_score, dict) else 0,
                'w_rp':          fp_score.get('weights', {}).get('rp', 0) if isinstance(fp_score, dict) else 0,
                'w_leveraged':   fp_score.get('weights', {}).get('leveraged', 0) if isinstance(fp_score, dict) else 0,
                'w_passive_etf': fp_score.get('weights', {}).get('passive_etf', 0) if isinstance(fp_score, dict) else 0,
                'w_buyback':     fp_score.get('weights', {}).get('buyback', 0) if isinstance(fp_score, dict) else 0,
                'w_cot':         fp_score.get('weights', {}).get('cot', 0) if isinstance(fp_score, dict) else 0,
                # Skew 25d (P25 put IV − C25 call IV), from BBG implied_volatility BQL
                'skew_25d':      round(skew * 100, 2) if pd.notna(skew) else 0,
                # Flow score total (0-100), from fp_score BBG-derived composite
                'flow_score_total': fp_score.get('score_total', 0) if isinstance(fp_score, dict) else 0,
                # VIX last price from BBG time series (fetched in analytics block)
                'vix':           float(vix_s.iloc[-1]) if ('vix_s' in dir() and not vix_s.empty) else 0,
                # Gamma Squeeze component breakdown (for bar chart)
                'squeeze_components': (_sq_result_v1['components'] if '_sq_result_v1' in dir() and _sq_result_v1 else {}),
            }

            # === Dados extras pra ZIP export (nao bloqueia dashboard) ==========
            # SPX Rebalancing prediction (probabilidades de entrada/saida)
            try:
                _snapshot['spx_prediction'] = {
                    'enabled': bool(spx_pred_ok) if 'spx_pred_ok' in dir() else False,
                    'top_in':  top_in_spx.to_dict('records')
                                 if 'top_in_spx' in dir() and top_in_spx is not None
                                    and hasattr(top_in_spx, 'to_dict') else [],
                    'top_out': top_out_spx.to_dict('records')
                                 if 'top_out_spx' in dir() and top_out_spx is not None
                                    and hasattr(top_out_spx, 'to_dict') else [],
                }
            except Exception:
                _snapshot['spx_prediction'] = {'enabled': False,
                                                'top_in': [], 'top_out': []}

            # Gamma Squeeze detalhado (curvas, componentes, score)
            try:
                _sq_full = _sq_result_v1 if '_sq_result_v1' in dir() and _sq_result_v1 else {}
                _snapshot['gamma_squeeze'] = {
                    'score':      float(_sq_full.get('score', 0)),
                    'components': dict(_sq_full.get('components', {})),
                    'interp':     str(_sq_full.get('interp', '')),
                    'net_gex_bn': float(_sq_gex_v1 * 0.1) if '_sq_gex_v1' in dir() else None,
                    'pc_ratio':   float(_sq_pc_v1) if '_sq_pc_v1' in dir() else None,
                    'gamma_flip': float(gamma_flip) if 'gamma_flip' in dir()
                                    and gamma_flip is not None else None,
                    'call_wall':  float(call_wall) if 'call_wall' in dir()
                                    and call_wall is not None else None,
                    'put_wall':   float(put_wall) if 'put_wall' in dir()
                                    and put_wall is not None else None,
                    'iv_30d':     float(iv_30d) if pd.notna(iv_30d) else None,
                    'rv_30d':     float(rv_30d) if pd.notna(rv_30d) else None,
                }
            except Exception:
                _snapshot['gamma_squeeze'] = {}

            # Tail Risk detalhado
            try:
                _snapshot['tail_risk'] = {
                    'score':      float(analytics.get('tail_score', 0)) if analytics else 0.0,
                    'components': dict(analytics.get('tail_components', {})) if analytics else {},
                    'interp':     str(analytics.get('tail_interp', '')) if analytics else '',
                    # log_returns serie historica pra CSV
                    'log_returns': analytics.get('log_returns').to_dict()
                                      if analytics and analytics.get('log_returns') is not None
                                         and hasattr(analytics.get('log_returns'), 'to_dict') else {},
                    'tail_metrics': dict(analytics.get('tail_metrics', {})) if analytics else {},
                }
            except Exception:
                _snapshot['tail_risk'] = {}

            # Dynamic Book / Ajuste Dinamico — lê de charts.LAST_DYN_BOOK_SNAPSHOT
            # (populado por build_dynamic_book_tab no baseline e cada 'Aplicar')
            try:
                from . import charts as _ch_mod
                _dyn_snap = getattr(_ch_mod, 'LAST_DYN_BOOK_SNAPSHOT', {}) or {}
                _snapshot['dynamic_book'] = dict(_dyn_snap) if _dyn_snap else {}
            except Exception:
                try:
                    import charts as _ch_mod
                    _dyn_snap = getattr(_ch_mod, 'LAST_DYN_BOOK_SNAPSHOT', {}) or {}
                    _snapshot['dynamic_book'] = dict(_dyn_snap) if _dyn_snap else {}
                except Exception:
                    _snapshot['dynamic_book'] = {}

            # Tab 2 (Exposições) usa matplotlib — captura separadamente
            try:
                mpl_items = _capture_matplotlib_figures(
                    plot_exposure_charts, agg, df, spot, from_strike,
                    to_strike, levels, model_curves, flip_points,
                    call_wall, put_wall)
                _snapshot['sections'].append({'name': 'Exposições', 'content': mpl_items})
            except Exception:
                pass

            # Todas as outras abas (recursivo via widget tree)
            for idx, (tab_w, tname) in enumerate(zip(dashboard.children, tab_names)):
                if tname == 'Exposições':
                    continue  # já capturado acima (matplotlib)
                try:
                    items = _collect_widget_content(tab_w)
                    if items:
                        _snapshot['sections'].append({'name': tname, 'content': items})
                except Exception:
                    pass

            display(title_html, dashboard)

        except Exception as e:
            clear_output(wait=True)
            print(f"ERRO NA ANÁLISE: {e}")
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — INTERFACE DE WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

run_btn.on_click(run_analysis)

# Botão de export / screenshot
export_btn = wd.Button(description='📸 Exportar HTML',
                       button_style='warning', icon='camera',
                       layout={'width': '180px'})
out_export = wd.Output()


def _on_export(_):
    with out_export:
        clear_output(wait=True)
        if not _snapshot['sections']:
            print("⚠️ Rode a análise primeiro antes de exportar.")
            return
        try:
            html_content = _export_dashboard_html()
            if not html_content:
                print("⚠️ Nenhum conteúdo para exportar.")
                return
            import base64
            fname = f"dashboard_{_snapshot['ticker'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
            b64 = base64.b64encode(html_content.encode('utf-8')).decode('ascii')
            size_mb = len(html_content) / (1024 * 1024)
            n_sections = len(_snapshot['sections'])
            n_items = sum(len(s['content']) for s in _snapshot['sections'])
            # Trigger browser download via JavaScript
            js_code = (
                f"var a = document.createElement('a');"
                f"a.href = 'data:text/html;base64,{b64}';"
                f"a.download = '{fname}';"
                f"document.body.appendChild(a);"
                f"a.click();"
                f"document.body.removeChild(a);")
            display(HTML(f"<script>{js_code}</script>"))
            display(wd.HTML(
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<p>✅ Download iniciado: <b>{fname}</b> ({size_mb:.1f} MB)</p>"
                f"<p><small>{n_sections} abas │ {n_items} itens (gráficos + tabelas)</small></p>"
                f"</div></div>"))
        except Exception as e:
            print(f"❌ Erro ao exportar: {e}")

export_btn.on_click(_on_export)

# ── ZIP Export ────────────────────────────────────────────────────────────────
zip_btn = wd.Button(
    description='📦 Export ZIP',
    button_style='info',
    icon='download',
    layout={'width': '160px'},
)
out_zip = wd.Output()


def _on_export_zip(_):
    with out_zip:
        clear_output(wait=True)
        if not _snapshot.get('ts'):
            print("⚠️ Rode a an\xe1lise primeiro.")
            return
        try:
            import io as _io
            import zipfile as _zf
            import json as _json
            import base64 as _b64

            ticker   = _snapshot['ticker']
            ts_safe  = str(_snapshot['ts']).replace(':', '').replace(' ', '_')
            zip_name = f"greeks_{ticker.replace(' ', '_')}_{ts_safe}.zip"

            buf = _io.BytesIO()
            with _zf.ZipFile(buf, 'w', _zf.ZIP_DEFLATED) as zf:
                # 1. metrics.json (summary)
                payload = {
                    'ticker':  ticker,
                    'spot':    _snapshot.get('spot'),
                    'ts':      _snapshot.get('ts'),
                    'metrics': _snapshot.get('metrics', {}),
                }
                zf.writestr('metrics.json',
                            _json.dumps(payload, indent=2, default=str))

                # 2. jarvis.html (full dashboard HTML)
                html = _export_dashboard_html()
                if html:
                    if isinstance(html, str):
                        html = html.encode('utf-8')
                    zf.writestr('jarvis.html', html)

                # 3. spx_prediction.json + CSVs (rebalanceamento SPX)
                _spx = _snapshot.get('spx_prediction') or {}
                if _spx.get('enabled') or _spx.get('top_in') or _spx.get('top_out'):
                    zf.writestr('spx_prediction.json',
                                _json.dumps(_spx, indent=2, default=str))
                    # CSVs separados se dados existirem
                    try:
                        import pandas as _pd
                        if _spx.get('top_in'):
                            _df_in = _pd.DataFrame(_spx['top_in'])
                            zf.writestr('spx_prediction_top_in.csv',
                                        _df_in.to_csv(index=False))
                        if _spx.get('top_out'):
                            _df_out = _pd.DataFrame(_spx['top_out'])
                            zf.writestr('spx_prediction_top_out.csv',
                                        _df_out.to_csv(index=False))
                    except Exception:
                        pass

                # 4. gamma_squeeze.json (score + components + niveis)
                _gs = _snapshot.get('gamma_squeeze') or {}
                if _gs:
                    zf.writestr('gamma_squeeze.json',
                                _json.dumps(_gs, indent=2, default=str))

                # 5. tail_risk.json + returns.csv
                _tr = _snapshot.get('tail_risk') or {}
                if _tr:
                    # JSON com score + components + metrics (sem log_returns,
                    # que vai separado como CSV)
                    _tr_summary = {k: v for k, v in _tr.items()
                                    if k != 'log_returns'}
                    zf.writestr('tail_risk.json',
                                _json.dumps(_tr_summary, indent=2, default=str))
                    # log_returns historicos em CSV
                    if _tr.get('log_returns'):
                        try:
                            import pandas as _pd
                            _lr = _pd.Series(_tr['log_returns'])
                            _lr.index = _pd.to_datetime(_lr.index)
                            _lr.name = 'log_return'
                            zf.writestr('tail_log_returns.csv',
                                        _lr.to_csv(header=True))
                        except Exception:
                            pass

                # 6. dynamic_book.json + CSV da expiry_agg
                _db = _snapshot.get('dynamic_book') or {}
                if _db:
                    zf.writestr('dynamic_book.json',
                                _json.dumps(_db, indent=2, default=str))
                    # CSV da agregacao por vencimento (expiry_agg)
                    if _db.get('expiry_agg'):
                        try:
                            import pandas as _pd
                            _df_agg = _pd.DataFrame(_db['expiry_agg'])
                            zf.writestr('dynamic_book_expiry_agg.csv',
                                        _df_agg.to_csv(index=False))
                        except Exception:
                            pass

            buf.seek(0)
            data = buf.read()
            b64  = _b64.b64encode(data).decode('ascii')

            js = (
                f"var a=document.createElement('a');"
                f"a.href='data:application/zip;base64,{b64}';"
                f"a.download='{zip_name}';"
                f"document.body.appendChild(a);a.click();"
                f"document.body.removeChild(a);"
            )
            display(HTML(f"<script>{js}</script>"))
            # Lista os arquivos realmente incluidos no ZIP
            try:
                _zf_read = _zf.ZipFile(_io.BytesIO(data), 'r')
                _files = sorted(_zf_read.namelist())
                _zf_read.close()
                _files_str = ', '.join(_files)
            except Exception:
                _files_str = 'metrics.json + jarvis.html + extras'
            display(wd.HTML(
                f"<div class='mm-dash'><div class='mm-card'>"
                f"<p>✅ ZIP gerado: <b>{zip_name}</b></p>"
                f"<p><small>{len(_files)} arquivos &middot; "
                f"{len(data)/1024:.0f} KB</small></p>"
                f"<p style='font-size:10px;color:#888;font-family:monospace;'>"
                f"{_files_str}</p>"
                f"</div></div>"
            ))
        except Exception as exc:
            print(f"❌ Erro ao gerar ZIP: {exc}")


zip_btn.on_click(_on_export_zip)


_ctrl_box_layout = wd.Layout(
    border=f'1px solid {_C["border"]}',
    border_radius='8px',
    padding='12px',
    margin='4px 0',
)

display(wd.HTML(DASH_CSS))
display(wd.VBox([
    wd.HTML(f"<div class='mm-dash'><div class='mm-section-label'>Parâmetros da Análise</div></div>"),
    wd.VBox([
        wd.HBox([ticker_w, dte_w]),
        mny_w,
        wd.HBox([run_btn, spx_pred_w, flow_pred_w, disp_w, cta_weight_w, export_btn, zip_btn]),
        wd.HBox([rebal_date_w]),
    ], layout=_ctrl_box_layout),
    wd.HTML(f"<div class='mm-dash'><div class='mm-section-label'>COT Controls</div></div>"),
    wd.VBox([
        wd.HBox([cot_type_w, cot_contract_w]),
        wd.HBox([cot_trader_w]),
        wd.HBox([cot_start_w, cot_end_w]),
        wd.HBox([cot_reload_btn, etf_reload_btn]),
    ], layout=_ctrl_box_layout),
    out_cot_reload,
    out_export,
    out_zip,
    out_main
]))
