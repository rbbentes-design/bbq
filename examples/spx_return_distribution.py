"""
SPX Daily Return Distribution
==============================
Visualizes the 1-day % return distribution for S&P 500 constituents,
showing equal-weighted and market-cap-weighted (free-float adjusted) KDEs,
with the top-10 heaviest names annotated.

Requires: bql, pandas, numpy, matplotlib, scipy
Environment: BQuant
"""

import bql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# 1. Data retrieval
# ---------------------------------------------------------------------------

bq = bql.Service()
universe = bq.univ.members('SPX Index')

# --- Weights: free-float shares outstanding → IWF-based weight -----------
weight_items = {
    'Free Float': bq.data.eqy_free_float_pct(fill='PREV')['VALUE'],
    'Shares':     bq.data.eqy_sh_out()['VALUE'],
}
weights = pd.concat(
    [item.df() for item in bq.execute(bql.Request(universe, weight_items))],
    axis=1,
)
weights['IWF'] = weights['Free Float'] * weights['Shares']
weights['w']   = weights['IWF'] / weights['IWF'].sum()

# --- Prices: 1-day return ------------------------------------------------
price_items = {
    'Price_prev': bq.data.px_last(dates='-1D', fill='PREV'),
    'Price_last': bq.data.px_last(fill='PREV'),
}
prices = pd.concat(
    [item.df() for item in bq.execute(bql.Request(universe, price_items))],
    axis=1,
)
prices['pct_chg'] = 100 * (prices['Price_last'] / prices['Price_prev'] - 1)

# --- Merge and derive adjusted weight -----------------------------------
merged = weights.merge(prices, how='inner', on='ID').dropna()
merged['w_adj'] = (
    (merged['Price_last'] * merged['w'])
    / (merged['Price_last'] * merged['w']).sum()
)

# ---------------------------------------------------------------------------
# 2. KDE estimation
# ---------------------------------------------------------------------------

returns_eq = prices['pct_chg'].dropna()          # equal-weighted
returns_wt = merged['pct_chg'].dropna()           # for weighted KDE

kde_eq = gaussian_kde(returns_eq)
x_eq   = np.linspace(returns_eq.min(), returns_eq.max(), 1000)
y_eq   = kde_eq(x_eq)

kde_wt = gaussian_kde(returns_wt, weights=merged['w'])
x_wt   = np.linspace(returns_wt.min(), returns_wt.max(), 1000)
y_wt   = kde_wt(x_wt)

# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------

DARK_BG = 'black'
TOP_N   = 10
SCATTER_SCALE = 3.5        # visual scale for scatter dot size axis

fig, ax1 = plt.subplots(figsize=(30, 15))
fig.patch.set_facecolor(DARK_BG)
ax1.set_facecolor(DARK_BG)
ax1.grid(axis='x', color='gray', linestyle='--')
ax1.tick_params(colors='white')
ax1.set_title(
    '1D% Return Distribution For SPX, Individual Constituents',
    color='white', fontsize='xx-large',
)

# Equal-weighted KDE fill (red / green split at zero)
ax1.plot(x_eq, y_eq, color='grey', alpha=0.5)
ax1.fill_between(x_eq[x_eq < 0], 0, y_eq[x_eq < 0], color='red',   alpha=0.5)
ax1.fill_between(x_eq[x_eq > 0], 0, y_eq[x_eq > 0], color='green', alpha=0.5)

# Up / down stock counts
n_up   = (returns_eq >= 0).sum()
n_down = (returns_eq < 0).sum()
ax1.plot([], [], ' ', label=f'Stocks Up: {n_up}')
ax1.plot([], [], ' ', label=f'Stocks Down: {n_down}')

# Secondary x-axis: weighted KDE + top-N annotations
ax2 = ax1.twiny()
ax2.tick_params(colors='white')
ax2.plot(x_wt, y_wt, color='yellow', label='Weighted distribution', alpha=1)

# Top-N heaviest names
top_n = merged.nlargest(TOP_N, 'w_adj')
top_x = top_n['pct_chg']
top_y = top_n['w_adj'] * SCATTER_SCALE
top_labels = (
    top_n.index
    .str.replace(r'\s*(Equity|UW|UQ|UN)\s*', '', regex=True)
    .str.strip()
)
ax2.scatter(top_x, top_y)
for xi, yi, label in zip(top_x, top_y, top_labels):
    ax2.text(xi, yi, label, color='white', fontsize='xx-large')

# Average return lines
avg_all  = merged['pct_chg'].mean()
avg_gain = merged.loc[merged['pct_chg'] >= 0, 'pct_chg'].mean()
avg_loss = merged.loc[merged['pct_chg'] <  0, 'pct_chg'].mean()

ax2.axvline(avg_all,  color='white', ls='--',           label=f'Avg. Daily Return: {avg_all:.1f}%')
ax2.axvline(avg_gain, color='green', ls='--', lw=2.5,   label=f'Avg. Gain: {avg_gain:.1f}%')
ax2.axvline(avg_loss, color='red',   ls='--', lw=2.5,   label=f'Avg. Decline: {avg_loss:.1f}%')

# Legends
legend_kw = dict(frameon=False, facecolor='white', edgecolor='white',
                 labelcolor='white', fontsize='xx-large')
ax1.legend(*ax1.get_legend_handles_labels(), loc='upper left',  **legend_kw)
ax2.legend(*ax2.get_legend_handles_labels(), loc='upper right', **legend_kw)

# Date stamp
current_date = datetime.now().strftime('%Y/%m/%d')
ax1.text(0.48, -0.05, f'Date: {current_date}',
         ha='center', va='center', transform=ax1.transAxes,
         color='white', fontsize='xx-large')

fig.tight_layout()
fig.savefig('spx_return_distribution.png', facecolor=DARK_BG, bbox_inches='tight')
plt.show()
