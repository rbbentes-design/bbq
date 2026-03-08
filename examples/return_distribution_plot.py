"""
Return Distribution Plot
========================
Reusable dark-theme KDE visualization for index constituent returns.
Expects pre-computed DataFrames (prices & merged) from a BQL pipeline.

Usage
-----
    from return_distribution_plot import plot_return_distribution
    fig = plot_return_distribution(prices, merged, title="SPX 1D Returns")
    fig.savefig("output.png")
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import gaussian_kde


def plot_return_distribution(
    prices: "pd.DataFrame",
    merged: "pd.DataFrame",
    *,
    return_col: str = "pct_chg",
    weight_col: str = "w",
    adj_weight_col: str = "w_adj",
    title: str = "1D% Return Distribution, Individual Constituents",
    top_n: int = 10,
    scatter_scale: float = 3.5,
    figsize: tuple = (30, 15),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot equal-weighted and market-cap-weighted return KDEs with annotations.

    Parameters
    ----------
    prices : DataFrame
        Must contain *return_col* (percent returns per security).
    merged : DataFrame
        prices merged with weights; must contain *return_col*,
        *weight_col*, and *adj_weight_col*.  Index = security ID.
    return_col : str
        Column name holding the 1-day % change.
    weight_col : str
        Column used as KDE weights.
    adj_weight_col : str
        Column used to rank the top-N names.
    title : str
        Chart title.
    top_n : int
        Number of heaviest names to annotate.
    scatter_scale : float
        Visual multiplier for top-N scatter dots.
    figsize : tuple
        Figure dimensions (width, height) in inches.
    save_path : str or None
        If provided, save the figure to this path before returning.

    Returns
    -------
    matplotlib.figure.Figure
    """
    BG = "black"

    # --- KDE estimation ---------------------------------------------------
    returns_eq = prices[return_col].dropna()
    kde_eq = gaussian_kde(returns_eq)
    x_eq = np.linspace(returns_eq.min(), returns_eq.max(), 1000)
    y_eq = kde_eq(x_eq)

    returns_wt = merged[return_col].dropna()
    kde_wt = gaussian_kde(returns_wt, weights=merged[weight_col])
    x_wt = np.linspace(returns_wt.min(), returns_wt.max(), 1000)
    y_wt = kde_wt(x_wt)

    # --- Figure setup -----------------------------------------------------
    fig, ax1 = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(BG)
    ax1.grid(axis="x", color="gray", linestyle="--")
    ax1.tick_params(colors="white")
    ax1.set_title(title, color="white", fontsize="xx-large")

    # Equal-weighted KDE (red/green split at zero)
    ax1.plot(x_eq, y_eq, color="grey", alpha=0.5)
    ax1.fill_between(x_eq[x_eq < 0], 0, y_eq[x_eq < 0], color="red",   alpha=0.5)
    ax1.fill_between(x_eq[x_eq > 0], 0, y_eq[x_eq > 0], color="green", alpha=0.5)

    n_up   = (returns_eq >= 0).sum()
    n_down = (returns_eq < 0).sum()
    ax1.plot([], [], " ", label=f"Stocks Up: {n_up}")
    ax1.plot([], [], " ", label=f"Stocks Down: {n_down}")

    # --- Secondary axis: weighted KDE + annotations -----------------------
    ax2 = ax1.twiny()
    ax2.tick_params(colors="white")
    ax2.plot(x_wt, y_wt, color="yellow", label="Weighted distribution")

    # Top-N heaviest names
    top = merged.nlargest(top_n, adj_weight_col)
    top_labels = (
        top.index
        .str.replace(r"\s*(Equity|UW|UQ|UN)\s*", "", regex=True)
        .str.strip()
    )
    ax2.scatter(top[return_col], top[adj_weight_col] * scatter_scale)
    for xi, yi, lbl in zip(
        top[return_col], top[adj_weight_col] * scatter_scale, top_labels
    ):
        ax2.text(xi, yi, lbl, color="white", fontsize="xx-large")

    # Average return lines
    avg_all  = merged[return_col].mean()
    avg_gain = merged.loc[merged[return_col] >= 0, return_col].mean()
    avg_loss = merged.loc[merged[return_col] <  0, return_col].mean()

    ax2.axvline(avg_all,  color="white", ls="--",         label=f"Avg. Daily Return: {avg_all:.1f}%")
    ax2.axvline(avg_gain, color="green", ls="--", lw=2.5, label=f"Avg. Gain: {avg_gain:.1f}%")
    ax2.axvline(avg_loss, color="red",   ls="--", lw=2.5, label=f"Avg. Decline: {avg_loss:.1f}%")

    # --- Legends ----------------------------------------------------------
    legend_kw = dict(
        frameon=False, facecolor="white", edgecolor="white",
        labelcolor="white", fontsize="xx-large",
    )
    ax1.legend(*ax1.get_legend_handles_labels(), loc="upper left",  **legend_kw)
    ax2.legend(*ax2.get_legend_handles_labels(), loc="upper right", **legend_kw)

    # Date stamp
    ax1.text(
        0.48, -0.05, f"Date: {datetime.now():%Y/%m/%d}",
        ha="center", va="center", transform=ax1.transAxes,
        color="white", fontsize="xx-large",
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, facecolor=BG, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Standalone usage (assumes `prices` and `merged` exist in the namespace)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # When run directly inside a BQuant notebook via %run, the calling
    # notebook must have `prices` and `merged` DataFrames in scope.
    try:
        fig = plot_return_distribution(
            prices, merged,
            title="1D% Return Distribution For SPX, Individual Constituents",
            save_path="spx_return_distribution.png",
        )
        plt.show()
    except NameError:
        print("Run the data pipeline first to create `prices` and `merged` DataFrames.")
