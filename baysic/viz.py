"""Visualization tools."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import rho_plus as rp

def ridge_plot(
    df: pd.DataFrame,
    group_var: str,
    num_var: str,
    order_by = lambda x: np.nanquantile(x, 0.1),
    height: int = 10,
    subplot_height: float = 0.8,
):
    if order_by is not None:
        order = df[[group_var, num_var]].groupby(group_var, observed=True).agg(order_by)[num_var].sort_values().index
        palette = 'rho_solara'

    else:
        order = pd.unique(df[group_var])
        palette = 'rho_iso_spectra'

    hspace = -0.5
    num_rows = int(height / ((1 + hspace) * subplot_height))
    num_cols = int(np.ceil(len(order) / num_rows))
    pad_order = np.concatenate([order.values, [np.nan for _ in range(num_rows * num_cols - len(order))]], dtype=object)
    pad_order = pad_order.reshape(num_rows, num_cols)
    order_i = 0
    for j in range(num_cols):
        for i in range(num_rows):
            if not pd.isnull(pad_order[i, j]):
                pad_order[i, j] = order[order_i]
                order_i += 1
    col_order = pad_order.flatten()
    col_order = col_order[~pd.isnull(col_order)]
    hue_order = df[[group_var, num_var]].groupby(group_var, observed=True).agg(order_by)[num_var].sort_values().index

    g = sns.FacetGrid(
        df, col=group_var, hue=group_var, aspect=8, height=subplot_height, col_wrap=num_cols,
        palette=palette, col_order=col_order, hue_order=hue_order, sharey=False)

    # Draw the densities in a few steps
    kde_params = dict(bw_adjust=0.5)
    g.map(sns.kdeplot, num_var, clip_on=False,
        fill=True, alpha=1, linewidth=1.5, edgecolor=plt.rcParams['figure.facecolor'], **kde_params)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.set_facecolor((0, 0, 0, 0))
        ax.set_ylabel(label, color=color, ha="right", rotation=0, y=0, va='bottom')

    g.map(label, "lattice_type")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=hspace)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.set_xlabels(num_var)
    g.despine(bottom=True, left=True)

    return g