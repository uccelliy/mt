"""Shared figure styling, so every plot in the repo looks like one family.

Data loading and aggregation deliberately live elsewhere
(`mt.evaluation.aggregate`); this module is presentation only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
VIOLET = "#4a3aa7"
SURFACE = "#fcfcfb"

STYLE = {
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.edgecolor': MUTED,
    'axes.labelcolor': INK,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'xtick.color': SECONDARY,
    'ytick.color': SECONDARY,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
}

def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

def dot_rows(ax, order, series):
    """Draw a dot-plot: one row per task, one dot per (value, style)."""

    y = range(len(order))
    for row, task in enumerate(order):
        values = [s.loc[task] for s, _ in series if pd.notna(s.loc[task])]
        ax.hlines(row, min(values), max(values), color=GRID, lw=0.9,
                  zorder=1)
    for values, kwargs in series:
        ax.scatter(values.loc[order], y, zorder=2, **kwargs)
    ax.set_yticks(list(y), [short_name(t) for t in order], fontsize=6.2)
    ax.set_ylim(-1, len(order))
    ax.invert_yaxis()
    ax.tick_params(axis='y', length=0)
    style_axis(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=0.8)

def short_name(experiment):
    return experiment.replace('/exp', ' e').replace('.csv', '')
