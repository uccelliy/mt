"""Render per-task appendix figures (E0/E1/E2/E3) from score CSVs.

One row per experiment across all 75 tasks: E0 full-context NLL for both
models against uniform, E1 within-session adaptation drop, E2 count
baselines, and E3 context-dependence gaps as paired heatmaps.  Also writes
a combined per-task table to outputs/analysis_per_task_report.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd

from build_report_figures import (
    BLUE,
    GRID,
    KEYS,
    MUTED,
    ORANGE,
    REPO,
    STYLE,
    WINDOWS,
    load_scores,
    replace_sessions,
    style_axis,
    task_macro,
)

AQUA = "#1baf7a"      # base_rate
MAGENTA = "#e87ba4"   # sticky
VIOLET = "#4a3aa7"    # bigram
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring", default=REPO / "outputs/scoring",
                        help="Directory holding the score CSVs")
    parser.add_argument("--figures", default=REPO / "outputs/figures",
                        help="Output directory for the PNG figures")
    args = parser.parse_args()
    scoring = Path(args.scoring)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    minitaur_e0 = replace_sessions(
        load_scores(scoring / "minitaur8b_e0_full_4bit.csv"),
        load_scores(scoring / "minitaur8b_e0_zorowitz_utf8_4bit.csv"))
    llama_e0 = load_scores(scoring / "llama31_8b_base_e0_full_4bit.csv")
    minitaur_e3 = load_scores(scoring / "minitaur8b_e3_e0grid5_4bit.csv")
    llama_e3 = load_scores(scoring / "llama31_8b_base_e3_e0grid5_4bit.csv")
    baselines = load_scores(scoring / "e2_all_tasks_s50.csv")

    table = pd.DataFrame({
        'llama_e0': task_macro(llama_e0),
        'minitaur_e0': task_macro(minitaur_e0),
        'uniform': task_macro(baselines, 'uniform'),
        'base_rate': task_macro(baselines, 'base_rate'),
        'sticky': task_macro(baselines, 'sticky'),
        'bigram': task_macro(baselines, 'bigram'),
        'llama_e1_drop': adaptation_drop(llama_e0),
        'minitaur_e1_drop': adaptation_drop(minitaur_e0),
    })
    gaps_minitaur = window_gaps(minitaur_e3)
    gaps_llama = window_gaps(llama_e3)
    for window in ['0', '1', '20']:
        table[f'minitaur_e3_gap_w{window}'] = gaps_minitaur[window]
        table[f'llama_e3_gap_w{window}'] = gaps_llama[window]

    fig_e0_per_task(table, figures)
    fig_e1_per_task(table, figures)
    fig_e2_per_task(table, figures)
    fig_e3_per_task(gaps_minitaur, gaps_llama, figures)

    out = REPO / "outputs/analysis_per_task_report.csv"
    table.round(4).to_csv(out)
    print(f"wrote per-task figures to {figures} and table to {out}")

def adaptation_drop(frame):
    """First-decile minus last-decile task-macro NLL (positive = adapts)."""

    n = frame.groupby(KEYS)['choice_index'].transform('max') + 1
    decile = (frame['choice_index'] * 10 // n).clip(upper=9)
    per_p = frame.assign(decile=decile).groupby(
        KEYS + ['decile'])['nll'].mean()
    per_e = per_p.groupby(['experiment', 'decile']).mean().unstack()
    last = per_e.apply(lambda row: row.dropna().iloc[-1], axis=1)
    return per_e[0] - last

def window_gaps(frame):
    """Per-experiment L_w - L_full for every window, token-normalized."""

    per_s = frame.groupby(KEYS + ['window']).apply(
        lambda g: g['nll'].sum() / g['num_tokens'].sum(),
        include_groups=False).rename('v').reset_index()
    per_e = per_s.groupby(['experiment', 'window'])['v'].mean().unstack()
    return per_e[WINDOWS].sub(per_e['full'], axis=0)

def short_name(experiment):
    return experiment.replace('/exp', ' e').replace('.csv', '')

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

def fig_e0_per_task(table, figures):
    order = table.sort_values('minitaur_e0', ascending=False).index
    fig, ax = plt.subplots(figsize=(7.6, 13.5))
    dot_rows(ax, order, [
        (table['uniform'], dict(s=22, facecolor='none', edgecolor=MUTED,
                                lw=1.0, label="uniform (E2)")),
        (table['llama_e0'], dict(s=22, color=BLUE, edgecolor='none',
                                 label="Llama-3.1-8B base")),
        (table['minitaur_e0'], dict(s=22, color=ORANGE, edgecolor='none',
                                    label="Minitaur-8B")),
    ])
    ax.set_xlabel("full-context task-macro choice NLL (nat)")
    ax.set_title("E0: full-context NLL per task (runtime NF4)\n"
                 "multi-token-response tasks not comparable across rows",
                 fontsize=10.5)
    ax.legend(frameon=False, loc='lower right', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(figures / "fig5_e0_per_task.png", dpi=200)
    plt.close(fig)

def fig_e1_per_task(table, figures):
    order = table.sort_values('minitaur_e1_drop', ascending=False).index
    fig, ax = plt.subplots(figsize=(7.6, 13.5))
    dot_rows(ax, order, [
        (table['llama_e1_drop'], dict(s=22, color=BLUE, edgecolor='none',
                                      label="Llama-3.1-8B base")),
        (table['minitaur_e1_drop'], dict(s=22, color=ORANGE,
                                         edgecolor='none',
                                         label="Minitaur-8B")),
    ])
    ax.axvline(0, color=MUTED, lw=1.0)
    ax.set_xlabel("within-session adaptation drop, first − last decile "
                  "(nat; positive = improves)")
    ax.set_title("E1: in-context adaptation per task\n"
                 "(from full-context E0 scores)", fontsize=10.5)
    ax.legend(frameon=False, loc='lower right', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(figures / "fig6_e1_adaptation_per_task.png", dpi=200)
    plt.close(fig)

def fig_e2_per_task(table, figures):
    order = (table['uniform'] - table['bigram']) \
        .sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(7.6, 13.5))
    dot_rows(ax, order, [
        (table['uniform'], dict(s=24, facecolor='none', edgecolor=MUTED,
                                lw=1.0, label="uniform")),
        (table['base_rate'], dict(s=20, color=AQUA, marker='^',
                                  edgecolor='none', label="base rate")),
        (table['sticky'], dict(s=20, color=MAGENTA, marker='s',
                               edgecolor='none', label="sticky")),
        (table['bigram'], dict(s=24, color=VIOLET, marker='D',
                               edgecolor='none', label="bigram")),
    ])
    ax.set_xlabel("prequential task-macro choice NLL (nat)")
    ax.set_title("E2: count-baseline NLL per task\n"
                 "sorted by bigram gain over uniform", fontsize=10.5)
    ax.legend(frameon=False, loc='lower right', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(figures / "fig7_e2_baselines_per_task.png", dpi=200)
    plt.close(fig)

def fig_e3_per_task(gaps_minitaur, gaps_llama, figures, vmax=1.5):
    order = gaps_minitaur['0'].sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 13.5), sharey=True)
    for ax, gaps, name in [(axes[0], gaps_minitaur, "Minitaur-8B"),
                           (axes[1], gaps_llama, "Llama-3.1-8B base")]:
        image = ax.imshow(gaps.loc[order, WINDOWS].values, cmap=SEQUENTIAL,
                          vmin=0, vmax=vmax, aspect='auto',
                          interpolation='nearest')
        ax.set_xticks(range(len(WINDOWS)), WINDOWS, fontsize=8)
        ax.set_xlabel("window $w$")
        ax.set_title(name, fontsize=10)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(len(order)),
                       [short_name(t) for t in order], fontsize=6.2)
    bar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02,
                       extend='max')
    bar.set_label(f"context gap  $L_w - L_{{full}}$  (nat, clipped at "
                  f"{vmax})", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    fig.suptitle("E3: context dependence per task — truncation cost by "
                 "window\n(token-normalized, five-anchor grid, runtime NF4)",
                 fontsize=10.5, y=0.995)
    fig.savefig(figures / "fig8_e3_context_gaps_per_task.png", dpi=200,
                bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()
