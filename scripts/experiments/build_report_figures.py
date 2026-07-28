"""Render the four preliminary-report figures from completed score CSVs.

Inputs are the full E0/E3 caches of both models plus the E2 baselines;
outputs are fig1-fig4 under outputs/figures/.  Aggregation is the design's
hierarchical task-macro NLL (trial -> participant -> task) throughout, with
the CP936 collsiöö name normalization and the UTF-8 zorowitz replacement
applied to the Minitaur E0 cache before any pairing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KEYS = ['experiment', 'participant']
WINDOWS = ['0', '1', '2', '5', '10', '20', 'full']

BLUE = "#2a78d6"      # Llama base
ORANGE = "#eb6834"    # Minitaur
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
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

    minitaur_e0 = load_scores(scoring / "minitaur8b_e0_full_4bit.csv")
    minitaur_e0 = replace_sessions(
        minitaur_e0,
        load_scores(scoring / "minitaur8b_e0_zorowitz_utf8_4bit.csv"))
    llama_e0 = load_scores(scoring / "llama31_8b_base_e0_full_4bit.csv")
    minitaur_e3 = load_scores(scoring / "minitaur8b_e3_e0grid5_4bit.csv")
    llama_e3 = load_scores(scoring / "llama31_8b_base_e3_e0grid5_4bit.csv")
    baselines = load_scores(scoring / "e2_all_tasks_s50.csv")

    fig1_context_by_finetuning(minitaur_e3, llama_e3, figures)
    fig2_waterfall(minitaur_e0, llama_e0, baselines, figures)
    fig3_early_trials(minitaur_e0, llama_e0, baselines, figures)
    fig4_gain_scatter(minitaur_e0, baselines, figures)
    print(f"wrote fig1-fig4 to {figures}")

def load_scores(path):
    frame = pd.read_csv(path, low_memory=False)
    # normalize the CP936 mojibake left by early Windows runs
    frame['experiment'] = frame['experiment'].str.replace(
        'collsi枚枚', 'collsiöö')
    frame['participant'] = frame['participant'].astype(str)
    return frame

def replace_sessions(source, replacement):
    replaced = set(map(tuple, replacement[KEYS].drop_duplicates()
                       .itertuples(index=False)))
    keep = ~source.set_index(KEYS).index.isin(replaced)
    return pd.concat([source[keep], replacement], ignore_index=True)

def task_macro(frame, value='nll'):
    per_participant = frame.groupby(KEYS)[value].mean()
    return per_participant.groupby('experiment').mean()

def window_task_macro(frame):
    per_session = frame.groupby(KEYS + ['window']).apply(
        lambda g: g['nll'].sum() / g['num_tokens'].sum(),
        include_groups=False).rename('v').reset_index()
    per_experiment = per_session.groupby(['experiment', 'window'])['v'].mean()
    return per_experiment.groupby('window').mean().reindex(WINDOWS)

def trial_macro(frame, value='nll', upto=10):
    sub = frame[frame['choice_index'] <= upto]
    per_participant = sub.groupby(KEYS + ['choice_index'])[value].mean()
    per_experiment = per_participant.groupby(
        ['experiment', 'choice_index']).mean()
    return per_experiment.groupby('choice_index').mean()

def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

def fig1_context_by_finetuning(minitaur_e3, llama_e3, figures):
    """E5: window curves and near-coincident context-gain curves."""

    minitaur = window_task_macro(minitaur_e3)
    llama = window_task_macro(llama_e3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    x = range(len(WINDOWS))

    ax1.plot(x, llama.values, '-o', color=BLUE, lw=2, ms=5,
             label="Llama-3.1-8B base")
    ax1.plot(x, minitaur.values, '-o', color=ORANGE, lw=2, ms=5,
             label="Minitaur-8B")
    ax1.set_xticks(x, WINDOWS)
    ax1.set_xlabel("context window $w$ (history segments)")
    ax1.set_ylabel("task-macro NLL (nat)")
    ax1.set_title("A    NLL vs. context window")
    ax1.legend(frameon=False, loc='upper right')
    style_axis(ax1)
    ax1.annotate("cold start: finetuning\nsaves 0.40 nat", xy=(0, 2.65),
                 xytext=(0.8, 2.35), fontsize=8.5, color=SECONDARY,
                 arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    ax1.annotate("$w\\geq 1$: constant offset\n$\\approx$ 0.087 nat",
                 xy=(3, 1.39), xytext=(3.2, 1.95), fontsize=8.5,
                 color=SECONDARY,
                 arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))

    gain_llama = llama - llama['full']
    gain_minitaur = minitaur - minitaur['full']
    ax2.plot(x, gain_llama.values, '-o', color=BLUE, lw=2, ms=5,
             label="Llama-3.1-8B base")
    ax2.plot(x, gain_minitaur.values, '-o', color=ORANGE, lw=2, ms=5,
             label="Minitaur-8B")
    ax2.set_xticks(x, WINDOWS)
    ax2.set_xlabel("context window $w$ (history segments)")
    ax2.set_ylabel("context gain  $L_w - L_{full}$  (nat)")
    ax2.set_title("B    context gain: curves nearly coincide")
    ax2.legend(frameon=False, loc='upper right')
    style_axis(ax2)
    ax2.annotate("interaction at $w=0$: $-0.318$\n"
                 "[$-0.347$, $-0.287$] bootstrap",
                 xy=(0.08, 1.60), fontsize=8.5, color=SECONDARY)
    ax2.annotate("$w\\geq 1$: |interaction| < 0.004,\nall CIs cross 0",
                 xy=(1.6, 0.55), fontsize=8.5, color=SECONDARY)
    fig.suptitle("Finetuning does not improve history use — it removes the "
                 "cold start (E5, 8B, runtime NF4)", y=1.00, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figures / "fig1_e5_context_x_finetuning.png", dpi=200)
    plt.close(fig)

def fig2_waterfall(minitaur_e0, llama_e0, baselines, figures):
    """Decomposition of the full-context advantage over uniform."""

    uniform = task_macro(baselines, 'uniform').mean()
    bigram = task_macro(baselines, 'bigram').mean()
    llama = task_macro(llama_e0).mean()
    minitaur = task_macro(minitaur_e0).mean()
    steps = [
        ("uniform\nbaseline", uniform, None),
        ("generic pretraining\n+ in-context learning", llama - uniform, BLUE),
        ("Psych-101\nfinetuning", minitaur - llama, ORANGE),
        ("Minitaur-8B\nfull context", minitaur, None),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    running = 0.0
    for i, (name, value, color) in enumerate(steps):
        if color is None:
            ax.bar(i, value, 0.55, color='#c3c2b7', edgecolor='none')
            ax.text(i, value + 0.03, f"{value:.3f}", ha='center',
                    fontsize=9, color=INK)
            running = value
        else:
            ax.bar(i, value, 0.55, bottom=running, color=color,
                   edgecolor='none')
            ax.text(i, running + value / 2, f"−{abs(value):.3f}",
                    ha='center', va='center', fontsize=9,
                    color=INK if abs(value) < 0.1 else SURFACE)
            running += value
    ax.axhline(bigram, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(3.35, bigram + 0.02, f"bigram baseline {bigram:.3f}",
            fontsize=8.5, color=SECONDARY, ha='right')
    ax.set_xticks(range(4), [s[0] for s in steps], fontsize=9)
    ax.set_ylabel("task-macro NLL (nat)")
    ax.set_ylim(0, 1.78)
    ax.set_title("98% of the advantage over baseline is generic pretraining "
                 "+ ICL\n(full context, 75 tasks, 8B runtime NF4)")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(figures / "fig2_waterfall_decomposition.png", dpi=200)
    plt.close(fig)

def fig3_early_trials(minitaur_e0, llama_e0, baselines, figures):
    """E1: both models converge to a tiny gap after one trial of context."""

    minitaur = trial_macro(minitaur_e0)
    llama = trial_macro(llama_e0)
    uniform = trial_macro(baselines, 'uniform')

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(llama.index, llama.values, '-o', color=BLUE, lw=2, ms=5,
            label="Llama-3.1-8B base")
    ax.plot(minitaur.index, minitaur.values, '-o', color=ORANGE, lw=2, ms=5,
            label="Minitaur-8B")
    ax.plot(uniform.index, uniform.values, ls=(0, (4, 3)), color=MUTED,
            lw=1.6, label="uniform (E2, session alphabet)")
    ax.set_xlabel("trial position in session")
    ax.set_ylabel("task-macro choice NLL (nat)")
    ax.set_title("One trial of context replaces finetuning\n"
                 "(E1, full-context scoring, first 11 trials)")
    ax.set_xticks(range(0, 11))
    ax.legend(frameon=False, loc='upper right')
    style_axis(ax)
    ax.annotate("trial 0: both models worse than uniform;\n"
                "finetuning gap 0.44 nat", xy=(0, 2.75), xytext=(0.5, 2.9),
                fontsize=8.5, color=SECONDARY,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    ax.annotate("from trial 1: gap $\\approx$ 0.01–0.02 nat",
                xy=(2, 1.21), xytext=(3.6, 1.80), fontsize=8.5,
                color=SECONDARY,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    fig.tight_layout()
    fig.savefig(figures / "fig3_early_trial_adaptation.png", dpi=200)
    plt.close(fig)

def fig4_gain_scatter(minitaur_e0, baselines, figures):
    """Per-task model gain against the bigram-reachable sequence gain."""

    tasks = pd.DataFrame({
        'minitaur': task_macro(minitaur_e0),
        'uniform': task_macro(baselines, 'uniform'),
        'bigram': task_macro(baselines, 'bigram')})
    tasks['model_gain'] = tasks['uniform'] - tasks['minitaur']
    tasks['bigram_gain'] = tasks['uniform'] - tasks['bigram']
    clean = ['wulff2018description/exp1.csv',
             'ruggeri2022globalizability/exp1.csv',
             'hebart2023things/exp1.csv']
    memory = ['enkavi2019digitspan/exp1.csv', 'popov2023intent/exp3.csv',
              'garcia2023experiential/exp1.csv']
    rest = tasks.drop(clean + memory)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.scatter(rest['bigram_gain'], rest['model_gain'], s=26, color=BLUE,
               alpha=0.55, edgecolor='none', label="task (n=75)")
    ax.scatter(tasks.loc[memory, 'bigram_gain'],
               tasks.loc[memory, 'model_gain'], s=42, color=ORANGE,
               edgecolor='none', label="memory / recall tasks")
    ax.scatter(tasks.loc[clean, 'bigram_gain'],
               tasks.loc[clean, 'model_gain'], s=52, facecolor='none',
               edgecolor=INK, lw=1.4, label="independent-trial flagships")
    limit = max(tasks['bigram_gain'].max(), tasks['model_gain'].max()) + 0.15
    ax.plot([-0.35, limit], [-0.35, limit], ls=(0, (4, 3)), color=MUTED,
            lw=1.2)
    ax.text(1.55, 1.42, "y = x: fully explained\nby bigram counting",
            fontsize=8.5, color=SECONDARY, rotation=38)
    labels = [(clean[0], "wulff2018\ndescription", 0.07, -0.06),
              (clean[1], "ruggeri2022", 0.09, -0.14),
              (clean[2], "hebart2023", -0.52, 0.09),
              (memory[0], "digitspan", 0.06, 0.02),
              (memory[1], "popov2023", 0.06, 0.02),
              (memory[2], "garcia2023", 0.06, 0.02)]
    for name, short, dx, dy in labels:
        ax.annotate(short,
                    xy=(tasks.loc[name, 'bigram_gain'],
                        tasks.loc[name, 'model_gain']),
                    xytext=(tasks.loc[name, 'bigram_gain'] + dx,
                            tasks.loc[name, 'model_gain'] + dy),
                    fontsize=7.5, color=SECONDARY)
    r = tasks['model_gain'].corr(tasks['bigram_gain'])
    ax.text(0.03, 0.97, f"r = {r:.2f}", transform=ax.transAxes, fontsize=10,
            color=INK, va='top')
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("bigram gain over uniform (nat)  — sequence statistics")
    ax.set_ylabel("Minitaur gain over uniform (nat)")
    ax.set_title("Model advantage tracks exploitable sequence statistics\n"
                 "(negative = worse than uniform)")
    ax.legend(frameon=False, loc='lower right', fontsize=8.5)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(figures / "fig4_gain_vs_sequence_stats.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
