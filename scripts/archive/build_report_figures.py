"""Render the main report figures from completed score CSVs.

Main figures (fig2-fig4) use the adapter-on-quantized-base run, which
reproduces the official Centaur-8B numbers to r=1.00000; the merged
Minitaur checkpoint underestimates the finetuning gain 13.7x and is kept
only for the appendix figure fig16.  fig1 (E5) still rests on Minitaur
because the adapter E3 has not been run yet, and is labelled accordingly.
Aggregation is the design's hierarchical task-macro NLL (trial ->
participant -> task) throughout, with the CP936 collsiöö name
normalization and the UTF-8 zorowitz replacement applied to the Minitaur
E0 cache before any pairing.
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
    centaur_e0 = load_scores(scoring / "centaur8b_adapter_e0_full_4bit.csv")
    llama_e0 = load_scores(scoring / "llama31_8b_base_e0_full_4bit.csv")
    minitaur_e3 = load_scores(scoring / "minitaur8b_e3_e0grid5_4bit.csv")
    llama_e3 = load_scores(scoring / "llama31_8b_base_e3_e0grid5_4bit.csv")
    baselines = load_scores(scoring / "e2_all_tasks_s50.csv")

    fig1_context_by_finetuning(minitaur_e3, llama_e3, figures)
    fig2_waterfall(centaur_e0, llama_e0, baselines, figures)
    fig3_early_trials(centaur_e0, llama_e0, baselines, figures)
    fig4_gain_scatter(centaur_e0, baselines, figures)
    fig16_quantization_damage(centaur_e0, minitaur_e0, llama_e0, figures)
    print(f"wrote fig1-fig4 and fig16 to {figures}")

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

def trial_macro(frame, value='nll', upto=10, min_length=None):
    """Task-macro NLL per absolute trial index.

    Sessions differ in length, so at high trial indices only long sessions
    remain and the curve mixes adaptation with a changing task mix. Pass
    min_length to hold the cohort (and hence the composition) fixed.
    """

    sub = frame
    if min_length is not None:
        length = sub.groupby(KEYS)['choice_index'].transform('max') + 1
        sub = sub[length >= min_length]
    sub = sub[sub['choice_index'] <= upto]
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
    ax2.text(0.5, -0.09, "⚠ measured on the merged Minitaur checkpoint, "
             "which loses 77% of the finetuning gain (fig16);\n"
             "re-run with the official adapter is pending — treat this "
             "panel as provisional", transform=ax2.transAxes, ha='center',
             va='top', fontsize=7.6, color=SECONDARY)
    fig.suptitle("E5 (provisional): on the merged checkpoint, finetuning "
                 "removes the cold start without improving history use",
                 y=1.00, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figures / "fig1_e5_context_x_finetuning.png", dpi=200)
    plt.close(fig)

def fig2_waterfall(centaur_e0, llama_e0, baselines, figures):
    """Decomposition of the full-context advantage over uniform."""

    uniform = task_macro(baselines, 'uniform').mean()
    bigram = task_macro(baselines, 'bigram').mean()
    llama = task_macro(llama_e0).mean()
    centaur = task_macro(centaur_e0).mean()
    steps = [
        ("uniform\nbaseline", uniform, None),
        ("generic pretraining\n+ in-context learning", llama - uniform, BLUE),
        ("Psych-101\nfinetuning", centaur - llama, ORANGE),
        ("Centaur-8B\nfull context", centaur, None),
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
    total = uniform - centaur
    ax.set_title(f"{(llama - uniform) / -total:.0%} of the advantage over "
                 f"baseline is generic pretraining + ICL\n"
                 f"(full context, 75 tasks, official Centaur-8B "
                 f"reproduced at runtime NF4)")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(figures / "fig2_waterfall_decomposition.png", dpi=200)
    plt.close(fig)

COHORT = 50  # keep the task mix fixed in the long-range panel

def fig3_early_trials(centaur_e0, llama_e0, baselines, figures):
    """E1: the finetuning gap is largest at the cold start but persists."""

    sessions = centaur_e0.groupby(KEYS).size()
    kept = int((sessions >= COHORT).sum())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    panels = [
        (ax1, 9, None,
         f"A    the cold start (all {len(sessions):,} sessions)"),
        (ax2, COHORT - 1, COHORT,
         f"B    the residual persists ({kept:,} sessions, "
         f"$\\geq${COHORT} trials)"),
    ]
    for ax, upto, cohort, title in panels:
        centaur = trial_macro(centaur_e0, upto=upto, min_length=cohort)
        llama = trial_macro(llama_e0, upto=upto, min_length=cohort)
        uniform = trial_macro(baselines, 'uniform', upto=upto,
                              min_length=cohort)
        ax.plot(llama.index, llama.values, '-o', color=BLUE, lw=1.8,
                ms=4 if cohort else 5, label="Llama-3.1-8B base")
        ax.plot(centaur.index, centaur.values, '-o', color=ORANGE, lw=1.8,
                ms=4 if cohort else 5,
                label="Centaur-8B (official adapter)")
        ax.plot(uniform.index, uniform.values, ls=(0, (4, 3)), color=MUTED,
                lw=1.6, label="uniform (E2, session alphabet)")
        ax.set_xlabel("trial position in session")
        ax.set_ylabel("task-macro choice NLL (nat)")
        ax.set_title(title, fontsize=10)
        style_axis(ax)
        if cohort is None:
            ax.set_xticks(range(0, upto + 1))
            ax.legend(frameon=False, loc='upper right', fontsize=8.5)
            ax.annotate(f"trial 0: base far worse than uniform,\n"
                        f"finetuned better; gap "
                        f"{llama.iloc[0] - centaur.iloc[0]:.2f} nat",
                        xy=(0, 2.2), xytext=(1.0, 2.55), fontsize=8.5,
                        color=SECONDARY,
                        arrowprops=dict(arrowstyle='-', color=MUTED,
                                        lw=0.8))
        else:
            gap = llama - centaur
            bands = [(1, 10), (10, 30), (30, COHORT)]
            for start, stop in bands:
                ax.hlines(gap[start:stop].mean() + centaur[start:stop].mean(),
                          start, stop - 1, color=SECONDARY, lw=0)
            ax.text(0.97, 0.93,
                    "gap (base − finetuned):\n"
                    + "\n".join(f"  trials {a}–{b - 1}: "
                                f"{gap[a:b].mean():.3f} nat"
                                for a, b in bands),
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8.5, color=INK)
            # a real phase boundary, not noise: popov2023intent switches
            # from encoding to free recall here (NLL 10-14 nat in 3 tasks)
            ax.annotate("recall phase begins in\npopov2023intent (3 tasks)",
                        xy=(30, llama[30]), xytext=(33, llama[30] + 0.30),
                        fontsize=8, color=SECONDARY,
                        arrowprops=dict(arrowstyle='-', color=MUTED,
                                        lw=0.8))
    fig.suptitle("Finetuning buys the cold start — and the residual does not "
                 "decay away (E1, full-context scoring)", y=0.99,
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures / "fig3_early_trial_adaptation.png", dpi=200)
    plt.close(fig)

def fig4_gain_scatter(centaur_e0, baselines, figures):
    """Per-task model gain against the bigram-reachable sequence gain."""

    tasks = pd.DataFrame({
        'centaur': task_macro(centaur_e0),
        'uniform': task_macro(baselines, 'uniform'),
        'bigram': task_macro(baselines, 'bigram')})
    tasks['model_gain'] = tasks['uniform'] - tasks['centaur']
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
    ax.set_ylabel("Centaur-8B gain over uniform (nat)")
    ax.set_title("Model advantage tracks exploitable sequence statistics\n"
                 "(negative = worse than uniform)")
    ax.legend(frameon=False, loc='lower right', fontsize=8.5)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(figures / "fig4_gain_vs_sequence_stats.png", dpi=200)
    plt.close(fig)

def fig16_quantization_damage(centaur_e0, minitaur_e0, llama_e0, figures):
    """Appendix: merging a QLoRA adapter then requantizing costs 0.15 nat."""

    tasks = pd.DataFrame({
        'llama': task_macro(llama_e0),
        'minitaur': task_macro(minitaur_e0),
        'centaur': task_macro(centaur_e0)})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.3))

    limit = 3.4
    ax1.plot([0, limit], [0, limit], ls=(0, (4, 3)), color=MUTED, lw=1.2)
    ax1.scatter(tasks['centaur'], tasks['minitaur'], s=28, color=ORANGE,
                alpha=0.7, edgecolor='none', label="merged Minitaur-8B")
    ax1.scatter(tasks['centaur'], tasks['llama'], s=28, color=BLUE,
                alpha=0.55, edgecolor='none', label="Llama-3.1-8B base")
    ax1.set_xlim(0, limit)
    ax1.set_ylim(0, limit)
    ax1.set_aspect('equal')
    ax1.set_xlabel("Centaur-8B: adapter on 4-bit base (nat)")
    ax1.set_ylabel("other checkpoint (nat)")
    ax1.set_title("A    per-task: merged weights lose most of the tuning")
    ax1.legend(frameon=False, loc='lower right', fontsize=8.5)
    style_axis(ax1)

    means = [tasks['llama'].mean(), tasks['minitaur'].mean(),
             tasks['centaur'].mean()]
    labels = ["Llama-3.1-8B\nbase", "Minitaur-8B\nmerged +\nrequantized",
              "Centaur-8B\nadapter on\n4-bit base"]
    ax2.bar(range(3), means, 0.55, color=[BLUE, '#c3c2b7', ORANGE],
            edgecolor='none')
    for i, value in enumerate(means):
        ax2.text(i - 0.32, value + 0.015, f"{value:.3f}", ha='center',
                 fontsize=9, color=INK)
    recovered = (means[0] - means[2]) - (means[0] - means[1])
    ax2.annotate("", xy=(2.42, means[2]), xytext=(2.42, means[0]),
                 arrowprops=dict(arrowstyle='<->', color=SECONDARY, lw=1.1))
    ax2.text(2.36, (means[0] + means[2]) / 2,
             f"true finetuning gain\n{means[0] - means[2]:.3f} nat",
             fontsize=8.5, color=SECONDARY, va='center', ha='right')
    ax2.text(1.0, means[1] + 0.10,
             f"merging + requantizing destroys\n{recovered:.3f} nat = "
             f"{recovered / (means[0] - means[2]):.0%} of the gain",
             fontsize=8.5, color=SECONDARY, ha='center')
    ax2.set_xticks(range(3), labels, fontsize=8)
    ax2.set_xlim(-0.6, 2.7)
    ax2.set_ylabel("task-macro NLL (nat)")
    ax2.set_ylim(0, 1.15)
    ax2.set_title("B    the same adapter, two deployment paths")
    style_axis(ax2)

    fig.suptitle("Appendix: a QLoRA adapter must stay on its quantized base\n"
                 "merging to BF16 and requantizing loses most of the "
                 "finetuning", fontsize=10.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(figures / "fig16_quantization_path_damage.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
