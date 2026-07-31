"""Render descriptive task-statistics figures for the test split.

fig11: session counts per task (all 75 experiments in one chart).
fig12: per-task response-label distributions (one panel per task).
fig13: Venn partition of history-dependent vs independent tasks, using
E2 (counting-exploitable sequence statistics) and E3 (causal truncation
cost beyond one segment) from the existing per-task results.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import pandas as pd

from build_per_task_figures import VIOLET, short_name
from build_report_figures import (
    BLUE,
    GRID,
    INK,
    ORANGE,
    REPO,
    SECONDARY,
    STYLE,
    style_axis,
)
from mt.models.llm.supervision import find_target_spans

# a task counts as sequence-exploitable / history-dependent above this
VENN_THRESHOLD = 0.05

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",
                        default=REPO / "data/psych-101-test/"
                                       "prompts_testing_t1.jsonl",
                        help="Psych-101-test prompts .jsonl")
    parser.add_argument("--scoring", default=REPO / "outputs/scoring",
                        help="Directory holding the score CSVs")
    parser.add_argument("--figures", default=REPO / "outputs/figures",
                        help="Output directory for the PNG figures")
    args = parser.parse_args()
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    # count from the per-choice cache with a uniform key dtype; the summary
    # CSV's mixed int/str participant ids overcount nunique by two
    scores = pd.read_csv(Path(args.scoring) / "minitaur8b_e0_full_4bit.csv",
                         low_memory=False)
    scores['experiment'] = scores['experiment'].str.replace(
        'collsi枚枚', 'collsiöö')
    scores['participant'] = scores['participant'].astype(str)
    fig11_sessions_per_task(scores, figures)

    labels = collect_labels(args.data)
    fig12_choice_distributions(labels, figures)

    report = pd.read_csv(REPO / "outputs/analysis_per_task_report.csv",
                         index_col=0)
    fig13_history_venn(report, figures)
    print(f"wrote fig11-fig13 to {figures}")

def collect_labels(path):
    """Return per-experiment Counter of marked response strings."""

    labels = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        counter = labels.setdefault(row['experiment'], Counter())
        text = row['text']
        counter.update(text[start:end].strip()
                       for start, end in find_target_spans(text))
    return labels

def fig11_sessions_per_task(scores, figures):
    counts = scores.groupby('experiment')['participant'].nunique() \
        .sort_values(ascending=False)
    n_choices = len(scores)
    fig, ax = plt.subplots(figsize=(7.6, 12.5))
    y = range(len(counts))
    ax.barh(y, counts.values, height=0.62, color=BLUE, edgecolor='none')
    ax.set_yticks(list(y), [short_name(t) for t in counts.index],
                  fontsize=6.2)
    ax.set_ylim(-1, len(counts))
    ax.invert_yaxis()
    ax.tick_params(axis='y', length=0)
    for row, value in enumerate(counts.values):
        ax.text(value + 3, row, str(value), va='center', fontsize=5.8,
                color=SECONDARY)
    style_axis(ax)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel("sessions (held-out participants)")
    ax.set_title(f"Psych-101-test: {len(counts)} tasks, "
                 f"{counts.sum():,} sessions, "
                 f"{n_choices:,} choices", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(figures / "fig11_sessions_per_task.png", dpi=200)
    plt.close(fig)

def fig12_choice_distributions(labels, figures, top=8, columns=8):
    tasks = sorted(labels)
    rows = -(-len(tasks) // columns)
    fig, axes = plt.subplots(rows, columns,
                             figsize=(1.72 * columns, 1.62 * rows))
    for ax in axes.flat[len(tasks):]:
        ax.set_visible(False)
    for ax, task in zip(axes.flat, tasks):
        counter = labels[task]
        total = sum(counter.values())
        common = counter.most_common(top)
        shares = [count / total for _, count in common]
        ax.bar(range(len(common)), shares, color=BLUE, edgecolor='none')
        ax.set_xticks(range(len(common)),
                      [label[:6] for label, _ in common],
                      fontsize=4.4, rotation=45)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_title(short_name(task), fontsize=5.8, pad=2)
        ax.text(0.97, 0.92, f"{len(counter)} labels",
                transform=ax.transAxes, fontsize=4.6, ha='right',
                color=SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    fig.suptitle("Response-label distribution per task (top "
                 f"{top} labels, share of choices; keys are randomized "
                 "per participant, so pooled spread partly reflects "
                 "randomization)", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(figures / "fig12_choice_distributions.png", dpi=200)
    plt.close(fig)

def fig13_history_venn(report, figures):
    sequence = report['uniform'] - report['bigram'] > VENN_THRESHOLD
    context = report['minitaur_e3_gap_w1'] > VENN_THRESHOLD
    both = report.index[sequence & context]
    only_sequence = report.index[sequence & ~context]
    only_context = report.index[~sequence & context]
    neither = report.index[~sequence & ~context]

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    ax.add_patch(Circle((3.9, 3.6), 2.4, facecolor=VIOLET, alpha=0.18,
                        edgecolor=VIOLET, lw=1.6))
    ax.add_patch(Circle((6.1, 3.6), 2.4, facecolor=ORANGE, alpha=0.18,
                        edgecolor=ORANGE, lw=1.6))
    ax.text(2.8, 6.15, "sequence statistics countable\n"
            f"(E2: bigram gain > {VENN_THRESHOLD})", fontsize=9,
            ha='center', color=VIOLET)
    ax.text(7.2, 6.15, "needs history beyond one trial\n"
            f"(E3: $L_{{w=1}}-L_{{full}}$ > {VENN_THRESHOLD})", fontsize=9,
            ha='center', color=ORANGE)

    def blurb(tasks, limit=3):
        names = [short_name(t) for t in tasks[:limit]]
        extra = f"\n+{len(tasks) - limit} more" if len(tasks) > limit else ""
        return "\n".join(names) + extra

    ax.text(2.7, 3.6, f"{len(only_sequence)}\n" + blurb(only_sequence),
            ha='center', va='center', fontsize=7.6, color=INK)
    ax.text(5.0, 3.6, f"{len(both)}\n" + blurb(both), ha='center',
            va='center', fontsize=7.6, color=INK)
    ax.text(7.3, 3.6, f"{len(only_context)}\n" + blurb(only_context),
            ha='center', va='center', fontsize=7.6, color=INK)
    names = [short_name(t) for t in neither]
    lines = [", ".join(names[start:start + 4])
             for start in range(0, len(names), 4)]
    ax.text(5.0, 0.55, f"neither (independent-trial): {len(neither)}\n"
            + "\n".join(lines), ha='center', va='top', fontsize=7.5,
            color=SECONDARY)
    ax.set_title("History dependence across the 75 tasks\n"
                 "(Minitaur E3 truncation cost x E2 counting baseline, "
                 "task-macro nat)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(figures / "fig13_history_dependence_venn.png", dpi=200)
    plt.close(fig)

    sets = pd.DataFrame({
        'sequence_countable': sequence,
        'needs_history': context})
    sets.to_csv(REPO / "outputs/analysis_history_dependence_sets.csv")
    print(f"venn: both={len(both)} sequence-only={len(only_sequence)} "
          f"context-only={len(only_context)} neither={len(neither)}")

if __name__ == "__main__":
    main()
