"""Compare our paper-protocol NF4 results with the official published CSVs.

The official numbers come from the Centaur repo's results/ directory
(github.com/marcelbinz/Llama-3.1-Centaur-70B), restricted to the 36 seen
task families evaluated with the standard eval_loss (custom_metric False,
unseen == participants).  Our numbers are the P0 paperstyle summaries.
Writes fig9 (validation scatter), fig10 (per-task dot plot with the
official cast) and outputs/analysis_official_vs_ours.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from build_per_task_figures import VIOLET, dot_rows
from build_report_figures import (
    BLUE,
    INK,
    MUTED,
    ORANGE,
    REPO,
    SECONDARY,
    STYLE,
    style_axis,
)

OFFICIAL_DEFAULT = Path(
    "~/wkspace/Llama-3.1-Centaur-70B/results").expanduser()
OFFICIAL_FILES = {
    'off_cent8': ("all_data_marcelbinz-Llama-3.1-Centaur-8B-adapter.csv",
                  "marcelbinz/Llama-3.1-Centaur-8B-adapter"),
    'off_cent70': ("all_data_marcelbinz-Llama-3.1-Centaur-70B-adapter.csv",
                   "marcelbinz/Llama-3.1-Centaur-70B-adapter"),
    'off_base8': ("all_data_unsloth-Meta-Llama-3.1-8B-bnb-4bit.csv",
                  "unsloth/Meta-Llama-3.1-8B-bnb-4bit"),
    'off_base70': ("all_data_unsloth-Meta-Llama-3.1-70B-bnb-4bit.csv",
                   "unsloth/Meta-Llama-3.1-70B-bnb-4bit"),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", default=OFFICIAL_DEFAULT,
                        help="Official results/ directory (repo clone)")
    parser.add_argument("--scoring", default=REPO / "outputs/scoring",
                        help="Directory holding our paperstyle summaries")
    parser.add_argument("--figures", default=REPO / "outputs/figures",
                        help="Output directory for the PNG figures")
    args = parser.parse_args()
    official = Path(args.official)
    scoring = Path(args.scoring)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    table = pd.DataFrame({
        'ours_llama': load_ours(
            scoring / "llama31_8b_base_paperstyle_nf4_summary.csv"),
        'ours_minitaur': load_ours(
            scoring / "minitaur8b_paperstyle_nf4_summary.csv"),
    })
    for column, (name, value_col) in OFFICIAL_FILES.items():
        table[column] = load_official(official / name, value_col)
    cog = pd.read_csv(official / "all_data_baseline.csv")
    # the cognitive baseline is missing for 2 of the 36 seen families
    table['off_cog'] = cog[cog['unseen'] == 'participants'] \
        .set_index('task')['baseline']
    missing = table.drop(columns='off_cog').isna().any(axis=1)
    if missing.any():
        raise SystemExit(f"Unmatched tasks: {list(table.index[missing])}")

    fig9_validation_scatter(table, figures)
    fig10_official_cast(table, figures)

    out = REPO / "outputs/analysis_official_vs_ours.csv"
    table.round(4).to_csv(out)
    report(table)
    print(f"wrote fig9/fig10 to {figures} and table to {out}")

def load_ours(path):
    frame = pd.read_csv(path)
    return frame.set_index('paper_task')['official_eval_loss']

def load_official(path, value_col):
    frame = pd.read_csv(path)
    frame = frame[frame['unseen'] == 'participants']
    if 'custom_metric' in frame.columns:
        frame = frame[~frame['custom_metric']]
    return frame.set_index('task')[value_col]

def report(table):
    print("36-task means:")
    print(table.mean().round(4).to_string())
    for label, ours, theirs in [
            ("base 8B (validation)", 'ours_llama', 'off_base8'),
            ("finetuned 8B", 'ours_minitaur', 'off_cent8')]:
        diff = table[ours] - table[theirs]
        print(f"{label}: mean diff {diff.mean():+.4f}, "
              f"mean |diff| {diff.abs().mean():.4f}, "
              f"max |diff| {diff.abs().max():.4f} ({diff.abs().idxmax()})")

def fig9_validation_scatter(table, figures):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.6))
    panels = [
        (ax1, 'off_base8', 'ours_llama', BLUE,
         "A    base 8B: pipeline validation",
         "official unsloth-8B-bnb-4bit eval loss (nat)",
         "our Llama-3.1-8B runtime NF4 (nat)"),
        (ax2, 'off_cent8', 'ours_minitaur', ORANGE,
         "B    finetuned 8B: Minitaur is not Centaur-8B",
         "official Centaur-8B-adapter eval loss (nat)",
         "our Minitaur-8B merged, runtime NF4 (nat)"),
    ]
    for ax, x_col, y_col, color, title, xlabel, ylabel in panels:
        limit = 1.32
        ax.plot([0, limit], [0, limit], ls=(0, (4, 3)), color=MUTED, lw=1.2)
        ax.scatter(table[x_col], table[y_col], s=30, color=color,
                   alpha=0.75, edgecolor='none')
        diff = table[y_col] - table[x_col]
        ax.text(0.03, 0.97,
                f"mean diff {diff.mean():+.4f}\n"
                f"max |diff| {diff.abs().max():.3f}\n"
                f"r = {table[x_col].corr(table[y_col]):.5f}",
                transform=ax.transAxes, fontsize=8.5, va='top', color=INK)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect('equal')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title)
        style_axis(ax)
    worst = (table['ours_minitaur'] - table['off_cent8']).idxmax()
    ax2.annotate(worst, xy=(table.loc[worst, 'off_cent8'],
                            table.loc[worst, 'ours_minitaur']),
                 xytext=(0.45, 1.18), fontsize=8, color=SECONDARY,
                 arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    fig.suptitle("Official published results vs. ours (36 seen families, "
                 "paper protocol)", y=0.99, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures / "fig9_official_vs_ours_scatter.png", dpi=200)
    plt.close(fig)

def fig10_official_cast(table, figures):
    order = table.sort_values('off_cent70', ascending=False).index
    fig, ax = plt.subplots(figsize=(8.0, 9.0))
    dot_rows(ax, order, [
        (table['off_cog'], dict(
            s=34, color=INK, marker='x', lw=1.2,
            label="cognitive models (official)")),
        (table['ours_llama'], dict(s=26, color=BLUE, edgecolor='none',
                                   label="Llama-8B base (ours, NF4)")),
        (table['off_base8'], dict(s=40, facecolor='none', edgecolor=BLUE,
                                  lw=1.1,
                                  label="Llama-8B base (official 4bit)")),
        (table['ours_minitaur'], dict(s=26, color=ORANGE, edgecolor='none',
                                      label="Minitaur-8B (ours, NF4)")),
        (table['off_cent8'], dict(s=40, facecolor='none', edgecolor=ORANGE,
                                  lw=1.1, label="Centaur-8B (official)")),
        (table['off_cent70'], dict(s=30, color=VIOLET, marker='D',
                                   edgecolor='none',
                                   label="Centaur-70B (official)")),
    ])
    ax.set_xlim(-0.02, 1.35)
    ax.set_xlabel("paper-protocol eval loss (nat)")
    ax.set_title("Per-task: official cast vs. our runs\n"
                 "(open/filled same hue = same class; blue pair coincides, "
                 "orange pair does not)", fontsize=10.5)
    ax.legend(frameon=False, loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "fig10_per_task_official_cast.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
