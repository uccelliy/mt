"""Render canonical (un-randomized) choice distributions from HF tables.

The per-task datasets under huggingface.co/marcelbinz/<family> hold the
original tabular data with a standardized `choice` column in canonical
option coding — unlike the transcripts, where options are mapped to
per-participant randomized key letters.  This script downloads those
tables (cached under data/psych101_tables/), and renders fig14: one
canonical choice-distribution panel per covered experiment.  fig12 (the
transcript-label view) is kept alongside; the two figures answer
different questions.  Tables contain all participants (train + test).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from build_per_task_figures import short_name
from build_report_figures import (
    BLUE,
    GRID,
    REPO,
    SECONDARY,
    STYLE,
)

from mt.models.baselines.canonical_tables import (
    COLUMN_OVERRIDES,
    FAMILIES,
    FAMILY_RENAMES,
    download_tables,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default=REPO / "data/psych101_tables",
                        help="Download cache for the parquet tables")
    parser.add_argument("--figures", default=REPO / "outputs/figures",
                        help="Output directory for the PNG figure")
    args = parser.parse_args()
    tables = Path(args.tables)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    download_tables(tables)
    distributions = collect_distributions(tables)
    fig14_canonical_distributions(distributions, figures)

    rows = [{'experiment': experiment, 'choice': str(value),
             'count': count, 'share': count / counts.sum()}
            for experiment, counts in sorted(distributions.items())
            for value, count in counts.items()]
    out = REPO / "outputs/analysis_canonical_choice_distributions.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote fig14 to {figures} and distributions to {out}")


def collect_distributions(tables):
    """Return {experiment id -> choice value_counts} from the tables."""

    distributions = {}
    for family in FAMILIES:
        name = FAMILY_RENAMES.get(family, family)
        columns = COLUMN_OVERRIDES.get(family, ["choice"])
        for config_dir in sorted((tables / family).iterdir()):
            if not config_dir.is_dir():
                continue
            frame = pd.concat([pd.read_parquet(p, columns=columns)
                               for p in sorted(config_dir.rglob(
                                   "*.parquet"))])
            values = pd.concat([frame[c].dropna() for c in columns])
            experiment = f"{name}/{config_dir.name}.csv"
            distributions[experiment] = values.value_counts()
    return distributions

def fig14_canonical_distributions(distributions, figures, top=8,
                                  columns=8):
    tasks = sorted(distributions)
    rows = -(-len(tasks) // columns)
    fig, axes = plt.subplots(rows, columns,
                             figsize=(1.72 * columns, 1.62 * rows))
    for ax in axes.flat[len(tasks):]:
        ax.set_visible(False)
    for ax, task in zip(axes.flat, tasks):
        counts = distributions[task]
        shares = counts.iloc[:top] / counts.sum()
        ax.bar(range(len(shares)), shares.values, color=BLUE,
               edgecolor='none')
        labels = [f"{v:g}" if isinstance(v, float) else str(v)[:6]
                  for v in shares.index]
        ax.set_xticks(range(len(shares)), labels, fontsize=4.4,
                      rotation=45)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_title(short_name(task), fontsize=5.8, pad=2)
        ax.text(0.97, 0.92, f"{len(counts)} options",
                transform=ax.transAxes, fontsize=4.6, ha='right',
                color=SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    fig.suptitle("Canonical choice distribution per task (original HF "
                 f"tables, un-randomized coding, top {top} options, all "
                 "participants; held-out generalization tasks have no "
                 "published tables)", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(figures / "fig14_canonical_choice_distributions.png",
                dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
