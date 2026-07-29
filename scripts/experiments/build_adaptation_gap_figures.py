"""Does Centaur's edge over cognitive models come from in-session adaptation?

The paper fits one parameter set across all training participants, so its
cognitive baselines cannot adapt to an individual within a session; Centaur
gets that for free in context.  This script uses the official per-choice
log-likelihood arrays (results/custom_metrics_full_log_likelihoods_*.pth),
positions each choice with our own aligned E0 cache, and plots the gap as a
function of within-session position.  A gap that widens with position is
adaptation the baselines were structurally forbidden from doing; a gap
already present at trial 0 is transferable behavioral knowledge.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from build_per_task_figures import VIOLET
from build_report_figures import (
    BLUE,
    INK,
    MUTED,
    ORANGE,
    REPO,
    SECONDARY,
    STYLE,
    load_scores,
    replace_sessions,
    style_axis,
)

OFFICIAL_DEFAULT = Path(
    "~/wkspace/Llama-3.1-Centaur-70B/results").expanduser()
SOURCES = {
    'cog': "baselines",
    'base70': "unsloth-Meta-Llama-3.1-70B-bnb-4bit",
    'cent70': "marcelbinz-Llama-3.1-Centaur-70B-adapter",
}
# supplementary models 13/14: oracle-fed rational model and lookup table
STATISTICAL = {"frey2017risk", "enkavi2019adaptivenback",
               "enkavi2019digitspan", "enkavi2019gonogo",
               "enkavi2019recentprobes", "wu2023chunking",
               "jansen2021dunningkruger"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", default=OFFICIAL_DEFAULT,
                        help="Official results/ directory (repo clone)")
    parser.add_argument("--scoring", default=REPO / "outputs/scoring",
                        help="Directory holding our score CSVs")
    parser.add_argument("--figures", default=REPO / "outputs/figures",
                        help="Output directory for the PNG figures")
    args = parser.parse_args()
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    frame = build_positioned_frame(Path(args.official), Path(args.scoring))
    families = sorted(frame['family'].unique())
    print(f"aligned families: {len(families)}")

    curves = position_curves(frame, 'decile')
    early = position_curves(frame[frame['choice_index'] <= 9],
                            'choice_index')
    fig_adaptation_gap(curves, early, figures)

    split = {name: position_curves(sub, 'decile')
             for name, sub in [
                 ("真认知模型", frame[~frame['family'].isin(STATISTICAL)]),
                 ("统计基线", frame[frame['family'].isin(STATISTICAL)])]}
    report(curves, early, split)
    out = REPO / "outputs/analysis_adaptation_gap.csv"
    curves.round(4).to_csv(out)
    print(f"\nwrote figure to {figures} and curve to {out}")

def build_positioned_frame(official, scoring):
    """Join official per-choice arrays onto our positioned E0 cache."""

    arrays = {key: torch.load(
        official / f"custom_metrics_full_log_likelihoods_{name}.pth",
        map_location="cpu", weights_only=False)
        for key, name in SOURCES.items()}

    ours = replace_sessions(
        load_scores(scoring / "minitaur8b_e0_full_4bit.csv"),
        load_scores(scoring / "minitaur8b_e0_zorowitz_utf8_4bit.csv"))
    ours['family'] = ours['experiment'].str.split('/').str[0]

    parts = []
    for family, group in ours.groupby('family', sort=True):
        lengths = {key: len(data[family]) for key, data in arrays.items()
                   if family in data}
        # position labels are only valid when every source has the same
        # per-choice ordering and length as our cache
        if len(lengths) < len(arrays) or set(lengths.values()) != {len(group)}:
            continue
        block = group[['family', 'participant', 'choice_index']].copy()
        for key, data in arrays.items():
            block[key] = list(map(float, data[family]))
        parts.append(block)
    frame = pd.concat(parts, ignore_index=True)
    n = frame.groupby(['family', 'participant'])['choice_index'].transform(
        'max') + 1
    frame['decile'] = (frame['choice_index'] * 10 // n).clip(upper=9)
    return frame

def position_curves(frame, bucket):
    """Hierarchical means per position: choice -> participant -> task."""

    columns = list(SOURCES)
    per_participant = frame.groupby(
        ['family', 'participant', bucket])[columns].mean()
    per_family = per_participant.groupby(['family', bucket]).mean()
    curves = per_family.groupby(bucket).mean()
    curves['gap_finetuned'] = curves['cog'] - curves['cent70']
    curves['gap_pretrained'] = curves['cog'] - curves['base70']
    return curves

def report(curves, early, split):
    print("\n=== within-session decile (task-macro NLL) ===")
    print(curves.round(4).to_string())
    print("\n=== absolute early trials ===")
    print(early.round(4).to_string())
    first, last = curves.iloc[0], curves.iloc[-1]
    print(f"\ngap vs cognitive models: decile 0 = "
          f"{first['gap_finetuned']:.4f} -> decile 9 = "
          f"{last['gap_finetuned']:.4f} "
          f"(change {last['gap_finetuned'] - first['gap_finetuned']:+.4f})")
    print(f"same for untuned base-70B: {first['gap_pretrained']:.4f} -> "
          f"{last['gap_pretrained']:.4f} "
          f"(change {last['gap_pretrained'] - first['gap_pretrained']:+.4f})")
    for name, sub in split.items():
        change = (sub['gap_finetuned'].iloc[-1]
                  - sub['gap_finetuned'].iloc[0])
        print(f"  [{name}] gap decile 0 {sub['gap_finetuned'].iloc[0]:.4f} "
              f"-> 9 {sub['gap_finetuned'].iloc[-1]:.4f} ({change:+.4f})")

def fig_adaptation_gap(curves, early, figures):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2))
    x = list(curves.index)
    for column, color, label in [
            ('cog', INK, "cognitive models (one shared parameter set)"),
            ('base70', BLUE, "Llama-3.1-70B base (untuned)"),
            ('cent70', ORANGE, "Centaur-70B")]:
        ax1.plot(x, curves[column], '-o', color=color, lw=2, ms=4,
                 label=label)
    ax1.set_xlabel("within-session position (decile)")
    ax1.set_ylabel("task-macro NLL (nat)")
    ax1.set_title("A    all three models over the session")
    ax1.legend(frameon=False, fontsize=8, loc='upper right')
    style_axis(ax1)

    ax2.plot(x, curves['gap_finetuned'], '-o', color=ORANGE, lw=2, ms=4,
             label="Centaur-70B − cognitive")
    ax2.plot(x, curves['gap_pretrained'], '-o', color=BLUE, lw=2, ms=4,
             label="base-70B − cognitive (no Psych-101)")
    ax2.axhline(0, color=MUTED, lw=1)
    ax2.set_xlabel("within-session position (decile)")
    ax2.set_ylabel("advantage over cognitive models (nat)")
    ax2.set_title("B    the advantage grows with accumulated history")
    ax2.legend(frameon=False, fontsize=8, loc='upper left')
    style_axis(ax2)
    change = (curves['gap_finetuned'].iloc[-1]
              - curves['gap_finetuned'].iloc[0])
    ax2.annotate(f"decile 0 → 9: {change:+.3f} nat",
                 xy=(4.5, curves['gap_finetuned'].iloc[4]), fontsize=8.5,
                 color=SECONDARY,
                 xytext=(2.2, curves['gap_finetuned'].max() * 0.55),
                 arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    fig.suptitle("Cognitive baselines are fitted with one parameter set for "
                 "all participants — they cannot adapt within a session\n"
                 f"(official published per-choice likelihoods, "
                 f"{len(curves)} deciles, aligned tasks only)",
                 fontsize=10.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures / "fig15_adaptation_gap_vs_cognitive.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
