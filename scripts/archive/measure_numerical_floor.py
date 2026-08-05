"""How small an effect may the benchmark assert? Measure the floor directly.

Every score in this project is a low-precision forward pass, so two runs of
the same model on the same data do not agree exactly: fp16 accumulation order
depends on how targets happen to be packed into batches.  L1 already measured
the cross-hardware floor at the per-experiment level (6.046e-4 nat, Server
handoff 1.6), but the benchmark reports a three-level macro over experiments,
and that is the level the "no effect below 5x the floor" rule has to police.

E0 and E3/full are exactly such a pair: same model, same hardware, same
targets, different packing (E0 packs whole sessions, E3 packs window cells).
Their difference on the shared keys is therefore a floor estimate at the
reporting level, with no extra GPU time.  It bounds packing noise only --
the cross-hardware floor at this level still has to be measured on the
cluster once the first job lands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_report_figures import KEYS, REPO, load_scores, replace_sessions, task_macro

PAIRS = {
    'minitaur8b': ("minitaur8b_e0_full_4bit.csv",
                   "minitaur8b_e3_e0grid5_4bit.csv"),
    'llama31_8b_base': ("llama31_8b_base_e0_full_4bit.csv",
                        "llama31_8b_base_e3_e0grid5_4bit.csv"),
}
ZOROWITZ = "minitaur8b_e0_zorowitz_utf8_4bit.csv"
CHOICE_KEYS = KEYS + ['choice_index']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring", default=REPO / "outputs/scoring",
                        help="Directory holding our score CSVs")
    parser.add_argument("--output", default=REPO / "outputs/analysis_numerical_floor.csv",
                        help="Where to write the per-model floor table")
    args = parser.parse_args()
    scoring = Path(args.scoring)

    records = []
    for model, (e0_name, e3_name) in PAIRS.items():
        e0 = load_scores(scoring / e0_name)
        if model == 'minitaur8b':
            e0 = replace_sessions(e0, load_scores(scoring / ZOROWITZ))
        e3 = load_scores(scoring / e3_name)
        records.append(compare(model, e0, e3))

    table = pd.DataFrame(records)
    table.to_csv(args.output, index=False)
    report(table)

def compare(model, e0, e3):
    """Align E0 against the E3 full-window cells and score both the same way."""

    full = e3[e3['window'].astype(str) == 'full']
    # E3 samples five anchors per session, so E0 is restricted, not the reverse
    shared = full[CHOICE_KEYS].drop_duplicates()
    left = shared.merge(e0, on=CHOICE_KEYS, how='inner', validate='one_to_one')
    right = shared.merge(full[CHOICE_KEYS + ['nll', 'num_tokens']],
                         on=CHOICE_KEYS, how='inner', validate='one_to_one')
    if len(left) != len(shared) or len(right) != len(shared):
        raise RuntimeError(
            f"{model}: expected {len(shared)} shared choices, matched "
            f"{len(left)} in E0 and {len(right)} in E3")
    if not left['num_tokens'].equals(right['num_tokens']):
        raise RuntimeError(f"{model}: token counts disagree between E0 and E3")

    delta = (left['nll'] - right['nll']).abs()
    macro_e0 = task_macro(left)
    macro_e3 = task_macro(right)
    per_experiment = (macro_e0 - macro_e3).abs()
    return {
        'model': model,
        'choices': len(shared),
        'experiments': len(macro_e0),
        'choice_mae': delta.mean(),
        'choice_max': delta.max(),
        'experiment_mae': per_experiment.mean(),
        'experiment_p95': per_experiment.quantile(0.95),
        'experiment_max': per_experiment.max(),
        'report_macro_e0': macro_e0.mean(),
        'report_macro_e3': macro_e3.mean(),
        'report_floor': abs(macro_e0.mean() - macro_e3.mean()),
    }

def report(table):
    pd.set_option('display.width', 120)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print()
    report_floor = table['report_floor'].max()
    task_floor = table['experiment_p95'].max()
    # L1's cross-hardware per-experiment floor vs the packing floor measured
    # here; the same ratio is the best available guess for how much worse the
    # cluster-vs-cluster case gets, and it is a guess, not a measurement
    ratio = 6.046e-4 / table['experiment_mae'].max()
    print(f"packing floor, report level        {report_floor:.3e} nat")
    print(f"packing floor, per task (p95)      {task_floor:.3e} nat")
    print(f"cross-hardware is ~{ratio:.0f}x worse per experiment (L1: 6.046e-4)")
    print()
    # the effects the benchmark is asked to resolve, from the handoff
    known = {
        'context w=0 -> full (E3)': 1.4405,
        'behavioral finetuning (official 8B)': 0.1305,
        'Minitaur deployment damage': 0.1170,
        'scale 8B -> 70B (official)': 0.0109,
    }
    print(f"{'effect':38s} {'nat':>8s} {'vs report':>10s} {'vs task':>9s}"
          f"  per-task verdict")
    for name, value in known.items():
        task = value / (5 * task_floor * ratio)
        verdict = "assertable" if task > 1 else "BELOW 5x FLOOR"
        print(f"  {name:36s} {value:8.4f} {value / (5 * report_floor):10.0f}x"
              f" {task:8.1f}x  {verdict}")

if __name__ == "__main__":
    main()
