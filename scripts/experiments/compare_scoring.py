"""Compare two score CSVs and judge them against a tolerance.

Used to validate a new environment (HPC) against a known-good local run.
Alignment keys adapt to the runner: E0 output keys on experiment /
participant / choice_index, E3 adds window and target_index.

The pass/fail criterion is per-experiment mean NLL, not per-row NLL. Two
GPUs running the same weights disagree per row by whatever their fp16
attention and matmul kernels accumulate differently -- on Volta, which has
neither FlashAttention nor cuDNN attention, that is around 0.02 nat per
choice against a Blackwell reference. Those differences are zero-mean and
cancel: the aggregates every downstream analysis uses agree three orders
of magnitude better. Row-level statistics are still reported and bounded,
because a genuinely broken pipeline (wrong tokenizer, wrong model, mangled
alignment) blows through them by an order of magnitude.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

KEY_COLUMNS = ("experiment", "participant", "window", "target_index",
               "choice_index")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True,
                        help="Baseline score CSV (the known-good run)")
    parser.add_argument("--candidate", required=True,
                        help="Score CSV to check against the baseline")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Max allowed |delta| in per-EXPERIMENT mean "
                             "NLL; this is the pass/fail criterion")
    parser.add_argument("--row-tolerance", type=float, default=0.25,
                        help="Max allowed |delta| in any single row; a "
                             "loose guard against gross breakage, not a "
                             "precision check")
    parser.add_argument("--min-correlation", type=float, default=0.99,
                        help="Min allowed Pearson r over matched rows; "
                             "catches misalignment, which collapses r")
    parser.add_argument("--strict", action="store_true",
                        help="Also require both files to cover the same "
                             "sessions; off by default so a one-experiment "
                             "candidate can be checked against a full "
                             "reference")
    args = parser.parse_args()

    reference = load_scores(args.reference)
    candidate = load_scores(args.candidate)
    keys = align_keys(reference, candidate)
    print(f"aligning on {', '.join(keys)}")

    merged = reference.merge(candidate, on=list(keys), how="inner",
                             suffixes=("_ref", "_cand"))
    report = summarize(reference, candidate, merged, keys)
    print_report(report)

    failures = judge(report, args.tolerance, args.row_tolerance,
                     args.min_correlation, args.strict)
    if failures:
        print(f"\nCOMPARE FAILED ({len(failures)} problems):")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\nCOMPARE PASSED")

def load_scores(path):
    """Read a score CSV with key columns forced to str for safe joins."""

    if not Path(path).exists():
        raise SystemExit(f"score CSV missing: {path}")
    # participant ids mix ints and strings across tasks; low_memory=False
    # keeps pandas from inferring per-chunk and warning
    frame = pd.read_csv(path, low_memory=False)
    if 'nll' not in frame.columns:
        raise SystemExit(f"{path} has no 'nll' column; not a score CSV")
    for name in KEY_COLUMNS:
        if name in frame.columns:
            frame[name] = frame[name].astype(str)
    return frame

def align_keys(reference, candidate):
    """Key columns present on both sides, in canonical order."""

    keys = tuple(name
                 for name in KEY_COLUMNS
                 if name in reference.columns and name in candidate.columns)
    if 'experiment' not in keys or 'participant' not in keys:
        raise SystemExit("both CSVs need experiment and participant columns")
    return keys

def sessions(frame):
    return set(map(tuple, frame[['experiment', 'participant']]
                   .drop_duplicates().itertuples(index=False)))

def summarize(reference, candidate, merged, keys):
    reference_sessions = sessions(reference)
    candidate_sessions = sessions(candidate)
    report = {
        'keys': keys,
        'reference_rows': len(reference),
        'candidate_rows': len(candidate),
        'matched_rows': len(merged),
        'only_reference': sorted(reference_sessions - candidate_sessions),
        'only_candidate': sorted(candidate_sessions - reference_sessions),
        'shared_sessions': len(reference_sessions & candidate_sessions),
    }
    if merged.empty:
        return report

    signed = merged['nll_cand'] - merged['nll_ref']
    delta = signed.abs()
    report['max_delta'] = float(delta.max())
    report['mean_delta'] = float(delta.mean())
    # a bias survives aggregation, zero-mean kernel noise does not; the two
    # need different responses, so report the signed mean separately
    report['signed_mean'] = float(signed.mean())
    report['worst'] = merged.loc[delta.idxmax(), list(keys)].to_dict()
    # a single-valued series has zero variance and no defined correlation
    if merged['nll_ref'].std() == 0 or merged['nll_cand'].std() == 0:
        report['correlation'] = None
    else:
        report['correlation'] = float(
            merged['nll_ref'].corr(merged['nll_cand']))
    report['experiment_delta'] = grouped_delta(merged, ['experiment'])
    report['session_delta'] = grouped_delta(merged,
                                            ['experiment', 'participant'])
    # identical token counts prove both runs tokenized the transcripts the
    # same way, which separates a numerical difference from a real one
    if 'num_tokens_ref' in merged.columns:
        mismatched = merged['num_tokens_ref'] != merged['num_tokens_cand']
        report['token_mismatches'] = int(mismatched.sum())
    return report

def grouped_delta(merged, by):
    """Largest |delta| between group means, and which group it is on."""

    means = merged.groupby(by)[['nll_ref', 'nll_cand']].mean()
    delta = (means['nll_cand'] - means['nll_ref']).abs()
    return {'max': float(delta.max()), 'mean': float(delta.mean()),
            'worst': str(delta.idxmax()), 'groups': len(delta)}

def print_report(report):
    print(f"reference rows: {report['reference_rows']}")
    print(f"candidate rows: {report['candidate_rows']}")
    print(f"matched rows:   {report['matched_rows']}")
    print(f"shared sessions: {report['shared_sessions']}, "
          f"only-reference {len(report['only_reference'])}, "
          f"only-candidate {len(report['only_candidate'])}")
    for label in ['only_reference', 'only_candidate']:
        for key in report[label][:5]:
            print(f"  {label}: {key[0]} p{key[1]}")
        if len(report[label]) > 5:
            print(f"  {label}: ... and {len(report[label]) - 5} more")
    if not report['matched_rows']:
        return
    if 'token_mismatches' in report:
        print(f"num_tokens mismatches: {report['token_mismatches']} "
              f"(non-zero means the two runs tokenized differently)")
    for label, key in [("per-experiment", 'experiment_delta'),
                       ("per-session   ", 'session_delta')]:
        stats = report[key]
        print(f"{label} mean-NLL |delta|: max {stats['max']:.3e}, "
              f"mean {stats['mean']:.3e}, over {stats['groups']} groups, "
              f"worst {stats['worst']}")
    print(f"per-row |delta|: max {report['max_delta']:.3e}, "
          f"mean {report['mean_delta']:.3e}, at {report['worst']}")
    print(f"per-row signed mean: {report['signed_mean']:+.3e} "
          f"(near zero = kernel noise; a bias would show here)")
    correlation = report['correlation']
    shown = ("undefined (constant series)" if correlation is None
             else f"{correlation:.8f}")
    print(f"pearson r:    {shown}")

def judge(report, tolerance, row_tolerance, min_correlation, strict):
    failures = []
    if not report['matched_rows']:
        failures.append("no rows matched; check the alignment keys and "
                        "whether the two runs cover the same sessions")
        return failures
    if report.get('token_mismatches'):
        failures.append(f"{report['token_mismatches']} rows disagree on "
                        f"num_tokens; the two runs tokenized differently, "
                        f"so the NLLs are not comparable at all")
    experiment = report['experiment_delta']
    if experiment['max'] > tolerance:
        failures.append(f"per-experiment mean NLL differs by "
                        f"{experiment['max']:.3e} > {tolerance:.3e} on "
                        f"{experiment['worst']}; this is the criterion that "
                        f"matters, a wrong model or checkpoint lands here")
    if report['max_delta'] > row_tolerance:
        failures.append(f"a single row differs by {report['max_delta']:.3e} "
                        f"> {row_tolerance:.3e} at {report['worst']}; too "
                        f"large for fp16 kernel differences")
    correlation = report['correlation']
    if correlation is not None and correlation < min_correlation:
        failures.append(f"pearson r {correlation:.8f} < "
                        f"{min_correlation:.8f}; rows may be misaligned")
    if strict:
        if report['only_reference']:
            failures.append(f"{len(report['only_reference'])} sessions only "
                            f"in the reference")
        if report['only_candidate']:
            failures.append(f"{len(report['only_candidate'])} sessions only "
                            f"in the candidate")
    return failures

if __name__ == "__main__":
    main()
