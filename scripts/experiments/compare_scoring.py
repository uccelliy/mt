"""Compare two score CSVs and judge them against a tolerance.

Used to validate a new environment (HPC) against a known-good local run.
Alignment keys adapt to the runner: E0 output keys on experiment /
participant / choice_index, E3 adds window and target_index.
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
                        help="Max allowed |delta| in per-choice NLL")
    parser.add_argument("--min-correlation", type=float, default=0.9999,
                        help="Min allowed Pearson r over matched rows")
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

    failures = judge(report, args.tolerance, args.min_correlation,
                     args.strict)
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

    delta = (merged['nll_cand'] - merged['nll_ref']).abs()
    report['max_delta'] = float(delta.max())
    report['mean_delta'] = float(delta.mean())
    report['worst'] = merged.loc[delta.idxmax(), list(keys)].to_dict()
    # a single-valued series has zero variance and no defined correlation
    if merged['nll_ref'].std() == 0 or merged['nll_cand'].std() == 0:
        report['correlation'] = None
    else:
        report['correlation'] = float(
            merged['nll_ref'].corr(merged['nll_cand']))
    return report

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
    if report['matched_rows']:
        print(f"max |delta|:  {report['max_delta']:.3e}")
        print(f"mean |delta|: {report['mean_delta']:.3e}")
        print(f"worst row:    {report['worst']}")
        correlation = report['correlation']
        shown = ("undefined (constant series)" if correlation is None
                 else f"{correlation:.8f}")
        print(f"pearson r:    {shown}")

def judge(report, tolerance, min_correlation, strict):
    failures = []
    if not report['matched_rows']:
        failures.append("no rows matched; check the alignment keys and "
                        "whether the two runs cover the same sessions")
        return failures
    if report['max_delta'] > tolerance:
        failures.append(f"max |delta| {report['max_delta']:.3e} > tolerance "
                        f"{tolerance:.3e} at {report['worst']}")
    correlation = report['correlation']
    if correlation is not None and correlation < min_correlation:
        failures.append(f"pearson r {correlation:.8f} < "
                        f"{min_correlation:.8f}")
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
