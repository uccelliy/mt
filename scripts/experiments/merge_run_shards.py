"""Merge a Track P run's three shard tables with integrity checks.

The merge is streaming: only the set of parent choice keys is retained in
memory, so the top-k table can contain tens of millions of rows without a
multi-gigabyte pandas concatenation. Each completed table is staged beside
its destination and published with ``os.replace`` only after all three
tables pass validation.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import tempfile

TABLES = ("predictions", "pred_topk", "pred_options")
PARENT_KEY = ("model", "dataset", "condition", "experiment",
              "participant", "choice_index")
CHILD_SUFFIX = {
    "pred_topk": ("token_index", "rank"),
    "pred_options": ("option",),
}


class MergeError(RuntimeError):
    """Shard tables are incomplete or violate the output contract."""


def _shard_paths(run_dir, table, num_shards):
    pattern = re.compile(rf"{re.escape(table)}_shard(\d+)\.csv")
    found = {}
    for path in run_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            found[int(match.group(1))] = path
    expected = set(range(num_shards))
    actual = set(found)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing shards {missing}")
        if unexpected:
            details.append(f"unexpected shards {unexpected}")
        raise MergeError(f"{run_dir}/{table}: " + ", ".join(details))
    return [found[index] for index in range(num_shards)]


def _nonempty_failure_logs(prediction_shards):
    nonempty = []
    for path in prediction_shards:
        failure = path.with_name(path.stem + ".failed.csv")
        if not failure.exists() or failure.stat().st_size == 0:
            continue
        with failure.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            if next(reader, None) is not None:
                nonempty.append(failure)
    return nonempty


def _staged_writer(run_dir, table):
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", delete=False,
        dir=run_dir, prefix=f".{table}.", suffix=".tmp")
    return handle, Path(handle.name)


def _reader(path, expected_header=None):
    handle = path.open(encoding="utf-8", newline="")
    reader = csv.DictReader(handle)
    header = reader.fieldnames
    if not header:
        handle.close()
        raise MergeError(f"{path}: missing CSV header")
    if expected_header is not None and header != expected_header:
        handle.close()
        raise MergeError(f"{path}: header differs from the other shards")
    return handle, reader, header


def _require_columns(path, header, columns):
    missing = [column for column in columns if column not in header]
    if missing:
        raise MergeError(f"{path}: missing columns {missing}")


def _merge_predictions(run_dir, shards):
    output, output_path = _staged_writer(run_dir, "predictions")
    header = None
    writer = None
    parents = set()
    rows = 0
    try:
        for path in shards:
            handle, reader, current_header = _reader(path, header)
            with handle:
                if header is None:
                    header = current_header
                    _require_columns(path, header, PARENT_KEY)
                    writer = csv.DictWriter(output, fieldnames=header)
                    writer.writeheader()
                for row in reader:
                    key = tuple(row[column] for column in PARENT_KEY)
                    if key in parents:
                        raise MergeError(
                            f"{path}: duplicate prediction key {key}")
                    parents.add(key)
                    writer.writerow(row)
                    rows += 1
        if not rows:
            raise MergeError(f"{run_dir}: predictions contain no rows")
    except Exception:
        output.close()
        output_path.unlink(missing_ok=True)
        raise
    output.close()
    return output_path, parents, {"rows": rows}


def _merge_child(run_dir, table, shards, parents):
    output, output_path = _staged_writer(run_dir, table)
    header = None
    writer = None
    seen_groups = set()
    rows = 0
    orphan_rows = 0
    duplicate_rows = 0
    suffix_columns = CHILD_SUFFIX[table]
    try:
        for path in shards:
            handle, reader, current_header = _reader(path, header)
            with handle:
                if header is None:
                    header = current_header
                    _require_columns(path, header,
                                     PARENT_KEY + suffix_columns)
                    writer = csv.DictWriter(output, fieldnames=header)
                    writer.writeheader()
                active_parent = None
                active_suffixes = set()
                duplicate_group = False
                for row in reader:
                    parent = tuple(row[column] for column in PARENT_KEY)
                    suffix = tuple(row[column] for column in suffix_columns)
                    if parent != active_parent:
                        active_parent = parent
                        active_suffixes = set()
                        duplicate_group = parent in seen_groups
                        if not duplicate_group:
                            seen_groups.add(parent)
                    elif not duplicate_group and suffix in active_suffixes:
                        duplicate_group = True
                    if duplicate_group:
                        duplicate_rows += 1
                        continue
                    active_suffixes.add(suffix)
                    if parent not in parents:
                        orphan_rows += 1
                        continue
                    writer.writerow(row)
                    rows += 1
    except Exception:
        output.close()
        output_path.unlink(missing_ok=True)
        raise
    output.close()
    return output_path, {
        "rows": rows,
        "orphan_rows_dropped": orphan_rows,
        "duplicate_rows_dropped": duplicate_rows,
    }


def merge_run(run_dir, num_shards):
    """Merge one run directory and return per-table row statistics."""

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise MergeError(f"run directory does not exist: {run_dir}")
    shards = {table: _shard_paths(run_dir, table, num_shards)
              for table in TABLES}
    failures = _nonempty_failure_logs(shards["predictions"])
    if failures:
        names = ", ".join(str(path) for path in failures)
        raise MergeError(f"failed sessions must be resolved before merge: "
                         f"{names}")

    staged = {}
    stats = {}
    try:
        staged["predictions"], parents, stats["predictions"] = (
            _merge_predictions(run_dir, shards["predictions"]))
        for table in TABLES[1:]:
            staged[table], stats[table] = _merge_child(
                run_dir, table, shards[table], parents)
        # predictions is the authoritative resume ledger, so publish it last.
        for table in TABLES[1:] + ("predictions",):
            os.replace(staged[table], run_dir / f"{table}.csv")
    except Exception:
        for path in staged.values():
            path.unlink(missing_ok=True)
        raise
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", nargs="+", required=True,
                        help="One or more outputs/runs/<tag> directories")
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    for run_dir in args.run_dir:
        try:
            stats = merge_run(run_dir, args.num_shards)
        except MergeError as error:
            parser.exit(1, f"merge failed: {error}\n")
        summary = ", ".join(
            f"{table}={values['rows']}" for table, values in stats.items())
        print(f"{run_dir}: {summary}", flush=True)
        for table in TABLES[1:]:
            values = stats[table]
            if values["orphan_rows_dropped"] or values["duplicate_rows_dropped"]:
                print(
                    f"  {table}: dropped "
                    f"{values['orphan_rows_dropped']} orphan and "
                    f"{values['duplicate_rows_dropped']} duplicate rows",
                    flush=True,
                )


if __name__ == "__main__":
    main()
