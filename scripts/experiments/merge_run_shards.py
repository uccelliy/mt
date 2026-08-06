"""Merge a Track P run's three shard tables with integrity checks.

The merge is streaming: only compact parent expectations and completion keys
are retained in memory, so the top-k table can contain tens of millions of
rows without a multi-gigabyte pandas concatenation. Each completed table is
staged beside its destination and published with ``os.replace`` only after
all three tables pass validation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from mt.evaluation.context_windows import grid_targets
from mt.evaluation.transcript_scoring import (
    OPTIONS_MULTI_TOKEN,
    OPTIONS_OFF,
    OPTIONS_SCORED,
    OPTIONS_TOKENIZATION_MISMATCH,
    OPTIONS_TOO_MANY,
)

TABLES = ("predictions", "pred_topk", "pred_options")
PARENT_KEY = ("model", "dataset", "condition", "experiment", "participant", "choice_index")
CHILD_SUFFIX = {
    "pred_topk": ("token_index", "rank"),
    "pred_options": ("option",),
}
PREDICTION_COUNTS = ("num_tokens", "k_options", "options_status")
PREDICTION_PAYLOAD = (
    "pred_choice",
    "pred_token_id",
    "human_choice",
    "raw_generation",
    "format_ok",
)
CHILD_PAYLOAD = {
    "pred_topk": ("token_id", "token", "logprob"),
    "pred_options": ("logprob", "n_tokens", "is_human", "is_correct"),
}
WINDOW_METADATA = (
    "window",
    "target_index",
    "effective_window",
    "position_grid",
    "n_segments",
    "target_fraction",
    "is_anchor",
)
SESSION_COMMIT_SCHEMA = 1
OPTIONS_STATUSES = frozenset(
    {
        OPTIONS_SCORED,
        OPTIONS_OFF,
        OPTIONS_TOO_MANY,
        OPTIONS_MULTI_TOKEN,
        OPTIONS_TOKENIZATION_MISMATCH,
    }
)


@dataclass(frozen=True, slots=True)
class E3Contract:
    """Protocol values needed to validate every E3 session structurally."""

    windows: tuple[int, ...]
    num_positions: int
    position_grid: str
    condition_prefix: str


@dataclass(frozen=True, slots=True)
class RunContract:
    top_k: int
    e3: E3Contract | None


@dataclass(frozen=True, slots=True)
class ParentExpectation:
    num_tokens: int
    expected_options: int
    shard: int
    row_digest: bytes
    window_metadata: tuple[str, ...] | None
    human_choice: str
    options_status: str


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
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=run_dir,
        prefix=f".{table}.",
        suffix=".tmp",
    )
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


def _nonnegative_int(value, *, path, column, key=None, positive=False):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        location = f" for prediction {key}" if key is not None else ""
        raise MergeError(f"{path}: {column} must be an integer{location}; got {value!r}") from error
    minimum = 1 if positive else 0
    if parsed < minimum:
        qualifier = "positive" if positive else "non-negative"
        location = f" for prediction {key}" if key is not None else ""
        raise MergeError(f"{path}: {column} must be {qualifier}{location}; got {parsed}")
    return parsed


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MergeError(f"{path}: invalid JSON: {error}") from error


def _normalize_e3_contract(path, payload):
    raw_windows = payload.get("windows")
    if not isinstance(raw_windows, list):
        raise MergeError(f"{path}: E3 metadata requires a windows list")
    windows = []
    saw_full = False
    for raw in raw_windows:
        if raw == "full":
            if saw_full:
                raise MergeError(f"{path}: duplicate full window")
            saw_full = True
            continue
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise MergeError(f"{path}: invalid finite E3 window {raw!r}")
        if raw in windows:
            raise MergeError(f"{path}: duplicate E3 window {raw}")
        windows.append(raw)
    if not saw_full or payload.get("full_condition") != "full":
        raise MergeError(f"{path}: E3 must declare one full window as condition=full")
    num_positions = payload.get("num_positions")
    if not isinstance(num_positions, int) or isinstance(num_positions, bool) or num_positions <= 0:
        raise MergeError(f"{path}: num_positions must be a positive integer")
    position_grid = payload.get("position_grid")
    if position_grid not in {"even", "e0-informed"}:
        raise MergeError(f"{path}: invalid E3 position_grid {position_grid!r}")
    if position_grid == "e0-informed" and num_positions != 5:
        raise MergeError(f"{path}: e0-informed E3 requires num_positions=5")
    condition_prefix = payload.get("condition_prefix", "e3")
    if condition_prefix != "e3":
        raise MergeError(f"{path}: E3 condition_prefix must be 'e3'")
    return E3Contract(tuple(windows), num_positions, position_grid, condition_prefix)


def _read_run_contract(run_dir, num_shards):
    """Cross-check run metadata and return the merge/structure contract."""

    values = []
    metadata = []
    launcher_path = run_dir / "launcher.json"
    if launcher_path.exists():
        launcher = _read_json(launcher_path)
        metadata.append((launcher_path, launcher))
        if "top_k" not in launcher:
            raise MergeError(f"{launcher_path}: missing top_k")
        values.append((launcher_path, launcher["top_k"]))
        if "num_shards" in launcher:
            recorded = _nonnegative_int(
                launcher["num_shards"],
                path=launcher_path,
                column="num_shards",
                positive=True,
            )
            if recorded != num_shards:
                raise MergeError(
                    f"{launcher_path}: num_shards={recorded}, merge requested {num_shards}"
                )

    manifest_paths = [run_dir / f"run_shard{index}.json" for index in range(num_shards)]
    present = [path for path in manifest_paths if path.exists()]
    if present and len(present) != num_shards:
        missing = [path.name for path in manifest_paths if not path.exists()]
        raise MergeError(f"{run_dir}: missing runner manifests {missing}")
    for path in present:
        manifest = _read_json(path)
        metadata.append((path, manifest))
        if "top_k" not in manifest:
            raise MergeError(f"{path}: missing top_k")
        values.append((path, manifest["top_k"]))

    if not values:
        raise MergeError(
            f"{run_dir}: cannot validate pred_topk without top_k in "
            "launcher.json or run_shard manifests"
        )
    parsed = [(path, _nonnegative_int(value, path=path, column="top_k")) for path, value in values]
    top_k = parsed[0][1]
    mismatches = [(path, value) for path, value in parsed if value != top_k]
    if mismatches:
        details = ", ".join(f"{path.name}={value}" for path, value in parsed)
        raise MergeError(f"{run_dir}: inconsistent top_k metadata: {details}")
    e3_values = []
    non_e3_paths = []
    for path, payload in metadata:
        declares_e3 = (
            payload.get("suite") == "e3"
            or payload.get("condition") == "e3"
            or payload.get("condition_prefix") == "e3"
        )
        if declares_e3:
            e3_values.append((path, _normalize_e3_contract(path, payload)))
        else:
            non_e3_paths.append(path)
    e3 = None
    if e3_values:
        if non_e3_paths:
            raise MergeError(
                f"{run_dir}: E3 metadata is missing from {[path.name for path in non_e3_paths]}"
            )
        e3 = e3_values[0][1]
        disagreements = [(path, value) for path, value in e3_values if value != e3]
        if disagreements:
            details = ", ".join(f"{path.name}={value}" for path, value in e3_values)
            raise MergeError(f"{run_dir}: inconsistent E3 metadata: {details}")
    return RunContract(top_k=top_k, e3=e3)


def _row_digest(row, header):
    """Return a stable digest of every serialized CSV field in header order."""

    digest = hashlib.sha256()
    for column in header:
        value = row[column]
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _merge_predictions(run_dir, shards, contract):
    output, output_path = _staged_writer(run_dir, "predictions")
    header = None
    writer = None
    parents = {}
    rows = 0
    duplicate_rows = 0
    try:
        for shard, path in enumerate(shards):
            handle, reader, current_header = _reader(path, header)
            with handle:
                if header is None:
                    header = current_header
                    _require_columns(
                        path,
                        header,
                        PARENT_KEY + PREDICTION_COUNTS + PREDICTION_PAYLOAD,
                    )
                    if contract.e3 is not None:
                        _require_columns(path, header, WINDOW_METADATA + ("human_choice",))
                    writer = csv.DictWriter(output, fieldnames=header)
                    writer.writeheader()
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise MergeError(f"{path}: malformed prediction CSV row")
                    key = tuple(row[column] for column in PARENT_KEY)
                    digest = _row_digest(row, header)
                    if key in parents:
                        previous = parents[key]
                        if previous.shard != shard:
                            raise MergeError(
                                f"{path}: duplicate prediction key {key} crosses shards"
                            )
                        if previous.row_digest != digest:
                            raise MergeError(
                                f"{path}: non-identical replay for prediction key {key}"
                            )
                        duplicate_rows += 1
                        continue
                    if contract.e3 is None and row["condition"].startswith("e3:"):
                        raise MergeError(
                            f"{path}: E3 prediction rows require E3 launcher/manifest metadata"
                        )
                    num_tokens = _nonnegative_int(
                        row["num_tokens"],
                        path=path,
                        column="num_tokens",
                        key=key,
                        positive=True,
                    )
                    k_options = _nonnegative_int(
                        row["k_options"],
                        path=path,
                        column="k_options",
                        key=key,
                        positive=True,
                    )
                    options_status = row["options_status"]
                    if options_status not in OPTIONS_STATUSES:
                        raise MergeError(
                            f"{path}: unknown options_status {options_status!r} "
                            f"for prediction {key}"
                        )
                    expected_options = k_options if options_status == OPTIONS_SCORED else 0
                    metadata = (
                        tuple(row[column] for column in WINDOW_METADATA)
                        if contract.e3 is not None
                        else None
                    )
                    parents[key] = ParentExpectation(
                        num_tokens=num_tokens,
                        expected_options=expected_options,
                        shard=shard,
                        row_digest=digest,
                        window_metadata=metadata,
                        human_choice=row["human_choice"],
                        options_status=options_status,
                    )
                    writer.writerow(row)
                    rows += 1
        if not rows:
            raise MergeError(f"{run_dir}: predictions contain no rows")
    except Exception:
        output.close()
        output_path.unlink(missing_ok=True)
        raise
    output.close()
    return output_path, parents, {"rows": rows, "duplicate_rows_dropped": duplicate_rows}


def _session_commit_name(experiment, participant):
    identity = json.dumps(
        [str(experiment), str(participant)], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".json"


def _read_session_commits(run_dir, num_shards):
    """Read the atomic receipts that make an E3 session resumably complete."""

    commits = {}
    for shard in range(num_shards):
        directory = run_dir / f"session_commits_shard{shard}"
        if not directory.is_dir():
            raise MergeError(f"{run_dir}: missing E3 session commit directory {directory.name}")
        for path in sorted(directory.glob("*.json")):
            payload = _read_json(path)
            required = {
                "schema",
                "experiment",
                "participant",
                "status",
                "prediction_rows",
                "conditions",
            }
            missing = sorted(required.difference(payload))
            if missing:
                raise MergeError(f"{path}: session commit is missing fields {missing}")
            if payload["schema"] != SESSION_COMMIT_SCHEMA:
                raise MergeError(f"{path}: unsupported session commit schema {payload['schema']!r}")
            key = (str(payload["experiment"]), str(payload["participant"]))
            if path.name != _session_commit_name(*key):
                raise MergeError(f"{path}: filename does not match its session identity")
            if key in commits:
                raise MergeError(f"{path}: duplicate session commit for {key}")
            if payload["status"] != "success":
                raise MergeError(f"{path}: E3 merge requires a successful session commit")
            prediction_rows = payload["prediction_rows"]
            conditions = payload["conditions"]
            if (
                not isinstance(prediction_rows, int)
                or isinstance(prediction_rows, bool)
                or prediction_rows <= 0
            ):
                raise MergeError(f"{path}: prediction_rows must be a positive integer")
            if not isinstance(conditions, dict) or any(
                not isinstance(condition, str)
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                for condition, count in conditions.items()
            ):
                raise MergeError(f"{path}: invalid condition counts")
            if sum(conditions.values()) != prediction_rows:
                raise MergeError(f"{path}: condition counts do not sum to prediction_rows")
            commits[key] = (shard, payload)
    return commits


def _strict_bool(value, *, run_dir, key, column):
    if value == "True":
        return True
    if value == "False":
        return False
    raise MergeError(f"{run_dir}: {column} must be True or False for prediction {key}")


def _validate_e3_sessions(run_dir, parents, contract, commits):
    """Require complete full choices and the exact finite-window anchor grid."""

    sessions = defaultdict(list)
    for key, expectation in parents.items():
        session = (key[0], key[1], key[3], key[4])
        sessions[session].append((key, expectation))

    marker_sessions = set(commits)
    prediction_sessions = {(session[2], session[3]) for session in sessions}
    if marker_sessions != prediction_sessions:
        missing = sorted(prediction_sessions - marker_sessions)[:3]
        extra = sorted(marker_sessions - prediction_sessions)[:3]
        raise MergeError(
            f"{run_dir}: E3 session commits disagree with predictions; "
            f"missing={missing}, extra={extra}"
        )

    expected_conditions = {"full"} | {
        f"{contract.condition_prefix}:w={window}" for window in contract.windows
    }
    for session, rows in sessions.items():
        model, dataset, experiment, participant = session
        del model, dataset
        session_key = (experiment, participant)
        marker_shard, marker = commits[session_key]
        shards = {expectation.shard for _, expectation in rows}
        if shards != {marker_shard}:
            raise MergeError(
                f"{run_dir}: E3 session {session_key} spans prediction shards {sorted(shards)} "
                f"but its commit is in shard {marker_shard}"
            )
        condition_counts = Counter(key[2] for key, _ in rows)
        if len(rows) != marker["prediction_rows"] or dict(condition_counts) != marker["conditions"]:
            raise MergeError(
                f"{run_dir}: E3 session {session_key} does not match its atomic commit counts"
            )
        if set(condition_counts) != expected_conditions:
            raise MergeError(
                f"{run_dir}: E3 session {session_key} has conditions "
                f"{sorted(condition_counts)}, expected {sorted(expected_conditions)}"
            )

        normalized = []
        n_segments_values = set()
        for key, expectation in rows:
            if expectation.window_metadata is None:
                raise MergeError(f"{run_dir}: missing E3 metadata for prediction {key}")
            metadata = dict(zip(WINDOW_METADATA, expectation.window_metadata))
            target = _nonnegative_int(
                metadata["target_index"], path=run_dir, column="target_index", key=key
            )
            effective = _nonnegative_int(
                metadata["effective_window"],
                path=run_dir,
                column="effective_window",
                key=key,
            )
            n_segments = _nonnegative_int(
                metadata["n_segments"],
                path=run_dir,
                column="n_segments",
                key=key,
                positive=True,
            )
            choice_index = _nonnegative_int(key[5], path=run_dir, column="choice_index", key=key)
            if target >= n_segments:
                raise MergeError(
                    f"{run_dir}: target_index {target} exceeds n_segments={n_segments} for {key}"
                )
            if metadata["position_grid"] != contract.position_grid:
                raise MergeError(f"{run_dir}: wrong position_grid for prediction {key}")
            try:
                target_fraction = float(metadata["target_fraction"])
            except ValueError as error:
                raise MergeError(
                    f"{run_dir}: invalid target_fraction for prediction {key}"
                ) from error
            expected_fraction = target / max(n_segments - 1, 1)
            if not math.isfinite(target_fraction) or not math.isclose(
                target_fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-12
            ):
                raise MergeError(f"{run_dir}: wrong target_fraction for prediction {key}")
            is_anchor = _strict_bool(
                metadata["is_anchor"], run_dir=run_dir, key=key, column="is_anchor"
            )
            n_segments_values.add(n_segments)
            normalized.append(
                {
                    "condition": key[2],
                    "choice_index": choice_index,
                    "human_choice": expectation.human_choice,
                    "window": metadata["window"],
                    "target": target,
                    "effective": effective,
                    "is_anchor": is_anchor,
                }
            )
        if len(n_segments_values) != 1:
            raise MergeError(f"{run_dir}: E3 session {session_key} has inconsistent n_segments")
        n_segments = next(iter(n_segments_values))
        targets = set(
            grid_targets(n_segments, contract.num_positions, strategy=contract.position_grid)
        )

        full_rows = [row for row in normalized if row["condition"] == "full"]
        full_by_choice = {}
        for row in full_rows:
            choice_index = row["choice_index"]
            if choice_index in full_by_choice:
                raise MergeError(
                    f"{run_dir}: E3 session {session_key} repeats numeric full choice {choice_index}"
                )
            if row["window"] != "full" or row["effective"] != row["target"]:
                raise MergeError(
                    f"{run_dir}: E3 session {session_key} has invalid full-window metadata"
                )
            if row["is_anchor"] != (row["target"] in targets):
                raise MergeError(
                    f"{run_dir}: E3 session {session_key} has wrong full is_anchor flag"
                )
            full_by_choice[choice_index] = (row["target"], row["human_choice"])
        if sorted(full_by_choice) != list(range(len(full_by_choice))):
            raise MergeError(
                f"{run_dir}: E3 session {session_key} full choice_index is not contiguous"
            )
        ordered_targets = [full_by_choice[index][0] for index in range(len(full_by_choice))]
        if ordered_targets != sorted(ordered_targets) or set(ordered_targets) != set(
            range(n_segments)
        ):
            raise MergeError(
                f"{run_dir}: E3 session {session_key} full rows do not cover every segment"
            )

        expected_anchor_choices = {
            index: value for index, value in full_by_choice.items() if value[0] in targets
        }
        for window in contract.windows:
            condition = f"{contract.condition_prefix}:w={window}"
            actual = {}
            for row in normalized:
                if row["condition"] != condition:
                    continue
                choice_index = row["choice_index"]
                if choice_index in actual:
                    raise MergeError(
                        f"{run_dir}: E3 session {session_key} repeats {condition} choice "
                        f"{choice_index}"
                    )
                if (
                    row["window"] != str(window)
                    or row["effective"] != min(row["target"], window)
                    or not row["is_anchor"]
                ):
                    raise MergeError(
                        f"{run_dir}: E3 session {session_key} has invalid {condition} metadata"
                    )
                actual[choice_index] = (row["target"], row["human_choice"])
            if actual != expected_anchor_choices:
                raise MergeError(
                    f"{run_dir}: E3 session {session_key} has incomplete {condition} anchor coverage"
                )


def _child_suffix(path, table, row):
    if table == "pred_options":
        return (row["option"],)
    return (
        _nonnegative_int(row["token_index"], path=path, column="token_index"),
        _nonnegative_int(row["rank"], path=path, column="rank"),
    )


def _expected_child_rows(table, expectation, top_k):
    if table == "pred_topk":
        return expectation.num_tokens * top_k
    return expectation.expected_options


def _complete_occurrence(table, suffixes, expectation, top_k):
    expected = _expected_child_rows(table, expectation, top_k)
    if len(suffixes) != expected or len(set(suffixes)) != expected:
        return False
    if table == "pred_options":
        # The prediction ledger records the option count, while the child
        # rows carry the strings. Exact unique count establishes structural
        # completeness; payload semantics are checked separately below.
        return True
    return all(
        0 <= token_index < expectation.num_tokens and 0 <= rank < top_k
        for token_index, rank in suffixes
    )


def _validate_option_occurrence(path, parent, rows, expectation):
    """Validate the all-single-token option payload for one scored choice."""

    if expectation.options_status != OPTIONS_SCORED:
        raise MergeError(f"{path}: pred_options rows exist for non-scored prediction {parent}")
    if not expectation.human_choice:
        raise MergeError(f"{path}: scored prediction {parent} has an empty human_choice")

    human_options = []
    for row in rows:
        if not row["option"]:
            raise MergeError(f"{path}: pred_options has an empty option for {parent}")
        n_tokens = _nonnegative_int(
            row["n_tokens"], path=path, column="n_tokens", key=parent, positive=True
        )
        if n_tokens != 1:
            raise MergeError(
                f"{path}: scored pred_options must have n_tokens=1 for {parent}; got {n_tokens}"
            )
        try:
            logprob = float(row["logprob"])
        except ValueError as error:
            raise MergeError(
                f"{path}: pred_options logprob must be numeric for {parent}; got {row['logprob']!r}"
            ) from error
        if not math.isfinite(logprob) or logprob > 0:
            raise MergeError(
                f"{path}: pred_options logprob must be finite and non-positive "
                f"for {parent}; got {row['logprob']!r}"
            )
        if _strict_bool(row["is_human"], run_dir=path, key=parent, column="is_human"):
            human_options.append(row["option"])

    if human_options != [expectation.human_choice]:
        raise MergeError(
            f"{path}: pred_options must mark exactly human_choice="
            f"{expectation.human_choice!r} for {parent}; got {human_options!r}"
        )


def _merge_child(run_dir, table, shards, parents, top_k):
    output, output_path = _staged_writer(run_dir, table)
    header = None
    writer = None
    completed = {}
    rows = 0
    orphan_rows = 0
    duplicate_rows = 0
    incomplete_rows = 0
    suffix_columns = CHILD_SUFFIX[table]
    active_parent = None
    active_rows = []
    active_suffixes = []

    def finish_occurrence():
        nonlocal active_parent, active_rows, active_suffixes
        nonlocal rows, duplicate_rows, incomplete_rows
        if active_parent is None:
            return
        expectation = parents[active_parent]
        complete = _complete_occurrence(table, active_suffixes, expectation, top_k)
        if complete and table == "pred_options":
            _validate_option_occurrence(run_dir / table, active_parent, active_rows, expectation)
        if active_parent in completed:
            if complete:
                replay_digest = hashlib.sha256(
                    b"".join(_row_digest(row, header) for row in active_rows)
                ).digest()
                if replay_digest != completed[active_parent]:
                    raise MergeError(
                        f"{run_dir}/{table}: non-identical complete replay for {active_parent}"
                    )
            duplicate_rows += len(active_rows)
        elif complete:
            writer.writerows(active_rows)
            rows += len(active_rows)
            completed[active_parent] = hashlib.sha256(
                b"".join(_row_digest(row, header) for row in active_rows)
            ).digest()
        else:
            # A pre-ledger crash can leave exactly this kind of partial first
            # occurrence. It is safe to drop only because merge later requires
            # another, complete occurrence for every parent that expects rows.
            incomplete_rows += len(active_rows)
        active_parent = None
        active_rows = []
        active_suffixes = []

    try:
        for shard, path in enumerate(shards):
            handle, reader, current_header = _reader(path, header)
            with handle:
                if header is None:
                    header = current_header
                    _require_columns(
                        path,
                        header,
                        PARENT_KEY + suffix_columns + CHILD_PAYLOAD[table],
                    )
                    if any(
                        expectation.window_metadata is not None for expectation in parents.values()
                    ):
                        _require_columns(path, header, WINDOW_METADATA)
                    writer = csv.DictWriter(output, fieldnames=header)
                    writer.writeheader()
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise MergeError(f"{path}: malformed {table} CSV row")
                    parent = tuple(row[column] for column in PARENT_KEY)
                    if parent not in parents:
                        finish_occurrence()
                        orphan_rows += 1
                        continue
                    expectation = parents[parent]
                    if expectation.shard != shard:
                        raise MergeError(
                            f"{path}: child row for {parent} crosses from prediction shard "
                            f"{expectation.shard}"
                        )
                    if expectation.window_metadata is not None:
                        metadata = tuple(row[column] for column in WINDOW_METADATA)
                        if metadata != expectation.window_metadata:
                            raise MergeError(
                                f"{path}: child E3 metadata differs from prediction {parent}"
                            )
                    expected = _expected_child_rows(table, parents[parent], top_k)
                    if expected == 0:
                        raise MergeError(
                            f"{path}: {table} row exists for prediction "
                            f"{parent}, which expects exactly 0 rows"
                        )
                    suffix = _child_suffix(path, table, row)
                    if parent != active_parent:
                        finish_occurrence()
                        active_parent = parent
                    elif suffix in active_suffixes:
                        # Replays restart the same parent at its first suffix.
                        # A repeated suffix therefore separates a partial old
                        # occurrence from the newly appended occurrence.
                        finish_occurrence()
                        active_parent = parent
                    active_rows.append(row)
                    active_suffixes.append(suffix)
                    if len(active_rows) == expected:
                        finish_occurrence()
            # Occurrences cannot span shard files, even when the last parent
            # key in one shard happens to equal the first key in another.
            finish_occurrence()

        missing_count = 0
        examples = []
        for parent, expectation in parents.items():
            if _expected_child_rows(table, expectation, top_k) > 0 and parent not in completed:
                missing_count += 1
                if len(examples) < 3:
                    examples.append(parent)
        if missing_count:
            raise MergeError(
                f"{run_dir}/{table}: {missing_count} predictions have no "
                f"complete child occurrence; examples: {examples}"
            )
    except Exception:
        output.close()
        output_path.unlink(missing_ok=True)
        raise
    output.close()
    return output_path, {
        "rows": rows,
        "orphan_rows_dropped": orphan_rows,
        "duplicate_rows_dropped": duplicate_rows,
        "incomplete_rows_dropped": incomplete_rows,
    }


def merge_run(run_dir, num_shards):
    """Merge one run directory and return per-table row statistics."""

    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise MergeError(f"run directory does not exist: {run_dir}")
    contract = _read_run_contract(run_dir, num_shards)
    shards = {table: _shard_paths(run_dir, table, num_shards) for table in TABLES}
    failures = _nonempty_failure_logs(shards["predictions"])
    if failures:
        names = ", ".join(str(path) for path in failures)
        raise MergeError(f"failed sessions must be resolved before merge: {names}")

    staged = {}
    stats = {}
    try:
        staged["predictions"], parents, stats["predictions"] = _merge_predictions(
            run_dir, shards["predictions"], contract
        )
        if contract.e3 is not None:
            commits = _read_session_commits(run_dir, num_shards)
            _validate_e3_sessions(run_dir, parents, contract.e3, commits)
        for table in TABLES[1:]:
            staged[table], stats[table] = _merge_child(
                run_dir, table, shards[table], parents, contract.top_k
            )
        # Session commits are the resume ledger; predictions still publishes
        # last so readers never observe a new parent before both child tables.
        for table in TABLES[1:] + ("predictions",):
            os.replace(staged[table], run_dir / f"{table}.csv")
    except Exception:
        for path in staged.values():
            path.unlink(missing_ok=True)
        raise
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--run-dir", nargs="+", required=True, help="One or more outputs/runs/<tag> directories"
    )
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    for run_dir in args.run_dir:
        try:
            stats = merge_run(run_dir, args.num_shards)
        except MergeError as error:
            parser.exit(1, f"merge failed: {error}\n")
        summary = ", ".join(f"{table}={values['rows']}" for table, values in stats.items())
        print(f"{run_dir}: {summary}", flush=True)
        for table in TABLES[1:]:
            values = stats[table]
            if (
                values["orphan_rows_dropped"]
                or values["duplicate_rows_dropped"]
                or values["incomplete_rows_dropped"]
            ):
                print(
                    f"  {table}: dropped "
                    f"{values['orphan_rows_dropped']} orphan and "
                    f"{values['duplicate_rows_dropped']} duplicate and "
                    f"{values['incomplete_rows_dropped']} incomplete rows",
                    flush=True,
                )


if __name__ == "__main__":
    main()
