"""Offline contracts for the Track P Slurm launchers."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import os
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]


def run_submit(*args):
    env = os.environ.copy()
    env.pop("MT_PROTOCOL_TAG", None)
    env.pop("MT_LOAD", None)
    return subprocess.run(
        ["bash", str(REPO / "scripts/submit_roster.sh"), *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )


def shell_function(relative, name):
    """Extract one shell function without sourcing the surrounding launcher."""

    source = (REPO / relative).read_text(encoding="utf-8")
    match = re.search(
        rf"^(?P<indent>[ \t]*){re.escape(name)}\(\) \{{\n", source, flags=re.MULTILINE
    )
    assert match is not None, f"missing {name} in {relative}"
    closing = f"\n{match.group('indent')}}}\n"
    end = source.index(closing, match.end()) + len(closing)
    return source[match.start() : end]


def test_track_p_launchers_pass_run_directories():
    expected = {
        "scripts/score_model.slurm": 5,
        "scripts/archive/smoke_e0_e3.slurm": 1,
        "scripts/archive/e0_e3_minitaur.slurm": 1,
        "scripts/template_gpu_job.slurm": 1,
        "scripts/run_e0_llama_base.ps1": 1,
    }
    for relative, count in expected.items():
        text = (REPO / relative).read_text(encoding="utf-8")
        assert text.count("--output-dir") == count, relative


def test_shard_wait_retries_after_signal_interrupt(tmp_path):
    source = (REPO / "scripts/score_model.slurm").read_text(encoding="utf-8")
    start = source.index("wait_for_pid() {")
    end = source.index("\n}\n\nwait_for_shards()", start) + 3
    function = source[start:end]
    probe = tmp_path / "wait_probe.sh"
    probe.write_text(
        "#!/usr/bin/bash\n"
        "set -u\n"
        f"{function}\n"
        "trap ':' USR1\n"
        "sleep 0.2 &\n"
        "worker=$!\n"
        "(sleep 0.05; kill -USR1 $$) &\n"
        "signaler=$!\n"
        'wait_for_pid "$worker"\n'
        'wait "$signaler"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(probe)], capture_output=True, text=True, timeout=2, check=False
    )
    assert result.returncode == 0, result.stderr


def test_track_p_submission_scripts_are_valid_bash():
    for relative in (
        "scripts/submit_roster.sh",
        "scripts/score_model.slurm",
        "scripts/merge_shards.slurm",
    ):
        result = subprocess.run(
            ["bash", "-n", str(REPO / relative)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        assert result.returncode == 0, f"{relative}: {result.stderr}"


@pytest.mark.parametrize(
    "relative",
    (
        "scripts/submit_roster.sh",
        "scripts/score_model.slurm",
        "scripts/merge_shards.slurm",
    ),
)
def test_live_slurm_query_distinguishes_expired_ids_from_controller_failure(relative):
    function = shell_function(relative, "query_live_slurm_job")
    probe = "\n".join(
        (
            "set -u",
            "squeue() {",
            '    [ "$#" -eq 3 ] && [ "$1" = -h ] && [ "$2" = -o ] && '
            '        [ "$3" = "%A|%T" ] || return 88',
            '    case "$SQUEUE_MODE" in',
            "        active) printf ' 1234 | RUNNING \\n 123 | PENDING \\n' ;;",
            "        expired) printf ' 1234 | RUNNING \\n' ;;",
            "        failure) return 72 ;;",
            "    esac",
            "}",
            function,
            'if state=$(query_live_slurm_job "$JOB_ID"); then',
            "    printf 'state=%s\\n' \"$state\"",
            "else",
            "    exit 42",
            "fi",
        )
    )
    cases = (
        ("active", "123", 0, "state=PENDING\n"),
        ("expired", "123", 0, "state=\n"),
        ("failure", "123", 42, ""),
    )
    for mode, job_id, returncode, stdout in cases:
        env = os.environ.copy()
        env.update(SQUEUE_MODE=mode, JOB_ID=job_id)
        result = subprocess.run(
            ["bash", "-c", probe],
            env=env,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        assert result.returncode == returncode, result.stderr
        assert result.stdout == stdout


@pytest.mark.parametrize(
    "relative",
    ("scripts/score_model.slurm", "scripts/merge_shards.slurm"),
)
def test_manual_locks_are_recoverable_only_when_safe(tmp_path, relative):
    live_query = shell_function(relative, "query_live_slurm_job")
    owner_query = shell_function(relative, "query_lock_owner_state")
    archive = shell_function(relative, "archive_stale_lock")
    active_lock = tmp_path / relative.replace("/", "_") / "active.lock"
    stale_lock = active_lock.parent / "stale.lock"
    active_lock.mkdir(parents=True)
    stale_lock.mkdir()
    (stale_lock / "owner").write_text("manual:test-node:99999999\n", encoding="utf-8")
    probe = "\n".join(
        (
            "set -u",
            "squeue() { return 91; }",
            'LOCK_HOST="test-node"',
            'LOCK_TOKEN="test-token"',
            live_query,
            owner_query,
            archive,
            'printf "manual:%s:%s\\n" "$LOCK_HOST" "$$" > "$ACTIVE_LOCK/owner"',
            'if archive_stale_lock "$ACTIVE_LOCK" score; then exit 10; fi',
            '[ -d "$ACTIVE_LOCK" ] || exit 11',
            'if query_lock_owner_state "manual:other-node:$$"; then exit 12; fi',
            'archive_stale_lock "$STALE_LOCK" score',
            '[ ! -e "$STALE_LOCK" ] || exit 13',
        )
    )
    env = os.environ.copy()
    env.update(ACTIVE_LOCK=str(active_lock), STALE_LOCK=str(stale_lock))
    result = subprocess.run(
        ["bash", "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(active_lock.parent.glob("stale.lock.stale.manual:test-node:99999999.*"))


def test_l3_resume_proof_propagates_commit_marker_failure(tmp_path):
    count_commits = shell_function("scripts/submit_roster.sh", "count_success_commits")
    prepare_proof = shell_function("scripts/submit_roster.sh", "prepare_l3_resume_proof")
    run_dir = tmp_path / "l3"
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    for shard in range(4):
        commit_dir = run_dir / f"session_commits_shard{shard}"
        commit_dir.mkdir(parents=True)
        status = "failed" if shard == 0 else "success"
        (commit_dir / "marker.json").write_text(f'{{"status": "{status}"}}\n', encoding="utf-8")
        (run_dir / f"predictions_shard{shard}.csv").write_text("header\nrow\n", encoding="utf-8")
    cancel_marker = gate_dir / "e3-l3.cancel-verified"
    probe = "\n".join(
        (
            "set -euo pipefail",
            'SUITE="e3"',
            'GATE_DIR="$TEST_GATE_DIR"',
            count_commits,
            prepare_proof,
            'if proof=$(prepare_l3_resume_proof "$TEST_RUN_DIR" "$TEST_CANCEL_MARKER"); then',
            "    exit 20",
            "fi",
            '[ ! -e "${TEST_CANCEL_MARKER%.cancel-verified}.resume-proof.tsv" ]',
        )
    )
    env = os.environ.copy()
    env.update(
        TEST_GATE_DIR=str(gate_dir),
        TEST_RUN_DIR=str(run_dir),
        TEST_CANCEL_MARKER=str(cancel_marker),
    )
    result = subprocess.run(
        ["bash", "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "non-success marker" in result.stderr


def test_l3_cancel_marker_is_bound_to_latest_job(tmp_path):
    job_state_file = shell_function("scripts/submit_roster.sh", "job_state_file")
    live_query = shell_function("scripts/submit_roster.sh", "query_live_slurm_job")
    l3_attempt = shell_function("scripts/submit_roster.sh", "l3_attempt")
    probe = "\n".join(
        (
            "set -u",
            "squeue() {",
            '    [ "$#" -eq 3 ] && [ "$1" = -h ] && [ "$2" = -o ] && '
            '        [ "$3" = "%A|%T" ] || return 88',
            "    return 0",
            "}",
            "sacct() {",
            '    [ "$1" = -n ] && [ "$2" = -X ] && [ "$3" = -j ] && '
            '        [ "$4" = 202 ] || return 89',
            '    printf "%s\\n" "$SACCT_STATE"',
            "}",
            'SUITE="e3"',
            "DRY_RUN=0",
            'GATE_DIR="$TEST_GATE_DIR"',
            job_state_file,
            live_query,
            l3_attempt,
            "l3_attempt centaur8b",
        )
    )
    for terminal_state, expected_returncode in (("CANCELLED", 0), ("COMPLETED", 1)):
        gate_dir = tmp_path / terminal_state.lower()
        gate_dir.mkdir()
        state_file = gate_dir / "e3-l3-centaur8b.jobs"
        cancel_marker = gate_dir / "e3-l3-centaur8b.cancel-verified"
        state_file.write_text("101\n202\n", encoding="utf-8")
        cancel_marker.write_text("101\n", encoding="utf-8")
        env = os.environ.copy()
        env.update(TEST_GATE_DIR=str(gate_dir), SACCT_STATE=terminal_state)
        result = subprocess.run(
            ["bash", "-c", probe],
            env=env,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        assert result.returncode == expected_returncode, result.stderr
        if terminal_state == "CANCELLED":
            assert cancel_marker.read_text(encoding="utf-8") == "202\n"
            assert result.stdout.startswith("3|")
        else:
            assert cancel_marker.read_text(encoding="utf-8") == "101\n"
            assert "requires an observed scancel" in result.stderr


def test_roster_exposes_gemma_as_deferred_but_does_not_submit_it():
    listed = run_submit("list")
    assert listed.returncode == 0
    assert "gemma4_e2b" in listed.stdout
    assert "deferred" in listed.stdout
    assert "minitaur8b" not in listed.stdout

    rejected = run_submit("dry-run", "smoke", "gemma4_e2b")
    assert rejected.returncode != 0
    assert "deferred" in rejected.stderr
    assert "submit:" not in rejected.stderr

    retired = run_submit("dry-run", "smoke", "minitaur8b")
    assert retired.returncode == 2
    assert "unknown roster tag" in retired.stderr
    assert "submit:" not in retired.stderr


def test_roster_requires_explicit_tags_instead_of_implicit_all():
    result = run_submit("dry-run", "smoke")
    assert result.returncode == 2
    assert "implicit all-roster" in result.stderr


def test_roster_rejects_duplicate_tags_before_printing_any_submission():
    result = run_submit("dry-run", "e3", "smoke", "centaur8b", "centaur8b")
    assert result.returncode == 2
    assert "duplicate roster tag" in result.stderr
    assert "submit:" not in result.stderr


def test_smoke_dry_run_requests_exactly_one_gpu_and_seven_cores():
    result = run_submit("dry-run", "smoke", "centaur8b")
    assert result.returncode == 0, result.stderr
    command = result.stderr
    assert "--gpus-per-task=1" in command
    assert "--cpus-per-task=7" in command
    assert "MT_RUN_MODE=smoke" in command
    assert "MT_NGPU=1" in command
    assert "greedy-unconstrained-1token" in command
    assert "trackp-greedy1tok-v1-" in command
    assert "Llama-3.1-Centaur-8B-adapter" in command


def test_full_dry_run_is_explicit_and_not_polluted_by_smoke_environment():
    env = os.environ.copy()
    env["MT_SMOKE"] = "999"
    env["MT_LIMIT"] = "1"
    result = subprocess.run(
        ["bash", str(REPO / "scripts/submit_roster.sh"), "dry-run", "full", "centaur8b"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    command = result.stderr
    assert "MT_RUN_MODE=full" in command
    assert "MT_NGPU=4" in command
    assert "MT_SMOKE" not in command
    assert "MT_LIMIT" not in command
    assert "Llama-3.1-Centaur-8B-adapter" in command


def test_full_launcher_has_l3_gate_and_merge_acceptance_contracts():
    submit = (REPO / "scripts/submit_roster.sh").read_text(encoding="utf-8")
    score = (REPO / "scripts/score_model.slurm").read_text(encoding="utf-8")
    merge = (REPO / "scripts/merge_shards.slurm").read_text(encoding="utf-8")

    assert "assert_full_gate" in submit
    assert "assert_smoke_gate" in submit
    assert "git status --porcelain" in submit
    assert "CANCELLED" in submit
    assert '--dependency="afterok:${score_job}"' in submit
    assert '--choice-readout "$CHOICE_READOUT"' in score
    assert score.count('--choice-readout "$CHOICE_READOUT"') == 4
    assert score.count("--max-options 256") == 4
    assert "export OMP_NUM_THREADS=7" in score
    assert 'for kind in ("failed", "skipped")' in score
    assert "MT_EXPECTED_SESSIONS" in merge
    assert "predictions*.skipped.csv" in merge
    assert "MT_GATE_FILE" in merge
    assert "session_commits_shard" in submit
    assert "MT_L3_PROOF_FILE" in submit
    assert "MT_L3_PROOF_FILE" in merge
    assert ".score.lock" in score
    assert ".merge.lock" in score
    assert ".score.lock" in merge
    assert ".merge.lock" in merge


def test_preflight_checks_the_adapter_cache_separately():
    score = (REPO / "scripts/score_model.slurm").read_text(encoding="utf-8")
    preflight = (REPO / "scripts/experiments/preflight.py").read_text(encoding="utf-8")

    assert "--adapter" in preflight
    assert 'check_model_cache(args.adapter, label="adapter")' in preflight
    assert 'scripts/experiments/preflight.py --model "$MODEL"' in score
    assert "ADAPTER_ARGS" in score
    assert '--load "$LOAD" --output-dir "$RUN_DIR"' in score


def test_attention_preflight_handles_query_allocation_oom(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "track_p_preflight", REPO / "scripts/experiments/preflight.py"
    )
    assert spec is not None and spec.loader is not None
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)

    class FakeOutOfMemoryError(Exception):
        pass

    emptied = []
    fake_cuda = SimpleNamespace(
        OutOfMemoryError=FakeOutOfMemoryError,
        is_available=lambda: True,
        reset_peak_memory_stats=lambda _device: None,
        max_memory_allocated=lambda _device: 0,
        empty_cache=lambda: emptied.append(True),
    )

    def raise_oom(*_args, **_kwargs):
        raise FakeOutOfMemoryError

    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        device=lambda value: value,
        float16="float16",
        randn=raise_oom,
    )
    transcript_scoring = ModuleType("mt.evaluation.transcript_scoring")

    @contextmanager
    def fake_sdpa_context(_device):
        yield

    transcript_scoring._cuda_sdpa_context = fake_sdpa_context
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "mt.evaluation.transcript_scoring", transcript_scoring)
    failures = []
    monkeypatch.setattr(preflight, "fail", failures.append)

    preflight.check_attention_backend(seq_len=8)

    assert failures and "ran out of memory" in failures[0]
    assert emptied == [True]


def test_formal_roster_is_pinned_to_nf4():
    env = os.environ.copy()
    env["MT_LOAD"] = "none"
    result = subprocess.run(
        ["bash", str(REPO / "scripts/submit_roster.sh"), "dry-run", "full", "centaur8b"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    assert result.returncode == 2
    assert "pinned to MT_LOAD=4bit" in result.stderr


def test_dry_run_covers_l3_full_and_merge_per_tag():
    l3 = run_submit("dry-run", "l3", "centaur8b")
    full = run_submit("dry-run", "full", "centaur8b")
    merge = run_submit("dry-run", "merge", "centaur8b")
    assert l3.returncode == full.returncode == merge.returncode == 0
    assert "MT_RUN_MODE=l3" in l3.stderr
    assert "MT_L3_LIMIT=200" in l3.stderr
    assert "Llama-3.1-Centaur-8B-adapter" in l3.stderr
    assert "MT_RUN_MODE=full" in full.stderr
    assert "MT_EXPECTED_SESSIONS=6561" in merge.stderr


def test_e3_dry_run_uses_separate_gates_paths_and_suite():
    smoke = run_submit("dry-run", "e3", "smoke", "centaur8b")
    l3 = run_submit("dry-run", "e3", "l3", "centaur8b")
    full = run_submit("dry-run", "e3", "full", "centaur8b")
    merge = run_submit("dry-run", "e3", "merge", "centaur8b")

    assert smoke.returncode == l3.returncode == full.returncode == merge.returncode == 0
    assert "MT_SUITE=e3" in smoke.stderr
    assert "mt-e3-smoke-centaur8b" in smoke.stderr
    assert "/centaur8b/e3" in smoke.stderr
    assert "MT_RUN_MODE=l3" in l3.stderr
    assert "e3-l3-passed" in full.stderr
    assert "MT_RUN_MODE=full" in full.stderr
    assert "/centaur8b/e3/full" in merge.stderr


def test_e3_smoke_uses_the_same_strict_shard_merger_as_formal_runs():
    score = (REPO / "scripts/score_model.slurm").read_text(encoding="utf-8")

    assert "--shard 0/1" in score
    assert "python scripts/experiments/merge_run_shards.py" in score
    assert '--num-shards 1 --run-dir "$RUN_DIR"' in score
    assert '"segmentation_protocol": "last-pre-marker-blank-v1"' in score
    assert '"window_unit": "marked-choice-segment"' in score


def test_e3_requires_an_explicit_model_tag():
    result = run_submit("dry-run", "e3", "smoke")
    assert result.returncode == 2
    assert "implicit all-roster" in result.stderr


def test_e3_all_contains_six_active_models_and_no_deferred_models():
    result = run_submit("dry-run", "e3", "smoke", "all")
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("scripts/score_model.slurm") == 6
    assert "Minitaur" not in result.stderr
    assert "gemma-4" not in result.stderr


def test_no_submittable_job_script_names_a_retired_model():
    # The merged Minitaur checkpoint left the formal roster (design §6.2).
    # Anything directly submittable with `sbatch scripts/<name>` must not be
    # able to start it; the v0 job scripts live under scripts/archive/.
    submittable = sorted(
        path
        for pattern in ("*.slurm", "*.sh")
        for path in (REPO / "scripts").glob(pattern)
    )
    assert submittable, "expected job scripts directly under scripts/"
    offenders = {
        path.name: [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if "minitaur" in line.lower()
        ]
        for path in submittable
    }
    assert not {name: lines for name, lines in offenders.items() if lines}


def test_every_submission_passes_mail_user_explicitly():
    # Exporting SBATCH_MAIL_USER fed the in-process progress notifier but
    # produced no Slurm BEGIN/END/FAIL mail, so the address is passed as a
    # flag now. Every sbatch this launcher issues must carry it.
    env = os.environ.copy()
    env.pop("MT_PROTOCOL_TAG", None)
    env.pop("MT_LOAD", None)
    env["SBATCH_MAIL_USER"] = "someone@uni.lu"
    env.pop("MT_NOTIFY_EMAIL", None)
    for args in (
        ("dry-run", "e3", "smoke", "centaur8b"),
        ("dry-run", "e3", "l3", "centaur8b"),
        ("dry-run", "e3", "full", "centaur8b"),
        ("dry-run", "e3", "merge", "centaur8b"),
    ):
        result = subprocess.run(
            ["bash", str(REPO / "scripts/submit_roster.sh"), *args],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        submits = [
            line for line in result.stderr.splitlines() if "score_model.slurm" in line
            or "merge_shards.slurm" in line
        ]
        assert submits, f"{args}: no sbatch command printed"
        for line in submits:
            assert "--mail-user=someone@uni.lu" in line.replace("\\", ""), (args, line)
