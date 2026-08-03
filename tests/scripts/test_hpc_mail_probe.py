from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "hpc_mail_probe.sh"
BASH = shutil.which("bash") or "/usr/bin/bash"


def write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def probe_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    capture_dir = tmp_path / "capture"
    bin_dir.mkdir()
    capture_dir.mkdir()

    write_executable(
        bin_dir / "timeout",
        "#!/bin/sh\nshift\nexec \"$@\"\n",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PROBE_CAPTURE_DIR"] = str(capture_dir)
    env.pop("SLURM_JOB_ID", None)
    return env, bin_dir, capture_dir


def run_probe(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [BASH, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_check_only_reports_capabilities_without_running_commands(tmp_path: Path) -> None:
    env, bin_dir, capture_dir = probe_environment(tmp_path)
    write_executable(
        bin_dir / "sendmail",
        "#!/bin/sh\ntouch \"$PROBE_CAPTURE_DIR/sendmail-ran\"\n",
    )
    write_executable(
        bin_dir / "sbatch",
        "#!/bin/sh\ntouch \"$PROBE_CAPTURE_DIR/sbatch-ran\"\n",
    )

    completed = run_probe("--check-only", env=env)

    assert completed.returncode == 0
    assert f"sendmail: {bin_dir / 'sendmail'}" in completed.stdout
    assert f"sbatch:   {bin_dir / 'sbatch'}" in completed.stdout
    assert not (capture_dir / "sendmail-ran").exists()
    assert not (capture_dir / "sbatch-ran").exists()


def test_send_test_uses_sendmail_with_fixed_headers(tmp_path: Path) -> None:
    env, bin_dir, capture_dir = probe_environment(tmp_path)
    write_executable(
        bin_dir / "sendmail",
        """#!/bin/sh
printf '%s\n' "$@" > "$PROBE_CAPTURE_DIR/sendmail-args"
cat > "$PROBE_CAPTURE_DIR/sendmail-body"
""",
    )

    completed = run_probe("--send-test", "person@example.com", env=env)

    assert completed.returncode == 0
    assert (capture_dir / "sendmail-args").read_text(encoding="utf-8") == "-t\n-oi\n"
    body = (capture_dir / "sendmail-body").read_text(encoding="utf-8")
    assert "To: person@example.com\n" in body
    assert "Subject: [mt] direct mail probe" in body
    assert "Execution context: outside a Slurm allocation" in body
    assert str(ROOT) not in body


def test_invalid_recipient_is_rejected_before_mailer_runs(tmp_path: Path) -> None:
    env, bin_dir, capture_dir = probe_environment(tmp_path)
    write_executable(
        bin_dir / "sendmail",
        "#!/bin/sh\ntouch \"$PROBE_CAPTURE_DIR/sendmail-ran\"\n",
    )

    completed = run_probe("--send-test", "person@example.com\nBcc:other@example.com", env=env)

    assert completed.returncode == 2
    assert "invalid email address" in completed.stderr
    assert not (capture_dir / "sendmail-ran").exists()


def test_recipient_starting_with_dash_is_rejected(tmp_path: Path) -> None:
    """mailx and mail take the recipient positionally, so '-x@y.z' is an option."""

    env, bin_dir, capture_dir = probe_environment(tmp_path)
    write_executable(
        bin_dir / "sendmail",
        "#!/bin/sh\ntouch \"$PROBE_CAPTURE_DIR/sendmail-ran\"\n",
    )

    completed = run_probe("--send-test", "--debug@example.com", env=env)

    assert completed.returncode == 2
    assert "must not start with '-'" in completed.stderr
    assert not (capture_dir / "sendmail-ran").exists()


def test_submit_slurm_test_uses_small_job_and_native_mail(tmp_path: Path) -> None:
    env, bin_dir, capture_dir = probe_environment(tmp_path)
    write_executable(
        bin_dir / "sbatch",
        """#!/bin/sh
printf '%s\n' "$@" > "$PROBE_CAPTURE_DIR/sbatch-args"
printf '%s\n' '12345;iris'
""",
    )

    completed = run_probe("--submit-slurm-test", "person@example.com", env=env)

    assert completed.returncode == 0
    assert "Submitted Job ID: 12345" in completed.stdout
    args = (capture_dir / "sbatch-args").read_text(encoding="utf-8").splitlines()
    assert "--partition=interactive" in args
    assert "--qos=debug" in args
    assert "--cpus-per-task=1" in args
    assert "--mem=128M" in args
    assert "--time=00:02:00" in args
    assert "--mail-user=person@example.com" in args
    assert "--mail-type=BEGIN,END,FAIL" in args
    assert '--wrap=bash "$MT_MAIL_PROBE_SCRIPT" --send-test "$MT_MAIL_PROBE_EMAIL"' in args
