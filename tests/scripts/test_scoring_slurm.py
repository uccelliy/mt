"""Offline contracts for the Track P Slurm launchers."""

from __future__ import annotations

from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]


def test_track_p_launchers_pass_run_directories():
    expected = {
        "scripts/score_model.slurm": 2,
        "scripts/smoke_e0_e3.slurm": 1,
        "scripts/e0_e3_minitaur.slurm": 1,
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
        "wait_for_pid \"$worker\"\n"
        "wait \"$signaler\"\n",
        encoding="utf-8",
    )
    result = subprocess.run(["bash", str(probe)], capture_output=True,
                            text=True, timeout=2, check=False)
    assert result.returncode == 0, result.stderr
