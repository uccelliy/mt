"""Terminal-friendly hardware monitoring helpers for training jobs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


GPU_QUERY_FIELDS = (
    "timestamp",
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "power.limit",
)

CSV_COLUMNS = (
    "sample_time",
    "gpu_timestamp",
    "gpu_index",
    "gpu_name",
    "gpu_util_pct",
    "memory_util_pct",
    "memory_used_mib",
    "memory_total_mib",
    "temperature_c",
    "power_draw_w",
    "power_limit_w",
    "cpu_load_1m",
    "cpu_load_5m",
    "cpu_load_15m",
    "host_memory_used_mib",
    "host_memory_total_mib",
)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    monitor_hardware(
        interval=args.interval,
        log_file=args.log_file,
        once=args.once,
        nvidia_smi=args.nvidia_smi,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor GPU/CPU usage during a training run.")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between samples.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional CSV log path.")
    parser.add_argument("--once", action="store_true", help="Print one sample and exit.")
    parser.add_argument("--nvidia-smi", default="nvidia-smi", help="Path to nvidia-smi.")
    return parser.parse_args(argv)


def monitor_hardware(
    *,
    interval: float = 10.0,
    log_file: Path | None = None,
    once: bool = False,
    nvidia_smi: str = "nvidia-smi",
) -> None:
    """Print and optionally log periodic GPU/CPU samples."""

    if interval <= 0:
        raise ValueError("interval must be positive.")

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _ensure_csv_header(log_file)

    while True:
        sample_time = datetime.now(timezone.utc).isoformat()
        gpu_rows = query_gpu_rows(nvidia_smi=nvidia_smi)
        host_metrics = sample_host_metrics()
        rows = [
            build_log_row(sample_time=sample_time, gpu_row=row, host_metrics=host_metrics)
            for row in gpu_rows
        ]
        print(format_status(rows, sample_time=sample_time), flush=True)
        if log_file is not None:
            append_rows(log_file, rows)
        if once:
            return
        time.sleep(interval)


def query_gpu_rows(*, nvidia_smi: str = "nvidia-smi") -> list[dict[str, str]]:
    """Return raw GPU rows from nvidia-smi query output."""

    executable = shutil.which(nvidia_smi) or nvidia_smi
    command = [
        executable,
        f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return parse_nvidia_smi_csv(completed.stdout)


def parse_nvidia_smi_csv(output: str) -> list[dict[str, str]]:
    """Parse nvidia-smi csv,noheader,nounits output into dictionaries."""

    rows: list[dict[str, str]] = []
    reader = csv.reader(line for line in output.splitlines() if line.strip())
    for values in reader:
        cleaned = [value.strip() for value in values]
        if len(cleaned) != len(GPU_QUERY_FIELDS):
            continue
        rows.append(dict(zip(GPU_QUERY_FIELDS, cleaned)))
    return rows


def sample_host_metrics() -> dict[str, float | None]:
    """Sample host load and memory without requiring optional dependencies."""

    try:
        import os

        load_1m, load_5m, load_15m = os.getloadavg()
    except (AttributeError, OSError):
        load_1m = load_5m = load_15m = None

    memory = read_linux_meminfo()
    return {
        "cpu_load_1m": load_1m,
        "cpu_load_5m": load_5m,
        "cpu_load_15m": load_15m,
        "host_memory_used_mib": memory.get("used_mib"),
        "host_memory_total_mib": memory.get("total_mib"),
    }


def read_linux_meminfo(path: str | Path = "/proc/meminfo") -> dict[str, float | None]:
    meminfo_path = Path(path)
    if not meminfo_path.exists():
        return {"used_mib": None, "total_mib": None}

    values: dict[str, float] = {}
    with meminfo_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].rstrip(":")
            try:
                values[key] = float(parts[1]) / 1024.0
            except ValueError:
                continue

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    used = total - available if total is not None and available is not None else None
    return {"used_mib": used, "total_mib": total}


def build_log_row(
    *,
    sample_time: str,
    gpu_row: dict[str, str],
    host_metrics: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "sample_time": sample_time,
        "gpu_timestamp": gpu_row.get("timestamp"),
        "gpu_index": gpu_row.get("index"),
        "gpu_name": gpu_row.get("name"),
        "gpu_util_pct": _to_float(gpu_row.get("utilization.gpu")),
        "memory_util_pct": _to_float(gpu_row.get("utilization.memory")),
        "memory_used_mib": _to_float(gpu_row.get("memory.used")),
        "memory_total_mib": _to_float(gpu_row.get("memory.total")),
        "temperature_c": _to_float(gpu_row.get("temperature.gpu")),
        "power_draw_w": _to_float(gpu_row.get("power.draw")),
        "power_limit_w": _to_float(gpu_row.get("power.limit")),
        **host_metrics,
    }


def format_status(rows: list[dict[str, Any]], *, sample_time: str) -> str:
    if not rows:
        return f"[{sample_time}] nvidia-smi unavailable or no GPUs visible."

    lines = [
        f"[{sample_time}] GPU  util  mem_used/total MiB  tempC  powerW  load1m",
    ]
    for row in rows:
        lines.append(
            "{idx:>3} {util:>5.0f}% {used:>8.0f}/{total:<8.0f} {temp:>5.0f} {power:>7.1f} {load:>6}".format(
                idx=row.get("gpu_index") or "?",
                util=row.get("gpu_util_pct") or 0.0,
                used=row.get("memory_used_mib") or 0.0,
                total=row.get("memory_total_mib") or 0.0,
                temp=row.get("temperature_c") or 0.0,
                power=row.get("power_draw_w") or 0.0,
                load=_format_optional_float(row.get("cpu_load_1m")),
            )
        )
    return "\n".join(lines)


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writerows(rows)


def _ensure_csv_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def _to_float(value: str | None) -> float | None:
    if value is None or value in {"", "[N/A]", "N/A"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "NA"
