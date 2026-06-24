from __future__ import annotations

from mt.utils.hardware_monitor import build_log_row, parse_nvidia_smi_csv


def test_parse_nvidia_smi_csv_maps_expected_fields() -> None:
    rows = parse_nvidia_smi_csv(
        "2026/06/24 12:00:00.000, 0, Tesla V100-SXM2-32GB, 87, 60, "
        "12000, 32768, 70, 210.5, 300.0\n"
    )

    assert rows == [
        {
            "timestamp": "2026/06/24 12:00:00.000",
            "index": "0",
            "name": "Tesla V100-SXM2-32GB",
            "utilization.gpu": "87",
            "utilization.memory": "60",
            "memory.used": "12000",
            "memory.total": "32768",
            "temperature.gpu": "70",
            "power.draw": "210.5",
            "power.limit": "300.0",
        }
    ]


def test_build_log_row_converts_gpu_metrics_to_numbers() -> None:
    row = build_log_row(
        sample_time="now",
        gpu_row={
            "timestamp": "gpu-now",
            "index": "0",
            "name": "Tesla V100-SXM2-32GB",
            "utilization.gpu": "87",
            "utilization.memory": "60",
            "memory.used": "12000",
            "memory.total": "32768",
            "temperature.gpu": "70",
            "power.draw": "210.5",
            "power.limit": "300.0",
        },
        host_metrics={"cpu_load_1m": 1.25, "cpu_load_5m": 1.1},
    )

    assert row["gpu_util_pct"] == 87.0
    assert row["memory_used_mib"] == 12000.0
    assert row["power_draw_w"] == 210.5
    assert row["cpu_load_1m"] == 1.25
