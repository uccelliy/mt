from __future__ import annotations

from mt.models.llm.finetune_artifacts import checkpoint_step, find_latest_checkpoint


def test_checkpoint_step_parses_transformers_checkpoint_names() -> None:
    assert checkpoint_step("checkpoint-300") == 300
    assert checkpoint_step("not-a-checkpoint") is None


def test_find_latest_checkpoint_uses_largest_step(tmp_path) -> None:
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-200").mkdir()
    (tmp_path / "notes").mkdir()

    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint-200"
