from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 导入 LLM 微调函数（如果需要用 LLM 微调）
# from mt.models.llm.finetuning import main

from mt.cli.train import main  # noqa: E402


if __name__ == "__main__":
    main()
