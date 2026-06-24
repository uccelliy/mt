"""Command-line LoRA finetuning entry point for behavioral LMs."""

from __future__ import annotations

import argparse
from pathlib import Path
import os

import torch

from mt.models.llm.finetune_artifacts import (
    TrainingArtifactCallback,
    find_latest_checkpoint,
    plot_metric_history,
)
from mt.data import (
    load_marked_text_supervision_dataset,
    make_lm_collate_fn,
)


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEFAULT_OUTPUT_DIR = "outputs/qwen25_05b_behavior_lora"
DEFAULT_DATA_SOURCE = "marcelbinz/Psych-101"


def main(argv: list[str] | None = None) -> None:
    """Run LoRA finetuning from the command line."""

    args = parse_args(argv)
    run_finetuning(args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM on behavioral trial supervision.",
    )

    data = parser.add_argument_group("data")
    data.add_argument("--data-source", default=DEFAULT_DATA_SOURCE)
    data.add_argument("--source-kind", choices=("hf", "file"), default="hf")
    data.add_argument("--dataset-config", default=None)
    data.add_argument("--data-split", default="train")
    data.add_argument("--file-pattern", default="*.csv")
    data.add_argument("--text-column", default="text")
    data.add_argument(
        "--max-samples",
        "--n-train",
        dest="max_samples",
        type=int,
        default=10000,
        help="Maximum rows to load; use 0 to disable the limit.",
    )
    data.add_argument("--eval-fraction", type=float, default=0.05)
    data.add_argument("--seq-len", type=int, default=4000)
    data.add_argument("--truncation", choices=("head", "tail"), default="head")
    data.add_argument("--preprocess-workers", type=int, default=1)

    model = parser.add_argument_group("model")
    model.add_argument("--model", default=DEFAULT_MODEL_NAME)
    model.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    model.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    model.add_argument("--dtype", choices=("auto", "float16", "float32", "bfloat16"), default="auto")
    model.add_argument("--lora-r", type=int, default=8)
    model.add_argument("--lora-alpha", type=int, default=16)
    model.add_argument("--lora-dropout", type=float, default=0.05)
    model.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    model.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    model.add_argument("--bnb-4bit-quant-type", default="nf4")
    model.add_argument(
        "--bnb-4bit-compute-dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )
    model.add_argument(
        "--bnb-4bit-use-double-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    train = parser.add_argument_group("training")
    train.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    train.add_argument("--max-steps", "--train-steps", dest="max_steps", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--eval-batch-size", type=int, default=1)
    train.add_argument("--gradient-accumulation-steps", "--gradient-acc", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--logging-steps", type=int, default=50)
    train.add_argument("--eval-steps", type=int, default=500)
    train.add_argument("--save-steps", type=int, default=300)
    train.add_argument("--save-total-limit", type=int, default=5)
    train.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="Use 'auto'/'latest' to resume the newest output checkpoint, 'none' to disable, or a path.",
    )
    train.add_argument("--report-to", default="none", help="Comma-separated HF integrations.")
    train.add_argument("--run-name", default=None)
    train.add_argument("--dataloader-workers", type=int, default=0)
    train.add_argument("--plot-steps", type=int, default=None)

    return parser.parse_args(argv)


def run_finetuning(args: argparse.Namespace) -> None:
    _set_seed(args.seed)

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Data: {args.data_source} ({args.source_kind})")
    print(f"Output: {output_dir}")
    print(f"Max steps: {args.max_steps}")
    print(f"Visible CUDA devices: {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = load_marked_text_supervision_dataset(
        args.data_source,
        tokenizer,
        source_kind=args.source_kind,
        dataset_config=args.dataset_config,
        split=args.data_split,
        file_pattern=args.file_pattern,
        text_column=args.text_column,
        max_samples=args.max_samples,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        max_length=args.seq_len,
        truncation=args.truncation,
        num_proc=args.preprocess_workers,
    )
    collate_fn = make_lm_collate_fn(tokenizer)

    print(
        "Dataset: "
        f"train={len(tokenized_dataset['train'])} "
        f"eval={len(tokenized_dataset['test'])}"
    )

    torch_dtype = _resolve_torch_dtype(args.dtype)
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.load_in_4bit:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model_kwargs["device_map"] = {"": local_rank} if torch.cuda.is_available() else None
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_resolve_torch_dtype(args.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_split_csv_arg(args.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    report_to = _split_csv_arg(args.report_to)
    if report_to == ["none"]:
        report_to = "none"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_num_workers=args.dataloader_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        report_to=report_to,
        run_name=args.run_name,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available() and torch_dtype is torch.float16,
        bf16=torch.cuda.is_available() and torch_dtype is torch.bfloat16,
        save_safetensors=True,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )

    artifact_callback = TrainingArtifactCallback(
        output_dir=output_dir,
        plot_steps=args.plot_steps or args.logging_steps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=collate_fn,
        callbacks=[artifact_callback],
    )

    resume_checkpoint = _resolve_resume_checkpoint(args.resume_from_checkpoint, output_dir)
    if resume_checkpoint is None:
        print("Starting from scratch.")
    else:
        print(f"Resuming from checkpoint: {resume_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    plot_metric_history(artifact_callback.records or trainer.state.log_history, output_dir / "plots")
    print(f"Saved model, trainer state, and training plots under {output_dir}")


def _split_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _resolve_torch_dtype(value: str) -> torch.dtype:
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    if value == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _resolve_resume_checkpoint(value: str | None, output_dir: Path) -> str | None:
    if value is None or value.lower() in {"none", "false", "0", "no"}:
        return None
    if value.lower() in {"auto", "latest", "true", "1", "yes"}:
        latest = find_latest_checkpoint(output_dir)
        return str(latest) if latest is not None else None
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    return str(path)


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
