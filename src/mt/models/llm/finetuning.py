
from experiments.llm.finetune_dataset import preprocess_example, preprocess_input, has_supervised_token, make_collate_fn

import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./outputs/qwen25_05b_centaur_lora_524"
PIC_PATH = './outputs/qwen25_05b_centaur_lora_524/pic'
train_steps=300
N_TRAIN = 10000
SEQ_LEN=4000
seed=42


def parse_args():
    """解析命令行参数，支持断点续训"""
    parser = argparse.ArgumentParser(description="LLM LoRA Fine-tuning with checkpoint resuming")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name or path")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--train-steps", type=int, default=train_steps, help="Total training steps")
    parser.add_argument("--n-train", type=int, default=N_TRAIN, help="Number of training samples")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-acc", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save-steps", type=int, default=300, help="Save checkpoint every N steps")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, 
                       help="Path to checkpoint to resume from, or 'latest' for automatic detection")
    parser.add_argument("--seed", type=int, default=seed, help="Random seed")
    return parser.parse_args()

def main():
    """主函数：支持参数化和断点续训"""
    args = parse_args()
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Total steps: {args.train_steps}")

    tokenizer = get_tokenizer(args.model)
    
    # 加载数据
    print(f"Loading {args.n_train} training samples...")
    tokenized_dataset = get_data(args.n_train, tokenizer, args.seed, args.seq_len)
    collate_fn = make_collate_fn(tokenizer)
    
    # 统计数据大小
    train_samples = len(tokenized_dataset["train"])
    train_tokens = sum(len(x.get("input_ids", [])) for x in tokenized_dataset["train"])
    print(f"Training samples: {train_samples}, Total tokens: {train_tokens:,}")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.train_steps,
        logging_steps=50,
        eval_steps=500,
        save_steps=args.save_steps,
        save_total_limit=5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_acc,
        dataloader_pin_memory=False,

        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,

        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=False,  # 不加载最佳模型，方便断点续训

        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=collate_fn,
    )

    # 从检查点恢复或开始新训练
    print("\n=== Starting Training ===")
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint == "latest":
        # 自动检测最新检查点
        resume_checkpoint = None  # Trainer会自动查找
        print("Looking for latest checkpoint to resume...")
    elif resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 保存最终模型
    print("\n=== Saving Final Model ===")
    plot_trainer_logs(trainer)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_data_ccs(text, tokenizer, seed):
    if isinstance(text, str):
        texts = [text]
    else:
        texts = list(text)

    raw_dataset = Dataset.from_list([{"text": item} for item in texts])

    if len(raw_dataset) >= 2:
        split_dataset = raw_dataset.train_test_split(
            test_size=0.05,
            seed=seed,
        )
    else:
        split_dataset = DatasetDict({
            "train": raw_dataset,
            "test": raw_dataset,
        })

    tokenized_dataset = split_dataset.map(
        lambda x: preprocess_input(x["text"], tokenizer, SEQ_LEN),
        remove_columns=split_dataset["train"].column_names,
        num_proc=1,
    )

    tokenized_dataset = tokenized_dataset.filter(has_supervised_token)
    return tokenized_dataset

def get_data(num,tokenizer,seed,SEQ_LEN):
    raw_dataset = load_dataset(
        "marcelbinz/Psych-101",
        split=f"train[:{num}]",
    )
    split_dataset = raw_dataset.train_test_split(
        test_size=0.05,
        seed=seed,
    )
    tokenized_dataset = split_dataset.map(
        lambda x: preprocess_example(x, tokenizer,SEQ_LEN),
        remove_columns=split_dataset["train"].column_names,
        num_proc=1,
    )

    tokenized_dataset = tokenized_dataset.filter(has_supervised_token)
    return tokenized_dataset

def plot_trainer_logs(trainer, save_path=PIC_PATH):
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    import os
    save_dir = save_path
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(trainer.state.log_history)

    print(df.columns)
    print(df.tail())

    train_df = df[df["loss"].notna()] if "loss" in df.columns else pd.DataFrame()
    eval_df = df[df["eval_loss"].notna()] if "eval_loss" in df.columns else pd.DataFrame()

    plt.figure()
    if not train_df.empty:
        plt.plot(train_df["step"], train_df["loss"], label="Train Loss")
    if not eval_df.empty:
        plt.plot(eval_df["step"], eval_df["eval_loss"], label="Eval Loss")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir,"loss.png"), dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure()
    if not train_df.empty:
        train_ppl = train_df["loss"].apply(lambda x: math.exp(x) if x < 20 else float("inf"))
        plt.plot(train_df["step"], train_ppl, label="Train Perplexity")
    if not eval_df.empty:
        eval_ppl = eval_df["eval_loss"].apply(lambda x: math.exp(x) if x < 20 else float("inf"))
        plt.plot(eval_df["step"], eval_ppl, label="Eval Perplexity")

    plt.xlabel("Step")
    plt.ylabel("Perplexity")
    plt.title("Training and Evaluation Perplexity")
    plt.legend()
    plt.savefig(os.path.join(save_dir,"perplexity.png"), dpi=300, bbox_inches="tight")
    plt.show()

    if "learning_rate" in df.columns:
        lr_df = df[df["learning_rate"].notna()]
        if not lr_df.empty:
            plt.figure()
            plt.plot(lr_df["step"], lr_df["learning_rate"], label="Learning Rate")
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.legend()
            plt.savefig(os.path.join(save_dir,"lr.png"), dpi=300, bbox_inches="tight")
            plt.show()

            
if __name__ == "__main__":
    main()
