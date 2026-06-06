
from mt.data.finetune_dataset import preprocess_example, preprocess_input, has_supervised_token, make_collate_fn

import torch
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

def main():
    device = get_device()
    print("device:", device)

    tokenizer = get_tokenizer(MODEL_NAME)
    #text_input = ccs_class.input_map2text(ccs_class.symptoms)
    #print(text_input)
    #tokenized_dataset = get_data_ccs(text_input, tokenizer, seed)
    tokenized_dataset = get_data(N_TRAIN,tokenizer,seed,SEQ_LEN)
    collate_fn = make_collate_fn(tokenizer)

    #inspect_preprocess_labels(tokenizer, n=10)


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "mps" else torch.float32,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

   

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

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)
    # collate_fn = make_collate_fn(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=train_steps,
        logging_steps=10,
        eval_steps=10,
        save_steps=300,
        save_total_limit=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        dataloader_pin_memory=False,

        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=50,

        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",

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

    trainer.train()

    plot_trainer_logs(trainer)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

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
