#!/usr/bin/env python3
"""
数据统计工具：计算训练数据中的实际token数
用于评估训练时间和成本
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

def count_tokens(dataset_name, tokenizer_name, num_samples=10000, seq_len=4096):
    """统计数据集中的token数"""
    print(f"Loading dataset: {dataset_name}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Max sequence length: {seq_len}")
    print()
    
    # 加载数据集
    dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    print(f"Total samples in split: {len(dataset)}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 统计tokens
    total_tokens = 0
    sample_tokens_list = []
    
    print("\nTokenizing samples...")
    for i, sample in enumerate(dataset):
        if i % 1000 == 0:
            print(f"  Processed: {i}/{len(dataset)}")
        
        # 假设数据格式为 {"text": "..."}，根据你的数据调整
        text = sample.get("text", str(sample))
        
        # 分词
        tokens = tokenizer(text, max_length=seq_len, truncation=True, return_tensors="pt")
        num_tokens = tokens["input_ids"].shape[1]
        
        total_tokens += num_tokens
        sample_tokens_list.append(num_tokens)
    
    # 统计
    avg_tokens = total_tokens / len(dataset) if dataset else 0
    min_tokens = min(sample_tokens_list) if sample_tokens_list else 0
    max_tokens = max(sample_tokens_list) if sample_tokens_list else 0
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {len(dataset):,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per sample: {avg_tokens:.0f}")
    print(f"Min tokens: {min_tokens:,}")
    print(f"Max tokens: {max_tokens:,}")
    print()
    
    # 时间预估
    print("TIME ESTIMATION (on 4×V100 SXM2)")
    print("-" * 60)
    
    # 假设不同的吞吐量
    throughputs = {
        "optimistic (1500 tok/s)": 1500,
        "moderate (1000 tok/s)": 1000,
        "conservative (500 tok/s)": 500,
    }
    
    for desc, tok_per_sec in throughputs.items():
        seconds = total_tokens / tok_per_sec
        hours = seconds / 3600
        days = hours / 24
        print(f"{desc:.<30} {hours:>6.1f} hours ({days:>4.2f} days)")
    
    print("="*60)
    
    return total_tokens, avg_tokens

def main():
    parser = argparse.ArgumentParser(description="Count tokens in training dataset")
    parser.add_argument("--dataset", type=str, default="marcelbinz/Psych-101", 
                       help="HuggingFace dataset name")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Tokenizer name or path")
    parser.add_argument("--num-samples", type=int, default=10000,
                       help="Number of samples to analyze")
    parser.add_argument("--seq-len", type=int, default=4096,
                       help="Max sequence length")
    
    args = parser.parse_args()
    
    count_tokens(args.dataset, args.tokenizer, args.num_samples, args.seq_len)

if __name__ == "__main__":
    main()
