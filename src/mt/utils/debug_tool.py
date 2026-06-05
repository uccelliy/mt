from datasets import load_dataset
from mt.data.finetune_dataset import preprocess_example


def inspect_preprocess_labels(tokenizer, n=10):
    raw_dataset = load_dataset(
        "marcelbinz/Psych-101",
        split=f"train[:{n}]",
    )

    for i in range(n):
        example = raw_dataset[i]

        processed = preprocess_example(example, tokenizer)

        input_ids = processed["input_ids"]
        labels = processed["labels"]

        kept_label_ids = [
            x for x in labels
            if x != -100
        ]

        print("=" * 80)
        print("FULL INPUT:")
        print(tokenizer.decode(input_ids))

        print("\nLABELS KEPT:")
        print(tokenizer.decode(kept_label_ids))