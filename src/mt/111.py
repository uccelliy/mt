import pandas as pd
import numpy as np
import torch
df = pd.read_parquet("hf://datasets/marcelbinz/enkavi2019digitspan/exp1/train-00000-of-00001.parquet")
uni_choice=df['ground_truth'].unique()
choice_to_idx = {c: i for i, c in enumerate(uni_choice)}
df["choice"] = df["choice"].map(choice_to_idx)
df["ground_truth"] = df["ground_truth"].map(choice_to_idx)
num_splits = 10
splits = np.array_split(df['participant'].unique(),num_splits)
predictive_nll = 0
def count_rational_nll(train_df, eval_df, d_c, alpha=1e-3):
    truth_train = torch.tensor(train_df["ground_truth"].values, dtype=torch.long)
    choice_train = torch.tensor(train_df["choice"].values, dtype=torch.long)

    counts = torch.full((d_c, d_c), alpha)

    for t, c in zip(truth_train, choice_train):
        counts[t, c] += 1

    probs = counts / counts.sum(dim=-1, keepdim=True)

    truth_eval = torch.tensor(eval_df["ground_truth"].values, dtype=torch.long)
    choice_eval = torch.tensor(eval_df["choice"].values, dtype=torch.long)

    nll = -torch.log(probs[truth_eval, choice_eval]).mean()
    return nll.item()

count_nll = 0

for split in splits:
    train_df = df[~df["participant"].isin(split.tolist())]
    eval_df = df[df["participant"].isin(split.tolist())]

    count_nll += count_rational_nll(train_df, eval_df, d_c=len(uni_choice))

count_nll = count_nll / num_splits
print(count_nll)


x = torch.tensor([float("nan")])
print(x)
print(x.long())