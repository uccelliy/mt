

import schedulefree
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_parquet("hf://datasets/marcelbinz/enkavi2019digitspan/exp1/train-00000-of-00001.parquet")
uni_choice=df['ground_truth'].unique()
choice_to_idx = {c: i for i, c in enumerate(uni_choice)}
df["choice"] = df["choice"].map(choice_to_idx)
df["ground_truth"] = df["ground_truth"].map(choice_to_idx)
num_splits = 10
splits = np.array_split(df['participant'].unique(),num_splits)
predictive_nll = 0
print("num rows:", len(df))
print("num participants:", df["participant"].nunique())
print("num tasks:", df["task"].nunique())
print("choice unique:", df["choice"].unique())
print("ground truth unique:", df["ground_truth"].unique())

def pd_to_pth(df, values, keys=['participant', 'task', 'trial']):
    column_names_list = [keys + [value] for value in values]
    wide_arrs = {}
    for column_names in column_names_list:
        arr = df[column_names].values
        dims = [np.unique(arr[:, i], return_inverse=True) for i in range(len(column_names)-1)]
        wide_arr = np.full([len(dims[i][0]) for i in range(len(column_names)-1)], np.nan)
        idx = tuple(dims[i][1] for i in range(len(column_names) - 1))
        wide_arr[idx] = arr[:, -1]
        wide_arrs[column_names[-1]] = torch.from_numpy(wide_arr).reshape(-1, wide_arr.shape[-1])
    return wide_arrs

def count_nll_from_tensor(train_data, eval_data, d_c, alpha=1e-6):
    train_truth = train_data["ground_truth"].flatten().long()
    train_choice = train_data["choice"].flatten().long()

    eval_truth = eval_data["ground_truth"].flatten().long()
    eval_choice = eval_data["choice"].flatten().long()

    train_mask = train_choice != -100
    eval_mask = eval_choice != -100

    train_truth = train_truth[train_mask]
    train_choice = train_choice[train_mask]

    eval_truth = eval_truth[eval_mask]
    eval_choice = eval_choice[eval_mask]

    counts = torch.full((d_c, d_c), alpha)

    for t, c in zip(train_truth, train_choice):
        counts[t, c] += 1

    # 注意：这里用 log(counts) 就够了，因为 cross_entropy 内部会做 log_softmax
    count_logits = torch.log(counts)

    eval_logits = count_logits[eval_truth]

    return F.cross_entropy(eval_logits, eval_choice)

class RationalModel(nn.Module):
    def __init__(self,d_c=len(uni_choice)):
        super().__init__()
        self.rational_params=nn.Parameter(torch.randn((d_c,d_c)))
        self.ignore_index=-100
    def preprocess_data(self,train_df,eval_df):
        train_data = pd_to_pth(train_df, ['choice','ground_truth'])
        print(train_data)
        eval_data = pd_to_pth(eval_df, ['choice','ground_truth'])
        print(eval_data)

        train_data['choice'] = torch.nan_to_num(
            train_data['choice'],
            nan=self.ignore_index
        ).long()

        eval_data['choice'] = torch.nan_to_num(
            eval_data['choice'],
            nan=self.ignore_index
        ).long()

        train_data['ground_truth'] = torch.nan_to_num(
            train_data['ground_truth'],
            nan=0
        ).long()

        eval_data['ground_truth'] = torch.nan_to_num(
            eval_data['ground_truth'],
            nan=0
        ).long()
        return train_data,eval_data

    def forward(self,data):
        action=data['ground_truth'].long()
        return self.rational_params[action]



class Trainer:
    def __init__(self, model, num_iter=50):
        self.model = model
        self.num_iter = num_iter
        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=0.1)

    def fit_and_evaluate(self, train_df, eval_df):
        ### PREPROCESS DATA ###
        train_data, eval_data = self.model.preprocess_data(train_df, eval_df)
        count_loss = count_nll_from_tensor(
            train_data,
            eval_data,
            d_c=self.model.rational_params.shape[0]
        )

        print("count loss on same tensor:", count_loss.item())
        ### FITTING ###
        self.model.train()
        self.optimizer.train()
        for _ in tqdm.tqdm(range(self.num_iter)):
            self.optimizer.zero_grad()
            logits = self.model(train_data)
            loss = F.cross_entropy(logits.flatten(0, -2), train_data['choice'].flatten().long())
            loss.backward()
            #print(loss.item(), flush=True)
            self.optimizer.step()

        ### EVALUATION ###
        self.model.eval()
        self.optimizer.train()
        with torch.no_grad():
            logits = self.model(eval_data)
            raw_eval_loss = F.cross_entropy(
                logits.flatten(0, -2),
                eval_data['choice'].flatten().long()
            )

        self.optimizer.eval()
        with torch.no_grad():
            logits = self.model(eval_data)
            avg_eval_loss = F.cross_entropy(
                logits.flatten(0, -2),
                eval_data['choice'].flatten().long()
            )
        logits = self.model(eval_data)
        print("raw eval:", raw_eval_loss.item())
        print("avg eval:", avg_eval_loss.item())
        return F.cross_entropy(logits.flatten(0, -2), eval_data['choice'].flatten().long())
fold_losses = []

for split in splits:
    train_df = df[~df['participant'].isin(split.tolist())]
    eval_df = df[df['participant'].isin(split.tolist())]
    trainer=Trainer(RationalModel())
    fold_loss = trainer.fit_and_evaluate(train_df, eval_df).item()
    predictive_nll += trainer.fit_and_evaluate(train_df, eval_df).item()
    fold_losses.append(fold_loss)

print(predictive_nll/num_splits,flush=False)
print("fold losses:", fold_losses)
print("sum:", sum(fold_losses))
print("mean:", np.mean(fold_losses))