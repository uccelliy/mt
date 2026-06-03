import tqdm
import numpy as np
import pandas as pd
from mt.models import Trainer
from mt.models import RationalModel
from mt.data.loading import load_dataframe


df,splits = load_dataframe("hf://datasets/marcelbinz/enkavi2019digitspan/exp1/train-00000-of-00001.parquet"
                    ,10,RationalModel().required_columns)

uni_choice=df['choice'].unique()
choice_to_idx = {c: i for i, c in enumerate(uni_choice)}
df["choice"] = df["choice"].map(choice_to_idx)
df["ground_truth"] = df["ground_truth"].map(choice_to_idx)
predictive_nll = 0
# print("num rows:", len(df))
# print("num participants:", df["participant"].nunique())
# print("num tasks:", df["task"].nunique())
# print("choice unique:", df["choice"].unique())
# print("ground truth unique:", df["ground_truth"].unique())

fold_losses = []
for split in splits:
    train_df = df[~df['participant'].isin(split.tolist())]
    eval_df = df[df['participant'].isin(split.tolist())]
    trainer=Trainer(RationalModel())
    fold_loss = trainer.fit_and_evaluate(train_df, eval_df).item()
    predictive_nll += trainer.fit_and_evaluate(train_df, eval_df).item()
    fold_losses.append(fold_loss)

print(predictive_nll/10,flush=False)
print("fold losses:", fold_losses)
print("sum:", sum(fold_losses))
print("mean:", np.mean(fold_losses))
