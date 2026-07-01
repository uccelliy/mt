import numpy as np
from mt.models import RationalModel
from mt.data import load, split_data_kfold
from mt.training.trainer import Trainer

path = "hf://datasets/marcelbinz/enkavi2019digitspan/exp1/train-00000-of-00001.parquet"
df = load(path)

uni_choice=df['choice'].unique()

predictive_nll = 0


print("num rows:", len(df))
print("num participants:", df["participant"].nunique())
print("num tasks:", df["task"].nunique())
print("choice unique:", df["choice"].unique())
print("ground truth unique:", df["ground_truth"].unique())

fold_losses = []
for train_df, eval_df in split_data_kfold(df, 10):
    trainer=Trainer(RationalModel(uni_choice.shape[0]))
    fold_loss = trainer.fit_and_evaluate(train_df, eval_df).item()
    predictive_nll += trainer.fit_and_evaluate(train_df, eval_df).item()
    fold_losses.append(fold_loss)

print(predictive_nll/10,flush=False)
print("fold losses:", fold_losses)
print("sum:", sum(fold_losses))
print("mean:", np.mean(fold_losses))
