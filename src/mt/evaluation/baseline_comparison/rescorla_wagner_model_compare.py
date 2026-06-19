from mt.models import RescorlaWagnerModel
from mt.data import load_dataframe, split_data_kfold
from mt.training.trainer import Trainer

path = "hf://datasets/marcelbinz/feng2021dynamics/exp1/train-00000-of-00001.parquet"
splits_num =10
df = load_dataframe(path, list(RescorlaWagnerModel.known_required_columns()))
predictive_nll = 0
for train_df,eval_df in split_data_kfold(df, splits_num):
    print("train_df shape:", train_df.shape)
    print("eval_df shape:", eval_df.shape)
    trainer=Trainer(RescorlaWagnerModel())
    fold_loss = trainer.fit_and_evaluate(train_df, eval_df).item()
    predictive_nll += fold_loss
    
predictive_nll = predictive_nll / splits_num
