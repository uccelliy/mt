import pandas as pd
import torch
import numpy as np


def split_data_by_participant(df, num_splits=10,participant_col='participant'):
    participants = df[participant_col].unique()
    splits = np.array_split(participants, num_splits)
    return splits

def split_data_kfold(df, num_splits=10,participant_col='participant'):
    participants = df[participant_col].unique()
    splits = np.array_split(participants, num_splits)
    
    for split in splits:
        train_df = df[~df[participant_col].isin(split.tolist())].copy()
        eval_df = df[df[participant_col].isin(split.tolist())].copy()
        yield train_df, eval_df