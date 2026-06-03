import numpy as np
import torch
import pandas as pd

def pd_to_pth(df, values, keys=None):
    if keys is None:
        keys = ['participant', 'task', 'trial']

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


def preprocess_rational_data(train_df, eval_df, ignore_index=-100):
    train_data = pd_to_pth(train_df, ['choice', 'ground_truth'])
    eval_data = pd_to_pth(eval_df, ['choice', 'ground_truth'])

    train_data['choice'] = torch.nan_to_num(train_data['choice'], nan=ignore_index).long()
    eval_data['choice'] = torch.nan_to_num(eval_data['choice'], nan=ignore_index).long()

    train_data['ground_truth'] = torch.nan_to_num(train_data['ground_truth'], nan=0).long()
    eval_data['ground_truth'] = torch.nan_to_num(eval_data['ground_truth'], nan=0).long()

    return train_data, eval_data


def preprocess_rescorla_wagner_data(train_df, eval_df, ignore_index=-100):
    values = ['reward', 'choice']
    if 'forced' in train_df:
        values.append('forced')

    train_data = pd_to_pth(train_df, values)
    eval_data = pd_to_pth(eval_df, values)

    train_data['choice'] = torch.nan_to_num(train_data['choice'], nan=ignore_index).long()
    eval_data['choice'] = torch.nan_to_num(eval_data['choice'], nan=ignore_index).long()

    train_data['choice_for_updating'] = train_data['choice'].clone().clamp(min=0)
    eval_data['choice_for_updating'] = eval_data['choice'].clone().clamp(min=0)

    if 'forced' in train_df:
        train_forced = torch.nan_to_num(train_data['forced'], nan=1).bool()
        eval_forced = torch.nan_to_num(eval_data['forced'], nan=1).bool()
        train_data['choice'][train_forced] = ignore_index
        eval_data['choice'][eval_forced] = ignore_index

    return train_data, eval_data


def preprocess_dual_system_data(train_df, eval_df, ignore_index=-100):
    train_data = _preprocess_two_step_df(train_df)
    eval_data = _preprocess_two_step_df(eval_df)

    train_data['choice'] = torch.nan_to_num(train_data['choice'], nan=ignore_index).long()
    eval_data['choice'] = torch.nan_to_num(eval_data['choice'], nan=ignore_index).long()

    return train_data, eval_data


def _preprocess_two_step_df(df):
    df = df.copy()
    df['choice'] = df['choice'].replace(2, -1)
    df = df.replace(-1, np.nan)

    step1_df = df[df['current_state'] == 999]
    step2_df = df[df['current_state'] != 999]

    step1_data = pd_to_pth(
        step1_df,
        ['current_state', 'reward', 'choice'],
        keys=['participant', 'trial'],
    )
    step2_data = pd_to_pth(
        step2_df,
        ['current_state', 'reward', 'choice'],
        keys=['participant', 'trial'],
    )

    return {
        key: torch.stack([step1_data[key], step2_data[key]], dim=-1)
        for key in step1_data.keys()
    }


def preprocess_dunning_kruger_data(train_df, eval_df):
    train_df = _encode_dunning_kruger_choices(train_df)
    eval_df = _encode_dunning_kruger_choices(eval_df)

    normalizer = torch.Tensor([2, 10, 1, 1,
                               1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1,
                               2, 10, 1, 1])

    train_data = {}
    num_train_participants = len(train_df.participant.unique())
    train_choices = train_df[train_df['trial'] != 24]['choice'].values.astype('float')
    train_data['choice'] = torch.from_numpy(train_choices)
    train_data['choice'] = (train_data['choice'] // normalizer.repeat(num_train_participants)).long()

    eval_data = {}
    num_eval_participants = len(eval_df.participant.unique())
    eval_choices = eval_df[eval_df['trial'] != 24]['choice'].values.astype('float')
    eval_data['choice'] = torch.from_numpy(eval_choices)
    eval_data['choice'] = (eval_data['choice'] // normalizer.repeat(num_eval_participants)).long()

    return train_data, eval_data


def _encode_dunning_kruger_choices(df):
    df = df.copy()
    for i in range(4, 24):
        trial_mask = df['trial'] == i
        df.loc[trial_mask, 'choice'] = df.loc[trial_mask, 'choice'].astype('category').cat.codes
    return df







num_splits = 10
