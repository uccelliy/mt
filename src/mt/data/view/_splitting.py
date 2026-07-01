import numpy as np

def split_values_kfold(values, num_splits=10, *, shuffle=False,
                       random_state=None):
    values = np.array(list(values))
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(values)

    splits = np.array_split(values, num_splits)
    return splits

def split_data_by_participant(df, num_splits=10,
                              participant_col="participant"):
    return split_values_kfold(df[participant_col].unique(), num_splits)

def split_data_kfold(df, num_splits=10, participant_col="participant"):
    yield from split_data_by_column(df, participant_col, num_splits=num_splits)

def split_data_by_column(
    df,
    column,
    *,
    num_splits=10,
    shuffle=False,
    random_state=None,
):
    if column not in df.columns:
        raise KeyError(f"Missing split column: {column}")

    splits = split_values_kfold(
        df[column].unique(),
        num_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    for split in splits:
        train_df = df[~df[column].isin(split.tolist())].copy()
        eval_df = df[df[column].isin(split.tolist())].copy()
        yield train_df, eval_df
