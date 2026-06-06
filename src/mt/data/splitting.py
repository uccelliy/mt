import numpy as np

from mt.data.requests import SplitSpec
from mt.data.views import DataSplit


def split_values_kfold(values, num_splits=10, *, shuffle=False, random_state=None):
    values = np.array(list(values))
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(values)

    splits = np.array_split(values, num_splits)
    return splits


def split_data_by_participant(df, num_splits=10,participant_col='participant'):
    return split_values_kfold(df[participant_col].unique(), num_splits)


def split_data_kfold(df, num_splits=10,participant_col='participant'):
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


def split_data_from_spec(df, spec: SplitSpec):
    """Yield :class:`DataSplit` objects from a declarative split spec."""

    for index, (train_df, eval_df) in enumerate(
        split_data_by_column(
            df,
            spec.column,
            num_splits=spec.num_splits,
            shuffle=spec.shuffle,
            random_state=spec.random_state,
        )
    ):
        yield DataSplit(
            train=train_df,
            eval=eval_df,
            name=f"{spec.strategy}_{index}",
            metadata={
                "strategy": spec.strategy,
                "column": spec.column,
                "fold": index,
            },
        )
