import re

import numpy as np
import torch


_FEATURE_RE = re.compile(r"^x(\d+)$")
_DEFAULT_INDEX_COLS = ["participant", "task", "trial"]


def df_to_tensors(df, values_cols, index_cols=None, *, fill_values=None):
    """Convert long-format trial data to model-ready tensors.

    `index_cols` must uniquely identify each row. By default this means one row
    per `(participant, task, trial)`. The last index column is kept as the tensor's
    second dimension, and all earlier index columns are flattened into the first
    dimension. For the default index columns, the output shape is
    `(participant-task sequences, trials)`.

    Do not remove `task` from `index_cols` unless the remaining columns still
    uniquely identify rows, usually by using a non-repeating `global_trial`.

    Args:
        df: The input dataframe.
        values_cols: Columns to convert to tensors. These are the values needed by
            models to compute logits and losses.
        index_cols: Columns used to place rows into the wide tensor. Defaults to
            `["participant", "task", "trial"]`.
        fill_values: Optional mapping from value column to the padding value used
            for missing index combinations. Columns not in the mapping use NaN.

    Returns:
        A dictionary mapping each value column to a tensor.
    """
    if index_cols is None:
        index_cols = _DEFAULT_INDEX_COLS
    else:
        index_cols = list(index_cols)

    duplicate_mask = df.duplicated(subset=index_cols, keep=False)
    if duplicate_mask.any():
        duplicate_examples = (
            df.loc[duplicate_mask, index_cols].drop_duplicates().head(5).to_dict("records")
        )
        raise ValueError(
            "index_cols must uniquely identify each row. "
            f"Found duplicate {index_cols} combinations, for example: {duplicate_examples}"
        )

    fill_values = fill_values or {}
    tensors = {}
    for value in values_cols:
        selected_cols = [*index_cols, value]
        tmp_values = df[selected_cols].values
        index_encoding = tuple(
            np.unique(tmp_values[:, i], return_inverse=True) for i in range(len(index_cols))
        )
        wide_shape = [len(unique_values) for unique_values, _ in index_encoding]
        fill_value = fill_values.get(value, np.nan)
        wide_values = np.full(wide_shape, fill_value)
        wide_idx = tuple(inverse_indices for _, inverse_indices in index_encoding)
        wide_values[wide_idx] = tmp_values[:, -1]
        tensors[value] = torch.from_numpy(wide_values).reshape(-1, wide_values.shape[-1])

    return tensors


def _df_pair_to_tensors(train_df, eval_df, values_cols, index_cols=None, *, fill_values=None):
    return (
        df_to_tensors(train_df, values_cols, index_cols=index_cols, fill_values=fill_values),
        df_to_tensors(eval_df, values_cols, index_cols=index_cols, fill_values=fill_values),
    )


def _fill_nan_with_long(data, key, fill_value):
    data[key] = torch.nan_to_num(data[key], nan=fill_value).long()


def _fill_train_eval_nan_with_long(train_data, eval_data, key, fill_value):
    _fill_nan_with_long(train_data, key, fill_value)
    _fill_nan_with_long(eval_data, key, fill_value)


def _value_to_index(values):
    return {value: idx for idx, value in enumerate(values)}


def _encode_columns(df, columns, value_to_idx):
    df = df.copy()
    for column in columns:
        df[column] = df[column].map(value_to_idx)
    return df


def _encode_train_eval_columns(train_df, eval_df, columns, value_to_idx):
    return (
        _encode_columns(train_df, columns, value_to_idx),
        _encode_columns(eval_df, columns, value_to_idx),
    )


def preprocess_rational_data(train_df, eval_df, ignore_index=-100):
    values = sorted(train_df["ground_truth"].unique())
    value_to_idx = _value_to_index(values)
    train_df, eval_df = _encode_train_eval_columns(
        train_df,
        eval_df,
        ["ground_truth", "choice"],
        value_to_idx,
    )

    train_data, eval_data = _df_pair_to_tensors(train_df, eval_df, ["choice", "ground_truth"])
    _fill_train_eval_nan_with_long(train_data, eval_data, "choice", ignore_index)
    _fill_train_eval_nan_with_long(train_data, eval_data, "ground_truth", 0)

    return train_data, eval_data


def preprocess_rescorla_wagner_data(train_df, eval_df, ignore_index=-100):
    values = ["reward", "choice"]
    if "forced" in train_df:
        values.append("forced")

    train_data, eval_data = _df_pair_to_tensors(train_df, eval_df, values)
    _fill_train_eval_nan_with_long(train_data, eval_data, "choice", ignore_index)

    train_data["choice_for_updating"] = train_data["choice"].clone().clamp(min=0)
    eval_data["choice_for_updating"] = eval_data["choice"].clone().clamp(min=0)

    if "forced" in train_df:
        train_forced = torch.nan_to_num(train_data["forced"], nan=1).bool()
        eval_forced = torch.nan_to_num(eval_data["forced"], nan=1).bool()
        train_data["choice"][train_forced] = ignore_index
        eval_data["choice"][eval_forced] = ignore_index

    return train_data, eval_data


def preprocess_dual_system_data(train_df, eval_df, ignore_index=-100):
    train_data = _preprocess_two_step_df(train_df)
    eval_data = _preprocess_two_step_df(eval_df)

    _fill_train_eval_nan_with_long(train_data, eval_data, "choice", ignore_index)

    return train_data, eval_data


def _preprocess_two_step_df(df):
    df = df.copy()
    df["choice"] = df["choice"].replace(2, -1)
    df = df.replace(-1, np.nan)

    step1_df = df[df["current_state"] == 999]
    step2_df = df[df["current_state"] != 999]

    values = ["current_state", "reward", "choice"]
    index_cols = ["participant", "trial"]
    step1_data = df_to_tensors(step1_df, values, index_cols=index_cols)
    step2_data = df_to_tensors(step2_df, values, index_cols=index_cols)

    return {
        key: torch.stack([step1_data[key], step2_data[key]], dim=-1) for key in step1_data.keys()
    }


def preprocess_dunning_kruger_data(train_df, eval_df):
    train_df = _encode_dunning_kruger_choices(train_df)
    eval_df = _encode_dunning_kruger_choices(eval_df)

    normalizer = torch.Tensor(
        [2, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 10, 1, 1]
    )

    train_data = {"choice": _dunning_kruger_choice_tensor(train_df, normalizer)}
    eval_data = {"choice": _dunning_kruger_choice_tensor(eval_df, normalizer)}

    return train_data, eval_data


def _dunning_kruger_choice_tensor(df, normalizer):
    num_participants = len(df.participant.unique())
    choices = df[df["trial"] != 24]["choice"].values.astype("float")
    choice_tensor = torch.from_numpy(choices)
    return (choice_tensor // normalizer.repeat(num_participants)).long()


def _encode_dunning_kruger_choices(df):
    df = df.copy()
    for i in range(4, 24):
        trial_mask = df["trial"] == i
        df.loc[trial_mask, "choice"] = df.loc[trial_mask, "choice"].astype("category").cat.codes
    return df


def _generalized_context_feature_columns(df):
    columns = []
    for column in df.columns:
        match = _FEATURE_RE.match(str(column))
        if match:
            columns.append((int(match.group(1)), column))

    if not columns:
        raise ValueError("No feature columns found. Expected columns of the form x1, x2, ...")
    return [column for _, column in sorted(columns)]


def preprocess_generalized_context_data(train_df, eval_df, ignore_index=-100):
    feature_columns = _generalized_context_feature_columns(train_df)
    if feature_columns != _generalized_context_feature_columns(eval_df):
        raise ValueError("Train and eval data have different feature columns.")

    train_df = train_df.copy()
    eval_df = eval_df.copy()
    class_values = sorted(
        np.unique(
            np.concatenate(
                [
                    train_df["ground_truth"].to_numpy(),
                    train_df["choice"].to_numpy(),
                    eval_df["ground_truth"].to_numpy(),
                    eval_df["choice"].to_numpy(),
                ]
            )
        ).tolist()
    )
    class_to_idx = _value_to_index(class_values)
    train_df, eval_df = _encode_train_eval_columns(
        train_df,
        eval_df,
        ["ground_truth", "choice"],
        class_to_idx,
    )

    train_data = _generalized_context_dataframe_to_tensors(
        train_df,
        feature_columns,
        num_classes=len(class_values),
        ignore_index=ignore_index,
    )
    eval_data = _generalized_context_dataframe_to_tensors(
        eval_df,
        feature_columns,
        num_classes=len(class_values),
        ignore_index=ignore_index,
    )
    return train_data, eval_data


def _generalized_context_dataframe_to_tensors(
    df,
    feature_columns,
    *,
    num_classes,
    ignore_index,
):
    feature_fill_values = {column: 0.0 for column in feature_columns}
    data = df_to_tensors(
        df,
        ["choice", "ground_truth", *feature_columns],
        fill_values=feature_fill_values,
    )
    _fill_nan_with_long(data, "choice", ignore_index)
    _fill_nan_with_long(data, "ground_truth", ignore_index)
    data["features"] = _stack_feature_columns(data, feature_columns)
    data["num_classes"] = num_classes
    return data


def _stack_feature_columns(data, feature_columns):
    return torch.stack(
        [data.pop(column).float() for column in feature_columns],
        dim=-1,
    )
