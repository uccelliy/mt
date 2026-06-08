"""Public API for data contracts, preparation, and validation."""

from mt.data._checking import (
    DEFAULT_CONTRACT_LOG_DIR,
    ContractCheckReport,
    SkippedDataSource,
    check_data_contract,
)
from mt.data._contracts import (
    DEFAULT_COLUMN_GROUPS,
    DEFAULT_INDEX_COLUMNS,
    ColumnSpec,
    DataContract,
    TensorSpec,
    cognitive_model_contract,
    make_contract,
    missing_required_columns,
    standard_behavior_contract,
    validate_dataframe,
)
from mt.data._filtering import drop_participants, filter_by_values, filter_dataframe
from mt.data._loading import (
    iter_contract_dataframes,
    iter_data_directory,
    load_dataframe,
    load_hf_dataset,
)
from mt.data._prepared import PreparedData, PreparedSplit
from mt.data._preparation import (
    prepare_data_view,
    prepare_dataframe,
    prepare_tensor_data_view,
    prepare_tensors,
)
from mt.data._requests import (
    DataRequest,
    FilterSpec,
    SplitSpec,
    TransformSpec,
    held_out_participant_split,
    held_out_task_split,
    held_out_trial_split,
)
from mt.data._splitting import (
    split_data_by_column,
    split_data_by_participant,
    split_data_from_spec,
    split_data_kfold,
    split_values_kfold,
)
from mt.data._tensors import dataframe_pair_to_tensors, dataframe_to_tensors
from mt.data._transforms import apply_transform, clear_history, mask_columns, replace_values


iter_valid_dataframes = iter_contract_dataframes


__all__ = [
    "DEFAULT_COLUMN_GROUPS",
    "DEFAULT_CONTRACT_LOG_DIR",
    "DEFAULT_INDEX_COLUMNS",
    "ColumnSpec",
    "ContractCheckReport",
    "DataContract",
    "DataRequest",
    "FilterSpec",
    "PreparedData",
    "PreparedSplit",
    "SkippedDataSource",
    "SplitSpec",
    "TensorSpec",
    "TransformSpec",
    "apply_transform",
    "check_data_contract",
    "clear_history",
    "cognitive_model_contract",
    "dataframe_pair_to_tensors",
    "dataframe_to_tensors",
    "drop_participants",
    "filter_by_values",
    "filter_dataframe",
    "held_out_participant_split",
    "held_out_task_split",
    "held_out_trial_split",
    "iter_contract_dataframes",
    "iter_data_directory",
    "iter_valid_dataframes",
    "load_dataframe",
    "load_hf_dataset",
    "make_contract",
    "mask_columns",
    "missing_required_columns",
    "prepare_data_view",
    "prepare_dataframe",
    "prepare_tensor_data_view",
    "prepare_tensors",
    "replace_values",
    "split_data_by_column",
    "split_data_by_participant",
    "split_data_from_spec",
    "split_data_kfold",
    "split_values_kfold",
    "standard_behavior_contract",
    "validate_dataframe",
]
