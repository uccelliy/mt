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
    cognitive_model_contract,
    make_contract,
    missing_required_columns,
    standard_behavior_contract,
    validate_dataframe,
)
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
from mt.data._reports import (
    DEFAULT_REPORT_TIMESTAMP_FORMAT,
    report_timestamp,
    save_json_report,
    save_text_report,
    timestamped_report_path,
)
from mt.data.view import (
    apply_transform,
    clear_history,
    drop_participants,
    filter_by_values,
    filter_dataframe,
    mask_columns,
    replace_values,
    split_data_by_column,
    split_data_by_participant,
    split_data_from_spec,
    split_data_kfold,
    split_values_kfold,
)


iter_valid_dataframes = iter_contract_dataframes


__all__ = [
    "DEFAULT_COLUMN_GROUPS",
    "DEFAULT_CONTRACT_LOG_DIR",
    "DEFAULT_INDEX_COLUMNS",
    "DEFAULT_REPORT_TIMESTAMP_FORMAT",
    "ColumnSpec",
    "ContractCheckReport",
    "DataContract",
    "DataRequest",
    "FilterSpec",
    "PreparedData",
    "PreparedSplit",
    "SkippedDataSource",
    "SplitSpec",
    "TransformSpec",
    "apply_transform",
    "check_data_contract",
    "clear_history",
    "cognitive_model_contract",
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
    "replace_values",
    "report_timestamp",
    "save_json_report",
    "save_text_report",
    "split_data_by_column",
    "split_data_by_participant",
    "split_data_from_spec",
    "split_data_kfold",
    "split_values_kfold",
    "standard_behavior_contract",
    "timestamped_report_path",
    "validate_dataframe",
]
