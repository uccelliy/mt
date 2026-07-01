"""Public API for loading and adapting cognitive trial data."""

from mt.data._adapter import AdaptationResult, DataAdapter
from mt.data._collection import TrialCollection
from mt.data._field_registry import (
    CANONICAL_COORDINATES,
    CANONICAL_PATHS,
    CANONICAL_SLOTS,
    FIELD_REGISTRY,
    REQUIRED_PATHS,
    FieldSpec,
    get_field_spec,
    is_registered_path,
)
from mt.data._loading import DataSource, load
from mt.data._mapping import ColumnMapping, MappingResolution
from mt.data._llm_supervision import (
    find_target_spans,
    format_record_as_marked_text,
    format_supervised_example,
    has_supervised_token,
    load_marked_text_supervision_dataset,
    load_raw_supervision_dataset,
    load_supervision_dataset,
    make_lm_collate_fn,
    preprocess_marked_text,
    preprocess_supervised_example,
    split_train_eval,
    truncate_tokenized_example,
)
from mt.data.view import (
    clear_history,
    mask_columns,
    replace_values,
    split_data_by_column,
    split_data_by_participant,
    split_data_kfold,
    split_values_kfold,
)


__all__ = [
    "AdaptationResult",
    "CANONICAL_COORDINATES",
    "CANONICAL_PATHS",
    "CANONICAL_SLOTS",
    "ColumnMapping",
    "DataAdapter",
    "DataSource",
    "FIELD_REGISTRY",
    "FieldSpec",
    "MappingResolution",
    "REQUIRED_PATHS",
    "TrialCollection",
    "clear_history",
    "find_target_spans",
    "format_record_as_marked_text",
    "format_supervised_example",
    "get_field_spec",
    "has_supervised_token",
    "is_registered_path",
    "load",
    "load_marked_text_supervision_dataset",
    "load_raw_supervision_dataset",
    "load_supervision_dataset",
    "make_lm_collate_fn",
    "mask_columns",
    "preprocess_marked_text",
    "preprocess_supervised_example",
    "replace_values",
    "split_data_by_column",
    "split_data_by_participant",
    "split_data_kfold",
    "split_train_eval",
    "split_values_kfold",
    "truncate_tokenized_example",
]
