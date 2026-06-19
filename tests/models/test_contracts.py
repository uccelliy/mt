from __future__ import annotations

from mt.models import get_model_class, list_model_names
from mt.models.common import MODEL_TENSOR_COLUMNS, model_data_spec


def test_every_registered_model_has_an_explicit_data_inventory() -> None:
    assert set(MODEL_TENSOR_COLUMNS) == set(list_model_names())


def test_dataframe_requirements_live_in_model_contracts() -> None:
    expected_required_columns = {
        "DualSystemsModel": ("choice", "current_state", "reward"),
        "DunningKruger": ("choice",),
        "GeneralizedContextModel": ("choice", "ground_truth"),
        "RationalModel": ("choice", "ground_truth"),
        "RescorlaWagnerModel": ("choice", "reward"),
    }

    for model_name, expected in expected_required_columns.items():
        model_class = get_model_class(model_name)
        assert "required_columns" not in model_class.__dict__
        assert model_data_spec(model_name).required_columns == expected


def test_noise_ceiling_dynamic_uid_remains_explicitly_unmapped() -> None:
    assert model_data_spec("NoiseCeiling").unmapped_tensor_keys == ("uid",)


def test_generalized_context_contract_declares_feature_column_pattern() -> None:
    spec = model_data_spec("GeneralizedContextModel")

    assert spec.column_patterns == {"features": r"^x\d+$"}
    assert spec.unmapped_tensor_keys == ()
