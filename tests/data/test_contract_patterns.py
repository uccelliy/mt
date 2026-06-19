from __future__ import annotations

import pandas as pd
import pytest

from mt.data import ColumnPatternSpec, make_contract, validate_dataframe


def test_validate_dataframe_accepts_required_column_pattern() -> None:
    contract = make_contract(
        "features",
        required_columns=("choice",),
        column_patterns=(ColumnPatternSpec(name="features", pattern=r"^x\d+$"),),
    )
    df = pd.DataFrame(
        {
            "participant": [1],
            "task": [1],
            "trial": [1],
            "choice": [0],
            "x1": [0.5],
        }
    )

    assert validate_dataframe(df, contract) is df


def test_validate_dataframe_rejects_missing_required_column_pattern() -> None:
    contract = make_contract(
        "features",
        required_columns=("choice",),
        column_patterns=(ColumnPatternSpec(name="features", pattern=r"^x\d+$"),),
    )
    df = pd.DataFrame(
        {
            "participant": [1],
            "task": [1],
            "trial": [1],
            "choice": [0],
        }
    )

    with pytest.raises(KeyError, match="features"):
        validate_dataframe(df, contract)
