import math
import pandas as pd
import pytest

from petquant_portfolio.features import extract_numeric_value, normalize_pet_features


def test_extract_nested_bigquery_numeric():
    assert extract_numeric_value({"float": 9.55, "integer": None, "provided": "float"}) == 9.55
    assert extract_numeric_value({"float": None, "integer": 9, "provided": "integer"}) == 9.0


def test_extract_invalid_is_nan():
    assert math.isnan(extract_numeric_value({"float": None, "integer": None}))


def test_normalize_pet_features_preserves_dose():
    df = pd.DataFrame(
        {
            "glucose_mg_dl": [180],
            "uptake_time_min": [65],
            "fdg_dose_mci": [{"float": None, "integer": 9}],
            "diabetic": [False],
            "weight_kg": [None],
        }
    )
    out = normalize_pet_features(df)
    assert out.loc[0, "fdg_dose_mci"] == 9.0
    assert out.loc[0, "weight_kg_missing"] == 1


def test_missing_required_columns_rejected():
    with pytest.raises(ValueError):
        normalize_pet_features(pd.DataFrame({"glucose_mg_dl": [100]}))
