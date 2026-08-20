import numpy as np
import pandas as pd
import pytest

from pima_diabetes_portfolio.data import prepare_pima_frame, ZeroAsMissingTransformer


def frame():
    return pd.DataFrame({
        "preg":[0,2], "plas":[0,120], "pres":[70,0], "skin":[0,22],
        "insu":[0,80], "mass":[0,31.2], "pedi":[0.2,0.5], "age":[21,45],
        "class":["tested_negative","tested_positive"],
    })


def test_prepare_schema_and_target():
    X, y = prepare_pima_frame(frame())
    assert X.columns.tolist() == ["preg","plas","pres","skin","insu","mass","pedi","age"]
    assert y.tolist() == [0,1]


def test_zero_transform_only_clinical_missing_measurements():
    X, _ = prepare_pima_frame(frame())
    transformed = ZeroAsMissingTransformer().fit_transform(X)
    assert transformed.loc[0, "preg"] == 0
    assert transformed.loc[0, "age"] == 21
    assert np.isnan(transformed.loc[0, "plas"])
    assert np.isnan(transformed.loc[0, "skin"])
    assert np.isnan(transformed.loc[0, "mass"])


def test_missing_required_column_rejected():
    with pytest.raises(ValueError, match="missing required"):
        prepare_pima_frame(frame().drop(columns=["plas"]))
