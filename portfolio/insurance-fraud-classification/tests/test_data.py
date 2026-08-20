import pandas as pd
import pytest
from fraud_portfolio.data import audit_frame, temporal_split


def sample():
    return pd.DataFrame({
        "PolicyNumber": [1,2,3,4,5,6],
        "Year": [1994,1994,1995,1995,1996,1996],
        "FraudFound_P": [0,1,0,1,0,1],
        "Age": [20,30,40,50,60,70],
        "Make": ["A","B","A","C","A","D"],
    })


def test_audit_counts():
    report = audit_frame(sample())
    assert report["rows"] == 6
    assert report["fraud_count"] == 3
    assert report["unique_policy_numbers"] == 6


def test_temporal_split_drops_id_and_year():
    train, validation, test = temporal_split(sample())
    for X, y in (train, validation, test):
        assert "PolicyNumber" not in X
        assert "Year" not in X
        assert len(X) == len(y) == 2


def test_missing_year_fails_closed():
    with pytest.raises(ValueError):
        temporal_split(sample().query("Year != 1996"))
