import pytest

from spark_portfolio.validation import dataframe_audit, header_match_fraction, validate_dataframe_contract


class FakeRow(dict):
    def asDict(self, recursive=True):
        return dict(self)


class FakeDataFrame:
    def __init__(self, columns, rows):
        self.columns = list(columns)
        self._rows = [FakeRow(r) for r in rows]

    def count(self):
        return len(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


def test_header_match_detects_saved_emr_failure_pattern():
    columns = ["sk_id_curr", "target", "name_contract_type", "code_gender", "occupation_type"]
    row = FakeRow(
        sk_id_curr=None,
        target=None,
        name_contract_type="NAME_CONTRACT_TYPE",
        code_gender="CODE_GENDER",
        occupation_type="OCCUPATION_TYPE",
    )
    assert header_match_fraction(row, columns) == pytest.approx(1.0)


def test_audit_flags_embedded_header():
    df = FakeDataFrame(
        ["id", "name", "category"],
        [
            {"id": None, "name": "NAME", "category": "CATEGORY"},
            {"id": 1, "name": "alpha", "category": "x"},
        ],
    )
    audit = dataframe_audit(df)
    assert audit["row_count"] == 2
    assert audit["embedded_header_suspected"] is True


def test_validation_accepts_clean_contract():
    df = FakeDataFrame(
        ["sk_id_curr", "target", "name_contract_type"],
        [{"sk_id_curr": 100001, "target": 1, "name_contract_type": "Cash loans"}],
    )
    audit = validate_dataframe_contract(
        df,
        required_columns=["sk_id_curr", "target"],
        expected_row_count=1,
    )
    assert audit["embedded_header_suspected"] is False


def test_validation_rejects_count_mismatch_before_analytics():
    df = FakeDataFrame(["id"], [{"id": 1}])
    with pytest.raises(ValueError, match="unexpected row count"):
        validate_dataframe_contract(df, expected_row_count=2)
