import pytest

from spark_portfolio.catalog import quote_catalog_identifier, resolve_existing_table


class FakeCatalog:
    def __init__(self, existing):
        self.existing = set(existing)

    def tableExists(self, table, db=None):
        key = (db, table) if db is not None else tuple(table.split(".", 1))
        return key in self.existing


class FakeSpark:
    def __init__(self, existing):
        self.catalog = FakeCatalog(existing)


def test_resolver_skips_missing_tutorial_table_and_finds_corrected_table():
    spark = FakeSpark({("hcdb", "hc_applications")})
    assert resolve_existing_table(
        spark, "hcdb", ["hc_application_csv", "hc_applications"]
    ) == "hc_applications"


def test_resolver_fails_when_no_approved_table_exists():
    with pytest.raises(LookupError):
        resolve_existing_table(FakeSpark(set()), "hcdb", ["a", "b"])


def test_identifier_rejects_sql_metacharacters():
    assert quote_catalog_identifier("hc_applications") == "`hc_applications`"
    with pytest.raises(ValueError):
        quote_catalog_identifier("hc_applications; drop table x")
