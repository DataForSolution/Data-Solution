"""Safe catalog-table selection using Spark-compatible catalog interfaces."""
from __future__ import annotations

import re
from collections.abc import Iterable

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_catalog_identifier(value: str) -> str:
    """Validate a simple catalog identifier and return Spark SQL backtick quoting."""
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"unsafe catalog identifier: {value!r}")
    return f"`{value}`"


def _table_exists(catalog, database: str, table: str) -> bool:
    """Support current and older Spark tableExists signatures without guessing SQL."""
    try:
        return bool(catalog.tableExists(table, database))
    except TypeError:
        return bool(catalog.tableExists(f"{database}.{table}"))


def resolve_existing_table(spark, database: str, candidates: Iterable[str]) -> str:
    """Return the first approved existing table, failing if no candidate exists."""
    quote_catalog_identifier(database)
    ordered = list(candidates)
    if not ordered:
        raise ValueError("at least one table candidate is required")
    for table in ordered:
        quote_catalog_identifier(table)
        if _table_exists(spark.catalog, database, table):
            return table
    raise LookupError(f"none of the approved tables exist in database {database!r}: {ordered!r}")
