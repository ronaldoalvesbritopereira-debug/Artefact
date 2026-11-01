from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError

from artefact_rag.database import (
    create_read_only_sql_database,
    initialize_database,
    validate_read_only_query,
)


def test_database_import_preserves_numeric_types_and_skips_unchanged_csvs(
    tmp_path: Path,
) -> None:
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    (csv_dir / "challenge - products.csv").write_text(
        "product_id,price_brl,stock_quantity,name\n1,599.90,12,Yamaha C40\n",
        encoding="utf-8",
    )
    database_path = tmp_path / "store.sqlite3"

    table_names = initialize_database(csv_dir, database_path)
    assert table_names == ["products"]

    with closing(sqlite3.connect(database_path)) as connection:
        column_types = {
            name: data_type
            for _, name, data_type, *_ in connection.execute(
                "PRAGMA table_info(products)"
            )
        }
    assert column_types["product_id"] == "INTEGER"
    assert column_types["price_brl"] == "REAL"
    assert column_types["stock_quantity"] == "INTEGER"

    database_path.write_bytes(b"not a valid database")
    assert initialize_database(csv_dir, database_path) == ["products"]


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM products",
        "DROP TABLE products",
        "PRAGMA table_info(products)",
        "SELECT * FROM products; SELECT * FROM orders",
    ],
)
def test_read_only_query_validation_rejects_unsafe_statements(query: str) -> None:
    with pytest.raises(ValueError):
        validate_read_only_query(query)


def test_read_only_database_executes_select_and_rejects_write(tmp_path: Path) -> None:
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    (csv_dir / "products.csv").write_text(
        "product_id,price_brl\n1,599.90\n",
        encoding="utf-8",
    )
    database_path = tmp_path / "store.sqlite3"
    table_names = initialize_database(csv_dir, database_path)
    database = create_read_only_sql_database(database_path, table_names)

    assert "599.9" in database.run("SELECT price_brl FROM products")
    with pytest.raises(OperationalError):
        database.run("UPDATE products SET price_brl = 1")


def test_read_only_validator_allows_select_and_single_trailing_semicolon() -> None:
    assert validate_read_only_query(" SELECT price_brl FROM products; ") == (
        "SELECT price_brl FROM products"
    )
