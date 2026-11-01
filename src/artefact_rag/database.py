from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path

import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)
METADATA_TABLE = "_app_metadata"
FINGERPRINT_KEY = "csv_fingerprint"


def _table_name(path: Path) -> str:
    name = path.stem.rsplit("-", maxsplit=1)[-1].strip()
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    table_name = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_name).strip("_").lower()
    if not table_name:
        raise ValueError(f"Não foi possível definir o nome da tabela para {path.name}.")
    return table_name


def _csv_fingerprint(csv_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in csv_paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _database_fingerprint(database_path: Path) -> str | None:
    if not database_path.exists():
        return None
    try:
        with closing(sqlite3.connect(database_path)) as connection:
            table_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (METADATA_TABLE,),
            ).fetchone()
            if table_exists is None:
                return None
            result = connection.execute(
                f"SELECT value FROM {METADATA_TABLE} WHERE key = ?",
                (FINGERPRINT_KEY,),
            ).fetchone()
    except sqlite3.DatabaseError:
        logger.warning("O banco existente está inválido; será recriado a partir dos CSVs.")
        return None
    return result[0] if result else None


def initialize_database(csv_dir: Path, database_path: Path) -> list[str]:
    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {csv_dir}.")

    table_names = [_table_name(path) for path in csv_paths]
    if len(table_names) != len(set(table_names)):
        raise ValueError("Há arquivos CSV com nomes de tabela duplicados.")

    fingerprint = _csv_fingerprint(csv_paths)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if _database_fingerprint(database_path) == fingerprint:
        logger.info("Banco SQLite atualizado; os CSVs não foram alterados.")
        return table_names

    temporary_path = database_path.with_suffix(".tmp.sqlite3")
    temporary_path.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(temporary_path)) as connection:
            with connection:
                for csv_path, table_name in zip(csv_paths, table_names, strict=True):
                    dataframe = pd.read_csv(csv_path)
                    dataframe.to_sql(
                        table_name,
                        connection,
                        if_exists="replace",
                        index=False,
                    )

                connection.execute(
                    f"CREATE TABLE {METADATA_TABLE} "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(
                    f"INSERT INTO {METADATA_TABLE} (key, value) VALUES (?, ?)",
                    (FINGERPRINT_KEY, fingerprint),
                )

        temporary_path.replace(database_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    logger.info("Banco SQLite criado com %d tabelas a partir dos CSVs.", len(table_names))
    return table_names


def create_read_only_sql_database(
    database_path: Path, table_names: list[str]
) -> SQLDatabase:
    database_uri = f"{database_path.resolve().as_uri()}?mode=ro"

    def connect_read_only() -> sqlite3.Connection:
        return sqlite3.connect(
            database_uri, uri=True, check_same_thread=False
        )

    engine = create_engine("sqlite://", creator=connect_read_only)
    return SQLDatabase(
        engine=engine,
        include_tables=table_names,
        sample_rows_in_table_info=0,
    )


def validate_read_only_query(query: str) -> str:
    statement = query.strip()
    if not statement:
        raise ValueError("A consulta SQL deve conter apenas uma instrução.")

    quote: str | None = None
    in_line_comment = False
    in_block_comment = False
    has_terminator = False
    index = 0
    while index < len(statement):
        character = statement[index]
        next_character = statement[index + 1] if index + 1 < len(statement) else ""

        if in_line_comment:
            if character in "\r\n":
                in_line_comment = False
        elif in_block_comment:
            if character == "*" and next_character == "/":
                in_block_comment = False
                index += 1
        elif quote is not None:
            if character == quote:
                if next_character == quote:
                    index += 1
                else:
                    quote = None
        elif character == "-" and next_character == "-":
            in_line_comment = True
            index += 1
        elif character == "/" and next_character == "*":
            in_block_comment = True
            index += 1
        elif character in ("'", '"', "`"):
            if has_terminator:
                raise ValueError("A consulta SQL deve conter apenas uma instrução.")
            quote = character
        elif character == "[":
            if has_terminator:
                raise ValueError("A consulta SQL deve conter apenas uma instrução.")
            quote = "]"
        elif character == ";":
            if has_terminator:
                raise ValueError("A consulta SQL deve conter apenas uma instrução.")
            has_terminator = True
        elif has_terminator and not character.isspace():
            raise ValueError("A consulta SQL deve conter apenas uma instrução.")
        index += 1

    if quote is not None or in_block_comment:
        raise ValueError("A consulta SQL está incompleta.")
    if has_terminator:
        statement = statement[: statement.rfind(";")].rstrip()
    if not re.match(r"^(SELECT|WITH)\b", statement, flags=re.IGNORECASE):
        raise ValueError("Somente consultas SQL de leitura (SELECT ou WITH) são aceitas.")
    return statement
