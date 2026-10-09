"""Content identities shared by QASPER freezing, retrieval, and reporting."""

from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3


CORPUS_TABLES = (
    "papers", "passages", "passage_search", "retrieval_units", "unit_search",
    "unit_sources", "unit_index_state", "passage_embeddings", "unit_embeddings",
)


def public_cases_fingerprint(cases):
    return sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def corpus_fingerprint(database):
    """Hash retrieval content in a caller's snapshot or a new read-only snapshot."""
    if not isinstance(database, sqlite3.Connection):
        with closing(sqlite3.connect(f"{Path(database).resolve().as_uri()}?mode=ro", uri=True)) as connection:
            connection.execute("BEGIN")
            return corpus_fingerprint(connection)
    digest = sha256()
    for table in CORPUS_TABLES:
        schema = database.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        digest.update(json.dumps([table, schema[0] if schema else None]).encode("utf-8"))
        digest.update(b"\n")
        if schema is None:
            continue
        columns = database.execute(f'PRAGMA table_info("{table}")').fetchall()
        primary = [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]]
        order = ",".join(f'"{name}"' for name in primary) if primary else "rowid"
        for row in database.execute(f'SELECT rowid,* FROM "{table}" ORDER BY {order}'):
            values = [{"blob": value.hex()} if isinstance(value, bytes) else value for value in row]
            digest.update(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()
