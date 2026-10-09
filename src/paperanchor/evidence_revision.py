"""A constant-cost content revision for the optional resident evidence engine."""

from pathlib import Path
import sqlite3


# These tables supply the engine's catalog, source mapping, text and vectors.
# Notes, conversations, jobs and space membership do not change that snapshot.
CONTENT_TABLES = (
    "papers", "passages", "retrieval_units", "unit_sources", "unit_index_state",
    "unit_embeddings",
)


def ensure_revision_tracking(database: Path) -> None:
    """Install metadata-only change tracking when expanded evidence is first used.

    Row triggers catch same-size reindexing and in-place vector updates, including
    writes from other SQLite connections. The revision changes in the writer's
    own transaction, so rolled-back writes never invalidate the engine.
    """
    connection = sqlite3.connect(database, timeout=5)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE IF NOT EXISTS evidence_content_revision ("
                           "id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL)")
        connection.execute("INSERT OR IGNORE INTO evidence_content_revision VALUES (1,0)")
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for table in CONTENT_TABLES:
            if table not in tables:
                continue
            for action in ("INSERT", "UPDATE", "DELETE"):
                connection.execute(
                    f"CREATE TRIGGER IF NOT EXISTS evidence_revision_{table}_{action.lower()} "
                    f"AFTER {action} ON {table} BEGIN "
                    "UPDATE evidence_content_revision SET revision=revision+1 WHERE id=1; END")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
