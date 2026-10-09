"""Consistent SQLite connection settings for local stores."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import sqlite3


class Database:
    """Open a fresh configured connection for each operation."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection within a transaction.

        Commits on success and rolls back on exception. The connection is
        always closed before this method returns.
        """
        connection = self._open()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def connect_only(self) -> sqlite3.Connection:
        """Return a connection without transaction management.

        Use this when the caller wants to manage the transaction itself.
        The caller is responsible for closing the connection.
        """
        return self._open()

    def _open(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection
