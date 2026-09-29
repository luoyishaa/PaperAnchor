"""Research-space membership over a shared paper library."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
import sqlite3
from typing import Iterator


ALL_PAPERS_SPACE_ID = 1


@dataclass(frozen=True)
class ResearchSpace:
    id: int
    name: str
    created_at: str
    paper_count: int


class SpaceStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS spaces (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS space_papers (
                    space_id INTEGER NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
                    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (space_id, paper_id)
                );
                CREATE TRIGGER IF NOT EXISTS paper_in_all_space
                AFTER INSERT ON papers
                BEGIN
                    INSERT INTO space_papers (space_id, paper_id, added_at)
                    SELECT id, NEW.id, NEW.indexed_at FROM spaces WHERE id = 1;
                END;
                """
            )
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                "INSERT OR IGNORE INTO spaces (id, name, created_at) VALUES (1, 'All papers', ?)",
                (now,),
            )
            # Backfill papers indexed before research spaces were introduced.
            connection.execute(
                """INSERT OR IGNORE INTO space_papers (space_id, paper_id, added_at)
                   SELECT 1, id, indexed_at FROM papers"""
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def list_spaces(self) -> tuple[ResearchSpace, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT s.id, s.name, s.created_at, COUNT(sp.paper_id) AS paper_count
                   FROM spaces AS s LEFT JOIN space_papers AS sp ON sp.space_id = s.id
                   GROUP BY s.id ORDER BY s.id = 1 DESC, s.name COLLATE NOCASE"""
            ).fetchall()
            return tuple(ResearchSpace(**dict(row)) for row in rows)

    def get_space(self, space_id: int) -> ResearchSpace | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT s.id, s.name, s.created_at, COUNT(sp.paper_id) AS paper_count
                   FROM spaces AS s LEFT JOIN space_papers AS sp ON sp.space_id = s.id
                   WHERE s.id = ? GROUP BY s.id""",
                (space_id,),
            ).fetchone()
            return ResearchSpace(**dict(row)) if row else None

    def create_space(self, name: str) -> ResearchSpace:
        name = name.strip()
        if not 1 <= len(name) <= 80:
            raise ValueError("Space name must contain 1 to 80 characters")
        with self._connect() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO spaces (name, created_at) VALUES (?, ?)",
                    (name, datetime.now(timezone.utc).isoformat()),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("A space with this name already exists") from error
            space_id = cursor.lastrowid
        return self.get_space(space_id)

    def add_paper(self, space_id: int, paper_id: int) -> None:
        with self._connect() as connection:
            if not connection.execute("SELECT 1 FROM spaces WHERE id=?", (space_id,)).fetchone():
                raise ValueError("Space not found")
            if not connection.execute("SELECT 1 FROM papers WHERE id=?", (paper_id,)).fetchone():
                raise ValueError("Paper not found")
            connection.execute(
                """INSERT OR IGNORE INTO space_papers (space_id, paper_id, added_at)
                   VALUES (?, ?, ?)""",
                (space_id, paper_id, datetime.now(timezone.utc).isoformat()),
            )

    def remove_paper(self, space_id: int, paper_id: int) -> None:
        if space_id == ALL_PAPERS_SPACE_ID:
            raise ValueError("Papers cannot be removed from All papers")
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM space_papers WHERE space_id=? AND paper_id=?",
                (space_id, paper_id),
            )

    def delete_space(self, space_id: int) -> None:
        if space_id == ALL_PAPERS_SPACE_ID:
            raise ValueError("All papers cannot be deleted")
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM spaces WHERE id=?", (space_id,))
            if cursor.rowcount == 0:
                raise ValueError("Space not found")
