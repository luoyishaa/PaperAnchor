"""Persist papers and passages; expose one small full-text retrieval interface."""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import re
import sqlite3

from .pdf import parse_pdf


@dataclass(frozen=True)
class IndexResult:
    paper_id: int
    passage_count: int
    changed: bool


@dataclass(frozen=True)
class PassageHit:
    passage_id: int
    paper_path: Path
    paper_title: str
    page_number: int
    text: str
    bbox: tuple[float, float, float, float]
    score: float


_STOP_WORDS = frozenset(
    "a an and are as at be by can do for from how in is it of on or the to was were what which who why with".split()
)


def _search_expression(question: str) -> str:
    terms = [
        term
        for term in re.findall(r"[A-Za-z][A-Za-z0-9]*", question.lower())
        if term not in _STOP_WORDS
    ]
    # Quoting each extracted word keeps user punctuation out of FTS5 syntax.
    return " OR ".join(f'"{term}"' for term in dict.fromkeys(terms[:12]))


class PaperLibrary:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY,
                    file_path TEXT NOT NULL UNIQUE,
                    file_hash TEXT NOT NULL,
                    title TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS passages (
                    id INTEGER PRIMARY KEY,
                    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL,
                    block_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    x0 REAL NOT NULL, y0 REAL NOT NULL,
                    x1 REAL NOT NULL, y1 REAL NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS passage_search
                    USING fts5(text, tokenize='porter unicode61');
                """
            )
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def index_pdf(self, pdf_path: Path) -> IndexResult:
        path = pdf_path.resolve(strict=True)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF: {path}")
        file_hash = sha256(path.read_bytes()).hexdigest()
        connection = self._connect()
        try:
            existing = connection.execute(
                "SELECT id, file_hash FROM papers WHERE file_path = ?", (str(path),)
            ).fetchone()
            if existing and existing["file_hash"] == file_hash:
                count = connection.execute(
                    "SELECT COUNT(*) FROM passages WHERE paper_id = ?", (existing["id"],)
                ).fetchone()[0]
                return IndexResult(existing["id"], count, changed=False)

            parsed = parse_pdf(path)
            with connection:
                if existing:
                    paper_id = existing["id"]
                    connection.execute(
                        "DELETE FROM passage_search WHERE rowid IN "
                        "(SELECT id FROM passages WHERE paper_id = ?)",
                        (paper_id,),
                    )
                    connection.execute("DELETE FROM passages WHERE paper_id = ?", (paper_id,))
                    connection.execute(
                        "UPDATE papers SET file_hash=?, title=?, page_count=?, indexed_at=? WHERE id=?",
                        (
                            file_hash,
                            parsed.title,
                            parsed.page_count,
                            datetime.now(timezone.utc).isoformat(),
                            paper_id,
                        ),
                    )
                else:
                    cursor = connection.execute(
                        "INSERT INTO papers (file_path, file_hash, title, page_count, indexed_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            str(path),
                            file_hash,
                            parsed.title,
                            parsed.page_count,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    paper_id = cursor.lastrowid

                for passage in parsed.passages:
                    cursor = connection.execute(
                        "INSERT INTO passages "
                        "(paper_id, page_number, block_number, text, x0, y0, x1, y1) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            paper_id,
                            passage.page_number,
                            passage.block_number,
                            passage.text,
                            *passage.bbox,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO passage_search (rowid, text) VALUES (?, ?)",
                        (cursor.lastrowid, passage.text),
                    )
            return IndexResult(paper_id, len(parsed.passages), changed=True)
        finally:
            connection.close()

    def search(self, question: str, limit: int = 5) -> tuple[PassageHit, ...]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        expression = _search_expression(question)
        if not expression:
            return ()
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT p.id, p.page_number, p.text, p.x0, p.y0, p.x1, p.y1,
                       d.file_path, d.title, bm25(passage_search) AS score
                FROM passage_search
                JOIN passages AS p ON p.id = passage_search.rowid
                JOIN papers AS d ON d.id = p.paper_id
                WHERE passage_search MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (expression, limit),
            ).fetchall()
            return tuple(
                PassageHit(
                    passage_id=row["id"],
                    paper_path=Path(row["file_path"]),
                    paper_title=row["title"],
                    page_number=row["page_number"],
                    text=row["text"],
                    bbox=(row["x0"], row["y0"], row["x1"], row["y1"]),
                    score=row["score"],
                )
                for row in rows
            )
        finally:
            connection.close()
