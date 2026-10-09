"""Versioned search representation over located PDF blocks.

The source passages remain the citation identity. Rebuilding search units
does not alter papers, annotations, or PDF coordinates.
"""

from contextlib import closing
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Callable

import numpy as np

from .database import Database
from .library import PaperLibrary, PassageHit, SourceSpan, _search_expression
from .pdf import ExtractedPassage
from . import representation
from .representation import build_units
from .semantic import BuildResult, EmbeddingProvider, _normalized


def _config_hash(tokenizer_model: str, max_tokens: int) -> str:
    """Fingerprint the unit version, tokenizer identity and token budget."""
    payload = f"{representation.REPRESENTATION_VERSION}|{tokenizer_model}|{max_tokens}"
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


class UnitStore:
    def __init__(self, library: PaperLibrary):
        self.library = library
        self.database_path = library.database_path
        self.database: Database = library.database
        self._refresh_lock = RLock()
        with self.database.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS retrieval_units (
                    id INTEGER PRIMARY KEY,
                    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL,
                    section TEXT NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS unit_sources (
                    unit_id INTEGER NOT NULL REFERENCES retrieval_units(id) ON DELETE CASCADE,
                    passage_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (unit_id, passage_id)
                );
                CREATE TABLE IF NOT EXISTS unit_index_state (
                    paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
                    file_hash TEXT NOT NULL,
                    representation_version INTEGER NOT NULL,
                    tokenizer_model TEXT NOT NULL,
                    config_hash TEXT NOT NULL DEFAULT ''
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS unit_search
                    USING fts5(section, text, tokenize='porter unicode61');
            """)
            columns = {row["name"] for row in connection.execute(
                "PRAGMA table_info(unit_index_state)")}
            if "config_hash" not in columns:
                connection.execute(
                    "ALTER TABLE unit_index_state ADD COLUMN config_hash TEXT NOT NULL DEFAULT ''")

    def _connect(self) -> sqlite3.Connection:
        return self.database.connect_only()

    def refresh(self, *, token_count: Callable[[str], int] | None = None,
                tokenizer_model: str = "",
                max_tokens: int | None = None) -> int:
        """Rebuild only stale papers.

        A paper is stale when its file hash, tokenizer model, or retrieval
        configuration fingerprint no longer matches the stored state.
        """
        with self._refresh_lock:
            return self._refresh_unlocked(token_count=token_count,
                                          tokenizer_model=tokenizer_model,
                                          max_tokens=max_tokens)

    def _refresh_unlocked(self, *, token_count: Callable[[str], int] | None,
                          tokenizer_model: str,
                          max_tokens: int | None) -> int:
        connection = self._connect()
        changed = 0
        try:
            papers = connection.execute("SELECT id, file_hash FROM papers ORDER BY id").fetchall()
            effective_max = (representation.MAX_UNIT_TOKENS if max_tokens is None
                             else max_tokens)
            if effective_max < 32:
                raise ValueError("max_tokens must be at least 32")
            fingerprint = _config_hash(tokenizer_model, effective_max)
            for paper in papers:
                state = connection.execute(
                    "SELECT * FROM unit_index_state WHERE paper_id=?", (paper["id"],)
                ).fetchone()
                stored_hash = (state["config_hash"] if state else "")
                if (state and state["file_hash"] == paper["file_hash"]
                        and state["representation_version"] == representation.REPRESENTATION_VERSION
                        and (stored_hash == fingerprint or
                             (token_count is None and state["tokenizer_model"]
                              and stored_hash == _config_hash(state["tokenizer_model"],
                                                               effective_max)))):
                    continue
                rows = connection.execute(
                    "SELECT * FROM passages WHERE paper_id=? ORDER BY page_number, block_number",
                    (paper["id"],),
                ).fetchall()
                source = [ExtractedPassage(row["page_number"], row["block_number"],
                                           row["text"], (row["x0"], row["y0"],
                                                         row["x1"], row["y1"]))
                          for row in rows]
                units = build_units(source, token_count=token_count,
                                    max_tokens=effective_max)
                with connection:
                    connection.execute(
                        "DELETE FROM unit_search WHERE rowid IN "
                        "(SELECT id FROM retrieval_units WHERE paper_id=?)", (paper["id"],)
                    )
                    connection.execute("DELETE FROM retrieval_units WHERE paper_id=?", (paper["id"],))
                    for unit in units:
                        cursor = connection.execute(
                            "INSERT INTO retrieval_units(paper_id,page_number,section,text) "
                            "VALUES (?,?,?,?)",
                            (paper["id"], source[unit.source_indexes[0]].page_number,
                             unit.section, unit.text),
                        )
                        unit_id = cursor.lastrowid
                        connection.execute(
                            "INSERT INTO unit_search(rowid,section,text) VALUES (?,?,?)",
                            (unit_id, unit.section, unit.text),
                        )
                        connection.executemany(
                            "INSERT INTO unit_sources(unit_id,passage_id,position) VALUES (?,?,?)",
                            [(unit_id, rows[index]["id"], position)
                             for position, index in enumerate(unit.source_indexes)],
                        )
                    connection.execute(
                        "INSERT INTO unit_index_state(paper_id,file_hash,representation_version,"
                        "tokenizer_model,config_hash) VALUES (?,?,?,?,?) "
                        "ON CONFLICT(paper_id) DO UPDATE SET "
                        "file_hash=excluded.file_hash, "
                        "representation_version=excluded.representation_version, "
                        "tokenizer_model=excluded.tokenizer_model, "
                        "config_hash=excluded.config_hash",
                        (paper["id"], paper["file_hash"], representation.REPRESENTATION_VERSION,
                         tokenizer_model, fingerprint),
                    )
                changed += 1
            return changed
        finally:
            connection.close()

    def count_units(self) -> int:
        with closing(self._connect()) as connection:
            return connection.execute("SELECT COUNT(*) FROM retrieval_units").fetchone()[0]

    def _hits(self, rows: list[sqlite3.Row], *,
              connection: sqlite3.Connection) -> tuple[PassageHit, ...]:
        if not rows:
            return ()
        ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in ids)
        locations = connection.execute(
            f"""SELECT us.unit_id, p.id AS passage_id, p.page_number,
                       p.x0,p.y0,p.x1,p.y1 FROM unit_sources AS us
                 JOIN passages AS p ON p.id=us.passage_id
                 WHERE us.unit_id IN ({placeholders}) ORDER BY us.unit_id, us.position""",
            ids,
        ).fetchall()
        spans: dict[int, list[SourceSpan]] = {}
        for item in locations:
            spans.setdefault(item["unit_id"], []).append(SourceSpan(
                item["passage_id"], item["page_number"],
                (item["x0"], item["y0"], item["x1"], item["y1"])))
        output = []
        for row in rows:
            links = tuple(spans.get(row["id"], ()))
            if not links:
                continue
            first = links[0]
            output.append(PassageHit(
                passage_id=-row["id"], paper_id=row["paper_id"],
                paper_path=Path(row["file_path"]), paper_title=row["title"],
                page_number=first.page_number,
                text=f"{row['section']}\n{row['text']}" if row["section"] else row["text"],
                bbox=first.bbox, score=float(row["score"]), source_spans=links,
            ))
        return tuple(output)

    def search(self, question: str, limit: int = 5, *,
               space_id: int | None = None,
               paper_id: int | None = None) -> tuple[PassageHit, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        self.refresh()
        expression = _search_expression(question)
        if not expression:
            return ()
        connection = self._connect()
        try:
            membership = ("JOIN space_papers AS sp ON sp.paper_id=d.id AND sp.space_id=?"
                          if space_id is not None else "")
            rows = connection.execute(
                f"""SELECT u.*, d.file_path, d.title, bm25(unit_search) AS score
                     FROM unit_search JOIN retrieval_units AS u ON u.id=unit_search.rowid
                     JOIN papers AS d ON d.id=u.paper_id {membership}
                     WHERE unit_search MATCH ?
                       {"AND u.paper_id=?" if paper_id is not None else ""}
                     ORDER BY score LIMIT ?""",
                ((space_id,) if space_id is not None else ())
                + (expression,) + ((paper_id,) if paper_id is not None else ())
                + (limit,),
            ).fetchall()
            return self._hits(rows, connection=connection)
        finally:
            connection.close()


class UnitSemanticIndex:
    def __init__(self, store: UnitStore):
        self.store = store
        self.database_path = store.database_path
        self.database: Database = store.database
        with self.database.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS unit_embeddings (
                unit_id INTEGER NOT NULL REFERENCES retrieval_units(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY(unit_id,model)
            )""")

    def count(self, model: str) -> int:
        with self.database.connect() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM unit_embeddings WHERE model=?", (model,)
            ).fetchone()[0]

    def build(self, provider: EmbeddingProvider, *, batch_size: int = 32,
              progress: Callable[[int, int], None] | None = None,
              max_tokens: int | None = None) -> BuildResult:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        count_tokens = getattr(provider, "count_untruncated_tokens", None)
        self.store.refresh(token_count=count_tokens,
                           tokenizer_model=provider.name if count_tokens else "",
                           max_tokens=max_tokens)
        connection = self._connect()
        indexed = 0
        try:
            total = connection.execute(
                """SELECT COUNT(*) FROM retrieval_units AS u
                     LEFT JOIN unit_embeddings AS e ON e.unit_id=u.id AND e.model=?
                     WHERE e.unit_id IS NULL""", (provider.name,)
            ).fetchone()[0]
            while True:
                rows = connection.execute(
                    """SELECT u.id,u.section,u.text FROM retrieval_units AS u
                         LEFT JOIN unit_embeddings AS e ON e.unit_id=u.id AND e.model=?
                         WHERE e.unit_id IS NULL ORDER BY u.id LIMIT ?""",
                    (provider.name, batch_size),
                ).fetchall()
                if not rows:
                    break
                texts = [f"{row['section']}\n{row['text']}" if row["section"]
                         else row["text"] for row in rows]
                vectors = provider.embed_passages(texts)
                if len(vectors) != len(rows):
                    raise ValueError("Embedding model returned the wrong number of vectors")
                normalized = [_normalized(vector) for vector in vectors]
                if len({vector.size for vector in normalized}) != 1:
                    raise ValueError("Embedding dimensions changed within a batch")
                with connection:
                    connection.executemany(
                        "INSERT INTO unit_embeddings(unit_id,model,dimension,vector) VALUES (?,?,?,?)",
                        [(row["id"], provider.name, vector.size, vector.tobytes())
                         for row, vector in zip(rows, normalized, strict=True)],
                    )
                indexed += len(rows)
                if progress:
                    progress(indexed, total)
            return BuildResult(indexed, total, provider.name)
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        return self.database.connect_only()

    def search(self, query: str, provider: EmbeddingProvider, *, limit: int = 20,
               space_id: int | None = None,
               paper_id: int | None = None) -> tuple[PassageHit, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query_vector = _normalized(provider.embed_query(query))
        connection = self._connect()
        try:
            membership = ("JOIN space_papers AS sp ON sp.paper_id=d.id AND sp.space_id=?"
                          if space_id is not None else "")
            rows = connection.execute(
                f"""SELECT u.*,d.file_path,d.title,e.vector,0.0 AS score
                     FROM unit_embeddings AS e
                     JOIN retrieval_units AS u ON u.id=e.unit_id
                     JOIN papers AS d ON d.id=u.paper_id {membership}
                     WHERE e.model=? AND e.dimension=?
                       {"AND u.paper_id=?" if paper_id is not None else ""}""",
                ((space_id,) if space_id is not None else ())
                + (provider.name, query_vector.size)
                + ((paper_id,) if paper_id is not None else ()),
            ).fetchall()
            if not rows:
                return ()
            vectors = np.stack([np.frombuffer(row["vector"], dtype=np.float32)
                                for row in rows])
            scores = vectors @ query_vector
            order = np.argsort(scores)[::-1][:limit]
            hits = self.store._hits([rows[i] for i in order], connection=connection)
            return tuple(replace(hit, score=float(scores[i]))
                         for hit, i in zip(hits, order, strict=True))
        finally:
            connection.close()
