"""Optional local embeddings and exact vector search for a small PDF library."""

from dataclasses import dataclass, replace
from contextlib import closing
from pathlib import Path
import sqlite3
from typing import Callable, Protocol, Sequence

import numpy as np

from .library import PaperLibrary, PassageHit


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingProvider(Protocol):
    name: str

    def embed_passages(self, texts: Sequence[str]) -> list[np.ndarray]: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedProvider:
    """Load model weights only when indexing or a vector query really needs them."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL,
                 cache_dir: Path | None = None):
        self.name = model_name
        self.cache_dir = cache_dir
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(
                model_name=self.name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
        return self._model

    def embed_passages(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [np.asarray(vector, dtype=np.float32)
                for vector in self.model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self.model.query_embed(text))), dtype=np.float32)


@dataclass(frozen=True)
class BuildResult:
    indexed: int
    total: int
    model: str


def _normalized(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Embedding model returned an invalid vector")
    norm = float(np.linalg.norm(values))
    if norm == 0:
        raise ValueError("Embedding model returned a zero vector")
    return values / norm


class SemanticIndex:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS passage_embeddings (
                     passage_id INTEGER NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
                     model TEXT NOT NULL,
                     dimension INTEGER NOT NULL,
                     vector BLOB NOT NULL,
                     PRIMARY KEY (passage_id, model)
                   )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def count(self, model: str) -> int:
        connection = self._connect()
        try:
            return connection.execute(
                "SELECT COUNT(*) FROM passage_embeddings WHERE model=?", (model,)
            ).fetchone()[0]
        finally:
            connection.close()

    def build(self, provider: EmbeddingProvider, *,
              batch_size: int = 32, progress: Callable[[int, int], None] | None = None
              ) -> BuildResult:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        connection = self._connect()
        indexed = 0
        try:
            total = connection.execute(
                """SELECT COUNT(*) FROM passages AS p
                   LEFT JOIN passage_embeddings AS e
                     ON e.passage_id=p.id AND e.model=?
                   WHERE e.passage_id IS NULL""",
                (provider.name,),
            ).fetchone()[0]
            while True:
                rows = connection.execute(
                    """SELECT p.id, p.text FROM passages AS p
                       LEFT JOIN passage_embeddings AS e
                         ON e.passage_id=p.id AND e.model=?
                       WHERE e.passage_id IS NULL ORDER BY p.id LIMIT ?""",
                    (provider.name, batch_size),
                ).fetchall()
                if not rows:
                    break
                vectors = provider.embed_passages([row["text"][:1600] for row in rows])
                if len(vectors) != len(rows):
                    raise ValueError("Embedding model returned the wrong number of vectors")
                normalized = [_normalized(vector) for vector in vectors]
                dimensions = {vector.size for vector in normalized}
                if len(dimensions) != 1:
                    raise ValueError("Embedding dimensions changed within a batch")
                with connection:
                    connection.executemany(
                        """INSERT INTO passage_embeddings
                           (passage_id, model, dimension, vector) VALUES (?, ?, ?, ?)""",
                        [(row["id"], provider.name, vector.size, vector.tobytes())
                         for row, vector in zip(rows, normalized, strict=True)],
                    )
                indexed += len(rows)
                if progress:
                    progress(indexed, total)
            return BuildResult(indexed, total, provider.name)
        finally:
            connection.close()

    def search(self, query: str, provider: EmbeddingProvider, *,
               limit: int = 20, space_id: int | None = None) -> tuple[PassageHit, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        query_vector = _normalized(provider.embed_query(query))
        membership = (
            "JOIN space_papers AS sp ON sp.paper_id=d.id AND sp.space_id=?"
            if space_id is not None else ""
        )
        connection = self._connect()
        try:
            rows = connection.execute(
                f"""SELECT p.id, p.paper_id, p.page_number, p.text,
                           p.x0, p.y0, p.x1, p.y1, d.file_path, d.title, e.vector
                    FROM passage_embeddings AS e
                    JOIN passages AS p ON p.id=e.passage_id
                    JOIN papers AS d ON d.id=p.paper_id
                    {membership}
                    WHERE e.model=? AND e.dimension=?""",
                ((space_id, provider.name, query_vector.size)
                 if space_id is not None else (provider.name, query_vector.size)),
            ).fetchall()
            if not rows:
                return ()
            vectors = np.stack([np.frombuffer(row["vector"], dtype=np.float32)
                                for row in rows])
            scores = vectors @ query_vector
            order = np.argsort(scores)[::-1][:limit]
            return tuple(
                PassageHit(
                    passage_id=rows[i]["id"], paper_id=rows[i]["paper_id"],
                    paper_path=Path(rows[i]["file_path"]),
                    paper_title=rows[i]["title"], page_number=rows[i]["page_number"],
                    text=rows[i]["text"],
                    bbox=(rows[i]["x0"], rows[i]["y0"], rows[i]["x1"], rows[i]["y1"]),
                    score=float(scores[i]),
                ) for i in order
            )
        finally:
            connection.close()


class HybridRetriever:
    """Merge two ranked lists by rank, leaving their incompatible scores alone."""

    def __init__(self, library: PaperLibrary, index: SemanticIndex,
                 provider: EmbeddingProvider):
        self.library = library
        self.index = index
        self.provider = provider

    def search(self, question: str, limit: int = 5, *,
               space_id: int | None = None) -> tuple[PassageHit, ...]:
        # The default top-five path needs a wider source pool than the reranker:
        # fusion otherwise discards useful one-channel passages too early.
        pool = max(limit * 8, 40)
        return self.candidates(question, pool, space_id=space_id)[:limit]

    def candidates(self, question: str, per_source_limit: int, *,
                   space_id: int | None = None) -> tuple[PassageHit, ...]:
        """Keep both sources' candidates available for a later reranker."""
        lexical = self.library.search(question, per_source_limit, space_id=space_id)
        if self.index.count(self.provider.name) == 0:
            return lexical
        semantic = self.index.search(question, self.provider, limit=per_source_limit,
                                     space_id=space_id)
        by_id: dict[int, PassageHit] = {}
        fused: dict[int, float] = {}
        for ranked in (lexical, semantic):
            for position, hit in enumerate(ranked, start=1):
                by_id[hit.passage_id] = hit
                fused[hit.passage_id] = fused.get(hit.passage_id, 0.0) + 1 / (60 + position)
        ordered = sorted(fused, key=lambda passage_id: (-fused[passage_id], passage_id))
        return tuple(replace(by_id[passage_id], score=fused[passage_id])
                     for passage_id in ordered)


class Reranker(Protocol):
    def score(self, question: str, texts: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base",
                 cache_dir: Path | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._model = TextCrossEncoder(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
        return list(self._model.rerank(question, list(texts)))


class RerankingRetriever:
    """Spend cross-encoder compute only on a bounded hybrid candidate set."""

    def __init__(self, initial: HybridRetriever, reranker: Reranker):
        self.initial = initial
        self.reranker = reranker

    def search(self, question: str, limit: int = 5, *,
               space_id: int | None = None) -> tuple[PassageHit, ...]:
        candidates = self.initial.candidates(question, max(limit * 4, 20),
                                             space_id=space_id)
        if not candidates:
            return ()
        scores = self.reranker.score(question, [hit.text[:1600] for hit in candidates])
        if len(scores) != len(candidates):
            raise ValueError("Reranker returned the wrong number of scores")
        if not np.all(np.isfinite(scores)):
            raise ValueError("Reranker returned an invalid score")
        order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
        return tuple(replace(candidates[index], score=float(scores[index]))
                     for index in order[:limit])


def build_retriever(library: PaperLibrary, index: SemanticIndex,
                    provider: EmbeddingProvider, *, rerank_model: str = "",
                    cache_dir: Path | None = None) -> HybridRetriever | RerankingRetriever:
    """Use one retrieval composition for the app and its end-to-end evaluation."""
    initial = HybridRetriever(library, index, provider)
    if not rerank_model:
        return initial
    return RerankingRetriever(initial, FastEmbedReranker(rerank_model, cache_dir))
