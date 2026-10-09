"""Optional local embeddings and exact vector search for a small PDF library."""

from dataclasses import dataclass, replace
from pathlib import Path
import sqlite3
from typing import Callable, Protocol, Sequence

import numpy as np

from .database import Database
from .library import PaperLibrary, PassageHit
from .telemetry import timed


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingProvider(Protocol):
    name: str

    def embed_passages(self, texts: Sequence[str]) -> list[np.ndarray]: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedProvider:
    """Load model weights only when indexing or a vector query really needs them."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL,
                 cache_dir: Path | None = None, *, local_files_only: bool = False):
        self.name = model_name
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._model = None
        self._counting_tokenizer = None
        self._last_query_embedding: tuple[str, np.ndarray] | None = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(
                model_name=self.name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                **({"local_files_only": True} if self.local_files_only else {}),
            )
        return self._model

    def embed_passages(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [np.asarray(vector, dtype=np.float32)
                for vector in self.model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> np.ndarray:
        cached = self._last_query_embedding
        if cached is not None and cached[0] == text:
            return cached[1].copy()
        vector = np.asarray(next(iter(self.model.query_embed(text))), dtype=np.float32)
        # Scoped comparisons search several papers with the same query. Keep
        # only that query; storing the pair together keeps concurrent reads safe.
        self._last_query_embedding = (text, vector.copy())
        return vector

    def count_untruncated_tokens(self, text: str) -> int:
        """Count with the embedding tokenizer before its 128-token truncation.

        FastEmbed's public token_count applies model truncation first, so a
        2,000-token paragraph can misleadingly report 128. Clone the same
        tokenizer and turn off truncation for chunk construction.
        """
        if self._counting_tokenizer is None:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_str(self.model.model.tokenizer.to_str())
            tokenizer.no_truncation()
            tokenizer.no_padding()
            self._counting_tokenizer = tokenizer
        return len(self._counting_tokenizer.encode(text).ids)


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
        self.database = Database(database_path)
        with self.database.connect() as connection:
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
        return self.database.connect_only()

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
               limit: int = 20, space_id: int | None = None,
               paper_id: int | None = None) -> tuple[PassageHit, ...]:
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
                    WHERE e.model=? AND e.dimension=?
                      {"AND p.paper_id=?" if paper_id is not None else ""}""",
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
                 provider: EmbeddingProvider, *, rrf_k: int = 60):
        if not 1 <= rrf_k <= 200:
            raise ValueError("rrf_k must be between 1 and 200")
        self.library = library
        self.index = index
        self.provider = provider
        self.rrf_k = rrf_k

    def search(self, question: str, limit: int = 5, *,
               space_id: int | None = None,
               paper_id: int | None = None) -> tuple[PassageHit, ...]:
        # The default top-five path needs a wider source pool than the reranker:
        # fusion otherwise discards useful one-channel passages too early.
        pool = max(limit * 8, 40)
        return self.candidates(question, pool, space_id=space_id,
                               paper_id=paper_id)[:limit]

    @timed("retrieval.candidates")
    def candidates(self, question: str, per_source_limit: int, *,
                   space_id: int | None = None,
                   paper_id: int | None = None) -> tuple[PassageHit, ...]:
        """Keep both sources' candidates available for a later reranker."""
        scope = {"space_id": space_id}
        if paper_id is not None:
            scope["paper_id"] = paper_id
        lexical = self.library.search(question, per_source_limit, **scope)
        if self.index.count(self.provider.name) == 0:
            return lexical
        semantic = self.index.search(question, self.provider, limit=per_source_limit,
                                     **scope)
        by_id: dict[int, PassageHit] = {}
        fused: dict[int, float] = {}
        for ranked in (lexical, semantic):
            for position, hit in enumerate(ranked, start=1):
                by_id[hit.passage_id] = hit
                fused[hit.passage_id] = (fused.get(hit.passage_id, 0.0)
                                         + 1 / (self.rrf_k + position))
        ordered = sorted(fused, key=lambda passage_id: (-fused[passage_id], passage_id))
        return tuple(replace(by_id[passage_id], score=fused[passage_id])
                     for passage_id in ordered)


class Reranker(Protocol):
    def score(self, question: str, texts: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base",
                 cache_dir: Path | None = None, *, local_files_only: bool = False):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._model = None

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._model = TextCrossEncoder(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                **({"local_files_only": True} if self.local_files_only else {}),
            )
        return list(self._model.rerank(question, list(texts)))


class RerankingRetriever:
    """Spend cross-encoder compute only on a bounded hybrid candidate set."""

    def __init__(self, initial: HybridRetriever, reranker: Reranker, *,
                 pool_factor: int = 4):
        if not 4 <= pool_factor <= 16:
            raise ValueError("pool_factor must be between 4 and 16")
        self.initial = initial
        self.reranker = reranker
        self.pool_factor = pool_factor

    @timed("retrieval.rerank")
    def search(self, question: str, limit: int = 5, *,
               space_id: int | None = None,
               paper_id: int | None = None) -> tuple[PassageHit, ...]:
        pool = min(80, max(limit * self.pool_factor, 20))
        candidates = self.initial.candidates(question, pool,
                                             space_id=space_id, paper_id=paper_id)
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
                    cache_dir: Path | None = None, rrf_k: int = 60,
                    rerank_pool_factor: int = 4) -> HybridRetriever | RerankingRetriever:
    """Use one retrieval composition for the app and its end-to-end evaluation."""
    initial = HybridRetriever(library, index, provider, rrf_k=rrf_k)
    if not rerank_model:
        return initial
    return RerankingRetriever(initial, FastEmbedReranker(rerank_model, cache_dir),
                              pool_factor=rerank_pool_factor)
