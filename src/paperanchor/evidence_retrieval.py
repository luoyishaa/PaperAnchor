"""Shared, read-only retrieval for evidence experiments and the optional profile.

Models are supplied by the caller. This module neither downloads models nor
builds an index. One instance reads a consistent index snapshot until closed.
"""

from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3
from threading import RLock
import time

import numpy as np

from .library import PassageHit, SourceSpan, _search_expression
from .semantic import _normalized
from .unit_index import UnitStore


def select_titles(title, catalog, limit=3):
    """Match the public catalog using the historical token-set Jaccard rule."""
    tokens = set(re.findall(r"\w+", title.casefold()))
    scores = []
    for paper_id, name in catalog:
        other = set(re.findall(r"\w+", name.casefold()))
        if tokens & other:
            scores.append((len(tokens & other) / len(tokens | other), paper_id))
    return [paper for _, paper in sorted(scores, key=lambda pair: (-pair[0], pair[1]))[:limit]]


def fuse(rankings):
    by_id, scores = {}, {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            by_id[item.passage_id] = item
            scores[item.passage_id] = scores.get(item.passage_id, 0) + 1 / (60 + rank)
    return [replace(by_id[key], score=scores[key])
            for key in sorted(scores, key=lambda key: (-scores[key], key))]


class EvidenceRetriever:
    """Reuse models, source text and scores across a batch of questions.

    Set ``cache_enabled = False`` for inference timing without result caches.
    Corpus text/vector matrices remain resident, as in a warmed product process.
    Explicit ``paper_ids=[]`` means an empty selection; ``None`` permits routing.
    A caller serving a mutable library must recreate this snapshot after indexing.
    """

    def __init__(self, database: Path, provider, reranker, *, score_cache_path=None):
        self.database = Path(database).resolve()
        self.provider, self.reranker = provider, reranker
        self._lock = RLock()
        self.score_cache_enabled = True
        self.query_cache_enabled = True
        self._scores, self._query_vectors, self._matrices = {}, {}, {}
        self._trace_tokenizers = None
        self._trace_model_identity = None
        self.last_score_trace = {}
        self._closed = False
        self._cache = None
        self._connection = sqlite3.connect(
            f"{self.database.as_uri()}?mode=ro", uri=True, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("BEGIN")
        try:
            paper_columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(papers)")}
            columns = "id,title,file_hash" if "file_hash" in paper_columns else "id,title"
            paper_rows = self._connection.execute(f"SELECT {columns} FROM papers").fetchall()
            self.catalog = [(row["id"], row["title"]) for row in paper_rows]
            self.paper_hashes = ({row["id"]: row["file_hash"] for row in paper_rows}
                                 if "file_hash" in paper_columns else {})
            self.blocks, self._blocks_by_id = {}, {}
            for row in self._connection.execute(
                "SELECT b.*,p.title,p.file_path FROM passages b JOIN papers p ON p.id=b.paper_id "
                "ORDER BY b.paper_id,b.page_number,b.block_number,b.id"
            ):
                box = tuple(row[key] for key in ("x0", "y0", "x1", "y1"))
                block = PassageHit(row["id"], row["paper_id"], Path(row["file_path"]),
                                   row["title"], row["page_number"], row["text"], box, 0.0,
                                   (SourceSpan(row["id"], row["page_number"], box),))
                self.blocks.setdefault((block.paper_id, block.page_number), []).append(block)
                self._blocks_by_id[block.passage_id] = block
            self._units = {}
            store = object.__new__(UnitStore)
            cursor = self._connection.execute(
                "SELECT u.*,p.file_path,p.title,0.0 AS score FROM retrieval_units u "
                "JOIN papers p ON p.id=u.paper_id ORDER BY u.id")
            while rows := cursor.fetchmany(512):
                self._units.update((item.passage_id, item) for item in
                                   store._hits(rows, connection=self._connection))
            self._papers_with_units = {item.paper_id for item in self._units.values()}
            self._papers_with_blocks = {item.paper_id for item in self._blocks_by_id.values()}
            if score_cache_path is not None:
                cache_path = Path(score_cache_path).resolve()
                if cache_path == self.database:
                    raise ValueError("Score cache must be separate from the corpus database")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._cache = sqlite3.connect(cache_path, check_same_thread=False)
                self._cache.execute("CREATE TABLE IF NOT EXISTS evidence_scores "
                                    "(key TEXT PRIMARY KEY, value REAL NOT NULL)")
                self._cache.execute("CREATE TABLE IF NOT EXISTS evidence_queries "
                                    "(key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
                self._cache.commit()
        except Exception:
            self.close()
            raise

    @property
    def cache_enabled(self):
        return self.score_cache_enabled and self.query_cache_enabled

    @cache_enabled.setter
    def cache_enabled(self, enabled):
        self.score_cache_enabled = self.query_cache_enabled = bool(enabled)

    @staticmethod
    def _identity(model):
        return (getattr(model, "model_name", None) or getattr(model, "name", None)
                or f"{type(model).__module__}.{type(model).__qualname__}")

    @staticmethod
    def _key(*parts):
        return sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
                      .encode("utf-8")).hexdigest()

    def _check_open(self):
        if self._closed:
            raise RuntimeError("EvidenceRetriever is closed")

    def begin_snapshot(self):
        """Start a read snapshot; the caller checks its revision before reuse."""
        with self._lock:
            self._check_open()
            if not self._connection.in_transaction:
                self._connection.execute("BEGIN")

    def release_snapshot(self):
        """Release the read transaction while retaining resident corpus caches."""
        with self._lock:
            self._check_open()
            if self._connection.in_transaction:
                self._connection.rollback()

    def snapshot_revision(self):
        """Read schema/content revision within the current consistent snapshot.

        Content tracking is installed by the product adapter, never by this
        read-only engine. Recreate the engine when the returned tuple changes.
        """
        with self._lock:
            self._check_open()
            if not self._connection.in_transaction:
                raise RuntimeError("A read snapshot is required to read its revision")
            schema = self._connection.execute("PRAGMA schema_version").fetchone()[0]
            row = self._connection.execute(
                "SELECT revision FROM evidence_content_revision WHERE id=1").fetchone()
            if row is None:
                raise RuntimeError("Evidence content revision is not initialized")
            return schema, row[0]

    def close(self):
        with self._lock:
            if not self._closed:
                self._connection.close()
                if self._cache is not None:
                    self._cache.close()
                self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()

    def _tokenizer_trace(self, query, documents):
        # Inspect already loaded wrappers without touching lazy model properties.
        identity = (id(self.reranker), self._identity(self.reranker))
        if identity != self._trace_model_identity:
            self._trace_tokenizers = None
            self._trace_model_identity = identity
        if self._trace_tokenizers is None:
            objects, seen, tokenizer = [self.reranker], set(), None
            while objects:
                model = objects.pop()
                if id(model) in seen:
                    continue
                seen.add(id(model))
                attributes = getattr(model, "__dict__", {})
                tokenizer = attributes.get("tokenizer")
                if tokenizer is not None and hasattr(tokenizer, "to_str"):
                    from tokenizers import Tokenizer

                    actual = Tokenizer.from_str(tokenizer.to_str())
                    actual.no_padding()
                    full = Tokenizer.from_str(tokenizer.to_str())
                    full.no_truncation()
                    full.no_padding()
                    self._trace_tokenizers = (actual, full)
                    break
                objects.extend(attributes[key] for key in ("_model", "model")
                               if attributes.get(key) is not None)
        trace = {"external_character_limit": None, "input_characters": [len(d) for d in documents],
                 "tokenizer_available": self._trace_tokenizers is not None}
        if self._trace_tokenizers is not None:
            actual, full = self._trace_tokenizers
            encoded = actual.encode_batch([(query, document) for document in documents])
            complete = full.encode_batch([(query, document) for document in documents])
            trace.update(model_truncation=actual.truncation,
                         pair_tokens_before_truncation=[len(item.ids) for item in complete],
                         pair_tokens_supplied=[len(item.ids) for item in encoded],
                         truncated_documents=sum(len(a.ids) < len(b.ids)
                                                 for a, b in zip(encoded, complete, strict=True)))
        return trace

    def score(self, query, documents):
        """Score full document text; cache by model, query and text, never case id."""
        with self._lock:
            self._check_open()
            documents = list(documents)
            if not documents:
                self.last_score_trace = {"documents": 0, "cache_hits": 0, "model_documents": 0,
                                         "external_character_limit": None}
                return []
            identity = self._identity(self.reranker)
            keys = [self._key("evidence-score-v1", identity, query, text) for text in documents]
            values = {}
            if self.score_cache_enabled:
                values.update((key, self._scores[key]) for key in keys if key in self._scores)
                if self._cache is not None:
                    missing = list(dict.fromkeys(key for key in keys if key not in values))
                    for start in range(0, len(missing), 500):
                        batch = missing[start:start + 500]
                        placeholders = ",".join("?" for _ in batch)
                        values.update(self._cache.execute(
                            f"SELECT key,value FROM evidence_scores WHERE key IN ({placeholders})", batch))
            cache_hits = sum(key in values for key in keys)
            missing_texts = dict((key, text) for key, text in zip(keys, documents, strict=True)
                                 if key not in values)
            if missing_texts:
                predicted = list(self.reranker.score(query, list(missing_texts.values())))
                if (len(predicted) != len(missing_texts)
                        or not all(math.isfinite(float(value)) for value in predicted)):
                    raise ValueError("Invalid crossencoder scores")
                newly_scored = {key: float(value) for key, value in
                                zip(missing_texts, predicted, strict=True)}
                values.update(newly_scored)
                if self.score_cache_enabled and self._cache is not None:
                    with self._cache:
                        self._cache.executemany("INSERT OR REPLACE INTO evidence_scores VALUES (?,?)",
                                                newly_scored.items())
            if self.score_cache_enabled:
                self._scores.update(values)
            self.last_score_trace = {"model": identity, "documents": len(documents),
                                     "cache_hits": cache_hits, "model_documents": len(missing_texts),
                                     **self._tokenizer_trace(query, documents)}
            return [values[key] for key in keys]

    def _query_vector(self, query):
        key = self._key("evidence-query-v1", self._identity(self.provider), query)
        if self.query_cache_enabled:
            if key in self._query_vectors:
                return self._query_vectors[key]
            if self._cache is not None:
                row = self._cache.execute("SELECT vector FROM evidence_queries WHERE key=?", (key,)).fetchone()
                if row:
                    vector = np.frombuffer(row[0], dtype=np.float32).copy()
                    if not vector.size or not np.all(np.isfinite(vector)):
                        raise ValueError("Invalid cached query embedding")
                    self._query_vectors[key] = vector
                    return vector
        elif hasattr(self.provider, "_last_query_embedding"):
            self.provider._last_query_embedding = None
        vector = _normalized(self.provider.embed_query(query))
        if self.query_cache_enabled:
            self._query_vectors[key] = vector
            if self._cache is not None:
                with self._cache:
                    self._cache.execute("INSERT OR REPLACE INTO evidence_queries VALUES (?,?)",
                                        (key, vector.tobytes()))
        return vector

    def score_with(self, reranker, query, documents):
        """Use another model's score cache, then restore the active model.

        On success ``last_score_trace`` describes this call for an ensemble
        wrapper to copy. A failed call preserves the preceding trace.
        """
        with self._lock:
            self._check_open()
            original = (self.reranker, self._trace_model_identity, self._trace_tokenizers)
            previous_trace = self.last_score_trace
            try:
                self.reranker = reranker
                return self.score(query, documents)
            except Exception:
                self.last_score_trace = previous_trace
                raise
            finally:
                self.reranker, self._trace_model_identity, self._trace_tokenizers = original

    @staticmethod
    def _scope_clause(papers):
        if papers is None:
            return "", ()
        return f"AND u.paper_id IN ({','.join('?' for _ in papers)})", tuple(papers)

    def _hybrid_channels(self, query, papers, depth):
        clause, parameters = self._scope_clause(papers)
        expression = _search_expression(query)
        rows = self._connection.execute(
            "SELECT u.id,bm25(unit_search) AS score FROM unit_search "
            "JOIN retrieval_units u ON u.id=unit_search.rowid JOIN papers p ON p.id=u.paper_id "
            f"WHERE unit_search MATCH ? {clause} ORDER BY score LIMIT ?",
            (expression, *parameters, depth)).fetchall() if expression else []
        lexical = [replace(self._units[-row["id"]], score=float(row["score"]))
                   for row in rows if -row["id"] in self._units]
        vector = self._query_vector(query)
        matrix_key = (self.provider.name, vector.size, None if papers is None else tuple(papers))
        if matrix_key not in self._matrices:
            rows = self._connection.execute(
                "SELECT u.id,e.vector FROM unit_embeddings e "
                "JOIN retrieval_units u ON u.id=e.unit_id JOIN papers p ON p.id=u.paper_id "
                f"WHERE e.model=? AND e.dimension=? {clause}",
                (self.provider.name, vector.size, *parameters)).fetchall()
            rows = [row for row in rows if -row["id"] in self._units]
            matrix = (np.stack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows])
                      if rows else np.empty((0, vector.size), dtype=np.float32))
            if matrix.shape[1] != vector.size or not np.all(np.isfinite(matrix)):
                raise ValueError("Invalid indexed vectors")
            self._matrices[matrix_key] = ([-row["id"] for row in rows], matrix)
        identities, matrix = self._matrices[matrix_key]
        if not identities:
            scoped_units = [item for item in self._units.values()
                            if papers is None or item.paper_id in papers]
            if scoped_units:
                raise ValueError("No indexed vectors for requested embedding model and dimension")
            return {"bm25": lexical, "dense": []}
        values = matrix @ vector
        order = np.argsort(values)[::-1][:depth]
        dense = [replace(self._units[identities[index]], score=float(values[index])) for index in order]
        return {"bm25": lexical, "dense": dense}

    def retrieve(self, question: str, title: str = "", paper_ids=None, *, route_top=3,
                 depth=40, query_mode="question", rerank_mode="question", pool="hybrid",
                 scope_paper_ids=None, title_aliases: dict[int, str] | None = None,
                 ranker=None, supplement_depth=0):
        with self._lock:
            self._check_open()
            if route_top < 1 or depth < 1:
                raise ValueError("route_top and depth must be positive")
            if supplement_depth < 0:
                raise ValueError("supplement_depth must be nonnegative")
            if query_mode not in ("question", "display", "dual"):
                raise ValueError("Unknown query_mode")
            if rerank_mode not in ("question", "display"):
                raise ValueError("Unknown rerank_mode")
            if pool not in ("hybrid", "all_units", "all_blocks"):
                raise ValueError("Unknown candidate pool")
            started = time.perf_counter()
            scope = None if scope_paper_ids is None else set(scope_paper_ids)
            aliases = title_aliases or {}
            catalog = [(paper, aliases.get(paper, name)) for paper, name in self.catalog
                       if scope is None or paper in scope]
            if paper_ids is not None:
                papers = list(dict.fromkeys(paper for paper in paper_ids
                                            if scope is None or paper in scope))
                fallback = False
            else:
                papers = select_titles(title, catalog, route_top) if title else []
                fallback = not papers
                if not papers:
                    papers = None if scope is None else sorted(scope)
            effective_pool = ("hybrid" if pool != "hybrid" and fallback and paper_ids is None
                              else pool)
            selected_scope = None if papers is None else set(papers)
            has_units = bool(self._papers_with_units if selected_scope is None else
                             self._papers_with_units & selected_scope)
            has_blocks = bool(self._papers_with_blocks if selected_scope is None else
                              self._papers_with_blocks & selected_scope)
            if (effective_pool != "all_blocks" or supplement_depth) and has_blocks and not has_units:
                raise ValueError("No reading-unit index for selected papers")
            display = f"{question}\nPaper: {title}" if title else question
            queries = ([question, display] if query_mode == "dual" else
                       [question if query_mode == "question" else display])
            queries = list(dict.fromkeys(queries))
            rerank_query = question if rerank_mode == "question" else display
            routed = time.perf_counter()
            channels = {}
            if papers == [] or (not has_units and effective_pool != "all_blocks"):
                union = []
            elif effective_pool == "hybrid":
                for query in queries:
                    current = self._hybrid_channels(query, papers, depth)
                    prefix = ("question_" if query == question else "display_") if query_mode == "dual" else ""
                    channels.update((prefix + name, ranking) for name, ranking in current.items())
                union = fuse(channels.values())
            else:
                source = (self._units.values() if effective_pool == "all_units"
                          else self._blocks_by_id.values())
                union = [item for item in source if papers is None or item.paper_id in papers]
                channels[effective_pool] = list(union)
            candidate_time = time.perf_counter()
            scorer = self if ranker is None else ranker

            def rank_candidates(candidates):
                values = list(scorer.score(rerank_query, [item.text for item in candidates]))
                if (len(values) != len(candidates)
                        or not all(math.isfinite(float(value)) for value in values)):
                    raise ValueError("Invalid candidate ranker scores")
                ordered = [replace(candidates[index], score=float(values[index])) for index in
                           sorted(range(len(candidates)), key=lambda index: (-values[index], index))]
                return ordered, deepcopy(scorer.last_score_trace)

            ranked, ranking_trace = rank_candidates(union)
            initial_rank_time = time.perf_counter()
            supplement = None
            supplement_candidate_seconds = supplement_rank_seconds = 0.0
            if supplement_depth:
                protected = [item.passage_id for item in ranked[:8]]
                initial_ids = [item.passage_id for item in union]
                initial_ranked_ids = [item.passage_id for item in ranked]
                current = (self._hybrid_channels(display, papers, supplement_depth)
                           if papers != [] and has_units else {"bm25": [], "dense": []})
                channels.update(("supplement_" + name, hits) for name, hits in current.items())
                additional = fuse(current.values())
                seen = set(initial_ids)
                added = [item for item in additional if item.passage_id not in seen]
                union = [*union, *added]
                supplement_candidate_time = time.perf_counter()
                combined_ranked, supplement_trace = rank_candidates(union)
                supplement_rank_time = time.perf_counter()
                by_id = {item.passage_id: item for item in combined_ranked}
                protected_ids = set(protected)
                ranked = ([by_id[identity] for identity in protected]
                          + [item for item in combined_ranked if item.passage_id not in protected_ids])
                ranking_trace = {"initial": ranking_trace, "supplement": supplement_trace}
                supplement = {
                    "depth": supplement_depth, "query": display,
                    "initial_candidate_ids": initial_ids, "initial_ranked_ids": initial_ranked_ids,
                    "protected_seed_ids": protected,
                    "supplement_candidate_ids": [item.passage_id for item in additional],
                    "added_candidate_ids": [item.passage_id for item in added],
                    "combined_ranked_ids": [item.passage_id for item in combined_ranked],
                }
                supplement_candidate_seconds = supplement_candidate_time - initial_rank_time
                supplement_rank_seconds = supplement_rank_time - supplement_candidate_time
            finished = time.perf_counter()
            result = {"ranked": ranked[:20], "full_ranked": ranked,
                    "candidates": {"channels": channels, "union": union},
                    "selected_papers": [] if papers is None else papers,
                    "title_fallback": fallback,
                    "requested_pool": pool, "effective_pool": effective_pool,
                    "seconds": finished - started,
                    "timings": {"route": routed - started,
                                "candidates": candidate_time - routed + supplement_candidate_seconds,
                                "rerank": finished - candidate_time - supplement_candidate_seconds},
                    "queries": {"bm25": queries if query_mode == "dual" else queries[0],
                                "dense": queries if query_mode == "dual" else queries[0],
                                "rerank": rerank_query},
                    "rerank_trace": ranking_trace}
            if supplement is not None:
                result["supplement"] = supplement
                result["queries"]["supplement"] = {"bm25": display, "dense": display,
                                                    "rerank": rerank_query}
                result["timings"].update(initial_rerank=initial_rank_time - candidate_time,
                                         supplement_candidates=supplement_candidate_seconds,
                                         supplement_rerank=supplement_rank_seconds)
            return result
