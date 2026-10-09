from pathlib import Path
from dataclasses import replace
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from paperanchor.evidence_retrieval import EvidenceRetriever, select_titles


class Provider:
    name = "fixture-vectors"

    def __init__(self):
        self.calls = []

    def embed_query(self, query):
        self.calls.append(query)
        return np.array([1, 0], dtype=np.float32)


class Reranker:
    model_name = "fixture-reranker"

    def __init__(self):
        self.calls = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [float(len(text)) for text in texts]


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "corpus.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE papers (id INTEGER PRIMARY KEY,file_path TEXT,title TEXT);
            CREATE TABLE passages (id INTEGER PRIMARY KEY,paper_id INTEGER,page_number INTEGER,
                                   block_number INTEGER,text TEXT,x0 REAL,y0 REAL,x1 REAL,y1 REAL);
            CREATE TABLE retrieval_units (id INTEGER PRIMARY KEY,paper_id INTEGER,page_number INTEGER,
                                          section TEXT,text TEXT);
            CREATE TABLE unit_sources (unit_id INTEGER,passage_id INTEGER,position INTEGER);
            CREATE TABLE unit_embeddings (unit_id INTEGER,model TEXT,dimension INTEGER,vector BLOB,
                                          PRIMARY KEY(unit_id,model));
            CREATE VIRTUAL TABLE unit_search USING fts5(section,text,tokenize='porter unicode61');
        """)
        for paper, title in [(1, "Alpha Findings"), (2, "Beta Findings"), (3, "Other Study")]:
            connection.execute("INSERT INTO papers VALUES (?,?,?)", (paper, f"{paper}.pdf", title))
        for identity, paper, text, vector in [
            (1, 1, "Alpha findings first unit.", [1, 0]),
            (2, 1, "Second important alpha result.", [0.8, 0.6]),
            (3, 2, "Beta findings show a much longer result.", [0, 1]),
            (4, 3, "Other study findings.", [-1, 0]),
        ]:
            connection.execute("INSERT INTO passages VALUES (?,?,1,?,?,1,2,3,4)",
                               (identity, paper, identity, text + " Source continuation."))
            connection.execute("INSERT INTO retrieval_units VALUES (?,?,1,'Results',?)",
                               (identity, paper, text))
            connection.execute("INSERT INTO unit_sources VALUES (?,?,0)", (identity, identity))
            connection.execute("INSERT INTO unit_search(rowid,section,text) VALUES (?,'Results',?)",
                               (identity, text))
            connection.execute("INSERT INTO unit_embeddings VALUES (?, ?,2,?)",
                               (identity, Provider.name, np.asarray(vector, dtype=np.float32).tobytes()))
    return path


def identities(items):
    return [item.passage_id for item in items]


def test_public_title_route_explicit_selection_and_scope(database):
    with EvidenceRetriever(database, Provider(), Reranker()) as engine:
        assert select_titles("Findings", engine.catalog) == [1, 2]
        result = engine.retrieve("findings?", "Alpha Findings", route_top=1)
        assert result["selected_papers"] == [1]
        assert {item.paper_id for item in result["ranked"]} == {1}
        explicit = engine.retrieve("findings?", "Alpha Findings", [2], scope_paper_ids=[1, 2])
        assert explicit["selected_papers"] == [2]
        assert {item.paper_id for item in explicit["ranked"]} == {2}
        scoped = engine.retrieve("findings?", "Alpha Findings", scope_paper_ids=[2], route_top=1)
        assert scoped["selected_papers"] == [2]
        empty = engine.retrieve("findings?", "Alpha Findings", [1], scope_paper_ids=[2])
        assert empty["ranked"] == []
        assert engine.retrieve("findings?", paper_ids=[])["ranked"] == []


def test_no_title_overlap_falls_back_within_allowed_scope(database):
    with EvidenceRetriever(database, Provider(), Reranker()) as engine:
        result = engine.retrieve("result", "Unmatched xyz")
        assert result["title_fallback"] is True
        assert result["selected_papers"] == []
        assert {item.paper_id for item in result["ranked"]} == {1, 2, 3}
        result = engine.retrieve("result", "Unmatched xyz", scope_paper_ids=[1, 3])
        assert result["selected_papers"] == [1, 3]
        assert {item.paper_id for item in result["ranked"]} == {1, 3}


def test_dual_queries_have_independent_channels_and_stable_ties(database):
    reranker = Reranker()
    reranker.score = lambda query, texts: [1.0] * len(texts)
    with EvidenceRetriever(database, Provider(), reranker) as engine:
        result = engine.retrieve("Which result?", "Alpha Findings", query_mode="dual", depth=1)
        assert set(result["candidates"]["channels"]) == {
            "question_bm25", "question_dense", "display_bm25", "display_dense"}
        assert all(len(items) <= 1 for items in result["candidates"]["channels"].values())
        assert result["queries"]["rerank"] == "Which result?"
        assert result["queries"]["dense"] == ["Which result?", "Which result?\nPaper: Alpha Findings"]
        assert identities(result["full_ranked"]) == identities(result["candidates"]["union"])


def test_all_units_and_blocks_preserve_source_identity(database):
    with EvidenceRetriever(database, Provider(), Reranker()) as engine:
        units = engine.retrieve("query", paper_ids=[1], pool="all_units")
        blocks = engine.retrieve("query", paper_ids=[1], pool="all_blocks")
        assert {hit.passage_id for hit in units["ranked"]} == {-1, -2}
        assert {hit.passage_id for hit in blocks["ranked"]} == {1, 2}
        assert all(hit.source_spans[0].passage_id == -hit.passage_id for hit in units["ranked"])
        assert all(hit.source_spans[0].passage_id == hit.passage_id for hit in blocks["ranked"])
        assert units["ranked"][0].text.startswith("Results\n")
        assert blocks["ranked"][0].text.endswith("Source continuation.")


@pytest.mark.parametrize("pool", ["all_units", "all_blocks"])
@pytest.mark.parametrize("title", ["", "unmatched xyzw"])
def test_unmatched_whole_paper_request_uses_bounded_hybrid_pool(database, pool, title):
    with EvidenceRetriever(database, Provider(), Reranker()) as engine:
        result = engine.retrieve("findings", title, pool=pool, depth=1)
        assert result["requested_pool"] == pool
        assert result["effective_pool"] == "hybrid"
        assert set(result["candidates"]["channels"]) == {"bm25", "dense"}
        assert len(result["candidates"]["union"]) <= 2
        scoped = engine.retrieve("findings", title, pool=pool, depth=1, scope_paper_ids=[2])
        assert scoped["effective_pool"] == "hybrid"
        assert {item.paper_id for item in scoped["ranked"]} == {2}
        selected = engine.retrieve("findings", title, [1], pool=pool, depth=1)
        assert selected["effective_pool"] == pool
        assert len(selected["candidates"]["union"]) == 2


def test_score_cache_reuses_text_across_runs_but_separates_model_and_query(database, tmp_path):
    cache = tmp_path / "score-cache.sqlite3"
    first = Reranker()
    with EvidenceRetriever(database, Provider(), first, score_cache_path=cache) as engine:
        assert engine.score("q", ["same", "same", "other"]) == [4.0, 4.0, 5.0]
        assert first.calls == [("q", ["same", "other"])]
        assert engine.score("q", ["other", "same"]) == [5.0, 4.0]
        assert engine.last_score_trace["cache_hits"] == 2
    resumed = Reranker()
    with EvidenceRetriever(database, Provider(), resumed, score_cache_path=cache) as engine:
        assert engine.score("q", ["same"]) == [4.0]
        assert resumed.calls == []
        engine.score("other query", ["same"])
        assert len(resumed.calls) == 1
        resumed.model_name = "different-model"
        engine.score("q", ["same"])
        assert len(resumed.calls) == 2


def test_timing_mode_bypasses_score_and_query_result_caches(database):
    provider, reranker = Provider(), Reranker()
    with EvidenceRetriever(database, provider, reranker) as engine:
        engine.retrieve("query", paper_ids=[1])
        engine.retrieve("query", paper_ids=[1])
        assert len(provider.calls) == len(reranker.calls) == 1
        engine.cache_enabled = False
        for _ in range(2):
            engine.retrieve("query", paper_ids=[1])
        assert len(provider.calls) == len(reranker.calls) == 3
        assert engine.last_score_trace["cache_hits"] == 0


def test_full_text_scoring_reports_actual_model_truncation(database):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.enable_truncation(max_length=12)
    reranker = Reranker()
    reranker._model = SimpleNamespace(model=SimpleNamespace(tokenizer=tokenizer))
    text = "long document " * 300
    with EvidenceRetriever(database, Provider(), reranker) as engine:
        engine.score("query", [text])
        trace = engine.last_score_trace
    assert reranker.calls[0][1] == [text]
    assert trace["external_character_limit"] is None
    assert trace["pair_tokens_supplied"] == [12]
    assert trace["pair_tokens_before_truncation"] == [601]
    assert trace["truncated_documents"] == 1


def test_missing_vectors_and_invalid_scores_are_visible_failures(database):
    provider = Provider()
    provider.name = "absent-model"
    with EvidenceRetriever(database, provider, Reranker()) as engine:
        with pytest.raises(ValueError, match="No indexed vectors"):
            engine.retrieve("query")
    reranker = Reranker()
    reranker.score = lambda query, texts: [float("nan")] * len(texts)
    with EvidenceRetriever(database, Provider(), reranker) as engine:
        with pytest.raises(ValueError, match="Invalid crossencoder"):
            engine.score("q", ["text"])


def test_missing_unit_index_is_distinguished_from_empty_selection(database):
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM retrieval_units WHERE paper_id=1")
    provider = Provider()
    with EvidenceRetriever(database, provider, Reranker()) as engine:
        for pool in ("hybrid", "all_units"):
            with pytest.raises(ValueError, match="No reading-unit index"):
                engine.retrieve("findings", paper_ids=[1], pool=pool)
        assert engine.retrieve("findings", paper_ids=[1], pool="all_blocks")["ranked"]
        assert engine.retrieve("findings", paper_ids=[999])["ranked"] == []
        assert provider.calls == []


def test_read_only_snapshot_and_lifecycle(database):
    before = database.read_bytes()
    engine = EvidenceRetriever(database, Provider(), Reranker())
    engine.retrieve("query", paper_ids=[1])
    engine.close()
    engine.close()
    assert database.read_bytes() == before
    with pytest.raises(RuntimeError, match="closed"):
        engine.retrieve("query")
    with pytest.raises(ValueError, match="separate"):
        EvidenceRetriever(database, Provider(), Reranker(), score_cache_path=database)


def test_snapshot_release_allows_writes_and_detects_new_content_revision(database):
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE evidence_content_revision (id INTEGER PRIMARY KEY, revision INTEGER)")
        connection.execute("INSERT INTO evidence_content_revision VALUES (1,0)")
    with EvidenceRetriever(database, Provider(), Reranker()) as engine:
        first = engine.snapshot_revision()
        engine.begin_snapshot()
        assert engine.snapshot_revision() == first
        engine.release_snapshot()
        engine.release_snapshot()
        with pytest.raises(RuntimeError, match="snapshot is required"):
            engine.snapshot_revision()
        with sqlite3.connect(database, timeout=0.1) as connection:
            connection.execute("UPDATE evidence_content_revision SET revision=revision+1 WHERE id=1")
        engine.begin_snapshot()
        assert engine.snapshot_revision() == (first[0], first[1] + 1)
        engine.release_snapshot()
        with sqlite3.connect(database, timeout=0.1) as connection:
            connection.execute("CREATE TABLE schema_change (id INTEGER)")
        engine.begin_snapshot()
        assert engine.snapshot_revision()[0] > first[0]


def test_title_aliases_only_change_catalog_routing_and_respect_scope(database):
    before = database.read_bytes()
    with EvidenceRetriever(database, Provider(), Reranker()) as engine:
        original_catalog = list(engine.catalog)
        ordinary = engine.retrieve("findings", "Alpha Findings", route_top=1)
        aliases = {1: "Corrected Different Title", 2: "Alpha Findings"}
        routed = engine.retrieve("findings", "Alpha Findings", route_top=1, title_aliases=aliases)
        assert ordinary["selected_papers"] == [1]
        assert routed["selected_papers"] == [2]
        assert {item.paper_title for item in routed["ranked"]} == {"Beta Findings"}
        assert engine.catalog == original_catalog
        selected = engine.retrieve("findings", "Alpha Findings", [1], title_aliases=aliases)
        assert selected["selected_papers"] == [1]
        scoped = engine.retrieve("findings", "Alpha Findings", scope_paper_ids=[1], title_aliases=aliases)
        assert {item.paper_id for item in scoped["ranked"]} == {1}
    assert database.read_bytes() == before


def test_paper_hashes_are_read_from_the_same_snapshot(database):
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE papers ADD COLUMN file_hash TEXT")
        connection.execute("UPDATE papers SET file_hash='original-hash' WHERE id=1")
    with EvidenceRetriever(database, Provider(), Reranker()) as engine:
        assert engine.paper_hashes[1] == "original-hash"


def test_score_with_uses_model_cache_and_restores_reranker_and_tokenizer_state(database):
    first, alternate = Reranker(), Reranker()
    alternate.model_name = "alternate-model"
    alternate.score = lambda query, texts: [float(len(text) * 2) for text in texts]
    with EvidenceRetriever(database, Provider(), first) as engine:
        assert engine.score("query", ["body"]) == [4.0]
        tokenizer_state = engine._trace_model_identity, engine._trace_tokenizers
        assert engine.score_with(alternate, "query", ["body"]) == [8.0]
        assert engine.reranker is first
        assert engine.last_score_trace["model"] == "alternate-model"
        assert (engine._trace_model_identity, engine._trace_tokenizers) == tokenizer_state
        assert engine.score_with(alternate, "query", ["body"]) == [8.0]
        assert engine.last_score_trace["cache_hits"] == 1
        assert engine.score("query", ["body"]) == [4.0]
        assert engine.last_score_trace["cache_hits"] == 1
        assert len(first.calls) == 1
        before_trace = engine.last_score_trace
        alternate.model_name = "broken-model"
        alternate.score = lambda query, texts: [float("nan") for _ in texts]
        with pytest.raises(ValueError, match="Invalid crossencoder"):
            engine.score_with(alternate, "query", ["body"])
        assert engine.reranker is first
        assert engine.last_score_trace is before_trace
        assert (engine._trace_model_identity, engine._trace_tokenizers) == tokenizer_state


def test_custom_ranker_receives_identical_union_and_controls_stable_order(database):
    original = Reranker()

    class Ranker:
        last_score_trace = {"policy": "fixture-equal-scores"}

        def score(self, query, documents):
            self.query, self.documents = query, list(documents)
            return [1.0] * len(documents)

    ranker = Ranker()
    with EvidenceRetriever(database, Provider(), original) as engine:
        result = engine.retrieve("findings", paper_ids=[1, 2], ranker=ranker)
        assert ranker.documents == [item.text for item in result["candidates"]["union"]]
        assert ranker.query == "findings"
        assert identities(result["full_ranked"]) == identities(result["candidates"]["union"])
        assert result["rerank_trace"] == {"policy": "fixture-equal-scores"}
        assert original.calls == []
        ranker.score = lambda query, texts: []
        with pytest.raises(ValueError, match="Invalid candidate ranker"):
            engine.retrieve("findings", paper_ids=[1], ranker=ranker)


def test_supplement_protects_first_eight_and_saves_both_ranking_traces(database, monkeypatch):
    class Ranker:
        def __init__(self):
            self.calls, self.last_score_trace = [], {}

        def score(self, query, documents):
            self.calls.append((query, list(documents)))
            phase = len(self.calls)
            self.last_score_trace.update(phase=phase, documents=len(documents))
            numbers = [int(text.split()[-1]) for text in documents]
            return [-number if phase == 1 else number for number in numbers]

    ranker, channel_calls = Ranker(), []
    with EvidenceRetriever(database, Provider(), Reranker()) as engine:
        hits = {number: replace(engine._units[-1], passage_id=-number,
                                text=f"document {number}") for number in range(1, 27)}

        def channels(query, papers, depth):
            channel_calls.append((query, papers, depth))
            selected = range(1, 13) if query == "question" else range(5, 27)
            return {"bm25": [hits[n] for n in selected], "dense": [hits[n] for n in selected]}

        monkeypatch.setattr(engine, "_hybrid_channels", channels)
        result = engine.retrieve("question", "Alpha Findings", [1], ranker=ranker,
                                 depth=12, supplement_depth=22)
        assert channel_calls == [("question", [1], 12), ("question\nPaper: Alpha Findings", [1], 22)]
        assert identities(result["ranked"]) == [-n for n in [*range(1, 9), *range(26, 14, -1)]]
        assert len(result["full_ranked"]) == len(set(identities(result["full_ranked"]))) == 26
        assert identities(result["candidates"]["union"]) == list(range(-1, -27, -1))
        assert set(result["candidates"]["channels"]) == {
            "bm25", "dense", "supplement_bm25", "supplement_dense"}
        assert result["rerank_trace"] == {"initial": {"phase": 1, "documents": 12},
                                          "supplement": {"phase": 2, "documents": 26}}
        assert result["supplement"]["protected_seed_ids"] == list(range(-1, -9, -1))
        assert result["supplement"]["added_candidate_ids"] == list(range(-13, -27, -1))
        assert result["supplement"]["combined_ranked_ids"] == list(range(-26, 0))
        assert result["queries"]["supplement"]["rerank"] == "question"


@pytest.mark.parametrize("model", ["BAAI/bge-reranker-base", "Xenova/ms-marco-MiniLM-L-6-v2"])
def test_extension_defaults_leave_existing_fake_model_outputs_identical(database, model):
    reranker = Reranker()
    reranker.model_name = model
    with EvidenceRetriever(database, Provider(), reranker) as engine:
        engine.cache_enabled = False
        ordinary = engine.retrieve("findings", "Alpha Findings")
        explicit = engine.retrieve("findings", "Alpha Findings", title_aliases=None,
                                   ranker=None, supplement_depth=0)
        for result in (ordinary, explicit):
            result.pop("seconds")
            result.pop("timings")
        assert ordinary == explicit


def test_supplement_empty_selection_and_invalid_depth(database):
    provider, reranker = Provider(), Reranker()
    with EvidenceRetriever(database, provider, reranker) as engine:
        empty = engine.retrieve("query", "Alpha Findings", [], supplement_depth=5)
        assert empty["ranked"] == []
        assert empty["supplement"]["protected_seed_ids"] == []
        assert provider.calls == reranker.calls == []
        with pytest.raises(ValueError, match="supplement_depth"):
            engine.retrieve("query", supplement_depth=-1)
