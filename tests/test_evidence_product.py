from dataclasses import asdict
from pathlib import Path
import json
import sqlite3

from fastapi.testclient import TestClient
import pymupdf
import pytest

from paperanchor.agent import direct_question
from paperanchor.evidence_profile import (EVIDENCE_V2_CONFIG, EvidenceProfile,
                                          EvidenceProfileUnavailable)
from paperanchor.library import PaperLibrary, PassageHit, SourceSpan
from paperanchor.rag import answer_from_context
from paperanchor.web import QueryRequest, create_app
from paperanchor.unit_index import UnitStore
from paperanchor.evidence_policies import BGE, MINILM


def _context(paper_id=1):
    hit = PassageHit(21, paper_id, Path("paper.pdf"), "A title", 3,
                     "Only the returned source slice.", (1, 2, 3, 4), 1,
                     (SourceSpan(21, 3, (1, 2, 3, 4)),))
    chunk = {**asdict(hit), "evidence_id": "E1"}
    return {"chunks": [chunk], "text": "[E1] A title (page 3)\n" + hit.text,
            "reference_tokens": 20, "budget": 4096}


def _make_library(tmp_path):
    database = tmp_path / "library.sqlite3"
    library = PaperLibrary(database)
    for name in ("first", "second"):
        path = tmp_path / f"{name}.pdf"
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 90), f"Research evidence for {name} methods.")
            pdf.new_page()
            pdf.new_page()
            pdf.save(path)
        library.index_pdf(path)
    return database, library


def _model(system, prompt):
    if "Return only JSON" in system:
        return '{"each_paper": false}'
    return "The result is supported. [E1]"


def _local_test_profile(database):
    class Tokenizer:
        def count(self, text):
            return len(text.split())

    class Reranker:
        model_name = "fixture-reranker"

        def score(self, question, texts):
            return [float(len(text)) for text in texts]

    profile = EvidenceProfile(database)
    # These fixtures exercise corpus lifetime, scoping and rendering without
    # embedding inference; their retrieval pool is independent of the winner.
    profile.config.update(pool="all_units", strategy="dedup_parent")
    profile.reranker = Reranker()
    profile._tokenizer = Tokenizer()
    return profile


def test_context_is_sent_verbatim_and_citations_match_returned_slices():
    context = _context()
    prompts = []
    result = answer_from_context("Why?", context,
                                 lambda _, prompt: prompts.append(prompt) or "Yes. [E1]")
    assert prompts == ["Question: Why?\n\nEvidence:\n" + context["text"]]
    assert result.evidence[0].text == context["chunks"][0]["text"]
    assert result.evidence[0].source_spans == (SourceSpan(21, 3, (1, 2, 3, 4)),)
    assert result.evidence[0].page_number == 3
    assert result.status == "answered"
    assert answer_from_context("Why?", context, lambda *_: "[E2]").status == "invalid_citation"


def test_prepared_context_skips_legacy_retrieval_and_retains_empty_fallback():
    class UnexpectedRetriever:
        def search(self, *_args, **_kwargs):
            raise AssertionError("The prepared context must be the only retrieval source")

    result = direct_question(UnexpectedRetriever(), "Why?", _model,
                             prepare_context=lambda _: _context())
    assert result.answer.status == "answered"
    empty = lambda _: {"chunks": [], "text": ""}
    assert direct_question(UnexpectedRetriever(), "Why?", _model,
                           prepare_context=empty).answer.status == "no_evidence"
    assert direct_question(UnexpectedRetriever(), "Why?", _model,
                           prepare_context=empty,
                           allow_general_knowledge=True).basis == "model"


def test_profile_defaults_and_validation():
    assert QueryRequest(question="Why?").retrieval_profile == "legacy"
    assert EVIDENCE_V2_CONFIG == {
        "route_top": 1, "depth": 40, "query_mode": "question",
        "rerank_mode": "question", "pool": "hybrid",
        "strategy": "ranked_expansion", "protect": 8,
        "rerank_model": "BAAI/bge-reranker-base",
    }
    with pytest.raises(ValueError):
        QueryRequest(question="Why?", retrieval_profile="unknown")


def test_api_profile_respects_selected_papers_and_space(tmp_path, monkeypatch):
    database, library = _make_library(tmp_path)
    calls, prompts = [], []
    papers = library.list_papers()

    def prepare(_self, question, **kwargs):
        calls.append((question, kwargs))
        return _context(papers[0].id)

    monkeypatch.setattr(EvidenceProfile, "prepare", prepare)
    client = TestClient(create_app(database, generate=lambda s, p: prompts.append(p) or _model(s, p)))
    space_id = client.post("/api/spaces", json={"name": "Scoped"}).json()["id"]
    client.post(f"/api/spaces/{space_id}/papers/{papers[0].id}")
    response = client.post("/api/ask", json={
        "question": "Which method?", "visual_mode": "off", "space_id": space_id,
        "retrieval_profile": "evidence_v2", "paper_ids": [papers[0].id]})
    assert response.status_code == 200
    assert calls == [("Which method?", {"title": papers[0].title,
                      "paper_ids": (papers[0].id,), "scope_paper_ids": (papers[0].id,),
                      "profile_name": "evidence_v2"})]
    assert prompts[-1].endswith(_context()["text"])
    assert response.json()["evidence"][0]["text"] == _context()["chunks"][0]["text"]
    calls.clear()
    outside = client.post("/api/ask", json={
        "question": "Which method?", "visual_mode": "off", "space_id": space_id,
        "retrieval_profile": "evidence_v2", "paper_ids": [papers[1].id]})
    assert outside.json()["status"] == "insufficient_evidence"
    assert calls == []


def test_api_title_hint_and_explicit_profile_override_agent_setting(tmp_path, monkeypatch):
    database, _ = _make_library(tmp_path)
    calls = []
    monkeypatch.setenv("PAPERANCHOR_ANSWER_MODE", "agent")
    monkeypatch.setattr(EvidenceProfile, "prepare", lambda _self, _question, **kwargs:
                        calls.append(kwargs) or _context())
    client = TestClient(create_app(database, generate=_model))
    response = client.post("/api/ask", json={"question": "Which method?",
        "retrieval_profile": "evidence_v2", "title_hint": "  The public title  ",
        "visual_mode": "off"})
    assert response.status_code == 200
    assert calls[0]["title"] == "The public title"
    assert calls[0]["paper_ids"] is None
    assert len(calls[0]["scope_paper_ids"]) == 2
    response = client.post("/api/ask", json={"question": "Which method?",
        "retrieval_profile": "evidence_v2", "enable_decomposition": True,
        "visual_mode": "off"})
    assert response.status_code == 422


def test_api_reports_unavailable_optional_profile_without_changing_legacy(tmp_path, monkeypatch):
    database, _ = _make_library(tmp_path)

    def unavailable(*_args, **_kwargs):
        raise EvidenceProfileUnavailable("A cached model is unavailable")

    monkeypatch.setattr(EvidenceProfile, "prepare", unavailable)
    client = TestClient(create_app(database, generate=_model))
    response = client.post("/api/ask", json={"question": "Research methods?",
        "retrieval_profile": "evidence_v2", "visual_mode": "off"})
    assert response.status_code == 503
    assert response.json()["detail"] == "A cached model is unavailable"
    assert client.post("/api/ask", json={"question": "Research methods?",
                        "visual_mode": "off"}).status_code == 200


def test_overview_keeps_existing_route_when_profile_selected(tmp_path, monkeypatch):
    database, library = _make_library(tmp_path)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Collection overviews retain their existing retrieval")

    monkeypatch.setattr(EvidenceProfile, "prepare", unexpected)
    client = TestClient(create_app(database, generate=lambda s, p:
        '{"each_paper": true}' if "Return only JSON" in s else "First [E1], second [E2]."))
    response = client.post("/api/ask", json={"question": "Summarize each paper",
        "retrieval_profile": "evidence_v2", "visual_mode": "off"})
    assert response.status_code == 200
    assert {item["paper_id"] for item in response.json()["evidence"]} == {
        paper.id for paper in library.list_papers()}


def test_product_adapter_uses_shared_engine_assembly_and_fresh_corpus(tmp_path):
    from paperanchor.evidence_context import assemble_evidence
    from paperanchor.evidence_retrieval import EvidenceRetriever

    database, library = _make_library(tmp_path)
    UnitStore(library).refresh()
    chosen = library.list_papers()[0]

    profile = EvidenceProfile(database)
    assert profile.provider.local_files_only is True
    assert profile.reranker.local_files_only is True
    profile = _local_test_profile(database)
    kwargs = {"paper_ids": (chosen.id,), "scope_paper_ids": (chosen.id,)}
    context = profile.prepare("What evidence?", **kwargs)
    assert context["chunks"]
    with EvidenceRetriever(database, profile.provider, profile.reranker) as engine:
        result = engine.retrieve("What evidence?", **kwargs,
            **{key: profile.config[key] for key in (
                "route_top", "depth", "query_mode", "rerank_mode", "pool")})
        expected = assemble_evidence(result["ranked"], engine.blocks, profile._tokenizer,
            strategy=profile.config["strategy"], budget=4096, reranker=engine,
            query="What evidence?", protect=profile.config["protect"])
    assert context == expected
    assert all(chunk["paper_id"] == chosen.id for chunk in context["chunks"])
    library.delete_paper(chosen.id)
    assert profile.prepare("What evidence?", **kwargs)["chunks"] == []
    profile.close()


def test_missing_reference_model_fails_locally(tmp_path):
    database, _ = _make_library(tmp_path)
    profile = EvidenceProfile(database)
    with pytest.raises(EvidenceProfileUnavailable, match="locally cached reference tokenizer"):
        profile.prepare("What evidence?")


def test_resident_engine_ignores_annotation_writes_and_releases_read_lock(tmp_path):
    database, library = _make_library(tmp_path)
    UnitStore(library).refresh()
    chosen = library.list_papers()[0]
    profile = _local_test_profile(database)
    try:
        context = profile.prepare("What evidence?", paper_ids=(chosen.id,))
        original = profile._engine
        revision = profile._revision
        # DELETE journal mode would time out here if the idle engine still held
        # its read transaction. Annotation writes must not invalidate content.
        with sqlite3.connect(database, timeout=0.1) as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
            connection.execute("INSERT INTO annotations (paper_id,page_number,x0,y0,x1,y1,"
                               "body,author,created_at) VALUES (?,1,1,2,3,4,'note','user','now')",
                               (chosen.id,))
        assert profile.prepare("What evidence?", paper_ids=(chosen.id,)) == context
        assert profile._engine is original
        assert profile._revision == revision
    finally:
        profile.close()


@pytest.mark.parametrize("statement", [
    "UPDATE papers SET title=title||' revised'",
    "UPDATE passages SET text=text||' Added source sentence.'",
    "UPDATE retrieval_units SET text=text||' Added unit sentence.'",
    "UPDATE unit_sources SET position=position+1",
    "UPDATE unit_index_state SET config_hash='updated-config'",
    "UPDATE unit_embeddings SET vector=x'0000803f00000000'",
])
def test_resident_engine_refreshes_same_count_content_updates(tmp_path, statement):
    from paperanchor.unit_index import UnitSemanticIndex

    database, library = _make_library(tmp_path)
    store = UnitStore(library)
    store.refresh()
    UnitSemanticIndex(store)
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO unit_embeddings SELECT id,'fixture',2,"
                           "x'000000000000803f' FROM retrieval_units")
    chosen = library.list_papers()[0]
    profile = _local_test_profile(database)
    try:
        profile.prepare("What evidence?", paper_ids=(chosen.id,))
        original, revision = profile._engine, profile._revision
        with sqlite3.connect(database, timeout=0.1) as connection:
            connection.execute(statement)
        context = profile.prepare("What evidence?", paper_ids=(chosen.id,))
        assert profile._engine is not original
        assert profile._revision[1] > revision[1]
        assert context["chunks"]
    finally:
        profile.close()


def test_resident_engine_detects_new_vector_table_and_tracks_future_updates(tmp_path):
    from paperanchor.unit_index import UnitSemanticIndex

    database, library = _make_library(tmp_path)
    store = UnitStore(library)
    store.refresh()
    chosen = library.list_papers()[0]
    profile = _local_test_profile(database)
    try:
        profile.prepare("What evidence?", paper_ids=(chosen.id,))
        before_schema = profile._engine
        UnitSemanticIndex(store)
        profile.prepare("What evidence?", paper_ids=(chosen.id,))
        assert profile._engine is not before_schema
        before_vector = profile._engine
        with sqlite3.connect(database) as connection:
            connection.execute("INSERT INTO unit_embeddings SELECT id,'fixture',2,"
                               "x'000000000000803f' FROM retrieval_units")
        profile.prepare("What evidence?", paper_ids=(chosen.id,))
        assert profile._engine is not before_vector
    finally:
        profile.close()


def test_resident_engine_sees_newly_indexed_paper(tmp_path):
    database, library = _make_library(tmp_path)
    store = UnitStore(library)
    store.refresh()
    chosen = library.list_papers()[0]
    profile = _local_test_profile(database)
    try:
        profile.prepare("What evidence?", paper_ids=(chosen.id,))
        original = profile._engine
        path = tmp_path / "new-paper.pdf"
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 90), "New paper records the next important finding.")
            pdf.save(path)
        new_paper = library.index_pdf(path).paper_id
        store.refresh()
        context = profile.prepare("What evidence?", paper_ids=(new_paper,))
        assert profile._engine is not original
        assert context["chunks"]
        assert {chunk["paper_id"] for chunk in context["chunks"]} == {new_paper}
    finally:
        profile.close()


def test_content_revision_rolls_back_with_failed_writes(tmp_path):
    from paperanchor.evidence_revision import ensure_revision_tracking

    database, _ = _make_library(tmp_path)
    ensure_revision_tracking(database)
    with sqlite3.connect(database) as connection:
        before = connection.execute("SELECT revision FROM evidence_content_revision").fetchone()[0]
        connection.execute("UPDATE papers SET title='temporary'")
        assert connection.execute("SELECT revision FROM evidence_content_revision").fetchone()[0] > before
        connection.rollback()
        assert connection.execute("SELECT revision FROM evidence_content_revision").fetchone()[0] == before


def test_legacy_app_does_not_install_content_tracking(tmp_path):
    database, _ = _make_library(tmp_path)
    client = TestClient(create_app(database, generate=_model))
    assert client.post("/api/ask", json={"question": "Research methods?",
        "visual_mode": "off"}).status_code == 200
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE "
                                  "name='evidence_content_revision'").fetchone() is None


def _write_optimized(profile, tmp_path, *, aliases=None, **config):
    path = tmp_path / "optimized.json"
    payload = {"schema_version": 1,
               "config": {**profile.config, "candidate_policy": "bge", "expansion_policy": "minilm",
                          "supplement_depth": 0, **config},
               "title_aliases": aliases or []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    profile.optimized_path = path
    return path


def test_profiles_are_available_without_loading_models_and_optimized_is_optional(tmp_path):
    database, library = _make_library(tmp_path)
    profile = EvidenceProfile(database)
    profile.optimized_path = tmp_path / "missing.json"
    status = profile.availability()
    assert status["profiles"] == ["legacy", "evidence_v2", "evidence_minilm"]
    assert "not been installed" in status["errors"]["evidence_optimized"]
    assert all(model._model is None and model.local_files_only for model in profile.models.values())
    _write_optimized(profile, tmp_path)
    assert "evidence_optimized" in profile.availability()["profiles"]
    with sqlite3.connect(database) as connection:
        paper_id, file_hash = connection.execute("SELECT id,file_hash FROM papers LIMIT 1").fetchone()
    _write_optimized(profile, tmp_path, aliases=[{
        "paper_id": paper_id, "file_hash": file_hash, "title": "Verified public title"}])
    assert "evidence_optimized" in profile.availability()["profiles"]
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE papers SET file_hash=? WHERE id=?", ("f" * 64, paper_id))
    status = profile.availability()
    assert "evidence_optimized" not in status["profiles"]
    assert "does not match" in status["errors"]["evidence_optimized"]
    assert "evidence_v2" in status["profiles"] and "evidence_minilm" in status["profiles"]


def test_profile_switch_reuses_corpus_and_keeps_model_scores_separate(tmp_path):
    database, library = _make_library(tmp_path)
    UnitStore(library).refresh()
    profile = _local_test_profile(database)

    class Scorer:
        def __init__(self, name, direction):
            self.model_name, self.direction, self.calls = name, direction, 0

        def score(self, question, texts):
            self.calls += 1
            return [self.direction * float(len(text)) for text in texts]

    profile.reranker = Scorer("fixture-bge", 1)
    profile.models[MINILM] = Scorer("fixture-minilm", -1)
    selected = tuple(paper.id for paper in library.list_papers())
    try:
        bge = profile.prepare("What evidence?", paper_ids=selected)
        engine = profile._engine
        mini = profile.prepare("What evidence?", paper_ids=selected, profile_name="evidence_minilm")
        assert profile._engine is engine
        assert profile.reranker.calls == 1 and profile.models[MINILM].calls == 1
        assert bge["chunks"][0]["paper_id"] != mini["chunks"][0]["paper_id"]
        assert profile.prepare("What evidence?", paper_ids=selected) == bge
        assert profile.reranker.calls == 1
    finally:
        profile.close()


def test_optimized_separate_policies_and_bound_alias_keep_source_titles(tmp_path):
    database, library = _make_library(tmp_path)
    UnitStore(library).refresh()
    chosen = library.list_papers()[0]
    with sqlite3.connect(database) as connection:
        file_hash = connection.execute("SELECT file_hash FROM papers WHERE id=?", (chosen.id,)).fetchone()[0]
        connection.execute("UPDATE retrieval_units SET text='Research evidence'")
    profile = _local_test_profile(database)

    class Scorer:
        def __init__(self, name):
            self.model_name, self.calls = name, []

        def score(self, question, texts):
            self.calls.append(list(texts))
            return [float(len(text)) for text in texts]

    profile.reranker = Scorer("fixture-bge")
    profile.models[MINILM] = Scorer("fixture-minilm")
    _write_optimized(profile, tmp_path, strategy="ranked_expansion", protect=1,
        aliases=[{"paper_id": chosen.id, "file_hash": file_hash, "title": "Verified replacement title"}])
    try:
        context = profile.prepare("What evidence?", title="Verified replacement title",
                                  profile_name="evidence_optimized")
        assert profile.reranker.calls and profile.models[MINILM].calls
        assert context["chunks"]
        assert {chunk["paper_id"] for chunk in context["chunks"]} == {chosen.id}
        assert {chunk["paper_title"] for chunk in context["chunks"]} == {chosen.title}
        assert any(chunk["context_kind"] == "parent" for chunk in context["chunks"])
    finally:
        profile.close()


def test_invalid_optimized_hash_does_not_disable_other_profiles(tmp_path):
    database, library = _make_library(tmp_path)
    UnitStore(library).refresh()
    chosen = library.list_papers()[0]
    profile = _local_test_profile(database)
    profile.models[MINILM] = profile.reranker
    _write_optimized(profile, tmp_path,
        aliases=[{"paper_id": chosen.id, "file_hash": "0" * 64, "title": "Wrong binding"}])
    try:
        with pytest.raises(EvidenceProfileUnavailable, match="does not match"):
            profile.prepare("What evidence?", paper_ids=(chosen.id,), profile_name="evidence_optimized")
        assert profile.prepare("What evidence?", paper_ids=(chosen.id,))["chunks"]
        assert profile.prepare("What evidence?", paper_ids=(chosen.id,), profile_name="evidence_minilm")["chunks"]
    finally:
        profile.close()


@pytest.mark.parametrize("name", ["evidence_v2", "evidence_minilm", "evidence_optimized"])
def test_api_dispatches_named_profiles(tmp_path, monkeypatch, name):
    database, library = _make_library(tmp_path)
    chosen = library.list_papers()[0]
    calls = []
    monkeypatch.setattr(EvidenceProfile, "prepare", lambda _self, question, **kwargs:
                        calls.append(kwargs) or _context(chosen.id))
    client = TestClient(create_app(database, generate=_model))
    response = client.post("/api/ask", json={"question": "Research methods?",
        "paper_ids": [chosen.id], "retrieval_profile": name, "visual_mode": "off"})
    assert response.status_code == 200
    assert calls[0]["profile_name"] == name


def test_api_reports_available_profiles_without_installing_revision(tmp_path, monkeypatch):
    from paperanchor import evidence_profile

    monkeypatch.setattr(evidence_profile, "OPTIMIZED_PROFILE_PATH", tmp_path / "missing.json")
    database, _ = _make_library(tmp_path)
    client = TestClient(create_app(database, generate=_model))
    status = client.get("/api/semantic/status").json()
    assert status["retrieval_profiles"] == ["legacy", "evidence_v2", "evidence_minilm"]
    assert "not been installed" in status["retrieval_profile_errors"]["evidence_optimized"]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE "
                                  "name='evidence_content_revision'").fetchone() is None
