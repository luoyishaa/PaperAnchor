from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.qasper_benchmark.retrieve import (
    CachedEmbeddingProvider, CachedReranker, FrozenUnitStore, ReferenceTokenizer,
    assemble_context, database_fingerprint, fixed_candidates, index_metadata, main,
    run_cases, run_fingerprints,
)
from paperanchor.library import PassageHit, SourceSpan


def hit(identity, text="Evidence is present."):
    return PassageHit(identity, 7, Path("paper.pdf"), "Paper", 2, text,
                      (1, 2, 3, 4), 0.5, (SourceSpan(90, 2, (1, 2, 3, 4)),))


class Lexical:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, query, limit, **scope):
        self.calls.append((query, limit, scope))
        return self.hits[:limit]


class Dense(Lexical):
    def count(self, model):
        return len(self.hits)

    def search(self, query, provider, *, limit, **scope):
        return super().search(query, limit, **scope)


class Reranker:
    def __init__(self):
        self.calls = []

    def score(self, query, texts):
        self.calls.append((query, texts))
        return list(range(len(texts)))


@pytest.fixture
def tokenizer(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    raw = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    raw.save(str(path))
    return ReferenceTokenizer(path)


def test_union_is_shared_and_depth_independent_of_topk(tokenizer):
    lexical = Lexical([hit(i, f"Lexical {i}.") for i in range(1, 51)])
    dense = Dense([hit(i, f"Dense {i}.") for i in range(30, 81)])
    reranker = Reranker()
    cases = [{"case_id": "q1", "split": "dev", "question": "Which finding?",
              "title": "Target title", "paper_id": 987654}]
    output = run_cases(cases, {"blocks": (lexical, dense)}, SimpleNamespace(name="fake"),
                       tokenizer, top_k=1, reranker=reranker)
    assert lexical.calls == [("Which finding?\nPaper: Target title", 40, {})]
    assert dense.calls == lexical.calls
    assert len(output) == 4
    assert all(len(record["ranked"]) == 1 for record in output)
    union = output[2]["candidates"]["union"]
    assert len(union) == 69
    assert reranker.calls[0][1] == [item["text"] for item in union]
    assert "987654" not in reranker.calls[0][0]
    assert output[3]["candidates"] == output[2]["candidates"]


def test_rrf_ranking_and_ties_are_deterministic():
    result = fixed_candidates("query", Lexical([hit(2), hit(1)]),
                              Dense([hit(1), hit(2)]), SimpleNamespace(name="fake"))
    assert [h.passage_id for h in result["rankings"]["hybrid"]] == [1, 2]
    assert result["rankings"]["hybrid"][0].score == 1 / 61 + 1 / 62


def test_selected_methods_preserve_candidates_and_resume(tokenizer):
    lexical, dense, reranker = Lexical([hit(1)]), Dense([hit(2)]), Reranker()
    cases = [{"case_id": "q", "split": "test", "display_question": "query"}]
    options = dict(split="test", allow_test=True, reranker=reranker, methods=["bm25", "rerank"])
    records = run_cases(cases, {"units": (lexical, dense)}, SimpleNamespace(name="fake"),
                        tokenizer, **options)
    assert [record["method"] for record in records] == ["bm25", "rerank"]
    assert len(records[1]["candidates"]["union"]) == 2
    resumed = run_cases(cases, {"units": (lexical, dense)}, SimpleNamespace(name="fake"),
                        tokenizer, previous_records=records, **options)
    assert resumed == records
    assert len(lexical.calls) == len(dense.calls) == len(reranker.calls) == 1
    with pytest.raises(ValueError, match="reranker"):
        run_cases(cases, {}, None, tokenizer, methods=["rerank"])


def test_no_vectors_is_failure_not_lexical_fallback(tokenizer):
    records = run_cases([{"case_id": "q", "split": "dev", "display_question": "query"}],
                        {"units": (Lexical([hit(-1)]), Dense([]))},
                        SimpleNamespace(name="missing"), tokenizer, reranker=Reranker())
    assert records[0]["status"] == "ok"
    assert [r["status"] for r in records[1:]] == ["failed"] * 3
    assert "No indexed vectors" in records[1]["failure"]["message"]
    assert all("ranked" not in record for record in records[1:])


def test_budget_saves_exact_text_and_provenance(tokenizer):
    original = hit(-9, "First sentence. Second sentence has the answer.")
    prefix = "[E1] Paper (page 2)\nFirst sentence."
    result = assemble_context([original], tokenizer, tokenizer.count(prefix))
    chunk = result["chunks"][0]
    assert result["text"] == prefix
    assert result["reference_tokens"] <= result["budget"]
    assert chunk["text"] == "First sentence."
    assert chunk["truncated"] is True
    assert chunk["retrieved_char_range"] == [0, len("First sentence.")]
    assert chunk["source_spans"][0]["passage_id"] == 90
    assert chunk["passage_id"] == -9
    assert "answer" not in chunk["text"]


def test_budget_accounts_for_headers_and_stops_at_unfitting_sentence(tokenizer):
    result = assemble_context([hit(1, "No sentence boundary in this long content"), hit(2)],
                              tokenizer, 1)
    assert result["chunks"] == []
    assert result["text"] == ""


def test_default_split_and_explicit_auxiliary_scope(tokenizer):
    lexical, dense = Lexical([hit(1)]), Dense([hit(1)])
    cases = [{"case_id": "dev", "split": "dev", "display_question": "q",
              "scope": "single_paper_auxiliary", "paper_id": 7},
             {"case_id": "test", "split": "test", "display_question": "secret"}]
    records = run_cases(cases, {"units": (lexical, dense)}, SimpleNamespace(name="fake"), tokenizer)
    assert {record["case_id"] for record in records} == {"dev"}
    assert lexical.calls == [("q", 40, {"paper_id": 7})]
    with pytest.raises(ValueError, match="allow_test"):
        run_cases(cases, {}, None, tokenizer, split="test")


def test_missing_reference_tokenizer_keeps_retrieval_but_marks_context_failed():
    records = run_cases([{"split": "dev", "display_question": "q"}],
                        {"blocks": (Lexical([hit(1)]), Dense([hit(1)]))},
                        SimpleNamespace(name="fake"), None,
                        tokenizer_error="missing tokenizer.json")
    assert all(record["status"] == "context_failed" for record in records)
    assert all(record["ranked"] for record in records)
    assert all("missing tokenizer.json" in record["context_failure"]["message"] for record in records)


def test_cached_models_use_local_files_only(monkeypatch, tmp_path):
    import fastembed
    import fastembed.rerank.cross_encoder

    calls = []

    def missing(**kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("model cache absent")

    monkeypatch.setattr(fastembed, "TextEmbedding", missing)
    monkeypatch.setattr(fastembed.rerank.cross_encoder, "TextCrossEncoder", missing)
    with pytest.raises(FileNotFoundError):
        CachedEmbeddingProvider("fake", tmp_path).embed_query("q")
    with pytest.raises(FileNotFoundError):
        CachedReranker("fake", tmp_path).score("q", ["text"])
    assert len(calls) == 2
    assert all(call["local_files_only"] is True for call in calls)


def test_invalid_reranker_scores_are_failures():
    broken = SimpleNamespace(score=lambda query, texts: [float("nan")])
    result = fixed_candidates("q", Lexical([hit(1)]), Dense([hit(1)]),
                              SimpleNamespace(name="fake"), reranker=broken)
    assert "rerank" in result["failures"]
    assert "rerank" not in result["rankings"]


def test_units_do_not_refresh_or_rebuild():
    store = object.__new__(FrozenUnitStore)
    assert store.refresh() == 0


def test_cli_missing_cache_records_incomplete_run(tmp_path):
    import json
    from paperanchor.library import PaperLibrary

    database = tmp_path / "library.sqlite3"
    PaperLibrary(database)
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"fingerprint": "frozen-input-id", "status": "diagnostic",
                                "cases": [{"case_id": "q", "split": "dev", "display_question": "q"}]}))
    output = tmp_path / "results.json"
    args = ["--database", str(database), "--cases", str(cases), "--output", str(output),
            "--reference-tokenizer", str(tmp_path / "absent.json"), "--representations", "blocks"]
    assert main(args) == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "incomplete"
    assert payload["config"]["split"] == "dev"
    assert payload["cases_fingerprint"] == "frozen-input-id"
    assert payload["cases_status"] == "diagnostic"
    assert payload["resume_supported"] is True
    assert payload["run_fingerprint"]
    assert payload["tokenizer_fingerprint"] is None
    assert payload["records"][0]["status"] == "context_failed"
    assert payload["records"][1]["status"] == "failed"
    with pytest.raises(SystemExit):
        main(args)


def test_product_blocks_and_units_keep_actual_text(tmp_path, tokenizer):
    import numpy as np
    import pymupdf
    from paperanchor.library import PaperLibrary
    from paperanchor.semantic import SemanticIndex
    from paperanchor.unit_index import UnitStore, UnitSemanticIndex

    pdf = tmp_path / "paper.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Introduction")
        page.insert_text((72, 110), "Graph retrieval finds relevant passages.")
        page.insert_text((72, 145), "It combines textual and semantic evidence.")
        document.save(pdf)
    library = PaperLibrary(tmp_path / "library.sqlite3")
    library.index_pdf(pdf)
    provider = SimpleNamespace(name="synthetic", embed_query=lambda text: np.array([1., 0.]),
                               embed_passages=lambda texts: [np.array([1., 0.]) for text in texts])
    block_index = SemanticIndex(library.database_path)
    block_index.build(provider)
    units = UnitStore(library)
    unit_index = UnitSemanticIndex(units)
    unit_index.build(provider)
    frozen = FrozenUnitStore(library)
    records = run_cases([{"case_id": "q", "split": "dev", "display_question": "graph"}],
                        {"blocks": (library, block_index),
                         "units": (frozen, UnitSemanticIndex(frozen))}, provider, tokenizer)
    assert len(records) == 6
    assert all(record["status"] == "ok" for record in records)
    unit_record = next(record for record in records if record["representation"] == "units")
    actual_hit = frozen.search("graph", 40)[0]
    assert unit_record["ranked"][0]["text"] == actual_hit.text
    assert unit_record["ranked"][0]["passage_id"] < 0
    assert len(unit_record["ranked"][0]["source_spans"]) == 2


@pytest.mark.parametrize("status", [None, "diagnostic", "incomplete"])
def test_cli_test_gate_requires_frozen_cases_even_with_flag(tmp_path, status, monkeypatch, capsys):
    import json
    import evals.qasper_benchmark.retrieve as retrieve

    database = tmp_path / "unused.sqlite3"
    database.touch()
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"status": status, "cases": []}), encoding="utf-8")
    monkeypatch.setattr(retrieve, "run_cases", lambda *args, **kwargs: pytest.fail("retrieval ran"))
    with pytest.raises(SystemExit) as error:
        main(["--database", str(database), "--cases", str(cases),
              "--output", str(tmp_path / "result.json"),
              "--reference-tokenizer", str(tmp_path / "absent.json"),
              "--split", "test", "--allow-test"])
    assert error.value.code == 2
    assert "test requires frozen cases" in capsys.readouterr().err


def test_partial_embedding_coverage_fails_dense_before_query(tmp_path, tokenizer):
    import sqlite3

    database = tmp_path / "partial.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE passages(id INTEGER PRIMARY KEY);
            CREATE TABLE passage_embeddings(passage_id INTEGER,model TEXT,dimension INTEGER);
            INSERT INTO passages VALUES(1),(2);
            INSERT INTO passage_embeddings VALUES(1,'fake',2);
        """)
    state = index_metadata(database, "blocks", "fake")
    assert state["total"] == 2
    assert state["vectors"] == 1
    assert state["vector_coverage_complete"] is False
    dense = Dense([hit(1)])
    records = run_cases([{"case_id": "q", "split": "dev", "display_question": "q"}],
                        {"blocks": (Lexical([hit(1)]), dense)}, SimpleNamespace(name="fake"),
                        tokenizer, index_states={"blocks": state}, reranker=Reranker())
    assert dense.calls == []
    assert records[0]["status"] == "ok"
    assert [record["status"] for record in records[1:]] == ["failed"] * 3
    assert "1/2" in records[1]["failure"]["message"]


def test_content_fingerprints_track_config_database_and_tokenizer(tmp_path):
    import sqlite3

    database = tmp_path / "state.sqlite3"
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("first tokenizer", encoding="utf-8")
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE state(value TEXT)")
        connection.execute("INSERT INTO state VALUES('one')")
    config = {"depth": 40, "output": "first.json"}
    original = run_fingerprints(config, database, tokenizer, "cases")
    assert run_fingerprints({**config, "output": "second.json", "resume": True}, database, tokenizer, "cases") == original
    changed_config = run_fingerprints({**config, "depth": 20}, database, tokenizer, "cases")
    assert changed_config["config_fingerprint"] != original["config_fingerprint"]
    tokenizer.write_text("second tokenizer", encoding="utf-8")
    assert run_fingerprints(config, database, tokenizer, "cases")["tokenizer_fingerprint"] != original["tokenizer_fingerprint"]
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO state VALUES('two')")
    changed_database = run_fingerprints(config, database, tokenizer, "cases")
    assert changed_database["database_fingerprint"] != original["database_fingerprint"]
    assert changed_database["run_fingerprint"] != original["run_fingerprint"]
    assert run_fingerprints(config, database, tokenizer, "different cases")["cases_content_fingerprint"] != original["cases_content_fingerprint"]


def test_database_fingerprint_includes_wal_and_survives_checkpoint(tmp_path):
    import sqlite3

    database = tmp_path / "wal.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE state(value TEXT)")
        connection.commit()
        before = database_fingerprint(database)
        connection.execute("INSERT INTO state VALUES('committed WAL row')")
        connection.commit()
        after = database_fingerprint(database)
        assert before != after
        connection.execute("PRAGMA wal_checkpoint(FULL)")
        assert database_fingerprint(database) == after


def test_interrupted_cli_resumes_missing_groups_and_preserves_failures(tmp_path, monkeypatch):
    import json
    import evals.qasper_benchmark.retrieve as retrieve
    from paperanchor.library import PaperLibrary

    database = tmp_path / "library.sqlite3"
    PaperLibrary(database)
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"status": "diagnostic", "fingerprint": "cases-1", "cases": [
        {"case_id": "q1", "split": "dev", "display_question": "first"},
        {"case_id": "q2", "split": "dev", "display_question": "second"},
    ]}), encoding="utf-8")
    output = tmp_path / "run.json"
    args = ["--database", str(database), "--cases", str(cases), "--output", str(output),
            "--reference-tokenizer", str(tmp_path / "missing.json"), "--representations", "blocks"]
    original = retrieve.fixed_candidates
    calls = []

    def interrupt(query, *positional, **kwargs):
        calls.append(query)
        saved = json.loads(output.read_text(encoding="utf-8"))
        assert saved["status"] == "incomplete"
        assert saved["expected_records"] == 6
        assert saved["expected_groups"] == 2
        if query == "second":
            raise KeyboardInterrupt()
        assert saved["records"] == []
        return original(query, *positional, **kwargs)

    monkeypatch.setattr(retrieve, "fixed_candidates", interrupt)
    with pytest.raises(KeyboardInterrupt):
        main(args)
    interrupted = json.loads(output.read_text(encoding="utf-8"))
    assert interrupted["completed_groups"] == 1
    assert len(interrupted["records"]) == 3
    assert interrupted["records"][1]["status"] == "failed"

    def resumed(query, *positional, **kwargs):
        calls.append(query)
        assert query == "second"
        return original(query, *positional, **kwargs)

    monkeypatch.setattr(retrieve, "fixed_candidates", resumed)
    assert main([*args, "--resume"]) == 1
    final = json.loads(output.read_text(encoding="utf-8"))
    assert final["records"][:3] == interrupted["records"]
    assert final["completed_groups"] == 2
    assert len(final["records"]) == 6
    assert calls == ["first", "second", "second"]
    assert main([*args, "--resume"]) == 1
    assert calls == ["first", "second", "second"]

    for extra in (["--depth", "41"], ["--budget", "10"]):
        with pytest.raises(SystemExit):
            main([*args, "--resume", *extra])
        assert json.loads(output.read_text(encoding="utf-8")) == final
    final["records"].append(final["records"][0])
    output.write_text(json.dumps(final), encoding="utf-8")
    with pytest.raises(SystemExit):
        main([*args, "--resume"])


def test_run_cases_checkpoints_whole_pools_and_reuses_success(tokenizer):
    lexical, dense = Lexical([hit(1)]), Dense([hit(1)])
    cases = [{"case_id": "q", "split": "dev", "display_question": "q"}]
    checkpoints = []
    records = run_cases(cases, {"blocks": (lexical, dense)}, SimpleNamespace(name="fake"),
                        tokenizer, checkpoint=checkpoints.append)
    assert len(checkpoints) == 1
    assert len(checkpoints[0]) == 3
    assert all(record["status"] == "ok" for record in records)
    assert run_cases(cases, {"blocks": (lexical, dense)}, SimpleNamespace(name="fake"),
                     tokenizer, previous_records=records) == records
    assert len(lexical.calls) == len(dense.calls) == 1
    with pytest.raises(ValueError, match="incomplete candidate-pool"):
        run_cases(cases, {"blocks": (lexical, dense)}, SimpleNamespace(name="fake"),
                  tokenizer, previous_records=records[:1])


@pytest.mark.parametrize("tamper", ["public", "corpus"])
def test_frozen_cli_rejects_public_or_corpus_tampering(tmp_path, monkeypatch, capsys, tamper):
    import json
    import sqlite3
    import evals.qasper_benchmark.retrieve as retrieve
    from evals.qasper_benchmark.identity import corpus_fingerprint, public_cases_fingerprint
    from paperanchor.library import PaperLibrary

    database = tmp_path / "library.sqlite3"
    PaperLibrary(database)
    public = [{"case_id": "opaque-1", "split": "dev", "display_question": "Original question"}]
    payload = {"status": "frozen", "fingerprint": "freeze-token", "cases": public,
               "corpus_fingerprint": corpus_fingerprint(database),
               "public_cases_fingerprint": public_cases_fingerprint(public)}
    if tamper == "public":
        public[0]["display_question"] = "Changed question"
    else:
        with sqlite3.connect(database) as connection:
            connection.execute("INSERT INTO papers VALUES (1,'paper.pdf','hash','Paper',1,'today')")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(retrieve, "CachedEmbeddingProvider", lambda *args: pytest.fail("model setup ran"))
    with pytest.raises(SystemExit) as error:
        main(["--database", str(database), "--cases", str(cases),
              "--output", str(tmp_path / "run.json"),
              "--reference-tokenizer", str(tmp_path / "tokenizer.json")])
    assert error.value.code == 2
    assert f"{'public_cases' if tamper == 'public' else 'corpus'}_fingerprint" in capsys.readouterr().err


def test_corpus_identity_ignores_unrelated_ddl_and_accepts_connection(tmp_path):
    import sqlite3
    from evals.qasper_benchmark.identity import corpus_fingerprint, public_cases_fingerprint

    database = tmp_path / "corpus.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE passages(id INTEGER PRIMARY KEY,text TEXT)")
        connection.execute("INSERT INTO passages VALUES(1,'Actual PDF evidence')")
        connection.commit()
        original = corpus_fingerprint(connection)
        assert corpus_fingerprint(database) == original
        connection.execute("CREATE TABLE metadata(paper_id INTEGER, abstract TEXT)")
        assert corpus_fingerprint(connection) == original
        connection.execute("UPDATE passages SET text='Changed evidence'")
        assert corpus_fingerprint(connection) != original
    assert public_cases_fingerprint([{"question": "q", "case_id": "opaque"}]) == public_cases_fingerprint(
        [{"case_id": "opaque", "question": "q"}])


@pytest.mark.parametrize("table", ["retrieval_units", "unit_sources", "unit_index_state",
                                  "passage_embeddings", "unit_embeddings", "passage_search", "unit_search"])
def test_corpus_identity_tracks_retrieval_tables(tmp_path, table):
    import sqlite3
    from evals.qasper_benchmark.identity import corpus_fingerprint

    database = tmp_path / "corpus.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(f'CREATE TABLE "{table}"(id INTEGER PRIMARY KEY, value BLOB)')
        connection.execute(f'INSERT INTO "{table}" VALUES(1,?)', (b"original",))
        before = corpus_fingerprint(connection)
        connection.execute(f'UPDATE "{table}" SET value=?', (b"changed",))
        assert corpus_fingerprint(connection) != before


@pytest.mark.parametrize("module", ["library", "semantic", "unit_index", "representation"])
def test_product_source_changes_invalidate_implementation_identity(tmp_path, monkeypatch, module):
    import evals.qasper_benchmark.retrieve as retrieve

    original = retrieve.implementation_fingerprint()
    dependency = getattr(retrieve, module)
    modified = tmp_path / Path(dependency.__file__).name
    modified.write_bytes(Path(dependency.__file__).read_bytes() + b"\n# synthetic implementation change\n")
    monkeypatch.setattr(dependency, "__file__", str(modified))
    assert retrieve.implementation_fingerprint() != original
