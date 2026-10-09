import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from evals.qasper_benchmark import run as benchmark
from evals.qasper_benchmark.align import align_annotations, build_document
from evals.qasper_benchmark.identity import corpus_fingerprint, public_cases_fingerprint
from paperanchor.evidence_profile import EvidenceProfile
from paperanchor.evidence_policies import BGE, MINILM


@pytest.fixture
def inputs(tmp_path):
    database = tmp_path / "library.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY, title TEXT)")
        connection.execute("INSERT INTO papers VALUES (1, 'Paper title')")
    document = build_document(1, [{"passage_id": 1, "text": "The result is 42."}])
    alignment = align_annotations([{"answer": {
        "unanswerable": False, "yes_no": None, "extractive_spans": ["42"],
        "highlighted_evidence": ["The result is 42."],
    }}], document)
    cases = [{"case_id": f"q{index:03}", "question": f"Question {index}?", "title": "Paper title",
              "paper_id": 1, "scope": "full_library" if index < 96 else "single_paper_auxiliary",
              "split": "dev" if index < 40 else "test"} for index in range(120)]
    identity = {"status": "frozen", "fingerprint": "fixture",
                "corpus_fingerprint": corpus_fingerprint(database),
                "public_cases_fingerprint": public_cases_fingerprint(cases)}
    public = {**identity, "cases": cases}
    gold = {**identity, "documents": {"1": document}, "cases": [
        {"case_id": case["case_id"], "index_paper_id": 1,
         "alignment": alignment if index < 96 else {"status": "unanswerable"}}
        for index, case in enumerate(cases)]}
    cases_path, gold_path = tmp_path / "cases.json", tmp_path / "gold.json"
    cases_path.write_text(json.dumps(public), encoding="utf-8")
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    return database, cases_path, gold_path, tokenizer_path


@pytest.mark.parametrize("name", ["minilm", "bge", "optimized"])
def test_fixed_profile_runs_all_cases_with_verbatim_inputs_and_failed_case_denominator(inputs, tmp_path, monkeypatch, name):
    database, cases_path, gold_path, tokenizer_path = inputs
    calls, instances = [], []

    class Profile:
        def __init__(self, path):
            assert path == database
            self.tokenizer_path = tokenizer_path
            self.config = {"strategy": "ranked_expansion", "protect": 8}
            self.provider = SimpleNamespace(name="fixture-embedding")
            self.models = {name: SimpleNamespace(model_name=name) for name in (BGE, MINILM)}
            self._revision = (1, 0)
            self.closed = False
            instances.append(self)

        def _optimized_spec(self):
            return {"config": {"strategy": "ranked_expansion", "protect": 4}}

        def prepare(self, question, **kwargs):
            calls.append((question, kwargs))
            if question == "Question 2?":
                raise RuntimeError("Synthetic retrieval failure")
            return {"budget": 4096, "reference_tokens": 9,
                    "text": "[E1] Paper title\nThe result is 42.",
                    "chunks": [{"evidence_id": "E1", "paper_id": 1, "passage_id": 1,
                                "paper_path": Path("paper.pdf"), "text": "The result is 42."}]}

        def close(self):
            self.closed = True

    monkeypatch.setattr(benchmark, "UncachedProfile", Profile)
    output = tmp_path / "results"
    result = benchmark.run(database, cases_path, gold_path, output, profile_name=name)
    assert (result["questions"], result["main"], result["auxiliary"]) == (120, 96, 24)
    assert result["status"] == "incomplete"
    assert result["failures"] == 1
    assert result["complete"] == 95
    assert result["ContextCompleteSupport"] == result["ContextEvidenceRecall"] == 95 / 96
    assert len(calls) == 121  # One warmup is excluded from results.
    assert calls[0] == calls[1]
    for index, (question, options) in enumerate(calls[1:]):
        assert question == f"Question {index}?"
        assert options == {"title": "Paper title", "profile_name": benchmark.PROFILES[name],
                           "paper_ids": None if index < 96 else (1,)}
    assert len(list(output.glob("q*.json"))) == 120
    saved = benchmark.read(output / "records.json")
    assert saved["protocol"]["result_cache"] is False
    assert saved["protocol"]["runtime_models"]["embedding"]["name"] == "fixture-embedding"
    assert saved["records"][2]["context"]["chunks"] == []
    assert saved["records"][0]["context"]["chunks"][0]["paper_path"] == "paper.pdf"
    assert instances[0].closed
    with pytest.raises(FileExistsError, match="new output"):
        benchmark.run(database, cases_path, gold_path, output, profile_name=name)


def test_frozen_inputs_reject_incomplete_set_changed_queries_or_wrong_corpus(inputs):
    database, cases_path, gold_path, _ = inputs
    public, gold = benchmark.read(cases_path), benchmark.read(gold_path)
    benchmark.validate_inputs(public, gold, database)
    public["cases"][0]["question"] = "Different question"
    with pytest.raises(ValueError, match="content changed"):
        benchmark.validate_inputs(public, gold, database)
    public = benchmark.read(cases_path)
    public["cases"].pop()
    with pytest.raises(ValueError, match="120 distinct"):
        benchmark.validate_inputs(public, gold, database)
    public = benchmark.read(cases_path)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE papers SET title='Changed'")
    with pytest.raises(ValueError, match="Indexed corpus"):
        benchmark.validate_inputs(public, gold, database)


def test_uncached_product_adapter_disables_caches_on_every_engine_refresh(monkeypatch):
    engines = [SimpleNamespace(cache_enabled=True), SimpleNamespace(cache_enabled=True)]
    pending = iter(engines)
    monkeypatch.setattr(EvidenceProfile, "_current_engine", lambda _: next(pending))
    profile = object.__new__(benchmark.UncachedProfile)
    assert profile._current_engine() is engines[0]
    assert profile._current_engine() is engines[1]
    assert all(engine.cache_enabled is False for engine in engines)


def test_runtime_identity_hashes_the_loaded_model_file(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"fixture model weights")
    native = SimpleNamespace(_model_dir=tmp_path, model_description=SimpleNamespace(model_file=path.name))
    model = SimpleNamespace(name="fixture", _model=SimpleNamespace(model=native))
    first = benchmark.model_identity(model)
    assert first["name"] == "fixture"
    assert first["file"] == str(path.resolve())
    path.write_bytes(b"changed model weights")
    assert benchmark.model_identity(model)["sha256"] != first["sha256"]
