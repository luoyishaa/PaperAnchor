import json
from hashlib import sha256
from pathlib import Path
import sqlite3

import pytest
import pymupdf

from evals.qasper_benchmark.cases import main, prepare_cases
from evals.qasper_benchmark.identity import corpus_fingerprint, public_cases_fingerprint
from evals.qasper_benchmark.score import score_retrieval


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fixture(tmp_path, *, complete=False):
    database = tmp_path / "library.sqlite3"
    sources = {s: {} for s in ("train", "validation", "test")}
    candidates, mappings = [], []
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "The result is 42.")
        pdf_bytes = document.tobytes()
    checksum = sha256(pdf_bytes).hexdigest()
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE papers(id INTEGER PRIMARY KEY,file_path TEXT,file_hash TEXT);
            CREATE TABLE passages(id INTEGER PRIMARY KEY,paper_id INTEGER,page_number INTEGER,
                block_number INTEGER,text TEXT,x0 REAL,y0 REAL,x1 REAL,y1 REAL);
        """)
        for index in range(300 if complete else 3):
            split = "validation" if index == 0 else "test" if index == 1 else "train"
            source_id = f"2000.{index:05d}"
            path = tmp_path / f"{index}.pdf"
            path.write_bytes(pdf_bytes)
            title = f"Research {index}"
            qas = []
            answerable, unanswerable = ((32, 8) if split == "validation" else (64, 16)) if complete else (12, 1)
            for number in range(answerable + unanswerable):
                answer = {"unanswerable": number >= answerable, "yes_no": None,
                          "extractive_spans": ["42"] if number < answerable else [],
                          "free_form_answer": "", "evidence": ["The result is 42."],
                          "highlighted_evidence": ["The result is 42."]}
                qas.append({"question_id": f"question-{index}-{number}",
                            "question": f"What result does experiment {number} report?",
                            "answers": [{"answer": answer, "worker_id": "original-worker"}]})
            sources[split][source_id] = {"title": title, "qas": qas}
            mapping = {"paper_id": source_id, "split": split, "title": title,
                       "pdf_path": str(path), "index_paper_id": index + 1}
            mappings.append(mapping)
            candidates.append({**mapping, "status": "ready", "sha256": checksum})
            connection.execute("INSERT INTO papers VALUES(?,?,?)", (index + 1, str(path), checksum))
            connection.execute("INSERT INTO passages VALUES(?,?,?,?,?,?,?,?,?)",
                               (index + 1, index + 1, 1, 0, "The result is 42.", 0, 0, 10, 10))
    raw_files = {}
    for split, source in sources.items():
        path = tmp_path / f"{split}.json"
        save(path, source)
        raw_files[split] = str(path)
    manifest, build = tmp_path / "manifest.json", tmp_path / "build.json"
    save(manifest, {"source": {"raw_files": raw_files}, "candidates": candidates})
    save(build, {"status": "ready", "fingerprint": "build-fixture", "mappings": mappings,
                 "validation": {"valid": True}})
    return manifest, build, database


def test_diagnostic_is_dev_only_and_scoring_gold_is_separate(tmp_path):
    inputs = fixture(tmp_path)
    output = tmp_path / "output"
    result = prepare_cases(*inputs, output, diagnostic_dev=True)
    assert result["status"] == "diagnostic"
    assert len(result["cases"]) == 10
    assert {c["split"] for c in result["cases"]} == {"dev"}
    gold = load(output / "gold.json")
    originals = load(tmp_path / "validation.json")["2000.00000"]["qas"]
    by_id = {g["case_id"]: g for g in gold["cases"]}
    for case in result["cases"]:
        assert case["display_question"] == case["question"] + "\nPaper: " + case["title"]
        assert not {"annotations", "alignment", "source_paper_id", "index_paper_id"} & case.keys()
        reference = by_id[case["case_id"]]
        assert reference["original_question"] in originals
        if case["scope"] == "full_library":
            assert "paper_id" not in case
            document = gold["documents"][str(reference["index_paper_id"])]
            scores = score_retrieval(reference["alignment"]["support_groups"], document["blocks"], document)
            assert scores["at_k"]["1"]["evidence_recall"] == 1
        else:
            assert case["paper_id"] == reference["index_paper_id"]
    assert "test_not_evaluated" in result["counters"]["decisions"]
    assert all("alignment" not in g for g in gold["candidates"] if g["split"] == "test")


def test_freeze_exact_quotas_and_corpus(tmp_path):
    inputs = fixture(tmp_path, complete=True)
    result = prepare_cases(*inputs, tmp_path / "frozen")
    assert result["status"] == "frozen"
    assert result["freeze_blockers"] == []
    assert result["counters"]["selected_by_split"] == {
        "dev": {"aligned": 32, "unanswerable": 8},
        "test": {"aligned": 64, "unanswerable": 16}}
    assert len(result["cases"]) == 120
    assert sum(c["scope"] == "single_paper_auxiliary" for c in result["cases"]) == 24
    assert len({c["case_id"] for c in result["cases"]}) == 120


def test_incomplete_and_all_candidate_exclusions_saved(tmp_path):
    inputs = fixture(tmp_path)
    raw = load(tmp_path / "validation.json")
    qas = raw["2000.00000"]["qas"]
    qas[0]["answers"][0]["answer"]["highlighted_evidence"] = ["The result is 43."]
    save(tmp_path / "validation.json", raw)
    output = tmp_path / "incomplete"
    result = prepare_cases(*inputs, output)
    assert result["status"] == "incomplete"
    assert set(result["freeze_blockers"]) == {"corpus_size_outside_300_500", "question_quotas_incomplete"}
    decisions = load(output / "alignment.json")["decisions"]
    assert len(decisions) == 39
    failed = next(d for d in decisions if d["question_id"] == qas[0]["question_id"])
    assert failed["reason"] == "no_complete_support_group"
    reference = next(g for g in load(output / "gold.json")["candidates"] if g["case_id"] == failed["case_id"])
    assert reference["original_question"] == qas[0]


def test_deterministic_resume_and_refuse_changed_inputs(tmp_path):
    inputs = fixture(tmp_path)
    output = tmp_path / "out"
    result = prepare_cases(*inputs, output, diagnostic_dev=True)
    assert prepare_cases(*inputs, tmp_path / "second", diagnostic_dev=True) == result
    assert prepare_cases(*inputs, output, diagnostic_dev=True, resume=True) == result
    with pytest.raises(ValueError, match="already exists"):
        prepare_cases(*inputs, output, diagnostic_dev=True)
    with pytest.raises(ValueError, match="differ"):
        prepare_cases(*inputs, output, diagnostic_dev=True, seed=1, resume=True)
    saved = load(output / "cases.json")
    saved["cases"][0]["question"] = "modified"
    save(output / "cases.json", saved)
    with pytest.raises(ValueError, match="differ"):
        prepare_cases(*inputs, output, diagnostic_dev=True, resume=True)


@pytest.mark.parametrize("change", ["path", "id", "checksum", "missing_checksum", "not_ready", "overlap"])
def test_invalid_mapping_is_rejected(tmp_path, change):
    manifest_path, build_path, database = fixture(tmp_path)
    manifest, build = load(manifest_path), load(build_path)
    if change == "path":
        with sqlite3.connect(database) as connection:
            connection.execute("UPDATE papers SET file_path='wrong.pdf' WHERE id=1")
    elif change == "id":
        build["mappings"][0]["index_paper_id"] = 100
    elif change == "checksum":
        manifest["candidates"][0]["sha256"] = "changed"
    elif change == "missing_checksum":
        del manifest["candidates"][0]["sha256"]
    elif change == "not_ready":
        manifest["candidates"][0]["status"] = "pending"
    else:
        build["mappings"][1]["paper_id"] = "2000.00000v2"
        manifest["candidates"][1]["paper_id"] = "2000.00000v2"
        raw = load(tmp_path / "test.json")
        raw["2000.00000v2"] = raw.pop("2000.00001")
        save(tmp_path / "test.json", raw)
    save(manifest_path, manifest)
    save(build_path, build)
    with pytest.raises(ValueError):
        prepare_cases(manifest_path, build_path, database, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_cli_diagnostic(tmp_path, capsys):
    manifest, build, database = fixture(tmp_path)
    assert main(["--manifest", str(manifest), "--build", str(build), "--database", str(database),
                 "--output", str(tmp_path / "cli"), "--diagnostic-dev"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "diagnostic"


def test_missing_indexed_mapping_is_exclusion_not_unanswerable(tmp_path):
    manifest, build_path, database = fixture(tmp_path)
    build = load(build_path)
    build["mappings"] = build["mappings"][1:]
    save(build_path, build)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM papers WHERE id=1")
        connection.execute("DELETE FROM passages WHERE paper_id=1")
    result = prepare_cases(manifest, build_path, database, tmp_path / "missing", diagnostic_dev=True)
    assert result["cases"] == []
    assert result["counters"]["decisions"]["not_indexed"] == 13


def test_full_quotas_cannot_freeze_unready_build(tmp_path):
    inputs = fixture(tmp_path, complete=True)
    build = load(inputs[1])
    build["status"] = "incomplete"
    save(inputs[1], build)
    result = prepare_cases(*inputs, tmp_path / "not-ready")
    assert len(result["cases"]) == 120
    assert result["status"] == "incomplete"
    assert result["freeze_blockers"] == ["build_not_ready"]


def test_changed_valid_pdf_cannot_freeze_with_matching_recorded_hashes(tmp_path):
    inputs = fixture(tmp_path, complete=True)
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "The result is 43.")
        document.save(tmp_path / "0.pdf")
    with pymupdf.open(tmp_path / "0.pdf") as document:
        assert "43" in document[0].get_text()
    with pytest.raises(ValueError, match="PDF bytes differ"):
        prepare_cases(*inputs, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_public_and_gold_identities_bind_content_and_index(tmp_path):
    inputs = fixture(tmp_path)
    output = tmp_path / "out"
    result = prepare_cases(*inputs, output, diagnostic_dev=True)
    gold = load(output / "gold.json")
    assert result["public_cases_fingerprint"] == public_cases_fingerprint(result["cases"])
    assert result["corpus_fingerprint"] == corpus_fingerprint(inputs[2])
    for key in ("corpus_fingerprint", "public_cases_fingerprint"):
        assert gold[key] == result[key]
    result["cases"][0]["display_question"] = "A different question"
    assert public_cases_fingerprint(result["cases"]) != gold["public_cases_fingerprint"]
    save(output / "cases.json", result)
    with pytest.raises(ValueError, match="differ"):
        prepare_cases(*inputs, output, diagnostic_dev=True, resume=True)
    with sqlite3.connect(inputs[2]) as connection:
        connection.execute("CREATE TABLE smoke_metadata (value TEXT)")
        connection.execute("INSERT INTO smoke_metadata VALUES ('checked')")
    assert corpus_fingerprint(inputs[2]) == gold["corpus_fingerprint"]
    with sqlite3.connect(inputs[2]) as connection:
        connection.execute("UPDATE passages SET text='Additional text. ' || text WHERE id=1")
    assert corpus_fingerprint(inputs[2]) != gold["corpus_fingerprint"]
    changed = prepare_cases(*inputs, tmp_path / "changed", diagnostic_dev=True)
    assert changed["public_cases_fingerprint"] == gold["public_cases_fingerprint"]
    assert changed["corpus_fingerprint"] != gold["corpus_fingerprint"]
    assert changed["fingerprint"] != gold["fingerprint"]
