from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import pymupdf
import pytest

from evals.qasper_benchmark import title_aliases


def fixture(tmp_path, *, title="Verified Public Title", original="I. Introduction",
            page_text=None):
    pdf = tmp_path / "paper.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), title if page_text is None else page_text)
        document.save(pdf)
    digest = sha256(pdf.read_bytes()).hexdigest()
    database = tmp_path / "library.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE papers(id INTEGER PRIMARY KEY,file_path TEXT,file_hash TEXT,title TEXT)")
        connection.execute("INSERT INTO papers VALUES(1,?,?,?)", (str(pdf), digest, original))
    candidate = {"paper_id": "1234.56789", "title": title, "pdf_path": str(pdf),
                 "sha256": digest, "status": "ready", "parse_status": "parsed",
                 "identity_status": "title_verified", "url": "https://arxiv.org/pdf/1234.56789v1"}
    manifest = tmp_path / "manifest.json"
    save_manifest(manifest, [candidate])
    return database, manifest, pdf, candidate


def save_manifest(path, candidates):
    path.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")


def save_aliases(tmp_path, payload):
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_verified_public_title_becomes_alias_without_database_mutation(tmp_path):
    database, manifest, pdf, candidate = fixture(tmp_path)
    before = database.read_bytes()
    result = title_aliases.build_aliases(database, manifest)
    assert set(result) == {"version", "aliases", "checks", "fingerprint"}
    assert result["version"] == 1
    assert len(result["aliases"]) == len(result["checks"]) == 1
    alias = result["aliases"][0]
    assert alias["paper_id"] == 1
    assert alias["file_hash"] == sha256(pdf.read_bytes()).hexdigest()
    assert alias["original_title"] == "I. Introduction"
    assert alias["title"] == candidate["title"]
    assert alias["source"]["manifest_sha256"] == sha256(manifest.read_bytes()).hexdigest()
    assert result["checks"][0]["status"] == "alias_added"
    assert result["checks"][0]["first_page_title_verified"] is True
    assert result["fingerprint"] == title_aliases._fingerprint(result["aliases"])
    assert title_aliases.load_aliases(save_aliases(tmp_path, result), database) == {1: candidate["title"]}
    assert database.read_bytes() == before


def test_equivalent_normalized_titles_are_verified_without_alias(tmp_path):
    database, manifest, _, _ = fixture(tmp_path, original="VERIFIED public:   Title")
    result = title_aliases.build_aliases(database, manifest)
    assert result["aliases"] == []
    assert result["checks"][0]["status"] == "verified_unchanged"
    assert result["checks"][0]["first_page_title_verified"] is True
    assert title_aliases._title_key("ＡＢＣ—Test") == title_aliases._title_key("abc test")


@pytest.mark.parametrize("field,value,reason", [
    ("status", "pending", "manifest_entry_not_verified"),
    ("parse_status", "pending", "manifest_entry_not_verified"),
    ("identity_status", "pending", "manifest_entry_not_verified"),
    ("title", "", "manifest_title_missing"),
    ("sha256", None, "manifest_hash_missing_or_invalid"),
    ("sha256", "not-a-hash", "manifest_hash_missing_or_invalid"),
    ("sha256", "0" * 64, "manifest_database_hash_mismatch"),
])
def test_unverified_or_inconsistent_manifest_keeps_original_title(tmp_path, field, value, reason):
    database, manifest, _, candidate = fixture(tmp_path)
    candidate[field] = value
    save_manifest(manifest, [candidate])
    result = title_aliases.build_aliases(database, manifest)
    assert result["aliases"] == []
    assert result["checks"][0]["status"] == "kept_original"
    assert result["checks"][0]["reason"] == reason


def test_first_page_text_must_confirm_manifest_title(tmp_path):
    database, manifest, _, _ = fixture(tmp_path, page_text="A Different Paper")
    result = title_aliases.build_aliases(database, manifest)
    assert result["aliases"] == []
    assert result["checks"][0]["reason"] == "title_not_confirmed_on_first_page"
    assert result["checks"][0]["first_page_title_verified"] is False


def test_changed_pdf_bytes_are_rejected_even_if_database_and_manifest_agree(tmp_path):
    database, manifest, pdf, _ = fixture(tmp_path)
    pdf.write_bytes(pdf.read_bytes() + b"\nchanged bytes")
    result = title_aliases.build_aliases(database, manifest)
    assert result["aliases"] == []
    assert result["checks"][0]["reason"] == "pdf_database_hash_mismatch"


def test_missing_pdf_is_recorded_and_does_not_remove_database_paper(tmp_path):
    database, manifest, pdf, _ = fixture(tmp_path)
    pdf.unlink()
    result = title_aliases.build_aliases(database, manifest)
    assert result["aliases"] == []
    assert result["checks"][0]["reason"] == "pdf_missing"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM papers").fetchone()[0] == 1


def test_unreadable_pdf_text_is_recorded(tmp_path):
    database, manifest, pdf, candidate = fixture(tmp_path)
    pdf.write_bytes(b"not a PDF")
    candidate["sha256"] = sha256(pdf.read_bytes()).hexdigest()
    save_manifest(manifest, [candidate])
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE papers SET file_hash=?", (candidate["sha256"],))
    result = title_aliases.build_aliases(database, manifest)
    assert result["aliases"] == []
    assert result["checks"][0]["reason"] == "pdf_text_unavailable"


def test_manifest_missing_or_ambiguous_entries_are_reported(tmp_path):
    database, manifest, _, candidate = fixture(tmp_path)
    save_manifest(manifest, [])
    result = title_aliases.build_aliases(database, manifest)
    assert result["checks"][0]["reason"] == "manifest_entry_missing"
    save_manifest(manifest, [candidate, {**candidate, "title": "Conflicting title"}])
    result = title_aliases.build_aliases(database, manifest)
    assert result["aliases"] == []
    assert result["checks"][0]["reason"] == "ambiguous_manifest_entries"


def test_moved_manifest_path_can_match_by_pdf_hash(tmp_path):
    database, manifest, _, candidate = fixture(tmp_path)
    candidate["pdf_path"] = str(tmp_path / "old-location" / "paper.pdf")
    save_manifest(manifest, [candidate])
    result = title_aliases.build_aliases(database, manifest)
    assert result["checks"][0]["manifest_match"] == "sha256"
    assert result["checks"][0]["status"] == "alias_added"


def test_load_rejects_tampered_alias_and_duplicate_identity(tmp_path):
    database, manifest, _, _ = fixture(tmp_path)
    result = title_aliases.build_aliases(database, manifest)
    result["aliases"][0]["title"] = "Unverified changed title"
    with pytest.raises(ValueError, match="fingerprint"):
        title_aliases.load_aliases(save_aliases(tmp_path, result), database)
    result["aliases"].append(dict(result["aliases"][0]))
    result["fingerprint"] = title_aliases._fingerprint(result["aliases"])
    with pytest.raises(ValueError, match="unique"):
        title_aliases.load_aliases(save_aliases(tmp_path, result), database)


@pytest.mark.parametrize("statement,value,error", [
    ("UPDATE papers SET file_hash=?", "0" * 64, "file hash mismatch"),
    ("UPDATE papers SET title=?", "Already corrected title", "original title mismatch"),
    ("DELETE FROM papers WHERE id=?", 1, "is missing"),
])
def test_load_fails_explicitly_when_database_alias_identity_changes(tmp_path, statement, value, error):
    database, manifest, _, _ = fixture(tmp_path)
    result = title_aliases.build_aliases(database, manifest)
    path = save_aliases(tmp_path, result)
    with sqlite3.connect(database) as connection:
        connection.execute(statement, (value,))
    with pytest.raises(ValueError, match=error):
        title_aliases.load_aliases(path, database)


def test_load_only_checks_metadata_and_never_reparses_pdfs(tmp_path, monkeypatch):
    database, manifest, _, candidate = fixture(tmp_path)
    result = title_aliases.build_aliases(database, manifest)
    def forbidden(*args):
        raise AssertionError("Loading must not parse PDF or read its bytes")
    monkeypatch.setattr(title_aliases, "_first_page_text", forbidden)
    monkeypatch.setattr(title_aliases, "_file_hash", forbidden)
    assert title_aliases.load_aliases(save_aliases(tmp_path, result), database) == {1: candidate["title"]}


def test_missing_database_is_not_created(tmp_path):
    _, manifest, _, _ = fixture(tmp_path)
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError):
        title_aliases.build_aliases(missing, manifest)
    assert not missing.exists()


def test_cli_saves_expected_schema_and_refuses_overwrite(tmp_path):
    database, manifest, _, _ = fixture(tmp_path)
    output = tmp_path / "run" / "title-aliases.json"
    arguments = ["--database", str(database), "--manifest", str(manifest), "--output", str(output)]
    assert title_aliases.main(arguments) == 0
    assert title_aliases.load_aliases(output, database) == {1: "Verified Public Title"}
    with pytest.raises(SystemExit):
        title_aliases.main(arguments)
