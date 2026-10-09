from hashlib import sha256
import json

import fitz
import numpy as np
import pytest

from evals.qasper_benchmark.build import build_library, validate_library
from evals.qasper_benchmark import build
from paperanchor.semantic import DEFAULT_EMBEDDING_MODEL


def test_staging_build_uses_final_pdf_path_and_reports_missing_vectors(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Evidence retrieval in research papers")
        page.insert_text((72, 100), "The experiment measures evidence retrieval across documents.")
        document.save(pdf_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"candidates": [{
        "paper_id": "1234.56789", "split": "validation", "title": "Evidence retrieval",
        "status": "ready", "pdf_path": str(pdf_path),
        "sha256": sha256(pdf_path.read_bytes()).hexdigest(),
    }]}), encoding="utf-8")
    database = tmp_path / "staging.sqlite3"
    result = build_library(manifest, database, cache_dir=tmp_path / "models",
                           embeddings=False, minimum_papers=1)
    assert result["status"] == "incomplete"
    assert result["validation"]["valid"]
    assert not result["validation"]["semantic_complete"]
    assert [s["papers"] for s in result["validation"]["spaces"]] == [1, 1]
    assert result["mappings"][0]["pdf_path"] == str(pdf_path.resolve())
    resumed = build_library(manifest, database, cache_dir=tmp_path / "models",
                            embeddings=False, minimum_papers=1)
    assert resumed["mappings"] == result["mappings"]
    with fitz.open() as replacement:
        page = replacement.new_page()
        page.insert_text((72, 72), "A different valid PDF must not pass validation")
        replacement.save(pdf_path)
    validation = validate_library(database, result["mappings"])
    assert not validation["valid"]
    assert any("content changed" in error for error in validation["errors"])
    pdf_path.unlink()
    assert not validate_library(database, result["mappings"])["valid"]


def test_preparation_threshold_does_not_create_database(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"candidates": []}', encoding="utf-8")
    database = tmp_path / "staging.sqlite3"
    with pytest.raises(ValueError, match="preparation incomplete"):
        build_library(manifest, database, cache_dir=tmp_path)
    assert not database.exists()


def test_complete_staging_builds_block_and_unit_vectors(tmp_path, monkeypatch):
    class Provider:
        name = DEFAULT_EMBEDDING_MODEL

        def count_untruncated_tokens(self, text):
            return len(text.split())

        def embed_passages(self, texts):
            return [np.array([1., 0., 0.], dtype=np.float32) for text in texts]

        def embed_query(self, text):
            return np.array([1., 0., 0.], dtype=np.float32)

    monkeypatch.setattr(build, "cached_provider", lambda path: Provider())
    pdf_path = tmp_path / "paper.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Evidence retrieval in research papers")
        page.insert_text((72, 100), "The experiment measures evidence retrieval across documents.")
        document.save(pdf_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"candidates": [{
        "paper_id": "1234.56789", "split": "validation", "title": "Evidence retrieval",
        "status": "ready", "pdf_path": str(pdf_path),
        "sha256": sha256(pdf_path.read_bytes()).hexdigest(),
    }]}), encoding="utf-8")
    database = tmp_path / "staging.sqlite3"
    result = build_library(manifest, database, cache_dir=tmp_path / "models", minimum_papers=1)
    assert result["status"] == "ready"
    assert validate_library(database, result["mappings"])["semantic_complete"]
