from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

import numpy as np

from paperanchor.discovery import parse_arxiv, parse_semantic_scholar
from paperanchor.web import create_app
from paperanchor.library import PaperLibrary


ARXIV_FEED = b'''<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry><id>http://arxiv.org/abs/2401.12345v2</id><title> Test  Paper </title>
    <summary> A useful abstract. </summary><published>2024-01-22T00:00:00Z</published>
    <author><name>Jane Doe</name></author><arxiv:doi>10.1/test</arxiv:doi>
  </entry>
</feed>'''


def test_discovery_adapters_normalize_identity_and_dates():
    arxiv = parse_arxiv(ARXIV_FEED)[0]
    assert (arxiv.arxiv_id, arxiv.year, arxiv.title) == (
        "2401.12345v2", 2024, "Test Paper")
    s2 = parse_semantic_scholar({"data": [{
        "paperId": "s2-1", "title": "Test Paper", "year": 2024,
        "externalIds": {"ArXiv": "2401.12345", "DOI": "10.1/test"},
        "citationCount": 7, "authors": [{"name": "Jane Doe"}],
    }]})[0]
    assert s2.importable and s2.arxiv_id == "2401.12345"
    assert s2.citation_count == 7


def test_remote_search_and_import_keep_metadata_with_local_pdf(tmp_path: Path):
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 90), "Evidence in downloaded paper")
        content = document.tobytes()

    class FakeDiscovery:
        def search(self, query, source, limit):
            assert (query, source, limit) == ("topic", "arxiv", 10)
            return parse_arxiv(ARXIV_FEED)

        def arxiv_record(self, arxiv_id):
            assert arxiv_id == "2401.12345v2"
            return parse_arxiv(ARXIV_FEED)[0]

        def arxiv_pdf(self, arxiv_id):
            return content

    class FakeEmbedding:
        name = "test-embedding"

        def embed_passages(self, texts):
            return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]

        def embed_query(self, text):
            return np.array([1.0, 0.0], dtype=np.float32)

    with TestClient(create_app(tmp_path / "library.sqlite3",
                               discovery=FakeDiscovery(),
                               embedding_provider=FakeEmbedding())) as client:
        space = client.post("/api/spaces", json={"name": "Topic"}).json()
        found = client.get("/api/discover", params={"q": "topic", "source": "arxiv"})
        assert found.status_code == 200
        assert found.json()[0]["published"] == "2024-01-22"
        imported = client.post("/api/discover/import", json={
            "arxiv_id": "2401.12345v2", "space_id": space["id"]})
        assert imported.status_code == 200
        paper = imported.json()["paper"]
        assert paper["year"] == 2024
        assert paper["title"] == "Test Paper"
        assert client.get("/api/papers", params={"space_id": space["id"]}).json()[0]["id"] == paper["id"]
        assert client.get(f"/api/papers/{paper['id']}/file").content == content
        assert client.post("/api/discover/import", json={
            "arxiv_id": "../../../local", "space_id": space["id"]}).status_code == 422


def test_arxiv_filename_year_is_labeled_as_an_inference(tmp_path: Path):
    path = tmp_path / "1706.03762v7.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 90), "Attention evidence")
        document.save(path)
    PaperLibrary(tmp_path / "library.sqlite3").index_pdf(path)
    with TestClient(create_app(tmp_path / "library.sqlite3")) as client:
        paper = client.get("/api/papers").json()[0]
    assert paper["year"] == 2017
    assert paper["year_basis"] == "arxiv_filename"
