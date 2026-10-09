from pathlib import Path

import pymupdf
import numpy as np
from fastapi.testclient import TestClient

from paperanchor import cli
from paperanchor.library import PaperLibrary
from paperanchor.web import create_app


def _pdf(path: Path) -> None:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 90), "Retrieval augmented generation combines evidence and generation.")
        document.save(path)


def _model(system: str, _prompt: str) -> str:
    if "Return only JSON" in system:
        return '{"sufficient": true}'
    return "It combines evidence and generation. [E1]"


def test_cli_ask_uses_web_evidence_workflow(tmp_path, monkeypatch, capsys) -> None:
    class Embedding:
        name = "test-cli-embedding"
        def embed_passages(self, texts):
            return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]
        def embed_query(self, _text):
            return np.array([1.0, 0.0], dtype=np.float32)

    database = tmp_path / "library.sqlite3"
    paper = tmp_path / "sample.pdf"
    _pdf(paper)
    monkeypatch.setattr(cli, "generate_with_config", _model)
    monkeypatch.setattr(cli, "FastEmbedProvider", lambda **_: Embedding())
    assert cli.main(["--db", str(database), "index", str(paper)]) == 0
    capsys.readouterr()

    question = "What does retrieval augmented generation combine?"
    assert cli.main(["--db", str(database), "ask", question]) == 0
    output = capsys.readouterr().out
    with TestClient(create_app(database, generate=_model, embedding_provider=Embedding())) as client:
        web_answer = client.post("/api/ask", json={"question": question}).json()

    assert "Status: answered | Basis: papers" in output
    assert web_answer["status"] == "answered"
    assert web_answer["basis"] == "papers"
    assert web_answer["text"] in output
    assert f"PDF page {web_answer['evidence'][0]['page_number']}" in output


def test_numbered_pdf_uses_prominent_first_page_title(tmp_path: Path) -> None:
    paper = tmp_path / "2404.16130v2.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 80), "A Useful Paper Title", fontsize=17)
        page.insert_text((72, 120), "Author Name", fontsize=10)
        document.save(paper)
    library = PaperLibrary(tmp_path / "library.sqlite3")
    result = library.index_pdf(paper)
    assert library.get_paper(result.paper_id).title == "A Useful Paper Title"
