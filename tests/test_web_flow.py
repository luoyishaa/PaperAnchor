from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from paperanchor.web import create_app


def _pdf_bytes() -> bytes:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 90), "Retrieval augmented generation combines retrieved evidence and generation.")
        return document.tobytes()


def test_upload_question_source_highlight_and_notes(tmp_path: Path) -> None:
    client = TestClient(create_app(
        tmp_path / "library.sqlite3",
        generate=lambda _system, _prompt: "It combines evidence and generation. [E1]",
    ))
    data = _pdf_bytes()
    upload = client.post("/api/papers", files={"file": ("sample.pdf", data, "application/pdf")})
    assert upload.status_code == 200
    paper = upload.json()["paper"]
    assert paper["title"] == "sample"
    assert paper["passage_count"] == 1
    assert client.post("/api/papers", files={"file": ("sample.pdf", data, "application/pdf")}).json()["changed"] is False

    answer_response = client.post("/api/ask", json={"question": "What is retrieval augmented generation?"})
    assert answer_response.status_code == 200
    answer = answer_response.json()
    assert answer["status"] == "answered"
    evidence = answer["evidence"][0]
    assert evidence["paper_id"] == paper["id"]
    assert evidence["page_number"] == 1
    assert evidence["bbox"][2] > evidence["bbox"][0]

    page = client.get(f"/api/papers/{paper['id']}/pages/1")
    assert page.status_code == 200
    assert page.json()["width"] > evidence["bbox"][2]
    image = client.get(f"/api/papers/{paper['id']}/pages/1/image")
    assert image.status_code == 200
    assert image.content.startswith(b"\x89PNG")
    assert client.get(f"/api/papers/{paper['id']}/file").content == data

    notes_url = f"/api/papers/{paper['id']}/annotations?page=1"
    assert [note["author"] for note in client.get(notes_url).json()] == ["agent"]
    user_note = client.post("/api/annotations", json={
        "paper_id": paper["id"], "page_number": 1,
        "bbox": evidence["bbox"], "body": "Check this passage.",
    })
    assert user_note.status_code == 200
    assert [note["author"] for note in client.get(notes_url).json()] == ["agent", "user"]


def test_empty_evidence_does_not_call_model(tmp_path: Path) -> None:
    def unexpected_call(_system: str, _prompt: str) -> str:
        raise AssertionError("model should not run without evidence")

    client = TestClient(create_app(tmp_path / "library.sqlite3", generate=unexpected_call))
    answer = client.post("/api/ask", json={"question": "Unknown topic?"})
    assert answer.status_code == 200
    assert answer.json()["status"] == "no_evidence"
    assert client.get("/api/papers/999/pages/1/image").status_code == 404


def test_invalid_upload_does_not_leave_a_paper(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "library.sqlite3"))
    response = client.post("/api/papers", files={"file": ("broken.pdf", b"not a PDF", "application/pdf")})
    assert response.status_code == 422
    assert client.get("/api/papers").json() == []
