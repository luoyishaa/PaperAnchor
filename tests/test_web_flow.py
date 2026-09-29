from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient
import numpy as np
import time
import json

from paperanchor.web import create_app
from paperanchor.library import PaperLibrary
from paperanchor.llm import ModelConfigError


def _pdf_bytes() -> bytes:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 90), "Retrieval augmented generation combines retrieved evidence and generation.")
        return document.tobytes()


def _grounded_model(system: str, _prompt: str) -> str:
    if "Return only JSON" in system:
        return '{"sufficient": true, "missing": "", "revised_query": "", "follow_up_question": ""}'
    return "It combines evidence and generation. [E1]"


def test_upload_question_source_highlight_and_notes(tmp_path: Path) -> None:
    client = TestClient(create_app(
        tmp_path / "library.sqlite3",
        generate=_grounded_model,
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
    assert client.post("/api/ask", json={
        "question": "What is retrieval augmented generation?"}).status_code == 200
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


def test_missing_model_key_is_reported_to_the_browser(tmp_path: Path) -> None:
    def missing_key(_system: str, _prompt: str) -> str:
        raise ModelConfigError("Set DEEPSEEK_API_KEY in .env")

    client = TestClient(create_app(tmp_path / "library.sqlite3", generate=missing_key))
    client.post("/api/papers", files={"file": ("sample.pdf", _pdf_bytes(), "application/pdf")})
    response = client.post("/api/ask", json={"question": "What is retrieval augmented generation?"})
    assert response.status_code == 400
    assert "DEEPSEEK_API_KEY" in response.json()["detail"]


def test_missing_pdf_is_reported_in_library_and_reader_api(tmp_path: Path) -> None:
    database = tmp_path / "library.sqlite3"
    client = TestClient(create_app(database))
    paper = client.post("/api/papers", files={
        "file": ("sample.pdf", _pdf_bytes(), "application/pdf")
    }).json()["paper"]
    PaperLibrary(database).get_paper(paper["id"]).file_path.unlink()
    assert client.get("/api/papers").json()[0]["file_available"] is False
    assert client.get(f"/api/papers/{paper['id']}/pages/1/image").status_code == 410


def test_ask_returns_subquestion_evidence_locations(tmp_path: Path) -> None:
    calls = []
    def model(system, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"sufficient": False, "subquestions": [
                {"question": "What is retrieved?", "query": "retrieval evidence"}]})
        return _grounded_model(system, prompt)
    client = TestClient(create_app(tmp_path / "library.sqlite3", generate=model))
    client.post("/api/papers", files={"file": ("sample.pdf", _pdf_bytes(), "application/pdf")})
    response = client.post("/api/ask", json={"question": "retrieval generation", "enable_decomposition": True})
    assert response.status_code == 200
    answer = response.json()
    assert answer["status"] == "answered"
    task = answer["subquestions"][0]
    assert task["sufficient"] is True
    assert task["passage_ids"] == [answer["evidence"][0]["passage_id"]]
    assert answer["evidence"][0]["page_number"] == 1


def test_invalid_upload_does_not_leave_a_paper(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "library.sqlite3"))
    response = client.post("/api/papers", files={"file": ("broken.pdf", b"not a PDF", "application/pdf")})
    assert response.status_code == 422
    assert client.get("/api/papers").json() == []


def test_space_filters_questions_and_can_reuse_an_imported_paper(tmp_path: Path) -> None:
    client = TestClient(create_app(
        tmp_path / "library.sqlite3",
        generate=_grounded_model,
    ))
    space = client.post("/api/spaces", json={"name": "Transformers"}).json()
    assert space["name"] == "Transformers"
    uploaded = client.post("/api/papers", data={"space_id": space["id"]},
                           files={"file": ("sample.pdf", _pdf_bytes(), "application/pdf")})
    assert uploaded.status_code == 200
    paper_id = uploaded.json()["paper"]["id"]
    assert len(client.get(f"/api/papers?space_id={space['id']}").json()) == 1
    assert client.post("/api/ask", json={
        "question": "What is retrieval augmented generation?", "space_id": space["id"]
    }).json()["status"] == "answered"

    other = client.post("/api/spaces", json={"name": "Other topic"}).json()
    assert client.post("/api/ask", json={
        "question": "What is retrieval augmented generation?", "space_id": other["id"]
    }).json()["status"] == "no_evidence"
    assert client.post(f"/api/spaces/{other['id']}/papers/{paper_id}").status_code == 200
    assert len(client.get(f"/api/papers?space_id={other['id']}").json()) == 1
    assert client.delete(f"/api/spaces/{other['id']}/papers/{paper_id}").status_code == 200
    assert client.delete(f"/api/spaces/{other['id']}").status_code == 200


def test_background_semantic_job_completes_and_is_visible_to_search(tmp_path: Path) -> None:
    class Embedding:
        name = "test-multilingual"

        def embed_passages(self, texts):
            return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]

        def embed_query(self, _text):
            return np.array([1.0, 0.0], dtype=np.float32)

    with TestClient(create_app(tmp_path / "library.sqlite3",
                               embedding_provider=Embedding())) as client:
        upload = client.post("/api/papers", files={
            "file": ("sample.pdf", _pdf_bytes(), "application/pdf")
        })
        assert upload.status_code == 200
        job_id = upload.json()["semantic_job_id"]
        assert job_id is not None
        for _ in range(100):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["state"] in ("succeeded", "failed"):
                break
            time.sleep(0.02)
        assert job["state"] == "succeeded"
        assert client.get("/api/semantic/status").json()["indexed_passages"] == 1
        assert client.get("/api/search?q=注意力").json()[0]["paper_id"] == upload.json()["paper"]["id"]
