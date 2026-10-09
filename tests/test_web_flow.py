from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient
import numpy as np
import sqlite3
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from paperanchor.web import create_app
from paperanchor.library import PaperLibrary
from paperanchor.semantic import SemanticIndex
from paperanchor.llm import ModelConfigError, ModelRequestError
from paperanchor.agent import ResearchResult
from paperanchor.rag import Answer, Evidence


def _pdf_bytes() -> bytes:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 90), "Retrieval augmented generation combines retrieved evidence and generation.")
        return document.tobytes()


def _grounded_model(system: str, _prompt: str) -> str:
    if "Return only JSON" in system:
        return '{"sufficient": true, "missing": "", "revised_query": "", "follow_up_question": ""}'
    return "It combines evidence and generation. [E1]"


def test_blank_question_is_rejected_before_model_call(tmp_path: Path) -> None:
    def unexpected(*_):
        raise AssertionError("Blank questions must not reach the model")

    client = TestClient(create_app(tmp_path / "library.sqlite3", generate=unexpected))
    response = client.post("/api/ask", json={"question": " \n\t "})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "question"]


def test_upload_indexes_pdf_off_event_loop(tmp_path: Path, monkeypatch) -> None:
    original = PaperLibrary.index_pdf
    indexing_threads = []
    loop_threads = []

    def index(self, *args, **kwargs):
        indexing_threads.append(threading.get_ident())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PaperLibrary, "index_pdf", index)
    app = create_app(tmp_path / "library.sqlite3", generate=_grounded_model)

    @app.middleware("http")
    async def record_loop(request, call_next):
        loop_threads.append(threading.get_ident())
        return await call_next(request)

    response = TestClient(app).post(
        "/api/papers", files={"file": ("sample.pdf", _pdf_bytes(), "application/pdf")})
    assert response.status_code == 200
    assert indexing_threads and loop_threads
    assert indexing_threads[0] != loop_threads[0]


def test_concurrent_duplicate_uploads_keep_one_indexed_pdf(tmp_path: Path) -> None:
    app = create_app(tmp_path / "library.sqlite3", generate=_grounded_model)
    barrier = threading.Barrier(4)
    data = _pdf_bytes()

    def upload():
        barrier.wait()
        return TestClient(app).post(
            "/api/papers", files={"file": ("sample.pdf", data, "application/pdf")})

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: upload(), range(4)))
    assert all(response.status_code == 200 for response in responses)
    assert sum(response.json()["changed"] for response in responses) == 1
    client = TestClient(app)
    papers = client.get("/api/papers").json()
    assert len(papers) == 1
    assert client.get(f"/api/papers/{papers[0]['id']}/file").content == data
    assert not list((tmp_path / "uploads").glob(".upload-*"))


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


def test_glm_auto_visual_question_uses_page_citation_without_fake_annotation(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("PAPERANCHOR_PROVIDER", "glm")
    monkeypatch.setattr("paperanchor.web.choose_visual_pages", lambda *_args, **_kwargs: (object(),))
    def visual_answer(_question, _pages):
        evidence = Evidence("E1", 0, 1, tmp_path / "sample.pdf", "sample", 1,
                            "PDF page", (0, 0, 0, 0))
        return ResearchResult(Answer("answered", "Figure result [E1]",
                                     (evidence,), ("E1",)), "papers", (), None)
    monkeypatch.setattr("paperanchor.web.answer_visual", visual_answer)
    client = TestClient(create_app(tmp_path / "library.sqlite3",
                                   generate=lambda *_: "unused"))
    paper = client.post("/api/papers", files={
        "file": ("sample.pdf", _pdf_bytes(), "application/pdf")
    }).json()["paper"]
    response = client.post("/api/ask", json={"question": "What does Figure 1 show?"})
    assert response.status_code == 200
    assert response.json()["text"] == "Figure result [E1]"
    assert client.get(f"/api/papers/{paper['id']}/annotations?page=1").json() == []


def test_visual_region_upload_answer_highlight_and_conversation(tmp_path, monkeypatch):
    from paperanchor.vision import answer_visual
    monkeypatch.setenv("PAPERANCHOR_PROVIDER", "glm")
    payloads = []
    def complete(_system, content):
        payloads.append(content)
        return "The blue method is the baseline. [E1]"
    monkeypatch.setattr("paperanchor.web.answer_visual",
        lambda question, pages: answer_visual(question, pages, complete=complete))
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.draw_rect((80, 100, 330, 270))
        page.insert_text((90, 240), "Blue baseline")
        page.insert_text((80, 300), "Figure 3: Model comparison.")
        data = doc.tobytes()
    client = TestClient(create_app(tmp_path / "library.sqlite3", generate=lambda *_: "unused"))
    paper = client.post("/api/papers", files={"file": ("figure.pdf", data, "application/pdf")}).json()["paper"]
    response = client.post("/api/ask", json={"question": "What does Figure 3 show?", "paper_ids": [paper["id"]]})
    assert response.status_code == 200
    answer = response.json()
    source = answer["evidence"][0]
    assert source["visual_region_id"] is not None
    assert source["source_kind"] == "visual_region"
    assert source["source_label"] == "Figure 3"
    from hashlib import sha256
    assert source["source_hash"] == sha256(data).hexdigest()
    assert source["bbox"] == source["source_spans"][0]["bbox"]
    assert len([p for p in payloads[0] if p["type"] == "image_url"]) == 2
    annotations = client.get(f"/api/papers/{paper['id']}/annotations?page=1").json()
    assert annotations[0]["bbox"] == source["bbox"]
    turns = client.get(f"/api/conversations/{answer['conversation_id']}/turns").json()
    assert turns[-1]["answer"]["evidence"][0]["visual_region_id"] == source["visual_region_id"]


def test_rotated_reader_geometry_image_and_original_use_explicit_frames(tmp_path):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=700)
        page.insert_text((80, 600), "Bottom of unrotated source")
        page.set_rotation(90)
        data = doc.tobytes()
    client = TestClient(create_app(tmp_path / "library.sqlite3"))
    pid = client.post("/api/papers", files={"file": ("rotated.pdf", data, "application/pdf")}).json()["paper"]["id"]
    geometry = client.get(f"/api/papers/{pid}/pages/1").json()
    assert (geometry["width"], geometry["height"]) == (400, 700)
    pix = pymupdf.Pixmap(client.get(f"/api/papers/{pid}/pages/1/image").content)
    assert (pix.width, pix.height) == (640, 1120)
    assert client.get(f"/api/papers/{pid}/file").content == data


def test_missing_model_key_is_reported_to_the_browser(tmp_path: Path) -> None:
    def missing_key(_system: str, _prompt: str) -> str:
        raise ModelConfigError("Set DEEPSEEK_API_KEY in .env")

    client = TestClient(create_app(tmp_path / "library.sqlite3", generate=missing_key))
    client.post("/api/papers", files={"file": ("sample.pdf", _pdf_bytes(), "application/pdf")})
    response = client.post("/api/ask", json={"question": "What is retrieval augmented generation?"})
    assert response.status_code == 400
    assert "DEEPSEEK_API_KEY" in response.json()["detail"]


def test_provider_balance_failure_is_actionable_without_leaking_raw_error(tmp_path: Path) -> None:
    def no_balance(_system: str, _prompt: str) -> str:
        raise ModelRequestError("deepseek", 402)

    client = TestClient(create_app(tmp_path / "library.sqlite3", generate=no_balance))
    client.post("/api/papers", files={"file": ("sample.pdf", _pdf_bytes(), "application/pdf")})
    response = client.post("/api/ask", json={"question": "What is retrieval augmented generation?"})
    assert response.status_code == 503
    assert "insufficient balance" in response.json()["detail"]


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


def test_search_and_bulk_add_existing_papers(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "library.sqlite3"))
    space_id = client.post("/api/spaces", json={"name": "Reading list"}).json()["id"]
    for title in ("Alpha methods", "Beta methods", "Gamma systems"):
        with pymupdf.open() as document:
            document.new_page().insert_text((72, 90), title)
            data = document.tobytes()
        assert client.post("/api/papers", files={
            "file": (f"{title}.pdf", data, "application/pdf")
        }).status_code == 200

    url = f"/api/spaces/{space_id}/available-papers"
    first = client.get(url, params={"q": "methods", "limit": 1}).json()
    assert first["total"] == 2
    assert [item["title"] for item in first["items"]] == ["Alpha methods"]
    second = client.get(url, params={"q": "methods", "limit": 1, "offset": 1}).json()
    assert [item["title"] for item in second["items"]] == ["Beta methods"]
    ids = [first["items"][0]["id"], second["items"][0]["id"]]
    assert client.post(f"/api/spaces/{space_id}/papers/batch", json={
        "paper_ids": ids
    }).json()["added"] == 2
    assert client.get(url).json()["total"] == 1
    assert client.post(f"/api/spaces/{space_id}/papers/batch", json={
        "query": "systems"
    }).json()["added"] == 1
    assert client.get(url).json()["total"] == 0
    assert len(client.get(f"/api/papers?space_id={space_id}").json()) == 3
    assert client.post(f"/api/spaces/{space_id}/papers/batch", json={
        "paper_ids": [ids[0], 99999]
    }).status_code == 400


def test_delete_imported_paper_removes_all_indexed_data_and_owned_pdf(tmp_path: Path) -> None:
    database = tmp_path / "library.sqlite3"
    client = TestClient(create_app(database, generate=_grounded_model))
    paper = client.post("/api/papers", files={
        "file": ("sample.pdf", _pdf_bytes(), "application/pdf")
    }).json()["paper"]
    paper_id = paper["id"]
    stored = PaperLibrary(database).get_paper(paper_id).file_path
    assert stored.exists()
    client.post("/api/ask", json={"question": "retrieval generation"})
    assert client.get(f"/api/papers/{paper_id}/annotations?page=1").json()
    SemanticIndex(database)
    with sqlite3.connect(database) as connection:
        passage_id = connection.execute(
            "SELECT id FROM passages WHERE paper_id=?", (paper_id,)
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO passage_embeddings (passage_id, model, dimension, vector) "
            "VALUES (?, 'test-model', 2, ?)",
            (passage_id, np.array([1.0, 0.0], dtype=np.float32).tobytes()),
        )
    assert SemanticIndex(database).count("test-model") == 1

    response = client.delete(f"/api/papers/{paper_id}")

    assert response.status_code == 200
    assert client.get("/api/papers").json() == []
    assert client.get(f"/api/papers/{paper_id}/file").status_code == 404
    assert client.get("/api/search?q=retrieval").json() == []
    assert SemanticIndex(database).count("test-model") == 0
    assert not stored.exists()
    assert client.delete(f"/api/papers/{paper_id}").status_code == 404


def test_delete_indexed_external_pdf_preserves_original_file(tmp_path: Path) -> None:
    database = tmp_path / "library.sqlite3"
    client = TestClient(create_app(database))
    external = tmp_path / "source.pdf"
    external.write_bytes(_pdf_bytes())
    paper_id = PaperLibrary(database).index_pdf(external).paper_id

    result = client.delete(f"/api/papers/{paper_id}")

    assert result.status_code == 200
    assert result.json()["file_was_managed"] is False
    assert external.exists()
    assert client.get("/api/papers").json() == []


def test_comparison_uses_each_paper_and_rejects_ambiguous_count(tmp_path: Path) -> None:
    def model(system: str, prompt: str) -> str:
        if "collection overview" in system:
            return '{"each_paper": true, "requested_count": 2}'
        if "Answer only from the supplied evidence" in system:
            assert "First paper" in prompt and "Second paper" in prompt
            return "First paper uses retrieval [E1]. Second paper uses graphs [E2]."
        raise AssertionError("Unexpected model call")

    def pdf(text: str) -> bytes:
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((72, 90), text)
            return document.tobytes()

    client = TestClient(create_app(tmp_path / "library.sqlite3", generate=model))
    first = client.post("/api/papers", files={"file": (
        "first.pdf", pdf("Abstract First paper uses retrieval."), "application/pdf")
    }).json()["paper"]["id"]
    second = client.post("/api/papers", files={"file": (
        "second.pdf", pdf("Abstract Second paper uses graphs."), "application/pdf")
    }).json()["paper"]["id"]

    result = client.post("/api/ask", json={"question": "Compare these two papers",
                                             "allow_general_knowledge": True}).json()
    assert result["status"] == "answered"
    assert result["basis"] == "papers"
    assert {item["paper_id"] for item in result["evidence"]} == {first, second}

    mismatch = client.post("/api/ask", json={"question": "Compare these two papers",
                                               "paper_ids": [first]}).json()
    assert mismatch["status"] == "insufficient_evidence"
    assert mismatch["basis"] == "none"
    assert "2" in mismatch["text"] and "1" in mismatch["text"]


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


def test_existing_block_index_builds_default_units_on_restart(tmp_path: Path,
                                                               monkeypatch) -> None:
    class Embedding:
        name = "test-multilingual"

        def embed_passages(self, texts):
            return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]

        def embed_query(self, _text):
            return np.array([1.0, 0.0], dtype=np.float32)

    database = tmp_path / "library.sqlite3"
    monkeypatch.setenv("PAPERANCHOR_REPRESENTATION", "blocks")
    with TestClient(create_app(database, embedding_provider=Embedding())) as client:
        upload = client.post("/api/papers", files={
            "file": ("sample.pdf", _pdf_bytes(), "application/pdf")
        })
        paper_id = upload.json()["paper"]["id"]
        for _ in range(100):
            status = client.get("/api/semantic/status").json()
            if status["indexed_passages"] == 1:
                break
            time.sleep(0.02)
        assert status["indexed_passages"] == 1

    monkeypatch.delenv("PAPERANCHOR_REPRESENTATION")
    with TestClient(create_app(database, embedding_provider=Embedding())) as client:
        for _ in range(100):
            status = client.get("/api/semantic/status").json()
            if status["indexed_passages"] == status["total_passages"]:
                break
            time.sleep(0.02)
        assert status["representation"] == "units"
        assert status["total_passages"] > 0
        assert status["indexed_passages"] == status["total_passages"]
        assert client.get("/api/search?q=retrieval").json()[0]["paper_id"] == paper_id
