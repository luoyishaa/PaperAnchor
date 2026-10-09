from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from paperanchor.web import create_app


def pdf_bytes() -> bytes:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 90), "Graph retrieval combines evidence and generation.")
        return document.tobytes()


def model(system: str, _user: str) -> str:
    if "Rewrite the new user question" in system:
        return '{"question":"How does graph retrieval combine evidence and generation?"}'
    if "collection overview" in system:
        return '{"each_paper":false,"requested_count":null}'
    if "Return only JSON" in system:
        return '{"sufficient":true,"missing":"","revised_query":"","follow_up_question":""}'
    return "It combines retrieved evidence and generation. [E1]"


def test_follow_up_resolves_for_retrieval_and_persists_per_space(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "library.sqlite3", generate=model))
    assert client.post("/api/papers", files={"file": ("paper.pdf", pdf_bytes(),
                                                 "application/pdf")}).status_code == 200
    first = client.post("/api/ask", json={"question": "What is graph retrieval?"}).json()
    assert first["status"] == "answered"
    second = client.post("/api/ask", json={
        "question": "How does it work?", "conversation_id": first["conversation_id"],
    }).json()
    assert second["conversation_id"] == first["conversation_id"]
    assert second["standalone_question"] == (
        "How does graph retrieval combine evidence and generation?")
    chats = client.get("/api/conversations?space_id=1").json()
    assert len(chats) == 1 and chats[0]["turn_count"] == 2
    history = client.get(f"/api/conversations/{first['conversation_id']}/turns?space_id=1").json()
    assert [turn["question"] for turn in history] == [
        "What is graph retrieval?", "How does it work?"]
    assert history[1]["answer"]["evidence"][0]["paper_path"].endswith("paper.pdf")
    other = client.post("/api/spaces", json={"name": "Other"}).json()
    assert client.get(f"/api/conversations/{first['conversation_id']}/turns?space_id={other['id']}").status_code == 404
