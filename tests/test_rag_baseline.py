from pathlib import Path

import pymupdf

from paperanchor.library import PaperLibrary
from paperanchor.rag import answer_question


def make_pdf(path: Path, text: str = "Retrieval augmented generation combines retrieved passages with a language model to answer questions.") -> None:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 90), text)
        document.save(path)


def test_indexed_paper_is_searchable_and_repeat_index_is_safe(tmp_path: Path) -> None:
    pdf = tmp_path / "example.pdf"
    make_pdf(pdf)
    library = PaperLibrary(tmp_path / "library.sqlite3")

    first = library.index_pdf(pdf)
    second = library.index_pdf(pdf)
    hits = library.search("retrieval augmented generation")

    assert first.changed is True
    assert first.passage_count > 0
    assert second.changed is False
    assert len(hits) == 1
    assert hits[0].paper_path == pdf.resolve()
    assert hits[0].page_number == 1
    assert "retrieved passages" in hits[0].text


def test_answer_keeps_a_checkable_evidence_reference(tmp_path: Path) -> None:
    pdf = tmp_path / "example.pdf"
    make_pdf(pdf)
    library = PaperLibrary(tmp_path / "library.sqlite3")
    library.index_pdf(pdf)

    result = answer_question(
        library,
        "What is retrieval augmented generation?",
        lambda _system, _user: "It combines retrieval and generation. [E1]",
    )

    assert result.status == "answered"
    assert result.cited_evidence_ids == ("E1",)
    assert result.evidence[0].page_number == 1
    assert result.evidence[0].paper_path == pdf.resolve()


def test_unavailable_evidence_never_becomes_a_paper_citation(tmp_path: Path) -> None:
    library = PaperLibrary(tmp_path / "library.sqlite3")
    result = answer_question(
        library,
        "What is the result of this paper?",
        lambda _system, _user: "This should not be called.",
    )

    assert result.status == "no_evidence"
    assert result.evidence == ()
    assert result.cited_evidence_ids == ()


def test_reindex_replaces_stale_passages(tmp_path: Path) -> None:
    pdf = tmp_path / "example.pdf"
    make_pdf(pdf, "Old topic: graph retrieval.")
    library = PaperLibrary(tmp_path / "library.sqlite3")
    library.index_pdf(pdf)

    make_pdf(pdf, "New topic: citation verification.")
    result = library.index_pdf(pdf)

    assert result.changed is True
    assert library.search("graph retrieval") == ()
    assert len(library.search("citation verification")) == 1


def test_answer_rejects_citation_outside_retrieved_evidence(tmp_path: Path) -> None:
    pdf = tmp_path / "example.pdf"
    make_pdf(pdf)
    library = PaperLibrary(tmp_path / "library.sqlite3")
    library.index_pdf(pdf)

    result = answer_question(
        library,
        "What is retrieval augmented generation?",
        lambda _system, _user: "An unsupported claim. [E99]",
    )

    assert result.status == "invalid_citation"
    assert result.cited_evidence_ids == ("E99",)
