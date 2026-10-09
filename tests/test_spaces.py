from pathlib import Path
import sqlite3

import pymupdf
import pytest

from paperanchor.library import PaperLibrary
from paperanchor.spaces import SpaceStore


def _paper(path: Path, text: str) -> None:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 90), text)
        document.save(path)


def test_existing_papers_are_migrated_and_queries_are_scoped(tmp_path: Path) -> None:
    library = PaperLibrary(tmp_path / "library.sqlite3")
    first_path, second_path = tmp_path / "first.pdf", tmp_path / "second.pdf"
    _paper(first_path, "Graph retrieval improves evidence selection.")
    _paper(second_path, "Graph retrieval and citations are studied.")
    first = library.index_pdf(first_path)
    second = library.index_pdf(second_path)

    spaces = SpaceStore(library.database_path)
    topic = spaces.create_space("Graph methods")
    spaces.add_paper(topic.id, first.paper_id)

    assert spaces.get_space(1).paper_count == 2
    assert spaces.get_space(topic.id).paper_count == 1
    assert {hit.paper_id for hit in library.search("graph retrieval")} == {
        first.paper_id, second.paper_id
    }
    assert {hit.paper_id for hit in library.search(
        "graph retrieval", space_id=topic.id
    )} == {first.paper_id}
    assert [paper.id for paper in library.list_papers(space_id=topic.id)] == [first.paper_id]

    spaces.remove_paper(topic.id, first.paper_id)
    assert library.search("graph retrieval", space_id=topic.id) == ()
    with pytest.raises(ValueError):
        spaces.remove_paper(1, first.paper_id)


def test_new_papers_remain_visible_in_all_papers(tmp_path: Path) -> None:
    library = PaperLibrary(tmp_path / "library.sqlite3")
    spaces = SpaceStore(library.database_path)
    path = tmp_path / "after-space.pdf"
    _paper(path, "Newly imported paper.")
    indexed = library.index_pdf(path)

    assert spaces.get_space(1).paper_count == 1
    assert library.list_papers(space_id=1)[0].id == indexed.paper_id


def test_available_papers_pages_and_bulk_add_large_library(tmp_path: Path) -> None:
    database = tmp_path / "library.sqlite3"
    PaperLibrary(database)
    spaces = SpaceStore(database)
    reading_list = spaces.create_space("Large reading list")
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """INSERT INTO papers (file_path, file_hash, title, page_count, indexed_at)
               VALUES (?, ?, ?, 1, '2026-01-01')""",
            ((f"/test/{index}.pdf", str(index), f"Paper {index:04d}")
             for index in range(1200)),
        )

    first, total = spaces.available_papers(reading_list.id, "Paper", limit=50)
    last, last_total = spaces.available_papers(reading_list.id, "Paper", limit=50, offset=1150)
    assert total == last_total == 1200
    assert len(first) == len(last) == 50
    assert first[0]["title"] == "Paper 0000"
    assert last[-1]["title"] == "Paper 1199"
    assert spaces.add_papers(reading_list.id, query="Paper") == 1200
    assert spaces.available_papers(reading_list.id)[1] == 0
    assert spaces.get_space(reading_list.id).paper_count == 1200
    assert spaces.available_papers(1, "%")[1] == 0
