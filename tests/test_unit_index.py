from pathlib import Path
import sqlite3

import numpy as np
import pymupdf

from paperanchor.library import PaperLibrary
from paperanchor import representation
from paperanchor.representation import build_units
from paperanchor.pdf import ExtractedPassage
from paperanchor.unit_index import UnitStore, UnitSemanticIndex


class TinyEmbeddings:
    name = "tiny"

    def embed_passages(self, texts):
        return [np.array([float("graph" in text.lower()), 1.0], dtype=np.float32)
                for text in texts]

    def embed_query(self, text):
        return np.array([float("graph" in text.lower()), 1.0], dtype=np.float32)


def make_pdf(path: Path) -> None:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Introduction")
        page.insert_text((72, 110), "Graph retrieval finds relevant passages.")
        page.insert_text((72, 145), "It combines textual and semantic evidence.")
        document.save(path)


def test_unit_links_multiple_real_source_blocks_and_preserves_notes(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf)
    library = PaperLibrary(tmp_path / "library.sqlite3")
    paper_id = library.index_pdf(pdf).paper_id
    note = library.add_annotation(paper_id, 1, (72, 90, 240, 120), "Read this")
    store = UnitStore(library)

    assert store.refresh() == 1
    hits = store.search("graph semantic")
    assert hits
    assert len(hits[0].source_spans) == 2
    assert {span.page_number for span in hits[0].source_spans} == {1}
    assert all(span.bbox[2] > span.bbox[0] for span in hits[0].source_spans)
    assert store.refresh() == 0
    assert library.list_annotations(paper_id, 1)[0].id == note.id

    index = UnitSemanticIndex(store)
    result = index.build(TinyEmbeddings())
    assert result.total == store.count_units()
    assert index.build(TinyEmbeddings()).indexed == 0
    assert index.search("graph", TinyEmbeddings())[0].source_spans


def test_token_budget_splits_long_block_without_losing_its_source() -> None:
    passage = ExtractedPassage(1, 0, "A sentence about graphs. " * 80,
                               (10.0, 10.0, 300.0, 500.0))
    units = build_units([passage], token_count=lambda text: len(text.split()),
                        max_tokens=32)
    assert len(units) > 1
    assert all(len((unit.section + " " + unit.text).split()) <= 32 for unit in units)
    assert all(unit.source_indexes == (0,) for unit in units)


def test_existing_unit_index_migrates_and_rebuilds_once(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf)
    library = PaperLibrary(tmp_path / "library.sqlite3")
    paper_id = library.index_pdf(pdf).paper_id
    with sqlite3.connect(library.database_path) as connection:
        connection.execute("""CREATE TABLE unit_index_state (
            paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
            file_hash TEXT NOT NULL, representation_version INTEGER NOT NULL,
            tokenizer_model TEXT NOT NULL)""")
        connection.execute(
            "INSERT INTO unit_index_state VALUES (?, ?, ?, ?)",
            (paper_id, "old", representation.REPRESENTATION_VERSION, ""),
        )

    store = UnitStore(library)
    assert store.refresh() == 1
    assert store.refresh() == 0
    with sqlite3.connect(library.database_path) as connection:
        assert connection.execute(
            "SELECT config_hash FROM unit_index_state WHERE paper_id=?", (paper_id,)
        ).fetchone()[0]


def test_tokenized_units_survive_search_and_budget_change_invalidates_vectors(
        tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf)
    library = PaperLibrary(tmp_path / "library.sqlite3")
    library.index_pdf(pdf)
    store = UnitStore(library)
    index = UnitSemanticIndex(store)

    class TokenizedEmbeddings(TinyEmbeddings):
        def count_untruncated_tokens(self, text):
            return len(text.split())

    provider = TokenizedEmbeddings()
    index.build(provider)
    count = index.count(provider.name)
    assert count > 0
    assert store.search("graph")
    assert index.count(provider.name) == count
    assert store.refresh(token_count=provider.count_untruncated_tokens,
                         tokenizer_model=provider.name, max_tokens=32) == 1
    assert index.count(provider.name) == 0
    assert index.build(provider).indexed == store.count_units()


def test_reindex_replaces_unit_vectors_without_orphans(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf)
    library = PaperLibrary(tmp_path / "library.sqlite3")
    library.index_pdf(pdf)
    store = UnitStore(library)
    index = UnitSemanticIndex(store)
    provider = TinyEmbeddings()
    index.build(provider)
    assert index.count(provider.name) > 0

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 90), "A different paper about graphs.")
        document.save(pdf)
    library.index_pdf(pdf)
    store.refresh()
    assert index.count(provider.name) == 0
    assert index.build(provider).indexed == store.count_units()
