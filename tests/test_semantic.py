from pathlib import Path

import numpy as np
import pymupdf
import pytest

from paperanchor.library import PaperLibrary, PassageHit
from paperanchor.semantic import HybridRetriever, RerankingRetriever, SemanticIndex
from paperanchor.spaces import SpaceStore


class SmallEmbedding:
    name = "test-multilingual-v1"

    def embed_passages(self, texts):
        return [np.array([1.0, 0.0] if "attention" in text.lower()
                         else [0.0, 1.0], dtype=np.float32) for text in texts]

    def embed_query(self, text):
        return np.array([1.0, 0.0] if "注意力" in text
                        else [0.0, 1.0], dtype=np.float32)


def _pdf(path: Path, text: str) -> None:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 90), text)
        document.save(path)


def test_chinese_question_finds_english_pdf_after_semantic_index(tmp_path: Path) -> None:
    library = PaperLibrary(tmp_path / "library.sqlite3")
    spaces = SpaceStore(library.database_path)
    attention = tmp_path / "attention.pdf"
    graph = tmp_path / "graph.pdf"
    _pdf(attention, "Attention mechanism connects tokens in a transformer.")
    _pdf(graph, "Graph retrieval walks edges in a network.")
    paper_id = library.index_pdf(attention).paper_id
    graph_id = library.index_pdf(graph).paper_id
    index = SemanticIndex(library.database_path)
    provider = SmallEmbedding()
    assert library.search("注意力") == ()

    result = index.build(provider, batch_size=1)
    assert result.indexed == result.total == 2
    assert index.build(provider).indexed == 0
    hits = HybridRetriever(library, index, provider).search("注意力如何工作?")
    assert hits[0].paper_id == paper_id
    assert hits[0].page_number == 1
    assert hits[0].bbox[2] > hits[0].bbox[0]

    topic = spaces.create_space("Graphs")
    spaces.add_paper(topic.id, graph_id)
    scoped = HybridRetriever(library, index, provider).search(
        "注意力如何工作?", space_id=topic.id
    )
    assert scoped and all(hit.paper_id != paper_id for hit in scoped)


def test_reindex_discards_old_embeddings(tmp_path: Path) -> None:
    library = PaperLibrary(tmp_path / "library.sqlite3")
    path = tmp_path / "paper.pdf"
    _pdf(path, "Attention mechanism.")
    library.index_pdf(path)
    index = SemanticIndex(library.database_path)
    provider = SmallEmbedding()
    index.build(provider)
    assert index.count(provider.name) == 1

    _pdf(path, "Graph retrieval.")
    library.index_pdf(path)
    assert index.count(provider.name) == 0
    assert index.build(provider).indexed == 1


def test_invalid_embedding_batch_does_not_leave_partial_vectors(tmp_path: Path) -> None:
    library = PaperLibrary(tmp_path / "library.sqlite3")
    path = tmp_path / "paper.pdf"
    _pdf(path, "Attention mechanism.")
    library.index_pdf(path)
    index = SemanticIndex(library.database_path)

    class BrokenEmbedding(SmallEmbedding):
        def embed_passages(self, texts):
            return []

    with pytest.raises(ValueError, match="wrong number"):
        index.build(BrokenEmbedding())
    assert index.count(BrokenEmbedding.name) == 0


def test_reranker_only_scores_hybrid_candidates(tmp_path: Path) -> None:
    library = PaperLibrary(tmp_path / "library.sqlite3")
    _pdf(tmp_path / "a.pdf", "Attention background.")
    _pdf(tmp_path / "b.pdf", "Attention method details.")
    for filename in ("a.pdf", "b.pdf"):
        library.index_pdf(tmp_path / filename)
    index = SemanticIndex(library.database_path)
    class FakeReranker:
        def score(self, question, texts):
            assert question == "Attention"
            assert len(texts) == 2
            return [1.0 if "method" in text else 0.0 for text in texts]

    hits = RerankingRetriever(HybridRetriever(library, index, SmallEmbedding()),
                              FakeReranker()).search("Attention", limit=1)
    assert "method details" in hits[0].text


def test_reranker_can_rescue_one_channel_evidence_before_fusion_truncation() -> None:
    def passage(number: int) -> PassageHit:
        return PassageHit(number, 1, Path("paper.pdf"), "Paper", 1,
                          "gold evidence" if number == 1 else f"distractor {number}",
                          (0.0, 0.0, 1.0, 1.0), 0.0)

    shared = tuple(passage(number) for number in range(2, 82))

    class Lexical:
        def search(self, question, limit, *, space_id=None):
            return (passage(1), *shared)[:limit]

    class Semantic:
        def count(self, model):
            return 81

        def search(self, question, provider, *, limit, space_id=None):
            return shared[:limit]

    class Reranker:
        def score(self, question, texts):
            return [1.0 if "gold evidence" in text else 0.0 for text in texts]

    initial = HybridRetriever(Lexical(), Semantic(), SmallEmbedding())
    assert all(hit.passage_id != 1 for hit in initial.search("question", limit=20))
    hits = RerankingRetriever(initial, Reranker()).search("question", limit=5)
    assert hits[0].passage_id == 1
