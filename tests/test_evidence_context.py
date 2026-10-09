from dataclasses import replace
from pathlib import Path

import pytest

from paperanchor.evidence_context import assemble_evidence
from paperanchor.library import PassageHit, SourceSpan


class Tokenizer:
    def count(self, text):
        return len(text)


def block(identifier, text, *, paper=1, page=1):
    bbox = (0.0, float(identifier), 10.0, float(identifier) + 1)
    return PassageHit(identifier, paper, Path("paper.pdf"), "T", page, text, bbox,
                      1.0, (SourceSpan(identifier, page, bbox),))


def child(parent, identifier, text):
    return replace(parent, passage_id=-identifier, text=text)


def pages(*items):
    result = {}
    for item in items:
        result.setdefault((item.paper_id, item.page_number), []).append(item)
    return result


def assemble(seeds, blocks, **kwargs):
    return assemble_evidence(seeds, blocks, Tokenizer(), strategy=kwargs.pop("strategy", "dedup_parent"),
                             **kwargs)


def test_seeds_precede_neighbors_and_selected_sources_are_not_neighbors():
    first, middle, last = block(1, "First."), block(2, "Between."), block(3, "Last.")
    result = assemble([child(first, 1, "First."), child(last, 2, "Last.")],
                      pages(first, middle, last), strategy="seed_first")
    assert [row["passage_id"] for row in result["chunks"]] == [-1, -2, 2]
    assert result["text"] == "[E1] T (page 1)\nFirst.\n\n[E2] T (page 1)\nLast.\n\n[E3] T (page 1)\nBetween."
    assert result["reference_tokens"] == len(result["text"])
    assert result["trace"]["neighbor_candidate_ids"] == [2]


def test_overlap_is_removed_by_source_interval_and_parent_missing_parts_restored():
    parent = block(1, "First. Second. Third. Fourth.")
    result = assemble([child(parent, 1, "First. Second."), child(parent, 2, "Second. Third.")],
                      pages(parent))
    assert [row["text"] for row in result["chunks"]] == ["First. Second.", "Third.", "Fourth."]
    assert all(result["text"].count(word) == 1 for word in ("First.", "Second.", "Third.", "Fourth."))
    for row in result["chunks"]:
        assert row["source_spans"][0]["passage_id"] == 1
        assert len(row["source_slices"]) == 1
        piece = row["source_slices"][0]
        assert parent.text[piece["start"]:piece["end"]] == row["text"]


def test_section_prefix_and_whitespace_fold_keep_source_mapping():
    parent = block(1, "First fact.\n\nSecond\t fact. Final fact.")
    result = assemble([child(parent, 1, "Results\nFirst fact. Second fact.")], pages(parent))
    first = result["chunks"][0]
    assert first["text"] == "Results\nFirst fact.\n\nSecond\t fact."
    assert first["source_mapping"] == "exact"
    assert result["chunks"][1]["text"] == "Final fact."


def test_ambiguous_occurrence_preserves_original_hit_without_claiming_intervals():
    parent = block(1, "Repeated. Interlude. Repeated.")
    result = assemble([child(parent, 1, "Repeated.")], pages(parent))
    assert result["chunks"][0]["text"] == "Repeated."
    assert result["chunks"][0]["source_mapping"] == "unlocated"
    assert result["chunks"][0]["source_slices"] == []
    assert result["chunks"][1]["text"] == parent.text


def test_arbitrary_body_prefix_is_not_discarded_and_missing_sources_remain_visible():
    parent = block(1, "Answer.")
    result = assemble([child(parent, 1, "A meaningful first line.\nAnswer.")], pages(parent))
    assert result["chunks"][0]["text"] == "A meaningful first line.\nAnswer."
    assert result["chunks"][0]["source_mapping"] == "unlocated"
    result = assemble([child(parent, 1, "Answer.")], {})
    assert result["chunks"][0]["text"] == "Answer."
    assert result["trace"]["mapped_seed_count"] == 0


def test_multiblock_source_slices_and_first_remaining_location_are_exact():
    first, second = block(1, "One."), block(2, "Two.")
    combined = replace(child(first, 2, "One.\nTwo."),
                       source_spans=first.source_spans + second.source_spans)
    result = assemble([child(first, 1, "One."), combined], pages(first, second))
    assert [row["text"] for row in result["chunks"]] == ["One.", "Two."]
    assert list(result["chunks"][1]["source_spans"]) == [{"passage_id": 2, "page_number": 1, "bbox": second.bbox}]
    assert result["chunks"][1]["bbox"] == second.bbox


def test_skip_oversized_seed_continues_and_sentence_prefix_also_continues():
    long = block(1, "x" * 200)
    small = block(2, "Fits.")
    result = assemble([long, small], pages(long, small), strategy="seed_first", budget=40)
    assert [row["passage_id"] for row in result["chunks"]] == [2]
    assert result["trace"]["events"][0]["action"] == "budget_skip"
    long = replace(long, text="A. " + "word " * 40 + ".")
    result = assemble([long, small], pages(long, small), strategy="seed_first", budget=50)
    assert [row["text"] for row in result["chunks"]] == ["A.", "Fits."]
    assert result["chunks"][0]["truncated"] is True
    assert result["chunks"][0]["source_slices"][0]["end"] == 2
    assert result["reference_tokens"] <= 50


def test_parent_budget_skip_does_not_prevent_smaller_later_neighbor():
    parent = block(1, "Seed. " + "left " * 20)
    neighbor = block(2, "Next.")
    result = assemble([child(parent, 1, "Seed.")], pages(parent, neighbor), budget=50)
    assert [row["text"] for row in result["chunks"]] == ["Seed.", "Next."]
    assert {row["kind"] for row in result["trace"]["events"] if row["action"] == "budget_skip"} == {"parent"}


def test_expansion_size_limits_and_same_page_are_respected():
    oversized, seed, neighbor = block(1, "x" * 513), block(2, "Seed."), block(3, "Next.")
    other = block(4, "Other page.", page=2)
    result = assemble([child(seed, 1, "Seed.")], pages(oversized, seed, neighbor, other))
    assert result["trace"]["neighbor_candidate_ids"] == [3]
    assert "Other page." not in result["text"]
    assert "x" * 513 not in result["text"]


def test_each_seed_adds_at_most_two_neighbors_in_existing_B_order():
    items = [block(index, f"Source {index}.") for index in range(1, 7)]
    combined = replace(child(items[1], 1, "Source 2.\nSource 5."),
                       source_spans=items[1].source_spans + items[4].source_spans)
    result = assemble([combined], pages(*items), strategy="seed_first")
    assert result["trace"]["neighbor_candidate_ids"] == [1, 3]


class Reranker:
    def __init__(self, scores=None):
        self.calls = []
        self.scores = scores

    def score(self, query, texts):
        self.calls.append((query, texts))
        return self.scores if self.scores is not None else [1.0] * len(texts)


def test_ranked_expansion_protects_seed_and_breaks_score_ties_by_incremental_tokens():
    first, second = block(1, "First. Long remaining content."), block(2, "Z.")
    reranker = Reranker()
    result = assemble([child(first, 1, "First."), child(second, 2, "Z.")], pages(first, second),
                      strategy="ranked_expansion", protect=1, query="Which?", reranker=reranker)
    assert [row["text"] for row in result["chunks"]] == ["First.", "Z.", "Long remaining content."]
    assert reranker.calls == [("Which?", ["Z.", "Long remaining content.", "Z."])]
    assert result["text"].count("Z.") == 1


def test_ranked_expansion_high_score_precedes_shorter_lower_score():
    first, second = block(1, "Protected. Important remaining answer."), block(2, "Z.")
    reranker = Reranker([1, 10, 1])
    result = assemble([child(first, 1, "Protected."), child(second, 2, "Z.")], pages(first, second),
                      strategy="ranked_expansion", protect=1, reranker=reranker)
    assert [row["text"] for row in result["chunks"]] == ["Protected.", "Important remaining answer.", "Z."]


def test_rerank_tie_cost_includes_headers():
    first = block(1, "Longer body.")
    second = replace(block(2, "Z.", paper=2), paper_title="A very long descriptive paper title")
    result = assemble([second, first], pages(first, second), strategy="ranked_expansion", protect=0,
                      reranker=Reranker())
    assert result["chunks"][0]["passage_id"] == 1


def test_budget_never_emits_only_a_section_prefix():
    parent = block(1, "Very long body with no sentence boundary")
    seed = child(parent, 1, "Introduction.\n" + parent.text)
    result = assemble([seed], pages(parent), budget=32)
    assert result["chunks"] == []


def test_restored_disjoint_source_pieces_are_compatible_with_existing_scorer():
    from evals.qasper_benchmark.align import align_evidence, build_document
    from evals.qasper_benchmark.retrieve import serialize_hit
    from evals.qasper_benchmark.score import _covered, chunk_intervals

    parent = block(1, "First. Second. Third. Fourth.")
    document = build_document(1, [serialize_hit(parent)])
    result = assemble([child(parent, 1, "Second. Third.")], pages(parent))
    intervals = [span for chunk in result["chunks"] for span in chunk_intervals(chunk, document)]
    assert _covered(align_evidence(parent.text, document), intervals, document)


@pytest.mark.parametrize("kwargs", [{"budget": 0}, {"protect": -1}, {"strategy": "unknown"},
                                   {"strategy": "ranked_expansion"}])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        assemble([], {}, **kwargs)


@pytest.mark.parametrize("scores", [[], [float("nan")]])
def test_invalid_rerank_scores_fail(scores):
    first = block(1, "Answer.")
    with pytest.raises(ValueError, match="scores"):
        assemble([first], pages(first), strategy="ranked_expansion", protect=0,
                 reranker=Reranker(scores))


def test_top20_limit_empty_and_wrong_source_page():
    sources = [block(index, f"Answer {index}.", page=index) for index in range(1, 22)]
    result = assemble(sources, pages(*sources), strategy="seed_first")
    assert len(result["chunks"]) == 20
    assert result["trace"]["seed_count"] == 20
    assert assemble([], {})["text"] == ""
    with pytest.raises(ValueError, match="paper/page"):
        assemble([], {(2, 1): [sources[0]]})
