"""Check that the benchmark scores evidence, not just paper identity."""

from pathlib import Path

from evals.retrieval_benchmark import (evaluate_depths, load_cases, normalized,
                                       resolve_support_sets, score_hits)
from paperanchor.library import PassageHit


def hit(passage_id: int) -> PassageHit:
    return PassageHit(passage_id, 1, Path("a.pdf"), "A", 2, "evidence",
                      (0.0, 0.0, 1.0, 1.0), 1.0)


def test_complete_support_requires_every_passage_in_a_group() -> None:
    groups = [{11, 12}, {21}]
    partial = score_hits((hit(11), hit(99)), groups, [1, 2])
    assert partial["any_support"]["1"] == 1
    assert partial["complete_support"]["2"] == 0

    alternative = score_hits((hit(99), hit(21)), groups, [1, 2])
    assert alternative["complete_support"]["1"] == 0
    assert alternative["complete_support"]["2"] == 1
    assert alternative["mrr"] == 0.5


def test_source_anchor_resolves_within_one_paper_page() -> None:
    case = {"support_sets": [[{"paper": "a.pdf", "page": 2,
                              "anchor": "Factual  evidence"}]]}
    by_page = {("a.pdf", 2): [(5, normalized("Factual\n evidence on page."))],
               ("b.pdf", 2): [(6, normalized("Factual evidence on page."))]}
    groups, missing = resolve_support_sets(case, by_page)
    assert groups == [{5}]
    assert missing == []


def test_duplicate_question_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    row = ('{"id":"Q1","answerability":"unanswerable","support_sets":[],"split":"dev"}\n')
    path.write_text(row * 2, encoding="utf-8")
    try:
        load_cases(path)
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("duplicate IDs accepted")


def test_each_at_k_uses_the_real_retriever_depth() -> None:
    class DepthSensitiveRetriever:
        def search(self, question, limit):
            assert question == "question"
            return ((hit(99), hit(11)) if limit == 5 else (hit(11), hit(99)))

    scored = evaluate_depths(DepthSensitiveRetriever(), "question", [{11}], [5, 10])
    assert scored["complete_support"]["5"] == 1
    assert scored["first_support_rank"] == 1

    class MissingAtFive:
        def search(self, question, limit):
            return ((hit(99),) if limit == 5 else (hit(11), hit(99)))

    scored = evaluate_depths(MissingAtFive(), "question", [{11}], [5, 10])
    assert scored["complete_support"] == {"5": 0, "10": 1}
