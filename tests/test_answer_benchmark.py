from pathlib import Path

import pytest
import json

from evals.answer_benchmark import (answer_fingerprint, evaluate_case,
                                    score_saved_row, summarize,
                                    write_report, write_review_template)
from evals.review_answers import score_reviews
from evals.summarize_answers import combine
from paperanchor.library import PassageHit


def case(answerability="answerable"):
    return {
        "id": "Q1", "split": "dev", "language": "en", "category": "mechanism",
        "answerability": answerability, "review_status": "single_pass_source_check",
        "question": "How?", "expected_answer": "The paper says X.",
    }


class Retriever:
    def search(self, question, limit=5, *, space_id=None):
        assert question == "How?" and limit == 5 and space_id is None
        return (PassageHit(7, 1, Path("paper.pdf"), "Paper", 2,
                           "The method uses X.", (1, 2, 3, 4), 0.8),)


def test_answer_report_keeps_gold_citation_and_model_trace_separate_from_factuality():
    def model(system, prompt):
        if "Return only JSON" in system:
            return '{"sufficient": true}'
        return "The method uses X [E1]."

    row = evaluate_case(case(), [{7}], Retriever(), model)
    assert row["first_retrieval_complete_support"] is True
    assert row["final_evidence_complete_support"] is True
    assert row["cited_complete_support"] is True
    assert row["diagnosis"] == "needs_human_content_review"
    assert [call["stage"] for call in row["model_calls"]] == ["assessment", "answer"]
    assert row["evidence"][0]["page"] == 2
    assert row["human_review"]["factuality"] is None
    assert summarize([row])["cited_complete_support"] == 1


def test_complete_retrieval_can_still_be_rejected_by_assessment():
    row = evaluate_case(case(), [{7}], Retriever(),
                        lambda _system, _prompt: '{"sufficient": false}')
    assert row["final_evidence_complete_support"] is True
    assert row["basis"] == "none"
    assert row["diagnosis"] == "assessment_abstained_with_gold_evidence"


def test_answer_report_serializes_subquestion_state():
    calls = []
    def model(system, prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"sufficient": False, "subquestions": [
                {"question": "Explain X", "query": "How?"}]})
        return '{"sufficient": true}' if "Return only JSON" in system else "X [E1]."
    row = evaluate_case(case(), [{7}], Retriever(), model, enable_decomposition=True)
    assert row["subquestions"][0]["passage_ids"] == (7,)
    assert row["subquestions"][0]["sufficient"] is True
    assert summarize([row])["decomposed_cases"] == 1
    assert summarize([row])["model_calls"] == 4
    json.dumps(row)


def test_provisional_absence_is_reported_without_claiming_ground_truth_accuracy():
    row = evaluate_case(case("unanswerable"), [], Retriever(),
                        lambda _system, _prompt: '{"sufficient": false}')
    assert row["diagnosis"] == "provisional_abstention"
    assert row["cited_complete_support"] is None
    assert summarize([row])["provisional_absent_cases"] == 1
    assert summarize([row])["human_factuality_reviewed"] == 0


def test_invalid_citation_is_not_counted_as_supported():
    def model(system, prompt):
        return ('{"sufficient": true}' if "Return only JSON" in system
                else "Claim [E9].")

    row = evaluate_case(case(), [{7}], Retriever(), model)
    assert row["status"] == "invalid_citation"
    assert row["cited_complete_support"] is False
    assert row["diagnosis"] == "citation_format_failure"


def test_review_is_bound_to_exact_answer_and_counts_only_explicit_judgments():
    row = evaluate_case(case(), [{7}], Retriever(),
                        lambda system, _prompt: ('{"sufficient": true}'
                                                if "Return only JSON" in system
                                                else "X [E1]."))
    report = {"benchmark": "fixture", "timestamp_utc": "2026-09-24T00:00:00Z",
              "rows": [row]}
    review = {"id": "Q1", "answer_sha256": answer_fingerprint(row),
              "factuality": True, "citation_support": None,
              "abstention": None, "absence_label": None}
    scored = score_reviews(report, [review])
    assert scored["factuality"] == {"reviewed": 1, "passed": 1}
    assert scored["citation_support"] == {"reviewed": 0, "passed": 0}

    with pytest.raises(ValueError, match="Answer changed"):
        score_reviews(report, [{**review, "answer_sha256": "old"}])
    with pytest.raises(ValueError, match="factuality"):
        score_reviews(report, [{**review, "factuality": 1}])
    with pytest.raises(ValueError, match="Duplicate"):
        score_reviews(report, [review, review])


def test_revised_gold_can_rescore_saved_trace_without_running_model():
    row = evaluate_case(case(), [{99}], Retriever(),
                        lambda system, _prompt: ('{"sufficient": true}'
                                                if "Return only JSON" in system
                                                else "X [E1]."))
    assert row["diagnosis"] == "paper_answer_without_gold_support"
    revised = score_saved_row(row, [{99}, {7}])
    assert revised["diagnosis"] == "needs_human_content_review"
    assert revised["cited_complete_support"] is True
    assert revised["answer"] == row["answer"]
    assert revised["model_calls"] == row["model_calls"]


def test_evaluation_uses_the_selected_research_space():
    class ScopedRetriever:
        def search(self, question, limit=5, *, space_id=None):
            assert question == "How?" and space_id == 3
            return ()

    row = evaluate_case(case(), [{7}], ScopedRetriever(),
                        lambda _system, _prompt: (_ for _ in ()).throw(
                            AssertionError("model must not run without evidence")),
                        space_id=3)
    assert row["status"] == "no_evidence"
    assert row["retrieval_calls"][0]["space_id"] == 3


def test_review_template_never_overwrites_existing_judgments(tmp_path):
    row = evaluate_case(case(), [{7}], Retriever(),
                        lambda system, _prompt: ('{"sufficient": true}'
                                                if "Return only JSON" in system
                                                else "X [E1]."))
    path = tmp_path / "reviews.jsonl"
    write_review_template({"rows": [row]}, path)
    original = path.read_text(encoding="utf-8")
    assert '"answer_sha256"' in original
    assert '"cited_locations"' in original
    with pytest.raises(FileExistsError):
        write_review_template({"rows": [row]}, path)
    assert path.read_text(encoding="utf-8") == original

    report_path = tmp_path / "report.json"
    write_report({"rows": [row]}, report_path)
    with pytest.raises(FileExistsError):
        write_report({"rows": []}, report_path)


def test_combining_runs_rejects_duplicate_cases_and_mixed_labels():
    row = evaluate_case(case(), [{7}], Retriever(),
                        lambda system, _prompt: ('{"sufficient": true}'
                                                if "Return only JSON" in system
                                                else "X [E1]."))
    base = {"benchmark": "fixture", "case_file_sha256": "gold-v1",
            "model": {"provider": "fake", "model": "fake"},
            "rerank_model": "", "top_k": 5, "rows": [row]}
    assert combine([base])["summary"]["cases"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        combine([base, base])
    with pytest.raises(ValueError, match="case_file_sha256"):
        combine([base, {**base, "case_file_sha256": "gold-v2", "rows": []}])
    with pytest.raises(ValueError, match="workflow_sha256"):
        combine([base, {**base, "workflow_sha256": "new-code", "rows": []}])
