import pytest

from evals.qasper_benchmark.align import build_document, align_annotations
from evals.qasper_benchmark.report import make_report, paired_comparison, summarize


def row(case, paper, score):
    return {"case_id": case, "paper_id": paper, "metrics": {"recall": score},
            "status": "ok", "seconds": .1}


def test_question_and_paper_macro_are_distinct():
    result = summarize([row("a", 1, 1), row("b", 1, 1), row("c", 2, 0)])
    assert result["question_macro"]["recall"] == 2 / 3
    assert result["paper_macro"]["recall"] == .5


def test_paired_cluster_sampling_is_repeatable():
    left = [row("a", 1, 0), row("b", 1, .5), row("c", 2, 1)]
    right = [row("a", 1, 1), row("b", 1, .5), row("c", 2, 0)]
    first = paired_comparison(left, right, metric="recall", samples=100)
    assert first == paired_comparison(left, right, metric="recall", samples=100)
    assert (first["wins"], first["losses"], first["ties"]) == (1, 1, 1)
    assert first["paper_clusters"] == 2


def test_report_missing_retrieval_is_zero_and_not_formal():
    document = build_document(1, [{"passage_id": 1, "text": "The result is 42."}])
    alignment = align_annotations([{"answer": {
        "unanswerable": False, "yes_no": None, "extractive_spans": ["42"],
        "highlighted_evidence": ["The result is 42."],
    }}], document)
    gold = {"fingerprint": "frozen-input", "corpus_fingerprint": "corpus",
            "public_cases_fingerprint": "public", "status": "frozen", "documents": {"1": document},
            "cases": [{"case_id": "q", "split": "dev", "index_paper_id": 1,
                       "alignment": alignment}]}
    run = {"status": "incomplete", "cases_fingerprint": "frozen-input",
           "corpus_fingerprint": "corpus", "public_cases_fingerprint": "public", "records": [{
        "case_id": "q", "scope": "full_library", "representation": "blocks",
        "method": "dense", "status": "failed",
    }]}
    report = make_report(run, gold)
    assert not report["formal_scores"]
    assert report["configurations"]["blocks/dense"]["failures"] == 1
    assert report["questions"][0]["metrics"]["EvidenceRecall@20"] == 0
    assert report["expected_records"] == 3
    assert report["not_attempted_records"] == 2
    assert report["configurations"]["blocks/bm25"]["not_attempted"] == 1
    run["status"] = "complete"
    run["records"][0]["status"] = "ok"
    assert not make_report(run, gold)["formal_scores"]
    run["records"] = [{**run["records"][0], "method": method}
                      for method in ("bm25", "dense", "hybrid")]
    assert make_report(run, gold)["formal_scores"]
    run["config"] = {"representations": ["blocks"], "methods": ["bm25", "rerank"],
                     "rerank_model": "fixture"}
    run["records"] = [{**run["records"][0], "method": method} for method in ("bm25", "rerank")]
    selected = make_report(run, gold)
    assert selected["formal_scores"]
    assert selected["expected_records"] == 2
    assert set(selected["configurations"]) == {"blocks/bm25", "blocks/rerank"}
    run["cases_fingerprint"] = "different"
    with pytest.raises(ValueError, match="fingerprints"):
        make_report(run, gold)


def test_all_split_scores_both_original_splits_and_keeps_auxiliary_separate():
    document = build_document(1, [{"passage_id": 1, "text": "The result is 42."}])
    aligned = align_annotations([{"answer": {
        "unanswerable": False, "yes_no": None, "extractive_spans": ["42"],
        "highlighted_evidence": ["The result is 42."],
    }}], document)
    gold = {"fingerprint": "f", "corpus_fingerprint": "c", "public_cases_fingerprint": "p",
            "status": "frozen", "documents": {"1": document}, "cases": [
                {"case_id": "a", "split": "dev", "index_paper_id": 1, "alignment": aligned},
                {"case_id": "b", "split": "test", "index_paper_id": 1, "alignment": aligned},
                {"case_id": "c", "split": "test", "index_paper_id": 1,
                 "alignment": {"status": "unanswerable"}},
            ]}
    records = [{"case_id": c["case_id"], "split": c["split"], "representation": "units",
                "method": "new", "scope": "single_paper_auxiliary" if c["case_id"] == "c" else "full_library",
                "status": "ok", "context": {"chunks": []}, "ranked": []} for c in gold["cases"]]
    run = {"status": "complete", "cases_fingerprint": "f", "corpus_fingerprint": "c",
           "public_cases_fingerprint": "p", "config": {"split": "all", "methods": ["new"]}, "records": records}
    result = make_report(run, gold)
    assert result["formal_scores"]
    assert result["expected_records"] == 3
    assert result["expected_questions"] == 2
    assert len(result["questions"]) == 2
    assert len(result["auxiliary"]) == 1
    run["records"] = records[:-1]
    assert not make_report(run, gold)["formal_scores"]
