"""Offline exact alignment, retrieval coverage and citation scoring fixtures."""

import pytest

from evals.qasper_benchmark.align import (
    align_annotations, align_evidence, build_document, normalize_text,
)
from evals.qasper_benchmark.score import (
    chunk_intervals, covered, score_citations, score_context, score_retrieval,
)


def document(*texts):
    return build_document(7, [{"text": text, "passage_id": index + 10,
                                "page_number": index + 1, "position": index,
                                "bbox": [1, 2, 3, 4]}
                               for index, text in enumerate(texts)])


def chunk(text, **kwargs):
    return {"paper_id": 7, "text": text, **kwargs}


def gold(doc, *texts):
    return [[align_evidence(text, doc) for text in texts]]


def test_unicode_whitespace_and_original_positions():
    doc = document("  Caf\u00e9\u00a0\n is", "not  5 mg. ")
    item = align_evidence("Cafe\u0301 is not 5 mg.", doc)
    assert item["status"] == "aligned"
    assert item["paper_id"] == 7
    assert item["text"] == "Cafe\u0301 is not 5 mg."
    assert item["normalized_text"] == "Cafe\u0301 is not 5 mg."
    assert [(span["page_number"], span["original_start"], span["original_end"])
            for span in item["spans"]] == [(1, 2, 11), (2, 0, 10)]
    assert all(span["bbox_scope"] == "block" for span in item["spans"])
    assert item["spans"][0]["passage_id"] == 10


def test_canonical_combining_order():
    assert normalize_text("a\u0301\u0327") == normalize_text("a\u0327\u0301")
    doc = document("a\u0301\u0327")
    assert align_evidence("a\u0327\u0301", doc)["spans"][0]["original_end"] == 3


def test_interior_offsets_without_invented_coordinates():
    doc = build_document("qasper-id", [{"text": "Before.  Gold\n sentence. After."}])
    item = align_evidence("Gold sentence.", doc)
    span = item["spans"][0]
    assert doc["blocks"][0]["text"][span["original_start"]:span["original_end"]] == "Gold\n sentence."
    assert item["paper_id"] == "qasper-id"
    assert "bbox" not in span
    assert "bbox_scope" not in span


@pytest.mark.parametrize("changed", ["Dose is 5 mg.", "Dose is not 6 mg.",
                                     "Dose is not 5 kg.", "dose is not 5 mg."])
def test_no_fuzzy_case_negation_number_or_unit_equivalence(changed):
    assert align_evidence(changed, document("Dose is not 5 mg."))["status"] == "missing"


def test_compatibility_unicode_does_not_change_units():
    assert normalize_text("m\u00b2") != normalize_text("m2")


@pytest.mark.parametrize("text,status", [("", "missing"), ("absent", "missing"),
                                         ("same", "ambiguous"),
                                         ("FLOAT SELECTED: Table 1", "unsupported")])
def test_alignment_rejections(text, status):
    assert align_evidence(text, document("same", "same", "FLOAT SELECTED: Table 1"))["status"] == status


def test_overlapping_matches_are_ambiguous():
    assert align_evidence("aaa", document("aaaa"))["match_count"] == 2


@pytest.mark.parametrize("boundary", [{"position": 3}, {"continuous": False}])
def test_evidence_cannot_cross_discontinuous_blocks(boundary):
    doc = build_document(7, [{"text": "first", "position": 0},
                             {"text": "second", "position": 1, **boundary}])
    assert align_evidence("first second", doc)["status"] == "missing"
    assert align_evidence("second", doc)["status"] == "aligned"


def test_document_rejects_mixed_papers():
    with pytest.raises(ValueError, match="same paper"):
        build_document(7, [{"text": "other", "paper_id": 8}])


def test_highlights_preferred_false_yes_no_and_alternative_groups():
    doc = document("No benefit.", "Independent support.")
    annotations = [{"answer": {"yes_no": False, "highlighted_evidence": ["No benefit."],
                                 "evidence": ["missing paragraph"]}},
                   {"answer": {"yes_no": False, "evidence": ["Independent support."]}}]
    aligned = align_annotations(annotations, doc)
    assert aligned["status"] == "aligned"
    assert aligned["answer_type"] == "yes_no"
    assert len(aligned["support_groups"]) == 2
    assert aligned["support_groups"][0][0]["granularity"] == "sentence"
    assert aligned["support_groups"][1][0]["granularity"] == "paragraph"
    result = score_context(aligned["support_groups"], [chunk("Independent support.")], doc)
    assert result["evidence_recall"] == result["complete_support"] == 1
    assert result["support_group_index"] == 1


@pytest.mark.parametrize("answers", [
    [{"yes_no": False}, {"yes_no": True}],
    [{"unanswerable": True}, {"yes_no": False}],
    [{"unanswerable": True, "yes_no": False}],
])
def test_conflicting_annotations_excluded(answers):
    result = align_annotations(answers, document("No benefit."))
    assert result["status"] == "excluded"
    assert result["reason"] == "conflicting_annotations"


def test_unanswerable_and_missing_annotations():
    doc = document("Text.")
    assert align_annotations([{"unanswerable": True}], doc)["status"] == "unanswerable"
    assert align_annotations([], doc)["status"] == "excluded"


def test_incomplete_groups_not_promoted_to_partial_gold():
    doc = document("Present.")
    annotations = [{"free_form_answer": "answer", "evidence": ["Present.", "Absent."]},
                   {"extractive_spans": ["Present"], "evidence": ["Present.", "Present."]}]
    aligned = align_annotations(annotations, doc)
    assert len(aligned["support_groups"]) == 1
    assert len(aligned["support_groups"][0]) == 1
    assert aligned["exclusions"][0]["annotation_index"] == 0
    assert aligned["exclusions"][0]["evidence"][0]["status"] == "missing"


def test_unaligned_highlights_do_not_fallback_to_paragraphs():
    result = align_annotations([{"yes_no": True, "highlighted_evidence": ["Missing."],
                                 "evidence": ["Present."]}], document("Present."))
    assert result["status"] == "excluded"


def test_same_source_is_not_evidence_and_other_paper_does_not_count():
    doc = document("Background. The answer is 42.")
    item = gold(doc, "The answer is 42.")[0][0]
    assert not covered(item, [chunk("Background.", passage_id=10)], doc)
    assert not covered(item, [chunk("The answer is 42.", paper_id=8, passage_id=10)], doc)
    assert not covered(item, [chunk("The answer is 41.", passage_id=10)], doc)


def test_partial_chunk_union_duplicates_and_single_hit_mrr():
    doc = document("The answer is 42.")
    groups = gold(doc, "The answer is 42.")
    chunks = [chunk("The answer"), chunk("The answer"), chunk("is 42.")]
    result = score_retrieval(groups, chunks, doc, ks=(1, 2, 3))
    assert result["at_k"]["2"]["evidence_recall"] == 0
    assert result["at_k"]["3"]["complete_support"] == 1
    assert result["mrr_at_20"] == 0
    assert result["first_support_rank"] is None
    chunks.append(chunk("The answer is 42."))
    assert score_retrieval(groups, chunks, doc)["mrr_at_20"] == 0.25


def test_overlap_union_missing_word_and_wrong_positions():
    doc = document("The answer is 42.")
    item = gold(doc, "The answer is 42.")[0][0]
    assert covered(item, [chunk("The answer is"), chunk("is 42.")], doc)
    assert not covered(item, [chunk("The answer"), chunk("42.")], doc)
    assert not covered(item, [chunk("The answer is 42.", start=10, end=26)], doc)


def test_repeated_actual_text_requires_location_disambiguation():
    doc = document("same ending", "same other")
    assert chunk_intervals(chunk("same"), doc) == []
    assert chunk_intervals(chunk("same", passage_id=10), doc) == [(0, 4)]
    assert chunk_intervals(chunk("same", source_spans=[{"passage_id": 11}]), doc) == [(12, 16)]
    assert chunk_intervals(chunk("same", start=12, end=16), doc) == [(12, 16)]


def test_cross_block_gold_and_chunk_union():
    doc = document("Answer starts", "and finishes.")
    groups = gold(doc, "Answer starts and finishes.")
    assert score_context(groups, [chunk("Answer starts"), chunk("and finishes.")], doc)["complete_support"] == 1


def test_unit_section_prefix_preserves_actual_body_coverage():
    doc = document("The answer is 42.")
    item = gold(doc, "The answer is 42.")[0][0]
    assert covered(item, [chunk("Results\nThe answer is 42.", source_spans=[{"passage_id": 10}])], doc)
    assert not covered(item, [chunk("The answer\nnot\nis 42.")], doc)


def test_product_unit_spatial_order_differs_from_sql_block_order(tmp_path):
    from dataclasses import asdict

    import pymupdf

    from paperanchor.library import PaperLibrary
    from paperanchor.pdf import ExtractedPassage
    from paperanchor.representation import build_units
    from paperanchor.unit_index import UnitStore

    pdf_path = tmp_path / "paper.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        for text, y in [("Introduction", 70), ("Alpha is not 5 mg.", 110),
                        ("Gamma is 30 mg.", 190), ("Beta is 20 mg.", 150)]:
            page.insert_text((72, y), text)
        pdf.save(pdf_path)
    library = PaperLibrary(tmp_path / "library.sqlite3")
    paper_id = library.index_pdf(pdf_path).paper_id
    with library.database.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM passages WHERE paper_id=? ORDER BY page_number,block_number,id",
            (paper_id,),
        ).fetchall()
    blocks = [{"text": row["text"], "passage_id": row["id"],
               "page_number": row["page_number"], "block_number": row["block_number"],
               "bbox": tuple(row[key] for key in ("x0", "y0", "x1", "y1"))} for row in rows]
    assert [block["text"] for block in blocks][1:] == [
        "Alpha is not 5 mg.", "Gamma is 30 mg.", "Beta is 20 mg."]
    units = build_units([ExtractedPassage(block["page_number"], block["block_number"],
                                         block["text"], block["bbox"]) for block in blocks])
    assert units[0].source_indexes == (1, 3, 2)
    hit = asdict(UnitStore(library).search("Alpha")[0])
    assert hit["text"] == "Introduction\n" + units[0].text
    doc = build_document(paper_id, blocks)
    groups = gold(doc, "Alpha is not 5 mg.", "Beta is 20 mg.", "Gamma is 30 mg.")
    result = score_retrieval(groups, [hit], doc)
    assert result["at_k"]["1"]["evidence_recall"] == 1
    assert result["at_k"]["1"]["complete_support"] == 1
    assert result["mrr_at_20"] == 1
    cited = score_citations(groups, [{**hit, "evidence_id": "E1"}], doc, ["E1"])
    assert cited["gold_citation_complete_support"] == 1
    assert cited["gold_citation_match_precision"] == 1
    partial = {**hit, "text": "Introduction\nAlpha is not 5 mg.\nBeta is"}
    assert score_context(groups, [partial], doc)["evidence_recall"] == pytest.approx(1 / 3)
    assert covered(groups[0][1], [partial, {**hit, "text": "20 mg."}], doc)


@pytest.mark.parametrize("alpha", ["Alpha is 5 mg.", "Alpha is not 6 mg.",
                                   "Alpha is not 5 kg.", "Alpha is\nnot changed\n5 mg."])
def test_reordered_unit_segments_do_not_repair_changed_evidence(alpha):
    doc = document("Alpha is not 5 mg.", "Gamma is 30 mg.", "Beta is 20 mg.")
    returned = chunk(f"Introduction\n{alpha}\nBeta is 20 mg.\nGamma is 30 mg.",
                     source_spans=[{"passage_id": identifier} for identifier in (10, 12, 11)])
    assert not covered(gold(doc, "Alpha is not 5 mg.")[0][0], [returned], doc)
    assert covered(gold(doc, "Beta is 20 mg.")[0][0], [returned], doc)


def test_segment_mapping_requires_known_unique_source_and_complete_tokens():
    doc = document("Alpha is 15 mg.", "same ending", "same other")
    links = [{"passage_id": identifier} for identifier in (10, 11, 12)]
    assert chunk_intervals(chunk("Unmatched heading\nsame", source_spans=links), doc) == []
    assert chunk_intervals(chunk("Unmatched heading\n5 mg.\nsame", source_spans=links), doc) == []
    assert chunk_intervals(chunk("Unmatched heading\nAlpha is 15 mg.\nother",
                                 source_spans=[{"passage_id": 999}]), doc) == []
    assert chunk_intervals(chunk("Unmatched heading\nAlpha is 15 mg.\nother",
                                 source_spans=links, paper_id=8), doc) == []
    assert chunk_intervals(chunk("Unmatched heading\nAlpha is 15 mg.\nother",
                                 source_spans=links, start=100, end=200), doc) == []


def test_repeated_unit_segments_do_not_add_or_remove_coverage():
    doc = document("Alpha evidence.", "Gamma evidence.", "Beta evidence.")
    returned = chunk("Heading\nAlpha evidence.\nBeta evidence.\nAlpha evidence.",
                     source_spans=[{"passage_id": identifier} for identifier in (10, 12, 11)])
    groups = gold(doc, "Alpha evidence.", "Gamma evidence.", "Beta evidence.")
    result = score_context(groups, [returned, returned], doc)
    assert result["evidence_recall"] == pytest.approx(2 / 3)
    assert result["complete_support"] == 0


@pytest.mark.parametrize("body", ["The answer\nnot\nis 42.", "is 42.\nThe answer"])
def test_segment_union_does_not_reconstruct_reordered_or_interrupted_block(body):
    doc = document("The answer is 42.", "Another block.")
    returned = chunk(f"Heading\n{body}\nAnother block.",
                     source_spans=[{"passage_id": 10}, {"passage_id": 11}])
    assert not covered(gold(doc, "The answer is 42.")[0][0], [returned], doc)
    assert covered(gold(doc, "Another block.")[0][0], [returned], doc)


def test_empty_retrieval_keeps_denominator():
    doc = document("Gold.")
    result = score_retrieval(gold(doc, "Gold."), [], doc)
    assert result["at_k"]["20"]["evidence_recall"] == 0
    assert result["at_k"]["20"]["paper_hit"] == 0
    assert result["mrr_at_20"] == 0


def test_best_alternative_fraction_and_mrr_cutoff():
    doc = document("First.", "Second.", "Third.")
    groups = [gold(doc, "First.", "Second.")[0], gold(doc, "Third.")[0]]
    result = score_context(groups, [chunk("First.")], doc)
    assert result["evidence_recall"] == 0.5
    assert result["complete_support"] == 0
    assert result["gold_evidence"] == 2
    ranked = [chunk("unrelated")] * 20 + [chunk("Third.")]
    result = score_retrieval(groups, ranked, doc, ks=(20, 21))
    assert result["mrr_at_20"] == 0
    assert result["at_k"]["20"]["paper_hit"] == 1
    assert result["at_k"]["20"]["evidence_recall"] == 0
    assert result["at_k"]["21"]["evidence_recall"] == 1


def test_citation_dedup_illegal_ids_and_actual_truncation():
    doc = document("The answer is 42.")
    groups = gold(doc, "The answer is 42.")
    context = [chunk("The answer", evidence_id="E1", passage_id=10),
               chunk("The answer is 42.", evidence_id="E2")]
    result = score_citations(groups, context, doc, ["E1", "E2", "E2", "E99"])
    assert result["citation_count"] == 3
    assert result["illegal_citation_count"] == 1
    assert result["citation_validity"] == pytest.approx(2 / 3)
    assert result["gold_citation_match_precision"] == pytest.approx(1 / 3)
    assert result["gold_citation_complete_support"] == 1
    assert score_citations(groups, context, doc, ["E1"])["gold_citation_evidence_recall"] == 0


def test_citation_union_can_cover_without_individual_precision_credit():
    doc = document("The answer is 42.")
    context = [chunk("The answer", evidence_id="E1"), chunk("is 42.", evidence_id="E2")]
    result = score_citations(gold(doc, "The answer is 42."), context, doc, ["E1", "E2"])
    assert result["gold_citation_match_precision"] == 0
    assert result["gold_citation_evidence_recall"] == 1


def test_no_citations_and_failed_outputs():
    doc = document("Gold.")
    groups = gold(doc, "Gold.")
    context = [chunk("Gold.", evidence_id="E1")]
    result = score_citations(groups, context, doc, [])
    assert result["gold_citation_match_precision"] is None
    assert result["citation_validity"] is None
    assert result["gold_citation_complete_support"] == 0
    assert result["no_citation"]
    failed = score_citations(groups, context, doc, ["E1"], failed=True)
    assert failed["failed"]
    assert failed["gold_citation_match_precision"] == 0
    assert failed["gold_citation_evidence_recall"] == 0


def test_invalid_gold_and_duplicate_context_ids_rejected():
    doc = document("Gold.")
    with pytest.raises(ValueError, match="nonempty"):
        score_context([[]], [], doc)
    with pytest.raises(ValueError, match="aligned"):
        score_context([[align_evidence("missing", doc)]], [], doc)
    with pytest.raises(ValueError, match="unique"):
        score_citations(gold(doc, "Gold."), [chunk("Gold.", evidence_id="E1")] * 2, doc, ["E1"])
