from evals.litqa_benchmark import article_blocks, score_context


def test_xml_body_and_tables_are_indexed_once_without_gold_or_references():
    raw = b'''<article><front><article-meta><title-group><article-title>A <italic>real</italic> title</article-title></title-group><abstract><p>Abstract.</p></abstract></article-meta></front><body><sec><title>Results</title><p>Full evidence.</p><table-wrap><caption><p>Caption.</p></caption><table><tr><td>42</td></tr></table></table-wrap></sec></body><back><ref-list><ref>Not body</ref></ref-list></back></article>'''
    title, blocks = article_blocks(raw)
    assert title == "A real title"
    texts = [text for _, text in blocks]
    assert texts == ["Abstract.", "Results", "Full evidence.", "Caption.42"]
    assert blocks[-1][0] == 2


def test_key_passage_must_be_from_annotated_source():
    context = {"chunks": [{"paper_id": 2, "text": "Mice received 5 mg of compound X."}]}
    result = score_context(context, {1}, "Mice received 5 mg of compound X.")
    assert not result["source_hit"]
    assert not result["key_passage_exact"]
    assert result["key_passage_character_coverage"] == 0


def test_multiple_sources_and_reordered_evidence_slices():
    first = "The experiment used twenty independently prepared specimens."
    second = "All specimens were measured after twenty four hours of treatment."
    context = {"chunks": [{"paper_id": 1, "text": second}, {"paper_id": 1, "text": first}]}
    result = score_context(context, {1, 2}, first + " " + second)
    assert result["source_hit"]
    assert not result["all_sources"]
    assert result["source_recall"] == 0.5
    assert not result["key_passage_exact"]
    assert result["key_passage_character_coverage"] == 1
