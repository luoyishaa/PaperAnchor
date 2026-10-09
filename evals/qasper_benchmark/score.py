"""Score actual returned text against exact aligned evidence, without models."""

from .align import exact_matches, normalize_text


def chunk_intervals(chunk: dict, document: dict) -> list[tuple[int, int]]:
    if chunk.get("paper_id") != document["paper_id"]:
        return []
    source_ids = {span["passage_id"] for span in chunk.get("source_spans", [])
                  if "passage_id" in span}
    if not source_ids and chunk.get("passage_id") is not None:
        source_ids = {chunk["passage_id"]}
    known_blocks = [block for block in document["blocks"]
                    if block.get("passage_id", block.get("id")) in source_ids]
    texts = [chunk["text"]]
    # Product reading units prepend one section-title line to the source text.
    if "\n" in chunk["text"] and "start" not in chunk and "end" not in chunk:
        texts.append(chunk["text"].split("\n", 1)[1])
    for text in texts:
        matches = exact_matches(text, document)
        if "start" in chunk and "end" in chunk:
            matches = [pair for pair in matches if pair == (chunk["start"], chunk["end"])]
        else:
            source = document["normalized_text"]
            matches = [(start, end) for start, end in matches
                       if (start == 0 or source[start - 1].isspace())
                       and (end == len(source) or source[end].isspace())]
        if known_blocks:
            matches = [(start, end) for start, end in matches
                       if all(block in known_blocks for block in document["blocks"]
                              if block["start"] < end and block["end"] > start)]
        if matches:
            return matches if len(matches) == 1 else []
    if not known_blocks or "start" in chunk or "end" in chunk:
        return []
    # Units join source pieces with newlines, possibly in a different block order.
    # Match complete returned pieces, never isolated characters or source IDs alone.
    located = {}
    for line_index, line in enumerate(chunk["text"].splitlines()):
        text = normalize_text(line)
        if not text:
            continue
        matches = []
        for block in known_blocks:
            source = block["normalized_text"]
            start = source.find(text)
            while start != -1:
                end = start + len(text)
                if ((start == 0 or source[start - 1].isspace())
                        and (end == len(source) or source[end].isspace())):
                    matches.append((block["block_index"], block["start"] + start,
                                    block["start"] + end))
                start = source.find(text, start + 1)
        if len(matches) == 1:
            block_index, start, end = matches[0]
            pieces = located.setdefault(block_index, [])
            if not any((left, right) == (start, end) for _, left, right in pieces):
                pieces.append((line_index, start, end))
    intervals = []
    for pieces in located.values():
        # Do not reconstruct one source block across omitted or reordered words.
        if any(line != previous_line + 1 or start < previous_start or end < previous_end
               for (previous_line, previous_start, previous_end), (line, start, end)
               in zip(pieces, pieces[1:])):
            continue
        intervals.extend((start, end) for _, start, end in pieces)
    return sorted(set(intervals))


def _covered(evidence: dict, intervals: list[tuple[int, int]], document: dict) -> bool:
    if evidence.get("paper_id") != document["paper_id"]:
        return False
    cursor, end = evidence["start"], evidence["end"]
    text = document["normalized_text"]
    for left, right in sorted(set(intervals)):
        if right <= cursor:
            continue
        # Normalized separators between independently returned chunks carry no words.
        if left > cursor and text[cursor:min(left, end)].strip():
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return cursor >= end


def covered(evidence: dict, chunks: list[dict], document: dict) -> bool:
    intervals = [pair for chunk in chunks for pair in chunk_intervals(chunk, document)]
    return _covered(evidence, intervals, document)


def _coverage(groups: list[list[dict]], intervals: list[tuple[int, int]], document: dict) -> dict:
    if not groups or any(not group for group in groups):
        raise ValueError("Scoring requires nonempty aligned support groups")
    if any(item.get("status") != "aligned" or item.get("paper_id") != document["paper_id"]
           for group in groups for item in group):
        raise ValueError("Gold evidence must be aligned to the scored paper")
    counts = [sum(_covered(item, intervals, document) for item in group) for group in groups]
    best = max(range(len(groups)), key=lambda index: counts[index] / len(groups[index]))
    return {"evidence_recall": counts[best] / len(groups[best]),
            "complete_support": int(counts[best] == len(groups[best])),
            "support_group_index": best, "covered_evidence": counts[best],
            "gold_evidence": len(groups[best]),
            "granularities": sorted({item["granularity"] for item in groups[best]
                                      if "granularity" in item})}


def score_context(support_groups: list[list[dict]], chunks: list[dict], document: dict) -> dict:
    return _coverage(support_groups,
                     [pair for chunk in chunks for pair in chunk_intervals(chunk, document)], document)


def score_retrieval(support_groups: list[list[dict]], chunks: list[dict], document: dict,
                    ks=(1, 5, 10, 20)) -> dict:
    intervals = [chunk_intervals(chunk, document) for chunk in chunks]
    at_k = {}
    for k in ks:
        if k < 1:
            raise ValueError("Retrieval depths must be positive")
        at_k[str(k)] = {**_coverage(support_groups,
                                    [pair for hit in intervals[:k] for pair in hit], document),
                        "paper_hit": int(any(chunk.get("paper_id") == document["paper_id"]
                                             for chunk in chunks[:k]))}
    first_rank = next((rank for rank, hit in enumerate(intervals[:20], 1)
                       if any(_covered(item, hit, document)
                              for group in support_groups for item in group)), None)
    return {"at_k": at_k, "mrr_at_20": 1 / first_rank if first_rank else 0.0,
            "first_support_rank": first_rank}


def score_citations(support_groups: list[list[dict]], context: list[dict], document: dict,
                    cited_ids: list[str], *, failed: bool = False) -> dict:
    available = {chunk["evidence_id"]: chunk for chunk in context}
    if len(available) != len(context):
        raise ValueError("Context evidence IDs must be unique")
    citations = list(dict.fromkeys(cited_ids))
    valid = [identifier for identifier in citations if identifier in available]
    chunks = [available[identifier] for identifier in valid] if not failed else []
    matched = sum(any(covered(item, [chunk], document)
                      for group in support_groups for item in group) for chunk in chunks)
    coverage = score_context(support_groups, chunks, document)
    return {"citation_count": len(citations), "valid_citation_count": len(valid),
            "illegal_citation_count": len(citations) - len(valid),
            "no_citation": not citations, "failed": failed,
            "citation_validity": len(valid) / len(citations) if citations else None,
            "gold_citation_match_precision": matched / len(citations) if citations else None,
            "gold_citation_evidence_recall": coverage["evidence_recall"],
            "gold_citation_complete_support": coverage["complete_support"],
            "support_group_index": coverage["support_group_index"]}
