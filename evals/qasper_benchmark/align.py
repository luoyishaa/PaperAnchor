"""Exact QASPER evidence alignment in an ordered, single-paper text stream."""

import re
import unicodedata


def _normalized_positions(text: str) -> tuple[str, list[tuple[int, int]]]:
    chars, positions = [], []
    for match in re.finditer(r"\s+|[^\s]", text):
        start, end = match.span()
        value = match.group()
        if value.isspace():
            chars.append(" ")
            positions.append((start, end))
        else:
            for char in unicodedata.normalize("NFD", value):
                chars.append(char)
                positions.append((start, end))
    # Canonical combining-mark ordering must also hold across original characters.
    index = 0
    while index < len(chars):
        end = index + 1
        while end < len(chars) and unicodedata.combining(chars[end]):
            end += 1
        begin = index if unicodedata.combining(chars[index]) else index + 1
        ordered = sorted(zip(chars[begin:end], positions[begin:end]),
                         key=lambda pair: unicodedata.combining(pair[0]))
        chars[begin:end] = [pair[0] for pair in ordered]
        positions[begin:end] = [pair[1] for pair in ordered]
        index = end
    start = 1 if chars and chars[0] == " " else 0
    end = len(chars) - (1 if chars and chars[-1] == " " else 0)
    return "".join(chars[start:end]), positions[start:end]


def normalize_text(text: str) -> str:
    """Canonical Unicode and whitespace only; preserve case, numbers and units."""
    return _normalized_positions(text)[0]


def build_document(paper_id, blocks: list[dict]) -> dict:
    """Use supplied reading order; positions, when supplied, must be consecutive.

    Offsets are half-open indices in normalized_text. A block's continuous=False
    starts a new run. Bboxes always describe source blocks, not evidence glyphs.
    """
    normalized, located, runs = [], [], []
    offset, run_start = 0, 0
    for index, block in enumerate(blocks):
        if "paper_id" in block and block["paper_id"] != paper_id:
            raise ValueError("Document blocks must belong to the same paper")
        text = normalize_text(block["text"])
        if index:
            previous = blocks[index - 1]
            gap = ("position" in block and "position" in previous
                   and block["position"] != previous["position"] + 1)
            if gap or block.get("continuous") is False:
                runs.append([run_start, offset])
                run_start = offset + 1
            normalized.append(" ")
            offset += 1
        located.append({**block, "block_index": index, "start": offset,
                        "end": offset + len(text), "normalized_text": text})
        normalized.append(text)
        offset += len(text)
    if blocks:
        runs.append([run_start, offset])
    return {"paper_id": paper_id, "normalized_text": "".join(normalized),
            "blocks": located, "runs": runs}


def exact_matches(text: str, document: dict) -> list[tuple[int, int]]:
    """Find every overlapping exact occurrence without crossing discontinuities."""
    needle = normalize_text(text)
    if not needle:
        return []
    matches = []
    for start, end in document["runs"]:
        index = document["normalized_text"].find(needle, start, end)
        while index != -1:
            matches.append((index, index + len(needle)))
            index = document["normalized_text"].find(needle, index + 1, end)
    return matches


def align_evidence(text: str, document: dict) -> dict:
    normalized = normalize_text(text)
    result = {"paper_id": document["paper_id"], "text": text,
              "normalized_text": normalized}
    if "FLOAT SELECTED" in normalized.upper():
        return {**result, "status": "unsupported", "reason": "float_placeholder"}
    matches = exact_matches(text, document)
    if len(matches) != 1:
        return {**result, "status": "ambiguous" if matches else "missing",
                "match_count": len(matches)}
    start, end = matches[0]
    spans = []
    for block in document["blocks"]:
        left, right = max(start, block["start"]), min(end, block["end"])
        if left >= right:
            continue
        _, positions = _normalized_positions(block["text"])
        selected = positions[left - block["start"]:right - block["start"]]
        span = {"block_index": block["block_index"], "start": left, "end": right,
                "original_start": min(item[0] for item in selected),
                "original_end": max(item[1] for item in selected)}
        for key in ("passage_id", "id", "page_number", "block_number", "bbox"):
            if key in block:
                span[key] = block[key]
        if "bbox" in span:
            span["bbox_scope"] = "block"
        spans.append(span)
    return {**result, "status": "aligned", "start": start, "end": end,
            "spans": spans}


def align_annotations(annotations: list[dict], document: dict) -> dict:
    """Accept QASPER answers (wrapped in 'answer' or direct answer dictionaries)."""
    answers = [item.get("answer", item) for item in annotations]
    result = {"paper_id": document["paper_id"], "support_groups": [],
              "exclusions": [], "annotations": annotations}
    answerability = {bool(answer.get("unanswerable", False)) for answer in answers}
    yes_no = {answer["yes_no"] for answer in answers
              if isinstance(answer.get("yes_no"), bool)}
    if len(answerability) > 1 or len(yes_no) > 1:
        return {**result, "status": "excluded", "reason": "conflicting_annotations"}
    if answerability == {True}:
        if any(answer.get("yes_no") is not None or answer.get("extractive_spans")
               or answer.get("free_form_answer") for answer in answers):
            return {**result, "status": "excluded", "reason": "conflicting_annotations"}
        return {**result, "status": "unanswerable", "answer_type": "unanswerable"}
    types = []
    for index, answer in enumerate(answers):
        if isinstance(answer.get("yes_no"), bool):
            answer_type = "yes_no"
        elif answer.get("extractive_spans"):
            answer_type = "extractive"
        elif answer.get("free_form_answer"):
            answer_type = "free_form"
        else:
            result["exclusions"].append({"annotation_index": index, "reason": "missing_answer"})
            continue
        highlighted = [text for text in answer.get("highlighted_evidence", [])
                       if normalize_text(text)]
        texts = highlighted or answer.get("evidence", [])
        granularity = "sentence" if highlighted else "paragraph"
        group, seen = [], set()
        for text in texts:
            evidence = align_evidence(text, document)
            if evidence["normalized_text"] in seen:
                continue
            seen.add(evidence["normalized_text"])
            group.append({**evidence, "granularity": granularity,
                          "annotation_index": index})
        failures = [item for item in group if item["status"] != "aligned"]
        if failures or not group:
            result["exclusions"].append({"annotation_index": index,
                                         "reason": "unaligned_evidence" if group else "missing_evidence",
                                         "evidence": failures})
        else:
            result["support_groups"].append(group)
            types.append(answer_type)
    if not result["support_groups"]:
        return {**result, "status": "excluded", "reason": "no_complete_support_group"}
    return {**result, "status": "aligned",
            "answer_type": types[0] if len(set(types)) == 1 else "mixed"}
