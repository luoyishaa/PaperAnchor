"""Budgeted evidence assembly shared by retrieval experiments and the product.

Source intervals are used only when the supplied text has one exact match in
its linked source blocks (allowing whitespace folding). Unlocated hits remain
intact; a source link alone is never treated as returned evidence.
"""

from dataclasses import asdict, dataclass, replace
import math
import re

from .library import PassageHit, SourceSpan
from .representation import _is_heading


STRATEGIES = ("seed_first", "dedup_parent", "ranked_expansion")


@dataclass(frozen=True)
class SourceSlice:
    paper_id: int
    passage_id: int
    page_number: int
    start: int
    end: int


@dataclass(frozen=True)
class _Candidate:
    hit: PassageHit
    kind: str
    slices: tuple[SourceSlice, ...] | None
    prefix: str = ""


@dataclass(frozen=True)
class _RenderedSlice:
    source: SourceSlice
    start: int
    end: int


def _fold(text):
    """Whitespace folding with a map back to original character positions."""
    chars, positions = [], []
    for match in re.finditer(r"\S|\s+", text):
        chars.append(" " if match.group().isspace() else match.group())
        positions.append(match.start())
    return "".join(chars), positions


def _locate(hit, by_id):
    ids = list(dict.fromkeys(span.passage_id for span in hit.source_spans))
    if not ids and hit.passage_id >= 0:
        ids = [hit.passage_id]
    parents = [by_id.get((hit.paper_id, identifier)) for identifier in ids]
    if not parents or any(parent is None for parent in parents):
        return None, ""
    if len(parents) == 1 and hit.passage_id == parents[0].passage_id and hit.text == parents[0].text:
        parent = parents[0]
        return (SourceSlice(parent.paper_id, parent.passage_id, parent.page_number,
                            0, len(parent.text)),), ""
    raw, positions = "", []
    for parent in parents:
        if raw:
            raw += "\n"
            positions.append(None)
        raw += parent.text
        positions.extend((parent, index) for index in range(len(parent.text)))
    source, offsets = _fold(raw)
    options = [(hit.text, "")]
    if "\n" in hit.text:
        prefix, body = hit.text.split("\n", 1)
        # UnitStore prepends only headings recognized by build_units. Do not
        # discard an arbitrary first body line merely because its tail matches.
        if _is_heading(prefix):
            options.append((body, prefix + "\n"))
    for body, prefix in options:
        needle = _fold(body)[0].strip()
        if not needle:
            continue
        start = source.find(needle)
        if start < 0 or source.find(needle, start + 1) >= 0:
            continue
        mapped = [positions[offsets[index]] for index in range(start, start + len(needle))]
        slices = []
        for location in mapped:
            if location is None:
                continue
            parent, offset = location
            if slices and slices[-1].passage_id == parent.passage_id:
                slices[-1] = replace(slices[-1], end=offset + 1)
            else:
                slices.append(SourceSlice(parent.paper_id, parent.passage_id,
                                          parent.page_number, offset, offset + 1))
        if slices:
            return tuple(slices), prefix
    return None, ""


def _remaining(piece, covered):
    cursor = piece.start
    result = []
    for start, end in covered.get((piece.paper_id, piece.passage_id), ()):
        if end <= cursor:
            continue
        if start >= piece.end:
            break
        if start > cursor:
            result.append(replace(piece, start=cursor, end=min(start, piece.end)))
        cursor = max(cursor, end)
        if cursor >= piece.end:
            break
    if cursor < piece.end:
        result.append(replace(piece, start=cursor))
    return result


def _cover(piece, covered):
    key = (piece.paper_id, piece.passage_id)
    merged = []
    for start, end in sorted([*covered.get(key, ()), (piece.start, piece.end)]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    covered[key] = merged


def _view(candidate, by_id, covered, fallback_seen):
    if candidate.slices is None:
        key = (candidate.hit.paper_id, candidate.hit.passage_id, candidate.hit.text)
        return ("", []) if key in fallback_seen else (candidate.hit.text, [])
    pieces = [part for piece in candidate.slices for part in _remaining(piece, covered)]
    text, rendered_slices = "", []
    for piece in pieces:
        raw = by_id[piece.paper_id, piece.passage_id].text
        # Whitespace has no evidence content, and should not create empty chunks.
        value = raw[piece.start:piece.end]
        left = len(value) - len(value.lstrip())
        right = len(value.rstrip())
        if left >= right:
            continue
        piece = replace(piece, start=piece.start + left, end=piece.start + right)
        if not text:
            text = candidate.prefix
        elif not text.endswith("\n"):
            text += "\n"
        start = len(text)
        text += raw[piece.start:piece.end]
        rendered_slices.append(_RenderedSlice(piece, start, len(text)))
    return text, rendered_slices


def _source_position(candidate):
    if candidate.slices:
        first = candidate.slices[0]
        return first.paper_id, first.page_number, first.passage_id, first.start
    hit = candidate.hit
    return hit.paper_id, hit.page_number, hit.passage_id, 0


def _expansions(ranked, blocks, by_id, tokenizer):
    locations = {(block.paper_id, block.passage_id): (page, index)
                 for page, items in blocks.items() for index, block in enumerate(items)}
    seen = {(item.paper_id, span.passage_id) for item in ranked for span in item.source_spans}
    seen.update((item.paper_id, item.passage_id) for item in ranked if item.passage_id >= 0)
    parents, neighbors, parent_seen = [], [], set()

    def eligible(block):
        return len(block.text) <= 2000 and tokenizer.count(block.text) <= 512

    def candidate(block, kind, score):
        return _Candidate(replace(block, score=score), kind,
                          (SourceSlice(block.paper_id, block.passage_id, block.page_number,
                                       0, len(block.text)),))

    for item in ranked:
        ids = list(dict.fromkeys(span.passage_id for span in item.source_spans))
        if not ids and item.passage_id >= 0:
            ids = [item.passage_id]
        adjacent = []
        for identifier in ids:
            key = (item.paper_id, identifier)
            block = by_id.get(key)
            if block is not None and key not in parent_seen and eligible(block):
                parents.append(candidate(block, "parent", item.score))
                parent_seen.add(key)
            if key not in locations:
                continue
            page, index = locations[key]
            for neighbor in (index - 1, index + 1):
                if 0 <= neighbor < len(blocks[page]):
                    adjacent.append(blocks[page][neighbor])
        added = 0
        for block in adjacent:
            key = (block.paper_id, block.passage_id)
            if key in seen or not eligible(block):
                continue
            neighbors.append(candidate(block, "neighbor", item.score))
            seen.add(key)
            added += 1
            if added == 2:
                break
    return parents, neighbors


def assemble_evidence(ranked, blocks, tokenizer, *, strategy, budget=4096,
                      reranker=None, query="", protect=8):
    """Assemble top-20 seeds and bounded source expansions within ``budget``.

    ``blocks`` maps ``(paper_id, page_number)`` to source blocks in extraction
    order. Tokenizer and reranker expose ``count(text)`` and
    ``score(query, texts)`` respectively. Only direct hits may be shortened to a
    complete sentence prefix; oversized expansions are skipped, not truncated.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown evidence strategy: {strategy}")
    if budget < 1 or protect < 0:
        raise ValueError("budget must be positive and protect must be nonnegative")
    if strategy == "ranked_expansion" and reranker is None:
        raise ValueError("ranked_expansion requires a reranker")
    ranked = list(ranked)[:20]
    by_id = {}
    for page, items in blocks.items():
        for block in items:
            if page != (block.paper_id, block.page_number):
                raise ValueError("Source block is stored under another paper/page")
            key = (block.paper_id, block.passage_id)
            if key in by_id:
                raise ValueError("Duplicate source block identity")
            by_id[key] = block
    seeds = []
    for item in ranked:
        pieces, prefix = _locate(item, by_id)
        seeds.append(_Candidate(item, "seed", pieces, prefix))
    parents, neighbors = _expansions(ranked, blocks, by_id, tokenizer)
    covered, fallback_seen, chunks, events = {}, set(), [], []
    rendered = ""

    def parts(candidate):
        _, slices = _view(candidate, by_id, covered, fallback_seen)
        if len(slices) <= 1:
            return [candidate]
        # Separate source excerpts carry separate citations. In particular,
        # disjoint portions of one parent must not look like one continuous
        # quote, and the existing scorer must not mistake the first excerpt
        # for a section-title prefix.
        return [replace(candidate, slices=(part.source,),
                        prefix=candidate.prefix if index == 0 else "")
                for index, part in enumerate(slices)]

    def add(candidate):
        nonlocal rendered
        separate = parts(candidate)
        if len(separate) > 1:
            for part in separate:
                add(part)
            return
        value, slices = _view(candidate, by_id, covered, fallback_seen)
        event = {"kind": candidate.kind, "passage_id": candidate.hit.passage_id,
                 "paper_id": candidate.hit.paper_id}
        if not value.strip():
            events.append({**event, "action": "duplicate_or_empty"})
            return
        evidence_id = f"E{len(chunks) + 1}"
        page = slices[0].source.page_number if slices else candidate.hit.page_number
        header = f"[{evidence_id}] {candidate.hit.paper_title} (page {page})\n"
        prefix = rendered + ("\n\n" if rendered else "") + header
        end = len(value)
        if tokenizer.count(prefix + value) > budget:
            end = 0
            if candidate.kind == "seed":
                for boundary in re.finditer(r"[.!?](?=\s|$)", value):
                    if tokenizer.count(prefix + value[:boundary.end()]) <= budget:
                        end = boundary.end()
            if not end:
                events.append({**event, "action": "budget_skip"})
                return
        returned_slices = []
        for part in slices:
            if part.start >= end:
                break
            returned_slices.append(replace(part.source,
                                          end=part.source.start + min(part.end, end) - part.start))
        if candidate.slices is not None and not returned_slices:
            events.append({**event, "action": "budget_skip"})
            return
        spans = []
        for piece in returned_slices:
            _cover(piece, covered)
            block = by_id[piece.paper_id, piece.passage_id]
            span = SourceSpan(block.passage_id, block.page_number, block.bbox)
            if span not in spans:
                spans.append(span)
        item = candidate.hit
        if spans:
            item = replace(item, page_number=spans[0].page_number,
                           bbox=spans[0].bbox, source_spans=tuple(spans))
        elif candidate.slices is None and end == len(value):
            fallback_seen.add((item.paper_id, item.passage_id, item.text))
        chunk = asdict(item)
        chunk.update(paper_path=str(item.paper_path), evidence_id=evidence_id,
                     text=value[:end], retrieved_char_range=[0, end],
                     original_char_count=len(value), input_char_count=len(candidate.hit.text),
                     truncated=end < len(value), reference_tokens=tokenizer.count(value[:end]),
                     source_slices=[asdict(piece) for piece in returned_slices],
                     source_mapping="exact" if candidate.slices is not None else "unlocated",
                     context_kind=candidate.kind)
        chunks.append(chunk)
        rendered = prefix + value[:end]
        events.append({**event, "action": "truncated" if end < len(value) else "added",
                       "evidence_id": evidence_id})

    if strategy in ("seed_first", "dedup_parent"):
        for candidate in seeds:
            add(candidate)
        if strategy == "dedup_parent":
            for candidate in parents:
                add(candidate)
        for candidate in neighbors:
            add(candidate)
    else:
        for candidate in seeds[:protect]:
            add(candidate)
        offered = []
        for candidate in [*seeds[protect:], *parents, *neighbors]:
            for part in parts(candidate):
                text, _ = _view(part, by_id, covered, fallback_seen)
                if text.strip():
                    offered.append((part, text))
        values = list(reranker.score(query, [text for _, text in offered])) if offered else []
        if len(values) != len(offered) or not all(math.isfinite(value) for value in values):
            raise ValueError("Reranker returned invalid expansion scores")
        remaining = [(candidate, float(value), index)
                     for index, ((candidate, _), value) in enumerate(zip(offered, values))]
        while remaining:
            def priority(entry):
                candidate, value, index = entry
                preview, count = rendered, len(chunks)
                for part in parts(candidate):
                    text, slices = _view(part, by_id, covered, fallback_seen)
                    if not text:
                        continue
                    count += 1
                    page = slices[0].source.page_number if slices else part.hit.page_number
                    header = f"[E{count}] {part.hit.paper_title} (page {page})\n"
                    preview += ("\n\n" if preview else "") + header + text
                cost = tokenizer.count(preview) - tokenizer.count(rendered)
                return cost, _source_position(candidate), index
            highest = max(entry[1] for entry in remaining)
            tied = [index for index, entry in enumerate(remaining) if entry[1] == highest]
            selected = (tied[0] if len(tied) == 1 else
                        min(tied, key=lambda index: priority(remaining[index])))
            add(remaining.pop(selected)[0])
    return {"chunks": chunks, "text": rendered, "reference_tokens": tokenizer.count(rendered),
            "budget": budget, "truncation_rule": "seed_sentence_prefix_expansion_skip_continue",
            "trace": {"strategy": strategy, "protect": protect, "seed_count": len(seeds),
                      "mapped_seed_count": sum(seed.slices is not None for seed in seeds),
                      "parent_candidate_ids": [item.hit.passage_id for item in parents],
                      "neighbor_candidate_ids": [item.hit.passage_id for item in neighbors],
                      "events": events}}
