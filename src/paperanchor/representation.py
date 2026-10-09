"""Searchable reading units with explicit links back to immutable PDF blocks.

A source block is the smallest located extraction unit. A retrieval unit may
combine adjacent blocks for meaning, but never invents a PDF rectangle.
"""

from dataclasses import dataclass
import re
from typing import Callable, Sequence

from .pdf import ExtractedPassage


REPRESENTATION_VERSION = 4
MAX_UNIT_TOKENS = 128


@dataclass(frozen=True)
class RetrievalUnit:
    text: str
    section: str
    source_indexes: tuple[int, ...]


_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?(?:abstract|introduction|background|"
    r"related work|method(?:s|ology)?|experiments?|results?|discussion|"
    r"conclusion(?:s)?|limitations?|references)\b",
    re.IGNORECASE,
)
_CAPTION = re.compile(r"^(?:figure|fig\.|table)\s*\d+\s*[:.]", re.IGNORECASE)


def _is_heading(text: str) -> bool:
    return len(text) <= 130 and bool(_HEADING.match(text))


def _split_sentences(text: str) -> list[str]:
    # Keep punctuation with the preceding sentence; long equations and URLs
    # are handled by the final word-level fallback below.
    sentences = re.split(r"(?<=[.!?。！？])\s+(?=[A-Z\d(\[\u4e00-\u9fff])", text)
    return [piece.strip() for piece in sentences if piece.strip()]


def _pieces(text: str, fits: Callable[[str], bool]) -> list[str]:
    if fits(text):
        return [text]
    result: list[str] = []
    current = ""
    for sentence in _split_sentences(text):
        if current and not fits(f"{current} {sentence}"):
            result.append(current)
            current = ""
        if fits(sentence):
            current = sentence if not current else f"{current} {sentence}"
            continue
        for word in sentence.split():
            trial = word if not current else f"{current} {word}"
            if current and not fits(trial):
                result.append(current)
                current = ""
            # An unbroken word can exceed the tokenizer limit. Split it only
            # then, rather than silently losing its tail at embedding time.
            while word and not fits(word):
                cut = max(1, len(word) // 2)
                while cut > 1 and not fits(word[:cut]):
                    cut //= 2
                result.append(word[:cut])
                word = word[cut:]
            if word:
                current = word if not current else f"{current} {word}"
    if current:
        result.append(current)
    return result


def _reading_order(passages: Sequence[ExtractedPassage]) -> list[int]:
    """Keep columns together instead of interleaving rows across columns."""
    if not passages:
        return []
    page_width = max(item.bbox[2] for item in passages)
    centre = page_width / 2
    left = [i for i, item in enumerate(passages) if item.bbox[2] < centre + 5]
    right = [i for i, item in enumerate(passages) if item.bbox[0] > centre - 5]
    if len(left) < 3 or len(right) < 3:
        return sorted(range(len(passages)), key=lambda i: (passages[i].bbox[1], passages[i].bbox[0]))
    both = set(left) & set(right)
    left = [i for i in left if i not in both]
    right = [i for i in right if i not in both]
    spanning = [i for i in range(len(passages)) if i not in set(left) | set(right)]
    top = min((passages[i].bbox[1] for i in left + right), default=0)
    head = [i for i in spanning if passages[i].bbox[1] < top]
    tail = [i for i in spanning if i not in head]
    key = lambda i: (passages[i].bbox[1], passages[i].bbox[0])
    return sorted(head, key=key) + sorted(left, key=key) + sorted(right, key=key) + sorted(tail, key=key)


def build_units(passages: Sequence[ExtractedPassage], *,
                token_count: Callable[[str], int] | None = None,
                max_tokens: int = MAX_UNIT_TOKENS) -> tuple[RetrievalUnit, ...]:
    """Pack neighboring paragraphs within a page, preserving every source link.

    The lightweight initial pass uses a conservative character budget. The
    background embedding pass can rebuild with the model's actual tokenizer.
    """
    if max_tokens < 32:
        raise ValueError("max_tokens must be at least 32")
    fits = (lambda value: token_count(value) <= max_tokens) if token_count else (
        lambda value: len(value) <= max_tokens * 2)
    by_page: dict[int, list[tuple[int, ExtractedPassage]]] = {}
    for index, passage in enumerate(passages):
        by_page.setdefault(passage.page_number, []).append((index, passage))

    units: list[RetrievalUnit] = []
    for page in sorted(by_page):
        # A heading from the previous page may no longer apply: many papers
        # place appendices after References without another recognized heading.
        section = ""
        entries = by_page[page]
        order = _reading_order([item for _, item in entries])
        buffer = ""
        source_indexes: list[int] = []

        def flush() -> None:
            nonlocal buffer, source_indexes
            if buffer:
                units.append(RetrievalUnit(buffer, section, tuple(source_indexes)))
            buffer, source_indexes = "", []

        for ordered_index in order:
            source_index, block = entries[ordered_index]
            value = block.text.strip()
            if not value:
                continue
            if _is_heading(value):
                flush()
                section = value
                continue
            if _CAPTION.match(value):
                flush()
            for piece in _pieces(value, lambda text: fits(f"{section}\n{text}")):
                trial = piece if not buffer else f"{buffer}\n{piece}"
                if buffer and not fits(f"{section}\n{trial}"):
                    flush()
                    trial = piece
                buffer = trial
                if not source_indexes or source_indexes[-1] != source_index:
                    source_indexes.append(source_index)
            if _CAPTION.match(value):
                flush()
        flush()
    return tuple(units)
