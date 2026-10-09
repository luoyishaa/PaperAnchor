"""Turn a PDF into located text passages without choosing a search engine."""

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import pymupdf
from .visual_index import DetectedVisual, detect_page


@dataclass(frozen=True)
class ExtractedPassage:
    page_number: int
    block_number: int
    text: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ParsedPdf:
    title: str
    page_count: int
    passages: tuple[ExtractedPassage, ...]
    visuals: tuple[DetectedVisual, ...] = ()


_ARXIV_FILENAME = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?", re.I)


def _first_page_title(page: pymupdf.Page) -> str | None:
    """Use prominent first-page type only when a numbered PDF has no title metadata."""
    candidates = []
    for block in page.get_text("dict")["blocks"][:15]:
        lines = block.get("lines", ())
        spans = [span for line in lines for span in line["spans"]]
        if not spans:
            continue
        text = " ".join("".join(span["text"] for span in line["spans"])
                        for line in lines)
        text = " ".join(unicodedata.normalize("NFKC", text).split())
        if text and not text.lower().startswith("arxiv:"):
            candidates.append((text, max(span["size"] for span in spans)))
    if not candidates:
        return None
    # Some article templates use a 12pt title while an arXiv stamp is 20pt.
    # Compare against non-stamp type rather than a fixed 13pt threshold.
    threshold = max(11.0, max(size for _, size in candidates) - 0.75)
    title_parts = []
    for text, size in candidates:
        if size >= threshold and 5 <= len(text) <= 200:
            title_parts.append(text)
        elif title_parts:
            break
    title = " ".join(title_parts)
    return title[:200] if title else None


def parse_pdf(path: Path) -> ParsedPdf:
    """Extract text blocks, retaining the PDF page and block rectangle."""
    passages: list[ExtractedPassage] = []
    visuals: list[DetectedVisual] = []
    with pymupdf.open(path) as document:
        title = (document.metadata.get("title") or "").strip() or path.stem
        if len(document) and title == path.stem and _ARXIV_FILENAME.fullmatch(path.stem):
            title = _first_page_title(document[0]) or title
        page_count = len(document)
        for page_index, page in enumerate(document):
            visuals.extend(detect_page(page))
            for block in page.get_text("blocks", sort=True):
                x0, y0, x1, y1, raw_text, block_number, block_type = block
                if block_type != 0:
                    continue
                text = " ".join(unicodedata.normalize("NFKC", raw_text).split())
                if not text:
                    continue
                passages.append(
                    ExtractedPassage(
                        page_number=page_index + 1,
                        block_number=block_number,
                        text=text,
                        bbox=(x0, y0, x1, y1),
                    )
                )
    if not passages:
        raise ValueError(f"No extractable text found in {path}")
    return ParsedPdf(title=title, page_count=page_count, passages=tuple(passages),
                     visuals=tuple(visuals))
