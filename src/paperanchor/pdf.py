"""Turn a PDF into located text passages without choosing a search engine."""

from dataclasses import dataclass
from pathlib import Path
import unicodedata

import pymupdf


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


def parse_pdf(path: Path) -> ParsedPdf:
    """Extract text blocks, retaining the PDF page and block rectangle."""
    passages: list[ExtractedPassage] = []
    with pymupdf.open(path) as document:
        title = (document.metadata.get("title") or "").strip() or path.stem
        page_count = len(document)
        for page_index, page in enumerate(document):
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
    return ParsedPdf(title=title, page_count=page_count, passages=tuple(passages))
