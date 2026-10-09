"""Caption-anchored visual regions, versioned against the original PDF bytes.

The index stores PDF rectangles, never model-generated coordinates. Detection
is conservative: a caption without a plausible body stays a page-level source.
"""

from dataclasses import dataclass
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata

import pymupdf

from .database import Database


DETECTOR_VERSION = 2
LABEL = re.compile(r"(?:figure|fig\.?|table|图|表)\s*([0-9]+)(?![0-9])", re.I)
CAPTION = re.compile(r"^(figure|fig\.?|table|图|表)\s*([0-9]+)\s*[:.：]", re.I)


def visual_label(text: str) -> tuple[str, str] | None:
    match = LABEL.search(text)
    if not match:
        return None
    return ("table" if match.group(0).lower().startswith(("table", "表"))
            else "figure", match.group(1))


@dataclass(frozen=True)
class DetectedVisual:
    page_number: int
    kind: str
    number: str
    caption: str
    bbox: tuple[float, float, float, float]
    context: str
    method: str


@dataclass(frozen=True)
class VisualHit:
    region_id: int
    paper_id: int
    paper_title: str
    path: Path
    source_hash: str
    region: DetectedVisual


def _overlap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def detect_page(page: pymupdf.Page) -> tuple[DetectedVisual, ...]:
    """Associate captions with graphics or a bounded run of table text.

    Coordinates are in the unrotated PDF frame, as returned by get_text.
    Mixed/ambiguous layouts deliberately fall back to whole-page reading.
    """
    blocks = []
    captions = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        lines = [(pymupdf.Rect(line["bbox"]), "".join(s["text"] for s in line["spans"]))
                 for line in block["lines"]]
        text = " ".join(unicodedata.normalize("NFKC", t).strip() for _, t in lines)
        starts = []
        for index, (rect, value) in enumerate(lines):
            # Some typesetters put 'Table' and its number on separate lines.
            candidate = " ".join(t for _, t in lines[index:])
            match = CAPTION.match(unicodedata.normalize("NFKC", candidate).strip())
            if match:
                starts.append(index)
        if len(starts) > 1:
            # A merged extraction block must not enlarge one crop to another caption.
            blocks.extend((rect, value, [(rect, value)]) for rect, value in lines)
        else:
            blocks.append((pymupdf.Rect(block["bbox"]), text, lines))
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(lines)
            segment = lines[start:end]
            box = pymupdf.Rect(segment[0][0])
            for extra, _ in segment[1:]:
                box |= extra
            candidate = " ".join(value for _, value in segment)
            captions.append((box, candidate, visual_label(candidate)))
    if not captions:
        return ()
    images = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
    graphics = list(images)
    # cluster_drawings is optional on older supported PyMuPDF versions.
    if hasattr(page, "cluster_drawings"):
        graphics.extend(page.cluster_drawings())
    else:
        graphics.extend(pymupdf.Rect(d["rect"]) for d in page.get_drawings())
    result = []
    bounds = page.rect * page.derotation_matrix
    for caption_box, caption, label in captions:
        kind, number = label
        body = None
        method = ""
        if kind == "figure" or images:
            # Raster tables can be associated directly; vector shapes alone
            # cannot distinguish a table from a neighboring chart.
            nearby = [g for g in (graphics if kind == "figure" else images)
                      if g.width >= 25 and g.height >= 20
                      and _overlap(g, caption_box) >= min(g.width, caption_box.width) * .25
                      and (-3 <= caption_box.y0 - g.y1 <= 75
                           or -3 <= g.y0 - caption_box.y1 <= 35)]
            if nearby:
                nearest = min(nearby, key=lambda g: min(abs(caption_box.y0 - g.y1),
                                                       abs(g.y0 - caption_box.y1)))
                # Preserve all panels sharing the caption and the same row.
                panels = [g for g in nearby if abs(g.y1 - nearest.y1) < 25]
                body = pymupdf.Rect(panels[0])
                for panel in panels[1:]:
                    body |= panel
                method = "caption_graphics"
        if kind == "table" and body is None:
            # Paragraphs are barriers; table cells are short line fragments.
            tabular = []
            barriers = []
            for rect, text, lines in blocks:
                if _overlap(rect, caption_box) < min(rect.width, caption_box.width) * .35:
                    continue
                if len(text) > 180 and sum(len(t) for _, t in lines) / max(1, len(lines)) > 45:
                    barriers.append(rect)
                    continue
                for line_rect, value in lines:
                    if not value.strip() or line_rect.intersects(caption_box):
                        continue
                    if len(value) <= 100:
                        tabular.append(line_rect)
            above = max((r.y1 for r in barriers if r.y1 <= caption_box.y0), default=bounds.y0 + 35)
            below = min((r.y0 for r in barriers if r.y0 >= caption_box.y1), default=bounds.y1 - 35)
            # Other captions delimit independent figures/tables in the same lane.
            for other, _, _ in captions:
                if other == caption_box or _overlap(other, caption_box) <= 0:
                    continue
                if other.y1 <= caption_box.y0:
                    above = max(above, other.y1)
                elif other.y0 >= caption_box.y1:
                    below = min(below, other.y0)
            upper = [r for r in tabular if above + 2 <= r.y0 and r.y1 < caption_box.y0]
            lower = [r for r in tabular if r.y0 > caption_box.y1 and r.y1 <= below - 2]
            options = [(min(caption_box.y0-r.y1 for r in upper), upper)] if upper else []
            if lower:
                options.append((min(r.y0-caption_box.y1 for r in lower), lower))
            if options:
                distance, cells = min(options, key=lambda item: item[0])
                if distance <= 35 and len(cells) >= 3:
                    # Walk from caption until a large whitespace gap; avoid a
                    # nearby heading becoming part of the table.
                    cells.sort(key=lambda r: abs((r.y0+r.y1)/2 - (caption_box.y0+caption_box.y1)/2))
                    body = pymupdf.Rect(cells[0])
                    for cell in cells[1:]:
                        gap = max(body.y0-cell.y1, cell.y0-body.y1, 0)
                        if gap > max(12, min(body.height, cell.height) * 1.3):
                            break
                        body |= cell
                    if body.height < 15:
                        body = None
                    method = "caption_table_text"
        if body is None:
            continue
        # Caption and nearby axis/panel labels are part of the readable region.
        box = body | caption_box
        box = (box + (-6, -6, 6, 6)) & bounds
        context = []
        for rect, text, lines in blocks:
            if rect.intersects(box):
                if len(text) < 180:
                    box |= rect & bounds
                context.append(text)
        box &= bounds
        if box.is_empty or box.width < 30 or box.height < 30 or box.get_area() > bounds.get_area() * .8:
            continue
        result.append(DetectedVisual(page.number + 1, kind, number, caption,
                                     tuple(box), "\n".join(context)[:2400], method))
    return tuple(result)


class VisualStore:
    """A derived index; PDF hash and detector version control its validity."""

    def __init__(self, library):
        self.library = library
        self.database: Database = library.database
        with self.database.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS visual_index_state (
                    paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
                    source_hash TEXT NOT NULL, version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS visual_regions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL, kind TEXT NOT NULL, number TEXT NOT NULL,
                    caption TEXT NOT NULL, bbox TEXT NOT NULL, context TEXT NOT NULL,
                    method TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS visual_region_paper ON visual_regions(paper_id, page_number);
            """)

    @contextmanager
    def _connect(self):
        with self.database.connect() as connection:
            yield connection

    def refresh(self, paper_ids: tuple[int, ...]) -> None:
        for paper_id in dict.fromkeys(paper_ids):
            with self._connect() as connection:
                paper = connection.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
                state = connection.execute("SELECT * FROM visual_index_state WHERE paper_id=?", (paper_id,)).fetchone()
            if paper is None:
                continue
            path = Path(paper["file_path"])
            if not path.is_file():
                raise FileNotFoundError(f"PDF is missing: {path.name}")
            data = path.read_bytes()
            source_hash = sha256(data).hexdigest()
            if source_hash != paper["file_hash"]:
                raise ValueError("PDF changed after import. Reimport it before visual reading.")
            if state and state["source_hash"] == source_hash and state["version"] == DETECTOR_VERSION:
                continue
            with pymupdf.open(stream=data, filetype="pdf") as document:
                regions = tuple(region for page in document for region in detect_page(page))
            with self._connect() as connection:
                # Detect outside the write transaction, then compare before publishing.
                current = connection.execute("SELECT file_hash FROM papers WHERE id=?", (paper_id,)).fetchone()
                if current is None or current[0] != source_hash:
                    continue
                self.replace_regions(connection, paper_id, source_hash, regions)

    @staticmethod
    def replace_regions(connection, paper_id: int, source_hash: str,
                        regions: tuple[DetectedVisual, ...]) -> None:
        connection.execute("DELETE FROM visual_regions WHERE paper_id=?", (paper_id,))
        connection.executemany("""INSERT INTO visual_regions
            (paper_id,page_number,kind,number,caption,bbox,context,method)
            VALUES (?,?,?,?,?,?,?,?)""", [(paper_id, r.page_number, r.kind, r.number,
                r.caption, json.dumps(r.bbox), r.context, r.method) for r in regions])
        connection.execute("INSERT OR REPLACE INTO visual_index_state VALUES (?,?,?)",
                           (paper_id, source_hash, DETECTOR_VERSION))

    def regions(self, paper_ids: tuple[int, ...]) -> tuple[VisualHit, ...]:
        self.refresh(paper_ids)
        hits = []
        with self._connect() as connection:
            for paper_id in dict.fromkeys(paper_ids):
                rows = connection.execute("""SELECT r.*, p.file_path, p.title, s.source_hash
                    FROM visual_regions r JOIN papers p ON p.id=r.paper_id
                    JOIN visual_index_state s ON s.paper_id=p.id
                    WHERE r.paper_id=? ORDER BY r.page_number, r.id""", (paper_id,)).fetchall()
                hits.extend(VisualHit(row["id"], paper_id, row["title"], Path(row["file_path"]),
                    row["source_hash"], DetectedVisual(row["page_number"], row["kind"], row["number"],
                        row["caption"], tuple(json.loads(row["bbox"])), row["context"], row["method"]))
                    for row in rows)
        return tuple(hits)
