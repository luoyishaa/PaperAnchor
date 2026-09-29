"""Normalize remote paper search results without coupling them to local indexing."""

from dataclasses import dataclass
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx
from .model_config import get_setting


ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z.-]+(?:/[a-z.-]+)?/\d{7})(?:v\d+)?$", re.I)


@dataclass(frozen=True)
class Candidate:
    source: str
    source_id: str
    title: str
    authors: tuple[str, ...]
    abstract: str
    published: str | None
    year: int | None
    venue: str | None
    citation_count: int | None
    paper_url: str
    arxiv_id: str | None
    doi: str | None
    importable: bool


def _clean(value: str | None) -> str:
    return " ".join((value or "").split())


def parse_arxiv(feed: bytes) -> tuple[Candidate, ...]:
    root = ElementTree.fromstring(feed)
    candidates = []
    for entry in root.findall(f"{ATOM}entry"):
        link = _clean(entry.findtext(f"{ATOM}id"))
        arxiv_id = urlparse(link).path.removeprefix("/abs/")
        if not ARXIV_ID.fullmatch(arxiv_id):
            continue
        published = _clean(entry.findtext(f"{ATOM}published"))[:10] or None
        candidates.append(Candidate(
            source="arxiv", source_id=arxiv_id,
            title=_clean(entry.findtext(f"{ATOM}title")),
            authors=tuple(_clean(node.findtext(f"{ATOM}name"))
                          for node in entry.findall(f"{ATOM}author")),
            abstract=_clean(entry.findtext(f"{ATOM}summary")),
            published=published, year=int(published[:4]) if published else None,
            venue=_clean(entry.findtext(f"{ARXIV}journal_ref")) or None,
            citation_count=None, paper_url=f"https://arxiv.org/abs/{arxiv_id}",
            arxiv_id=arxiv_id, doi=_clean(entry.findtext(f"{ARXIV}doi")) or None,
            importable=True,
        ))
    return tuple(candidates)


def parse_semantic_scholar(data: dict) -> tuple[Candidate, ...]:
    candidates = []
    for item in data.get("data", []):
        paper_id = item.get("paperId")
        if not paper_id or not item.get("title"):
            continue
        ids = item.get("externalIds") or {}
        arxiv_id = ids.get("ArXiv")
        if arxiv_id and not ARXIV_ID.fullmatch(arxiv_id):
            arxiv_id = None
        published = item.get("publicationDate")
        year = item.get("year")
        record_url = item.get("url") or ""
        if urlparse(record_url).scheme != "https":
            record_url = f"https://www.semanticscholar.org/paper/{paper_id}"
        candidates.append(Candidate(
            source="semantic_scholar", source_id=paper_id,
            title=_clean(item["title"]),
            authors=tuple(_clean(author.get("name")) for author in item.get("authors") or []),
            abstract=_clean(item.get("abstract")),
            published=published, year=year,
            venue=_clean(item.get("venue")) or None,
            citation_count=item.get("citationCount"),
            paper_url=record_url,
            arxiv_id=arxiv_id, doi=ids.get("DOI"), importable=bool(arxiv_id),
        ))
    return tuple(candidates)


class Discovery:
    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=15, follow_redirects=False)
        self.owns_client = client is None
        self._arxiv_lock = Lock()
        self._last_arxiv_call = 0.0

    def close(self) -> None:
        if self.owns_client:
            self.client.close()

    def _arxiv_metadata(self, params: dict) -> httpx.Response:
        # arXiv asks clients to leave at least three seconds between API calls.
        with self._arxiv_lock:
            delay = 3.0 - (time.monotonic() - self._last_arxiv_call)
            if delay > 0:
                time.sleep(delay)
            self._last_arxiv_call = time.monotonic()
            return self.client.get("https://export.arxiv.org/api/query", params=params)

    def search(self, query: str, source: str, limit: int = 10) -> tuple[Candidate, ...]:
        query = query.strip()
        if not query or not 1 <= limit <= 20:
            raise ValueError("Provide a query and a limit from 1 to 20")
        if source == "arxiv":
            # One small request per user action; no automated paging or bulk harvesting.
            terms = re.findall(r"[\w-]+", query, flags=re.UNICODE)[:8]
            expression = " AND ".join(f"all:{term}" for term in terms)
            response = self._arxiv_metadata({
                "search_query": expression, "start": 0, "max_results": limit,
                "sortBy": "relevance", "sortOrder": "descending",
            })
            response.raise_for_status()
            return parse_arxiv(response.content)
        if source == "semantic_scholar":
            key = get_setting("SEMANTIC_SCHOLAR_API_KEY")
            response = self.client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": limit,
                        "fields": "title,abstract,authors,year,publicationDate,venue,"
                                  "citationCount,url,externalIds"},
                headers={"x-api-key": key} if key else {},
            )
            response.raise_for_status()
            return parse_semantic_scholar(response.json())
        raise ValueError("Unknown discovery source")

    def arxiv_pdf(self, arxiv_id: str) -> bytes:
        if not ARXIV_ID.fullmatch(arxiv_id):
            raise ValueError("Invalid arXiv ID")
        response = self.client.get(f"https://arxiv.org/pdf/{arxiv_id}")
        response.raise_for_status()
        if len(response.content) > 40 * 1024 * 1024:
            raise ValueError("PDF exceeds 40 MB")
        if not response.content.startswith(b"%PDF-"):
            raise ValueError("arXiv did not return a PDF")
        return response.content

    def arxiv_record(self, arxiv_id: str) -> Candidate:
        if not ARXIV_ID.fullmatch(arxiv_id):
            raise ValueError("Invalid arXiv ID")
        response = self._arxiv_metadata({"id_list": arxiv_id})
        response.raise_for_status()
        matches = parse_arxiv(response.content)
        if not matches:
            raise ValueError("arXiv paper was not found")
        return matches[0]


class MetadataStore:
    """Imported discovery metadata is optional; the local PDF remains source of truth."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        with closing(self._connect()) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS paper_metadata (
                    paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
                    source TEXT NOT NULL, source_id TEXT NOT NULL,
                    authors TEXT NOT NULL, abstract TEXT NOT NULL,
                    published TEXT, year INTEGER, venue TEXT,
                    citation_count INTEGER, paper_url TEXT NOT NULL,
                    arxiv_id TEXT, doi TEXT
                )
            """)
            connection.commit()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def put(self, paper_id: int, candidate: Candidate) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("""
                INSERT INTO paper_metadata
                (paper_id, source, source_id, authors, abstract, published, year,
                 venue, citation_count, paper_url, arxiv_id, doi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    source=excluded.source, source_id=excluded.source_id,
                    authors=excluded.authors, abstract=excluded.abstract,
                    published=excluded.published, year=excluded.year,
                    venue=excluded.venue, citation_count=excluded.citation_count,
                    paper_url=excluded.paper_url, arxiv_id=excluded.arxiv_id,
                    doi=excluded.doi
            """, (paper_id, candidate.source, candidate.source_id,
                  ", ".join(candidate.authors), candidate.abstract,
                  candidate.published, candidate.year, candidate.venue,
                  candidate.citation_count, candidate.paper_url,
                  candidate.arxiv_id, candidate.doi))

    def get(self, paper_id: int) -> dict | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM paper_metadata WHERE paper_id=?",
                                     (paper_id,)).fetchone()
            return dict(row) if row else None
