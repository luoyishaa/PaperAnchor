"""Verify public manifest titles against indexed PDF bytes and first-page text.

Aliases change routing labels only. The database, indexed passages, and titles
rendered in evidence remain untouched. Building records every indexed paper;
loading fails closed when a saved alias no longer matches its database row.
"""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

import pymupdf


VERSION = 1
_HASH = re.compile(r"[0-9a-f]{64}")


def _title_key(text):
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold()
                   if char.isalnum())


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _path_key(path, base):
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _first_page_text(path):
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError("PDF requires a password")
        if not len(document):
            raise ValueError("PDF has no pages")
        return document[0].get_text("text", sort=True)


def _fingerprint(aliases):
    ordered = sorted(aliases, key=lambda row: row["paper_id"])
    canonical = json.dumps(ordered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _connect(database):
    path = Path(database).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Existing database required: {path}")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def build_aliases(database, manifest_path) -> dict:
    """Check all indexed papers and return routing aliases verified without gold.

    Manifest entries must already be ready, parsed, and title-verified. Match by
    stored PDF path, falling back to the immutable PDF hash when files moved.
    Ambiguous or unverifiable entries keep their existing routing title and
    receive an explicit reason in ``checks``.
    """
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Manifest must contain a candidates list")
    manifest_hash = sha256(raw).hexdigest()
    by_path, by_hash = defaultdict(list), defaultdict(list)
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Manifest candidates must be objects")
        if candidate.get("pdf_path"):
            by_path[_path_key(candidate["pdf_path"], manifest_path.parent)].append(candidate)
        if isinstance(candidate.get("sha256"), str):
            by_hash[candidate["sha256"].lower()].append(candidate)
    aliases, checks = [], []
    connection = _connect(database)
    try:
        connection.execute("BEGIN")
        papers = connection.execute("SELECT id,file_path,file_hash,title FROM papers ORDER BY id").fetchall()
        for paper in papers:
            # PaperLibrary stores absolute file paths; resolve relative fixtures
            # or legacy rows against the process working directory.
            path = _path_key(paper["file_path"], Path.cwd())
            check = {"paper_id": paper["id"], "file_path": str(path),
                     "file_hash": paper["file_hash"], "original_title": paper["title"],
                     "status": "kept_original", "reason": None}
            checks.append(check)
            matches = by_path.get(path, [])
            check["manifest_match"] = "path"
            if not matches:
                matches = by_hash.get(paper["file_hash"].lower(), [])
                check["manifest_match"] = "sha256"
            if len(matches) != 1:
                check["reason"] = "manifest_entry_missing" if not matches else "ambiguous_manifest_entries"
                continue
            candidate = matches[0]
            check["manifest_paper_id"] = candidate.get("paper_id")
            check["manifest_title"] = candidate.get("title")
            if (candidate.get("status") != "ready" or candidate.get("parse_status") != "parsed"
                    or candidate.get("identity_status") != "title_verified"):
                check["reason"] = "manifest_entry_not_verified"
                continue
            title = candidate.get("title")
            if not isinstance(title, str) or not _title_key(title):
                check["reason"] = "manifest_title_missing"
                continue
            expected = candidate.get("sha256")
            if not isinstance(expected, str) or not _HASH.fullmatch(expected.lower()):
                check["reason"] = "manifest_hash_missing_or_invalid"
                continue
            expected = expected.lower()
            check["manifest_sha256"] = expected
            if expected != paper["file_hash"].lower():
                check["reason"] = "manifest_database_hash_mismatch"
                continue
            if not path.is_file():
                check["reason"] = "pdf_missing"
                continue
            try:
                actual = _file_hash(path)
            except OSError as error:
                check.update(reason="pdf_unreadable", error=f"{type(error).__name__}: {error}")
                continue
            check["actual_pdf_sha256"] = actual
            if actual != expected:
                check["reason"] = "pdf_database_hash_mismatch"
                continue
            try:
                first_page = _first_page_text(path)
            except Exception as error:
                check.update(reason="pdf_text_unavailable", error=f"{type(error).__name__}: {error}")
                continue
            check["first_page_text_sha256"] = sha256(first_page.encode("utf-8")).hexdigest()
            matched = _title_key(title) in _title_key(first_page)
            check["first_page_title_verified"] = matched
            if not matched:
                check["reason"] = "title_not_confirmed_on_first_page"
                continue
            if _title_key(title) == _title_key(paper["title"]):
                check.update(status="verified_unchanged", reason="normalized_titles_equal")
                continue
            source = {"kind": "verified_qasper_manifest_and_pdf_first_page",
                      "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
                      "manifest_paper_id": candidate.get("paper_id"),
                      "manifest_pdf_path": candidate.get("pdf_path"),
                      "url": candidate.get("url"), "actual_pdf_sha256": actual,
                      "first_page_text_sha256": check["first_page_text_sha256"],
                      "title_normalization": "NFKC_casefold_alphanumeric"}
            aliases.append({"paper_id": paper["id"], "file_hash": paper["file_hash"],
                            "original_title": paper["title"], "title": title, "source": source})
            check.update(status="alias_added", reason="verified_public_title_differs")
    finally:
        connection.close()
    return {"version": VERSION, "aliases": aliases, "checks": checks,
            "fingerprint": _fingerprint(aliases)}


def load_aliases(path, database) -> dict[int, str]:
    """Load aliases after checking their digest and current DB hash/title.

    A missing paper, hash mismatch, changed original title, or malformed alias
    fails the whole load explicitly. PDF bytes are verified when building;
    loading reads only SQLite metadata, so it does not reparse the corpus.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != VERSION or not isinstance(payload.get("aliases"), list):
        raise ValueError("Unsupported title-alias schema")
    if not isinstance(payload.get("checks"), list):
        raise ValueError("Title-alias checks must be a list")
    aliases, seen = payload["aliases"], set()
    for alias in aliases:
        if not isinstance(alias, dict):
            raise ValueError("Title alias must be an object")
        identifier = alias.get("paper_id")
        if type(identifier) is not int or identifier < 1 or identifier in seen:
            raise ValueError("Title aliases require unique positive paper IDs")
        seen.add(identifier)
        if not isinstance(alias.get("file_hash"), str) or not _HASH.fullmatch(alias["file_hash"].lower()):
            raise ValueError(f"Invalid title-alias file hash for paper {identifier}")
        if (not isinstance(alias.get("title"), str) or not _title_key(alias["title"])
                or not isinstance(alias.get("original_title"), str)
                or not isinstance(alias.get("source"), dict)):
            raise ValueError(f"Invalid title alias for paper {identifier}")
        if _title_key(alias["title"]) == _title_key(alias["original_title"]):
            raise ValueError(f"Title alias does not change routing title for paper {identifier}")
    if payload.get("fingerprint") != _fingerprint(aliases):
        raise ValueError("Title-alias fingerprint mismatch")
    connection = _connect(database)
    try:
        connection.execute("BEGIN")
        result = {}
        for alias in aliases:
            row = connection.execute("SELECT file_hash,title FROM papers WHERE id=?",
                                     (alias["paper_id"],)).fetchone()
            if row is None:
                raise ValueError(f"Title-alias paper {alias['paper_id']} is missing")
            if row["file_hash"].lower() != alias["file_hash"].lower():
                raise ValueError(f"Title-alias database file hash mismatch for paper {alias['paper_id']}")
            if row["title"] != alias["original_title"]:
                raise ValueError(f"Title-alias original title mismatch for paper {alias['paper_id']}")
            result[alias["paper_id"]] = alias["title"]
        return result
    finally:
        connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be a new file")
    result = build_aliases(args.database, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"papers_checked": len(result["checks"]), "aliases": len(result["aliases"]),
                      "statuses": dict(Counter(row["status"] for row in result["checks"])),
                      "fingerprint": result["fingerprint"], "output": str(args.output)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
