"""Build and validate a separate library at the PDFs' final paths."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import fitz

from paperanchor.conversations import ConversationStore
from paperanchor.jobs import JobStore
from paperanchor.library import PaperLibrary
from paperanchor.semantic import DEFAULT_EMBEDDING_MODEL, FastEmbedProvider, SemanticIndex
from paperanchor.spaces import SpaceStore
from paperanchor.unit_index import UnitSemanticIndex, UnitStore


SPACE_NAME = "QASPER \u00b7 NLP & IR"


def cached_provider(cache_dir: Path) -> FastEmbedProvider:
    from fastembed import TextEmbedding

    provider = FastEmbedProvider(cache_dir=cache_dir)
    provider._model = TextEmbedding(
        model_name=provider.name, cache_dir=str(cache_dir), local_files_only=True,
        threads=4,
    )
    return provider


def validate_library(database: Path, mappings: list[dict], *,
                     model: str = DEFAULT_EMBEDDING_MODEL) -> dict:
    library = PaperLibrary(database)
    spaces = SpaceStore(database).list_spaces()
    expected = {item["index_paper_id"]: Path(item["pdf_path"]).resolve()
                for item in mappings}
    prepared_hashes = {item["index_paper_id"]: item["sha256"] for item in mappings}
    errors = []
    with library.database.connect() as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            errors.append(integrity)
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            errors.append("foreign_key_check failed")
        papers = connection.execute("SELECT id,file_path,file_hash FROM papers").fetchall()
        if {row["id"]: Path(row["file_path"]).resolve() for row in papers} != expected:
            errors.append("Paper IDs or paths differ from manifest")
        for row in papers:
            try:
                if row["file_hash"] != prepared_hashes.get(row["id"]):
                    errors.append(f"Indexed PDF differs from prepared revision: {row['id']}")
                if sha256(Path(row["file_path"]).read_bytes()).hexdigest() != row["file_hash"]:
                    errors.append(f"PDF content changed after indexing: {row['id']}")
                with fitz.open(row["file_path"]) as pdf:
                    if not pdf.page_count:
                        errors.append(f"Empty PDF: {row['id']}")
            except Exception as error:
                errors.append(f"PDF unavailable: {row['id']}: {type(error).__name__}")
        if len(spaces) != 2 or {space.name for space in spaces} != {"All papers", SPACE_NAME}:
            errors.append("Unexpected research spaces")
        for space in spaces:
            members = {row[0] for row in connection.execute(
                "SELECT paper_id FROM space_papers WHERE space_id=?", (space.id,))}
            if members != set(expected):
                errors.append(f"Incomplete space membership: {space.name}")
        for table in ("annotations", "conversations", "conversation_turns", "jobs"):
            if connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]:
                errors.append(f"Unexpected old records in {table}")
        orphan_units = connection.execute(
            "SELECT COUNT(*) FROM retrieval_units u WHERE NOT EXISTS "
            "(SELECT 1 FROM unit_sources s WHERE s.unit_id=u.id)").fetchone()[0]
        if orphan_units:
            errors.append("Reading units without sources")
        wrong_sources = connection.execute(
            "SELECT COUNT(*) FROM unit_sources s JOIN retrieval_units u ON u.id=s.unit_id "
            "JOIN passages p ON p.id=s.passage_id WHERE p.paper_id!=u.paper_id").fetchone()[0]
        if wrong_sources:
            errors.append("Reading units have sources in another paper")
        counts = {}
        for table in ("papers", "passages", "retrieval_units"):
            counts[table] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        vectors = {}
        for table, source in (("passage_embeddings", "passages"),
                              ("unit_embeddings", "retrieval_units")):
            rows = connection.execute(
                f"SELECT dimension,COUNT(*) AS n FROM {table} WHERE model=? GROUP BY dimension",
                (model,),
            ).fetchall()
            count = sum(row["n"] for row in rows)
            malformed = connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE model=? AND "
                "(dimension<=0 OR length(vector)!=dimension*4)", (model,),
            ).fetchone()[0]
            vectors[table] = {"count": count, "total": counts[source],
                              "dimensions": [row["dimension"] for row in rows],
                              "complete": count == counts[source] and count > 0
                              and len(rows) == 1 and malformed == 0}
        sample = connection.execute("SELECT text FROM passages ORDER BY id LIMIT 1").fetchone()
    if not counts["retrieval_units"]:
        errors.append("No reading units")
    if sample and not library.search(sample[0], limit=1):
        errors.append("Full-text search smoke check failed")
    return {"valid": not errors, "errors": errors, "integrity": integrity,
            "counts": counts, "vectors": vectors,
            "semantic_complete": all(item["complete"] for item in vectors.values()),
            "spaces": [{"id": s.id, "name": s.name, "papers": s.paper_count} for s in spaces]}


def build_library(manifest_path: Path, database: Path, *, cache_dir: Path,
                  embeddings: bool = True, minimum_papers: int = 300) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = [paper for paper in manifest["candidates"] if paper["status"] == "ready"]
    if len(papers) < minimum_papers:
        raise ValueError(f"preparation incomplete: {len(papers)} PDFs; need {minimum_papers}")
    if database.name == "library.sqlite3":
        raise ValueError("Build into a staging database, not the running library")
    fingerprint = sha256(json.dumps(papers, sort_keys=True).encode()).hexdigest()
    state_path = database.with_suffix(".build.json")
    if database.exists():
        if not state_path.exists():
            raise ValueError("Existing database has no matching build state")
        if json.loads(state_path.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            raise ValueError("Manifest changed; choose a new staging database")
    database.parent.mkdir(parents=True, exist_ok=True)
    state = {"fingerprint": fingerprint, "status": "building", "mappings": [], "failures": []}

    def save():
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(state_path)

    save()
    library = PaperLibrary(database)
    spaces = SpaceStore(database)
    ConversationStore(database)
    JobStore(database)
    research = next((space for space in spaces.list_spaces() if space.name == SPACE_NAME), None)
    if research is None:
        research = spaces.create_space(SPACE_NAME)
    units = UnitStore(library)
    block_vectors = SemanticIndex(database)
    unit_vectors = UnitSemanticIndex(units)
    for paper in papers:
        path = Path(paper["pdf_path"]).resolve()
        try:
            if sha256(path.read_bytes()).hexdigest() != paper["sha256"]:
                raise ValueError("PDF changed after preparation")
            result = library.index_pdf(path, display_name=paper["title"])
            if not result.passage_count:
                raise ValueError("PDF contains no parsed passages")
            spaces.add_paper(research.id, result.paper_id)
            state["mappings"].append({"paper_id": paper["paper_id"], "split": paper["split"],
                                      "title": paper["title"], "pdf_path": str(path),
                                      "sha256": paper["sha256"], "index_paper_id": result.paper_id})
            print(f"Indexed {len(state['mappings'])}/{len(papers)}: {paper['paper_id']}", flush=True)
        except Exception as error:
            state["failures"].append({"paper_id": paper["paper_id"], "error": str(error)})
        save()
    units.refresh()
    state["embedding_status"] = "not_requested"
    if embeddings and not state["failures"]:
        try:
            provider = cached_provider(cache_dir)
            progress = lambda done, total: print(f"Vectors {done}/{total}", flush=True)
            block_vectors.build(provider, progress=progress)
            unit_vectors.build(provider, progress=progress)
            state["embedding_status"] = "complete"
        except Exception as error:
            state["embedding_status"] = "incomplete"
            state["embedding_error"] = f"{type(error).__name__}: {error}"
    state["validation"] = validate_library(database, state["mappings"])
    state["status"] = ("ready" if not state["failures"] and state["validation"]["valid"]
                       and state["validation"]["semantic_complete"] else "incomplete")
    save()
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/qasper/manifest.json"))
    parser.add_argument("--database", type=Path, default=Path(".paperanchor/qasper-staging.sqlite3"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".paperanchor/models"))
    parser.add_argument("--without-embeddings", action="store_true")
    args = parser.parse_args()
    result = build_library(args.manifest.resolve(), args.database.resolve(),
                           cache_dir=args.cache_dir.resolve(), embeddings=not args.without_embeddings)
    print(json.dumps({key: result[key] for key in ("status", "embedding_status", "validation")},
                     ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
