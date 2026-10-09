"""Command-line interface for PDF indexing and evidence-grounded answers."""

import argparse
import logging
from pathlib import Path
import sys

from .library import PaperLibrary
from .llm import ModelConfigError, ModelRequestError, generate_with_config
from .model_config import get_setting
from .agent import direct_question, research_question
from .semantic import FastEmbedProvider, SemanticIndex, build_retriever
from .unit_index import UnitStore, UnitSemanticIndex
from .spaces import ALL_PAPERS_SPACE_ID, SpaceStore


def _pdf_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        return [path]
    if path.is_dir():
        return sorted(file for file in path.rglob("*") if file.suffix.lower() == ".pdf")
    raise ValueError(f"Provide a PDF or a directory containing PDFs: {path}")


def _result_limit(value: str) -> int:
    limit = int(value)
    if not 1 <= limit <= 20:
        raise argparse.ArgumentTypeError("limit must be between 1 and 20")
    return limit


def main(argv: list[str] | None = None) -> int:
    # Some Windows shells still default to GBK, while PDFs contain other Unicode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Ask research PDFs with cited evidence")
    parser.add_argument("--db", type=Path, default=Path(".paperanchor/library.sqlite3"))
    parser.add_argument("--log-level", choices=("WARNING", "INFO"), default="WARNING",
                        help="Enable safe stage timing records with INFO")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="Extract and index a PDF or directory")
    index.add_argument("path", type=Path)
    search = commands.add_parser("search", help="Inspect BM25 passages without an LLM")
    search.add_argument("question")
    search.add_argument("--limit", type=_result_limit, default=5)
    ask = commands.add_parser("ask", help="Retrieve passages and generate a cited answer")
    ask.add_argument("question")
    ask.add_argument("--limit", type=_result_limit, default=8)
    serve = commands.add_parser("serve", help="Open the local research workspace")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("paperanchor.operations").setLevel(args.log_level)
    if args.command in ("search", "ask") and not args.question.strip():
        parser.error("question must not be empty")
    if args.command == "serve":
        import uvicorn

        from .web import create_app

        uvicorn.run(create_app(args.db), host=args.host, port=args.port)
        return 0
    library = PaperLibrary(args.db)

    if args.command == "index":
        try:
            files = _pdf_files(args.path)
        except ValueError as error:
            parser.error(str(error))
        if not files:
            parser.error(f"No PDF files found in {args.path}")
        for file in files:
            try:
                result = library.index_pdf(file)
            except (ValueError, OSError) as error:
                print(f"FAILED {file}: {error}")
                return 1
            action = "indexed" if result.changed else "unchanged"
            print(f"{action}: {file.name} ({result.passage_count} passages)")
        return 0

    if args.command == "search":
        hits = library.search(args.question, args.limit)
        if not hits:
            print("No matching passages. Try English terms present in the PDFs.")
            return 0
        for number, hit in enumerate(hits, 1):
            print(f"\n[{number}] {hit.paper_title} | page {hit.page_number} | {hit.paper_path}")
            print(hit.text[:700])
        return 0

    try:
        SpaceStore(args.db)
        provider = FastEmbedProvider(cache_dir=args.db.parent / "models")
        representation = get_setting("PAPERANCHOR_REPRESENTATION").lower() or "units"
        if representation not in ("blocks", "units"):
            parser.error("PAPERANCHOR_REPRESENTATION must be 'blocks' or 'units'")
        source = UnitStore(library) if representation == "units" else library
        index = UnitSemanticIndex(source) if representation == "units" else SemanticIndex(args.db)
        if representation == "units":
            source.refresh()
            if index.count(provider.name) < source.count_units():
                index.build(provider)
        retriever = build_retriever(
            source, index, provider,
            rerank_model=get_setting("PAPERANCHOR_RERANK_MODEL"),
            cache_dir=args.db.parent / "models",
        )
        mode = get_setting("PAPERANCHOR_ANSWER_MODE").lower() or "direct"
        if mode not in ("direct", "agent"):
            parser.error("PAPERANCHOR_ANSWER_MODE must be direct or agent")
        result = (research_question if mode == "agent" else direct_question)(
            retriever, args.question, generate_with_config,
            limit=args.limit, space_id=ALL_PAPERS_SPACE_ID,
            collection=library,
        )
    except (ModelConfigError, ModelRequestError) as error:
        print(error)
        return 2
    answer = result.answer
    print(f"Status: {answer.status} | Basis: {result.basis}\n\n{answer.text}")
    if result.follow_up_question:
        print(f"\nFollow-up: {result.follow_up_question}")
    if answer.evidence:
        print("\nRetrieved evidence:")
        for item in answer.evidence:
            print(f"[{item.evidence_id}] {item.paper_title}, PDF page {item.page_number}")
            print(f"    {item.paper_path}")
    return 0 if answer.status in ("answered", "no_evidence") else 1
