"""Command-line interface for PDF indexing and evidence-grounded answers."""

import argparse
from pathlib import Path
import sys

from .library import PaperLibrary
from .llm import ModelConfigError, generate_with_config
from .rag import answer_question


def _pdf_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".pdf":
        return [path]
    if path.is_dir():
        return sorted(file for file in path.rglob("*") if file.suffix.lower() == ".pdf")
    raise ValueError(f"Provide a PDF or a directory containing PDFs: {path}")


def main(argv: list[str] | None = None) -> int:
    # Some Windows shells still default to GBK, while PDFs contain other Unicode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Ask research PDFs with cited evidence")
    parser.add_argument("--db", type=Path, default=Path(".paperanchor/library.sqlite3"))
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="Extract and index a PDF or directory")
    index.add_argument("path", type=Path)
    search = commands.add_parser("search", help="Inspect retrieved passages without an LLM")
    search.add_argument("question")
    search.add_argument("--limit", type=int, default=5)
    ask = commands.add_parser("ask", help="Retrieve passages and generate a cited answer")
    ask.add_argument("question")
    ask.add_argument("--limit", type=int, default=5)
    serve = commands.add_parser("serve", help="Open the local research workspace")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
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
        answer = answer_question(
            library, args.question, generate_with_config, limit=args.limit
        )
    except ModelConfigError as error:
        print(error)
        return 2
    print(f"Status: {answer.status}\n\n{answer.text}")
    if answer.evidence:
        print("\nRetrieved evidence:")
        for item in answer.evidence:
            print(f"[{item.evidence_id}] {item.paper_title}, PDF page {item.page_number}")
            print(f"    {item.paper_path}")
    return 0 if answer.status in ("answered", "no_evidence") else 1
