# PaperAnchor

PaperAnchor is a command-line RAG application for asking questions about local research PDFs. It extracts page-located passages, indexes them for full-text search, and supplies retrieved evidence to a configurable language model. Answers carry evidence IDs and PDF page references so the source can be checked.

## Capabilities

- Index one PDF or a directory of PDFs. Re-indexing an unchanged file is a no-op.
- Search indexed passages with SQLite FTS5 and BM25 ranking, without a model account.
- Generate answers through a Chat Completions-compatible API using retrieved passages.
- Check that generated evidence IDs refer to passages supplied to the model.
- Store the PDF page number and text-block rectangle for each passage.

## Architecture

```text
PDF → text blocks with page coordinates → SQLite passages + FTS5 index
                                              ↓
Question → BM25 retrieval → numbered evidence → model → cited answer
```

The PDF parser, library, answer workflow, model adapter, and CLI are separate modules. See [Architecture](docs/architecture.md) for their responsibilities and data contracts.

## Requirements

- Python 3.11+
- Text-based PDFs (scanned pages without a text layer are not supported)
- A Chat Completions-compatible API for `ask`; `index` and `search` work offline

## Install and run

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\paperanchor.exe index data
.\.venv\Scripts\paperanchor.exe search "retrieval augmented generation"
```

On macOS or Linux, activate the virtual environment and run `pip install -e '.[dev]'`, then use `paperanchor` in place of `.\.venv\Scripts\paperanchor.exe`.

`data/` is a local input directory and is ignored by Git. Supply your own PDFs there, or pass an individual PDF path to `index`. The default index is `.paperanchor/library.sqlite3`; use `--db PATH` before the subcommand to choose another location.

For generated answers, set the endpoint, model ID, and API key in your local environment:

```powershell
$env:PAPERANCHOR_BASE_URL = 'https://your-provider.example/v1'
$env:PAPERANCHOR_API_KEY = 'your-key'
$env:PAPERANCHOR_MODEL = 'your-model-id'
.\.venv\Scripts\paperanchor.exe ask "What problem does retrieval-augmented generation address?"
```

Use the values documented by your provider. `.env` and the local index are ignored by Git. The application reads environment variables; it does not load `.env` automatically.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Tests cover PDF indexing, repeated and changed-file indexing, retrieval, cited answers, and missing or invalid evidence. They use generated PDFs and a model stub; a live model call requires provider credentials.

## Current scope

This implementation uses English-term keyword retrieval and PDF text blocks. It does not yet provide semantic search, reranking, OCR, paper discovery, a PDF viewer, or multi-user research spaces. A passage reference locates a PDF page and block; it does not highlight an exact sentence. Citation validation checks evidence IDs, not whether every factual claim is actually supported. Bibliographies and short blocks can rank above explanatory text; improving retrieval quality requires a measured evaluation set.

## References

- [PyMuPDF text extraction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html)
- [SQLite FTS5 and BM25](https://www.sqlite.org/fts5.html)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
