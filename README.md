# PaperAnchor

Ask questions about research papers and follow each citation back to the PDF.

PaperAnchor is a local research workspace built with Python and React. Collect papers, search their contents, and read answers alongside highlighted source passages. Your library, index, conversations, and notes stay on your computer; configured model providers receive the excerpts or page images needed to answer a question.

## Features

- **A paper library for each research topic.** Upload PDFs or discover papers through arXiv and Semantic Scholar. Reuse papers across research spaces.
- **Answers you can inspect.** Click a citation to open the original PDF page and highlight its source passages. Continue the conversation or ask about selected papers.
- **Hybrid retrieval.** Combine BM25 and local vector search, rerank with MiniLM or BGE, and restore surrounding evidence within a 4,096-token context budget.
- **Figures and tables.** Ask about a detected figure or table, or select a region in the reader, using a configured vision model.
- **Local or hosted models.** Use Ollama locally or configure a supported API provider in `.env`.

## Quickstart

Requires Python 3.11 or 3.12. From a checkout, create and activate a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with `.\.venv\Scripts\Activate.ps1` instead.

```sh
pip install .
paperanchor serve
```

Open [localhost:8000](http://127.0.0.1:8000), upload a PDF, and create a research space. The web interface is included in the Python package. PDFs need a selectable text layer.

Copy [`.env.example`](.env.example) to `.env` and fill in the key for your chosen answer provider. The example selects GLM with `ZHIPU_API_KEY`; local Ollama needs no API key. Importing, reading, and keyword search work without an answer model. Semantic indexing downloads a local embedding model on first use and runs in the background.

The browser defaults to **Fast (MiniLM)** and remembers your choice. Fast and Quality (BGE) require their local reranker models to be cached. Use **Standard** if those models are not available. **Optimized** uses the benchmark configuration and is enabled only when its saved corpus mapping matches the library. See [retrieval profiles](docs/architecture.md#retrieval-profiles) for details.

You can also work from the command line:

```sh
paperanchor index ./papers
paperanchor search "retrieval augmented generation"
paperanchor ask "What evidence supports the proposed method?"
```

`index` extracts PDF text; `search` runs keyword search; `ask` builds missing semantic embeddings and generates a cited answer. CLI questions use the Standard pipeline. Add `--db PATH` before a subcommand to use a different library. The default database is `.paperanchor/library.sqlite3`.

## Evaluation

Retrieval was evaluated on **120 QASPER questions over 400 papers**, with a fixed **4,096-token** evidence budget. Complete evidence coverage measures whether the returned context contains a full annotated evidence set for each of the 96 answerable questions.

| Profile | Complete evidence coverage | Mean retrieval time |
| --- | ---: | ---: |
| MiniLM | 82.29% (79/96) | 2.19 s |
| BGE | 83.33% (80/96) | 8.30 s |
| Optimized | **90.63% (87/96)** | 32.15 s |

The remaining 24 questions are unanswerable auxiliary cases. Timings cover retrieval and context assembly, excluding model loading and answer generation; MiniLM and BGE use the preceding evaluation run, while Optimized was rerun on all 120 questions with result caches disabled. These are retrieval results, not answer-accuracy scores.

See [evaluation details](docs/evaluation.md), [per-question results](evals/results/qasper120.json), and the [benchmark guide](evals/README.md).

A separate **question-only LitQA2 evaluation** uses no injected paper title or
source filter. On 95 questions with obtainable full text in a 93-paper local
corpus, Fast retrieval returned the annotated source for all 95 questions, and
local Qwen3-8B answered 77 correctly (**81.05%**), with 7 wrong choices and 11
abstentions. This is an accessible-subset, XML-text evaluation; 104 of the 199
public questions lacked obtainable source text in this run. It does not evaluate
PDF parsing or web-scale search. See the [protocol and results](docs/litqa2-evaluation.md).

## Development

The backend uses FastAPI, SQLite FTS5, PyMuPDF, and FastEmbed. The frontend uses React and TypeScript. See [architecture](docs/architecture.md) for the retrieval and citation pipeline.

```sh
pip install -e '.[dev]'
python -m pytest
cd frontend
npm ci
npm run dev
```

The Vite server proxies `/api` to `paperanchor serve`. Node.js 22.12+ is needed for frontend development. Run `npm test`, `npm run lint`, and `npm run build` for frontend checks. Browser tests use `npm run test:browser` after `npx playwright install chromium`.

Before packaging, run `npm run build` in `frontend/`, then `python scripts/build_wheel.py` from the repository root with the Python `build` package installed. This verifies that the wheel contains the current frontend assets. Tests use generated PDFs and model stubs, so they need no provider keys.

## Scope

PaperAnchor is intended for one researcher and a personal paper library. It has no authentication or multi-user isolation. Scanned PDFs without text are not indexed, and vision answers can misread figures. Citations identify the supplied evidence; they do not independently prove an answer is correct.

## License

[MIT](LICENSE)
