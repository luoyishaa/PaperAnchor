# PaperAnchor

PaperAnchor is a local research workspace for finding papers, building a PDF library, and asking questions against checkable evidence. Answers link to a highlighted source block in the original PDF. The project is designed for one researcher on one computer.

## Capabilities

- Create research spaces, reuse papers between spaces, and scope questions to one space.
- Search arXiv or Semantic Scholar for candidate papers. Review the date, abstract, venue, and citation count when available. Import arXiv-hosted PDFs, or upload local PDFs.
- Extract source blocks with PDF page coordinates. Index in SQLite FTS5 and a local multilingual vector index. Semantic indexing runs as a durable background job.
- Retrieve with BM25 and vector search, then fuse rankings. Optional BGE cross-encoder reranking is available when enabled.
- Check evidence sufficiency, try one revised search query when needed, and distinguish paper-based answers from optional model-knowledge answers.
- Click answer citations to open and highlight the source PDF page. Keep user and assistant notes visually distinct.

## Architecture

```text
PDF → page-located blocks → SQLite FTS5 + local vectors
                            ↓
Question → hybrid search → evidence check → answer + validated citation
                                               ↓
                                    PDF page + highlighted block
```

The browser interface is React and TypeScript; the Python API also supports a command-line path. CLI `ask` uses the same hybrid retrieval and evidence-checking workflow as the browser. See [Architecture](docs/architecture.md) for module boundaries and data contracts.

## Requirements

- Python 3.11+
- Text-based PDFs (scanned pages without a text layer are not supported)
- Node.js 20.19+ or 22.12+ only if you change and rebuild the frontend
- A supported model provider API key for answers; importing, reading, and searching do not require one
- Internet access for the first local embedding-model download and online paper discovery

## Install and run

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\paperanchor.exe serve
```

Open http://127.0.0.1:8000 after starting `serve`. You can upload a PDF there, inspect the library, ask a question, click a citation, move between pages, and add notes. The local server listens only on your computer by default. On macOS or Linux, activate the virtual environment and run `pip install .`, then use `paperanchor` in place of `.\.venv\Scripts\paperanchor.exe`.

For a directory of existing PDFs, run `paperanchor index data` before starting the server. `paperanchor search "retrieval augmented generation"` is a BM25-only inspection command that works before semantic indexing. `paperanchor ask "..."` runs the browser's answer workflow over **All papers**, using vectors once they have been indexed. The browser builds missing embeddings in the background and shows progress; CLI `index` itself only indexes PDF text.

The first semantic indexing job downloads a multilingual embedding model (about 220 MB). Existing BM25 search remains available while it builds. Semantic Scholar may rate-limit requests without its optional API key; arXiv search works without a key. Only candidates with an arXiv ID can currently be imported directly; open other records and upload their PDF manually.

To edit the frontend:

```powershell
cd frontend
npm ci
npm run dev
# Before packaging: npm run build
```

The Vite development server proxies `/api` to the Python server. The production build is served from the Python package.

`data/` is a local input directory and is ignored by Git. Supply your own PDFs there, or pass an individual PDF path to `index`. The default index is `.paperanchor/library.sqlite3`; use `--db PATH` before the subcommand to choose another location.

For generated answers, create a local configuration file and fill in the key for the selected provider:

```powershell
Copy-Item .env.example .env
# Edit .env and fill DEEPSEEK_API_KEY for the default provider.
.\.venv\Scripts\paperanchor.exe ask "What problem does retrieval-augmented generation address?"
```

To change providers, set `PAPERANCHOR_PROVIDER` in `.env` and fill its matching key. `PAPERANCHOR_MODEL_TIER=fast` is the default; set it to `pro` for the higher-capability preset. `PAPERANCHOR_MODEL_ID` optionally selects an exact model ID. Process environment variables take precedence over `.env`.

`SEMANTIC_SCHOLAR_API_KEY` is optional. `PAPERANCHOR_RERANK_MODEL=BAAI/bge-reranker-base` enables local cross-encoder reranking; this downloads roughly 1 GB of weights on first use and increases query latency. It is off by default until you evaluate its value on your own papers.

| Provider | Key variable | `fast` | `pro` |
| --- | --- | --- | --- |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-flash` | `deepseek-v4-pro` |
| GLM | `ZHIPU_API_KEY` | `glm-5.3-flash` | `glm-5.3` |
| Qwen | `DASHSCOPE_API_KEY` | `qwen3.8-flash` | `qwen3.8-max` |
| Kimi | `MOONSHOT_API_KEY` | `kimi-k2.6` | `kimi-k3` |
| OpenAI | `OPENAI_API_KEY` | `gpt-6-luna` | `gpt-6-sol` |
| Gemini | `GEMINI_API_KEY` | `gemini-3.8-flash` | `gemini-3.1-pro-preview` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-haiku-4-5` | `claude-opus-5-5` |

These presets call each provider directly. Qwen uses the China (Beijing) endpoint, so its key must belong to that region. DeepSeek currently routes `deepseek-v4-pro` to Flash pending its next Pro release; selecting `pro` does not currently provide a distinct Pro model. Provider model IDs and availability change, so check the linked official docs before relying on a preset long term.

### Local keys and GitHub

`.env.example` contains empty key fields and is committed. Your filled `.env` stays on your computer because `.gitignore` excludes it. GitHub does not read your local environment variables or `.env` when you push code. CI runs tests with model stubs and needs no provider key. If a key is ever committed or pasted into a public issue or log, revoke it and create a new one; adding `.gitignore` afterward does not erase Git history.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m pytest
```

Tests cover PDF indexing, repeated and changed-file indexing, retrieval, cited answers, missing or invalid evidence, and the web flow from upload through source location and notes. They use generated PDFs and a model stub; a live model call requires provider credentials.

The [evidence retrieval benchmark](evals/README.md) pins six public PDFs by checksum and contains source-located questions, including Chinese queries, multi-paper evidence, and no-answer cases. It reports complete evidence-group recall separately for BM25, semantic search, and hybrid retrieval. The local corpus and detailed result files remain outside Git.

The frontend also has `npm run lint` and `npm run build` checks. CI rebuilds the committed frontend bundle and checks that the wheel contains it. See [Evaluation](docs/evaluation.md) for measured results and their limits. A retrieval comparison should use a source-labeled question/evidence set from the papers you actually research, rather than assuming a model name guarantees quality.

## Current scope

The PDF parser reads selectable text. It does not interpret scanned pages, equations, figures, or tables as a multimodal model would. A citation highlights a source block, which can contain several sentences; citation validation checks the ID, not factual entailment. The six-paper benchmark measures evidence coverage in that narrow corpus; it does not establish general answer accuracy. Local vector search scans stored vectors exactly and is intended for a small personal collection. The workspace has no login or multi-user isolation. Online discovery returns source-ranked candidates and does not claim to judge paper quality automatically.

## References

- [PyMuPDF text extraction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html)
- [SQLite FTS5 and BM25](https://www.sqlite.org/fts5.html)
- [arXiv API](https://info.arxiv.org/help/api/user-manual.html)
- [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/api-docs/)
- [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
- [DeepSeek models](https://api-docs.deepseek.com/quick_start/pricing/)
- [GLM model overview](https://docs.bigmodel.cn/cn/guide/start/model-overview)
- [Qwen model list](https://help.aliyun.com/zh/model-studio/text-generation-model)
- [Kimi API overview](https://platform.kimi.com/docs/api/overview)
- [OpenAI models](https://developers.openai.com/api/docs/models)
- [Gemini models](https://ai.google.dev/gemini-api/docs/models)
- [Claude models](https://platform.claude.com/docs/en/models/overview)
