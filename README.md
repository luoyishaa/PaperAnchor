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
- A supported provider API key for `ask`; `index` and `search` work offline

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

For generated answers, create a local configuration file and fill in the key for the selected provider:

```powershell
Copy-Item .env.example .env
# Edit .env and fill DEEPSEEK_API_KEY for the default provider.
.\.venv\Scripts\paperanchor.exe ask "What problem does retrieval-augmented generation address?"
```

To change providers, set `PAPERANCHOR_PROVIDER` in `.env` and fill its matching key. `PAPERANCHOR_MODEL_TIER=fast` is the default; set it to `pro` for the higher-capability preset. `PAPERANCHOR_MODEL_ID` optionally selects an exact model ID. Process environment variables take precedence over `.env`.

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
.\.venv\Scripts\python.exe -m pytest
```

Tests cover PDF indexing, repeated and changed-file indexing, retrieval, cited answers, and missing or invalid evidence. They use generated PDFs and a model stub; a live model call requires provider credentials.

## Current scope

This implementation uses English-term keyword retrieval and PDF text blocks. It does not yet provide semantic search, reranking, OCR, paper discovery, a PDF viewer, or multi-user research spaces. A passage reference locates a PDF page and block; it does not highlight an exact sentence. Citation validation checks evidence IDs, not whether every factual claim is actually supported. Bibliographies and short blocks can rank above explanatory text; improving retrieval quality requires a measured evaluation set.

## References

- [PyMuPDF text extraction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html)
- [SQLite FTS5 and BM25](https://www.sqlite.org/fts5.html)
- [DeepSeek models](https://api-docs.deepseek.com/quick_start/pricing/)
- [GLM model overview](https://docs.bigmodel.cn/cn/guide/start/model-overview)
- [Qwen model list](https://help.aliyun.com/zh/model-studio/text-generation-model)
- [Kimi API overview](https://platform.kimi.com/docs/api/overview)
- [OpenAI models](https://developers.openai.com/api/docs/models)
- [Gemini models](https://ai.google.dev/gemini-api/docs/models)
- [Claude models](https://platform.claude.com/docs/en/models/overview)
