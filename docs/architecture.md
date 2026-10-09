# Architecture

PaperAnchor is a local paper library with a FastAPI backend and a React/TypeScript reader. SQLite stores paper metadata, located text, retrieval indexes, research spaces, annotations, conversations, and background job state. PDFs remain the source of every citation.

## From PDF to searchable evidence

1. **Import.** `pdf.py` extracts text blocks with one-based page numbers and PDF rectangles. `library.py` identifies files by path and SHA-256, skips unchanged PDFs, and replaces changed content transactionally.
2. **Represent.** `representation.py` builds reading units of up to 128 model tokens, joining or splitting text while retaining links to the original source blocks. `unit_index.py` tracks the PDF hash, representation version, and tokenizer identity so stale units can be rebuilt.
3. **Index.** SQLite FTS5 provides BM25 search. `semantic.py` embeds units with `paraphrase-multilingual-MiniLM-L12-v2` and searches normalized vectors by cosine similarity. Vector rows retain their model identity and dimension. The current implementation uses an exact vector scan for a small local library.
4. **Locate.** Retrieval results retain source block IDs, pages, and rectangles. Rebuilding the search representation does not change the coordinate system used by PDF annotations.

A research space is a collection of papers that constrains both lexical and vector search. A paper may belong to multiple spaces; removing a space does not delete its PDFs. Indexing runs in a local background worker with persistent job state and incremental embedding batches.

## Retrieval profiles

The expanded evidence profiles share `evidence_retrieval.py` and `evidence_context.py` with the evaluation harness:

- A title hint selects the closest catalog title by token-set overlap. Explicit paper selections take precedence; without a matching title, retrieval falls back to the selected research space.
- The question retrieves 40 BM25 and 40 dense candidates. Their deduplicated union is reranked, and the first 20 hits enter context assembly.
- The assembler first reserves space for protected direct hits. It then ranks the remaining hits, missing text from their source blocks, and eligible same-page neighbors together.
- Overlapping source intervals are removed before adding text. Evidence headers and body text share a 4,096-token reference budget. Each emitted slice keeps its own evidence ID and source location.

Candidate ranking and expansion ranking are independent policies. Weighted reciprocal rank fusion (RRF) combines model positions as `sum(weight / (60 + rank))`; it does not average incompatible raw model scores.

| Web profile | API value | Candidate / expansion ranking | Protected hits |
| --- | --- | --- | ---: |
| Fast (MiniLM) | `evidence_minilm` | MiniLM / MiniLM | 8 |
| Quality (BGE) | `evidence_v2` | BGE / BGE | 8 |
| Optimized | `evidence_optimized` | Equal-weight MiniLM + BGE RRF / MiniLM | 4 |
| Standard | `legacy` | Configured block or unit retriever | — |

The rerankers are `Xenova/ms-marco-MiniLM-L-6-v2` and `BAAI/bge-reranker-base`. The optimized profile also uses verified title aliases from [`evidence_optimized.json`](../src/paperanchor/evidence_optimized.json). Each alias is bound to a paper ID and PDF hash; this packaged profile is available only when the local library matches those bindings. Aliases affect routing, not indexed text or citation headers.

`evidence_profile.py` keeps models and the corpus index resident, serializes requests, and releases the SQLite read snapshot after assembly. Content revisions invalidate the index after paper, passage, unit, or vector changes. Annotation and conversation updates do not rebuild it.

The browser defaults to Fast and persists the user's selection. API requests that omit `retrieval_profile` retain Standard behavior. Expanded profiles require cached local models; choose Standard when those models are not installed. Expanded profiles cannot be combined with experimental question decomposition.

After semantic indexing has installed the embedding model and reference tokenizer, cache the Fast reranker from the repository root (requires a download):

```sh
python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='.paperanchor/models')"
```

For Quality or Optimized, also cache `BAAI/bge-reranker-base` using the same command with that model name. With a custom database, use a `models` directory beside the database instead. Optimized still requires the matching corpus bindings described above.

## Answers and the PDF reader

`rag.py` assigns evidence IDs such as `E1`, sends the assembled context to the selected language model, and checks returned citation IDs against the supplied evidence. A citation links an answer to that evidence; the reader opens the stored PDF page and highlights its source region. Valid IDs do not independently verify every generated claim.

`agent.py` implements the direct answer path, collection overviews, follow-up rewriting, and an optional bounded search-recovery mode. A follow-up can use the last three exchanges to resolve references before retrieval. Previous answers are conversation context, not additions to the paper index. `conversations.py` stores the original question, rewritten question, and answer snapshot per research space.

Figure and table requests use a separate visual path. `visual_index.py` associates captions with embedded images, vector graphics, or table-text rows. `vision.py` selects at most two sources and sends page context plus region crops to a vision model; a user-selected rectangle supplies its crop directly. Source hashes and detector versions invalidate stale regions. Page rendering and highlights use the same unrotated PDF coordinates.

## Module map

| Modules | Responsibility |
| --- | --- |
| `pdf`, `library`, `database` | PDF extraction, source identity, storage, annotations |
| `representation`, `unit_index`, `semantic` | Reading units, lexical/vector indexes, standard retrieval |
| `evidence_retrieval`, `evidence_context`, `evidence_policies` | Candidate selection, model ranking, source deduplication, context budgets |
| `evidence_profile`, `evidence_revision` | Product profiles, model reuse, index invalidation |
| `rag`, `agent`, `conversations` | Cited answers, question workflows, conversation history |
| `visual_index`, `vision` | Located figure/table evidence and multimodal requests |
| `discovery`, `spaces`, `jobs` | External paper search, library organization, indexing jobs |
| `model_config`, `llm`, `ollama` | Model settings and remote/local model transports |
| `web`, `cli`, `frontend/` | HTTP API, command line, browser workspace |
| `evals/qasper_benchmark/` | Dataset preparation, evidence alignment, retrieval evaluation |

## Operating scope

Run one application process for one trusted local user. Paper discovery uses arXiv and Semantic Scholar; imported papers remain searchable when those services are unavailable. Text and vision models can be configured separately, including local Ollama models.

Text extraction does not OCR scanned pages. Visual region detection is based on captions and geometry rather than a learned layout model; page-level reading is the fallback when a region cannot be located. Retrieval evaluation measures evidence coverage, not answer correctness or visual understanding. See [Evaluation](evaluation.md) for the measured scope and results.
