# Architecture

PaperAnchor keeps paper identity, passages, source locations, research spaces, and job progress in one SQLite database. A PDF is the source artifact. Retrieval and model calls consume located passages; they never define where a citation points.

## Module boundaries

| Module | Owns |
| --- | --- |
| `pdf.py` | Text extraction, page numbers, and source rectangles. |
| `library.py` | Paper and passage transactions, FTS5 search, annotations. |
| `spaces.py` | Space membership; one paper can belong to several spaces. |
| `semantic.py` | Embeddings, exact vector search, rank fusion, optional reranking. |
| `jobs.py` | Durable semantic-index job states and a single local worker. |
| `discovery.py` | arXiv and Semantic Scholar adapters, normalized candidates, imported metadata. |
| `rag.py` | Evidence IDs and citation validation. |
| `agent.py` | Bounded search, evidence assessment, and answer-basis decision. |
| `model_config.py` and `llm.py` | Provider settings and model transport. |
| `web.py` and `cli.py` | HTTP and command-line entry points. |
| `frontend/` | React and TypeScript research workspace and PDF page viewer. |
| `evals/` | Source-anchored retrieval labels, end-to-end traces, deterministic scoring, and human-review exports. |

The application workflow depends on a small passage-retriever interface and a model-generation function. Tests supply in-memory fakes at these boundaries. Paper discovery is separate from local indexing: source failures do not make imported papers unavailable.

## Source and retrieval contracts

1. The library identifies an imported PDF by its resolved path and SHA-256 content hash. Reindexing unchanged bytes is a no-op. Replacing changed bytes updates passages and full-text entries in one transaction; old annotations and vectors attached to replaced passages are removed.
2. A passage retains the paper ID, one-based PDF page, text, and rectangle in PDF coordinates. The browser maps the rectangle onto the rendered page. The stable passage ID is shared by FTS, vectors, answer evidence, and notes.
3. SQLite FTS5 BM25 and normalized-vector cosine search independently return ranked passages. Reciprocal rank fusion combines positions, avoiding direct addition of incompatible scores. The optional cross-encoder sees the bounded, deduplicated union of both source lists before fusion can discard one-channel candidates.
4. Vector rows include the embedding model ID and dimension. Missing rows can be built incrementally; a failed job retains completed batches and can resume. The current exact vector scan is chosen for a small local library, not a large shared corpus.
5. Research spaces filter both retrieval paths before fusion. The default **All papers** space contains every imported paper; deleting a custom space does not delete its PDFs.

## Evidence workflow

Dynamic decomposition is experimental and disabled by default after a focused
regression showed lower fixed-label coverage and higher latency. API callers can
explicitly set `enable_decomposition: true`; benchmark runs use `--decompose`.
The default retains the single-rewrite workflow and its original assessment prompt.

The question workflow first retrieves passages and checks support for every requested part. Supported questions proceed directly to generation. Insufficient evidence can trigger either one revised query or a model-proposed plan of at most three independently verifiable subquestions. There are no predefined question categories. Each subquestion receives its own search results plus the original evidence and a separate sufficiency check. Duplicate queries reuse their results. Nested plans are never executed; malformed or oversized plans fail closed rather than being silently truncated.

The workflow also accepts an optional recovery retriever through the same passage-search interface. When the first evidence check fails, it searches the original question with that retriever, retains original passage IDs and PDF coordinates, and checks the combined evidence once more. If evidence remains insufficient, one revised query can still run. This opt-in path uses at most three searches and retains at most three times the per-search passage limit; it skips a redundant assessment if recovery finds no new passages. If the initial search finds nothing, recovery can run before returning no evidence. The app does not currently configure this experimental option, and decomposition takes precedence when enabled. The ordinary app path is unchanged.

The workflow retains the deduplicated union of passages, preserving PDF coordinates. Every subquestion must pass before a final check against the original question; this catches omitted qualifiers in an otherwise successful plan. At the default five passages per search, the rewrite path retains at most ten passages and the decomposition path at most twenty. Decomposition uses at most four searches and six model calls, including the final answer (model transport retries are separate). A `SubquestionResult` records the question, query, candidate passage IDs, sufficiency decision, and gap. Candidate IDs describe what was assessed, not verified claim-level support. These records are returned by the API and exported in evaluation traces.

A sufficient set is numbered `E1`, `E2`, and so on. The generated answer is checked for references to IDs that were actually supplied. Weak evidence yields an explicit gap and an optional follow-up question. A model-only answer requires caller opt-in and is labeled separately with no paper citations. The model still controls semantic sufficiency judgments; execution budgets and evidence retention are enforced in code.

An evidence ID proves which passage was supplied; it does not prove that every sentence in a generated answer follows from that passage. The reader opens the stored PDF location so the user can check it.

## Evaluation boundary

The retrieval benchmark anchors each gold evidence set to a PDF filename, page, and short source text. The end-to-end runner calls the same retriever composition and question workflow as the app, recording searches, model assessments, answers, citations, and timing. Its deterministic scorer checks whether a complete gold set entered the initial retrieval, final evidence, and cited evidence. These are coverage measurements, not answer accuracy or entailment judgments. A separate review sheet records factuality, per-claim citation support, and the appropriateness of abstention. Saved traces can be rescored after a source-label correction without repeating model calls; the report records the label-file hash, and human reviews are bound to the exact answer and evidence by a fingerprint. Local reports contain paper excerpts and stay under ignored `.study/`.

## External discovery and time

Both search adapters return a common candidate containing title, authors, date or year, venue, abstract, external identifiers, and a source URL when available. Search order comes from each source. The user reviews candidates before import. For a direct import, the server accepts a validated arXiv ID, fetches arXiv metadata and PDF, and records the publication date. Semantic Scholar candidates without an arXiv ID remain links for manual PDF upload. For an existing local file named with a modern arXiv ID, the timeline infers its submission year from the ID and labels that provenance. Other local PDFs appear as **Undated**.

## Execution and limits

FastAPI serves the compiled frontend and runs one background semantic worker by default. SQLite job rows preserve queued, running, succeeded, and failed state; a restarted worker recovers an interrupted running job. The server is for one trusted local user and should run with one process. The PDF parser is text-only; equation, table, figure, scan, and claim-level verification require additional work. Schema changes beyond the current additive tables will need explicit migrations.
