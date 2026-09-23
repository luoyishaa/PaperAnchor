# Architecture

PaperAnchor separates PDF extraction, storage and retrieval, answer construction, model access, and the command-line interface. Each layer has a small data contract so the retrieval strategy or model provider can change without rewriting PDF parsing.

| Module | Responsibility |
| --- | --- |
| `pdf.py` | Extract text blocks with one-based PDF page numbers and bounding boxes. |
| `library.py` | Persist PDF records and passages; maintain the FTS5 index; return ranked hits. |
| `rag.py` | Turn hits into numbered evidence, request an answer, and validate cited IDs. |
| `llm.py` | Call a configured Chat Completions-compatible model. |
| `cli.py` | Expose indexing, retrieval inspection, and answering as commands. |

## Data flow and invariants

1. `index_pdf` identifies a PDF by its resolved file path and SHA-256 content hash. The same path with the same content is left unchanged. Changed content replaces its old passages and search entries in one transaction.
2. Each passage records text, page number, block number, and bounding box. Page numbers are one-based for users; rectangles use PDF page coordinates.
3. `search` ranks matching passages with SQLite FTS5 BM25 and returns their source locations. The current query parser extracts English alphanumeric terms and quotes them before constructing an FTS expression.
4. `answer_question` labels hits `E1`, `E2`, etc. The generator receives only these passages as evidence. The returned answer is marked `answered`, `uncited`, or `invalid_citation` according to its evidence IDs. When retrieval finds nothing, the generator is not called.

The current database stores one PDF file per paper record. A later implementation of paper versions or alternate files will require separate paper and file identities. The model client is injected into the answer workflow, allowing deterministic tests without network access.

## Known boundaries

An evidence ID proves that a cited passage was retrieved, not that the answer's claim follows from it. PDF text extraction does not interpret figures, formulas, or scans. The CLI prints page-level evidence; the stored bounding boxes are available for a future PDF viewer. The synchronous indexer is suited to small local collections.
