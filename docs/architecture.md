# Architecture

PaperAnchor separates PDF extraction, storage and retrieval, answer construction, model access, and presentation. Each layer has a small data contract so the retrieval strategy or model provider can change without rewriting PDF parsing.

| Module | Responsibility |
| --- | --- |
| `pdf.py` | Extract text blocks with one-based PDF page numbers and bounding boxes. |
| `library.py` | Persist PDF records and passages; maintain the FTS5 index; return ranked hits. |
| `rag.py` | Turn hits into numbered evidence, request an answer, and validate cited IDs. |
| `model_config.py` | Resolve provider presets and the selected local credential. |
| `llm.py` | Call an official provider through Chat Completions or native Claude Messages. |
| `cli.py` | Expose indexing, retrieval inspection, and answering as commands. |
| `web.py` | Expose local upload, paper, question, page image, and annotation endpoints. |
| `static/` | Present the library, answer, evidence, PDF page, highlight, and notes. |

## Data flow and invariants

1. `index_pdf` identifies a PDF by its resolved file path and SHA-256 content hash. The same path with the same content is left unchanged. Changed content replaces its old passages and search entries in one transaction; annotations attached to the old PDF layout are cleared.
2. Each passage records text, page number, block number, and bounding box. Page numbers are one-based for users; rectangles use PDF page coordinates. The reader divides those coordinates by the page dimensions to place a highlight over the rendered page.
3. `search` ranks matching passages with SQLite FTS5 BM25 and returns their source locations. The current query parser extracts English alphanumeric terms and quotes them before constructing an FTS expression.
4. `answer_question` labels hits `E1`, `E2`, etc. The generator receives only these passages as evidence. The returned answer is marked `answered`, `uncited`, or `invalid_citation` according to its evidence IDs. When retrieval finds nothing, the generator is not called.
5. The web API returns each evidence item's paper ID, page, and rectangle. The browser links a citation to that exact location. Notes are stored separately with an author value of `user` or `agent`.

The current database stores one PDF file per paper record. A later implementation of paper versions or alternate files will require separate paper and file identities. The model generator is injected into the answer workflow, allowing deterministic tests without network access. Provider selection and credential lookup remain behind that generator interface.

## Known boundaries

An evidence ID proves that a cited passage was retrieved, not that the answer's claim follows from it. PDF text extraction does not interpret figures, formulas, or scans. The synchronous indexer is suited to small local collections. The local web API has no user authentication and is bound to `127.0.0.1` by default.
