# Evidence retrieval benchmark

This is a small, **source-anchored local benchmark**, not a claim of parity with
PaperQA2/LitQA2 or a general scientific-QA benchmark. It tests whether the
retriever returns the PDF passages needed to answer realistic questions about
six related research papers. The questions were authored from the source PDFs,
independently of the retriever's answers.

## Contents

- `corpus.json` pins the six public arXiv PDF versions by filename and SHA-256.
  The PDFs themselves stay in `data/` and are not committed.
- `retrieval_cases.jsonl` has 71 questions: 60 answerable and 11 deliberately
  unanswerable in this corpus. Questions include English and Chinese, methods,
  numbers, ablations, false premises, and evidence from more than one paper.
- `retrieval_benchmark.py` validates the corpus and gold evidence, then compares
  BM25, semantic search, and rank fusion without calling a paid LLM API.
  `--mode rerank` separately benchmarks the optional BGE cross-encoder; it may
  download large weights and takes longer, so it is not part of default `all`.

An answerable case stores one or more `support_sets`. Each set contains every
source location needed to support the reference answer; different sets are
alternative ways to support it. A location is a PDF filename, one-based page,
and short text anchor. The runner resolves it to the current passage ID. This
keeps the labels independent of database IDs and makes parser changes visible.
An absent or ambiguous anchor stops the run; it is not counted as a retrieval
miss. The current cases were built from extractable PDF text, so this version
does **not** measure OCR, figure understanding, or initial parser coverage.

## Run on the exact corpus

Place the six PDF versions named in `corpus.json` in `data/`. Their hashes must
match; a different arXiv revision is a different benchmark corpus. Index only
these six files into a dedicated database, or use the default local library if
it contains exactly these files:

```powershell
.\.venv\Scripts\paperanchor.exe --db .study\benchmark.sqlite3 index data
.\.venv\Scripts\python.exe evals\retrieval_benchmark.py --db .study\benchmark.sqlite3 --mode bm25 --validate-only
```

For semantic and hybrid comparisons, build the semantic index with the same
embedding model as the application and wait for all passages to be indexed.
The easiest path is to start the app on this database, trigger **Build semantic
index**, and wait for the counter to match the passage count. Then:

```powershell
.\.venv\Scripts\python.exe evals\retrieval_benchmark.py --db .study\benchmark.sqlite3 --split dev --json-report .study\retrieval-dev.json
.\.venv\Scripts\python.exe evals\retrieval_benchmark.py --db .study\benchmark.sqlite3 --split test --json-report .study\retrieval-test.json
```

`--split dev` is for making design decisions. Freeze the code and labels before
reporting `test`; do not repeatedly tune on test failures. The detailed JSON
contains every question's ranked paper/page locations and latency so errors can
be audited. For a lexical-only run, use `--mode bm25` while embeddings are
unavailable. The default `--ks` are 1, 5, and 10.

Each `k` is a separate retrieval call. Hybrid retrieval expands its source
candidate pool according to the requested limit, so requesting top 10 and
truncating the result to five does **not** measure the product's top-five path.
Older reports made with that shortcut must be rerun.

The first baseline was used to audit some alternative gold evidence in both
splits. Therefore this `test` split is useful for regression tracking but is
not a fully blind holdout. A future external capability comparison needs a
new, independently reviewed holdout that has not been inspected during design.

## Interpret the numbers

- **Complete-support recall@k**: top k contains every passage in at least one
  valid support set. This is the primary metric. A two-passage question does
  not pass merely because one of its passages was returned.
- **Any-support recall@k**: at least one gold passage appears. This diagnoses
  partial evidence but overstates performance on multi-evidence questions.
- **MRR**: reciprocal rank of the first gold passage within the largest k.
- **By-language and by-category breakdowns** prevent an average from hiding
  cross-language or multi-paper failures. Very small categories are diagnostic,
  not stable estimates.
- The 11 **unanswerable** questions are listed but excluded from retrieval
  recall. A retriever always returns something; refusal and false-answer rates
  require a separate end-to-end answer evaluation. Their absence labels remain
  provisional until another reviewer checks the full six-paper corpus.

The answerable cases have been checked against the source passages during
construction, but this first release has no independent second annotation and
uses a narrow, related six-paper corpus. Some
questions may have additional valid supporting passages. Review the PDF page
and add alternative gold support sets when a retrieved passage genuinely
supports the entire answer. Record label changes before interpreting score
changes. The reference answer is a human-readable rubric, not an exact-string
target or an automated grading claim.

This benchmark isolates the **retrieval** stage. It does not score the agent's
evidence-sufficiency decision, generated answer correctness, or citation
faithfulness. Those should be measured separately on the same frozen corpus.

## End-to-end answer evaluation

`answer_benchmark.py` runs the same retrieval composition and `research_question`
workflow as the application, with the model configured in local `.env`. It
records each retrieval query and ranked passage, each model prompt and response,
the final answer, cited PDF locations, and per-stage timing. It disables
model-only answers so unsupported questions test the paper-evidence boundary.
The report includes complete gold-evidence coverage at first retrieval, in
the final evidence shown to the model, and among cited passages. These are
**traceability proxies**, not answer accuracy or citation entailment scores.

```powershell
.\.venv\Scripts\python.exe evals\answer_benchmark.py --split dev --validate-only --json-report .study\answer-validity.json
.\.venv\Scripts\python.exe evals\answer_benchmark.py --split dev --json-report .study\answer-dev.json --review-template .study\answer-dev-reviews.jsonl
```

Use `--case-id T01` for a single real request before a full run. Live evaluation
calls the configured provider and can incur charges. Reports and review sheets
contain paper excerpts and model responses; keep them under ignored `.study/`.
The review template is created only if it does not already exist, preserving
earlier human judgments.

Review each case against the cited PDF and fill these JSONL fields with JSON
`true`, `false`, or `null` (unreviewed):

- `factuality`: the response answers the question accurately without unsupported
  factual claims. Mark `null` for an abstention rather than calling it factual.
- `citation_support`: every substantive claim attributed to papers is supported
  by its cited passage. A valid `[E1]` label alone is insufficient.
- `abstention`: the decision to withhold a paper-grounded answer was appropriate.
  Mark `null` for a substantive answer.
- `absence_label`: for provisional no-answer cases only, set `confirmed_absent`,
  `has_answer`, or `uncertain` after checking the corpus. Leave `null` until
  reviewed. The first pass in this project is not an independent second review.
- `notes`: briefly record the disputed claim or PDF location. Do not change
  `id` or `answer_sha256`; the hash prevents grading a different run's answer.

Score completed reviews without calling the model again:

```powershell
.\.venv\Scripts\python.exe evals\review_answers.py --report .study\answer-dev.json --reviews .study\answer-dev-reviews.jsonl --output .study\answer-dev-reviewed.json
```

If source review adds a valid alternative passage to `support_sets`, rescore the
saved model trace instead of paying for another, nondeterministic model run:

```powershell
.\.venv\Scripts\python.exe evals\answer_benchmark.py --split dev --rescore-from .study\answer-dev.json --json-report .study\answer-dev-rescored.json
```

The runner checks that the saved questions and answerability labels still match.
The new report records the current case-file hash and the path to the original
run. A reviewer should use the rescored report with its matching answer review
template; the answer fingerprint stays the same when only gold labels change.

The denominators are only explicitly reviewed judgments. A provisional
no-answer question is never counted as a confirmed hallucination merely because
the system answered it. Keep dev for diagnosis; this test split has already
been inspected during retrieval work, so a later external claim needs a new
independently reviewed holdout.

To combine separately saved dev and test runs without additional model calls:

```powershell
.\.venv\Scripts\python.exe evals\summarize_answers.py --reports .study\answer-dev-rescored.json .study\answer-test.json --output .study\answer-summary.json
```

The combiner rejects mixed case-file versions, model configurations, or
duplicate question IDs. A paper-style answer without complete gold evidence is
reported separately from a refusal caused by missing retrieval evidence; it
may indicate a partial answer, an overconfident sufficiency decision, or a
missing alternative gold passage and needs source review.

New answer reports also record a hash of the executed workflow source files,
including uncommitted edits; the combiner rejects differing workflow hashes.
Older reports have no such fingerprint and should not be treated as proof that
two runs executed identical code. `top_k` is the limit for each retrieval call;
after a retry, the workflow may retain up to twice that many distinct passages.
Dynamic decomposition instead permits up to three subqueries and retains up to
four times `top_k` passages. Reports include per-subquestion assessments and
candidate passage IDs, plus total retrieval/model calls and decomposed-case
counts. Compare latency and cost alongside fixed-label citation coverage.
It is disabled by default; pass `--decompose` to evaluate the experimental path.
The report combiner rejects runs with differing decomposition settings.

For a readable review queue with each answer and its cited PDF pages:

```powershell
.\.venv\Scripts\python.exe evals\render_review_packet.py --reports .study\answer-dev-rescored.json .study\answer-test.json --output .study\answer-review-packet.md
```

Use the packet to inspect the PDFs, then enter judgments in the JSONL review
sheets. The packet itself contains no grades and is not a replacement for the
source PDFs.
