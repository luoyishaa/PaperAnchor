# LitQA2 question-only evaluation

This evaluates the existing PaperAnchor text retrieval and answering pipeline on
independent scientific questions. Unlike the QASPER run, **no paper title, DOI,
source ID, gold passage, or per-question paper filter is supplied to retrieval**.

## Full-text recovery follow-up — 2026-10-09

Corpus acquisition now covers **192/199 questions with every annotated source
available**, and **194/199 with at least one annotated source available**. These
are availability counts, not new retrieval or answer scores. The original
`manifest.json`, evaluation database and 95-question results remain unchanged.
The expanded manifest is `data/litqa2/manifest-expanded.json`.

| Full-text source | Available source entries |
| --- | ---: |
| Original Europe PMC XML | 93 |
| Official PMC AWS XML | 30 |
| Public bioRxiv parsed full-text archive | 34 |
| PMC article HTML | 3 |
| Author/institution/publisher/public article PDFs | 24 |
| Public article PDF requiring OCR | 1 |
| Total | 185 |

The initial collector required a PMC identifier and used only Europe PMC's
`fullTextXML` endpoint. It stopped at 62 unmatched source entries and 36 HTTP 500
responses. These were limitations of that collector, not proof that papers were
unavailable. The official LAB-Bench LitQA2 directory contains questions, split
metadata and task code; it does not bundle the referenced full papers.

The largest recovery came from
[bethgelab/paper_parsed_jsons](https://huggingface.co/datasets/bethgelab/paper_parsed_jsons),
revision `86b3ec0effd80065738dabf44cc9b36f5355b9f5`. All 34 missing bioRxiv DOI
filenames were found inside its approximately 7 GB `bioarxiv.zip`. HTTP byte-range
requests downloaded the ZIP directory and required entries only; decompressed
lengths and CRC32 checks were verified. The files contain S2ORC PDF-extracted
body paragraphs, not generated summaries. DOI filenames and official bioRxiv
titles were cross-checked; three title differences were manually inspected.
The archive does not reliably identify exact preprint revisions. Its extraction
can omit text, figures or layout, so availability is not a guarantee of complete
evidence. Original JSON files and hashes are retained.

Other routes included [PMC AWS](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/),
accessible PMC article pages, author laboratory pages and institutional
repositories. PDFs are checked against the article DOI and title; Unicode
normalization handles typographic ligatures. Raw PDFs/HTML are preserved, and
extracted paragraphs are wrapped in article XML for adapter compatibility.
An eLife identity mapping is recorded explicitly: its publisher's version-of-
record DOI `10.7554/eLife.91357.3` corresponds to PMC's unversioned DOI, matching
publication date, title and version-specific links.

Further recovery used Caltech, Oxford, Basel and Manchester repositories,
author laboratory publication lists, and Elsevier's publicly downloadable
article-plus-supplement PDF bundles. The bundles contain the article itself,
not just supplementary figures. Their additional material is retained and must
be acknowledged when comparing a future expanded run against the original run.
Basel's public file API returned the article despite a broken authorization
check in its page rendering. Author sites with broken HTTPS also served their
public PDFs through ordinary HTTP; no authentication was bypassed.

The Loomba article (`10.1126/science.abo0924`) is a 22-page Science first-release
PDF from 23 June 2022, hosted publicly at
[Gwern's article archive](https://gwern.net/doc/psychology/neuroscience/2022-loomba.pdf).
Its body was image-based and was extracted using full-page Tesseract English OCR
at 200 DPI. Original PDF and page-level OCR JSON are retained with hashes; OCR
can introduce character errors. The Manchester enzyme-cascade paper is an
accepted author manuscript, not the final typeset version. These version and
extraction differences should be retained in future evaluation reporting.

There remain 7 questions without all sources, including two with one source
already available and five with none. The unresolved list contains 6 source URL
entries. Multi-source rows include preprint/published-version links and a
conference abstract/full-article pair; the strict all-source count has not been
silently changed to an any-source count. Different-DOI preprints have not been
substituted for journal articles. Publisher access failures, browser challenges,
metadata-only responses and unresolved repository downloads remain recorded.

The CsrA article (`10.1128/jb.00479-21`) was recovered from PMC article HTML.
The HTML collector now rejects cached challenge pages and retries the public
article endpoint rather than treating a challenge as a DOI mismatch.

Recovery scripts, discovery records, raw archive entries and validation are in
`.study/litqa2-recovery/`. `validate.py` verifies every available file hash, raw
source hash and article parser output; `audit.json` lists exact missing question
IDs and source URLs. The expanded corpus has **not yet been indexed or
reevaluated**. The scores below apply exclusively to the original 95-question
corpus.

## Retrieval results — original 95-question run, 2026-10-09

The 199 public questions reference 191 unique source DOIs. Full texts were
obtained for 93 sources, covering 95 questions. The remaining 104 questions have
missing corpus sources: 62 source DOIs had no matching PMC full text and 36
full-text endpoints returned HTTP 500. These are access outcomes, not proof that
the papers are unavailable everywhere. No questions were excluded for poor
retrieval performance.

The isolated corpus contains 9,957 text blocks and 18,427 reading units. All
18,427 units have 384-dimensional vectors. The current 95-question subset has one annotated source
per question, so it tests finding an unknown source rather than multi-paper synthesis.

| Retrieval metric | Result |
| --- | ---: |
| Annotated source in hybrid candidate pool | 95/95 |
| Annotated source in reranked top-20 units | 95/95 |
| Annotated source in final context | 95/95 |
| Complete normalized key passage in context, on exact-alignable annotations | 43/83 (51.81%) |
| Mean retrieval and context assembly time | 2.44 s |
| Median retrieval and context assembly time | 2.37 s |
| Maximum reference tokens | 4,096 |

Of the 95 questions, 94 have a key passage; 83 key passages match the downloaded
full text exactly after normalization. The other 12 cases remain in retrieval
and answering evaluation. The 43/83 metric requires the entire annotated excerpt
in emitted order; an answer may need only part of a long excerpt, and evidence
slices may be reordered. It is a conservative text diagnostic, not semantic
evidence recall or answer accuracy.

All 95 saved retrieval traces were checked: BM25, dense retrieval and reranking
queries equal the original question, and no per-question paper scope was supplied.
The 100% source hit rate applies only to this 93-paper closed corpus. It does not
estimate retrieval over a much larger collection or demonstrate complete evidence.

Local hardware: Intel Core i7-14650HX and NVIDIA GeForce RTX 5060 Laptop GPU
(8,151 MiB). Retrieval uses the cached local FastEmbed/ONNX models; the GPU is used
by the separate local answer model. Timings are single-run observations.

## Answer results — all 95 available-source questions

The saved retrieval contexts were passed to local Qwen3-8B through the product
answer generator. No retrieval settings, prompt or model settings were tuned
after observing these outcomes. The final selected letter is graded strictly.

| Answer metric | Result |
| --- | ---: |
| Correct | 77/95 — **81.05%** |
| Incorrect choice | 7/95 |
| Insufficient-information choice | 11/95 |
| Invalid output / request error | 0 / 0 |
| Non-abstention coverage | 84/95 — 88.42% |
| Precision among non-abstained answers | 77/84 — 91.67% |
| Mean answer generation time, excluding retrieval | 4.76 s |

Of the 18 incorrect or abstained answers, six already had the complete normalized
gold excerpt in the returned context. Improving retrieval alone would not address
those six outcomes. Manual inspection also found final-letter errors: for example,
case `100b570f-1c4f-4214-8402-f1b15f9989b4` states the correct 30% in prose but
selects C (25%), rather than F (30%). It remains incorrect under strict grading.
Case `487539f9-2f17-4009-aa4a-c41322445f11` receives the complete deletion-study
evidence but abstains. These observations are diagnosis, not post-hoc score repairs.

The product citation-ID check reports valid cited outputs for 93 responses and
no citation for two. This checks ID validity only; citation faithfulness was not
independently scored. Choice accuracy also does not establish faithfulness or
measure the incremental benefit of retrieval over a no-context model baseline.

An independent pass regraded all 95 saved answer letters against the original
choices and gold answers, checked every retrieval query and lack of paper filter,
and checked the context token budget. All passed. Adapter and relevant product
tests: 31 passed. Compact results and per-question metrics are in
[`evals/results/litqa2-question-only.json`](../evals/results/litqa2-question-only.json).
Full contexts, answers, source-availability audit and verification records remain
in `.study/litqa2-question-only-20261009/`.

The main finding is that the current pipeline can retrieve and answer without
injected paper identity on this accessible biomedical subset. The small,
source-derived corpus, availability-based selection, XML input and single-source
questions limit generalization. These results cannot be compared directly with
the QASPER 90.63% evidence-coverage score or an official LitQA2 leaderboard.

## Dataset selection

| Dataset | Suitable task | Decision |
| --- | --- | --- |
| [LAB-Bench LitQA2](https://huggingface.co/datasets/futurehouse/lab-bench) | Scientific literature questions with answer choices, source DOIs and key passages | First local evaluation; all 199 public questions are downloaded |
| [ScholarQABench](https://github.com/AkariAsai/ScholarQABench) | Open-ended, multi-paper literature synthesis | Valuable complementary evaluation; needs a suitable broad corpus and rubric/citation evaluation |
| [PaperFindingBench](https://github.com/allenai/asta-bench/tree/main/astabench/evals/paper_finder) | Finding sets of papers from semantic and metadata requirements | Complementary paper-discovery task; official setup uses a much larger scholarly index |
| [LitTraceQA](https://huggingface.co/datasets/LitTraceQA/LitTraceQA) | Paper retrieval, evidence grounding, tables and figures | Public examples include many explicitly named papers; not selected as the primary no-title test |

The 100 ScholarQA-CS question metadata records and published rubric configurations
were also downloaded to `data/scholarqabench/`, pinned to upstream commit
`95e6fc52b0a8a0ce0a74956029991e3bb00c38b9`. They have not been evaluated; they are
the complementary candidate for multi-paper synthesis, not part of the LitQA2 score.

LitQA2's public split is named `train` on Hugging Face; here it is used only for
evaluation, with no parameter tuning. Question text is preserved verbatim. Some
questions naturally mention methods, authors or papers; the adapter adds no
source hints. This is a bounded local-corpus adaptation, not the original full
literature search benchmark and not directly comparable to published leaderboards.

## Full-text corpus and isolation

Source DOIs are resolved through Europe PMC and the available JATS XML full texts
are downloaded. All obtainable source papers share one corpus; every other paper
is a distractor for a given question. Questions whose annotated sources cannot
all be obtained are listed explicitly, not silently removed from the total.
Corpus membership depends on source availability, never on retrieval success.

The adapter imports article abstracts and full bodies, including table text and
figure captions, into an independent `.paperanchor/litqa2.sqlite3` database. It
does not index answer choices, questions, key passages or answer labels. It uses
the product's `UnitStore`, local embeddings, `EvidenceProfile`, and
`answer_from_context`. Original library contents and application settings remain
unchanged.

XML section ordinals occupy the internal page field, with zero rectangles.
These are **not PDF pages or valid visual coordinates**. This experiment measures
text retrieval/QA and does not evaluate PDF parsing, visual reasoning, or PDF
highlighting. XML structure also changes the expansion neighborhoods relative to
PDF extraction, so scores should not be compared directly with PDF-based QASPER.

## Fixed protocol

- Default product profile: Fast / `evidence_minilm`.
- BM25 and dense retrieval each supply 40 candidates; MiniLM reranks the union.
- The top 20 hits enter source deduplication, parent/neighbor expansion and the
  product's 4,096-reference-token context assembler.
- No title routing or research-space restriction; a runtime assertion checks this.
- Models/index warm up before timing. Query/score result caches are disabled.
- Every obtainable-source question is used for retrieval evaluation.
- All 95 obtainable-source questions are also answered. Choices are
  deterministically shuffled per question, including an insufficient-information
  option. Retrieval receives only the original question; answer generation
  receives that question and choices. The runner also supports a reproducible
  bounded sample via `--answer-limit`, sorted by `SHA256("20261009:" + question_id)`.
- Local Qwen3-8B answers through the existing Ollama adapter. The run explicitly
  records thinking mode, output budget and model context size. These process-local
  settings do not modify `.env`.
- No tuning on this evaluation set; preserve this first-run baseline.

## Metrics

1. **Candidate source hit:** at least one annotated paper appears in the hybrid
   candidate pool.
2. **Top-20 source hit:** at least one annotated paper remains after reranking.
3. **Context source hit / all sources:** final context contains text from at least
   one / all annotated papers. This does not establish that the text answers the question.
4. **Key-passage exact presence:** the normalized annotated passage appears in
   returned text from its source. This conservative string diagnostic is neither
   semantic evidence recall nor answer accuracy. An additional character-coverage
   diagnostic accounts for split/reordered evidence, but is not a correctness score.
5. **Answer accuracy, abstention and wrong choices:** exact grading of the selected
   option on the fixed sample. Invalid outputs and request errors remain visible.
   Correct options alone do not prove citation faithfulness.

Record original dataset revision, source hashes, corpus fingerprint, exclusions,
queries, ranked hits, final context, per-question latency and answer outputs.

## Reproduction

From the project root, using the existing development environment and model cache:

```powershell
.venv\Scripts\python.exe -m evals.litqa_benchmark prepare
.venv\Scripts\python.exe -m evals.litqa_benchmark build
.venv\Scripts\python.exe -m evals.litqa_benchmark run
```

Add `--answer --answer-limit 199` to run all available questions with the configured
provider. Raw source data and complete traces stay in ignored `data/litqa2/` and
`.study/litqa2-question-only-20261009/`. This avoids redistributing the benchmark
or third-party articles as project code. LAB-Bench declares CC BY-SA 4.0; article
reuse remains subject to each article's license.

The initial answer run uses process-local settings
`PAPERANCHOR_OLLAMA_THINKING=false`, `PAPERANCHOR_TEXT_MAX_TOKENS=768`, and
`PAPERANCHOR_OLLAMA_CONTEXT_TOKENS=8192`, with the configured `qwen3:8b` model.
Answer timing and quality apply to these settings, not the application's default
thinking/output budget. Sampling otherwise follows the installed model defaults.

The script resumes completed records, checks corpus/question/profile identities,
and rejects incompatible output directories. Run adapter tests with:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_litqa_benchmark.py
```
