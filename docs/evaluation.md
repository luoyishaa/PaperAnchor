# Evaluation

PaperAnchor evaluates retrieval on **120 QASPER questions against a 400-PDF library**, using a fixed **4,096-reference-token context budget**. The main metric is whether the final context fully covers an annotated evidence set for each answerable question.

## Results

| Profile | Complete evidence coverage | Mean latency | P50 | P95 |
| --- | ---: | ---: | ---: | ---: |
| Fast (MiniLM) | 79/96 · 82.29% | 2.19 s | 2.11 s | 3.22 s |
| Quality (BGE) | 80/96 · 83.33% | 8.30 s | 7.86 s | 11.86 s |
| Optimized (R2) | **87/96 · 90.625%** | 32.15 s | 31.20 s | 38.70 s |

All profiles ran all 120 questions. Coverage is calculated over **96 answerable questions**; the other **24 unanswerable questions** are auxiliary runs and do not enter that denominator. R2 achieved **91.41% macro evidence recall** and used a mean of **4,049.8 reference tokens**. See the [machine-readable results](../evals/results/qasper120.json) for per-question outcomes, configuration, and fingerprints.

Latency covers retrieval, reranking, and context assembly after model/index warm-up, with result caches disabled. It excludes startup and answer generation. MiniLM and BGE timings are from 2026-10-07; R2 was measured on 2026-10-08. These are local measurements from separate runs, not a controlled hardware comparison. R2 is slower in the recorded runs, so the browser defaults to MiniLM.

R2 combines verified title routing, equal-weight RRF of MiniLM and BGE candidate rankings, MiniLM expansion ranking, and four protected direct hits. It covers eight more questions than MiniLM without losing any of MiniLM's successful cases; against BGE it gains ten and loses three. The shipped configuration is [`evidence_optimized.json`](../src/paperanchor/evidence_optimized.json).

## Dataset and protocol

- **Source:** QASPER v0.3, dataset revision `fdc9d8214fbab5dd782958601db4d678e6934a54`, seed `20261005`.
- **Corpus:** 200 train, 80 validation, and 120 test papers; 86,224 extracted blocks and 44,199 reading units. Both block and unit embeddings are 384-dimensional. Train papers serve as distractors.
- **Questions:** 40 selected from validation and 80 from test, totaling 96 answerable and 24 unanswerable questions. All 120 are used together for the reported configuration comparisons.
- **Inputs:** the original English question and public paper title. Main retrieval receives no gold paper ID. Title hints guide paper selection within the library. Auxiliary unanswerable questions are scoped to their source paper.
- **Retrieval:** BM25 and dense search each return 40 candidates; reranking selects 20 hits before parent/neighbor expansion. All profiles use the same frozen corpus, questions, scoring rules, and context budget.
- **Budget:** the cached multilingual MiniLM tokenizer counts evidence IDs, titles, pages, and body text with truncation disabled. The 4,096 tokens exclude the question and system prompt and are not the answer model's native token count.

PDF download, version, parsing, and exact evidence alignment must succeed before a question is admitted. Alignment failures and exclusion reasons are recorded. Retrieval inputs and gold annotations are stored separately.

## What the metrics mean

**Complete evidence coverage** requires the actual returned text to cover every evidence item in at least one annotated support set. Multiple annotations are alternative support sets, not one combined requirement.

**Evidence recall** is the proportion of fully covered evidence items in the best-matching support set, averaged across questions. A partially covered item does not count as fully recovered. Multiple returned slices may jointly cover an item; duplicate text does not increase its score.

The scorer matches normalized, non-whitespace character intervals within the same paper. A matching source ID or page is insufficient. Context metrics use only the final text after budget trimming. Separate top-k metrics describe the ranked hits before expansion and should not be substituted for context coverage.

## Reproduction and artifacts

The [evaluation harness guide](../evals/README.md) documents preparation, alignment, retrieval, and scoring commands. Core entry points are:

| Module | Purpose |
| --- | --- |
| `evals.qasper_benchmark.prepare` | Download and validate the selected corpus |
| `evals.qasper_benchmark.build` | Build an isolated library and embedding indexes |
| `evals.qasper_benchmark.cases` | Freeze questions and aligned gold evidence |
| `evals.qasper_benchmark.run` | Evaluate a fixed MiniLM, BGE, or Optimized product profile on all 120 questions |
| `evals.qasper_benchmark.retrieve` | Save candidates, ranked hits, and final contexts |
| `evals.qasper_benchmark.report` | Score saved text against gold annotations |

The fixed-profile runner warms the selected profile, disables query/score result caches, and saves each returned context and latency. Historical experiment traces additionally retain actual queries, model scores, ranks, and source slices. Use a fresh output directory for a new run; do not combine records with different input or configuration fingerprints. Local corpus, model caches, and full run traces are generated artifacts rather than repository contents.

R2 was rerun on all 120 questions with result caches disabled. Candidate ordering, model ranks, queries, final context text, source references, and token counts matched the saved result for every question. Product-level checks also matched ten representative cases across the three profiles.

## Scope and limitations

- **This is evidence retrieval coverage, not 90.625% answer accuracy.** No generated-answer correctness, citation faithfulness, or refusal-accuracy result is reported. The metric is project-defined exact coverage, not the official QASPER evidence-selection F1.
- Title hints and selection for exact PDF alignment make this a specific retrieval task. It does not measure unknown-paper discovery, cross-paper synthesis, general PDF parsing quality, or reading chart values.
- Annotated evidence is not exhaustive: another passage may answer the question without matching the selected annotation. Conversely, finding all annotated text does not prove that a language model will interpret it correctly.
- The remaining nine coverage failures comprise eight cases with incomplete initial candidate evidence and one with evidence present in the pool but missing from the top 20. This diagnosis concerns the fixed corpus and candidate configuration.

Implementation details are in [Architecture](architecture.md); scoring rules are implemented in [`score.py`](../evals/qasper_benchmark/score.py).
