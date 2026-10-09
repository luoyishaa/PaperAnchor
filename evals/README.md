# Evaluation

PaperAnchor evaluates retrieval on 120 QASPER questions over a 400-paper PDF
library. The question and paper title are supplied separately. Of the 120 cases,
96 have annotated evidence and 24 are unanswerable auxiliary cases. Every profile
uses a 4,096-token reference context budget.

See [results and metric definitions](../docs/evaluation.md) and the
[120-question results](results/qasper120.json).

## Evaluate a product profile

Run commands from the repository root with the development environment activated.
The prepared library, frozen cases, gold annotations, and local model cache must
be available. The model cache lives beside the database in `models/` and contains
the multilingual MiniLM embedding model, the reference tokenizer, and the selected
reranker: `BAAI/bge-reranker-base` or `Xenova/ms-marco-MiniLM-L-6-v2`. The optimized
profile requires both rerankers and its matching PDF title metadata.

```sh
python -m evals.qasper_benchmark.run --profile minilm --database .paperanchor/library.sqlite3 --cases .study/qasper-benchmark/frozen-20261005/cases/cases.json --gold .study/qasper-benchmark/frozen-20261005/cases/gold.json --output .study/qasper-benchmark/minilm-reproduction
```

Choose `minilm`, `bge`, or `optimized`; use a new output directory for each run.
The runner calls the same `EvidenceProfile` implementation as the application.
It verifies the frozen input fingerprints, runs all 120 cases, and writes:

- `protocol.json`: profile configuration, input identities, and model/source hashes.
- `q*.json`: each question's actual context, citations, token usage, and latency.
- `records.json` and `summary.json`: complete records and aggregate results.

Models warm up before timing. Query and score result caches are disabled; the
warmup is excluded. Retrieval failures remain in the denominator and make the
run incomplete. No answer-generation model is called.

## Prepare a library and question set

The preparation step downloads QASPER v0.3 annotations and source PDFs. QASPER is
licensed under [CC BY 4.0](https://allenai.org/data/qasper). The seed and corpus
quotas are fixed; training papers serve as retrieval distractors.

```sh
python -m evals.qasper_benchmark.prepare prepare --data-dir data/qasper --seed 20261005 --train 200 --validation 80 --test 120
python -m evals.qasper_benchmark.build --manifest data/qasper/manifest.json --database .paperanchor/qasper-staging.sqlite3 --cache-dir .paperanchor/models
python -m evals.qasper_benchmark.cases --manifest data/qasper/manifest.json --build .paperanchor/qasper-staging.build.json --database .paperanchor/qasper-staging.sqlite3 --output .study/qasper-benchmark/prepared/cases
```

Building requires the cached embedding model and creates both block and reading
unit indexes. It does not replace the application's library. Case preparation
aligns annotated evidence to extracted PDF text and writes `cases.json`,
`gold.json`, and `alignment.json`. Only complete inputs receive `frozen` status.
Point the evaluation command at this database and these case files when using a
newly prepared library. The installed optimized title aliases are bound to PDF
hashes and local paper IDs; a different index must provide matching metadata.

## Modules

| Module | Purpose |
| --- | --- |
| `run` | Evaluate the installed MiniLM, BGE, or optimized product profile. |
| `prepare`, `build`, `cases` | Download papers, build indexes, and freeze questions. |
| `align`, `score`, `identity` | Match source intervals, score evidence, and verify inputs. |
| `retrieve`, `report` | Run and score BM25, dense, and hybrid retrieval baselines. |
| `title_aliases` | Verify routing titles against public metadata and source PDFs. |

`ContextCompleteSupport` requires all evidence in at least one annotated support
set to appear in the returned context. `ContextEvidenceRecall` measures the
fraction covered. These are retrieval metrics; answer accuracy is not measured.
Auxiliary cases are reported separately from the 96-question coverage score.
Historical run artifacts and source snapshots remain in the ignored `.study/`
directory; the compact published results are in `evals/results/`.

## Tests

```sh
python -m pytest tests/test_qasper_scoring.py tests/test_qasper_run.py
```

Unit tests use synthetic inputs and temporary databases, without downloading the
corpus or running model inference.
