# Evaluation and limits

PaperAnchor has a reproducible, source-located evaluation set, but no claim of parity with a general scientific question-answering benchmark. The [benchmark instructions](../evals/README.md) define the corpus, checksum validation, question labels, and commands.

## Recorded end-to-end run

The September 24, 2026 development snapshot used six related, text-based arXiv PDFs and 71 questions: 60 labeled answerable and 11 provisionally labeled absent. It used the default hybrid retriever, no cross-encoder reranker, and the `deepseek-flash` preset. The median end-to-end latency was 7.6 seconds in that environment.

| Trace measure | Result | Interpretation |
| --- | ---: | --- |
| Complete labeled evidence in first retrieval | 39 / 60 | A valid full support set was among the first search results. |
| Complete labeled evidence among final cited passages | 49 / 60 | The answer cited every passage in one labeled support set. |
| Paper-style answers to provisionally absent questions | 0 / 11 | All 11 withheld a paper-based answer; absence still needs independent review. |

These measures track **source coverage**, not factual correctness or whether every sentence is supported by its citation. The labels and initial audit were produced by the same assistant that developed the system; the test split was subsequently inspected and is not blind.

A separate three-paper, 12-question probe used previously unseen DPR, BEIR, and E5 PDFs. On its ten provisionally answerable questions, the default workflow cited a complete labeled support set in 3 / 10. Some labels may omit valid alternative passages, but this result makes the current limit clear: the six-paper score should not be generalized to other collections. The experimental recovery strategy reached 4 / 10 at added latency and is not enabled by default.

## Source inspection of representative answers

For V1, an assistant manually compared 12 saved answers with their cited PDF excerpts and inspected the relevant PDFs. Nine were substantive answers (`T08`, `C06`, `V04`, `X01`, `X06`, `G06`, `R09`, `S04`, `G08`); their main claims appeared supported by the cited passages. Three (`N01`, `N04`, `N10`) withheld paper-based answers; targeted full-text searches of the respective source PDFs found no requested measurement. This is an **assistant-conducted spot check**, not independent human annotation or a 9 / 9 accuracy estimate.

Two useful cautions from that check:

- `G08` asks about “high-level” summaries, while the cited cost figure is specifically for root-level `C0` summaries. The answer distinguishes `C0` from `C1`, but the reference wording should be tightened before using that case in a comparative score.
- `G06` and `V04` include extra, largely irrelevant cited material. The relevant conclusions are sourced, yet citation selection and answer brevity still need improvement.

The benchmark's human-review fields remain ungraded. An independent reviewer should check individual claims against the PDF, confirm the absence labels, and add valid alternative support passages before reporting answer accuracy or citation faithfulness. Scanned documents, mathematical notation, figures, and tables are outside this evaluation.
