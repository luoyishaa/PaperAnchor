"""Run the real evidence workflow and export auditable, deliberately limited scores.

This runner does not use an LLM to grade itself. A cited gold passage is a
traceability check, not proof that the answer or individual claim is correct.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperanchor.agent import ASSESS_PROMPT, DECOMPOSE_PROMPT, GENERAL_PROMPT, research_question
from paperanchor.library import PaperLibrary
from paperanchor.llm import generate_with_config
from paperanchor.model_config import ModelConfigError, get_setting, load_model_config
from paperanchor.rag import Generate, PassageRetriever, SYSTEM_PROMPT
from paperanchor.semantic import (DEFAULT_EMBEDDING_MODEL, FastEmbedProvider,
                                  SemanticIndex, build_retriever)

from evals.retrieval_benchmark import (HERE, ROOT, load_cases, load_passages,
                                       resolve_support_sets, verify_corpus)


class TracingRetriever:
    def __init__(self, inner: PassageRetriever):
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def search(self, question: str, limit: int = 5, *,
               space_id: int | None = None):
        started = time.perf_counter()
        hits = self.inner.search(question, limit=limit, space_id=space_id)
        self.calls.append({
            "query": question, "limit": limit, "space_id": space_id,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "hits": [{"passage_id": hit.passage_id, "paper": hit.paper_path.name,
                      "page": hit.page_number, "score": hit.score, "text": hit.text}
                     for hit in hits],
        })
        return hits


class TracingGenerator:
    def __init__(self, inner: Generate):
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def __call__(self, system: str, prompt: str) -> str:
        stage = ({ASSESS_PROMPT: "assessment", DECOMPOSE_PROMPT: "planning", SYSTEM_PROMPT: "answer",
                  GENERAL_PROMPT: "general"}).get(system, "unknown")
        started = time.perf_counter()
        try:
            output = self.inner(system, prompt)
        except Exception as error:
            self.calls.append({"stage": stage, "system": system, "prompt": prompt,
                               "error": {"type": type(error).__name__,
                                         "message": str(error)},
                               "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
            raise
        self.calls.append({"stage": stage, "system": system, "prompt": prompt,
                           "output": output,
                           "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
        return output


def _complete(ids: set[int], groups: list[set[int]]) -> bool:
    return any(group <= ids for group in groups)


def workflow_fingerprint() -> str:
    """Include local edits: git HEAD alone does not identify the executed code."""
    digest = sha256()
    for name in ("agent.py", "rag.py", "semantic.py", "library.py", "llm.py"):
        digest.update(name.encode())
        digest.update((ROOT / "src" / "paperanchor" / name).read_bytes())
    return digest.hexdigest()


def score_saved_row(row: dict, groups: list[set[int]]) -> dict:
    """Reapply changed gold labels to an immutable model trace without new API calls."""
    if "error" in row:
        return {**row, "gold_support_sets": [sorted(group) for group in groups]}
    evidence = {item["label"]: item["passage_id"] for item in row["evidence"]}
    cited_ids = {evidence[label] for label in row["cited_evidence_ids"]
                 if label in evidence}
    first_ids = ({hit["passage_id"] for hit in row["retrieval_calls"][0]["hits"]}
                 if row["retrieval_calls"] else set())
    answerable = row["answerability"] == "answerable"
    first_support = _complete(first_ids, groups) if answerable else None
    final_support = _complete(set(evidence.values()), groups) if answerable else None
    cited_support = _complete(cited_ids, groups) if answerable else None
    if not answerable:
        diagnosis = ("provisional_paper_answer" if row["basis"] == "papers"
                     else "provisional_abstention")
    elif not final_support and row["basis"] == "papers":
        diagnosis = "paper_answer_without_gold_support"
    elif not final_support:
        diagnosis = "retrieval_or_query_gap"
    elif row["basis"] != "papers":
        diagnosis = "assessment_abstained_with_gold_evidence"
    elif row["status"] != "answered":
        diagnosis = "citation_format_failure"
    elif not cited_support:
        diagnosis = "citation_gold_coverage_gap"
    else:
        diagnosis = "needs_human_content_review"
    return {**row, "gold_support_sets": [sorted(group) for group in groups],
            "first_retrieval_complete_support": first_support,
            "final_evidence_complete_support": final_support,
            "cited_complete_support": cited_support, "diagnosis": diagnosis}


def evaluate_case(case: dict, groups: list[set[int]],
                  retriever: PassageRetriever, generate: Generate,
                  *, space_id: int | None = None, enable_decomposition: bool = False,
                  recovery_retriever: PassageRetriever | None = None) -> dict:
    """Score observable workflow facts; leave factuality and entailment to review."""
    traced_retriever = TracingRetriever(retriever)
    traced_recovery = (TracingRetriever(recovery_retriever)
                       if recovery_retriever is not None else None)
    traced_generator = TracingGenerator(generate)
    started = time.perf_counter()
    try:
        result = research_question(traced_retriever, case["question"], traced_generator,
                                   limit=5, space_id=space_id,
                                   allow_general_knowledge=False,
                                   enable_decomposition=enable_decomposition,
                                   recovery_retriever=traced_recovery)
    except Exception as error:
        return {
            "id": case["id"], "question": case["question"],
            "split": case["split"], "language": case["language"],
            "category": case["category"], "answerability": case["answerability"],
            "review_status": case["review_status"],
            "expected_answer": case["expected_answer"],
            "error": {"type": type(error).__name__, "message": str(error)},
            "retrieval_calls": traced_retriever.calls,
            "recovery_calls": traced_recovery.calls if traced_recovery else [],
            "model_calls": traced_generator.calls,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "human_review": {"factuality": None, "citation_support": None,
                             "abstention": None, "notes": ""},
        }

    answer = result.answer
    row = {
        "id": case["id"], "question": case["question"],
        "split": case["split"], "language": case["language"],
        "category": case["category"], "answerability": case["answerability"],
        "review_status": case["review_status"],
        "expected_answer": case["expected_answer"],
        "retrieval_calls": traced_retriever.calls,
        "recovery_calls": traced_recovery.calls if traced_recovery else [],
        "model_calls": traced_generator.calls,
        "attempted_queries": result.attempted_queries,
        "subquestions": [asdict(task) for task in result.subquestions],
        "basis": result.basis, "status": answer.status,
        "answer": answer.text, "follow_up_question": result.follow_up_question,
        "evidence": [{"label": item.evidence_id, "passage_id": item.passage_id,
                      "paper": item.paper_path.name, "page": item.page_number,
                      "bbox": item.bbox, "text": item.text}
                     for item in answer.evidence],
        "cited_evidence_ids": answer.cited_evidence_ids,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "human_review": {"factuality": None, "citation_support": None,
                         "abstention": None, "notes": ""},
    }
    return score_saved_row(row, groups)


def summarize(rows: list[dict]) -> dict:
    answered = [row for row in rows if row["answerability"] == "answerable"
                and "error" not in row]
    absent = [row for row in rows if row["answerability"] == "unanswerable"
              and "error" not in row]
    def count_true(key: str) -> int:
        return sum(row[key] is True for row in answered)

    return {
        "cases": len(rows), "errors": sum("error" in row for row in rows),
        "decomposed_cases": sum(bool(row.get("subquestions")) for row in rows),
        "retrieval_calls": sum(len(row.get("retrieval_calls", [])) +
                               len(row.get("recovery_calls", [])) for row in rows),
        "recovery_calls": sum(len(row.get("recovery_calls", [])) for row in rows),
        "model_calls": sum(len(row.get("model_calls", [])) for row in rows),
        "median_latency_ms": (sorted(row["latency_ms"] for row in rows)[len(rows) // 2]
                              if rows else None),
        "answerable_scored": len(answered),
        "first_retrieval_complete_support": count_true("first_retrieval_complete_support"),
        "final_evidence_complete_support": count_true("final_evidence_complete_support"),
        "paper_answers": sum(row["basis"] == "papers" for row in answered),
        "cited_complete_support": count_true("cited_complete_support"),
        "provisional_absent_cases": len(absent),
        "provisional_paper_answers": sum(row["basis"] == "papers" for row in absent),
        "diagnoses": dict(Counter(row["diagnosis"] for row in rows if "error" not in row)),
        "human_factuality_reviewed": 0,
        "human_citation_support_reviewed": 0,
    }


def answer_fingerprint(row: dict) -> str:
    """Bind a human judgment to this exact answer and cited evidence."""
    payload = {"status": row.get("status"), "answer": row.get("answer"),
               "cited_evidence_ids": row.get("cited_evidence_ids"),
               "evidence": [(item["label"], item["passage_id"])
                            for item in row.get("evidence", [])]}
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_review_template(report: dict, path: Path) -> None:
    """Never overwrite a reviewer's existing judgments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        for row in report["rows"]:
            item = {"id": row["id"], "answer_sha256": answer_fingerprint(row),
                    "question": row["question"],
                    "expected_answer": row["expected_answer"],
                    "answer": row.get("answer", ""),
                    "cited_locations": [
                        {"paper": evidence["paper"], "page": evidence["page"],
                         "excerpt": evidence["text"][:400]}
                        for evidence in row.get("evidence", [])
                        if evidence["label"] in row.get("cited_evidence_ids", [])
                    ],
                    "factuality": None, "citation_support": None,
                    "abstention": None, "absence_label": None, "notes": ""}
            output.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)


def run(args: argparse.Namespace, generate: Generate = generate_with_config) -> dict:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = [case for case in load_cases(args.cases)
             if (args.split == "all" or case["split"] == args.split)
             and (not args.case_id or case["id"] in args.case_id)]
    if not cases:
        raise ValueError("No questions match the selected split and IDs")
    verify_corpus(args.data, args.db, manifest)
    by_page = load_passages(args.db)
    groups_by_id = {}
    for case in cases:
        groups, missing = resolve_support_sets(case, by_page)
        if missing:
            raise ValueError(f"Gold anchors unresolved for {case['id']}: {missing}")
        groups_by_id[case["id"]] = groups

    report = {
        "benchmark": manifest["name"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case_file_sha256": sha256(args.cases.read_bytes()).hexdigest(),
        "workflow_sha256": workflow_fingerprint(),
        "split": args.split, "selected_case_ids": [case["id"] for case in cases],
        "evaluation_scope": "workflow trace and gold citation coverage; no automatic factuality grade",
        "allow_general_knowledge": False, "top_k": 5, "space_id": args.space_id,
        "enable_decomposition": getattr(args, "decompose", False),
        "limit_semantics": "per retrieval call; one rewrite or up to three subqueries; deduplicated union",
        "model": None, "rerank_model": get_setting("PAPERANCHOR_RERANK_MODEL"),
        "rows": [],
    }
    if args.validate_only:
        report["summary"] = {"cases_validated": len(cases)}
        return report

    if args.rescore_from:
        previous = json.loads(args.rescore_from.read_text(encoding="utf-8"))
        if previous["benchmark"] != manifest["name"] or [row["id"] for row in previous["rows"]] != [case["id"] for case in cases]:
            raise ValueError("Saved answer report does not match this corpus and case selection")
        if any(row["question"] != case["question"] or
               row["answerability"] != case["answerability"]
               for row, case in zip(previous["rows"], cases, strict=True)):
            raise ValueError("Saved questions or answerability labels changed")
        report = {**previous, "gold_scored_at_utc": datetime.now(timezone.utc).isoformat(),
                  "gold_rescored_from": str(args.rescore_from),
                  "case_file_sha256": sha256(args.cases.read_bytes()).hexdigest()}
        report["rows"] = [score_saved_row(row, groups_by_id[row["id"]])
                          for row in previous["rows"]]
        report["summary"] = summarize(report["rows"])
        report["by_language"] = {
            language: summarize([row for row in report["rows"] if row["language"] == language])
            for language in sorted({case["language"] for case in cases})
        }
        return report

    config = load_model_config() if generate is generate_with_config else None
    if config:
        report["model"] = {"provider": config.provider, "model": config.model}
    index = SemanticIndex(args.db)
    with sqlite3.connect(args.db) as connection:
        scoped = connection.execute(
            "SELECT COUNT(*) FROM space_papers WHERE space_id=?", (args.space_id,)
        ).fetchone()[0]
    if scoped != len(manifest["papers"]):
        raise ValueError("Selected research space must contain all benchmark papers")
    if index.count(DEFAULT_EMBEDDING_MODEL) != sum(map(len, by_page.values())):
        raise ValueError("Semantic index incomplete. Finish the indexing job first.")
    provider = FastEmbedProvider(cache_dir=args.db.parent / "models")
    retriever = build_retriever(PaperLibrary(args.db), index, provider,
                                rerank_model=report["rerank_model"],
                                cache_dir=args.db.parent / "models")
    for position, case in enumerate(cases, start=1):
        report["rows"].append(evaluate_case(case, groups_by_id[case["id"]],
                                            retriever, generate, space_id=args.space_id,
                                            enable_decomposition=getattr(args, "decompose", False)))
        print(f"Completed {position}/{len(cases)}: {case['id']}", file=sys.stderr,
              flush=True)
    report["summary"] = summarize(report["rows"])
    report["by_language"] = {
        language: summarize([row for row in report["rows"] if row["language"] == language])
        for language in sorted({case["language"] for case in cases})
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--db", type=Path, default=ROOT / ".paperanchor/library.sqlite3")
    parser.add_argument("--manifest", type=Path, default=HERE / "corpus.json")
    parser.add_argument("--cases", type=Path, default=HERE / "retrieval_cases.jsonl")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--space-id", type=int, default=1)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--decompose", action="store_true",
                        help="Enable experimental dynamic subquestion recovery")
    parser.add_argument("--rescore-from", type=Path,
                        help="Apply revised gold labels to a saved run; makes no model calls")
    parser.add_argument("--json-report", type=Path, required=True)
    parser.add_argument("--review-template", type=Path)
    args = parser.parse_args()
    if args.json_report.exists():
        parser.error(f"Report already exists: {args.json_report}")
    if args.review_template and args.review_template.exists():
        parser.error(f"Review template already exists: {args.review_template}")
    try:
        report = run(args)
    except (ValueError, OSError, sqlite3.Error, ModelConfigError) as error:
        print(f"Benchmark cannot run: {error}", file=sys.stderr)
        return 2
    try:
        write_report(report, args.json_report)
        if args.review_template and not args.validate_only:
            write_review_template(report, args.review_template)
    except FileExistsError as error:
        print(f"Output already exists: {error.filename}", file=sys.stderr)
        return 2
    print(json.dumps({"report": str(args.json_report), "summary": report["summary"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
