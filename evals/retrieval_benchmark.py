"""Reproducible source-passage retrieval benchmark for the six-paper corpus.

The benchmark never uses a model to decide its own ground truth. Gold anchors
refer to the original PDF page and are resolved against the current parser at
run time, so a parser change is visible instead of silently changing labels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
import time
import unicodedata

import pymupdf

from paperanchor.library import PaperLibrary, PassageHit
from paperanchor.semantic import (DEFAULT_EMBEDDING_MODEL, FastEmbedProvider,
                                  FastEmbedReranker, HybridRetriever,
                                  RerankingRetriever, SemanticIndex)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


class SemanticRetriever:
    def __init__(self, index: SemanticIndex, provider: FastEmbedProvider):
        self.index = index
        self.provider = provider

    def search(self, question: str, limit: int) -> tuple[PassageHit, ...]:
        return self.index.search(question, self.provider, limit=limit)


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_cases(path: Path) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark question ID")
    for case in cases:
        if case["answerability"] == "answerable" and not case["support_sets"]:
            raise ValueError(f"{case['id']}: answerable question has no support")
        if case["answerability"] == "unanswerable" and case["support_sets"]:
            raise ValueError(f"{case['id']}: unanswerable question has support")
        if case["split"] not in {"dev", "test"}:
            raise ValueError(f"{case['id']}: unknown split")
    return cases


def verify_corpus(data_dir: Path, db: Path, manifest: dict) -> dict[str, int]:
    expected = {item["file"]: item["sha256"] for item in manifest["papers"]}
    actual_files = {path.name for path in data_dir.glob("*.pdf")}
    if actual_files != set(expected):
        raise ValueError(f"Corpus files differ: missing={set(expected)-actual_files}, "
                         f"extra={actual_files-set(expected)}")
    for filename, digest in expected.items():
        if sha256((data_dir / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"PDF bytes changed: {filename}")
    if not db.is_file():
        raise ValueError(f"Index not found: {db}. Index the six PDFs first.")
    with sqlite3.connect(db) as connection:
        rows = connection.execute("SELECT id, file_path, file_hash FROM papers").fetchall()
    indexed = {Path(path).name: (paper_id, Path(path), digest)
               for paper_id, path, digest in rows}
    if set(indexed) != set(expected) or len(rows) != len(expected):
        raise ValueError("Index must contain exactly the six benchmark PDFs")
    for filename, (_, path, digest) in indexed.items():
        if path.resolve() != (data_dir / filename).resolve() or digest != expected[filename]:
            raise ValueError(f"Indexed PDF differs from corpus manifest: {filename}")
    return {filename: indexed[filename][0] for filename in expected}


def load_passages(db: Path) -> dict[tuple[str, int], list[tuple[int, str]]]:
    with sqlite3.connect(db) as connection:
        rows = connection.execute(
            "SELECT p.id, d.file_path, p.page_number, p.text "
            "FROM passages p JOIN papers d ON p.paper_id=d.id"
        ).fetchall()
    by_page: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    for passage_id, path, page, text in rows:
        by_page[(Path(path).name, page)].append((passage_id, normalized(text)))
    return by_page


def resolve_support_sets(case: dict, by_page: dict) -> tuple[list[set[int]], list[dict]]:
    resolved = []
    missing = []
    for group in case["support_sets"]:
        passage_ids = set()
        group_missing = False
        for locator in group:
            anchor = normalized(locator["anchor"])
            matches = [passage_id for passage_id, text in
                       by_page.get((locator["paper"], locator["page"]), [])
                       if anchor in text]
            if len(matches) != 1:
                missing.append({**locator, "matches": len(matches)})
                group_missing = True
            else:
                passage_ids.add(matches[0])
        if not group_missing:
            resolved.append(passage_ids)
    return resolved, missing


def score_hits(hits: tuple[PassageHit, ...], support_sets: list[set[int]],
               ks: list[int]) -> dict:
    ranks = {hit.passage_id: rank for rank, hit in enumerate(hits, start=1)}
    gold_ids = set().union(*support_sets) if support_sets else set()
    first = min((ranks[passage_id] for passage_id in gold_ids if passage_id in ranks),
                default=None)
    return {
        "first_support_rank": first,
        "mrr": 1 / first if first else 0.0,
        "any_support": {str(k): int(first is not None and first <= k) for k in ks},
        "complete_support": {
            str(k): int(any(all(ranks.get(passage_id, k + 1) <= k
                                for passage_id in group) for group in support_sets))
            for k in ks
        },
    }


def evaluate_depths(retriever, question: str, support_sets: list[set[int]],
                    ks: list[int]) -> dict:
    """Run each k separately: some retrievers change their candidate pool with k."""
    any_support = {}
    complete_support = {}
    latency = {}
    deepest_hits = ()
    deepest_score = None
    for k in ks:
        started = time.perf_counter()
        hits = retriever.search(question, limit=k)
        latency[str(k)] = round((time.perf_counter() - started) * 1000, 2)
        result = score_hits(hits, support_sets, [k])
        any_support[str(k)] = result["any_support"][str(k)]
        complete_support[str(k)] = result["complete_support"][str(k)]
        deepest_hits = hits
        deepest_score = result
    return {"any_support": any_support, "complete_support": complete_support,
            "first_support_rank": deepest_score["first_support_rank"],
            "mrr": deepest_score["mrr"], "latency_ms_by_k": latency,
            "top_hits": [{"rank": rank, "paper": hit.paper_path.name,
                          "page": hit.page_number, "passage_id": hit.passage_id}
                         for rank, hit in enumerate(deepest_hits, 1)]}


def aggregate(results: list[dict], ks: list[int]) -> dict:
    if not results:
        return {"n": 0}
    n = len(results)
    return {
        "n": n,
        "mrr": round(sum(row["mrr"] for row in results) / n, 4),
        "any_support": {str(k): round(sum(row["any_support"][str(k)]
                                           for row in results) / n, 4) for k in ks},
        "complete_support": {str(k): round(sum(row["complete_support"][str(k)]
                                                for row in results) / n, 4) for k in ks},
    }


def run(args: argparse.Namespace) -> dict:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = [case for case in load_cases(args.cases)
             if args.split == "all" or case["split"] == args.split]
    if not cases:
        raise ValueError("No questions in selected split")
    verify_corpus(args.data, args.db, manifest)
    by_page = load_passages(args.db)
    gold = {}
    unresolved = []
    for case in cases:
        if case["answerability"] == "unanswerable":
            continue
        support_sets, missing = resolve_support_sets(case, by_page)
        gold[case["id"]] = support_sets
        if missing:
            unresolved.append({"id": case["id"], "locators": missing})
    if unresolved:
        raise ValueError("Gold source anchors could not be resolved uniquely; "
                         "inspect parser/page mapping: " + json.dumps(unresolved, ensure_ascii=False))

    answerable = [case for case in cases if case["answerability"] == "answerable"]
    unanswerable = [case for case in cases if case["answerability"] == "unanswerable"]
    report = {
        "benchmark": manifest["name"], "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "db": str(args.db.resolve()),
        "parser_version": pymupdf.VersionBind,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "ks": args.ks,
        "counts": {"all": len(cases), "answerable": len(answerable),
                   "unanswerable_excluded_from_retrieval": len(unanswerable),
                   "review_status": dict((status, sum(case["review_status"] == status
                                                     for case in cases))
                                         for status in sorted({c["review_status"] for c in cases}))},
        "modes": {},
    }
    if args.validate_only:
        return report

    library = PaperLibrary(args.db)
    index = SemanticIndex(args.db)
    retrievers = {"bm25": library}
    if args.mode != "bm25":
        total = sum(len(rows) for rows in by_page.values())
        if index.count(DEFAULT_EMBEDDING_MODEL) != total:
            raise ValueError("Semantic index incomplete. Finish the background indexing job first.")
        provider = FastEmbedProvider(cache_dir=args.db.parent / "models")
        retrievers["semantic"] = SemanticRetriever(index, provider)
        hybrid = HybridRetriever(library, index, provider)
        retrievers["hybrid"] = hybrid
        if args.mode == "rerank":
            retrievers["rerank"] = RerankingRetriever(
                hybrid, FastEmbedReranker(args.rerank_model, args.db.parent / "models")
            )
    modes = (list(retrievers) if args.mode == "all" else [args.mode])
    for name in modes:
        retriever = retrievers[name]
        rows = []
        for case in answerable:
            scores = evaluate_depths(retriever, case["question"], gold[case["id"]],
                                     args.ks)
            source_papers = {locator["paper"] for locator in case["support_sets"][0]}
            paper_group = next(iter(source_papers)) if len(source_papers) == 1 else "cross-paper"
            rows.append({"id": case["id"], "split": case["split"],
                         "language": case["language"], "category": case["category"],
                         "paper_group": paper_group,
                         **scores})
        report["modes"][name] = {
            "overall": aggregate(rows, args.ks),
            "by_language": {language: aggregate([r for r in rows if r["language"] == language],
                                               args.ks)
                            for language in sorted({r["language"] for r in rows})},
            "by_category": {category: aggregate([r for r in rows if r["category"] == category],
                                               args.ks)
                            for category in sorted({r["category"] for r in rows})},
            "by_paper": {paper: aggregate([r for r in rows if r["paper_group"] == paper],
                                          args.ks)
                         for paper in sorted({r["paper_group"] for r in rows})},
            "median_latency_ms_by_k": {
                str(k): sorted(r["latency_ms_by_k"][str(k)] for r in rows)[len(rows) // 2]
                for k in args.ks
            },
            "cases": rows,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--db", type=Path, default=ROOT / ".paperanchor/library.sqlite3")
    parser.add_argument("--manifest", type=Path, default=HERE / "corpus.json")
    parser.add_argument("--cases", type=Path, default=HERE / "retrieval_cases.jsonl")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--mode", choices=("bm25", "semantic", "hybrid", "rerank", "all"),
                        default="all")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args()
    if not args.ks or any(k < 1 or k > 20 for k in args.ks):
        parser.error("--ks values must be between 1 and 20")
    args.ks = sorted(set(args.ks))
    try:
        report = run(args)
    except (ValueError, OSError, sqlite3.Error) as error:
        print(f"Benchmark cannot run: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"benchmark": report["benchmark"], "split": report["split"],
                      "counts": report["counts"],
                      "modes": {name: {"overall": data["overall"],
                                       "median_latency_ms_by_k": data["median_latency_ms_by_k"]}
                                for name, data in report["modes"].items()}},
                     ensure_ascii=False, indent=2))
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(f"Detailed report: {args.json_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
