"""Score saved retrieval text separately from retrieval and generation."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from .score import score_context, score_retrieval


def summarize(rows: list[dict]) -> dict:
    metrics = defaultdict(list)
    papers = defaultdict(list)
    for row in rows:
        papers[row["paper_id"]].append(row)
        for key, value in row["metrics"].items():
            metrics[key].append(value)
    per_paper = {str(paper): {key: statistics.mean(row["metrics"][key] for row in values)
                             for key in metrics}
                 for paper, values in papers.items()}
    durations = sorted(row["seconds"] for row in rows if row.get("seconds") is not None)
    return {"questions": len(rows), "papers": len(papers),
            "attempted": sum(row["status"] != "not_attempted" for row in rows),
            "not_attempted": sum(row["status"] == "not_attempted" for row in rows),
            "failures": sum(row["status"] not in ("ok", "not_attempted") for row in rows),
            "question_macro": {key: statistics.mean(values) for key, values in metrics.items()},
            "paper_macro": {key: statistics.mean(values[key] for values in per_paper.values())
                            for key in metrics},
            "per_paper": per_paper,
            "seconds": {"count": len(durations),
                        "p50": statistics.median(durations) if durations else None,
                        "p95": durations[min(len(durations) - 1, int(len(durations) * .95))]
                        if durations else None}}


def paired_comparison(left: list[dict], right: list[dict], *,
                      metric: str, seed: int = 20261005, samples: int = 2000) -> dict:
    a, b = ({row["case_id"]: row for row in rows} for rows in (left, right))
    if a.keys() != b.keys() or not a:
        raise ValueError("Paired comparison requires the same nonempty question set")
    clusters = defaultdict(list)
    for key in sorted(a):
        if a[key]["paper_id"] != b[key]["paper_id"]:
            raise ValueError("Paired question target papers differ")
        clusters[a[key]["paper_id"]].append(b[key]["metrics"][metric] - a[key]["metrics"][metric])
    differences = [value for values in clusters.values() for value in values]
    groups = list(clusters.values())
    rng = random.Random(seed)
    boot = sorted(statistics.mean(value for group in rng.choices(groups, k=len(groups))
                                  for value in group) for _ in range(samples))
    return {"metric": metric, "direction": "right_minus_left", "questions": len(a),
            "paper_clusters": len(groups), "wins": sum(value > 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "mean_difference": statistics.mean(differences),
            "cluster_bootstrap_95_percent": [boot[int(samples * .025)], boot[int(samples * .975)]],
            "seed": seed, "samples": samples, "interpretation": "descriptive"}


def make_report(run: dict, gold: dict) -> dict:
    for actual, reference in (("cases_fingerprint", "fingerprint"),
                              ("corpus_fingerprint", "corpus_fingerprint"),
                              ("public_cases_fingerprint", "public_cases_fingerprint")):
        if not gold.get(reference) or run.get(actual) != gold[reference]:
            raise ValueError(f"Retrieval and gold {reference} fingerprints differ")
    references = {case["case_id"]: case for case in gold["cases"]}
    config = run.get("config", {})
    split = config.get("split", "dev")
    representations = config.get("representations", sorted({r["representation"] for r in run["records"]}))
    methods = config.get("methods") or (["bm25", "dense", "hybrid"]
                                        + (["rerank"] if config.get("rerank_model") else []))
    expected_keys = {(case["case_id"], representation, method)
                     for case in gold["cases"] if split == "all" or case.get("split") == split
                     for representation in representations for method in methods}
    indexed = {}
    for record in run["records"]:
        key = (record["case_id"], record["representation"], record["method"])
        if key in indexed:
            raise ValueError("Duplicate retrieval result")
        if key not in expected_keys:
            raise ValueError("Unexpected retrieval case or configuration")
        expected_scope = ("full_library" if references[key[0]]["alignment"]["status"] == "aligned"
                          else "single_paper_auxiliary")
        if record["scope"] != expected_scope:
            raise ValueError("Retrieval scope differs from reference task")
        indexed[key] = record
    rows, auxiliary = [], []
    for key in sorted(expected_keys):
        reference = references[key[0]]
        scope = ("full_library" if reference["alignment"]["status"] == "aligned"
                 else "single_paper_auxiliary")
        record = indexed.get(key, {"case_id": key[0], "representation": key[1], "method": key[2],
                                   "scope": scope, "status": "not_attempted"})
        reference = references[record["case_id"]]
        if record["scope"] != "full_library":
            auxiliary.append({"case_id": record["case_id"], "status": record["status"],
                              "representation": record["representation"], "method": record["method"]})
            continue
        document = gold["documents"][str(reference["index_paper_id"])]
        alignment = reference["alignment"]
        groups = alignment["support_groups"]
        retrieval = score_retrieval(groups, record.get("ranked", []), document)
        context = score_context(groups, record.get("context", {}).get("chunks", []), document)
        metrics = {"MRR@20": retrieval["mrr_at_20"],
                   "ContextEvidenceRecall": context["evidence_recall"],
                   "ContextCompleteSupport": context["complete_support"]}
        for k, values in retrieval["at_k"].items():
            for public, field in (("PaperHit", "paper_hit"), ("EvidenceRecall", "evidence_recall"),
                                  ("CompleteSupport", "complete_support")):
                metrics[f"{public}@{k}"] = values[field]
        rows.append({"case_id": record["case_id"], "paper_id": document["paper_id"],
                     "representation": record["representation"], "method": record["method"],
                     "status": record["status"], "seconds": record.get("seconds"),
                     "answer_type": alignment.get("answer_type", "unknown"),
                     "granularity": "+".join(sorted({item["granularity"] for group in groups
                                                       for item in group})),
                     "metrics": metrics, "support": retrieval, "context_support": context})
    configurations = defaultdict(list)
    for row in rows:
        configurations[f"{row['representation']}/{row['method']}"].append(row)
    summaries = {}
    for name, values in configurations.items():
        summaries[name] = summarize(values)
        for field in ("answer_type", "granularity"):
            summaries[name][field] = {label: summarize([row for row in values if row[field] == label])
                                       for label in sorted({row[field] for row in values})}
    expected = {case["case_id"] for case in gold["cases"]
                if (split == "all" or case.get("split") == split)
                and case["alignment"]["status"] == "aligned"}
    all_present = bool(expected_keys) and indexed.keys() == expected_keys
    status = ("complete" if run["status"] == "complete" and gold["status"] == "frozen"
              and config.get("top_k", 20) >= 20 and all_present
              and all(record["status"] == "ok" for record in indexed.values()) else "incomplete")
    comparisons = {}
    names = sorted(configurations)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if {row["case_id"] for row in configurations[left]} == {
                    row["case_id"] for row in configurations[right]}:
                comparisons[f"{right} vs {left}"] = paired_comparison(
                    configurations[left], configurations[right], metric="EvidenceRecall@20")
    return {"status": status, "formal_scores": status == "complete",
            "scope": "title-hinted full-library retrieval under current PDF extraction",
            "gold_status": gold["status"], "expected_questions": len(expected),
            "expected_records": len(expected_keys), "attempted_records": len(indexed),
            "not_attempted_records": len(expected_keys) - len(indexed),
            "failed_records": sum(record["status"] != "ok" for record in indexed.values()),
            "provenance": {key: value for key, value in run.items()
                           if key.endswith("fingerprint") or key in ("config", "index_metadata", "cases_status")},
            "configurations": summaries, "paired_comparisons": comparisons,
            "questions": rows, "auxiliary": auxiliary,
            "generation": {"status": "not_evaluated", "reason": "Retrieval-only evaluation."}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = make_report(json.loads(args.retrieval.read_text(encoding="utf-8")),
                         json.loads(args.gold.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": report["status"], "formal_scores": report["formal_scores"]}))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
