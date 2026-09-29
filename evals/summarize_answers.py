"""Combine saved answer runs without repeating paid model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.answer_benchmark import summarize


def combine(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("No reports supplied")
    reference = reports[0]
    rows = []
    seen = set()
    for report in reports:
        for field in ("benchmark", "case_file_sha256", "workflow_sha256",
                      "model", "rerank_model", "top_k", "enable_decomposition"):
            if report.get(field) != reference.get(field):
                raise ValueError(f"Incompatible answer reports: {field}")
        for row in report["rows"]:
            if row["id"] in seen:
                raise ValueError(f"Duplicate case across reports: {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    return {
        "benchmark": reference["benchmark"],
        "case_file_sha256": reference.get("case_file_sha256"),
        "workflow_sha256": reference.get("workflow_sha256"),
        "model": reference.get("model"),
        "rerank_model": reference.get("rerank_model"),
        "case_ids": [row["id"] for row in rows],
        "summary": summarize(rows),
        "by_language": {
            language: summarize([row for row in rows if row["language"] == language])
            for language in sorted({row["language"] for row in rows})
        },
        "by_category": {
            category: summarize([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
        summary = combine(reports)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cannot combine reports: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "summary": summary["summary"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
