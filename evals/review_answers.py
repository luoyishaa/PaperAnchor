"""Apply independent human judgments to a saved answer benchmark without API calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.answer_benchmark import answer_fingerprint


REVIEW_FIELDS = ("factuality", "citation_support", "abstention")
ABSENCE_LABELS = {None, "confirmed_absent", "has_answer", "uncertain"}


def score_reviews(report: dict, review_lines: list[dict]) -> dict:
    rows = {row["id"]: row for row in report["rows"]}
    seen = set()
    details = []
    for item in review_lines:
        case_id = item["id"]
        if case_id in seen:
            raise ValueError(f"Duplicate review ID: {case_id}")
        if case_id not in rows:
            raise ValueError(f"Unknown review ID: {case_id}")
        seen.add(case_id)
        row = rows[case_id]
        if item.get("answer_sha256") != answer_fingerprint(row):
            raise ValueError(f"Answer changed since review template: {case_id}")
        for field in REVIEW_FIELDS:
            value = item.get(field)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{case_id}: {field} must be true, false, or null")
        if item.get("absence_label") not in ABSENCE_LABELS:
            raise ValueError(f"{case_id}: unknown absence_label")
        if row["answerability"] == "answerable" and item.get("absence_label") is not None:
            raise ValueError(f"{case_id}: absence_label applies only to absent cases")
        details.append({"id": case_id, "answerability": row["answerability"],
                        "basis": row.get("basis"),
                        **{field: item.get(field) for field in REVIEW_FIELDS},
                        "absence_label": item.get("absence_label"),
                        "notes": item.get("notes", "")})

    def verdict(field: str) -> dict:
        reviewed = [item for item in details if item[field] is not None]
        return {"reviewed": len(reviewed), "passed": sum(item[field] is True
                                                           for item in reviewed)}

    confirmed_absent = [item for item in details
                        if item["absence_label"] == "confirmed_absent"]
    return {
        "benchmark": report["benchmark"],
        "source_report_timestamp_utc": report["timestamp_utc"],
        "review_entries": len(details),
        "reviewed_cases": sum(any(item[field] is not None for field in REVIEW_FIELDS)
                              or item["absence_label"] is not None for item in details),
        "factuality": verdict("factuality"),
        "citation_support": verdict("citation_support"),
        "abstention": verdict("abstention"),
        "absence_labels": {label: sum(item["absence_label"] == label for item in details)
                           for label in ("confirmed_absent", "has_answer", "uncertain")},
        "paper_answers_on_confirmed_absent": sum(item["basis"] == "papers"
                                                  for item in confirmed_absent),
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        reviews = [json.loads(line) for line in
                   args.reviews.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = score_reviews(report, reviews)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cannot score reviews: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "details"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
