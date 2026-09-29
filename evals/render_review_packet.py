"""Render saved answer traces as a readable source-review packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def render(reports: list[dict], data_dir: Path) -> str:
    rows = [row for report in reports for row in report["rows"]]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs in review packet")
    lines = ["# PaperAnchor answer review packet", "",
             "Check the linked original PDF before recording a judgment in the matching",
             "JSONL review sheet. A matching gold passage does not prove that every claim",
             "in the answer is true or supported by its citation.", ""]
    for row in rows:
        lines += [f"## {row['id']} · {row['category']} · {row['language']}", "",
                  f"**Question:** {row['question']}", "",
                  f"**Reference rubric:** {row['expected_answer']}", "",
                  f"**Automatic diagnosis:** {row.get('diagnosis', 'request_error')}", "",
                  "**Model answer:**", "", row.get("answer", "Request failed."), "",
                  "**Cited PDF passages:**", ""]
        cited = set(row.get("cited_evidence_ids", []))
        evidence = [item for item in row.get("evidence", [])
                    if item["label"] in cited]
        if not evidence:
            lines += ["No cited passages.", ""]
        for item in evidence:
            pdf = data_dir / item["paper"]
            lines += [f"- **{item['label']}** · [PDF page {item['page']}]({pdf.as_uri()}#page={item['page']})",
                      "", f"  > {item['text'][:700].replace(chr(10), ' ')}", ""]
        lines += ["**Review:** factuality ___ · citation support ___ · abstention ___ · absence label ___", "",
                  "**Notes:**", "", "---", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
        body = render(reports, args.data.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cannot render review packet: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(body)
    print(f"Review packet: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
